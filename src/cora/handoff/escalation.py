"""Turn-level escalation triggers and the E1 dispute-intake slot-filler (task 4.5, REQ-15/REQ-17).

The policy engine (`policy/engine.py`) already decides escalation for the triggers it can see in
a single `PolicyInput`: an escalation intent E1-E4 (POL-050), a fraud signal on a referenced
transaction (POL-040), and low confidence (POL-060). Two REQ-15 triggers are CROSS-TURN state, not
a single-turn fact, so they do not fit `PolicyInput`:

- `Very Negative` sentiment on two CONSECUTIVE turns;
- repeated tool failure after retries (a tool kept failing across turns).

`update_turn_signals` folds this turn's sentiment / tool-failure signal into two counters on
`SessionState` and returns a `HandoffReason` when either crosses its threshold - deterministic
code, never the model (the sentiment label is an untrusted signal, treated as data). The graph
calls it after the policy decision; if it fires, the turn escalates to Handoff even when the
policy did not. The two in-policy triggers stay in the policy engine (single source of truth);
this module only adds what a single `PolicyInput` cannot express.

The E1 dispute intake (REQ-17) is a small deterministic slot-filler: it reports which slots are
still open (transaction identified VIA A TOOL - never a free-text id - plus the reason and whether
the card is still in the customer's possession), and whether a freeze should be offered because
fraud is suspected. It NEVER promises a refund or an outcome: the only customer-facing copy on the
dispute path is the handoff-acknowledgement template, which promises nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from cora.tools.models import HandoffReason

if TYPE_CHECKING:
    from cora.agent.state import SessionState

__all__ = [
    "DisputeIntake",
    "Sentiment",
    "VERY_NEGATIVE_STREAK",
    "TOOL_FAILURE_STREAK",
    "dispute_intake",
    "update_turn_signals",
]

# Two consecutive Very-Negative turns escalate (REQ-15). A single bad turn is not enough - a
# customer can be curt once; two in a row is the signal the design calls out.
VERY_NEGATIVE_STREAK = 2
# A tool that keeps failing across this many turns (after its own `tenacity` retries) escalates
# rather than looping on an unavailable tool (REQ-15 "repeated tool failure after retries").
TOOL_FAILURE_STREAK = 2


class Sentiment(StrEnum):
    """The sentiment signal a turn carries. Only `VERY_NEGATIVE` drives the REQ-15 streak."""

    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


def update_turn_signals(
    state: SessionState,
    *,
    sentiment: Sentiment | None = None,
    tool_failed: bool = False,
) -> HandoffReason | None:
    """Fold this turn's cross-turn signals into state counters; return a reason if either fires.

    Deterministic and fail-safe: an unknown/absent sentiment resets the Very-Negative streak (a
    non-very-negative turn breaks the "consecutive" chain), and a turn with no tool failure resets
    the tool-failure streak. Returns `HandoffReason.HUMAN_REQUEST` (E4) when either counter reaches
    its threshold - both are "the automated path cannot proceed, hand to a human" situations.
    """
    if sentiment is Sentiment.VERY_NEGATIVE:
        state.very_negative_streak += 1
    else:
        state.very_negative_streak = 0

    if tool_failed:
        state.tool_failure_streak += 1
    else:
        state.tool_failure_streak = 0

    if state.very_negative_streak >= VERY_NEGATIVE_STREAK:
        return HandoffReason.HUMAN_REQUEST
    if state.tool_failure_streak >= TOOL_FAILURE_STREAK:
        return HandoffReason.HUMAN_REQUEST
    return None


class _Slot(StrEnum):
    TRANSACTION = auto()
    REASON = auto()
    CARD_IN_POSSESSION = auto()


@dataclass(frozen=True)
class DisputeIntake:
    """The state of an E1 dispute intake: which slots are filled and whether to offer a freeze.

    `transaction_id` is only ever set from a TOOL-verified transaction (the orchestrator resolves
    it via `search_transactions`/reference and verifies ownership before it reaches here); a
    free-text id the customer typed is rejected upstream and never fills this slot (REQ-17).
    `offer_freeze` is True when fraud is suspected, so the node can offer A1. `complete` is True
    only when every required slot is filled; until then `open_slots` lists what is still missing.
    `promises_outcome` is always False - the dispute path never promises a refund or an outcome.
    """

    transaction_id: str | None
    reason: str | None
    card_in_possession: bool | None
    offer_freeze: bool
    open_slots: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.open_slots

    @property
    def promises_outcome(self) -> bool:
        """A dispute intake NEVER promises a refund or outcome (REQ-17). Structurally always False."""
        return False


def dispute_intake(
    *,
    verified_transaction_id: str | None,
    reason: str | None,
    card_in_possession: bool | None,
    fraud_suspected: bool,
) -> DisputeIntake:
    """Build the E1 dispute-intake state from already-verified slots (REQ-17).

    `verified_transaction_id` MUST be a tool-identified id (resolved and ownership-checked by the
    orchestrator); pass `None` when the customer only gave free text, which leaves the transaction
    slot open rather than trusting the text. Offers a freeze (A1) when fraud is suspected. The
    returned intake promises no outcome by construction.
    """
    open_slots: list[str] = []
    if not verified_transaction_id:
        open_slots.append(_Slot.TRANSACTION.value)
    if not reason:
        open_slots.append(_Slot.REASON.value)
    if card_in_possession is None:
        open_slots.append(_Slot.CARD_IN_POSSESSION.value)
    return DisputeIntake(
        transaction_id=verified_transaction_id,
        reason=reason,
        card_in_possession=card_in_possession,
        offer_freeze=fraud_suspected,
        open_slots=open_slots,
    )


if __name__ == "__main__":  # self-check: streak thresholds + dispute never promises an outcome
    from cora.agent.state import SessionState

    st = SessionState(jti="j")
    assert update_turn_signals(st, sentiment=Sentiment.VERY_NEGATIVE) is None  # one strike
    assert update_turn_signals(st, sentiment=Sentiment.VERY_NEGATIVE) is HandoffReason.HUMAN_REQUEST
    st2 = SessionState(jti="j")
    assert update_turn_signals(st2, sentiment=Sentiment.VERY_NEGATIVE) is None
    assert update_turn_signals(st2, sentiment=Sentiment.NEUTRAL) is None  # chain broken -> reset
    assert st2.very_negative_streak == 0
    st3 = SessionState(jti="j")
    assert update_turn_signals(st3, tool_failed=True) is None
    assert update_turn_signals(st3, tool_failed=True) is HandoffReason.HUMAN_REQUEST

    free_text = dispute_intake(
        verified_transaction_id=None,
        reason="no reconozco el cargo",
        card_in_possession=True,
        fraud_suspected=True,
    )
    assert "transaction" in free_text.open_slots and not free_text.complete  # free text != verified id
    assert free_text.offer_freeze and not free_text.promises_outcome
    done = dispute_intake(
        verified_transaction_id="TRX-1",
        reason="no reconozco el cargo",
        card_in_possession=True,
        fraud_suspected=False,
    )
    assert done.complete and not done.promises_outcome
    print("escalation self-check OK")
