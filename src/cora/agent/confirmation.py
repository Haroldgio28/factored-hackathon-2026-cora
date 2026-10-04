"""The Confirm -> Execute -> Verify protocol around the card-action tools (task 4.4, REQ-09, REQ-14).

The tool layer already owns the mechanics: `ToolLayer._card_action` writes the overlay, reads it
back and reports OK only if the post-condition holds (REQ-09), and it refuses to execute without a
valid confirmation bound to the exact customer+action+product (REQ-14, `tools/confirmations.py`).
4.4 is the PROTOCOL that drives those seams across two turns, and it is DETERMINISTIC CODE - the
LLM never decides whether the customer said yes (security steering P1).

Two turns, two functions:

- `request_confirmation(state, action, product_id, masked_number)` records what the machine is
  about to do on `SessionState` and leaves the restatement to the 4.2 `Outcome.CONFIRM` template
  (exact action + masked product number). Nothing is executed yet.
- `resolve_confirmation(state, utterance_masked, tools, *, now)` runs on the NEXT turn while a
  pending confirmation is open. It classifies the reply as affirmative / negative / ambiguous with
  a pinned es+pt lexicon (the reply is data, never an instruction), and ONLY on an explicit
  affirmative in the same still-open pending mints a single-use confirmation
  (`ConfirmationStore.issue`) and calls `freeze_card`/`unfreeze_card`. Negative, ambiguous or
  absent -> cancelled, nothing executed (REQ-14). After one execution the pending is cleared, so a
  replayed "sí" does nothing (single-use, closing the task-2.3 note).

Expiry is handled upstream by the orchestrator/`SessionStore` (an expired session discards all
state, dropping the pending with it), so this module only sees live sessions. Fail closed on a
non-affirmative reply or a tool that does not verify: disclose nothing done, offer a human.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto

from cora.agent.state import SessionState
from cora.tools import Result, Status, ToolLayer
from cora.tools.confirmations import CardAction
from cora.tools.models import CardActionData, CardFreezeInput

__all__ = [
    "ConfirmationOutcome",
    "PendingConfirmation",
    "Reply",
    "classify_reply",
    "request_confirmation",
    "resolve_confirmation",
]


@dataclass(frozen=True)
class PendingConfirmation:
    """What the machine restated and is waiting for an explicit yes/no on (REQ-14).

    Bound to ONE action + product so the executed write cannot drift from what was restated; the
    masked number is kept only so the trace/cancel copy can refer to the same card the customer saw.
    """

    action: CardAction
    product_id: str
    masked_number: str


class Reply(StrEnum):
    """The deterministic classification of the customer's confirmation reply (code, not the LLM)."""

    AFFIRMATIVE = auto()
    NEGATIVE = auto()
    AMBIGUOUS = auto()


class ConfirmationOutcome(StrEnum):
    """What `resolve_confirmation` did, so the orchestrator renders the right node/template."""

    EXECUTED_OK = auto()  # affirmative + tool verified the post-condition (REQ-09) -> report success
    NOT_COMPLETED = auto()  # affirmative but the tool did not verify -> not done, offer a human
    CANCELLED = auto()  # negative/ambiguous/absent -> nothing executed (REQ-14)


# es/pt/en affirmative and negative lexicons. Matched as whole words on accent-folded text so
# "sí"/"si"/"SÍ" and "não"/"nao" all hit. Ordered: a reply that carries a negative marker is
# treated as negative even if it also contains an affirmative token (fail closed on mixed intent).
_AFFIRMATIVE_RE = re.compile(
    r"\b(si|sim|claro|dale|correcto|confirmo|confirmar|afirmativo|adelante|hazlo|"
    r"procede|continua|de acuerdo|esta bien|ok|okay|yes|sure|go ahead)\b"
)
_NEGATIVE_RE = re.compile(
    r"\b(no|nao|nunca|cancela|cancelar|cancele|detente|para|mejor no|ahora no|"
    r"nope|stop|cancel)\b"
)


def _normalize(text: str) -> str:
    """Lowercase and strip accents so one spelling matches every variant (es MX/CO/AR + pt BR)."""
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def classify_reply(utterance_masked: str) -> Reply:
    """Classify a confirmation reply deterministically (REQ-14). Negative wins over affirmative.

    Treats the reply strictly as data. Anything that is not an unambiguous yes/no - empty, both
    markers, or neither - is AMBIGUOUS, which the protocol treats like a no (fail closed).
    """
    text = _normalize(utterance_masked)
    has_no = bool(_NEGATIVE_RE.search(text))
    has_yes = bool(_AFFIRMATIVE_RE.search(text))
    if has_no:
        return Reply.NEGATIVE
    if has_yes:
        return Reply.AFFIRMATIVE
    return Reply.AMBIGUOUS


def request_confirmation(
    state: SessionState, action: CardAction, product_id: str, masked_number: str
) -> PendingConfirmation:
    """Open a pending confirmation on the session (the restatement itself is a 4.2 template)."""
    pending = PendingConfirmation(action=action, product_id=product_id, masked_number=masked_number)
    state.pending_action = pending
    # Keep the legacy string field in lockstep so the expiry/discard paths that touch it still work.
    state.pending_confirmation_id = f"{action.value}:{product_id}"
    return pending


def resolve_confirmation(
    state: SessionState,
    utterance_masked: str,
    tools: ToolLayer,
    *,
    now: datetime,
) -> tuple[ConfirmationOutcome, Result[CardActionData] | None]:
    """Resolve an open pending confirmation on this turn (REQ-09, REQ-14).

    Clears the pending FIRST (single-use: a replayed "sí" after this turn finds nothing to act on),
    then executes only on an explicit affirmative by minting a confirmation bound to this exact
    customer+action+product and calling the matching tool. The tool itself read-backs and reports
    OK only if the post-condition holds; `EXECUTED_OK` is returned only for that OK result, else
    `NOT_COMPLETED` (offer a human). Negative/ambiguous -> `CANCELLED`, nothing executed.
    """
    pending = state.pending_action
    # Single-use: consume the pending now so no later turn can replay this affirmative.
    state.clear_pending_confirmation()
    if pending is None:
        return ConfirmationOutcome.CANCELLED, None

    if classify_reply(utterance_masked) is not Reply.AFFIRMATIVE:
        # Negative, ambiguous or empty -> do not execute, disclose nothing done (REQ-14).
        return ConfirmationOutcome.CANCELLED, None

    # Explicit affirmative in the same (still-open) session: mint a single-use confirmation bound
    # to exactly what was restated, then execute. The tool re-validates the binding and expiry.
    confirmation_id = tools.confirmations.issue(
        customer_id=tools.customer_id,
        action=pending.action,
        product_id=pending.product_id,
        now=now,
    )
    tool = tools.freeze_card if pending.action is CardAction.FREEZE else tools.unfreeze_card
    result = tool(CardFreezeInput(product_id=pending.product_id, confirmation_id=confirmation_id))
    # REQ-09: report success only on the tool's verified OK read-back; otherwise not completed.
    outcome = (
        ConfirmationOutcome.EXECUTED_OK if result.status is Status.OK else ConfirmationOutcome.NOT_COMPLETED
    )
    return outcome, result


if __name__ == "__main__":  # self-check: the security-critical yes/no lexicon (REQ-14)
    assert classify_reply("sí, adelante") is Reply.AFFIRMATIVE
    assert classify_reply("no, mejor no") is Reply.NEGATIVE
    assert classify_reply("sim claro") is Reply.AFFIRMATIVE
    assert classify_reply("no estoy seguro") is Reply.NEGATIVE  # carries "no" -> fail closed
    assert classify_reply("mmm") is Reply.AMBIGUOUS
    # A missing pending cancels (nothing to execute); the full execute+verify path is covered in
    # tests/unit/test_confirmation.py with a real ToolLayer + in-memory overlay.
    assert SessionState(jti="j", customer_id="CUST-1").pending_action is None
    print("confirmation stub self-check OK")
