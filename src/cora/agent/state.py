"""Per-session conversation state and its in-memory store (task 4.1, design section 4, REQ-06).

The orchestrator (`graph.py`) is a deterministic state machine; between turns it needs to
remember exactly the fields design section 4 lists: the session language, the verified
customer, the product/transaction the conversation last referred to, a pending confirmation,
the open clarification slots, and the turn history. `SessionState` holds precisely those and
nothing more.

Two invariants are structural here, not left to a caller to remember:

- `customer_id` is only ever copied FROM a verified `Session` (`SessionState.authenticate`),
  never from model output or user text (security steering, REQ-12). There is no setter that
  takes a free-standing id.
- The store is keyed by the session `jti` and fails closed: `require(session)` returns the live
  state only while the session matches an unexpired entry; on expiry, a `jti` mismatch (a new
  login) or an absent entry it DISCARDS any state and forces re-authentication, dropping the
  pending confirmation with it (design section 4 `Expired -> Unauthenticated: state discarded`).

The store mirrors the existing in-memory-store pattern (`handoff_store.py`,
`confirmations.py`): a dict plus an injectable `clock`, no new dependency. Turn history records
are already-masked (the orchestrator masks before storing), so the state never holds raw PII.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cora.identity import Session

if TYPE_CHECKING:
    from cora.agent.confirmation import PendingConfirmation

__all__ = ["SessionState", "SessionStore", "TurnRecord"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class TurnRecord:
    """One masked turn in the history (design section 4 "turn history").

    `utterance_masked` has already been through `mask_pii`; no raw PII is ever stored. `intent`
    and `decision` are the classifier/policy outputs for the turn, kept for cross-turn reference
    resolution and the handoff transcript (never re-interpreted as instructions).
    """

    utterance_masked: str
    intent: str | None
    decision: str | None


@dataclass
class SessionState:
    """The REQ-06 per-session store: exactly the fields design section 4 enumerates.

    Created empty for a `jti`; `authenticate` binds it to a verified `Session`. Everything the
    machine remembers between turns lives here and is discarded wholesale when the session
    expires (the store drops the whole entry), so there is no partial-state fail-open path.
    """

    jti: str
    language: str = "es"
    # Only ever set by `authenticate` from a verified Session (never from model/user text).
    customer_id: str | None = None
    # The product/transaction the conversation last referred to, for deterministic cross-turn
    # reference resolution ("that card", "o ultimo"); set by the orchestrator from tool results.
    referenced_product_id: str | None = None
    referenced_transaction_id: str | None = None
    # Ordered ids of the products/transactions last listed to the customer, so an ordinal
    # reference ("the second one", "la segunda") resolves deterministically from state.
    last_listed_product_ids: list[str] = field(default_factory=list)
    last_listed_transaction_ids: list[str] = field(default_factory=list)
    # A confirmation awaiting the customer's explicit yes/no (the full protocol is task 4.4).
    pending_confirmation_id: str | None = None
    # The exact action+product restated to the customer, awaiting their explicit affirmative
    # (task 4.4). `None` whenever no confirmation is open; cleared on resolve/cancel/expiry.
    pending_action: PendingConfirmation | None = None
    # Open clarification slots the machine is waiting on (design section 4 "open slots").
    open_slots: list[str] = field(default_factory=list)
    # E1 dispute-intake slots (task 4.5, REQ-17): the free-text reason and whether the card is
    # still in the customer's possession. The disputed transaction is NOT stored here - it is only
    # ever taken from a tool-resolved id (`referenced_transaction_id`), never from free text.
    dispute_reason: str | None = None
    card_in_possession: bool | None = None
    # Count of consecutive failed clarifications; 2 -> escalate (E4) per REQ-07.
    clarification_count: int = 0
    # Cross-turn escalation streaks (REQ-15, task 4.5): consecutive Very-Negative-sentiment turns
    # and consecutive turns a tool kept failing. Both reset on a turn that breaks the chain; 2 of
    # either escalates. Updated by `handoff.escalation.update_turn_signals` (deterministic code).
    very_negative_streak: int = 0
    tool_failure_streak: int = 0
    turn_history: list[TurnRecord] = field(default_factory=list)

    @property
    def authenticated(self) -> bool:
        return self.customer_id is not None

    def authenticate(self, session: Session) -> None:
        """Bind this state to a verified session, copying ONLY the customer id (REQ-12).

        The `customer_id` enters the conversation here and nowhere else; `jti` is checked by the
        store, not trusted from here.
        """
        self.customer_id = session.customer_id

    def record_turn(self, utterance_masked: str, intent: str | None, decision: str | None) -> None:
        """Append an already-masked turn record (the orchestrator masks before calling)."""
        self.turn_history.append(
            TurnRecord(utterance_masked=utterance_masked, intent=intent, decision=decision)
        )

    def clear_pending_confirmation(self) -> None:
        self.pending_confirmation_id = None
        self.pending_action = None


class SessionStore:
    """In-memory per-session state store keyed by `jti`, fail-closed on expiry/mismatch.

    Mirrors the `InMemory*` store pattern (dict + injectable clock). The clock is only used to
    decide whether a verified session is still live; expiry itself is owned by the identity
    service (`Session.expires_at`), so this store never invents a TTL.
    """

    def __init__(self, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self._states: dict[str, SessionState] = {}
        self._clock = clock

    def require(self, session: Session) -> SessionState:
        """Return the live state for a verified, unexpired session, creating it on first use.

        Fail closed: if the session has expired by this store's clock, any state for that `jti`
        is discarded (dropping the pending confirmation with it) and a fresh, UNauthenticated
        state is returned so the machine routes back through re-auth. A `jti` never seen before
        yields a fresh authenticated-on-first-use state.
        """
        if self._clock() >= session.expires_at:
            self.discard(session.jti)
            return SessionState(jti=session.jti)
        state = self._states.get(session.jti)
        if state is None:
            state = SessionState(jti=session.jti)
            state.authenticate(session)
            self._states[session.jti] = state
        return state

    def discard(self, jti: str) -> None:
        """Drop all state for a session (expiry / logout). Idempotent; pending confirmation goes too."""
        self._states.pop(jti, None)

    def peek(self, jti: str) -> SessionState | None:
        """Non-mutating lookup for tests/inspection; does not create or expire anything."""
        return self._states.get(jti)
