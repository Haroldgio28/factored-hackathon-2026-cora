"""Tests for task 4.5: escalation triggers, dispute intake and the end-to-end handoff (REQ-15/16/17).

Two layers:

- unit: `update_turn_signals` fires on the two cross-turn REQ-15 triggers (Very-Negative streak,
  repeated tool failure) and resets on a broken chain; `dispute_intake` rejects a free-text id,
  keeps the transaction slot open until a TOOL-identified id is supplied, offers a freeze when
  fraud is suspected, and NEVER promises an outcome;
- end-to-end: driving the orchestrator with an E1-E4 intent lands in Handoff and persists a REQ-16
  package to the store; a Very-Negative streak escalates a turn the policy would have answered.

The classifier is not loaded in Phase 4, so intent/confidence is injected by monkeypatching
`Orchestrator._classify` (same seam as `test_agent_graph.py`). Tools are an in-memory fake with an
in-memory handoff store; the LLM is `StubLLMClient`; identity is the real service with an offset
clock (never `sleep`).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cora.agent import StubLLMClient
from cora.agent.graph import Orchestrator, TurnSignals
from cora.agent.state import SessionState, SessionStore
from cora.handoff.escalation import (
    TOOL_FAILURE_STREAK,
    VERY_NEGATIVE_STREAK,
    Sentiment,
    dispute_intake,
    update_turn_signals,
)
from cora.identity import MockIdentityService, Session
from cora.policy import Decision, Intent, PolicyEngine, Thresholds
from cora.tools import InMemoryHandoffStore, Result, Status
from cora.tools.models import BalanceData, CardDetailsData, HandoffReason

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CLI-000123"
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


# -- unit: cross-turn escalation triggers (REQ-15) -------------------------------------


def test_very_negative_two_consecutive_turns_escalates() -> None:
    state = SessionState(jti="j")
    assert update_turn_signals(state, sentiment=Sentiment.VERY_NEGATIVE) is None
    reason = update_turn_signals(state, sentiment=Sentiment.VERY_NEGATIVE)
    assert reason is HandoffReason.HUMAN_REQUEST
    assert state.very_negative_streak >= VERY_NEGATIVE_STREAK


def test_very_negative_streak_resets_on_neutral_turn() -> None:
    state = SessionState(jti="j")
    update_turn_signals(state, sentiment=Sentiment.VERY_NEGATIVE)
    update_turn_signals(state, sentiment=Sentiment.NEUTRAL)  # breaks the consecutive chain
    assert state.very_negative_streak == 0
    assert update_turn_signals(state, sentiment=Sentiment.VERY_NEGATIVE) is None  # only one again


def test_repeated_tool_failure_escalates() -> None:
    state = SessionState(jti="j")
    for _ in range(TOOL_FAILURE_STREAK - 1):
        assert update_turn_signals(state, tool_failed=True) is None
    assert update_turn_signals(state, tool_failed=True) is HandoffReason.HUMAN_REQUEST


def test_tool_failure_streak_resets_on_success() -> None:
    state = SessionState(jti="j")
    update_turn_signals(state, tool_failed=True)
    update_turn_signals(state, tool_failed=False)
    assert state.tool_failure_streak == 0


# -- unit: dispute intake (REQ-17) -----------------------------------------------------


def test_dispute_intake_rejects_free_text_keeps_transaction_slot_open() -> None:
    # A customer-typed id is NOT a tool-identified transaction: it never fills the slot.
    intake = dispute_intake(
        verified_transaction_id=None,
        reason="no reconozco el cargo",
        card_in_possession=True,
        fraud_suspected=False,
    )
    assert "transaction" in intake.open_slots
    assert not intake.complete
    assert intake.transaction_id is None


def test_dispute_intake_complete_with_tool_identified_transaction() -> None:
    intake = dispute_intake(
        verified_transaction_id="TRX-9",
        reason="no reconozco el cargo",
        card_in_possession=True,
        fraud_suspected=False,
    )
    assert intake.complete
    assert intake.transaction_id == "TRX-9"


def test_dispute_intake_offers_freeze_when_fraud_suspected() -> None:
    intake = dispute_intake(
        verified_transaction_id="TRX-9",
        reason="cargo sospechoso",
        card_in_possession=True,
        fraud_suspected=True,
    )
    assert intake.offer_freeze is True


def test_dispute_intake_never_promises_outcome() -> None:
    for fraud in (True, False):
        intake = dispute_intake(
            verified_transaction_id="TRX-9",
            reason="cargo",
            card_in_possession=True,
            fraud_suspected=fraud,
        )
        assert intake.promises_outcome is False


# -- end-to-end: handoff package built + persisted through the graph -------------------


class _FakeToolLayer:
    """Duck-typed `ToolLayer` for the graph: ownership answers + an in-memory handoff store."""

    def __init__(self, owned: set[str], store: InMemoryHandoffStore) -> None:
        self._owned = owned
        self.handoff_store = store

    def get_balance(self, tool_input):  # noqa: ANN001 - duck-typed test double
        if tool_input.product_id in self._owned:
            return Result[BalanceData](
                status=Status.OK,
                data=BalanceData(
                    product_id=tool_input.product_id,
                    currency="USD",
                    current_balance=100.0,
                    credit_limit=None,
                    available_credit=None,
                ),
            )
        return Result[BalanceData](status=Status.FORBIDDEN, message="not authorized")

    def get_card_details(self, tool_input):  # noqa: ANN001 - duck-typed test double
        return Result[CardDetailsData](status=Status.FORBIDDEN, message="not authorized")


def _issue_session(clock: _Clock) -> Session:
    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)
    challenge = service.begin_authentication(_CUSTOMER)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


def _orchestrator(clock: _Clock, store: InMemoryHandoffStore, *, owned: set[str] | None = None):
    tools = _FakeToolLayer(owned or set(), store)
    return Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=lambda _session: tools,
        llm=StubLLMClient(),
        clock=clock,
    )


def _patch_intent(monkeypatch: pytest.MonkeyPatch, intent: Intent, confidence: float = 0.99) -> None:
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (intent, confidence))


@pytest.mark.parametrize(
    ("intent", "expected_reason"),
    [
        (Intent.E1, HandoffReason.DISPUTE),
        (Intent.E2, HandoffReason.COMPLAINT),
        (Intent.E3, HandoffReason.FRAUD),
        (Intent.E4, HandoffReason.HUMAN_REQUEST),
    ],
)
def test_each_escalation_intent_builds_and_persists_package(
    monkeypatch: pytest.MonkeyPatch, intent: Intent, expected_reason: HandoffReason
) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _issue_session(clock)
    store = InMemoryHandoffStore()
    orch = _orchestrator(clock, store)
    _patch_intent(monkeypatch, intent)

    result = orch.step(session, "necesito ayuda con un problema")
    assert result.node == "Handoff"
    assert result.decision is Decision.ESCALATE
    # The package was built with the right reason and persisted to the store.
    assert result.handoff_case_id is not None
    assert result.handoff_package is not None
    assert result.handoff_package.reason is expected_reason
    assert result.handoff_case_id in store.cases
    # The verbatim request is in the package (masked by construction).
    assert result.handoff_package.customer_request == "necesito ayuda con un problema"


def test_e1_dispute_offers_freeze_and_promises_no_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _issue_session(clock)
    store = InMemoryHandoffStore()
    orch = _orchestrator(clock, store)
    _patch_intent(monkeypatch, Intent.E1)

    # Seed a fraud signal so the policy escalates with priority high (POL-040) -> fraud suspected.
    state = orch._store.require(session)  # noqa: SLF001 - test seam
    state.referenced_transaction_id = "TRX-9"
    monkeypatch.setattr(
        Orchestrator,
        "_build_policy_input",
        lambda self, **kw: (_fraud_policy_input(kw), [], False),
    )

    result = orch.step(session, "no reconozco este cargo")
    assert result.node == "Handoff"
    assert result.dispute is not None
    assert result.dispute.offer_freeze is True  # fraud suspected -> offer A1
    assert result.dispute.promises_outcome is False  # never promises an outcome (REQ-17)
    assert result.dispute.transaction_id == "TRX-9"  # tool-resolved id, not free text
    # The customer-facing copy is the handoff ack; it carries no refund/outcome promise.
    assert result.response is not None
    assert "reembolso" not in result.response.text.lower()


def _fraud_policy_input(kw: dict):
    """Build a PolicyInput carrying a fraud signal, for the E1-with-fraud end-to-end test."""
    from cora.policy import PolicyInput, ReferencedTransaction

    return PolicyInput(
        session_valid=True,
        intent=kw["intent"],
        intent_confidence=kw["confidence"],
        referenced_txn=ReferencedTransaction(is_fraud=True),
    )


def test_very_negative_streak_escalates_a_would_answer_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _issue_session(clock)
    store = InMemoryHandoffStore()
    orch = _orchestrator(clock, store, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"

    # First Very-Negative turn: still answers (one strike).
    first = orch.step(
        session, "esto es un desastre", turn_signals=TurnSignals(sentiment=Sentiment.VERY_NEGATIVE)
    )
    assert first.node == "Answer"
    # Second consecutive Very-Negative turn: escalates even though the policy would answer.
    second = orch.step(
        session, "sigue sin servir", turn_signals=TurnSignals(sentiment=Sentiment.VERY_NEGATIVE)
    )
    assert second.node == "Handoff"
    assert second.handoff_case_id is not None
