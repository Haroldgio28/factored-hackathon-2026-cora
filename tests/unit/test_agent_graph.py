"""Tests for task 4.1: the design-section-4 orchestrator (REQ-06, REQ-07, REQ-13).

Every edge of the state machine is driven by crafting the signals a turn would produce:

- expired / absent session -> re-auth node, pending confirmation discarded (fail closed);
- an owned read -> Answer; an owned actionable card -> Confirm;
- low confidence -> Clarify, and two failed clarifications -> Escalate/Handoff (REQ-07);
- an escalation intent -> Escalate -> Handoff (ends);
- credit (X1) -> abstain_route; money movement (X2) -> refuse (both land in Abstain);
- an LLM proposal outside the allow-list is rejected while the decision still comes from the
  rules (REQ-13);
- cross-turn reference resolution feeds the ownership check from session state.

The classifier is not loaded in 4.1, so the intent/confidence are injected per turn by
monkeypatching `Orchestrator._classify` - this isolates the state machine from the (separate)
NLU model exactly as the plan intends. Tools are an in-memory fake (no DataSource/AWS); the LLM
is `StubLLMClient`; identity uses the real service with an offset clock (never `sleep`).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cora.agent import StubLLMClient
from cora.agent.graph import Orchestrator
from cora.agent.state import SessionStore
from cora.identity import MockIdentityService, Session
from cora.policy import Decision, Intent, PolicyEngine, Thresholds
from cora.tools import InMemoryHandoffStore, Result, Status
from cora.tools.models import BalanceData, CardDetailsData

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CUST-000123"
# REQ-07 pins tau_clarify at 0.50; inject it so the test does not depend on the file's default.
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _FakeToolLayer:
    """Duck-typed stand-in for `ToolLayer`: answers only the two ownership questions 4.1 asks.

    `owned` is the set of product ids the session customer owns; `status_by_id` overrides the
    card status so the POL-080 "state allows" branch can be exercised. Anything else fails closed
    with FORBIDDEN, mirroring the real layer.
    """

    def __init__(self, owned: set[str], status_by_id: dict[str, str] | None = None) -> None:
        self._owned = owned
        self._status = status_by_id or {}
        # The Handoff node (task 4.5) persists the package here; mirror the real `ToolLayer`.
        self.handoff_store = InMemoryHandoffStore()

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
        if tool_input.product_id in self._owned:
            return Result[CardDetailsData](
                status=Status.OK,
                data=CardDetailsData(
                    product_id=tool_input.product_id,
                    product_type="Tarjeta Crédito",
                    product_number_masked="****1234",
                    currency="USD",
                    product_status=self._status.get(tool_input.product_id, "Active"),
                    credit_limit=1000.0,
                    days_past_due=0,
                    expiration_date=None,
                ),
            )
        return Result[CardDetailsData](status=Status.FORBIDDEN, message="not authorized")


def _issue_session(clock: _Clock, ttl_minutes: int = 15) -> tuple[MockIdentityService, Session, str]:
    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=ttl_minutes), clock=clock)
    challenge = service.begin_authentication(_CUSTOMER)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service, service.verify_token(token), token


def _orchestrator(
    clock: _Clock,
    *,
    owned: set[str] | None = None,
    status_by_id: dict[str, str] | None = None,
) -> Orchestrator:
    store = SessionStore(clock=clock)
    fake_tools = _FakeToolLayer(owned or {"PRD-1"}, status_by_id)
    return Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=store,
        tool_layer_factory=lambda _session: fake_tools,
        llm=StubLLMClient(),
        clock=clock,
    )


def _patch_intent(monkeypatch: pytest.MonkeyPatch, intent: Intent | None, confidence: float) -> None:
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (intent, confidence))


# -- expiry / re-auth (fail closed) ----------------------------------------------------


def test_expired_session_routes_to_reauth_and_discards_state(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    # Seed state with a pending confirmation, then let the session expire.
    orch.step(session, "quiero mi saldo")
    orch._store.require(session).pending_confirmation_id = "CONF-X"  # noqa: SLF001 - test seam
    clock.now = session.expires_at + timedelta(seconds=1)

    result = orch.step(session, "quiero mi saldo")
    assert result.node == "Unauthenticated"
    assert result.decision is None  # no policy decision taken on the expiry path
    # State discarded: the pending confirmation is gone with it (fail closed, state dropped).
    assert orch._store.peek(session.jti) is None  # noqa: SLF001 - test seam


# -- Understand -> Decide edges --------------------------------------------------------


def test_owned_read_intent_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    result = orch.step(session, "cual es mi saldo")
    assert result.node == "Answer"
    assert result.decision is Decision.ANSWER
    assert result.rule_id == "POL-090"


def test_unowned_read_intent_does_not_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned=set())
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-OTHER"
    result = orch.step(session, "cual es el saldo")
    # Not owned -> POL-090 does not fire; falls through to the POL-999 abstain default.
    assert result.decision is Decision.ABSTAIN
    assert result.node == "Abstain"


def test_owned_actionable_card_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"}, status_by_id={"PRD-1": "Active"})
    _patch_intent(monkeypatch, Intent.A1, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    result = orch.step(session, "congela mi tarjeta")
    assert result.node == "Confirm"
    assert result.decision is Decision.CONFIRM
    assert result.rule_id == "POL-080"


def test_closed_card_does_not_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"}, status_by_id={"PRD-1": "Closed"})
    _patch_intent(monkeypatch, Intent.A1, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    result = orch.step(session, "congela mi tarjeta")
    # Owned but state does not allow -> POL-080 does not fire -> abstain default.
    assert result.decision is Decision.ABSTAIN


# -- Clarify / escalate (REQ-07) -------------------------------------------------------


def test_low_confidence_clarifies_then_escalates_after_two(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    # Confidence between tau_escalate (0.40) and tau_clarify (0.50) -> POL-070 clarify.
    _patch_intent(monkeypatch, Intent.I1, 0.45)

    first = orch.step(session, "mmm no se")
    assert first.node == "Clarify" and first.decision is Decision.CLARIFY

    second = orch.step(session, "sigue sin estar claro")
    # Second consecutive failed clarification -> escalate (E4), ending in Handoff.
    assert second.node == "Handoff"
    assert second.decision is Decision.ESCALATE
    # Counter resets so a later clarify starts a fresh pair.
    assert orch._store.require(session).clarification_count == 0  # noqa: SLF001


def test_escalation_intent_goes_to_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.E2, 0.99)

    result = orch.step(session, "quiero poner una queja")
    assert result.node == "Handoff"
    assert result.decision is Decision.ESCALATE
    assert result.rule_id == "POL-050"


# -- credit / money-movement guards ----------------------------------------------------


def test_credit_intent_abstains_out_of_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.X1, 0.99)

    result = orch.step(session, "quiero aumentar mi limite de credito")
    assert result.decision is Decision.ABSTAIN_ROUTE
    assert result.rule_id == "POL-030"
    assert result.node == "Abstain"


def test_money_movement_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.X2, 0.99)

    result = orch.step(session, "transfiere dinero")
    assert result.decision is Decision.REFUSE
    assert result.rule_id == "POL-020"


# -- allow-list: proposal rejected, decision still from the rules (REQ-13) -------------


def test_out_of_allow_list_proposal_is_rejected_but_rules_decide(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    # Propose CONFIRM for a read intent: not in the allow-list for I1.
    result = orch.step(session, "cual es mi saldo", proposed_action=Decision.CONFIRM)
    assert result.proposal_rejected is True
    # The rules still decide ANSWER regardless of the rejected proposal.
    assert result.decision is Decision.ANSWER


# -- every node records a trace span ---------------------------------------------------


def test_each_turn_records_trace_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.E3, 0.99)

    result = orch.step(session, "creo que hay un fraude")
    assert result.spans, "every node must leave a trace span"
    assert result.spans[-1].node == "Handoff"
    assert all(span.node for span in result.spans)


# -- stable per-turn trace id (task 5.1, REQ-39) ---------------------------------------


def test_each_turn_has_a_stable_unique_trace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1, 0.99)
    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"

    first = orch.step(session, "cual es mi saldo")
    second = orch.step(session, "cual es mi saldo")
    # 32-char uuid4 hex, present on every turn, and different between turns.
    assert len(first.trace_id) == 32 and first.trace_id.isalnum()
    assert len(second.trace_id) == 32
    assert first.trace_id != second.trace_id


def test_handoff_turn_threads_trace_id_into_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.E2, 0.99)

    result = orch.step(session, "quiero poner una queja")
    assert result.node == "Handoff"
    # The package a human/UI reads carries the SAME trace id as the turn (5.1 end-to-end tie).
    assert result.handoff_package is not None
    assert result.handoff_package.trace_ref == result.trace_id
    assert result.trace_id


# -- explicit UI language choice (input side, REQ-05/REQ-18) ---------------------------


def _ban_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `language.detect` fail the test if it is ever called (proves detection was skipped)."""
    from cora.nlu import language as language_mod

    def _boom(_text: str):  # noqa: ANN202 - test double, never returns
        raise AssertionError("language.detect must NOT run when an explicit valid choice is given")

    monkeypatch.setattr(language_mod, "detect", _boom)


def test_explicit_language_pt_overrides_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1, 0.99)
    _ban_detect(monkeypatch)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    # A deliberately Spanish-ish utterance: detection (banned) would say es, but the explicit
    # choice must win and set the turn language to pt without running the detector at all.
    result = orch.step(session, "cual es mi saldo", language="pt")
    assert result.language == "pt"
    assert state.language == "pt"


def test_explicit_language_es_sets_spanish(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1, 0.99)
    _ban_detect(monkeypatch)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    result = orch.step(session, "quero saber meu saldo", language="es")
    assert result.language == "es"
    assert state.language == "es"


def test_invalid_language_falls_back_to_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    from cora.nlu import language as language_mod

    real_detect = language_mod.detect
    seen: list[str] = []

    def _spy(text: str):  # noqa: ANN202 - delegates to the real stub detector
        seen.append(text)
        return real_detect(text)

    monkeypatch.setattr(language_mod, "detect", _spy)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    # An invalid value ('en') is ignored (not a 422 here, not an error): the detector runs on the
    # Portuguese utterance and drives the turn language, proving fail-closed fallback.
    result = orch.step(session, "quero saber meu saldo por favor", language="en")
    assert seen == ["quero saber meu saldo por favor"]  # detection ran on the raw text
    assert result.language == "pt"  # the stub detector classified the pt utterance


def test_no_language_keeps_detection_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock, owned={"PRD-1"})
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"
    # No explicit choice (today's behavior): the stub detector classifies the es utterance.
    result = orch.step(session, "cual es mi saldo por favor")
    assert result.language == "es"
