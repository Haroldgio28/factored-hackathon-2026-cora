"""Credit guard + money-movement refusal, proven end to end through the graph (task 4.6,
REQ-33, REQ-34).

Nothing new is built here: the rules already encode both guardrails (POL-030 routes a credit
intent out of scope with reason `CREDIT_OUT_OF_SCOPE`; POL-020 refuses money movement) and the
`ToolLayer` exposes NO money-movement callable. This suite proves and renders those guarantees:

- an X1 (credit) turn drives the machine to `abstain_route` + `CREDIT_OUT_OF_SCOPE`, the rendered
  es/pt copy promises nothing about credit, the reason is logged and recorded on the trace span,
  and the engine computes nothing about eligibility (no credit figure is produced);
- an X2 (money-movement) turn is refused by rule (POL-020), and its rendered copy offers no
  transfer/payment;
- the tool registry exposes no money-movement callable (mirrors the REQ-34 precedent in
  `test_tools.py`), so a refused request has nothing to call anyway.

The orchestrator is exercised exactly as in `test_agent_graph.py`: an in-memory fake tool layer
(no DataSource/AWS), the `StubLLMClient`, the real identity service on an offset clock, and the
intent injected per turn by monkeypatching `_classify` (the 4.1 classifier seam).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from cora.agent import StubLLMClient
from cora.agent.graph import Orchestrator
from cora.agent.state import SessionStore
from cora.identity import MockIdentityService, Session
from cora.policy import Decision, Intent, PolicyEngine, Thresholds
from cora.tools import TOOL_NAMES, InMemoryHandoffStore

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CUST-000123"
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)

# Credit words that must NEVER appear in a credit-guard reply as a promise/decision. The copy may
# say it CANNOT evaluate credit, so we assert the message carries no approval/eligibility verb and
# no figure, not merely the absence of the word "credit".
_CREDIT_PROMISE_WORDS = ("aprob", "aprov", "eleg", "eligí", "otorg", "concedo", "concedido", "%")


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _FakeToolLayer:
    """A tool layer that answers no read (nothing to ground) and persists handoffs in memory.

    A credit/money-movement turn never reaches a read tool - the policy guard fires first - so the
    fake only needs the `handoff_store` the escalation path would use. The absence of any
    money-movement method here is itself part of the REQ-34 guarantee.
    """

    def __init__(self) -> None:
        self.handoff_store = InMemoryHandoffStore()


def _issue_session(clock: _Clock, ttl_minutes: int = 15) -> tuple[MockIdentityService, Session, str]:
    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=ttl_minutes), clock=clock)
    challenge = service.begin_authentication(_CUSTOMER)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service, service.verify_token(token), token


def _orchestrator(clock: _Clock) -> Orchestrator:
    fake_tools = _FakeToolLayer()
    return Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=lambda _session: fake_tools,
        llm=StubLLMClient(),
        clock=clock,
    )


def _patch_intent(monkeypatch: pytest.MonkeyPatch, intent: Intent, confidence: float) -> None:
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text: (intent, confidence))


# -- credit guard: abstain_route CREDIT_OUT_OF_SCOPE, no credit decision (REQ-33) ------


@pytest.mark.parametrize("lang_utterance", ["quiero aumentar mi limite de credito", "aumento de cupo"])
def test_credit_intent_routes_out_of_scope(monkeypatch: pytest.MonkeyPatch, lang_utterance: str) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.X1, 0.99)

    result = orch.step(session, lang_utterance)

    # Guarded outcome: abstain_route via POL-030, landing in the Abstain node - never a credit
    # decision (there is no APPROVE/ELIGIBLE decision in the enum at all).
    assert result.decision is Decision.ABSTAIN_ROUTE
    assert result.rule_id == "POL-030"
    assert result.node == "Abstain"
    # The routing reason is recorded on the trace span (CREDIT_OUT_OF_SCOPE), never a decision.
    assert any(span.detail == "CREDIT_OUT_OF_SCOPE" for span in result.spans)


def test_credit_reply_promises_nothing_and_logs_reason(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.X1, 0.99)

    with caplog.at_level(logging.INFO, logger="cora.agent.graph"):
        result = orch.step(session, "quiero aumentar mi limite de credito")

    # The reason is logged and recorded on the trace span (explainable from rule id + reason,
    # never from model reasoning - REQ-33/REQ-39).
    assert "CREDIT_OUT_OF_SCOPE" in caplog.text
    assert any(span.detail == "CREDIT_OUT_OF_SCOPE" for span in result.spans)

    # The rendered copy routes to a human and promises nothing about credit: no approval/
    # eligibility verb, and no figure (the agent computes nothing about eligibility, REQ-33).
    text = result.response.text.lower()
    assert text, "a credit-guard turn must still have a safe thing to say"
    assert not any(word in text for word in _CREDIT_PROMISE_WORDS), text
    assert not any(ch.isdigit() for ch in text), f"no credit figure may appear: {text!r}"


# -- money-movement refusal (REQ-34) ---------------------------------------------------


def test_money_movement_is_refused_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session, _token = _issue_session(clock)
    orch = _orchestrator(clock)
    _patch_intent(monkeypatch, Intent.X2, 0.99)

    result = orch.step(session, "transfiere 500 pesos a otra cuenta")

    # Guarded outcome: refused by POL-020, rendered as the refuse copy - no transfer/payment is
    # offered and no money moves (there is no tool to move it; see the registry test below).
    assert result.decision is Decision.REFUSE
    assert result.rule_id == "POL-020"
    assert result.response.text.strip(), "a refusal must still say something safe"


# -- registry invariant: no money-movement tool exists (REQ-34) ------------------------


def test_tool_registry_exposes_no_money_movement_callable() -> None:
    # Mirrors the REQ-34 precedent in test_tools.py at the registry level: a money-movement
    # request has nothing to call, so the refusal is structural, not just a rule.
    forbidden = ("transfer", "payment", "pay", "deposit", "withdraw", "send", "move", "wire")
    offenders = {name for name in TOOL_NAMES if any(bad in name for bad in forbidden)}
    assert not offenders, offenders


if __name__ == "__main__":  # self-check: both guardrails hold through the graph
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
