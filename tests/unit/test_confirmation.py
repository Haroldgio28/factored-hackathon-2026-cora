"""Tests for task 4.4: the Confirm -> Execute -> Verify protocol (REQ-09, REQ-14).

The protocol spans two turns and is DETERMINISTIC CODE, never the LLM:

- turn 1 (intent A1/A2 on an owned, actionable card) -> Confirm: the machine restates the exact
  action + MASKED product number and opens a pending confirmation; nothing is executed;
- turn 2 (the customer's reply) -> Execute+Verify: only an EXPLICIT affirmative in the same
  still-open session mints a confirmation and calls the freeze/unfreeze tool, which read-backs and
  reports OK only if the post-condition holds (REQ-09); negative/ambiguous/absent -> cancelled,
  nothing executed (REQ-14); an expired session between the two turns drops the pending and routes
  to re-auth; a replayed "yes" after execution is a no-op (single-use).

These drive the REAL `ToolLayer.freeze_card`/`unfreeze_card` against a tiny self-built Parquet
landing (never the real `data/`) with REAL verified sessions, so `customer_id` is injected exactly
as in production and the overlay read-back is the real one. The classifier is not loaded in Phase
4, so the per-turn intent/confidence is injected by monkeypatching `Orchestrator._classify`
(same seam as `test_agent_graph.py`). The LLM is `StubLLMClient`; no network, no AWS.

`classify_reply` is also unit-tested directly for the es+pt yes/no lexicon, including the
fail-closed rule that a reply carrying a negative marker is never read as affirmative.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from cora.agent import StubLLMClient
from cora.agent.confirmation import Reply, classify_reply
from cora.agent.graph import Orchestrator
from cora.agent.state import SessionStore
from cora.data.datasource import LocalSource
from cora.identity import MockIdentityService, Session
from cora.policy import Intent, PolicyEngine, Thresholds
from cora.tools import (
    CardFreezeState,
    InMemoryCardOverlay,
    InMemoryConfirmationStore,
    ToolLayer,
)
from cora.tools.card_overlay import CardOverlay

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CLI-OWNER0000001"
_CARD = "PRD-OWNERCARD01"
_MASKED = "****3827"  # last four of the fixture PAN below
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)
_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


# -- fixture landing (one owned, active card) ------------------------------------------


def _build_landing(root: Path) -> None:
    products = pd.DataFrame(
        [
            {
                "product_id": _CARD,
                "customer_id": _CUSTOMER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            }
        ]
    )
    root.mkdir(parents=True, exist_ok=True)
    products.astype("string").to_parquet(root / "products.parquet", index=False)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("raw_parquet")
    _build_landing(root)
    return root


@pytest.fixture
def source(landing: Path) -> LocalSource:
    return LocalSource(root=landing)


def _issue_session(clock: _Clock) -> Session:
    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)
    challenge = service.begin_authentication(_CUSTOMER)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


def _orchestrator(
    source: LocalSource,
    clock: _Clock,
    *,
    overlay: CardOverlay | None = None,
    confirmations: InMemoryConfirmationStore | None = None,
) -> Orchestrator:
    overlay = overlay if overlay is not None else InMemoryCardOverlay()
    confirmations = confirmations if confirmations is not None else InMemoryConfirmationStore()

    def _factory(session: Session) -> ToolLayer:
        return ToolLayer(
            session,
            source,
            card_overlay=overlay,
            confirmations=confirmations,
            clock=clock,
        )

    return Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=_factory,
        llm=StubLLMClient(),
        clock=clock,
    )


def _patch_intent(monkeypatch: pytest.MonkeyPatch, intent: Intent | None, confidence: float) -> None:
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (intent, confidence))


def _confirm_turn(orch: Orchestrator, session: Session) -> None:
    """Drive turn 1 (A1 freeze) to the Confirm node with the card referenced from state."""
    state = orch._store.require(session)  # noqa: SLF001 - test seam
    state.referenced_product_id = _CARD
    result = orch.step(session, "congela mi tarjeta")
    assert result.node == "Confirm"
    # The restatement masks the product number and names the action (REQ-14).
    assert result.response is not None
    assert _MASKED in result.response.text
    assert "congelar" in result.response.text
    assert state.pending_action is not None


# -- the lexicon (deterministic yes/no, REQ-14) ----------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("sí", Reply.AFFIRMATIVE),
        ("si, adelante", Reply.AFFIRMATIVE),
        ("claro que sí", Reply.AFFIRMATIVE),
        ("sim, pode continuar", Reply.AFFIRMATIVE),
        ("dale", Reply.AFFIRMATIVE),
        ("no", Reply.NEGATIVE),
        ("não", Reply.NEGATIVE),
        ("mejor no", Reply.NEGATIVE),
        ("cancela por favor", Reply.NEGATIVE),
        ("no estoy seguro", Reply.NEGATIVE),  # carries "no" -> fail closed to negative
        ("mmm", Reply.AMBIGUOUS),
        ("", Reply.AMBIGUOUS),
        ("qué significa congelar", Reply.AMBIGUOUS),
    ],
)
def test_classify_reply_lexicon(reply: str, expected: Reply) -> None:
    assert classify_reply(reply) is expected


# -- explicit yes -> executes and verifies (REQ-09) ------------------------------------


def test_affirmative_executes_and_verifies(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock)
    overlay = InMemoryCardOverlay()
    orch = _orchestrator(source, clock, overlay=overlay)
    _patch_intent(monkeypatch, Intent.A1, 0.99)

    _confirm_turn(orch, session)
    result = orch.step(session, "sí, adelante")

    assert result.node == "Verify"
    # The overlay was actually written and read back frozen (verified post-condition, REQ-09).
    frozen = overlay.get(_CARD)
    assert frozen is not None and frozen.frozen is True
    # Pending is consumed (single-use).
    assert orch._store.require(session).pending_action is None  # noqa: SLF001


def test_unfreeze_affirmative_executes(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock)
    overlay = InMemoryCardOverlay()
    # Seed the card as already frozen so an unfreeze has something to reverse.
    overlay.put(
        CardFreezeState(
            product_id=_CARD,
            customer_id=_CUSTOMER,
            frozen=True,
            confirmation_id="CONF-SEED",
            updated_at=_NOW,
        )
    )
    orch = _orchestrator(source, clock, overlay=overlay)
    _patch_intent(monkeypatch, Intent.A2, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = _CARD
    confirm = orch.step(session, "descongela mi tarjeta")
    assert confirm.node == "Confirm" and "descongelar" in confirm.response.text  # type: ignore[union-attr]

    orch.step(session, "sim")
    unfrozen = overlay.get(_CARD)
    assert unfrozen is not None and unfrozen.frozen is False


# -- no / ambiguous -> not executed (REQ-14) -------------------------------------------


@pytest.mark.parametrize("reply", ["no, mejor no", "mmm no sé", ""])
def test_non_affirmative_does_not_execute(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource, reply: str
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock)
    overlay = InMemoryCardOverlay()
    orch = _orchestrator(source, clock, overlay=overlay)
    _patch_intent(monkeypatch, Intent.A1, 0.99)

    _confirm_turn(orch, session)
    result = orch.step(session, reply)

    assert result.node == "Verify"  # cancelled is rendered here, not an error node
    assert overlay.get(_CARD) is None  # nothing executed
    assert orch._store.require(session).pending_action is None  # noqa: SLF001 - pending cleared


# -- expired session between request and reply -> not executed, re-auth ----------------


def test_expiry_between_request_and_reply_does_not_execute(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock)
    overlay = InMemoryCardOverlay()
    orch = _orchestrator(source, clock, overlay=overlay)
    _patch_intent(monkeypatch, Intent.A1, 0.99)

    _confirm_turn(orch, session)
    clock.now = session.expires_at + timedelta(seconds=1)  # expire before the reply

    result = orch.step(session, "sí")
    assert result.node == "Unauthenticated"  # fail closed to re-auth
    assert overlay.get(_CARD) is None  # nothing executed
    assert orch._store.peek(session.jti) is None  # noqa: SLF001 - state (and pending) discarded


# -- single-use: a second "yes" after execution is a no-op -----------------------------


def test_second_yes_after_execution_is_noop(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock)
    overlay = InMemoryCardOverlay()
    orch = _orchestrator(source, clock, overlay=overlay)
    _patch_intent(monkeypatch, Intent.A1, 0.99)

    _confirm_turn(orch, session)
    orch.step(session, "sí")  # executes
    assert overlay.get(_CARD).frozen is True  # type: ignore[union-attr]

    # No pending is open now; a replayed "sí" is just a normal (A1) turn that re-confirms, never a
    # second execution off the first affirmative. Prove the pending was consumed, not re-fired.
    state = orch._store.require(session)  # noqa: SLF001
    assert state.pending_action is None
    replay = orch.step(session, "sí")
    # With no pending, the "sí" turn is classified as A1 again -> a fresh Confirm (restate), it does
    # NOT execute off the earlier affirmative.
    assert replay.node == "Confirm"


# -- post-condition false -> not completed + escalation (REQ-09) -----------------------


class _DroppingOverlay:
    """Writes never land, so the tool's read-back fails the post-condition (REQ-09)."""

    def get(self, product_id: str) -> CardFreezeState | None:
        return None

    def put(self, state: CardFreezeState) -> None:
        return None


def test_post_condition_false_reports_not_completed_and_escalates(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock)
    orch = _orchestrator(source, clock, overlay=_DroppingOverlay())
    _patch_intent(monkeypatch, Intent.A1, 0.99)

    _confirm_turn(orch, session)
    result = orch.step(session, "sí")

    # The tool could not verify the post-condition -> not completed, offer a human (Escalate).
    assert result.node == "Escalate"
    assert result.response is not None and result.response.text.strip()
