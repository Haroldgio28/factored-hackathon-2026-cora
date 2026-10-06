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
from cora.tools.models import BalanceData, CardDetailsData, ProductsData, ProductSummary

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CUST-000123"
# REQ-07 pins tau_clarify at 0.50; inject it so the test does not depend on the file's default.
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _FakeSource:
    """Minimal `DataSource` stand-in: answers only the `customers` existence `count` 4.1 asks.

    `count` returns the seeded customer-record count so the deliverable-2 non-customer branch can
    tell a known id (>=1) from an unknown one (0) without a real Parquet landing.
    """

    def __init__(self, customer_record_count: int) -> None:
        self._count = customer_record_count

    def count(self, table, *, where=None, **_kwargs):  # noqa: ANN001, ANN003 - duck-typed test double
        assert table == "customers"
        return self._count


class _FakeToolLayer:
    """Duck-typed stand-in for `ToolLayer`: answers only the two ownership questions 4.1 asks.

    `owned` is the set of product ids the session customer owns; `status_by_id` overrides the
    card status so the POL-080 "state allows" branch can be exercised. Anything else fails closed
    with FORBIDDEN, mirroring the real layer.
    """

    def __init__(
        self,
        owned: set[str],
        status_by_id: dict[str, str] | None = None,
        *,
        customer_id: str = _CUSTOMER,
        customer_record_count: int = 1,
    ) -> None:
        self._owned = owned
        self._status = status_by_id or {}
        # The Handoff node (task 4.5) persists the package here; mirror the real `ToolLayer`.
        self.handoff_store = InMemoryHandoffStore()
        # The non-customer branch (deliverable 2) reads `customer_id` + `source.count("customers")`.
        # These doubles default to a customer whose record EXISTS, so every existing edge test runs
        # the ordinary classify/policy path (never the non-customer greet).
        self.customer_id = customer_id
        self.source = _FakeSource(customer_record_count)

    def customer_record_exists(self):  # noqa: ANN202 - duck-typed test double
        # Mirror the real `ToolLayer.customer_record_exists`: OK when the id has >=1 `customers`
        # row, NOT_FOUND on zero rows, UNAVAILABLE when the lookup raises (fail closed).
        try:
            count = self.source.count("customers", where=f"customer_id = '{self.customer_id}'")
        except Exception:  # noqa: BLE001 - any lookup error fails closed to UNAVAILABLE
            return Result[ProductsData](status=Status.UNAVAILABLE, message="tool unavailable")
        status = Status.OK if count > 0 else Status.NOT_FOUND
        return Result[ProductsData](status=status)

    def list_products(self, tool_input):  # noqa: ANN001 - duck-typed test double
        # Read by `_build_policy_input` for I1/I6: `owned` seeds the session's products.
        products = [
            ProductSummary(
                product_id=product_id,
                product_type="Tarjeta Crédito",
                product_number_masked="****1234",
                currency="USD",
                product_status=self._status.get(product_id, "Active"),
            )
            for product_id in sorted(self._owned)
        ]
        return Result[ProductsData](status=Status.OK, data=ProductsData(products=products))

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
    # The session IS a real customer (owns PRD-1) but references a FOREIGN product it does not own,
    # so the non-customer branch (deliverable 2) does not fire and the foreign read is a correct
    # deny - distinct from an identity that owns nothing (that is the non-customer path, tested
    # separately below).
    orch = _orchestrator(clock, owned={"PRD-1"})
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


# -- non-customer: greet + route to a human (deliverable 2, REQ-16) --------------------
#
# These exercise the deterministic non-customer branch against the REAL `ToolLayer` over a tiny
# Parquet landing (same pattern as `test_confirmation.py`). Existence is a `customers`-record
# lookup (`ToolLayer.customer_record_exists`) scoped to the verified `customer_id`, read through
# `_safe_read`: a KNOWN customer HAS a `customers` row (`Status.OK`) and takes the normal path -
# even one who owns zero products - while a wholly unknown id (`Status.NOT_FOUND`) is routed to a
# human with no account data disclosed. A lookup in DOUBT (`Status.UNAVAILABLE`) fails closed.

import pandas as pd  # noqa: E402 - test-local import kept next to the fixture that uses it

from cora.data.datasource import LocalSource  # noqa: E402
from cora.tools import ToolLayer  # noqa: E402
from cora.tools.models import HandoffReason  # noqa: E402

_NONCUSTOMER_KEY = "test-signing-key-not-a-real-secret"
# (a) a known customer WITH a customers row AND a product.
_KNOWN_CUSTOMER = "CLI-KNOWN0000001"
_KNOWN_CARD = "PRD-KNOWNCARD01"
# (b) a known customer WITH a customers row but ZERO products -> still a real customer.
_NOPRODUCTS_CUSTOMER = "CLI-NOPRODUCTS01"
# (c) an unknown id with NO customers row -> the non-customer greet + handoff.
_UNKNOWN_CUSTOMER = "CLI-UNKNOWN00001"


def _build_noncustomer_landing(root) -> None:  # noqa: ANN001 - a Path (pytest tmp)
    # The `customers` table drives the deterministic existence check: both known ids have a row;
    # the unknown id has none. Only the first known id owns a product, proving "has a record" is
    # independent of "owns a product".
    customers = pd.DataFrame(
        [
            {"customer_id": _KNOWN_CUSTOMER, "full_name": "Known One"},
            {"customer_id": _NOPRODUCTS_CUSTOMER, "full_name": "Known Two"},
        ]
    )
    products = pd.DataFrame(
        [
            {
                "product_id": _KNOWN_CARD,
                "customer_id": _KNOWN_CUSTOMER,
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
    # A valid USD->COP rate dated on the turn clock. This exists so the FX fail-closed regression
    # (`test_noncustomer_check_unavailable_blocks_independent_fx`) can prove that a tool which
    # WOULD succeed on its own (convert_currency) still discloses nothing once the customer-record
    # existence check is in DOUBT.
    rates = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "source_currency": "USD",
                "target_currency": "COP",
                "exchange_rate": "4000.0",
            }
        ]
    )
    root.mkdir(parents=True, exist_ok=True)
    customers.astype("string").to_parquet(root / "customers.parquet", index=False)
    products.astype("string").to_parquet(root / "products.parquet", index=False)
    rates.astype("string").to_parquet(root / "daily_exchange_rates.parquet", index=False)


@pytest.fixture(scope="module")
def _noncustomer_landing(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN202 - a Path
    root = tmp_path_factory.mktemp("noncustomer_landing")
    _build_noncustomer_landing(root)
    return root


def _noncustomer_source(_noncustomer_landing) -> LocalSource:  # noqa: ANN001 - a Path fixture
    return LocalSource(root=_noncustomer_landing)


def _real_orchestrator(
    source,  # noqa: ANN001 - a DataSource
    clock: _Clock,
    *,
    tool_layer_factory=None,  # noqa: ANN001 - optional override for the fail-closed test
) -> Orchestrator:
    store = InMemoryHandoffStore()

    def _default_factory(session: Session) -> ToolLayer:
        return ToolLayer(session, source, handoff_store=store, clock=clock)

    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=tool_layer_factory or _default_factory,
        llm=StubLLMClient(),
        clock=clock,
    )
    orch._handoff_store = store  # noqa: SLF001 - test seam to read the persisted case back
    return orch


def _issue_real_session(clock: _Clock, customer_id: str) -> Session:
    service = MockIdentityService(
        signing_key=_NONCUSTOMER_KEY, session_ttl=timedelta(minutes=15), clock=clock
    )
    challenge = service.begin_authentication(customer_id)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


def test_noncustomer_first_turn_greets_and_handsoff(_noncustomer_landing) -> None:  # noqa: ANN001
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    orch = _real_orchestrator(_noncustomer_source(_noncustomer_landing), clock)
    session = _issue_real_session(clock, _UNKNOWN_CUSTOMER)

    result = orch.step(session, "hola, necesito ayuda")

    assert result.node == "Handoff"
    assert result.handoff_case_id  # the case reaches the console with an id
    assert result.handoff_package is not None
    assert result.handoff_package.reason is HandoffReason.NON_CUSTOMER
    assert result.handoff_package.priority == "normal"
    assert result.response is not None
    # The es greeting welcomes the person and routes them to a human (no figure).
    text = result.response.text.lower()
    assert "cora" in text
    assert "cliente" in text and "persona" in text


def test_noncustomer_greeting_in_portuguese(_noncustomer_landing) -> None:  # noqa: ANN001
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    orch = _real_orchestrator(_noncustomer_source(_noncustomer_landing), clock)
    session = _issue_real_session(clock, _UNKNOWN_CUSTOMER)

    # Explicit pt choice drives the greeting language (deterministic, no detector).
    result = orch.step(session, "ola, preciso de ajuda", language="pt")

    assert result.node == "Handoff"
    assert result.language == "pt"
    assert result.response is not None
    text = result.response.text.lower()
    assert "cora" in text
    assert "cliente" in text and "pessoa" in text


def test_noncustomer_turn_discloses_no_account_data(_noncustomer_landing) -> None:  # noqa: ANN001
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    orch = _real_orchestrator(_noncustomer_source(_noncustomer_landing), clock)
    session = _issue_real_session(clock, _UNKNOWN_CUSTOMER)

    result = orch.step(session, "cual es mi saldo")

    assert result.handoff_package is not None
    # No account fact can leak: the package is built from NO tool results.
    assert result.handoff_package.verified_facts == []
    assert result.handoff_package.supporting_transactions == []
    # The greeting carries no digit (no balance/limit/figure).
    assert result.response is not None
    assert not any(ch.isdigit() for ch in result.response.text)


def test_noncustomer_handoff_reaches_console_store(_noncustomer_landing) -> None:  # noqa: ANN001
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    orch = _real_orchestrator(_noncustomer_source(_noncustomer_landing), clock)
    session = _issue_real_session(clock, _UNKNOWN_CUSTOMER)

    result = orch.step(session, "hola")

    stored = orch._handoff_store.cases[result.handoff_case_id]  # noqa: SLF001 - test seam
    assert stored["reason"] == "E5"
    assert stored["priority"] == "normal"
    assert stored["verified_facts"] == []
    # Shared handoff path (review finding 2): the persisted package carries THIS turn's trace id,
    # so a human agent can correlate the case to the trace through the normal REQ-16 contract.
    assert stored["trace_ref"] == result.trace_id
    assert result.trace_id


def test_known_customer_not_treated_as_noncustomer(
    monkeypatch: pytest.MonkeyPatch, _noncustomer_landing
) -> None:  # noqa: ANN001
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    orch = _real_orchestrator(_noncustomer_source(_noncustomer_landing), clock)
    session = _issue_real_session(clock, _KNOWN_CUSTOMER)
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = _KNOWN_CARD
    result = orch.step(session, "cual es mi saldo")

    # A real customer runs the ordinary read path and is answered - the non-customer branch did
    # not fire, so ownership for real customers is unchanged.
    assert result.node == "Answer"
    assert result.decision is Decision.ANSWER


def test_known_customer_with_zero_products_is_not_a_noncustomer(
    monkeypatch: pytest.MonkeyPatch, _noncustomer_landing
) -> None:  # noqa: ANN001
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    orch = _real_orchestrator(_noncustomer_source(_noncustomer_landing), clock)
    # Case (b): a real customer WITH a `customers` row but ZERO products. The existence check is a
    # customer-record lookup, NOT a product count, so this id is NOT a non-customer: the normal I6
    # path answers the grounded "no products" copy (behaviour preserved for real customers).
    session = _issue_real_session(clock, _NOPRODUCTS_CUSTOMER)
    _patch_intent(monkeypatch, Intent.I6, 0.99)

    result = orch.step(session, "¿qué productos tengo?")

    assert result.node == "Answer"
    assert result.decision is Decision.ANSWER
    assert result.response is not None
    assert "No tienes productos" in result.response.text


def _existence_outage_factory(source: LocalSource, clock: _Clock):  # noqa: ANN202 - a factory
    """A tool-layer factory whose `customers` existence `count` RAISES but every other read works.

    Only the existence lookup is broken, so a downstream tool (e.g. `convert_currency`) can still
    succeed on its own - exactly the shape that must NOT be allowed to disclose a figure once the
    customer-record check is in DOUBT.
    """

    def _factory(session: Session) -> ToolLayer:
        layer = ToolLayer(session, source, handoff_store=InMemoryHandoffStore(), clock=clock)

        class _BoomSource:
            def count(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202 - existence outage
                raise RuntimeError("customers backend down")

            def __getattr__(self, name):  # noqa: ANN001, ANN204 - delegate every other read
                return getattr(source, name)

        layer.source = _BoomSource()  # type: ignore[assignment]
        return layer

    return _factory


def test_noncustomer_check_fails_closed_on_lookup_error(
    monkeypatch: pytest.MonkeyPatch, _noncustomer_landing
) -> None:  # noqa: ANN001
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    source = _noncustomer_source(_noncustomer_landing)
    orch = _real_orchestrator(source, clock, tool_layer_factory=_existence_outage_factory(source, clock))
    # An UNKNOWN id, but the existence lookup is UNAVAILABLE (raises): the identity is in DOUBT, so
    # the turn must NOT be read as "not a client" (no non-customer handoff) AND must NOT answer -
    # it fails closed to the honest tool-unavailable copy (no figure).
    session = _issue_real_session(clock, _UNKNOWN_CUSTOMER)
    _patch_intent(monkeypatch, Intent.I1, 0.99)

    result = orch.step(session, "cual es mi saldo")

    # A lookup outage must never be read as "not a client": no non-customer handoff.
    assert not (
        result.node == "Handoff"
        and result.handoff_package is not None
        and result.handoff_package.reason is HandoffReason.NON_CUSTOMER
    )
    # It also never answers with a figure: identity doubt fails the turn closed.
    assert result.node != "Answer"
    assert result.decision is not Decision.ANSWER
    assert result.response is not None
    assert not any(ch.isdigit() for ch in result.response.text)


def test_noncustomer_check_unavailable_blocks_independent_fx(
    monkeypatch: pytest.MonkeyPatch, _noncustomer_landing
) -> None:  # noqa: ANN001
    # Regression (review finding 1): a tool that WOULD succeed independently of the customer-record
    # lookup (I5 FX conversion, backed by a valid rate in the landing) must still disclose NOTHING
    # when the existence check is in DOUBT. Identity uncertainty fails the whole turn closed.
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    source = _noncustomer_source(_noncustomer_landing)
    orch = _real_orchestrator(source, clock, tool_layer_factory=_existence_outage_factory(source, clock))
    session = _issue_real_session(clock, _KNOWN_CUSTOMER)
    _patch_intent(monkeypatch, Intent.I5, 0.99)

    # Inject the FX entities the I5 path reads (LLM-proposed data): convert 100 USD -> COP. The
    # rate dated on the clock exists, so `convert_currency` returns OK on its own.
    from cora.agent.references import Reference, ReferenceKind  # noqa: PLC0415 - test-local
    from cora.nlu.entities import ExtractedEntities  # noqa: PLC0415

    entities = ExtractedEntities(amount=100, currency="USD", to_currency="COP")
    monkeypatch.setattr(
        Orchestrator,
        "_resolve_reference",
        lambda self, state, masked: (Reference(kind=ReferenceKind.NONE), entities),
    )

    # Sanity: the FX tool really does succeed on its own with these inputs (so the block below is
    # meaningful, not a vacuous pass because FX was unavailable anyway).
    from cora.tools.models import ConvertCurrencyInput  # noqa: PLC0415

    probe = ToolLayer(session, source, handoff_store=InMemoryHandoffStore(), clock=clock)
    fx = probe.convert_currency(
        ConvertCurrencyInput(amount=100, from_currency="USD", to_currency="COP", on_date=clock.now.date())
    )
    assert fx.status is Status.OK

    result = orch.step(session, "cuanto son 100 dolares en pesos")

    # Despite FX being answerable, the turn fails closed: no ANSWER, no figure leaks.
    assert result.node != "Answer"
    assert result.decision is not Decision.ANSWER
    assert result.response is not None
    assert not any(ch.isdigit() for ch in result.response.text)
