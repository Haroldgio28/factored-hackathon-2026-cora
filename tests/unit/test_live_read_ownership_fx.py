"""Tests for the live read-ownership + FX wiring fix (REQ-04, REQ-28).

Two orchestrator gaps are closed here, both driven through the REAL `ToolLayer` over a tiny
self-built Parquet landing (never the real `data/`) with REAL verified sessions, exactly like
`test_confirmation.py` - so `customer_id` is injected from the token as in production and the
ownership checks are the real ones:

- NON-REFERENCED READS (A): a read intent with no specific product (I6 "list my products", a
  bare account-level balance read) now resolves ownership at the customer level via
  `list_products` and answers, instead of falling through to abstain.
- FX CONVERSION (B): intent I5 now calls `convert_currency` from the proposed amount+currencies
  and answers the grounded figure.

The safety invariants are pinned alongside the fix: a foreign/non-owned resource is STILL denied
(ownership not weakened into "answer anything"), and a tool outage FAILS CLOSED to the honest
tool-unavailable path, never a false "not owned".

The classifier is injected per turn by monkeypatching `Orchestrator._classify`; the FX entities
are injected by monkeypatching `Orchestrator._resolve_reference` (same seams the other graph
tests use). The LLM is `StubLLMClient`; no network, no AWS.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from cora.agent import StubLLMClient
from cora.agent.graph import Orchestrator
from cora.agent.references import Reference, ReferenceKind
from cora.agent.state import SessionStore
from cora.agent.templates import Outcome
from cora.data.datasource import LocalSource
from cora.identity import MockIdentityService, Session
from cora.nlu.entities import ExtractedEntities
from cora.policy import Decision, Intent, PolicyEngine, Thresholds
from cora.tools import ToolLayer

_KEY = "test-signing-key-not-a-real-secret"
_OWNER = "CLI-OWNER0000001"
_OTHER = "CLI-OTHER0000002"
_OWNED_CARD = "PRD-OWNERCARD01"
_FOREIGN_CARD = "PRD-OTHERCARD02"
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)
_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _build_landing(root: Path) -> None:
    # Both ids have a `customers` row so the deliverable-2 existence check (`customer_record_exists`)
    # returns `Status.OK` and these real owners take the ordinary read/FX path. Without the table
    # the existence lookup would be UNAVAILABLE and the turn would (correctly) fail closed.
    customers = pd.DataFrame(
        [
            {"customer_id": _OWNER, "full_name": "Owner One"},
            {"customer_id": _OTHER, "full_name": "Other Two"},
            # A real, registered customer who owns NO products: still has a `customers` row, so the
            # existence check is OK and I6 answers the grounded empty-list copy (not a non-customer).
            {"customer_id": "CLI-NOPRODUCTS01", "full_name": "No Products"},
        ]
    )
    products = pd.DataFrame(
        [
            {
                "product_id": _OWNED_CARD,
                "customer_id": _OWNER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "COP",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            },
            {
                "product_id": _FOREIGN_CARD,
                "customer_id": _OTHER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030860000",
                "currency": "COP",
                "current_balance": "100.00",
                "credit_limit": "1000.00",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            },
        ]
    )
    rates = pd.DataFrame(
        [
            {
                "date": "2026-06-18",
                "source_currency": "USD",
                "target_currency": "COP",
                "exchange_rate": "4000.0",
                "buy_rate": None,
                "sell_rate": None,
            }
        ]
    )
    root.mkdir(parents=True, exist_ok=True)
    customers.astype("string").to_parquet(root / "customers.parquet", index=False)
    products.astype("string").to_parquet(root / "products.parquet", index=False)
    rates.astype("string").to_parquet(root / "daily_exchange_rates.parquet", index=False)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("live_read_raw_parquet")
    _build_landing(root)
    return root


@pytest.fixture
def source(landing: Path) -> LocalSource:
    return LocalSource(root=landing)


def _issue_session(clock: _Clock, customer_id: str) -> Session:
    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)
    challenge = service.begin_authentication(customer_id)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


def _orchestrator(source: LocalSource, clock: _Clock) -> Orchestrator:
    def _factory(session: Session) -> ToolLayer:
        return ToolLayer(session, source, clock=clock)

    return Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=_factory,
        llm=StubLLMClient(),
        clock=clock,
    )


def _patch_intent(monkeypatch: pytest.MonkeyPatch, intent: Intent, confidence: float = 0.99) -> None:
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (intent, confidence))


def _patch_entities(monkeypatch: pytest.MonkeyPatch, entities: ExtractedEntities | None) -> None:
    """Inject the (data-only) extracted entities a turn would carry, with no referential phrase."""
    ref = Reference(kind=ReferenceKind.NONE)
    monkeypatch.setattr(Orchestrator, "_resolve_reference", lambda self, state, masked: (ref, entities))


# -- (A) non-referenced reads: I6 list products ----------------------------------------


def test_list_products_answers_for_owner(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I6)
    _patch_entities(monkeypatch, None)  # "list my products" carries no product reference

    result = orch.step(session, "¿qué productos tengo?")
    # Ownership resolved at the customer level via list_products -> POL-090 answers.
    assert result.decision is Decision.ANSWER
    assert result.rule_id == "POL-090"
    assert result.node == "Answer"
    # The customer-visible answer is non-empty and grounded in the product the fixture owns: the
    # masked card tail (from product_number 4717188030863827) and the product status appear, and
    # no figure outside the tool result leaked (grounding gate ran against the same tool_results).
    assert result.response is not None
    text = result.response.text
    assert text.strip()
    assert "3827" in text  # masked tail of the owned card number
    assert result.response.outcome is Outcome.ANSWER


def test_bare_account_read_answers_balance_for_owner(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    # A bare account-level balance read (I1) with no resolved product reference. The owner has
    # exactly one product, so ownership resolves to it and its REAL balance is read and shown.
    _patch_intent(monkeypatch, Intent.I1)
    _patch_entities(monkeypatch, None)

    result = orch.step(session, "mi saldo")
    assert result.decision is Decision.ANSWER
    assert result.rule_id == "POL-090"
    # The grounded balance figure (952.03 COP from the fixture product) is in the answer, proving
    # the turn fetched the balance (not merely "has a product") and rendered a source-backed value.
    assert result.response is not None
    assert "952.03" in result.response.text
    assert "COP" in result.response.text


def test_list_products_answers_empty_list_for_customer_without_products(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource
) -> None:
    clock = _Clock(_NOW)
    # A customer the landing has no products for: list_products is OK but empty. An OK empty list
    # is still a grounded I6 answer ("you have no products"), not an ownership failure.
    session = _issue_session(clock, "CLI-NOPRODUCTS01")
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I6)
    _patch_entities(monkeypatch, None)

    result = orch.step(session, "¿qué productos tengo?")
    # OK empty list -> POL-090 answers with the explicit empty-list copy (grounded, no figure).
    assert result.decision is Decision.ANSWER
    assert result.rule_id == "POL-090"
    assert result.node == "Answer"
    assert result.response is not None
    assert result.response.outcome is Outcome.ANSWER
    # The bilingual empty-list copy appears (es default here); no product tail is invented.
    assert "No tienes productos" in result.response.text


# -- adversarial polish: a factual answer is template-only, so a tampered rephrase cannot leak --


class _CurrencySwapLLM(StubLLMClient):
    """A hostile polish client: it returns a rephrase that KEEPS every figure but swaps the
    currency code (COP -> USD). The grounding gate only validates numbers/dates/statuses, so this
    tampering would pass grounding; the only thing that stops it reaching the customer is that a
    factual ANSWER is rendered template-only (`polish=False`). If the orchestrator ever polished a
    factual answer again, `complete` would be called and its tampered text would surface - this
    test fails in that regression.
    """

    def __init__(self) -> None:
        super().__init__(default="")
        self.called = False

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.called = True
        return "Tu saldo es 952.03 USD."  # same number, wrong currency


def test_factual_answer_is_template_only_so_currency_cannot_be_tampered(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    hostile = _CurrencySwapLLM()

    def _factory(sess: Session) -> ToolLayer:
        return ToolLayer(sess, source, clock=clock)

    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=_factory,
        llm=hostile,
        clock=clock,
    )
    _patch_intent(monkeypatch, Intent.I1)
    _patch_entities(monkeypatch, None)  # bare balance read -> owner's single COP product

    result = orch.step(session, "mi saldo")
    assert result.decision is Decision.ANSWER
    assert result.response is not None
    text = result.response.text
    # The real currency (COP) from the tool result is shown; the swapped USD rephrase never leaks.
    assert "COP" in text
    assert "USD" not in text
    assert not result.response.polished  # factual answer rendered template-only, no LLM polish
    # The hostile polish client was never invoked for the factual answer (polish disabled).
    assert hostile.called is False


# -- (B) FX conversion: I5 convert_currency --------------------------------------------


def test_fx_conversion_answers_with_grounded_figure(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I5)
    _patch_entities(
        monkeypatch,
        ExtractedEntities(amount=100, currency="USD", to_currency="COP"),
    )

    result = orch.step(session, "conviérteme 100 dólares a pesos")
    # convert_currency served the latest-prior rate within 7 days -> POL-090 answers.
    assert result.decision is Decision.ANSWER
    assert result.rule_id == "POL-090"
    assert result.node == "Answer"
    # The grounded converted figure is shown: 100 USD * 4000 (the only fixture rate) = 400000 COP.
    # Grounding ran against the convert_currency Result, so these figures are source-backed.
    assert result.response is not None
    text = result.response.text
    assert "400000" in text
    assert "4000" in text  # the rate itself
    assert "COP" in text and "USD" in text
    # REQ-28/ADR-016: no rate existed for the requested date (2026-06-20); the latest prior rate
    # (2026-06-18, within 7 days) was substituted, so the answer MUST state it to the customer.
    assert "anterior" in text.lower()  # es "tasa anterior" / pt "taxa anterior"


def test_fx_without_entities_clarifies(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I5)
    _patch_entities(monkeypatch, None)  # no amount/currencies -> cannot invent values

    result = orch.step(session, "quiero convertir plata")
    # Incomplete FX entities carry no amount+currencies to convert: the turn marks the entity
    # ambiguous so POL-070 asks one focused question (clarify), never guessing a rate (REQ-28).
    assert result.decision is Decision.CLARIFY
    assert result.rule_id == "POL-070"
    assert result.response is not None and result.response.text.strip()


def test_fx_ambiguous_target_currency_clarifies(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I5)
    # An unqualified "pesos" target is ambiguous across MXN/COP/ARS. The extraction prompt tells
    # the model to return null for it, so `to_currency` is None here: deterministic code (never a
    # guessed code) then decides there is nothing unambiguous to convert and clarifies (POL-070).
    _patch_entities(
        monkeypatch,
        ExtractedEntities(amount=100, currency="USD", to_currency=None),
    )

    result = orch.step(session, "conviérteme 100 dólares a pesos")
    assert result.decision is Decision.CLARIFY
    assert result.rule_id == "POL-070"
    assert result.response is not None and result.response.text.strip()


def test_fx_stale_rate_fails_closed(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    # Move the clock far past the only rate date (2026-06-18) so the latest-prior rate is >7 days
    # stale: the tool abstains (UNAVAILABLE) and the turn surfaces the honest tool-unavailable copy.
    clock = _Clock(datetime(2026, 9, 1, 12, 0, tzinfo=UTC))
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I5)
    _patch_entities(
        monkeypatch,
        ExtractedEntities(amount=100, currency="USD", to_currency="COP"),
    )

    result = orch.step(session, "conviérteme 100 dólares a pesos")
    # No fresh rate -> not owned -> abstain, and the honest tool-unavailable copy is rendered
    # (never a guessed rate). The rendered outcome proves it is the TOOL_UNAVAILABLE template.
    assert result.decision is Decision.ABSTAIN
    assert result.response is not None
    assert result.response.outcome is Outcome.TOOL_UNAVAILABLE


# -- ownership NOT weakened: a foreign resource is still denied ------------------------


def test_foreign_product_read_is_still_denied(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I1)
    # A read explicitly scoped to a product the OWNER does not own (belongs to _OTHER).
    ref = Reference(kind=ReferenceKind.PRODUCT, product_id=_FOREIGN_CARD)
    monkeypatch.setattr(Orchestrator, "_resolve_reference", lambda self, state, masked: (ref, None))

    result = orch.step(session, "cuál es el saldo de esa cuenta")
    # FORBIDDEN read -> resource not owned -> abstain (ownership check intact).
    assert result.decision is Decision.ABSTAIN
    assert result.node == "Abstain"


# -- bare I1 partial outage: list OK but balance UNAVAILABLE must fail closed ----------


class _BalanceOutageSource:
    """Wraps a real source but makes only the get_balance read fail (list_products stays OK).

    `list_products` filters by `customer_id`, while `get_balance` loads the product by
    `product_id` (`_load_owned_product`). Raising on a `product_id = ...` lookup simulates a
    balance-read outage while `list_products` still succeeds - the exact list-OK /
    balance-UNAVAILABLE sequence a bare I1 can hit.
    """

    def __init__(self, inner: LocalSource) -> None:
        self._inner = inner

    def fetch_df(self, table: str, *, where=None, **kwargs):  # noqa: ANN001, ANN201
        if where is not None and "product_id =" in where:
            raise RuntimeError("simulated balance-read outage")
        return self._inner.fetch_df(table, where=where, **kwargs)

    def scan(self, *args: object, **kwargs: object):  # noqa: ANN201
        return self._inner.scan(*args, **kwargs)


def test_bare_i1_partial_outage_fails_closed(monkeypatch: pytest.MonkeyPatch, source: LocalSource) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)

    def _factory(sess: Session) -> ToolLayer:
        return ToolLayer(sess, _BalanceOutageSource(source), clock=clock)  # type: ignore[arg-type]

    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=_factory,
        llm=StubLLMClient(),
        clock=clock,
    )
    _patch_intent(monkeypatch, Intent.I1)
    _patch_entities(monkeypatch, None)  # bare account-level balance read, no product reference

    result = orch.step(session, "mi saldo")
    # list_products succeeded but the required balance read is UNAVAILABLE: ownership for a bare
    # I1 is gated on the balance read, so resource_owned stays False (no false product-list
    # answer) and the turn fails closed to the honest tool-unavailable copy - never a guessed or
    # unrelated figure.
    assert result.response is not None and result.response.outcome is Outcome.TOOL_UNAVAILABLE
    assert result.decision in {Decision.ABSTAIN, Decision.ESCALATE}
    # The owned card tail must NOT leak as a substitute "answer" for the balance that failed.
    assert "3827" not in result.response.text


# -- non-referenced I2-I4 have no account-level renderer: they must abstain ------------


@pytest.mark.parametrize("intent", [Intent.I2, Intent.I3, Intent.I4])
def test_non_referenced_i2_i4_abstain_not_product_list(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource, intent: Intent
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, intent)
    _patch_entities(monkeypatch, None)  # no product reference

    result = orch.step(session, "algo sobre mi cuenta")
    # I2-I4 have no account-level tool/renderer wired: a non-referenced one must NOT be answered
    # with an unrelated product list. It stays not-owned and fails closed to abstain.
    assert result.decision is Decision.ABSTAIN
    assert result.node == "Abstain"


# -- referenced I2-I4 have no renderer either: they must NOT return a balance answer ---


@pytest.mark.parametrize("intent", [Intent.I2, Intent.I3, Intent.I4])
def test_referenced_i2_i4_abstain_not_balance_answer(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource, intent: Intent
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, intent)
    # The turn references the owner's REAL product. I2/I3 (transactions) and I4 (card status)
    # have no renderer wired, so even a referenced, owned one must NOT be answered with the
    # (unrelated) balance figure - it stays not-owned and fails closed to abstain.
    ref = Reference(kind=ReferenceKind.PRODUCT, product_id=_OWNED_CARD)
    monkeypatch.setattr(Orchestrator, "_resolve_reference", lambda self, state, masked: (ref, None))

    result = orch.step(session, "y sobre esa tarjeta")
    assert result.decision is Decision.ABSTAIN
    assert result.node == "Abstain"
    # The owned card's balance must never surface for a transaction/card-status request.
    if result.response is not None:
        assert "952.03" not in result.response.text


def test_referenced_i6_answers_product_list_not_balance(
    monkeypatch: pytest.MonkeyPatch, source: LocalSource
) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)
    orch = _orchestrator(source, clock)
    _patch_intent(monkeypatch, Intent.I6)
    # Even with a resolved product reference in state, I6 ("list my products") is a customer-level
    # question and must be served by list_products, never redirected to that product's balance.
    ref = Reference(kind=ReferenceKind.PRODUCT, product_id=_OWNED_CARD)
    monkeypatch.setattr(Orchestrator, "_resolve_reference", lambda self, state, masked: (ref, None))

    result = orch.step(session, "¿qué productos tengo con esa tarjeta?")
    assert result.decision is Decision.ANSWER
    assert result.rule_id == "POL-090"
    assert result.node == "Answer"
    assert result.response is not None
    text = result.response.text
    # The grounded product list (masked card tail) is shown, not the balance figure.
    assert "3827" in text
    assert "952.03" not in text


# -- tool outage fails closed to tool-unavailable, not a false deny --------------------


class _RaisingSource:
    """A DataSource whose reads always raise, to simulate a transport outage (REQ-40)."""

    def fetch_df(self, *args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("simulated remote outage")

    def scan(self, *args: object, **kwargs: object):  # noqa: ANN201 - never returns
        raise RuntimeError("simulated remote outage")


def test_tool_outage_fails_closed_to_tool_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(_NOW)
    session = _issue_session(clock, _OWNER)

    def _factory(sess: Session) -> ToolLayer:
        return ToolLayer(sess, _RaisingSource(), clock=clock)  # type: ignore[arg-type]

    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=_factory,
        llm=StubLLMClient(),
        clock=clock,
    )
    _patch_intent(monkeypatch, Intent.I6)
    _patch_entities(monkeypatch, None)

    result = orch.step(session, "¿qué productos tengo?")
    # list_products raised through _safe_read -> UNAVAILABLE: the turn renders the honest
    # tool-unavailable copy and never claims the customer owns nothing.
    assert result.response is not None and result.response.text.strip()
    # The decision falls through to abstain (no owned resource proven), but the customer-facing
    # text is the fail-closed TOOL_UNAVAILABLE template - NOT the generic abstain/"not owned" copy
    # and not a guessed value. Asserting the rendered outcome pins the fail-closed path exactly.
    assert result.decision in {Decision.ABSTAIN, Decision.ESCALATE}
    assert result.response.outcome is Outcome.TOOL_UNAVAILABLE
