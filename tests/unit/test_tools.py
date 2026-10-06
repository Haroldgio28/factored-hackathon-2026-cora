"""Tests for task 2.2: the tool layer with per-customer authorization (REQ-12/28/34/38).

The suite runs against a tiny, self-built Parquet landing (never the real `data/`): it mirrors
the raw landing's all-VARCHAR shape so the tools' `TRY_CAST` typing path is exercised, and it
makes the FX rate-date fallback and the 7-day abstain deterministic. Sessions are REAL verified
sessions issued by the task-2.1 `MockIdentityService`, so `customer_id` is injected exactly as
in production - no tool is ever handed a `customer_id` directly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from cora.data.datasource import LocalSource
from cora.identity import MockIdentityService, Session
from cora.tools import (
    TOOL_NAMES,
    ConvertCurrencyInput,
    CreateHandoffInput,
    GetBalanceInput,
    GetCardDetailsInput,
    HandoffReason,
    InMemoryHandoffStore,
    ListProductsInput,
    SearchTransactionsInput,
    Status,
    ToolLayer,
    mask_product_number,
)
from cora.tools.models import CardFreezeInput

OWNER = "CLI-OWNER0000001"
OTHER = "CLI-OTHER0000002"

OWNER_CARD = "PRD-OWNERCARD01"
OWNER_SAVINGS = "PRD-OWNERSAV001"
OWNER_LOAN = "PRD-OWNERLOAN01"
OWNER_OVERLIMIT_CARD = "PRD-OWNEROVER01"
OTHER_CARD = "PRD-OTHERCARD1"


# -- fixture landing -------------------------------------------------------------------


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mirror the real landing: every column is VARCHAR. Casting to the pandas "string" dtype
    # keeps nulls as proper string nulls so DuckDB infers one consistent schema across the
    # Hive-partitioned fact files (an all-None object column would otherwise land as a null
    # type and clash with a sibling partition's VARCHAR).
    df.astype("string").to_parquet(path, index=False)


def _build_landing(root: Path) -> None:
    # products (all columns VARCHAR, like the real landing).
    products = pd.DataFrame(
        [
            {
                "product_id": OWNER_CARD,
                "customer_id": OWNER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            },
            {
                "product_id": OWNER_SAVINGS,
                "customer_id": OWNER,
                "product_type": "Cuenta Ahorro",
                "product_number": "7887710499",
                "currency": "USD",
                "current_balance": "60.44",
                "credit_limit": None,
                "days_past_due": None,
                "expiration_date": None,
                "product_status": "Active",
                "last_updated": "2026-06-14 09:00:00",
            },
            {
                # Loan: carries a credit_limit (original principal), but it is NOT revolving
                # credit, so get_balance must NOT derive available_credit for it (B1).
                "product_id": OWNER_LOAN,
                "customer_id": OWNER,
                "product_type": "Préstamo Personal",
                "product_number": "9900112233",
                "currency": "USD",
                "current_balance": "5000.0",
                "credit_limit": "20000.0",
                "days_past_due": "0.0",
                "expiration_date": None,
                "product_status": "Active",
                "last_updated": "2026-06-13 09:00:00",
            },
            {
                # Over-limit revolving card: current_balance > credit_limit, so the derived
                # available_credit is negative and is surfaced as-is (B1).
                "product_id": OWNER_OVERLIMIT_CARD,
                "customer_id": OWNER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030869999",
                "currency": "USD",
                "current_balance": "6000.0",
                "credit_limit": "5000.0",
                "days_past_due": "15.0",
                "expiration_date": "2027-05-01",
                "product_status": "Active",
                "last_updated": "2026-06-12 09:00:00",
            },
            {
                "product_id": OTHER_CARD,
                "customer_id": OTHER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4000000000000002",
                "currency": "COP",
                "current_balance": "100.0",
                "credit_limit": "5000.0",
                "days_past_due": "0.0",
                "expiration_date": "2027-01-01",
                "product_status": "Active",
                "last_updated": "2026-06-10 08:00:00",
            },
        ]
    )
    _write(root / "products.parquet", products)

    # customers: both OWNER and OTHER are current customers (have a record). The tool layer gates
    # every account read/action on this table (an id absent from `customers` owns no resources,
    # REQ-12 fail-closed), so a landing that exercises reads must register the ids it reads for.
    customers = pd.DataFrame(
        [
            {"customer_id": OWNER, "full_name": "Owner One"},
            {"customer_id": OTHER, "full_name": "Other Two"},
        ]
    )
    _write(root / "customers.parquet", customers)

    # transactions (Hive-partitioned facts; year/month/day come from the path).
    tx_cols = [
        "transaction_id",
        "transaction_date",
        "process_date",
        "product_id",
        "customer_id",
        "amount",
        "currency",
        "transaction_type",
        "transaction_status",
        "merchant_name",
        "transaction_category",
    ]

    def _tx(root: Path, day: int, rows: list[dict[str, object]]) -> None:
        df = pd.DataFrame(rows, columns=tx_cols)
        part = root / "transactions" / "year=2026" / "month=06" / f"day={day:02d}"
        _write(part / f"transactions_202606{day:02d}.parquet", df)

    _tx(
        root,
        3,
        [
            {
                "transaction_id": "TRX-OWNER0003",
                "transaction_date": "2026-06-03 05:48:32",
                "process_date": "2026-06-03",
                "product_id": OWNER_SAVINGS,
                "customer_id": OWNER,
                "amount": "6600.04",
                "currency": "USD",
                "transaction_type": "Transfer",
                "transaction_status": "Approved",
                "merchant_name": None,
                "transaction_category": None,
            }
        ],
    )
    _tx(
        root,
        8,
        [
            {
                "transaction_id": "TRX-OWNER0001",
                "transaction_date": "2026-06-08 21:31:05",
                "process_date": "2026-06-08",
                "product_id": OWNER_CARD,
                "customer_id": OWNER,
                "amount": "267.32",
                "currency": "USD",
                "transaction_type": "Purchase",
                "transaction_status": "Approved",
                "merchant_name": "Tienda General",
                "transaction_category": "Food",
            }
        ],
    )
    _tx(
        root,
        9,
        [
            {
                "transaction_id": "TRX-OWNER0002",
                "transaction_date": "2026-06-09 05:46:31",
                "process_date": "2026-06-09",
                "product_id": OWNER_CARD,
                "customer_id": OWNER,
                "amount": "430.37",
                "currency": "USD",
                "transaction_type": "Purchase",
                "transaction_status": "Approved",
                "merchant_name": "Internet Plus",
                "transaction_category": None,
            }
        ],
    )
    _tx(
        root,
        5,
        [
            {
                "transaction_id": "TRX-OTHER0001",
                "transaction_date": "2026-06-05 12:00:00",
                "process_date": "2026-06-05",
                "product_id": OTHER_CARD,
                "customer_id": OTHER,
                "amount": "50.0",
                "currency": "COP",
                "transaction_type": "Purchase",
                "transaction_status": "Approved",
                "merchant_name": "Foreign Shop",
                "transaction_category": None,
            }
        ],
    )

    # daily_exchange_rates: USD->COP present on 2026-06-01 and 2026-06-10 only.
    rates = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "source_currency": "USD",
                "target_currency": "COP",
                "exchange_rate": "3900.0",
                "buy_rate": None,
                "sell_rate": None,
                "source": "fixture",
            },
            {
                "date": "2026-06-10",
                "source_currency": "USD",
                "target_currency": "COP",
                "exchange_rate": "3954.98",
                "buy_rate": None,
                "sell_rate": None,
                "source": "fixture",
            },
        ]
    )
    _write(root / "daily_exchange_rates.parquet", rates)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("raw_parquet")
    _build_landing(root)
    return root


@pytest.fixture
def source(landing: Path) -> LocalSource:
    return LocalSource(root=landing)


def _session(customer_id: str) -> Session:
    service = MockIdentityService(signing_key="test-key", session_ttl=timedelta(minutes=15))
    challenge = service.begin_authentication(customer_id)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


@pytest.fixture
def owner(source: LocalSource) -> ToolLayer:
    return ToolLayer(_session(OWNER), source)


# -- registry / no money movement (REQ-34) ---------------------------------------------


def test_tool_registry_matches_declared_names(owner: ToolLayer) -> None:
    assert set(owner.tools()) == set(TOOL_NAMES)


def test_no_money_movement_tool_exists() -> None:
    # REQ-34: the layer exposes nothing that could move money or change a balance.
    forbidden = ("transfer", "payment", "pay", "deposit", "withdraw", "send", "move", "wire")
    for name in TOOL_NAMES:
        assert not any(bad in name for bad in forbidden), name


def test_datasource_exposes_no_write_methods(source: LocalSource) -> None:
    # The real guarantee behind REQ-34 (N5): the layer's only data access is read-only, so no
    # tool could change a balance even if misnamed. Assert the source has no mutating surface.
    mutating = ("insert", "update", "delete", "write", "execute", "drop", "merge", "upsert")
    public = {name for name in dir(source) if not name.startswith("_")}
    offenders = {name for name in public if any(m in name.lower() for m in mutating)}
    assert not offenders, offenders


# -- PII masking (REQ-36) --------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4717188030863827", "****3827"),
        ("7887710499", "****0499"),
        ("1234", "****"),
        ("", "****"),
        (None, None),
    ],
)
def test_mask_product_number(raw: str | None, expected: str | None) -> None:
    assert mask_product_number(raw) == expected


# -- inputs cannot smuggle a customer_id (REQ-12) --------------------------------------


def test_tool_input_rejects_customer_id_field() -> None:
    # The customer_id is injected from the session; an input that tries to carry one is rejected.
    with pytest.raises(ValidationError):
        GetBalanceInput(product_id=OWNER_CARD, customer_id=OTHER)  # type: ignore[call-arg]


# -- I6: list_products ------------------------------------------------------------------


def test_list_products_happy_path(owner: ToolLayer) -> None:
    result = owner.list_products(ListProductsInput())
    assert result.status is Status.OK
    assert result.data is not None
    ids = {p.product_id for p in result.data.products}
    # every product the session customer owns; the foreign product is never returned
    assert ids == {OWNER_CARD, OWNER_SAVINGS, OWNER_LOAN, OWNER_OVERLIMIT_CARD}
    card = next(p for p in result.data.products if p.product_id == OWNER_CARD)
    assert card.product_number_masked == "****3827"  # full PAN never exposed
    assert card.currency == "USD"
    # as_of is normalised to UTC-aware at the layer boundary (N2).
    assert result.as_of == datetime(2026, 6, 15, 10, 0, tzinfo=UTC)


# -- I1: get_balance --------------------------------------------------------------------


def test_get_balance_happy_path(owner: ToolLayer) -> None:
    result = owner.get_balance(GetBalanceInput(product_id=OWNER_CARD))
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.current_balance == pytest.approx(952.03)
    assert result.data.credit_limit == pytest.approx(40451.75)
    assert result.data.available_credit == pytest.approx(40451.75 - 952.03)
    assert result.data.currency == "USD"
    assert result.source_refs[0].ref == OWNER_CARD


def test_get_balance_loan_has_no_available_credit(owner: ToolLayer) -> None:
    # B1: a loan carries a credit_limit (original principal) but is not revolving credit, so
    # available_credit must be suppressed even though both balance and limit are present.
    result = owner.get_balance(GetBalanceInput(product_id=OWNER_LOAN))
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.current_balance == pytest.approx(5000.0)
    assert result.data.credit_limit == pytest.approx(20000.0)
    assert result.data.available_credit is None


def test_get_balance_over_limit_card_reports_negative_available(owner: ToolLayer) -> None:
    # B1: for a revolving card whose balance exceeds its limit, the derived available_credit is
    # the truthful negative over-limit position and is surfaced as-is (not floored).
    result = owner.get_balance(GetBalanceInput(product_id=OWNER_OVERLIMIT_CARD))
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.available_credit == pytest.approx(5000.0 - 6000.0)


def test_get_balance_forbidden_on_foreign_product_logs_attempt(owner: ToolLayer) -> None:
    result = owner.get_balance(GetBalanceInput(product_id=OTHER_CARD))
    assert result.status is Status.FORBIDDEN
    assert result.data is None  # fail closed: discloses nothing
    assert len(owner.access_log.attempts) == 1
    attempt = owner.access_log.attempts[0]
    assert attempt.tool == "get_balance"
    assert attempt.session_customer_id == OWNER
    assert attempt.resource_id == OTHER_CARD


def test_get_balance_not_found(owner: ToolLayer) -> None:
    result = owner.get_balance(GetBalanceInput(product_id="PRD-DOESNOTEXIST"))
    assert result.status is Status.NOT_FOUND
    assert not owner.access_log.attempts  # a non-existent product is not a foreign-access attempt


def test_get_balance_invalid_id(owner: ToolLayer) -> None:
    result = owner.get_balance(GetBalanceInput(product_id="PRD'; DROP TABLE products;--"))
    assert result.status is Status.INVALID


# -- I4: get_card_details ---------------------------------------------------------------


def test_get_card_details_happy_path(owner: ToolLayer) -> None:
    result = owner.get_card_details(GetCardDetailsInput(product_id=OWNER_CARD))
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.product_number_masked == "****3827"
    assert result.data.product_status == "Active"
    assert result.data.credit_limit == pytest.approx(40451.75)
    assert result.data.days_past_due == 0
    assert result.data.expiration_date == datetime(2027, 9, 12)


def test_get_card_details_forbidden_on_foreign(owner: ToolLayer) -> None:
    result = owner.get_card_details(GetCardDetailsInput(product_id=OTHER_CARD))
    assert result.status is Status.FORBIDDEN
    assert owner.access_log.attempts[0].tool == "get_card_details"


# -- I2, I3: search_transactions --------------------------------------------------------


def test_search_transactions_newest_first(owner: ToolLayer) -> None:
    result = owner.search_transactions(SearchTransactionsInput())
    assert result.status is Status.OK
    assert result.data is not None
    ids = [t.transaction_id for t in result.data.transactions]
    # Only the owner's three transactions, newest first; the foreign one never appears.
    assert ids == ["TRX-OWNER0002", "TRX-OWNER0001", "TRX-OWNER0003"]


def test_search_transactions_product_filter(owner: ToolLayer) -> None:
    result = owner.search_transactions(SearchTransactionsInput(product_id=OWNER_CARD))
    assert result.status is Status.OK
    assert result.data is not None
    assert {t.transaction_id for t in result.data.transactions} == {"TRX-OWNER0001", "TRX-OWNER0002"}


def test_search_transactions_merchant_filter(owner: ToolLayer) -> None:
    result = owner.search_transactions(SearchTransactionsInput(merchant="internet"))
    assert result.data is not None
    assert [t.transaction_id for t in result.data.transactions] == ["TRX-OWNER0002"]


def test_search_transactions_forbidden_on_foreign_product(owner: ToolLayer) -> None:
    result = owner.search_transactions(SearchTransactionsInput(product_id=OTHER_CARD))
    assert result.status is Status.FORBIDDEN
    assert owner.access_log.attempts[0].resource_id == OTHER_CARD


def test_search_transactions_limit_cap_enforced() -> None:
    # REQ: the search limit is capped at 20 by the contract itself.
    with pytest.raises(ValidationError):
        SearchTransactionsInput(limit=21)
    assert SearchTransactionsInput(limit=20).limit == 20


def test_search_transactions_respects_limit(owner: ToolLayer) -> None:
    result = owner.search_transactions(SearchTransactionsInput(limit=1))
    assert result.data is not None
    assert len(result.data.transactions) == 1
    assert result.data.transactions[0].transaction_id == "TRX-OWNER0002"  # newest


# -- I5, REQ-28: convert_currency -------------------------------------------------------


def test_convert_currency_exact_date(owner: ToolLayer) -> None:
    result = owner.convert_currency(
        ConvertCurrencyInput(amount=100, from_currency="USD", to_currency="COP", on_date=date(2026, 6, 10))
    )
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.rate == pytest.approx(3954.98)
    assert result.data.rate_date == date(2026, 6, 10)
    assert result.data.used_prior_rate is False
    assert result.data.converted_amount == pytest.approx(395498.0)
    assert result.source_refs[0].ref == "USD-COP@2026-06-10"


def test_convert_currency_uses_latest_prior_rate_within_7_days(owner: ToolLayer) -> None:
    # No rate on 2026-06-12; the latest prior (2026-06-10, 2 days) is used and flagged.
    result = owner.convert_currency(
        ConvertCurrencyInput(amount=10, from_currency="USD", to_currency="COP", on_date=date(2026, 6, 12))
    )
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.rate_date == date(2026, 6, 10)
    assert result.data.used_prior_rate is True


def test_convert_currency_abstains_when_rate_older_than_7_days(owner: ToolLayer) -> None:
    # Latest prior rate (2026-06-10) is 10 days before the requested date -> abstain.
    result = owner.convert_currency(
        ConvertCurrencyInput(amount=10, from_currency="USD", to_currency="COP", on_date=date(2026, 6, 20))
    )
    assert result.status is Status.UNAVAILABLE
    assert result.data is None


def test_convert_currency_accepts_rate_exactly_7_days_prior(owner: ToolLayer) -> None:
    # Boundary (N5): a rate dated exactly 7 days before the requested date is still accepted;
    # 2026-06-10 is exactly 7 days before 2026-06-17, so it is used and flagged as a prior rate.
    result = owner.convert_currency(
        ConvertCurrencyInput(amount=1, from_currency="USD", to_currency="COP", on_date=date(2026, 6, 17))
    )
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.rate_date == date(2026, 6, 10)
    assert result.data.used_prior_rate is True


def test_convert_currency_identity_same_currency(owner: ToolLayer) -> None:
    result = owner.convert_currency(
        ConvertCurrencyInput(amount=42, from_currency="USD", to_currency="USD", on_date=date(2026, 6, 10))
    )
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.rate == 1.0
    assert result.data.converted_amount == pytest.approx(42)


def test_convert_currency_unknown_currency_invalid(owner: ToolLayer) -> None:
    result = owner.convert_currency(
        ConvertCurrencyInput(amount=1, from_currency="USD", to_currency="JPY", on_date=date(2026, 6, 10))
    )
    assert result.status is Status.INVALID


# -- A1, A2: freeze_card / unfreeze_card (contract + authZ; write is task 2.3) ----------


def test_freeze_card_without_valid_confirmation_fails_closed(owner: ToolLayer) -> None:
    # REQ-14: a confirmation id that was never issued is invalid, so the owned-card freeze is
    # NOT executed and nothing is disclosed. (The confirmed happy path lives in
    # tests/unit/test_card_overlay.py.)
    result = owner.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id="CONF-UNKNOWN"))
    assert result.status is Status.INVALID
    assert result.data is None


def test_unfreeze_card_forbidden_on_foreign(owner: ToolLayer) -> None:
    result = owner.unfreeze_card(CardFreezeInput(product_id=OTHER_CARD, confirmation_id="CONF-1"))
    assert result.status is Status.FORBIDDEN
    assert owner.access_log.attempts[0].tool == "unfreeze_card"


def test_freeze_card_requires_a_confirmation_id() -> None:
    with pytest.raises(ValidationError):
        CardFreezeInput(product_id=OWNER_CARD, confirmation_id="")


# -- E1-E4: create_handoff --------------------------------------------------------------


def test_create_handoff_writes_store_with_injected_customer_id(source: LocalSource) -> None:
    store = InMemoryHandoffStore()
    layer = ToolLayer(_session(OWNER), source, handoff_store=store)
    result = layer.create_handoff(
        CreateHandoffInput(reason=HandoffReason.HUMAN_REQUEST, summary="quiere un humano")
    )
    assert result.status is Status.OK
    assert result.data is not None
    case_id = result.data.case_id
    assert case_id in store.cases
    # The persisted package carries the session customer id, not anything from the input.
    assert store.cases[case_id]["customer_id"] == OWNER
    assert store.cases[case_id]["reason"] == "E4"


def test_create_handoff_forbidden_on_foreign_product(owner: ToolLayer) -> None:
    result = owner.create_handoff(
        CreateHandoffInput(reason=HandoffReason.DISPUTE, summary="disputa", product_id=OTHER_CARD)
    )
    assert result.status is Status.FORBIDDEN
    assert owner.access_log.attempts[0].tool == "create_handoff"


def test_create_handoff_forbidden_on_foreign_transaction(source: LocalSource) -> None:
    # B1 (REQ-12): a transaction id is a resource too; citing another customer's transaction in
    # a handoff must fail closed, be logged, and never reach the store a human agent reads.
    store = InMemoryHandoffStore()
    layer = ToolLayer(_session(OWNER), source, handoff_store=store)
    result = layer.create_handoff(
        CreateHandoffInput(reason=HandoffReason.FRAUD, summary="fraude", transaction_id="TRX-OTHER0001")
    )
    assert result.status is Status.FORBIDDEN
    assert result.data is None
    assert not store.cases  # nothing persisted
    attempt = layer.access_log.attempts[0]
    assert attempt.tool == "create_handoff"
    assert attempt.resource_type == "transaction"
    assert attempt.resource_id == "TRX-OTHER0001"


def test_create_handoff_allows_owned_transaction(source: LocalSource) -> None:
    store = InMemoryHandoffStore()
    layer = ToolLayer(_session(OWNER), source, handoff_store=store)
    result = layer.create_handoff(
        CreateHandoffInput(reason=HandoffReason.DISPUTE, summary="disputa", transaction_id="TRX-OWNER0001")
    )
    assert result.status is Status.OK
    assert result.data is not None
    assert not layer.access_log.attempts
    assert store.cases[result.data.case_id]["transaction_id"] == "TRX-OWNER0001"


def test_create_handoff_not_found_on_unknown_transaction(source: LocalSource) -> None:
    store = InMemoryHandoffStore()
    layer = ToolLayer(_session(OWNER), source, handoff_store=store)
    result = layer.create_handoff(
        CreateHandoffInput(reason=HandoffReason.FRAUD, summary="fraude", transaction_id="TRX-NOPE0000")
    )
    assert result.status is Status.NOT_FOUND
    assert not store.cases  # a non-existent transaction is not a foreign-access attempt
    assert not layer.access_log.attempts


def test_create_handoff_rejects_orchestrator_only_e5_reason(source: LocalSource) -> None:
    # Review finding 2: E5 (NON_CUSTOMER) is orchestrator-only (set deterministically by the
    # non-customer branch after the existence check). The public tool must reject it fail-closed
    # so the model/user cannot assert the same identity-derived reason, and persist NOTHING.
    store = InMemoryHandoffStore()
    layer = ToolLayer(_session(OWNER), source, handoff_store=store)
    result = layer.create_handoff(
        CreateHandoffInput(reason=HandoffReason.NON_CUSTOMER, summary="should never persist")
    )
    assert result.status is Status.INVALID
    assert result.data is None
    assert not store.cases  # nothing written


# -- security: an id absent from `customers` owns NO resources (review finding 1) ------
#
# Even if the local/AWS data is inconsistent and an ORPHAN product/transaction row carries an id
# that has no `customers` record, that verified non-customer must NOT be able to read or act on
# it. Referential integrity is evidence, not an authorization boundary: every account read/action
# gates on the deterministic `customers`-record lookup, NOT on a matching resource row.

_ORPHAN_CUSTOMER = "CLI-ORPHAN000001"  # NO `customers` row, but owns orphan account rows below
_ORPHAN_CARD = "PRD-ORPHANCARD1"
_ORPHAN_TX = "TRX-ORPHAN0001"


def _build_orphan_landing(root: Path) -> None:
    # A registered customer (has a `customers` row AND an owned product) plus an ORPHAN id that
    # has account rows but NO `customers` row. The orphan rows must stay inaccessible.
    customers = pd.DataFrame([{"customer_id": OWNER, "full_name": "Owner One"}])
    products = pd.DataFrame(
        [
            {
                "product_id": OWNER_CARD,
                "customer_id": OWNER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            },
            {
                "product_id": _ORPHAN_CARD,
                "customer_id": _ORPHAN_CUSTOMER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4000111122223333",
                "currency": "USD",
                "current_balance": "7777.77",
                "credit_limit": "10000.0",
                "days_past_due": "0.0",
                "expiration_date": "2028-01-01",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            },
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "transaction_id": _ORPHAN_TX,
                "transaction_date": "2026-06-08 21:31:05",
                "process_date": "2026-06-08",
                "product_id": _ORPHAN_CARD,
                "customer_id": _ORPHAN_CUSTOMER,
                "amount": "267.32",
                "currency": "USD",
                "transaction_type": "Purchase",
                "transaction_status": "Approved",
                "merchant_name": "Tienda General",
                "transaction_category": "Food",
            }
        ]
    )
    _write(root / "customers.parquet", customers)
    _write(root / "products.parquet", products)
    tx_part = root / "transactions" / "year=2026" / "month=06" / "day=08"
    _write(tx_part / "transactions_20260608.parquet", transactions)


@pytest.fixture(scope="module")
def _orphan_landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("orphan_landing")
    _build_orphan_landing(root)
    return root


@pytest.fixture
def _orphan_source(_orphan_landing: Path) -> LocalSource:
    return LocalSource(root=_orphan_landing)


def test_orphan_customer_list_products_returns_empty(_orphan_source: LocalSource) -> None:
    layer = ToolLayer(_session(_ORPHAN_CUSTOMER), _orphan_source)
    result = layer.list_products(ListProductsInput())
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.products == []  # the orphan product is never listed


def test_orphan_customer_get_balance_denied(_orphan_source: LocalSource) -> None:
    layer = ToolLayer(_session(_ORPHAN_CUSTOMER), _orphan_source)
    result = layer.get_balance(GetBalanceInput(product_id=_ORPHAN_CARD))
    assert result.status is Status.NOT_FOUND
    assert result.data is None  # no balance figure leaks for an unregistered id


def test_orphan_customer_get_card_details_denied(_orphan_source: LocalSource) -> None:
    layer = ToolLayer(_session(_ORPHAN_CUSTOMER), _orphan_source)
    result = layer.get_card_details(GetCardDetailsInput(product_id=_ORPHAN_CARD))
    assert result.status is Status.NOT_FOUND
    assert result.data is None


def test_orphan_customer_search_transactions_returns_empty(_orphan_source: LocalSource) -> None:
    layer = ToolLayer(_session(_ORPHAN_CUSTOMER), _orphan_source)
    result = layer.search_transactions(SearchTransactionsInput())
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.transactions == []  # the orphan transaction is never returned


def test_orphan_customer_freeze_card_denied(_orphan_source: LocalSource) -> None:
    layer = ToolLayer(_session(_ORPHAN_CUSTOMER), _orphan_source)
    result = layer.freeze_card(CardFreezeInput(product_id=_ORPHAN_CARD, confirmation_id="CONF-1"))
    # The ownership gate denies before any confirmation/overlay write is considered.
    assert result.status is Status.NOT_FOUND
    assert result.data is None


def test_orphan_customer_create_handoff_cannot_reference_orphan_resources(
    _orphan_source: LocalSource,
) -> None:
    store = InMemoryHandoffStore()
    layer = ToolLayer(_session(_ORPHAN_CUSTOMER), _orphan_source, handoff_store=store)
    by_product = layer.create_handoff(
        CreateHandoffInput(reason=HandoffReason.DISPUTE, summary="x", product_id=_ORPHAN_CARD)
    )
    by_tx = layer.create_handoff(
        CreateHandoffInput(reason=HandoffReason.FRAUD, summary="x", transaction_id=_ORPHAN_TX)
    )
    assert by_product.status is Status.NOT_FOUND
    assert by_tx.status is Status.NOT_FOUND
    assert not store.cases  # neither orphan resource reaches the agent console


def test_registered_customer_unaffected_by_orphan_gate(_orphan_source: LocalSource) -> None:
    # Known-customer behavior is unchanged: the registered OWNER still reads its own product.
    layer = ToolLayer(_session(OWNER), _orphan_source)
    products = layer.list_products(ListProductsInput())
    balance = layer.get_balance(GetBalanceInput(product_id=OWNER_CARD))
    assert products.status is Status.OK
    assert products.data is not None
    assert [p.product_id for p in products.data.products] == [OWNER_CARD]
    assert balance.status is Status.OK
    assert balance.data is not None
    assert balance.data.current_balance == pytest.approx(952.03)


# -- fail closed on a `customers` lookup OUTAGE (review finding 1) ----------------------
#
# The customer-record gate has THREE states and must not collapse an UNAVAILABLE existence
# lookup into a definitive absence. A REGISTERED customer whose `customers` lookup RAISES must
# receive UNAVAILABLE on every path (collections AND point/action reads) - never a successful
# empty collection or a false NOT_FOUND, which would silently disclose "no data" after a backend
# failure. The ordinary resource reads must be reachable (so the failure is attributable to the
# existence gate alone), so only the `customers` count is forced to raise.


class _CustomersCountFails:
    """Delegating `DataSource` whose only failure is the `customers` existence count.

    Every other read passes straight through to the real `LocalSource`, so a healthy registered
    customer would otherwise read normally; forcing just the `customers` count to raise isolates
    the existence-gate outage (review finding 1).
    """

    def __init__(self, inner: LocalSource) -> None:
        self._inner = inner

    def count(self, table: str, *, where: str | None = None) -> int:
        if table == "customers":
            raise RuntimeError("simulated customers lookup outage")
        return self._inner.count(table, where=where)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def test_registered_customer_list_products_unavailable_on_customers_outage(source: LocalSource) -> None:
    layer = ToolLayer(_session(OWNER), _CustomersCountFails(source))  # type: ignore[arg-type]
    result = layer.list_products(ListProductsInput())
    # Not a successful empty list: the existence lookup was in doubt, so fail closed.
    assert result.status is Status.UNAVAILABLE
    assert result.data is None


def test_registered_customer_search_transactions_unavailable_on_customers_outage(
    source: LocalSource,
) -> None:
    layer = ToolLayer(_session(OWNER), _CustomersCountFails(source))  # type: ignore[arg-type]
    result = layer.search_transactions(SearchTransactionsInput())
    assert result.status is Status.UNAVAILABLE
    assert result.data is None


def test_registered_customer_get_balance_unavailable_on_customers_outage(source: LocalSource) -> None:
    layer = ToolLayer(_session(OWNER), _CustomersCountFails(source))  # type: ignore[arg-type]
    result = layer.get_balance(GetBalanceInput(product_id=OWNER_CARD))
    # Point read: UNAVAILABLE, not a false NOT_FOUND, so no "product not found" is implied.
    assert result.status is Status.UNAVAILABLE
    assert result.data is None


def test_registered_customer_freeze_card_unavailable_on_customers_outage(source: LocalSource) -> None:
    layer = ToolLayer(_session(OWNER), _CustomersCountFails(source))  # type: ignore[arg-type]
    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id="CONF-1"))
    # Action path: the existence gate fails closed before any confirmation/overlay write.
    assert result.status is Status.UNAVAILABLE
    assert result.data is None
