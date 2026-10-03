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
