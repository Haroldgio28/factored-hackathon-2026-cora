"""Contract tests for the 13 Pandera schemas (CORA task 1.2, REQ-21/REQ-22).

Positive tests read a small bounded sample per table from the local Parquet landing via
`LocalSource` and assert it VALIDATES after coercion (so the raw-string-landing ->
canonical-typed design works on real data). They skip cleanly when the landing is absent,
mirroring tests/unit/test_datasource.py, so CI without the data still passes. Facts are
read from a single present partition (year=2026, month=6) with a small limit so no fact
table is ever loaded in full (host RAM ~4 GB).

Negative tests build tiny hand-crafted raw-string frames (no local data needed) and assert
`validate` RETURNS the violation as a failure case rather than raising, and that the bad
rows are excluded from `valid_df`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cora.data.contracts import CONTRACTS_VERSION, SCHEMAS, validate
from cora.data.datasource import FACTS, TABLES, LocalSource

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw_parquet"

local_data = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="local Parquet landing (data/raw_parquet) is absent",
)

# Bounded sample size: small enough to stay memory-safe for the fact tables.
SAMPLE_LIMIT = 200


def test_registry_covers_all_tables() -> None:
    assert set(SCHEMAS) == set(TABLES)
    assert len(SCHEMAS) == 13


def test_contracts_version() -> None:
    assert CONTRACTS_VERSION == "1.0"


@local_data
@pytest.mark.parametrize("table", sorted(TABLES))
def test_sample_validates_against_contract(table: str) -> None:
    """A bounded local sample of every table validates after coercion."""
    source = LocalSource(root=DATA_ROOT)
    kwargs: dict[str, int] = {"year": 2026, "month": 6} if table in FACTS else {}
    df = source.fetch_df(table, limit=SAMPLE_LIMIT, **kwargs)
    assert len(df) <= SAMPLE_LIMIT  # guard: never load a fact table in full
    valid_df, failure_cases = validate(table, df)
    assert failure_cases.empty, (
        f"{table} sample failed validation:\n"
        f"{failure_cases[['column', 'check', 'failure_case']].drop_duplicates().head(20)}"
    )
    assert len(valid_df) == len(df)


# --- Negative tests: raw-string frames that violate the contract ----------------------
#
# Each builds the minimal columns a schema needs, with the whole frame in the raw-string
# landing shape, then injects one violation and asserts it is reported (not raised).


def _customers_frame(n: int = 2) -> pd.DataFrame:
    """A minimal valid raw-string `customers` frame of `n` rows."""
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "document_number": [f"D{i:05d}" for i in range(n)],
            "document_type": ["DNI"] * n,
            "first_name": ["Ana"] * n,
            "last_name": ["García"] * n,
            "date_of_birth": ["1990-01-01"] * n,
            "gender": ["F"] * n,
            "email": ["ana@example.com"] * n,
            "mobile_phone": ["+521234567"] * n,
            "landline_phone": [None] * n,
            "address": ["Calle 1"] * n,
            "city": ["CDMX"] * n,
            "state": ["CDMX"] * n,
            "country": ["México"] * n,
            "postal_code": ["01000"] * n,
            "detected_accent": ["mexican"] * n,
            "segment": ["Basic"] * n,
            "credit_score": ["700"] * n,
            "estimated_monthly_income": ["50000.0"] * n,
            "occupation": ["Engineer"] * n,
            "marital_status": ["Single"] * n,
            "education_level": ["University"] * n,
            "registration_date": ["2020-01-01"] * n,
            "registration_branch_id": ["B001"] * n,
            "customer_status": ["Active"] * n,
            "last_updated": ["2026-01-01"] * n,
            "accepts_marketing": ["True"] * n,
        }
    )


def _products_frame(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [f"P{i:05d}" for i in range(n)],
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "product_type": ["Cuenta Ahorro"] * n,
            "product_number": [f"N{i:05d}" for i in range(n)],
            "currency": ["USD"] * n,
            "current_balance": ["100.0"] * n,
            "credit_limit": [None] * n,
            "interest_rate": [None] * n,
            "opening_date": ["2021-01-01"] * n,
            "expiration_date": [None] * n,
            "opening_branch_id": ["B001"] * n,
            "product_status": ["Active"] * n,
            "opening_channel": ["Branch"] * n,
            "has_linked_app": ["True"] * n,
            "days_past_due": [None] * n,
            "last_transaction_date": [None] * n,
            "last_updated": ["2026-01-01"] * n,
        }
    )


def _transactions_frame(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:05d}" for i in range(n)],
            "transaction_date": ["2026-06-01"] * n,
            "process_date": ["2026-06-01"] * n,
            "product_id": ["P00000"] * n,
            "customer_id": ["C00000"] * n,
            "transaction_type": ["Purchase"] * n,
            "transaction_category": ["Food"] * n,
            "amount": ["10.0"] * n,
            "currency": ["USD"] * n,
            "amount_usd": ["10.0"] * n,
            "channel": ["POS"] * n,
            "branch_id": [None] * n,
            "merchant_name": [None] * n,
            "merchant_category": [None] * n,
            "transaction_country": ["MX"] * n,
            "transaction_city": ["CDMX"] * n,
            "transaction_status": ["Approved"] * n,
            "response_code": ["00"] * n,
            "is_fraud": ["False"] * n,
            "fraud_score": ["10.0"] * n,
            "latitude": [None] * n,
            "longitude": [None] * n,
        }
    )


def _columns_in_failures(failure_cases: pd.DataFrame) -> set[str]:
    return set(failure_cases["column"].dropna().unique())


def test_negative_bad_enum_value() -> None:
    """An out-of-vocabulary product_status is reported, not raised."""
    df = _products_frame()
    df.loc[0, "product_status"] = "Frozen"  # not in the observed enum
    valid_df, failure_cases = validate("products", df)
    assert not failure_cases.empty
    assert "product_status" in _columns_in_failures(failure_cases)
    assert len(valid_df) == 0


def test_negative_null_in_non_nullable() -> None:
    """A null customer_id (non-nullable PK) is reported."""
    df = _customers_frame(n=1)
    df.loc[0, "customer_id"] = None
    valid_df, failure_cases = validate("customers", df)
    assert not failure_cases.empty
    assert "customer_id" in _columns_in_failures(failure_cases)
    assert len(valid_df) == 0


def test_negative_fraud_score_out_of_range() -> None:
    """fraud_score above 100 is reported."""
    df = _transactions_frame()
    df.loc[0, "fraud_score"] = "150"
    valid_df, failure_cases = validate("transactions", df)
    assert not failure_cases.empty
    assert "fraud_score" in _columns_in_failures(failure_cases)
    assert len(valid_df) == 0


def test_negative_credit_score_out_of_range() -> None:
    """credit_score above 850 is reported."""
    df = _customers_frame(n=1)
    df.loc[0, "credit_score"] = "900"
    valid_df, failure_cases = validate("customers", df)
    assert not failure_cases.empty
    assert "credit_score" in _columns_in_failures(failure_cases)
    assert len(valid_df) == 0


def test_negative_duplicate_primary_key() -> None:
    """Two rows with the same customer_id are reported as a uniqueness failure."""
    df = _customers_frame(n=2)
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    valid_df, failure_cases = validate("customers", df)
    assert not failure_cases.empty
    assert "customer_id" in _columns_in_failures(failure_cases)
    # The duplicated rows are excluded from the valid frame.
    assert len(valid_df) < len(df)
