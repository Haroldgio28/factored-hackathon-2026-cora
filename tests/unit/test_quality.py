"""Unit tests for the data-quality package (CORA task 1.3, REQ-21/REQ-22/REQ-23).

The pure-pandas paths (quarantine, dedup) and the DuckDB checks run on tiny in-memory
frames with KNOWN violations, so no local Parquet is needed. Checks are exercised through a
small `InMemorySource` that registers each DataFrame as a DuckDB view and returns the view
name as the read expression — the same `DataSource` surface the real sources expose, so the
check SQL is unchanged. A test asserts `RANGE_CHECKS` agrees with the contracts so they
never drift. Data-dependent tests skip cleanly when the landing is absent (mirrors
tests/unit/test_datasource.py).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from cora.data.contracts import get_schema, validate
from cora.data.datasource import LocalSource
from cora.data.quality import (
    RANGE_CHECKS,
    build_report,
    deduplicate,
    quarantine_batch,
    report_to_json,
    report_to_markdown,
)
from cora.data.quality.checks import (
    currency_checks,
    enum_violation_checks,
    null_rate_checks,
    orphan_fk_checks,
    out_of_range_checks,
    pk_duplicate_check,
    row_hash_duplicate_check,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw_parquet"

local_data = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="local Parquet landing (data/raw_parquet) is absent",
)


class InMemorySource:
    """A `DataSource`-shaped stub backed by in-memory DataFrames registered in DuckDB.

    Exposes `_source(table)` and `_connect()` so the quality checks run their real SQL over
    tiny frames without touching Parquet. Each frame is registered as a DuckDB view named
    `mem_<table>`; `_source` returns that view name.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._con = duckdb.connect()
        self._views: dict[str, str] = {}
        for table, df in frames.items():
            view = f"mem_{table}"
            self._con.register(view, df)
            self._views[table] = view

    def _source(self, table: str) -> str:
        return self._views[table]

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return self._con


# --- Checks over tiny in-memory frames with known violations --------------------------


def test_pk_duplicate_check_counts_extra_rows() -> None:
    df = pd.DataFrame({"customer_id": ["C1", "C1", "C2"]})
    src = InMemorySource({"customers": df})
    result = pk_duplicate_check(src, "customers")
    assert result.denominator == 3
    assert result.numerator == 1  # one repeated PK row
    assert "C1" in result.examples


def test_pk_duplicate_check_composite_key() -> None:
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "source_currency": ["USD", "USD", "USD"],
            "target_currency": ["COP", "COP", "COP"],
        }
    )
    src = InMemorySource({"daily_exchange_rates": df})
    result = pk_duplicate_check(src, "daily_exchange_rates")
    assert result.numerator == 1
    assert result.denominator == 3


def test_row_hash_duplicate_check() -> None:
    frame = _products_frame(2)
    frame.loc[1] = frame.loc[0]  # full-row duplicate
    frame["product_id"] = ["P1", "P1"]
    src = InMemorySource({"products": frame})
    result = row_hash_duplicate_check(src, "products")
    assert result.numerator == 1
    assert result.denominator == 2


def test_null_rate_flags_non_nullable_null() -> None:
    df = pd.DataFrame({"customer_id": ["C1", None], "country": ["México", "México"]})
    # Only the two columns we care about are present; checks iterate contract columns.
    src = InMemorySource({"customers": _pad_customers(df)})
    results = null_rate_checks(src, "customers")
    pk_null = next(r for r in results if r.column == "customer_id")
    assert pk_null.numerator == 1  # one null in a non-nullable column => violation
    assert pk_null.denominator == 2


def test_orphan_fk_check_detects_missing_parent() -> None:
    products = pd.DataFrame({"customer_id": ["C1", "C2", "C_missing"]})
    customers = pd.DataFrame({"customer_id": ["C1", "C2"]})
    # pad products with the PK so the relation has the needed columns
    products["product_id"] = ["P1", "P2", "P3"]
    src = InMemorySource({"products": products, "customers": customers})
    results = orphan_fk_checks(src, "products")
    fk = next(r for r in results if r.column == "customer_id")
    assert fk.numerator == 1
    assert fk.denominator == 3
    assert "C_missing" in fk.examples


def test_out_of_range_credit_score() -> None:
    df = pd.DataFrame({"customer_id": ["C1", "C2"], "credit_score": [700, 900]})
    src = InMemorySource({"customers": df})
    results = out_of_range_checks(src, "customers")
    cs = next(r for r in results if r.column == "credit_score")
    assert cs.numerator == 1  # 900 > 850
    assert cs.denominator == 2
    assert 900 in cs.examples


def test_enum_violation_check() -> None:
    df = pd.DataFrame({"product_id": ["P1", "P2"], "product_status": ["Active", "Frozen"]})
    src = InMemorySource({"products": df})
    results = enum_violation_checks(src, "products")
    status = next(r for r in results if r.column == "product_status")
    assert status.numerator == 1
    assert "Frozen" in status.examples


def test_currency_check_products_rejects_mxn() -> None:
    df = pd.DataFrame({"product_id": ["P1", "P2"], "currency": ["USD", "MXN"]})
    src = InMemorySource({"products": df})
    results = currency_checks(src, "products")
    cur = next(r for r in results if r.column == "currency")
    assert cur.numerator == 1  # MXN not allowed for products (F3)
    assert "MXN" in cur.examples


def test_currency_check_complaints_allows_mxn() -> None:
    df = pd.DataFrame({"complaint_id": ["X1", "X2"], "currency": ["USD", "MXN"]})
    src = InMemorySource({"complaints": df})
    results = currency_checks(src, "complaints")
    cur = next(r for r in results if r.column == "currency")
    assert cur.numerator == 0  # MXN IS allowed for complaints (ADR-017)


# --- Quarantine: split, reason, pipeline continues ------------------------------------


def test_quarantine_splits_and_attaches_reason(tmp_path: Path) -> None:
    df = _products_frame(2)
    df.loc[0, "product_status"] = "Frozen"  # bad enum
    valid_df, failure_cases = validate("products", df)
    kept, result = quarantine_batch("products", df, failure_cases, out_dir=tmp_path, run_id="run1")
    assert result.quarantined == 1
    assert result.valid == 1
    assert result.total == 2
    assert len(kept) == 1
    # The quarantined file exists and names the offending column/check.
    written = pd.read_parquet(result.out_path)
    assert len(written) == 1
    assert "product_status" in written["quarantine_reason"].iloc[0]


def test_quarantine_all_good_writes_nothing(tmp_path: Path) -> None:
    df = _products_frame(1)
    valid_df, failure_cases = validate("products", df)
    kept, result = quarantine_batch("products", df, failure_cases, out_dir=tmp_path, run_id="run2")
    assert result.quarantined == 0
    assert result.out_path is None
    assert len(kept) == 1


def test_quarantine_all_bad_continues(tmp_path: Path) -> None:
    df = _products_frame(2)
    df["product_status"] = ["Frozen", "Melted"]  # both bad enums
    valid_df, failure_cases = validate("products", df)
    kept, result = quarantine_batch("products", df, failure_cases, out_dir=tmp_path, run_id="run3")
    # Pipeline continues: no raise, all rows quarantined, zero valid.
    assert result.quarantined == 2
    assert len(kept) == 0


# --- Dedup: latest wins per PK, counts removals ---------------------------------------


def test_dedup_keeps_latest_by_last_updated() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "last_updated": pd.to_datetime(["2026-01-01", "2026-06-01", "2026-03-01"]),
            "segment": ["Basic", "Premium", "Plus"],
        }
    )
    deduped, result = deduplicate("customers", df)
    assert result.removed == 1
    assert result.tie_break == "last_updated_desc"
    kept = deduped.set_index("customer_id")["segment"].to_dict()
    assert kept["C1"] == "Premium"  # latest last_updated wins


def test_dedup_keeps_latest_by_process_date_fact() -> None:
    df = pd.DataFrame(
        {
            "transaction_id": ["T1", "T1"],
            "process_date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "amount": [10.0, 20.0],
        }
    )
    deduped, result = deduplicate("transactions", df)
    assert result.removed == 1
    assert result.tie_break == "process_date_desc"
    assert deduped["amount"].iloc[0] == 20.0


def test_dedup_input_order_tie_break_branches() -> None:
    df = pd.DataFrame({"branch_id": ["B1", "B1", "B2"], "branch_name": ["a", "b", "c"]})
    deduped, result = deduplicate("branches", df)
    assert result.removed == 1
    assert result.tie_break == "input_order"
    assert deduped.set_index("branch_id")["branch_name"]["B1"] == "a"


# --- RANGE_CHECKS agrees with the contracts -------------------------------------------


def test_range_checks_agree_with_contracts() -> None:
    for (table, column), (low, high) in RANGE_CHECKS.items():
        col = get_schema(table).columns[column]
        bounds = [
            (c.statistics.get("min_value"), c.statistics.get("max_value"))
            for c in col.checks
            if getattr(c, "statistics", None) and "min_value" in c.statistics and "max_value" in c.statistics
        ]
        assert (low, high) in bounds, f"{table}.{column} range drifted from contract"


# --- Smoke test over a bounded real sample --------------------------------------------


@local_data
def test_build_report_over_real_sample_bounded() -> None:
    source = LocalSource(root=DATA_ROOT)
    tables = ["customers", "transactions"]
    report = build_report(source, tables, sample_limit=200)
    assert set(report.tables) == set(tables)
    # Fact table result must be labeled sampled (never loaded whole).
    tx_results = report.for_table("transactions")
    assert tx_results and all(r.scope == "sampled" for r in tx_results)
    # JSON and markdown are produced and non-empty.
    assert report_to_json(report)["results"]
    md = report_to_markdown(report)
    assert "customers" in md and "transactions" in md


# --- Helpers --------------------------------------------------------------------------


def _products_frame(n: int = 1) -> pd.DataFrame:
    """A minimal valid raw-string `products` frame of `n` rows (mirrors contract tests)."""
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


def _pad_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tiny customers frame has every contract column (nulls where unset)."""
    out = df.copy()
    for name in get_schema("customers").columns:
        if name not in out.columns:
            out[name] = [None] * len(out)
    return out
