"""Unit + integration tests for schema-evolution handling (CORA task 1.5, REQ-25).

TEAM-GENERATED. All tests use TINY synthetic raw-string frames (a handful of rows) so they
run WITHOUT the real landing and stay fast + memory-safe. They cover the REQ-25 behaviour:
an additive nullable column is accepted and logged (processing continues, the row is curated);
a missing REQUIRED contract column fails the table loudly with a message naming the column;
the all-VARCHAR landing does NOT false-positive on type (a column the contract coerces is
accepted, `type_changes` stays empty); the schema-event log appends correctly and round-trips;
and `RunSummary` surfaces the events. The integration test proves fail-closed: a missing
required column raises and leaves NO curated partition written.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.data.pipeline import (
    SchemaChangeEvent,
    SchemaEvolutionError,
    append_schema_events,
    check_schema,
    load_schema_events,
    process_partition,
    raise_if_breaking,
    run_table,
)

# --- Helpers (local copies, mirroring tests/unit/test_pipeline.py) --------------------


def _transactions_frame(ids: list[str], process_date: str, *, amount: str = "10.0") -> pd.DataFrame:
    """A minimal valid raw-string `transactions` frame (mirrors the contract/pipeline tests)."""
    n = len(ids)
    return pd.DataFrame(
        {
            "transaction_id": ids,
            "transaction_date": [process_date] * n,
            "process_date": [process_date] * n,
            "product_id": ["P00000"] * n,
            "customer_id": ["C00000"] * n,
            "transaction_type": ["Purchase"] * n,
            "transaction_category": ["Food"] * n,
            "amount": [amount] * n,
            "currency": ["USD"] * n,
            "amount_usd": [amount] * n,
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


def _write_partition(raw_root: Path, table: str, d: date, df: pd.DataFrame) -> None:
    """Write `df` into the Hive layout the real landing uses (zero-padded month/day)."""
    part = raw_root / table / f"year={d.year}" / f"month={d.month:02d}" / f"day={d.day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    df.to_parquet(part / f"{table}_{d:%Y%m%d}.parquet", index=False)


def _curated_dir(curated_root: Path, table: str, d: date) -> Path:
    return curated_root / table / f"process_date={d.isoformat()}"


def _curated_rows(curated_root: Path, table: str, d: date) -> pd.DataFrame:
    return pd.read_parquet(_curated_dir(curated_root, table, d) / "part.parquet")


def _run(raw_root: Path, tmp_path: Path, table: str = "transactions", **kwargs: object):
    source = LocalSource(root=raw_root)
    return run_table(
        source,
        table,
        root=raw_root,
        curated_root=tmp_path / "curated",
        quarantine_root=tmp_path / "quarantine",
        state_dir=tmp_path / "_state",
        **kwargs,  # type: ignore[arg-type]
    )


def _event(**overrides: object) -> SchemaChangeEvent:
    base = dict(
        table="transactions",
        kind="added",
        column="promo_flag",
        detail="d",
        process_date="2026-06-01",
        detected_at="2026-06-01T00:00:00Z",
        pipeline_version="1.0",
        contracts_version="1.0",
    )
    base.update(overrides)
    return SchemaChangeEvent(**base)  # type: ignore[arg-type]


# --- check_schema: pure diff behaviour ------------------------------------------------


def test_added_nullable_column_accepted_and_logged() -> None:
    df = _transactions_frame(["T1"], "2026-06-01")
    df["promo_flag"] = ["spring"]
    result = check_schema("transactions", df, process_date="2026-06-01")
    assert result.is_breaking is False
    assert result.added == ["promo_flag"]
    added_events = [e for e in result.events if e.kind == "added"]
    assert [e.column for e in added_events] == ["promo_flag"]
    assert added_events[0].table == "transactions"


def test_missing_required_column_is_breaking_and_names_column() -> None:
    df = _transactions_frame(["T1"], "2026-06-01").drop(columns=["customer_id"])
    result = check_schema("transactions", df)
    assert result.is_breaking is True
    assert result.breaking_detail is not None
    assert "customer_id" in result.breaking_detail
    assert "transactions" in result.breaking_detail
    with pytest.raises(SchemaEvolutionError) as excinfo:
        raise_if_breaking(result)
    message = str(excinfo.value)
    assert "customer_id" in message
    assert "transactions" in message


def test_missing_optional_partition_column_is_not_breaking() -> None:
    # The frame has no year/month/day columns (required=False) — their absence is informational.
    df = _transactions_frame(["T1"], "2026-06-01")
    result = check_schema("transactions", df)
    assert result.is_breaking is False
    # year/month/day are optional contract columns; absence yields `removed` events, not failure.
    removed_cols = {e.column for e in result.events if e.kind == "removed"}
    assert {"year", "month", "day"}.issubset(removed_cols)


def test_all_varchar_landing_no_false_positive_on_type() -> None:
    # Standard raw-string frame: every value is a string the contract coerces. Name-based diff
    # must not flag type changes, and nothing is breaking.
    df = _transactions_frame(["T1", "T2"], "2026-06-01")
    result = check_schema("transactions", df)
    assert result.type_changes == []
    assert result.is_breaking is False


# --- Schema-event log append-correctness + round-trip ---------------------------------


def test_schema_event_log_appends_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "_state" / "schema_events.json"
    e = _event()
    append_schema_events([e], path)
    append_schema_events([e], path)
    loaded = load_schema_events(path)
    assert len(loaded) == 2
    assert loaded[0] == e
    # Round-trip equality through to_dict/from_dict.
    assert SchemaChangeEvent.from_dict(e.to_dict()) == e
    # Absent file initialises empty; empty append is a no-op.
    assert load_schema_events(tmp_path / "nope.json") == []
    append_schema_events([], path)
    assert len(load_schema_events(path)) == 2


# --- Integration through the runner ---------------------------------------------------


def test_added_column_curates_and_surfaces_event_on_run_summary(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    df = _transactions_frame(["T1"], d.isoformat())
    df["promo_flag"] = ["spring"]
    _write_partition(raw, "transactions", d, df)
    summary = _run(raw, tmp_path)
    # Additive column accepted: the row is curated and the extra column survives (strict=False).
    assert summary.schema_breaking is False
    assert len(summary.schema_events) >= 1
    assert any(ev.kind == "added" and ev.column == "promo_flag" for ev in summary.schema_events)
    curated = _curated_rows(tmp_path / "curated", "transactions", d)
    assert len(curated) == 1
    assert "promo_flag" in curated.columns
    # The event was persisted to the runtime log under the run's state dir.
    log = load_schema_events(tmp_path / "_state" / "schema_events.json")
    assert any(ev.column == "promo_flag" for ev in log)


def test_missing_required_raises_and_writes_no_curated_partition(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    df = _transactions_frame(["T1"], d.isoformat()).drop(columns=["customer_id"])
    _write_partition(raw, "transactions", d, df)
    source = LocalSource(root=raw)
    with pytest.raises(SchemaEvolutionError) as excinfo:
        process_partition(
            source,
            "transactions",
            d,
            curated_root=tmp_path / "curated",
            quarantine_root=tmp_path / "quarantine",
            run_id="test_run",
        )
    assert "customer_id" in str(excinfo.value)
    # Fail-closed: no curated partition directory was written.
    assert not _curated_dir(tmp_path / "curated", "transactions", d).exists()
