"""Unit tests for row-level lineage + the per-run lineage manifest (CORA task 1.6, REQ-27).

TEAM-GENERATED. All synthetic tests write TINY partitioned Parquet into `tmp_path` and point
a `LocalSource` at it, so they run WITHOUT the real landing and stay fast + memory-safe (a
handful of rows per partition), mirroring tests/unit/test_pipeline.py. They cover REQ-27:
every curated row carries `_source_file`/`_ingested_at`/`_pipeline_version` with the correct
values; all rows of a run share one `_ingested_at`; two runs of the same partition keep
identical DATA columns (idempotency re-scoped to DATA, lineage excluded); the per-run manifest
is written at `data/_state/manifests/<table>/<run_id>.json` with inputs->outputs and counts
matching the RunSummary, and includes task-1.5 schema events when a schema change occurs; a
dry-run writes no manifest; and `RunManifest.from_run(...)`/`to_dict()` round-trips (pure).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from cora.data.contracts import CONTRACTS_VERSION
from cora.data.datasource import LocalSource
from cora.data.pipeline import (
    LINEAGE_COLUMNS,
    PIPELINE_VERSION,
    FreshnessRecord,
    PartitionResult,
    RunManifest,
    RunSummary,
    raw_partition_dir,
    run_table,
)

# --- Helpers (mirror tests/unit/test_pipeline.py) -------------------------------------


def _transactions_frame(ids: list[str], process_date: str, *, amount: str = "10.0") -> pd.DataFrame:
    """A minimal valid raw-string `transactions` frame (mirrors the pipeline tests)."""
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


def _curated_rows(curated_root: Path, table: str, d: date) -> pd.DataFrame:
    path = curated_root / table / f"process_date={d.isoformat()}" / "part.parquet"
    return pd.read_parquet(path)


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


# --- Row-level lineage columns --------------------------------------------------------


def test_curated_rows_carry_lineage_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    _write_partition(raw, "transactions", d, _transactions_frame(["T1", "T2"], d.isoformat()))
    summary = _run(raw, tmp_path)
    curated = _curated_rows(tmp_path / "curated", "transactions", d)

    # All three lineage columns are present on every curated row.
    assert set(LINEAGE_COLUMNS).issubset(curated.columns)
    assert len(curated) == 2

    # `_source_file` is the raw Hive partition directory this curated partition was read from.
    expected_source = raw_partition_dir(raw, "transactions", d)
    assert (curated["_source_file"] == expected_source).all()

    # `_pipeline_version` is the committed PIPELINE_VERSION.
    assert (curated["_pipeline_version"] == PIPELINE_VERSION).all()

    # `_ingested_at` is one run-constant UTC ISO timestamp shared by all rows of the run.
    assert curated["_ingested_at"].nunique() == 1
    assert curated["_ingested_at"].iloc[0].endswith("Z")
    # It agrees with the freshness record's last_ingested_at (same run timestamp).
    assert curated["_ingested_at"].iloc[0] == summary.freshness.last_ingested_at


def test_source_file_matches_dir_per_partition(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d1, d2 = date(2026, 6, 1), date(2026, 6, 2)
    _write_partition(raw, "transactions", d1, _transactions_frame(["T1"], d1.isoformat()))
    _write_partition(raw, "transactions", d2, _transactions_frame(["T2"], d2.isoformat()))
    _run(raw, tmp_path)
    first = _curated_rows(tmp_path / "curated", "transactions", d1)
    second = _curated_rows(tmp_path / "curated", "transactions", d2)
    assert first["_source_file"].iloc[0] == raw_partition_dir(raw, "transactions", d1)
    assert second["_source_file"].iloc[0] == raw_partition_dir(raw, "transactions", d2)
    # The two partitions share one run timestamp.
    assert first["_ingested_at"].iloc[0] == second["_ingested_at"].iloc[0]


# --- Idempotency re-scoped to DATA columns --------------------------------------------


def test_idempotency_preserved_excluding_lineage(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    _write_partition(raw, "transactions", d, _transactions_frame(["T1", "T2"], d.isoformat()))
    _run(raw, tmp_path)
    first = _curated_rows(tmp_path / "curated", "transactions", d)
    _run(raw, tmp_path)  # second run of the same partition
    second = _curated_rows(tmp_path / "curated", "transactions", d)

    pk = "transaction_id"
    a = first.drop(columns=list(LINEAGE_COLUMNS)).sort_values(pk).reset_index(drop=True)
    b = second.drop(columns=list(LINEAGE_COLUMNS)).sort_values(pk).reset_index(drop=True)
    assert a.equals(b)  # DATA columns are identical across runs (overwrite, not append)
    assert len(first) == len(second) == 2
    # Each run's `_ingested_at` is internally uniform (one value per run).
    assert first["_ingested_at"].nunique() == 1
    assert second["_ingested_at"].nunique() == 1


# --- Per-run lineage manifest ---------------------------------------------------------


def test_manifest_written_with_inputs_outputs_and_counts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    _write_partition(raw, "transactions", d, _transactions_frame(["T1", "T2"], d.isoformat()))
    summary = _run(raw, tmp_path)

    # The manifest path is surfaced on the summary and points at a real file.
    assert summary.manifest_path is not None
    manifest_file = Path(summary.manifest_path)
    assert manifest_file.exists()
    assert manifest_file.parent == tmp_path / "_state" / "manifests" / "transactions"

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["table"] == "transactions"
    assert data["pipeline_version"] == PIPELINE_VERSION
    assert data["contracts_version"] == CONTRACTS_VERSION
    assert data["created_at"] == summary.freshness.last_ingested_at

    # Totals match the RunSummary (not recomputed).
    assert data["totals"]["partitions_processed"] == summary.partitions_processed
    assert data["totals"]["rows_curated"] == summary.rows_curated
    assert data["totals"]["rows_quarantined"] == summary.rows_quarantined
    assert data["totals"]["duplicates_removed"] == summary.duplicates_removed

    # Per-partition INPUT -> OUTPUT and counts match the RunSummary partition.
    assert len(data["partitions"]) == 1
    part = data["partitions"][0]
    pr = summary.partitions[0]
    assert part["process_date"] == pr.process_date
    assert part["source"] == raw_partition_dir(raw, "transactions", d)
    assert part["curated_path"] == pr.curated_path
    assert part["rows_in"] == pr.rows_in
    assert part["rows_curated"] == pr.rows_curated
    assert part["rows_quarantined"] == pr.rows_quarantined
    assert part["duplicates_removed"] == pr.duplicates_removed


def test_manifest_includes_schema_events_on_change(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    df = _transactions_frame(["T1"], d.isoformat())
    df["promo_flag"] = ["spring"]  # additive column -> a task-1.5 schema event
    _write_partition(raw, "transactions", d, df)
    summary = _run(raw, tmp_path)
    assert summary.manifest_path is not None

    data = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
    added = [e for e in data["schema_events"] if e["kind"] == "added"]
    assert any(e["column"] == "promo_flag" for e in added)
    # The per-partition record also carries the schema event.
    part_events = data["partitions"][0]["schema_events"]
    assert any(e["column"] == "promo_flag" for e in part_events)


def test_dry_run_writes_no_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    _write_partition(raw, "transactions", d, _transactions_frame(["T1"], d.isoformat()))
    summary = _run(raw, tmp_path, dry_run=True)
    assert summary.manifest_path is None
    assert not (tmp_path / "_state" / "manifests").exists()


# --- RunManifest.from_run / to_dict round-trip (pure, no I/O) --------------------------


def test_run_manifest_from_run_round_trips() -> None:
    freshness = FreshnessRecord(
        table="transactions",
        max_partition_date="2026-06-01",
        max_event_date="2026-06-01",
        watermark="2026-06-01",
        last_ingested_at="2026-06-01T00:00:00Z",
        curated_rows=2,
    )
    pr = PartitionResult(
        table="transactions",
        process_date="2026-06-01",
        rows_in=3,
        rows_valid=2,
        rows_quarantined=1,
        duplicates_removed=0,
        rows_curated=2,
        curated_path="data/curated/transactions/process_date=2026-06-01/part.parquet",
        quarantine_path=None,
    )
    summary = RunSummary(
        table="transactions",
        partitions_processed=1,
        rows_curated=2,
        rows_quarantined=1,
        duplicates_removed=0,
        old_watermark=None,
        new_watermark="2026-06-01",
        freshness=freshness,
        dry_run=False,
        partitions=[pr],
    )
    manifest = RunManifest.from_run(
        summary,
        run_id="run_20260601T000000Z",
        ingested_at="2026-06-01T00:00:00Z",
        window_days=3,
        contracts_version=CONTRACTS_VERSION,
        source_fn=lambda t, d: raw_partition_dir(Path("data/raw_parquet"), t, date.fromisoformat(d)),
    )
    out = manifest.to_dict()
    assert out["run_id"] == "run_20260601T000000Z"
    assert out["table"] == "transactions"
    assert out["created_at"] == "2026-06-01T00:00:00Z"
    assert out["window"] == {
        "old_watermark": None,
        "new_watermark": "2026-06-01",
        "window_days": 3,
    }
    assert out["totals"]["rows_curated"] == 2
    assert out["partitions"][0]["source"] == "data/raw_parquet/transactions/year=2026/month=06/day=01"
    assert out["partitions"][0]["curated_path"] == pr.curated_path
    # Fully JSON-serializable.
    json.dumps(out, default=str)
