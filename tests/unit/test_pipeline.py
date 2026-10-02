"""Unit tests for the incremental processing pipeline (CORA task 1.4, REQ-24).

TEAM-GENERATED. All synthetic tests write TINY partitioned Parquet into `tmp_path` and point
a `LocalSource` at it, so they run WITHOUT the real landing and stay fast + memory-safe (a
handful of rows per partition). They cover: watermark init + advancement; the reprocessing
window (late row within `watermark - W` picked up next run; a partition older than the window
is NOT reprocessed); idempotency (same partition twice -> identical curated row count AND
content, proving overwrite not append); orchestration (bad rows -> quarantine, good rows
curated, duplicates deduped latest-wins); the freshness record; and watermark/freshness JSON
round-trip. A smoke test dry-runs the real `transactions` landing bounded to one partition and
skips cleanly when the landing is absent (mirrors test_datasource/test_contracts/test_quality).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.data.pipeline import (
    LINEAGE_COLUMNS,
    FreshnessRecord,
    Watermarks,
    available_partitions,
    load_freshness,
    load_watermarks,
    partitions_to_process,
    run_table,
    save_freshness,
    save_watermarks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw_parquet"

local_data = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="local Parquet landing (data/raw_parquet) is absent",
)


# --- Helpers --------------------------------------------------------------------------


def _transactions_frame(ids: list[str], process_date: str, *, amount: str = "10.0") -> pd.DataFrame:
    """A minimal valid raw-string `transactions` frame (mirrors the contract tests)."""
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


# --- Partition discovery + window math (pure) -----------------------------------------


def test_available_partitions_discovers_sorted_unique(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for d in (date(2026, 6, 3), date(2026, 6, 1), date(2026, 6, 2)):
        _write_partition(raw, "transactions", d, _transactions_frame(["T1"], d.isoformat()))
    assert available_partitions(raw, "transactions") == [
        date(2026, 6, 1),
        date(2026, 6, 2),
        date(2026, 6, 3),
    ]


def test_window_first_run_takes_all_or_from_start() -> None:
    avail = [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]
    assert partitions_to_process(avail, None, 3) == avail
    assert partitions_to_process(avail, None, 3, start_date=date(2026, 6, 2)) == avail[1:]


def test_window_reprocesses_within_W_not_older() -> None:
    avail = [date(2026, 6, 5), date(2026, 6, 8), date(2026, 6, 10)]
    watermark = date(2026, 6, 10)
    # W=3 -> lower bound 2026-06-07: 06-08 (within window) reprocessed, 06-05 (older) is not.
    got = partitions_to_process(avail, watermark, 3)
    assert date(2026, 6, 8) in got
    assert date(2026, 6, 5) not in got


# --- Watermark init + advancement -----------------------------------------------------


def test_first_run_processes_all_and_advances_watermark(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i, d in enumerate((date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3))):
        _write_partition(raw, "transactions", d, _transactions_frame([f"T{i}"], d.isoformat()))
    summary = _run(raw, tmp_path)
    assert summary.old_watermark is None
    assert summary.partitions_processed == 3
    assert summary.new_watermark == "2026-06-03"
    # Watermark state was persisted and reloads to the latest processed date.
    wm = load_watermarks(tmp_path / "_state" / "watermarks.json")
    assert wm.get("transactions") == "2026-06-03"


def test_second_run_only_touches_window(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i, d in enumerate((date(2026, 6, 1), date(2026, 6, 5), date(2026, 6, 10))):
        _write_partition(raw, "transactions", d, _transactions_frame([f"T{i}"], d.isoformat()))
    first = _run(raw, tmp_path, window_days=3)
    assert first.new_watermark == "2026-06-10"
    # Late row lands at 2026-06-10 (the watermark) and a new partition appears far in the past.
    _write_partition(raw, "transactions", date(2026, 6, 10), _transactions_frame(["T9"], "2026-06-10"))
    _write_partition(raw, "transactions", date(2026, 1, 1), _transactions_frame(["T0"], "2026-01-01"))
    second = _run(raw, tmp_path, window_days=3)
    processed = {p.process_date for p in second.partitions}
    assert "2026-06-10" in processed  # within window -> reprocessed
    assert "2026-01-01" not in processed  # older than watermark - W -> never reprocessed


def test_late_row_within_window_picked_up_next_run(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_partition(raw, "transactions", date(2026, 6, 8), _transactions_frame(["T1"], "2026-06-08"))
    _write_partition(raw, "transactions", date(2026, 6, 10), _transactions_frame(["T2"], "2026-06-10"))
    _run(raw, tmp_path, window_days=3)
    assert len(_curated_rows(tmp_path / "curated", "transactions", date(2026, 6, 8))) == 1
    # A late row is appended to the raw 2026-06-08 partition (within watermark - W = 06-07).
    _write_partition(
        raw,
        "transactions",
        date(2026, 6, 8),
        _transactions_frame(["T1", "T3"], "2026-06-08"),
    )
    second = _run(raw, tmp_path, window_days=3)
    assert "2026-06-08" in {p.process_date for p in second.partitions}
    assert len(_curated_rows(tmp_path / "curated", "transactions", date(2026, 6, 8))) == 2


# --- Idempotency: re-running a partition overwrites, never appends --------------------


def test_idempotent_partition_overwrite(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    _write_partition(raw, "transactions", d, _transactions_frame(["T1", "T2"], d.isoformat()))
    _run(raw, tmp_path)
    first = _curated_rows(tmp_path / "curated", "transactions", d)
    _run(raw, tmp_path)  # second run of the same window
    second = _curated_rows(tmp_path / "curated", "transactions", d)
    assert len(first) == len(second) == 2
    # The 1.6 lineage columns are stamped on every curated row...
    assert set(LINEAGE_COLUMNS).issubset(first.columns)
    pk = "transaction_id"
    # ...but they are run-stamped metadata (`_ingested_at` is wall-clock) and are EXCLUDED
    # from the idempotency comparison: idempotency is asserted on the business/DATA columns,
    # so re-running overwrites (not appends) and the DATA is byte-identical across runs.
    a = first.drop(columns=list(LINEAGE_COLUMNS)).sort_values(pk).reset_index(drop=True)
    b = second.drop(columns=list(LINEAGE_COLUMNS)).sort_values(pk).reset_index(drop=True)
    assert a.equals(b)  # identical DATA content, not duplicated/appended


# --- Orchestration: quarantine bad rows, dedup latest-wins ----------------------------


def test_orchestration_quarantine_and_dedup(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d = date(2026, 6, 1)
    # Three rows: T1 good, T2 bad enum (-> quarantine), T1 duplicate PK with a later value.
    df = _transactions_frame(["T1", "T2", "T1"], d.isoformat())
    df.loc[1, "transaction_status"] = "Teleported"  # bad enum
    df.loc[2, "amount"] = "20.0"
    df.loc[2, "amount_usd"] = "20.0"
    df.loc[2, "process_date"] = "2026-06-01"
    _write_partition(raw, "transactions", d, df)
    summary = _run(raw, tmp_path)
    pr = summary.partitions[0]
    assert pr.rows_in == 3
    assert pr.rows_quarantined == 1  # bad enum row quarantined
    assert pr.duplicates_removed == 1  # duplicate PK deduped latest-wins
    assert pr.rows_curated == 1  # one unique good row remains
    curated = _curated_rows(tmp_path / "curated", "transactions", d)
    assert len(curated) == 1
    assert curated["transaction_id"].iloc[0] == "T1"
    # The quarantine file exists and names the offending column.
    assert pr.quarantine_path is not None
    written = pd.read_parquet(pr.quarantine_path)
    assert "transaction_status" in written["quarantine_reason"].iloc[0]


# --- Freshness record -----------------------------------------------------------------


def test_freshness_record_fields(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    d1, d2 = date(2026, 6, 1), date(2026, 6, 2)
    # transaction_date differs from process_date so max_event_date is exercised distinctly.
    f1 = _transactions_frame(["T1"], d1.isoformat())
    f1.loc[0, "transaction_date"] = "2026-05-30"
    _write_partition(raw, "transactions", d1, f1)
    _write_partition(raw, "transactions", d2, _transactions_frame(["T2"], d2.isoformat()))
    summary = _run(raw, tmp_path)
    fr = summary.freshness
    assert fr.max_partition_date == "2026-06-02"
    assert fr.watermark == "2026-06-02"
    assert fr.max_event_date == "2026-06-02"  # max transaction_date in the latest partition
    # last_ingested_at parses as ISO-8601 (ends in Z).
    assert fr.last_ingested_at.endswith("Z")
    pd.Timestamp(fr.last_ingested_at)


# --- State JSON round-trip ------------------------------------------------------------


def test_watermark_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "_state" / "watermarks.json"
    wm = Watermarks({"transactions": "2026-06-03"})
    save_watermarks(wm, path)
    assert load_watermarks(path).to_dict() == {"transactions": "2026-06-03"}
    # Absent file initialises empty (init-when-absent).
    assert load_watermarks(tmp_path / "nope.json").to_dict() == {}


def test_freshness_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "_state" / "freshness.json"
    rec = FreshnessRecord(
        table="transactions",
        max_partition_date="2026-06-03",
        max_event_date="2026-06-03",
        watermark="2026-06-03",
        last_ingested_at="2026-06-03T00:00:00Z",
        curated_rows=5,
    )
    save_freshness({"transactions": rec}, path)
    loaded = load_freshness(path)
    assert loaded["transactions"] == rec


# --- Smoke test over a bounded real partition (dry-run) -------------------------------


@local_data
def test_dry_run_real_transactions_single_partition_bounded(tmp_path: Path) -> None:
    source = LocalSource(root=DATA_ROOT)
    summary = run_table(
        source,
        "transactions",
        root=DATA_ROOT,
        curated_root=tmp_path / "curated",
        quarantine_root=tmp_path / "quarantine",
        state_dir=tmp_path / "_state",
        start_date=date(2026, 6, 17),  # bound to the last available day only
        dry_run=True,
    )
    assert summary.dry_run is True
    assert summary.partitions_processed >= 1
    assert all(p.rows_in >= 0 for p in summary.partitions)
    # dry-run writes nothing.
    assert not (tmp_path / "_state" / "watermarks.json").exists()
    assert not (tmp_path / "curated").exists()
