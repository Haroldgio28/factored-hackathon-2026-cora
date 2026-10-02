"""Per-partition orchestration + per-table run loop (task 1.4, REQ-24).

This module ORCHESTRATES the existing pieces one `process_date` partition at a time; it does
NOT re-implement validation, quarantine or dedup:

    DataSource.scan(table, year=, month=, day=).df()   # one partition only (memory-safe)
      -> quality.deduplicate(table, df)                # task 1.3 (latest-wins, collapse redeliveries)
      -> contracts.validate(table, deduped)            # task 1.2
      -> quality.quarantine_batch(...)                 # task 1.3 (genuinely-bad rows -> quarantine)
      -> idempotent curated partition-overwrite write  # task 1.4

Stage order is dedup -> validate -> quarantine (design §7 / REQ-23): a duplicate re-delivery
is a repeated arrival to COLLAPSE (latest-wins), not a bad row to quarantine, so dedup runs
first on the raw batch. The PK-uniqueness contract then validates a batch that already has
unique PKs; a genuine non-redelivery uniqueness violation still surfaces and is quarantined.

Memory safety (host RAM ~4 GB): every fact read is bounded to a single Hive partition via
`source.scan(table, year=, month=, day=)` before `.df()`; a whole fact table is never
materialized. `max_event_date` is computed with a DuckDB `max(...)` aggregate over the
just-written curated partition, not by loading rows. Partition discovery is a directory
listing only.

Idempotency (the core correctness property): for a given `(table, process_date)` the curated
output is a single file `data/curated/<table>/process_date=YYYY-MM-DD/part.parquet`.
Re-running a partition deletes that partition directory and rewrites it (overwrite, never
append). Because the raw partition is immutable and validate -> quarantine -> dedup are
deterministic, the same input yields identical curated row count and content on every run.

Seams for later tasks are marked inline:
- `# seam: 1.5` — schema-evolution check, before `contracts.validate`.
- `# seam: 1.6` — lineage stamping (`_source_file`/`_ingested_at`/`_pipeline_version`) before
  the curated write, and a run-manifest JSON writer consuming the returned `RunSummary`.
"""

from __future__ import annotations

import shutil
import time
from datetime import date
from pathlib import Path

import duckdb

from cora.data import contracts, quality
from cora.data.datasource import DataSource, LocalSource

from . import state
from .partitions import (
    EVENT_DATE_COLUMN,
    available_partitions,
    partitions_to_process,
    validate_fact,
)
from .types import FreshnessRecord, PartitionResult, RunSummary

CURATED_ROOT = Path("data/curated")
QUARANTINE_ROOT = Path("data/quarantine")


def _utc_now_iso() -> str:
    """UTC ISO-8601 timestamp, same format `scripts/data/fetch_to_parquet.py` uses."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_run_id() -> str:
    """Filename-safe UTC run id (no `:` — illegal in Windows paths; used in quarantine names)."""
    return time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime())


def _curated_partition_dir(curated_root: Path, table: str, process_date: date) -> Path:
    return Path(curated_root) / table / f"process_date={process_date.isoformat()}"


def _max_event_date(curated_path: Path, table: str) -> str | None:
    """Return the max business event date in a curated partition via a bounded aggregate.

    Uses the `EVENT_DATE_COLUMN` entry for the table. Reads only a single `max(...)` scalar
    through DuckDB — the rows are never loaded into pandas. Returns `None` when the table has
    no distinct event-date column or the partition is empty.
    """
    column = EVENT_DATE_COLUMN.get(table)
    if column is None:
        return None
    path = curated_path.as_posix()
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='512MB'")
        row = con.execute(
            f"SELECT max({column}) FROM read_parquet(?)",  # noqa: S608 - column from fixed registry
            [path],
        ).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    value = row[0]
    # DuckDB returns a datetime/date; normalise to an ISO date string.
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]


def process_partition(
    source: DataSource,
    table: str,
    process_date: date,
    *,
    curated_root: Path = CURATED_ROOT,
    quarantine_root: Path = QUARANTINE_ROOT,
    run_id: str,
) -> PartitionResult:
    """Orchestrate one `(table, process_date)` partition and write it idempotently.

    Reads exactly one Hive partition (never the whole fact table), validates, quarantines
    bad rows, dedups latest-wins, and writes the curated partition by deleting-then-rewriting
    its directory so a re-run overwrites rather than appends.
    """
    # Read exactly one partition (memory-safe: year/month/day prune to a single directory).
    df = source.scan(
        table,
        year=process_date.year,
        month=process_date.month,
        day=process_date.day,
    ).df()
    rows_in = len(df)

    # seam: 1.5 — schema_check(table, df) will slot in here, before contract validation.

    # Dedup FIRST (design §7 / REQ-23): collapse duplicate-redeliveries latest-wins on the
    # raw batch before validation, so a repeated arrival is deduped (not quarantined) and the
    # PK-uniqueness contract then validates a batch that already has unique PKs — a genuine
    # non-redelivery uniqueness violation still surfaces as a failure and is quarantined.
    deduped, dres = quality.deduplicate(table, df)

    # validate() returns the COERCED valid rows (curated output) plus Pandera failure cases.
    # quarantine_batch() is handed the FULL deduped batch + those failure cases so it WRITES
    # the bad rows with a reason (handing it valid_df would silently drop them instead). It
    # returns the valid split; curated_count == len(valid_df) == len(kept).
    valid_df, failure_cases = contracts.validate(table, deduped)
    _kept, qres = quality.quarantine_batch(
        table, deduped, failure_cases, out_dir=quarantine_root, run_id=run_id
    )

    # seam: 1.6 — stamp _source_file / _ingested_at / _pipeline_version here, before writing.

    out_dir = _curated_partition_dir(curated_root, table, process_date)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # idempotent overwrite: drop the whole partition dir first
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "part.parquet"
    valid_df.to_parquet(out_file, index=False)

    return PartitionResult(
        table=table,
        process_date=process_date.isoformat(),
        rows_in=rows_in,
        rows_valid=qres.valid,
        rows_quarantined=qres.quarantined,
        duplicates_removed=dres.removed,
        rows_curated=len(valid_df),
        curated_path=out_file.as_posix(),
        quarantine_path=qres.out_path,
    )


def run_table(
    source: DataSource,
    table: str,
    *,
    root: Path = Path("data/raw_parquet"),
    curated_root: Path = CURATED_ROOT,
    quarantine_root: Path = QUARANTINE_ROOT,
    state_dir: Path = state.STATE_DIR,
    window_days: int = 3,
    start_date: date | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
) -> RunSummary:
    """Run the incremental pipeline for one fact table and persist watermark + freshness.

    Loads the watermark, discovers available partitions, computes the reprocessing window,
    processes each partition in order, advances the watermark (only forward), builds the
    freshness record, and persists both state files (unless `dry_run`). Returns a
    `RunSummary` with the per-partition results.
    """
    validate_fact(table)
    run_id = run_id or _default_run_id()

    watermarks_path = Path(state_dir) / "watermarks.json"
    freshness_path = Path(state_dir) / "freshness.json"
    watermarks = state.load_watermarks(watermarks_path)
    old_wm_str = watermarks.get(table)
    old_watermark = date.fromisoformat(old_wm_str) if old_wm_str else None

    available = available_partitions(root, table)
    targets = partitions_to_process(available, old_watermark, window_days, start_date)

    results: list[PartitionResult] = []
    for process_date in targets:
        if dry_run:
            rows_in = source.count(
                table, year=process_date.year, month=process_date.month, day=process_date.day
            )
            results.append(
                PartitionResult(
                    table=table,
                    process_date=process_date.isoformat(),
                    rows_in=rows_in,
                    rows_valid=0,
                    rows_quarantined=0,
                    duplicates_removed=0,
                    rows_curated=0,
                )
            )
            continue
        results.append(
            process_partition(
                source,
                table,
                process_date,
                curated_root=curated_root,
                quarantine_root=quarantine_root,
                run_id=run_id,
            )
        )

    new_watermark = _advance_watermark(old_watermark, targets)
    freshness = _build_freshness(
        table,
        new_watermark=new_watermark,
        results=results,
        curated_root=curated_root,
        dry_run=dry_run,
    )

    if not dry_run and new_watermark is not None:
        watermarks.set(table, new_watermark.isoformat())
        state.save_watermarks(watermarks, watermarks_path)
        all_freshness = state.load_freshness(freshness_path)
        all_freshness[table] = freshness
        state.save_freshness(all_freshness, freshness_path)

    return RunSummary(
        table=table,
        partitions_processed=len(results),
        rows_curated=sum(r.rows_curated for r in results),
        rows_quarantined=sum(r.rows_quarantined for r in results),
        duplicates_removed=sum(r.duplicates_removed for r in results),
        old_watermark=old_wm_str,
        new_watermark=new_watermark.isoformat() if new_watermark else old_wm_str,
        freshness=freshness,
        dry_run=dry_run,
        partitions=results,
    )


def _advance_watermark(old: date | None, targets: list[date]) -> date | None:
    """Return the new watermark: max of old and the latest processed partition date."""
    if not targets:
        return old
    latest = max(targets)
    return latest if old is None else max(old, latest)


def _build_freshness(
    table: str,
    *,
    new_watermark: date | None,
    results: list[PartitionResult],
    curated_root: Path,
    dry_run: bool,
) -> FreshnessRecord:
    """Assemble the per-table `FreshnessRecord` for this run."""
    wm_str = new_watermark.isoformat() if new_watermark else None
    max_event_date = wm_str
    if not dry_run and new_watermark is not None:
        curated_path = _curated_partition_dir(curated_root, table, new_watermark) / "part.parquet"
        if curated_path.exists():
            event = _max_event_date(curated_path, table)
            max_event_date = event if event is not None else wm_str
    return FreshnessRecord(
        table=table,
        max_partition_date=wm_str,
        max_event_date=max_event_date,
        watermark=wm_str,
        last_ingested_at=_utc_now_iso(),
        curated_rows=sum(r.rows_curated for r in results),
    )


def default_source(root: Path = Path("data/raw_parquet")) -> LocalSource:
    """Convenience `LocalSource` for the CLI (kept out of the pure orchestration path)."""
    return LocalSource(root=root)


__all__ = [
    "CURATED_ROOT",
    "QUARANTINE_ROOT",
    "default_source",
    "process_partition",
    "run_table",
]
