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
from dataclasses import replace
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
from .schema_evolution import append_schema_events, check_schema, raise_if_breaking
from .types import PIPELINE_VERSION, FreshnessRecord, PartitionResult, RunManifest, RunSummary

CURATED_ROOT = Path("data/curated")
QUARANTINE_ROOT = Path("data/quarantine")
RAW_ROOT = Path("data/raw_parquet")


def _utc_now_iso() -> str:
    """UTC ISO-8601 timestamp, same format `scripts/data/fetch_to_parquet.py` uses."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_run_id() -> str:
    """Filename-safe UTC run id (no `:` — illegal in Windows paths; used in quarantine names)."""
    return time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime())


def _curated_partition_dir(curated_root: Path, table: str, process_date: date) -> Path:
    return Path(curated_root) / table / f"process_date={process_date.isoformat()}"


def raw_partition_dir(root: Path, table: str, process_date: date) -> str:
    """Return the raw Hive partition DIRECTORY as a POSIX string (lineage `_source_file`).

    `<root>/<table>/year=Y/month=MM/day=DD` with zero-padded month/day, matching
    `datasource.LocalSource._source` / `partitions._PARTITION_RE`. This is the real unit the
    pipeline reads (`DataSource.scan` prunes to exactly this directory), so it is the stable
    identifier stamped on `_source_file` and reused as the manifest's per-partition INPUT —
    one helper, so the row stamp and the manifest never drift (task 1.6, REQ-27).
    """
    return (
        Path(root)
        / table
        / f"year={process_date.year}"
        / f"month={process_date.month:02d}"
        / f"day={process_date.day:02d}"
    ).as_posix()


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
    state_dir: Path = state.STATE_DIR,
    run_id: str,
    ingested_at: str | None = None,
    root: Path = RAW_ROOT,
) -> PartitionResult:
    """Orchestrate one `(table, process_date)` partition and write it idempotently.

    Reads exactly one Hive partition (never the whole fact table), validates, quarantines
    bad rows, dedups latest-wins, and writes the curated partition by deleting-then-rewriting
    its directory so a re-run overwrites rather than appends.
    """
    # One UTC ISO timestamp for every row of this call. In a normal run `run_table` computes
    # it ONCE and threads it in so all rows of the run share it; a standalone call falls back
    # to a fresh timestamp here.
    ingested_at = ingested_at or _utc_now_iso()

    # Read exactly one partition (memory-safe: year/month/day prune to a single directory).
    df = source.scan(
        table,
        year=process_date.year,
        month=process_date.month,
        day=process_date.day,
    ).df()
    rows_in = len(df)

    # seam: 1.5 — schema-evolution check (REQ-25), BEFORE dedup/validate and reading only
    # column NAMES (never row content). A BREAKING change (a required contract column missing)
    # raises SchemaEvolutionError, which propagates out so NO curated partition is written
    # (fail-closed). An ADDITIVE change (extra column, or an optional column absent) is logged
    # as a schema-change event and processing CONTINUES — the extra nullable column flows
    # through contract validation (contracts are strict=False). Stage order below is unchanged.
    schema_result = check_schema(table, df, process_date=process_date.isoformat())
    raise_if_breaking(schema_result)
    append_schema_events(schema_result.events, Path(state_dir) / "schema_events.json")

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

    # seam: 1.6 — stamp row-level lineage onto the per-partition frame already in memory
    # (bounded; never a full-table pass), BEFORE the idempotent write (REQ-27). `_source_file`
    # is the raw Hive partition directory this curated partition was read from; `_ingested_at`
    # is the run-constant UTC timestamp (one value shared by all rows of the run);
    # `_pipeline_version` is the committed PIPELINE_VERSION. These are appended as the LAST
    # columns and flow into the overwrite write below. They are run-stamped metadata and are
    # EXCLUDED from the idempotency comparison (see types.LINEAGE_COLUMNS): idempotency is
    # asserted on the business/DATA columns, not on this lineage. `.assign(...)` on a copy so
    # the empty-frame (0 valid rows) case still produces the three columns.
    valid_df = valid_df.assign(
        _source_file=raw_partition_dir(root, table, process_date),
        _ingested_at=ingested_at,
        _pipeline_version=PIPELINE_VERSION,
    )

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
        schema_events=schema_result.events,
        schema_breaking=schema_result.is_breaking,
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
    # One UTC timestamp for the whole run: stamped on every curated row (_ingested_at),
    # recorded in the manifest and reused for the freshness record so all three agree.
    ingested_at = _utc_now_iso()

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
                state_dir=state_dir,
                run_id=run_id,
                ingested_at=ingested_at,
                root=root,
            )
        )

    new_watermark = _advance_watermark(old_watermark, targets)
    freshness = _build_freshness(
        table,
        new_watermark=new_watermark,
        results=results,
        curated_root=curated_root,
        dry_run=dry_run,
        ingested_at=ingested_at,
    )

    if not dry_run and new_watermark is not None:
        watermarks.set(table, new_watermark.isoformat())
        state.save_watermarks(watermarks, watermarks_path)
        all_freshness = state.load_freshness(freshness_path)
        all_freshness[table] = freshness
        state.save_freshness(all_freshness, freshness_path)

    summary = RunSummary(
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
        schema_events=[ev for r in results for ev in r.schema_events],
        schema_breaking=any(r.schema_breaking for r in results),
    )

    # seam: 1.6 — per-run lineage manifest (REQ-27). Assembled from the already-collected
    # RunSummary AFTER the loop, so it touches NO row data (memory-safe by construction). A
    # dry-run writes no curated data and no manifest. The manifest's per-partition INPUT path
    # is computed with the SAME `raw_partition_dir` helper used to stamp `_source_file`, so
    # the rows and the manifest can never drift.
    if not dry_run:
        out_path = state.manifest_path(table, run_id, state_dir)
        manifest = RunManifest.from_run(
            summary,
            run_id=run_id,
            ingested_at=ingested_at,
            window_days=window_days,
            contracts_version=contracts.CONTRACTS_VERSION,
            source_fn=lambda t, d: raw_partition_dir(root, t, date.fromisoformat(d)),
        )
        state.save_manifest(manifest, out_path)
        summary = replace(summary, manifest_path=out_path.as_posix())

    return summary


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
    ingested_at: str,
) -> FreshnessRecord:
    """Assemble the per-table `FreshnessRecord` for this run.

    `ingested_at` is the run-constant timestamp (reused so the freshness record, the manifest
    and every row's `_ingested_at` agree instead of each getting its own wall-clock read).
    """
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
        last_ingested_at=ingested_at,
        curated_rows=sum(r.rows_curated for r in results),
    )


def default_source(root: Path = Path("data/raw_parquet")) -> LocalSource:
    """Convenience `LocalSource` for the CLI (kept out of the pure orchestration path)."""
    return LocalSource(root=root)


__all__ = [
    "CURATED_ROOT",
    "QUARANTINE_ROOT",
    "RAW_ROOT",
    "default_source",
    "process_partition",
    "raw_partition_dir",
    "run_table",
]
