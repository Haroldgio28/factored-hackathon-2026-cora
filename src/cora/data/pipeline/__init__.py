"""CORA incremental processing pipeline (task 1.4, REQ-24).

This package ORCHESTRATES the already-built pieces one `process_date` partition at a time;
it does NOT re-implement them and does NOT implement 1.5 (schema-evolution detection) or 1.6
(row-level lineage columns + run-manifest JSON) — the runner leaves clearly-marked seams for
both.

Design summary:
- `process_date` IS the Hive partition date. The raw landing partitions each fact under
  `year=YYYY/month=MM/day=DD` and also carries a row-level `process_date` column; the two are
  the same calendar day (verified). The watermark is keyed on the partition date because it
  prunes to a single directory (memory-safe) and cannot be corrupted by a bad row value.
- High-watermark: per table, the latest successfully-curated `process_date`, persisted in
  `data/_state/watermarks.json`. Absent -> first run processes all available partitions (or
  from a configurable start date).
- Reprocessing window W (default 3 days): each run reprocesses partitions from
  `watermark - W` through the latest available, so late arrivals within the window are
  absorbed. Partitions older than `watermark - W` are never reprocessed; the watermark only
  advances.
- Per partition: `DataSource.scan` (one partition) -> dedup latest-wins (collapse
  redeliveries) -> `contracts.validate` -> quarantine genuinely-bad rows -> idempotent
  curated partition-overwrite write (design §7 / REQ-23: a duplicate re-delivery is deduped,
  not quarantined).
- Idempotency: a curated partition is `data/curated/<table>/process_date=YYYY-MM-DD/part.parquet`;
  re-running deletes-then-rewrites that directory, so the same input yields identical output.
- Freshness: per table (`max_partition_date`, `max_event_date`, `watermark`,
  `last_ingested_at`, `curated_rows`) persisted in `data/_state/freshness.json`.

`data/` is gitignored: the state/curated/quarantine FILES are runtime data; this CODE and the
record SCHEMA (`types.py`) are the committed artefact.
"""

from __future__ import annotations

from .partitions import (
    EVENT_DATE_COLUMN,
    available_partitions,
    partitions_to_process,
)
from .runner import process_partition, run_table
from .state import (
    load_freshness,
    load_watermarks,
    save_freshness,
    save_watermarks,
)
from .types import (
    PIPELINE_VERSION,
    FreshnessRecord,
    PartitionResult,
    RunSummary,
    Watermarks,
)

__all__ = [
    "EVENT_DATE_COLUMN",
    "PIPELINE_VERSION",
    "FreshnessRecord",
    "PartitionResult",
    "RunSummary",
    "Watermarks",
    "available_partitions",
    "load_freshness",
    "load_watermarks",
    "partitions_to_process",
    "process_partition",
    "run_table",
    "save_freshness",
    "save_watermarks",
]
