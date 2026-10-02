"""Partition discovery + reprocessing-window math (task 1.4, REQ-24).

`process_date` IS the Hive partition date. The raw landing partitions each fact under
`<table>/year=YYYY/month=MM/day=DD/<table>_YYYYMMDD.parquet` and also carries a row-level
`process_date` column; verified against the local landing, the two are the same calendar
day (one distinct `process_date` per partition). The pipeline therefore keys the watermark
on the Hive partition date (D1 in the plan): it cannot be corrupted by a bad row value and
`DataSource.scan(table, year=, month=, day=)` prunes to exactly one partition, which is what
keeps the pipeline memory-safe.

Partition discovery is a directory listing only — it never reads Parquet content, so it is
cheap and memory-free. Hive dtypes observed in the landing: `year` is int, `month`/`day` are
zero-padded VARCHAR; the directory-name regex tolerates both by parsing with `int(...)`
(mirrors `scripts/data/validate_landing.DAY_RE`).

Reprocessing window W semantics: on each run the pipeline reprocesses partitions from
`watermark - W days` through the latest available partition, so late-arriving rows that land
within the window are absorbed on the next run. Partitions older than `watermark - W` are
never reprocessed; the watermark only advances, never regresses.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

from cora.data.datasource import FACTS

# Matches the Hive partition path `.../year=YYYY/month=MM/day=DD` (month/day zero-padded).
_PARTITION_RE = re.compile(r"year=(\d{4})/month=(\d{1,2})/day=(\d{1,2})")

# Business event-date column per fact where it differs from `process_date`. The watermark is
# ALWAYS keyed on the partition date (D1); this map only feeds
# `FreshnessRecord.max_event_date`, computed with a bounded DuckDB `max(...)` aggregate over
# the curated partition — never by loading rows into pandas. `None` means the fact has no
# business event date distinct from `process_date` (then max_event_date == max_partition_date).
EVENT_DATE_COLUMN: dict[str, str | None] = {
    "transactions": "transaction_date",
    "call_center_interactions": "interaction_date",
    "call_transcripts": None,
    "satisfaction_surveys": "survey_date",
    "digital_events": "event_date",
    "complaints": "creation_date",
    "campaign_sends": "send_date",
}


def available_partitions(root: Path, table: str) -> list[date]:
    """Return the sorted unique Hive partition dates present for `table` under `root`.

    Directory listing only — no Parquet content is read. Tolerates int `year` and
    zero-padded VARCHAR `month`/`day` by parsing the directory names with `int(...)`.
    """
    table_root = Path(root) / table
    if not table_root.exists():
        return []
    dates: set[date] = set()
    for day_dir in table_root.glob("year=*/month=*/day=*"):
        if not day_dir.is_dir():
            continue
        match = _PARTITION_RE.search(day_dir.as_posix())
        if match:
            year, month, day = (int(g) for g in match.groups())
            dates.add(date(year, month, day))
    return sorted(dates)


def partitions_to_process(
    available: list[date],
    watermark: date | None,
    window_days: int,
    start_date: date | None = None,
) -> list[date]:
    """Return the partition dates to (re)process this run, oldest first.

    First run (`watermark is None`): every available partition, optionally filtered to those
    `>= start_date`. Incremental run: every available partition `>= watermark - window_days`
    up to the latest available, so late arrivals within the window `W` are absorbed and
    partitions older than `watermark - W` are never reprocessed.
    """
    if watermark is None:
        if start_date is not None:
            return [d for d in available if d >= start_date]
        return list(available)
    lower_bound = watermark - timedelta(days=window_days)
    return [d for d in available if d >= lower_bound]


def validate_fact(table: str) -> None:
    """Reject a table that is not a Hive-partitioned fact (task 1.4 targets facts only)."""
    if table not in FACTS:
        raise ValueError(
            f"pipeline table {table!r} is not a partitioned fact; expected one of {sorted(FACTS)}"
        )
