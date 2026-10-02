"""Run the CORA incremental processing pipeline (task 1.4, REQ-24).

Orchestrates the existing DataSource (1.1), contracts (1.2) and quality/quarantine/dedup
(1.3) one `process_date` partition at a time, with a high-watermark, a reprocessing window
`W`, idempotent curated partition-overwrite writes, and a per-table freshness record. Facts
only; a non-fact table is rejected with a clear error. Memory-safe: one Hive partition is
read at a time (never a whole fact table).

Quality issues are not fatal (bad rows are quarantined, the run continues); the script exits
non-zero only on an operational error, e.g. the raw landing is absent.

Usage:
    uv run python scripts/data/run_pipeline.py --tables transactions --window 3
    uv run python scripts/data/run_pipeline.py --dry-run --tables transactions
    uv run python scripts/data/run_pipeline.py --tables transactions --start-date 2026-06-10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from cora.data.datasource import FACTS, LocalSource
from cora.data.pipeline import RunSummary, run_table


def _format_summary(summary: RunSummary) -> str:
    """One line per table: partitions, curated/quarantined/dedup counts, watermark, freshness."""
    fr = summary.freshness
    mode = " [dry-run]" if summary.dry_run else ""
    return (
        f"{summary.table}{mode}: "
        f"partitions={summary.partitions_processed} "
        f"curated={summary.rows_curated} "
        f"quarantined={summary.rows_quarantined} "
        f"dedup_removed={summary.duplicates_removed} "
        f"watermark {summary.old_watermark} -> {summary.new_watermark} "
        f"| freshness max_event_date={fr.max_event_date} last_ingested_at={fr.last_ingested_at}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the CORA incremental processing pipeline.")
    ap.add_argument("--dest", type=Path, default=Path("data/raw_parquet"), help="Raw landing root.")
    ap.add_argument("--curated", type=Path, default=Path("data/curated"), help="Curated output root.")
    ap.add_argument(
        "--quarantine", type=Path, default=Path("data/quarantine"), help="Quarantine output root."
    )
    ap.add_argument(
        "--state", type=Path, default=Path("data/_state"), help="State dir for watermarks/freshness."
    )
    ap.add_argument(
        "--tables",
        nargs="*",
        default=sorted(FACTS),
        help="Fact tables to process (default: all seven facts).",
    )
    ap.add_argument("--window", type=int, default=3, help="Reprocessing window W in days (default 3).")
    ap.add_argument("--start-date", type=str, default=None, help="ISO start date for the first run.")
    ap.add_argument("--dry-run", action="store_true", help="Discover + count only; write nothing.")
    args = ap.parse_args()

    if not args.dest.exists():
        print(f"error: raw landing not found at {args.dest}", file=sys.stderr)
        return 1

    unknown = [t for t in args.tables if t not in FACTS]
    if unknown:
        print(f"error: not partitioned facts: {unknown}; expected {sorted(FACTS)}", file=sys.stderr)
        return 1

    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    source = LocalSource(root=args.dest)

    for table in args.tables:
        summary = run_table(
            source,
            table,
            root=args.dest,
            curated_root=args.curated,
            quarantine_root=args.quarantine,
            state_dir=args.state,
            window_days=args.window,
            start_date=start_date,
            dry_run=args.dry_run,
        )
        print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
