"""Run the CORA data-quality checks and write the committed evidence (task 1.3).

Builds a `LocalSource` over the raw Parquet landing, runs every REQ-22 quality check per
table through the `DataSource` (DuckDB aggregate SQL — no fact table is materialized in
pandas), and writes BOTH `documentation/reports/data_quality.json` and
`documentation/reports/data_quality.md`. Fact tables are bounded to one Hive partition when
`--sample-limit` is set (default), so the run stays within the host's ~4 GB RAM budget.

Quality issues are REPORTED, not fatal: the script exits non-zero only on an operational
error (e.g. the landing is absent). Read-only on the data.

Usage:
    uv run python scripts/data/quality_report.py --dest data/raw_parquet --out documentation/reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cora.data.datasource import TABLES, LocalSource
from cora.data.quality import build_report, report_to_json, report_to_markdown


def main() -> int:
    ap = argparse.ArgumentParser(description="Run CORA data-quality checks and emit the report.")
    ap.add_argument("--dest", type=Path, default=Path("data/raw_parquet"))
    ap.add_argument("--out", type=Path, default=Path("documentation/reports"))
    ap.add_argument(
        "--sample-limit",
        type=int,
        default=50_000,
        help="Row cap per fact table (bounded to one Hive partition, memory-safe). "
        "Use 0 for full-history scans of fact tables.",
    )
    ap.add_argument("--year", type=int, default=None, help="Fact partition year for sampled runs.")
    ap.add_argument("--month", type=int, default=None, help="Fact partition month for sampled runs.")
    ap.add_argument(
        "--tables",
        nargs="*",
        default=sorted(TABLES),
        help="Tables to check (default: all 13).",
    )
    args = ap.parse_args()

    if not args.dest.exists():
        print(f"error: landing not found at {args.dest}", file=sys.stderr)
        return 1

    sample_limit = None if args.sample_limit == 0 else args.sample_limit
    source = LocalSource(root=args.dest)
    report = build_report(
        source, list(args.tables), sample_limit=sample_limit, year=args.year, month=args.month
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_quality.json").write_text(
        json.dumps(report_to_json(report), indent=1, default=str), encoding="utf-8"
    )
    markdown = report_to_markdown(report)
    (args.out / "data_quality.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
