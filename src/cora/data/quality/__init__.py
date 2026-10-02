"""CORA data-quality package (task 1.3): checks + quarantine + dedup + run report.

Builds ON task 1.2 contracts (`cora.data.contracts.validate`) and the `DataSource`
abstraction. It does NOT re-implement contract validation and does NOT build the
incremental watermark (1.4), schema-evolution (1.5) or lineage/manifest (1.6) layers.

- REQ-22 quality checks (`run_checks` and the individual check functions) run as DuckDB
  aggregate SQL through the `DataSource`; a fact table is never materialized in pandas.
- REQ-21 quarantine (`quarantine_batch`) consumes the contract `failure_cases`, splits a
  bounded batch into valid vs quarantined with a per-row reason, writes the quarantined rows
  to Parquet, and lets the pipeline continue.
- REQ-23 dedup (`deduplicate`) keeps the latest record per PK over a bounded batch and
  records how many duplicates were removed.
- The run report (`build_report`, `report_to_json`, `report_to_markdown`) emits the
  committed evidence at `documentation/reports/data_quality.{md,json}`.

Public API is re-exported here.
"""

from __future__ import annotations

from .checks import (
    currency_checks,
    enum_violation_checks,
    null_rate_checks,
    orphan_fk_checks,
    out_of_range_checks,
    pk_duplicate_check,
    row_hash_duplicate_check,
    run_checks,
)
from .dedup import deduplicate
from .quarantine import quarantine_batch
from .registries import (
    ARRIVAL_COLUMN,
    CURRENCY_SETS,
    FK_MAP,
    PK_COLUMNS,
    RANGE_CHECKS,
)
from .report import build_report, report_to_json, report_to_markdown
from .types import CheckResult, DedupResult, QualityReport, QuarantineResult

__all__ = [
    "ARRIVAL_COLUMN",
    "CURRENCY_SETS",
    "FK_MAP",
    "PK_COLUMNS",
    "RANGE_CHECKS",
    "CheckResult",
    "DedupResult",
    "QualityReport",
    "QuarantineResult",
    "build_report",
    "currency_checks",
    "deduplicate",
    "enum_violation_checks",
    "null_rate_checks",
    "orphan_fk_checks",
    "out_of_range_checks",
    "pk_duplicate_check",
    "quarantine_batch",
    "report_to_json",
    "report_to_markdown",
    "row_hash_duplicate_check",
    "run_checks",
]
