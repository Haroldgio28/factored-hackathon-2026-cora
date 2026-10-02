"""Typed records for the incremental processing pipeline (task 1.4, REQ-24).

These frozen dataclasses are the committed SCHEMA for the pipeline's runtime state and run
summary. They mirror the `quality/types.py` convention (frozen dataclasses with a
`.to_dict()` for machine-readable JSON) so later tasks (tools, agent, eval) can consume the
watermark/freshness state and the run summary without re-deriving the shapes.

Record shapes:
- `PartitionResult` — the per-`process_date` orchestration outcome (counts + output paths).
- `FreshnessRecord` — one per table: the max partition date processed, the max business
  event date, the watermark, the UTC ingest timestamp and the curated row count. Persisted
  in `data/_state/freshness.json`.
- `Watermarks` — a typed wrapper over `{table: ISO-date}`, the high-watermark per table.
  Persisted in `data/_state/watermarks.json`.
- `RunSummary` — everything one `run_table(...)` produced, including the list of
  `PartitionResult`, so the CLI can print a summary and 1.6 can later write a run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Pipeline schema/behaviour version. Defined now so the 1.6 lineage seam can stamp it; it
# also lets later readers detect a state file written by an older pipeline.
PIPELINE_VERSION = "1.0"


@dataclass(frozen=True)
class SchemaChangeEvent:
    """One accepted (non-breaking) schema-change observation on a partition (task 1.5, REQ-25).

    Emitted by `schema_evolution.check_schema` when an additive column appears (kind=`added`)
    or an OPTIONAL contract column is absent (kind=`removed`). The `type_change` kind is
    defined for a future curated/typed reference layer but is NEVER emitted from the raw
    all-VARCHAR landing path (type incompatibility is handled by `contracts.validate` /
    quarantine); see `schema_evolution` for the landing-aware rationale.

    Persisted append-style to `data/_state/schema_events.json` (runtime data, gitignored).
    """

    table: str
    kind: str  # "added" | "removed" | "type_change"
    column: str
    detail: str
    process_date: str | None
    detected_at: str
    pipeline_version: str
    contracts_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "kind": self.kind,
            "column": self.column,
            "detail": self.detail,
            "process_date": self.process_date,
            "detected_at": self.detected_at,
            "pipeline_version": self.pipeline_version,
            "contracts_version": self.contracts_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SchemaChangeEvent:
        """Rebuild an event from its JSON object (inverse of `to_dict`)."""
        return cls(
            table=str(data["table"]),
            kind=str(data["kind"]),
            column=str(data["column"]),
            detail=str(data["detail"]),
            process_date=_opt_str(data.get("process_date")),
            detected_at=str(data["detected_at"]),
            pipeline_version=str(data["pipeline_version"]),
            contracts_version=str(data["contracts_version"]),
        )


@dataclass(frozen=True)
class PartitionResult:
    """Outcome of orchestrating one `(table, process_date)` partition."""

    table: str
    process_date: str
    rows_in: int
    rows_valid: int
    rows_quarantined: int
    duplicates_removed: int
    rows_curated: int
    curated_path: str | None = None
    quarantine_path: str | None = None
    schema_events: list[SchemaChangeEvent] = field(default_factory=list)
    schema_breaking: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "process_date": self.process_date,
            "rows_in": self.rows_in,
            "rows_valid": self.rows_valid,
            "rows_quarantined": self.rows_quarantined,
            "duplicates_removed": self.duplicates_removed,
            "rows_curated": self.rows_curated,
            "curated_path": self.curated_path,
            "quarantine_path": self.quarantine_path,
            "schema_events": [e.to_dict() for e in self.schema_events],
            "schema_breaking": self.schema_breaking,
        }


@dataclass(frozen=True)
class FreshnessRecord:
    """Per-table freshness: how current the curated data is after a run.

    `max_partition_date` is the max Hive partition date (`process_date`) that is curated.
    `max_event_date` is the max of the table's business event-date column when one exists
    distinct from `process_date` (see `partitions.EVENT_DATE_COLUMN`), else equal to
    `max_partition_date`. `watermark` is the high-watermark after the run (ISO date).
    `last_ingested_at` is a UTC ISO-8601 timestamp. `curated_rows` is the total curated row
    count across the partitions processed in the run.
    """

    table: str
    max_partition_date: str | None
    max_event_date: str | None
    watermark: str | None
    last_ingested_at: str
    curated_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "max_partition_date": self.max_partition_date,
            "max_event_date": self.max_event_date,
            "watermark": self.watermark,
            "last_ingested_at": self.last_ingested_at,
            "curated_rows": self.curated_rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FreshnessRecord:
        """Rebuild a record from its JSON object (inverse of `to_dict`)."""
        return cls(
            table=str(data["table"]),
            max_partition_date=_opt_str(data.get("max_partition_date")),
            max_event_date=_opt_str(data.get("max_event_date")),
            watermark=_opt_str(data.get("watermark")),
            last_ingested_at=str(data["last_ingested_at"]),
            curated_rows=int(data["curated_rows"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Watermarks:
    """High-watermark per table: the latest successfully-curated `process_date` (ISO date).

    A missing table means "never processed" (first run processes all available partitions,
    or from a configurable start date). The on-disk JSON is a flat `{table: ISO-date}`
    object so later tasks can read it trivially.
    """

    values: dict[str, str] = field(default_factory=dict)

    def get(self, table: str) -> str | None:
        return self.values.get(table)

    def set(self, table: str, iso_date: str) -> None:
        self.values[table] = iso_date

    def to_dict(self) -> dict[str, str]:
        return dict(self.values)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Watermarks:
        return cls(values={str(k): str(v) for k, v in data.items()})


@dataclass(frozen=True)
class RunSummary:
    """Everything one `run_table(...)` call produced, for the CLI and the 1.6 manifest."""

    table: str
    partitions_processed: int
    rows_curated: int
    rows_quarantined: int
    duplicates_removed: int
    old_watermark: str | None
    new_watermark: str | None
    freshness: FreshnessRecord
    dry_run: bool
    partitions: list[PartitionResult] = field(default_factory=list)
    schema_events: list[SchemaChangeEvent] = field(default_factory=list)
    schema_breaking: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "partitions_processed": self.partitions_processed,
            "rows_curated": self.rows_curated,
            "rows_quarantined": self.rows_quarantined,
            "duplicates_removed": self.duplicates_removed,
            "old_watermark": self.old_watermark,
            "new_watermark": self.new_watermark,
            "freshness": self.freshness.to_dict(),
            "dry_run": self.dry_run,
            "partitions": [p.to_dict() for p in self.partitions],
            "schema_events": [e.to_dict() for e in self.schema_events],
            "schema_breaking": self.schema_breaking,
        }


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)
