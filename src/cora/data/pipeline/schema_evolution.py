"""Schema-evolution handling for the incremental pipeline (task 1.5, REQ-25).

REQ-25 (verbatim): "WHEN a new nullable column appears, THE pipeline SHALL accept it and log
a schema-change event; WHEN a required column disappears or changes type, THE pipeline SHALL
fail the table with a clear error." (design.md §7: additive nullable columns accepted with an
event; breaking changes fail loudly.)

The check slots into `runner.process_partition()` at the `# seam: 1.5`, BEFORE
dedup/validate. The CONTRACT (`cora.data.contracts.get_schema`) is the single source of truth
for the EXPECTED schema; this module does NOT hard-code a second column list.

Landing-aware comparison semantics (the subtle part — read before changing this)
---------------------------------------------------------------------------------
The raw landing stores EVERY column as a VARCHAR string (EDA finding F1); the contracts set
`coerce=True` and pre-cast via `_checks.normalize_landing`. A physical-dtype comparison of
incoming-vs-contract would therefore flag EVERY column as a "type change" — a false positive
on 100% of the landing. So the diff here is NAME-BASED, and type-safety is delegated to the
already-built coercion/validation path:

- ADDED (incoming column not in the contract): always ADDITIVE-ACCEPTABLE. Fact contracts are
  `strict=False`, so an extra column does NOT fail `contracts.validate` by itself — it flows
  through curation untouched. In a `strict=False` schema an unknown extra column is
  effectively nullable/optional (the contract neither requires nor type-constrains it), which
  directly satisfies REQ-25's "a new nullable column appears → accept + log event". Emit an
  `added` event and CONTINUE.
- MISSING (contract column not in incoming):
    * required column (`col.required is True`, e.g. a PK / NOT-NULL business column) →
      BREAKING. This is REQ-25's "a required column disappears → fail". Raise a clear error
      naming the table + column.
    * optional column (`col.required is False`, e.g. the `year`/`month`/`day` Hive partition
      columns that a `scan` may prune out) → NOT breaking: emit a `removed` event and continue.
- TYPE change: NOT detected here. Because the landing is all-VARCHAR, independent dtype
  comparison is meaningless. REQ-25's "changes type → fail" is realized by the EXISTING
  contract layer: a value whose string form cannot be coerced to the contract's canonical
  dtype fails `contracts.validate` and is quarantined. The `SchemaChangeEvent.kind` enum still
  includes `type_change` so a future curated/typed reference layer (arrow logical types) can
  emit it without a shape change; for 1.5 `type_changes` is always empty from the landing path
  and this is intentional.

Memory safety (host RAM ~4 GB): the check reads only `set(df.columns)` — column NAMES —
never row content. The runner already holds the single partition `df`, so no extra scan.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cora.data import contracts

from . import state
from .types import PIPELINE_VERSION, SchemaChangeEvent

SCHEMA_EVENTS_FILE = state.STATE_DIR / "schema_events.json"


class SchemaEvolutionError(RuntimeError):
    """Raised when a partition's schema has a BREAKING change versus its contract (REQ-25).

    A breaking change (a required contract column missing) must fail the table loudly and
    leave nothing curated (fail-closed); the runner lets this propagate out of
    `process_partition` so no curated partition is written.
    """


@dataclass(frozen=True)
class SchemaCheckResult:
    """Outcome of diffing one partition's columns against its contract (task 1.5, REQ-25)."""

    table: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    type_changes: list[str] = field(default_factory=list)
    events: list[SchemaChangeEvent] = field(default_factory=list)
    is_breaking: bool = False
    breaking_detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "added": list(self.added),
            "removed": list(self.removed),
            "type_changes": list(self.type_changes),
            "events": [e.to_dict() for e in self.events],
            "is_breaking": self.is_breaking,
            "breaking_detail": self.breaking_detail,
        }


def _utc_now_iso() -> str:
    """UTC ISO-8601 timestamp (same format as `runner._utc_now_iso`)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _incoming_columns(incoming: Iterable[str] | pd.DataFrame) -> list[str]:
    """Column NAMES from either a DataFrame (uses `.columns`) or an iterable of names."""
    if isinstance(incoming, pd.DataFrame):
        return [str(c) for c in incoming.columns]
    return [str(c) for c in incoming]


def check_schema(
    table: str,
    incoming: Iterable[str] | pd.DataFrame,
    *,
    process_date: str | None = None,
) -> SchemaCheckResult:
    """Diff a partition's columns against `table`'s contract (name-based, landing-aware).

    `incoming` may be a DataFrame (its `.columns` are used) or an iterable of column names;
    only NAMES are read, never row content. Returns a `SchemaCheckResult`; it NEVER raises on
    a breaking change — the caller decides via `raise_if_breaking`. See the module docstring
    for the landing-aware semantics.
    """
    schema = contracts.get_schema(table)
    expected = schema.columns
    incoming_cols = set(_incoming_columns(incoming))
    expected_cols = set(expected)

    added = sorted(incoming_cols - expected_cols)
    removed = sorted(expected_cols - incoming_cols)

    detected_at = _utc_now_iso()
    events: list[SchemaChangeEvent] = []

    def _event(kind: str, column: str, detail: str) -> SchemaChangeEvent:
        return SchemaChangeEvent(
            table=table,
            kind=kind,
            column=column,
            detail=detail,
            process_date=process_date,
            detected_at=detected_at,
            pipeline_version=PIPELINE_VERSION,
            contracts_version=contracts.CONTRACTS_VERSION,
        )

    for column in added:
        events.append(
            _event(
                "added",
                column,
                f"column {column!r} present in partition, absent from contract "
                f"(strict=False) — accepted as additive nullable",
            )
        )

    breaking_columns: list[str] = []
    for column in removed:
        if expected[column].required:
            breaking_columns.append(column)
        else:
            events.append(
                _event(
                    "removed",
                    column,
                    f"optional contract column {column!r} absent from partition — "
                    f"not breaking (informational)",
                )
            )

    is_breaking = bool(breaking_columns)
    breaking_detail: str | None = None
    if is_breaking:
        breaking_detail = (
            f"schema evolution: table {table!r} is missing required contract column(s) "
            f"{breaking_columns!r} — failing the table (REQ-25)"
        )

    return SchemaCheckResult(
        table=table,
        added=added,
        removed=removed,
        type_changes=[],  # never emitted from the all-VARCHAR landing path (see module docstring)
        events=events,
        is_breaking=is_breaking,
        breaking_detail=breaking_detail,
    )


def raise_if_breaking(result: SchemaCheckResult) -> None:
    """Raise `SchemaEvolutionError` with `result.breaking_detail` when the result is breaking."""
    if result.is_breaking:
        raise SchemaEvolutionError(result.breaking_detail or f"breaking schema change in {result.table!r}")


def load_schema_events(path: Path = SCHEMA_EVENTS_FILE) -> list[SchemaChangeEvent]:
    """Load the schema-events log, returning `[]` when the file is absent/empty."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SchemaChangeEvent.from_dict(item) for item in data]


def append_schema_events(events: list[SchemaChangeEvent], path: Path = SCHEMA_EVENTS_FILE) -> None:
    """Append `events` to the on-disk schema-events log (load-existing + extend + atomic rewrite).

    The on-disk shape is a JSON LIST of event objects. A no-op for an empty `events` list.
    Reuses the atomic tmp-file + `os.replace` pattern from `state.atomic_write_json`; the log
    lives under `data/_state/` (gitignored runtime data, consistent with watermarks/freshness).
    """
    if not events:
        return
    path = Path(path)
    existing = load_schema_events(path)
    existing.extend(events)
    state.atomic_write_json(path, [e.to_dict() for e in existing])


__all__ = [
    "SCHEMA_EVENTS_FILE",
    "SchemaCheckResult",
    "SchemaEvolutionError",
    "append_schema_events",
    "check_schema",
    "load_schema_events",
    "raise_if_breaking",
]
