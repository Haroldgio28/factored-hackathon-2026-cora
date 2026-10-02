"""Shared result types for the data-quality package (task 1.3, REQ-22/REQ-23).

`CheckResult` is the uniform shape every quality check returns: a numerator AND a
denominator (not just a rate), a few example offending keys, and a scope label so the run
report can honestly say whether a number is full-history or sampled. `QualityReport`
aggregates per-table results for the JSON + markdown emitters. All dataclasses are frozen
and expose `.to_dict()` for machine-readable JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckResult:
    """One quality-check outcome for one table (and column/FK where applicable).

    `rate` is `numerator / denominator`, or 0.0 when the denominator is 0 (nothing to
    check is not a violation). `examples` holds up to five offending key values/dicts.
    `scope` is "full" or "sampled"; `sample_size` records the row budget for sampled runs.
    `expected` is a short human note (e.g. "~2% documented") for the report.
    """

    check: str
    table: str
    numerator: int
    denominator: int
    column: str | None = None
    fk: str | None = None
    examples: list[object] = field(default_factory=list)
    scope: str = "full"
    sample_size: int | None = None
    expected: str | None = None

    @property
    def rate(self) -> float:
        return (self.numerator / self.denominator) if self.denominator else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "table": self.table,
            "column": self.column,
            "fk": self.fk,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
            "examples": self.examples,
            "scope": self.scope,
            "sample_size": self.sample_size,
            "expected": self.expected,
        }


@dataclass(frozen=True)
class QualityReport:
    """All check results across the requested tables plus run metadata."""

    results: list[CheckResult]
    tables: list[str]
    scope: str = "full"
    sample_size: int | None = None

    def for_table(self, table: str) -> list[CheckResult]:
        return [r for r in self.results if r.table == table]

    def to_dict(self) -> dict[str, object]:
        return {
            "tables": self.tables,
            "scope": self.scope,
            "sample_size": self.sample_size,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass(frozen=True)
class QuarantineResult:
    """Outcome of splitting a bounded batch into valid vs quarantined rows."""

    table: str
    total: int
    quarantined: int
    valid: int
    out_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "total": self.total,
            "quarantined": self.quarantined,
            "valid": self.valid,
            "out_path": self.out_path,
        }


@dataclass(frozen=True)
class DedupResult:
    """Outcome of deterministic dedup over a bounded batch (REQ-23)."""

    table: str
    kept: int
    removed: int
    tie_break: str
    examples: list[object] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "kept": self.kept,
            "removed": self.removed,
            "tie_break": self.tie_break,
            "examples": self.examples,
        }
