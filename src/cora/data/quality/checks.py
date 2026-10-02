"""REQ-22 quality checks as DuckDB aggregate SQL through the DataSource.

Every full-table check is one `SELECT ... GROUP BY` / `count(*)` built against the
`DataSource` read expression. Only small aggregates and a handful of example keys are
pulled into Python; a fact table is never materialized in pandas (host RAM ~4 GB). The
checks consume contract metadata (nullability, enum sets, numeric ranges) from
`cora.data.contracts` so bounds live in exactly one place, plus the small registries for
the cross-table relationships, currency allow-sets and range bounds the report needs.

Each check returns a `CheckResult` with numerator AND denominator (not just a rate) and up
to five example offending keys. Facts accept `year`/`month` filters so a check can run over
one bounded partition when `sample_limit` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from cora.data.contracts import get_schema
from cora.data.datasource import FACTS

from .registries import CURRENCY_SETS, FK_MAP, PK_COLUMNS, RANGE_CHECKS
from .types import CheckResult

if TYPE_CHECKING:
    from cora.data.datasource import DataSource

_EXAMPLE_LIMIT = 5


def _quote(col: str) -> str:
    return f'"{col}"'


def _read_expr(source: DataSource, table: str) -> str:
    """Return the engine `read_parquet(...)` expression for `table` (never re-opened)."""
    return source._source(table)


def _partition_predicate(table: str, year: int | None, month: int | None) -> str:
    """Optional Hive partition predicate for a bounded fact read."""
    if table not in FACTS:
        return ""
    preds = [f"{name} = {int(v)}" for name, v in (("year", year), ("month", month)) if v is not None]
    return (" WHERE " + " AND ".join(preds)) if preds else ""


def _relation(
    source: DataSource,
    table: str,
    *,
    year: int | None,
    month: int | None,
    sample_limit: int | None = None,
) -> str:
    """A subquery string selecting the (optionally bounded) table rows.

    For facts in sampled mode the read is bounded to one Hive partition AND capped at
    `sample_limit` rows, so no fact scan ever exceeds the row budget (host RAM ~4 GB).
    Dimensions are small and always read whole.
    """
    where = _partition_predicate(table, year, month)
    limit = ""
    if sample_limit is not None and table in FACTS:
        limit = f" LIMIT {int(sample_limit)}"
    return f"(SELECT * FROM {_read_expr(source, table)}{where}{limit})"


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _examples(con: duckdb.DuckDBPyConnection, sql: str) -> list[object]:
    rows = con.execute(sql).fetchall()
    return [r[0] if len(r) == 1 else list(r) for r in rows]


def _scope(sample_limit: int | None) -> tuple[str, int | None]:
    return ("sampled", sample_limit) if sample_limit is not None else ("full", None)


def _columns_present(source: DataSource, table: str, rel: str) -> set[str]:
    """Column names actually present in the relation (robust to partial test frames)."""
    con = source._connect()
    rows = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()  # noqa: S608
    return {r[0] for r in rows}


def _not_in_subquery(rel: str, col: str, allowed_sql: str) -> str:
    """Subquery yielding non-null `col` values (as `_v`) that are NOT in `allowed_sql`."""
    q = _quote(col)
    return (  # noqa: S608 - rel is engine-built, col is a known contract column
        f"SELECT {q} AS _v FROM {rel} WHERE {q} IS NOT NULL AND {q} NOT IN ({allowed_sql})"
    )


def pk_duplicate_check(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> CheckResult:
    """Rows that repeat an already-seen PK value (composite for daily_exchange_rates)."""
    con = source._connect()
    rel = _relation(source, table, year=year, month=month, sample_limit=sample_limit)
    pk = PK_COLUMNS[table]
    key = ", ".join(_quote(c) for c in pk)
    total = _scalar(con, f"SELECT count(*) FROM {rel}")  # noqa: S608 - rel is engine-built
    distinct = _scalar(con, f"SELECT count(*) FROM (SELECT DISTINCT {key} FROM {rel})")  # noqa: S608
    scope, size = _scope(sample_limit)
    examples = _examples(
        con,
        f"SELECT {key} FROM {rel} GROUP BY {key} HAVING count(*) > 1 LIMIT {_EXAMPLE_LIMIT}",  # noqa: S608
    )
    return CheckResult(
        check="pk_duplicate",
        table=table,
        column=", ".join(pk),
        numerator=max(total - distinct, 0),
        denominator=total,
        examples=examples,
        scope=scope,
        sample_size=size,
        expected="~2% documented; 0% observed (F4)",
    )


def row_hash_duplicate_check(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> CheckResult:
    """Fully duplicated rows via md5 over all contract columns."""
    con = source._connect()
    rel = _relation(source, table, year=year, month=month, sample_limit=sample_limit)
    present = _columns_present(source, table, rel)
    cols = [c for c in get_schema(table).columns if c in present]
    casted = ", ".join(f"CAST({_quote(c)} AS VARCHAR)" for c in cols)
    hashed = f"(SELECT md5(concat_ws('|', {casted})) AS _h FROM {rel})"  # noqa: S608
    total = _scalar(con, f"SELECT count(*) FROM {hashed}")  # noqa: S608
    distinct = _scalar(con, f"SELECT count(DISTINCT _h) FROM {hashed}")  # noqa: S608
    scope, size = _scope(sample_limit)
    examples = _examples(
        con,
        f"SELECT _h FROM {hashed} GROUP BY _h HAVING count(*) > 1 LIMIT {_EXAMPLE_LIMIT}",  # noqa: S608
    )
    return CheckResult(
        check="row_hash_duplicate",
        table=table,
        numerator=max(total - distinct, 0),
        denominator=total,
        examples=examples,
        scope=scope,
        sample_size=size,
        expected="~2% documented; 0% observed (F4)",
    )


def null_rate_checks(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> list[CheckResult]:
    """Null rate per contract column; a null in a NON-nullable column is a violation."""
    con = source._connect()
    rel = _relation(source, table, year=year, month=month, sample_limit=sample_limit)
    schema = get_schema(table)
    present = _columns_present(source, table, rel)
    total = _scalar(con, f"SELECT count(*) FROM {rel}")  # noqa: S608
    scope, size = _scope(sample_limit)
    results: list[CheckResult] = []
    for name, column in schema.columns.items():
        if name not in present:
            continue
        nulls = _scalar(
            con,
            f"SELECT count(*) FROM {rel} WHERE {_quote(name)} IS NULL",  # noqa: S608
        )
        # Non-nullable columns: the violation count is the null count; nullable columns
        # report the null rate informationally (numerator=0 so it is not a violation).
        if column.nullable:
            numerator = 0
            expected = "~5% documented; many structural (F6)"
        else:
            numerator = nulls
            expected = "non-nullable per contract"
        results.append(
            CheckResult(
                check="null_rate",
                table=table,
                column=name,
                numerator=numerator if not column.nullable else nulls,
                denominator=total,
                scope=scope,
                sample_size=size,
                expected=expected if column.nullable else f"non-nullable; {nulls} nulls",
            )
        )
    return results


def orphan_fk_checks(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> list[CheckResult]:
    """Child FK values absent from the parent PK set (anti-join), non-null denominator."""
    relationships = FK_MAP.get(table, [])
    if not relationships:
        return []
    con = source._connect()
    child_rel = _relation(source, table, year=year, month=month, sample_limit=sample_limit)
    scope, size = _scope(sample_limit)
    results: list[CheckResult] = []
    for child_col, parent_table, parent_col in relationships:
        parent_rel = f"(SELECT DISTINCT {_quote(parent_col)} AS _k FROM {_read_expr(source, parent_table)})"
        child = (
            f"(SELECT {_quote(child_col)} AS _v FROM {child_rel} "  # noqa: S608
            f"WHERE {_quote(child_col)} IS NOT NULL)"
        )
        denom = _scalar(con, f"SELECT count(*) FROM {child}")  # noqa: S608
        orphan_sql = f"SELECT c._v FROM {child} c LEFT JOIN {parent_rel} p ON c._v = p._k WHERE p._k IS NULL"
        numerator = _scalar(con, f"SELECT count(*) FROM ({orphan_sql})")  # noqa: S608
        examples = _examples(con, f"{orphan_sql} LIMIT {_EXAMPLE_LIMIT}")  # noqa: S608
        results.append(
            CheckResult(
                check="orphan_fk",
                table=table,
                column=child_col,
                fk=f"{child_col} -> {parent_table}.{parent_col}",
                numerator=numerator,
                denominator=denom,
                examples=examples,
                scope=scope,
                sample_size=size,
                expected="0% observed (F5/§10.1)",
            )
        )
    return results


def _enum_sets(table: str) -> dict[str, list[str]]:
    """Observed enum allow-set per column, read from the contract `isin` checks."""
    out: dict[str, list[str]] = {}
    for name, column in get_schema(table).columns.items():
        for check in column.checks:
            stats = getattr(check, "statistics", None) or {}
            allowed = stats.get("allowed_values")
            if allowed is not None:
                out[name] = list(allowed)
    return out


def enum_violation_checks(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> list[CheckResult]:
    """Values outside the contract's observed enum set (non-null denominator)."""
    enum_cols = _enum_sets(table)
    if not enum_cols:
        return []
    con = source._connect()
    rel = _relation(source, table, year=year, month=month, sample_limit=sample_limit)
    present = _columns_present(source, table, rel)
    scope, size = _scope(sample_limit)
    results: list[CheckResult] = []
    for col, allowed in enum_cols.items():
        if col not in present:
            continue
        quoted = ", ".join("'" + v.replace("'", "''") + "'" for v in allowed)
        denom = _scalar(
            con,
            f"SELECT count(*) FROM {rel} WHERE {_quote(col)} IS NOT NULL",  # noqa: S608
        )
        bad = _not_in_subquery(rel, col, quoted)
        numerator = _scalar(con, f"SELECT count(*) FROM ({bad})")  # noqa: S608
        examples = _examples(con, f"SELECT DISTINCT _v FROM ({bad}) LIMIT {_EXAMPLE_LIMIT}")  # noqa: S608
        results.append(
            CheckResult(
                check="enum_violation",
                table=table,
                column=col,
                numerator=numerator,
                denominator=denom,
                examples=examples,
                scope=scope,
                sample_size=size,
            )
        )
    return results


def _is_temporal(value: object) -> bool:
    return isinstance(value, pd.Timestamp)


def _contract_ranges(table: str) -> dict[str, tuple[object | None, object | None]]:
    """(low, high) bounds introspected from the contract Pandera checks.

    Bounds may be numeric (`credit_score` 300-850) or temporal (`process_date` /
    `date` fact window as `pd.Timestamp`); the SQL builder handles both.
    """
    out: dict[str, tuple[object | None, object | None]] = {}
    for name, column in get_schema(table).columns.items():
        low: object | None = None
        high: object | None = None
        for check in column.checks:
            stats = getattr(check, "statistics", None) or {}
            if "min_value" in stats and "max_value" in stats:
                low, high = stats["min_value"], stats["max_value"]
            elif check.name == "greater_than_or_equal_to" and "min_value" in stats:
                low = stats["min_value"]
            elif check.name == "greater_than" and "min_value" in stats:
                low = stats["min_value"]
        if low is not None or high is not None:
            out[name] = (low, high)
    return out


def _range_bound_sql(col: str, bound: object, op: str) -> str:
    """Build one `col <op> bound` predicate, casting numeric or temporal appropriately."""
    quoted = _quote(col)
    if _is_temporal(bound):
        return f"TRY_CAST({quoted} AS TIMESTAMP) {op} TIMESTAMP '{bound}'"
    return f"TRY_CAST({quoted} AS DOUBLE) {op} {bound}"


def out_of_range_checks(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> list[CheckResult]:
    """Values outside the contract-encoded numeric/temporal ranges (non-null denominator)."""
    ranges = _contract_ranges(table)
    if not ranges:
        return []
    con = source._connect()
    rel = _relation(source, table, year=year, month=month, sample_limit=sample_limit)
    present = _columns_present(source, table, rel)
    scope, size = _scope(sample_limit)
    results: list[CheckResult] = []
    for col, (low, high) in ranges.items():
        if col not in present:
            continue
        conds = []
        if low is not None:
            conds.append(_range_bound_sql(col, low, "<"))
        if high is not None:
            conds.append(_range_bound_sql(col, high, ">"))
        bad_cond = " OR ".join(conds)
        denom = _scalar(
            con,
            f"SELECT count(*) FROM {rel} WHERE {_quote(col)} IS NOT NULL",  # noqa: S608
        )
        bad = f"SELECT {_quote(col)} AS _v FROM {rel} WHERE {_quote(col)} IS NOT NULL AND ({bad_cond})"
        numerator = _scalar(con, f"SELECT count(*) FROM ({bad})")  # noqa: S608
        examples = _examples(con, f"SELECT DISTINCT _v FROM ({bad}) LIMIT {_EXAMPLE_LIMIT}")  # noqa: S608
        results.append(
            CheckResult(
                check="out_of_range",
                table=table,
                column=col,
                numerator=numerator,
                denominator=denom,
                examples=examples,
                scope=scope,
                sample_size=size,
                expected=f"[{low}, {high}]",
            )
        )
    return results


def currency_checks(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> list[CheckResult]:
    """Currency values outside the per-table allowed set (ADR-016/017)."""
    sets = CURRENCY_SETS.get(table, {})
    if not sets:
        return []
    con = source._connect()
    rel = _relation(source, table, year=year, month=month, sample_limit=sample_limit)
    present = _columns_present(source, table, rel)
    scope, size = _scope(sample_limit)
    results: list[CheckResult] = []
    for col, allowed in sets.items():
        if col not in present:
            continue
        quoted = ", ".join("'" + v + "'" for v in sorted(allowed))
        denom = _scalar(
            con,
            f"SELECT count(*) FROM {rel} WHERE {_quote(col)} IS NOT NULL",  # noqa: S608
        )
        bad = _not_in_subquery(rel, col, quoted)
        numerator = _scalar(con, f"SELECT count(*) FROM ({bad})")  # noqa: S608
        examples = _examples(con, f"SELECT DISTINCT _v FROM ({bad}) LIMIT {_EXAMPLE_LIMIT}")  # noqa: S608
        results.append(
            CheckResult(
                check="currency",
                table=table,
                column=col,
                numerator=numerator,
                denominator=denom,
                examples=examples,
                scope=scope,
                sample_size=size,
                expected=f"allowed {sorted(allowed)}",
            )
        )
    return results


def run_checks(
    source: DataSource,
    table: str,
    *,
    year: int | None = None,
    month: int | None = None,
    sample_limit: int | None = None,
) -> list[CheckResult]:
    """Run every REQ-22 check for `table`, returning a flat list of `CheckResult`.

    When `sample_limit` is set and `table` is a fact, callers pass `year`/`month` to bound
    the scan to one partition; the results are labeled `sampled`.
    """
    kw = {"year": year, "month": month, "sample_limit": sample_limit}
    results = [
        pk_duplicate_check(source, table, **kw),
        row_hash_duplicate_check(source, table, **kw),
    ]
    results += null_rate_checks(source, table, **kw)
    results += orphan_fk_checks(source, table, **kw)
    results += enum_violation_checks(source, table, **kw)
    results += out_of_range_checks(source, table, **kw)
    results += currency_checks(source, table, **kw)
    return results


# Re-export the range registry so a test can assert it agrees with the contract.
__all__ = [
    "CheckResult",
    "RANGE_CHECKS",
    "currency_checks",
    "enum_violation_checks",
    "null_rate_checks",
    "orphan_fk_checks",
    "out_of_range_checks",
    "pk_duplicate_check",
    "row_hash_duplicate_check",
    "run_checks",
]
