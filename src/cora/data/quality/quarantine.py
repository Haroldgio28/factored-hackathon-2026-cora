"""Quarantine: split a bounded batch into valid vs quarantined rows (task 1.3, REQ-21).

This consumes the `failure_cases` frame returned by `cora.data.contracts.validate()`; it
does NOT re-implement contract validation. Rows that any contract check failed are written
to `data/quarantine/<table>/<run_id>.parquet` with an inline `quarantine_reason` naming the
offending column/check/value, so the row and its reason stay joined. Valid rows are returned
untouched for the dedup step. The function never raises on bad data: an all-bad batch writes
everything to quarantine and returns zero valid rows; an all-good batch writes nothing. The
pipeline MUST continue after quarantining.

`data/` and `*.parquet` are gitignored — the written Parquet is runtime output, this code is
the committed artifact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .types import QuarantineResult

_MAX_REASONS_PER_ROW = 3


def _reason_for_row(group: pd.DataFrame) -> str:
    """Build a compact reason string from this row's failure cases."""
    parts: list[str] = []
    for _, case in group.head(_MAX_REASONS_PER_ROW).iterrows():
        column = case.get("column")
        check = case.get("check")
        value = case.get("failure_case")
        col_txt = f"column={column}" if pd.notna(column) else "column=<frame>"
        parts.append(f"{col_txt} check={check} value={value}")
    if len(group) > _MAX_REASONS_PER_ROW:
        parts.append(f"(+{len(group) - _MAX_REASONS_PER_ROW} more)")
    return "; ".join(parts)


def quarantine_batch(
    table: str,
    df: pd.DataFrame,
    failure_cases: pd.DataFrame,
    *,
    out_dir: Path = Path("data/quarantine"),
    run_id: str,
) -> tuple[pd.DataFrame, QuarantineResult]:
    """Split `df` into (valid_df, quarantine_result), writing quarantined rows to Parquet.

    `failure_cases` is the Pandera failure-cases frame from `contracts.validate(table, df)`
    (columns: schema_context, column, check, check_number, failure_case, index). Rows whose
    `.index` label appears in `failure_cases["index"]` are quarantined with a reason; the
    rest are returned as the valid batch. Returns the valid DataFrame plus a
    `QuarantineResult`. Tolerates an empty `failure_cases` (writes nothing).
    """
    total = len(df)
    if failure_cases is None or failure_cases.empty:
        return df, QuarantineResult(table=table, total=total, quarantined=0, valid=total, out_path=None)

    cases = failure_cases.dropna(subset=["index"])
    bad_index = pd.Index(cases["index"].unique())
    present = df.index.intersection(bad_index)
    if len(present) == 0:
        return df, QuarantineResult(table=table, total=total, quarantined=0, valid=total, out_path=None)

    reasons = {idx: _reason_for_row(group) for idx, group in cases.groupby("index")}
    quarantined = df.loc[present].copy()
    quarantined["quarantine_reason"] = [reasons.get(idx, "unknown") for idx in present]
    quarantined["quarantine_run_id"] = run_id
    quarantined["quarantine_table"] = table

    out_path = Path(out_dir) / table / f"{run_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quarantined.to_parquet(out_path, index=False)

    valid_df = df.drop(index=present)
    return valid_df, QuarantineResult(
        table=table,
        total=total,
        quarantined=len(present),
        valid=len(valid_df),
        out_path=str(out_path).replace("\\", "/"),
    )
