"""Deterministic dedup over a bounded batch, latest wins (task 1.3, REQ-23).

Keeps the latest record per PK. Preference, per table (see `ARRIVAL_COLUMN`):
- an arrival column (`last_updated` for customers/products, `process_date` for facts): sort
  by it descending (stable) then keep the first row per PK;
- no arrival column (branches, service_agents, marketing_campaigns, daily_exchange_rates):
  keep the first occurrence in input order (`tie_break="input_order"`), documented in the
  result.

Observed PK-dup is 0% (EDA F4), so this is defensive but must be correct and testable. It
operates on an already-bounded in-memory batch (the valid rows from quarantine), never a
full fact table.
"""

from __future__ import annotations

import pandas as pd

from .registries import ARRIVAL_COLUMN, PK_COLUMNS
from .types import DedupResult

_MAX_EXAMPLES = 5


def deduplicate(table: str, df: pd.DataFrame) -> tuple[pd.DataFrame, DedupResult]:
    """Return (deduped_df, DedupResult) keeping the latest record per PK."""
    pk = PK_COLUMNS[table]
    total = len(df)
    arrival = ARRIVAL_COLUMN.get(table)

    if total == 0:
        return df, DedupResult(table=table, kept=0, removed=0, tie_break="n/a")

    # Example duplicated PK values (before dedup), for the report.
    dup_mask = df.duplicated(subset=pk, keep=False)
    if len(pk) == 1:
        examples = df.loc[dup_mask, pk[0]].drop_duplicates().head(_MAX_EXAMPLES).tolist()
    else:
        examples = df.loc[dup_mask, pk].drop_duplicates().head(_MAX_EXAMPLES).to_dict(orient="records")

    if arrival is not None and arrival in df.columns:
        tie_break = f"{arrival}_desc"
        # Stable sort by arrival desc so the latest arrival is first per PK; nulls sort last.
        ordered = df.sort_values(by=arrival, ascending=False, kind="stable", na_position="last")
        deduped = ordered.drop_duplicates(subset=pk, keep="first")
        # Restore original row order among the kept rows for determinism/readability.
        deduped = deduped.sort_index()
    else:
        tie_break = "input_order"
        deduped = df.drop_duplicates(subset=pk, keep="first")

    kept = len(deduped)
    return deduped, DedupResult(
        table=table,
        kept=kept,
        removed=total - kept,
        tie_break=tie_break,
        examples=examples,
    )
