"""Generator for the incremental update-correctness fixture (CORA task 1.7, REQ-26).

TEAM-GENERATED. Every frame produced here is SYNTHETIC and TEAM-AUTHORED; NONE of it comes
from the organizer dataset. The values are tiny, hand-built, all-VARCHAR `transactions` rows
(a handful per partition) so the committed fixture is inspectable and the end-to-end tests run
fast and memory-safe without the real landing.

Why `transactions`: it is the table every existing pipeline test models
(`tests/unit/test_pipeline.py`, `test_lineage.py`, `test_schema_evolution.py`), so this fixture
reuses the proven synthetic shape (`_transactions_frame`) and inherits a known-correct baseline.
It has a real PK (`transaction_id`, `unique=True`) and required contract columns
(`customer_id`, `product_id`) — dropping a required column triggers the BREAKING schema case.

Dedup order (grounded): `quality/registries.py` sets `ARRIVAL_COLUMN["transactions"] =
"process_date"`. Within ONE raw partition every row shares the same `process_date`, so
`deduplicate()` sorts stably by `process_date` desc (all equal) and keeps the FIRST occurrence
per `transaction_id` in input order. Consequence: a duplicate re-delivery of the SAME PK inside
one partition collapses latest-wins to the FIRST row of that PK in the written frame. The
duplicate delivery below therefore places the surviving row FIRST.

This module is import-safe (no top-level I/O). Running it as a script
(`uv run python tests/fixtures/incremental/generate.py`) materializes the committed Parquet
tree under `tests/fixtures/incremental/raw/` for inspection/reproducibility. The tests do NOT
depend on that tree: they stage each per-run delivery frame into a pytest `tmp_path` raw root
in order, because deliveries OVERWRITE the same partition across runs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

TABLE = "transactions"

# Delivery dates used across the scenario (all inside the contract fact window
# 2023-06-17 .. 2026-06-17). Window W = 3 days (the pipeline default).
DATE_OUT_OF_WINDOW = date(2026, 1, 1)  # T0 — far in the past, never reprocessed after R2
DATE_DAY_N = date(2026, 6, 10)  # T1, T2 — the day-N normal delivery, later gets late T4
DATE_SECOND = date(2026, 6, 12)  # T3 — advances watermark; later duplicate + additive
DATE_BREAKING = date(2026, 6, 13)  # T6 — the breaking (required-column-missing) delivery


def _transactions_frame(ids: list[str], process_date: str, *, amount: str = "10.0") -> pd.DataFrame:
    """A minimal valid raw-string `transactions` frame (mirrors the pipeline tests).

    All values are strings because the real landing is all-VARCHAR; writing strings makes the
    `normalize_landing`/coerce path run exactly as it does on real data.
    """
    n = len(ids)
    return pd.DataFrame(
        {
            "transaction_id": ids,
            "transaction_date": [process_date] * n,
            "process_date": [process_date] * n,
            "product_id": ["P00000"] * n,
            "customer_id": ["C00000"] * n,
            "transaction_type": ["Purchase"] * n,
            "transaction_category": ["Food"] * n,
            "amount": [amount] * n,
            "currency": ["USD"] * n,
            "amount_usd": [amount] * n,
            "channel": ["POS"] * n,
            "branch_id": [None] * n,
            "merchant_name": [None] * n,
            "merchant_category": [None] * n,
            "transaction_country": ["MX"] * n,
            "transaction_city": ["CDMX"] * n,
            "transaction_status": ["Approved"] * n,
            "response_code": ["00"] * n,
            "is_fraud": ["False"] * n,
            "fraud_score": ["10.0"] * n,
            "latitude": [None] * n,
            "longitude": [None] * n,
        }
    )


def write_partition(raw_root: Path, table: str, d: date, df: pd.DataFrame) -> Path:
    """Write `df` into the Hive layout the real landing uses (zero-padded month/day).

    Returns the written Parquet file path. Overwrites any file already in the partition so a
    later delivery for the same `process_date` replaces the earlier one (as a re-delivery does).
    """
    part = raw_root / table / f"year={d.year}" / f"month={d.month:02d}" / f"day={d.day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    out = part / f"{table}_{d:%Y%m%d}.parquet"
    df.to_parquet(out, index=False)
    return out


# --- Per-run delivery frames (synthetic, team-authored) -------------------------------
#
# Each function returns the RAW frame delivered for one scenario step. Tests stage these into
# their own tmp_path raw root in order; the committed tree (see `materialize`) holds the final
# state of each partition for inspection only.


def delivery_out_of_window() -> pd.DataFrame:
    """Row T0 on 2026-01-01 — a good partition far older than any later window.

    Written once before the watermark advances to 2026-06-12. After that advance its lower
    reprocessing bound (watermark - W = 2026-06-09) excludes it, so it is NEVER reprocessed.
    Used to assert an out-of-window partition stays untouched.
    """
    return _transactions_frame(["T0"], DATE_OUT_OF_WINDOW.isoformat())


def delivery_r1_day_n() -> pd.DataFrame:
    """(a) Day-N normal delivery: 2026-06-10 with two good rows T1, T2."""
    return _transactions_frame(["T1", "T2"], DATE_DAY_N.isoformat())


def delivery_r2_second_day() -> pd.DataFrame:
    """Second good partition: 2026-06-12 with row T3. Advances the watermark to 2026-06-12."""
    return _transactions_frame(["T3"], DATE_SECOND.isoformat())


def delivery_r3_late_arrival() -> pd.DataFrame:
    """(b) Late arrival: 2026-06-10 re-delivered as T1, T2, T4.

    T4 is a late good row for an EARLIER day. With watermark 2026-06-12 and W=3 the lower
    bound is 2026-06-09, so 2026-06-10 is inside the window and gets reprocessed, correcting
    the earlier curated 2026-06-10 partition to three rows.
    """
    return _transactions_frame(["T1", "T2", "T4"], DATE_DAY_N.isoformat())


def delivery_r4_duplicate() -> pd.DataFrame:
    """(c) Duplicate re-delivery: 2026-06-12 re-delivered with T3 twice (same PK).

    The SURVIVING row (original amount 10.0) is placed FIRST; the duplicate (amount 99.0) is
    second. Because the arrival column is `process_date` (constant within the partition),
    latest-wins collapses to the first-in-input-order row, so curated keeps the 10.0 row and
    the duplicate is COLLAPSED (deduplicated), NOT quarantined.
    """
    frame = _transactions_frame(["T3", "T3"], DATE_SECOND.isoformat())
    frame.loc[1, "amount"] = "99.0"
    frame.loc[1, "amount_usd"] = "99.0"
    return frame


def delivery_r5_additive() -> pd.DataFrame:
    """(d-additive) Additive schema change: 2026-06-12 re-delivered as T3, T5 with a new
    nullable column `promo_flag`.

    The extra column is accepted (contracts are strict=False), a `SchemaChangeEvent`
    kind=`added` column=`promo_flag` is logged, and the rows are curated with the new column.
    This overwrites the 06-12 partition, so curated afterwards reflects {T3, T5}.
    """
    frame = _transactions_frame(["T3", "T5"], DATE_SECOND.isoformat())
    frame["promo_flag"] = ["spring", "spring"]
    return frame


def delivery_r6_breaking() -> pd.DataFrame:
    """(d-breaking) Breaking schema change: 2026-06-13 delivered as T6 WITHOUT `customer_id`.

    A required contract column is missing, so the schema-evolution check raises
    `SchemaEvolutionError` (fail-closed) and NO curated partition is written for this delivery.
    """
    return _transactions_frame(["T6"], DATE_BREAKING.isoformat()).drop(columns=["customer_id"])


# --- Committed-tree materialization (inspection / reproducibility) --------------------

# The committed Parquet tree holds the FINAL state of each partition. 06-10 ends at its late
# arrival state (T1, T2, T4); 06-12 ends at its additive state (T3, T5). The breaking delivery
# is also materialized so a reviewer can inspect the missing-column input.
_FINAL_STATE: dict[date, pd.DataFrame] = {}


def _final_state() -> dict[date, pd.DataFrame]:
    if not _FINAL_STATE:
        _FINAL_STATE.update(
            {
                DATE_OUT_OF_WINDOW: delivery_out_of_window(),
                DATE_DAY_N: delivery_r3_late_arrival(),
                DATE_SECOND: delivery_r5_additive(),
                DATE_BREAKING: delivery_r6_breaking(),
            }
        )
    return _FINAL_STATE


def materialize(raw_root: Path) -> list[Path]:
    """Write the committed Parquet tree (final per-partition states) under `raw_root`.

    Returns the list of written file paths. Used by `main()` to (re)create the committed
    fixture; tests do NOT call this (they stage per-run frames instead).
    """
    return [write_partition(raw_root, TABLE, d, df) for d, df in _final_state().items()]


def main() -> None:
    """(Re)create the committed fixture Parquet tree under this folder's `raw/` directory."""
    raw_root = Path(__file__).resolve().parent / "raw"
    written = materialize(raw_root)
    for path in written:
        print(f"wrote {path.as_posix()}")


if __name__ == "__main__":
    main()
