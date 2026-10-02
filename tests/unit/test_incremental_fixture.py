"""End-to-end update-correctness tests over the TEAM-GENERATED incremental fixture (task 1.7,
REQ-26).

TEAM-GENERATED. These tests drive the full incremental pipeline (task 1.4 dedup/watermark/
window, task 1.5 schema-evolution, task 1.6 lineage/manifest) as a BLACK BOX over the
synthetic deliveries defined in `tests/fixtures/incremental/generate.py`, and assert the
curated output is correct AFTER EACH incremental step. Every delivery frame is synthetic and
team-authored (see the fixture README) — NOT from the organizer dataset.

All tests write TEMP `curated_root`, `quarantine_root`, `state_dir` and raw `root` via pytest
`tmp_path` and point `LocalSource(root=...)` at the staged raw tree, so they never touch the
real `data/`. Deliveries OVERWRITE the same partition across runs (06-10 at R1 then R3; 06-12
at R2, R4, R5), so each test stages its prerequisite deliveries into its own raw root in order
before the step it asserts, mirroring the existing pipeline test modules.

Scenarios (REQ-26), each asserted with the EXPECTED-correct outcome stated inline:
  (a) day-N normal delivery; (b) late arrival within the window; (c) duplicate re-delivery
  collapsed latest-wins (not quarantined); (d) schema change BOTH kinds — additive accepted +
  logged, breaking fails closed with no curated partition.

BUG-001 (FIXED). The two schema-change scenarios (additive + breaking) exercise heterogeneous
schemas under one fact table. They used to be blocked by a data-layer bug — `LocalSource._source`
read a fact as ONE DuckDB glob over `<table>/**/*.parquet` with no per-partition schema isolation
— and were marked `xfail(strict=True)`. The fix (per-partition schema-isolated reads in
`src/cora/data/datasource.py`, `LocalSource._source`/`S3Source._source`) makes them pass as normal
tests, so the markers are gone. The assertions were NOT weakened: they still encode the
EXPECTED-CORRECT behaviour (additive accepted + logged, breaking fails closed with a clear
`SchemaEvolutionError` and no curated partition).
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.data.pipeline import (
    LINEAGE_COLUMNS,
    SchemaEvolutionError,
    load_schema_events,
    load_watermarks,
    process_partition,
    run_table,
)


def _load_fixture_generator():
    """Load the fixture generator by file path.

    `tests/` is not an importable package in this repo (no `__init__.py`, and pytest runs in
    the default `prepend` import mode), so a plain `import tests.fixtures...` fails. The
    generator is loaded by its absolute path instead, keeping it the SINGLE source of the
    synthetic delivery frames without adding package plumbing to the test tree.
    """
    gen_path = Path(__file__).resolve().parents[1] / "fixtures" / "incremental" / "generate.py"
    spec = importlib.util.spec_from_file_location("cora_incremental_fixture", gen_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fx = _load_fixture_generator()
TABLE = fx.TABLE


# --- Helpers (mirror the existing pipeline test modules) ------------------------------


def _run(raw_root: Path, tmp_path: Path, **kwargs: object):
    source = LocalSource(root=raw_root)
    return run_table(
        source,
        TABLE,
        root=raw_root,
        curated_root=tmp_path / "curated",
        quarantine_root=tmp_path / "quarantine",
        state_dir=tmp_path / "_state",
        **kwargs,  # type: ignore[arg-type]
    )


def _curated_rows(curated_root: Path, d: date) -> pd.DataFrame:
    return pd.read_parquet(curated_root / TABLE / f"process_date={d.isoformat()}" / "part.parquet")


def _curated_dir(curated_root: Path, d: date) -> Path:
    return curated_root / TABLE / f"process_date={d.isoformat()}"


def _data_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the run-stamped lineage columns and sort by PK for a stable DATA comparison."""
    return df.drop(columns=list(LINEAGE_COLUMNS)).sort_values("transaction_id").reset_index(drop=True)


def _ids(df: pd.DataFrame) -> set[str]:
    return set(df["transaction_id"])


def _stage_through_r2(raw: Path, tmp_path: Path) -> None:
    """Stage + run R1 and R2 so the watermark is at 2026-06-12 (shared setup).

    Also stages the out-of-window partition (T0 @ 2026-01-01) ONCE before anything advances
    the watermark, so later steps can prove it is never reprocessed.
    """
    fx.write_partition(raw, TABLE, fx.DATE_OUT_OF_WINDOW, fx.delivery_out_of_window())
    fx.write_partition(raw, TABLE, fx.DATE_DAY_N, fx.delivery_r1_day_n())
    _run(raw, tmp_path)
    fx.write_partition(raw, TABLE, fx.DATE_SECOND, fx.delivery_r2_second_day())
    _run(raw, tmp_path)


# --- (a) Day-N normal delivery --------------------------------------------------------


def test_day_n_delivery(tmp_path: Path) -> None:
    """R1: a normal good partition is curated, stamped and watermarked."""
    raw = tmp_path / "raw"
    curated = tmp_path / "curated"
    state = tmp_path / "_state"
    fx.write_partition(raw, TABLE, fx.DATE_DAY_N, fx.delivery_r1_day_n())
    summary = _run(raw, tmp_path)

    day_n = _curated_rows(curated, fx.DATE_DAY_N)
    assert _ids(day_n) == {"T1", "T2"}  # EXPECTED: the two good day-N rows, nothing else
    assert set(LINEAGE_COLUMNS).issubset(day_n.columns)  # EXPECTED: lineage stamped on rows
    # EXPECTED: watermark persisted at the latest processed partition (2026-06-10).
    assert load_watermarks(state / "watermarks.json").get(TABLE) == "2026-06-10"
    # EXPECTED: a run manifest file was written and surfaced on the summary.
    assert summary.manifest_path is not None
    assert Path(summary.manifest_path).exists()


# --- (b) Late arrival within the window -----------------------------------------------


def test_late_arrival_reprocessed_within_window(tmp_path: Path) -> None:
    """R3: a late row for an earlier day inside the window corrects the earlier curated
    partition, while an out-of-window partition is left untouched."""
    raw = tmp_path / "raw"
    curated = tmp_path / "curated"
    _stage_through_r2(raw, tmp_path)  # watermark now 2026-06-12

    # Snapshot the out-of-window curated DATA so we can prove it is untouched by R3.
    out_of_window_before = _data_only(_curated_rows(curated, fx.DATE_OUT_OF_WINDOW))

    # Overwrite 2026-06-10 with the late row T4. Watermark 2026-06-12, W=3 -> lower bound
    # 2026-06-09, so 2026-06-10 is inside the window and gets reprocessed; 2026-01-01 is not.
    fx.write_partition(raw, TABLE, fx.DATE_DAY_N, fx.delivery_r3_late_arrival())
    summary = _run(raw, tmp_path)

    # EXPECTED: the earlier 2026-06-10 curated partition now carries the late row -> 3 rows.
    assert _ids(_curated_rows(curated, fx.DATE_DAY_N)) == {"T1", "T2", "T4"}
    processed = {p.process_date for p in summary.partitions}
    assert "2026-06-10" in processed  # EXPECTED: within window -> reprocessed
    assert "2026-01-01" not in processed  # EXPECTED: out of window -> never reprocessed
    # EXPECTED: the out-of-window partition's curated DATA is byte-identical to before R3.
    assert _data_only(_curated_rows(curated, fx.DATE_OUT_OF_WINDOW)).equals(out_of_window_before)


# --- (c) Duplicate re-delivery collapsed latest-wins (NOT quarantined) ----------------


def test_duplicate_redelivery_collapsed_not_quarantined(tmp_path: Path) -> None:
    """R4: a duplicate re-delivery of the same PK is deduped latest-wins, keeping exactly one
    curated row — the duplicate is collapsed, NOT quarantined."""
    raw = tmp_path / "raw"
    curated = tmp_path / "curated"
    _stage_through_r2(raw, tmp_path)  # watermark now 2026-06-12, curated 06-12 == {T3}

    # Overwrite 2026-06-12 with T3 twice; the surviving (amount 10.0) row is first in input
    # order, the duplicate (amount 99.0) is second.
    fx.write_partition(raw, TABLE, fx.DATE_SECOND, fx.delivery_r4_duplicate())
    summary = _run(raw, tmp_path)

    second_pr = next(p for p in summary.partitions if p.process_date == "2026-06-12")
    assert second_pr.duplicates_removed == 1  # EXPECTED: exactly one duplicate collapsed
    assert second_pr.rows_quarantined == 0  # EXPECTED: a duplicate is deduped, not quarantined
    dup_curated = _curated_rows(curated, fx.DATE_SECOND)
    assert _ids(dup_curated) == {"T3"}  # EXPECTED: exactly one row for PK T3
    # EXPECTED: the surviving row is the first-in-input-order one (original amount 10.0).
    assert dup_curated.loc[dup_curated["transaction_id"] == "T3", "amount"].iloc[0] == 10.0
    # EXPECTED: the duplicate was not written to quarantine (no quarantine file references T3).
    if second_pr.quarantine_path is not None:
        quarantined = pd.read_parquet(second_pr.quarantine_path)
        assert "T3" not in set(quarantined.get("transaction_id", pd.Series(dtype=str)))


# --- (d-additive) New nullable column accepted + logged -------------------------------


def test_additive_column_accepted_and_logged(tmp_path: Path) -> None:
    """R5: a new nullable column is accepted (strict=False), a SchemaChangeEvent is logged,
    and the column reaches curated.

    EXPECTED-CORRECT behaviour (asserted, NOT weakened). Unblocked by the BUG-001 fix
    (per-partition schema-isolated reads): the additive `promo_flag` present only in the
    2026-06-12 partition now survives the read, so the event is emitted and the column reaches
    curated.
    """
    raw = tmp_path / "raw"
    curated = tmp_path / "curated"
    state = tmp_path / "_state"
    _stage_through_r2(raw, tmp_path)  # watermark now 2026-06-12

    # Overwrite 2026-06-12 with {T3, T5} carrying a new nullable column `promo_flag`.
    fx.write_partition(raw, TABLE, fx.DATE_SECOND, fx.delivery_r5_additive())
    summary = _run(raw, tmp_path)

    assert summary.schema_breaking is False  # EXPECTED: an additive change is NOT breaking
    # EXPECTED: a SchemaChangeEvent kind=added column=promo_flag is surfaced on the summary.
    assert any(ev.kind == "added" and ev.column == "promo_flag" for ev in summary.schema_events)
    # EXPECTED: the same event is persisted to the runtime schema-event log under state.
    persisted = load_schema_events(state / "schema_events.json")
    assert any(ev.kind == "added" and ev.column == "promo_flag" for ev in persisted)
    additive_curated = _curated_rows(curated, fx.DATE_SECOND)
    assert "promo_flag" in additive_curated.columns  # EXPECTED: new column flows to curated
    assert _ids(additive_curated) == {"T3", "T5"}  # EXPECTED: one row per PK after overwrite


# --- (d-breaking) Required column missing -> fail-closed ------------------------------


def test_breaking_column_fails_closed(tmp_path: Path) -> None:
    """R6: a partition missing a required column fails closed with a clear
    `SchemaEvolutionError` naming the column, writes NO curated partition, and leaves prior
    curated state intact.

    EXPECTED-CORRECT behaviour (asserted, NOT weakened). Unblocked by the BUG-001 fix
    (per-partition schema-isolated reads): the 2026-06-13 partition missing `customer_id` is
    read in isolation, so the column is genuinely absent (not NULL-filled from a sibling) and
    the pipeline's check_schema/raise_if_breaking fail-closed gate produces the designed
    SchemaEvolutionError.
    """
    raw = tmp_path / "raw"
    curated = tmp_path / "curated"
    state = tmp_path / "_state"
    _stage_through_r2(raw, tmp_path)  # watermark now 2026-06-12; curated 06-10/06-12 exist

    # Snapshot prior curated DATA so we can prove the breaking delivery leaves it intact.
    day_n_before = _data_only(_curated_rows(curated, fx.DATE_DAY_N))
    second_before = _data_only(_curated_rows(curated, fx.DATE_SECOND))

    # Drive the bad 2026-06-13 partition with process_partition directly (isolates the failure
    # instead of aborting a whole run_table loop), matching the existing schema-evolution test.
    fx.write_partition(raw, TABLE, fx.DATE_BREAKING, fx.delivery_r6_breaking())
    source = LocalSource(root=raw)
    with pytest.raises(SchemaEvolutionError) as excinfo:
        process_partition(
            source,
            TABLE,
            fx.DATE_BREAKING,
            curated_root=curated,
            quarantine_root=tmp_path / "quarantine",
            state_dir=state,
            run_id="fixture_breaking_run",
            root=raw,
        )
    # EXPECTED: the error names the missing required column.
    assert "customer_id" in str(excinfo.value)
    # EXPECTED: fail-closed -> NO curated partition directory for the breaking delivery.
    assert not _curated_dir(curated, fx.DATE_BREAKING).exists()
    # EXPECTED: prior curated state (06-10, 06-12) is unchanged (DATA columns, lineage excluded).
    assert _data_only(_curated_rows(curated, fx.DATE_DAY_N)).equals(day_n_before)
    assert _data_only(_curated_rows(curated, fx.DATE_SECOND)).equals(second_before)
