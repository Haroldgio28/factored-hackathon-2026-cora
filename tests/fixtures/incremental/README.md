# Incremental update-correctness fixture — TEAM-GENERATED (REQ-26)

> **TEAM-GENERATED / SYNTHETIC.** Every byte in this folder is hand-built by the CORA team.
> **None of it comes from the organizer dataset.** It exists only to drive the incremental
> pipeline end-to-end in tests and prove the curated output is correct after each step.

This fixture satisfies **REQ-26**: a clearly labeled synthetic fixture simulating day-N
delivery, late arrival, duplicate re-delivery and a schema change, with tests proving the
curated output is correct after each step.

## What is committed

- `generate.py` — the generator (TEAM-AUTHORED header). Builds every raw delivery frame and
  can (re)write the committed Parquet tree. Import-safe (no top-level I/O).
- `raw/transactions/year=.../month=.../day=.../transactions_YYYYMMDD.parquet` — the committed
  Parquet partitions (the **final** state of each partition, for inspection/reproducibility).
  Re-create them with `uv run python tests/fixtures/incremental/generate.py`.
- `.gitignore` with a single `!*.parquet` line — a scoped, test-only negation that re-includes
  the fixture Parquet (the root `.gitignore` ignores `*.parquet`). The root `.gitignore` is
  untouched; only this fixture folder's Parquet is committed.

The **tests** (`tests/unit/test_incremental_fixture.py`) do NOT read the committed tree. They
import the per-run frame builders from `generate.py` and stage each delivery into a pytest
`tmp_path` raw root in order, because deliveries overwrite the same partition across runs. All
tests use TEMP `curated_root`, `quarantine_root`, `state_dir` and raw `root` — they never touch
the real `data/`.

## Table modeled: `transactions` (and why)

`transactions` is the table every existing pipeline test models, so this fixture reuses the
proven synthetic row shape and inherits a known-correct baseline. It has a real PK
(`transaction_id`, `unique=True`) and required contract columns (`customer_id`, `product_id`),
so dropping a required column cleanly triggers the BREAKING schema case. All raw values are
VARCHAR strings so the `normalize_landing`/coerce path runs exactly as on real data.

## Dedup order note (drives the duplicate scenario)

The arrival column for `transactions` is `process_date` (`quality/registries.py`), which is
constant within one partition. So latest-wins dedup = **keep the first occurrence per PK in
input order**. The duplicate delivery places the row that should survive FIRST.

## Reprocessing window

`W = 3` days (the pipeline default). After the watermark reaches `2026-06-12`, the lower
reprocessing bound is `2026-06-09`; partitions older than that are never reprocessed.

## Delivery order and expected curated outcome after each step

| Run | Raw state staged before the run | Scenario | Expected curated outcome asserted |
|-----|---------------------------------|----------|-----------------------------------|
| R1 | `day=2026-06-10` = {T1, T2} (plus `day=2026-01-01` = {T0} staged once, out of window) | (a) Day-N normal delivery | curated `2026-06-10` == {T1, T2}; all three lineage columns present; watermark == `2026-06-10`; manifest file written and `summary.manifest_path` set |
| R2 | `day=2026-06-12` = {T3} | advance watermark | watermark advances to `2026-06-12`; curated `2026-06-12` == {T3} |
| R3 | overwrite `day=2026-06-10` = {T1, T2, T4} | (b) Late arrival within window | curated `2026-06-10` reprocessed == {T1, T2, T4}; out-of-window `2026-01-01` untouched; `2026-06-10` in processed set, `2026-01-01` not |
| R4 | overwrite `day=2026-06-12` = {T3 (amount 10.0), T3 (amount 99.0)} | (c) Duplicate re-delivery | `duplicates_removed == 1`; curated `2026-06-12` keeps exactly one T3 with the original amount; `rows_quarantined == 0`, duplicate NOT in quarantine |
| R5 | overwrite `day=2026-06-12` = {T3, T5} + new nullable `promo_flag` | (d) Additive schema change | `schema_breaking is False`; `SchemaChangeEvent` kind=`added` column=`promo_flag` surfaced on the summary and persisted; curated `2026-06-12` includes `promo_flag` |
| R6 | `day=2026-06-13` = {T6} with `customer_id` dropped | (d) Breaking schema change | `process_partition` raises `SchemaEvolutionError` naming `customer_id`; NO curated partition for `2026-06-13`; prior curated `2026-06-10`/`2026-06-12` DATA unchanged |

PKs: `T0` (out-of-window), `T1`/`T2` (day-N), `T3` (second day + duplicate + additive), `T4`
(late arrival), `T5` (additive), `T6` (breaking).

## Known limitation (BUG-001) — FIXED

All five REQ-26 scenarios now pass as normal tests, including the two **schema-change**
scenarios (additive column, breaking column). They previously exposed a real data-layer bug
tracked as **BUG-001** (`.agents/tasks/BUG-001-multifile-scan.md`), and their tests
(`test_additive_column_accepted_and_logged`, `test_breaking_column_fails_closed`) were marked
`pytest.mark.xfail(strict=True)`. The fix landed, so the markers are gone and the tests pass
as normal tests with their original (unweakened) EXPECTED-correct assertions.

Original root cause: `LocalSource._source` read a fact as one DuckDB glob over
`<table>/**/*.parquet` with `hive_partitioning=true` and no per-partition schema isolation, so a
column present in only one partition was dropped (additive) and a partition missing a required
column raised a raw DuckDB read error before the pipeline's fail-closed schema gate ran
(breaking). `union_by_name=true` was the WRONG fix because it NULL-fills the missing required
column and defeats the fail-closed check.

The fix (`src/cora/data/datasource.py`, `LocalSource._source` and the mirrored
`S3Source._source`): when a fact is read with a FULLY-specified single partition (year AND
month AND day), `read_parquet` is pointed at that partition's files
(`<table>/year=Y/month=MM/day=DD/*.parquet`) instead of the whole-table `**` glob. The added
column then survives (its own file carries it) and a missing required column is genuinely absent
from the read (not NULL-filled), so the pipeline's `check_schema`/`raise_if_breaking` fires the
designed `SchemaEvolutionError` → fail-closed, no curated partition. Partial specs (year only,
year+month, or none) keep the whole/subtree `**` glob, and the narrowing keeps reads bounded.
