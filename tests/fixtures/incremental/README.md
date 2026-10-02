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

## Known limitation (BUG-001)

The fixture DATA represents all five REQ-26 scenarios, but the two **schema-change** scenarios
(additive column, breaking column) do not yet pass against the current pipeline — they expose a
real data-layer bug, tracked separately as **BUG-001**
(`.agents/tasks/BUG-001-multifile-scan.md`). Their tests
(`test_additive_column_accepted_and_logged`, `test_breaking_column_fails_closed`) are marked
`pytest.mark.xfail(strict=True)`: they assert the EXPECTED-correct behaviour (the assertions are
NOT weakened to match the bug) and will flip to XPASS — failing the suite — once the fix lands,
forcing them to become normal passing tests then.

Root cause: `LocalSource._source` reads a fact as one DuckDB glob over
`<table>/**/*.parquet` with `hive_partitioning=true` and no per-partition schema isolation, so a
column present in only one partition is dropped (additive) and a partition missing a required
column raises a raw DuckDB read error before the pipeline's fail-closed schema gate runs
(breaking). `union_by_name=true` is the WRONG fix because it NULL-fills the missing required
column and defeats the fail-closed check — see BUG-001 for the proposed per-partition read fix.
The other three scenarios (day-N, late arrival within/out of window, duplicate collapsed
latest-wins not quarantined) pass as normal tests.
