# CORA - Implementation Tasks

**Spec:** `cora` · **Phase:** Tasks (3 of 3) · **Status:** Draft v1.0 · **Date:** 2026-10-01
Implements [`design.md`](design.md) for [`requirements.md`](requirements.md). Schedule and process: [`documentation/DEVELOPMENT_PLAN.md`](../../../documentation/DEVELOPMENT_PLAN.md).

Rules for every task:
- One branch per task group (`feat/<area>-<short>`), PR to `main`, never commit on `main`.
- A task is **done** only when its *Evidence* exists in the repo (test, report, figure or doc) and the traceability matrix is updated.
- Code and docs in English; customer-facing strings in es/pt.

---

## Phase 0 - Foundations (Day 1)

- [x] **0.1 Project skeleton & tooling**
  - `pyproject.toml` with `uv`, pinned deps, `ruff`, `pytest`, `src/cora` package, `Makefile` + `tasks.ps1` (Windows) with `setup / data / test / eval / demo`.
  - Evidence: `make test` green on an empty suite in CI.
  - Done 2026-09-30. Verified locally: `uv run ruff check .` passes, `ruff format --check` clean
    (33 files), `uv run pytest` green (5 passed). `analysis/profile_tables.py` lint fixed
    (E501 per-file-ignore for the data-dictionary strings).
  - _Requirements: REQ-48_
- [x] **0.2 CI**
  - GitHub Actions: lint, unit tests, contract tests on the fixture; secret scanning (gitleaks).
  - Done 2026-09-30. `.github/workflows/ci.yml`: jobs `test` (uv sync --locked, ruff check +
    format --check, pytest), `secrets` (gitleaks), `no-pdfs` (fails if any PDF is tracked);
    actions pinned by SHA. Lint/format/tests confirmed green locally.
  - _Requirements: REQ-36, REQ-48_
- [x] **0.3 Config & secrets**
  - Evidence: `src/cora/settings.py`, `tests/unit/test_settings.py` (template keys match settings, no credential shapes, blank template parses).
  - `settings.py` (pydantic-settings), `.env.example` with variable names only; AWS profile name configurable.
  - _Requirements: REQ-36, REQ-48_

- [x] **0.4 AWS account bootstrap & Bedrock smoke test** (owner executes, Kiro assists)
  - Follow [`documentation/AWS_SETUP.md`](../../../documentation/AWS_SETUP.md) steps 1-7: root MFA, budget + anomaly alerts, IAM Identity Center with `CoraAdmin` / `CoraDeveloper` (least privilege, `infra/iam/cora-developer-policy.json`), SSO profile `cora-dev`, Anthropic FTU form, model ids in `.env`.
  - Run `python scripts/aws/verify_bedrock.py`; record masked output (latency, tokens, es/pt answers) in `documentation/reports/bedrock_smoke_test.md`.
  - Done 2026-09-30. Account is "Sign up for AWS (new)" (Paid Plan, USD 20 spend limit), so the
    plan deviated from the guide: no IAM Identity Center (unsupported) -> Bedrock API key bearer
    token instead of SSO profile `cora-dev`; region `us-east-2`; `global.` inference profiles.
    See the addendum in `AWS_SETUP.md` and evidence in `documentation/reports/bedrock_smoke_test.md`.
  - _Requirements: REQ-36, REQ-40, REQ-43, REQ-48_

## Phase 1 - Data platform (Days 1-3)

- [x] **1.1 DataSource interface**
  - `LocalSource` (sample folder) and `S3Source` (profile `cora-datathon`); `scripts/fetch_sample` pulls a reproducible sample (fixed months, all dimensions).
  - Raw landing (2026-10-01): the full history of all 13 tables is landed as raw Parquet by `scripts/data/fetch_to_parquet.py` (replaces the sample fetch; ~0.9 GB) and checked by `scripts/data/validate_landing.py` (`documentation/reports/landing_validation.md`).
  - Done (2026-10-01): `src/cora/data/` with `DataSource` (ABC), `LocalSource` (DuckDB over `data/raw_parquet/`, INT year/month/day partition filters) and `S3Source` (read-only profile `cora-datathon`, no network on construction, live read opt-in via `CORA_RUN_S3_LIVE=1`). Methods `scan` (lazy DuckDB relation) / `count` / `fetch_df` (required positive `limit`); table names validated against a FACTS/DIMENSIONS registry (SQL-injection guard). Evidence: `tests/unit/test_datasource.py` (9 passed, 1 skipped); `uv run pytest` 14 passed / 1 skipped, ruff clean.
  - _Requirements: REQ-20_
- [x] **1.2 Contracts for all 13 tables**
  - Pandera schemas from the data dictionary (types, nullability, PK, enums, ranges); contract tests.
  - Done (2026-10-01): `src/cora/data/contracts/` — one `coerce=True` Pandera `DataFrameSchema` per table (6 dimensions in `dimensions.py`, 7 facts in `facts.py`, shared primitives in `_checks.py`). Encodes OBSERVED Spanish enums (F2), observed structural nullability (F6), PK uniqueness (composite PK for `daily_exchange_rates`), and justified ranges (`fraud_score` 0-100, `credit_score` 300-850). `normalize_landing()` pre-casts the all-VARCHAR raw landing (string booleans, float-string ints) since Pandera coerces before parsers. `SCHEMAS` registry asserts coverage == `datasource.TABLES` at import; `CONTRACTS_VERSION="1.0"`; `validate(table, df) -> (valid_df, failure_cases)` uses `lazy=True` and does NOT raise, so task 1.3 can build quarantine on top. Reads go through the `DataSource`. Deps: `pandera==0.20.4`, `pandas==2.2.3` promoted to runtime (shipped code). Evidence: `tests/contract/test_contracts.py` (13 positive bounded-sample + 2 registry/version + 5 negative); `uv run pytest` 34 passed / 1 skipped, ruff clean. Note: `complaints.currency` includes MXN (verified: 5,487 rows) — F3's "no MXN" is specific to `products`/`transactions`, not `complaints`. Approved by semantic review.
  - _Requirements: REQ-21_
- [x] **1.3 Quality checks + quarantine**
  - Duplicates (PK and row hash), null rates, orphan FKs, enums, ranges, currency; quarantine with reason; run report.
  - Evidence: `documentation/reports/data_quality.md` with measured rates vs documented (~2% dup, ~5% null).
  - Done (2026-10-01): `src/cora/data/quality/` package (`checks.py`, `quarantine.py`, `dedup.py`, `report.py`, `registries.py`, `types.py`) on top of the `DataSource` + contracts. Checks are DuckDB aggregate SQL (memory-safe) with numerator/denominator/examples: PK-dup + full-row-hash dup (composite key for FX), null-vs-contract, orphan-FK anti-join, enum violations, contract-introspected ranges (`credit_score` 300-850, `fraud_score` 0-100, non-negative, date window), per-table currency rules (ADR-016/017: no MXN on products/transactions, MXN on complaints/FX). Quarantine consumes `contracts.validate()` failure_cases, splits valid/invalid with a per-row reason, writes to gitignored `data/quarantine/<table>/<run_id>.parquet`, never crashes. Deterministic latest-wins dedup (`last_updated`→`process_date`→input-order tie-break, REQ-23). CLI `scripts/data/quality_report.py` emits `documentation/reports/data_quality.{md,json}`. Measured: PK-dup / row-hash-dup / orphan-FK ~0%; structural nulls high in expected columns (`credit_limit` 68.7%, `origin_interaction_id` 100%). Evidence: `tests/unit/test_quality.py` (17 tests); `uv run pytest` 51 passed / 1 skipped, ruff clean; no new dependency. Approved by semantic review (3 non-blocking notes).
  - _Requirements: REQ-22, REQ-23_
- [x] **1.4 Incremental pipeline**
  - Watermark by `process_date`, reprocessing window W=3, idempotent partition overwrite, dedup latest-wins, freshness record.
  - Done (2026-10-01): `src/cora/data/pipeline/` (`types.py`, `state.py`, `partitions.py`, `runner.py`) orchestrating the existing DataSource + contracts (1.2) + quality/quarantine/dedup (1.3) one partition at a time. Per-table high-watermark (latest curated `process_date`) in `data/_state/watermarks.json` (runtime; init-when-absent, only advances). Reprocessing window `W` default 3 (`--window`): reprocesses `watermark-W .. latest` to absorb late arrivals; older partitions never reprocessed. Per-partition stage order **dedup → validate → quarantine** (owner-approved deviation, ADR rationale: PK `unique=True` would quarantine legitimate re-deliveries under validate-first, violating REQ-23; design §7 makes dedup its own stage). Idempotent partition-overwrite writes to `data/curated/<table>/process_date=YYYY-MM-DD/` (two-run SHA-256 byte-identical evidence). Typed freshness record (`max_partition_date`, `max_event_date`, watermark, `last_ingested_at`, `curated_rows`) in `data/_state/freshness.json`. CLI `scripts/data/run_pipeline.py` (flags `--window`, `--start-date`, dry-run). `process_date` == Hive partition date (verified 0 mismatches). Memory-safe (one partition at a time). 1.5/1.6 left as clean seams; `PIPELINE_VERSION` defined. Evidence: `tests/unit/test_pipeline.py` (12 tests incl. watermark init/advance, window late-arrival, idempotency, quarantine+dedup orchestration, freshness); `uv run pytest` 63 passed / 1 skipped, ruff clean; no new dependency. Approved by semantic review.
  - _Requirements: REQ-24_
- [x] **1.5 Schema-evolution handling**
  - Additive vs breaking change detection; events in the manifest.
  - Done (2026-10-01): `src/cora/data/pipeline/schema_evolution.py` — `check_schema(table, incoming) -> SchemaCheckResult`, `SchemaEvolutionError`, `raise_if_breaking()`, and append/load of `data/_state/schema_events.json` (runtime). Wired at the `# seam: 1.5` in `runner.process_partition()` BEFORE dedup/validate (stage order unchanged, ADR-018). The Pandera contract (`get_schema`) is the sole expected-schema authority; diff is NAME-based (the all-VARCHAR landing makes dtype comparison meaningless, so type-safety stays with `contracts.validate`/quarantine — documented). ADDED column → accept + log event + continue (facts are `strict=False`, extra column survives); MISSING required contract column (via `Column.required`) → fail loudly, fail-closed (raise before any curated write, nothing curated); MISSING optional (`year`/`month`/`day`) → non-breaking removed event. Schema inspection reads only column names, never rows. `SchemaChangeEvent` typed record; events surfaced on `PartitionResult`/`RunSummary` (so task 1.6 can fold them into the run manifest). Evidence: `tests/unit/test_schema_evolution.py` (7 tests: added-nullable accepted, missing-required raises with clear message, type non-false-positive, event-log append, RunSummary surfacing, integration curates vs fail-closed); `uv run pytest` 70 passed / 1 skipped, ruff clean; no new dependency. Approved by semantic review.
  - _Requirements: REQ-25_
- [x] **1.6 Lineage**
  - Row-level `_source_file`, `_ingested_at`, `_pipeline_version`; run manifest JSON.
  - Done (2026-10-01): row-level lineage stamped at the `# seam: 1.6` in `runner.process_partition()` (after validate/quarantine, before the idempotent curated write) onto the per-partition frame already in memory (bounded, never a full-table pass). `_source_file` = the raw Hive partition DIRECTORY `data/raw_parquet/<table>/year=Y/month=MM/day=DD` (POSIX, via the shared `raw_partition_dir(root, table, process_date)` helper — the unit `DataSource.scan` actually prunes to); `_ingested_at` = one run-constant UTC ISO timestamp computed once in `run_table` and threaded into every partition + reused for the freshness record so rows/manifest/freshness agree; `_pipeline_version` = `PIPELINE_VERSION`. Idempotency re-scoped (documented): asserted on business/DATA columns, the three run-stamped lineage columns (`types.LINEAGE_COLUMNS`) EXCLUDED from the comparison; the task-1.4 `test_idempotent_partition_overwrite` adjusted to drop them before `.equals` and still proves overwrite-not-append. Per-run manifest `RunManifest` (frozen dataclass + `to_dict`/`from_run`) REUSES `RunSummary`/`PartitionResult`/`SchemaChangeEvent` (no recompute), written via `state.save_manifest`→`atomic_write_json` to runtime `data/_state/manifests/<table>/<run_id>.json`; captures run id, pipeline/contracts version, window/watermark, per-partition INPUT(raw dir)→OUTPUT(curated path) + row counts + task-1.5 schema events + freshness; surfaced on `RunSummary.manifest_path` and in the CLI summary line. Dry-run writes no manifest. Evidence: `tests/unit/test_lineage.py` (7 tests: lineage columns + correct values, per-partition source match, DATA idempotency excluding lineage, manifest inputs→outputs/counts match RunSummary, schema events included on change, dry-run writes none, `RunManifest.from_run`/`to_dict` round-trip) + adjusted `tests/unit/test_pipeline.py`; `uv run pytest` 77 passed / 1 skipped, ruff clean; verified over a bounded real partition (2026-06-17, 5342 curated rows: three lineage columns present, manifest JSON written, second run DATA-identical). No new dependency. See `.agents/tasks/task-1.6-verification.md`.
  - _Requirements: REQ-27_
- [x] **1.7 Labeled update-correctness fixture**
  - `tests/fixtures/incremental/` (TEAM-GENERATED banner): day N, late arrival, duplicate re-delivery, new column, broken column; tests assert the curated state after each step.
  - Done (2026-10-01): committed TEAM-GENERATED fixture under `tests/fixtures/incremental/` (scoped `.gitignore` with `!*.parquet` to un-ignore ONLY the fixture Parquet; root `.gitignore` untouched) — `generate.py` (team-authored header, R1-R6 delivery frames + materializer), `README.md` (provenance, delivery table, expected curated outcome per step, Known-limitation section), and 4 committed raw partitions modeling the `transactions` table. `tests/unit/test_incremental_fixture.py` drives `run_table`/`process_partition` over the deliveries in order with `tmp_path` dirs (never touches real `data/`) and asserts the curated state after each step. All 5 REQ-26 scenarios present in the fixture data and now PASSING as normal tests: (a) day-N, (b) late arrival within window + out-of-window untouched, (c) duplicate collapsed latest-wins NOT quarantined, (d-additive new nullable column accepted + event logged + curated), (d-breaking missing required column fails closed with `SchemaEvolutionError`, no curated partition, prior curated intact). Driven over `tmp_path`; no new dependency. Approved by semantic review.
  - **BUG-001 (resolved):** the two schema-change scenarios initially exposed a data-layer bug — `LocalSource._source` read facts as one DuckDB glob over `<table>/**/*.parquet` without per-partition schema isolation (added column silently dropped; missing-required raised a raw DuckDB IOException before the fail-closed gate). Fixed in `src/cora/data/datasource.py`: a fully-specified single-partition fact read now narrows `read_parquet` to that partition's files (`year=Y/month=MM/day=DD/*.parquet`), restoring REQ-25 schema evolution and the fail-closed property. Mirrored in `S3Source._source` (network-free); `_SampledFactSource` in `quality/report.py` updated for the new `_source` signature. `union_by_name=true` deliberately NOT used (would NULL-fill and defeat fail-closed). The two `xfail(strict=True)` markers removed — they pass as normal tests; added a direct DataSource isolation unit test. `uv run pytest` 83 passed / 1 skipped / 0 xfailed, ruff clean; no new dependency. Report: `.agents/tasks/BUG-001-multifile-scan.md` (resolved).
  - _Requirements: REQ-26_
- [x] **1.8 Extend EDA (problem evidence)**
  - Transcripts (utterance lengths, language, intents vocabulary, label quality), transactions (status mix, fraud score distribution, currencies), surveys (CSAT by category), data-quality findings; update `analysis/EDA_FINDINGS.md`.
  - Done (2026-10-01): `analysis/eda_full_history.py` (memory-safe DuckDB over Parquet, `memory_limit=2GB`, aggregated/sampled SQL; `digital_events` 15.6M as aggregates). Added `analysis/EDA_FINDINGS.md` section 10 (10.1 cross-table joinability, 10.2 temporal/behavioral, 10.3 capability→table→column map, 10.4 escalation evidence, 10.5 full-history data-quality); sections 1-9 untouched. Evidence: `analysis/figures/{10_..15_}.png` + `metrics.json`. Business findings: FCR Transaccional 91.5% / Producto 89.6% vs Queja 43.6%; `complaints.origin_interaction_id` 0% full-history (no dispute-intake join); freeze/unfreeze PARTIAL (no native field, via `product_status` + sandbox); all raw columns landed as VARCHAR (contracts must cast). Approved by semantic review.
  - _Requirements: REQ-01, REQ-02_

## Phase 2 - Identity, tools & policy (Days 3-4)

- [x] **2.1 Mock identity service**
  - OTP step + HMAC JWT with TTL and `jti`; tampered/expired token tests; "ID only" refusal test.
  - Done (2026-10-02): `src/cora/identity/` (`service.py`, `__init__.py`). OTP step then an HMAC-signed JWT with `sub=customer_id`, absolute-TTL `exp` (default 15 min, configurable) and a unique `jti`; verification FAILS CLOSED on tampered or expired tokens (discloses nothing, discards pending confirmations, requires re-auth); supplying only a customer id/number yields refusal, not data access. Signing key + TTL come from `settings.py` (names only in `.env.example`; the mock fails closed when the key is unset). Tokens are never echoed to the LLM. New dep `pyjwt==2.10.1`. Evidence: `tests/unit/test_identity.py` (valid round-trip, tampered rejected, expired rejected via injected clock, ID-only refusal, OTP guard). `uv run pytest` green, ruff clean. Approved by semantic review (clean, no blocking findings).
  - _Requirements: REQ-10, REQ-11_
- [x] **2.2 Tool layer with authorization**
  - Tools of design §6 with Pydantic contracts, `customer_id` injected from session, `FORBIDDEN` on foreign resources, `as_of` and `source_refs`.
  - Evidence: `documentation/TOOL_CONTRACTS.md` (inputs, outputs, errors, side effects, limitations vs real API).
  - Done (2026-10-02): `src/cora/tools/` (`base.py` Status/Result/SourceRef/PII-masking/AccessLog, `models.py` Pydantic contracts with `extra="forbid"` and NO `customer_id` field, `layer.py` the 7 design-§6 tools bound to a verified Session, `handoff_store.py` Phase-4 seam, `__init__.py`). `customer_id` is injected from the task-2.1 session and is a read-only property (a smuggled `customer_id` is rejected by validation); foreign resources return `FORBIDDEN` and are logged (REQ-12); PANs masked to `****1234`; bounded DataSource reads over the VARCHAR landing; `convert_currency` follows I5/REQ-28 (exact-date rate, latest-prior-within-7-days flagged, >7-day abstain, no relabeling per ADR-016); NO money-movement tool exists (REQ-34, registry test). `freeze/unfreeze` declare the real contract + authZ (write lands in 2.3); `create_handoff` has real contract + authZ with ownership-checked `transaction_id`. Evidence: `documentation/TOOL_CONTRACTS.md`; `tests/unit/test_tools.py`. `uv run pytest` green, ruff clean. Approved by semantic review (blocking `create_handoff` authZ + `get_balance` available-credit findings fixed; 6 non-blocking notes carried to Phase 4).
  - _Requirements: REQ-12, REQ-28, REQ-34, REQ-38_
- [x] **2.3 Sandbox overlay + freeze/unfreeze**
  - Idempotent writes requiring a `confirmation_id`; read-back verification.
  - Done (2026-10-02): `src/cora/tools/card_overlay.py` (`CardFreezeState`, `CardOverlay` protocol, `InMemoryCardOverlay`, `JsonCardOverlay` with atomic write) + `confirmations.py` (`CardAction`, `Confirmation` bound to customer+action+product+TTL, `ConfirmationStore`). `freeze_card`/`unfreeze_card` in `layer.py` now do a real overlay write: authorize → require a valid, unexpired confirmation bound to THIS customer+action+product (REQ-14; else INVALID, no write) → idempotent `put` (double-freeze is a no-op success) → read-back; reports OK only if the post-condition holds, else UNAVAILABLE "not completed, offering a human" (REQ-09). A raising overlay (corrupt/unwritable `data/_state/`) fails closed the same way. Source/curated data stays read-only; freeze is an overlay flag (no native field — documented limitation). Overlay persisted to gitignored `data/_state/card_overlay.json`. Evidence: `tests/unit/test_card_overlay.py` (freeze→read-back frozen, idempotent double-freeze, unfreeze reverses, unknown/expired/wrong-product/wrong-action confirmation refuse, foreign card FORBIDDEN, post-condition-fail + overlay-error → UNAVAILABLE+escalation, JSON persist/reload). `uv run pytest` green, ruff clean. Approved by semantic review (blocking REQ-09 tool-error branch fixed; non-blocking partial-write handoff marker deferred to Phase 4).
  - _Requirements: REQ-09, REQ-14_
- [x] **2.4 Policy engine**
  - `rules.yaml` (design §5), engine with rule ids, allow-list, unit tests per rule (table-driven).
  - Done (2026-10-02): `src/cora/policy/` (`rules.yaml` v0.1.0 encoding the design §5 table exactly POL-000..POL-999 with placeholder thresholds `tau_fraud=70.0`/`tau_escalate=0.40`/`tau_clarify=0.60` and an explicit allow-list; `engine.py` loads via `yaml.safe_load`, evaluates top-down first-match-wins, returns `PolicyDecision` with the matched `rule_id` + `rules_version` for traces; `__init__.py`). Typed `PolicyInput` (no free text — instruction-like input is data, REQ-35); `AllowList.permits` rejects an LLM proposal outside the allow-list (REQ-13, LLM proposes/policy decides); injectable `Thresholds` (final values set in Phase 3); credit→`abstain_route` `CREDIT_OUT_OF_SCOPE` never evaluated (REQ-33); money→`refuse` (REQ-34); fail-closed YAML validation. New dep `pyyaml==6.0.3`; `uv.lock` updated for both Phase-2 deps. Evidence: `tests/unit/test_policy.py` (table-driven, one case per rule id proving first-match-wins, allow-list rejection, credit→abstain_route, money→refuse, fraud high-priority, injection-is-data, injectable thresholds, YAML-coverage guard). `uv run pytest` 175 passed / 1 skipped, ruff clean. Approved by semantic review (6 non-blocking findings deferred to Phase 4 wiring).
  - _Requirements: REQ-13, REQ-15, REQ-33, REQ-34_

## Phase 3 - NLU & learned component (Days 4-6)

- [x] **3.1 Labeling guide & gold set**
  - `documentation/LABELING_GUIDE.md`; **team-written** labeled utterances (es MX/CO/AR + pt) - dataset transcripts are templated and only seed examples (EDA F7); provenance column; double-label 20% and report κ.
  - Done: `src/cora/nlu/labels.py` (re-exports policy `Intent`; `Provenance`/`Variant` StrEnums), `src/cora/nlu/goldset.py` (fail-closed TSV loader + Cohen's κ via a direct counting formula, no ML dep yet), `data/nlu/gold.tsv` (240 team-generated ES rows, 15/class balanced across all 16 classes, MX/CO/AR, authoring-wave `month`, `scenario_id` grouping) authored by `data/nlu/build_gold.py`, and `documentation/LABELING_GUIDE.md` (per-class guidance + anchors + provenance scheme + double-label protocol). 48 rows double-labeled (20.0%), **Cohen's κ = 0.733**. Evidence: `tests/unit/test_goldset.py` (real-file balance/κ + 11 fail-closed cases). `ruff` clean; `uv run pytest` green.
  - _Requirements: REQ-30, REQ-37_
- [x] **3.2 Portuguese set**
  - ES->PT translation of the gold set (marked TRANSLATED) + native-style rewrites of 20%; reviewer notes; documented limitation.
  - Done: `data/nlu/gold_pt.tsv` (240 BR rows mirroring the ES set 1:1 - 192 `translated` + 48 `team-generated` native rewrites = 20%) authored by `data/nlu/build_gold_pt.py`; each row's `reviewer_note` records provenance + source `scenario_id`. `cora.nlu.goldset.check_pairing` (new, fail-closed) proves every ES scenario has a PT counterpart; double-labeling mirrors ES so Cohen's κ = 0.733 on both. `documentation/LABELING_GUIDE.md` gains a "Language coverage & limitations" section (synthetic/translationese caveat, per-language reporting deferred to 3.6). Evidence: `tests/unit/test_goldset.py` (+6 PT cases: balance, 20% rewrites, provenance notes, pairing, 2 fail-closed pairing cases). `ruff` clean; `uv run pytest` green.
  - _Requirements: REQ-18, REQ-19_
- [x] **3.3 Leakage-safe splits**
  - Group-by-customer + time split; near-duplicate removal; split report.
  - Done: `src/cora/nlu/splits.py` (deterministic, seeded) groups rows by the **source scenario** - an ES row and its PT translation share one group key (`scenario_id` with the PT `BR` segment stripped), the team-written analogue of REQ-31's `customer_id` grouping (D1) - then class-stratifies into 70/15/15 train/val/test so every intent and language is present in each split where it has enough groups (a thin class is flagged, never dropped), and removes cross-split near-duplicates by embedding cosine > 0.95 (train wins). The embedder is a lazy seam: `cora.nlu.embeddings.encode` if present (3.5), else a deterministic char-trigram hash fallback. `data/nlu/splits.json` records per-split counts, the authoring-wave composition, the seed, the 0.95 threshold, the 39 removed near-duplicates with examples, and the honest flag that X3 is emptied from validation by dedup (`missing_after_dedup`). Evidence: `tests/unit/test_splits.py` (no source scenario spans two splits, held-out near-dup removed only from the held-out side, every class/language present-or-flagged, thin class flagged-not-dropped, real-gold-set smoke). `ruff` clean; `uv run pytest` green.
  - _Requirements: REQ-31_
- [x] **3.4 Baselines**
  - Majority, keyword rules (es/pt), zero-shot LLM.
  - Done: `src/cora/agent/llm.py` adds the single shared `LLMClient` (Converse-API `BedrockLLMClient`, model id from `settings.bedrock_model_id` only, bounded `tenacity` retry, fail-closed `LLMUnavailable`, no silent model-id fallback - D3; plus `StubLLMClient` and a `get_llm_client` factory that returns the stub under `CORA_NLU_STUB=1` or when Bedrock is unconfigured). `src/cora/nlu/baselines.py` ships three `predict(...) -> list[Intent]` baselines: `MajorityBaseline` (B-maj, stdlib `Counter`, deterministic modal class), `KeywordBaseline` (B-kw, ordered es+pt regex rules, first-match-wins, no match -> `OTHER`; also the Phase-5 fallback), and `ZeroShotBaseline` (B-zs, prompts the LLM via versioned `src/cora/agent/prompts/zero_shot_intent.{es,pt}.txt`, masks PII first, validates the untrusted reply against `Intent`, invalid/empty -> `OTHER`). Offline measurement on the 480-row gold set (recorded in `data/nlu/model_card.json`): B-maj acc 0.062, B-kw acc 0.744 (es 0.812 / pt 0.675), B-zs skipped (Bedrock unconfigured, fails closed). New `nlu` dependency group pinned exactly (`scikit-learn` 1.5.2, `sentence-transformers` 3.0.1, `torch` 2.4.1, `fasttext-wheel` 0.9.2, `joblib` 1.4.2, `numpy` 1.26.4, `tenacity` 9.0.0), `uv.lock` resolved and installs clean on Windows/Py3.12 (fasttext fallback not needed). Evidence: `tests/unit/test_baselines.py` (8 cases). `ruff` clean; `uv run pytest` green.
  - _Requirements: REQ-29_
- [x] **3.5 Intent classifier**
  - Multilingual embeddings + calibrated logistic regression; TF-IDF variant for comparison; macro-F1, per-class recall, ECE, confusion matrix, es vs pt.
  - Done: `src/cora/nlu/embeddings.py` is the representation seam - a lazy `@lru_cache` `paraphrase-multilingual-MiniLM-L12-v2` loader (never imported at module import), a seeded hash `StubEncoder` selected under `CORA_NLU_STUB=1` (tests/offline, no download), and a `char_wb` 3-5 `TfidfVectorizer` builder (the "no cross-lingual transfer" comparison, NIT-3). `src/cora/nlu/classifier.py` fits `CalibratedClassifierCV(LogisticRegression(multinomial), cv=5, method="sigmoid")` (sigmoid/Platt per D2, not isotonic) over frozen embeddings and, as a Pipeline, over TF-IDF; `IntentClassifier` exposes `predict`/`predict_proba` with `classes_` as a fixed `Intent` column order and joblib save/load of the calibrated head only (`data/nlu/intent_head.joblib`, 51 KB; no embedding weights committed - NIT-6). Offline run `data/nlu/train_intent_model.py` regenerates `splits.json`, trains both heads on the leakage-safe train split and evaluates on test: macro-F1, per-class recall (escalation E1-E4 called out), ECE, confusion matrix, es-vs-pt breakdown, and sigmoid-vs-isotonic reliability (figures in `documentation/reports/figures/`); numbers in `data/nlu/model_card.json.intent_model_results`. **Honest provenance:** the sandbox could not download the MiniLM weights (HF SSL blocked), so the embedding-head numbers are from the `StubEncoder` and labeled as such; the TF-IDF numbers are download-free and real (macro-F1 0.9321). The real-MiniLM path in the script is unchanged and re-runs on a host with HF access. Train carries 15/16 classes (X3 emptied by 3.3 dedup; stated, not inflated). Evidence: `tests/unit/test_classifier.py` (6 cases, stub). `ruff` clean; `uv run pytest` 195 passed / 21 skipped. The full report prose lands in `intent_model.md` at 3.6.
  - _Requirements: REQ-29, REQ-32_
- [x] **3.6 Threshold selection**
  - Cost-matrix optimization of `τ_clarify`, `τ_escalate` on validation; frozen before test.
  - Evidence: `documentation/reports/intent_model.md` with figures and error analysis.
  - Done: `src/cora/nlu/thresholds.py` selects `τ_escalate`/`τ_clarify` by an exhaustive cost-matrix sweep over the policy engine's **three** POL-060/070/090 confidence bands (`conf<τ_escalate`→escalate, `τ_escalate≤conf<τ_clarify`→clarify, else answer - the real top-down rule semantics, not a free 3-way argmin), minimising expected cost under the named matrix `unsafe_answer=100 >> unnecessary_escalate=10 >> clarify=1 >> correct=0`; the invariant `τ_escalate ≤ τ_clarify` is enforced as a hard grid constraint **and** a final `assert` so the frozen pair is always representable by POL-060/POL-070. Selection runs on the **validation** split and `freeze(...)` writes `data/nlu/thresholds.json` **before** the test split is touched (REQ-47 freeze discipline). Offline run (`data/nlu/train_intent_model.py`, extended) froze `τ_escalate=0.20`, `τ_clarify=0.50` (validation n=45, cost 33.0), produced `documentation/reports/figures/intent_cost_vs_threshold.png`, and generated `documentation/reports/intent_model.md` (representation + metrics justification, baseline-vs-classifier table, reliability + confusion figures, cost-vs-threshold figure, es/pt breakdown with sizes, the frozen taus citing POL-060/070, the freeze-order honesty note, and an **error-analysis section with real misclassified test examples**). `tau_fraud` (POL-040) is NOT re-derived. Evidence: `tests/unit/test_thresholds.py` (6 cases). `ruff` check + format clean; `uv run pytest` 201 passed / 21 skipped.
  - _Requirements: REQ-32, REQ-47_
- [ ] **3.7 Language detection, entity extraction, injection screen**
  - fastText lid; LLM structured entities validated by schema; injection rules + classifier with logged hits.
  - _Requirements: REQ-18, REQ-35, REQ-07_

## Phase 4 - Agent orchestration (Days 5-7)

- [ ] **4.1 LangGraph state machine**
  - Nodes/edges of design §4; session state store; reference resolution across turns.
  - _Requirements: REQ-06, REQ-07, REQ-13_
- [ ] **4.2 Response generation + templates**
  - Versioned prompts (hash in trace), es/pt templates for every decision, PII masking before LLM.
  - _Requirements: REQ-18, REQ-36, REQ-40_
- [ ] **4.3 Grounding checker**
  - Number/entity diff against tool results; block + template fallback; tests with adversarial generations.
  - _Requirements: REQ-08_
- [ ] **4.4 Confirmation protocol & verified actions**
  - _Requirements: REQ-09, REQ-14_
- [ ] **4.5 Escalation & handoff package + dispute intake**
  - Package schema, store, agent-console view.
  - _Requirements: REQ-15, REQ-16, REQ-17_
- [ ] **4.6 Credit guard & money-movement refusal**
  - _Requirements: REQ-33, REQ-34_

## Phase 5 - Reliability, observability & service (Days 6-7)

- [ ] **5.1 Tracing** - OTel spans per node/tool, JSONL exporter, optional Langfuse; trace id in UI and handoff. _Requirements: REQ-39_
- [ ] **5.2 Retries, timeouts, circuit breaker, fallbacks** - fault-injection tests. _Requirements: REQ-40_
- [ ] **5.3 FastAPI service** - `/auth`, `/chat`, `/handoffs`; stateless handlers. _Requirements: REQ-49, REQ-51_
- [ ] **5.4 Demo UI** - Streamlit customer view (language toggle, trace link) + agent console. _Requirements: REQ-05, REQ-49_

## Phase 6 - Evaluation (Days 7-9)

- [ ] **6.1 Scenario builder** - ~400 held-out cases per design §9 with deterministic reference outcomes; fixed seed; `scenarios.jsonl` versioned. _Requirements: REQ-42_
- [ ] **6.2 Naive baseline B1** - single-prompt LLM with pasted data. _Requirements: REQ-41_
- [ ] **6.3 Runner** - B1 and CORA ×3 repeats, fault injection on. _Requirements: REQ-41, REQ-44_
- [ ] **6.4 Deterministic judges** - facts, policy compliance, disclosure/authorization, action verification, escalation correctness, handoff completeness. _Requirements: REQ-43_
- [ ] **6.5 LLM judge + validation** - rubric doc; 50 human labels; agreement. _Requirements: REQ-45_
- [ ] **6.6 Metrics & statistics** - SAR, attempted share, containment, escalation P/R (missed/unnecessary), unsafe with CI and rule of three, p50/p95, cost per attempted/per SAR. _Requirements: REQ-43_
- [ ] **6.7 Fairness breakdown** - by language, country/accent, segment; disparity investigation. _Requirements: REQ-46_
- [ ] **6.8 Historical baseline B0** - FCR/AHT from data, labeled historical. _Requirements: REQ-41, REQ-47_
- [ ] **6.9 Evaluation report** - `documentation/reports/EVALUATION.md`: setup, versions, results, failures, error analysis, labeled evidence types. _Requirements: REQ-44, REQ-47_

## Phase 7 - Production readiness & submission (Days 9-10)

- [ ] **7.1 Load test** - locust/k6 against local API; capacity and bottlenecks. _Requirements: REQ-50, REQ-51_
- [ ] **7.2 Production-readiness dossier** - monitoring, alerting, access control, retention, AWS target architecture, remaining work. _Requirements: REQ-50_
- [ ] **7.3 Trade-off register & outcomes** - finalize REQ-02 / REQ-03 tables with measured numbers. _Requirements: REQ-02, REQ-03_
- [ ] **7.4 Limitations** - data, language, synthetic labels, small samples. _Requirements: REQ-19, REQ-47_
- [ ] **7.5 Demo script & video** - all five demo paths in es and pt. _Requirements: REQ-05_
- [ ] **7.6 Submission audit** - traceability 100%, no secrets/PDFs (gitleaks + manual), clean-clone reproduction. _Requirements: REQ-48, REQ-52_

## Phase 8 - AWS migration (after submission or if time allows)

- [ ] **8.1** CDK stack: S3 + Glue + Athena, DynamoDB, Fargate/Lambda, Cognito, Secrets Manager, CloudWatch/X-Ray. _Requirements: REQ-50, REQ-51_
- [ ] **8.2** Switch adapters via config; rerun eval in AWS; label results as a separate environment.
