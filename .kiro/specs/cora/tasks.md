# CORA - Implementation Tasks

**Spec:** `cora` · **Phase:** Tasks (3 of 3) · **Status:** Draft v1.0 · **Date:** 2026-10-01
Implements [`design.md`](design.md) for [`requirements.md`](requirements.md). Schedule and process: [`documentation/DEVELOPMENT_PLAN.md`](../../../documentation/DEVELOPMENT_PLAN.md).

Rules for every task:
- One branch per task group (`feat/<area>-<short>`), PR to `main`, never commit on `main`.
- A task is **done** only when its *Evidence* exists in the repo (test, report, figure or doc) and the traceability matrix is updated.
- Code and docs in English; customer-facing strings in es/pt.

---

## Phase 0 - Foundations (Day 1)

- [ ] **0.1 Project skeleton & tooling**
  - `pyproject.toml` with `uv`, pinned deps, `ruff`, `pytest`, `src/cora` package, `Makefile` + `tasks.ps1` (Windows) with `setup / data / test / eval / demo`.
  - Evidence: `make test` green on an empty suite in CI.
  - _Requirements: REQ-48_
- [ ] **0.2 CI**
  - GitHub Actions: lint, unit tests, contract tests on the fixture; secret scanning (gitleaks).
  - _Requirements: REQ-36, REQ-48_
- [x] **0.3 Config & secrets**
  - Evidence: `src/cora/settings.py`, `tests/unit/test_settings.py` (template keys match settings, no credential shapes, blank template parses).
  - `settings.py` (pydantic-settings), `.env.example` with variable names only; AWS profile name configurable.
  - _Requirements: REQ-36, REQ-48_

- [ ] **0.4 AWS account bootstrap & Bedrock smoke test** (owner executes, Kiro assists)
  - Follow [`documentation/AWS_SETUP.md`](../../../documentation/AWS_SETUP.md) steps 1-7: root MFA, budget + anomaly alerts, IAM Identity Center with `CoraAdmin` / `CoraDeveloper` (least privilege, `infra/iam/cora-developer-policy.json`), SSO profile `cora-dev`, Anthropic FTU form, model ids in `.env`.
  - Run `python scripts/aws/verify_bedrock.py`; record masked output (latency, tokens, es/pt answers) in `documentation/reports/bedrock_smoke_test.md`.
  - _Requirements: REQ-36, REQ-40, REQ-43, REQ-48_

## Phase 1 - Data platform (Days 1-3)

- [ ] **1.1 DataSource interface**
  - `LocalSource` (sample folder) and `S3Source` (profile `cora-datathon`); `scripts/fetch_sample` pulls a reproducible sample (fixed months, all dimensions).
  - _Requirements: REQ-20_
- [ ] **1.2 Contracts for all 13 tables**
  - Pandera schemas from the data dictionary (types, nullability, PK, enums, ranges); contract tests.
  - _Requirements: REQ-21_
- [ ] **1.3 Quality checks + quarantine**
  - Duplicates (PK and row hash), null rates, orphan FKs, enums, ranges, currency; quarantine with reason; run report.
  - Evidence: `documentation/reports/data_quality.md` with measured rates vs documented (~2% dup, ~5% null).
  - _Requirements: REQ-22, REQ-23_
- [ ] **1.4 Incremental pipeline**
  - Watermark by `process_date`, reprocessing window W=3, idempotent partition overwrite, dedup latest-wins, freshness record.
  - _Requirements: REQ-24_
- [ ] **1.5 Schema-evolution handling**
  - Additive vs breaking change detection; events in the manifest.
  - _Requirements: REQ-25_
- [ ] **1.6 Lineage**
  - Row-level `_source_file`, `_ingested_at`, `_pipeline_version`; run manifest JSON.
  - _Requirements: REQ-27_
- [ ] **1.7 Labeled update-correctness fixture**
  - `tests/fixtures/incremental/` (TEAM-GENERATED banner): day N, late arrival, duplicate re-delivery, new column, broken column; tests assert the curated state after each step.
  - _Requirements: REQ-26_
- [ ] **1.8 Extend EDA (problem evidence)**
  - Transcripts (utterance lengths, language, intents vocabulary, label quality), transactions (status mix, fraud score distribution, currencies), surveys (CSAT by category), data-quality findings; update `analysis/EDA_FINDINGS.md`.
  - _Requirements: REQ-01, REQ-02_

## Phase 2 - Identity, tools & policy (Days 3-4)

- [ ] **2.1 Mock identity service**
  - OTP step + HMAC JWT with TTL and `jti`; tampered/expired token tests; "ID only" refusal test.
  - _Requirements: REQ-10, REQ-11_
- [ ] **2.2 Tool layer with authorization**
  - Tools of design §6 with Pydantic contracts, `customer_id` injected from session, `FORBIDDEN` on foreign resources, `as_of` and `source_refs`.
  - Evidence: `documentation/TOOL_CONTRACTS.md` (inputs, outputs, errors, side effects, limitations vs real API).
  - _Requirements: REQ-12, REQ-28, REQ-34, REQ-38_
- [ ] **2.3 Sandbox overlay + freeze/unfreeze**
  - Idempotent writes requiring a `confirmation_id`; read-back verification.
  - _Requirements: REQ-09, REQ-14_
- [ ] **2.4 Policy engine**
  - `rules.yaml` (design §5), engine with rule ids, allow-list, unit tests per rule (table-driven).
  - _Requirements: REQ-13, REQ-15, REQ-33, REQ-34_

## Phase 3 - NLU & learned component (Days 4-6)

- [ ] **3.1 Labeling guide & gold set**
  - `documentation/LABELING_GUIDE.md`; **team-written** labeled utterances (es MX/CO/AR + pt) - dataset transcripts are templated and only seed examples (EDA F7); provenance column; double-label 20% and report κ.
  - _Requirements: REQ-30, REQ-37_
- [ ] **3.2 Portuguese set**
  - ES->PT translation of the gold set (marked TRANSLATED) + native-style rewrites of 20%; reviewer notes; documented limitation.
  - _Requirements: REQ-18, REQ-19_
- [ ] **3.3 Leakage-safe splits**
  - Group-by-customer + time split; near-duplicate removal; split report.
  - _Requirements: REQ-31_
- [ ] **3.4 Baselines**
  - Majority, keyword rules (es/pt), zero-shot LLM.
  - _Requirements: REQ-29_
- [ ] **3.5 Intent classifier**
  - Multilingual embeddings + calibrated logistic regression; TF-IDF variant for comparison; macro-F1, per-class recall, ECE, confusion matrix, es vs pt.
  - _Requirements: REQ-29, REQ-32_
- [ ] **3.6 Threshold selection**
  - Cost-matrix optimization of `τ_clarify`, `τ_escalate` on validation; frozen before test.
  - Evidence: `documentation/reports/intent_model.md` with figures and error analysis.
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
