# Traceability Matrix

Maps every source obligation ([`SOURCE_REQUIREMENTS.md`](SOURCE_REQUIREMENTS.md)) to the spec
requirement that satisfies it ([`requirements.md`](../.kiro/specs/cora/requirements.md)), the task
that implements it ([`tasks.md`](../.kiro/specs/cora/tasks.md)) and the evidence a judge can open.

**Why this exists:** the brief says *"depth, demonstrated behavior, and engineering judgment determine
the score"*. A matrix proves nothing in the brief was skipped and points straight at the proof.
It is updated in the same PR as any scope change; a row is `Done` only when its evidence exists.

Status: `Planned` · `Partial` · `Done`. Coverage today: **84 / 84 SRC mapped** (0 unmapped).

| SRC | Obligation (short) | REQ | Task | Evidence | Status |
|---|---|---|---|---|---|
| SRC-01 | Working AI-first system | 04, 49 | 5.3, 5.4 | API + demo UI | Planned |
| SRC-02 | Understand complex interactions | 06 | 4.1 | multi-turn scenarios | Planned |
| SRC-03 | Use data/tools securely | 10, 12 | 2.1, 2.2 | `tests/unit/test_identity.py` (authN) + `tests/unit/test_tools.py` (authZ, session-injected `customer_id`) | Done |
| SRC-04 | Complete service workflows | 04 | 4.1 | state machine + scenarios | Planned |
| SRC-05 | Involve humans when needed | 15, 16 | 4.5 | handoff packages | Planned |
| SRC-06 | Focused problem, end-to-end | 01, 04 | 1.8 | `analysis/EDA_FINDINGS.md` | Partial |
| SRC-07 | Data explains why it matters | 01 | 1.8 | `analysis/EDA_FINDINGS.md` | Partial |
| SRC-08 | Establish baseline | 02, 41 | 6.2, 6.8 | EVALUATION.md | Planned |
| SRC-09 | Measure quality & efficiency gains | 02, 43 | 6.6, 6.9 | EVALUATION.md | Planned |
| SRC-10 | Privacy, explainability, fairness, reliability, scalability | 36, 39, 46, 50, 51 | 4.2, 5.1, 6.7, 7.2 | dossier + reports | Planned |
| SRC-11 | Explicit trade-offs | 03 | 7.3 | trade-off register | Planned |
| SRC-12 | Justify AI vs deterministic; how evaluated | 03 | 7.3 | design §2, DECISIONS | Partial |
| SRC-13 | Prototype + readiness + honest remaining work | 48, 50, 52 | 7.2, 7.6 | dossier | Planned |
| SRC-14 | No live service expected | 50 | 7.2 | dossier scope note | Planned |
| SRC-15 | One coherent workflow | 01, 04, 17 | 1.8 | ADR-001 | Done |
| SRC-16 | Depth over breadth | 01, 04 | - | ADR-001 | Done |
| SRC-17 | Normal resolution path | 05 | 7.5 | demo + scenarios | Planned |
| SRC-18 | Ambiguous / unsupported request | 05, 07 | 7.5 | demo + scenarios | Planned |
| SRC-19 | Human-intervention case | 05, 15, 17 | 4.5, 7.5 | demo + handoff | Planned |
| SRC-20 | Spanish and Portuguese | 05, 18 | 3.2, 7.5 | per-language results | Planned |
| SRC-21 | Report data/language limitations | 19, 52 | 7.4 | LIMITATIONS section | Planned |
| SRC-22 | Analyze reasons, demand, quality, constraints | 01 | 1.8 | EDA + data_quality.md | Partial |
| SRC-23 | Prioritize workflow; define outcomes | 01, 02 | 1.8, 7.3 | outcomes table | Partial |
| SRC-24 | Conversational context | 06 | 4.1 | context tests | Planned |
| SRC-25 | Clarify ambiguity | 07 | 4.1 | clarification scenarios | Planned |
| SRC-26 | Grounded factual answers | 08, 28 | 4.3 | grounding tests | Planned |
| SRC-27 | Use tools when they serve | 09 | 2.2, 4.4 | tool layer `tests/unit/test_tools.py` (2.2 done); agent-wired traces pending (4.4) | Partial |
| SRC-28 | Report only verified actions | 09 | 2.3, 4.4 | `tests/unit/test_card_overlay.py` read-back + fail-closed (2.3 done); agent flow pending (4.4) | Partial |
| SRC-29 | Define answerable requests | 04, 13 | 2.4 | `src/cora/policy/rules.yaml` + `tests/unit/test_policy.py` | Done |
| SRC-30 | Define confirmation-required actions | 13, 14 | 4.4 | confirmation tests | Planned |
| SRC-31 | Define abstain / transfer | 13, 15 | 2.4, 4.5 | `rules.yaml` POL-050/999 + `tests/unit/test_policy.py` (2.4 done); handoff execution pending (4.5) | Partial |
| SRC-32 | Policy enforced outside prose | 12, 13 | 2.2, 2.4 | `src/cora/policy/engine.py` first-match + allow-list, `tests/unit/test_policy.py` + `test_tools.py` | Done |
| SRC-33 | Handoff contents | 16 | 4.5 | handoff schema | Planned |
| SRC-34 | Contracts, checks, lineage, freshness | 20-24, 27 | 1.1-1.6 | manifests + reports | Planned |
| SRC-35 | Learned component vs baseline | 29 | 3.4, 3.5 | intent_model.md | Planned |
| SRC-36 | Valid labels | 30, 37 | 3.1 | `documentation/LABELING_GUIDE.md` + `data/nlu/gold.tsv` (team-written, provenance-tagged, 240 rows/16 classes) + `cora.nlu.goldset` loader (fail-closed) + `tests/unit/test_goldset.py`; Cohen's κ = 0.733 | Done |
| SRC-37 | Prevent leakage | 31 | 3.3 | split report | Planned |
| SRC-38 | Justify representation, metrics, thresholds, splits | 31, 32 | 3.3, 3.6 | intent_model.md | Planned |
| SRC-39 | Held-out evaluation | 42 | 6.1 | scenarios.jsonl | Planned |
| SRC-40 | Incorrect / missing data cases | 42 | 6.1 | adversarial results | Planned |
| SRC-41 | Expired sessions | 11, 42 | 2.1, 6.1 | `tests/unit/test_identity.py` expired/tampered token fail-closed (2.1 done); eval cases pending (6.1) | Partial |
| SRC-42 | Unauthorized access attempts | 12, 42 | 2.2, 6.1 | `tests/unit/test_tools.py` FORBIDDEN + access log (2.2 done); eval cases pending (6.1) | Partial |
| SRC-43 | Prompt injection | 35, 42 | 3.7, 6.1 | adversarial results | Planned |
| SRC-44 | Tool failures | 40, 42 | 5.2, 6.1 | fault-injection results | Planned |
| SRC-45 | Multilingual ambiguity | 07, 18, 42 | 3.7, 6.1 | adversarial results | Planned |
| SRC-46 | Report outcomes w/ sizes & limits | 43 | 6.6 | EVALUATION.md | Planned |
| SRC-47 | Tracing | 39 | 5.1 | trace samples | Planned |
| SRC-48 | Bounded retries | 40 | 5.2 | retry tests | Planned |
| SRC-49 | Safe fallback | 40 | 5.2 | fallback tests | Planned |
| SRC-50 | Reproducible setup | 20, 48 | 0.1, 7.6 | `pyproject.toml` + `uv.lock`, `Makefile`/`tasks.ps1` (0.1 done: lint/format/pytest green); clean-clone run pending (7.6) | Partial |
| SRC-51 | Capacity, monitoring, access, retention, remaining work | 50, 51 | 7.1, 7.2 | dossier + load test | Planned |
| SRC-52 | Explanations from sources/rules/records, not CoT | 08, 27, 39 | 5.1 | trace-based explanations | Planned |
| SRC-53 | Justified combination of techniques | 03 | 7.3 | design §2, DECISIONS | Partial |
| SRC-54 | DE + AI/ML rigor (labels, repr., leakage, error analysis) | 29, 47 | 3.5, 3.6 | intent_model.md | Planned |
| SRC-55 | Batch/incremental/streaming as needed | 24 | 1.4 | ADR-006 | Partial |
| SRC-56 | Labeled update-correctness fixture | 26 | 1.7 | fixture tests | Planned |
| SRC-57 | Organizer-approved data only | 37 | 3.1 | data provenance table | Planned |
| SRC-58 | Label real/synthetic/team-generated | 19, 30, 37 | 3.1, 3.2 | provenance column | Planned |
| SRC-59 | No private data/credentials in submission or LLM calls | 36, 52 | 0.2, 4.2, 7.6 | gitleaks job in `.github/workflows/ci.yml` (0.2 done); PII masking (4.2) + clean-clone audit (7.6) pending | Partial |
| SRC-60 | Mock tools with documented contracts | 38 | 2.2 | `documentation/TOOL_CONTRACTS.md` (inputs/outputs/errors/side-effects/limitations per tool) | Done |
| SRC-61 | Trusted authentication | 10, 11 | 2.1 | `tests/unit/test_identity.py` (OTP + HMAC JWT, TTL, jti, ID-only refusal) | Done |
| SRC-62 | Per-customer access in tool layer | 12 | 2.2 | `tests/unit/test_tools.py` FORBIDDEN on foreign resource + read-only `customer_id` | Done |
| SRC-63 | Credit separation / no invented eligibility | 33 | 2.4, 4.6 | `rules.yaml` POL-030 abstain_route + `test_policy.py` (2.4 done); agent guard pending (4.6) | Partial |
| SRC-64 | No lending or money movement | 17, 34 | 2.2, 4.6 | `test_tools.py` registry test: no money-movement tool exists (2.2 done); policy refusal path pending (4.6) | Partial |
| SRC-65 | Baseline vs system on same workload | 41 | 6.2, 6.3 | EVALUATION.md | Planned |
| SRC-66 | Cases mix, label quality, versions, variability | 42, 44 | 6.1, 6.3 | run metadata | Planned |
| SRC-67 | Include failures | 43, 47 | 6.9 | error analysis | Planned |
| SRC-68 | LLM-judge rubric + validation | 30, 45 | 6.5 | judge validation | Planned |
| SRC-69 | Safe automated resolution + attempted share | 43 | 6.6 | EVALUATION.md | Planned |
| SRC-70 | Containment (not sufficient) | 43 | 6.6 | EVALUATION.md | Planned |
| SRC-71 | Escalation quality, missed/unnecessary | 16, 43 | 6.4, 6.6 | EVALUATION.md | Planned |
| SRC-72 | Unsafe outcomes w/ counts & denominators | 43 | 6.4, 6.6 | EVALUATION.md | Planned |
| SRC-73 | p50/p95, cost per case and per SAR | 43 | 6.6 | EVALUATION.md | Planned |
| SRC-74 | By language & segment; disparities | 19, 46 | 6.7 | fairness section | Planned |
| SRC-75 | Label offline / simulation / projection | 47 | 6.9 | evidence labels | Planned |
| SRC-76 | ~2% duplicates | 22, 23 | 1.3 | `documentation/reports/data_profile.md` (observed 0%) | Partial |
| SRC-77 | ~5% nulls | 21, 22 | 1.2, 1.3 | `documentation/reports/data_profile.md` (structural nulls) | Partial |
| SRC-78 | Late arrivals | 24 | 1.4, 1.7 | fixture tests | Planned |
| SRC-79 | Schema evolution | 25 | 1.5, 1.7 | fixture tests | Planned |
| SRC-80 | Referential integrity + orphans | 22 | 1.3 | `documentation/reports/data_profile.md` (observed 0%) | Partial |
| SRC-81 | Regional Spanish variants | 18 | 3.1, 3.7 | per-country results | Planned |
| SRC-82 | Multi-currency + FX | 28 | 2.2 | `tests/unit/test_tools.py` `convert_currency` (exact-date rate, latest-prior-within-7-days, >7-day abstain, no relabeling) | Done |
| SRC-83 | Date-partitioned facts | 20 | 1.1, 1.4 | pipeline | Planned |
| SRC-84 | Synthetic, read-only, credentials not shared | 36 | 0.2, 0.3 | `.gitignore`, gitleaks + no-PDF jobs in `.github/workflows/ci.yml`, `tests/unit/test_settings.py` | Done |
