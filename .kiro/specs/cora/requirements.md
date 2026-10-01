# CORA - Requirements Specification

**Spec:** `cora` · **Phase:** Requirements (1 of 3) · **Status:** Draft v1.0 · **Date:** 2026-10-01
Next: [`design.md`](design.md) -> [`tasks.md`](tasks.md). Process: [`documentation/DEVELOPMENT_PLAN.md`](../../../documentation/DEVELOPMENT_PLAN.md).

Acceptance criteria use **EARS** notation (*Easy Approach to Requirements Syntax*):
`WHEN <trigger>` (event) · `IF <condition> THEN` (unwanted behavior) · `WHILE <state>` (state) ·
`THE SYSTEM SHALL` (ubiquitous). Every requirement lists the source obligations it satisfies
(`Traces:` -> [`documentation/SOURCE_REQUIREMENTS.md`](../../../documentation/SOURCE_REQUIREMENTS.md)).

> **Why EARS:** each criterion is a single testable sentence, so it maps 1:1 to an evaluation
> case or unit test. That is exactly what the judges ask for ("demonstrated behavior",
> "measured quality"); vague requirements cannot be measured.

---

## 0. Glossary

| Term | Meaning |
|---|---|
| **Customer** | A synthetic `customers` record interacting with CORA. |
| **Session** | A short-lived, signed, authenticated context bound to exactly one `customer_id`. |
| **Tool** | A deterministic Python function with a typed contract that reads or changes sandbox banking state. |
| **Policy engine** | Deterministic rule layer that decides what is allowed (answer / confirm / abstain / escalate). Not an LLM. |
| **Handoff package** | Structured record given to a human agent on escalation. |
| **In-scope case** | A test case whose correct outcome is automated resolution by policy. |
| **Safe automated resolution (SAR)** | In-scope case reaching the correct, policy-compliant outcome with no human. |
| **Unsafe outcome** | Unauthorized disclosure, unauthorized action, or materially incorrect outcome. |

---

## 1. Problem definition & outcomes (Axis 1)

### REQ-01 Data-supported problem statement
**User story:** As a judge, I want the chosen workflow justified with data, so that I trust it matters.
- THE SYSTEM's documentation SHALL quantify demand by contact reason, channel, resolution rate (FCR), handle time and data-quality issues from the supplied data, with reproducible code.
- THE documentation SHALL state the chosen workflow, the rejected alternatives and why.
- THE documentation SHALL list operational constraints found in the data (e.g. empty `origin_interaction_id`, Spanish-only text, flat complaint categories).
- **Traces:** SRC-06, SRC-07, SRC-15, SRC-16, SRC-22, SRC-23 · **Status:** largely done (`analysis/EDA_FINDINGS.md`), to be extended to transcripts, transactions and surveys.

### REQ-02 Intended outcomes & success targets
- THE documentation SHALL define customer outcomes (correct answer, time-to-resolution, fewer repeat contacts) and business outcomes (containment with correctness, cost per resolution, agent time saved) with explicit numeric targets set **before** evaluation.
- THE targets SHALL be compared against the baselines of REQ-41.
- **Traces:** SRC-08, SRC-09, SRC-23.

### REQ-03 Explicit trade-off register
- THE documentation SHALL contain a trade-off table covering autonomy, accuracy, latency, cost and human oversight, stating the chosen operating point and the evidence for it.
- THE documentation SHALL state, per component, whether AI or deterministic logic is used and why.
- **Traces:** SRC-11, SRC-12, SRC-53.

---

## 2. Workflow scope (Scope)

### REQ-04 Focused workflow: Account & Card Servicing
THE SYSTEM SHALL support exactly one coherent workflow - *account & card servicing* - composed of:

| Code | Capability | Mode |
|---|---|---|
| I1 | Account / card balance and available credit | Answer (read) |
| I2 | Recent transactions and transaction search (date, merchant, amount) | Answer (read) |
| I3 | Explain a transaction's status (approved / declined / pending / reversed) | Answer (read) |
| I4 | Card status, credit limit, payment due / days past due | Answer (read) |
| I5 | Currency conversion of an amount using the day's official rate | Answer (read) |
| I6 | List the customer's products | Answer (read) |
| A1 | Temporarily freeze a card | Action, **requires confirmation** |
| A2 | Unfreeze a card frozen by the customer | Action, **requires confirmation** |
| E1 | Dispute / unrecognized charge -> structured intake | **Escalate** with handoff |
| E2 | Complaint | **Escalate** with handoff |
| E3 | Suspected fraud signal | **Escalate** (priority) |
| E4 | Explicit request for a human, or repeated failure | **Escalate** |
| X1 | Credit-limit increase / loan eligibility | **Abstain + route** (credit guard, REQ-33) |
| X2 | Money movement (transfers, payments) | **Refuse** (out of scope by rule, REQ-34) |
| X3 | Anything else (advice, other customers, chit-chat) | **Abstain / redirect** |

- **Traces:** SRC-04, SRC-06, SRC-15, SRC-16, SRC-29.
- **Why this scope:** see ADR-001. Read paths I1-I6 are ~57% of demand with ~90% FCR; A1/A2 give a *real* confirmed action without moving money; E1-E4 give the required human path; X1-X3 give the required unsupported path.

### REQ-05 Mandatory demo paths
- THE demo SHALL include, in both Spanish and Portuguese: (a) a normal resolution (I1-I6), (b) a confirmed action (A1), (c) an ambiguous request that is clarified, (d) an unsupported request that is abstained, (e) an escalation with a handoff package.
- **Traces:** SRC-17, SRC-18, SRC-19, SRC-20.

---

## 3. Conversation (Axis 2)

### REQ-06 Conversational context
- THE SYSTEM SHALL keep per-session state (language, authenticated customer, referenced product, pending confirmation, open slots, turn history) across turns.
- WHEN the customer refers back ("that card", "o último", "the second one"), THE SYSTEM SHALL resolve the reference from session state.
- WHEN a session ends or expires, THE SYSTEM SHALL discard conversational state except the audit record.
- **Traces:** SRC-02, SRC-24.

### REQ-07 Ambiguity clarification
- IF a request maps to more than one product, intent or entity with confidence below threshold `τ_clarify`, THEN THE SYSTEM SHALL ask one targeted clarifying question listing the candidate options.
- IF the request remains ambiguous after 2 clarification turns, THEN THE SYSTEM SHALL escalate (E4).
- **Traces:** SRC-18, SRC-25, SRC-45.

### REQ-08 Grounded answers
- THE SYSTEM SHALL produce every factual value (amount, date, status, limit, rate) from a tool result in the current turn; the LLM SHALL NOT originate figures.
- THE response SHALL carry citations to the tool calls/records it used (`source_refs`), stored in the trace.
- WHEN a generated response contains a number not present in the tool results, THE grounding checker SHALL block it and fall back to a templated answer.
- **Traces:** SRC-26, SRC-52.

### REQ-09 Verified actions only
- WHEN THE SYSTEM performs an action (A1/A2), it SHALL re-read state after the write and report success only if the post-condition holds.
- IF the post-condition fails or the tool errors, THEN THE SYSTEM SHALL tell the customer the action was **not** completed and offer escalation.
- **Traces:** SRC-27, SRC-28.

---

## 4. Identity, authorization & permissions (Boundaries, Axis 3)

### REQ-10 Authentication
- THE SYSTEM SHALL require a trusted test session issued by a mock identity service (signed token + one-time code step) before any customer data is read.
- IF a user supplies only a customer number or national ID, THEN THE SYSTEM SHALL refuse data access and request authentication.
- **Traces:** SRC-03, SRC-61.

### REQ-11 Session expiry
- THE session token SHALL expire after a configurable TTL (default 15 min idle).
- IF a request arrives with an expired or tampered token, THEN THE SYSTEM SHALL disclose nothing, discard pending confirmations and ask to re-authenticate.
- **Traces:** SRC-41, SRC-61.

### REQ-12 Per-customer authorization in the tool layer
- EVERY tool SHALL receive the `customer_id` from the verified session, never from model output or user text.
- IF a tool is asked for a resource (product, transaction) not owned by the session customer, THEN it SHALL return `FORBIDDEN` and the event SHALL be logged as an access attempt.
- **Traces:** SRC-32, SRC-42, SRC-62.

### REQ-13 Action permissions
- THE policy engine SHALL hold an explicit allow-list mapping each intent to {answer, confirm, escalate, abstain, refuse}.
- THE LLM SHALL only *propose* a next step; THE policy engine SHALL decide. An LLM proposal outside the allow-list SHALL be rejected.
- **Traces:** SRC-29, SRC-30, SRC-31, SRC-32.

### REQ-14 Confirmation protocol
- WHEN an action requires confirmation, THE SYSTEM SHALL restate the exact action and target (masked product number) and wait for an explicit affirmative in the same session.
- IF the confirmation is absent, negative, ambiguous or arrives after expiry, THEN THE SYSTEM SHALL NOT execute.
- **Traces:** SRC-30.

---

## 5. Escalation & handoff (Axis 3)

### REQ-15 Escalation triggers
THE SYSTEM SHALL escalate WHEN any of: intent E1-E4; fraud signal (`is_fraud` or `fraud_score` ≥ `τ_fraud` on a referenced transaction); sentiment `Very Negative` for 2 consecutive turns; 2 failed clarifications; repeated tool failure after retries; policy engine returns `escalate`.
- **Traces:** SRC-05, SRC-19, SRC-31.

### REQ-16 Handoff package
WHEN escalating, THE SYSTEM SHALL create a handoff package containing: the customer request (verbatim + language), verified facts with source refs, actions taken and their verified outcomes, supporting evidence (transactions involved), unresolved questions, escalation reason code, priority, and the session transcript reference.
- THE package SHALL be visible in an agent console view.
- **Traces:** SRC-33, SRC-71.

### REQ-17 Dispute intake (E1)
- WHEN a customer disputes a charge, THE SYSTEM SHALL collect the transaction (identified via tool, not free text), the reason, and whether the card is still in possession; SHALL offer a card freeze (A1) when fraud is suspected; and SHALL hand off - it SHALL NOT promise refunds or outcomes.
- **Traces:** SRC-15, SRC-19, SRC-64.

---

## 6. Multilingual support

### REQ-18 Spanish and Portuguese
- THE SYSTEM SHALL detect the language per turn and reply in the customer's language (es / pt); code and docs remain English.
- WHEN language detection confidence is low or the turn mixes languages, THE SYSTEM SHALL keep the session language and confirm understanding.
- THE regional Spanish variants (MX, CO, AR) SHALL be handled (vocabulary like *tarjeta / plástico*, *saldo*, *cuotas*).
- **Traces:** SRC-20, SRC-45, SRC-81.

### REQ-19 Language coverage limitations
- THE documentation SHALL state that the supplied data is Spanish-only, that Portuguese evaluation data is team-generated (translated + reviewed), and report results per language with sample sizes.
- **Traces:** SRC-21, SRC-58, SRC-74.

---

## 7. Data platform (Axis 4)

### REQ-20 Repeatable ingestion
- THE pipeline SHALL ingest from a `DataSource` interface with two implementations (local sample / S3) and produce the same curated Parquet tables, re-runnable idempotently by one command.
- **Traces:** SRC-34, SRC-50, SRC-83.

### REQ-21 Data contracts
- EACH table used SHALL have a versioned contract (columns, types, nullability, PK, enums, ranges) matching the data dictionary.
- IF a batch violates a contract, THEN the pipeline SHALL quarantine the offending rows with a reason and continue; the run report SHALL show counts.
- **Traces:** SRC-34, SRC-77.

### REQ-22 Quality checks
THE pipeline SHALL check and report per run: duplicates (by PK and by full-row hash), null rates vs contract, orphan FKs, enum violations, out-of-range values (e.g. `credit_score` 300-850), currency validity.
- **Traces:** SRC-34, SRC-76, SRC-77, SRC-80.

### REQ-23 Deduplication
- THE pipeline SHALL deduplicate deterministically (keep latest `last_updated` / latest arrival per PK) and record removed duplicates.
- **Traces:** SRC-76.

### REQ-24 Incremental processing, late arrivals & freshness
- THE pipeline SHALL process fact tables incrementally by `process_date` partition with a high-watermark and a re-processing window `W` (default 3 days) to absorb late arrivals.
- THE pipeline SHALL publish a freshness record per table (`max_event_date`, `last_ingested_at`); WHILE data is older than the freshness SLA, THE SYSTEM SHALL tell the customer the data's as-of date.
- **Traces:** SRC-34, SRC-55, SRC-78.

### REQ-25 Schema evolution
- WHEN a new nullable column appears, THE pipeline SHALL accept it and log a schema-change event; WHEN a required column disappears or changes type, THE pipeline SHALL fail the table with a clear error.
- **Traces:** SRC-79.

### REQ-26 Update-correctness test fixture
- THE repository SHALL contain a clearly labeled synthetic fixture (`tests/fixtures/incremental/`, marked TEAM-GENERATED) simulating day-N delivery, late arrival, duplicate re-delivery and a schema change, with tests proving the curated output is correct after each step.
- **Traces:** SRC-56.

### REQ-27 Lineage
- EVERY curated row SHALL carry `_source_file`, `_ingested_at`, `_pipeline_version`; every run SHALL write a lineage manifest (inputs -> outputs, row counts, checks).
- **Traces:** SRC-34, SRC-52.

### REQ-28 Multi-currency correctness
- THE SYSTEM SHALL present amounts in the product's currency, and SHALL convert only with the `daily_exchange_rates` row of the relevant date, citing the rate and date.
- IF no rate exists for that date, THEN THE SYSTEM SHALL use the latest prior rate and state it, or abstain if older than 7 days.
- THE SYSTEM SHALL report the currency exactly as recorded and SHALL NOT relabel it (the data has no MXN; Mexican products are recorded in USD - ADR-016).
- **Traces:** SRC-26, SRC-82.

---

## 8. Learned component (Axis 4)

### REQ-29 Intent & routing classifier
- THE SYSTEM SHALL include a learned intent classifier mapping a customer utterance (es/pt) to {I1..I6, A1, A2, E1..E4, X1..X3, OTHER} with a calibrated confidence.
- IT SHALL be evaluated against baselines: (B-maj) majority class, (B-kw) keyword/regex rules, (B-zs) zero-shot LLM prompt.
- **Traces:** SRC-35, SRC-54.

### REQ-30 Valid labels
- THE labeled set SHALL be **team-written** utterances (Spanish MX/CO/AR variants and Portuguese) labeled with a written guideline - dataset transcripts are templated (42 distinct utterances, see `analysis/EDA_FINDINGS.md` F7) and may only seed examples, never test cases; a subset SHALL be double-labeled and agreement (Cohen's κ) reported; label provenance (dataset-seed / team-generated / translated) SHALL be recorded per row.
- **Traces:** SRC-36, SRC-58, SRC-68.

### REQ-31 Leakage prevention & splits
- Splits SHALL be grouped by `customer_id` and ordered in time (train on earlier months, test on later), with near-duplicate utterances removed across splits.
- Label-bearing fields (`detected_intents`, `reason_category`) SHALL NOT be model inputs.
- **Traces:** SRC-37, SRC-38.

### REQ-32 Justified representation, metrics & thresholds
- THE documentation SHALL justify the text representation (multilingual sentence embeddings), the metrics (macro-F1, per-class recall, ECE calibration), and the thresholds `τ_clarify` and `τ_escalate` chosen on the validation split by minimizing expected cost of unsafe automation.
- **Traces:** SRC-38.

---

## 9. Credit & money guardrails (Boundaries)

### REQ-33 Credit guard
- THE SYSTEM SHALL NOT evaluate, predict, promise or approve credit eligibility, limit increases or loans.
- WHEN asked, THE SYSTEM SHALL explain it cannot decide that in this channel and route to a human or the (out-of-scope) credit service, logging reason `CREDIT_OUT_OF_SCOPE`.
- **Rationale:** the brief requires that any credit flow separates conversation, risk model and eligibility policy behind a labeled synthetic service; we deliberately keep credit out of scope (depth over breadth) and enforce that with a rule, which is itself an evaluated behavior.
- **Traces:** SRC-63.

### REQ-34 No money movement
- THE tool layer SHALL expose no tool that moves money or changes balances; requests for transfers/payments SHALL be refused by policy.
- **Traces:** SRC-64.

---

## 10. Security & privacy

### REQ-35 Prompt-injection resistance
- Instructions found in user text, transcripts or tool results SHALL be treated as data; THE policy engine SHALL be unaffected by them.
- THE SYSTEM SHALL screen inputs with an injection detector (rules + classifier) and log hits; permissions SHALL hold even when the detector misses.
- **Traces:** SRC-43.

### REQ-36 Data minimization to external models
- Requests to any external LLM SHALL contain only the fields needed for the turn; document numbers, emails, phones, addresses and full card numbers SHALL be masked or omitted.
- No credentials or secrets SHALL ever be sent or committed.
- **Traces:** SRC-10, SRC-59, SRC-84.

### REQ-37 Data provenance labeling
- THE documentation SHALL label every dataset used as synthetic (organizer), team-generated, or translated, and confirm only organizer-approved data is used.
- **Traces:** SRC-57, SRC-58.

### REQ-38 Mock tool contracts
- EACH mock tool SHALL document inputs, outputs, errors, side effects, latency profile and limitations vs a real banking API.
- **Traces:** SRC-60.

---

## 11. Reliability & observability (Axis 6)

### REQ-39 Tracing & execution records
- EVERY turn SHALL produce a trace: input (masked), language, intent+confidence, policy decision+rule id, tool calls (args, result status, latency), retries, model+prompt version, tokens, cost, output, grounding check.
- Explanations shown to judges/agents SHALL be generated from these records and policy rules, not from model reasoning text.
- **Traces:** SRC-47, SRC-52.

### REQ-40 Bounded retries & safe fallback
- Tool and LLM calls SHALL use timeouts and at most N=2 retries with exponential backoff and jitter, only for idempotent operations.
- IF the LLM fails, THEN THE SYSTEM SHALL use deterministic templates; IF a tool fails after retries, THEN THE SYSTEM SHALL say so and offer escalation; THE SYSTEM SHALL never guess.
- **Traces:** SRC-44, SRC-48, SRC-49.

---

## 12. Evaluation (Axis 5 + Evaluation evidence)

### REQ-41 Baselines on the same workload
THE evaluation SHALL compare on the identical held-out scenario set:
- **B0 - historical human operation** (offline, from data: FCR, AHT by category) - labeled *historical, not comparable 1:1*;
- **B1 - naive LLM** (single prompt, data pasted, no policy engine, no grounding check);
- **CORA** (proposed).
- **Traces:** SRC-08, SRC-65.

### REQ-42 Held-out scenario suite
- THE suite SHALL be built from held-out customers/periods and cover: every intent; es and pt; MX/CO/AR; all segments; and adversarial categories (missing/incorrect data, expired session, unauthorized access, prompt injection, tool failure, multilingual ambiguity, credit request, money-movement request).
- EACH case SHALL have a reference outcome (expected decision, expected facts computed deterministically from data, expected escalation yes/no).
- **Traces:** SRC-39, SRC-40..45, SRC-66.

### REQ-43 Outcome metrics
THE report SHALL include, with numerators/denominators and 95% CIs:
- SAR rate over all in-scope cases + share of cases where automation was attempted;
- containment (explicitly noted as insufficient alone);
- escalation quality: precision/recall of transfers, missed and unnecessary transfers, handoff completeness score;
- unsafe outcomes by type with counts/denominators and an upper confidence bound (rule of three when zero);
- p50/p95 end-to-end latency; cost per attempted case and per SAR ("not defined" if SAR = 0), with token prices and workload assumptions.
- **Traces:** SRC-46, SRC-67, SRC-69, SRC-70, SRC-71, SRC-72, SRC-73.

### REQ-44 Versions & variability
- THE report SHALL record model ids, prompt versions (hash), code commit, data snapshot, and run each configuration ≥3 times reporting mean ± spread.
- **Traces:** SRC-66.

### REQ-45 LLM-as-judge (only where deterministic judging is impossible)
- Factual correctness and policy compliance SHALL be judged deterministically. An LLM judge MAY score tone/clarity/handoff usefulness; its rubric SHALL be published and validated on ≥50 human-labeled samples (agreement reported).
- **Traces:** SRC-68.

### REQ-46 Fairness / disparity analysis
- THE report SHALL break down SAR, unsafe rate, escalation and latency by language, country/accent and customer segment, flag gaps beyond a stated tolerance, investigate causes and note small-sample limits.
- **Traces:** SRC-10, SRC-74.

### REQ-47 Honest labeling of evidence
- EVERY figure SHALL be labeled *offline measurement*, *simulation* or *projection*; projected savings SHALL show the formula and assumptions; no claim of production improvement.
- Failures SHALL be included and analyzed (error analysis section with examples).
- **Traces:** SRC-54, SRC-67, SRC-75.

---

## 13. Operability & deliverables (Axis 6, Scope)

### REQ-48 Reproducible setup
- `make setup && make data && make eval && make demo` (or documented equivalents on Windows) SHALL reproduce results from a clean clone with pinned dependencies and a sample-data fetch script; secrets via env vars only.
- **Traces:** SRC-50, SRC-13.

### REQ-49 Service interface
- THE SYSTEM SHALL run as an HTTP API (chat turn, auth, handoff queue) plus a demo UI with customer and agent views.
- **Traces:** SRC-01, SRC-13.

### REQ-50 Production-readiness dossier
THE documentation SHALL cover: capacity limits and a load test result; monitoring/alerting plan; access controls; data retention (transcripts, traces, PII masking, TTLs); the AWS target architecture; and an honest "remaining work before deployment" list.
- **Traces:** SRC-10, SRC-13, SRC-14, SRC-51.

### REQ-51 Scalability path
- THE design SHALL be stateless per request (session state in a store), so it scales horizontally on AWS; the dossier SHALL state expected throughput and bottlenecks.
- **Traces:** SRC-10, SRC-51.

### REQ-52 Submission package
- THE final submission SHALL include: README, problem & data evidence, architecture, evaluation report, trade-offs, limitations, demo video/script, and this spec with full traceability - with no credentials or PDFs.
- **Traces:** SRC-13, SRC-21, SRC-59.

---

## Coverage

All 84 `SRC` items are mapped in [`documentation/TRACEABILITY.md`](../../../documentation/TRACEABILITY.md). Any change to scope must update that matrix in the same commit.
