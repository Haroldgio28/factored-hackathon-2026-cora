# CORA - Exploratory Data Analysis & Workflow Selection

**Date:** 2026-10-01
**Author:** Harold Uribe (with Kiro)
**Scope:** Factored AI & Data Hackathon 2026 - evidence for *"a problem supported by data"* (evaluation axis 1).

Reproduce with:

```bash
python analysis/eda_sample.py --outdir analysis/figures
```

Raw metrics: [`analysis/figures/metrics.json`](figures/metrics.json).

---

## 1. Sample

A local, multi-day sample was pulled from the organizer S3 bucket (`us-east-2`, read-only) to drive workflow selection without scanning all 19M rows.

| Table | Rows | Days | Date range |
|---|---|---|---|
| `call_center_interactions` | 105,485 | 168 | 2026-01-01 to 2026-06-18 |
| `complaints` | 10,332 | 168 | 2026 (Jan-Jun) |

> This is the full Jan-Jun 2026 window for both tables (they are small fact tables). It is a recent, representative demand slice, not a cherry-picked day.

## 2. Where the demand is

Call-center demand splits into six generic reason categories. Transactional + Product dominate (~57% combined).

![Demand by reason category](figures/01_demand_by_reason_category.png)

| reason_category | % of interactions |
|---|---|
| Transaccional | 34.9% |
| Producto | 22.2% |
| Queja (complaint) | 16.9% |
| Técnico | 15.0% |
| Comercial | 8.0% |
| Retención | 3.0% |

## 3. What is safely automatable vs. what must escalate

First-contact resolution (FCR = `was_resolved`) varies sharply by category, while the *escalation* flag is flat (~10% everywhere - synthetic noise). FCR is the real signal.

![FCR vs escalation](figures/02_fcr_vs_escalation.png)

| reason_category | FCR % | Median handle time (s) |
|---|---|---|
| Transaccional | **91.8%** | 203 |
| Producto | **89.7%** | 263 |
| Técnico | 69.8% | 361 |
| Comercial | 66.1% | 542 |
| Retención | 61.6% | 478 |
| Queja | **43.3%** | 431 |

**Reading:** Transactional and Product inquiries already resolve on first contact ~90% of the time and are the shortest calls -> highest-value, lowest-risk automation targets. Complaints resolve only 43% first-contact and take longer -> a natural *escalate-to-human* path, not an automation target.

## 4. Channel & language

![Channel mix](figures/03_channel_mix.png)

Phone dominates at 85%; digital text channels (App, Web Chat, WhatsApp, Email) together ~14.5%. A text-first assistant maps cleanly onto the digital channels and onto transcribed phone calls (`call_transcripts`).

![Customer accent](figures/04_customer_accent.png)

Customer accent is Mexican 34.9% / Colombian 21.1% / Argentine 14.0% / undetected 30.0%. **All text in the dataset is Spanish - there is no Portuguese data.** The hackathon still requires ES + PT interactions, so **Portuguese must be covered with our own labeled evaluation fixtures**, reported as a known data limitation.

## 5. Data-quality findings that shape the design

- **`contact_reason` == `reason_category`** in this sample (fully redundant; `contact_reason` carries no finer signal).
- **`complaints.origin_interaction_id` is 0% populated** -> there is *no* usable join between complaints and the originating call-center interaction in this data. This **weakens a "transaction-dispute intake" workflow** that would rely on that link.
- **`complaints.category` is near-uniform** (~2,000 each across Fees / Transactions / Service / Technical / Branch) -> little signal about "where it hurts"; synthetic flat distribution.
- `complaints.status` is mostly open/in-process (Open 29% + In Process 40%); `sla_breached` 20.5%.
- These are consistent with the documented quality challenges (~2% dup, ~5% null, late arrivals, schema evolution) and must be handled by data contracts + quality checks (axis 4).

## 6. Workflow decision

**Chosen workflow: Account & transaction information + payment/card inquiries (Transactional + Product).**

Rationale, grounded in the data above:
1. **Largest demand** - ~57% of all interactions.
2. **Highest safe-automation ceiling** - 90-92% FCR today; these are factual, policy-bounded lookups that an AI-first assistant can ground in `transactions`, `products`, `customers`, `daily_exchange_rates`.
3. **Shortest handle time** - best latency/cost profile for the efficiency metrics (axis 5).
4. **Clean escalation contrast** - Complaints (43% FCR) give us a credible, data-justified *human handoff* path to demonstrate (required by the brief), without depending on the broken `origin_interaction_id` link.
5. **Guardrail-friendly** - these flows read permitted account/transaction data; they do **not** require inventing eligibility rules or moving money (which the brief forbids).

**Explicitly rejected:** *transaction-dispute intake* as the primary flow - the complaints<->interaction join is empty (0%) and complaint categories are flat, so the data cannot support a rigorous dispute pipeline. (A dispute *intake-and-escalate* path can still appear as the human-handoff demo case.)

**Demo cases to build (required by the brief):**
- *Normal resolution:* "What's my checking balance / last 5 transactions / this card's limit?" grounded in account data, with authn + per-customer authorization.
- *Ambiguous / unsupported:* vague or out-of-scope request -> clarify, then abstain/redirect.
- *Human intervention:* a complaint or disputed charge -> escalate with full handoff context (request, verified facts, actions taken, open questions).
- *Multilingual:* the same flows in ES and PT (PT via labeled fixtures).

## 7. Next steps

1. Define data contracts (Pandera) + quality checks for the tables this workflow reads.
2. Stand up the tool/policy layer: identity/session, per-customer authorization, read-only account/transaction tools with documented contracts.
3. Build the LangGraph decision flow (answer / confirm / abstain / escalate).
4. Build the evaluation harness (held-out cases incl. adversarial: prompt injection, unauthorized access, expired session, tool failure, ES/PT ambiguity) and a baseline to compare against.

---

## 8. All 13 tables profiled vs the data dictionary (2026-10-01)

Reproduce: `python analysis/profile_tables.py --sample <sample_dir> --out documentation/reports`.
Full output: [`documentation/reports/data_profile.md`](../documentation/reports/data_profile.md) / `.json`.

**Bucket coverage (full, not sampled):** every fact table has 1,097 daily partitions from 2023-06-17 to 2026-06-17, except `campaign_sends` (1,083, starts 2023-07-01). Raw size ≈ 5.35 GB of CSV, 70% of it `digital_events` (3.76 GB).

**Profiled:** the six dimension/reference tables in full; `transactions`, `digital_events`, `campaign_sends` for 2026-06 (17 days); `call_transcripts`, `satisfaction_surveys`, `call_center_interactions`, `complaints` for 2026-H1.

| # | Finding | Evidence | Impact on CORA |
|---|---|---|---|
| F1 | **Schemas match the dictionary exactly** in all 13 tables (no missing/extra columns). | data_profile.md | Contracts can be generated from the dictionary. |
| F2 | **Categorical values are in Spanish, not the English listed in the dictionary**: `Cuenta Ahorro`, `Tarjeta Crédito`, `México`, `Pasaporte`, `Transaccional`, `Queja`... | enums in data_profile | Contracts must encode the *observed* vocabulary + a mapping to canonical English codes. |
| F3 | **No MXN anywhere.** All 200,398 products of Mexican customers are in `USD`; Argentina = ARS (+ some USD), Colombia = COP (+ some USD). Transactions likewise have no MXN. | country × currency crosstab | Likely data defect (MXN labeled as USD). CORA must state the currency exactly as recorded and must not silently relabel; flagged as a known data limitation and a policy decision (ADR-016). |
| F4 | **Duplicates observed: 0%** (PK and full-row, within and across the sampled days) vs "~2%" documented. | data_profile | Keep dedup (defensive, contract-driven), report observed rate honestly; the fixture (REQ-26) injects duplicates to prove the logic. |
| F5 | **Orphan FKs observed: 0%** for customer/product/agent in all sampled facts vs "small %" documented. | data_profile | Same: check stays, observed rate reported. |
| F6 | **Null rates vary widely, many are structural**, not the uniform "~5%": `credit_limit`/`days_past_due` ~69% (only credit products), `landline_phone` 50%, `customer_detected_accent` ~30%, `complaints.origin_interaction_id` **100%**. | top_null_columns | Contracts declare nullability per column, conditional on product type where structural. |
| F7 | **Call transcripts are fully templated:** 26,552 transcripts contain only **42 distinct `customer_text`** strings; `detected_intents` = `consulta_general` in 95%; `detected_language` = `es` in 100%. | transcripts check | Transcripts are **not valid training/test labels** for the intent classifier (would leak and look perfect). Gold set must be team-written (ADR-008 updated). Confirms no Portuguese. |
| F8 | `daily_exchange_rates` has **13,164 rows** (1,097 days × 12 currency pairs), not 3,000 as documented. | row count | FX conversion has full daily coverage for all pairs. |
| F9 | Fraud is rare: `is_fraud` = 0.10% of June transactions (69 / 70,691). | enums | Fraud escalation (E3) needs synthetic scenarios to be evaluated with a meaningful sample. |
| F10 | `satisfaction_surveys.nps_category` has **no Promoters** (only Detractor / Passive among non-null). | enums | Survey data is not a reliable outcome label; used only descriptively. |
| F11 | Transactions are mostly card/cash (POS 35%, ATM 30%); merchant fields ~77% null (non-purchase types). | enums, nulls | Merchant-based search must tolerate missing merchants. |

These findings were not visible from the dictionary alone; each one changes a contract, a decision or a stated limitation.

## 9. Full-history landing validated against the dictionary (2026-10-01)

All 13 tables were landed in full (2023-06-17 to 2026-06-17) as raw Parquet by
`scripts/data/fetch_to_parquet.py` and recounted by `scripts/data/validate_landing.py`.
Full output: [`documentation/reports/landing_validation.md`](../documentation/reports/landing_validation.md) / `.json`.

- **Landing integrity:** every table's Parquet row count equals its ingestion manifest, with 0 failed files.
  Each file was verified (rows and columns equal to its CSV) before the CSV was deleted.
- **Dimensions match the dictionary exactly** (customers, products, branches, service_agents, marketing_campaigns).
- **Fact tables are 11-16% below the stated counts** (e.g. transactions 4.43M vs 5M, complaints 67K vs 80K).
  Every daily partition is present, so this is not a download gap: the dictionary figures are rounded
  targets, not exact counts. We use the observed counts in all baselines.
- **`digital_events` is 56% above** the stated 10M (15.6M rows), and **`daily_exchange_rates` has 13,164 rows
  vs 3,000** (every currency pair per day; already noted in section 8). Total landed: 23.5M rows vs "~19M".
- **No PK duplicates and no null PKs in any table**, over the full history. The dictionary's "~2% duplicates"
  is not observed at the key level; record-level dedup checks stay in the curated layer (task 1.2) anyway.
- **`campaign_sends` has no partitions for 2023-06-17..2023-06-30** (14 days). This is expected, not a gap:
  the earliest campaign in `marketing_campaigns` starts on 2023-07-01.

## 10. Full-history cross-table EDA (behavior, correlation, capability mapping) (2026-10-01)

Reproduce: `uv run python analysis/eda_full_history.py --outdir analysis/figures`.
Raw metrics: [`analysis/figures/metrics.json`](figures/metrics.json). All numbers below are **full-history**
(2023-06-17 to 2026-06-17, every landed row) unless explicitly labelled *sampled*. The driver runs entirely
in DuckDB (`memory_limit='2GB'`) with aggregated/sampled SQL and converts only small aggregates to pandas,
so it completes on the ~4 GB-RAM host without `MemoryError`. The default `--sample-frac 0.02` is used only for
one illustrative sampled metric (see 10.5); every structural figure/number is exact full-history.

### 10.1 Cross-table joinability & correlation

Joins across the operational tables are clean enough to ground the chosen servicing workflow
(all values from `metrics.json["joinability"]`):

| Join | Metric | Value |
|---|---|---|
| customers↔products | orphan `products.customer_id` | 0.0% |
| customers↔products | products per customer p50/p90/p99 | 3 / 5 / 7 |
| customers↔transactions | orphan `customer_id` / `product_id` | 0.0% / 0.0% |
| customers↔transactions | customers with ≥1 transaction | 89.68% |
| cci↔transcripts | interactions with a transcript | 24.96% |
| transcripts↔cci | transcript `interaction_id` matches an interaction | 100.0% |
| transcripts↔customers | transcript `customer_id` matches a customer | 100.0% |
| transactions↔FX | convertible to USD | 100.0% (12 pairs, see F8) |
| products | `product_status` = Active / Closed / Blocked / Suspended | 84.99 / 8.01 / 4.98 / 2.02% |

FCR by reason category over the full history matches the sample (`fcr_by_reason_category_pct`): Transaccional
91.5%, Producto 89.6% are the safe-automation targets; Queja 43.6% is the escalation path — confirming the
section-3 workflow decision at full scale.

**`origin_interaction_id` full-history verdict:** `cci_complaints.origin_interaction_id_nonnull_pct` = **0.0%**
and `origin_interaction_id_match_pct` = **0.0%**. The sample-era finding (section 5 / F6) is **confirmed, not
refuted**: there is no usable complaints→interaction link in the full history either, so a dispute-intake flow
keyed on that link is not possible. Disputes/complaints remain a human-handoff path, as decided in section 6.

### 10.2 Behavioral & temporal patterns

- **Interaction demand** is flat-seasonal at ~18k–20k/month across three years
  ([`10_interaction_demand_over_time.png`](figures/10_interaction_demand_over_time.png)); the first and last
  months are partial (range boundaries). Weekday mix (`interaction_weekday_pct`) is mid-week heavy (Tue–Fri
  ~16.5% each, Sunday lowest at ~8.4%).
- **Transaction volume & value** are stable at ~120k txns and ~$87M USD/month with a ~$466 median
  ([`11_transaction_volume_value_over_time.png`](figures/11_transaction_volume_value_over_time.png)).
- **Digital events** (15.6M rows, aggregates only) are dominated by PageView 38.2%, Click 23.0%, Login/Logout
  ~15.6% each ([`12_digital_events_type_mix.png`](figures/12_digital_events_type_mix.png)); Purchase is 1.5%.
- **Channel mix** is stable over time with Phone ~85% and digital text channels (App/Web Chat/WhatsApp/Email/Web)
  ~15% combined ([`13_channel_mix_over_time.png`](figures/13_channel_mix_over_time.png)), consistent with
  section 4 — a text-first assistant maps onto the digital channels and transcribed calls.

### 10.3 Capability → table → column map

From `metrics.json["capability_map"]`. Read capabilities are fully grounded; freeze/unfreeze are **PARTIAL** by
design because `products` has **no native freeze field** (`has_native_freeze_field=false`; the 17 `product_columns`
are listed in `metrics.json`).

| Capability | Status | Tables | Key columns | Evidence |
|---|---|---|---|---|
| I1 Account balance | SUPPORTED | products | current_balance, currency | 0% orphan products |
| I2 Recent transactions | SUPPORTED | transactions | transaction_date, amount(_usd), currency | 0% orphan FK |
| I3 Transaction status | SUPPORTED | transactions | transaction_status, response_code | per-row enum |
| I4 Card status & limit | SUPPORTED | products | product_status, credit_limit, days_past_due | limits structural-null off credit |
| I5 FX conversion | SUPPORTED | daily_exchange_rates, transactions | source/target_currency, exchange_rate | 100% convertible, 12 pairs |
| I6 Product list | SUPPORTED | products | product_type, product_number, product_status | 3/5/7 products per customer |
| A1 Card freeze | **PARTIAL** | products | product_status | no native freeze field; via status + sandbox overlay (design §6) |
| A2 Card unfreeze | **PARTIAL** | products | product_status | no native freeze field; via status + sandbox overlay (design §6) |

No capability is UNSUPPORTED by data; A1/A2 are the only PARTIAL rows and the gap is a missing *field*, not
missing join integrity.

### 10.4 Escalation evidence

From `metrics.json["escalation"]`:

- Complaints by case type: Complaint 60.3%, Claim 24.7%, Request 10.1%, Suggestion 4.9%.
- Status mix is mostly unresolved (In Process 40.0% + Open 30.0%); `sla_breached` **20.11%** full-history
  ([`14_complaints_by_status_sla.png`](figures/14_complaints_by_status_sla.png)), matching the ~20.5% sample.
- Low-FCR categories (<70% FCR) are Queja, Retención, Comercial, Técnico — the data-justified human-handoff set.
- **Fraud is rare:** `fraud_rate_pct` = **0.0975%** full-history; `fraud_score` (0–100 scale) p50/p90/p99 =
  15.0 / 27.0 / 29.7, i.e. a long thin tail ([`15_fraud_score_distribution.png`](figures/15_fraud_score_distribution.png)).
  Fraud scenarios need synthetic cases to be evaluated meaningfully (confirms F9 at full scale).

### 10.5 Updated full-history data-quality numbers

From `metrics.json["data_quality"]`; each number carries a `measurement_scope` label.

| Metric | Scope | Value |
|---|---|---|
| PK duplicate rate (customers/products/transactions/complaints/cci) | full-history | 0.0% all |
| Orphan FK (products.customer_id, transactions.customer_id/product_id) | full-history | 0.0% all |
| Structural null `credit_limit` by product type | full-history | 100% off-credit; ~5% on credit products |
| Distinct `customer_text` in transcripts | full-history | **42** (templated — confirms F7, not training labels) |
| No MXN anywhere (products + transactions) | full-history | confirmed true (confirms F3) |
| `daily_exchange_rates` rows | full-history | 13,164 (confirms F8) |
| `campaign_sends` missing early days / first day | full-history | 14 days / 2023-07-01 (expected, confirms §9) |
| Fraud rate | sampled frac=0.02 | 0.0872% (vs 0.0975% full-history — sampling sanity check) |

Every quantitative claim above is backed by a `metrics.json` key or a figure file; the full-history numbers
confirm (do not overturn) the sample-era findings F1–F11 while adding the cross-table correlation, temporal
and capability-mapping evidence the servicing workflow and its escalation path depend on.
