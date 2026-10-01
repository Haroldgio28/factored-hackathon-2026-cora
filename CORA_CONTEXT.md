# CORA - Project Context

> **CORA** = *Customer-Oriented Resolution Agent*.
> Self-contained context document for the **Factored AI & Data Hackathon 2026**.
> Everything needed lives here: **no need to reopen the source PDFs**.
> Language policy: code and docs in English; customer-facing interactions in Spanish + Portuguese.
> Last updated: 2026-10-01.

---

## 1. The challenge

**Factored AI & Data Hackathon 2026 - Problem Statement.**

Build an **AI-first banking customer-service system** that works end to end in a realistic banking environment. The solution must:

- Understand complex customer interactions.
- Use data and tools **securely**.
- Complete appropriate service workflows.
- **Involve human agents when needed** (human-in-the-loop).

Choose **one focused service workflow** and demonstrate an end-to-end solution. With the supplied data: explain **why** the problem matters, establish a **baseline**, and **measure** whether the approach improves service quality and operational efficiency.

### Scope

- Deliver a **working prototype** with evidence of production readiness and an honest account of the work remaining before deployment.
- The **ten-day submission is NOT expected to operate a live banking service**.
- Pick a coherent workflow, e.g.:
  - Account or payment inquiries.
  - Card-service support.
  - Transaction-dispute intake.
  - Credit-product information and eligibility support.
- **These are examples, not separate tracks.** Score is driven by **depth, demonstrated behavior and engineering judgment**; implementing more workflows earns **no** automatic bonus.
- The demo must include: a **normal resolution path**, an **ambiguous or unsupported request**, and a **case requiring human intervention**.
- Demonstrate interactions in **Spanish and Portuguese** and report limitations in the supplied data or language coverage.

### What the solution must demonstrate (6 evaluation axes)

1. **A problem supported by data.** Analyze contact reasons, demand patterns, data quality and operational constraints. Use that evidence to prioritize the workflow and define intended customer and business outcomes.
2. **A functioning AI system.** Maintain relevant conversational context, clarify ambiguity, and **ground factual responses** in permitted account/transaction/policy information. Use tools when they serve the workflow and **report only actions whose outcomes the system has verified**.
3. **Controlled automation.** Define which requests the system can answer, which actions require **confirmation**, and when it must **abstain or transfer to a human**. **Enforce permissions and policy OUTSIDE model-generated prose.** Give the human agent: the request, verified facts, actions taken, supporting evidence and unresolved questions.
4. **Sound data and ML practice.** Repeatable data preparation with **contracts, quality checks, lineage** and an update/freshness policy. Evaluate **at least one learned component** against an appropriate **baseline**. Use valid labels or relevance judgments, prevent **leakage**, and justify representations, metrics, thresholds and splits.
5. **Measured quality and failure handling.** Evaluate on **held-out** cases. Include: incorrect or missing data, **expired sessions**, **unauthorized access attempts**, **prompt injection**, **tool failures** and **multilingual ambiguity**. Report successful outcomes, unsafe outcomes, handoff behavior, latency and cost, with sample sizes and limitations.
6. **A credible route to operation.** Demonstrate **tracing**, **bounded retries**, safe **fallback** and reproducible setup. Explain capacity limits, monitoring, access controls, data retention and remaining deployment work. Explanations are based on **sources, policy rules and execution records**; *hidden model chain-of-thought is NOT an audit artifact*.

### Architecture freedom

- You may use conventional ML, pretrained LLMs, retrieval, deterministic workflows, agents, or a justified combination.
- **NOT mandatory:** training a new model, using multiple agents, hitting a tool-count target, implementing streaming, forecasting demand, building a dashboard.
- Every team is assessed on **data-engineering and AI/ML rigor**. For a pretrained/retrieval solution, demonstrate those competencies via component selection, relevance/intent labels, representations, leakage prevention, held-out evaluation and error analysis.
- **Batch, incremental or streaming** processing as the inputs and the workflow's latency/freshness needs require. **Incremental file delivery does NOT by itself require streaming.** If only static data is supplied, demonstrate update correctness with a clearly labeled test fixture.

### Data and execution boundaries

- Use **only organizer-approved data** and permitted external resources. Identify which inputs are real, de-identified, synthetic or team-generated, and follow the data-use terms.
- **Do not include** private customer records, credentials or restricted data in public submissions or external model requests.
- Sandbox services and mock banking tools are acceptable when their contracts and limitations are documented.
- **Authentication:** demonstrate with a **trusted test session or identity service**; a national ID or customer number **alone does NOT prove identity**. Enforce access to each customer's records and action permissions in the service/tool layer.
- **Credit workflows:** separate conversation handling, predictive risk estimates, and eligibility policy. Use approved rules or a **clearly labeled synthetic policy service** to produce a simulated eligibility outcome. **The conversational model must NOT invent eligibility rules or independently approve credit.** Show explanations, uncertainty and review paths for missing data or borderline cases.
- **No live lending decisions or movement of money** is required or authorized.

### Evaluation evidence (metrics to report)

Compare **baseline vs. proposed system** on the **same held-out workload**. Report the number and mix of cases, label quality, model and prompt versions, and repeated-run variability. **Include failures in the results.** If using an LLM-as-judge, document its rubric and validate a sample against human or deterministic judgments.

Distinguish these outcomes:

- **Safe automated resolution:** an eligible case reaches the correct, policy-compliant outcome **without human intervention**. Report this rate over all in-scope test cases, plus the share of cases where automation was **attempted**.
- **Containment:** a case ends without transfer. *Containment alone does NOT demonstrate the problem was solved.*
- **Escalation quality:** cases that require escalation are transferred correctly and with **useful handoff context**. Report **missed** and **unnecessary** transfers where reference labels permit.
- **Unsafe outcomes:** unauthorized disclosures/actions or materially incorrect outcomes, reported with counts and denominators. *Zero observed failures in a small test set does NOT establish zero risk.*
- **Operating efficiency:** end-to-end **p50/p95** latency and **cost per attempted case** and **per successful automated resolution**. State workload, sample size and cost assumptions; use "not defined" when there are no successful resolutions.

Compare relevant service outcomes **by language and authorized customer segment**, state small-sample limitations and investigate disparities. **Label separately** offline measurements, simulations and projected business savings. *Do not describe an offline comparison as a measured production improvement.*

---

## 2. The dataset - LATAM Bank (Version 1.0.0)

Synthetic regional banking system operating across three countries: **Mexico, Colombia, Argentina**.

| Attribute | Value |
|---|---|
| Total records | ~19,000,000 rows |
| Tables | 13 |
| Countries | Mexico, Colombia, Argentina |
| Date range | 2023-06-17 to 2026-06-17 (3 years) |
| Currencies | MXN, COP, ARS, USD |
| Languages | Spanish (Mexican, Colombian, Argentine accents) |
| Generation | Synthetic, for the Factored Datathon 2026 |

> **Language note:** the dataset text is **Spanish only**. The challenge additionally requires **demonstrating interactions in Spanish and Portuguese** and reporting language-coverage limitations (supplied data is ES; PT is a challenge requirement to be covered with our own fixtures).

### Data-quality challenges (intentional)

| Challenge | Rate | Description |
|---|---|---|
| Duplicate records | ~2% | Realistic duplicates across tables |
| Null values | ~5% | Missing data in non-mandatory fields |
| Late arrivals | Yes | Partitioned data may arrive late |
| Schema evolution | Yes | Table schemas may evolve over time |

Also: **referential integrity** maintained, with a small % of orphans on purpose for testing. Spanish text with regional variants (MX/CO/AR accents). Large fact tables **partitioned by date** (year/month/day).

### Data access (S3)

- Hosted on **Amazon S3**, **read-only** for participants.
- **Bucket:** `factored-datathon-2026-s3-157725502942-us-east-2-an`
- **Region:** `us-east-2`
- Data prefix: `data/` (e.g. `s3://<bucket>/data/customers.csv`).
- Layout: dimension tables are flat CSVs under `data/`; fact tables are partitioned Hive-style `data/<table>/year=YYYY/month=MM/day=DD/<table>_YYYYMMDD.csv`.
- Typical commands: `aws s3 ls s3://<bucket>/data/`, `aws s3 cp s3://<bucket>/data/<file>.csv ./`, `aws s3 sync s3://<bucket>/data/ ./data/`.

> **CREDENTIALS:** the original PDF contained a plaintext Access Key ID + Secret Access Key. **They are NOT stored in this repo or any note.** Treat them as potentially compromised (they were in a shared file). Load them via an AWS profile or env var (`aws configure --profile cora-datathon`), never committed. If the organizer rotates keys, use the new one. The source PDFs under `Docs/` are gitignored for this reason.

### The 13 tables

**Dimension Tables (5)**

| Table | Rows | Source | Partition | Description |
|---|---|---|---|---|
| `customers` | 150,000 | Core Banking | monthly_snapshot | Customer dimension |
| `products` | 400,000 | Core Banking | monthly_snapshot | Active financial products |
| `branches` | 350 | Internal | full_snapshot | Physical branches |
| `service_agents` | 1,200 | Internal | monthly_snapshot | Customer-service agents |
| `marketing_campaigns` | 200 | Internal | full_snapshot | Marketing campaigns |

**Fact Tables (7)**

| Table | Rows | Source | Partition | Description |
|---|---|---|---|---|
| `transactions` | 5,000,000 | Core Banking | daily | Daily financial transactions |
| `call_center_interactions` | 800,000 | Contact Center | daily | Call-center interactions |
| `call_transcripts` | 200,000 | Contact Center | daily | Call transcripts |
| `satisfaction_surveys` | 250,000 | Contact Center | daily | Post-interaction surveys (CSAT, NPS, CES) |
| `digital_events` | 10,000,000 | Digital Banking | daily | Digital channel events (app, web) |
| `complaints` | 80,000 | PQR | daily | Complaints and claims (PQR) |
| `campaign_sends` | 2,000,000 | Internal | daily | Individual campaign sends |

**Reference Tables (1)**

| Table | Rows | Source | Partition | Description |
|---|---|---|---|---|
| `daily_exchange_rates` | 3,000 | Reference | daily | Daily FX rates for currency conversion |

### Detailed schema per table

#### `customers` [DIMENSION] - 150,000
`customer_id` VARCHAR(20) PK NOT NULL · `document_number` VARCHAR(20) NOT NULL UNIQUE · `document_type` VARCHAR(10) NOT NULL (DNI, CURP, CC, CE, Passport) · `first_name` · `last_name` · `date_of_birth` DATE · `gender` (M,F,O) · `email` · `mobile_phone` · `landline_phone` · `address` · `city` NOT NULL · `state` NOT NULL · `country` NOT NULL (Mexico, Colombia, Argentina) · `postal_code` · `detected_accent` (mexican, colombian, argentine, neutral) · `segment` NOT NULL (Premium, Plus, Basic, Student) · `credit_score` INT (300-850) · `estimated_monthly_income` DECIMAL(12,2) · `occupation` · `marital_status` · `education_level` · `registration_date` TIMESTAMP NOT NULL · `registration_branch_id` VARCHAR(20) FK NOT NULL · `customer_status` NOT NULL (Active, Inactive, Suspended, Closed) · `last_updated` TIMESTAMP NOT NULL · `accepts_marketing` BOOLEAN NOT NULL.

#### `products` [DIMENSION] - 400,000
`product_id` PK · `customer_id` FK NOT NULL · `product_type` NOT NULL (Checking Account, Savings Account, Credit Card, Debit Card, Personal Loan, Mortgage, Investment, Insurance) · `product_number` NOT NULL UNIQUE · `currency` (MXN,COP,ARS,USD) · `current_balance` DECIMAL(15,2) · `credit_limit` · `interest_rate` DECIMAL(5,2) · `opening_date` DATE · `expiration_date` · `opening_branch_id` FK · `product_status` (Active, Blocked, Closed, Suspended) · `opening_channel` (Branch, Web, App, Call Center) · `has_linked_app` BOOLEAN · `days_past_due` INT · `last_transaction_date` · `last_updated`.

#### `branches` [DIMENSION] - 350
`branch_id` PK · `branch_code` UNIQUE · `branch_name` · `branch_type` (Main, Express, Premium, Corporate) · `address` · `city` · `state` · `country` · `postal_code` · `geographic_zone` (Urban, Suburban, Rural) · `phone` · `email` · `opening_time` TIME · `closing_time` TIME · `has_atms` BOOLEAN · `atm_count` INT · `has_teller_windows` BOOLEAN · `teller_window_count` INT · `latitude` · `longitude` · `branch_opening_date` DATE · `branch_status` (Active, Temporarily Closed, Closed).

#### `service_agents` [DIMENSION] - 1,200
`agent_id` PK · `employee_code` UNIQUE · `first_name` · `last_name` · `email` · `phone` · `native_accent` (mexican, colombian, argentine) · `country_of_origin` · `assigned_branch_id` FK · `agent_type` (Phone, In-Person, Digital, Hybrid) · `experience_level` (Junior, Mid-Senior, Senior, Specialist) · `languages` · `specialty` · `hire_date` · `avg_csat` DECIMAL(3,2) (1-5) · `total_monthly_interactions` INT · `agent_status` (Active, Vacation, Leave, Inactive) · `work_shift` (Morning, Afternoon, Night, Rotating).

#### `marketing_campaigns` [DIMENSION] - 200
`campaign_id` PK · `campaign_name` · `description` TEXT · `campaign_type` (Email, SMS, Push, WhatsApp, Voice, Mix) · `campaign_objective` (Acquisition, Retention, Cross-sell, Up-sell, Reactivation) · `promoted_product` · `target_segment` · `target_country` · `start_date` · `end_date` · `budget` DECIMAL(12,2) · `campaign_status` (Planned, Active, Paused, Completed) · `expected_conversion_rate` DECIMAL(5,2).

#### `transactions` [FACT] - 5,000,000 (partition `process_date`, daily)
`transaction_id` PK · `transaction_date` TIMESTAMP · `process_date` DATE (partition key) · `product_id` FK · `customer_id` FK · `transaction_type` (Deposit, Withdrawal, Transfer, Payment, Purchase, Adjustment) · `transaction_category` (Food, Transport, Services, Entertainment, Health, Other) · `amount` DECIMAL(15,2) · `currency` · `amount_usd` · `channel` (ATM, Branch, Web, App, POS, Transfer) · `branch_id` FK · `merchant_name` · `merchant_category` (MCC) · `transaction_country` · `transaction_city` · `transaction_status` (Approved, Declined, Pending, Reversed) · `response_code` · `is_fraud` BOOLEAN · `fraud_score` DECIMAL(5,2) (0-100) · `latitude` · `longitude`.

#### `call_center_interactions` [FACT] - 800,000 (partition `process_date`, daily)
`interaction_id` PK · `interaction_date` TIMESTAMP · `process_date` DATE · `customer_id` FK · `agent_id` FK · `interaction_type` (Inbound Call, Outbound Call, Chat, Email, Video) · `channel` (Phone, Web Chat, WhatsApp, Email, App) · `contact_reason` · `reason_category` (Transactional, Product, Technical, Commercial, Complaint) · `duration_seconds` · `wait_time_seconds` · `was_resolved` BOOLEAN (FCR) · `requires_followup` BOOLEAN · `detected_sentiment` (Positive, Neutral, Negative, Very Negative) · `sentiment_score` DECIMAL(3,2) (-1 to 1) · `customer_detected_accent` · `agent_used_accent` · `was_escalated` BOOLEAN · `mentioned_products` (comma-separated IDs) · `has_transcript` BOOLEAN · `has_recording` BOOLEAN.

#### `call_transcripts` [FACT] - 200,000 (partition `process_date`, daily)
`transcript_id` PK · `interaction_id` FK · `process_date` DATE · `customer_id` FK · `agent_id` FK · `full_text` TEXT (Spanish) · `customer_text` TEXT · `agent_text` TEXT · `detected_language` · `detected_accent` · `accent_confidence` DECIMAL(3,2) (0-1) · `detected_keywords` · `mentioned_entities` TEXT (JSON) · `detected_intents` · `main_topics` · `transcription_model` (Whisper, Google STT, etc.) · `audio_quality` (High, Medium, Low) · `duration_seconds`.

#### `satisfaction_surveys` [FACT] - 250,000 (partition `process_date`, daily)
`survey_id` PK · `survey_date` TIMESTAMP · `process_date` DATE · `interaction_id` FK · `customer_id` FK · `agent_id` FK · `survey_type` (CSAT, NPS, CES) · `send_channel` (Email, SMS, IVR, App, Web) · `main_score` INT (1-5 CSAT, 0-10 NPS) · `nps_category` (Promoter, Passive, Detractor) · `question_1_text`/`question_1_response` (1-5) · `question_2_*` · `question_3_*` · `open_comments` TEXT (Spanish) · `comment_sentiment` · `response_time_hours` DECIMAL(8,2) · `campaign_response_rate` DECIMAL(5,2).

#### `digital_events` [FACT] - 10,000,000 (partition `process_date`, daily)
`event_id` PK · `event_date` TIMESTAMP · `process_date` DATE · `customer_id` FK · `session_id` · `event_type` (PageView, Click, FormSubmit, Login, Logout, Error, Purchase) · `event_category` (Navigation, Transaction, Authentication, Product) · `channel` (Android App, iOS App, Desktop Web, Mobile Web) · `platform` (Android, iOS, Windows, MacOS, Linux) · `browser` · `app_version` · `page_url` · `page_title` · `action` · `element_id` · `product_id` FK · `event_value` DECIMAL(15,2) · `duration_seconds` · `ip_address` · `ip_country` · `ip_city` · `is_mobile` BOOLEAN · `referrer` · `utm_source` · `utm_medium` · `utm_campaign`.

#### `complaints` [FACT] - 80,000 (PQR, partition `process_date`, daily)
`complaint_id` PK · `creation_date` TIMESTAMP · `process_date` DATE · `customer_id` FK · `case_type` (Complaint, Claim, Request, Suggestion) · `category` · `subcategory` · `reception_channel` (Call Center, Email, Web, App, Branch, Regulator) · `affected_product_id` FK · `related_branch_id` FK · `origin_interaction_id` FK · `description` TEXT (Spanish) · `claimed_amount` · `currency` · `priority` (Low, Medium, High, Critical) · `status` (Open, In Process, Escalated, Resolved, Closed, Rejected) · `assigned_agent_id` FK · `assignment_date` · `first_response_date` · `resolution_date` · `closing_date` · `sla_breached` BOOLEAN · `resolution_days` INT · `resolution` TEXT · `compensation_granted` · `resolution_satisfaction` (1-5) · `is_repeat_complainer` BOOLEAN (previous complaints in 90 days).

#### `campaign_sends` [FACT] - 2,000,000 (partition `process_date`, daily)
`send_id` PK · `send_date` TIMESTAMP · `process_date` DATE · `campaign_id` FK · `customer_id` FK · `send_channel` (Email, SMS, Push, WhatsApp, Voice) · `template_used` · `subject` · `send_status` (Sent, Failed, Bounced, Blocked) · `was_delivered` BOOLEAN · `was_opened` BOOLEAN · `open_date` · `was_clicked` BOOLEAN · `click_date` · `click_count` · `had_conversion` BOOLEAN · `conversion_date` · `conversion_value` · `open_device` · `open_country` · `failure_reason` · `send_cost` DECIMAL(10,4).

#### `daily_exchange_rates` [REFERENCE] - 3,000 (partition `date`, daily)
`date` DATE PK · `source_currency` VARCHAR(3) PK · `target_currency` VARCHAR(3) PK · `exchange_rate` DECIMAL(12,6) · `buy_rate` · `sell_rate` · `source`.

### Relationships (Foreign Keys)

- **customers** <- products, transactions, call_center_interactions, call_transcripts, satisfaction_surveys, digital_events, complaints, campaign_sends (all via `customer_id`).
- **branches** <- customers.`registration_branch_id`, products.`opening_branch_id`, service_agents.`assigned_branch_id`, transactions.`branch_id`, complaints.`related_branch_id`.
- **service_agents** <- call_center_interactions.`agent_id`, call_transcripts.`agent_id`, satisfaction_surveys.`agent_id`, complaints.`assigned_agent_id`.
- **products** <- transactions.`product_id`, digital_events.`product_id`, complaints.`affected_product_id`.
- **marketing_campaigns** <- campaign_sends.`campaign_id`.
- **call_center_interactions** <- call_transcripts.`interaction_id`, satisfaction_surveys.`interaction_id`, complaints.`origin_interaction_id`.

> **Measured caveat (see `analysis/EDA_FINDINGS.md`):** in the Jan-Jun 2026 sample, `complaints.origin_interaction_id` is **0% populated** - the complaint<->interaction join is effectively empty - and `contact_reason` duplicates `reason_category`. Design around this.

### Potential use cases (from the dataset docs)

- **Customer Analytics:** segmentation/clustering, churn, CLV, cross/up-sell.
- **Contact Center Optimization:** FCR improvement, agent performance, sentiment trends, accent-based routing.
- **Fraud Detection:** transaction fraud, spend anomalies, geographic risk.
- **Marketing Analytics:** campaign effectiveness, channel attribution, personalization, A/B testing.
- **NLP / Text Analytics:** transcript topic modeling, intent classification, entity extraction, multilingual accent detection.
- **Product Analytics:** product adoption/usage, digital engagement funnel, feature usage.

---

## 3. CORA - solution definition

**Name:** CORA - *Customer-Oriented Resolution Agent*.

**Vision:** an AI-first banking customer-service assistant for LATAM Bank, bilingual (ES/PT), with **controlled automation** and **human-in-the-loop**, that safely resolves a focused service workflow and escalates to a human with full context when appropriate.

### Chosen workflow

**Account & transaction information + payment/card inquiries** (reason categories Transactional + Product).

Data-driven rationale (full evidence in `analysis/EDA_FINDINGS.md`):
- Largest demand (~57% of interactions).
- Highest safe-automation ceiling (90-92% first-contact resolution today) and shortest handle time.
- Complaints (43% FCR) give a credible, data-justified human-escalation path - without relying on the empty `origin_interaction_id` link.
- Reads permitted account/transaction data; does not invent eligibility rules or move money (brief forbids it).

**Demo cases (required):** normal resolution (balance / recent transactions / card limit), ambiguous or unsupported request (clarify then abstain), human intervention (complaint / disputed charge -> escalate with handoff context), and the same flows in ES and PT.

### Architecture principles (from the axes)

1. **Policy and permissions live OUTSIDE the LLM** - a deterministic service/tool layer. The model never approves credit, moves money, or invents rules.
2. **Real authentication** - test session / identity service; a customer number is not enough. Per-customer authorization in the tool layer.
3. **Only verified actions are reported** - every action goes through a tool that confirms its outcome before informing the customer.
4. **Handoff with context** - on escalation the human receives: request, verified facts, actions taken, evidence, open questions.
5. **Auditable by design** - tracing + execution records + policy rules as the explanation source (not the chain-of-thought).
6. **Baseline first** - measure against a baseline on the same held-out set; include failures.

### Technical stack

See `ARCHITECTURE.md`. In one line:
`Python 3.12 + uv/ruff/pytest` · data `DuckDB+Parquet (local) -> S3/Athena (AWS)` with `Pandera` · agent `LangGraph + Bedrock` · deterministic tool/policy layer · custom eval harness + `Langfuse/OTel`. Local-first with local<->AWS parity so migrating to a personal AWS account is a config change.

---

## 4. Project status

- **Local repo:** `C:\Users\1872146\Documents\CORA`
- **Structure:** `README.md`, `ARCHITECTURE.md`, `CORA_CONTEXT.md`, `analysis/` (EDA code + figures + findings), `requirements.txt`, `.gitignore` (excludes `.env`, `Docs/`, data samples).
- **Git remote:** _pending (GitHub)._
- **Decided:** name = CORA · language policy (English code/docs, ES+PT interactions) · stack · focused workflow.
- **Pending:** LLM provider (Bedrock vs. direct); GitHub remote + branch/PR flow; Portuguese evaluation fixtures.
- **Vault note:** `Personal/Carrera/CORA/` (index `CORA.md`), updated as we progress. (Authorized exception: KiroCrew may write under `Personal\Carrera` **only** for CORA; the rest of `Personal\` stays out of scope.)

---

## 5. Inherited working rules (in force for CORA)

- **Git:** never commit/push directly to `main`/`master`. Feature branch + PR. Any impactful action (commit, push, PR, merge, delete) needs Harold's explicit approval.
- **Secrets:** never credentials/tokens/keys in the repo or notes. Local `.env` or a password manager only. The repo `.gitignore` excludes `.env` and `Docs/`.
- **Data:** organizer-approved data only; do not expose restricted data in public submissions or external model requests.

---

*End of context. This document is the project's source of truth; update it when scope changes or decisions are made.*
