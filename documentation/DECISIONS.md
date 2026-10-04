# Architecture Decision Records (ADR)

Each decision states **context, decision, alternatives considered, and consequences**, so the
"why" is auditable (brief: *"justify where AI is appropriate, where deterministic logic is
preferable"*, *"make explicit trade-offs"*). Status: `Accepted` · `Proposed` (needs owner sign-off) · `Superseded`.

---

## ADR-001 - Workflow: Account & Card Servicing · Accepted
- **Context:** the brief rewards depth on one workflow (SRC-16) justified by data (SRC-22). EDA on Jan-Jun 2026 (105k interactions, 10k complaints) shows Transactional 34.9% + Product 22.2% = ~57% of demand with 91.8% / 89.7% FCR and the shortest handle times (203 s / 263 s); complaints have 43.3% FCR. `complaints.origin_interaction_id` is 0% populated and complaint categories are flat.
- **Decision:** read inquiries (balance, transactions, card status, FX, products) + one confirmed reversible action (card freeze/unfreeze) + dispute/complaint/fraud intake as the escalation path; credit and money movement explicitly out of scope.
- **Alternatives:** *dispute intake as primary* - rejected, the data cannot link complaints to interactions; *credit eligibility* - rejected, requires a separate risk model + labeled synthetic policy service (SRC-63) for little demand evidence; *several workflows* - rejected, no bonus for breadth.
- **Consequences:** maximal automatable volume with low risk; the action gives a real "confirm + verify" demonstration without moving money (SRC-64); disputes still appear as the human-handoff demo.

## ADR-002 - Spec-driven development with Kiro specs + EARS · Accepted
- **Context:** 10 days, 84 source obligations, judges scoring rigor.
- **Decision:** requirements -> design -> tasks in `.kiro/specs/cora/`, EARS acceptance criteria, a traceability matrix from every `SRC` to evidence, project steering in `.kiro/steering/`.
- **Alternatives:** ad-hoc backlog (risk of silently missing obligations); heavy formal methods (too slow).
- **Consequences:** every criterion is testable; scope changes are visible in diffs; agents (Kiro) and humans share the same source of truth.

## ADR-003 - Language policy · Accepted
- **Decision:** code, comments, docs, commit messages in English; customer-facing interaction in Spanish and Portuguese.
- **Why:** English maximizes reviewability by an international jury; es/pt is a product requirement (SRC-20).

## ADR-004 - Orchestration: LangGraph state machine · Accepted
- **Context:** controlled automation needs explicit, testable transitions (SRC-29..32).
- **Decision:** LangGraph graph whose nodes call deterministic components and the LLM only where language is involved; the policy engine owns transitions.
- **Alternatives:** free-form tool-calling agent (non-deterministic control flow, hard to prove safety); hand-rolled state machine (feasible, but LangGraph gives checkpoints, retries and visualization for free); multi-agent (not required, adds failure modes).
- **Consequences:** each path in the demo is a reproducible graph path, visible in traces.
- **Phase 4 deviation (task 4.1):** the section-4 machine is implemented as a thin stdlib dispatcher (nodes = functions, edges = a `decision -> node` table) rather than adopting the `langgraph` runtime. LangGraph is unavailable on this RAM-constrained host (`importlib.util.find_spec('langgraph')` is `None`) and heavy to add; the machine is small and fully deterministic, so none of its runtime value-adds (checkpointer, streaming, concurrent branches, tool-calling loop) is needed - `PolicyEngine.decide(...)` picks every edge, never the model. The node/edge structure mirrors section 4 one-to-one, so swapping in LangGraph later is mechanical. The ADR-004 invariant "each transition is code, not prompt" is satisfied identically. Precedent: the Phase-3 `CORA_NLU_STUB` offline path.

## ADR-005 - LLM provider: Amazon Bedrock behind an `LLMClient` interface · Accepted (2026-10-01, owner confirmed)
- **Context:** target is migration to a personal AWS account; data-minimization rules apply to any external model call (SRC-59).
- **Decision:** Bedrock (Claude Haiku-class for generation/extraction, larger model only for the B-zs baseline and LLM-judge); a local stub client for tests; provider swappable by config.
- **Alternatives:** direct OpenAI/Anthropic API (faster setup, second migration later); local open model (no external calls, weaker es/pt quality on a laptop with limited RAM).
- **Consequences:** one integration from dev to AWS; costs computed from Bedrock list prices. Personal account in `us-east-1`, SSO profile `cora-dev`, cross-region inference profiles; model ids in `.env` (see `documentation/AWS_SETUP.md`).

## ADR-006 - Batch + incremental processing, no streaming · Accepted
- **Context:** data arrives as daily partitioned files with late arrivals; SRC-55 says incremental delivery does not by itself require streaming.
- **Decision:** incremental batch with watermark + 3-day reprocessing window; real-time state only for the sandbox overlay (card status).
- **Consequences:** simpler, cheaper, testable with the labeled fixture (SRC-56); freshness stated to customers as "as of" date.

## ADR-007 - DuckDB + Parquet locally, S3 + Athena on AWS · Accepted
- **Decision:** a `DataSource` interface with local and AWS adapters; Parquet partitioned by `process_date` in both.
- **Alternatives:** Postgres (needs a server, poor fit for partitioned files); Spark (overkill for 19M rows on a laptop).
- **Consequences:** identical SQL semantics locally and in Athena (both Presto/ANSI-like); low memory footprint.

## ADR-008 - Intent classifier: multilingual embeddings + calibrated logistic regression · Accepted
- **Decision:** `paraphrase-multilingual-MiniLM-L12-v2` embeddings, calibrated LR; compared to majority, keyword rules, TF-IDF+LR and zero-shot LLM.
- **Why:** shared es/pt space enables cross-lingual generalization without Portuguese training data; calibration makes `τ` thresholds meaningful; CPU-only; interpretable errors.
- **Consequences:** must prove it beats baselines on a leakage-safe test split, or we report that it does not and keep the better one.
- **Update 2026-10-01 (data profile F7):** call transcripts contain only 42 distinct customer utterances and 95% `consulta_general` intents, so they are **not** usable as training or test labels. The gold set is **team-written** (es variants MX/CO/AR + pt), with the 42 dataset utterances used only as seed examples and kept out of the test split; provenance recorded per row.

## ADR-016 - Currency as recorded (no MXN in the data) · Accepted
- **Context:** profile F3 - Mexican customers' products and transactions are all labeled `USD`; MXN never appears although the dictionary lists it.
- **Decision:** CORA reports amounts in the currency recorded on the product and never relabels; FX conversion uses `daily_exchange_rates` on explicit request. The anomaly is documented as a data limitation and covered by an evaluation case.
- **Alternatives:** relabel Mexican USD to MXN (invents data, unverifiable); exclude Mexico (drops 50% of customers).
- **Consequences:** answers are faithful to the source; judges see the defect was detected and handled transparently.

## ADR-009 - Mock identity: OTP + HMAC-signed JWT with TTL · Accepted
- **Decision:** a mock IdP issues short-lived tokens after an OTP step; tools read `customer_id` only from the verified token.
- **Why:** SRC-61 forbids identity by ID number; tokens enable expiry/tampering tests (SRC-41). Maps to Cognito on AWS.

## ADR-010 - Observability: OpenTelemetry -> JSONL (Langfuse optional) · Accepted
- **Decision:** OTel spans per node/tool exported to JSONL; Langfuse docker optional; X-Ray/CloudWatch on AWS.
- **Why:** zero infra and low RAM locally; traces are the audit artifact (SRC-52) and the input of the evaluation judges.

## ADR-011 - Portuguese coverage via team-generated fixtures · Accepted
- **Context:** the dataset is Spanish-only (measured).
- **Decision:** translate the gold set and scenarios to Portuguese, rewrite 20% natively, label provenance, report PT results separately as a known limitation.
- **Consequences:** honest coverage claim; PT performance is measured, not assumed.

## ADR-012 - Credit out of scope, enforced by rule · Accepted
- **Decision:** policy rule `POL-030` abstains and routes any credit/eligibility request.
- **Why:** SRC-63 forbids the model from inventing eligibility; building a separate risk model + policy service would dilute depth. The guard is itself tested.

## ADR-013 - Actions on a sandbox overlay · Accepted
- **Decision:** freeze/unfreeze writes to an overlay table, never to source data; read-back verification.
- **Why:** source data is read-only (SRC-84); overlay makes actions reversible, auditable and resettable between evaluation runs.

## ADR-014 - Deterministic judges first, LLM judge only for subjective qualities · Accepted
- **Decision:** facts, policy compliance, disclosure and actions judged by code against reference outcomes; LLM judge only for tone/clarity/handoff usefulness, validated on 50 human labels.
- **Why:** SRC-68; deterministic judges have no variance and no self-preference bias.

## ADR-015 - Repository hygiene for source PDFs · Accepted
- **Decision:** the original PDFs stay in `Docs/`, which is gitignored (the data dictionary contains plaintext credentials). Versioned documentation lives in `documentation/` - a distinct name because Windows paths are case-insensitive and `docs/` would collide with the ignored `Docs/`.
- **Consequences:** `CORA_CONTEXT.md` and `documentation/SOURCE_REQUIREMENTS.md` (built from the PDF text extraction) replace the PDFs for anyone reading the repo.

## ADR-017 - Data contracts follow observed data over the dictionary · Accepted (2026-10-01)
- **Context:** task 1.2 built Pandera contracts for all 13 tables. REQ-21 says contracts match the data dictionary, but the dictionary disagrees with the landed data in several places (consistent with profile findings F1-F6: schemas match but categorical values are Spanish, nullability is structural, and the raw landing stored every column as a string). Five concrete conflicts surfaced while validating real samples through `LocalSource`.
- **Decision:** where the dictionary and the observed data conflict, the contract encodes the **observed** reality (the observed-structure rule), and the divergence is recorded. The five divergences from the task 1.2 plan:
  1. `call_center_interactions.detected_sentiment` uses observed Spanish values `{Muy Positivo, Positivo, Neutral, Negativo, Muy Negativo}`, not the English in the plan.
  2. `complaints.currency` includes **MXN** (observed `{USD, MXN, COP, ARS}`; measured 5,487 MXN rows). This refines ADR-016, which scopes "no MXN" to `products` and `transactions`; complaints carry MXN and are not relabeled.
  3. `branches.geographic_zone` uses observed Spanish `{Urbana, Suburbana, Rural}`.
  4. `digital_events.customer_id` is `nullable=True` (observed ~24% null, anonymous events: 60,518 / 251,967 rows in 2026-06), not non-nullable.
  5. `service_agents.employee_code` is not unique (1,187 distinct / 1,200 rows); `agent_id` is the unique PK.
- **Alternatives:** *encode the dictionary verbatim* - rejected, contracts would reject valid landed rows and give false quality signals; *silently diverge without a record* - rejected, the engineering-rigor axis needs the "why" to be auditable.
- **Consequences:** contracts validate real samples after coercion (34 passed / 1 skipped); the contract vocabulary and nullability are faithful to the data; the quality layer (task 1.3) and the labeled fixture (task 1.7) exercise the violation paths. Each divergence was verified against the data, not assumed. ADR-016's "currency as recorded" principle still holds; this ADR only clarifies MXN's scope.

## ADR-018 - Pipeline stage order: dedup before contract validation · Accepted (2026-10-01)
- **Context:** task 1.4's incremental pipeline orchestrates the existing stages (contract validation, quarantine, dedup) per `process_date` partition. The task-1.2 fact contracts declare the PK `unique=True`, so Pandera's `validate()` reports BOTH rows of any duplicate PK as failures. With a validate-first order, a legitimate duplicate re-delivery (a late re-send of the same record, within the reprocessing window W) would be sent to quarantine and never reach the latest-wins dedup — discarding a valid record and contradicting REQ-23.
- **Decision:** run the per-partition stages in the order **dedup → validate → quarantine**. Dedup collapses re-deliveries latest-wins first (reusing the task-1.3 dedup verbatim: tie-break `last_updated` → `process_date` → input-order); the contract then validates a batch that already has unique PKs; genuinely-bad rows are quarantined. The PK uniqueness check still catches a true uniqueness violation that dedup's deterministic tie-break cannot legitimately collapse.
- **Alternatives:** *validate → quarantine → dedup (the original 1.4 plan order)* - rejected, it quarantines legitimate re-deliveries and violates REQ-23 (duplicates must be deduplicated, not discarded); *drop `unique=True` from the fact contracts* - rejected, out of task-1.4 scope and loses the uniqueness check entirely.
- **Consequences:** matches the design §7 pipeline where "Dedup by PK (latest wins)" is its own stage before curation; a duplicate PK is removed by dedup (kept = latest, `duplicates_removed` increments) and the curated partition keeps one row, rather than being quarantined and dropped. Verified by task-1.4 tests (duplicate re-delivery deduped, good rows curated) and the two-run SHA-256 idempotency evidence.
