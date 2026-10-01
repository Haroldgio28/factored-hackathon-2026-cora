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
