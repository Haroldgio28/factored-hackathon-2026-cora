# Source Requirements Inventory

Every obligation, constraint and evaluation criterion stated in the three source
documents, extracted one by one and given a stable ID (`SRC-xx`). This file is the
**root of traceability**: every `SRC` must map to at least one spec requirement
(`REQ-xx`) in [`.kiro/specs/cora/requirements.md`](../.kiro/specs/cora/requirements.md), and from
there to tasks and evidence (see [`TRACEABILITY.md`](TRACEABILITY.md)).

> **Provenance:** built from the full-text extraction (pypdf) of the three PDFs, page by page.
> The PDFs themselves stay out of git (`Docs/` is ignored: the data dictionary contains
> plaintext credentials); this inventory plus `CORA_CONTEXT.md` replace them.

**Sources**

- **[PS]** *Factored AI & Data Hackathon 2026 - Problem Statement* (6 pages).
- **[DD]** *LATAM Bank Complete Data Dictionary v1.0.0* (18 pages).
- **[DS]** *LATAM Bank Dataset Summary v1.0.0* (5 pages).

Wording is paraphrased closely; *italic quotes* keep the original emphasis where the
nuance matters for scoring.

---

## A. Mission (PS p.2)

| ID | Requirement |
|---|---|
| SRC-01 | Build a **working AI-first customer service system** for a real-world banking environment. |
| SRC-02 | The solution **understands complex customer interactions**. |
| SRC-03 | It **uses data and tools securely**. |
| SRC-04 | It **completes appropriate service workflows**. |
| SRC-05 | It **involves human agents when needed**. |
| SRC-06 | Choose a **focused** customer-service problem and demonstrate an **end-to-end** solution. |
| SRC-07 | Use the supplied data to **explain why the problem matters**. |
| SRC-08 | **Establish a baseline**. |
| SRC-09 | **Measure** whether the approach improves **service quality** and **operational efficiency**. |
| SRC-10 | Design for **privacy, explainability, fairness, reliability and scalability**. |
| SRC-11 | Make **explicit trade-offs** across **autonomy, accuracy, latency, cost and human oversight**. |
| SRC-12 | **Justify where AI is appropriate, where deterministic logic is preferable**, and **how the system is evaluated for quality and safety**. |

## B. Scope (PS p.2-3)

| ID | Requirement |
|---|---|
| SRC-13 | Deliver a **working prototype** with **evidence of production readiness** and an **honest account of the work required before deployment**. |
| SRC-14 | Ten-day submission; **not expected to operate a live banking service**. |
| SRC-15 | Select **one coherent workflow** (examples: account/payment inquiries, card-service support, transaction-dispute intake, credit-product information & eligibility). |
| SRC-16 | Examples are not tracks: *"depth, demonstrated behavior, and engineering judgment determine the score; implementing more workflows does not earn an automatic bonus."* |
| SRC-17 | Include a **normal resolution path**. |
| SRC-18 | Include an **ambiguous or unsupported request**. |
| SRC-19 | Include a **case requiring human intervention**. |
| SRC-20 | **Demonstrate interactions in Spanish and Portuguese**. |
| SRC-21 | **Report limitations in the supplied data or language coverage**. |

## C. What the solution must demonstrate (PS p.3-4)

### C1. A problem supported by data

| ID | Requirement |
|---|---|
| SRC-22 | Analyze **contact reasons, demand patterns, data quality and operational constraints**. |
| SRC-23 | Use that evidence to **prioritize the workflow** and **define intended customer and business outcomes**. |

### C2. A functioning AI system

| ID | Requirement |
|---|---|
| SRC-24 | **Maintain relevant conversational context**. |
| SRC-25 | **Clarify ambiguity**. |
| SRC-26 | **Ground factual responses** in permitted account, transaction or policy information. |
| SRC-27 | **Use tools when they serve the workflow**. |
| SRC-28 | **Report only actions whose outcomes the system has verified**. |

### C3. Controlled automation

| ID | Requirement |
|---|---|
| SRC-29 | Define **which requests the system can answer**. |
| SRC-30 | Define **which actions require confirmation**. |
| SRC-31 | Define **when it must abstain or transfer to a human**. |
| SRC-32 | **Enforce permissions and policy outside model-generated prose**. |
| SRC-33 | Provide the human agent with **the request, verified facts, actions taken, supporting evidence and unresolved questions**. |

### C4. Sound data and ML practice

| ID | Requirement |
|---|---|
| SRC-34 | **Repeatable data preparation** with **contracts**, **quality checks**, **lineage** and an **update/freshness policy**. |
| SRC-35 | **Evaluate at least one learned component against an appropriate baseline**. |
| SRC-36 | Use **valid labels or relevance judgments**. |
| SRC-37 | **Prevent leakage**. |
| SRC-38 | **Justify representations, metrics, thresholds and evaluation splits**. |

### C5. Measured quality and failure handling

| ID | Requirement |
|---|---|
| SRC-39 | **Evaluate on held-out cases**. |
| SRC-40 | Include **incorrect or missing data**. |
| SRC-41 | Include **expired sessions**. |
| SRC-42 | Include **unauthorized access attempts**. |
| SRC-43 | Include **prompt injection**. |
| SRC-44 | Include **tool failures**. |
| SRC-45 | Include **multilingual ambiguity**. |
| SRC-46 | Report **successful outcomes, unsafe outcomes, handoff behavior, latency and cost**, **with sample sizes and limitations**. |

### C6. A credible route to operation

| ID | Requirement |
|---|---|
| SRC-47 | Demonstrate **tracing**. |
| SRC-48 | Demonstrate **bounded retries**. |
| SRC-49 | Demonstrate **safe fallback**. |
| SRC-50 | Demonstrate **reproducible setup**. |
| SRC-51 | Explain **capacity limits, monitoring, access controls, data retention** and the **remaining deployment work**. |
| SRC-52 | Provide explanations based on **sources, policy rules and execution records**; *"hidden model chain-of-thought is not an audit artifact."* |

## D. Architecture freedom (PS p.4)

| ID | Requirement |
|---|---|
| SRC-53 | Any of conventional ML, pretrained LMs, retrieval, deterministic workflows, agents, or a **justified combination**. Training a model, multi-agent, tool-count targets, streaming, demand forecasting and dashboards are **not mandatory**. |
| SRC-54 | Every team is assessed on **data engineering and AI/ML rigor**; for pretrained/retrieval solutions show it via **component selection, relevance or intent labels, representations, leakage prevention, held-out evaluation and error analysis**. |
| SRC-55 | Use **batch, incremental or streaming** processing according to inputs and the workflow's **latency and freshness** needs; incremental file delivery does not by itself require streaming. |
| SRC-56 | If only static data is supplied, **demonstrate update correctness with a clearly labeled test fixture**. |

## E. Data and execution boundaries (PS p.5)

| ID | Requirement |
|---|---|
| SRC-57 | Use **only organizer-approved data and permitted external resources**. |
| SRC-58 | **Identify which inputs are real, de-identified, synthetic or team-generated**, and follow the published data-use terms. |
| SRC-59 | **Do not include private customer records, credentials or restricted data** in public submissions **or external model requests**. |
| SRC-60 | Sandbox services and mock banking tools are acceptable **when their contracts and limitations are documented**. |
| SRC-61 | **Demonstrate authentication** with a trusted test session or identity service; *"a national ID or customer number alone does not prove identity."* |
| SRC-62 | **Enforce access to each customer's records and action permissions in the service or tool layer**. |
| SRC-63 | Credit workflows: **separate conversation handling, predictive risk estimates and eligibility policy**; use approved rules or a **clearly labeled synthetic policy service**; the model **must not invent eligibility rules or independently approve credit**; show **explanations, uncertainty and review paths** for missing data or borderline cases. |
| SRC-64 | **No live lending decisions or movement of money** is required or authorized. |

## F. Evaluation evidence (PS p.5-6)

| ID | Requirement |
|---|---|
| SRC-65 | **Compare baseline and proposed system on the same held-out workload**. |
| SRC-66 | Report **number and mix of cases, label quality, model and prompt versions, and repeated-run variability**. |
| SRC-67 | **Include failures in the results**. |
| SRC-68 | If a model judges answers: **document its rubric** and **validate a sample against human or deterministic judgments**. |
| SRC-69 | **Safe automated resolution** rate over **all in-scope test cases**, plus the **share of cases on which automation was attempted**. |
| SRC-70 | **Containment** (case ends without transfer) - *"does not demonstrate that the problem was solved."* |
| SRC-71 | **Escalation quality**: correct transfers with useful handoff context; report **missed and unnecessary transfers**. |
| SRC-72 | **Unsafe outcomes** (unauthorized disclosures/actions, materially incorrect outcomes) **with counts and denominators**; *"zero observed failures in a small test set does not establish zero risk."* |
| SRC-73 | **Operating efficiency**: end-to-end **p50/p95 latency** and **cost per attempted case and per successful automated resolution**; state **workload, sample size and cost assumptions**; use **"not defined"** when there are no successful resolutions. |
| SRC-74 | **Compare service outcomes by language and authorized customer segments**, state small-sample limitations and **investigate disparities**. |
| SRC-75 | **Label offline measurements, simulations and projected business savings separately**; never present an offline comparison as a measured production improvement. |

## G. Dataset characteristics that impose engineering obligations (DD, DS)

| ID | Requirement |
|---|---|
| SRC-76 | Handle **~2% duplicate records** across tables. |
| SRC-77 | Handle **~5% null values** in nullable fields. |
| SRC-78 | Handle **late-arriving** partitioned data. |
| SRC-79 | Handle **schema evolution** over time. |
| SRC-80 | Respect **referential integrity** while tolerating a **small % of orphan records** (left there for testing). |
| SRC-81 | Text is **Spanish with regional variants** (Mexican, Colombian, Argentine). |
| SRC-82 | **Multi-currency** (MXN, COP, ARS, USD) with daily exchange rates; amounts must be presented in the right currency. |
| SRC-83 | Large fact tables are **date-partitioned** (year/month/day). |
| SRC-84 | Data is **fully synthetic**; access is **read-only**; credentials **must not be shared** outside participants. |

---

**Total: 84 source requirements.** Coverage is verified in [`TRACEABILITY.md`](TRACEABILITY.md).
