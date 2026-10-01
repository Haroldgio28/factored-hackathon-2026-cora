# CORA - Architecture & Technical Stack

**CORA** = *Customer-Oriented Resolution Agent*.
AI-first banking customer-service assistant for the Factored AI & Data Hackathon 2026.

> **Language policy:** all code and documentation are in **English**. Only the
> system's **customer-facing interactions** are multilingual (**Spanish + Portuguese**).

## Guiding principle: local-first, AWS-ready

Develop locally against a **sample** of the dataset, but keep **local<->AWS parity**
from day one so migrating to a personal AWS account is a config change, not a rewrite.
Data access and tools sit behind stable interfaces with two implementations
(local / AWS).

## Stack

| Layer | Local (development) | AWS (target migration) |
|---|---|---|
| Language / tooling | Python 3.12, `uv` (env + lockfile), `ruff` (lint+format), `pytest` | same |
| Data query | **DuckDB** over local Parquet/CSV sample | Athena (or DuckDB over S3 Parquet) |
| Data format | Parquet partitioned by `process_date` (mirrors the real layout) | S3 + Glue Catalog |
| Data contracts / quality | **Pandera** schemas; checks for dup ~2%, null ~5%, late arrivals, schema evolution | same |
| Agent orchestration | **LangGraph** - explicit state graph (answer / confirm / abstain / escalate) | same |
| LLM | **Amazon Bedrock** (Claude / Nova) via a thin `LLMClient` interface | Bedrock (same account) |
| Tools / policy layer | Pure-Python tools with documented contracts; auth, per-customer authorization, synthetic eligibility-policy service - **enforced outside the LLM** | same, backed by real services/mocks |
| Retrieval (if needed) | FAISS / Chroma over policy/FAQ docs | OpenSearch / Bedrock Knowledge Base |
| Language detection | `fastText` lid or `langdetect` per turn -> respond in customer language | same |
| Evaluation | Custom harness on `pytest` over held-out cases (incl. adversarial) | same |
| Tracing / observability | **Langfuse** (self-host) or OpenTelemetry; execution records = audit artifact | same |

## Why these choices (mapped to evaluation axes)

- **DuckDB + Parquet** -> axis 4 (repeatable data prep) and a drop-in local stand-in for Athena; cheap, no infra.
- **Pandera contracts** -> axis 4 (contracts, quality checks, update/freshness via test fixtures).
- **LangGraph** -> axis 3 (controlled automation): the decide/confirm/abstain/escalate transitions are graph edges, not model prose; permissions and policy are enforced in the tool layer, not generated text.
- **Bedrock** -> single integration that works locally and in the target AWS account.
- **Custom eval harness + Langfuse/OTel** -> axes 5 & 6: held-out metrics (safe-resolution, containment, escalation-quality, unsafe outcomes, p50/p95 latency, cost), tracing, bounded retries, safe fallback. Explanations come from sources + policy rules + execution records, never from hidden chain-of-thought.

## Security & data boundaries (from the brief)

- Authentication via a trusted test session / identity service; a national ID or customer number alone does not prove identity.
- Per-customer record access and action permissions enforced in the service/tool layer.
- The conversational model never invents eligibility rules, approves credit, or moves money.
- No real credentials in the repo. Datathon S3 keys load from an AWS profile / env var only (`.gitignore` excludes `.env`).

## Repository layout (planned)

```
CORA/
  analysis/            # EDA: scripts, figures, findings (evidence, axis 1)
  src/cora/            # application code (English)
    data/              # DataSource interface: local DuckDB vs AWS/Athena
    tools/             # banking tools + policy/permission layer
    agent/             # LangGraph graph, nodes, prompts
    eval/              # evaluation harness + metrics
  tests/               # pytest + data-update fixtures
  Docs/                # original hackathon PDFs
  CORA_CONTEXT.md      # self-contained project context
  ARCHITECTURE.md      # this file
  README.md
```

## Open decisions

- LLM provider: Bedrock (recommended) vs. a direct provider for faster dev.
- Git remote (GitHub) + branch/PR flow.
- Portuguese coverage: build labeled ES->PT evaluation fixtures (dataset is ES-only).
