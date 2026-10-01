# CORA - Customer-Oriented Resolution Agent

AI-first banking customer-service assistant built for the **Factored AI & Data Hackathon 2026**, on the synthetic **LATAM Bank** dataset (19M rows, 13 tables, Mexico/Colombia/Argentina).

> **Language policy:** code and documentation are in **English**; the assistant's **customer-facing interactions** support **Spanish and Portuguese**.

## What CORA does

A focused, safe, measurable customer-service workflow with **controlled automation** and **human-in-the-loop**:
- grounds factual answers in permitted account/transaction data,
- enforces authentication, per-customer authorization and policy **outside** the LLM,
- escalates to a human with full handoff context when needed.

**Chosen workflow:** account & transaction information + payment/card inquiries (the ~57% of demand with ~90% first-contact resolution), with complaints as the human-escalation path. See [`analysis/EDA_FINDINGS.md`](analysis/EDA_FINDINGS.md) for the data-driven justification.

## Documentation

**Spec-driven plan** (Kiro specs, EARS criteria, full traceability):
- [`.kiro/specs/cora/requirements.md`](.kiro/specs/cora/requirements.md) - 52 testable requirements.
- [`.kiro/specs/cora/design.md`](.kiro/specs/cora/design.md) - architecture, state machine, policy engine, contracts.
- [`.kiro/specs/cora/tasks.md`](.kiro/specs/cora/tasks.md) - phased implementation tasks.
- [`documentation/SOURCE_REQUIREMENTS.md`](documentation/SOURCE_REQUIREMENTS.md) - all 84 obligations from the brief and data docs.
- [`documentation/TRACEABILITY.md`](documentation/TRACEABILITY.md) - source -> requirement -> task -> evidence.
- [`documentation/DECISIONS.md`](documentation/DECISIONS.md) - ADRs with alternatives and rationale.
- [`documentation/DEVELOPMENT_PLAN.md`](documentation/DEVELOPMENT_PLAN.md) - process, 10-day schedule, quality gates, risks.
- [`.kiro/steering/`](.kiro/steering/) - standing rules for every Kiro session on this repo.

**Context & evidence:**
- [`CORA_CONTEXT.md`](CORA_CONTEXT.md) - self-contained project context (challenge + dataset schema + rules). No PDFs needed.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - technical stack and local->AWS migration plan.
- [`analysis/EDA_FINDINGS.md`](analysis/EDA_FINDINGS.md) - exploratory analysis, charts and workflow selection.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

| Make (Linux/macOS) | PowerShell (Windows) | What |
|---|---|---|
| `make setup` | `.\tasks.ps1 setup` | `uv sync --locked --all-groups` |
| `make lint` | `.\tasks.ps1 lint` | ruff check + format check |
| `make format` | `.\tasks.ps1 format` | ruff autofix + format |
| `make test` | `.\tasks.ps1 test` | pytest |
| `make data` / `eval` / `demo` | `.\tasks.ps1 data` / `eval` / `demo` | pending (phases 1, 6, 5) |

Configuration: copy `.env.example` to `.env` (gitignored); values are loaded by `src/cora/settings.py`.
Only profile names, regions and model ids go there; AWS credentials come from SSO profiles.

## Reproduce the analysis

```bash
uv sync --locked --group analysis
# Configure the datathon S3 profile first (read-only; never commit keys):
#   aws configure set aws_access_key_id    <KEY>    --profile cora-datathon
#   aws configure set aws_secret_access_key <SECRET> --profile cora-datathon
#   aws configure set region us-east-2               --profile cora-datathon
# Then download a sample and run:
uv run python analysis/eda_sample.py --outdir analysis/figures
```

## Status

- [x] Project named, context captured, dataset mapped.
- [x] Bucket analyzed; workflow chosen with evidence.
- [x] Stack defined (`ARCHITECTURE.md`).
- [x] Spec-driven plan: requirements, design, tasks, traceability (84/84 mapped), ADRs, steering.
- [ ] Data contracts + quality checks.
- [ ] Tool/policy layer (auth, authorization, eligibility service).
- [ ] LangGraph agent flow.
- [ ] Evaluation harness + baseline.
- [x] Project skeleton (uv, ruff, pytest), settings, CI workflow (lint, tests, gitleaks, no-PDF check).

## Security

Synthetic data only. No real credentials in the repo - datathon S3 keys load from an AWS profile/env var; `.gitignore` excludes `.env`.
