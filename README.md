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

- [`CORA_CONTEXT.md`](CORA_CONTEXT.md) - self-contained project context (challenge + dataset schema + rules). No PDFs needed.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - technical stack and local->AWS migration plan.
- [`analysis/EDA_FINDINGS.md`](analysis/EDA_FINDINGS.md) - exploratory analysis, charts and workflow selection.

## Reproduce the analysis

```bash
pip install -r requirements.txt
# Configure the datathon S3 profile first (read-only; never commit keys):
#   aws configure set aws_access_key_id    <KEY>    --profile cora-datathon
#   aws configure set aws_secret_access_key <SECRET> --profile cora-datathon
#   aws configure set region us-east-2               --profile cora-datathon
# Then download a sample and run:
python analysis/eda_sample.py --outdir analysis/figures
```

## Status

- [x] Project named, context captured, dataset mapped.
- [x] Bucket analyzed; workflow chosen with evidence.
- [x] Stack defined (`ARCHITECTURE.md`).
- [ ] Data contracts + quality checks.
- [ ] Tool/policy layer (auth, authorization, eligibility service).
- [ ] LangGraph agent flow.
- [ ] Evaluation harness + baseline.
- [ ] GitHub remote + CI.

## Security

Synthetic data only. No real credentials in the repo - datathon S3 keys load from an AWS profile/env var; `.gitignore` excludes `.env`.
