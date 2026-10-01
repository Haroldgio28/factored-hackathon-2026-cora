---
inclusion: always
---

# Tech stack

- Python 3.12, `uv` (pinned lockfile), `ruff` (lint+format), `pytest`. Windows dev machine: provide PowerShell equivalents for Make targets.
- Data: DuckDB + Parquet partitioned by `process_date` locally; S3 + Glue + Athena on AWS, behind a `DataSource` interface. Contracts with Pandera.
- Agent: LangGraph state machine; LLM via an `LLMClient` interface (Bedrock proposed, local stub for tests).
- NLU: fastText language id; multilingual MiniLM embeddings + calibrated logistic regression for intents.
- Service: FastAPI; demo UI Streamlit. Tracing: OpenTelemetry -> JSONL (Langfuse optional). Retries: `tenacity`.
- AWS target: Fargate/Lambda, DynamoDB, Cognito, Secrets Manager, CloudWatch/X-Ray, CDK (Python).
- Host RAM is limited (~4 GB free): prefer sampled data, CPU-small models, targeted tests; no local large LLMs.
- AWS data access uses the profile `cora-datathon` (read-only). For S3 downloads prefer `aws s3api get-object`.
- Pin exact dependency versions; justify any new dependency in the PR.
