# Production-readiness dossier — CORA (task 7.2, REQ-50 / REQ-51)

An honest readiness assessment of the CORA prototype: what it can carry today, how it is
observed, who can reach what, how long data lives, the AWS target it is wired toward, and the
work that remains before any deployment. No live service is expected (SRC-14); this is the
engineering-judgement artifact, not an operations runbook.

Numbers here are **not re-measured**: capacity figures reference the task 7.1 load test
([`load_test.md`](load_test.md)) and are not duplicated or re-estimated. Everything else is read
from the code and setup docs already in the repo — no invented metrics, endpoints or SLAs.

## Security invariant (holds across every section below)

The architecture that makes this readiness assessment meaningful is: **the LLM proposes,
deterministic code decides.** Identity, authorization, policy and confirmation never depend on
model output. The system **fails closed** — on expiry, tampering, ambiguity or tool failure it
discloses nothing, does nothing, and offers a human. **No tool moves money or changes balances**,
and **credit eligibility is never evaluated** (POL-030). These are structural guarantees, not
runtime toggles, so they do not degrade under load, during an incident, or when an LLM call fails.

## 1. Capacity limits

Measured on the **local stub path** (`CORA_NLU_STUB=1`, single-process uvicorn, one worker,
Windows dev host) in task 7.1. The full table and reproduction steps live in
[`load_test.md`](load_test.md); the readiness-relevant takeaways:

- **Stable under test load.** Zero errors across all runs (concurrency 1, 16, 32): no 5xx, no
  timeouts, no dropped turns.
- **Throughput plateaus at ~110–125 turns/s** between concurrency 16 and 32. Doubling concurrency
  (16→32) buys ~16 % more throughput while roughly doubling p50/p95 latency — the classic
  single-process queueing signature.
- **A full auth+chat turn costs ~12 ms end to end at concurrency 1** on the stub path, so the
  orchestration/policy/template work itself is cheap; latency growth is queueing, not per-turn cost.

Two bottlenecks are identified in 7.1 and both are understood rather than defects:

1. **Single-process uvicorn** is the dominant limit. The HTTP handlers are stateless wiring around
   the `create_app` composition root, so adding workers/replicas is a configuration change at the
   process boundary (`--workers N` locally, Fargate task count on AWS).
2. **Per-`jti` turn lock.** `Orchestrator.step()` runs under a per-session lock so overlapping
   same-token turns cannot interleave the shared mutable `SessionState` (REQ-51). This serializes
   *same-token* turns by design (correctness over throughput); it is not the test bottleneck
   because each virtual user holds its own token.

**Horizontal scaling is NOT currently safe and NOT proven for REQ-51.** Although the handlers are
stateless, the per-session *state* is not shared: `create_app()` builds a fresh in-memory
`SessionStore` per process, and both the mutable `SessionState` and the per-`jti` lock registry
live in that process's memory (`src/cora/agent/state.py`, `src/cora/agent/graph.py`). With multiple
workers or Fargate tasks and no sticky routing, two requests carrying the **same token** can land
on different processes, observe different conversation/confirmation state, and bypass the
per-process turn serialization entirely. Scaling out therefore requires real code work, not just a
worker/task count: move session + handoff state to a shared store (the documented DynamoDB target,
section 5) and implement + test cross-worker same-session concurrency/idempotency. The 7.1 numbers
describe the **measured single-process** service only; the multi-process target is unimplemented and
unvalidated (section 6).

**Not measured and must not be inferred from the above:** live Bedrock `/chat` latency. With
Bedrock configured and the breaker closed, `/chat` adds a Converse-API round trip for response
polish. This host has no Bedrock load quota, so that figure is **owner-run on a Bedrock-quota
host** (re-run the 7.1 harness with `CORA_NLU_STUB` unset). Capacity planning for a deployed
service must use that measurement, not the stub numbers.

## 2. Monitoring & alerting plan

CORA attempts to append one execution record per turn (best-effort exporter). The plan reuses that, it does not add a stack.

- **Local, today:** `src/cora/obs/tracing.py` appends one JSON record per turn to
  `<trace_dir>/traces.jsonl` (default `data/_state/traces`, gitignored). Each record carries a
  stable per-turn `trace_id` (threaded into the handoff `trace_ref` and the agent console),
  language, intent + confidence, the policy decision + POL rule id + rules version, the per-node
  spans, and the output signal (template outcome / polished flag / grounding result). Export is
  **best-effort** — a trace write failure is logged and swallowed and never gates, reorders or
  rolls back an already-verified turn (a trace is an audit artifact, not a control-flow gate).
- **No raw PII in traces, enforced twice:** the record never reads the rendered customer text, and
  every free-form field (`masked_input`, `message`, span `detail`) is run through `mask_pii` at the
  serialization boundary. `tests/unit/test_tracing.py` drives all five PII categories through to
  prove it.
- **AWS target (documented seams, not yet wired):** the OTel → X-Ray path is kept behind a single
  seam — swap the sink inside `export_turn`, no caller change. Langfuse export is a documented,
  intentionally no-op stub (`_emit_to_langfuse`): there is no SDK, no `CORA_LANGFUSE_*` setting and
  no conditional today — the stub is called unconditionally and returns immediately, so the JSONL
  sink stays the single source of truth. A settings flag (e.g. `CORA_LANGFUSE_ENABLED`) is the
  documented upgrade path in `tracing.py`, proposed remaining work, not an existing control. On
  AWS, ship traces to **CloudWatch Logs / X-Ray** (the ARCHITECTURE.md observability row) with
  Langfuse optional for a hosted trace UI.
- **Alerting (target, proposed):** CloudWatch alarms on service error rate (any 5xx), p95 `/chat`
  latency against the owner-run Bedrock baseline, Bedrock `ThrottlingException` rate, and the
  circuit-breaker-open signal (`src/cora/obs/resilience.py`) so repeated LLM failure that trips
  template-only fallback pages an operator. **Cost control today is a hard monthly spend limit**
  on the dev account (the actual control per [`AWS_SETUP.md`](../AWS_SETUP.md) addendum), not an
  active alarm. The Step 2 Budgets template and CloudWatch billing alerts are an **unchecked
  recommended setup**, not proven-enabled, and Cost Anomaly Detection is enterprise-only and
  unavailable on this account — so a Budgets email alert is a recommended/target control, not a
  current one.
- **Still missing from REQ-39 telemetry:** tool-call latency, retry counts, model id, token counts
  and per-turn cost are not yet on the turn record. They attach to the same record via the 5.2/5.3
  producers (see remaining work) — no new record type.

## 3. Access controls

- **Customer identity (today):** the mock IdP (`src/cora/identity/service.py`) gates every
  customer-data read behind a one-time OTP step and an HMAC-signed session JWT (`sub=customer_id`,
  `exp`, unique `jti`). A national ID or customer number alone never proves identity. The token is
  never echoed to the LLM and never logged.
- **`customer_id` provenance:** tools receive `customer_id` only from the verified session token,
  never from model output or user text. This is the structural core of per-customer authorization.
- **Agent console (today):** the `/handoffs` read endpoints are the agent-facing surface and
  require a static key in the `X-Agent-Key` header, compared **constant-time** against
  `CORA_AGENT_CONSOLE_KEY`. It **fails closed**: a missing, empty, wrong, or entirely unconfigured
  key returns a 401 that discloses nothing — the 401 never distinguishes "no key set" from "wrong
  key". The key is a secret, so only its name ships in `.env.example` (empty).
- **AWS target (documented seams):** the mock IdP is replaced by **Cognito**, and the static
  agent-console key is replaced by a **Cognito group/role claim** — a documented seam in
  `settings.py`, not JWT/role machinery to be written here.
- **AWS credentials (least privilege, no long-lived keys):** the baseline rule is short-lived
  credentials only — no root access keys, no long-lived access keys, ever.
  [`AWS_SETUP.md`](../AWS_SETUP.md) documents the **intended classic-account** setup: IAM Identity
  Center (SSO) with `cora-dev` (`CoraDeveloper` permission set, 8 h sessions) for daily
  least-privilege work and `cora-admin` (`CoraAdmin`, 4 h) for infrastructure only.
  **Actual dev-account exception (task 0.4, AWS_SETUP.md addendum, `bedrock_smoke_test.md`):** the
  "Sign up for AWS (new)" account used for CORA does **not** support IAM Identity Center, so there
  is no `cora-dev`/`cora-admin` SSO profile today; it is in `us-east-2`, uses `global.` inference
  profiles, and authenticates to Bedrock with a **short-term API bearer token** (≤12 h, exported
  per shell session, never written to `.env` or committed) — still short-lived, still no long-lived
  keys. `cora-datathon` is the organizer's separate read-only data profile and is never mixed with
  this account. The SSO path remains the target once an account that supports Identity Center is
  used.

## 4. Data retention

- **PII masking before any LLM call** is the baseline rule: document number, email, phone, address
  and full card number are masked before any model boundary, and the same `mask_pii` is applied
  defensively at the trace serialization boundary (section 2).
- **Session JWT TTL: 15 minutes** (absolute expiry fixed at issuance, `CORA_IDENTITY_SESSION_TTL_
  MINUTES`, default 15). On expiry the token is rejected and the system fails closed. The OTP
  challenge is single-use and expires after 5 minutes. **The 15-minute TTL bounds *access*, not
  in-memory state lifetime.** `/chat` calls `verify_token()` first, so an expired token returns 401
  *before* the orchestrator runs. Runtime callers of `discard()` do exist — the orchestrator calls
  it when it observes an expired session (`src/cora/agent/graph.py`), and `SessionStore.require()`
  calls it on its own expiry branch (`src/cora/agent/state.py`) — but both only fire when a request
  carrying the same `jti` actually reaches them (e.g. a clock-skew edge within the window). The gap
  is the common case: once a token is rejected at the API boundary by `verify_token()`, nothing
  routes to those branches, and there is **no background expiry sweep, no TTL eviction in
  `SessionStore`, and no logout endpoint** to proactively drop the entry. So an expired entry can
  linger in process memory until the process restarts. Enforced local state deletion on
  expiry/logout is remaining work (section 6).
- **Transcripts:** CORA keeps per-session mutable state in memory (`SessionStore`) keyed by `jti`;
  it does not persist a durable transcript today. Access to that state fails closed once the token
  expires, but as noted above the entry is not actively deleted on expiry/logout.
- **Traces:** JSONL append under the gitignored runtime dir; PII-masked, never committed. No TTL
  is enforced locally today.
- **Handoffs:** `JsonHandoffStore` persists each package to a gitignored runtime JSON file under
  `data/_state/` so the agent console can read open cases across restarts.
- **Proposed retention policy (target):** on AWS, store traces and handoff records with explicit
  TTLs rather than keeping them forever — a DynamoDB TTL attribute on session and handoff items,
  and a lifecycle/retention policy on the trace log store (e.g. CloudWatch Logs retention or an S3
  lifecycle rule). Concrete day counts are a product/compliance decision and are left for the owner;
  the mechanism is the seam, not a hard-coded number.

## 5. AWS target architecture

The complete target table is from [`.kiro/specs/cora/design.md`](../../.kiro/specs/cora/design.md)
§12 (deployment views), and the service list is repeated in `tasks.md` task 8.1 — not invented.
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) supplies the guiding principle (local↔AWS parity: data
access and tools sit behind stable interfaces with local and AWS implementations, so migration is a
config change at each seam, not a rewrite) and the observability row (Langfuse/OTel).
[`AWS_SETUP.md`](../AWS_SETUP.md) supplies the account/bootstrap facts. Where the **current
implementation** differs from stale local entries in the design, the deviation is labelled below.

| Concern | Local (today) | AWS target (design §12) |
|---|---|---|
| Compute | single-process uvicorn | **Fargate / Lambda** (horizontal scale, section 1) |
| Session + handoff store | in-memory `SessionStore`, `JsonHandoffStore` (gitignored JSON) — deviates from design's SQLite row | **DynamoDB** (with TTL, section 4) |
| Data | DuckDB over local Parquet partitioned by `process_date` | **S3 + Glue + Athena** behind the `DataSource` interface |
| LLM | Bedrock via `LLMClient` (stub for tests) | **Amazon Bedrock** (ADR-005) |
| Identity | mock IdP (OTP + HMAC JWT) | **Cognito** |
| Agent-console auth | static `X-Agent-Key` | **Cognito group/role claim** |
| Secrets | `.env` / short-term credentials (section 3) | **Secrets Manager** |
| Observability | JSONL traces, best-effort | **CloudWatch / X-Ray** (OTel collector), Langfuse optional |
| Infra as code | — | **CDK (Python)**, Phase 8 |

The intended dev region is `us-east-1` (Bedrock coverage) per `AWS_SETUP.md`; the actual CORA
dev account runs in `us-east-2` (section 3), and the organizer bucket keeps its own `us-east-2`
read-only profile. Infra provisioning (CDK bootstrap, project bucket, Glue DB, Athena workgroup,
extended `CoraDeveloper` policy) is Phase 8 (task 8.1) and out of scope here.

## 6. Remaining work before deployment (honest)

These are real gaps, stated plainly. None is hidden behind an optimistic SLA.

- **Orchestration is a stdlib dispatcher, not LangGraph.** ADR-004 chose a deterministic stdlib
  state dispatcher over the LangGraph named in ARCHITECTURE.md. The decide/confirm/abstain/escalate
  transitions are code, not a graph library. Fine for the prototype; revisit if graph tooling buys
  something concrete.
- **Thresholds come from a 98-scenario subset.** The evaluation ran on a reduced 98-case stratified
  subset with 1 repeat (see `EVALUATION.md`), below the ≥3-run, full-workload bar. Treat the
  reported metrics as a reduced offline measurement, not a production baseline.
- **Portuguese data is synthetic/translated.** The dataset is Spanish-only; PT coverage is
  team-generated/translated (192 translated + 48 team-generated), each labeled with provenance. PT
  behavior is not validated on real PT traffic.
- **No human-eval labels.** The deterministic judges carry the evaluation; there is no ≥50-label
  human-validated rubric behind any LLM-judge dimension, so subjective quality is unmeasured.
- **Single-process, local only.** No horizontal scaling, no replicas, no deployed environment has
  been exercised. The scale-out path is understood (section 1) but unproven.
- **No shared session/handoff state or cross-worker concurrency.** Session state and the per-`jti`
  turn locks are process-local; same-session serialization and idempotency do not span workers.
  Moving to the DynamoDB target (section 5) and testing cross-worker same-session behavior is
  required before any multi-process deployment (REQ-51).
- **Mock IdP and static agent key, not Cognito.** Customer identity (OTP + HMAC JWT) and the static
  `X-Agent-Key` agent-console gate must be replaced by Cognito customer and agent/group
  authorization (section 3). Only seams exist today; no Cognito integration is implemented.
- **No real observability export or alarms.** The CloudWatch/X-Ray path and the Langfuse export are
  seams/no-ops (section 2); actual trace shipping and the proposed CloudWatch alarms are not wired.
- **Retention/deletion not enforced.** Local session/transcript cleanup on expiry/logout is not
  implemented (section 4), and the proposed trace/handoff TTLs and owner-approved retention
  durations are unset. The mechanisms (DynamoDB TTL, log/S3 lifecycle) are documented, not applied.
- **No Secrets Manager, CDK/IaC, deployment, or AWS-environment validation.** Secrets still come
  from `.env`/short-term credentials; infra-as-code, provisioning and an end-to-end AWS run are
  Phase 8 (task 8.1) and have not been executed.
- **Load test is on the stub path.** Capacity numbers isolate service/orchestration overhead with a
  deterministic stub LLM; live-Bedrock `/chat` latency under load is owner-run and not yet measured.
- **REQ-39 telemetry is partial.** Tool latency, retries, model id, tokens and per-turn cost are not
  yet on the trace record (5.2/5.3 producers).
- **Clean-clone reproducibility pending** (task 7.6): the lockfile and task runners are in place, but
  a from-scratch clone-and-run has not been recorded.

Until these close, CORA is a demonstrable prototype with a credible operations route — not a
production service.
