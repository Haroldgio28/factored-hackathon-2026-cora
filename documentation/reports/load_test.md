# Load test — CORA FastAPI service (task 7.1, REQ-50 / REQ-51)

A lazy concurrent load-test harness (`scripts/load_test.py`) driven against the **local** CORA
FastAPI service on the **offline stub path**. It measures service + orchestration overhead
honestly, without live Bedrock: with `CORA_NLU_STUB=1` the LLM is the deterministic
`StubLLMClient` and intents use the keyword baseline, so every number below reflects routing,
policy, the per-session lock, tool reads and template rendering — not model latency.

Bedrock-dependent latency (LLM polish on the `/chat` path when the breaker is closed and Bedrock
is configured) is **owner-run on a Bedrock-quota host**; it is deliberately NOT in these numbers
and must not be inferred from them.

## Harness

- Stdlib `concurrent.futures.ThreadPoolExecutor` + the already-present dev dep `httpx` +
  stdlib `statistics`. No locust/k6 (no heavy new dependency added — ponytail rung 5).
- Per virtual user, per request: `POST /auth` (start, no code) → `POST /auth` (with the
  mock-delivered OTP) → `POST /chat`. One auth+chat pair counts as one "turn".
- Latency is split into an `auth` phase (two `/auth` calls) and a `chat` phase (one `/chat` turn);
  p50/p95 are nearest-rank over the **successful** turns; throughput is completed turns / wall;
  error rate is failed turns / total.

## How to reproduce

Start the service on the stub path (separate shell):

```powershell
$env:CORA_NLU_STUB=1
uv run uvicorn cora.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Then drive it:

```powershell
uv run python scripts/load_test.py --base-url http://127.0.0.1:8000 `
    --customer-id CLI-0Y0HOQ83ZSO4 --concurrency 16 --requests 20
```

`--customer-id` is any id the mock IdP will issue a token for; the id above is a real id from the
local `data/raw_parquet` landing, so the chat turn exercises a grounded product read.

## Measured results (real local stub-path runs, this host)

Windows dev host, single-process uvicorn (one worker), `CORA_NLU_STUB=1`, local landing.
These are actual runs executed for this task, not estimates.

| Concurrency | Turns | Errors | Error rate | Wall (s) | Throughput (turns/s) | auth p50 / p95 (ms) | chat p50 / p95 (ms) |
|-------------|-------|--------|------------|----------|----------------------|---------------------|---------------------|
| 1           | 50    | 0      | 0.0 %      | 0.84     | 59.6                 | 6.0 / 12.4          | 6.4 / 10.8          |
| 16          | 320   | 0      | 0.0 %      | 2.96     | 108.2                | 65.8 / 120.1        | 46.3 / 102.7        |
| 32          | 640   | 0      | 0.0 %      | 5.10     | 125.5                | 135.0 / 199.6       | 98.1 / 149.3        |

## Capacity observations

- **Zero errors across all three runs** — the service is stable under this load; no 5xx, no
  timeouts, no dropped turns.
- At **concurrency 1** a full auth+chat turn costs ~12 ms end to end (6 ms auth, 6 ms chat); the
  stub orchestration path is cheap.
- **Throughput saturates** around ~110–125 turns/s between concurrency 16 and 32: raising
  concurrency 2× (16→32) lifts throughput only ~16 % while roughly **doubling p50/p95 latency**.
  That is the classic single-process queueing signature — added clients wait, they don't get
  served faster.
- Latency grows with concurrency because requests queue behind the one process; the per-turn work
  itself did not get slower (the c=1 numbers show the real service cost).

## Identified bottleneck

1. **Single-process uvicorn.** The dominant limit here. One worker serializes CPU-bound Python
   work (orchestration, policy, template render) on one core; throughput plateaus and latency
   climbs with concurrency. Fix: run multiple workers / replicas behind the ASGI server
   (`--workers N`, or Fargate task count on AWS) — the handlers are already built as stateless
   wiring (`create_app` composition root), so horizontal scaling needs no code change.
2. **Per-`jti` turn lock.** `Orchestrator.step()` runs under a per-session (`jti`) lock so
   overlapping same-token turns can't interleave the shared mutable `SessionState` (REQ-51,
   per-session turn isolation). This serializes *same-token* turns by design (correctness over
   throughput). It is NOT the bottleneck in this test — every virtual user authenticates its own
   token, so turns run under distinct locks — but it would serialize a single customer firing
   concurrent turns. That is the intended safety trade-off, not a defect.

Neither bottleneck involves Bedrock; adding live LLM polish shifts the `/chat` phase from
template-render time to model round-trip time, which is the owner-run measurement below.

## Owner-run (not measured in this sandbox)

- **Live Bedrock `/chat` latency.** With Bedrock configured and the circuit breaker closed, the
  `/chat` turn adds a Converse-API round trip for response polish. This host has no Bedrock quota
  wired for a load run, so it is **owner-run on a Bedrock-quota host**: re-run the same harness
  with `CORA_NLU_STUB` unset and `CORA_BEDROCK_MODEL_ID` set, and compare the `chat` p50/p95
  against the stub numbers above to isolate model latency from orchestration overhead.

## Verification (gates run for this task, from the repo root)

```
uv run ruff check .          -> All checks passed!
uv run ruff format --check . -> 190 files already formatted
uv run pytest                -> 509 passed, 1 skipped (in ~88s)
```

The 1 skip is the pre-existing data-dependent S3 test (not a regression; baseline was
504 passed / 1 skipped). This subtask adds 5 tests (`tests/unit/test_load_test.py`),
raising the pass count to 509.
