"""Lazy concurrent load-test harness against a LOCAL CORA FastAPI service (task 7.1, REQ-50/51).

ponytail: no locust/k6 (heavy new deps). `concurrent.futures.ThreadPoolExecutor` + the already-dev
`httpx` client cover a throughput/latency/error-rate driver in a few lines of stdlib.

What it drives, per virtual user, `--requests` times:
  1. POST /auth (no code)  -> {challenge_id, otp_code}  (the mock IdP delivers the code inline)
  2. POST /auth (with code) -> {token}
  3. POST /chat (token + utterance) -> one orchestrator turn

It measures the OFFLINE STUB path: start the service with `CORA_NLU_STUB=1` so the LLM is the
deterministic `StubLLMClient` and intents use the keyword baseline. That honestly isolates
service + orchestration overhead (routing, policy, session lock, tool reads, template render)
WITHOUT live Bedrock latency, which is owner-run on a Bedrock-quota host (see the report).

The server must already be running (this script does not spawn it), e.g.:

    $env:CORA_NLU_STUB=1
    uv run uvicorn cora.api.app:create_app --factory

Then:

    uv run python scripts/load_test.py --base-url http://127.0.0.1:8000 \
        --customer-id CLI-0Y0HOQ83ZSO4 --concurrency 16 --requests 20

The customer id only needs to be a well-formed id the mock IdP will issue a token for; a real id
from the landing exercises the grounded product read, but any id drives the same orchestration.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx

_DEFAULT_UTTERANCE = "¿Cuál es el saldo de mi tarjeta?"


@dataclass(frozen=True)
class TurnResult:
    """One virtual-user iteration: auth + chat, with per-phase latencies in milliseconds."""

    ok: bool
    auth_ms: float
    chat_ms: float
    error: str | None = None


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100]) over `values`; `values` must be non-empty.

    Nearest-rank (not interpolated) is the honest choice for small samples: it always returns an
    observed latency, never a value between two measurements. `statistics.quantiles` would need
    n >= 2 and interpolates; nearest-rank is one sort + one index and is correct at the edges.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be in [0, 100]")
    ordered = sorted(values)
    if pct == 0:
        return ordered[0]
    # Rank = ceil(pct/100 * n), 1-based; clamp into range and convert to a 0-based index.
    rank = -(-int(pct) * len(ordered) // 100)  # ceil without float error
    index = min(max(rank, 1), len(ordered)) - 1
    return ordered[index]


def _run_turn(client: httpx.Client, customer_id: str, utterance: str, language: str) -> TurnResult:
    """Drive one auth (two-step OTP) + one chat turn, timing each phase. Never raises."""
    try:
        t0 = time.perf_counter()
        start = client.post("/auth", json={"customer_id": customer_id})
        start.raise_for_status()
        challenge = start.json()
        verify = client.post(
            "/auth",
            json={
                "customer_id": customer_id,
                "challenge_id": challenge["challenge_id"],
                "otp_code": challenge["otp_code"],
            },
        )
        verify.raise_for_status()
        token = verify.json()["token"]
        auth_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        chat = client.post("/chat", json={"token": token, "utterance": utterance, "language": language})
        chat.raise_for_status()
        chat_ms = (time.perf_counter() - t1) * 1000.0
        return TurnResult(ok=True, auth_ms=auth_ms, chat_ms=chat_ms)
    except Exception as exc:  # noqa: BLE001 - a load driver records failures, never crashes the run
        return TurnResult(ok=False, auth_ms=0.0, chat_ms=0.0, error=f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class Summary:
    """Aggregate report over every completed turn."""

    total: int
    errors: int
    wall_seconds: float
    auth_p50_ms: float
    auth_p95_ms: float
    chat_p50_ms: float
    chat_p95_ms: float

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0

    @property
    def throughput_rps(self) -> float:
        """Completed turns per second over the wall clock (auth+chat counts as one turn)."""
        return self.total / self.wall_seconds if self.wall_seconds else 0.0


def summarize(results: list[TurnResult], wall_seconds: float) -> Summary:
    """Aggregate throughput, p50/p95 latency (over SUCCESSFUL turns), and error rate."""
    ok = [r for r in results if r.ok]
    auth = [r.auth_ms for r in ok] or [0.0]
    chat = [r.chat_ms for r in ok] or [0.0]
    return Summary(
        total=len(results),
        errors=sum(1 for r in results if not r.ok),
        wall_seconds=wall_seconds,
        auth_p50_ms=percentile(auth, 50),
        auth_p95_ms=percentile(auth, 95),
        chat_p50_ms=percentile(chat, 50),
        chat_p95_ms=percentile(chat, 95),
    )


def run_load(
    base_url: str,
    customer_id: str,
    *,
    concurrency: int,
    requests: int,
    utterance: str,
    language: str,
    timeout: float,
) -> tuple[Summary, list[TurnResult]]:
    """Fire `concurrency * requests` turns across a thread pool and summarize the results."""
    total = concurrency * requests
    results: list[TurnResult] = []
    # One client per worker thread (httpx.Client is not meant to be shared concurrently); the pool
    # has exactly `concurrency` workers, so we hand each a long-lived client via a thread-local.
    import threading

    local = threading.local()

    def worker(_: int) -> TurnResult:
        client = getattr(local, "client", None)
        if client is None:
            client = httpx.Client(base_url=base_url, timeout=timeout)
            local.client = client
        return _run_turn(client, customer_id, utterance, language)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(total)]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - start
    return summarize(results, wall), results


def _format(summary: Summary) -> str:
    return (
        f"turns={summary.total} errors={summary.errors} "
        f"error_rate={summary.error_rate:.1%} wall={summary.wall_seconds:.2f}s "
        f"throughput={summary.throughput_rps:.1f} turns/s\n"
        f"auth  p50={summary.auth_p50_ms:.1f}ms p95={summary.auth_p95_ms:.1f}ms\n"
        f"chat  p50={summary.chat_p50_ms:.1f}ms p95={summary.chat_p95_ms:.1f}ms"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Concurrent stub-path load test for the CORA API.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="Running service base URL.")
    ap.add_argument("--customer-id", required=True, help="Customer id to authenticate (mock OTP).")
    ap.add_argument("--concurrency", type=int, default=16, help="Parallel virtual users.")
    ap.add_argument("--requests", type=int, default=20, help="Turns per virtual user.")
    ap.add_argument("--utterance", default=_DEFAULT_UTTERANCE, help="Chat utterance to send.")
    ap.add_argument("--language", default="es", help="Customer language preference (es/pt).")
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout (seconds).")
    args = ap.parse_args()

    summary, _ = run_load(
        args.base_url,
        args.customer_id,
        concurrency=args.concurrency,
        requests=args.requests,
        utterance=args.utterance,
        language=args.language,
        timeout=args.timeout,
    )
    print(_format(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
