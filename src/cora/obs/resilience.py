"""A tiny stdlib circuit breaker for the LLM polish path (task 5.2, REQ-40, design section 10/13).

Design section 13 puts resilience helpers in `obs/`. This is the laziest honest breaker: a
consecutive-failure counter plus a three-state machine (closed -> open -> half-open), no library
(ponytail rung 5 declined - a dependency buys nothing a counter and a timestamp do not).

The breaker SELECTS which path a turn runs (LLM polish vs deterministic template); it is a routing
switch, NOT a gate. It never fails a turn: `generate(...)` already falls back to the grounded
template on `LLMUnavailable`, so the breaker is an optimisation + protection layer on top of an
already-safe fallback, never the thing that makes the fallback safe. Its state is in memory on the
instance (one breaker per orchestrator/process); it is best-effort, exactly like the trace sink
(security steering P5): a state update must never raise into a turn, and no state is persisted.

State machine:
- `closed`: calls flow to the LLM. `failure_threshold` CONSECUTIVE failures -> `open`.
- `open`: short-circuit to the template without touching the LLM. After `cooldown` elapses the next
  `allow()` returns True and moves to `half-open` (one trial call).
- `half-open`: one trial. A success -> `closed` (recovered); a failure -> `open` (cooldown restarts).

The clock is injected via the same `Callable[[], datetime]` seam the orchestrator and identity
service already use (do NOT reinvent a clock), so a test advances time without `sleep`.

# ponytail: in-memory per-process breaker, no library, no shared store. Upgrade path = a shared
#           store (e.g. DynamoDB/Redis) ONLY if multi-process breaker coordination is ever needed;
#           for a single Fargate task / local demo one instance is correct.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum

logger = logging.getLogger("cora.obs.resilience")

__all__ = ["BreakerState", "CircuitBreaker"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Consecutive-failure circuit breaker with a cooldown and an injectable clock.

    `failure_threshold` consecutive failures open the breaker; after `cooldown` it half-opens for
    one trial. `allow()` is the only question callers ask ("may I attempt the LLM this turn?");
    `record_success`/`record_failure` feed the state machine from the attempt's result.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(seconds=30),
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: datetime | None = None

    @property
    def state(self) -> BreakerState:
        """The current state, promoting open -> half-open if the cooldown has elapsed (read-only)."""
        if self._state is BreakerState.OPEN and self._cooldown_elapsed():
            self._state = BreakerState.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        """True when a call may attempt the LLM: closed, or half-open after the cooldown elapsed.

        While open and still cooling down, returns False so the caller short-circuits to the
        template WITHOUT attempting the LLM.
        """
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        """A successful call: reset the failure count and close the breaker (recovered)."""
        self._consecutive_failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        """A failed call: in half-open re-open immediately; in closed open on the threshold."""
        if self.state is BreakerState.HALF_OPEN:
            self._open()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._open()

    def _open(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        logger.info("circuit breaker opened after %d failure(s)", self._consecutive_failures)

    def _cooldown_elapsed(self) -> bool:
        return self._opened_at is not None and self._clock() - self._opened_at >= self._cooldown


if __name__ == "__main__":  # self-check: open-after-threshold and half-open-after-cooldown
    clock_now = datetime(2026, 1, 1, tzinfo=UTC)
    breaker = CircuitBreaker(failure_threshold=3, cooldown=timedelta(seconds=30), clock=lambda: clock_now)
    assert breaker.allow() and breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.allow(), "below threshold must still allow"
    breaker.record_failure()  # third consecutive -> open
    assert not breaker.allow() and breaker.state is BreakerState.OPEN

    clock_now += timedelta(seconds=31)  # cooldown elapsed -> half-open on the next check
    assert breaker.allow() and breaker.state is BreakerState.HALF_OPEN
    breaker.record_failure()  # trial failed -> open again
    assert not breaker.allow()

    clock_now += timedelta(seconds=31)
    assert breaker.allow() and breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()  # trial ok -> closed
    assert breaker.allow() and breaker.state is BreakerState.CLOSED
    print("resilience circuit-breaker self-check OK")
