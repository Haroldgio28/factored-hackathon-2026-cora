"""Seeded fault injection for the evaluation runner (task 6.3, REQ-44, design section 9).

The runner exercises B1 and CORA with faults ON so the report can show how each handles a flaky
backend (REQ-44 "variability", design section 9 "fault-injection wrapper raising timeouts/5xx at a
set rate"). This is the only genuinely new resilience code the phase adds; `obs/resilience.py`
already has the `CircuitBreaker` that reacts to failures, so faults.py only needs to CAUSE them.

`FaultInjectingToolLayer` wraps a real `ToolLayer` and, at a seeded per-call rate, turns a tool
READ into `Status.UNAVAILABLE` instead of calling the underlying tool - exactly the shape a
timeout / 5xx surfaces as after the layer's own bounded retries (`Status.UNAVAILABLE`), so the
orchestrator's existing fail-closed path (`graph.py`: tool_failed -> TOOL_UNAVAILABLE + escalation
streak) is what gets exercised. Only reads are injected: there is no money-movement tool to fault,
and a write (freeze/unfreeze) already has its own read-back fail-closed path.

Determinism (REQ-44): one `random.Random(seed)` drives the per-call draw, so a fixed seed yields
the SAME injected failures across the x3 repeats - a repeat is a true replay, not a reroll. The
proxy delegates every non-faulted attribute (`handoff_store`, `confirmations`, `customer_id`, the
write tools) straight through, so wrapping a layer changes nothing except that some reads fail.

    # ponytail: a __getattr__ proxy over the real ToolLayer, not a reimplementation. Faulting the
    # five READ tools covers design section 9; writes keep their own read-back guard, so there is
    # nothing to add there.
"""

from __future__ import annotations

import random

from cora.tools import Result, Status, ToolLayer

__all__ = ["FAULTABLE_TOOLS", "FaultInjectingToolLayer"]

# The read tools a fault may hit. A faulted call returns `Status.UNAVAILABLE` (the post-retry
# timeout/5xx shape) WITHOUT touching the backend. The write tools (`freeze_card`/`unfreeze_card`)
# and `create_handoff` are left real: a write already fails closed on its own read-back, and
# faulting a handoff write would drop the escalation the fault is supposed to cause.
FAULTABLE_TOOLS: frozenset[str] = frozenset(
    {"list_products", "get_balance", "search_transactions", "get_card_details", "convert_currency"}
)

_UNAVAILABLE_MESSAGE = "injected fault: backend timeout/5xx"


class FaultInjectingToolLayer:
    """A `ToolLayer` proxy that fails a seeded fraction of read calls with `Status.UNAVAILABLE`.

    `rate` is the per-call failure probability in [0, 1] (0 = never fault, a pass-through). The
    draw uses the injected `random.Random`, so the SAME seed replays the SAME failures across
    repeats. Every attribute other than a faultable read is delegated to the wrapped layer, so the
    orchestrator sees an ordinary tool layer that is simply unreliable on reads.
    """

    def __init__(self, inner: ToolLayer, *, rate: float, rng: random.Random) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("fault rate must be in [0, 1]")
        self._inner = inner
        self._rate = rate
        self._rng = rng

    def __getattr__(self, name: str) -> object:
        """Delegate to the wrapped layer, wrapping only the faultable read tools.

        `__getattr__` fires only for attributes not set on this proxy, so `_inner`/`_rate`/`_rng`
        (set in `__init__`) are never routed here - no recursion. A faultable tool is returned as
        a closure that draws once and either short-circuits to UNAVAILABLE or calls through.
        """
        attr = getattr(self._inner, name)
        if name in FAULTABLE_TOOLS and callable(attr):

            def faulted(*args: object, **kwargs: object) -> object:
                if self._rng.random() < self._rate:
                    return Result(status=Status.UNAVAILABLE, message=_UNAVAILABLE_MESSAGE)
                return attr(*args, **kwargs)

            return faulted
        return attr


if __name__ == "__main__":  # self-check: rate=0 never faults, rate=1 always, same seed replays
    import random as _random

    class _StubInner:
        customer_id = "CLI-1"

        def get_balance(self, _inp: object) -> Result:
            return Result(status=Status.OK)

    inner = _StubInner()
    always = FaultInjectingToolLayer(inner, rate=1.0, rng=_random.Random(0))  # type: ignore[arg-type]
    never = FaultInjectingToolLayer(inner, rate=0.0, rng=_random.Random(0))  # type: ignore[arg-type]
    assert always.get_balance(None).status is Status.UNAVAILABLE
    assert never.get_balance(None).status is Status.OK
    assert always.customer_id == "CLI-1"

    one = FaultInjectingToolLayer(inner, rate=0.5, rng=_random.Random(7))  # type: ignore[arg-type]
    two = FaultInjectingToolLayer(inner, rate=0.5, rng=_random.Random(7))  # type: ignore[arg-type]
    seq_one = [one.get_balance(None).status for _ in range(20)]
    seq_two = [two.get_balance(None).status for _ in range(20)]
    assert seq_one == seq_two, "same seed must replay the same fault pattern"
    print("faults self-check OK")
