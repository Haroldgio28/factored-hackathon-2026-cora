"""Tests for task 7.1: the load-test driver's aggregation logic (REQ-50/51).

Only the non-trivial pure logic is pinned - the nearest-rank `percentile` and the `summarize`
rollup (error rate, throughput, p50/p95 over SUCCESSFUL turns). The HTTP driver and the thread
pool are I/O against a running server; they are not unit-tested here (nothing to assert offline).
"""

from __future__ import annotations

import pytest

from scripts.load_test import TurnResult, percentile, summarize


def test_percentile_nearest_rank_returns_observed_values() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    # Nearest-rank must return an OBSERVED latency, never an interpolated one.
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 50.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 50.0


def test_percentile_single_value() -> None:
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summarize_error_rate_and_throughput() -> None:
    results = [
        TurnResult(ok=True, auth_ms=10.0, chat_ms=100.0),
        TurnResult(ok=True, auth_ms=20.0, chat_ms=200.0),
        TurnResult(ok=False, auth_ms=0.0, chat_ms=0.0, error="boom"),
    ]
    summary = summarize(results, wall_seconds=2.0)
    assert summary.total == 3
    assert summary.errors == 1
    assert summary.error_rate == pytest.approx(1 / 3)
    assert summary.throughput_rps == pytest.approx(1.5)
    # p50/p95 are computed over the SUCCESSFUL turns only, so the failed turn's zeros don't skew.
    assert summary.chat_p50_ms == 100.0
    assert summary.chat_p95_ms == 200.0
    assert summary.auth_p50_ms == 10.0


def test_summarize_all_errors_does_not_divide_by_zero() -> None:
    results = [TurnResult(ok=False, auth_ms=0.0, chat_ms=0.0, error="x")]
    summary = summarize(results, wall_seconds=0.0)
    assert summary.error_rate == 1.0
    assert summary.throughput_rps == 0.0  # wall_seconds == 0 -> guarded, no ZeroDivisionError
