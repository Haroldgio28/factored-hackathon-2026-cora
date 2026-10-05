"""Observability (task 5.1, REQ-39): local JSONL trace export over the existing TurnResult spans.

Zero infra by design: one JSON record per turn appended to a local file. OpenTelemetry/X-Ray is
the AWS target (design section 12) and sits behind the single `export_turn` sink seam.
"""

from __future__ import annotations

from cora.obs.resilience import BreakerState, CircuitBreaker
from cora.obs.tracing import export_turn

__all__ = ["BreakerState", "CircuitBreaker", "export_turn"]
