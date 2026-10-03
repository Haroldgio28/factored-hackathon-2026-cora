"""CORA policy engine (task 2.4, design section 5, REQ-13/REQ-15/REQ-33/REQ-34).

A deterministic, versioned rule table (`rules.yaml`) evaluated top-down, first match wins. The
LLM only proposes; this layer decides (principle P1). Credit eligibility is never evaluated
(REQ-33) and money movement is refused by rule (REQ-34). The matched rule id travels with the
decision for traces and agent explanations (REQ-39).
"""

from __future__ import annotations

from cora.policy.engine import (
    AllowList,
    Decision,
    Intent,
    PolicyDecision,
    PolicyEngine,
    PolicyError,
    PolicyInput,
    ReferencedTransaction,
    Thresholds,
)

__all__ = [
    "AllowList",
    "Decision",
    "Intent",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyError",
    "PolicyInput",
    "ReferencedTransaction",
    "Thresholds",
]
