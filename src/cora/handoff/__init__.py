"""Handoff package: build the REQ-16 package, persist it durably, and run the E1 dispute intake.

Phase 2 shipped the `create_handoff` tool and its in-memory store seam (`tools/handoff_store.py`).
Phase 4 fills in the rest (task 4.5): `package.build_package` assembles the REQ-16 handoff package
from VERIFIED facts only (never the model), `store.JsonHandoffStore` persists it durably for the
agent console, and `escalation` adds the two cross-turn REQ-15 triggers the policy engine cannot
see in one turn (Very-Negative-sentiment streak, repeated tool failure) plus the E1 dispute-intake
slot-filler (REQ-17) that collects a tool-identified transaction, the reason and card-in-possession
and NEVER promises an outcome.
"""

from __future__ import annotations

from cora.handoff.escalation import (
    TOOL_FAILURE_STREAK,
    VERY_NEGATIVE_STREAK,
    DisputeIntake,
    Sentiment,
    dispute_intake,
    update_turn_signals,
)
from cora.handoff.package import HandoffPackage, Priority, VerifiedFact, build_package
from cora.handoff.store import DEFAULT_HANDOFF_PATH, JsonHandoffStore

__all__ = [
    "DEFAULT_HANDOFF_PATH",
    "TOOL_FAILURE_STREAK",
    "VERY_NEGATIVE_STREAK",
    "DisputeIntake",
    "HandoffPackage",
    "JsonHandoffStore",
    "Priority",
    "Sentiment",
    "VerifiedFact",
    "build_package",
    "dispute_intake",
    "update_turn_signals",
]
