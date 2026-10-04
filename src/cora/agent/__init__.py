"""Agent package: the LLM seam (Phase 3) plus the Phase-4 conversation orchestrator.

The LLM proposes; deterministic code decides (principle P1). Phase 3 added `llm.py` (the one
`LLMClient` seam) and its callers treat the reply as an untrusted suggestion. Phase 4 adds the
design-section-4 state machine as a thin stdlib dispatcher (`graph.Orchestrator`): the policy
engine picks every edge, the model never does. `SessionState`/`SessionStore` hold the REQ-06
per-session memory and `resolve_reference` resolves cross-turn references deterministically.
"""

from __future__ import annotations

from cora.agent.graph import Node, Orchestrator, TraceSpan, TurnResult
from cora.agent.llm import (
    BedrockLLMClient,
    LLMClient,
    LLMUnavailable,
    StubLLMClient,
    get_llm_client,
)
from cora.agent.references import Reference, ReferenceKind, resolve_reference
from cora.agent.state import SessionState, SessionStore, TurnRecord

__all__ = [
    "BedrockLLMClient",
    "LLMClient",
    "LLMUnavailable",
    "Node",
    "Orchestrator",
    "Reference",
    "ReferenceKind",
    "SessionState",
    "SessionStore",
    "StubLLMClient",
    "TraceSpan",
    "TurnRecord",
    "TurnResult",
    "get_llm_client",
    "resolve_reference",
]
