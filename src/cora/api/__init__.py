"""CORA service surface (task 5.3, REQ-49/REQ-51).

A thin FastAPI app that WIRES the existing components - the mock identity service, the Phase-4
`Orchestrator` and the durable handoff store - behind `/auth`, `/chat` and `/handoffs`. The
handlers hold NO business logic: identity, authorization, policy and confirmation all stay in the
components the handlers call (principle P1, "the LLM proposes, deterministic code decides"). See
`cora.api.app.create_app`.
"""

from __future__ import annotations

from cora.api.app import create_app

__all__ = ["create_app"]
