"""The CORA FastAPI service: `/auth`, `/chat`, `/handoffs` (task 5.3, REQ-49/REQ-51).

Design sections 7/12. The app is a THIN wiring layer: `create_app()` builds the shared singletons
once (the mock identity service, a `SessionStore`, the `PolicyEngine`, a per-session `ToolLayer`
factory bound to a durable `JsonHandoffStore`, and one `Orchestrator`) and the handlers only move
data between the HTTP boundary and those components. No identity, authorization, policy or
confirmation logic lives here - every decision is made by the component the handler calls (P1).

Security invariants enforced AT the boundary:
- `/chat` derives `customer_id` ONLY from the verified session token (its `jti`/`sub`); the request
  model forbids unknown fields, so a smuggled `customer_id` (or any extra body field) is REJECTED
  with 422 before anything runs - the per-turn `ToolLayer` is bound to the token's session, so no
  body field can widen access (REQ-12).
- Token verification fails CLOSED: an expired / tampered / missing token returns 401 and discloses
  nothing (no customer data, no stack detail).
- The session token is never echoed to any LLM and never logged.
- The `/handoffs` read endpoints are the AGENT console surface, so they require a static agent
  console key in the `X-Agent-Key` header, compared constant-time against `CORA_AGENT_CONSOLE_KEY`.
  A missing/empty/wrong key - AND the fail-closed case where the key is not configured at all -
  returns 401 disclosing nothing (the 401 never distinguishes "no key set" from "wrong key"), so
  the handoff queue is never exposed unauthenticated. This static key is the LOCAL stand-in for a
  real agent principal (AWS Cognito group/role, design section 12 / Phase 8); on AWS a Cognito
  claim replaces it - a documented seam, not JWT/role machinery here.
- A handoff `{id}` that does not exist (for an authorized caller) returns 404 - no leak, no partial
  disclosure.

    # ponytail: no custom HTTP server, no router abstraction - FastAPI's `APIRouter` is the native
    # feature (rung 4) and its `TestClient` gives offline tests with no running server.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from cora.agent.graph import Orchestrator
from cora.agent.llm import get_llm_client
from cora.agent.state import SessionStore
from cora.data.datasource import DataSource, LocalSource
from cora.handoff.store import JsonHandoffStore
from cora.identity import IdentityError, MockIdentityService, Session
from cora.policy import PolicyEngine
from cora.settings import get_settings
from cora.tools import ToolLayer

__all__ = ["create_app"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


# -- request/response models (the HTTP contract; untrusted input is parsed, never trusted) ------


class AuthRequest(BaseModel):
    """Start or complete the OTP flow. Without `otp_code` -> start; with it -> verify.

    A single endpoint with optional fields is the lazy two-step: the first call returns a
    `challenge_id`, the second returns a token (the mock delivers the code in the challenge).
    """

    customer_id: str = Field(min_length=1)
    challenge_id: str | None = None
    otp_code: str | None = None


class AuthStartResponse(BaseModel):
    challenge_id: str
    # The mock IdP "delivers" the one-time code here so the demo/tests can complete the flow; a
    # real IdP would send it out-of-band and never return it.
    otp_code: str


class AuthTokenResponse(BaseModel):
    token: str


class ChatRequest(BaseModel):
    """A chat turn. Identity comes ONLY from the token; the body cannot carry identity.

    `extra="forbid"`: an unknown field (e.g. a smuggled `customer_id`) is REJECTED with 422, not
    silently dropped. Identity is sourced solely from the verified session, so forbidding unknown
    fields makes a trust-boundary injection fail loudly before any orchestrator/tool runs, instead
    of letting a probing or malformed client request look valid (REQ-12).
    """

    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1)
    utterance: str


class ChatResponse(BaseModel):
    response_text: str
    trace_id: str
    outcome: str | None = None
    decision: str | None = None
    rule_id: str | None = None
    handoff_case_id: str | None = None


# -- composition root -----------------------------------------------------------------------------


class _Deps:
    """The shared singletons the handlers wire. Built once per app (not per request)."""

    def __init__(
        self,
        *,
        identity: MockIdentityService,
        store: SessionStore,
        orchestrator: Orchestrator,
        handoff_store: JsonHandoffStore,
        agent_console_key: str,
    ) -> None:
        self.identity = identity
        self.store = store
        self.orchestrator = orchestrator
        self.handoff_store = handoff_store
        # Secret: compared constant-time, never logged, never returned. Empty => fail closed.
        self.agent_console_key = agent_console_key


def create_app(
    *,
    identity: MockIdentityService | None = None,
    source: DataSource | None = None,
    handoff_store: JsonHandoffStore | None = None,
    policy: PolicyEngine | None = None,
    llm=None,  # noqa: ANN001 - an LLMClient; resolved from settings (StubLLMClient under the stub)
    clock: Callable[[], datetime] = _utcnow,
    agent_console_key: str | None = None,
) -> FastAPI:
    """Build the CORA FastAPI app with its handlers wired to the existing components.

    Every collaborator is injectable so tests supply offline doubles (an offset-clock identity
    service, a `LocalSource` over a tiny fixture landing, a tmp-path handoff store, the stub LLM);
    production calls `create_app()` with no args and the components resolve from settings/env. The
    `clock` must be the SAME injectable clock the identity service uses, so the session store's
    liveness check and the token's `exp` agree (production uses wall-clock for both).
    """
    identity = identity or MockIdentityService.from_settings(clock=clock)
    source = source or LocalSource()
    handoff_store = handoff_store or JsonHandoffStore()
    policy = policy or PolicyEngine.from_yaml()
    llm = llm or get_llm_client()
    store = SessionStore(clock=clock)
    # The agent console key resolves from settings in production; tests inject it. An empty key
    # (unset setting or injected "") means the /handoffs endpoints fail CLOSED for every caller.
    agent_console_key = (
        agent_console_key if agent_console_key is not None else get_settings().agent_console_key
    )

    def _tool_layer_factory(session: Session) -> ToolLayer:
        # The ToolLayer is bound to the VERIFIED session, so its `customer_id` is the only identity
        # any tool sees. The durable handoff store is shared so `/handoffs` reads what a turn wrote.
        return ToolLayer(session, source, handoff_store=handoff_store, clock=clock)

    orchestrator = Orchestrator(
        policy=policy, store=store, tool_layer_factory=_tool_layer_factory, llm=llm, clock=clock
    )
    deps = _Deps(
        identity=identity,
        store=store,
        orchestrator=orchestrator,
        handoff_store=handoff_store,
        agent_console_key=agent_console_key,
    )

    app = FastAPI(title="CORA", version="0.1.0")
    app.state.deps = deps

    def get_deps() -> _Deps:
        return app.state.deps

    @app.post("/auth")
    def auth(req: AuthRequest, deps: _Deps = Depends(get_deps)) -> AuthStartResponse | AuthTokenResponse:
        """Two-step OTP: start (no code) returns a challenge; verify (code) returns a token.

        Fails closed to 401 on an OTP failure (unknown/expired challenge, wrong code), disclosing
        nothing. The issued token is returned to the caller but NEVER logged or sent to an LLM.
        """
        if req.otp_code is None:
            challenge = deps.identity.begin_authentication(req.customer_id)
            return AuthStartResponse(challenge_id=challenge.challenge_id, otp_code=challenge.code)
        if not req.challenge_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "challenge_id is required to verify")
        try:
            token = deps.identity.complete_authentication(req.challenge_id, req.otp_code)
        except IdentityError as exc:
            # Fail closed: do not distinguish the OTP failure modes to the caller.
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication failed") from exc
        return AuthTokenResponse(token=token)

    @app.post("/chat")
    def chat(req: ChatRequest, deps: _Deps = Depends(get_deps)) -> ChatResponse:
        """Verify the token (fail closed), run one orchestrator turn, return the customer reply.

        `customer_id` is taken ONLY from the verified session; the request model forbids unknown
        fields, so a smuggled `customer_id` is rejected with 422 before this handler runs (the body
        is untrusted data). The orchestrator masks PII before any LLM call, so the handler adds no
        raw-PII path. On a bad token: 401, disclose nothing.
        """
        try:
            session = deps.identity.verify_token(req.token)
        except IdentityError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid session") from exc

        result = deps.orchestrator.step(session, req.utterance)
        response_text = result.response.text if result.response is not None else ""
        outcome = result.response.outcome.value if result.response is not None else None
        return ChatResponse(
            response_text=response_text,
            trace_id=result.trace_id,
            outcome=outcome,
            decision=result.decision.value if result.decision is not None else None,
            rule_id=result.rule_id,
            handoff_case_id=result.handoff_case_id,
        )

    def _require_agent(deps: _Deps, agent_key: str | None) -> None:
        """Authorize an agent-console caller or FAIL CLOSED with an indistinguishable 401.

        Compares the `X-Agent-Key` header constant-time against the configured key. The empty
        string is never a valid key, so an unset `CORA_AGENT_CONSOLE_KEY` (expected == "") rejects
        every caller - the queue is never exposed unauthenticated. The 401 carries no detail that
        tells "no key configured" apart from "wrong key"; the key is never logged or echoed.
        """
        expected = deps.agent_console_key
        supplied = agent_key or ""
        if not expected or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "agent authorization required")

    @app.get("/handoffs")
    def list_handoffs(
        deps: _Deps = Depends(get_deps),
        x_agent_key: str | None = Header(default=None),
    ) -> dict[str, dict[str, object]]:
        """List every persisted handoff case for the authorized agent console (REQ-16).

        Requires a valid `X-Agent-Key` (fail closed, 401). The store is NOT customer-scoped and
        this is an AGENT-facing surface, so once authorized, listing all open cases is the intended
        behaviour (a human agent triages the queue).
        """
        _require_agent(deps, x_agent_key)
        return deps.handoff_store.list_cases()

    @app.get("/handoffs/{case_id}")
    def get_handoff(
        case_id: str,
        deps: _Deps = Depends(get_deps),
        x_agent_key: str | None = Header(default=None),
    ) -> dict[str, object]:
        """Fetch one handoff package by case id for an authorized agent; else fail closed.

        Authorization is checked BEFORE the lookup, so an unauthorized caller cannot even probe
        which case ids exist (401, never 404). An authorized caller gets 404 for an unknown id.
        """
        _require_agent(deps, x_agent_key)
        package = deps.handoff_store.get(case_id)
        if package is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "handoff case not found")
        return package

    return app
