"""The conversation orchestrator: design section 4 as a thin stdlib state machine (task 4.1).

Design section 4 draws a small, fully deterministic state machine. ADR-004 adopts LangGraph for
it; the Phase-4 plan records the deviation (see `documentation/DECISIONS.md`): LangGraph is not
installed on this RAM-constrained host and none of its runtime value-adds (checkpointer,
streaming, concurrent branches, a tool-calling loop) is needed, because every transition here is
ALREADY code - `PolicyEngine.decide(...)` picks the edge, never the model. So the machine is a
one-module dispatcher: one function per node plus a `_EDGES: {Decision: node}` table. The
node/edge structure mirrors section 4 one-to-one, so swapping in LangGraph later is mechanical.

The non-negotiable invariants (security steering, P1) are threaded through `step()` in order:

1. Expire-check the session against the store's clock; an expired/absent session discards all
   state and routes to re-auth (fail closed, `Expired -> Unauthenticated`).
2. Detect the language (keep the session language on low confidence; never guess).
3. `mask_pii` the utterance BEFORE anything else sees it, and screen it for injection - both on
   the masked text. Nothing after this point reads the raw utterance.
4. Classify intent, extract candidate entities, resolve any cross-turn reference DETERMINISTICALLY
   from session state (`references.resolve_reference`), never from the model.
5. Build a TYPED `PolicyInput`: ownership/state/fraud flags come from the `ToolLayer` (the
   verified session's `customer_id`), the LLM proposal (if any) is passed through for the
   allow-list check but is NEVER adopted as the decision.
6. `PolicyEngine.decide(...)` returns the `Decision`; `_EDGES` dispatches to the matching node.

The LLM is consulted for a proposal only; a proposal outside the allow-list is rejected by the
engine (`proposal_rejected`) and the decision is taken from the rules regardless (REQ-13).

REQ-07: when the decision is `clarify`, the Clarify node bumps `clarification_count`; the SECOND
consecutive failed clarification escalates (E4) instead of asking again. A non-clarify turn
resets the counter.

Node bodies for the generation (4.2), confirm/execute/verify (4.4), handoff (4.5) and
credit/money-movement rendering (4.6) subtasks are thin, honest stubs here: they set the node
name and the decision/rule on the `TurnResult` and leave the user-facing text to those subtasks.
Every node records a trace span (design section 4 "every node writes a trace span").
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from cora.agent.generator import GeneratedResponse, generate
from cora.agent.llm import LLMClient
from cora.agent.references import Reference, resolve_reference
from cora.agent.state import SessionState, SessionStore
from cora.agent.templates import Outcome
from cora.identity import Session
from cora.nlu import entities as nlu_entities
from cora.nlu import injection, language
from cora.policy import (
    Decision,
    Intent,
    PolicyDecision,
    PolicyEngine,
    PolicyInput,
    ReferencedTransaction,
)
from cora.tools import Status, ToolLayer
from cora.tools.base import mask_pii
from cora.tools.models import GetBalanceInput, GetCardDetailsInput

logger = logging.getLogger("cora.agent.graph")

__all__ = ["Node", "Orchestrator", "TraceSpan", "TurnResult"]

# Intent groups the orchestrator uses to decide which ownership flag the policy needs. Mirrors
# the policy engine's own groupings (kept here so the orchestrator asks the ToolLayer only the
# question the decision needs - no speculative lookups).
_READ_INTENTS = frozenset({Intent.I1, Intent.I2, Intent.I3, Intent.I4, Intent.I5, Intent.I6})
_ACTION_INTENTS = frozenset({Intent.A1, Intent.A2})

# Card statuses an action may be applied to (design POL-080 "state allows"). A freeze/unfreeze is
# only offered for a card that is currently in a normal servicing state; a closed card cannot be
# frozen. The overlay projects Active/Blocked; the base product_status carries the rest.
_ACTIONABLE_CARD_STATUSES = frozenset({"Active", "Blocked", "Suspended"})

# The second consecutive failed clarification escalates instead of asking again (REQ-07, E4).
_MAX_CLARIFICATIONS = 2


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class TraceSpan:
    """One node's trace span (design section 4 "every node writes a trace span").

    Deliberately tiny: the node name, the decision/rule that routed into it, and the policy
    rules version. The full OpenTelemetry->JSONL exporter is a later observability subtask; this
    is the structured record each node leaves behind so a turn is explainable from traces and
    rule ids, never from model reasoning (REQ-39). Carries no raw PII.
    """

    node: str
    decision: str | None = None
    rule_id: str | None = None
    rules_version: str | None = None
    detail: str | None = None


@dataclass
class TurnResult:
    """The typed outcome of one `step(...)`: where the machine landed and why.

    `node` is the design-section-4 node reached; `decision`/`rule_id` are the policy outcome that
    routed there (`None` only before the policy runs, e.g. the re-auth path). `message` is a
    short, non-sensitive status string; customer-facing text is produced by the 4.2 generator,
    not here. `spans` is the ordered trace of nodes visited this turn.
    """

    node: str
    decision: Decision | None = None
    rule_id: str | None = None
    rules_version: str | None = None
    intent: Intent | None = None
    reference: Reference | None = None
    proposal_rejected: bool = False
    message: str | None = None
    # The rendered customer-facing response (task 4.2). `None` on nodes whose text another
    # subtask fills (e.g. Answer needs the read tool's facts, Confirm the 4.4 restatement).
    response: GeneratedResponse | None = None
    spans: list[TraceSpan] = field(default_factory=list)


# A node takes the orchestrator, the live session state and the policy decision, and returns the
# turn result. Keeping the signature uniform is what makes `_EDGES` a plain dispatch table.
Node = Callable[["Orchestrator", SessionState, PolicyDecision], TurnResult]


class Orchestrator:
    """The design-section-4 machine as a deterministic dispatcher over the policy decision.

    Constructed with the policy engine and the per-session store; a `ToolLayer` is built per turn
    from the VERIFIED session (its `customer_id` is the only identity any tool ever sees). The
    `tool_layer_factory` seam lets tests inject an in-memory tool layer with no DataSource/AWS.
    """

    def __init__(
        self,
        *,
        policy: PolicyEngine,
        store: SessionStore,
        tool_layer_factory: Callable[[Session], ToolLayer],
        llm: LLMClient | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._policy = policy
        self._store = store
        self._tool_layer_factory = tool_layer_factory
        self._llm = llm
        self._clock = clock

    # -- public entry point ------------------------------------------------------------

    def step(
        self, session: Session, utterance: str, *, proposed_action: Decision | None = None
    ) -> TurnResult:
        """Run one conversation turn through the machine (design section 4, in order).

        `session` is already VERIFIED by the identity service (fail-closed verification is its
        job); here we only check it is still live against the store's clock. `proposed_action` is
        an optional LLM proposal, validated against the allow-list but never adopted as the
        decision (REQ-13).
        """
        now = self._clock()
        # 1. Expiry / fail-closed re-auth. An expired or absent session discards all state.
        if now >= session.expires_at:
            self._store.discard(session.jti)
            return self._expired()

        state = self._store.require(session)

        # 2. Language: detect on the raw text, but never let a low-confidence guess flip the
        #    session language (keep it; the generator confirms). Empty/low-confidence -> keep.
        lang = language.detect(utterance)
        if not lang.low_confidence:
            state.language = lang.lang

        # 3. Mask BEFORE anything else reads the utterance; screen the masked text for injection.
        masked = mask_pii(utterance)
        injection_hit = injection.screen(masked).injection_hit

        # 4. Classify + extract + resolve reference (deterministic; LLM only proposes entities).
        intent, confidence = self._classify(masked)
        reference = self._resolve_reference(state, masked)

        # 5. Build the typed PolicyInput from tool-resolved facts (never the model).
        tools = self._tool_layer_factory(session)
        policy_input = self._build_policy_input(
            state=state,
            tools=tools,
            intent=intent,
            confidence=confidence,
            injection_hit=injection_hit,
            reference=reference,
            proposed_action=proposed_action,
        )

        # 6. Decide + dispatch. The decision is the policy's, never the model's.
        decision = self._policy.decide(policy_input)
        result = _EDGES[decision.decision](self, state, decision)
        result.intent = intent
        result.reference = reference
        result.proposal_rejected = decision.proposal_rejected

        # Record the (masked) turn and reset the clarification counter on any non-clarify outcome.
        state.record_turn(masked, intent.value if intent else None, decision.decision.value)
        if decision.decision is not Decision.CLARIFY:
            state.clarification_count = 0
        return result

    # -- understand helpers (deterministic) --------------------------------------------

    def _classify(self, masked_utterance: str) -> tuple[Intent | None, float]:
        """Classify the masked utterance into an intent + confidence.

        4.1 wires the seam; the trained classifier is loaded by a later subtask/offline artifact.
        Absent a model the turn is low-confidence `OTHER`, which routes to clarify/escalate by
        policy - the safe default, never a guessed answer.
        """
        return None, 0.0

    def _resolve_reference(self, state: SessionState, masked_utterance: str) -> Reference:
        """Extract candidate entities (LLM proposes) then resolve the reference from state (code)."""
        extracted = nlu_entities.extract_entities(masked_utterance, client=self._llm)
        product_ref = extracted.entities.product_ref if extracted.entities is not None else None
        return resolve_reference(state, product_ref)

    def _build_policy_input(
        self,
        *,
        state: SessionState,
        tools: ToolLayer,
        intent: Intent | None,
        confidence: float,
        injection_hit: bool,
        reference: Reference,
        proposed_action: Decision | None,
    ) -> PolicyInput:
        """Resolve every policy flag deterministically via the ToolLayer (never the model).

        Ownership/state/fraud are facts, so they come from the session-bound tool layer: the
        `customer_id` is the verified session's, so neither user text nor the model can widen
        access. A referential phrase that resolved to nothing is surfaced as `ambiguous_entity`
        so POL-070 clarifies.
        """
        product_id = reference.product_id or state.referenced_product_id
        resource_owned = False
        product_owned = False
        state_allows = False
        referenced_txn: ReferencedTransaction | None = None

        if intent in _READ_INTENTS and product_id is not None:
            resource_owned = self._owns_product(tools, product_id)
        if intent in _ACTION_INTENTS and product_id is not None:
            product_owned, state_allows = self._owns_actionable_card(tools, product_id)

        return PolicyInput(
            session_valid=state.authenticated,
            intent=intent,
            intent_confidence=confidence,
            injection_hit=injection_hit,
            referenced_txn=referenced_txn,
            ambiguous_entity=reference.ambiguous,
            product_owned=product_owned,
            state_allows=state_allows,
            resource_owned=resource_owned,
            proposed_action=proposed_action,
        )

    @staticmethod
    def _owns_product(tools: ToolLayer, product_id: str) -> bool:
        """True iff the product belongs to the session customer (fail closed on any non-OK)."""
        return tools.get_balance(GetBalanceInput(product_id=product_id)).status is Status.OK

    @staticmethod
    def _owns_actionable_card(tools: ToolLayer, product_id: str) -> tuple[bool, bool]:
        """`(owned, state_allows)` for a card action: owned via the tool, state via its status."""
        result = tools.get_card_details(GetCardDetailsInput(product_id=product_id))
        if result.status is not Status.OK or result.data is None:
            return False, False
        return True, result.data.product_status in _ACTIONABLE_CARD_STATUSES

    # -- nodes (design section 4). Each records a trace span. ---------------------------
    #
    # The Answer/Confirm/Execute/Verify/Escalate/Handoff/Abstain bodies are the 4.1 skeleton:
    # they land the machine in the right node with the policy decision attached and leave the
    # user-facing rendering / tool execution to subtasks 4.2/4.4/4.5/4.6. Each is a real node in
    # the graph now, so those subtasks fill a body rather than add a state.

    def _expired(self) -> TurnResult:
        """`Expired -> Unauthenticated`: state already discarded; force re-auth, disclose nothing."""
        return _single("Unauthenticated", detail="session expired; state discarded, re-auth required")

    def _render(self, state: SessionState, outcome: Outcome, **fields: str) -> GeneratedResponse:
        """Render the customer-facing response for `outcome` in the session language (task 4.2).

        Deterministic template first, LLM polish via the orchestrator's client when configured;
        fails closed to the template on any LLM error (REQ-40). Figures come only from `fields`,
        which the node fills from tool facts - the model never originates a value.
        """
        return generate(outcome, state.language, fields=fields, client=self._llm)

    def _node_reauth(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """POL-000: the session is not valid for this turn -> re-authenticate (fail closed)."""
        state.clear_pending_confirmation()
        result = _from_decision("Unauthenticated", decision)
        # Re-auth copy carries nothing sensitive and must not be reworded into a disclosure, so
        # skip the LLM and return the deterministic template verbatim.
        result.response = generate(Outcome.REAUTH, state.language, polish=False)
        return result

    def _node_answer(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide -> Answer` (POL-090): render from tool facts (4.2). Returns to Authenticated.

        The grounded `facts` string is assembled from the read tool's `Result` by the read
        subtask; until it is wired the node lands with no `response` (never a guessed answer).
        """
        return _from_decision("Answer", decision)

    def _node_clarify(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Understand -> Clarify` (POL-070), with the REQ-07 two-strikes escalation guard.

        A clarify decision bumps the consecutive-failure counter; the SECOND one escalates (E4)
        through the same Escalate->Handoff path instead of asking a third time
        (`Clarify -> Escalate: 2 failed clarifications`).
        """
        state.clarification_count += 1
        if state.clarification_count >= _MAX_CLARIFICATIONS:
            state.clarification_count = 0
            result = _node_handoff(self, state, decision)
            result.decision = Decision.ESCALATE
            result.message = "E4: 2 failed clarifications"
            result.response = self._render(state, Outcome.ESCALATE)
            return result
        result = _from_decision("Clarify", decision)
        result.response = self._render(state, Outcome.CLARIFY)
        return result

    def _node_confirm(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide -> Confirm` (POL-080): restate + await explicit yes (protocol in 4.4)."""
        return _from_decision("Confirm", decision)

    def _node_escalate(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide/Verify/Clarify -> Escalate`: build the handoff package next (4.5)."""
        return _node_handoff(self, state, decision)

    def _node_abstain(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide -> Abstain` (POL-010/030/999): disclose nothing / route out of scope (4.6).

        The decision distinguishes a plain abstain (disclose nothing, offer a human) from an
        `abstain_route` (credit out of scope, POL-030/REQ-33 -> route to a human); the outcome
        maps 1:1 so the right es/pt copy is rendered.
        """
        result = _from_decision("Abstain", decision)
        result.response = self._render(state, Outcome.from_decision(decision.decision))
        return result

    def _node_refuse(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide -> Abstain` for a refuse decision (POL-020 money movement, 4.6)."""
        result = _from_decision("Abstain", decision)
        result.response = self._render(state, Outcome.REFUSE)
        return result


# -- module-level node helpers (keep the Orchestrator surface small) -----------------------


def _node_handoff(orch: Orchestrator, state: SessionState, decision: PolicyDecision) -> TurnResult:
    """`Escalate -> Handoff -> [*]`: assemble the package and end the conversation (4.5)."""
    result = _from_decision("Escalate", decision)
    result.spans.append(TraceSpan(node="Handoff", decision=decision.decision.value, rule_id=decision.rule_id))
    result.node = "Handoff"
    result.response = orch._render(state, Outcome.ESCALATE)
    return result


def _single(node: str, *, detail: str | None = None) -> TurnResult:
    """A node reached without a policy decision (the re-auth-on-expiry path)."""
    return TurnResult(node=node, message=detail, spans=[TraceSpan(node=node, detail=detail)])


def _from_decision(node: str, decision: PolicyDecision, *, detail: str | None = None) -> TurnResult:
    """Build a `TurnResult` for `node` carrying the policy decision + a trace span."""
    span = TraceSpan(
        node=node,
        decision=decision.decision.value,
        rule_id=decision.rule_id,
        rules_version=decision.rules_version,
        detail=detail,
    )
    return TurnResult(
        node=node,
        decision=decision.decision,
        rule_id=decision.rule_id,
        rules_version=decision.rules_version,
        message=detail,
        spans=[span],
    )


# The design-section-4 edge table: each policy Decision routes to exactly one node. This is the
# whole state machine's transition function - the LLM never picks a key here.
_EDGES: dict[Decision, Node] = {
    Decision.REAUTH: Orchestrator._node_reauth,
    Decision.ANSWER: Orchestrator._node_answer,
    Decision.CLARIFY: Orchestrator._node_clarify,
    Decision.CONFIRM: Orchestrator._node_confirm,
    Decision.ESCALATE: Orchestrator._node_escalate,
    Decision.ABSTAIN: Orchestrator._node_abstain,
    Decision.ABSTAIN_ROUTE: Orchestrator._node_abstain,
    Decision.REFUSE: Orchestrator._node_refuse,
}

# Fail closed if a Decision ever has no edge (keeps the table and the enum in lockstep).
assert set(_EDGES) == set(Decision), "every policy Decision must map to exactly one node"
