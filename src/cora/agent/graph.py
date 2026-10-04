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
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from cora.agent.confirmation import (
    ConfirmationOutcome,
    request_confirmation,
    resolve_confirmation,
)
from cora.agent.generator import GeneratedResponse, generate
from cora.agent.llm import LLMClient, LLMUnavailable, get_llm_client
from cora.agent.references import Reference, resolve_reference
from cora.agent.state import SessionState, SessionStore
from cora.agent.templates import Outcome, action_verb
from cora.handoff.escalation import DisputeIntake, Sentiment, dispute_intake, update_turn_signals
from cora.handoff.package import HandoffPackage, build_package
from cora.identity import Session
from cora.nlu import entities as nlu_entities
from cora.nlu import injection, language
from cora.obs.resilience import CircuitBreaker
from cora.obs.tracing import export_turn
from cora.policy import (
    Decision,
    Intent,
    PolicyDecision,
    PolicyEngine,
    PolicyInput,
    ReferencedTransaction,
)
from cora.tools import Result, Status, ToolLayer
from cora.tools.base import mask_pii
from cora.tools.confirmations import CardAction
from cora.tools.models import GetBalanceInput, GetCardDetailsInput, HandoffReason

logger = logging.getLogger("cora.agent.graph")

__all__ = ["Node", "Orchestrator", "TraceSpan", "TurnResult", "TurnSignals"]

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


@dataclass(frozen=True)
class TurnSignals:
    """Cross-turn escalation inputs the policy engine cannot see in a single `PolicyInput` (REQ-15).

    `sentiment` is this turn's sentiment label (a classifier output, treated strictly as data);
    `tool_failed` is True when a tool kept failing this turn after its own retries. Both default to
    the neutral/no-failure case so a caller that has neither simply passes nothing.
    """

    sentiment: Sentiment | None = None
    tool_failed: bool = False


@dataclass
class TurnResult:
    """The typed outcome of one `step(...)`: where the machine landed and why.

    `node` is the design-section-4 node reached; `decision`/`rule_id` are the policy outcome that
    routed there (`None` only before the policy runs, e.g. the re-auth path). `message` is a
    short, non-sensitive status string; customer-facing text is produced by the 4.2 generator,
    not here. `spans` is the ordered trace of nodes visited this turn. `trace_id` is a stable
    per-turn id (stdlib uuid4 hex) the JSONL exporter, the handoff package and the UI all carry
    so one turn is explainable end to end (task 5.1, REQ-39); it holds no PII.
    """

    node: str
    trace_id: str = ""
    decision: Decision | None = None
    rule_id: str | None = None
    rules_version: str | None = None
    intent: Intent | None = None
    # Execution-record fields for the REQ-39 trace: the masked input (never raw), the session
    # language this turn ran in, and the intent confidence the policy decided on. Kept on the
    # turn (not the span) because they are turn-level, and PII-safe (`masked_input` is already
    # `mask_pii`-ed before it is set; the exporter masks again as defence in depth).
    masked_input: str | None = None
    language: str | None = None
    intent_confidence: float | None = None
    reference: Reference | None = None
    proposal_rejected: bool = False
    message: str | None = None
    # The rendered customer-facing response (task 4.2). `None` on nodes whose text another
    # subtask fills (e.g. Answer needs the read tool's facts, Confirm the 4.4 restatement).
    response: GeneratedResponse | None = None
    # Set on a Handoff turn (task 4.5): the persisted case id, the built package and - for an E1
    # dispute - the intake slot state. `None` on every non-handoff turn.
    handoff_case_id: str | None = None
    handoff_package: HandoffPackage | None = None
    dispute: DisputeIntake | None = None
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
        breaker: CircuitBreaker | None = None,
        classifier: Callable[[str, str], tuple[Intent | None, float]] | None = None,
        settings=None,  # noqa: ANN001 - a cora.settings.Settings; injected in tests for the trace dir
    ) -> None:
        self._policy = policy
        self._store = store
        self._tool_layer_factory = tool_layer_factory
        self._llm = llm
        self._clock = clock
        # The intent classifier seam (task 5.2/5.3). `_classify` calls this and FAILS CLOSED to
        # `keyword_fallback_intent` on any error. The learned calibrated head is wired here in 5.3;
        # until then the default IS the keyword baseline, so there is already a real production
        # classifier (not the old `(None, 0.0)` stub) and the fallback path has a real caller. The
        # import is LOCAL to break an import cycle (cora.agent package init -> graph -> nlu.fallback
        # -> nlu.baselines -> cora.agent.llm -> cora.agent package init); it stays importable when
        # `nlu.fallback` is the entry module (its own `__main__` self-check).
        from cora.nlu.fallback import keyword_fallback_intent

        self._classifier = classifier if classifier is not None else keyword_fallback_intent
        # The session language for the current turn, set by `step` before `_classify` so the
        # one-arg classifier seam (patched by every Phase-4 test) still works unchanged.
        self._turn_language = "es"
        # One breaker per orchestrator (task 5.2). After repeated LLM-polish failure it opens and
        # `_render` forces template mode WITHOUT attempting the LLM. Best-effort/in-memory: a
        # breaker update NEVER fails a turn (it only selects LLM-vs-template, see `_render`).
        self._breaker = breaker if breaker is not None else CircuitBreaker()
        # The trace exporter reads `settings.trace_dir`; `None` falls back to `get_settings()`
        # inside the exporter (the production default). Injected in tests to redirect the file.
        self._settings = settings

    # -- public entry point ------------------------------------------------------------

    def step(
        self,
        session: Session,
        utterance: str,
        *,
        proposed_action: Decision | None = None,
        turn_signals: TurnSignals | None = None,
    ) -> TurnResult:
        """Run one conversation turn through the machine (design section 4, in order).

        `session` is already VERIFIED by the identity service (fail-closed verification is its
        job); here we only check it is still live against the store's clock. `proposed_action` is
        an optional LLM proposal, validated against the allow-list but never adopted as the
        decision (REQ-13). `turn_signals` carries the cross-turn escalation inputs the policy
        engine cannot see in one turn (Very-Negative sentiment, a tool that keeps failing); they
        feed `update_turn_signals`, which can escalate on a streak (REQ-15, task 4.5).
        """
        # One stable id per turn (stdlib uuid4). Set on EVERY return path below so the trace, the
        # handoff package pointer and the UI all carry the same id (task 5.1, REQ-39). It is a
        # plain opaque hex string and never derived from user text, so it leaks no PII.
        trace_id = uuid.uuid4().hex

        now = self._clock()
        # 1. Expiry / fail-closed re-auth. An expired or absent session discards all state.
        if now >= session.expires_at:
            self._store.discard(session.jti)
            result = self._expired()
            return self._finish(result, trace_id)

        state = self._store.require(session)

        # 1b. Confirm -> Execute -> Verify (task 4.4): while a confirmation is open, THIS turn is
        #     the customer's yes/no. Resolve it deterministically (never the LLM) and execute only
        #     on an explicit affirmative; the policy engine is not consulted for a reply.
        if state.pending_action is not None:
            masked = mask_pii(utterance)
            result = self._resolve_pending(session, state, masked)
            result.masked_input = masked
            result.language = state.language
            return self._finish(result, trace_id)

        # 2. Language: detect on the raw text, but never let a low-confidence guess flip the
        #    session language (keep it; the generator confirms). Empty/low-confidence -> keep.
        lang = language.detect(utterance)
        if not lang.low_confidence:
            state.language = lang.lang

        # 3. Mask BEFORE anything else reads the utterance; screen the masked text for injection.
        masked = mask_pii(utterance)
        injection_hit = injection.screen(masked).injection_hit

        # 4. Classify + extract + resolve reference (deterministic; LLM only proposes entities).
        #    `_classify` keeps its one-arg seam (every Phase-4 test patches `(self, text)`), so the
        #    session language it needs for the keyword fallback is handed over on the instance for
        #    THIS turn only - the machine runs one turn at a time, so there is no cross-turn race.
        self._turn_language = state.language
        intent, confidence = self._classify(masked)
        reference = self._resolve_reference(state, masked)

        # 5. Build the typed PolicyInput from tool-resolved facts (never the model). The OK tool
        #    results are kept so an escalation can build the handoff package from the SAME facts.
        tools = self._tool_layer_factory(session)
        policy_input, tool_results, tool_failed = self._build_policy_input(
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
        result.trace_id = trace_id
        result.intent = intent
        # Turn-level execution-record fields for the REQ-39 trace (masked input only).
        result.masked_input = masked
        result.language = state.language
        result.intent_confidence = confidence
        result.reference = reference
        result.proposal_rejected = decision.proposal_rejected

        # A CONFIRM decision opens the Confirm->Execute->Verify protocol (task 4.4): restate the
        # exact action + MASKED product number and open the pending confirmation. The restatement
        # text and the pending both come from facts resolved above (never the model).
        if decision.decision is Decision.CONFIRM and intent is not None:
            product_id = reference.product_id or state.referenced_product_id
            if product_id is not None:
                self._open_confirmation(state, result, tools, intent, product_id)

        # A tool read came back UNAVAILABLE (after `_safe_read`'s bounded retries): the tool
        # outcome is in DOUBT, so FAIL CLOSED (REQ-40, security steering P4). This is distinct from
        # a not-owned read (FORBIDDEN/NOT_FOUND is a definite deny the policy already handles): the
        # policy, seeing only `resource_owned=False`, routed to a generic abstain/clarify, but the
        # honest thing to say is the es/pt TOOL_UNAVAILABLE copy - never guess a value, never claim
        # the resource is not owned. We only override the CUSTOMER-FACING text; the policy decision,
        # rule id and trace are unchanged. An escalation/handoff/confirm/answer turn is left alone
        # (it already discloses nothing or has grounded facts); the streak below drives escalation.
        if tool_failed and result.node not in ("Escalate", "Handoff", "Confirm", "Answer"):
            result.response = self._render(state, Outcome.TOOL_UNAVAILABLE)

        # Record the (masked) turn BEFORE any handoff so the package's verbatim request is this
        # turn's text; reset the clarification counter on any non-clarify outcome.
        state.record_turn(masked, intent.value if intent else None, decision.decision.value)
        if decision.decision is not Decision.CLARIFY:
            state.clarification_count = 0

        # 7. Cross-turn escalation (REQ-15, task 4.5): fold this turn's sentiment / tool-failure
        #    signal into the streak counters. A streak escalates even when the policy did not. A
        #    tool read that failed THIS turn (`tool_failed`, detected automatically from a
        #    Status.UNAVAILABLE read after retries) feeds the streak on its own - the caller does
        #    not have to pass `tool_failed=True`; a real repeated tool outage escalates by itself.
        signals = turn_signals or TurnSignals()
        streak_reason = update_turn_signals(
            state, sentiment=signals.sentiment, tool_failed=signals.tool_failed or tool_failed
        )
        if result.node not in ("Escalate", "Handoff") and streak_reason is not None:
            result = _node_handoff(self, state, decision)
            result.trace_id = trace_id
            result.intent = intent
            result.masked_input = masked
            result.language = state.language
            result.intent_confidence = confidence
            result.message = f"{streak_reason.value}: escalation streak"

        # 8. On any escalation (policy or streak), build + persist the REQ-16 handoff package from
        #    the verified facts gathered this turn and run the E1 dispute intake (REQ-17). The
        #    turn `trace_id` is threaded into the package (as `trace_ref`) so a human agent can
        #    pull the trace. The package is persisted HERE, synchronously (its pre-5.1 timing):
        #    persisting a handoff does not depend on the trace file having been written - the
        #    trace is a best-effort audit artifact, not a precondition of the handoff.
        if result.node == "Handoff":
            self._handoff(session, state, result, tools, intent, reference, decision, tool_results)
        return self._finish(result, trace_id)

    def _finish(self, result: TurnResult, trace_id: str) -> TurnResult:
        """Emit the one REQ-39 trace record (best-effort), then return the real turn result.

        Every return path of `step` funnels through here, so a real turn emits exactly one JSONL
        record on the happy path - not only the test calls. The export is BEST-EFFORT observability
        (security steering P5, "everything is a record"): a trace is an audit artifact, never a
        gate on an already-verified turn. A telemetry sink must never crash or alter a turn, so a
        write failure (an `OSError`, or any broad `Exception`) is logged and SWALLOWED and the
        real, unmodified `result` is returned - the verified card action still happened, the
        confirmation prompt was still delivered, and the `trace_id` stays on the result (the id is
        valid and correlates the handoff/UI even if this one JSONL append failed).
        """
        result.trace_id = trace_id
        try:
            export_turn(result, settings=self._settings)
        except Exception:  # noqa: BLE001 - a best-effort telemetry sink must never fail a turn
            logger.warning("trace export failed for turn %s; continuing", trace_id, exc_info=True)
        return result

    # -- understand helpers (deterministic) --------------------------------------------

    def _classify(self, masked_utterance: str) -> tuple[Intent | None, float]:
        """Classify the masked utterance into an intent + confidence (task 5.2 fallback wired).

        Calls the injected `classifier` (the learned calibrated head is wired in 5.3; until then
        the injected default IS the keyword baseline, so there is a real production classifier).
        On ANY classifier failure - a missing/corrupt artifact, a load error, an inference error -
        the turn FAILS CLOSED to `keyword_fallback_intent` (REQ-40, design section 10): the
        deterministic `KeywordBaseline` at the conservative rules.yaml stand-in band. So a
        classifier outage still yields a classified-or-safely-clarified turn, never a crash and
        never a guessed answer. The session language for this turn is read from `_turn_language`
        (set by `step` before this call) so the keyword table uses the right es/pt rules.
        """
        from cora.nlu.fallback import keyword_fallback_intent  # local: break the import cycle

        language = getattr(self, "_turn_language", "es")
        try:
            return self._classifier(masked_utterance, language)
        except Exception:  # noqa: BLE001 - classifier unavailable: fail closed to the baseline
            logger.warning("intent classifier unavailable; using keyword fallback", exc_info=True)
            return keyword_fallback_intent(masked_utterance, language)

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
    ) -> tuple[PolicyInput, list[Result], bool]:
        """Resolve every policy flag deterministically via the ToolLayer (never the model).

        Ownership/state/fraud are facts, so they come from the session-bound tool layer: the
        `customer_id` is the verified session's, so neither user text nor the model can widen
        access. A referential phrase that resolved to nothing is surfaced as `ambiguous_entity`
        so POL-070 clarifies.

        Also returns the OK tool `Result`s fetched this turn so an escalation can build the
        handoff package from the SAME verified facts (task 4.5) without re-querying - the facts a
        human sees are exactly the ones the policy decided on - and a `tool_failed` flag that is
        True when a read this turn came back `Status.UNAVAILABLE` (the tool outcome is in DOUBT,
        after `_safe_read`'s bounded retries). `tool_failed` is DISTINCT from not-owned
        (FORBIDDEN/NOT_FOUND): a not-owned read is a definite answer (deny), a failed read is doubt,
        so the orchestrator renders the honest tool-unavailable message and feeds the escalation
        streak only for the latter (REQ-40, security steering P4).
        """
        product_id = reference.product_id or state.referenced_product_id
        resource_owned = False
        product_owned = False
        state_allows = False
        tool_failed = False
        referenced_txn: ReferencedTransaction | None = None
        tool_results: list[Result] = []

        if intent in _READ_INTENTS and product_id is not None:
            balance = _safe_read(lambda: tools.get_balance(GetBalanceInput(product_id=product_id)))
            resource_owned = balance.status is Status.OK
            tool_failed = tool_failed or balance.status is Status.UNAVAILABLE
            if resource_owned:
                tool_results.append(balance)
        if intent in _ACTION_INTENTS and product_id is not None:
            details = _safe_read(lambda: tools.get_card_details(GetCardDetailsInput(product_id=product_id)))
            tool_failed = tool_failed or details.status is Status.UNAVAILABLE
            if details.status is Status.OK and details.data is not None:
                product_owned = True
                state_allows = details.data.product_status in _ACTIONABLE_CARD_STATUSES
                tool_results.append(details)

        policy_input = PolicyInput(
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
        return policy_input, tool_results, tool_failed

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

        Task 5.2 adds the circuit breaker as a routing switch on top of this already-safe fallback:
        while the breaker is OPEN, polish is forced off and the LLM is NOT attempted (short-circuit
        straight to the template). While closed/half-open the LLM is attempted through a thin
        breaker-observing wrapper that records a success/failure so repeated outages open the
        breaker. The breaker only SELECTS the path; `generate` is unchanged and still fails closed.
        """
        # GUARDRAIL (task 5.2): the breaker INSPECTION is best-effort too, not just its writes. A
        # breaker (or its injected clock) that raises on `allow()` must never fail the turn - it only
        # SELECTS LLM-vs-template. If inspection raises, log it and fall through to the normal
        # generator path (which itself fails closed to the template on `LLMUnavailable`), so a broken
        # breaker degrades to "attempt the LLM" rather than crashing a verified turn. (A persist/
        # journal/transactional breaker would be the WRONG fix here - the breaker is advisory.)
        try:
            breaker_open = not self._breaker.allow()
        except Exception:  # noqa: BLE001 - breaker inspection is advisory; never fail a turn on it
            logger.warning("circuit-breaker allow() failed; continuing via generator", exc_info=True)
            breaker_open = False
        if breaker_open:
            # Breaker open: skip the LLM entirely, render the grounded template (REQ-40).
            return generate(outcome, state.language, fields=fields, polish=False, client=self._llm)
        return generate(
            outcome, state.language, fields=fields, client=_BreakerClient(self._llm, self._breaker)
        )

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
        """`Decide -> Confirm` (POL-080): the restatement + pending are opened by `step` (4.4)."""
        return _from_decision("Confirm", decision)

    # -- Confirm -> Execute -> Verify protocol (task 4.4, REQ-09/REQ-14) ----------------

    def _open_confirmation(
        self,
        state: SessionState,
        result: TurnResult,
        tools: ToolLayer,
        intent: Intent,
        product_id: str,
    ) -> None:
        """Open the pending confirmation and render the restatement (exact action + masked number).

        The masked number comes from the card-details tool (never the model), so the restatement
        shows the customer the same `****1234` the overlay will act on. If the tool cannot supply
        it the turn fails closed to the tool-unavailable copy rather than restating a blank card.
        """
        action = CardAction.FREEZE if intent is Intent.A1 else CardAction.UNFREEZE
        # Idempotent read -> same bounded-retry fail-closed boundary as every other read (task 5.2,
        # REQ-40). A transient 5xx/timeout on THIS read becomes `Status.UNAVAILABLE` rather than
        # escaping `step()`, so the turn renders the honest tool-unavailable copy below and no
        # pending confirmation is opened (never restate a blank/guessed card).
        details = _safe_read(lambda: tools.get_card_details(GetCardDetailsInput(product_id=product_id)))
        masked_number = details.data.product_number_masked if details.data is not None else None
        if details.status is not Status.OK or not masked_number:
            result.response = self._render(state, Outcome.TOOL_UNAVAILABLE)
            return
        request_confirmation(state, action, product_id, masked_number)
        result.response = self._render(
            state,
            Outcome.CONFIRM,
            action=action_verb(action, state.language),
            product_number_masked=masked_number,
        )

    def _resolve_pending(self, session: Session, state: SessionState, masked: str) -> TurnResult:
        """Execute+verify an open confirmation on the customer's reply (task 4.4, REQ-09/REQ-14).

        Deterministic end to end: `resolve_confirmation` classifies the yes/no (never the LLM),
        executes only on an explicit affirmative, and reports success only from the tool's verified
        OK read-back. Negative/ambiguous -> cancelled (nothing executed); a non-verifying tool ->
        not completed + escalate (offer a human). The pending is single-use (cleared on resolve).
        """
        tools = self._tool_layer_factory(session)
        outcome, _result = resolve_confirmation(state, masked, tools, now=self._clock())
        if outcome is ConfirmationOutcome.EXECUTED_OK:
            node = "Verify"
            response = self._render(state, Outcome.ACTION_DONE)
        elif outcome is ConfirmationOutcome.CANCELLED:
            node = "Verify"
            response = self._render(state, Outcome.ACTION_CANCELLED)
        else:  # NOT_COMPLETED: the tool did not verify -> offer a human (REQ-09)
            node = "Escalate"
            response = self._render(state, Outcome.ESCALATE)
        turn = TurnResult(
            node=node,
            message=f"confirmation resolved: {outcome.name}",
            response=response,
            spans=[TraceSpan(node="Execute", detail=outcome.name), TraceSpan(node=node)],
        )
        state.record_turn(masked, None, None)
        return turn

    def _node_escalate(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide/Verify/Clarify -> Escalate`: build the handoff package next (4.5)."""
        return _node_handoff(self, state, decision)

    # -- Handoff package + dispute intake (task 4.5, REQ-15/REQ-16/REQ-17) --------------

    def _handoff(
        self,
        session: Session,
        state: SessionState,
        result: TurnResult,
        tools: ToolLayer,
        intent: Intent | None,
        reference: Reference,
        decision: PolicyDecision,
        tool_results: list[Result],
    ) -> None:
        """Build the REQ-16 package from verified facts, run the E1 intake, and persist it (4.5).

        The package is assembled by `handoff.build_package` from the SAME OK tool results the
        policy decided on (never the model); priority is `high` on a fraud escalation (POL-040).
        For an E1 dispute the slot-filler records the tool-identified transaction (never free
        text), the reason and card-in-possession, and offers a freeze when fraud is suspected -
        but the customer-facing copy stays the handoff acknowledgement, so no outcome is promised
        (REQ-17). The full package is written to the durable handoff store for the agent console.

        The turn's `trace_id` is threaded in as the package `trace_ref` so a human agent can pull
        the trace; it is just the id string and does not require the trace file to exist (the trace
        is a best-effort audit artifact, written separately in `_finish`).
        """
        reason = _handoff_reason(intent, decision)
        unresolved: list[str] = []
        if intent is Intent.E1:
            # Fraud is "suspected" when the policy escalated on a fraud signal (POL-040, priority
            # high) - a deterministic read of the decision, not a model judgement.
            fraud_suspected = decision.priority == "high"
            dispute = dispute_intake(
                verified_transaction_id=reference.transaction_id or state.referenced_transaction_id,
                reason=state.dispute_reason,
                card_in_possession=state.card_in_possession,
                fraud_suspected=fraud_suspected,
            )
            result.dispute = dispute
            unresolved = list(dispute.open_slots)
            # Offer the freeze as a follow-up action note in the package; the customer-facing text
            # is still only the handoff ack (no refund/outcome ever promised, REQ-17).
            actions = ["offer_freeze(A1)"] if dispute.offer_freeze else []
        else:
            actions = []

        package = build_package(
            state,
            tool_results,
            reason,
            priority=decision.priority,
            unresolved_questions=unresolved,
            actions_taken=actions,
            trace_ref=result.trace_id,
            now=self._clock(),
        )
        case_id = tools.handoff_store.create(package.to_store_dict())
        result.handoff_package = package
        result.handoff_case_id = case_id
        result.spans.append(TraceSpan(node="Handoff", decision=reason.value, detail=case_id))

    def _node_abstain(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide -> Abstain` (POL-010/030/999): disclose nothing / route out of scope (4.6).

        The decision distinguishes a plain abstain (disclose nothing, offer a human) from an
        `abstain_route` (credit out of scope, POL-030/REQ-33 -> route to a human); the outcome
        maps 1:1 so the right es/pt copy is rendered. On an `abstain_route` the routing reason
        (e.g. `CREDIT_OUT_OF_SCOPE`) is logged and recorded on the trace span, so the credit
        guardrail is explainable from the rule id + reason and never from model reasoning
        (REQ-33, REQ-39). The engine computes nothing about credit eligibility.
        """
        result = _from_decision("Abstain", decision, detail=decision.route)
        if decision.decision is Decision.ABSTAIN_ROUTE and decision.route is not None:
            logger.info("policy %s -> abstain_route reason=%s", decision.rule_id, decision.route)
        result.response = self._render(state, Outcome.from_decision(decision.decision))
        return result

    def _node_refuse(self, state: SessionState, decision: PolicyDecision) -> TurnResult:
        """`Decide -> Abstain` for a refuse decision (POL-020 money movement, 4.6)."""
        result = _from_decision("Abstain", decision)
        result.response = self._render(state, Outcome.REFUSE)
        return result


# Bounded retry for an IDEMPOTENT read tool (task 5.2, REQ-40, design section 10). A read is
# side-effect-free, so a transient remote error (S3/httpfs throttle or 5xx) is safe to re-attempt:
# at most TWO retries (three total attempts) with exponential backoff + jitter, exactly the Bedrock
# policy in `agent/llm.py` (same pinned `tenacity`, no new dependency, no second retry library).
# The bounded WALL-CLOCK TIMEOUT lives at the transport edge, NOT here: the remote `DataSource`
# read goes through DuckDB `httpfs`, whose `http_timeout` (set on the `S3Source` connection, with
# `http_retries=0` so it does not double-retry under this policy) bounds a hung HTTP operation, so
# a stuck socket surfaces here as a RAISED timeout - which this retry then re-attempts and finally
# fails closed on. (The Bedrock client bounds its own Converse call with botocore's
# `read_timeout`/`connect_timeout` + disabled boto retries, the same single-retry-policy idea.) We do NOT
# wrap the read in a worker-thread watchdog: a synchronous DuckDB read is bound to the connection's
# thread, so running it off-thread would break DuckDB's single-thread-per-connection contract (the
# wrong fix). Non-idempotent operations (the confirmed card action) are NOT routed through here;
# they keep their exactly-once confirm/read-back protocol.
_READ_MAX_ATTEMPTS = 3


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(_READ_MAX_ATTEMPTS),
    wait=wait_exponential_jitter(initial=0.1, max=2.0),
    reraise=True,
)
def _read_with_retry(read: Callable[[], Result]) -> Result:
    """Run one idempotent read; a raised transient/timeout error is retried by the decorator.

    A read that raises (a transient remote error or a transport-level timeout) raises out of here,
    so `tenacity` re-attempts up to `_READ_MAX_ATTEMPTS` with exponential backoff + jitter. The
    final failure propagates to `_safe_read`, which fails closed.
    """
    return read()


def _safe_read(read: Callable[[], Result]) -> Result:
    """Run an idempotent read with bounded retries, failing closed on exhaustion (task 5.2, REQ-40).

    The tool layer already returns a non-OK `Result` for the errors it can see (invalid/not-found/
    forbidden, overlay I/O). But a read backed by a remote `DataSource` (S3/httpfs on AWS) can RAISE
    on a timeout or 5xx, and that would otherwise crash the turn. This is a genuine fail-closed path
    (a tool outcome is in DOUBT, security steering P1/P4 - distinct from best-effort observability):
    after at most two retries a still-raising read becomes `Status.UNAVAILABLE`, which the
    orchestrator routes to the honest tool-unavailable message + escalation. The turn discloses
    NOTHING and offers a human; no figure can leak from a failure.
    """
    try:
        return _read_with_retry(read)
    except Exception as exc:  # noqa: BLE001 - any tool/transport/timeout error fails closed uniformly
        logger.warning("read tool failed after retries, failing closed: %s", type(exc).__name__)
        return Result(status=Status.UNAVAILABLE, message="tool unavailable")


class _BreakerClient:
    """Wraps the orchestrator's LLM client so each `complete` call feeds the circuit breaker.

    A successful `complete` records a success (closing a half-open breaker); an `LLMUnavailable`
    records a failure (opening the breaker on a streak) and is re-raised so `generate` still fails
    closed to the grounded template. GUARDRAIL (task 5.2): the breaker update is best-effort - it
    is wrapped so a bookkeeping error can NEVER raise into the turn; the real LLM result (or the
    real `LLMUnavailable`) always propagates unchanged, so the breaker only ever SELECTS the path.
    """

    def __init__(self, inner: LLMClient | None, breaker: CircuitBreaker) -> None:
        # `generate` resolves the default client when passed None, but it is passed THIS wrapper,
        # so resolve the real client here when the orchestrator has none (stub under CORA_NLU_STUB).
        self._inner = inner if inner is not None else get_llm_client()
        self._breaker = breaker

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        try:
            reply = self._inner.complete(system=system, user=user, max_tokens=max_tokens)
        except LLMUnavailable:
            self._record(self._breaker.record_failure)
            raise
        self._record(self._breaker.record_success)
        return reply

    @staticmethod
    def _record(update: Callable[[], None]) -> None:
        """Run a breaker state update, swallowing any error (best-effort; never fails the turn)."""
        try:
            update()
        except Exception:  # noqa: BLE001 - a best-effort breaker update must never fail a turn
            logger.warning("circuit-breaker state update failed; continuing", exc_info=True)


# -- module-level node helpers (keep the Orchestrator surface small) -----------------------


def _node_handoff(orch: Orchestrator, state: SessionState, decision: PolicyDecision) -> TurnResult:
    """`Escalate -> Handoff -> [*]`: assemble the package and end the conversation (4.5)."""
    result = _from_decision("Escalate", decision)
    result.spans.append(TraceSpan(node="Handoff", decision=decision.decision.value, rule_id=decision.rule_id))
    result.node = "Handoff"
    result.response = orch._render(state, Outcome.ESCALATE)
    return result


# Map the escalating intent to the handoff reason code recorded in the package (REQ-16). The
# escalation intents E1-E4 map 1:1 onto `HandoffReason`; any other escalation path (a fraud signal
# on a non-E intent, a clarify/verify/streak escalation) has no dispute/complaint/fraud intent, so
# it is recorded as a human-request handoff (E4) - the catch-all "a human must take over" reason.
_INTENT_HANDOFF_REASON: dict[Intent, HandoffReason] = {
    Intent.E1: HandoffReason.DISPUTE,
    Intent.E2: HandoffReason.COMPLAINT,
    Intent.E3: HandoffReason.FRAUD,
    Intent.E4: HandoffReason.HUMAN_REQUEST,
}


def _handoff_reason(intent: Intent | None, decision: PolicyDecision) -> HandoffReason:
    """The REQ-16 reason code for this escalation: the E-intent if any, else fraud or human-request."""
    if intent is not None and intent in _INTENT_HANDOFF_REASON:
        return _INTENT_HANDOFF_REASON[intent]
    # A fraud signal (POL-040) escalates with priority high even without an E3 intent -> FRAUD.
    if decision.priority == "high":
        return HandoffReason.FRAUD
    return HandoffReason.HUMAN_REQUEST


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
