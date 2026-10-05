"""Fault-injection tests for task 5.2 (REQ-40): retries, timeouts, circuit breaker, fallbacks.

These are REAL fault injections, not smoke tests: each drives a turn through the orchestrator with
a deliberately broken dependency and asserts the FALLBACK ACTUALLY ENGAGED (a real response still
comes back, the right path ran) and that nothing is disclosed on a failure.

- LLM repeatedly fails -> breaker opens -> template mode, a grounded response STILL comes back, and
  while open the LLM is NOT attempted (short-circuit). After a cooldown the breaker half-opens and a
  healthy LLM closes it again.
- Classifier unavailable -> `keyword_fallback_intent` (KeywordBaseline) still classifies the turn.
- A read tool times out / 5xx -> honest es/pt message + escalation (handoff), disclosing nothing.
- GUARDRAIL: a breaker-state write that raises never fails the turn (telemetry/bookkeeping is
  best-effort and only SELECTS the path; it never gates a turn outcome).

Offline under `CORA_NLU_STUB=1` (autouse conftest). No Bedrock, no network. Reuses the harness
patterns from `test_agent_graph.py` (`_Clock`, `_FakeToolLayer`, `_issue_session`, `_patch_intent`).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cora.agent.graph import _READ_MAX_ATTEMPTS, Orchestrator
from cora.agent.llm import LLMUnavailable, StubLLMClient
from cora.agent.state import SessionStore
from cora.agent.templates import Outcome, render_template
from cora.identity import MockIdentityService, Session
from cora.nlu.fallback import MATCH_CONFIDENCE, keyword_fallback_intent
from cora.nlu.labels import Intent
from cora.obs.resilience import BreakerState, CircuitBreaker
from cora.policy import Decision, PolicyEngine, Thresholds
from cora.tools import InMemoryHandoffStore, Result, Status
from cora.tools.models import BalanceData, CardDetailsData

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CUST-000123"
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)

# Figures a tool-down turn must NEVER disclose (they would only exist if a tool had answered).
_SENSITIVE_FIGURES = ("100", "1000", "1234", "USD")


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _FakeToolLayer:
    """In-memory tool layer: owns `PRD-1`, answers the two ownership reads 4.1/5.2 need."""

    def __init__(self, owned: set[str]) -> None:
        self._owned = owned
        self.handoff_store = InMemoryHandoffStore()

    def get_balance(self, tool_input):  # noqa: ANN001 - duck-typed test double
        if tool_input.product_id in self._owned:
            return Result[BalanceData](
                status=Status.OK,
                data=BalanceData(
                    product_id=tool_input.product_id,
                    currency="USD",
                    current_balance=100.0,
                    credit_limit=None,
                    available_credit=None,
                ),
            )
        return Result[BalanceData](status=Status.FORBIDDEN, message="not authorized")

    def get_card_details(self, tool_input):  # noqa: ANN001 - duck-typed test double
        if tool_input.product_id in self._owned:
            return Result[CardDetailsData](
                status=Status.OK,
                data=CardDetailsData(
                    product_id=tool_input.product_id,
                    product_type="Tarjeta Crédito",
                    product_number_masked="****1234",
                    currency="USD",
                    product_status="Active",
                    credit_limit=1000.0,
                    days_past_due=0,
                    expiration_date=None,
                ),
            )
        return Result[CardDetailsData](status=Status.FORBIDDEN, message="not authorized")


class _TimingOutToolLayer(_FakeToolLayer):
    """A tool layer whose reads RAISE, simulating a remote DataSource timeout / 5xx (S3/Athena)."""

    def get_balance(self, tool_input):  # noqa: ANN001 - duck-typed test double
        raise TimeoutError("read timed out")

    def get_card_details(self, tool_input):  # noqa: ANN001 - duck-typed test double
        raise TimeoutError("read timed out")


class _AlwaysDownLLM:
    """An LLM client that always fails - every polish attempt raises `LLMUnavailable`."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.calls += 1
        raise LLMUnavailable("bedrock down")


def _issue_session(clock: _Clock) -> tuple[MockIdentityService, Session]:
    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)
    challenge = service.begin_authentication(_CUSTOMER)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service, service.verify_token(token)


def _orchestrator(
    clock: _Clock,
    *,
    llm,  # noqa: ANN001 - an LLMClient-shaped double
    breaker: CircuitBreaker | None = None,
    tools: _FakeToolLayer | None = None,
) -> Orchestrator:
    fake_tools = tools if tools is not None else _FakeToolLayer({"PRD-1"})
    return Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=lambda _session: fake_tools,
        llm=llm,
        clock=clock,
        breaker=breaker,
    )


def _patch_intent(monkeypatch: pytest.MonkeyPatch, intent: Intent, confidence: float) -> None:
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (intent, confidence))


# -- 1. breaker opens after repeated LLM failures -> template mode ---------------------


def test_breaker_opens_after_repeated_failures_and_forces_template_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    down = _AlwaysDownLLM()
    breaker = CircuitBreaker(failure_threshold=3, cooldown=timedelta(seconds=30), clock=clock)
    orch = _orchestrator(clock, llm=down, breaker=breaker)
    # X2 (money movement) renders REFUSE via the LLM polish path on EVERY turn and never trips the
    # clarify counter, so repeated failures accumulate on the breaker.
    _patch_intent(monkeypatch, Intent.X2, 0.99)

    # Drive failure_threshold turns: each attempts the LLM polish (and fails) while closed.
    for _ in range(3):
        result = orch.step(session, "transfiere dinero")
        assert result.decision is Decision.REFUSE
        assert result.response is not None and not result.response.polished
        assert result.response.text.strip()  # a real grounded template still came back
    assert breaker.state is BreakerState.OPEN

    # Next turn: breaker open -> short-circuit to the template WITHOUT attempting the LLM polish.
    # The orchestrator also calls the LLM once per turn for entity extraction (not breaker-gated),
    # so compare the DELTA: an open breaker adds exactly 0 polish calls (delta == 1, extraction
    # only) rather than 2 (extraction + polish).
    calls_before = down.calls
    result = orch.step(session, "transfiere dinero")
    assert down.calls - calls_before == 1, "breaker open: the LLM polish must NOT be attempted"
    assert result.response is not None and not result.response.polished
    assert result.response.text.strip(), "a real template response still comes back"


# -- 2. breaker half-opens after cooldown and closes on success ------------------------


def test_breaker_half_opens_after_cooldown_and_closes_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    # A healthy stub that rephrases faithfully (no figure, no language flip) so polish is KEPT.
    healthy = StubLLMClient(default="No puedo ayudarte con eso, gracias.")
    breaker = CircuitBreaker(failure_threshold=1, cooldown=timedelta(seconds=30), clock=clock)
    breaker.record_failure()  # force it open at t0
    assert breaker.state is BreakerState.OPEN
    orch = _orchestrator(clock, llm=healthy, breaker=breaker)
    _patch_intent(monkeypatch, Intent.X2, 0.99)

    clock.now += timedelta(seconds=31)  # cooldown elapsed -> next allow() half-opens
    result = orch.step(session, "transfiere dinero")
    assert result.response is not None and result.response.polished, "half-open trial attempts the LLM"
    assert breaker.state is BreakerState.CLOSED, "a successful trial closes the breaker"


# -- 3. classifier unavailable -> KeywordBaseline fallback classifies the turn ---------


def test_classifier_unavailable_uses_keyword_baseline() -> None:
    # A known es action utterance still classifies (A1) at the conservative band.
    es_intent, es_conf = keyword_fallback_intent("quiero congelar mi tarjeta", "es")
    assert es_intent is Intent.A1 and es_conf == MATCH_CONFIDENCE
    # A known pt escalation utterance still classifies (E4 - human request).
    pt_intent, pt_conf = keyword_fallback_intent("quero falar com uma pessoa", "pt")
    assert pt_intent is Intent.E4 and pt_conf == MATCH_CONFIDENCE
    # An unmatched utterance -> OTHER at 0.0 (routes to a safe clarify, never a guessed answer).
    other_intent, other_conf = keyword_fallback_intent("asdf qwer zxcv", "es")
    assert other_intent is Intent.OTHER and other_conf == 0.0


def test_production_classify_falls_back_to_keyword_when_classifier_raises() -> None:
    # Inject a classifier dependency that ALWAYS raises (a missing/corrupt artifact or an inference
    # error). The PRODUCTION `_classify` path must catch it and fall back to `keyword_fallback_intent`
    # on its own - no monkeypatch of `_classify`, so this proves the real except-path, not a stub.
    def _broken_classifier(masked: str, language: str):  # noqa: ANN202
        raise RuntimeError("model artifact missing")

    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=lambda _session: _FakeToolLayer({"PRD-1"}),
        llm=StubLLMClient(),
        clock=clock,
        classifier=_broken_classifier,
    )

    # A known es complaint utterance still classifies (E2) via the keyword baseline at the
    # conservative band, so the turn is classified despite the classifier failure.
    result = orch.step(session, "quiero poner una queja")
    assert result.intent is Intent.E2, "production _classify fell back to the keyword baseline"
    assert result.intent_confidence == MATCH_CONFIDENCE
    # A pt utterance uses the pt keyword rules via the session language threaded by `step`.
    state = orch._store.require(session)  # noqa: SLF001
    state.language = "pt"
    pt_result = orch.step(session, "quero falar com uma pessoa")
    assert pt_result.intent is Intent.E4 and pt_result.intent_confidence == MATCH_CONFIDENCE


def test_production_classify_default_is_the_keyword_baseline() -> None:
    # With NO injected classifier the default IS the keyword baseline (the learned head is wired in
    # 5.3): a real production classifier, not the old `(None, 0.0)` stub. A known utterance classifies.
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    orch = _orchestrator(clock, llm=StubLLMClient())
    result = orch.step(session, "quiero congelar mi tarjeta")
    assert result.intent is Intent.A1, "default classifier classified the turn (not None)"
    assert result.intent_confidence == MATCH_CONFIDENCE


# -- 4. tool timeout/5xx -> honest es/pt escalation, nothing disclosed -----------------


@pytest.mark.parametrize(
    ("language", "utterance"),
    [("es", "cuanto dinero tengo disponible"), ("pt", "quanto dinheiro tenho disponível")],
)
def test_tool_timeout_renders_honest_unavailable_message_and_discloses_nothing(
    monkeypatch: pytest.MonkeyPatch, language: str, utterance: str
) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    orch = _orchestrator(clock, llm=StubLLMClient(), tools=_TimingOutToolLayer({"PRD-1"}))
    _patch_intent(monkeypatch, Intent.I1, 0.99)  # a read intent -> hits the (timing-out) read tool

    state = orch._store.require(session)  # noqa: SLF001
    state.language = language
    state.referenced_product_id = "PRD-1"

    result = orch.step(session, utterance)
    # Fail closed: the read raised -> UNAVAILABLE after retries -> the honest TOOL_UNAVAILABLE copy
    # is rendered (NOT a generic abstain, NOT an answer). The exact es/pt template text is selected.
    assert result.decision is not Decision.ANSWER
    assert result.node != "Answer"
    assert result.response is not None and result.response.outcome is Outcome.TOOL_UNAVAILABLE
    assert result.response.lang == language, "the honest message is in the session language (es/pt)"
    assert result.response.text == render_template(Outcome.TOOL_UNAVAILABLE, language), (
        "the exact bilingual tool-unavailable template is rendered"
    )
    for figure in _SENSITIVE_FIGURES:
        assert figure not in result.response.text, f"disclosed {figure!r} on a tool failure"


def test_injected_tool_failures_escalate_to_handoff_without_manual_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    orch = _orchestrator(clock, llm=StubLLMClient(), tools=_TimingOutToolLayer({"PRD-1"}))
    _patch_intent(monkeypatch, Intent.I1, 0.99)
    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"

    # The INJECTED tool failure feeds the escalation streak automatically - the caller passes NO
    # `tool_failed=True`. Turn 1 renders the honest tool-unavailable copy (streak below threshold);
    # the second consecutive injected failure reaches the TOOL_FAILURE_STREAK and escalates to a
    # human (REQ-15), building a handoff package that still discloses no figure.
    first = orch.step(session, "cual es mi saldo")
    assert first.node != "Handoff", "a single tool failure does not escalate yet"
    assert first.response is not None and first.response.outcome is Outcome.TOOL_UNAVAILABLE

    second = orch.step(session, "cual es mi saldo")
    assert second.node == "Handoff", "the injected tool-failure streak escalated on its own"
    assert second.handoff_case_id is not None
    for figure in _SENSITIVE_FIGURES:
        assert second.response is None or figure not in second.response.text


def test_tool_retry_recovers_within_bounded_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    # A read that fails transiently twice then succeeds must RECOVER within the bounded attempts
    # (3 total = 2 retries), so the turn answers and no fallback engages. Proves the retry policy
    # actually re-attempts rather than failing on the first error.
    from cora.tools.models import BalanceData

    attempts = {"n": 0}

    class _FlakyToolLayer(_FakeToolLayer):
        def get_balance(self, tool_input):  # noqa: ANN001, ANN202
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("transient 5xx")
            return Result[BalanceData](
                status=Status.OK,
                data=BalanceData(
                    product_id=tool_input.product_id,
                    currency="USD",
                    current_balance=100.0,
                    credit_limit=None,
                    available_credit=None,
                ),
            )

    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    orch = _orchestrator(clock, llm=StubLLMClient(), tools=_FlakyToolLayer({"PRD-1"}))
    _patch_intent(monkeypatch, Intent.I1, 0.99)
    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"

    result = orch.step(session, "cual es mi saldo")
    assert attempts["n"] == 3, "the read was retried up to the bounded attempt count"
    # The read eventually succeeded, so the turn is NOT a tool-unavailable fallback.
    assert result.response is None or result.response.outcome is not Outcome.TOOL_UNAVAILABLE


def test_tool_read_timeout_fails_closed_after_bounded_retries() -> None:
    # A remote read whose transport TIMES OUT surfaces as a raised timeout; after the bounded
    # attempts (3 total = 2 retries) it fails closed to UNAVAILABLE rather than crashing the turn.
    # The injected backoff is near-zero so the test does not sleep in real time.
    from cora.agent import graph as graph_mod
    from cora.agent.graph import _safe_read

    attempts = {"n": 0}

    def _always_times_out() -> Result:
        attempts["n"] += 1
        raise TimeoutError("remote read timed out")  # the shape boto/httpfs raises on read_timeout

    result = _safe_read(_always_times_out)
    assert result.status is Status.UNAVAILABLE, "a timed-out read fails closed"
    assert attempts["n"] == graph_mod._READ_MAX_ATTEMPTS, "the read was retried up to the bound"


def test_s3source_connect_configures_a_bounded_http_timeout_at_the_transport_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 1: the bounded wall-clock timeout must be CONFIGURED by the PRODUCTION transport setup
    # (`S3Source._connect`), not merely mirrored on a throwaway connection. Invoke the real
    # `_connect` with the credential seam stubbed (no SSO, no network) and read back the settings
    # DuckDB actually stored, so this test would FAIL if the production `SET http_timeout` /
    # `SET http_retries=0` statements were removed - proving a hung remote op is genuinely bounded
    # and that tenacity in `_safe_read` is the single retry owner (httpfs's own retries are off).
    import boto3

    from cora.data.datasource import S3Source

    # Stub the credential seam so `_connect` does NOT resolve a real SSO profile or touch the
    # network: a session whose `get_credentials()` returns None (the no-credentials branch). This
    # exercises the production profile/credential path deterministically and offline.
    class _NoCredsSession:
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def get_credentials(self):  # noqa: ANN202
            return None

    monkeypatch.setattr(boto3, "Session", _NoCredsSession)

    src = S3Source(bucket="example-bucket", region="us-east-1", profile=None, http_timeout_ms=15_000)
    con = src._connect()  # noqa: SLF001 - exercising the real production transport setup
    try:
        timeout = con.execute("SELECT current_setting('http_timeout')").fetchone()[0]
        retries = con.execute("SELECT current_setting('http_retries')").fetchone()[0]
    finally:
        con.close()
    assert int(timeout) == 15_000, "`_connect` configured a bounded http_timeout on the httpfs transport"
    assert int(retries) == 0, "`_connect` disabled httpfs's own retries so tenacity is the single policy"


def test_confirmation_card_read_timeout_fails_closed_and_opens_no_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 2: a freeze (A1) reads the card TWICE - once in `_build_policy_input` to establish
    # ownership (so the policy reaches CONFIRM) and again in `_open_confirmation` to restate the
    # masked number. A transient timeout on the SECOND read must fail closed through the same
    # bounded-retry boundary: render TOOL_UNAVAILABLE, exhaust the retries, open NO pending
    # confirmation, disclose nothing - and NOT escape step(). The double must therefore SUCCEED on
    # the first (ownership) read and raise only on the subsequent (confirmation) read, otherwise the
    # policy never reaches CONFIRM and the confirmation-read fix is never exercised.
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)

    reads = {"n": 0}

    class _CardReadTimesOutOnConfirmation(_FakeToolLayer):
        """Owns PRD-1: the first card read (ownership) succeeds; every later read (confirmation,
        with its bounded retries) times out, so the fault lands squarely on the confirmation read."""

        def get_card_details(self, tool_input):  # noqa: ANN001 - duck-typed test double
            reads["n"] += 1
            if reads["n"] == 1:
                return super().get_card_details(tool_input)  # ownership read -> OK -> CONFIRM
            raise TimeoutError("card details read timed out")  # confirmation read fails closed

    orch = _orchestrator(clock, llm=StubLLMClient(), tools=_CardReadTimesOutOnConfirmation({"PRD-1"}))
    _patch_intent(monkeypatch, Intent.A1, 0.99)  # freeze -> CONFIRM -> second card-details read
    state = orch._store.require(session)  # noqa: SLF001
    state.referenced_product_id = "PRD-1"

    result = orch.step(session, "quiero congelar mi tarjeta")
    # The confirm path WAS reached: the ownership read succeeded (read #1) and the policy decided
    # CONFIRM, then the confirmation read was attempted and retried up to the bound before failing.
    assert result.decision is Decision.CONFIRM, "the ownership read succeeded -> policy reached CONFIRM"
    assert reads["n"] == 1 + _READ_MAX_ATTEMPTS, (
        "ownership read (1) + the confirmation read exhausted its bounded retries"
    )
    # The confirmation read failed closed: honest tool-unavailable copy, no pending opened, no leak.
    assert result.response is not None and result.response.outcome is Outcome.TOOL_UNAVAILABLE
    assert state.pending_action is None, "a failed confirmation read must NOT open a pending confirmation"
    for figure in _SENSITIVE_FIGURES:
        assert figure not in result.response.text, f"disclosed {figure!r} on a confirmation read failure"


# -- 5. GUARDRAIL: a breaker-state write failure never fails the turn ------------------


def test_breaker_state_write_failure_never_fails_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    breaker = CircuitBreaker(clock=clock)

    def _boom() -> None:
        raise RuntimeError("breaker store unreachable")

    # Both state updates raise: the turn must still return its real result unchanged (the breaker
    # update is best-effort bookkeeping, it only SELECTS a path and never gates a turn outcome).
    monkeypatch.setattr(breaker, "record_success", _boom)
    monkeypatch.setattr(breaker, "record_failure", _boom)
    orch = _orchestrator(clock, llm=StubLLMClient(default="No puedo ayudarte con eso."), breaker=breaker)
    _patch_intent(monkeypatch, Intent.X2, 0.99)

    result = orch.step(session, "transfiere dinero")
    assert result.decision is Decision.REFUSE, "the real turn outcome is returned despite the breaker error"
    assert result.response is not None and result.response.text.strip()


def test_breaker_allow_failure_never_fails_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    # Finding 3: breaker INSPECTION is advisory, not just its writes. A breaker whose `allow()`
    # raises (a broken clock/state read, or a custom implementation that throws) must not escape
    # into the turn: _render logs it best-effort and continues through the normal generator path,
    # returning the same deterministic policy outcome and a safe response.
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    _service, session = _issue_session(clock)
    breaker = CircuitBreaker(clock=clock)

    def _boom() -> bool:
        raise RuntimeError("breaker state read unreachable")

    monkeypatch.setattr(breaker, "allow", _boom)
    orch = _orchestrator(clock, llm=StubLLMClient(default="No puedo ayudarte con eso."), breaker=breaker)
    _patch_intent(monkeypatch, Intent.X2, 0.99)

    result = orch.step(session, "transfiere dinero")
    assert result.decision is Decision.REFUSE, "the deterministic policy outcome is unchanged"
    assert result.response is not None and result.response.text.strip(), "a safe response still comes back"
