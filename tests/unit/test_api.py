"""Tests for task 5.3: the FastAPI service `/auth`, `/chat`, `/handoffs` (REQ-49, REQ-51).

Everything runs OFFLINE through FastAPI's `TestClient` (no running server, no network, no
Bedrock): `CORA_NLU_STUB=1` is forced by the suite `conftest`, so the classifier uses the
deterministic keyword baseline and the LLM is `StubLLMClient`. The app is built with injected
offline doubles - an offset-clock `MockIdentityService`, a `LocalSource` over a tiny one-card
Parquet landing, and a tmp-path `JsonHandoffStore` - so `customer_id` is injected exactly as in
production and the handoff store the handlers read is the one a turn writes to.

The security contract is asserted at the boundary: a valid token runs a turn and returns a trace
id; a tampered / expired / missing token is refused with 401 and discloses nothing; a smuggled
`customer_id` in the body is REJECTED (422) at the trust boundary, so no turn runs under a forged
identity (identity comes only from the verified token); `/handoffs` lists persisted cases and 404s
a bogus id; and a known utterance classifies to a real intent (proving `_classify` is wired, not
the old `(None, 0.0)` stub). Overlapping concurrent turns on the same token in different languages
each classify in their own language (per-session turn isolation, no cross-request bleed on the
shared orchestrator).
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cora.agent import StubLLMClient
from cora.agent.graph import Orchestrator
from cora.agent.state import SessionStore
from cora.api.app import create_app
from cora.data.datasource import LocalSource
from cora.handoff.store import JsonHandoffStore
from cora.identity import MockIdentityService
from cora.policy import Intent, PolicyEngine, Thresholds

_KEY = "test-signing-key-not-a-real-secret"
_AGENT_KEY = "test-agent-console-key-not-a-real-secret"
_CUSTOMER = "CLI-OWNER0000001"
_OTHER_CUSTOMER = "CLI-OTHER0000002"
_CARD = "PRD-OWNERCARD01"
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)
_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _build_landing(root: Path) -> None:
    products = pd.DataFrame(
        [
            {
                "product_id": _CARD,
                "customer_id": _CUSTOMER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            }
        ]
    )
    root.mkdir(parents=True, exist_ok=True)
    products.astype("string").to_parquet(root / "products.parquet", index=False)


@pytest.fixture
def source(tmp_path: Path) -> LocalSource:
    root = tmp_path / "raw_parquet"
    _build_landing(root)
    return LocalSource(root=root)


@pytest.fixture
def clock() -> _Clock:
    return _Clock(_NOW)


@pytest.fixture
def identity(clock: _Clock) -> MockIdentityService:
    return MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)


@pytest.fixture
def handoff_store(tmp_path: Path) -> JsonHandoffStore:
    return JsonHandoffStore(path=tmp_path / "handoffs.json")


@pytest.fixture
def client(
    source: LocalSource,
    identity: MockIdentityService,
    handoff_store: JsonHandoffStore,
    clock: _Clock,
) -> TestClient:
    app = create_app(
        identity=identity,
        source=source,
        handoff_store=handoff_store,
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        llm=StubLLMClient(),
        clock=clock,
        agent_console_key=_AGENT_KEY,
    )
    return TestClient(app)


def _authenticate(client: TestClient, customer_id: str = _CUSTOMER) -> str:
    """Drive the two-step OTP flow and return the issued session token."""
    start = client.post("/auth", json={"customer_id": customer_id})
    assert start.status_code == 200, start.text
    body = start.json()
    verify = client.post(
        "/auth",
        json={
            "customer_id": customer_id,
            "challenge_id": body["challenge_id"],
            "otp_code": body["otp_code"],
        },
    )
    assert verify.status_code == 200, verify.text
    return verify.json()["token"]


# -- /auth -----------------------------------------------------------------------------


def test_auth_issues_a_token(client: TestClient) -> None:
    token = _authenticate(client)
    assert isinstance(token, str) and token


def test_auth_wrong_code_is_refused_disclosing_nothing(client: TestClient) -> None:
    start = client.post("/auth", json={"customer_id": _CUSTOMER}).json()
    bad = client.post(
        "/auth",
        json={"customer_id": _CUSTOMER, "challenge_id": start["challenge_id"], "otp_code": "000000"},
    )
    assert bad.status_code == 401
    assert "token" not in bad.json()


# -- /chat (valid turn) ----------------------------------------------------------------


def test_chat_with_valid_token_runs_a_turn_and_returns_trace_id(client: TestClient) -> None:
    token = _authenticate(client)
    # Seed the referenced card so the A1 freeze utterance reaches a real, rendered Confirm turn
    # (the orchestrator resolves the product from session state, never from the body).
    session = client.app.state.deps.identity.verify_token(token)
    client.app.state.deps.store.require(session).referenced_product_id = _CARD
    resp = client.post("/chat", json={"token": token, "utterance": "quiero congelar mi tarjeta"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["response_text"]  # non-empty customer-facing reply
    assert len(body["trace_id"]) == 32  # stable uuid4 hex per turn (REQ-39)
    assert body["outcome"] is not None


# -- /chat fail-closed on a bad token --------------------------------------------------


def test_chat_refuses_tampered_token(client: TestClient) -> None:
    token = _authenticate(client)
    # Flip the FIRST character of the signature segment (header.payload.signature). That byte
    # carries meaningful signed bits, so the change deterministically breaks the HS256 signature -
    # unlike flipping the final base64url char, whose unused low padding bits can decode to the
    # same signature bytes and let a "tampered" token still verify (a flaky security test).
    header, payload, signature = token.split(".")
    flipped = ("B" if signature[0] != "B" else "C") + signature[1:]
    tampered = f"{header}.{payload}.{flipped}"
    assert tampered != token
    resp = client.post("/chat", json={"token": tampered, "utterance": "quiero mi saldo"})
    assert resp.status_code == 401
    _assert_discloses_nothing(resp)


def test_chat_refuses_expired_token(
    source: LocalSource, handoff_store: JsonHandoffStore, clock: _Clock
) -> None:
    # Issue with a short TTL, then advance the shared clock past expiry before the chat turn.
    identity = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)
    app = create_app(
        identity=identity,
        source=source,
        handoff_store=handoff_store,
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        llm=StubLLMClient(),
        clock=clock,
    )
    client = TestClient(app)
    token = _authenticate(client)
    clock.now = _NOW + timedelta(minutes=16)  # past the 15-minute absolute TTL
    resp = client.post("/chat", json={"token": token, "utterance": "quiero mi saldo"})
    assert resp.status_code == 401
    _assert_discloses_nothing(resp)


def test_chat_refuses_missing_token(client: TestClient) -> None:
    # Three "no usable token" shapes, all fail closed with no disclosure:
    #  - the token field OMITTED entirely -> 422 (required field), never reaches the orchestrator;
    #  - an empty token -> 422 (model min_length), never reaches the orchestrator;
    #  - a syntactically-present junk token -> 401 (verification fails closed).
    omitted = client.post("/chat", json={"utterance": "hola"})
    assert omitted.status_code == 422
    _assert_discloses_nothing(omitted)
    empty = client.post("/chat", json={"token": "", "utterance": "hola"})
    assert empty.status_code == 422
    junk = client.post("/chat", json={"token": "not-a-jwt", "utterance": "hola"})
    assert junk.status_code == 401
    _assert_discloses_nothing(junk)


def _assert_discloses_nothing(resp) -> None:  # noqa: ANN001 - httpx.Response from TestClient
    """A refused turn leaks no customer data and no stack detail."""
    text = resp.text.lower()
    assert _CARD.lower() not in text
    assert "952.03" not in text  # the fixture balance
    assert "traceback" not in text
    assert "4717188030863827" not in text  # the fixture PAN


# -- customer_id cannot be injected via the body ---------------------------------------


def test_customer_id_cannot_be_injected_via_body(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `customer_id` in the body is REJECTED (422), not silently accepted: it is an unknown field.

    `ChatRequest` forbids unknown fields, so a body smuggling `_OTHER_CUSTOMER` fails validation at
    the trust boundary (422) and the orchestrator is NEVER invoked - a probing/malformed request
    cannot look valid. The refusal discloses no customer data. (Identity is only ever sourced from
    the verified token; the body has no identity field at all.)
    """
    # Trip-wire: if validation ever let a smuggled field through, `_classify` would run and this
    # would fire. The 422 must happen BEFORE any orchestrator work.
    called = False

    def _tripwire(self: Orchestrator, text: str, language: str):  # noqa: ANN202 - test seam
        nonlocal called
        called = True
        return (Intent.A1, 0.99)

    monkeypatch.setattr(Orchestrator, "_classify", _tripwire)
    token = _authenticate(client, customer_id=_CUSTOMER)
    session = client.app.state.deps.identity.verify_token(token)
    client.app.state.deps.store.require(session).referenced_product_id = _CARD  # noqa: SLF001

    resp = client.post(
        "/chat",
        json={
            "token": token,
            "utterance": "quiero congelar mi tarjeta",
            "customer_id": _OTHER_CUSTOMER,  # smuggled unknown field: must be refused
        },
    )
    assert resp.status_code == 422, resp.text
    assert not called  # the orchestrator (and so the tool layer) never ran
    _assert_discloses_nothing(resp)


# -- /handoffs -------------------------------------------------------------------------


def test_handoffs_returns_packages_and_404s_a_bogus_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force an escalation intent (E2 complaint) so the turn builds + persists a handoff package.
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))
    token = _authenticate(client)
    chat = client.post("/chat", json={"token": token, "utterance": "quiero poner una queja"})
    assert chat.status_code == 200, chat.text
    case_id = chat.json()["handoff_case_id"]
    assert case_id

    agent = {"X-Agent-Key": _AGENT_KEY}
    listing = client.get("/handoffs", headers=agent)
    assert listing.status_code == 200
    assert case_id in listing.json()

    one = client.get(f"/handoffs/{case_id}", headers=agent)
    assert one.status_code == 200
    assert one.json()  # the persisted package

    assert client.get("/handoffs/CASE-does-not-exist", headers=agent).status_code == 404


def test_handoffs_fail_closed_without_a_valid_agent_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both /handoffs reads fail closed (401, no leak) for a missing or wrong agent key.

    A real case is persisted first (so a leak WOULD disclose data if the guard were absent), then
    every unauthorized shape - no header, empty header, wrong key - is refused with 401 and the
    response discloses no package content, for both the list and the by-id endpoint.
    """
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))
    token = _authenticate(client)
    chat = client.post("/chat", json={"token": token, "utterance": "quiero poner una queja"})
    case_id = chat.json()["handoff_case_id"]
    assert case_id

    for headers in ({}, {"X-Agent-Key": ""}, {"X-Agent-Key": "wrong-key"}):
        listing = client.get("/handoffs", headers=headers)
        assert listing.status_code == 401, headers
        assert case_id not in listing.text  # the queue is not disclosed
        # Authorization is checked BEFORE the lookup, so even a REAL id is 401, never 404 - an
        # unauthorized caller cannot probe which case ids exist.
        one = client.get(f"/handoffs/{case_id}", headers=headers)
        assert one.status_code == 401, headers
        assert "traceback" not in one.text.lower()


def test_handoffs_fail_closed_when_agent_key_is_not_configured(
    source: LocalSource, identity: MockIdentityService, handoff_store: JsonHandoffStore, clock: _Clock
) -> None:
    """With `CORA_AGENT_CONSOLE_KEY` unset (empty), /handoffs rejects EVERY caller (never open).

    The empty string is never a valid key, so an app built with no configured key fails closed for
    a request that sends no header AND for one that sends an empty header - the queue is never
    exposed just because the key was not configured.
    """
    app = create_app(
        identity=identity,
        source=source,
        handoff_store=handoff_store,
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        llm=StubLLMClient(),
        clock=clock,
        agent_console_key="",  # not configured
    )
    unconfigured = TestClient(app)
    assert unconfigured.get("/handoffs").status_code == 401
    assert unconfigured.get("/handoffs", headers={"X-Agent-Key": ""}).status_code == 401


# -- the classifier is really wired (no longer (None, 0.0)) ----------------------------


def test_known_utterance_classifies_to_expected_intent() -> None:
    """A known fixture utterance classifies to a real intent deterministically offline.

    Under `CORA_NLU_STUB=1` the orchestrator's default classifier uses the keyword baseline (the
    64-d stub encoder cannot drive the 384-d trained head), which STILL returns a real intent -
    proving `_classify` is wired and no longer the 4.1 `(None, 0.0)` stub.
    """
    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(),
        tool_layer_factory=lambda _s: None,  # _classify never touches the tool layer
        llm=StubLLMClient(),
    )
    intent, confidence = orch._classify("quiero congelar mi tarjeta", "es")  # noqa: SLF001 - test seam
    assert intent is Intent.A1
    assert confidence > 0.0


def test_overlapping_turns_classify_in_their_own_language(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two CONCURRENT same-token turns in different languages each classify in their OWN language.

    The `Orchestrator` is a per-process singleton shared across concurrent FastAPI requests, and a
    turn mutates the SAME mutable `SessionState` for a `jti` (it writes `state.language` from the
    utterance, then later classifies and renders using it). Without per-session isolation an
    overlapping pt/es turn could flip `state.language` between the first turn's language choice and
    its classification/rendering, so a turn would classify in the OTHER request's language and
    misroute deterministic policy (REQ-51). `step` prevents this by holding the per-`jti` lock for
    the WHOLE turn body (`_step_locked`); different sessions hold different locks and still run
    concurrently.

    Deterministic concurrency test (no sleeps/races), engineered so deleting the `with
    self._session_lock(...)` wrapper in `step` makes it FAIL. The seam is `_step_locked` itself -
    the FIRST thing the locked turn runs, so it executes INSIDE the lock. The first thread to enter
    parks there WHILE STILL HOLDING the session lock; the test then asserts the second thread has
    NOT been able to enter `_step_locked` within a bounded wait. With the lock that entry is blocked
    (the assertion holds); WITHOUT the lock the second turn enters concurrently and its entry event
    fires, so the assertion fails. That load-bearing check is what proves whole-turn isolation. The
    captured (utterance, language) pairing is the functional check layered on top.
    """
    real_step_locked = Orchestrator._step_locked  # bound below; capture the real impl first.
    entered = {"pt": threading.Event(), "es": threading.Event()}
    first_entered = threading.Event()
    release_first = threading.Event()
    captured: list[tuple[str, str]] = []
    captured_guard = threading.Lock()
    first_lang: list[str] = []

    def _recording_classify(self: Orchestrator, text: str, language: str):  # noqa: ANN202 - test seam
        with captured_guard:
            captured.append((text, language))
        return (Intent.E2, 0.99)

    def _barrier_step_locked(self: Orchestrator, session, utterance: str, **kwargs):  # noqa: ANN001, ANN202 - test seam
        # Runs INSIDE `step`'s per-`jti` lock and is the first thing the locked turn does. Signal
        # entry per language, then the FIRST turn to enter parks here (still holding the lock) until
        # the test releases it; the second turn runs through normally.
        lang = "pt" if utterance == utterances["pt"] else "es"
        entered[lang].set()
        with captured_guard:
            is_first = not first_lang
            if is_first:
                first_lang.append(lang)
        if is_first:
            first_entered.set()
            release_first.wait(timeout=5)
        return real_step_locked(self, session, utterance, **kwargs)

    monkeypatch.setattr(Orchestrator, "_classify", _recording_classify)
    monkeypatch.setattr(Orchestrator, "_step_locked", _barrier_step_locked)

    token = _authenticate(client)
    utterances = {
        "pt": "quero fazer uma reclamacao",  # detects pt -> turn sets state.language = "pt"
        "es": "quiero poner una queja",  # detects es -> turn sets state.language = "es"
    }
    results: dict[str, int] = {}

    def _post(lang: str) -> None:
        resp = client.post("/chat", json={"token": token, "utterance": utterances[lang]})
        results[lang] = resp.status_code

    t_pt = threading.Thread(target=_post, args=("pt",))
    t_es = threading.Thread(target=_post, args=("es",))
    t_pt.start()
    # Wait until the first turn is parked inside `_step_locked` (holding the session lock), then
    # start the second so it genuinely contends for the SAME session lock while the first is parked.
    assert first_entered.wait(timeout=5), "the first turn never entered the locked region"
    t_es.start()
    first = first_lang[0]
    other = "es" if first == "pt" else "pt"
    # LOAD-BEARING: while the first turn holds the lock, the second must be unable to enter
    # `_step_locked`. With the per-`jti` lock this stays unset; delete the lock and the second turn
    # enters concurrently, this event fires, and the test FAILS - which is the mutation sensitivity.
    assert not entered[other].wait(timeout=1), (
        "second turn entered the locked region while the first held the session lock - "
        "per-session isolation is broken (the per-`jti` lock was bypassed)"
    )
    release_first.set()
    t_pt.join(timeout=5)
    t_es.join(timeout=5)

    assert results == {"pt": 200, "es": 200}, results
    assert len(captured) == 2, captured
    # Each classify call saw the language of its OWN utterance - no cross-request bleed. Order is
    # not asserted (the lock may run either turn first); the (utterance, language) pairing is.
    by_text = dict(captured)
    assert by_text[utterances["pt"]] == "pt"
    assert by_text[utterances["es"]] == "es"


# -- the token / PII / secret never cross the API -> LLM or log boundary ---------------


class _RecordingLLM(StubLLMClient):
    """A stub LLM that records every `user` prompt it is asked to complete.

    Every call the orchestrator makes to the LLM (entity extraction, response polish) passes
    through here, so `seen` is the complete record of what crossed the API -> LLM boundary this
    turn. It returns a benign same-language reply so the polish path runs normally.
    """

    def __init__(self) -> None:
        super().__init__(default="Voy a transferir tu caso a una persona de nuestro equipo.")
        self.seen: list[str] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.seen.append(user)
        return super().complete(system=system, user=user, max_tokens=max_tokens)


def test_raw_pii_token_and_secret_never_reach_the_llm_or_logs(
    source: LocalSource,
    identity: MockIdentityService,
    handoff_store: JsonHandoffStore,
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raw HTTP-supplied PII, the bearer token, and the signing secret stay out of the LLM and logs.

    An E2 complaint is pinned so the turn runs entity extraction AND an LLM polish (both call the
    recording client), guaranteeing the LLM boundary is exercised. The utterance carries raw PII of
    every masked kind (email, phone, document number, address, PAN). We then assert that none of the
    raw PII, the session token, or the signing key appears in ANY prompt sent to the LLM or in any
    captured log record - the orchestrator masks before the LLM and nothing logs the credential.
    """
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))
    recorder = _RecordingLLM()
    app = create_app(
        identity=identity,
        source=source,
        handoff_store=handoff_store,
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        llm=recorder,
        clock=clock,
    )
    client = TestClient(app)
    token = _authenticate(client)

    raw_pii = {
        "email": "juan.perez@example.com",
        "phone": "+52 55 1234 5678",
        "document": "DNI 30.123.456",
        "address": "Calle Falsa 123, Bogota",
        "pan": "4111 1111 1111 1111",
    }
    utterance = (
        "Quiero poner una queja. Soy Juan y mi correo es juan.perez@example.com, "
        "mi telefono +52 55 1234 5678, documento DNI 30.123.456, vivo en Calle Falsa 123, Bogota, "
        "mi tarjeta es 4111 1111 1111 1111."
    )

    with caplog.at_level(0):  # capture every log record regardless of level
        resp = client.post("/chat", json={"token": token, "utterance": utterance})
    assert resp.status_code == 200, resp.text
    assert recorder.seen, "the turn must exercise the LLM boundary (entity extraction + polish)"

    llm_inputs = "\n".join(recorder.seen)
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    for kind, value in raw_pii.items():
        assert value not in llm_inputs, f"raw {kind} reached the LLM"
        assert value not in log_text, f"raw {kind} reached the logs"
    # The bearer token and the signing secret must never cross either boundary.
    assert token not in llm_inputs and token not in log_text, "the session token leaked"
    assert _KEY not in llm_inputs and _KEY not in log_text, "the signing secret leaked"
