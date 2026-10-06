"""Tests for task 5.1: the JSONL trace exporter and the per-turn trace id (REQ-39).

The exporter consumes the EXISTING `TurnResult`/`TraceSpan` the orchestrator already produces and
appends one JSON record per turn to a local JSONL file. These prove the hard requirement and the
security invariant (fail closed, no raw PII in a trace):

- `export_turn` writes exactly one JSON line per call and APPENDS (two calls -> two lines);
- the record round-trips via `json.loads` and carries the REQ-39 execution fields (masked input,
  language, intent + confidence, decision + rule id, output signal, timestamp, spans);
- NO raw PII in ANY serialised field: all five PII categories (email, document number, phone,
  address, full card number) are placed in the fields the exporter actually writes
  (`masked_input`, `message`, span `detail`) and in the never-read `response.text`, and none leaks;
- a REAL orchestrator turn writes exactly one record (tracing is wired into `step`, not detached),
  and a trace-write failure is BEST-EFFORT (security steering P5, "everything is a record"): a
  trace is an audit artifact, never a gate on an already-verified turn, so an exporter `OSError`
  is swallowed and the turn result is returned UNCHANGED - a verified card freeze still executes
  and reads back OK, a confirmation prompt is still delivered, the response is not rewritten to an
  escalation, and the `trace_id` stays on the result (it correlates the handoff/UI either way);
- the orchestrator gives each turn a stable, unique `trace_id`, threads it into the persisted
  handoff package, and the agent console's pure formatter surfaces it (handoff + UI retrieval).

`CORA_NLU_STUB=1` is autouse (conftest); nothing here needs the network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cora.agent.generator import GeneratedResponse
from cora.agent.graph import Orchestrator, TraceSpan, TurnResult
from cora.agent.state import SessionStore
from cora.agent.templates import Outcome
from cora.handoff.store import JsonHandoffStore
from cora.identity import MockIdentityService
from cora.obs import export_turn
from cora.obs.tracing import TRACE_FILENAME, build_record
from cora.policy import Decision, Intent, PolicyEngine, Thresholds
from cora.tools import InMemoryHandoffStore, Result, Status, ToolLayer
from cora.tools.models import BalanceData, CardDetailsData

# PII probes that must NEVER appear raw in a trace record. All five categories the security
# steering names (email, document number, phone, address, full card number) are covered, and the
# test drives them through the fields the exporter actually serialises (masked_input, message,
# span detail) as well as the never-read response text.
_PII_EMAIL = "juan.perez@mail.com"
_PII_CARD = "4111111111111111"
_PII_PHONE = "+52 55 1234 5678"
_PII_DOC = "DNI 12345678"
_PII_ADDRESS = "Calle Falsa 123"
_PII_PROBES = (_PII_EMAIL, _PII_CARD, _PII_PHONE, _PII_DOC, _PII_ADDRESS)


def _turn_result(trace_id: str = "a" * 32) -> TurnResult:
    """A TurnResult whose EVERY free-form field is stuffed with PII - the trace must stay clean.

    PII is placed in the fields the exporter serialises (`masked_input`, `message`, span `detail`)
    AND in the never-read `response.text`, so the test proves the two defences together: the
    response text is never read, and the serialised free-form fields are masked at the boundary.
    """
    pii_blob = f"{_PII_EMAIL} {_PII_CARD} {_PII_PHONE} {_PII_DOC} {_PII_ADDRESS}"
    return TurnResult(
        node="Answer",
        trace_id=trace_id,
        decision=Decision.ANSWER,
        rule_id="POL-090",
        rules_version="v1",
        intent=Intent.I1,
        masked_input=pii_blob,
        language="es",
        intent_confidence=0.91,
        # The response text (what the customer sees) carries PII; the exporter must not read it.
        response=GeneratedResponse(
            text=f"Hola, tu correo {_PII_EMAIL}, tarjeta {_PII_CARD}, tel {_PII_PHONE}",
            outcome=Outcome.ANSWER,
            lang="es",
            polished=False,
            prompt_hash=None,
        ),
        # `message` should be an orchestrator-authored status string; feed it PII to prove the
        # exporter masks it defensively rather than trusting the producer's convention.
        message=f"status: answered {pii_blob}",
        spans=[
            TraceSpan(node="Understand", detail=pii_blob),
            TraceSpan(node="Answer", decision="answer", rule_id="POL-090", rules_version="v1"),
        ],
    )


def test_export_writes_one_json_line_and_appends(tmp_path: Path) -> None:
    target = tmp_path / "traces" / TRACE_FILENAME
    export_turn(_turn_result(trace_id="t1" + "0" * 30), path=target)
    export_turn(_turn_result(trace_id="t2" + "0" * 30), path=target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # appended, not overwritten
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["trace_id"].startswith("t1")
    assert second["trace_id"].startswith("t2")


def test_record_round_trips_with_the_req39_execution_fields(tmp_path: Path) -> None:
    target = tmp_path / TRACE_FILENAME
    export_turn(_turn_result(trace_id="r" * 32), path=target)
    record = json.loads(target.read_text(encoding="utf-8").strip())

    # REQ-39: masked input, language, intent + confidence, decision + rule id, output signal, ts.
    assert record["trace_id"] == "r" * 32
    assert record["node"] == "Answer"
    assert record["decision"] == "answer"
    assert record["intent"] == Intent.I1.value
    assert record["intent_confidence"] == 0.91
    assert record["language"] == "es"
    assert record["masked_input"]  # present (and masked - asserted by the PII test)
    assert record["ts"]  # a timestamp is recorded
    assert record["output"]["outcome"] == Outcome.ANSWER.value
    assert record["output"]["polished"] is False
    assert isinstance(record["spans"], list) and len(record["spans"]) == 2
    assert record["spans"][-1]["rule_id"] == "POL-090"


def test_no_raw_pii_reaches_the_trace_in_any_serialised_field(tmp_path: Path) -> None:
    target = tmp_path / TRACE_FILENAME
    export_turn(_turn_result(), path=target)
    raw = target.read_text(encoding="utf-8")

    # None of the five PII categories leak - and they were placed in masked_input, message and a
    # span detail (the fields actually serialised), not only in the never-read response text.
    for probe in _PII_PROBES:
        assert probe not in raw, f"PII probe {probe!r} leaked into the trace file"
    # The record never serialises the raw customer-facing text under any key.
    record = json.loads(raw.strip())
    assert record["output"]["outcome"]  # output is metadata only...
    assert "text" not in record["output"]  # ...never the rendered reply


def test_build_record_masks_every_free_form_field() -> None:
    # Guard the invariant directly on the dict: masked_input, message and each span detail are
    # masked; the response text is never a field at all.
    record = build_record(_turn_result())
    serialised = json.dumps(record)
    for probe in _PII_PROBES:
        assert probe not in serialised, f"PII probe {probe!r} not masked in the record"


def test_export_defaults_to_settings_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # With no explicit path, the file lands under settings.trace_dir / traces.jsonl.
    monkeypatch.setenv("CORA_TRACE_DIR", str(tmp_path / "runtime_traces"))
    from cora.settings import get_settings

    get_settings.cache_clear()
    try:
        written = export_turn(_turn_result())
    finally:
        get_settings.cache_clear()
    assert written == tmp_path / "runtime_traces" / TRACE_FILENAME
    assert written.exists()


# -- end-to-end: stable trace id + handoff/UI retrieval --------------------------------

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CUST-000123"
_THRESHOLDS = Thresholds(tau_fraud=70.0, tau_escalate=0.40, tau_clarify=0.50)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _ExistingCustomerSource:
    """A `DataSource` stand-in whose `customers` count is 1 (so the double is a real customer).

    The deliverable-2 non-customer branch calls `source.count("customers", where=...)` every turn;
    returning 1 keeps the tracing turns on the ordinary (real-customer) path.
    """

    def count(self, table, *, where=None, **_kwargs):  # noqa: ANN001, ANN003 - test double
        assert table == "customers"
        return 1


class _FakeToolLayer:
    def __init__(self, owned: set[str]) -> None:
        self._owned = owned
        self.handoff_store = InMemoryHandoffStore()
        # The non-customer branch reads `customer_id` + `source.count("customers")` every turn.
        self.customer_id = _CUSTOMER
        self.source = _ExistingCustomerSource()

    # Reuse the REAL existence contract (needs only `customer_id` + `source.count`).
    customer_record_exists = ToolLayer.customer_record_exists

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
        return Result[CardDetailsData](status=Status.FORBIDDEN, message="not authorized")


class _Settings:
    """Minimal settings stand-in: the exporter only reads `trace_dir`."""

    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir


def _orchestrator(clock: _Clock, trace_dir: Path) -> Orchestrator:
    return Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=lambda _session: _FakeToolLayer({"PRD-1"}),
        clock=clock,
        settings=_Settings(trace_dir),
    )


def _authenticated_session(clock: _Clock) -> object:
    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)
    challenge = service.begin_authentication(_CUSTOMER)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


def test_turn_trace_id_is_retrievable_via_handoff_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _authenticated_session(clock)
    orch = _orchestrator(clock, tmp_path)
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))

    result = orch.step(session, "quiero poner una queja")
    # The id is stable within the turn and retrievable from the handoff package a human/UI reads.
    assert result.trace_id and len(result.trace_id) == 32
    assert result.handoff_package is not None
    assert result.handoff_package.trace_ref == result.trace_id


def test_every_turn_writes_exactly_one_trace_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A real orchestrator turn (not a direct export_turn call) must produce one JSONL line, and a
    # second turn appends a second line - proving tracing is wired into step(), not detached.
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _authenticated_session(clock)
    orch = _orchestrator(clock, tmp_path)
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))

    orch.step(session, "quiero poner una queja")
    orch.step(session, "sigo molesto")

    trace_file = tmp_path / TRACE_FILENAME
    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # one record per turn, appended
    assert all(json.loads(line)["trace_id"] for line in lines)


def test_trace_write_failure_does_not_change_the_handoff_turn_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Best-effort tracing (P5): a trace is an audit artifact, not a gate on a verified turn. An
    # exporter OSError must be swallowed and the real turn result returned UNCHANGED - the handoff
    # is still built and persisted (its trace_ref is the valid id), the trace_id still rides the
    # result, and nothing is rewritten to a different escalation. No trace line is written.
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _authenticated_session(clock)
    store = JsonHandoffStore(path=tmp_path / "handoffs.json")

    class _StoringToolLayer(_FakeToolLayer):
        def __init__(self) -> None:
            super().__init__({"PRD-1"})
            self.handoff_store = store

    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=lambda _session: _StoringToolLayer(),
        clock=clock,
        settings=_Settings(tmp_path),
    )
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("trace sink unavailable")

    monkeypatch.setattr("cora.agent.graph.export_turn", _boom)
    result = orch.step(session, "quiero poner una queja")

    # Unchanged outcome: the real handoff still completed and persisted, the id still rides.
    assert result.node == "Handoff"
    assert result.handoff_case_id  # a real case id, present on the returned result
    assert result.trace_id and len(result.trace_id) == 32  # id stays - it still correlates
    assert result.handoff_package is not None
    assert result.handoff_package.trace_ref == result.trace_id
    assert store.get(result.handoff_case_id) is not None  # persisted despite the failed trace
    # The sink failed, so no trace line exists - but the turn still happened.
    assert not (tmp_path / TRACE_FILENAME).exists()


def test_handoff_record_matches_the_returned_final_spans_and_case_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The handoff is finalized before export, so the JSONL record snapshots the returned
    # `TurnResult.spans` and carries the real case id - not a null id with a missing span.
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _authenticated_session(clock)
    orch = _orchestrator(clock, tmp_path)
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))

    result = orch.step(session, "quiero poner una queja")
    assert result.node == "Handoff"
    assert result.handoff_case_id  # a real case id, present on the returned result

    record = json.loads((tmp_path / TRACE_FILENAME).read_text(encoding="utf-8").strip())
    # The record's case id matches the returned result (not null for a successful handoff).
    assert record["handoff_case_id"] == result.handoff_case_id
    # The record's spans are the SAME ordered node list the result returned, including the final
    # Handoff span.
    assert [s["node"] for s in record["spans"]] == [s.node for s in result.spans]
    assert record["spans"][-1]["node"] == "Handoff"
    assert record["spans"][-1]["detail"] == result.handoff_case_id


def test_handoff_console_renders_the_stored_trace_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # End-to-end UI retrieval (task 5.4 tie): a turn's trace id is persisted to the durable store
    # and the agent console's pure formatter surfaces it from the stored package.
    import ui.agent_console as console

    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    session = _authenticated_session(clock)

    store = JsonHandoffStore(path=tmp_path / "handoffs.json")

    class _StoringToolLayer(_FakeToolLayer):
        def __init__(self) -> None:
            super().__init__({"PRD-1"})
            self.handoff_store = store

    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=lambda _session: _StoringToolLayer(),
        clock=clock,
        settings=_Settings(tmp_path),
    )
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.E2, 0.99))

    result = orch.step(session, "quiero poner una queja")
    stored = store.get(result.handoff_case_id)
    assert stored is not None
    assert stored["trace_ref"] == result.trace_id
    # The console's pure formatter surfaces the stored id read-only (no new UI abstraction).
    assert result.trace_id in console.trace_label(stored["trace_ref"])
    assert console.trace_label("") == "Trace id: (none)"


# -- best-effort tracing: a verified card action still runs when the trace sink is down ---------

_ACTION_CUSTOMER = "CLI-ACTION000001"
_ACTION_CARD = "PRD-ACTIONCARD1"


def _action_landing(root: Path) -> None:
    import pandas as pd

    # A `customers` row so the deliverable-2 existence check returns OK and this real customer
    # takes the ordinary action path (without it the lookup is UNAVAILABLE -> fail closed).
    customers = pd.DataFrame([{"customer_id": _ACTION_CUSTOMER, "full_name": "Action One"}])
    products = pd.DataFrame(
        [
            {
                "product_id": _ACTION_CARD,
                "customer_id": _ACTION_CUSTOMER,
                "product_type": "Tarjeta Credito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "100.0",
                "credit_limit": "5000.0",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            }
        ]
    )
    root.mkdir(parents=True, exist_ok=True)
    customers.astype("string").to_parquet(root / "customers.parquet", index=False)
    products.astype("string").to_parquet(root / "products.parquet", index=False)


def test_confirmed_card_action_still_runs_when_trace_sink_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Best-effort tracing (P5): a trace is an audit artifact, never a gate on a verified action.
    # Drive a real A1 turn to the Confirm node, then make the trace exporter raise on the
    # affirmative turn and assert the freeze STILL executes and reads back OK (Verify), the
    # response is NOT rewritten to an escalation, and the trace_id still rides the result.
    import pandas as pd  # noqa: F401 - ensures the parquet engine is importable in this env

    from cora.agent import StubLLMClient
    from cora.data.datasource import LocalSource
    from cora.identity import Session
    from cora.tools import InMemoryCardOverlay, InMemoryConfirmationStore, ToolLayer

    _action_landing(tmp_path / "raw_parquet")
    source = LocalSource(root=tmp_path / "raw_parquet")
    overlay = InMemoryCardOverlay()
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    service = MockIdentityService(signing_key=_KEY, session_ttl=timedelta(minutes=15), clock=clock)
    challenge = service.begin_authentication(_ACTION_CUSTOMER)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    session = service.verify_token(token)

    def _factory(sess: Session) -> ToolLayer:
        return ToolLayer(
            sess, source, card_overlay=overlay, confirmations=InMemoryConfirmationStore(), clock=clock
        )

    orch = Orchestrator(
        policy=PolicyEngine.from_yaml(thresholds=_THRESHOLDS),
        store=SessionStore(clock=clock),
        tool_layer_factory=_factory,
        llm=StubLLMClient(),
        clock=clock,
        settings=_Settings(tmp_path / "traces"),
    )
    monkeypatch.setattr(Orchestrator, "_classify", lambda self, text, language: (Intent.A1, 0.99))

    # Turn 1: reference the owned active card and reach Confirm (pending opened, nothing executed).
    state = orch._store.require(session)  # noqa: SLF001 - test seam
    state.referenced_product_id = _ACTION_CARD
    confirm = orch.step(session, "congela mi tarjeta")
    assert confirm.node == "Confirm" and state.pending_action is not None
    assert overlay.get(_ACTION_CARD) is None  # not frozen yet

    # Turn 2 (the affirmative): the trace sink is down. The verified action STILL runs.
    def _export_boom(*_a: object, **_k: object) -> None:
        raise OSError("trace sink unavailable")

    monkeypatch.setattr("cora.agent.graph.export_turn", _export_boom)
    result = orch.step(session, "sí, adelante")

    assert result.node == "Verify"  # the action completed and read back OK
    assert result.node != "Escalate"  # the response was NOT rewritten to an escalation
    assert overlay.get(_ACTION_CARD) is not None  # the card WAS frozen (overlay changed)
    # The pending is single-use and consumed by the resolved confirmation.
    assert orch._store.require(session).pending_action is None  # noqa: SLF001
    # The trace_id still rides the result even though this JSONL append failed.
    assert result.trace_id and len(result.trace_id) == 32
