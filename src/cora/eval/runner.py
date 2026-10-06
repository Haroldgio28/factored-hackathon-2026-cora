"""Run B1 and full CORA over the held-out suite, with faults on, x3 repeats (task 6.3, REQ-41/44).

The runner is the harness's execution engine: it drives every scenario through each configured
system (`b1` and `cora`), `repeats` times, with `FaultInjectingToolLayer` turning a seeded share of
reads into `Status.UNAVAILABLE` so the report shows how each system handles a flaky backend. Each
(scenario, config, repeat) yields one `RunRecord` the deterministic judges (6.4) consume.

Reuses, never reinvents (ponytail rung 2): CORA is `Orchestrator.step` built exactly as the graph
tests build it (`MockIdentityService` -> `SessionStore` -> `tool_layer_factory`); B1 is `B1NaiveLLM`
from 6.2; faults come from `faults.FaultInjectingToolLayer`; the real tools come from `ToolLayer`
over the injected `DataSource`. The runner adds only the loop, the record, and the version capture.

Determinism (REQ-44): one `random.Random(seed)` seeds a per-(scenario, config, repeat) child RNG
for the fault layer, so a repeat is a true replay of the same injected failures - not a reroll - and
the whole run reproduces from the seed. The ONLY non-determinism is a real Bedrock LLM; under the
stub (tests / `CORA_NLU_STUB=1`) there is none, and the record carries the model id so a real run is
labelled honestly (REQ-47).

Versions captured per record (REQ-44): the LLM model id (from settings, never hard-coded), the
B1/CORA prompt-version hashes, the code commit (`git rev-parse HEAD`) and the data snapshot id.

Security: the runner is a harness, not a decision-maker. It mints each session through the real
identity service (so a scenario's `customer_id` enters only via a verified token, never free text),
and it treats every scenario `utterance` as DATA passed to `step`/`answer` - it never executes
anything from a transcript or a model reply (security steering P1/P7).
"""

from __future__ import annotations

import logging
import random
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from cora.agent.graph import Orchestrator, TurnResult
from cora.agent.state import SessionStore
from cora.eval.baselines import B1NaiveLLM
from cora.eval.faults import FaultInjectingToolLayer
from cora.eval.scenarios import EVAL_NOW, Scenario
from cora.identity import MockIdentityService, Session
from cora.policy import PolicyEngine
from cora.tools import ToolLayer

logger = logging.getLogger("cora.eval.runner")

__all__ = ["CONFIGS", "RunRecord", "RunVersions", "run_suite", "write_runs"]

# The two systems the suite is run through: the naive baseline and full CORA (design section 9).
CONFIGS: tuple[str, ...] = ("b1", "cora")

# A fixed signing key for the harness's mock sessions. It is NOT a real secret (the mock IdP is a
# local test double); a scenario's customer_id still only enters via a verified token minted here.
_HARNESS_SIGNING_KEY = "eval-runner-key"
_SESSION_TTL = timedelta(minutes=15)


@dataclass(frozen=True)
class RunVersions:
    """The provenance a run is reproducible from (REQ-44): model, prompts, commit, data snapshot."""

    model_id: str
    code_commit: str
    data_snapshot: str
    b1_prompt_hash: str | None = None
    cora_prompt_hash: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """One (scenario, config, repeat) outcome - the uniform record the 6.4 judges score.

    Carries the scenario id + its reference fields (so a judge needs only the record), the system
    that produced it, the typed decision/escalation signals (from CORA's `TurnResult`; `None` for
    B1, which has no policy decision by design), the produced customer-facing text and any produced
    facts, the latency, whether the system was reachable, and the run versions (REQ-44). No raw PII:
    CORA's text/decision are already PII-safe, and B1's text is the model reply which the judges
    treat as data.
    """

    scenario_id: str
    config: str
    repeat: int
    language: str
    category: str
    # Reference outcome copied from the scenario so a judge scores from one record (6.4).
    expected_decision: str
    expected_escalation: bool
    # Produced signals. CORA fills node/decision/rule/escalation/handoff from the TurnResult; B1
    # leaves them None (it has no policy engine) and only produces `produced_text`. `produced_node`
    # is the state-machine node CORA landed on (e.g. "Unauthenticated" on the fail-closed re-auth
    # path, where `produced_decision` is None because the turn short-circuits before policy runs).
    produced_node: str | None = None
    produced_decision: str | None = None
    produced_rule_id: str | None = None
    produced_escalation: bool = False
    handoff_case_id: str | None = None
    handoff_complete: bool | None = None
    produced_text: str | None = None
    latency_ms: float = 0.0
    available: bool = True
    versions: dict[str, object] = field(default_factory=dict)


def code_commit() -> str:
    """Current commit hash via `git rev-parse HEAD`, or 'unknown' off a repo (REQ-44)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _mint_session(service: MockIdentityService, customer_id: str) -> Session:
    """Mint a verified session for a customer id (the production construction, mirrors the builder)."""
    challenge = service.begin_authentication(customer_id)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


def _child_rng(seed: int, scenario_id: str, config: str, repeat: int) -> random.Random:
    """A per-cell RNG so faults are seed-deterministic AND replay identically across repeats.

    The repeat index is deliberately NOT mixed in, so repeat 0/1/2 of the same (scenario, config)
    see the SAME injected failures - a repeat is a replay, the variability REQ-44 asks for comes
    from a real LLM, not from re-rolling the fault pattern.
    """
    return random.Random(f"{seed}:{scenario_id}:{config}")


def _run_cora(
    scenario: Scenario,
    source: object,
    *,
    rng: random.Random,
    fault_rate: float,
    now: datetime,
    policy: PolicyEngine,
) -> Callable[[], TurnResult]:
    """Build the CORA turn for one scenario and return a thunk that runs ONLY `step`.

    All setup (identity service, session mint, orchestrator) happens here, OUTSIDE the caller's
    timed region, so the recorded latency covers just the `step` call (REQ-44 latency fidelity).
    The shared `PolicyEngine` is passed in so its uncached `from_yaml()` read is not paid per cell.

    An `expired_session` adversarial case is realised by advancing the orchestrator/store clock
    PAST the freshly-minted session's expiry (reusing the identity `clock` seam, no sleeping), so
    `step` takes its fail-closed re-auth path. The tool layer is wrapped so a seeded share of reads
    fail with `Status.UNAVAILABLE`.
    """
    expired = scenario.adversarial_kind == "expired_session"
    issue_at = now
    turn_clock_time = now + _SESSION_TTL + timedelta(minutes=1) if expired else now

    # A tool_failure case is made CONCRETE: force every read to fail (rate=1.0) so the turn
    # deterministically takes its fail-closed TOOL_UNAVAILABLE path - the suite builder derives
    # the same not-owned reference from an always-UNAVAILABLE read, so expected and produced agree
    # instead of relying on the generic probabilistic fault_rate (which may or may not fire).
    effective_rate = 1.0 if scenario.force_tool_fault else fault_rate

    clock_box = {"t": issue_at}
    clock = lambda: clock_box["t"]  # noqa: E731 - a one-line injectable clock seam
    service = MockIdentityService(signing_key=_HARNESS_SIGNING_KEY, session_ttl=_SESSION_TTL, clock=clock)
    session = _mint_session(service, scenario.customer_id)
    clock_box["t"] = turn_clock_time  # advance AFTER issuing so an expired case is past expiry

    def factory(sess: Session) -> FaultInjectingToolLayer:
        return FaultInjectingToolLayer(ToolLayer(sess, source), rate=effective_rate, rng=rng)

    store = SessionStore(clock=clock)
    orchestrator = Orchestrator(
        policy=policy,
        store=store,
        tool_layer_factory=factory,
        clock=clock,
    )
    if scenario.referenced_product_id and not expired:
        # An unauthorized-access case references a product owned by ANOTHER customer. Seeding it
        # as the last-referenced product makes the turn attempt a REAL foreign read via the
        # orchestrator's existing deterministic resolution (`reference.product_id or
        # state.referenced_product_id`), so the ownership check fails closed on FORBIDDEN/NOT_FOUND
        # - the concrete unauthorized access the reference outcome is derived from (never a
        # label-forced abstain). `customer_id` is still bound only from the verified session; only
        # the referenced product id is seeded, and never on an expired case (state is discarded).
        store.require(session).referenced_product_id = scenario.referenced_product_id
    return lambda: orchestrator.step(session, scenario.utterance, language=scenario.language)


def _record_from_cora(scenario: Scenario, result: TurnResult, versions: RunVersions) -> dict[str, object]:
    """Pull the judge-relevant signals off a CORA `TurnResult` into record fields."""
    package = result.handoff_package
    # Handoff completeness is a 6.4 judge's job (all REQ-16 fields); here only note whether a
    # package was built at all, so the record is self-contained. `None` = no handoff this turn.
    handoff_complete = None if package is None else bool(package.customer_request and package.reason)
    produced_escalation = result.node in ("Escalate", "Handoff")
    return {
        "produced_node": result.node,
        "produced_decision": result.decision.value if result.decision is not None else None,
        "produced_rule_id": result.rule_id,
        "produced_escalation": produced_escalation,
        "handoff_case_id": result.handoff_case_id,
        "handoff_complete": handoff_complete,
        "produced_text": result.response.text if result.response is not None else None,
        "available": True,
    }


def run_suite(
    scenarios: Sequence[Scenario],
    source: object,
    *,
    configs: Sequence[str] = CONFIGS,
    repeats: int = 3,
    seed: int = 42,
    fault_rate: float = 0.1,
    versions: RunVersions,
    now: datetime | None = None,
    pace_s: float = 0.0,
) -> list[RunRecord]:
    """Run the suite through each config, `repeats` times, with faults on; return the run records.

    `source` is the `DataSource` the real tools read; `versions` carries the REQ-44 provenance to
    stamp on every record. `now` fixes the harness clock (defaults to a fixed epoch so a run is
    reproducible). Latency is `time.perf_counter` around the single `step`/`answer` call.

    `pace_s` sleeps that many seconds AFTER each turn (outside the timed region, so latency is
    unaffected). It defaults to 0.0 - a no-op for tests and the stub - and is set >0 only for a
    real-Bedrock run to stay under a low per-minute quota and avoid throttling turns into template
    fallback (which would silently contaminate the real numbers). ponytail: one stdlib sleep, not a
    rate-limiter class.
    """
    # The suite builder derives its date-sensitive FX reference at EVAL_NOW, so the runner must
    # execute every turn at the SAME clock or expected and produced FX rates would describe
    # different rows (REQ-28/44). A caller may still override `now` for a targeted experiment.
    base_now = now or EVAL_NOW
    records: list[RunRecord] = []
    version_dict = asdict(versions)
    # Build the policy engine once (its `from_yaml()` is uncached) so no CORA cell pays the YAML
    # read+parse inside the timed region (REQ-44 latency fidelity).
    policy = PolicyEngine.from_yaml() if "cora" in configs else None
    for scenario in scenarios:
        for config in configs:
            for repeat in range(repeats):
                rng = _child_rng(seed, scenario.id, config, repeat)
                extra: dict[str, object]
                # Setup OUTSIDE the timed region; time only the step/answer call.
                if config == "cora":
                    assert policy is not None  # built above when "cora" in configs
                    run_cora = _run_cora(
                        scenario, source, rng=rng, fault_rate=fault_rate, now=base_now, policy=policy
                    )
                    start = time.perf_counter()
                    result = run_cora()
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    extra = _record_from_cora(scenario, result, versions)
                elif config == "b1":
                    run_b1 = _run_b1(scenario, source, rng=rng, fault_rate=fault_rate, now=base_now)
                    start = time.perf_counter()
                    extra = run_b1()
                    latency_ms = (time.perf_counter() - start) * 1000.0
                else:
                    raise ValueError(f"unknown config {config!r}")
                records.append(
                    RunRecord(
                        scenario_id=scenario.id,
                        config=config,
                        repeat=repeat,
                        language=scenario.language,
                        category=scenario.category,
                        expected_decision=scenario.expected_decision.value,
                        expected_escalation=scenario.expected_escalation,
                        latency_ms=latency_ms,
                        versions=version_dict,
                        **extra,  # type: ignore[arg-type]
                    )
                )
                if pace_s > 0.0:
                    time.sleep(pace_s)  # outside the timed region: quota pacing, not latency
    return records


def _run_b1(
    scenario: Scenario,
    source: object,
    *,
    rng: random.Random,
    fault_rate: float,
    now: datetime,
) -> Callable[[], dict[str, object]]:
    """Build the B1 turn for one scenario and return a thunk that runs ONLY `answer`.

    Setup (identity service, session mint, fault-wrapped tool layer, B1 client) happens here,
    OUTSIDE the timed region, so latency covers just the `answer` call (REQ-44 latency fidelity).
    B1's read tool layer is wrapped in the SAME seeded `FaultInjectingToolLayer` as CORA's, so both
    configs face identical injected failures (REQ-41 identical-conditions comparison): a naive agent
    answering confidently off a failed read is exactly the danger the eval surfaces.

    The adversarial conditions are presented to B1 exactly as to CORA (REQ-41 same conditions): a
    `tool_failure` case forces every read to fail (rate=1.0), and an `unauthorized_access` case
    makes B1 attempt the SAME foreign product read. The session boundary is NOT weakened - B1's
    tool layer is still bound to the scenario's own verified session, so the foreign read fails
    closed with FORBIDDEN/NOT_FOUND; the eval then surfaces whether the naive agent leaks it.
    """
    clock = lambda: now  # noqa: E731 - fixed harness clock
    effective_rate = 1.0 if scenario.force_tool_fault else fault_rate
    service = MockIdentityService(signing_key=_HARNESS_SIGNING_KEY, session_ttl=_SESSION_TTL, clock=clock)
    session = _mint_session(service, scenario.customer_id)
    tools = FaultInjectingToolLayer(ToolLayer(session, source, clock=clock), rate=effective_rate, rng=rng)
    baseline = B1NaiveLLM()

    def run() -> dict[str, object]:
        result = baseline.answer(
            scenario.utterance,
            scenario.language,
            tools=tools,
            referenced_product_id=scenario.referenced_product_id,
        )
        return {
            "produced_decision": None,  # B1 has no policy decision by design
            "produced_text": result.text,
            "available": result.available,
        }

    return run


def write_runs(records: Sequence[RunRecord], path: Path | str) -> None:
    """Write run records as JSONL (one per line), creating the directory if absent."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
