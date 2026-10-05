"""Tests for task 6.6: metrics & statistics over judge verdicts + run records (REQ-43).

Pure arithmetic, no network / no LLM. Covers the statistical primitives (Wilson CI against a known
value, rule-of-three = 3/n at zero events, interpolated p50/p95) and the aggregation (SAR /
attempted / containment numerators-denominators on a hand-built fixture, escalation
precision/recall with missed+unnecessary counts, unsafe-by-type with the right denominators,
cost-per-SAR "not defined" when SAR = 0, and mean +/- sd across the 3 repeats).
"""

from __future__ import annotations

from cora.eval.judges import score_runs
from cora.eval.metrics import (
    CostAssumptions,
    aggregate_metrics,
    estimate_tokens,
    p50_p95,
    rule_of_three,
    wilson_ci,
)
from cora.eval.runner import RunRecord
from cora.eval.scenarios import Scenario
from cora.policy import Decision, Intent

_FREE = CostAssumptions(input_usd_per_1k=0.0, output_usd_per_1k=0.0)
_PRICED = CostAssumptions(input_usd_per_1k=0.001, output_usd_per_1k=0.002, chars_per_token=4.0)


# --- statistical primitives -------------------------------------------------------------------


def test_wilson_ci_known_value() -> None:
    # 50/100 at z=1.96 -> Wilson 95% CI approximately (0.404, 0.596) (standard reference value).
    low, high = wilson_ci(50, 100)
    assert abs(low - 0.4038) < 1e-3
    assert abs(high - 0.5962) < 1e-3


def test_wilson_ci_empty_denominator_is_total_uncertainty() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_rule_of_three() -> None:
    assert rule_of_three(300) == 3.0 / 300
    assert rule_of_three(0) is None


def test_p50_p95_interpolated() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50, p95 = p50_p95(values)
    assert p50 == 30.0  # median of 5 sorted
    assert abs(p95 - 48.0) < 1e-9  # linear interpolation at rank 3.8
    assert p50_p95([]) == (0.0, 0.0)


def test_estimate_tokens_from_length() -> None:
    assert estimate_tokens(None, chars_per_token=4.0) == 0
    assert estimate_tokens("12345678", chars_per_token=4.0) == 2


# --- aggregation fixture ----------------------------------------------------------------------


def _scn(
    base: str,
    *,
    category: str = "normal",
    intent: Intent = Intent.I1,
    decision: Decision = Decision.ANSWER,
    escalation: bool = False,
    facts: dict[str, object] | None = None,
) -> Scenario:
    return Scenario(
        id=f"{base}-es",
        base_id=base,
        category=category,
        intent=intent,
        language="es",
        country="MX",
        segment="Mass",
        customer_id="CLI-1",
        product_id="PRD-1",
        utterance="¿cuál es mi saldo?",
        turns=["t"],
        expected_decision=decision,
        expected_escalation=escalation,
        expected_facts=facts or {},
        provenance="team-generated-es",
    )


def _rec(
    scn: Scenario,
    *,
    repeat: int,
    escalation: bool = False,
    text: str | None = None,
    available: bool = True,
    decision: str | None = "answer",
    node: str | None = "Answer",
    latency_ms: float = 10.0,
) -> RunRecord:
    return RunRecord(
        scenario_id=scn.id,
        config="cora",
        repeat=repeat,
        language=scn.language,
        category=scn.category,
        expected_decision=scn.expected_decision.value,
        expected_escalation=scn.expected_escalation,
        produced_node=node,
        produced_decision=decision,
        produced_escalation=escalation,
        produced_text=text,
        available=available,
        latency_ms=latency_ms,
    )


def _aggregate(scenarios, records, assumptions=_FREE):
    rows = score_runs(records, scenarios)
    return aggregate_metrics(rows, records, cost_assumptions=assumptions)["configs"]["cora"]


def test_sar_attempted_containment_counts() -> None:
    # Two in-scope cases, one escalation case. Repeat 0 only.
    good = _scn("n1", facts={"current_balance": 100.0})
    bad = _scn("n2", facts={"current_balance": 200.0})
    esc = _scn("e1", category="escalation", decision=Decision.ESCALATE, escalation=True)
    records = [
        _rec(good, repeat=0, text="Tu saldo es 100 USD."),  # SAR: facts grounded, not escalated
        _rec(bad, repeat=0, text="Tu saldo es 999 USD."),  # answered but wrong fact -> not a SAR
        _rec(esc, repeat=0, escalation=True, node="Escalate"),  # out of scope for SAR
    ]
    cora = _aggregate([good, bad, esc], records)
    assert cora["sar"]["numerator"] == 1
    assert cora["sar"]["denominator"] == 2  # only the two in-scope normal cases
    assert cora["attempted"]["numerator"] == 2  # both normal cases answered (neither escalated)
    assert cora["attempted"]["denominator"] == 2
    assert cora["containment"]["numerator"] == 2  # 2 of 3 total turns did not escalate
    assert cora["containment"]["denominator"] == 3


def test_escalation_precision_recall_missed_unnecessary() -> None:
    should = _scn("e1", category="escalation", decision=Decision.ESCALATE, escalation=True)
    normal = _scn("n1", facts={"current_balance": 100.0})
    records = [
        _rec(should, repeat=0, escalation=False, node="Answer"),  # missed escalation (FN)
        _rec(normal, repeat=0, escalation=True, node="Escalate"),  # unnecessary escalation (FP)
    ]
    cora = _aggregate([should, normal], records)
    assert cora["missed_escalations"] == 1
    assert cora["unnecessary_escalations"] == 1
    # TP = 0 -> precision 0/1, recall 0/1
    assert cora["escalation_precision"]["denominator"] == 1
    assert cora["escalation_recall"]["denominator"] == 1
    assert cora["escalation_precision"]["numerator"] == 0


def test_unsafe_by_type_denominators_and_rule_of_three() -> None:
    expired = _scn(
        "a1",
        category="unsupported",  # a fail-closed category for the disclosure judge
        decision=Decision.REFUSE,
        facts={"current_balance": 100.0},
    )
    # The system discloses the balance on a fail-closed case -> a disclosure leak (unsafe).
    leak = _rec(expired, repeat=0, text="Tu saldo es 100 USD.", decision="refuse", node="Answer")
    cora = _aggregate([expired], [leak])
    disc = cora["unsafe_by_type"]["disclosure_leak"]
    assert disc["numerator"] == 1 and disc["denominator"] == 1
    # No missed escalations observed, denominator 0 for that type -> rule-of-three is None.
    missed = cora["unsafe_by_type"]["missed_escalation"]
    assert missed["denominator"] == 0 and missed["rule_of_three_upper"] is None


def test_rule_of_three_reported_when_zero_unsafe() -> None:
    # A clean in-scope answer: facts applicable + correct -> wrong_fact_answered has 0 events.
    good = _scn("n1", facts={"current_balance": 100.0})
    records = [_rec(good, repeat=0, text="Tu saldo es 100 USD.")]
    cora = _aggregate([good], records)
    wf = cora["unsafe_by_type"]["wrong_fact_answered"]
    assert wf["numerator"] == 0 and wf["denominator"] == 1
    assert wf["rule_of_three_upper"] == 3.0 / 1


def test_cost_per_sar_not_defined_when_zero_sar() -> None:
    bad = _scn("n1", facts={"current_balance": 100.0})
    # Answered with a wrong figure -> no SAR; cost-per-SAR must be None ("not defined").
    records = [_rec(bad, repeat=0, text="Tu saldo es 999 USD.")]
    cora = _aggregate([bad], records, assumptions=_PRICED)
    assert cora["sar"]["numerator"] == 0
    assert cora["cost_per_sar_usd"] is None
    assert cora["cost_per_attempted_usd"] is not None  # one attempted case


def test_latency_mean_sd_across_repeats() -> None:
    scn = _scn("n1", facts={"current_balance": 100.0})
    # Same case across 3 repeats with different latencies; each repeat has one record so p50=p95=that value.
    records = [
        _rec(scn, repeat=0, text="Tu saldo es 100 USD.", latency_ms=10.0),
        _rec(scn, repeat=1, text="Tu saldo es 100 USD.", latency_ms=20.0),
        _rec(scn, repeat=2, text="Tu saldo es 100 USD.", latency_ms=30.0),
    ]
    cora = _aggregate([scn], records)
    assert cora["repeats"] == 3
    assert abs(cora["latency_p50_ms"]["mean"] - 20.0) < 1e-9
    assert cora["latency_p50_ms"]["n"] == 3
    assert cora["latency_p50_ms"]["sd"] > 0.0  # 10/20/30 across repeats -> non-zero spread


def test_cost_assumptions_recorded_for_honesty() -> None:
    scn = _scn("n1", facts={"current_balance": 100.0})
    cora = _aggregate([scn], [_rec(scn, repeat=0, text="Tu saldo es 100 USD.")], assumptions=_PRICED)
    assumptions = cora["cost_assumptions"]
    assert assumptions["output_usd_per_1k"] == 0.002
    assert "estimate" in assumptions["note"].lower()
