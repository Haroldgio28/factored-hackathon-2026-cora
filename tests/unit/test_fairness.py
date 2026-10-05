"""Tests for task 6.7: fairness breakdown by language / country / segment (REQ-46).

Pure arithmetic, no network / no LLM. Covers: subgroup rates reconcile with the overall 6.6 metric
(a single-subgroup slice equals the whole), a best-vs-worst gap beyond tolerance is flagged over
the non-small-sample subgroups, a thin subgroup is marked `small_sample` and excluded from the gap
vote, and the breakdown carries the REQ-47 honesty labels.
"""

from __future__ import annotations

from cora.eval.fairness import DIMENSIONS, fairness_breakdown
from cora.eval.judges import score_runs
from cora.eval.metrics import CostAssumptions, aggregate_metrics
from cora.eval.runner import RunRecord
from cora.eval.scenarios import Scenario
from cora.policy import Decision, Intent

_FREE = CostAssumptions(input_usd_per_1k=0.0, output_usd_per_1k=0.0)


def _scn(
    base: str,
    *,
    language: str = "es",
    country: str = "MX",
    segment: str = "Mass",
    facts: dict[str, object] | None = None,
) -> Scenario:
    provenance = "team-generated-pt" if language == "pt" else "team-generated-es"
    return Scenario(
        id=f"{base}-{language}",
        base_id=base,
        category="normal",
        intent=Intent.I1,
        language=language,
        country=country,
        segment=segment,
        customer_id="CLI-1",
        product_id="PRD-1",
        utterance="saldo",
        turns=["t"],
        expected_decision=Decision.ANSWER,
        expected_escalation=False,
        expected_facts=facts or {},
        provenance=provenance,
    )


def _rec(scn: Scenario, *, repeat: int = 0, text: str | None = None) -> RunRecord:
    return RunRecord(
        scenario_id=scn.id,
        config="cora",
        repeat=repeat,
        language=scn.language,
        category=scn.category,
        expected_decision=scn.expected_decision.value,
        expected_escalation=scn.expected_escalation,
        produced_node="Answer",
        produced_decision="answer",
        produced_escalation=False,
        produced_text=text,
        available=True,
        latency_ms=10.0,
    )


def _run(scenarios, records):
    rows = score_runs(records, scenarios)
    return fairness_breakdown(rows, records, scenarios, configs=("cora",), min_n=1)


def test_breakdown_has_all_three_dimensions() -> None:
    scn = _scn("n1", facts={"current_balance": 100.0})
    out = _run([scn], [_rec(scn, text="Tu saldo es 100 USD.")])
    assert set(out["dimensions"]) == set(DIMENSIONS)
    assert out["label"] == "offline measurement"


def test_single_subgroup_reconciles_with_overall_sar() -> None:
    # One ES/MX case answered correctly -> SAR 1/1. The language=es subgroup must equal the overall.
    scn = _scn("n1", facts={"current_balance": 100.0})
    records = [_rec(scn, text="Tu saldo es 100 USD.")]
    rows = score_runs(records, scenarios := [scn])
    overall = aggregate_metrics(rows, records, cost_assumptions=_FREE)["configs"]["cora"]["sar"]["rate"]
    out = fairness_breakdown(rows, records, scenarios, configs=("cora",), min_n=1)
    es = next(s for s in out["dimensions"]["language"]["cora"]["subgroups"] if s["subgroup"] == "es")
    assert es["sar_rate"] == overall == 1.0


def test_language_gap_flagged_when_beyond_tolerance() -> None:
    # ES answers correctly (SAR 1.0), PT answers with a wrong figure (SAR 0.0) -> a 1.0 gap > 0.10.
    es = _scn("n1", language="es", facts={"current_balance": 100.0})
    pt = _scn("n1", language="pt", facts={"current_balance": 100.0})
    records = [
        _rec(es, text="Tu saldo es 100 USD."),
        _rec(pt, text="O seu saldo e 999 USD."),  # wrong fact -> not a SAR
    ]
    rows = score_runs(records, [es, pt])
    out = fairness_breakdown(rows, records, [es, pt], configs=("cora",), min_n=1)
    gap = out["dimensions"]["language"]["cora"]["gaps"]["sar_rate"]
    assert gap["computable"] is True
    assert gap["flagged"] is True
    assert gap["best"]["subgroup"] == "es"
    assert gap["worst"]["subgroup"] == "pt"
    assert abs(gap["spread"] - 1.0) < 1e-9


def test_thin_subgroup_marked_small_sample_and_excluded_from_gap() -> None:
    # Two countries, one record each, min_n=2 -> both thin -> no gap vote, gap not computable.
    mx = _scn("n1", country="MX", facts={"current_balance": 100.0})
    co = _scn("n2", country="CO", facts={"current_balance": 100.0})
    records = [_rec(mx, text="Tu saldo es 100 USD."), _rec(co, text="Tu saldo es 50 USD.")]
    rows = score_runs(records, [mx, co])
    out = fairness_breakdown(rows, records, [mx, co], configs=("cora",), min_n=2)
    subgroups = out["dimensions"]["country"]["cora"]["subgroups"]
    assert all(s["small_sample"] for s in subgroups)
    gap = out["dimensions"]["country"]["cora"]["gaps"]["sar_rate"]
    assert gap["computable"] is False and gap["flagged"] is False


def test_unsafe_rate_pooled_over_types() -> None:
    # A fail-closed (unsupported) case that leaks a fact -> one disclosure-leak unsafe event.
    leak_scn = Scenario(
        id="a1-es",
        base_id="a1",
        category="unsupported",
        intent=Intent.X1,
        language="es",
        country="AR",
        segment="Premium",
        customer_id="CLI-1",
        product_id="PRD-1",
        utterance="algo",
        turns=["t"],
        expected_decision=Decision.REFUSE,
        expected_escalation=False,
        expected_facts={"current_balance": 100.0},
        provenance="team-generated-es",
    )
    leak = RunRecord(
        scenario_id="a1-es",
        config="cora",
        repeat=0,
        language="es",
        category="unsupported",
        expected_decision="refuse",
        expected_escalation=False,
        produced_node="Answer",
        produced_decision="refuse",
        produced_escalation=False,
        produced_text="Tu saldo es 100 USD.",
        available=True,
        latency_ms=5.0,
    )
    rows = score_runs([leak], [leak_scn])
    out = fairness_breakdown(rows, [leak], [leak_scn], configs=("cora",), min_n=1)
    ar = next(s for s in out["dimensions"]["country"]["cora"]["subgroups"] if s["subgroup"] == "AR")
    # Pooled over types: 1 disclosure leak out of 2 applicable turns (disclosure applicable + facts
    # applicable; facts passed because "100" grounds the balance) -> 1 unsafe event / 2 = 0.5.
    assert ar["unsafe_rate"] == 0.5
