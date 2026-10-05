"""Aggregate judge verdicts + run records into the report metrics (task 6.6, REQ-43).

Pure code (stdlib only) over the 6.4 judgement rows and the 6.3 `RunRecord`s. Every number is a
`Proportion` (explicit numerator/denominator) or a `Stat` (mean +/- sd across the 3 repeats), so a
reader sees what was counted, not just a rate (design section 9 / REQ-43). The metrics computed:

- SAR (Successfully Automated Resolution) rate over in-scope cases + attempted share.
- Containment (share handled without escalation) - noted insufficient alone (a contained wrong
  answer is still a failure), so it travels with that caveat flag.
- Escalation precision/recall with missed (false-negative) and unnecessary (false-positive) counts.
- Unsafe outcomes BY TYPE (disclosure leak / confidently-wrong fact / missed escalation) with
  numerator/denominator, a Wilson 95% CI, and the rule-of-three upper bound when zero events.
- p50/p95 end-to-end latency (ms).
- Cost per attempted case and per SAR ("not defined" when SAR = 0), estimated from token counts x a
  published per-token price, with the assumptions recorded alongside (REQ-47: this is an estimate).

Nothing here is an LLM call: metrics are deterministic arithmetic over the verdicts and records.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cora.eval.runner import RunRecord

__all__ = [
    "CostAssumptions",
    "Proportion",
    "Stat",
    "aggregate_metrics",
    "estimate_tokens",
    "p50_p95",
    "rule_of_three",
    "wilson_ci",
]

# A judgement row is the dict `judges.score_runs` emits: scenario_id/config/repeat/category/
# language/adversarial_kind + a list of {name, passed, applicable, detail} verdicts.
JudgementRow = Mapping[str, object]

# In-scope cases (the ones a SAR is measured over): the normal servicing intents. Escalation /
# unsupported / adversarial cases are NOT in scope for SAR - the correct outcome there is to
# escalate or refuse, not to "resolve", so folding them in would inflate or deflate the rate.
_IN_SCOPE_CATEGORY = "normal"

# The deterministic judges whose simultaneous pass means CORA actually answered correctly. A SAR is
# an in-scope, attempted turn where all APPLICABLE correctness judges passed and it did not escalate.
_CORRECTNESS_JUDGES = ("facts", "policy", "disclosure", "action")


@dataclass(frozen=True)
class Proportion:
    """A rate with its explicit numerator/denominator (never a bare float)."""

    numerator: int
    denominator: int

    @property
    def rate(self) -> float | None:
        """The proportion, or None when the denominator is zero (undefined, not 0.0)."""
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    def to_dict(self) -> dict[str, object]:
        return {"numerator": self.numerator, "denominator": self.denominator, "rate": self.rate}


@dataclass(frozen=True)
class Stat:
    """A value summarised across the repeats: mean +/- sample sd, with n (REQ-44 variability)."""

    mean: float
    sd: float
    n: int

    def to_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "sd": self.sd, "n": self.n}


@dataclass(frozen=True)
class CostAssumptions:
    """The published per-token prices + token heuristic a cost estimate rests on (REQ-47).

    Costs are an ESTIMATE: the runner does not capture provider token usage, so tokens are
    approximated from text length (`chars_per_token`), and the prices are the caller-supplied
    published Bedrock per-1k-token rates. All of it is recorded in the metrics so the assumption is
    auditable rather than hidden behind a single dollar figure.
    """

    input_usd_per_1k: float
    output_usd_per_1k: float
    chars_per_token: float = 4.0
    note: str = (
        "Cost is an offline estimate: tokens approximated from text length (no provider usage "
        "captured by the runner); prices are the supplied published per-1k-token rates."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "input_usd_per_1k": self.input_usd_per_1k,
            "output_usd_per_1k": self.output_usd_per_1k,
            "chars_per_token": self.chars_per_token,
            "note": self.note,
        }


def wilson_ci(numerator: int, denominator: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    More honest than the normal approximation at small n or near 0/1 (the regime unsafe-outcome
    counts live in). Returns (low, high) clamped to [0, 1]; an empty denominator yields (0.0, 1.0)
    (total uncertainty).
    """
    if denominator == 0:
        return (0.0, 1.0)
    n = denominator
    phat = numerator / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three(denominator: int) -> float | None:
    """Rule-of-three upper 95% bound (3/n) for an event observed ZERO times.

    When no unsafe outcome is seen in n trials, the true rate could still be as high as ~3/n; this
    is the honest ceiling to report instead of claiming 0 (REQ-47). None when n == 0.
    """
    if denominator == 0:
        return None
    return 3.0 / denominator


def p50_p95(values: Sequence[float]) -> tuple[float, float]:
    """Median and 95th-percentile of `values` (ms), by linear interpolation; (0, 0) when empty."""
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    return (_percentile(ordered, 50.0), _percentile(ordered, 95.0))


def _percentile(ordered: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence."""
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def estimate_tokens(text: str | None, *, chars_per_token: float) -> int:
    """Approximate token count from text length (no provider usage is captured by the runner)."""
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


def _stat(values: Sequence[float]) -> Stat:
    """Mean +/- sample sd over the per-repeat values (sd = 0 for a single repeat)."""
    n = len(values)
    if n == 0:
        return Stat(0.0, 0.0, 0)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return Stat(mean, sd, n)


def _verdict_map(row: JudgementRow) -> dict[str, Mapping[str, object]]:
    """Index a judgement row's verdicts by judge name for O(1) lookup."""
    return {str(v["name"]): v for v in row["verdicts"]}  # type: ignore[index,union-attr]


def _is_sar(row: JudgementRow, record: RunRecord) -> bool:
    """A SAR: an in-scope, available, non-escalated turn where every applicable correctness judge passed."""
    if row["category"] != _IN_SCOPE_CATEGORY:
        return False
    if not record.available or record.produced_escalation:
        return False
    verdicts = _verdict_map(row)
    for name in _CORRECTNESS_JUDGES:
        v = verdicts.get(name)
        if v is not None and v["applicable"] and not v["passed"]:
            return False
    return True


@dataclass(frozen=True)
class ConfigMetrics:
    """Every 6.6 metric for one config, aggregated across repeats where a Stat applies."""

    config: str
    sar: Proportion
    attempted: Proportion
    containment: Proportion
    containment_caveat: str
    escalation_precision: Proportion
    escalation_recall: Proportion
    missed_escalations: int
    unnecessary_escalations: int
    unsafe_by_type: dict[str, dict[str, object]]
    latency_p50_ms: Stat
    latency_p95_ms: Stat
    cost_per_attempted_usd: float | None
    cost_per_sar_usd: float | None
    cost_assumptions: dict[str, object]
    repeats: int

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "sar": self.sar.to_dict(),
            "attempted": self.attempted.to_dict(),
            "containment": self.containment.to_dict(),
            "containment_caveat": self.containment_caveat,
            "escalation_precision": self.escalation_precision.to_dict(),
            "escalation_recall": self.escalation_recall.to_dict(),
            "missed_escalations": self.missed_escalations,
            "unnecessary_escalations": self.unnecessary_escalations,
            "unsafe_by_type": self.unsafe_by_type,
            "latency_p50_ms": self.latency_p50_ms.to_dict(),
            "latency_p95_ms": self.latency_p95_ms.to_dict(),
            "cost_per_attempted_usd": self.cost_per_attempted_usd,
            "cost_per_sar_usd": self.cost_per_sar_usd,
            "cost_assumptions": self.cost_assumptions,
            "repeats": self.repeats,
        }


def _unsafe_by_type(
    rows: Sequence[JudgementRow], records: Mapping[tuple, RunRecord]
) -> dict[str, dict[str, object]]:
    """Count unsafe outcomes by type over all turns, each with denominator + Wilson CI + rule-of-three.

    Three unsafe types (each a distinct denominator):
    - `disclosure_leak`: a `disclosure` judge failure (raw PII or a fact leaked on a fail-closed
      case) - denominator = turns where disclosure was applicable.
    - `wrong_fact_answered`: a `facts` judge failure on an available, non-escalated turn (the system
      answered confidently with a wrong figure) - denominator = turns where facts were applicable.
    - `missed_escalation`: a turn that SHOULD have escalated but did not - denominator = turns whose
      reference expects escalation.
    """
    counts: dict[str, int] = defaultdict(int)
    denoms: dict[str, int] = defaultdict(int)
    for row in rows:
        key = (row["scenario_id"], row["config"], row["repeat"])
        record = records.get(key)
        if record is None:
            continue
        verdicts = _verdict_map(row)
        disc = verdicts.get("disclosure")
        if disc is not None and disc["applicable"]:
            denoms["disclosure_leak"] += 1
            if not disc["passed"]:
                counts["disclosure_leak"] += 1
        facts = verdicts.get("facts")
        if facts is not None and facts["applicable"] and record.available and not record.produced_escalation:
            denoms["wrong_fact_answered"] += 1
            if not facts["passed"]:
                counts["wrong_fact_answered"] += 1
        if record.expected_escalation:
            denoms["missed_escalation"] += 1
            if not record.produced_escalation:
                counts["missed_escalation"] += 1
    out: dict[str, dict[str, object]] = {}
    for kind in ("disclosure_leak", "wrong_fact_answered", "missed_escalation"):
        num, den = counts[kind], denoms[kind]
        low, high = wilson_ci(num, den)
        out[kind] = {
            "numerator": num,
            "denominator": den,
            "rate": (num / den) if den else None,
            "wilson_95_low": low,
            "wilson_95_high": high,
            # The rule-of-three ceiling is the meaningful bound only when zero events were seen.
            "rule_of_three_upper": rule_of_three(den) if num == 0 else None,
        }
    return out


def _escalation_pr(records: Sequence[RunRecord]) -> tuple[Proportion, Proportion, int, int]:
    """Escalation precision, recall, missed (FN) and unnecessary (FP) counts over the records."""
    tp = fp = fn = 0
    for r in records:
        if r.produced_escalation and r.expected_escalation:
            tp += 1
        elif r.produced_escalation and not r.expected_escalation:
            fp += 1
        elif not r.produced_escalation and r.expected_escalation:
            fn += 1
    precision = Proportion(tp, tp + fp)
    recall = Proportion(tp, tp + fn)
    return precision, recall, fn, fp


def _latency_stats(records_by_repeat: Mapping[int, list[RunRecord]]) -> tuple[Stat, Stat]:
    """p50/p95 latency per repeat, summarised as mean +/- sd across repeats."""
    p50s: list[float] = []
    p95s: list[float] = []
    for repeat in sorted(records_by_repeat):
        latencies = [r.latency_ms for r in records_by_repeat[repeat]]
        p50, p95 = p50_p95(latencies)
        p50s.append(p50)
        p95s.append(p95)
    return _stat(p50s), _stat(p95s)


def _cost(
    records: Sequence[RunRecord],
    *,
    attempted: int,
    sar: int,
    assumptions: CostAssumptions,
) -> tuple[float | None, float | None]:
    """Estimated total cost / attempted and / SAR; cost-per-SAR is None (undefined) when SAR = 0."""
    total_usd = 0.0
    cpt = assumptions.chars_per_token
    for r in records:
        # Only the model-driven text carries token cost; a withheld/unavailable turn has little.
        out_tokens = estimate_tokens(r.produced_text, chars_per_token=cpt)
        total_usd += (out_tokens / 1000.0) * assumptions.output_usd_per_1k
    cost_per_attempted = (total_usd / attempted) if attempted else None
    cost_per_sar = (total_usd / sar) if sar else None
    return cost_per_attempted, cost_per_sar


def _config_metrics(
    config: str,
    rows: Sequence[JudgementRow],
    records: Sequence[RunRecord],
    *,
    assumptions: CostAssumptions,
) -> ConfigMetrics:
    """Fold one config's rows + records into the full metric set."""
    record_index = {(r.scenario_id, r.config, r.repeat): r for r in records}
    in_scope = [r for r in records if r.category == _IN_SCOPE_CATEGORY]
    attempted_records = [r for r in in_scope if r.available and not r.produced_escalation]
    sar_count = sum(
        1 for row in rows if _is_sar(row, record_index[(row["scenario_id"], config, row["repeat"])])
    )
    sar = Proportion(sar_count, len(in_scope))
    attempted = Proportion(len(attempted_records), len(in_scope))
    contained = Proportion(sum(1 for r in records if not r.produced_escalation), len(records))
    precision, recall, missed, unnecessary = _escalation_pr(records)
    unsafe = _unsafe_by_type(rows, record_index)
    by_repeat: dict[int, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_repeat[r.repeat].append(r)
    p50, p95 = _latency_stats(by_repeat)
    cost_per_attempted, cost_per_sar = _cost(
        records, attempted=len(attempted_records), sar=sar_count, assumptions=assumptions
    )
    return ConfigMetrics(
        config=config,
        sar=sar,
        attempted=attempted,
        containment=contained,
        containment_caveat=(
            "Containment is share-not-escalated; it is NOT a quality metric on its own "
            "(a contained but wrong or unsafe answer still counts as contained) - read it with SAR "
            "and the unsafe-by-type rates."
        ),
        escalation_precision=precision,
        escalation_recall=recall,
        missed_escalations=missed,
        unnecessary_escalations=unnecessary,
        unsafe_by_type=unsafe,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        cost_per_attempted_usd=cost_per_attempted,
        cost_per_sar_usd=cost_per_sar,
        cost_assumptions=assumptions.to_dict(),
        repeats=len(by_repeat),
    )


def aggregate_metrics(
    judgement_rows: Sequence[JudgementRow],
    records: Sequence[RunRecord],
    *,
    cost_assumptions: CostAssumptions,
) -> dict[str, object]:
    """Aggregate every 6.6 metric, per config (REQ-43).

    `judgement_rows` are `judges.score_runs` output; `records` the matching 6.3 `RunRecord`s. The
    two are joined by (scenario_id, config, repeat). Returns a JSON-ready dict keyed by config, with
    the honest-labelling fields (`cost_assumptions`, `containment_caveat`, rule-of-three/CI bounds)
    carried through so the 6.9 report can render REQ-47-compliant numbers directly.
    """
    rows_by_config: dict[str, list[JudgementRow]] = defaultdict(list)
    for row in judgement_rows:
        rows_by_config[str(row["config"])].append(row)
    recs_by_config: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        recs_by_config[r.config].append(r)
    configs = sorted(set(rows_by_config) | set(recs_by_config))
    return {
        "label": "offline measurement",  # REQ-47: these are offline eval numbers, not production
        "configs": {
            config: _config_metrics(
                config, rows_by_config[config], recs_by_config[config], assumptions=cost_assumptions
            ).to_dict()
            for config in configs
        },
    }
