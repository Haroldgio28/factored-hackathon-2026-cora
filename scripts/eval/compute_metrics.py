"""Score the run records and aggregate the report metrics (task 6.6, REQ-43).

Loads the 6.3 run records (`data/eval/runs.jsonl`) and the 6.1 scenario suite, scores every record
with the 6.4 deterministic judges, then folds the verdicts + records into the 6.6 metric set
(SAR / attempted / containment, escalation precision/recall with missed+unnecessary counts, unsafe
outcomes by type with Wilson CI + rule-of-three, p50/p95 latency, cost per attempted/per SAR) and
writes `data/eval/metrics.json`.

Cost is an offline ESTIMATE (REQ-47): the runner captures no provider token usage, so the per-token
prices below are the caller-supplied published Bedrock rates and tokens are approximated from text
length; all of it is recorded in `metrics.json` under `cost_assumptions`.

Exits non-zero (non-destructively) when the runs or the suite are absent, so a partial pipeline
reports the blocker rather than writing empty metrics.

Usage:
    uv run python scripts/eval/compute_metrics.py
    uv run python scripts/eval/compute_metrics.py --input-usd-per-1k 0.0008 --output-usd-per-1k 0.004
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cora.eval.judges import score_runs
from cora.eval.metrics import CostAssumptions, aggregate_metrics
from cora.eval.runner import RunRecord
from cora.eval.scenarios import load_scenarios


def _load_runs(path: Path) -> list[RunRecord]:
    """Load run records from JSONL (the inverse of `runner.write_runs`)."""
    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(RunRecord(**json.loads(line)))
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate the held-out evaluation metrics (task 6.6).")
    ap.add_argument("--runs", type=Path, default=Path("data/eval/runs.jsonl"), help="Run records.")
    ap.add_argument("--suite", type=Path, default=Path("data/eval/scenarios.jsonl"), help="Scenario suite.")
    ap.add_argument("--out", type=Path, default=Path("data/eval/metrics.json"), help="Output JSON.")
    ap.add_argument("--input-usd-per-1k", type=float, default=0.0, help="Published input price / 1k tokens.")
    ap.add_argument(
        "--output-usd-per-1k", type=float, default=0.0, help="Published output price / 1k tokens."
    )
    args = ap.parse_args()

    if not args.runs.exists():
        print(f"error: run records not found at {args.runs} (run run_eval first)", file=sys.stderr)
        return 1
    if not args.suite.exists():
        print(f"error: scenario suite not found at {args.suite} (run build_scenarios first)", file=sys.stderr)
        return 1

    records = _load_runs(args.runs)
    scenarios = load_scenarios(args.suite)
    judgement_rows = score_runs(records, scenarios)
    assumptions = CostAssumptions(
        input_usd_per_1k=args.input_usd_per_1k, output_usd_per_1k=args.output_usd_per_1k
    )
    metrics = aggregate_metrics(judgement_rows, records, cost_assumptions=assumptions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    configs = ", ".join(sorted(metrics["configs"]))  # type: ignore[arg-type]
    print(f"wrote metrics for configs [{configs}] to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
