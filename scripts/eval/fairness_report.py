"""Break the held-out eval outcomes down by subgroup and write the fairness report (task 6.7, REQ-46).

Loads the 6.3 run records (`data/eval/runs.jsonl`) and the 6.1 scenario suite, scores every record
with the 6.4 deterministic judges, then breaks SAR / unsafe rate / escalation / latency down by
language, country (regional-accent proxy) and segment, flagging gaps beyond the stated tolerance
and marking thin subgroups as small-sample. Writes `data/eval/fairness.json`.

Exits non-zero (non-destructively) when the runs or the suite are absent, so a partial pipeline
reports the blocker rather than writing an empty breakdown.

Usage:
    uv run python scripts/eval/fairness_report.py
    uv run python scripts/eval/fairness_report.py --min-n 50 --rate-tolerance 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cora.eval.fairness import fairness_breakdown
from cora.eval.judges import score_runs
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
    ap = argparse.ArgumentParser(description="Fairness breakdown of the held-out evaluation (task 6.7).")
    ap.add_argument("--runs", type=Path, default=Path("data/eval/runs.jsonl"), help="Run records.")
    ap.add_argument("--suite", type=Path, default=Path("data/eval/scenarios.jsonl"), help="Scenario suite.")
    ap.add_argument("--out", type=Path, default=Path("data/eval/fairness.json"), help="Output JSON.")
    ap.add_argument(
        "--min-n", type=int, default=30, help="Below this many records a subgroup is small-sample."
    )
    ap.add_argument("--rate-tolerance", type=float, default=0.10, help="Best-vs-worst gap flag threshold.")
    args = ap.parse_args()

    if not args.runs.exists():
        print(f"error: run records not found at {args.runs} (run run_eval first)", file=sys.stderr)
        return 1
    if not args.suite.exists():
        print(f"error: scenario suite not found at {args.suite} (run build_scenarios first)", file=sys.stderr)
        return 1

    records = _load_runs(args.runs)
    scenarios = load_scenarios(args.suite)
    rows = score_runs(records, scenarios)
    breakdown = fairness_breakdown(
        rows, records, scenarios, min_n=args.min_n, rate_tolerance=args.rate_tolerance
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    dims = ", ".join(sorted(breakdown["dimensions"]))  # type: ignore[arg-type]
    print(f"wrote fairness breakdown over [{dims}] to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
