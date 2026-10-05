"""Run B1 and CORA over the held-out suite with faults on, x3 repeats, and write run records (6.3).

Loads the scenario suite built by `build_scenarios.py`, reads the curated/landing Parquet via
`LocalSource`, runs every scenario through each config `--repeats` times with a seeded fault rate,
and writes `data/eval/runs.jsonl` (the per-(scenario, config, repeat) `RunRecord`s the 6.4 judges
consume). The run versions (model id, prompt hashes, commit, data snapshot) are stamped on every
record for REQ-44 reproducibility; the model id comes from settings (the stub when Bedrock is
unconfigured), never hard-coded.

Exits non-zero (non-destructively) when the suite or the landing is absent, so a bare host reports
the blocker clearly rather than writing empty runs.

Usage:
    uv run python scripts/eval/run_eval.py
    uv run python scripts/eval/run_eval.py --repeats 3 --fault-rate 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cora.data.datasource import LocalSource
from cora.eval.baselines import _prompt_hash as _b1_prompt_hash
from cora.eval.runner import RunVersions, code_commit, run_suite, write_runs
from cora.eval.scenarios import load_scenarios
from cora.settings import get_settings


def main() -> int:
    ap = argparse.ArgumentParser(description="Run B1 and CORA over the held-out suite (task 6.3).")
    ap.add_argument("--suite", type=Path, default=Path("data/eval/scenarios.jsonl"), help="Scenario suite.")
    ap.add_argument("--dest", type=Path, default=Path("data/raw_parquet"), help="Raw landing root.")
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs.jsonl"), help="Output JSONL.")
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per (scenario, config).")
    ap.add_argument("--fault-rate", type=float, default=0.1, help="Per-read fault probability [0,1].")
    ap.add_argument("--seed", type=int, default=42, help="Fault seed (whole run reproduces from it).")
    args = ap.parse_args()

    if not args.suite.exists():
        print(f"error: scenario suite not found at {args.suite} (run build_scenarios first)", file=sys.stderr)
        return 1
    if not args.dest.exists():
        print(f"error: raw/curated landing not found at {args.dest}", file=sys.stderr)
        return 1

    settings = get_settings()
    scenarios = load_scenarios(args.suite)
    versions = RunVersions(
        model_id=settings.bedrock_model_id or "stub",
        code_commit=code_commit(),
        data_snapshot=str(args.dest),
        b1_prompt_hash=_b1_prompt_hash("es"),
        cora_prompt_hash=None,  # CORA records its polish-prompt hash per turn in the trace, not here
    )
    records = run_suite(
        scenarios,
        LocalSource(root=args.dest),
        repeats=args.repeats,
        seed=args.seed,
        fault_rate=args.fault_rate,
        versions=versions,
    )
    write_runs(records, args.out)
    print(
        f"wrote {len(records)} run records to {args.out} "
        f"(model={versions.model_id}, commit={versions.code_commit[:8]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
