"""Assemble documentation/reports/EVALUATION.md from the eval artifacts (task 6.9, REQ-44/47).

Loads whatever JSON artifacts exist under `data/eval/` and renders the report with
`cora.eval.report.render_report`. Unlike the other eval stages this one NEVER fails closed on a
missing input: a report that renders only the sections whose artifacts exist (with clearly-labelled
`pending data-host run` placeholders for the rest) is the whole point - it is regenerable, so a
later data-host run fills the placeholders in. Numbers are labelled per REQ-47 by the renderer.

Usage:
    uv run python scripts/eval/make_report.py
    uv run python scripts/eval/make_report.py --eval-dir data/eval --out documentation/reports/EVALUATION.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cora.eval.report import render_report

# Artifact name -> filename under the eval dir. Any absent file renders as a placeholder.
_ARTIFACTS = {
    "metrics": "metrics.json",
    "fairness": "fairness.json",
    "b0": "b0_historical.json",
    "thresholds": "thresholds_derivation.json",
    "manifest": "scenarios_manifest.json",
    "agreement": "judge_agreement.json",
}


def _load(path: Path) -> Any | None:
    """Load a JSON artifact, or None when it is absent (a pending data-host stage)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the Phase 6 evaluation report (task 6.9).")
    ap.add_argument("--eval-dir", type=Path, default=Path("data/eval"), help="Artifact directory.")
    ap.add_argument(
        "--out", type=Path, default=Path("documentation/reports/EVALUATION.md"), help="Report path."
    )
    args = ap.parse_args()

    artifacts = {name: _load(args.eval_dir / filename) for name, filename in _ARTIFACTS.items()}
    report = render_report(artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    present = sorted(name for name, value in artifacts.items() if value is not None)
    pending = sorted(name for name, value in artifacts.items() if value is None)
    print(f"wrote {args.out} (present: {present or 'none'}; pending data-host run: {pending or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
