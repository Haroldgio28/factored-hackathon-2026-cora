"""Compute the historical B0 baseline (FCR/AHT by category) and write it (task 6.8, REQ-41/47).

Reads the local Parquet landing through `LocalSource` and writes `data/eval/b0_historical.json`
labelled `offline measurement` with the "not comparable 1:1" caveat.

Fail-closed (never fabricate): if the `call_center_interactions` landing is absent on this host the
script prints a clear "data not present - B0 paused" message and exits non-zero, so a partial
pipeline reports the blocker rather than inventing numbers. The ~0.9 GB landing is gitignored and
lives on the owner's data host (the MAIN repo root), not in a bare worktree.

Usage:
    uv run python scripts/eval/historical_baseline.py
    uv run python scripts/eval/historical_baseline.py --root data/raw_parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cora.data.datasource import LocalSource
from cora.eval.historical import compute_b0


def main() -> int:
    ap = argparse.ArgumentParser(description="Historical B0 baseline (FCR/AHT by category, task 6.8).")
    ap.add_argument("--root", type=Path, default=Path("data/raw_parquet"), help="Parquet landing root.")
    ap.add_argument("--out", type=Path, default=Path("data/eval/b0_historical.json"), help="Output JSON.")
    args = ap.parse_args()

    # The call-center fact lives under <root>/call_center_interactions/. Absent -> paused, not faked.
    fact_dir = args.root / "call_center_interactions"
    if not fact_dir.exists():
        print(
            f"B0 paused: call-center data not present at {fact_dir}. "
            "Run on the data host (MAIN repo root) where the landing exists; do not fabricate B0.",
            file=sys.stderr,
        )
        return 1

    b0 = compute_b0(LocalSource(root=args.root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(b0, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    n_cat = len(b0["by_category"])  # type: ignore[arg-type]
    print(f"wrote B0 historical baseline over {n_cat} categories to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
