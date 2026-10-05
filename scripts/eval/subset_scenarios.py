"""Draw a reproducible stratified subset of the held-out suite for a quota-limited real run.

The full 400-case suite saturates a low Bedrock per-minute quota (turns throttle into template
fallback, contaminating the real numbers). This takes a smaller sample that PRESERVES the strata
(language x category) proportionally, so B1-vs-CORA is still measured on a representative, honestly
labelled slice. Deterministic: a fixed seed and round-trips through the same JSONL loader/writer.

Usage:
    uv run python scripts/eval/subset_scenarios.py --n 100
    uv run python scripts/eval/subset_scenarios.py --in data/eval/scenarios.jsonl \
        --out data/eval/scenarios_subset.jsonl --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from math import floor
from pathlib import Path

from cora.eval.scenarios import load_scenarios, write_scenarios


def main() -> int:
    ap = argparse.ArgumentParser(description="Stratified subset of the scenario suite.")
    ap.add_argument("--in", dest="src", type=Path, default=Path("data/eval/scenarios.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/scenarios_subset.jsonl"))
    ap.add_argument("--n", type=int, default=100, help="Target subset size.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.src.exists():
        print(f"error: suite not found at {args.src} (run build_scenarios first)")
        return 1

    scenarios = load_scenarios(args.src)
    total = len(scenarios)
    if args.n >= total:
        print(f"n={args.n} >= suite size {total}; nothing to subset")
        return 1

    # Group by (language, category) and take a proportional share of each stratum, so the subset
    # mirrors the full suite's composition. A fixed-seed RNG makes the draw reproducible.
    rng = random.Random(args.seed)
    strata: dict[tuple[str, str], list] = defaultdict(list)
    for s in scenarios:
        strata[(s.language, s.category)].append(s)

    frac = args.n / total
    picked: list = []
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        take = max(1, floor(len(group) * frac))  # at least one per non-empty stratum
        picked.extend(group[:take])

    rng.shuffle(picked)
    write_scenarios(picked, args.out)

    by_lang: dict[str, int] = defaultdict(int)
    by_cat: dict[str, int] = defaultdict(int)
    for s in picked:
        by_lang[s.language] += 1
        by_cat[s.category] += 1
    print(
        f"wrote {len(picked)} scenarios (of {total}) to {args.out} "
        f"langs={dict(by_lang)} categories={dict(by_cat)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
