"""Build the held-out REQ-42 scenario suite and write it as versioned evidence (task 6.1).

Reads the curated/landing Parquet via `LocalSource`, builds the ~400-case bilingual suite with a
FIXED seed, and writes:

- `data/eval/scenarios.jsonl` - the suite itself (byte-stable for a given seed + data snapshot).
- `data/eval/scenarios_manifest.json` - seed, data snapshot, sampled customer ids and the
  stratification counts (category / language / country / adversarial kind), plus the held-out
  claim: the NLU training ids drawn from `data/nlu/` and whether the suite is disjoint from them.

Exits non-zero (non-destructively) when the raw/curated landing is absent, so a bare host without
the ~0.9 GB Parquet reports the blocker clearly rather than writing an empty suite.

Usage:
    uv run python scripts/eval/build_scenarios.py
    uv run python scripts/eval/build_scenarios.py --seed 42 --dest data/raw_parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cora.data.datasource import LocalSource
from cora.eval.builder import DEFAULT_SEED, build_suite, built_at
from cora.eval.scenarios import write_scenarios
from cora.policy import PolicyEngine


def _nlu_training_customer_ids(nlu_dir: Path) -> list[str]:
    """Return any real `customer_id` present in the NLU training TSVs (held-out claim, REQ-42).

    The gold TSVs are keyed by `scenario_id` and carry NO real customer id, so this is expected to
    be empty - the suite is then trivially disjoint. Recording it makes the held-out claim checked,
    not assumed: if a future training file ever gained a `customer_id` column, it would surface here.
    """
    ids: set[str] = set()
    for tsv in sorted(nlu_dir.glob("*.tsv")):
        header = tsv.read_text(encoding="utf-8").splitlines()[:1]
        if not header or "customer_id" not in header[0].split("\t"):
            continue
        columns = header[0].split("\t")
        col = columns.index("customer_id")
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) > col and fields[col].strip():
                ids.add(fields[col].strip())
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the held-out REQ-42 scenario suite.")
    ap.add_argument("--dest", type=Path, default=Path("data/raw_parquet"), help="Raw landing root.")
    ap.add_argument("--out", type=Path, default=Path("data/eval"), help="Output directory.")
    ap.add_argument("--nlu-dir", type=Path, default=Path("data/nlu"), help="NLU training dir.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed sampling seed.")
    args = ap.parse_args()

    if not args.dest.exists():
        print(f"error: raw/curated landing not found at {args.dest}", file=sys.stderr)
        return 1

    source = LocalSource(root=args.dest)
    policy = PolicyEngine.from_yaml()
    data_snapshot = str(args.dest)
    scenarios, manifest = build_suite(source, policy, seed=args.seed, data_snapshot=data_snapshot)

    scenarios_path = args.out / "scenarios.jsonl"
    write_scenarios(scenarios, scenarios_path)

    training_ids = _nlu_training_customer_ids(args.nlu_dir)
    overlap = sorted(set(manifest.customer_ids) & set(training_ids))
    manifest_dict = asdict(manifest)
    manifest_dict["built_at"] = built_at()
    manifest_dict["held_out_disjoint_from_nlu"] = not overlap
    manifest_dict["nlu_training_customer_ids"] = training_ids
    manifest_dict["held_out_overlap"] = overlap
    (args.out / "scenarios_manifest.json").write_text(
        json.dumps(manifest_dict, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"wrote {manifest.total} scenarios to {scenarios_path} "
        f"(es/pt {manifest.counts_by_language}, countries {manifest.counts_by_country}, "
        f"held-out disjoint={not overlap})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
