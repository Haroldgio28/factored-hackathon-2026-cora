"""Re-derive the operational confidence thresholds on the held-out suite (task 6.1b, REQ-32/42/47).

Builds the REQ-42 held-out suite (same `build_suite` as 6.1), classifies each case's first-turn
utterance with the trained intent head (`cora.nlu.classifier.learned_intent`) to get a
`(confidence, correct)` pair per case, SELECTS `(tau_escalate, tau_clarify)` on a seeded VALIDATION
slice via `cora.eval.thresholds_eval.derive_thresholds` (reusing the 3.6 cost-matrix sweep), then:

- re-freezes `data/nlu/thresholds.json` with the derived pair (BEFORE the test slice is scored), and
- promotes the pair into `src/cora/policy/rules.yaml` (replacing the Phase-3 stand-in), and
- writes `data/eval/thresholds_derivation.json` recording old stand-in vs new values, the freeze
  order and the validation cost - the note folded into `EVALUATION.md` at 6.9.

Requires the ~0.9 GB raw/curated landing (for `build_suite`) AND the real multilingual MiniLM
encoder (the committed head is 384-d; the offline `StubEncoder` cannot run it). Exits non-zero,
non-destructively, when either is absent - no file is touched in that case.

Usage:
    uv run python scripts/eval/rederive_thresholds.py
    uv run python scripts/eval/rederive_thresholds.py --seed 42 --val-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cora.data.datasource import LocalSource
from cora.eval.builder import DEFAULT_SEED, build_suite
from cora.eval.thresholds_eval import (
    STANDIN_TAU_CLARIFY,
    STANDIN_TAU_ESCALATE,
    derive_thresholds,
    promote_to_rules,
    refreeze,
)
from cora.policy import PolicyEngine


def _classify_suite(scenarios) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001 - list[Scenario]
    """Classify each case's utterance; return (top confidence, top==reference intent) arrays.

    Uses the real trained head via `learned_intent`; raises on a missing/incompatible encoder so
    the CLI fails closed rather than deriving thresholds from stub (64-d) noise.
    """
    from cora.nlu.classifier import learned_intent  # local import: needs the real encoder

    confidences: list[float] = []
    correct: list[bool] = []
    for s in scenarios:
        predicted, confidence = learned_intent(s.utterance, s.language)
        confidences.append(confidence)
        correct.append(predicted == s.intent)
    return np.asarray(confidences, dtype=float), np.asarray(correct, dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-derive operational confidence thresholds (6.1b).")
    ap.add_argument("--dest", type=Path, default=Path("data/raw_parquet"), help="Raw landing root.")
    ap.add_argument("--out", type=Path, default=Path("data/eval"), help="Output directory.")
    ap.add_argument("--thresholds", type=Path, default=Path("data/nlu/thresholds.json"))
    ap.add_argument("--rules", type=Path, default=Path("src/cora/policy/rules.yaml"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed split/sampling seed.")
    ap.add_argument("--val-fraction", type=float, default=0.5, help="Validation share for selection.")
    args = ap.parse_args()

    if not args.dest.exists():
        print(
            f"error: raw/curated landing not found at {args.dest}; cannot build the held-out "
            "suite. Re-run on a data-present host (6.1b stays pending).",
            file=sys.stderr,
        )
        return 1

    source = LocalSource(root=args.dest)
    policy = PolicyEngine.from_yaml()
    scenarios, _ = build_suite(source, policy, seed=args.seed, data_snapshot=str(args.dest))

    try:
        confidences, correct = _classify_suite(scenarios)
    except Exception as exc:  # noqa: BLE001 - fail closed with a clear operator message
        print(
            f"error: could not classify the suite with the real intent head ({exc}); the 384-d "
            "MiniLM encoder is required (the stub cannot run the committed head). 6.1b stays "
            "pending.",
            file=sys.stderr,
        )
        return 1

    result = derive_thresholds(confidences, correct, val_fraction=args.val_fraction, seed=args.seed)

    # Freeze BEFORE the test slice is ever scored (REQ-47), then promote into rules.yaml.
    frozen = refreeze(result.choice, args.thresholds)
    old = promote_to_rules(result.choice, args.rules)

    note = {
        "task": "6.1b",
        "derived_at": datetime.now(UTC).isoformat(),
        "n_held_out": int(confidences.size),
        "n_validation": result.choice.n_validation,
        "n_test_held_back": int(result.test_index.size),
        "old_standin": {"tau_escalate": STANDIN_TAU_ESCALATE, "tau_clarify": STANDIN_TAU_CLARIFY},
        "old_rules_yaml": old,
        "new": {"tau_escalate": result.choice.tau_escalate, "tau_clarify": result.choice.tau_clarify},
        "validation_cost": result.choice.validation_cost,
        "invariant": "tau_escalate <= tau_clarify (POL-060 before POL-070)",
        "freeze_order": "select on validation -> freeze thresholds.json -> promote rules.yaml -> "
        "test slice scored only at 6.6",
        "frozen_json": frozen,
        "label": "offline measurement",
    }
    (args.out / "thresholds_derivation.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"re-derived tau_escalate={result.choice.tau_escalate} tau_clarify={result.choice.tau_clarify} "
        f"(was stand-in {old}); froze {args.thresholds}, promoted {args.rules}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
