"""Re-derive the operational confidence thresholds on the held-out suite (task 6.1b, REQ-32/42/47).

Phase 3's cost-matrix sweep ran on the tiny 480-row gold split and returned a DEGENERATE pair
(`tau_escalate = tau_clarify = 0.0`): with tau=0 the POL-060/POL-070 low-confidence bands never
fire, so nothing routes to clarify/escalate by confidence. `rules.yaml` therefore ships a
conservative STAND-IN (`tau_escalate=0.40 / tau_clarify=0.60`) that is NOT operational. This module
re-derives the pair on the NEW REQ-42 held-out scenarios, promotes it into `rules.yaml`, and
re-freezes `data/nlu/thresholds.json`.

It reuses `cora.nlu.thresholds` wholesale - the same three-band `assign_bands`
(conf < tau_escalate -> escalate / tau_escalate <= conf < tau_clarify -> clarify / else answer),
the same `CostMatrix` ordering (`unsafe_answer >> unnecessary_escalate >> clarify >> correct`),
the same `select_thresholds` grid sweep, the same hard invariant `tau_escalate <= tau_clarify`, and
the same `freeze`. The ONLY new behaviour here is (a) a seeded validation/test split so the pair is
SELECTED on validation and the test slice stays untouched until 6.6 (REQ-47 freeze discipline), and
(b) a surgical promotion of the frozen pair into `rules.yaml` (3.6 left it a stand-in).

    # ponytail: reuses nlu.thresholds entirely; this file is split + freeze + a two-line YAML edit,
    # no new selection logic. The scenario->(confidence, correct) extraction lives in the CLI
    # (scripts/eval/rederive_thresholds.py) because it needs the real landing + MiniLM encoder;
    # the derivation math here is tested on synthetic score arrays with no data/network.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cora.nlu.thresholds import COST, CostMatrix, ThresholdChoice, freeze, select_thresholds
from cora.policy.engine import PolicyEngine, Thresholds

logger = logging.getLogger("cora.eval.thresholds_eval")

__all__ = [
    "DerivationResult",
    "derive_thresholds",
    "promote_to_rules",
    "refreeze",
]

# The stand-in rules.yaml ships (recorded so the derivation note can show old vs new, REQ-47).
STANDIN_TAU_ESCALATE = 0.40
STANDIN_TAU_CLARIFY = 0.60


@dataclass(frozen=True)
class DerivationResult:
    """The selected pair plus the test slice held back from selection (fed to 6.6, never to it)."""

    choice: ThresholdChoice
    test_confidences: np.ndarray
    test_correct: np.ndarray
    val_index: np.ndarray  # row indices (into the input) used for selection
    test_index: np.ndarray  # row indices held back


def derive_thresholds(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    val_fraction: float = 0.5,
    seed: int = 42,
    cost: CostMatrix = COST,
) -> DerivationResult:
    """Select `(tau_escalate, tau_clarify)` on a seeded VALIDATION slice of the held-out scores.

    `confidences[i]` is the model's top calibrated probability for held-out case i; `correct[i]`
    whether that top prediction matched the reference label. A single `np.random.default_rng(seed)`
    permutes the rows and takes the first `val_fraction` for selection; the remaining rows are the
    TEST slice, returned untouched so 6.6 can score the frozen pair on data that never influenced
    it (REQ-47 freeze-before-test). Selection reuses `nlu.thresholds.select_thresholds`, so the
    cost matrix, grid and the hard `tau_escalate <= tau_clarify` invariant are identical to 3.6.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")
    n = confidences.size
    if n < 2:
        raise ValueError("need at least two held-out cases to split validation/test")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, min(n - 1, round(n * val_fraction)))
    val_index = np.sort(order[:n_val])
    test_index = np.sort(order[n_val:])

    choice = select_thresholds(confidences[val_index], correct[val_index], cost=cost)
    logger.info(
        "re-derived tau_escalate=%.2f tau_clarify=%.2f on %d validation cases (%d held back for test)",
        choice.tau_escalate,
        choice.tau_clarify,
        n_val,
        test_index.size,
    )
    return DerivationResult(
        choice=choice,
        test_confidences=confidences[test_index],
        test_correct=correct[test_index],
        val_index=val_index,
        test_index=test_index,
    )


def refreeze(choice: ThresholdChoice, out: Path | str) -> dict:
    """Re-freeze `data/nlu/thresholds.json` with the newly derived pair (reuses `nlu.thresholds.freeze`)."""
    return freeze(choice, out)


# Matches a `  tau_escalate: 0.40` line (optional trailing comment), capturing the indent, key and
# the current value so the comment block and `version` above the thresholds stay byte-for-byte
# unchanged and the old pair can be reported for the derivation note.
_TAU_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<key>tau_escalate|tau_clarify):\s*(?P<value>[-+0-9.eE]+)\s*(?P<comment>#.*)?$"
)


def promote_to_rules(choice: ThresholdChoice, rules_path: Path | str) -> dict[str, float]:
    """Replace the stand-in `tau_escalate`/`tau_clarify` in `rules.yaml` with the derived pair.

    A surgical two-line text edit (not a YAML round-trip) so every comment, the `version` pin and
    `tau_fraud` are preserved exactly - only the two confidence taus change. The old values are read
    off the matched lines in the same pass (returned for the note). A file missing either threshold
    line fails LOUD here (`ValueError`); after writing, the file is re-parsed through
    `PolicyEngine.from_yaml` so a malformed edit is caught and the engine's `tau_escalate <=
    tau_clarify` invariant is re-asserted on what was actually written.
    """
    rules_path = Path(rules_path)
    new = {"tau_escalate": choice.tau_escalate, "tau_clarify": choice.tau_clarify}
    if new["tau_escalate"] > new["tau_clarify"]:
        raise ValueError(f"derived tau_escalate {new['tau_escalate']} > tau_clarify {new['tau_clarify']}")

    lines = rules_path.read_text(encoding="utf-8").splitlines(keepends=True)
    old: dict[str, float] = {}
    for i, line in enumerate(lines):
        m = _TAU_LINE.match(line.rstrip("\n"))
        if not m:
            continue
        key = m.group("key")
        old[key] = float(m.group("value"))
        newline = "\n" if line.endswith("\n") else ""
        comment = f"  {m.group('comment')}" if m.group("comment") else ""
        lines[i] = f"{m.group('indent')}{key}: {new[key]}{comment}{newline}"
    missing = {"tau_escalate", "tau_clarify"} - set(old)
    if missing:
        raise ValueError(f"rules.yaml did not contain {sorted(missing)} threshold lines to update")
    rules_path.write_text("".join(lines), encoding="utf-8")

    # Fail-closed re-parse: the engine must still load, and the invariant must hold on the real file.
    engine = PolicyEngine.from_yaml(rules_path)
    assert isinstance(engine.thresholds, Thresholds)
    assert engine.thresholds.tau_escalate == new["tau_escalate"]
    assert engine.thresholds.tau_clarify == new["tau_clarify"]
    assert engine.thresholds.tau_escalate <= engine.thresholds.tau_clarify
    logger.info("promoted thresholds into %s: %s -> %s", rules_path, old, new)
    return old
