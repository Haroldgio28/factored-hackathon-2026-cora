"""Cost-matrix threshold selection for the policy confidence bands (task 3.6, REQ-32/REQ-47).

The policy engine does NOT argmin three actions on confidence; it evaluates `rules.yaml` top-down
and the FIRST match wins (verified in `policy/engine.py`): POL-060 `confidence_below_escalate`
fires first (-> escalate), then POL-070 `confidence_below_clarify_or_ambiguous_entity` (-> clarify),
else POL-090 answers. So the real decision surface is three confidence BANDS, with escalate as the
*lowest*-confidence outcome:

    conf < tau_escalate                 -> escalate   (POL-060)
    tau_escalate <= conf < tau_clarify  -> clarify     (POL-070)
    conf >= tau_clarify                 -> answer      (POL-090)

`select_thresholds` sweeps candidate `(tau_escalate, tau_clarify)` pairs over a grid, assigns each
validation prediction to its band, and sums the per-row cost under the named cost matrix. The
ordering `unsafe_answer >> unnecessary_escalate >> clarify` encodes the brief's priority directly
(an answer that should have escalated/clarified is the thing we most avoid). The invariant
`tau_escalate <= tau_clarify` is a HARD grid constraint (violating pairs are rejected before
scoring) AND a final `assert`, so the frozen pair is always representable by POL-060/POL-070 - the
engine's ordering can never be handed a `tau_escalate > tau_clarify` it cannot express.

Scope: only `tau_clarify` and `tau_escalate` are chosen here. `tau_fraud` (POL-040) is a data-driven
fraud-score cutoff on a 0-100 scale, not an intent confidence, and is NOT re-derived.

    # ponytail: a dense O(grid^2 * n) sweep over a ~21x21 grid on a <100-row validation split -
    # a few thousand cheap comparisons. No optimizer, no gradient; the surface is tiny and the
    # exhaustive sweep is both the simplest and the most auditable choice (every candidate's cost
    # can be printed). Widen GRID or switch to a coarse-then-fine search only if the grid ever grows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("cora.nlu.thresholds")

__all__ = ["COST", "GRID", "CostMatrix", "ThresholdChoice", "assign_bands", "select_thresholds"]

# Band labels (mirror the policy decisions the three rules produce; kept as plain strings so this
# module has no import dependency on the engine - the mapping to POL-060/070/090 is the docstring).
ESCALATE = "escalate"
CLARIFY = "clarify"
ANSWER = "answer"


@dataclass(frozen=True)
class CostMatrix:
    """Per-outcome costs. The ordering (not the absolute values) encodes the brief's priority:
    an unsafe automated answer is 10x worse than an unnecessary escalation, which is 10x worse than
    a clarification turn; a correct answer is free."""

    unsafe_answer: float = 100.0  # answered when the top prediction was wrong (should not automate)
    unnecessary_escalate: float = 10.0  # escalated a turn the model would have gotten right
    clarify: float = 1.0  # a clarification turn (mild friction, always "safe")
    correct_answer: float = 0.0  # answered and the top prediction was right


COST = CostMatrix()

# Candidate grid for both taus: 0.00..1.00 step 0.05. rules.yaml ships 0.40/0.60 as placeholders;
# this grid brackets them with room on both sides.
GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0 + 1e-9, 0.05))


@dataclass(frozen=True)
class ThresholdChoice:
    """The frozen pair plus the validation cost it was selected on (evidence for the report)."""

    tau_escalate: float
    tau_clarify: float
    validation_cost: float
    n_validation: int
    grid: tuple[float, ...]
    cost_matrix: CostMatrix


def assign_bands(confidences: np.ndarray, tau_escalate: float, tau_clarify: float) -> list[str]:
    """Map each confidence to its policy band exactly as the engine's rule order would (POL-060/070/090).

    Fail-closed on an impossible pair: `tau_escalate > tau_clarify` cannot be expressed by the
    top-down rules (POL-060 would swallow the clarify band), so it is rejected rather than silently
    reinterpreted.
    """
    if tau_escalate > tau_clarify:
        raise ValueError(f"tau_escalate ({tau_escalate}) must be <= tau_clarify ({tau_clarify})")
    bands: list[str] = []
    for conf in confidences:
        if conf < tau_escalate:
            bands.append(ESCALATE)
        elif conf < tau_clarify:
            bands.append(CLARIFY)
        else:
            bands.append(ANSWER)
    return bands


def _band_cost(band: str, correct: bool, cost: CostMatrix) -> float:
    if band == ANSWER:
        return cost.correct_answer if correct else cost.unsafe_answer
    if band == CLARIFY:
        return cost.clarify
    # ESCALATE: safe either way, but costly when the model would have answered correctly.
    return cost.unnecessary_escalate if correct else 0.0


def _total_cost(
    confidences: np.ndarray, correct: np.ndarray, tau_escalate: float, tau_clarify: float, cost: CostMatrix
) -> float:
    bands = assign_bands(confidences, tau_escalate, tau_clarify)
    return float(sum(_band_cost(b, bool(c), cost) for b, c in zip(bands, correct, strict=True)))


def select_thresholds(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    cost: CostMatrix = COST,
    grid: tuple[float, ...] = GRID,
) -> ThresholdChoice:
    """Pick (tau_escalate, tau_clarify) minimizing expected cost on the VALIDATION split.

    `confidences[i]` is the model's top calibrated probability for validation row i; `correct[i]`
    is whether that top prediction matched the gold label. Only pairs with
    `tau_escalate <= tau_clarify` are scored (the invariant the engine requires); ties break toward
    the lower (safer, more-escalating) `tau_escalate` then the lower `tau_clarify`.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")
    if confidences.size == 0:
        raise ValueError("cannot select thresholds on an empty validation split")

    best: tuple[float, float, float] | None = None  # (cost, tau_escalate, tau_clarify)
    for tau_e in grid:
        for tau_c in grid:
            if tau_e > tau_c:  # hard constraint: reject before scoring
                continue
            total = _total_cost(confidences, correct, tau_e, tau_c, cost)
            cand = (total, tau_e, tau_c)
            if best is None or cand < best:
                best = cand

    assert best is not None  # the grid always contains at least (0.0, 0.0)
    total, tau_e, tau_c = best
    # Final guard: the frozen pair MUST satisfy the engine invariant.
    assert tau_e <= tau_c, f"selected tau_escalate {tau_e} > tau_clarify {tau_c}"
    logger.info("selected tau_escalate=%.2f tau_clarify=%.2f (val cost=%.1f)", tau_e, tau_c, total)
    return ThresholdChoice(
        tau_escalate=tau_e,
        tau_clarify=tau_c,
        validation_cost=total,
        n_validation=int(confidences.size),
        grid=grid,
        cost_matrix=cost,
    )


def freeze(choice: ThresholdChoice, out: Path | str) -> dict:
    """Write the chosen pair to `thresholds.json` BEFORE the test split is touched (REQ-47 honesty)."""
    payload = {
        "tau_escalate": choice.tau_escalate,
        "tau_clarify": choice.tau_clarify,
        "selected_on": "validation",
        "validation_cost": choice.validation_cost,
        "n_validation": choice.n_validation,
        "cost_matrix": {
            "unsafe_answer": choice.cost_matrix.unsafe_answer,
            "unnecessary_escalate": choice.cost_matrix.unnecessary_escalate,
            "clarify": choice.cost_matrix.clarify,
            "correct_answer": choice.cost_matrix.correct_answer,
        },
        "grid": {"start": choice.grid[0], "stop": choice.grid[-1], "n": len(choice.grid)},
        "invariant": "tau_escalate <= tau_clarify (POL-060 before POL-070); asserted at selection",
        "rules": {"POL-060": "confidence_below_escalate", "POL-070": "confidence_below_clarify"},
        "note": (
            "Frozen on the validation split before the test split was evaluated; feeds the "
            "rules.yaml POL-060/POL-070 placeholders. tau_fraud (POL-040) is NOT re-derived here."
        ),
    }
    out = Path(out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("froze thresholds to %s", out)
    return payload
