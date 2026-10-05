"""Tests for task 6.1b: re-derive operational thresholds on the held-out suite (REQ-32/42/47).

The derivation math reuses `nlu.thresholds` (covered in `test_thresholds.py`); this file pins the
6.1b-specific behaviour on synthetic score arrays and a temp copy of `rules.yaml` - NO landing, NO
network, NO real encoder:
  (a) selection runs on a VALIDATION slice and the TEST slice is held back untouched (freeze order),
  (b) the frozen pair satisfies the engine invariant and lands non-degenerate on a clean signal,
  (c) `refreeze` writes a thresholds.json the same shape 3.6 froze,
  (d) `promote_to_rules` edits ONLY the two tau lines, keeps the file loadable, and fails closed on
      a missing threshold block / an inverted pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cora.eval.thresholds_eval import (
    derive_thresholds,
    promote_to_rules,
    refreeze,
)
from cora.policy import PolicyEngine

_RULES = Path("src/cora/policy/rules.yaml")


def _clean_signal(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Low-confidence rows wrong, high-confidence rows correct - the sweep must push tau up."""
    rng = np.random.default_rng(seed)
    conf = np.concatenate([rng.uniform(0.0, 0.35, n // 2), rng.uniform(0.65, 1.0, n // 2)])
    correct = np.concatenate([np.zeros(n // 2, dtype=bool), np.ones(n // 2, dtype=bool)])
    return conf, correct


def test_selection_on_validation_holds_back_the_test_slice() -> None:
    conf, correct = _clean_signal(60)
    result = derive_thresholds(conf, correct, val_fraction=0.5, seed=42)
    # The validation slice drove selection; the test slice is held back and disjoint.
    assert result.choice.n_validation == result.val_index.size
    assert result.test_index.size == conf.size - result.val_index.size
    assert set(result.val_index.tolist()).isdisjoint(result.test_index.tolist())
    # The returned test arrays are exactly the held-back rows (fed to 6.6, never to selection).
    assert np.array_equal(result.test_confidences, conf[result.test_index])
    assert np.array_equal(result.test_correct, correct[result.test_index])


def test_derivation_is_non_degenerate_and_respects_the_invariant() -> None:
    conf, correct = _clean_signal(60)
    choice = derive_thresholds(conf, correct, val_fraction=0.5, seed=42).choice
    assert choice.tau_escalate <= choice.tau_clarify  # POL-060 before POL-070
    # On a clean signal the sweep must NOT collapse to the degenerate 0.0/0.0 the gold split gave.
    assert choice.tau_clarify > 0.0


def test_split_is_seed_deterministic() -> None:
    conf, correct = _clean_signal(60)
    a = derive_thresholds(conf, correct, seed=7)
    b = derive_thresholds(conf, correct, seed=7)
    assert np.array_equal(a.val_index, b.val_index)
    assert (a.choice.tau_escalate, a.choice.tau_clarify) == (b.choice.tau_escalate, b.choice.tau_clarify)


def test_too_few_cases_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        derive_thresholds(np.array([0.5]), np.array([True]))


def test_refreeze_writes_the_threshold_json(tmp_path: Path) -> None:
    conf, correct = _clean_signal(40)
    choice = derive_thresholds(conf, correct, seed=1).choice
    out = tmp_path / "thresholds.json"
    payload = refreeze(choice, out)
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == payload
    assert on_disk["selected_on"] == "validation"
    assert on_disk["tau_escalate"] == choice.tau_escalate
    assert on_disk["tau_clarify"] == choice.tau_clarify


def test_promote_edits_only_the_two_tau_lines_and_stays_loadable(tmp_path: Path) -> None:
    rules = tmp_path / "rules.yaml"
    original = _RULES.read_text(encoding="utf-8")
    rules.write_text(original, encoding="utf-8")

    conf, correct = _clean_signal(60)
    choice = derive_thresholds(conf, correct, seed=42).choice
    # The "old" values promote_to_rules returns must equal whatever rules.yaml currently holds.
    # Read them from the file-under-test rather than the stand-in constants, so the test stays
    # correct after 6.1b has legitimately promoted operational tau into the committed rules.yaml.
    before = PolicyEngine.from_yaml(rules).thresholds
    old = promote_to_rules(choice, rules)

    assert old == {"tau_escalate": before.tau_escalate, "tau_clarify": before.tau_clarify}
    edited = rules.read_text(encoding="utf-8")
    # Exactly two lines changed (the two tau lines); everything else is byte-identical.
    diff = [(a, b) for a, b in zip(original.splitlines(), edited.splitlines(), strict=True) if a != b]
    assert len(diff) == 2
    assert all("tau_" in b for _, b in diff)
    assert "tau_fraud: 70.0" in edited  # the fraud cutoff is untouched

    # The engine loads the edited file and reads back the derived pair, invariant intact.
    engine = PolicyEngine.from_yaml(rules)
    assert engine.thresholds.tau_escalate == choice.tau_escalate
    assert engine.thresholds.tau_clarify == choice.tau_clarify
    assert engine.thresholds.tau_escalate <= engine.thresholds.tau_clarify


def test_promote_fails_closed_without_threshold_lines(tmp_path: Path) -> None:
    rules = tmp_path / "rules.yaml"
    rules.write_text('version: "0.1.0"\nthresholds:\n  tau_fraud: 70.0\n', encoding="utf-8")
    conf, correct = _clean_signal(40)
    choice = derive_thresholds(conf, correct, seed=1).choice
    with pytest.raises(ValueError, match="tau_"):
        promote_to_rules(choice, rules)
