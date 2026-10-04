"""Tests for task 3.6: cost-matrix threshold selection on the validation bands (REQ-32/REQ-47).

The sweep is pure logic (no model), so it is pinned on tiny synthetic score frames:
  (a) band assignment matches the engine's POL-060/070/090 rule order,
  (b) an impossible `tau_escalate > tau_clarify` pair is rejected, never silently reinterpreted,
  (c) the selected pair always satisfies the invariant and minimizes cost toward the safe choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from cora.nlu.thresholds import (
    ANSWER,
    CLARIFY,
    ESCALATE,
    CostMatrix,
    assign_bands,
    select_thresholds,
)


def test_bands_match_pol_rule_order() -> None:
    # tau_escalate=0.4, tau_clarify=0.6 (the rules.yaml placeholders).
    conf = np.array([0.10, 0.40, 0.55, 0.60, 0.95])
    bands = assign_bands(conf, 0.4, 0.6)
    # <0.4 escalate (POL-060); [0.4,0.6) clarify (POL-070); >=0.6 answer (POL-090).
    assert bands == [ESCALATE, CLARIFY, CLARIFY, ANSWER, ANSWER]


def test_impossible_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="tau_escalate"):
        assign_bands(np.array([0.5]), 0.7, 0.3)


def test_sweep_only_scores_valid_pairs_and_asserts_invariant() -> None:
    # A frame where high-confidence rows are correct and low-confidence rows are wrong: answering
    # the low-confidence rows is penalised by unsafe_answer, so the sweep must pull tau up to
    # escalate/clarify them. The chosen pair must satisfy tau_escalate <= tau_clarify.
    rng = np.random.default_rng(0)
    conf = np.concatenate([rng.uniform(0.0, 0.4, 20), rng.uniform(0.6, 1.0, 20)])
    correct = np.concatenate([np.zeros(20, dtype=bool), np.ones(20, dtype=bool)])

    choice = select_thresholds(conf, correct)
    assert choice.tau_escalate <= choice.tau_clarify  # the engine invariant, guaranteed
    # The wrong low-confidence rows must NOT land in the answer band (that would cost unsafe_answer).
    bands = assign_bands(conf, choice.tau_escalate, choice.tau_clarify)
    for b, is_correct in zip(bands, correct, strict=True):
        if not is_correct:
            assert b != ANSWER  # a wrong row answered is the outcome the cost matrix most avoids


def test_unsafe_answer_cost_dominates_escalation() -> None:
    # One row, wrong, at mid confidence. Answering it costs 100; escalating costs 0 (safe). So the
    # minimiser must push tau_escalate above the row's confidence rather than answer it.
    conf = np.array([0.5])
    correct = np.array([False])
    choice = select_thresholds(conf, correct)
    assert assign_bands(conf, choice.tau_escalate, choice.tau_clarify) == [ESCALATE]


def test_empty_validation_split_fails_closed() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_thresholds(np.array([]), np.array([], dtype=bool))


def test_cost_matrix_ordering_is_the_brief_priority() -> None:
    c = CostMatrix()
    assert c.unsafe_answer > c.unnecessary_escalate > c.clarify > c.correct_answer
