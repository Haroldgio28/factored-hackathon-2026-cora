"""Validate the LLM judge against robust-LLM SILVER labels (task 6.5, REQ-45; owner decision 2026-10).

REQ-45 asks for a judge validated against human labels. The owner decided (see
`.agents/tasks/phase6-6.5-owner-decision.md`) NOT to hand-label: the >=50 reference labels are
generated programmatically by a MORE ROBUST model (the owner points `CORA_BEDROCK_JUDGE_MODEL_ID` at
a higher-capability model for the reference run). This harness therefore computes agreement between
the base judge and the robust-LLM reference judge - NOT human validation.

Honesty (REQ-47, mandatory and reviewer-enforced): the agreement number is labelled `simulation`
and every artifact carries, verbatim, the caveat in `SILVER_NOT_HUMAN_NOTE`. The 50 human-labelled
samples REQ-45 asks for remain a documented limitation. This module does NOT read any human-labels
file and does NOT pause the subtask.

Agreement reuses the ONE Cohen's-kappa counting formula (`nlu.goldset.cohen_kappa_pairs`) rather
than re-deriving it (ponytail rung 2), per style field, plus a plain percent-agreement. Both judge
runs mask PII before the LLM call (that is `llm_judge`'s job); this module only pairs their outputs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from cora.agent.llm import LLMClient
from cora.eval.llm_judge import SCORE_FIELDS, StyleScore, score_tone_clarity_handoff
from cora.eval.runner import RunRecord
from cora.nlu.goldset import cohen_kappa_pairs

logger = logging.getLogger("cora.eval.judge_validation")

__all__ = ["MIN_VALIDATION_SAMPLES", "SILVER_NOT_HUMAN_NOTE", "validate_judge"]

# The verbatim honesty caveat REQ-47 requires on the rubric doc and in EVALUATION.md. Defined once
# here so the code, the rubric and the report cannot drift apart.
SILVER_NOT_HUMAN_NOTE = (
    "Agreement is judge-vs-robust-LLM silver labels, not human validation; the 50 human-labelled "
    "samples REQ-45 asks for remain a documented limitation."
)

# REQ-45 asks for >=50 validation samples. The harness runs on whatever it is given but records how
# many usable (both-judges-available) pairs backed the number, and flags when it is under this bar.
MIN_VALIDATION_SAMPLES = 50


def _usable_pairs(
    base: Sequence[StyleScore], reference: Sequence[StyleScore], field_name: str
) -> tuple[list[int], list[int]]:
    """Pair base/reference scores for one field, keeping only records both judges scored.

    Pairs by `(scenario_id, config, repeat)` so a filtered or reordered input still matches, and
    drops any pair where either side is unavailable (a malformed/absent score is not an agreement
    signal). Returns the two aligned integer sequences.
    """
    ref_by_key = {(s.scenario_id, s.config, s.repeat): s for s in reference}
    a: list[int] = []
    b: list[int] = []
    for base_score in base:
        ref = ref_by_key.get((base_score.scenario_id, base_score.config, base_score.repeat))
        if ref is None:
            continue
        base_val = getattr(base_score, field_name)
        ref_val = getattr(ref, field_name)
        if base_val is None or ref_val is None:
            continue
        a.append(base_val)
        b.append(ref_val)
    return a, b


def validate_judge(
    records: Sequence[RunRecord],
    *,
    base_client: LLMClient,
    reference_client: LLMClient,
    base_model_id: str,
    reference_model_id: str,
) -> dict[str, object]:
    """Score `records` with the base and the robust reference judge; return their agreement.

    Runs `score_tone_clarity_handoff` twice over the SAME records (base client/id vs the robust
    reference client/id - the owner points the latter at a higher-capability model) and computes,
    per style field, Cohen's kappa (reused from `goldset`) and percent agreement over the pairs both
    judges could score. The result is a JSON-ready dict labelled `simulation` and carrying the
    verbatim silver-not-human caveat (REQ-47). Does NOT read human labels and does NOT pause.
    """
    base_scores = [score_tone_clarity_handoff(r, client=base_client, model_id=base_model_id) for r in records]
    reference_scores = [
        score_tone_clarity_handoff(r, client=reference_client, model_id=reference_model_id) for r in records
    ]

    per_field: dict[str, object] = {}
    sample_counts: list[int] = []
    for field_name in SCORE_FIELDS:
        a, b = _usable_pairs(base_scores, reference_scores, field_name)
        n = len(a)
        sample_counts.append(n)
        if n == 0:
            per_field[field_name] = {"n": 0, "kappa": None, "percent_agreement": None}
            continue
        percent = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
        per_field[field_name] = {
            "n": n,
            "kappa": cohen_kappa_pairs(a, b),
            "percent_agreement": percent,
        }

    n_samples = min(sample_counts) if sample_counts else 0
    return {
        "label": "simulation",  # REQ-47: judge-vs-robust-LLM, not a measured/human-validated number
        "note": SILVER_NOT_HUMAN_NOTE,
        "base_model_id": base_model_id,
        "reference_model_id": reference_model_id,
        "n_records": len(records),
        "n_usable_samples": n_samples,
        "meets_min_samples": n_samples >= MIN_VALIDATION_SAMPLES,
        "per_field": per_field,
    }


def write_agreement(agreement: dict[str, object], path: Path | str) -> None:
    """Write the agreement result as JSON, creating the directory if absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(agreement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
