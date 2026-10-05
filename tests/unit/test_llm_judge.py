"""Tests for the LLM judge + silver-label validation harness (task 6.5, REQ-45).

Everything runs offline against `StubLLMClient` - no network, no Bedrock. The tests pin the
security invariants a review enforces: the judge scores ONLY tone/clarity/handoff (never
facts/policy), masks PII before the call, fails closed on a malformed reply, reads the model id
from injection (no literal), and the validation harness reuses the gold-set kappa and carries the
`simulation` label + the verbatim silver-not-human note.
"""

from __future__ import annotations

import json

from cora.agent.llm import LLMClient, LLMUnavailable, StubLLMClient
from cora.eval.judge_validation import (
    MIN_VALIDATION_SAMPLES,
    SILVER_NOT_HUMAN_NOTE,
    validate_judge,
)
from cora.eval.llm_judge import SCORE_FIELDS, score_tone_clarity_handoff
from cora.eval.runner import RunRecord


def _record(scenario_id: str, text: str, language: str = "es") -> RunRecord:
    return RunRecord(
        scenario_id=scenario_id,
        config="cora",
        repeat=0,
        language=language,
        category="normal",
        expected_decision="ANSWER",
        expected_escalation=False,
        produced_text=text,
    )


def _good_reply() -> str:
    return json.dumps({"tone": 3, "clarity": 2, "handoff_usefulness": 1})


def test_judge_returns_only_style_fields_never_facts_or_policy() -> None:
    client = StubLLMClient(default=_good_reply())
    score = score_tone_clarity_handoff(
        _record("S1", "Su saldo es 100 USD."), client=client, model_id="base-id"
    )
    assert score.available
    assert (score.tone, score.clarity, score.handoff_usefulness) == (3, 2, 1)
    # The serialized score carries no facts/policy verdict - only the three style keys (+ metadata).
    keys = set(score.to_dict())
    assert set(SCORE_FIELDS) <= keys
    assert not ({"facts", "policy", "decision", "grounding", "rule_id"} & keys)


def test_model_id_comes_from_injection_not_a_literal() -> None:
    client = StubLLMClient(default=_good_reply())
    score = score_tone_clarity_handoff(_record("S1", "hola"), client=client, model_id="owner-chosen-id")
    assert score.model_id == "owner-chosen-id"  # recorded as given, no hard-coded id


def test_masks_pii_before_the_llm_call() -> None:
    # A spy client captures the user prompt; the raw email must never reach it (security steering).
    seen: list[str] = []

    class _Spy:
        def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
            seen.append(user)
            return _good_reply()

    spy: LLMClient = _Spy()
    score_tone_clarity_handoff(
        _record("S1", "Escríbeme a juan.perez@mail.com por favor"), client=spy, model_id="base-id"
    )
    assert seen, "the judge must call the LLM"
    assert "juan.perez@mail.com" not in seen[0]
    assert "j***@mail.com" in seen[0]  # mask_pii applied


def test_malformed_reply_fails_closed() -> None:
    for bad in [
        "not json",
        json.dumps({"tone": 3}),
        json.dumps({"tone": 5, "clarity": 2, "handoff_usefulness": 1}),
        json.dumps({"tone": True, "clarity": 2, "handoff_usefulness": 1}),
        json.dumps({"tone": 2, "clarity": 2, "handoff_usefulness": 1, "facts": 3}),
    ]:
        client = StubLLMClient(default=bad)
        score = score_tone_clarity_handoff(_record("S1", "hola"), client=client, model_id="base-id")
        assert not score.available
        assert score.tone is None and score.clarity is None and score.handoff_usefulness is None


def test_llm_unavailable_fails_closed() -> None:
    class _Down:
        def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
            raise LLMUnavailable("down")

    score = score_tone_clarity_handoff(_record("S1", "hola"), client=_Down(), model_id="base-id")
    assert not score.available


def test_both_languages_scored() -> None:
    client = StubLLMClient(default=_good_reply())
    es = score_tone_clarity_handoff(_record("S-es", "hola", "es"), client=client, model_id="base-id")
    pt = score_tone_clarity_handoff(_record("S-pt", "ola", "pt"), client=client, model_id="base-id")
    assert es.available and pt.available  # es and pt prompts both present or a FileNotFoundError


def test_validation_reuses_kappa_and_labels_silver_not_human() -> None:
    # Base judge always says tone=3; reference judge always says tone=3 too -> perfect agreement.
    records = [_record(f"S{i}", f"respuesta numero {i}") for i in range(3)]
    base = StubLLMClient(default=json.dumps({"tone": 3, "clarity": 2, "handoff_usefulness": 1}))
    reference = StubLLMClient(default=json.dumps({"tone": 3, "clarity": 2, "handoff_usefulness": 1}))
    out = validate_judge(
        records,
        base_client=base,
        reference_client=reference,
        base_model_id="base-id",
        reference_model_id="robust-id",
    )
    assert out["label"] == "simulation"  # REQ-47: not a measured/human number
    assert out["note"] == SILVER_NOT_HUMAN_NOTE  # verbatim silver-not-human caveat
    assert out["base_model_id"] == "base-id" and out["reference_model_id"] == "robust-id"
    # Perfect agreement on every field.
    for field_name in SCORE_FIELDS:
        cell = out["per_field"][field_name]
        assert cell["n"] == 3
        assert cell["percent_agreement"] == 1.0
        assert cell["kappa"] == 1.0  # all identical -> perfect by convention
    # Under the 50-sample bar, honestly flagged (not fabricated).
    assert out["n_usable_samples"] == 3
    assert out["meets_min_samples"] is False
    assert MIN_VALIDATION_SAMPLES == 50


def test_validation_drops_unavailable_pairs_from_agreement() -> None:
    # The reference judge returns a malformed reply for one record -> that pair is dropped, not
    # counted as a disagreement (a missing score is not a signal).
    records = [_record("S0", "respuesta cero"), _record("S1", "respuesta uno")]
    base = StubLLMClient(default=json.dumps({"tone": 2, "clarity": 2, "handoff_usefulness": 2}))
    reference = StubLLMClient(
        responses={"respuesta cero": json.dumps({"tone": 2, "clarity": 2, "handoff_usefulness": 2})},
        default="garbage",
    )
    out = validate_judge(
        records, base_client=base, reference_client=reference, base_model_id="b", reference_model_id="r"
    )
    for field_name in SCORE_FIELDS:
        assert out["per_field"][field_name]["n"] == 1  # only the matched, both-available pair
