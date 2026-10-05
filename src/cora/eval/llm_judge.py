"""The LLM judge: scores ONLY tone / clarity / handoff usefulness (task 6.5, REQ-45).

This is the ONE place in the evaluation where a model's opinion is read, and it is deliberately
boxed in. The deterministic judges (6.4) own every safety-relevant verdict - facts, policy,
disclosure, actions, escalation, handoff completeness - and they are pure code. This judge NEVER
touches any of those: it scores the STYLE of a produced answer (is the tone warm, is it clear, would
the handoff summary actually help a human agent) on a 1-3 ordinal scale. A security review enforces
that nothing here can flip a factual or policy outcome: `StyleScore` carries only the three style
fields, and the judge's output is never fed back into a decision.

Security invariants (reviewer-enforced):
- The produced answer is `mask_pii`-masked BEFORE the LLM call (security steering, REQ-36): the
  judge sees only masked text, exactly like the generator and B1.
- The model reply is untrusted DATA: it is parsed and validated into a typed score, and ANYTHING
  malformed fails closed (`available=False`, no score) rather than being guessed at (REQ-35).
- The model id comes from the injected client / settings only, never a literal (tech/security
  steering, design D3). The same function serves the base judge and the robust reference judge
  (6.5 silver labels); only the injected `LLMClient` and `model_id` differ.

The prompt files (`agent/prompts/llm_judge.{es,pt}.txt`) are versioned and their SHA-256 is recorded
on every score (REQ-44), reusing the `hashlib` pattern from `baselines.py` / `generator.py`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from cora.agent.llm import LLMClient, LLMUnavailable
from cora.eval.runner import RunRecord
from cora.tools.base import mask_pii

logger = logging.getLogger("cora.eval.llm_judge")

__all__ = ["SCORE_FIELDS", "StyleScore", "prompt_hash", "score_tone_clarity_handoff"]

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"

# The ONLY three dimensions the LLM judge may score. Facts/policy/disclosure are 6.4's (deterministic)
# job and must never appear here - a security test asserts the output keys are exactly these.
SCORE_FIELDS: tuple[str, ...] = ("tone", "clarity", "handoff_usefulness")

# The ordinal scale each field uses (1 = poor, 3 = good). A reply with any field outside this set is
# malformed and fails closed.
_SCALE = frozenset({1, 2, 3})


@dataclass(frozen=True)
class StyleScore:
    """One answer's STYLE score - tone/clarity/handoff only, never facts or policy.

    `available` is False when the LLM could not be reached or returned a malformed reply (fail
    closed): the three scores are then `None` so a missing judgement is never read as a zero. The
    `model_id` and `prompt_hash` make the score reproducible/auditable (REQ-44); `label` carries the
    REQ-47 honesty tag.
    """

    scenario_id: str
    config: str
    repeat: int
    language: str
    tone: int | None
    clarity: int | None
    handoff_usefulness: int | None
    model_id: str
    prompt_hash: str
    available: bool = True
    label: str = "simulation"  # REQ-47: an LLM opinion, not a measured outcome

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "config": self.config,
            "repeat": self.repeat,
            "language": self.language,
            "tone": self.tone,
            "clarity": self.clarity,
            "handoff_usefulness": self.handoff_usefulness,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "available": self.available,
            "label": self.label,
        }


def _prompt_path(language: str) -> Path:
    path = _PROMPTS_DIR / f"llm_judge.{language}.txt"
    if not path.exists():  # es/pt ship; any other language fails the pairing test, not a silent swap
        raise FileNotFoundError(f"missing llm_judge prompt for language {language!r}")
    return path


def prompt_hash(language: str) -> str:
    """SHA-256 of the versioned judge prompt file bytes (REQ-44 run metadata)."""
    return hashlib.sha256(_prompt_path(language).read_bytes()).hexdigest()


def _parse_score(text: str) -> dict[str, int] | None:
    """Parse the model reply into the three style scores, or `None` if malformed (fail closed).

    The reply is untrusted DATA: it must be a JSON object carrying exactly the three `SCORE_FIELDS`,
    each an int in {1,2,3}. Anything else (non-JSON, missing/extra key, out-of-range value) returns
    `None` so the caller records an unavailable score rather than a guessed one (REQ-35).
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.info("llm judge: non-JSON reply, failing closed")
        return None
    if not isinstance(payload, dict) or set(payload) != set(SCORE_FIELDS):
        logger.info("llm judge: reply keys are not exactly the style fields, failing closed")
        return None
    scores: dict[str, int] = {}
    for field_name in SCORE_FIELDS:
        value = payload[field_name]
        # bool is an int subclass; reject it so `true`/`false` is not read as 1/0.
        if isinstance(value, bool) or not isinstance(value, int) or value not in _SCALE:
            logger.info("llm judge: field %s out of the 1-3 scale, failing closed", field_name)
            return None
        scores[field_name] = value
    return scores


def score_tone_clarity_handoff(
    record: RunRecord,
    *,
    client: LLMClient,
    model_id: str,
    max_tokens: int = 64,
) -> StyleScore:
    """Score ONE run record's produced answer for tone/clarity/handoff usefulness only (REQ-45).

    Masks the produced text, fills the versioned rubric prompt for the record's language, makes one
    LLM call via the injected `client`, and validates the reply into a `StyleScore`. Fails closed to
    `available=False` (no scores) on an unavailable LLM or a malformed reply. `model_id` is recorded
    as given (the caller reads it from settings - never a literal here); the SAME function is reused
    for the base judge and the robust reference judge, differing only by the injected client/id.
    """
    language = record.language if record.language in ("es", "pt") else "es"
    answer = mask_pii(record.produced_text or "")
    prompt = _prompt_path(language).read_text(encoding="utf-8").format(answer=answer)
    phash = prompt_hash(language)

    def _unavailable() -> StyleScore:
        return StyleScore(
            scenario_id=record.scenario_id,
            config=record.config,
            repeat=record.repeat,
            language=record.language,
            tone=None,
            clarity=None,
            handoff_usefulness=None,
            model_id=model_id,
            prompt_hash=phash,
            available=False,
        )

    try:
        reply = client.complete(system="", user=prompt, max_tokens=max_tokens)
    except LLMUnavailable:
        logger.info("llm judge: LLM unavailable for %s/%s", record.scenario_id, record.config)
        return _unavailable()

    scores = _parse_score(reply.strip())
    if scores is None:
        return _unavailable()
    return StyleScore(
        scenario_id=record.scenario_id,
        config=record.config,
        repeat=record.repeat,
        language=record.language,
        tone=scores["tone"],
        clarity=scores["clarity"],
        handoff_usefulness=scores["handoff_usefulness"],
        model_id=model_id,
        prompt_hash=phash,
    )
