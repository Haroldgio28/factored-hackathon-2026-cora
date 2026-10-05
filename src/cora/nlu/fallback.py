"""Classifier-unavailable fallback (task 5.2, REQ-40, design section 10).

When the learned intent classifier cannot answer (missing artifact, load error, encode error),
the turn must still be classified-or-safely-clarified rather than crash. The documented fallback
(design section 10, `nlu/baselines.py` docstring) is the deterministic `KeywordBaseline`: an
ordered es/pt regex table that already covers the 16 intents. This module wraps it with the
conservative confidence the policy expects, so 5.3's `_classify` except-path calls one function.

Confidence uses the rules.yaml stand-in bands (0.40 escalate / 0.60 answer): a keyword match is a
real but low-certainty signal, so it is reported ABOVE the escalate band but BELOW the answer band
(0.50) - enough to route an action/escalation intent through the policy, not enough to answer
without clarifying. No match -> `OTHER` at 0.0, which the policy routes to clarify (the safe
default, never a guessed answer). Thresholds are NOT re-derived here (Phase 6 owns thresholds); the
stand-in bands are the documented interim contract.

# ponytail: reuse KeywordBaseline (already the documented fallback), do NOT re-derive thresholds.
"""

from __future__ import annotations

from functools import lru_cache

from cora.nlu.baselines import KeywordBaseline
from cora.nlu.labels import Intent

__all__ = ["MATCH_CONFIDENCE", "keyword_fallback_intent"]

# Conservative confidence for a keyword hit: above the 0.40 escalate band, below the 0.50 clarify
# band (rules.yaml stand-in), so an action/escalation intent routes but a read still clarifies.
MATCH_CONFIDENCE = 0.45


@lru_cache(maxsize=1)
def _baseline() -> KeywordBaseline:
    """One shared compiled KeywordBaseline (regex compilation is done once)."""
    return KeywordBaseline()


def keyword_fallback_intent(masked_utterance: str, language: str) -> tuple[Intent, float]:
    """Classify a masked utterance with the keyword baseline when the learned model is unavailable.

    Returns `(intent, MATCH_CONFIDENCE)` on a keyword match and `(Intent.OTHER, 0.0)` otherwise, so
    the policy either routes the recognised intent or clarifies - never guesses an answer.
    """
    intent = _baseline().predict_one(masked_utterance, language)
    confidence = MATCH_CONFIDENCE if intent is not Intent.OTHER else 0.0
    return intent, confidence


if __name__ == "__main__":  # self-check: known es/pt keyword utterances hit the expected intent
    es_intent, es_conf = keyword_fallback_intent("quiero congelar mi tarjeta", "es")
    assert es_intent is Intent.A1 and es_conf == MATCH_CONFIDENCE, (es_intent, es_conf)
    pt_intent, pt_conf = keyword_fallback_intent("quero bloquear meu cartão", "pt")
    assert pt_intent is Intent.A1 and pt_conf == MATCH_CONFIDENCE, (pt_intent, pt_conf)
    other_intent, other_conf = keyword_fallback_intent("xyzzy nonsense words", "es")
    assert other_intent is Intent.OTHER and other_conf == 0.0, (other_intent, other_conf)
    print("nlu keyword-fallback self-check OK")
