"""Language detection for the es/pt customer channel (task 3.7, REQ-18).

One public seam: `detect(text) -> LanguageResult(lang, confidence, low_confidence)`, where `lang`
is normalized to the closed set `{es, pt, other}` so callers never see a third language. On
low or mixed confidence the result is flagged `low_confidence=True`; the design's rule is to then
keep the session language and confirm, never to guess (fail closed, REQ-18).

Backends, in order of preference (design section 7a):

- `fasttext-wheel` loading the COMPRESSED `lid.176.ftz` (~917 KB, NOT the ~126 MB `.bin`). The
  model file is not committed (model-card records the URL); it is cached under `data/nlu/` and
  downloaded once on first offline use.
- `lingua-language-detector` (pure Python, no native build) if no fasttext wheel installs on the
  host - behind the same `detect` signature so callers do not change.
- a deterministic `StubDetector` (short es/pt stop-word counts) selected under `CORA_NLU_STUB=1`
  so tests run with no download and no network.

    # ponytail: a stop-word stub is enough for the TESTED path; the real lid.176 model is the
    #           offline/production backend. If the stub ever needs more than es/pt discrimination,
    #           it should not grow - wire the real backend instead.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("cora.nlu.language")

__all__ = ["SUPPORTED", "LanguageResult", "detect", "is_supported", "stub_enabled"]

# Normalized output vocabulary. Anything the backend returns that is not es/pt collapses to OTHER;
# the policy/agent layer treats OTHER like low confidence (keep session language, confirm).
_ES = "es"
_PT = "pt"
_OTHER = "other"

# The closed set of languages the channel supports, as a single source of truth for every caller
# that must VALIDATE a language value (not just normalize a detector reply) - the API body and the
# orchestrator's explicit-choice path both check against this, so there is one list, not two.
SUPPORTED: tuple[str, str] = (_ES, _PT)


def is_supported(lang: str | None) -> bool:
    """True only for an exact supported value ('es' or 'pt'); None/empty/other/wrong case -> False.

    Used to validate an EXPLICIT language preference fail-closed: anything that is not an exact
    member of `SUPPORTED` is rejected so the caller falls back to detection (never guesses).
    """
    return lang in SUPPORTED


# Below this the detection is treated as unreliable -> low_confidence=True (confirm, don't guess).
_MIN_CONFIDENCE = 0.60

# Compressed fastText language-id model (NOT the 126 MB .bin). Cached under data/nlu/, not committed.
_LID_FTZ_NAME = "lid.176.ftz"
_LID_FTZ_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_LID_FTZ_PATH = Path(__file__).resolve().parents[2].parent / "data" / "nlu" / _LID_FTZ_NAME


@dataclass(frozen=True)
class LanguageResult:
    """Detected language normalized to {es, pt, other}, its confidence, and the fail-closed flag."""

    lang: str
    confidence: float
    low_confidence: bool


def stub_enabled() -> bool:
    """True when tests/offline runs force the deterministic stub detector (`CORA_NLU_STUB=1`)."""
    return os.getenv("CORA_NLU_STUB") == "1"


def _normalize(raw_lang: str, confidence: float) -> LanguageResult:
    """Collapse a backend's (lang, confidence) to the closed vocabulary + fail-closed flag."""
    lang = raw_lang if raw_lang in (_ES, _PT) else _OTHER
    low = lang == _OTHER or confidence < _MIN_CONFIDENCE
    return LanguageResult(lang=lang, confidence=confidence, low_confidence=low)


# -- stub backend (tested path) --------------------------------------------------------

# Tiny, deliberately small stop-word sets: enough to separate es from pt in a unit test without a
# model download. NOT a real detector - the real backend is fastText/lingua on the offline path.
_ES_MARKERS = frozenset(
    "el la los las un una y o pero cuanto cuánto tengo quiero saldo tarjeta cuenta por favor"
    " dinero gracias hola donde dónde está mi es con para que qué".split()
)
_PT_MARKERS = frozenset(
    "o a os as um uma e ou mas quanto tenho quero saldo cartão conta por favor dinheiro"
    " obrigado olá onde está meu é com para que não minha".split()
)


class StubDetector:
    """Count es vs pt marker words; the larger count wins. Ties / no markers -> low confidence."""

    def detect(self, text: str) -> LanguageResult:
        words = [w.strip(".,:;¿?¡!\"'()").lower() for w in text.split()]
        es = sum(w in _ES_MARKERS for w in words)
        pt = sum(w in _PT_MARKERS for w in words)
        total = es + pt
        if total == 0 or es == pt:
            return LanguageResult(lang=_OTHER, confidence=0.0, low_confidence=True)
        lang, hits = (_ES, es) if es > pt else (_PT, pt)
        return _normalize(lang, hits / total)


# -- fastText backend (offline/production) ---------------------------------------------


def _ensure_lid_model() -> Path:
    """Return the path to the cached lid.176.ftz, downloading it once if absent (offline only)."""
    if not _LID_FTZ_PATH.exists():
        _LID_FTZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading %s to %s (one-time)", _LID_FTZ_NAME, _LID_FTZ_PATH)
        urllib.request.urlretrieve(_LID_FTZ_URL, _LID_FTZ_PATH)  # noqa: S310 - pinned fasttext URL
    return _LID_FTZ_PATH


class _FastTextDetector:
    """fastText lid.176 wrapper. Loaded lazily; the model file is cached under data/nlu/."""

    def __init__(self) -> None:
        import fasttext

        # fasttext prints a load warning to stderr; silence it, we only want the predictor.
        fasttext.FastText.eprint = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        self._model = fasttext.load_model(str(_ensure_lid_model()))

    def detect(self, text: str) -> LanguageResult:
        # fasttext needs single-line input; it labels like "__label__es" with a probability.
        labels, probs = self._model.predict(text.replace("\n", " "), k=1)
        raw = labels[0].removeprefix("__label__") if labels else _OTHER
        confidence = float(probs[0]) if len(probs) else 0.0
        return _normalize(raw, confidence)


# -- lingua backend (pure-Python fallback) ---------------------------------------------


class _LinguaDetector:
    """lingua fallback (no native build) behind the same signature; es/pt only."""

    def __init__(self) -> None:
        from lingua import Language, LanguageDetectorBuilder

        self._lang = Language
        self._detector = LanguageDetectorBuilder.from_languages(Language.SPANISH, Language.PORTUGUESE).build()

    def detect(self, text: str) -> LanguageResult:
        values = self._detector.compute_language_confidence_values(text)
        if not values:
            return LanguageResult(lang=_OTHER, confidence=0.0, low_confidence=True)
        top = values[0]
        code = {self._lang.SPANISH: _ES, self._lang.PORTUGUESE: _PT}.get(top.language, _OTHER)
        return _normalize(code, float(top.value))


@lru_cache(maxsize=1)
def _detector():  # noqa: ANN202 - backend type varies (stub / fasttext / lingua)
    """Pick and cache the detector backend. Stub under the test flag, else fastText, else lingua."""
    if stub_enabled():
        return StubDetector()
    try:
        return _FastTextDetector()
    except Exception as exc:  # noqa: BLE001 - any import/load failure falls back, then stub
        logger.warning("fasttext lid unavailable (%s); trying lingua", type(exc).__name__)
    try:
        return _LinguaDetector()
    except Exception as exc:  # noqa: BLE001
        logger.warning("lingua unavailable (%s); using stub detector", type(exc).__name__)
        return StubDetector()


def detect(text: str) -> LanguageResult:
    """Detect the language of `text`, normalized to {es, pt, other} with a fail-closed flag.

    Empty/whitespace input is low confidence by definition. The backend reply is untrusted data:
    only its normalized (lang, confidence) is used; nothing in `text` can steer the caller.
    """
    if not text or not text.strip():
        return LanguageResult(lang=_OTHER, confidence=0.0, low_confidence=True)
    return _detector().detect(text)


if __name__ == "__main__":  # tiny self-check on the stub (no download)
    os.environ["CORA_NLU_STUB"] = "1"
    _detector.cache_clear()
    assert detect("quiero saber mi saldo por favor").lang == "es"
    assert detect("quero saber meu saldo por favor").lang == "pt"
    assert detect("asdf qwerty 123").low_confidence
    assert detect("").low_confidence
    print("language stub self-check OK")
