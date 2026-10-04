"""Tests for task 3.7: language detection normalized to {es, pt, other} (REQ-18).

Runs on the deterministic `StubDetector` under `CORA_NLU_STUB=1` so there is no model download and
no network. The contract under test is the fail-closed one: es/pt are detected, and ambiguous or
empty input is flagged `low_confidence` (keep session language, confirm - never guess).
"""

from __future__ import annotations

import pytest

from cora.nlu import language


@pytest.fixture(autouse=True)
def _force_stub(monkeypatch) -> None:
    monkeypatch.setenv("CORA_NLU_STUB", "1")
    language._detector.cache_clear()


def test_detects_spanish() -> None:
    result = language.detect("quiero saber mi saldo por favor")
    assert result.lang == "es"
    assert not result.low_confidence


def test_detects_portuguese() -> None:
    result = language.detect("quero saber meu saldo por favor")
    assert result.lang == "pt"
    assert not result.low_confidence


def test_empty_input_is_low_confidence() -> None:
    result = language.detect("   ")
    assert result.lang == "other"
    assert result.low_confidence


def test_unknown_language_is_low_confidence_other() -> None:
    # No es/pt markers -> other + low confidence (fail closed, never guess a third language).
    result = language.detect("asdf qwerty 123 zzz")
    assert result.lang == "other"
    assert result.low_confidence


def test_normalized_vocabulary_is_closed() -> None:
    # Whatever a backend emits, the public result is only ever es / pt / other.
    for utterance in ("hola saldo", "olá saldo", "random noise"):
        assert language.detect(utterance).lang in {"es", "pt", "other"}
