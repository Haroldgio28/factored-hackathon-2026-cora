"""Tests for task 3.5: the calibrated intent classifier plumbing (REQ-29/REQ-32).

No network and no 470 MB download: `CORA_NLU_STUB=1` forces the deterministic `StubEncoder`, so
these cover the fit -> save -> load -> predict_proba *plumbing* and the calibration/label contracts.
The REAL macro-F1/recall/ECE numbers come from the offline run on the MiniLM encoder (recorded in
`data/nlu/model_card.json`); 'code looks right' is not that evidence (REQ-47), so this file only
asserts the mechanics the offline script relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from cora.nlu import embeddings
from cora.nlu.classifier import (
    CALIBRATION_METHOD,
    IntentClassifier,
    fit_embedding_classifier,
    fit_tfidf_classifier,
)
from cora.nlu.labels import Intent


@pytest.fixture(autouse=True)
def _force_stub(monkeypatch) -> None:
    # Force the deterministic stub encoder and clear its lru_cache so no run leaks the real model.
    monkeypatch.setenv("CORA_NLU_STUB", "1")
    embeddings._stub_encoder.cache_clear()


# Two classes, >=5 samples each so CalibratedClassifierCV(cv=5) has a fold member per class.
_UTTERANCES = [
    "cuanto dinero tengo en mi cuenta",
    "cual es mi saldo disponible",
    "quiero ver mi saldo",
    "cuanto me queda en la cuenta",
    "saldo de mi cuenta por favor",
    "muestrame el saldo",
    "congela mi tarjeta ahora",
    "bloquea mi tarjeta de credito",
    "quiero congelar la tarjeta",
    "desactiva mi tarjeta ya",
    "suspende mi tarjeta por favor",
    "bloquea la tarjeta de inmediato",
]
_LABELS = [Intent.I1] * 6 + [Intent.A1] * 6


def test_stub_is_active_so_no_model_download() -> None:
    assert embeddings.stub_enabled()
    v = embeddings.encode(["hola", "hola"])
    assert v.shape == (2, embeddings.STUB_DIM)
    assert np.allclose(v[0], v[1])  # identical text -> identical stub vector


def test_embedding_classifier_predicts_and_calibrates() -> None:
    clf = fit_embedding_classifier(_UTTERANCES, [str(x) for x in _LABELS])
    x = embeddings.encode(_UTTERANCES)
    proba = clf.predict_proba(x)
    assert proba.shape == (len(_UTTERANCES), len(clf.classes_))
    # Calibrated probabilities form a distribution per row.
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    # classes_ are policy Intents in a fixed order.
    assert set(clf.classes_) == {Intent.I1, Intent.A1}


def test_embedding_classifier_requires_encoded_input() -> None:
    clf = fit_embedding_classifier(_UTTERANCES, [str(x) for x in _LABELS])
    with pytest.raises(TypeError):
        clf.predict(["raw text, not an array"])  # embedded head must get an (n, d) array


def test_save_load_round_trips_predictions(tmp_path) -> None:
    clf = fit_embedding_classifier(_UTTERANCES, [str(x) for x in _LABELS])
    x = embeddings.encode(_UTTERANCES)
    out = tmp_path / "head.joblib"
    clf.save(out)
    reloaded = IntentClassifier.load(out)
    assert reloaded.embedded is True
    assert reloaded.classes_ == clf.classes_
    np.testing.assert_allclose(reloaded.predict_proba(x), clf.predict_proba(x))


def test_tfidf_variant_fits_on_raw_text() -> None:
    clf = fit_tfidf_classifier(_UTTERANCES, [str(x) for x in _LABELS])
    assert clf.embedded is False
    preds = clf.predict(_UTTERANCES)  # raw utterances; the pipeline vectorizes internally
    assert len(preds) == len(_UTTERANCES)
    assert all(isinstance(p, Intent) for p in preds)


def test_calibration_method_is_sigmoid_per_d2() -> None:
    # D2 is a frozen decision; guard it so a silent swap to isotonic is caught.
    assert CALIBRATION_METHOD == "sigmoid"
