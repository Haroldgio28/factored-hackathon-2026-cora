"""Calibrated intent classifier: MiniLM embeddings + multinomial LogReg (task 3.5, REQ-29/REQ-32).

The design locks a frozen multilingual embedding (`embeddings.encode`) plus a linear head. On a
small gold set (~20 utterances/class) a high-capacity model would memorize; a linear model over a
fixed 384-d space generalizes and stays calibratable. Two heads are built from this one module:

- `fit_embedding_classifier` - the real model: `encode` the utterances, then fit
  `CalibratedClassifierCV(LogisticRegression(multinomial), cv=5, method="sigmoid")`. Sigmoid/Platt
  (not isotonic/temperature) is the recorded decision D2: at N ~= 20/class an isotonic map
  overfits, while sigmoid fits two parameters per class. The report shows both reliability curves
  as evidence, but sigmoid is the frozen default.
- `fit_tfidf_classifier` - the "no cross-lingual transfer" comparison: the same calibrated LogReg
  over the char-ngram TF-IDF features (an sklearn `Pipeline`, so the vectorizer travels with the
  head). Saved/loaded whole; the embedding head saves only the calibrated classifier (the encoder
  is referenced by HF id in the model card, never committed - design NIT-6).

`IntentClassifier` wraps a fitted head and exposes `predict` / `predict_proba` returning columns in
a FIXED label order (`classes_`) so 3.6's thresholds and `PolicyInput` read a stable layout. Both
heads degrade through the same object; the only difference is whether inputs are pre-embedded.

    # ponytail: joblib persistence + sklearn CalibratedClassifierCV, both already pulled by the
    # nlu group; no custom serializer or calibration code. The encoder stays out of the saved
    # artifact (referenced by id), so the committed head is a few hundred KB.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from cora.nlu.embeddings import build_tfidf, encode
from cora.nlu.labels import Intent

__all__ = [
    "CALIBRATION_METHOD",
    "IntentClassifier",
    "fit_embedding_classifier",
    "fit_tfidf_classifier",
]

CALIBRATION_METHOD = "sigmoid"  # D2: Platt, not isotonic - small N per class
_CV = 5  # CalibratedClassifierCV folds; train has 20/class, so >=5 per class is satisfied


def _calibrated_logreg():  # noqa: ANN202 - sklearn estimator
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    # multinomial LogReg over frozen features; lbfgs handles the multiclass softmax directly.
    base = LogisticRegression(max_iter=2000, C=1.0)
    return CalibratedClassifierCV(base, cv=_CV, method=CALIBRATION_METHOD)


class IntentClassifier:
    """A fitted calibrated head over one representation. `embedded` says if inputs are pre-encoded.

    `embedded=True`  -> `predict`/`predict_proba` take an (n, d) embedding array (the MiniLM head).
    `embedded=False` -> they take a list of raw utterances (the TF-IDF pipeline encodes internally).
    """

    def __init__(self, estimator, *, embedded: bool) -> None:
        self._est = estimator
        self.embedded = embedded
        # classes_ is the frozen column order for predict_proba; map to Intent once, fail loud if
        # the estimator ever learned a label outside the policy vocabulary.
        self.classes_ = [Intent(c) for c in estimator.classes_]

    def _features(self, utterances: list[str] | np.ndarray):  # noqa: ANN202
        if self.embedded:
            if not isinstance(utterances, np.ndarray):
                raise TypeError("embedding classifier expects a pre-encoded (n, d) array")
            return utterances
        return list(utterances)

    def predict_proba(self, utterances: list[str] | np.ndarray) -> np.ndarray:
        return self._est.predict_proba(self._features(utterances))

    def predict(self, utterances: list[str] | np.ndarray) -> list[Intent]:
        return [Intent(c) for c in self._est.predict(self._features(utterances))]

    def save(self, path: Path | str) -> None:
        """Persist the calibrated head only (not the encoder) via joblib (design NIT-6)."""
        joblib.dump({"estimator": self._est, "embedded": self.embedded}, path)

    @classmethod
    def load(cls, path: Path | str) -> IntentClassifier:
        blob = joblib.load(path)
        return cls(blob["estimator"], embedded=blob["embedded"])


def fit_embedding_classifier(utterances: list[str], labels: list[str]) -> IntentClassifier:
    """Encode the utterances with the (lazy) MiniLM encoder and fit the calibrated LogReg head."""
    x = encode(utterances)
    est = _calibrated_logreg()
    est.fit(x, labels)
    return IntentClassifier(est, embedded=True)


def fit_tfidf_classifier(utterances: list[str], labels: list[str]) -> IntentClassifier:
    """Fit the char-ngram TF-IDF + calibrated LogReg comparison pipeline on raw utterances."""
    from sklearn.pipeline import Pipeline

    est = Pipeline([("tfidf", build_tfidf()), ("clf", _calibrated_logreg())])
    est.fit(utterances, labels)
    return IntentClassifier(est, embedded=False)
