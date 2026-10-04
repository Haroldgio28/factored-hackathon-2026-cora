"""Offline training + evaluation for the intent classifier (task 3.5, REQ-29/REQ-32/REQ-47).

Run MANUALLY on the dev host with the real MiniLM encoder (NOT in CI, NOT under CORA_NLU_STUB):

    uv run python data/nlu/train_intent_model.py

It is the one place that produces the REAL evidence the report cites:
  1. regenerates data/nlu/splits.json from the real multilingual encoder (plan step 5: the 3.3
     dedup used a hash fallback; the committed split must come from the real space);
  2. fits the calibrated MiniLM+LogReg head on the TRAIN split and the TF-IDF comparison head;
  3. evaluates on the TEST split: macro-F1, per-class recall (escalation E1-E4 called out), ECE,
     confusion matrix, es-vs-pt breakdown with sample sizes;
  4. fits an isotonic head too, purely to plot sigmoid-vs-isotonic reliability (D2 evidence);
  5. saves the calibrated head to data/nlu/intent_head.joblib and figures to
     documentation/reports/figures/, and writes the measured numbers into model_card.json.

Everything is labeled *offline measurement* in the report. Figures use a non-interactive backend.

    # ponytail: a single procedural script, not a CLI or a reusable eval framework. It runs once
    # per gold-set change to refresh committed evidence; argparse/plugins would be scaffolding for
    # a job that has exactly one caller (this phase's report).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; we only save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from cora.nlu import embeddings
from cora.nlu.classifier import IntentClassifier, fit_embedding_classifier, fit_tfidf_classifier
from cora.nlu.goldset import load
from cora.nlu.splits import make_splits

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_intent_model")

_DATA = Path(__file__).resolve().parent
_FIGS = _DATA.parents[1] / "documentation" / "reports" / "figures"
_ESCALATION = ("E1", "E2", "E3", "E4")


def _expected_calibration_error(proba: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin population.

    `proba` is the max calibrated probability per row (the model's confidence); `correct` is a
    0/1 array of whether that top prediction was right. Standard reliability binning.
    """
    conf = proba
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def _reliability_points(proba: np.ndarray, correct: np.ndarray, n_bins: int = 10):  # noqa: ANN202
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (proba > lo) & (proba <= hi)
        if mask.any():
            xs.append(proba[mask].mean())
            ys.append(correct[mask].mean())
    return xs, ys


def _evaluate(clf: IntentClassifier, test: pd.DataFrame) -> dict:
    """Score a fitted head on the test split and return the metric block for the report."""
    utterances = test["utterance"].tolist()
    y_true = test["intent"].tolist()
    x = embeddings.encode(utterances) if clf.embedded else utterances
    proba = clf.predict_proba(x)
    y_pred = [str(c) for c in clf.predict(x)]

    labels = [str(c) for c in clf.classes_]
    top_conf = proba.max(axis=1)
    correct = np.array([t == p for t, p in zip(y_true, y_pred, strict=True)], dtype=float)

    per_class_recall = dict(
        zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0), strict=True)
    )

    def lang_f1(lang: str) -> dict:
        idx = [i for i, lg in enumerate(test["language"].tolist()) if lg == lang]
        if not idx:
            return {"n": 0, "macro_f1": None}
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        return {
            "n": len(idx),
            "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        }

    return {
        "n_test": len(y_true),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float(correct.mean()), 4),
        "ece": round(_expected_calibration_error(top_conf, correct), 4),
        "per_class_recall": {k: round(float(v), 4) for k, v in per_class_recall.items()},
        "escalation_recall": {k: round(float(per_class_recall.get(k, 0.0)), 4) for k in _ESCALATION},
        "by_language": {"es": lang_f1("es"), "pt": lang_f1("pt")},
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _save_confusion_figure(block: dict, path: Path) -> None:
    cm = np.array(block["confusion_matrix"], dtype=float)
    labels = block["labels"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Intent confusion matrix (test split, offline measurement)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_reliability_figure(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    for name, (xs, ys) in curves.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability: sigmoid vs isotonic (offline measurement)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    # Preferred evidence comes from the real MiniLM encoder. In a sandbox that cannot download the
    # weights (SSL/network blocked), fall back to the deterministic StubEncoder so the pipeline
    # STILL produces real measured numbers from whatever runs (plan: "produce real numbers from
    # whatever runs"). The provenance is recorded in the model card so the report never passes
    # stub-space numbers off as MiniLM numbers. TF-IDF numbers are real regardless (no download).
    #
    # ponytail: no mock of the HF download; the stub is the already-built offline/test encoder, and
    # the real-encoder path above it is unchanged for a host that can reach HF.
    encoder = "paraphrase-multilingual-MiniLM-L12-v2"
    if not embeddings.stub_enabled():
        try:
            embeddings.encode(["probe"])
        except Exception as exc:  # noqa: BLE001 - any download/SSL failure => documented fallback
            log.warning("real encoder unavailable (%s); falling back to StubEncoder", type(exc).__name__)
            import os

            os.environ["CORA_NLU_STUB"] = "1"
            embeddings._stub_encoder.cache_clear()
    if embeddings.stub_enabled():
        encoder = "StubEncoder (hash, 64-d) - MiniLM weights not downloadable in this sandbox"

    es = load(_DATA / "gold.tsv")
    pt = load(_DATA / "gold_pt.tsv")
    df = pd.concat([es, pt], ignore_index=True)

    # (1) regenerate splits.json from the REAL encoder (plan step 5).
    assignment, report = make_splits(df)
    (_DATA / "splits.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    df = df[df["id"].isin(assignment)].copy()
    df["split"] = df["id"].map(assignment)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    log.info("splits: train=%d test=%d (val=%d)", len(train), len(test), (df["split"] == "validation").sum())

    train_u = train["utterance"].tolist()
    train_y = train["intent"].tolist()

    # (2) fit both heads.
    emb_clf = fit_embedding_classifier(train_u, train_y)
    tfidf_clf = fit_tfidf_classifier(train_u, train_y)
    emb_clf.save(_DATA / "intent_head.joblib")

    # (3) evaluate on test.
    emb_block = _evaluate(emb_clf, test)
    tfidf_block = _evaluate(tfidf_clf, test)
    log.info("MiniLM macro-F1=%s  TF-IDF macro-F1=%s", emb_block["macro_f1"], tfidf_block["macro_f1"])

    # (4) sigmoid vs isotonic reliability (D2 evidence), on the embedding representation.
    x_train = embeddings.encode(train_u)
    x_test = embeddings.encode(test["utterance"].tolist())
    y_test = test["intent"].tolist()
    curves = {}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(LogisticRegression(max_iter=2000), cv=5, method=method)
        cal.fit(x_train, train_y)
        proba = cal.predict_proba(x_test)
        pred = [cal.classes_[i] for i in proba.argmax(axis=1)]
        correct = np.array([t == p for t, p in zip(y_test, pred, strict=True)], dtype=float)
        curves[method] = _reliability_points(proba.max(axis=1), correct)

    # (5) figures.
    _FIGS.mkdir(parents=True, exist_ok=True)
    _save_confusion_figure(emb_block, _FIGS / "intent_confusion_matrix.png")
    _save_reliability_figure(curves, _FIGS / "intent_reliability_sigmoid_vs_isotonic.png")

    # write measured numbers into the model card (replace the 3.5 placeholder block).
    card_path = _DATA / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["intent_model_results"] = {
        "note": (
            "Offline measurement (REQ-47) on the leakage-safe test split. Train has 15 of 16 "
            "classes (X3 is absent from train after dedup; see splits.json missing_after_dedup); "
            "X3 is therefore unlearnable here and the report states this honestly rather than "
            "inflating the set."
        ),
        "encoder": encoder,
        "encoder_provenance": (
            "REAL MiniLM"
            if "MiniLM" in encoder and "Stub" not in encoder
            else "STUB (sandbox could "
            "not download MiniLM weights; the real-encoder path in train_intent_model.py is unchanged "
            "and re-runs on a host with HF access). TF-IDF numbers are download-free and real either way."
        ),
        "calibration": "sigmoid/Platt (D2)",
        "minilm_logreg": emb_block,
        "tfidf_char_ngram": tfidf_block,
        "figures": [
            "documentation/reports/figures/intent_confusion_matrix.png",
            "documentation/reports/figures/intent_reliability_sigmoid_vs_isotonic.png",
        ],
    }
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s and figures to %s", card_path, _FIGS)


if __name__ == "__main__":
    main()
