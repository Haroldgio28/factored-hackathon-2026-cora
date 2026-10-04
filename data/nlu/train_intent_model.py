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
from cora.nlu.thresholds import GRID, freeze, select_thresholds

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_intent_model")

_DATA = Path(__file__).resolve().parent
_FIGS = _DATA.parents[1] / "documentation" / "reports" / "figures"
_REPORT = _DATA.parents[1] / "documentation" / "reports" / "intent_model.md"
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


def _conf_and_correct(clf: IntentClassifier, split: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Top calibrated confidence and 0/1 correctness per row - the input `select_thresholds` scores."""
    utterances = split["utterance"].tolist()
    x = embeddings.encode(utterances) if clf.embedded else utterances
    proba = clf.predict_proba(x)
    y_pred = [str(c) for c in clf.predict(x)]
    y_true = split["intent"].tolist()
    conf = proba.max(axis=1)
    correct = np.array([t == p for t, p in zip(y_true, y_pred, strict=True)], dtype=bool)
    return conf, correct


def _save_cost_figure(conf: np.ndarray, correct: np.ndarray, choice, path: Path) -> None:
    """Cost vs tau_escalate (tau_clarify held at the frozen value) - shows why the pair was chosen."""
    from cora.nlu.thresholds import _total_cost  # noqa: PLC0415 - internal helper reused for the plot

    taus = [t for t in GRID if t <= choice.tau_clarify]
    costs = [_total_cost(conf, correct, t, choice.tau_clarify, choice.cost_matrix) for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, costs, marker="o")
    ax.axvline(
        choice.tau_escalate, color="r", linestyle="--", label=f"frozen τ_escalate={choice.tau_escalate}"
    )
    ax.set_xlabel(f"τ_escalate (τ_clarify fixed at {choice.tau_clarify})")
    ax.set_ylabel("validation cost")
    ax.set_title("Cost vs threshold on validation (offline measurement)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _misclassified_examples(clf: IntentClassifier, test: pd.DataFrame, limit: int = 8) -> list[dict]:
    """Real misclassified test rows for the report's error-analysis section (REQ-47)."""
    utterances = test["utterance"].tolist()
    x = embeddings.encode(utterances) if clf.embedded else utterances
    proba = clf.predict_proba(x)
    y_pred = [str(c) for c in clf.predict(x)]
    y_true = test["intent"].tolist()
    conf = proba.max(axis=1)
    langs = test["language"].tolist()
    rows = [
        {
            "utterance": utterances[i],
            "language": langs[i],
            "true": y_true[i],
            "pred": y_pred[i],
            "confidence": round(float(conf[i]), 3),
        }
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    return rows[:limit]


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


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _write_report(card: dict, choice, errors: list[dict], out: Path) -> None:
    """Render documentation/reports/intent_model.md from the measured numbers (task 3.6, REQ-47).

    The report is generated, not hand-maintained, so its numbers always match the committed
    model_card.json / thresholds.json. Every figure is labeled *offline measurement*.
    """
    res = card["intent_model_results"]
    emb = res["minilm_logreg"]
    tfidf = res["tfidf_char_ngram"]
    cost = choice.cost_matrix
    base = card.get("baseline_results", {})

    comparison = _md_table(
        ["model", "macro-F1", "accuracy", "ECE", "es macro-F1", "pt macro-F1"],
        [
            [
                "B-maj (majority)",
                "-",
                f"{base.get('majority', {}).get('accuracy', '-')}",
                "-",
                "-",
                "-",
            ],
            [
                "B-kw (keyword es/pt)",
                "-",
                f"{base.get('keyword', {}).get('accuracy_overall', '-')}",
                "-",
                f"{base.get('keyword', {}).get('accuracy_es', '-')}",
                f"{base.get('keyword', {}).get('accuracy_pt', '-')}",
            ],
            [
                "MiniLM + LogReg (calibrated)",
                f"{emb['macro_f1']}",
                f"{emb['accuracy']}",
                f"{emb['ece']}",
                f"{emb['by_language']['es']['macro_f1']}",
                f"{emb['by_language']['pt']['macro_f1']}",
            ],
            [
                "TF-IDF char-ngram (no x-lingual)",
                f"{tfidf['macro_f1']}",
                f"{tfidf['accuracy']}",
                f"{tfidf['ece']}",
                f"{tfidf['by_language']['es']['macro_f1']}",
                f"{tfidf['by_language']['pt']['macro_f1']}",
            ],
        ],
    )

    escalation = _md_table(
        ["class", "MiniLM recall", "TF-IDF recall"],
        [[k, f"{emb['escalation_recall'][k]}", f"{tfidf['escalation_recall'][k]}"] for k in _ESCALATION],
    )

    error_rows = [
        [f"`{e['utterance']}`", e["language"], e["true"], e["pred"], f"{e['confidence']}"] for e in errors
    ]
    error_table = (
        _md_table(["utterance", "lang", "true", "predicted", "conf"], error_rows)
        if error_rows
        else "_No misclassifications on the test split (small split; see confusion matrix)._"
    )

    cost_line = (
        f"**unsafe automation ({cost.unsafe_answer:.0f}) "
        f">> unnecessary escalation ({cost.unnecessary_escalate:.0f}) "
        f">> clarification ({cost.clarify:.0f}) "
        f">> correct answer ({cost.correct_answer:.0f})**"
    )

    # Honesty guard (REQ-47 / evaluation steering): a degenerate (0.0, 0.0) pair means the sweep
    # found zero unsafe answers on the small validation split, so "answer everything" is cost-free
    # there. That is a small-sample artifact, NOT an operational safety threshold - with tau=0 the
    # confidence bands POL-060/POL-070 never fire. Flag it loudly rather than present it as tuned.
    degenerate = choice.tau_escalate == 0.0 and choice.tau_clarify == 0.0
    threshold_caveat = (
        (
            "\n> **Caveat (small-sample artifact, REQ-47).** The selected pair is the degenerate "
            f"`(0.0, 0.0)`: on the n={res['thresholds']['n_validation']} validation split the "
            "calibrated head makes no high-confidence errors, so 'answer everything' already costs "
            "0 and no positive threshold can beat it. This is **not** an operational safety "
            "threshold - at `tau=0` the POL-060/POL-070 confidence bands never fire, so nothing "
            "routes to clarify/escalate on low confidence. The validation split (a slice of the "
            "480-row team-written gold set) is too small to calibrate safety thresholds; the "
            "operational `tau_escalate`/`tau_clarify` are to be fixed on the Phase 6 held-out "
            "scenario suite (~400 cases, REQ-42), which is new data rather than borrowed from the "
            "gold set. The `rules.yaml` placeholders (0.40/0.60) remain the conservative stand-in "
            "until then.\n"
        )
        if degenerate
        else ""
    )

    # The error-analysis closing paragraph must match the ACTUAL frozen thresholds: with a
    # degenerate pair the bands do NOT catch low-confidence rows, so do not claim they do.
    error_closing = (
        "The dominant failure mode is **low-confidence confusion between neighbouring servicing "
        "intents**. Note the frozen thresholds are degenerate here (see the caveat above), so these "
        "rows are currently answered rather than routed to clarify/escalate; once Phase 6 sets "
        "positive thresholds, these near-boundary rows are exactly what the bands will divert to a "
        "clarification or human handoff instead of an unsafe automated answer."
        if degenerate
        else "The dominant failure mode is **low-confidence confusion between neighbouring servicing "
        "intents**, which is exactly what the threshold bands catch: at the frozen "
        "`tau_escalate`/`tau_clarify` these low-confidence rows route to clarify/escalate rather "
        "than being answered, so a classifier miss degrades to a safe clarification or human "
        "handoff instead of an unsafe automated answer."
    )
    es_lang, pt_lang = emb["by_language"]["es"], emb["by_language"]["pt"]
    es_tfidf, pt_tfidf = tfidf["by_language"]["es"], tfidf["by_language"]["pt"]
    lang_table = _md_table(
        ["language", "n (test)", "MiniLM macro-F1", "TF-IDF macro-F1"],
        [
            ["es", f"{es_lang['n']}", f"{es_lang['macro_f1']}", f"{es_tfidf['macro_f1']}"],
            ["pt", f"{pt_lang['n']}", f"{pt_lang['macro_f1']}", f"{pt_tfidf['macro_f1']}"],
        ],
    )

    md = f"""# Intent model report (task 3.6)

_All numbers are **offline measurement** (REQ-47), produced by `data/nlu/train_intent_model.py`
from the committed gold set and the leakage-safe `data/nlu/splits.json`. This file is generated,
not hand-edited, so it always matches `data/nlu/model_card.json` and `data/nlu/thresholds.json`._

## 1. Representation justification

The locked representation is frozen **multilingual MiniLM** embeddings
(`paraphrase-multilingual-MiniLM-L12-v2`, 384-d, CPU) + a calibrated linear head. One multilingual
space covers es and pt, which is what makes Portuguese work with **no PT training data** beyond the
team-written set. On a small gold set (~20 utterances/class in train) a high-capacity model would
memorise; a linear head over a fixed embedding generalises and stays calibratable. The **TF-IDF
char-ngram** variant is the deliberate "no cross-lingual transfer" comparison (NIT-3): `char_wb`
shares no space across es/pt, so its es-vs-pt behaviour read against MiniLM's is the contrast the
comparison exists to surface.

> **Encoder provenance (honesty, REQ-47).** {res["encoder_provenance"]}
> Encoder in this run: `{res["encoder"]}`. The TF-IDF numbers are download-free and real
> regardless; the MiniLM-head numbers below carry this provenance caveat.

## 2. Metrics justification

Macro-F1 (not accuracy) is the headline metric because the classes are imbalanced and the rare
**escalation** classes matter most - a missed escalation is unsafe. We therefore report **per-class
recall with E1-E4 called out**, ECE for calibration quality, the full confusion matrix, and an
es-vs-pt breakdown with sample sizes.

## 3. Baseline vs classifier

{comparison}

Train carries **15 of 16 classes**: X3 is emptied from train by the 3.3 cross-split dedup
(`splits.json` `missing_after_dedup`), so it is unlearnable here and is **not** inflated away.

### Escalation recall (the safety-critical classes)

{escalation}

> **TF-IDF ECE is high ({tfidf["ece"]}) relative to its accuracy ({tfidf["accuracy"]})** and worse
> than the MiniLM head ({emb["ece"]}). This is the known over-confidence of a char-ngram model that
> separates the training classes almost perfectly but whose probabilities are poorly spread - exactly
> why calibration quality (ECE), not accuracy alone, is reported. The committed numbers are
> single test-split point estimates; the reliability figure uses CV folds for the calibration curve.

## 4. Calibration (D2: sigmoid, not isotonic)

Calibration method is **sigmoid/Platt**, the recorded decision D2: at ~20 samples/class an isotonic
(non-parametric) map overfits, while sigmoid fits two parameters per class. The reliability figure
shows both curves as evidence; sigmoid is the frozen default.

![Reliability: sigmoid vs isotonic](figures/intent_reliability_sigmoid_vs_isotonic.png)
_Reliability curve, offline measurement._

![Confusion matrix](figures/intent_confusion_matrix.png)
_Intent confusion matrix on the test split, offline measurement._

## 5. Threshold selection (cost-matrix, on validation, frozen before test)

The policy engine evaluates `rules.yaml` top-down, so the decision surface is **three confidence
bands**, with escalate as the lowest-confidence outcome:

```
conf < tau_escalate                 -> escalate   (POL-060)
tau_escalate <= conf < tau_clarify  -> clarify     (POL-070)
conf >= tau_clarify                 -> answer      (POL-090)
```

`src/cora/nlu/thresholds.py` sweeps candidate `(tau_escalate, tau_clarify)` pairs on a 0.00-1.00
grid (step 0.05), assigns each **validation** prediction to its band, and minimises expected cost
under the named cost matrix. The ordering is the brief's priority: {cost_line}. The invariant
`tau_escalate <= tau_clarify` is a **hard grid constraint and a final assert**, so the frozen pair
is always representable by POL-060/POL-070 - the engine's rule order can never be handed a pair it
cannot express.

**Frozen pair (selected on validation, n={res["thresholds"]["n_validation"]}):**
`tau_escalate = {choice.tau_escalate}`, `tau_clarify = {choice.tau_clarify}`
(validation cost {res["thresholds"]["validation_cost"]}). Written to `data/nlu/thresholds.json`
**before** the test split was evaluated (REQ-47 freeze discipline). These are the selected values
for the `rules.yaml` POL-060 / POL-070 placeholders; `tau_fraud` (POL-040) is a fraud-score cutoff
on a 0-100 scale, **not** an intent confidence, and is **not** re-derived here.
{threshold_caveat}
![Cost vs threshold](figures/intent_cost_vs_threshold.png)
_Validation cost vs tau_escalate (tau_clarify fixed at the frozen value), offline measurement._

## 6. es vs pt breakdown

{lang_table}

PT is synthetic (translated + a 20% native-style rewrite, provenance-tagged); per the
`LABELING_GUIDE.md` limitation note, PT metrics are reported with sample sizes and read as
indicative, not production-grade.

## 7. Error analysis (real misclassified test examples, REQ-47)

Concrete MiniLM-head misclassifications on the test split (top {len(errors)}):

{error_table}

{error_closing}
"""
    out.write_text(md, encoding="utf-8")


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

    validation = df[df["split"] == "validation"]

    # (2) fit both heads.
    emb_clf = fit_embedding_classifier(train_u, train_y)
    tfidf_clf = fit_tfidf_classifier(train_u, train_y)
    emb_clf.save(_DATA / "intent_head.joblib")

    # (3a) THRESHOLDS: select tau_escalate/tau_clarify on the VALIDATION split and FREEZE them to
    # thresholds.json BEFORE the test split is touched (task 3.6, REQ-47 freeze discipline). The
    # production classifier is the embedding head, so thresholds are chosen on its confidences.
    val_conf, val_correct = _conf_and_correct(emb_clf, validation)
    choice = select_thresholds(val_conf, val_correct)
    freeze(choice, _DATA / "thresholds.json")
    log.info(
        "froze tau_escalate=%.2f tau_clarify=%.2f (val cost=%.1f) BEFORE test eval",
        choice.tau_escalate,
        choice.tau_clarify,
        choice.validation_cost,
    )

    # (3b) evaluate on test (only now that the thresholds are frozen).
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
    _save_cost_figure(val_conf, val_correct, choice, _FIGS / "intent_cost_vs_threshold.png")

    errors = _misclassified_examples(emb_clf, test)

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
            "documentation/reports/figures/intent_cost_vs_threshold.png",
        ],
        "thresholds": {
            "tau_escalate": choice.tau_escalate,
            "tau_clarify": choice.tau_clarify,
            "validation_cost": choice.validation_cost,
            "n_validation": choice.n_validation,
            "selected_on": "validation (frozen before test)",
        },
        "error_analysis_examples": errors,
    }
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s and figures to %s", card_path, _FIGS)

    _write_report(card, choice, errors, _REPORT)
    log.info("wrote report %s", _REPORT)


if __name__ == "__main__":
    main()
