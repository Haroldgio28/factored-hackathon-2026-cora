"""Load and validate the team-written gold set, and report inter-annotator agreement (task 3.1).

The gold set is a TSV (utterances contain commas, so not CSV). It is small (<2k rows), so the
contract is plain `assert`-style checks here rather than Pandera (ponytail: no schema framework for
a file this size). The loader is the trust boundary for every downstream ML step, so it does NOT
simplify validation away: a malformed file raises `GoldSetError` and NEVER silently drops a row
(fail closed, security steering).

Rejected by the contract:
- any column missing, or an unexpected extra column (the opposite of the entity schema at 3.7,
  which ignores extra keys - the two loaders deliberately do not share leniency);
- an unknown `intent`, `provenance`, `language` or `variant`;
- a blank `utterance` or `scenario_id`;
- a `dataset-seed` row (those are seed material, never a split row - EDA F7, REQ-30);
- a `double_labeled` row with a blank/invalid `label_b`, or a `label_b` on a row not marked
  double-labeled (the 20% second-annotator protocol must be self-consistent).

`cohen_kappa` reports agreement on the double-labeled subset, so the κ figure the labeling guide
promises is produced from the real file, not asserted. κ is a four-line counting formula over two
label columns, so it is computed directly rather than pulling the `nlu` ML group (sklearn) into
this subtask - that group lands at 3.4 (ponytail: stdlib arithmetic beats a forward dependency).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from cora.nlu.labels import Intent, Provenance, Variant

__all__ = ["COLUMNS", "GoldSetError", "cohen_kappa", "load"]

# Exact column order of data/nlu/gold.tsv and gold_pt.tsv (plan NIT-3: scenario_id and month are
# their own explicit columns, not parsed out of `id`).
COLUMNS: tuple[str, ...] = (
    "id",
    "scenario_id",
    "utterance",
    "intent",
    "language",
    "variant",
    "provenance",
    "month",
    "double_labeled",
    "label_b",
    "reviewer_note",
)

_LANGUAGES = frozenset({"es", "pt"})
# Spanish rows take a Spanish variant; Portuguese rows take BR. Pairing language<->variant stops a
# mislabeled row (e.g. an es row tagged BR) slipping through.
_VARIANTS_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "es": frozenset({Variant.MX, Variant.CO, Variant.AR}),
    "pt": frozenset({Variant.BR}),
}
_INTENTS = frozenset(Intent)
_PROVENANCES = frozenset(Provenance)
# Truthy tokens accepted in the boolean `double_labeled` column (TSV has no native bool).
_TRUE = frozenset({"1", "true", "yes", "y"})
_FALSE = frozenset({"0", "false", "no", "n", ""})


class GoldSetError(ValueError):
    """The gold TSV is malformed; the loader refuses it rather than dropping rows (fail closed)."""


def _as_bool(raw: str, row_id: str) -> bool:
    token = raw.strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise GoldSetError(f"row {row_id!r}: double_labeled must be a boolean token, got {raw!r}")


def load(path: Path | str) -> pd.DataFrame:
    """Load, validate and return the gold TSV as a DataFrame (fail closed on any violation).

    All columns are read as strings (no type coercion that could mask a bad value); the returned
    frame carries a parsed boolean `double_labeled` plus the original string columns.
    """
    path = Path(path)
    if not path.exists():
        raise GoldSetError(f"gold set not found: {path}")

    # keep_default_na=False so an empty cell stays "" (a blank utterance must be caught, not read
    # as NaN); dtype=str so no column is silently coerced.
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")

    if tuple(df.columns) != COLUMNS:
        raise GoldSetError(f"columns must be exactly {COLUMNS}, got {tuple(df.columns)}")
    if df.empty:
        raise GoldSetError(f"gold set {path} has no rows")

    ids: set[str] = set()
    double_labeled: list[bool] = []
    for row in df.itertuples(index=False):
        row_id = row.id.strip()
        if not row_id:
            raise GoldSetError("a row has a blank id")
        if row_id in ids:
            raise GoldSetError(f"duplicate row id: {row_id!r}")
        ids.add(row_id)

        if not row.scenario_id.strip():
            raise GoldSetError(f"row {row_id!r}: blank scenario_id")
        if not row.utterance.strip():
            raise GoldSetError(f"row {row_id!r}: blank utterance")
        if row.intent not in _INTENTS:
            raise GoldSetError(f"row {row_id!r}: unknown intent {row.intent!r}")
        if row.provenance not in _PROVENANCES:
            raise GoldSetError(f"row {row_id!r}: unknown provenance {row.provenance!r}")
        if row.provenance == Provenance.DATASET_SEED:
            # Seed rows are paraphrased from the 42 templated transcripts; they must never become a
            # split row (EDA F7, REQ-30). Keeping them out of the gold file entirely is the simplest
            # guarantee, so the loader rejects them here.
            raise GoldSetError(
                f"row {row_id!r}: dataset-seed is seed-only and must not appear in the gold set"
            )
        if row.language not in _LANGUAGES:
            raise GoldSetError(f"row {row_id!r}: unknown language {row.language!r}")
        if row.variant not in _VARIANTS_BY_LANGUAGE[row.language]:
            raise GoldSetError(
                f"row {row_id!r}: variant {row.variant!r} invalid for language {row.language!r}"
            )
        if not row.month.strip():
            raise GoldSetError(f"row {row_id!r}: blank month (authoring wave)")

        is_double = _as_bool(row.double_labeled, row_id)
        has_label_b = bool(row.label_b.strip())
        if is_double:
            if row.label_b not in _INTENTS:
                raise GoldSetError(
                    f"row {row_id!r}: double_labeled row needs a valid label_b, got {row.label_b!r}"
                )
        elif has_label_b:
            raise GoldSetError(f"row {row_id!r}: label_b present but row is not double_labeled")
        double_labeled.append(is_double)

    df["double_labeled"] = double_labeled
    return df


def cohen_kappa(df: pd.DataFrame) -> float:
    """Cohen's κ between the two annotators on the double-labeled subset.

    `df` is a frame returned by `load`. κ = (p_o - p_e) / (1 - p_e), where p_o is the observed
    agreement rate and p_e the agreement expected by chance from each annotator's label marginals.
    Returns 1.0 when the two annotators never disagree AND use only one label (p_e == 1, perfect by
    convention). Raises `GoldSetError` if no rows are double-labeled, so a missing second-annotator
    pass is a loud failure rather than a silent nan.
    """
    subset = df[df["double_labeled"]]
    n = len(subset)
    if n == 0:
        raise GoldSetError("no double-labeled rows; cannot compute Cohen's kappa")

    a = subset["intent"].tolist()
    b = subset["label_b"].tolist()
    p_o = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    count_a = Counter(a)
    count_b = Counter(b)
    p_e = sum(count_a[label] * count_b[label] for label in set(count_a) | set(count_b)) / (n * n)
    if p_e == 1.0:
        return 1.0  # both annotators used a single, identical label; perfect agreement by convention
    return (p_o - p_e) / (1 - p_e)
