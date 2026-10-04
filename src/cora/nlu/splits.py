"""Leakage-safe train/validation/test splits for the gold set (task 3.3, REQ-31).

REQ-31 asks for three guarantees against evaluation leakage; this module owns all three and is the
single place that decides which utterance lands in which split:

1. **Group isolation.** The gold set has no real `customer_id`, so the grouping key is the
   *source scenario* (design D1): an ES row `S-I1-MX-0` and its PT translation `S-I1-BR-MX-0` are
   two phrasings of the same imagined request, so they share one group key `S-I1-MX-0` and can
   never fall on opposite sides of the split. Grouping by the source scenario is the team-written
   analogue of "group by customer_id".
2. **Time ordering.** Each group carries an authoring-wave `month` tag; the report records the
   wave composition of each split (REQ-31 "train earlier, test later"). The gold set was authored
   in two class-segregated waves (W1 = servicing intents, W2 = escalation/other), so a split *by
   wave alone* would leave whole classes absent from a split. Stratifying by class is therefore
   required for a usable split and takes precedence; the honest consequence (waves do not cleanly
   order train-before-test for this synthetic set) is written into the report, not hidden.
3. **Cross-split near-duplicate removal.** Every validation/test utterance is compared by embedding
   cosine against every train utterance; anything above `DEDUP_THRESHOLD` (0.95, the REQ-31 value)
   is dropped from the held-out side so train wins and the held-out set stays clean.

The split is deterministic (seeded) so `splits.json` is reproducible from the committed gold set.

Encoder seam (ponytail, no new abstraction): dedup needs an embedder, but `cora.nlu.embeddings`
only lands at task 3.5. So `_encode` imports it lazily *if present* and otherwise falls back to a
local deterministic hash encoder - good enough to catch the identical/near-identical rows the dedup
step targets, and swapped for the real multilingual encoder when 3.5 regenerates `splits.json`.

    # ponytail: O(n_val+test * n_train) cosine scan + a hash fallback encoder. Fine at <1k rows;
    # when 3.5 lands, _encode picks up the real MiniLM encoder and splits.json is regenerated.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cora.nlu.goldset import GoldSetError, load

__all__ = ["DEDUP_THRESHOLD", "SEED", "SPLIT_RATIOS", "group_key", "make_splits", "write_report"]

logger = logging.getLogger("cora.nlu.splits")

SEED = 20261003  # fixed so the split (and splits.json) is reproducible from the gold set
DEDUP_THRESHOLD = 0.95  # REQ-31 cross-split near-duplicate cosine cutoff; a logged constant
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}  # by group count, per class

_SPLITS = ("train", "validation", "test")
_HASH_DIM = 64  # fallback encoder dimensionality; only used until the real encoder (3.5) is wired


def group_key(scenario_id: str) -> str:
    """Map a scenario id to the leakage group shared by an ES row and its PT translation.

    PT scenarios are `S-<intent>-BR-<src-variant>-<idx>`, derived from the ES source
    `S-<intent>-<src-variant>-<idx>` (see `goldset.check_pairing`). Dropping the `BR` segment gives
    the shared source key, so a translation never crosses the split boundary away from its origin.
    ES scenarios (no `BR` segment) are their own group key unchanged.
    """
    parts = scenario_id.split("-")
    if len(parts) == 5 and parts[0] == "S" and parts[2] == "BR":
        return f"{parts[0]}-{parts[1]}-{parts[3]}-{parts[4]}"
    return scenario_id


def _hash_encode(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-char-trigram hash vectors, L2-normalized (fallback dedup encoder).

    Not a semantic encoder: it maps near-identical strings to near-identical vectors, which is all
    the cross-split dedup needs (identical ES/PT duplicates, trivial paraphrases). Replaced by the
    real multilingual encoder when 3.5 is wired into `_encode`.
    """
    vecs = np.zeros((len(texts), _HASH_DIM), dtype=np.float64)
    for i, text in enumerate(texts):
        norm = " ".join(text.lower().split())
        for j in range(len(norm) - 2):
            tri = norm[j : j + 3]
            bucket = int(hashlib.blake2b(tri.encode("utf-8"), digest_size=8).hexdigest(), 16) % _HASH_DIM
            vecs[i, bucket] += 1.0
    lengths = np.linalg.norm(vecs, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0  # a (near-)empty string stays the zero vector, not a divide-by-zero
    return vecs / lengths


def _encode(texts: list[str]) -> np.ndarray:
    """Embed utterances for dedup, preferring the real encoder (3.5) over the hash fallback.

    Lazy import so task 3.3 commits before `cora.nlu.embeddings` exists (ponytail: no forward
    dependency, no new abstraction). When 3.5 lands, `embeddings.encode` is picked up automatically
    and `splits.json` is regenerated from the real multilingual space.
    """
    try:
        from cora.nlu.embeddings import encode  # type: ignore[attr-defined]
    except ImportError:
        logger.info("splits: embeddings module absent, using hash fallback encoder for dedup")
        return _hash_encode(texts)
    return np.asarray(encode(texts), dtype=np.float64)


def _assign_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group row ids by leakage key, deterministically ordered."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in df.itertuples(index=False):
        groups[group_key(row.scenario_id)].append(row.id)
    return {key: groups[key] for key in sorted(groups)}


def _split_groups_for_class(group_keys: list[str], rng: np.random.Generator) -> dict[str, list[str]]:
    """Partition one class's groups into train/val/test by SPLIT_RATIOS (deterministic).

    A class with too few groups to fill all three splits keeps train first, then validation, then
    test; the shortfall is flagged by the caller (design error-table: "flagged, not dropped").
    """
    keys = list(group_keys)
    rng.shuffle(keys)
    n = len(keys)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_val = round(n * SPLIT_RATIOS["validation"])
    # test gets the remainder so the three counts always sum to n (no row lost to rounding).
    return {
        "train": keys[:n_train],
        "validation": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :],
    }


def _dedup(df: pd.DataFrame, assignment: dict[str, str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Drop any validation/test row whose embedding cosine to any train row exceeds the threshold.

    Returns the pruned assignment and a list of removed-row records (for the report). Train wins:
    only held-out rows are ever dropped, so the training set is untouched and the held-out sets stay
    free of train look-alikes.
    """
    by_id = df.set_index("id")
    train_ids = [rid for rid, split in assignment.items() if split == "train"]
    heldout_ids = [rid for rid, split in assignment.items() if split != "train"]
    if not train_ids or not heldout_ids:
        return assignment, []

    train_vecs = _encode([by_id.at[rid, "utterance"] for rid in train_ids])
    heldout_vecs = _encode([by_id.at[rid, "utterance"] for rid in heldout_ids])
    # Vectors are L2-normalized, so the dot product is the cosine similarity.
    sims = heldout_vecs @ train_vecs.T

    pruned = dict(assignment)
    removed: list[dict[str, str]] = []
    for i, rid in enumerate(heldout_ids):
        j = int(np.argmax(sims[i]))
        best = float(sims[i, j])
        if best > DEDUP_THRESHOLD:
            match_id = train_ids[j]
            removed.append(
                {
                    "removed_id": rid,
                    "removed_split": assignment[rid],
                    "train_match_id": match_id,
                    "cosine": round(best, 4),
                    "removed_utterance": str(by_id.at[rid, "utterance"]),
                    "train_utterance": str(by_id.at[match_id, "utterance"]),
                }
            )
            del pruned[rid]
    return pruned, removed


def make_splits(df: pd.DataFrame) -> tuple[dict[str, str], dict]:
    """Assign every gold row to a split and return (row_id -> split, report dict).

    `df` is the concatenation of the ES and PT gold frames (both from `goldset.load`). Groups are
    class-stratified so every intent and language is represented in each split where it has enough
    groups; cross-split near-duplicates are then removed. The report captures the per-split counts,
    the wave composition, the removed duplicates, the seed and the thresholds so `splits.json` is a
    self-describing evidence artifact.
    """
    if df.empty:
        raise GoldSetError("cannot split an empty gold set")

    rng = np.random.default_rng(SEED)
    groups = _assign_groups(df)
    group_intent = {key: df[df["id"].isin(ids)]["intent"].iloc[0] for key, ids in groups.items()}

    # Stratify by intent: split each class's groups independently so a class is never absent from a
    # split merely because the global shuffle starved it.
    by_class: dict[str, list[str]] = defaultdict(list)
    for key in groups:
        by_class[group_intent[key]].append(key)

    group_split: dict[str, str] = {}
    thin_classes: list[dict] = []
    for intent in sorted(by_class):
        class_groups = sorted(by_class[intent])
        parts = _split_groups_for_class(class_groups, rng)
        for split_name, keys in parts.items():
            for key in keys:
                group_split[key] = split_name
        empty = [s for s in _SPLITS if not parts[s]]
        if empty:
            thin_classes.append({"intent": intent, "n_groups": len(class_groups), "empty_splits": empty})
            logger.warning(
                "splits: class %s has %d groups, missing from splits %s", intent, len(class_groups), empty
            )

    assignment = {
        rid: group_split[group_key(sid)] for rid, sid in zip(df["id"], df["scenario_id"], strict=True)
    }
    assignment, removed = _dedup(df, assignment)

    report = _build_report(df, assignment, removed, thin_classes)
    return assignment, report


def _build_report(
    df: pd.DataFrame,
    assignment: dict[str, str],
    removed: list[dict[str, str]],
    thin_classes: list[dict],
) -> dict:
    """Assemble the self-describing split report (counts, waves, dedup, provenance of the recipe)."""
    kept = df[df["id"].isin(assignment)].copy()
    kept["split"] = kept["id"].map(assignment)

    def crosstab(col: str) -> dict:
        return {
            split: kept[kept["split"] == split][col].value_counts().sort_index().to_dict()
            for split in _SPLITS
        }

    # Coverage AFTER dedup: cross-split near-duplicate removal can empty a thin class or language
    # out of a held-out split (REQ-31 correctly drops the leaking row; the honest consequence is
    # that the class is now under-represented there). Flag it so the report states it rather than
    # the reader discovering a 0 in the matrix (REQ-47 honesty). Nothing is silently dropped: the
    # removed rows are listed in `dedup.removed`.
    def missing(col: str) -> list[dict]:
        all_values = set(df[col])
        return [
            {"value": v, "column": col, "empty_splits": empties}
            for v in sorted(all_values)
            if (empties := [s for s in _SPLITS if v not in set(kept[kept["split"] == s][col])])
        ]

    return {
        "seed": SEED,
        "dedup_threshold": DEDUP_THRESHOLD,
        "split_ratios": SPLIT_RATIOS,
        "grouping": {
            "key": "source scenario (scenario_id with the PT 'BR' segment stripped)",
            "note": (
                "REQ-31 customer_id grouping is realised as the source scenario id (design D1); an "
                "ES row and its PT translation share one group and never cross the split boundary. "
                "No real customer ids or timestamps exist in the synthetic gold set."
            ),
        },
        "time_ordering": {
            "wave_column": "month",
            "note": (
                "The gold set was authored in two class-segregated waves (W1 servicing, W2 "
                "escalation/other), so splitting by wave alone would leave whole classes absent "
                "from a split. Class-stratification takes precedence for a usable split; the wave "
                "composition per split is reported below for transparency."
            ),
            "waves_per_split": crosstab("month"),
        },
        "counts": {
            "total_rows_in": int(len(df)),
            "total_rows_kept": int(len(kept)),
            "per_split": {split: int((kept["split"] == split).sum()) for split in _SPLITS},
            "per_split_groups": {
                split: int(kept[kept["split"] == split]["scenario_id"].map(group_key).nunique())
                for split in _SPLITS
            },
        },
        "per_split_intent": crosstab("intent"),
        "per_split_language": crosstab("language"),
        "thin_classes": thin_classes,
        "missing_after_dedup": missing("intent") + missing("language"),
        "dedup": {
            "removed_count": len(removed),
            "removed": removed,
        },
    }


def write_report(gold_es: Path | str, gold_pt: Path | str, out: Path | str) -> dict:
    """Build splits from the committed ES+PT gold files and write the report JSON. Returns the report.

    This is the offline entry point that regenerates `data/nlu/splits.json`.
    """
    es = load(gold_es)
    pt = load(gold_pt)
    df = pd.concat([es, pt], ignore_index=True)
    _assignment, report = make_splits(df)
    out = Path(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("splits: wrote %s (%d rows kept of %d)", out, report["counts"]["total_rows_kept"], len(df))
    return report


if __name__ == "__main__":  # offline regeneration of data/nlu/splits.json
    _data = Path(__file__).resolve().parents[3] / "data" / "nlu"
    write_report(_data / "gold.tsv", _data / "gold_pt.tsv", _data / "splits.json")
