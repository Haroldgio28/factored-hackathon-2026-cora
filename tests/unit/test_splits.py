"""Tests for task 3.3: leakage-safe group+time splits with cross-split 0.95 dedup (REQ-31).

The three REQ-31 guarantees are each pinned on a tiny synthetic frame (no model, hash fallback
encoder): (a) a source scenario never spans two splits, (b) a near-duplicate of a train row is
dropped from the held-out side only, (c) every class and language is present in each split (or the
shortfall is flagged, never silently dropped). A smoke test over the real committed gold set checks
the recipe runs end to end and keeps the invariants on real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cora.nlu.goldset import COLUMNS, load
from cora.nlu.labels import Intent
from cora.nlu.splits import DEDUP_THRESHOLD, group_key, make_splits

_DATA = Path(__file__).resolve().parents[2] / "data" / "nlu"


def _row(rid: str, scenario: str, utterance: str, intent: str, language: str, variant: str) -> dict:
    return dict(
        zip(
            COLUMNS,
            [
                rid,
                scenario,
                utterance,
                intent,
                language,
                variant,
                "team-generated",
                "2026-10-W1",
                "false",
                "",
                "n",
            ],
            strict=True,
        )
    )


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df["double_labeled"] = False
    return df


def _synthetic() -> pd.DataFrame:
    """Several groups per class so every class can fill all three splits, plus one ES/PT pair."""
    rows: list[dict] = []
    for intent in ("I1", "A1"):
        for g in range(10):
            rows.append(
                _row(
                    f"{intent}-{g}-1",
                    f"S-{intent}-MX-{g}",
                    f"{intent} utterance number {g}",
                    intent,
                    "es",
                    "MX",
                )
            )
    # One PT translation of S-I1-MX-0: same group key, near-identical text -> dedup target.
    rows.append(_row("I1-br-1", "S-I1-BR-MX-0", "I1 utterance number 0", "I1", "pt", "BR"))
    return _frame(rows)


def test_group_key_strips_pt_br_segment() -> None:
    assert group_key("S-I1-BR-MX-0") == "S-I1-MX-0"  # PT maps to its ES source group
    assert group_key("S-I1-MX-0") == "S-I1-MX-0"  # ES is its own group, unchanged


def test_no_source_scenario_spans_two_splits() -> None:
    df = _synthetic()
    assignment, _ = make_splits(df)
    # Build group -> set of splits; every group must sit in exactly one split.
    by_group: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        if row.id in assignment:
            by_group.setdefault(group_key(row.scenario_id), set()).add(assignment[row.id])
    assert all(len(splits) == 1 for splits in by_group.values()), by_group


def test_near_duplicate_dropped_from_heldout_only() -> None:
    df = _synthetic()
    assignment, report = make_splits(df)
    # S-I1-MX-0 (ES) and S-I1-BR-MX-0 (PT) are the same group and identical text. They share a
    # split, so the PT row is never a *cross-split* dup of its own ES source; instead craft a
    # cross-group near-duplicate to force a removal.
    df2 = pd.concat(
        [df, _frame([_row("DUP-1", "S-I1-MX-99", "I1 utterance number 0", "I1", "es", "MX")])],
        ignore_index=True,
    )
    assignment2, report2 = make_splits(df2)
    # The duplicate text exists in train (S-I1-MX-0); if DUP landed in val/test it is removed.
    removed_ids = {r["removed_id"] for r in report2["dedup"]["removed"]}
    # Either DUP shares train with its twin (not removed) or it was held out and removed - never
    # kept on the held-out side alongside an identical train row.
    if "DUP-1" in assignment2 and assignment2["DUP-1"] != "train":
        raise AssertionError("a held-out near-duplicate of a train row survived dedup")
    assert all(r["cosine"] > DEDUP_THRESHOLD for r in report2["dedup"]["removed"])
    assert removed_ids.issubset(set(df2["id"]))


def test_every_class_and_language_present_or_flagged() -> None:
    df = _synthetic()
    _assignment, report = make_splits(df)
    # Guarantee: a class/language is in every split, OR its shortfall is flagged (thin at
    # stratification, or emptied by cross-split dedup) - never silently absent.
    flagged_thin = {c["intent"] for c in report["thin_classes"]}
    for entry in report["missing_after_dedup"]:
        assert entry["empty_splits"], entry  # a flag with no empty split would be meaningless
    flagged_missing = {
        (e["column"], e["value"], s) for e in report["missing_after_dedup"] for s in e["empty_splits"]
    }
    for split in ("train", "validation", "test"):
        present = set(report["per_split_intent"][split])
        for intent in ("I1", "A1"):
            flagged = intent in flagged_thin or ("intent", intent, split) in flagged_missing
            assert intent in present or flagged, f"{intent} missing from {split} and not flagged"


def test_thin_class_is_flagged_not_dropped() -> None:
    # A class with a single group cannot fill three splits; it must be flagged, and its row kept.
    df = _frame(
        [_row(f"I1-{g}-1", f"S-I1-MX-{g}", f"saldo {g}", "I1", "es", "MX") for g in range(6)]
        + [_row("A1-0-1", "S-A1-MX-0", "congelar tarjeta", "A1", "es", "MX")]
    )
    assignment, report = make_splits(df)
    assert "A1" in {c["intent"] for c in report["thin_classes"]}
    assert "A1-0-1" in assignment  # flagged, but not dropped from the splits


def test_real_gold_set_splits_cleanly() -> None:
    df = pd.concat([load(_DATA / "gold.tsv"), load(_DATA / "gold_pt.tsv")], ignore_index=True)
    assignment, report = make_splits(df)
    # No source scenario spans two splits on the real data either.
    by_group: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        if row.id in assignment:
            by_group.setdefault(group_key(row.scenario_id), set()).add(assignment[row.id])
    assert all(len(s) == 1 for s in by_group.values())
    # Every intent appears in the train split; a held-out split may be emptied of a heavily
    # duplicated class by cross-split dedup, in which case it is flagged in missing_after_dedup.
    assert len(report["per_split_intent"]["train"]) == 16
    flagged = {(e["column"], e["value"], s) for e in report["missing_after_dedup"] for s in e["empty_splits"]}
    for split in ("validation", "test"):
        present = set(report["per_split_intent"][split])
        for intent in {i.value for i in Intent}:
            assert intent in present or ("intent", intent, split) in flagged, f"{intent} unflagged in {split}"
    # Dedup only ever removes rows; it never invents them.
    assert report["counts"]["total_rows_kept"] <= report["counts"]["total_rows_in"]
    assert (
        report["dedup"]["removed_count"]
        == report["counts"]["total_rows_in"] - report["counts"]["total_rows_kept"]
    )
