"""Tests for task 3.1: the gold-set loader, its fail-closed contract, and Cohen's kappa.

Two halves: (1) the real committed `data/nlu/gold.tsv` loads, is balanced across the 16 intents,
is ~20% double-labeled, and yields a plausible kappa; (2) a battery of malformed in-memory TSVs
each raise `GoldSetError` so the loader is proven to fail closed rather than drop a bad row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cora.nlu.goldset import COLUMNS, GoldSetError, check_pairing, cohen_kappa, load
from cora.nlu.labels import Intent

_DATA = Path(__file__).resolve().parents[2] / "data" / "nlu"
GOLD = _DATA / "gold.tsv"
GOLD_PT = _DATA / "gold_pt.tsv"

# A minimal valid two-row TSV used as the base for the malformed-file cases; each bad case mutates
# exactly one field so the test pins which rule rejected it.
_HEADER = "\t".join(COLUMNS)
_GOOD_ROW = "S-I1-MX-0-1\tS-I1-MX-0\t¿Cuánto tengo?\tI1\tes\tMX\tteam-generated\t2026-10-W1\tfalse\t\tnote"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "g.tsv"
    path.write_text(f"{_HEADER}\n{body}\n", encoding="utf-8")
    return path


# -- the real committed gold set ------------------------------------------------------------


def test_real_gold_set_loads_and_is_balanced() -> None:
    df = load(GOLD)
    # Every intent is present and the set is balanced (the authoring target: ~equal per class).
    counts = df["intent"].value_counts()
    assert set(counts.index) == {i.value for i in Intent}
    assert counts.min() >= 10, "each intent should have at least 10 utterances"
    # No dataset-seed row ever reaches a split (REQ-30 / EDA F7): the loader forbids them.
    assert "dataset-seed" not in set(df["provenance"])


def test_real_gold_set_double_labeling_and_kappa() -> None:
    df = load(GOLD)
    share = df["double_labeled"].mean()
    assert 0.15 <= share <= 0.25, f"double-labeling should be ~20%, got {share:.2%}"
    kappa = cohen_kappa(df)
    # A real, non-degenerate agreement figure: some disagreements exist, so it is neither 1.0 nor
    # negative. 0.4 is the floor for "moderate" agreement on the Landis-Koch scale.
    assert 0.4 <= kappa < 1.0, f"kappa should be substantial-but-imperfect, got {kappa}"


# -- the real committed Portuguese gold set (task 3.2) --------------------------------------


def test_real_pt_gold_set_loads_and_is_balanced() -> None:
    df = load(GOLD_PT)
    counts = df["intent"].value_counts()
    assert set(counts.index) == {i.value for i in Intent}
    assert set(df["language"]) == {"pt"}
    assert set(df["variant"]) == {"BR"}
    # Only translated/team-generated provenance (no dataset-seed ever); both are present.
    assert set(df["provenance"]) == {"translated", "team-generated"}


def test_pt_native_rewrites_are_about_twenty_percent() -> None:
    df = load(GOLD_PT)
    share = (df["provenance"] == "team-generated").mean()
    assert 0.15 <= share <= 0.25, f"native rewrites should be ~20%, got {share:.2%}"


def test_every_pt_row_records_its_provenance_in_the_note() -> None:
    # REQ-19/REQ-37: a translated or rewritten row must carry a non-blank reviewer_note.
    df = load(GOLD_PT)
    assert (df["reviewer_note"].str.strip() != "").all()
    translated = df[df["provenance"] == "translated"]
    assert translated["reviewer_note"].str.contains("translation").all()


def test_pt_pairs_every_spanish_row() -> None:
    es, pt = load(GOLD), load(GOLD_PT)
    check_pairing(es, pt)  # raises GoldSetError if any es scenario lacks a pt counterpart


def test_pairing_fails_when_a_pt_row_is_missing() -> None:
    es, pt = load(GOLD), load(GOLD_PT)
    with pytest.raises(GoldSetError, match="no pt translation"):
        check_pairing(es, pt.iloc[1:])  # drop one pt row -> an es scenario is now uncovered


def test_pairing_rejects_es_frame_as_pt() -> None:
    es = load(GOLD)
    with pytest.raises(GoldSetError, match="no BR rows"):
        check_pairing(es, es)


# -- fail-closed contract -------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldSetError):
        load(tmp_path / "does_not_exist.tsv")


def test_unknown_intent_rejected(tmp_path: Path) -> None:
    bad = _GOOD_ROW.replace("\tI1\t", "\tZZ\t", 1)
    with pytest.raises(GoldSetError, match="unknown intent"):
        load(_write(tmp_path, bad))


def test_dataset_seed_rejected(tmp_path: Path) -> None:
    bad = _GOOD_ROW.replace("team-generated", "dataset-seed")
    with pytest.raises(GoldSetError, match="dataset-seed"):
        load(_write(tmp_path, bad))


def test_blank_utterance_rejected(tmp_path: Path) -> None:
    bad = _GOOD_ROW.replace("¿Cuánto tengo?", "")
    with pytest.raises(GoldSetError, match="blank utterance"):
        load(_write(tmp_path, bad))


def test_blank_scenario_id_rejected(tmp_path: Path) -> None:
    bad = _GOOD_ROW.replace("S-I1-MX-0\t", "\t", 1)
    with pytest.raises(GoldSetError, match="blank scenario_id"):
        load(_write(tmp_path, bad))


def test_variant_language_mismatch_rejected(tmp_path: Path) -> None:
    # An es row tagged BR (a Portuguese variant) must be caught.
    bad = _GOOD_ROW.replace("\tMX\t", "\tBR\t", 1)
    with pytest.raises(GoldSetError, match="variant"):
        load(_write(tmp_path, bad))


def test_unknown_variant_rejected(tmp_path: Path) -> None:
    bad = _GOOD_ROW.replace("\tMX\t", "\tUS\t", 1)
    with pytest.raises(GoldSetError, match="variant"):
        load(_write(tmp_path, bad))


def test_double_labeled_without_label_b_rejected(tmp_path: Path) -> None:
    bad = _GOOD_ROW.replace("\tfalse\t\t", "\ttrue\t\t")  # marked double but label_b blank
    with pytest.raises(GoldSetError, match="label_b"):
        load(_write(tmp_path, bad))


def test_label_b_without_double_labeled_rejected(tmp_path: Path) -> None:
    bad = _GOOD_ROW.replace("\tfalse\t\t", "\tfalse\tI2\t")  # label_b set but not double-labeled
    with pytest.raises(GoldSetError, match="not double_labeled"):
        load(_write(tmp_path, bad))


def test_duplicate_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(GoldSetError, match="duplicate row id"):
        load(_write(tmp_path, f"{_GOOD_ROW}\n{_GOOD_ROW}"))


def test_extra_column_rejected(tmp_path: Path) -> None:
    path = tmp_path / "g.tsv"
    path.write_text(f"{_HEADER}\textra\n{_GOOD_ROW}\tx\n", encoding="utf-8")
    with pytest.raises(GoldSetError, match="columns must be exactly"):
        load(path)


def test_kappa_raises_without_double_labeled_rows(tmp_path: Path) -> None:
    df = load(_write(tmp_path, _GOOD_ROW))  # the single row is not double-labeled
    with pytest.raises(GoldSetError, match="no double-labeled rows"):
        cohen_kappa(df)
