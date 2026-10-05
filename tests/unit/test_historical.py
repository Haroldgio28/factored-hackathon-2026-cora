"""Tests for task 6.8: the historical baseline B0 (REQ-41/47).

B0 is exercised over a TINY self-built `call_center_interactions` Parquet partition (the
`test_eval_baselines` landing pattern), with NO network and NO full-history dependency. Asserted:
FCR and AHT are computed correctly per `reason_category`; a category with no known `was_resolved`
reports `fcr=None` (not 0.0); `was_escalated` is NOT used as an escalation-truth signal; and the
output carries the `offline measurement` label plus the "not comparable 1:1" caveat.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.eval.historical import B0_LABEL, B0_NOT_COMPARABLE_NOTE, compute_b0


def _write_fact(root: Path, df: pd.DataFrame) -> None:
    """Write df as the single Hive partition call_center_interactions/year=2026/month=06/day=15."""
    part = root / "call_center_interactions" / "year=2026" / "month=06" / "day=15"
    part.mkdir(parents=True, exist_ok=True)
    df.to_parquet(part / "call_center_interactions_20260615.parquet", index=False)


@pytest.fixture
def landing(tmp_path: Path) -> Path:
    # (category, was_resolved, was_escalated, duration_seconds). Two Transaccional (one resolved,
    # one not -> FCR 0.5), one Queja resolved (FCR 1.0), one Producto with was_resolved NULL (FCR
    # undefined). was_escalated is deliberately the OPPOSITE of was_resolved to prove it is not read
    # as resolution/escalation truth.
    rows = [
        ("Transaccional", True, False, 100.0),
        ("Transaccional", False, True, 300.0),
        ("Queja", True, True, 600.0),
        ("Producto", None, False, None),
    ]
    df = pd.DataFrame(rows, columns=["reason_category", "was_resolved", "was_escalated", "duration_seconds"])
    df["was_resolved"] = df["was_resolved"].astype("boolean")
    df["was_escalated"] = df["was_escalated"].astype("boolean")
    _write_fact(tmp_path, df)
    return tmp_path


def test_fcr_and_aht_by_category(landing: Path) -> None:
    b0 = compute_b0(LocalSource(root=landing))
    by_cat = {row["category"]: row for row in b0["by_category"]}

    assert by_cat["Transaccional"]["fcr"] == pytest.approx(0.5)  # 1 of 2 resolved
    assert by_cat["Transaccional"]["aht_seconds"] == pytest.approx(200.0)  # mean(100, 300)
    assert by_cat["Queja"]["fcr"] == pytest.approx(1.0)
    assert by_cat["Queja"]["aht_seconds"] == pytest.approx(600.0)


def test_undefined_fcr_is_none_not_zero(landing: Path) -> None:
    by_cat = {row["category"]: row for row in compute_b0(LocalSource(root=landing))["by_category"]}
    # Producto has only a NULL was_resolved -> denominator 0 -> undefined, reported as None.
    assert by_cat["Producto"]["fcr"] is None
    assert by_cat["Producto"]["fcr_denominator"] == 0
    assert by_cat["Producto"]["aht_seconds"] is None  # all duration_seconds NULL


def test_was_escalated_is_not_used_as_truth(landing: Path) -> None:
    b0 = compute_b0(LocalSource(root=landing))
    # The output must not expose a was_escalated-derived rate, and must say so explicitly.
    assert "not used" in b0["escalation_signal"]
    for row in b0["by_category"]:
        assert "escalat" not in " ".join(row.keys()).lower()
    # FCR tracks was_resolved, not was_escalated: Transaccional FCR is 0.5 (resolved), and would
    # be 0.5 the other way too, so pin Queja where resolved(1.0) != not-escalated(0.0).
    queja = next(row for row in b0["by_category"] if row["category"] == "Queja")
    assert queja["fcr"] == pytest.approx(1.0)


def test_output_carries_honest_labels(landing: Path) -> None:
    b0 = compute_b0(LocalSource(root=landing))
    assert b0["label"] == B0_LABEL == "offline measurement"
    assert b0["note"] == B0_NOT_COMPARABLE_NOTE
    assert "not comparable 1:1" in b0["note"]
