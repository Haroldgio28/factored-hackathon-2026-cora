"""Tests for task 6.9: the evaluation report renderer (REQ-44/47).

`render_report` is pure (dicts in, markdown out), so it is exercised from small fixtures with no
filesystem. Asserted: every produced number carries an honesty label; an ABSENT artifact renders a
clearly-labelled pending-run placeholder (never a fabricated number); the owner reproduction section
is at the top; and the honest assumptions/limitations are surfaced, not buried.
"""

from __future__ import annotations

from cora.eval.report import REPRODUCTION_SECTION, render_report


def _metrics_fixture() -> dict:
    return {
        "label": "offline measurement",
        "configs": {
            "cora": {
                "sar": {"numerator": 180, "denominator": 240, "rate": 0.75},
                "attempted": {"numerator": 230, "denominator": 240, "rate": 0.9583},
                "containment": {"numerator": 300, "denominator": 400, "rate": 0.75},
                "escalation_precision": {"numerator": 55, "denominator": 60, "rate": 0.9167},
                "escalation_recall": {"numerator": 55, "denominator": 58, "rate": 0.9483},
                "missed_escalations": 3,
                "unnecessary_escalations": 5,
                "unsafe_by_type": {
                    "disclosure_leak": {
                        "numerator": 0,
                        "denominator": 60,
                        "rate": None,
                        "wilson_95_low": 0.0,
                        "wilson_95_high": 0.06,
                        "rule_of_three_upper": 0.05,
                    }
                },
                "latency_p50_ms": {"mean": 120.0, "sd": 4.0, "n": 3},
                "latency_p95_ms": {"mean": 300.0, "sd": 10.0, "n": 3},
                "cost_per_attempted_usd": 0.0012,
                "cost_per_sar_usd": None,
            }
        },
    }


def test_full_report_labels_numbers_and_renders_tables() -> None:
    md = render_report(
        {
            "metrics": _metrics_fixture(),
            "b0": {
                "label": "offline measurement",
                "note": "Historical human-agent baseline, not comparable 1:1 with CORA's SAR.",
                "escalation_signal": "not used",
                "by_category": [
                    {
                        "category": "Transaccional",
                        "n": 10,
                        "fcr": 0.9,
                        "fcr_numerator": 9,
                        "fcr_denominator": 10,
                        "aht_seconds": 200.0,
                    }
                ],
            },
            "thresholds": {
                "n_held_out": 400,
                "n_validation": 200,
                "old_standin": {"tau_escalate": 0.40, "tau_clarify": 0.60},
                "new": {"tau_escalate": 0.35, "tau_clarify": 0.55},
                "invariant": "tau_escalate <= tau_clarify",
                "freeze_order": "select -> freeze -> promote -> test",
                "label": "offline measurement",
            },
            "manifest": {
                "total": 400,
                "counts_by_language": {"es": 200, "pt": 200},
                "counts_by_country": {"MX": 134, "CO": 133, "AR": 133},
                "seed": 42,
                "held_out_disjoint_from_nlu": True,
            },
        }
    )
    assert "offline measurement" in md
    assert "0.750 (180/240)" in md  # SAR rendered with numerator/denominator
    assert "cost / SAR".lower() in md.lower() and "not defined (SAR=0)" in md  # None -> honest text
    assert "n/a (0/60)" in md  # a None rate is never printed as 0.0
    assert "0.050" in md  # rule-of-three ceiling shown when zero events
    # The sections whose artifacts ARE present must not render the pending placeholder.
    results = md.split("## Results")[1].split("## Historical baseline B0")[0]
    assert "pending data-host run" not in results
    b0 = md.split("## Historical baseline B0")[1].split("## LLM style-judge")[0]
    assert "pending data-host run" not in b0


def test_absent_artifacts_render_pending_not_fabricated() -> None:
    md = render_report({})  # nothing produced yet
    # Every data-dependent section must say pending, and must not invent a rate.
    for marker in ["## Results", "## Historical baseline B0", "## Fairness", "## LLM style-judge"]:
        assert marker in md
    assert md.count("pending data-host run") >= 4
    # The silver-not-human caveat is stated even with no agreement artifact.
    assert "not human validation" in md


def test_reproduction_section_is_at_the_top() -> None:
    md = render_report({})
    assert REPRODUCTION_SECTION in md
    # It appears before the results section (owner requirement: reproduction at the top).
    assert md.index("## Reproduction") < md.index("## Results")
    # It names the main repo root, not the worktree, and both shells.
    assert "MAIN repo root" in md
    assert ".\\tasks.ps1 eval" in md and "make eval" in md


def test_assumptions_are_surfaced() -> None:
    md = render_report({"metrics": _metrics_fixture()})
    assert "## Assumptions & limitations" in md
    for needle in ["Handoff completeness", "Containment caveat", "output-tokens-only"]:
        assert needle in md
