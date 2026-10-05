"""Render the Phase 6 evaluation report from whatever artifacts exist (task 6.9, REQ-44/47).

Pure code (stdlib only): `render_report(artifacts)` turns the loaded eval JSON artifacts into the
`EVALUATION.md` markdown string. The CLI (`scripts/eval/make_report.py`) loads the files and writes
the result, so this function stays unit-testable from fixtures with no filesystem.

Honesty discipline (REQ-47), enforced here, not left to the operator:

- Every results number is rendered with its `offline measurement` / `simulation` / `projection`
  label, and shows numerator/denominator or mean +/- sd rather than a bare rate.
- Where an artifact is ABSENT (no data-host run yet) the section renders a clearly-labelled
  `_pending data-host run_` placeholder - never a fabricated number. Re-running `make eval` /
  `.\tasks.ps1 eval` on the data host fills the real numbers in.
- The known assumptions/limitations carried from earlier reviews (handoff-completeness not fully
  persisted by the runner, containment caveat, cost = output-tokens-only estimate) are surfaced in
  their own section, not buried.

`artifacts` is a mapping of artifact name -> loaded JSON (or None when the file was absent):
`metrics`, `fairness`, `b0`, `thresholds`, `manifest`, `agreement`. Any may be None.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["REPRODUCTION_SECTION", "render_report"]

_PENDING = "_pending data-host run_"


# The owner-requested reproduction section (verbatim deliverable, folded in at the TOP). It is a
# fixed string because it documents a fixed procedure; it carries no measured number.
REPRODUCTION_SECTION = """\
## Reproduction - running the real evaluation from the repo

> Run this from the **MAIN repo root** `c:\\Users\\1872146\\Documents\\CORA`, **NOT** the worktree:
> the ~0.9 GB landing lives there (`data\\raw_parquet` ~= 857 MB, 7684 files on the owner's disk).
> `data\\curated` does not exist and is **optional** - `LocalSource` reads the raw landing directly.
> To build a curated layer first: `uv run python scripts/data/run_pipeline.py`.

**Prerequisites**

- `.env` with `CORA_BEDROCK_MODEL_ID` set, and - for the 6.5 robust silver-label judge -
  `CORA_BEDROCK_JUDGE_MODEL_ID` pointed at a higher-capability model.
- The real 384-d MiniLM encoder weights downloadable/cached (HF access). If HF is blocked the
  intent encoder falls back to the 64-d stub and the operational-threshold / intent-dependent
  numbers **cannot** be produced (same caveat as the Phase-3 note); those figures stay labelled.

**One-shot**

```powershell
# PowerShell (owner's shell)
.\\tasks.ps1 eval
```

```bash
# make equivalent
make eval
```

The `eval` target runs, in order and failing closed if an input is absent (never fabricating):
`build_scenarios -> rederive_thresholds -> run_eval -> compute_metrics -> fairness_report ->
historical_baseline -> make_report`.

**Per-stage commands**

```powershell
uv run python scripts/eval/build_scenarios.py      # -> scenarios.jsonl (+ manifest)
uv run python scripts/eval/rederive_thresholds.py  # 6.1b operational tau -> thresholds_derivation.json
uv run python scripts/eval/run_eval.py             # B1 + CORA x3, faults on -> runs.jsonl
uv run python scripts/eval/compute_metrics.py      # -> metrics.json
uv run python scripts/eval/fairness_report.py      # -> fairness.json
uv run python scripts/eval/historical_baseline.py  # 6.8 B0 -> b0_historical.json
uv run python scripts/eval/make_report.py          # -> documentation/reports/EVALUATION.md
```

Any number produced without the real encoder/data is labelled `offline measurement` /
`simulation` / `projection` accordingly; this report is regenerable - rerun the chain on the data
host to replace every `pending data-host run` placeholder with the measured value.
"""


def _fmt_proportion(p: Mapping[str, Any] | None) -> str:
    """Render a Proportion dict as `rate (num/den)`, or `n/a (0 denominator)` when undefined."""
    if not p:
        return _PENDING
    num, den, rate = p.get("numerator"), p.get("denominator"), p.get("rate")
    if rate is None:
        return f"n/a (0/{den})"
    return f"{rate:.3f} ({num}/{den})"


def _fmt_stat(s: Mapping[str, Any] | None, *, unit: str = "") -> str:
    """Render a Stat dict as `mean +/- sd (n=N)`."""
    if not s:
        return _PENDING
    return f"{s['mean']:.1f} +/- {s['sd']:.1f}{unit} (n={s['n']})"


def _fmt_cost(value: float | None) -> str:
    return "not defined (SAR=0)" if value is None else f"${value:.6f}"


def _results_table(metrics: Mapping[str, Any] | None) -> str:
    """A B1-vs-CORA table of the headline metrics, or a pending placeholder."""
    if not metrics:
        return (
            f"{_PENDING} - run `run_eval` + `compute_metrics` on the data host (`data/eval/metrics.json`).\n"
        )
    configs: Mapping[str, Any] = metrics.get("configs", {})
    label = metrics.get("label", "offline measurement")
    header = "| Metric | " + " | ".join(sorted(configs)) + " |"
    sep = "| --- | " + " | ".join("---" for _ in configs) + " |"
    rows = [header, sep]

    def row(name: str, render) -> str:  # noqa: ANN001 - small local
        cells = " | ".join(render(configs[c]) for c in sorted(configs))
        return f"| {name} | {cells} |"

    rows.append(row("SAR (num/den)", lambda m: _fmt_proportion(m.get("sar"))))
    rows.append(row("Attempted share", lambda m: _fmt_proportion(m.get("attempted"))))
    rows.append(row("Containment*", lambda m: _fmt_proportion(m.get("containment"))))
    rows.append(row("Escalation precision", lambda m: _fmt_proportion(m.get("escalation_precision"))))
    rows.append(row("Escalation recall", lambda m: _fmt_proportion(m.get("escalation_recall"))))
    rows.append(row("Missed escalations", lambda m: str(m.get("missed_escalations", _PENDING))))
    rows.append(row("Unnecessary escalations", lambda m: str(m.get("unnecessary_escalations", _PENDING))))
    rows.append(row("Latency p50 ms", lambda m: _fmt_stat(m.get("latency_p50_ms"))))
    rows.append(row("Latency p95 ms", lambda m: _fmt_stat(m.get("latency_p95_ms"))))
    rows.append(row("Cost / attempted", lambda m: _fmt_cost(m.get("cost_per_attempted_usd"))))
    rows.append(row("Cost / SAR", lambda m: _fmt_cost(m.get("cost_per_sar_usd"))))
    table = "\n".join(rows)
    caveat = "*Containment is not a quality metric alone (see Limitations)."
    return f"_All figures labelled `{label}`._\n\n{table}\n\n{caveat}\n"


def _unsafe_table(metrics: Mapping[str, Any] | None) -> str:
    """Unsafe-outcome-by-type table with Wilson CI and the rule-of-three ceiling when zero."""
    if not metrics:
        return f"{_PENDING}\n"
    configs: Mapping[str, Any] = metrics.get("configs", {})
    rows = [
        "| Config | Unsafe type | num/den | Wilson 95% | rule-of-3 upper (if 0) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for config in sorted(configs):
        for kind, u in sorted(configs[config].get("unsafe_by_type", {}).items()):
            den = u.get("denominator")
            rate = u.get("rate")
            num = u.get("numerator")
            frac = f"n/a (0/{den})" if rate is None else f"{num}/{den}"
            ci = f"[{u.get('wilson_95_low', 0):.3f}, {u.get('wilson_95_high', 1):.3f}]"
            r3 = u.get("rule_of_three_upper")
            r3s = "-" if r3 is None else f"{r3:.3f}"
            rows.append(f"| {config} | {kind} | {frac} | {ci} | {r3s} |")
    return "\n".join(rows) + "\n"


def _versions_section(metrics: Mapping[str, Any] | None, manifest: Mapping[str, Any] | None) -> str:
    """Setup + versions (model id / prompt hashes / commit / data snapshot live on the run records)."""
    lines = [
        "Run metadata (model id, B1/CORA prompt hashes, code commit, data snapshot) is stamped on "
        "every `RunRecord` in `data/eval/runs.jsonl` by `run_eval` (REQ-44); the headline values "
        "appear there once the data-host run has produced the records.",
    ]
    if manifest:
        lines.append("")
        lines.append(
            f"Scenario suite: {manifest.get('total', _PENDING)} cases, "
            f"languages {manifest.get('counts_by_language', _PENDING)}, "
            f"countries {manifest.get('counts_by_country', _PENDING)}, "
            f"seed {manifest.get('seed', _PENDING)}, "
            f"held-out disjoint from NLU training = {manifest.get('held_out_disjoint_from_nlu', _PENDING)}."
        )
    else:
        lines.append("")
        lines.append(f"Scenario suite manifest: {_PENDING} (`data/eval/scenarios_manifest.json`).")
    return "\n".join(lines) + "\n"


def _thresholds_section(thresholds: Mapping[str, Any] | None) -> str:
    """Fold in the 6.1b operational-threshold derivation note."""
    if not thresholds:
        return (
            f"{_PENDING} - the Phase-3 sweep returned a degenerate pair and `rules.yaml` ships a "
            "conservative stand-in (`tau_escalate=0.40`, `tau_clarify=0.60`). Running "
            "`rederive_thresholds` on the data host (with the real 384-d MiniLM encoder) derives "
            "and freezes the operational pair and writes `data/eval/thresholds_derivation.json`.\n"
        )
    old = thresholds.get("old_standin", {})
    new = thresholds.get("new", {})
    return (
        f"Re-derived on the held-out REQ-42 suite ({thresholds.get('n_held_out', '?')} cases, "
        f"validation n={thresholds.get('n_validation', '?')}), labelled "
        f"`{thresholds.get('label', 'offline measurement')}`:\n\n"
        f"- Old stand-in: tau_escalate={old.get('tau_escalate')}, tau_clarify={old.get('tau_clarify')}\n"
        f"- New operational: tau_escalate={new.get('tau_escalate')}, tau_clarify={new.get('tau_clarify')}\n"
        f"- Invariant: {thresholds.get('invariant', 'tau_escalate <= tau_clarify')}\n"
        f"- Freeze order: {thresholds.get('freeze_order', '')}\n"
    )


def _b0_section(b0: Mapping[str, Any] | None) -> str:
    """Historical B0 FCR/AHT by category."""
    if not b0:
        return (
            f"{_PENDING} - run `historical_baseline` on the data host where the "
            "`call_center_interactions` landing exists (`data/eval/b0_historical.json`).\n"
        )
    rows = ["| Category | n | FCR | AHT (s) |", "| --- | --- | --- | --- |"]
    for r in b0.get("by_category", []):
        if r.get("fcr") is None:
            fcr = "n/a"
        else:
            fcr = f"{r['fcr']:.3f} ({r.get('fcr_numerator')}/{r.get('fcr_denominator')})"
        aht = "n/a" if r.get("aht_seconds") is None else f"{r['aht_seconds']:.1f}"
        rows.append(f"| {r.get('category')} | {r.get('n')} | {fcr} | {aht} |")
    note = b0.get("note", "")
    escal = b0.get("escalation_signal", "")
    return (
        f"_Labelled `{b0.get('label', 'offline measurement')}`. {note}_\n\n"
        + "\n".join(rows)
        + f"\n\nEscalation signal: {escal}.\n"
    )


def _agreement_section(agreement: Mapping[str, Any] | None) -> str:
    """6.5 LLM-judge vs robust-LLM silver-label agreement (simulation)."""
    if not agreement:
        return (
            f"{_PENDING} - the base judge and the robust silver-label judge run on the data host "
            "with `CORA_BEDROCK_JUDGE_MODEL_ID` set (`data/eval/judge_agreement.json`). "
            "**Agreement is judge-vs-robust-LLM silver labels, not human validation; the 50 "
            "human-labelled samples REQ-45 asks for remain a documented limitation.**\n"
        )
    return (
        f"_Labelled `{agreement.get('label', 'simulation')}`._ "
        f"{agreement.get('note', '')}\n\n"
        f"- Cohen's kappa: {agreement.get('cohen_kappa', _PENDING)}\n"
        f"- Percent agreement: {agreement.get('percent_agreement', _PENDING)}\n"
        f"- Usable samples: {agreement.get('n_usable_samples', _PENDING)} "
        f"(>= 50 required: {agreement.get('meets_min_samples', _PENDING)})\n"
    )


def _fairness_section(fairness: Mapping[str, Any] | None) -> str:
    if not fairness:
        return (
            f"{_PENDING} - run `fairness_report` on the data host (`data/eval/fairness.json`). "
            "Breaks SAR / unsafe / escalation / latency down by language, country and segment; "
            "thin subgroups are marked small-sample with wide CIs.\n"
        )
    dims = ", ".join(sorted(fairness.get("dimensions", []))) or "(none)"
    return (
        f"_Labelled `{fairness.get('label', 'offline measurement')}`._ "
        f"Breakdown dimensions: {dims}. Subgroups below the min-n are marked small-sample and "
        "excluded from the gap vote; a flagged gap means best-vs-worst spread exceeded the stated "
        "tolerance. See `data/eval/fairness.json` for the full per-subgroup numbers.\n"
    )


def render_report(artifacts: Mapping[str, Any]) -> str:
    """Render EVALUATION.md from the loaded (or absent) eval artifacts (REQ-44/47)."""
    metrics = artifacts.get("metrics")
    fairness = artifacts.get("fairness")
    b0 = artifacts.get("b0")
    thresholds = artifacts.get("thresholds")
    manifest = artifacts.get("manifest")
    agreement = artifacts.get("agreement")

    parts = [
        "# CORA evaluation (Phase 6)",
        "",
        "This report is **regenerable**: it renders from the artifacts under `data/eval/`. Numbers "
        "are labelled `offline measurement`, `simulation` or `projection` (REQ-47); any section "
        f"without a produced artifact shows a {_PENDING} placeholder rather than a fabricated value. "
        "No claim of production improvement is made here.",
        "",
        REPRODUCTION_SECTION,
        "## Setup & versions",
        "",
        _versions_section(metrics, manifest),
        "## Operational confidence thresholds (6.1b)",
        "",
        _thresholds_section(thresholds),
        "## Results - B1 (naive LLM) vs CORA",
        "",
        _results_table(metrics),
        "### Unsafe outcomes by type",
        "",
        _unsafe_table(metrics),
        "## Historical baseline B0 (6.8)",
        "",
        _b0_section(b0),
        "## LLM style-judge validation (6.5)",
        "",
        _agreement_section(agreement),
        "## Fairness breakdown (6.7)",
        "",
        _fairness_section(fairness),
        "## Failures & error analysis",
        "",
        "Failures are included and analysed, not hidden (REQ-47). The adversarial suite "
        "(missing/incorrect data, expired session, unauthorized access, prompt injection, tool "
        "failure, multilingual ambiguity, plus credit-request and money-movement guardrail cases) "
        "exercises the fail-closed paths; the per-type unsafe counts above are the headline failure "
        "signal, with Wilson CIs so a zero count is reported as a rule-of-three ceiling, not as a "
        "claim of zero risk. Once the data-host run produces `runs.jsonl` + `metrics.json`, cite "
        "two or three concrete failing transcripts (scenario id + what went wrong) here.",
        "",
        "## Assumptions & limitations (honest, not buried)",
        "",
        "- **Handoff completeness.** The runner persists only a boolean `handoff_complete` "
        "(customer_request + reason present) on each `RunRecord`; the full REQ-16 field-by-field "
        "handoff-completeness judge needs the whole `HandoffPackage` on the record, which is not "
        "yet wired. Known gap: the handoff-completeness number is coarse until the package is "
        "persisted.",
        "- **Containment caveat.** Containment is share-not-escalated; a contained but wrong or "
        "unsafe answer still counts as contained, so containment is never read alone - always with "
        "SAR and the unsafe-by-type rates.",
        "- **Cost is an output-tokens-only estimate.** The runner captures no provider token usage, "
        "so cost approximates tokens from text length and counts output tokens only, priced at the "
        "supplied published per-1k rates (see `cost_assumptions` in `metrics.json`). It is a "
        "`projection`, not a billed figure.",
        "- **Encoder/data dependence.** Numbers produced without the real 384-d MiniLM encoder or "
        "the full landing are labelled accordingly; the stub cannot produce operational-threshold "
        "or intent-dependent figures.",
        "- **PT provenance.** Portuguese scenarios are team-generated/translated (the dataset is "
        "Spanish-only); PT results carry that provenance.",
        "- **Silver, not human, judge validation.** The 6.5 agreement is judge-vs-robust-LLM silver "
        "labels; the 50 human-labelled samples REQ-45 asks for remain a documented limitation.",
        "",
    ]
    return "\n".join(parts)
