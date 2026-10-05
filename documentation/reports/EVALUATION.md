# CORA evaluation (Phase 6)

This report is **regenerable**: it renders from the artifacts under `data/eval/`. Numbers are labelled `offline measurement`, `simulation` or `projection` (REQ-47); any section without a produced artifact shows a _pending data-host run_ placeholder rather than a fabricated value. No claim of production improvement is made here.

## Reproduction - running the real evaluation from the repo

> Run this from the **MAIN repo root** `c:\Users\1872146\Documents\CORA`, **NOT** the worktree:
> the ~0.9 GB landing lives there (`data\raw_parquet` ~= 857 MB, 7684 files on the owner's disk).
> `data\curated` does not exist and is **optional** - `LocalSource` reads the raw landing directly.
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
.\tasks.ps1 eval
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

## Setup & versions

Run metadata (model id, B1/CORA prompt hashes, code commit, data snapshot) is stamped on every `RunRecord` in `data/eval/runs.jsonl` by `run_eval` (REQ-44); the headline values appear there once the data-host run has produced the records.

Scenario suite manifest: _pending data-host run_ (`data/eval/scenarios_manifest.json`).

## Operational confidence thresholds (6.1b)

_pending data-host run_ - the Phase-3 sweep returned a degenerate pair and `rules.yaml` ships a conservative stand-in (`tau_escalate=0.40`, `tau_clarify=0.60`). Running `rederive_thresholds` on the data host (with the real 384-d MiniLM encoder) derives and freezes the operational pair and writes `data/eval/thresholds_derivation.json`.

## Results - B1 (naive LLM) vs CORA

_pending data-host run_ - run `run_eval` + `compute_metrics` on the data host (`data/eval/metrics.json`).

### Unsafe outcomes by type

_pending data-host run_

## Historical baseline B0 (6.8)

_pending data-host run_ - run `historical_baseline` on the data host where the `call_center_interactions` landing exists (`data/eval/b0_historical.json`).

## LLM style-judge validation (6.5)

_pending data-host run_ - the base judge and the robust silver-label judge run on the data host with `CORA_BEDROCK_JUDGE_MODEL_ID` set (`data/eval/judge_agreement.json`). **Agreement is judge-vs-robust-LLM silver labels, not human validation; the 50 human-labelled samples REQ-45 asks for remain a documented limitation.**

## Fairness breakdown (6.7)

_pending data-host run_ - run `fairness_report` on the data host (`data/eval/fairness.json`). Breaks SAR / unsafe / escalation / latency down by language, country and segment; thin subgroups are marked small-sample with wide CIs.

## Failures & error analysis

Failures are included and analysed, not hidden (REQ-47). The adversarial suite (missing/incorrect data, expired session, unauthorized access, prompt injection, tool failure, multilingual ambiguity, plus credit-request and money-movement guardrail cases) exercises the fail-closed paths; the per-type unsafe counts above are the headline failure signal, with Wilson CIs so a zero count is reported as a rule-of-three ceiling, not as a claim of zero risk. Once the data-host run produces `runs.jsonl` + `metrics.json`, cite two or three concrete failing transcripts (scenario id + what went wrong) here.

## Assumptions & limitations (honest, not buried)

- **Handoff completeness.** The runner persists only a boolean `handoff_complete` (customer_request + reason present) on each `RunRecord`; the full REQ-16 field-by-field handoff-completeness judge needs the whole `HandoffPackage` on the record, which is not yet wired. Known gap: the handoff-completeness number is coarse until the package is persisted.
- **Containment caveat.** Containment is share-not-escalated; a contained but wrong or unsafe answer still counts as contained, so containment is never read alone - always with SAR and the unsafe-by-type rates.
- **Cost is an output-tokens-only estimate.** The runner captures no provider token usage, so cost approximates tokens from text length and counts output tokens only, priced at the supplied published per-1k rates (see `cost_assumptions` in `metrics.json`). It is a `projection`, not a billed figure.
- **Encoder/data dependence.** Numbers produced without the real 384-d MiniLM encoder or the full landing are labelled accordingly; the stub cannot produce operational-threshold or intent-dependent figures.
- **PT provenance.** Portuguese scenarios are team-generated/translated (the dataset is Spanish-only); PT results carry that provenance.
- **Silver, not human, judge validation.** The 6.5 agreement is judge-vs-robust-LLM silver labels; the 50 human-labelled samples REQ-45 asks for remain a documented limitation.
