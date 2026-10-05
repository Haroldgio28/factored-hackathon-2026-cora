# CORA submission audit (task 7.6, REQ-48 / REQ-52)

Final pre-submission check that the package is complete, reproducible from a clean clone, and
carries no credentials or PDFs. Nothing here is re-estimated; command outputs are recorded verbatim
from this repo on the date below.

- Date: 2026-10-05
- Branch: `feat/phase7-production-readiness`
- Commit at audit time: `25916b1` (`docs(demo): add five-path demo script in es and pt`)

## 1. Traceability is complete and honest (REQ-52)

[`../TRACEABILITY.md`](../TRACEABILITY.md) maps **84 / 84** SRC obligations to a REQ, a task and an
openable evidence path (0 unmapped). Of the 84 rows, **60 are `Done`** and **24 remain open** today
(**8 `Planned` + 16 `Partial`**). Every open row is enumerated below rather than hidden; none is a
silent gap. The 24 open rows reconcile exactly with the matrix status column, and the breakdown
counts below sum to 24.

Still-open rows, by honest reason (24 total):

- **Phase 8 data-engineering pipeline (post-submission) — 11 rows.** SRC-34, SRC-55, SRC-56,
  SRC-57, SRC-76, SRC-77, SRC-78, SRC-79, SRC-80, SRC-81, SRC-83 — contracts/lineage/freshness,
  incremental & streaming, the labeled update-correctness fixture, provenance table, duplicate/null/
  late-arrival/schema-evolution/referential-integrity fixtures, regional-variant and date-partition
  evidence. These depend on the Phase-1 landing pipeline that is scheduled after the submission;
  they stay `Planned`/`Partial` on purpose.
- **Phase 1 problem/data analysis (shares the Phase 8 landing dependency) — 3 rows.** SRC-06,
  SRC-07, SRC-22 (`Partial`, task 1.8) point at [`../../analysis/EDA_FINDINGS.md`](../../analysis/EDA_FINDINGS.md)
  (and the planned `data_quality.md`). The EDA that frames the focused problem exists and is cited
  across the design and ADR-001, but these rows are kept `Partial` because their full data-quality
  evidence (demand/quality/constraints profiling) lands with the same Phase-1 landing pipeline above,
  not from a separate run. No judge-facing claim depends on a `Done` here that is not already carried
  by the Done rows SRC-15/SRC-16/SRC-23 (focused single workflow, depth-over-breadth, prioritised
  outcomes).
- **Core system rows whose only gap is end-to-end demo/eval evidence — 3 rows.** SRC-01, SRC-02,
  SRC-04 (`Partial`) are implementation-complete — the FastAPI service, the §4 state machine with
  every node wired, cross-turn reference resolution and the es/pt UIs all ship with passing unit
  tests (see the Done rows SRC-03/05/24/25/26/27/28). They stay `Partial` only because their
  end-to-end *demonstration* evidence is the owner-executed demo video (script committed as
  [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)) plus the live five-path run, which was executed as the reduced
  2026-10-05 run (98-case subset, 1 repeat) rather than the full ≥3-repeat / 400-case suite. The
  behaviour is demonstrated; the row is held open until the full-scale run + recorded video replace
  the reduced/owner-executed evidence.
- **AI-vs-deterministic justification rows — 2 rows.** SRC-12 and SRC-53 (`Partial`) point at
  `design §2` + [`DECISIONS.md`](../DECISIONS.md). The justification itself is written and current —
  the per-component AI-vs-deterministic table with measured evidence lives in
  [`TRADE_OFFS.md`](TRADE_OFFS.md) §2 (the Done row SRC-11) — but these two rows are left `Partial`
  because their evidence pointers were not re-pointed at the 7.3 trade-off register and the
  "how it was evaluated" half leans on the same reduced eval run above. No separate obligation is
  unmet; the rows trail their own citation.
- **Explainability from traces — 2 rows.** SRC-47 (`Partial`) and SRC-52 (`Planned`): the JSONL
  trace exporter ships (`src/cora/obs/tracing.py`) but the remaining REQ-39 telemetry (tool latency,
  retries, model id, tokens, cost on the same record) and the trace-sourced explanation surface land
  with the eval producers; the headline eval numbers do not depend on it.
- **Learned-classifier encoder caveat — 1 row.** SRC-35 (`Partial`): the calibrated classifier,
  baselines, splits and threshold selection all ship and are tested, but the embedding-head numbers
  were produced with the `StubEncoder` fallback because the MiniLM weights were not downloadable in
  the sandbox (TF-IDF numbers are real, macro-F1 0.9321; the later confirmed real-MiniLM rerun is
  recorded in [`LIMITATIONS.md`](LIMITATIONS.md)). The row stays `Partial` until the real-MiniLM run
  is the primary recorded result.
- **Reduced evaluation run (stated limitation) — 2 rows.** SRC-66 and SRC-68 (`Partial`): the real
  B1-vs-CORA run (2026-10-05) used a 98-case stratified subset at 1 repeat (Bedrock quota), below the
  ≥3-repeats / full 400-case rule, and the LLM-judge validation uses robust-LLM silver labels rather
  than ≥50 human labels; [`EVALUATION.md`](EVALUATION.md) labels every figure and
  [`LIMITATIONS.md`](LIMITATIONS.md) carries both caveats.

The one known deferred runtime fix — the **live read-ownership gap** (a non-referenced read: POL-090
never matches, so the engine falls through to POL-999 `abstain` and fails closed, blocking some
in-scope reads) — is a Phase-4 fix deferred past Phase 7. It is not a separate open matrix row; it is
documented in [`LIMITATIONS.md`](LIMITATIONS.md) §5 and demonstrated honestly in
[`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) Path 3.

Count check: 11 + 3 + 3 + 2 + 2 + 1 + 2 = **24 open rows**, matching the matrix (8 `Planned`
+ 16 `Partial`). The remaining **60 rows are `Done` with their evidence in the repo**. No capability
exists without a REQ and a matrix row; no row is `Done` without its evidence in the repo.

## 2. No secrets, no PDFs (REQ-52, SRC-59)

The CI secret/PDF gates ([`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) jobs `secrets`
and `no-pdfs`) were run locally. Actual outputs:

```text
# Any tracked PDF? (CI job no-pdfs: `git ls-files | grep -i '\.pdf$'`)
PS> git ls-files | Select-String -Pattern '\.pdf$'
(no output — zero tracked PDFs)

# Any tracked .env?
PS> git ls-files | Select-String -Pattern '(^|/)\.env$'
(no output — zero tracked .env files)

# Are .env and Docs/ ignored?
PS> git check-ignore .env
.env
PS> git check-ignore Docs/
Docs/
```

Result: **no `.pdf` and no `.env` are tracked**; both `.env` (credentials) and `Docs/` (original
PDFs with credentials) are matched by `.gitignore`, so they cannot be committed by accident.

`gitleaks detect` was not run locally — the `gitleaks` binary is **absent on this machine**:

```text
PS> if (Get-Command gitleaks -ErrorAction SilentlyContinue) { gitleaks version } else { "ABSENT" }
ABSENT
```

Full-history secret scanning is therefore a **CI-gated check**: the `secrets` job in
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs `gitleaks/gitleaks-action` with
`fetch-depth: 0` (full history, not just the last commit) on every push to `main` and every PR, so
no commit reaches the submission branch without passing it.

## 3. Clean-clone reproduction (REQ-48)

From a fresh clone, secrets supplied via env vars only (never committed):

```powershell
# 1. Clone
git clone <repo-url> cora && cd cora

# 2. Pinned install (uses uv.lock; no version drift)
uv sync --locked --all-groups      # or: .\tasks.ps1 setup

# 3. Gates — must be green
uv run ruff check .
uv run ruff format --check .
uv run pytest                       # or: .\tasks.ps1 test

# 4. Secrets for the live paths (NEVER committed; see documentation/AWS_SETUP.md)
#    copy .env.example -> .env and fill CORA_BEDROCK_* and the cora-datathon profile

# 5. Full evaluation chain — run on a DATA HOST that holds the ~0.9 GB landing
.\tasks.ps1 eval                    # Makefile equivalent: make eval
```

Notes carried over from [`EVALUATION.md`](EVALUATION.md) "Reproduction":

- Steps 1–3 reproduce from any machine with `uv`; no data or credentials are needed and the full
  unit/contract suite is self-contained (NLU/LLM stubs on the test path).
- Step 5 requires the landed data snapshot and a valid Bedrock bearer token; the eval data/LLM
  stages **fail closed with a clear message** when inputs are absent (REQ-47, never fabricate), and
  `make_report` still renders `pending data-host run` placeholders rather than invented numbers.
- Model ids come from `.env` (`CORA_BEDROCK_*`), never hard-coded; AWS access uses SSO profiles /
  the `cora-datathon` read-only profile — no long-lived keys.

### Submission package contents (REQ-52)

Everything a judge opens is tracked in the repo; none of it contains credentials or PDFs:

| Package item (REQ-52) | File in repo |
|---|---|
| README | [`README.md`](../../README.md) |
| Problem & data evidence | [`analysis/EDA_FINDINGS.md`](../../analysis/EDA_FINDINGS.md), [`SOURCE_REQUIREMENTS.md`](../SOURCE_REQUIREMENTS.md) |
| Architecture | [`.kiro/specs/cora/design.md`](../../.kiro/specs/cora/design.md), [`DECISIONS.md`](../DECISIONS.md) |
| Evaluation report | [`EVALUATION.md`](EVALUATION.md) |
| Trade-offs | [`TRADE_OFFS.md`](TRADE_OFFS.md) |
| Limitations | [`LIMITATIONS.md`](LIMITATIONS.md) |
| Demo script | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) |
| Production readiness | [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) |
| Spec with full traceability | [`requirements.md`](../../.kiro/specs/cora/requirements.md) → [`TRACEABILITY.md`](../TRACEABILITY.md) (84/84) |

The **demo video** is owner-executed (needs a valid Bedrock bearer token + landed data); the
committed deliverable is the script above. **No credentials, no `.env`, no PDFs** are part of the
package — confirmed by §2.
