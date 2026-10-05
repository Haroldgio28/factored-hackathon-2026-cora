# CORA developer tasks for Windows (mirror of the Makefile).
# Usage: .\tasks.ps1 setup|lint|format|test|data|eval|demo
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("setup", "lint", "format", "test", "data", "eval", "demo")]
    [string]$Target
)
# No $ErrorActionPreference = "Stop": PowerShell 5.1 treats native stderr (uv progress) as fatal.
# Failures are detected through $LASTEXITCODE instead.

function Invoke-Step([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Write-Pending([string]$Where) {
    Write-Host "not implemented yet - $Where"
    exit 1
}

# Run a stage but DO NOT abort the chain on a non-zero exit: the eval data/LLM stages fail closed
# with a clear message when their inputs are absent (REQ-47, never fabricate), and a partial run
# must still reach make_report (which renders pending-run placeholders).
function Invoke-Tolerant([string[]]$Arguments) {
    & uv @Arguments
    if ($LASTEXITCODE -ne 0) { Write-Host "  (stage skipped - inputs absent; see message above)" }
}

switch ($Target) {
    "setup"  { Invoke-Step uv @("sync", "--locked", "--all-groups") }
    "lint"   { Invoke-Step uv @("run", "ruff", "check", "."); Invoke-Step uv @("run", "ruff", "format", "--check", ".") }
    "format" { Invoke-Step uv @("run", "ruff", "check", "--fix", "."); Invoke-Step uv @("run", "ruff", "format", ".") }
    "test"   { Invoke-Step uv @("run", "pytest") }
    "data"   { Write-Pending "tasks 1.1-1.4" }
    "eval"   {
        # Full evaluation chain (Phase 6). Run from the MAIN repo root where the ~0.9 GB landing
        # lives (see documentation/reports/EVALUATION.md "Reproduction"). Data/LLM stages fail
        # closed if inputs are absent; make_report always renders.
        Invoke-Tolerant @("run", "python", "scripts/eval/build_scenarios.py")
        Invoke-Tolerant @("run", "python", "scripts/eval/rederive_thresholds.py")
        Invoke-Tolerant @("run", "python", "scripts/eval/run_eval.py")
        Invoke-Tolerant @("run", "python", "scripts/eval/compute_metrics.py")
        Invoke-Tolerant @("run", "python", "scripts/eval/fairness_report.py")
        Invoke-Tolerant @("run", "python", "scripts/eval/historical_baseline.py")
        Invoke-Step uv @("run", "python", "scripts/eval/make_report.py")
    }
    "demo"   { Write-Pending "task 5.4" }
}
