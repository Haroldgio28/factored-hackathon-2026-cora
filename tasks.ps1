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

switch ($Target) {
    "setup"  { Invoke-Step uv @("sync", "--locked", "--all-groups") }
    "lint"   { Invoke-Step uv @("run", "ruff", "check", "."); Invoke-Step uv @("run", "ruff", "format", "--check", ".") }
    "format" { Invoke-Step uv @("run", "ruff", "check", "--fix", "."); Invoke-Step uv @("run", "ruff", "format", ".") }
    "test"   { Invoke-Step uv @("run", "pytest") }
    "data"   { Write-Pending "tasks 1.1-1.4" }
    "eval"   { Write-Pending "phase 6" }
    "demo"   { Write-Pending "task 5.4" }
}
