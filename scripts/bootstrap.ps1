<#
.SYNOPSIS
    Prepare a fresh Windows PC to run the CORA demo.

.DESCRIPTION
    Checks for the four prerequisites (Python 3.12, uv, AWS CLI, ngrok), offers to install
    any that are missing via winget, then installs the Python dependencies from the pinned
    uv.lock and creates a .env from the template if one does not exist.

    What it does NOT do (you must do these manually - they involve secrets):
      - fill in .env values (Bedrock model ids, signing keys)       -> see section 5 of DEPLOY_NEW_PC.md
      - configure the cora-datathon AWS profile for the dataset     -> see section 4
      - set the ngrok authtoken                                     -> ngrok config add-authtoken <token>
      - download the dataset                                        -> scripts/data/fetch_to_parquet.py

    Full guide: documentation/DEPLOY_NEW_PC.md

.PARAMETER All
    Run `uv sync --locked --all-groups` (dev + analysis + nlu + ui). Default installs only
    the `ui` group needed for the demo (API + UI), which is much lighter (no torch).

.PARAMETER AutoInstall
    Install missing tools via winget without prompting for each one.

.EXAMPLE
    .\scripts\bootstrap.ps1
    Check prerequisites, prompt before installing, sync the demo (ui) deps.

.EXAMPLE
    .\scripts\bootstrap.ps1 -All -AutoInstall
    Install everything missing without prompts and sync all dependency groups.
#>
[CmdletBinding()]
param(
    [switch]$All,
    [switch]$AutoInstall
)

# No strict stop: winget and uv write progress to stderr, which PS 5.1 can treat as fatal.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== CORA bootstrap ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot`n"

# toolName -> winget id. Checked in order.
$tools = [ordered]@{
    "python" = "Python.Python.3.12"
    "uv"     = "astral-sh.uv"
    "aws"    = "Amazon.AWSCLI"
    "ngrok"  = "Ngrok.Ngrok"
}

function Test-Tool([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Install-Tool([string]$name, [string]$wingetId) {
    if (-not (Test-Tool "winget")) {
        Write-Host "  winget not available - install $name manually (id: $wingetId)" -ForegroundColor Red
        return
    }
    $go = $AutoInstall
    if (-not $go) {
        $ans = Read-Host "  '$name' is missing. Install $wingetId via winget now? [y/N]"
        $go = ($ans -eq "y" -or $ans -eq "Y")
    }
    if ($go) {
        Write-Host "  installing $wingetId ..." -ForegroundColor Yellow
        winget install --id $wingetId --accept-source-agreements --accept-package-agreements --silent
    } else {
        Write-Host "  skipped $name - the demo will not run without it" -ForegroundColor DarkYellow
    }
}

Write-Host "--- Checking prerequisites ---" -ForegroundColor Cyan
$missing = @()
foreach ($name in $tools.Keys) {
    if (Test-Tool $name) {
        Write-Host "  [ok]      $name" -ForegroundColor Green
    } else {
        Write-Host "  [missing] $name" -ForegroundColor Yellow
        $missing += $name
    }
}

foreach ($name in $missing) {
    Install-Tool $name $tools[$name]
}

if ($missing.Count -gt 0) {
    Write-Host "`nSome tools were just installed. If any command is still 'not found'," -ForegroundColor DarkYellow
    Write-Host "close and reopen PowerShell so PATH refreshes, then re-run this script." -ForegroundColor DarkYellow
}

# --- Python dependencies via uv.lock ---
if (Test-Tool "uv") {
    Write-Host "`n--- Installing Python dependencies (uv.lock) ---" -ForegroundColor Cyan
    if ($All) {
        Write-Host "  uv sync --locked --all-groups" -ForegroundColor DarkGray
        uv sync --locked --all-groups
    } else {
        Write-Host "  uv sync --locked --group ui   (demo only; use -All for everything)" -ForegroundColor DarkGray
        uv sync --locked --group ui
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  uv sync failed - see the error above." -ForegroundColor Red
    } else {
        Write-Host "  dependencies installed." -ForegroundColor Green
    }
} else {
    Write-Host "`nSkipping dependency install: uv not available yet." -ForegroundColor DarkYellow
}

# --- .env from template ---
Write-Host "`n--- Environment file ---" -ForegroundColor Cyan
$envPath = Join-Path $RepoRoot ".env"
if (Test-Path $envPath) {
    Write-Host "  .env already exists - leaving it untouched." -ForegroundColor Green
} else {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envPath
    Write-Host "  created .env from .env.example - now fill in the secrets (see DEPLOY_NEW_PC.md section 5)." -ForegroundColor Yellow
}

Write-Host "`n=== Next steps (manual, involve secrets) ===" -ForegroundColor Cyan
Write-Host '  1. Edit .env: Bedrock model ids, CORA_IDENTITY_SIGNING_KEY, CORA_AGENT_CONSOLE_KEY'
Write-Host '  2. Set the Bedrock key:   $env:AWS_BEARER_TOKEN_BEDROCK = "your-short-term-key"'
Write-Host '  3. (data layer) configure the cora-datathon profile + run scripts/data/fetch_to_parquet.py'
Write-Host '  4. ngrok config add-authtoken your-token'
Write-Host '  5. Smoke test:  uv run python scripts/aws/verify_bedrock.py'
Write-Host '  6. Launch:      .\scripts\demo.ps1 -Domain shabby-backer-language.ngrok-free.dev'
Write-Host "`nFull guide: documentation/DEPLOY_NEW_PC.md" -ForegroundColor DarkGray
