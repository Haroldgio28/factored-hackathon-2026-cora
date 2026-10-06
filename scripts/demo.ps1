<#
.SYNOPSIS
    Launch the CORA demo locally and expose the UI through a public ngrok URL.

.DESCRIPTION
    Starts three processes for a live demo:
      1. FastAPI service  (uvicorn, cora.api.app:create_app)  on 127.0.0.1:8000
      2. Streamlit UI      (ui/customer_app.py)                on 127.0.0.1:8501
      3. ngrok tunnel      exposing the UI port publicly over HTTPS

    The UI talks to the API over localhost, so only the Streamlit port (8501) is
    tunneled. The API stays private. Nothing here changes app logic or secrets:
    Bedrock credentials and all other config come from the gitignored .env, loaded
    by the app itself.

    This is a DEMO tunnel, not a deployment. The public URL works only while this
    script runs and your machine is on. Press Ctrl+C to tear everything down.

.PARAMETER Domain
    Optional reserved ngrok domain (e.g. "cora-demo.ngrok-free.app"). Reserve one
    once at https://dashboard.ngrok.com/domains so the URL is the SAME every run and
    can be shared with judges ahead of time. Omit it to get a random ephemeral URL.

.PARAMETER ApiPort
    Port for the FastAPI service. Default 8000.

.PARAMETER UiPort
    Port for the Streamlit UI (the port that gets tunneled). Default 8501.

.EXAMPLE
    .\scripts\demo.ps1
    Random public URL, printed by ngrok in its own window.

.EXAMPLE
    .\scripts\demo.ps1 -Domain cora-demo.ngrok-free.app
    Stable reserved URL you can send in advance.

.NOTES
    One-time setup:
      winget install --id Ngrok.Ngrok
      ngrok config add-authtoken <YOUR_TOKEN>   # from https://dashboard.ngrok.com
#>
[CmdletBinding()]
param(
    [string]$Domain,
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"

# Resolve the repo root from this script's location so the command works from any cwd.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Assert-Command {
    param([string]$Name, [string]$Hint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "'$Name' not found on PATH. $Hint"
        exit 1
    }
}

Assert-Command -Name "uv"    -Hint "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
Assert-Command -Name "ngrok" -Hint "Install ngrok: winget install --id Ngrok.Ngrok  then  ngrok config add-authtoken <TOKEN>"

$processes = @()

function Stop-All {
    Write-Host "`nShutting down demo processes..." -ForegroundColor Yellow
    foreach ($p in $processes) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch { }
        }
    }
}

try {
    Write-Host "Starting CORA API on 127.0.0.1:$ApiPort ..." -ForegroundColor Cyan
    $api = Start-Process -FilePath "uv" `
        -ArgumentList @(
            "run", "uvicorn", "cora.api.app:create_app", "--factory",
            "--host", "127.0.0.1", "--port", "$ApiPort"
        ) `
        -WorkingDirectory $RepoRoot -PassThru
    $processes += $api

    # Give the API a moment to bind before the UI starts calling it.
    Start-Sleep -Seconds 3

    Write-Host "Starting Streamlit UI on 127.0.0.1:$UiPort ..." -ForegroundColor Cyan
    # Tunnel-friendly server flags also live in .streamlit/config.toml; passed here too
    # so the script is self-contained even if that file changes.
    $ui = Start-Process -FilePath "uv" `
        -ArgumentList @(
            "run", "streamlit", "run", "ui/customer_app.py",
            "--server.port", "$UiPort",
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ) `
        -WorkingDirectory $RepoRoot -PassThru
    $processes += $ui

    Start-Sleep -Seconds 4

    $ngrokArgs = @("http", "$UiPort")
    if ($Domain) {
        $ngrokArgs += @("--domain", $Domain)
        Write-Host "Opening public tunnel at https://$Domain ..." -ForegroundColor Green
    } else {
        Write-Host "Opening public tunnel (random URL - see the ngrok window / http://127.0.0.1:4040) ..." -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "CORA demo is starting. Share the ngrok HTTPS URL with the judges." -ForegroundColor Green
    Write-Host "Inspect traffic and copy the URL at: http://127.0.0.1:4040" -ForegroundColor DarkGray
    Write-Host "Press Ctrl+C in this window to stop the API, UI and tunnel." -ForegroundColor DarkGray
    Write-Host ""

    # ngrok runs in the foreground; its own output shows the public URL. When it exits
    # (Ctrl+C), the finally block tears down the API and UI too.
    & ngrok @ngrokArgs
}
finally {
    Stop-All
}
