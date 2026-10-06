# Running CORA on a fresh (non-corporate) PC

This guide brings CORA up from zero on a clean Windows machine and exposes the demo
through a public ngrok URL. Use it on a personal / non-corporate network: a corporate
proxy that intercepts TLS blocks ngrok's session (`x509: certificate signed by unknown
authority`) and also blocks model-weight downloads from Hugging Face.

Everything here is reproducible from the repo plus two things that never live in git:
your `.env` secrets and your AWS credentials.

---

## 0. What you need to bring (NOT in the repo)

The repo is self-contained for code, but these are gitignored and must be supplied:

| Item | Why | Where it goes |
|---|---|---|
| `.env` file | Bedrock model ids, signing keys, console key | repo root (copy from `.env.example`) |
| Bedrock API key (bearer token) | the LLM calls | `AWS_BEARER_TOKEN_BEDROCK` env var or `.env` |
| `cora-datathon` AWS profile | read-only S3 download of the dataset | `aws configure --profile cora-datathon` |
| ngrok authtoken | the public tunnel | `ngrok config add-authtoken <token>` |

The dataset itself (~0.9 GB Parquet) is NOT in git. You either re-download it from S3
(section 4) or copy the `data/` folder from this machine on a USB drive.

---

## 1. Install prerequisites

Run `scripts/bootstrap.ps1` (section 7) to check/install these automatically, or do it by hand:

```powershell
# Python 3.12 (the project pins ==3.12.*)
winget install --id Python.Python.3.12

# uv - the package manager that reads uv.lock (reproducible installs)
winget install --id astral-sh.uv

# AWS CLI v2 - needed for the S3 dataset download and Bedrock
winget install --id Amazon.AWSCLI

# ngrok - the public tunnel for the demo
winget install --id Ngrok.Ngrok

# Git (if cloning rather than copying the folder)
winget install --id Git.Git
```

Close and reopen PowerShell after installing so PATH updates.

Verify:

```powershell
python --version   # 3.12.x
uv --version
aws --version
ngrok version
```

---

## 2. Get the code

```powershell
# Option A - clone (public demo state is on main / the demo branch)
git clone https://github.com/Haroldgio28/CORA.git
cd CORA

# Option B - copy the whole folder from the corporate PC on a USB drive,
# then DELETE the local-only caches so there are no conflicts:
#   .venv  .pytest_cache  .ruff_cache  __pycache__
# uv rebuilds the venv cleanly in section 3. See section 8.
```

---

## 3. Install dependencies (reproducible, no conflicts)

`uv` creates an isolated `.venv` from the pinned `uv.lock`, so versions are identical to
this machine. Do NOT reuse a `.venv` copied from another PC - it holds absolute paths and
platform-specific binaries (torch, pyarrow) and will conflict. Always let uv rebuild it.

```powershell
# Everything (dev + analysis + nlu + ui) - matches `.\tasks.ps1 setup`
uv sync --locked --all-groups
```

If you only want to run the demo (API + UI) and skip the heavy ML/analysis groups:

```powershell
uv sync --locked --group ui
```

Note on model weights: the intent classifier can use MiniLM embeddings from Hugging Face.
On a network that blocks HF the code falls back to its download-free path (documented in
the project notes). The demo (API + UI + Bedrock) does NOT need HF weights.

---

## 4. Download the dataset (only if you need the data layer / eval)

The customer-facing demo talks to Bedrock and the tool layer; for a basic chat demo you
need the data only if you exercise balance/transaction lookups against real tables.

Configure the read-only datathon profile first (keys are provided by the organizer; never
commit them):

```powershell
aws configure set aws_access_key_id     <KEY>    --profile cora-datathon
aws configure set aws_secret_access_key <SECRET> --profile cora-datathon
aws configure set region us-east-2                --profile cora-datathon
```

Then land the dataset as Parquet (~0.9 GB; takes a while):

```powershell
uv run python scripts/data/fetch_to_parquet.py --dest data/raw_parquet --staging $env:TEMP\cora_staging
uv run python scripts/data/validate_landing.py
```

Faster alternative: copy the existing `data/` folder from the corporate PC on a USB drive
and skip the download entirely.

---

## 5. Recreate the `.env` (secrets - never in git)

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and fill in (see `.env.example` comments and `documentation/AWS_SETUP.md`):

- `CORA_BEDROCK_MODEL_ID`, `CORA_BEDROCK_JUDGE_MODEL_ID`, `CORA_BEDROCK_FALLBACK_MODEL_ID`
  - exact inference-profile ids (`global.` prefix on this account), region `us-east-2`.
- `CORA_IDENTITY_SIGNING_KEY` - any strong random string (signs session JWTs; fails closed if blank).
- `CORA_AGENT_CONSOLE_KEY` - any strong random string (guards the `/handoffs` endpoints).
- `AWS_REGION=us-east-2` for Bedrock on this account.

Bedrock auth (pick one, in `.env` or the shell):

```powershell
# Bedrock API key (this account has no IAM Identity Center) - prefer a short-term key:
$env:AWS_BEARER_TOKEN_BEDROCK = "<short-term Bedrock API key>"
```

Smoke-test Bedrock before the demo:

```powershell
uv run python scripts/aws/verify_bedrock.py
```

A successful run prints status, latency and token usage for one ES and one PT prompt.

---

## 6. Run the demo with a public URL

One-time ngrok setup:

```powershell
ngrok config add-authtoken <YOUR_NGROK_TOKEN>
```

Then launch everything (API + UI + tunnel) with the reserved domain:

```powershell
.\scripts\demo.ps1 -Domain shabby-backer-language.ngrok-free.dev
```

The public URL is `https://shabby-backer-language.ngrok-free.dev`. Share it with the judges.
Press `Ctrl+C` in that window to stop the API, UI and tunnel.

Reminder: the URL only works while this script runs and the PC is on. Each judge message
consumes real Bedrock quota.

Local-only fallback (no tunnel, you share your screen): the UI is at `http://localhost:8501`.

---

## 7. One-shot prerequisite check

`scripts/bootstrap.ps1` checks for Python/uv/AWS CLI/ngrok, offers to install any that are
missing via winget, runs `uv sync`, and copies `.env.example` to `.env` if absent. It does
NOT fill in secrets or AWS credentials (you do that manually - sections 4 and 5).

```powershell
.\scripts\bootstrap.ps1
```

---

## 8. Avoiding conflicts when copying the folder (not cloning)

If you copy the repo folder instead of cloning, delete these local-only, machine-specific
artifacts BEFORE running `uv sync` so nothing conflicts:

```powershell
Remove-Item -Recurse -Force .venv, .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

`uv sync --locked` then rebuilds `.venv` from `uv.lock`, giving byte-identical dependency
versions with no cross-machine conflicts. Keep `uv.lock` - it is the source of truth.

Do NOT copy `.env` to a shared location or commit it; recreate it per machine (section 5).
```
