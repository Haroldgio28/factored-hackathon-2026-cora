# AWS Account Setup (from scratch) - CORA

Step-by-step guide to prepare a **personal AWS account** for CORA development with
**Amazon Bedrock** (ADR-005, accepted 2026-10-01). Designed to be executed from the
**Kiro IDE**: every step is a checkbox, every command runs in the Kiro terminal (PowerShell),
and task `0.4` in [`.kiro/specs/cora/tasks.md`](../.kiro/specs/cora/tasks.md) points here.

> **Security rules for this guide**
> - Never paste access keys, passwords or MFA codes into chat, code, notes or `.env` committed files.
> - We use **IAM Identity Center (SSO)** short-lived credentials - no long-lived access keys.
> - The organizer's datathon bucket keeps using its own profile `cora-datathon`; this guide creates a
>   separate profile `cora-dev` for *your* account. Never mix them.

**Region:** `us-east-1` (N. Virginia) for everything in this account - widest Bedrock model and
cross-region inference coverage. *Why not us-east-2 (the organizer bucket's region)?* Your account
does not read that bucket with these credentials; Bedrock availability matters more.

---

## Step 1 - Create the account and lock down root (console, ~15 min)

- [ ] 1.1 Sign up at [aws.amazon.com](https://aws.amazon.com/) with a dedicated email (e.g. `you+aws-cora@...`), choose **Personal** account, add a payment card, pick the **Basic (free) support plan**.
- [ ] 1.2 Sign in as **root** -> top-right account menu -> **Security credentials** -> **Assign MFA device** (authenticator app or passkey).
- [ ] 1.3 On the same page confirm there are **no root access keys** (create none, ever).
- [ ] 1.4 Account menu -> **Account** -> set **Alternate contacts** (billing/security) to your email.

## Step 2 - Cost guardrails before anything else (console, ~5 min)

Bedrock is **pay-per-token, not free tier**. Set alarms first.

- [ ] 2.1 **Billing and Cost Management -> Budgets -> Create budget -> Use a template -> Monthly cost budget**. Amount: e.g. **USD 30**. Alerts to your email at 50%, 80%, 100% (actual) and 100% (forecasted).
- [ ] 2.2 **Billing preferences -> Alert preferences**: enable *Receive Free Tier usage alerts* and *Receive CloudWatch billing alerts*.
- [ ] 2.3 **Cost Anomaly Detection -> Create monitor** (AWS services, daily summary to your email).

Expected CORA dev cost: low single-digit USD/day with a Haiku-class model (we log tokens and cost per turn, REQ-39).

## Step 3 - IAM Identity Center: your everyday identity (console, ~15 min)

- [ ] 3.1 Switch the console region to **us-east-1**. Open **IAM Identity Center -> Enable** (organization instance; this also creates an AWS Organization - free).
- [ ] 3.2 **Settings -> Authentication**: require **MFA** for every sign-in.
- [ ] 3.3 **Users -> Add user**: your name + email (a different identity than root). Accept the invitation email and set a password + MFA.
- [ ] 3.4 **Groups -> Create group** `cora-admins`, add your user.
- [ ] 3.5 **Permission sets -> Create**:
  - `CoraAdmin` - predefined **AdministratorAccess**, session duration **4 h**. Used only for setup and infrastructure (Phase 8).
  - `CoraDeveloper` - **custom**: paste the inline policy from [`infra/iam/cora-developer-policy.json`](../infra/iam/cora-developer-policy.json), session duration **8 h**. Used for daily development (least privilege).
- [ ] 3.6 **AWS accounts -> select your account -> Assign users or groups** -> `cora-admins` -> both permission sets.
- [ ] 3.7 Copy the **AWS access portal URL** from the Identity Center dashboard (looks like `https://d-xxxxxxxxxx.awsapps.com/start`). It is not a secret, but keep it out of the repo.

*Why Identity Center and not an IAM user with access keys:* credentials expire automatically, MFA is enforced, nothing long-lived can leak from a laptop or a commit.

## Step 4 - CLI profile `cora-dev` (Kiro terminal, ~5 min)

AWS CLI v2 is already installed on this PC (`aws --version` to confirm; otherwise install the MSI from the AWS docs).

- [ ] 4.1 Configure SSO (interactive - answer the prompts):

```powershell
aws configure sso --profile cora-dev
# SSO session name:        cora
# SSO start URL:           <your access portal URL from 3.7>
# SSO region:              us-east-1
# SSO registration scopes: sso:account:access   (default)
# -> browser opens: approve
# Account:                 <your account>
# Role:                    CoraDeveloper
# Default client region:   us-east-1
# Output format:           json
```

- [ ] 4.2 Verify:

```powershell
aws sso login --profile cora-dev
aws sts get-caller-identity --profile cora-dev
```

Expected: an ARN containing `AWSReservedSSO_CoraDeveloper`. Repeat 4.1 with `--profile cora-admin` and role `CoraAdmin` when you need infrastructure rights.

## Step 5 - Bedrock access (console + CLI, ~10 min)

Since Oct 2025 serverless models are **enabled automatically on first invocation**; the old
"Model access" page is retired. **Anthropic (Claude) models still require a one-time First Time Use (FTU) form** per account.

- [ ] 5.1 Console (us-east-1) -> **Amazon Bedrock -> Model catalog** -> open an **Anthropic Claude** model (Haiku-class) -> submit the **use-case details** form. Suggested text: *"Personal development of a banking customer-service assistant prototype for the Factored AI & Data Hackathon 2026, using synthetic data only."* Approval is usually immediate.
- [ ] 5.2 List the inference profiles available to you and pick the model IDs (do not hard-code from memory - versions change):

```powershell
aws bedrock list-inference-profiles --profile cora-dev --region us-east-1 `
  --query "inferenceProfileSummaries[?contains(inferenceProfileId,'claude') || contains(inferenceProfileId,'nova')].[inferenceProfileId,status]" `
  --output table
```

  Choose, and write down:
  - `CORA_BEDROCK_MODEL_ID` - generation/extraction: newest **Claude Haiku** `us.anthropic...` profile (cheap, fast, strong es/pt).
  - `CORA_BEDROCK_JUDGE_MODEL_ID` - LLM-judge / zero-shot baseline: a **Claude Sonnet** `us.anthropic...` profile.
  - `CORA_BEDROCK_FALLBACK_MODEL_ID` - fallback: an **Amazon Nova Lite** `us.amazon.nova-lite...` profile (no FTU needed).

  *Why inference profiles (`us.` prefix):* cross-region routing gives higher throughput and fewer throttles than a single-region model id.

- [ ] 5.3 Smoke test from the CLI (replace the id):

```powershell
aws bedrock-runtime converse --profile cora-dev --region us-east-1 `
  --model-id <CORA_BEDROCK_MODEL_ID> `
  --messages '[{\"role\":\"user\",\"content\":[{\"text\":\"Responde en una frase: ¿qué es un saldo disponible?\"}]}]' `
  --inference-config '{\"maxTokens\":120,\"temperature\":0}'
```

If you get `AccessDeniedException ... aws-marketplace`, the permission set lacks the marketplace subscribe permissions (they are included in the provided policy - re-check step 3.5). If you get a use-case error, finish 5.1.

## Step 6 - Wire the repo (Kiro terminal, ~5 min)

- [ ] 6.1 Create your local env file from the template (the real `.env` is gitignored):

```powershell
Copy-Item .env.example .env
```

  Fill in `.env`: `AWS_PROFILE=cora-dev`, `AWS_REGION=us-east-1`, and the three model ids from 5.2. **No keys go in this file** - SSO provides them.

- [ ] 6.2 Install the smoke-test dependency and run the Python check (sends a synthetic es + pt question, prints latency, tokens and estimated cost; sends **no customer data**):

```powershell
python -m pip install boto3==1.35.36 python-dotenv==1.0.1
python scripts/aws/verify_bedrock.py
```

  Expected: two answers (Spanish, Portuguese), `status=OK`, latency and token counts. The output is evidence for task 0.4 - save it to `documentation/reports/bedrock_smoke_test.md` (masked, no account id).

## Step 7 - Using it from the Kiro IDE

- [ ] 7.1 Open the folder `C:\Users\1872146\Documents\CORA` in Kiro. Steering in `.kiro/steering/` loads automatically (product, tech, security, language, git workflow).
- [ ] 7.2 Open `.kiro/specs/cora/tasks.md` -> task **0.4** -> *Start task*. Kiro follows this guide and the spec.
- [ ] 7.3 Each work day: run `aws sso login --profile cora-dev` once in the Kiro terminal (sessions last 8 h).
- [ ] 7.4 Keep the Kiro agent away from secrets: do not ask it to open `.env` or `~/.aws/*`; it only needs the variable *names*.

## Step 8 - Later (Phase 8, AWS migration) - not needed now

- `aws sso login --profile cora-admin` then `cdk bootstrap aws://<account>/us-east-1 --profile cora-admin`.
- Project bucket `cora-<account>-us-east-1` (block public access, SSE-S3, versioning), Glue database `cora_curated`, Athena workgroup `cora` with a results bucket and per-query scan limit.
- Extend `CoraDeveloper` with the S3/Glue/Athena statements already scoped in the policy file.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Token has expired` / `SSO session expired` | 8 h session ended | `aws sso login --profile cora-dev` |
| `AccessDeniedException` on `converse` mentioning marketplace | missing subscribe permission | re-apply policy (step 3.5) |
| `ResourceNotFoundException` / `ValidationException` model id | wrong id or not an inference profile | redo step 5.2 and copy the exact id |
| `ThrottlingException` | low default quota | use the `us.` profile; add retries (REQ-40); request a quota increase in Service Quotas |
| Budget email | spend crossed threshold | check Cost Explorer by service; lower `maxTokens`, cache, switch to Nova Lite |
