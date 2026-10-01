# Bedrock Smoke Test - Evidence (task 0.4)

Evidence that the personal AWS account can reach Amazon Bedrock through the Converse API
with the configured inference profiles. No customer data was sent; prompts are synthetic.

## Environment

- Account type: Sign up for AWS (new), upgraded to a Paid Plan with a USD 20 monthly spend limit.
- Region: `us-east-2` (the only region enabled on this account).
- Auth: Amazon Bedrock API key (bearer token) via `AWS_BEARER_TOKEN_BEDROCK`.
  IAM Identity Center is not supported on this account type, so there is no SSO profile.
- Model: `global.anthropic.claude-haiku-4-5-20251001-v1:0` (global cross-region inference profile).

## Command

```powershell
$env:AWS_BEARER_TOKEN_BEDROCK = "<short-term Bedrock API key>"
python scripts/aws/verify_bedrock.py
```

## Result (masked)

```
model=global.anthropic.claude-haiku-4-5-20251001-v1:0 region=us-east-2 auth=bedrock-api-key
[es] status=OK latency_ms=2547 tokens_in=37 tokens_out=55 est_cost=n/a (set CORA_PRICE_* in .env)
[es] answer: El saldo es el total de dinero en tu cuenta, mientras que el saldo disponible es
     la cantidad que puedes usar inmediatamente, ya que excluye fondos congelados, retenciones
     o transacciones pendientes.
[pt] status=OK latency_ms=1235 tokens_in=33 tokens_out=45 est_cost=n/a (set CORA_PRICE_* in .env)
[pt] answer: O saldo é o valor total em sua conta, enquanto o saldo disponível é apenas a parte
     que você pode usar imediatamente, excluindo valores bloqueados ou em processamento.
```

Both languages returned `status=OK`. Latency and token counts recorded; estimated cost is `n/a`
until `CORA_PRICE_INPUT_PER_1K` / `CORA_PRICE_OUTPUT_PER_1K` are set in `.env` from the Bedrock
pricing page.

## Notes / deviations from AWS_SETUP.md

- The guide assumed a classic account with IAM Identity Center and an SSO profile `cora-dev`.
  This account type does not support Identity Center, so authentication uses a Bedrock API key
  (short-term bearer token) instead. `verify_bedrock.py` was updated to support this path.
- The guide assumed region `us-east-1`; this account only exposes `us-east-2`.
- Inference profiles use the `global.` prefix (global cross-region routing), not `us.`.
