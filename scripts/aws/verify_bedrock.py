"""Bedrock smoke test for CORA (task 0.4).

Sends one synthetic Spanish and one synthetic Portuguese question to the configured
Bedrock model through the Converse API and prints status, latency, token usage and
estimated cost. Sends NO customer data. Reads configuration from .env (see .env.example);
credentials come from the AWS SSO profile, never from this file.

Authentication (resolved in this order):
    1. If AWS_BEARER_TOKEN_BEDROCK is set, boto3 uses that Bedrock API key (bearer token).
       Used on accounts without IAM Identity Center; prefer a short-term key (<=12 h).
    2. Otherwise the named AWS_PROFILE (SSO) is used.

Usage (SSO):
    aws sso login --profile cora-dev
    python scripts/aws/verify_bedrock.py

Usage (Bedrock API key, PowerShell):
    $env:AWS_BEARER_TOKEN_BEDROCK = "<short-term key from the Bedrock console>"
    python scripts/aws/verify_bedrock.py
"""

from __future__ import annotations

import os
import sys
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

PROMPTS = {
    "es": "Responde en una sola frase y en español: ¿qué diferencia hay entre saldo y saldo disponible?",
    "pt": "Responda em uma única frase e em português: qual é a diferença entre saldo e saldo disponível?",
}


def _price(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


def main() -> int:
    load_dotenv()
    profile = os.getenv("AWS_PROFILE", "cora-dev")
    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("CORA_BEDROCK_MODEL_ID", "").strip()
    if not model_id:
        print("CORA_BEDROCK_MODEL_ID is empty - complete step 5.2 of documentation/AWS_SETUP.md")
        return 2

    price_in, price_out = _price("CORA_PRICE_INPUT_PER_1K"), _price("CORA_PRICE_OUTPUT_PER_1K")

    # A Bedrock API key (bearer token) takes precedence over the named profile. boto3 reads
    # AWS_BEARER_TOKEN_BEDROCK automatically, so we build a plain session without a profile.
    # We must also drop AWS_PROFILE from the environment: boto3 still honours it when set, and
    # a profile that does not exist in ~/.aws/config (the case on this account) raises
    # ProfileNotFound while the session loads its config.
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        auth = "bedrock-api-key"
        os.environ.pop("AWS_PROFILE", None)
        client = boto3.Session(region_name=region).client("bedrock-runtime")
    else:
        auth = f"profile={profile}"
        client = boto3.Session(profile_name=profile, region_name=region).client("bedrock-runtime")

    failures = 0
    print(f"model={model_id} region={region} auth={auth}")
    for lang, text in PROMPTS.items():
        start = time.perf_counter()
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": text}]}],
                inferenceConfig={"maxTokens": 120, "temperature": 0},
            )
        except (ClientError, BotoCoreError) as exc:
            failures += 1
            print(f"[{lang}] status=ERROR {type(exc).__name__}: {exc}")
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.get("usage", {})
        tin, tout = usage.get("inputTokens", 0), usage.get("outputTokens", 0)
        answer = resp["output"]["message"]["content"][0]["text"].strip()
        cost = (
            f"{tin / 1000 * price_in + tout / 1000 * price_out:.6f} USD"
            if price_in is not None and price_out is not None
            else "n/a (set CORA_PRICE_* in .env)"
        )
        print(
            f"[{lang}] status=OK latency_ms={latency_ms:.0f} "
            f"tokens_in={tin} tokens_out={tout} est_cost={cost}"
        )
        print(f"[{lang}] answer: {answer}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
