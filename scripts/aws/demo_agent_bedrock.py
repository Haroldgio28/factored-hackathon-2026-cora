"""End-to-end Phase 4 demo against REAL Bedrock (not the stub).

What the stubbed unit tests cannot show: that the 4.2 generator's LLM-polish path, the es<->pt
language guard and the 4.3 grounding checker behave correctly on REAL model output. This script
exercises exactly that path with `BedrockLLMClient`, in es and pt, and prints for each turn the
LLM-polished text, whether it was polished, whether grounding or the language guard blocked it,
and the prompt hash recorded for the trace. The model id is printed up front (evaluation rules:
model ids recorded next to prompt hashes) and the whole transcript is written under
`documentation/reports/` so it is citable submission evidence.

It sends NO customer data: every "fact" is a synthetic literal. Auth is the same Bedrock API key
(bearer token) the smoke test uses - export AWS_BEARER_TOKEN_BEDROCK first (see
documentation/AWS_SETUP.md). CORA_NLU_STUB must NOT be set, or we would be testing the stub again.

The script exits non-zero if the model was never actually reached (every turn fell back to a
template with polished=False): a run with a bad token, wrong region or missing model id must NOT
be mistaken for working evidence.

Usage (PowerShell):
    $env:AWS_BEARER_TOKEN_BEDROCK = "<short-term key>"
    uv run python scripts/aws/demo_agent_bedrock.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from cora.agent.generator import GeneratedResponse, generate
from cora.agent.llm import BedrockLLMClient, LLMUnavailable
from cora.agent.templates import Outcome
from cora.settings import Settings
from cora.tools.base import Result, SourceRef, Status
from cora.tools.models import BalanceData

_REPORT_PATH = Path(__file__).resolve().parents[2] / "documentation" / "reports" / "demo_agent_bedrock.txt"


def _banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _show(label: str, response: GeneratedResponse, *, latency_ms: float) -> None:
    print(
        f"[{label}] polished={response.polished} grounding_blocked={response.grounding_blocked} "
        f"language_blocked={response.language_blocked} "
        f"prompt_hash={(response.prompt_hash or 'none')[:12]} latency_ms={latency_ms:.0f}"
    )
    print(f"[{label}] text: {response.text}")


def _run_demo(client: BedrockLLMClient, model_id: str) -> list[GeneratedResponse]:
    """Run every demo turn, printing each, and return the responses so main can check them."""
    responses: list[GeneratedResponse] = []
    print(f"model={model_id}")

    # A grounded ANSWER: the facts come from a (synthetic) tool Result for THIS turn. The generator
    # fills the template with these figures, the LLM only rephrases, and grounding checks the
    # rephrasing against the same tool Result.
    balance_result = Result[BalanceData](
        status=Status.OK,
        data=BalanceData(
            product_id="PRD-demo",
            currency="COP",
            current_balance=1250.75,
            credit_limit=None,
            available_credit=1100.00,
        ),
        source_refs=[SourceRef(table="products", ref="PRD-demo")],
        as_of=datetime(2026, 10, 4, 9, 0),
    )
    es_facts = "Tu saldo es 1.250,75 COP y tu saldo disponible es 1.100,00 COP."
    pt_facts = "Seu saldo é 1.250,75 COP e seu saldo disponível é 1.100,00 COP."

    _banner("1. Grounded ANSWER, real LLM polish (es + pt)")
    for label, lang, facts in (("es", "es", es_facts), ("pt", "pt", pt_facts)):
        start = time.perf_counter()
        resp = generate(
            Outcome.ANSWER,
            lang,
            fields={"facts": facts},
            tool_results=[balance_result],
            client=client,
        )
        _show(label, resp, latency_ms=(time.perf_counter() - start) * 1000)
        responses.append(resp)

    _banner("2. REFUSE (money movement) + ABSTAIN_ROUTE (credit) polished (es + pt)")
    for outcome in (Outcome.REFUSE, Outcome.ABSTAIN_ROUTE):
        for lang in ("es", "pt"):
            start = time.perf_counter()
            resp = generate(outcome, lang, tool_results=[], client=client)
            _show(f"{outcome.value}/{lang}", resp, latency_ms=(time.perf_counter() - start) * 1000)
            responses.append(resp)

    # 3. Figure-free smoke test on real model output. The CLARIFY template carries no figures, so a
    # faithful polish stays figure-free and grounding passes - this section does NOT exercise the
    # grounding REJECTION branch (that deterministic block is covered by tests/unit/test_grounding.py
    # with injected ungrounded figures). What it DOES show on real output is the honest common case:
    # the polished text alongside whether any figure was extracted, i.e. that the model added none.
    _banner("3. Figure-free polish smoke test on real model output (es)")
    start = time.perf_counter()
    clarify = generate(Outcome.CLARIFY, "es", tool_results=[], client=client)
    _show("clarify/es", clarify, latency_ms=(time.perf_counter() - start) * 1000)
    outcome_note = (
        "grounding blocked an invented figure"
        if clarify.grounding_blocked
        else "polish was figure-free (nothing to ground), so the template-equivalent text passed"
    )
    print(f"[clarify/es] grounding outcome: {outcome_note}")
    responses.append(clarify)

    print(
        "\nDemo complete. Grounding and template fallbacks are deterministic; the LLM only ever "
        "rephrased a fact-filled, PII-masked draft."
    )
    return responses


def main() -> int:
    load_dotenv()
    if os.getenv("CORA_NLU_STUB") == "1":
        print("CORA_NLU_STUB=1 is set - unset it to exercise real Bedrock. Aborting.")
        return 2
    if not os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        print("AWS_BEARER_TOKEN_BEDROCK is empty - export your short-term key first (AWS_SETUP.md).")
        return 2

    # The bearer-token path needs boto3 to NOT look up a named profile. Two independent places read
    # it: (1) BedrockLLMClient passes Settings.aws_profile to boto3, so we pass a profile-less
    # Settings; (2) botocore ALSO reads AWS_PROFILE straight from the process env, and load_dotenv()
    # above just loaded the .env's empty `AWS_PROFILE=` into os.environ as "", which botocore treats
    # as a profile named "" and fails with ProfileNotFound. So drop an empty AWS_PROFILE from the env
    # too (verify_bedrock.py does the same). A non-empty AWS_PROFILE is left alone.
    if not os.environ.get("AWS_PROFILE", "").strip():
        os.environ.pop("AWS_PROFILE", None)
    settings = Settings(aws_profile="")
    model_id = settings.bedrock_model_id.strip()
    if not model_id:
        print("CORA_BEDROCK_MODEL_ID is empty - complete step 5.2 of documentation/AWS_SETUP.md")
        return 2

    try:
        client = BedrockLLMClient(settings)
    except LLMUnavailable as exc:
        print(f"Bedrock not configured: {exc}")
        return 2

    # Capture the whole transcript so it can be both printed and written as citable evidence.
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        responses = _run_demo(client, model_id)
    transcript = buffer.getvalue()
    print(transcript, end="")

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"CORA Phase 4.2 real-Bedrock demo\nmodel={model_id}\n"
        f"generated={datetime.now().isoformat(timespec='seconds')}\n{'=' * 70}\n"
    )
    _REPORT_PATH.write_text(header + transcript, encoding="utf-8")
    print(f"\nTranscript written to {_REPORT_PATH}")

    # Honest-evidence guard: if NO turn was actually polished, the model was never really reached
    # (bad token / wrong region / missing id) and every line is a template fallback. Fail loudly so
    # a broken configuration is never filed as a working run.
    if not any(r.polished for r in responses):
        print("\nNo turn was polished by the model - treating this run as a FAILURE (see above).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
