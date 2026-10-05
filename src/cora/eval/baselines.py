"""The naive single-prompt baseline B1 (task 6.2, REQ-41).

B1 is the explicit "what does CORA add?" contrast for the evaluation. It is a SINGLE LLM call with
the customer's account data pasted into the prompt and NOTHING else: no policy engine, no grounding
check, no confirmation protocol, no tool-call loop. Whatever the model says is the answer. The
deterministic judges (6.4) then score B1 and full CORA on the identical scenario suite, so the
report can attribute the safety/grounding gap to the engineering CORA adds, not to a different model.

Two security invariants still bind B1 (they are not "CORA features", they are hard rules):
- PII is masked (`tools.base.mask_pii`) before the single LLM call - the pasted account data and the
  customer utterance both pass through the masker, and a spy test asserts no raw PII reaches the LLM.
- B1 has NO money-movement / card-action capability: it only READS the customer's own products via
  the session-bound `ToolLayer` (the same per-customer authorization CORA uses to fetch data). It
  cannot move money or change a balance because no such tool exists to paste a result from.

The versioned prompt files (`agent/prompts/naive_baseline.{es,pt}.txt`) have their SHA-256 recorded
on every result (REQ-44 run metadata), reusing the `hashlib` pattern from `agent/generator.py`.

    # ponytail: B1 reuses ToolLayer.list_products for the "pasted data" (already masked, already
    # authorized) and the shared LLMClient; it is one prompt + one call, deliberately no CORA
    # machinery. The RunRecord shape is 6.3's concern - B1 returns a thin typed result the runner
    # adapts, so this file does not import a not-yet-existing runner type.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cora.agent.llm import LLMClient, LLMUnavailable, get_llm_client
from cora.tools.base import Status, mask_pii
from cora.tools.layer import ToolLayer
from cora.tools.models import GetBalanceInput

logger = logging.getLogger("cora.eval.baselines")

__all__ = ["B1NaiveLLM", "B1Result"]

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"

# Honest fallback text when the LLM is unavailable (unconfigured Bedrock / transport failure). B1
# has no template layer to fall back to - that is a CORA feature - so it simply records the outage.
_UNAVAILABLE_ES = "No puedo responder en este momento."
_UNAVAILABLE_PT = "Não consigo responder neste momento."


@dataclass(frozen=True)
class B1Result:
    """One B1 turn: the model's answer plus the run metadata the judges + runner consume.

    `text` is what B1 would show the customer (ungrounded, unchecked). `prompt_hash` is the SHA-256
    of the versioned prompt file (REQ-44). `pasted_data` is the MASKED account data that went into
    the prompt (kept for the trace/judge; never raw PII). `available` is False when the LLM could
    not be reached (so metrics can separate a bad answer from an outage).
    """

    text: str
    language: str
    prompt_hash: str
    pasted_data: str
    available: bool = True
    account_source_refs: list[str] = field(default_factory=list)


def _prompt_path(language: str) -> Path:
    path = _PROMPTS_DIR / f"naive_baseline.{language}.txt"
    if not path.exists():  # es/pt ship; any other language fails the pairing test, not a silent swap
        raise FileNotFoundError(f"missing naive_baseline prompt for language {language!r}")
    return path


def _prompt_hash(language: str) -> str:
    """SHA-256 of the versioned prompt file bytes (REQ-44 run metadata)."""
    return hashlib.sha256(_prompt_path(language).read_bytes()).hexdigest()


class B1NaiveLLM:
    """Single-prompt LLM baseline: pasted (masked) data in, direct answer out. No CORA machinery."""

    def __init__(self, client: LLMClient | None = None, *, max_tokens: int = 256) -> None:
        # Default to the shared factory: stub under CORA_NLU_STUB=1 / unconfigured Bedrock (no network).
        self._client = client if client is not None else get_llm_client()
        self._max_tokens = max_tokens

    def answer(
        self,
        utterance: str,
        language: str,
        *,
        tools: ToolLayer,
        referenced_product_id: str | None = None,
    ) -> B1Result:
        """Answer `utterance` with the customer's own (masked) product data pasted into one prompt.

        `tools` is the session-bound `ToolLayer` for THIS scenario's customer - B1 reads products
        through it (per-customer authorization), masks the rendered data AND the utterance, fills
        the versioned prompt and makes one LLM call. The reply is returned verbatim (ungrounded):
        B1 does no grounding/policy/confirmation, which is exactly the behaviour the eval contrasts.

        `referenced_product_id` is set only for an unauthorized-access scenario, so B1 faces the
        SAME foreign-resource condition as CORA (REQ-41 identical conditions). The read goes through
        the same session-bound tool layer, so the authorization boundary still denies it
        (FORBIDDEN/NOT_FOUND) and no foreign data is pasted - the eval surfaces whether the naive
        agent nonetheless answers about a resource it could not read.
        """
        pasted, refs = self._render_account_data(tools, referenced_product_id)
        prompt = (
            _prompt_path(language)
            .read_text(encoding="utf-8")
            .format(
                account_data=pasted,
                utterance=mask_pii(utterance),  # security: mask the utterance before the LLM call
            )
        )
        prompt_hash = _prompt_hash(language)
        try:
            reply = self._client.complete(system="", user=prompt, max_tokens=self._max_tokens)
        except LLMUnavailable:
            logger.info("B1 LLM unavailable; recording outage for language=%s", language)
            fallback = _UNAVAILABLE_PT if language == "pt" else _UNAVAILABLE_ES
            return B1Result(
                text=fallback,
                language=language,
                prompt_hash=prompt_hash,
                pasted_data=pasted,
                available=False,
                account_source_refs=refs,
            )
        return B1Result(
            text=reply.strip(),
            language=language,
            prompt_hash=prompt_hash,
            pasted_data=pasted,
            account_source_refs=refs,
        )

    @staticmethod
    def _render_account_data(
        tools: ToolLayer, referenced_product_id: str | None = None
    ) -> tuple[str, list[str]]:
        """Render the customer's products as the masked text B1 pastes into the prompt.

        Reuses `ToolLayer.list_products` (product numbers already masked by the tool layer); the
        whole string is run through `mask_pii` again as defence in depth before it reaches the LLM.
        Returns the pasted text and the product source refs (for the run record).

        When `referenced_product_id` is set (an unauthorized-access scenario), B1 attempts to read
        that product through the SAME session-bound tool layer, so it faces the identical foreign
        read CORA does. The authorization check denies it (FORBIDDEN/NOT_FOUND), so no foreign
        balance is pasted - only a denial note - and the session boundary is never weakened.
        """
        result = tools.list_products()
        products = result.data.products if result.data is not None else []
        lines = [
            f"- {p.product_type} {p.product_number_masked} ({p.currency}), estado: {p.product_status}"
            for p in products
        ]
        refs = [ref.ref for ref in result.source_refs]
        if referenced_product_id is not None:
            # Same foreign read as CORA; the session-bound layer denies it, so nothing leaks.
            foreign = tools.get_balance(GetBalanceInput(product_id=referenced_product_id))
            if foreign.status is not Status.OK:
                lines.append(f"- (acceso denegado al producto solicitado: {foreign.status.value})")
        if not lines:
            return "(sin productos)", refs
        return mask_pii("\n".join(lines)), refs
