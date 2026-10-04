"""The one LLM interface Phase 3+ shares (design section 8, ADR-005, tech steering).

`LLMClient` is the single abstraction Phase 3 introduces. It has three real callers - the
zero-shot intent baseline (3.4), fail-closed entity extraction (3.7) and the Phase-5 agent
generator - so it clears the ponytail "no interface with one implementation" bar.

Two implementations:

- `BedrockLLMClient` calls Amazon Bedrock through the Converse API with a cross-region inference
  profile. The model id comes ONLY from `settings.bedrock_model_id` (`CORA_BEDROCK_*`); it is
  NEVER hard-coded (tech/security steering, design D3). It fails closed: unconfigured, or still
  failing after a bounded `tenacity` retry, raises `LLMUnavailable`; it does not silently fall back
  to another model id (the fallback id is a Phase-5 orchestrator concern, not this stateless call).
- `StubLLMClient` returns deterministic canned text for tests and offline runs, so CI needs no AWS
  and no network. `get_llm_client()` returns it whenever Bedrock is unconfigured or `CORA_NLU_STUB=1`.

Callers (B-zs, entities) catch `LLMUnavailable` and degrade to a safe default (`OTHER` /
`UNAVAILABLE`); the raw reply is always treated as data and validated before use (REQ-35).
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cora.settings import Settings, get_settings

logger = logging.getLogger("cora.agent.llm")

__all__ = [
    "BedrockLLMClient",
    "LLMClient",
    "LLMUnavailable",
    "StubLLMClient",
    "get_llm_client",
]

# Bounded retry for transient Bedrock errors (throttling, timeouts). Small and capped: a learned-NLU
# call that keeps failing degrades to the safe default rather than blocking the turn (REQ-40 scope
# for Phase 3; the model-level fallback id is wired in the Phase-5 orchestrator, not here - D3).
_MAX_ATTEMPTS = 3


class LLMUnavailable(RuntimeError):
    """The LLM could not be reached or is not configured; callers must fail closed."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-in/text-out contract. The reply is untrusted data the caller must validate."""

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str: ...


class StubLLMClient:
    """Deterministic canned client for tests/offline runs - no network, no AWS.

    `responses` maps a lowercased substring of the user prompt to the reply to return; the first
    matching key wins. With no match it returns `default`. This lets a test pin "this utterance ->
    this label" and "this bad utterance -> malformed reply" without any model.
    """

    def __init__(self, responses: dict[str, str] | None = None, *, default: str = "") -> None:
        self.responses = responses or {}
        self.default = default

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        haystack = user.lower()
        for needle, reply in self.responses.items():
            if needle.lower() in haystack:
                return reply
        return self.default


class BedrockLLMClient:
    """Amazon Bedrock Converse client. Model id from settings only; fails closed (design D3)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.bedrock_configured:
            raise LLMUnavailable("CORA_BEDROCK_MODEL_ID is not set")
        # boto3 is a core dep; import locally so this module stays importable where boto3 is absent.
        import boto3

        self._client = boto3.Session(
            profile_name=self._settings.aws_profile or None,
            region_name=self._settings.aws_region,
        ).client("bedrock-runtime")

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        try:
            return self._converse(system=system, user=user, max_tokens=max_tokens)
        except LLMUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any boto/transport error fails closed uniformly
            logger.warning("bedrock converse failed, failing closed: %s", type(exc).__name__)
            raise LLMUnavailable(str(exc)) from exc

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _converse(self, *, system: str, user: str, max_tokens: int) -> str:
        # Converse rejects a `system` block whose text is empty, so omit the field entirely when
        # the caller passes no system prompt (the generator does: system="").
        kwargs: dict[str, object] = {
            "modelId": self._settings.bedrock_model_id,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
        }
        if system.strip():
            kwargs["system"] = [{"text": system}]
        response = self._client.converse(**kwargs)
        blocks = response["output"]["message"]["content"]
        return "".join(block.get("text", "") for block in blocks).strip()


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the stub when Bedrock is unconfigured or `CORA_NLU_STUB=1`, else the Bedrock client.

    Fails closed by preference: anything that would need the network in a test or an unconfigured
    environment yields the stub, so no caller accidentally reaches out.
    """
    settings = settings or get_settings()
    if os.getenv("CORA_NLU_STUB") == "1" or not settings.bedrock_configured:
        return StubLLMClient()
    return BedrockLLMClient(settings)
