"""Fail-closed entity extraction from a customer utterance (task 3.7, design section 7b).

The LLM only PROPOSES candidate entities (dates, an amount+currency, a merchant, a product
reference); it never decides anything. Values are resolved against real data by deterministic
tool lookups in Phase 5 (P1 "the LLM proposes, deterministic code decides").

The flow is, in order (security steering):

1. `mask_pii(utterance)` FIRST - no raw PII (email, phone, document number, address) reaches
   Bedrock. The masker is the single one in `cora.tools.base` (one PII home, HIGH-1).
2. Call the shared `LLMClient` with the versioned `entity_extraction.txt` prompt on the masked
   text, requesting JSON.
3. Parse and validate against `ExtractedEntities`. On ANY malformed JSON, schema violation, empty
   reply or transport failure -> `EntitiesResult(status=UNAVAILABLE, entities=None)`: disclose
   nothing, extract nothing. The caller treats that as ambiguous and routes to clarify via
   `PolicyInput.ambiguous_entity`.

Schema leniency scope (NIT-5): `ExtractedEntities` uses `extra="ignore"` - the opposite of the
tool-layer `extra="forbid"` in `tools/models.py` - BECAUSE the model may volunteer fields we do not
consume; ignoring them is safe, while missing required structure or a bad value still fails closed.
This leniency is scoped to the entity JSON only; the gold-set TSV loader rejects unknown columns.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cora.agent.llm import LLMClient, LLMUnavailable, get_llm_client
from cora.tools.base import mask_pii

logger = logging.getLogger("cora.nlu.entities")

__all__ = ["EntitiesResult", "ExtractedEntities", "ExtractionStatus", "extract_entities"]

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "agent" / "prompts" / "entity_extraction.txt"

# Currencies the FX/tool layer knows (mirrors tools.models._CURRENCIES). An amount in any other
# currency fails validation -> fail closed, rather than silently passing an unknown code downstream.
_CURRENCIES = frozenset({"USD", "MXN", "COP", "ARS"})


class ExtractionStatus(StrEnum):
    """Outcome of an extraction attempt. UNAVAILABLE is the single fail-closed sink."""

    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"


class ExtractedEntities(BaseModel):
    """Candidate entities the LLM proposed. All optional; `extra="ignore"` (NIT-5, see module doc).

    These are candidates only - never trusted as fact and never used for authorization. Downstream
    tools validate them against real data (P1/P2).
    """

    # extra="ignore": the model may volunteer fields we do not use; dropping them is safe. This is
    # DELIBERATELY the opposite of tools.models (extra="forbid"), scoped to the entity JSON only.
    model_config = ConfigDict(extra="ignore", frozen=True)

    date: dt.date | None = None
    date_from: dt.date | None = None
    date_to: dt.date | None = None
    amount: Decimal | None = Field(default=None, ge=0)
    # The source currency of an amount (I1-I4) and, for an FX conversion (I5), the target to
    # convert it into. Both candidates only; the FX tool validates and resolves the real rate.
    currency: str | None = None
    to_currency: str | None = None
    merchant: str | None = None
    product_ref: str | None = None

    def model_post_init(self, _context: object) -> None:
        # A currency outside the known set fails closed (invalid, not silently kept).
        for code in (self.currency, self.to_currency):
            if code is not None and code not in _CURRENCIES:
                raise ValueError(f"unknown currency: {code!r}")


class EntitiesResult(BaseModel):
    """The envelope the caller receives: a status plus entities only on OK (fail closed otherwise)."""

    model_config = ConfigDict(frozen=True)

    status: ExtractionStatus
    entities: ExtractedEntities | None = None


_UNAVAILABLE = EntitiesResult(status=ExtractionStatus.UNAVAILABLE, entities=None)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _parse(reply: str) -> EntitiesResult:
    """Validate an (untrusted) LLM reply into entities; anything invalid -> UNAVAILABLE."""
    text = reply.strip()
    if not text:
        return _UNAVAILABLE
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.info("entity extraction: non-JSON reply, failing closed")
        return _UNAVAILABLE
    if not isinstance(payload, dict):
        logger.info("entity extraction: JSON reply was not an object, failing closed")
        return _UNAVAILABLE
    try:
        entities = ExtractedEntities.model_validate(payload)
    except (ValidationError, ValueError):
        logger.info("entity extraction: schema violation, failing closed")
        return _UNAVAILABLE
    return EntitiesResult(status=ExtractionStatus.OK, entities=entities)


def extract_entities(utterance: str, *, client: LLMClient | None = None) -> EntitiesResult:
    """Extract candidate entities from `utterance`, failing closed on any error (design section 7b).

    PII is masked before the LLM call; a malformed/empty reply or an unavailable LLM yields
    `UNAVAILABLE` so the turn routes to clarify. The reply is always treated as data and validated.
    """
    masked = mask_pii(utterance)
    llm = client if client is not None else get_llm_client()
    prompt = _load_prompt().format(utterance=masked)
    try:
        reply = llm.complete(system="", user=prompt, max_tokens=256)
    except LLMUnavailable:
        return _UNAVAILABLE
    return _parse(reply)


if __name__ == "__main__":  # tiny self-check with a canned stub (no network)
    from cora.agent.llm import StubLLMClient

    ok = extract_entities(
        "quiero ver mis compras del 2024-01-05",
        client=StubLLMClient(default='{"date": "2024-01-05", "amount": null}'),
    )
    assert ok.status is ExtractionStatus.OK and ok.entities is not None
    assert ok.entities.date == dt.date(2024, 1, 5)
    bad = extract_entities("texto", client=StubLLMClient(default="no soy json"))
    assert bad.status is ExtractionStatus.UNAVAILABLE and bad.entities is None
    print("entities stub self-check OK")
