"""Tests for task 3.7: fail-closed entity extraction (design section 7b, security steering).

Driven by `StubLLMClient` with canned replies: no network, no AWS. The contract is fail closed -
good JSON parses into typed candidates, and ANY malformed/empty/invalid reply yields UNAVAILABLE
with no entities, which the caller treats as ambiguous (routes to clarify).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from cora.agent.llm import StubLLMClient
from cora.nlu.entities import ExtractionStatus, extract_entities


def test_valid_json_parses_into_typed_entities() -> None:
    client = StubLLMClient(
        default='{"date": "2024-01-05", "amount": "150.50", "currency": "MXN", "merchant": "OXXO"}'
    )
    result = extract_entities("quiero ver la compra en OXXO del 5 de enero", client=client)
    assert result.status is ExtractionStatus.OK
    assert result.entities is not None
    assert result.entities.date == date(2024, 1, 5)
    assert result.entities.amount == Decimal("150.50")
    assert result.entities.currency == "MXN"
    assert result.entities.merchant == "OXXO"


def test_malformed_json_fails_closed() -> None:
    result = extract_entities("texto", client=StubLLMClient(default="no soy json"))
    assert result.status is ExtractionStatus.UNAVAILABLE
    assert result.entities is None


def test_empty_reply_fails_closed() -> None:
    result = extract_entities("texto", client=StubLLMClient(default=""))
    assert result.status is ExtractionStatus.UNAVAILABLE
    assert result.entities is None


def test_json_array_fails_closed() -> None:
    # A JSON array is valid JSON but not the expected object -> fail closed.
    result = extract_entities("texto", client=StubLLMClient(default="[1, 2, 3]"))
    assert result.status is ExtractionStatus.UNAVAILABLE


def test_unknown_currency_fails_closed() -> None:
    client = StubLLMClient(default='{"amount": "10", "currency": "BRL"}')
    assert extract_entities("texto", client=client).status is ExtractionStatus.UNAVAILABLE


def test_negative_amount_fails_closed() -> None:
    client = StubLLMClient(default='{"amount": "-5"}')
    assert extract_entities("texto", client=client).status is ExtractionStatus.UNAVAILABLE


def test_extra_keys_are_ignored_not_errored() -> None:
    # NIT-5: the entity schema is lenient (extra="ignore") - a volunteered field is dropped, not
    # a failure. This is deliberately opposite to the tool-layer extra="forbid".
    client = StubLLMClient(default='{"date": "2024-02-02", "unexpected_field": "whatever"}')
    result = extract_entities("texto", client=client)
    assert result.status is ExtractionStatus.OK
    assert result.entities is not None
    assert result.entities.date == date(2024, 2, 2)


def test_pii_is_masked_before_the_llm_sees_it() -> None:
    # The stub records the user prompt it last saw; assert the raw email never reached it.
    class RecordingStub(StubLLMClient):
        last_user = ""

        def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
            RecordingStub.last_user = user
            return "{}"

    extract_entities("mi correo es juan.perez@mail.com", client=RecordingStub())
    assert "juan.perez@mail.com" not in RecordingStub.last_user
    assert "@mail.com" in RecordingStub.last_user
