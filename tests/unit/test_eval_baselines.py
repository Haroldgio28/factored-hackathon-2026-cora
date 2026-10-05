"""Tests for task 6.2: the naive single-prompt baseline B1 (REQ-41).

B1 is exercised over a TINY self-built Parquet landing + a verified session (the `test_card_overlay`
/ `test_scenario_builder` pattern), with a spy `LLMClient` so the test can inspect exactly what text
reached the model - NO network, NO Bedrock. Asserted: B1 produces a `B1Result` from one LLM call;
PII in BOTH the pasted account data and the utterance is masked before the call; the prompt hash is
recorded; an LLM outage is recorded (not raised); and both es and pt prompts exist (a missing one
fails loud, mirroring the bilingual steering).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.eval.baselines import B1NaiveLLM, B1Result
from cora.identity import MockIdentityService, Session
from cora.tools.layer import ToolLayer


class _Spy:
    """Records every user prompt it is asked to complete; returns a fixed canned answer."""

    def __init__(self, reply: str = "Su tarjeta está activa.") -> None:
        self.reply = reply
        self.seen_user: list[str] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.seen_user.append(user)
        return self.reply


class _Down:
    """An LLM client that always fails closed (unavailable)."""

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        from cora.agent.llm import LLMUnavailable

        raise LLMUnavailable("down")


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.astype("string").to_parquet(path, index=False)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("b1_raw_parquet")
    customers = pd.DataFrame(
        [{"customer_id": "CLI-0000000000001", "country": "MX", "customer_segment": "Mass"}]
    )
    products = pd.DataFrame(
        [
            {
                "product_id": "PRD-00000000001",
                "customer_id": "CLI-0000000000001",
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            }
        ]
    )
    _write(root / "customers.parquet", customers)
    _write(root / "products.parquet", products)
    return root


def _session(customer_id: str = "CLI-0000000000001") -> Session:
    service = MockIdentityService(signing_key="b1-key", session_ttl=timedelta(minutes=15))
    challenge = service.begin_authentication(customer_id)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


@pytest.fixture
def tools(landing: Path) -> ToolLayer:
    return ToolLayer(_session(), LocalSource(root=landing))


def test_b1_produces_a_result_from_one_call(tools: ToolLayer) -> None:
    spy = _Spy()
    result = B1NaiveLLM(client=spy).answer("¿Mi tarjeta está activa?", "es", tools=tools)
    assert isinstance(result, B1Result)
    assert result.text == "Su tarjeta está activa."
    assert result.available is True
    assert len(spy.seen_user) == 1  # exactly one LLM call, no tool-call loop
    # The pasted data is the customer's own product (masked number), proving data was pasted in.
    assert "****3827" in result.pasted_data
    assert result.account_source_refs == ["PRD-00000000001"]


def test_b1_masks_pii_before_the_llm_call(tools: ToolLayer) -> None:
    spy = _Spy()
    # An utterance carrying raw PII must be masked before it reaches the model.
    B1NaiveLLM(client=spy).answer(
        "Mi correo es juan.perez@mail.com y mi tarjeta es 4717188030863827",
        "es",
        tools=tools,
    )
    sent = spy.seen_user[0]
    assert "juan.perez@mail.com" not in sent, "raw email reached the LLM"
    assert "4717188030863827" not in sent, "raw card number reached the LLM"


def test_b1_records_the_prompt_hash(tools: ToolLayer) -> None:
    spy = _Spy()
    result = B1NaiveLLM(client=spy).answer("hola", "es", tools=tools)
    assert len(result.prompt_hash) == 64  # a SHA-256 hex digest
    # The hash is of the real versioned prompt file bytes.
    import hashlib

    expected = hashlib.sha256((Path("src/cora/agent/prompts/naive_baseline.es.txt")).read_bytes()).hexdigest()
    assert result.prompt_hash == expected


def test_b1_both_languages_have_a_prompt(tools: ToolLayer) -> None:
    spy = _Spy(reply="Seu cartão está ativo.")
    es = B1NaiveLLM(client=spy).answer("¿saldo?", "es", tools=tools)
    pt = B1NaiveLLM(client=spy).answer("qual meu saldo?", "pt", tools=tools)
    assert es.prompt_hash != pt.prompt_hash  # distinct es/pt prompt files
    assert es.language == "es" and pt.language == "pt"


def test_b1_unknown_language_fails_loud(tools: ToolLayer) -> None:
    with pytest.raises(FileNotFoundError):
        B1NaiveLLM(client=_Spy()).answer("hello", "en", tools=tools)


def test_b1_records_outage_without_raising(tools: ToolLayer) -> None:
    result = B1NaiveLLM(client=_Down()).answer("¿saldo?", "es", tools=tools)
    assert result.available is False
    assert result.text  # an honest "cannot answer" message, not an empty string
