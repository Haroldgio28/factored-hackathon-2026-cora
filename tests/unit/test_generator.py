"""Tests for task 4.2 response generation (REQ-18, REQ-36, REQ-40, REQ-08).

The generator renders a grounded template first and only lets the LLM REPHRASE it. These tests
pin the security- and resilience-critical guarantees:

- PII is masked BEFORE the LLM is called (a spy client records its `user` arg has no raw PII).
- `LLMUnavailable` and an empty reply both fall back to the deterministic template (REQ-40).
- The polish prompt's SHA-256 is recorded on a polished result and `None` on a template-only one.
- The LLM never originates a figure: with polish off, the output is the fact-filled template, and
  the template contributes no digits the caller did not supply.
"""

from __future__ import annotations

from cora.agent.generator import generate
from cora.agent.llm import LLMUnavailable, StubLLMClient
from cora.agent.templates import Outcome


class _SpyLLM(StubLLMClient):
    """Records the exact `user` prompt the generator sends, so a test can assert it carries no PII."""

    def __init__(self, reply: str = "texto reformulado") -> None:
        super().__init__(default=reply)
        self.seen_user: str | None = None

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.seen_user = user
        return super().complete(system=system, user=user, max_tokens=max_tokens)


class _DownLLM(StubLLMClient):
    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        raise LLMUnavailable("down")


def test_pii_is_masked_before_the_llm_call() -> None:
    """REQ-36: a raw email/phone in the grounded facts must not reach the LLM prompt."""
    spy = _SpyLLM()
    generate(
        Outcome.ANSWER,
        "es",
        fields={"facts": "Tu correo registrado es juan.perez@mail.com y tu telefono +52 55 1234 5678"},
        client=spy,
    )
    assert spy.seen_user is not None
    assert "juan.perez@mail.com" not in spy.seen_user
    assert "5678" in spy.seen_user  # masked tail is fine; the raw local-part/number are not
    assert "1234 5678" not in spy.seen_user


def test_llm_unavailable_falls_back_to_template() -> None:
    """REQ-40: LLM down -> the deterministic, grounded template verbatim, no polish, no hash."""
    out = generate(Outcome.REFUSE, "pt", client=_DownLLM())
    assert not out.polished
    assert out.prompt_hash is None
    assert out.text.strip()


def test_empty_reply_falls_back_to_template() -> None:
    """An empty LLM reply is treated as a failure: return the grounded template (never a blank)."""
    out = generate(Outcome.CLARIFY, "es", client=StubLLMClient(default="   "))
    assert not out.polished
    assert out.prompt_hash is None
    assert out.text.strip()


def test_prompt_hash_recorded_only_when_polished() -> None:
    """REQ-08/design section 4: the polish prompt hash is on polished turns, absent otherwise."""
    polished = generate(Outcome.CLARIFY, "es", client=_SpyLLM("mensaje mas calido"))
    assert polished.polished
    assert polished.prompt_hash is not None and len(polished.prompt_hash) == 64

    template_only = generate(Outcome.REAUTH, "es", polish=False)
    assert not template_only.polished
    assert template_only.prompt_hash is None


def test_no_polish_returns_fact_filled_template() -> None:
    """With polish off the output is exactly the template filled from tool facts (no LLM figure)."""
    out = generate(Outcome.ANSWER, "pt", fields={"facts": "Seu saldo e 100 BRL"}, polish=False)
    assert out.text == "Seu saldo e 100 BRL"
    assert not out.polished


def test_language_is_passed_through_to_the_polish_prompt() -> None:
    """REQ-18: the session language reaches the polish prompt so the reply stays in that language."""
    spy = _SpyLLM()
    generate(Outcome.REFUSE, "pt", client=spy)
    assert spy.seen_user is not None and "pt" in spy.seen_user
