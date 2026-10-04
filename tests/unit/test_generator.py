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

from cora.agent.generator import _flipped_language, generate
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


def test_pt_draft_translated_to_es_is_blocked() -> None:
    """REQ-18 / language steering: a pt draft the LLM translates to es is discarded for the template.

    Deterministic defence in depth - the polish prompt asks the model to keep the language, but the
    generator must not depend on the model obeying. This is the EXACT balance-answer shape the real-
    Bedrock demo caught the model translating pt -> es; the guard's es-exclusive lexical markers
    ("su", "disponibles", "es de") catch it even though it uses no Spanish-only character.
    """
    flipped = generate(
        Outcome.REFUSE,
        "pt",
        client=StubLLMClient(
            default="Su saldo actual es de 1.250,75 COP y tiene 1.100,00 COP disponibles para usar"
        ),
    )
    assert not flipped.polished
    assert flipped.language_blocked
    assert flipped.text.strip()  # the grounded pt template, not a blank


def test_es_draft_translated_to_pt_is_blocked() -> None:
    """pt-exclusive orthography ("não", "você", "ç") in an es turn also falls back to the template."""
    flipped = generate(
        Outcome.REFUSE,
        "es",
        client=StubLLMClient(default="Não posso ajudar você com isso, obrigado."),
    )
    assert not flipped.polished
    assert flipped.language_blocked


# Ordinary Spanish replies that CONTAIN a token the OLD guard wrongly treated as Portuguese
# (`está`, `consigo`, `sim`, `voces`). The redesigned guard keys only on pt-exclusive signals, so
# a Spanish draft must show NO flip for any of these (regression guard for review issues #1/#3).
# Asserted against `_flipped_language` directly, since routing through `generate` with empty
# tool_results would (correctly) grounding-block a status word like "activa" for an unrelated reason.
_ES_REPHRASINGS_THAT_MUST_NOT_FLIP = [
    "Tu tarjeta está activa.",
    "Esa información no está disponible por este canal.",
    "Reporté el robo de tu tarjeta SIM.",
    "Lo consigo por este canal sin problema.",
    "No puedo ayudarte con eso, gracias por tu comprensión.",
]

# pt->es translations in the EXACT shapes the real-Bedrock demo produced: the guard MUST catch
# these (es-exclusive lexical markers), even though they use no Spanish-only character (issue #2).
_PT_DRAFTS_TRANSLATED_TO_ES = [
    "Su saldo actual es de 1.250,75 COP y tiene 1.100,00 COP disponibles para usar",
    "Voy a transferir tu caso a una persona del equipo",
]


def test_ordinary_spanish_is_never_read_as_a_flip() -> None:
    """Review #1/#3: common Spanish containing a shared es/pt token must not read as a pt flip."""
    for reply in _ES_REPHRASINGS_THAT_MUST_NOT_FLIP:
        assert _flipped_language(reply, "es") is None, f"false positive on {reply!r}"


def test_pt_to_es_translations_in_demo_shapes_are_caught() -> None:
    """Review #2/#3: the balance/escalation translations the demo produced must be caught."""
    for reply in _PT_DRAFTS_TRANSLATED_TO_ES:
        assert _flipped_language(reply, "pt") is not None, f"missed flip on {reply!r}"


def test_non_es_pt_language_returns_no_flip() -> None:
    """Review #3: the guard is out of scope outside es/pt and returns None (early return)."""
    assert _flipped_language("I cannot help with that", "en") is None


def test_faithful_portuguese_polish_is_kept() -> None:
    """A pt rephrase of the escalation shape (no es-exclusive marker) is NOT blocked (issue #2)."""
    kept = generate(
        Outcome.ESCALATE,
        "pt",
        client=StubLLMClient(default="Vou encaminhar o seu caso a uma pessoa da equipe."),
    )
    assert kept.polished
    assert not kept.language_blocked


def test_unsupported_language_normalizes_to_es_and_guard_is_silent() -> None:
    """Review #11: an unsupported lang normalizes to es once; the guard returns no flip for it.

    `render_template` already falls back to es, so the es template renders and a faithful es polish
    is kept with `lang == "es"` recorded (not the unsupported value passed in).
    """
    out = generate(
        Outcome.REFUSE,
        "fr",
        client=StubLLMClient(default="No puedo ayudarte con eso, gracias."),
    )
    assert out.lang == "es"
    assert out.polished
    assert not out.language_blocked
