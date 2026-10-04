"""Response generation: deterministic template first, optional LLM polish, fail-closed (task 4.2,
design sections 4/5, REQ-18, REQ-36, REQ-40).

`generate` is the single entry point the orchestrator's Answer/Clarify/Confirm/... nodes call. It:

1. Selects the fixed `(outcome, lang)` template (`templates.render_template`) and fills it from
   TOOL-RESULT FACTS ONLY (`fields`). The deterministic layer owns every figure; the LLM never
   originates one (P1, REQ-08). A missing required field is a render failure -> fail closed.
2. Optionally asks the LLM to REPHRASE the already-filled, grounded draft for a warmer tone. The
   draft is `mask_pii`-masked BEFORE the `complete(...)` call (security steering, REQ-36) - no raw
   PII ever reaches Bedrock. The prompt tells the model to invent no values and to treat the draft
   as data.
3. GROUNDS the polished text against the current turn's tool results (`grounding.check_grounding`,
   task 4.3, REQ-08): if the LLM introduced any figure/date/status not present in those tool
   results, the polished text is DISCARDED and the deterministic template is returned instead.
   Deterministic code, not the model, makes this block decision.
4. Fails closed to the deterministic template on ANYTHING unexpected: LLM unavailable, an empty
   reply, or a render error (REQ-40 "LLM down -> templates; honest fallback"). The template is
   grounded by construction, so the fallback is always safe.

The versioned prompt file's SHA-256 is recorded on the result (`GeneratedResponse.prompt_hash`)
for the trace (design section 4 "prompts' hash recorded in traces", REQ-08), reusing `hashlib`
over the file bytes - no new dependency. When no polish happens (template-only / fallback) the
hash is `None`, so a trace can tell a pure-template turn from a polished one.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from cora.agent.grounding import check_grounding
from cora.agent.llm import LLMClient, LLMUnavailable, get_llm_client
from cora.agent.templates import SUPPORTED_LANGUAGES, Outcome, render_template
from cora.tools.base import Result, mask_pii

logger = logging.getLogger("cora.agent.generator")

__all__ = ["GeneratedResponse", "generate"]

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "response_polish.txt"

# Deterministic es<->pt flip guard for the LLM polish. This is NOT a language detector (that is
# `nlu.language.detect`, which false-positives on short service phrases es/pt share) - it is a
# HIGH-PRECISION safety net: the session language is already known with certainty (the user-chosen
# toggle, 5.4), so the guard only has to catch a CONFIDENT flip to the other language and stay
# silent on everything ambiguous. The English polish prompt is the PRIMARY fix (validated against
# real Bedrock); this guard is the deterministic backstop ("the LLM proposes, code decides").
#
# Precision comes from keying on signals EXCLUSIVE to one language. The two sides are deliberately
# ASYMMETRIC because the languages are not symmetric on paper:
#
# - Portuguese has ORTHOGRAPHY Spanish never uses: `ã`, `õ`, `ç`, the digraphs `nh`/`lh`, and a
#   small set of accented function words (`não`, `você`, `cartão`, `então`, standalone `é`). Any
#   of these in a reply that should be Spanish is near-certain proof of an es->pt flip. Crucially
#   it does NOT list words es and pt SHARE in spelling (`está`, `consigo`, `sim`, `voces`): those
#   are ordinary Spanish and blocking them discarded valid polish (the bug this redesign fixes).
#
# - Spanish has only three exclusive characters (`ñ`, `¿`, `¡`), and a translated balance/escalation
#   answer uses none of them - so the pt->es direction (the flip actually observed) CANNOT be caught
#   orthographically. It is caught LEXICALLY instead, with accent-independent function words that are
#   Spanish and never Portuguese: `voy`, `una`, `persona`, `equipo`, `disponibles`, `su`, `dinero`,
#   `es de`, `usted(es)`. Portuguese writes these differently (`vou`, `uma`, `pessoa`, `equipe`,
#   `disponíveis`, `seu/sua`, `dinheiro`, `é de`), so a faithful pt polish never trips them.
#
# This is honest about its ceiling: the pt->es list is a curated set of common forms, not a proof
# of exhaustiveness. A flip in wording that avoids all of them survives the guard - and is then the
# English prompt's job. A missed flip never leaks a wrong figure either way: the grounding gate
# runs regardless, so the worst case is drier wording, never an invented number.
#
# ponytail: curated marker lists, no model, no new dependency. If a real flip slips through, add the
#           missing exclusive form here; the grounding gate still guarantees figures are correct.
_PT_ONLY = re.compile(
    r"(?:[ãõ]|ç|nh|lh|\bnão\b|\bvocê\b|\bvoc[êe]s\b|\bcartão\b|\bé\b|\bobrigad[oa]\b"
    r"|\bendere[çc]o\b|\bentão\b)",
    re.IGNORECASE,
)
_ES_ONLY = re.compile(
    r"(?:ñ|¿|¡|\bvoy\b|\buna\b|\bpersona\b|\bequipo\b|\bdisponibles\b|\bsu\b|\bdinero\b"
    r"|\bes de\b|\busted(?:es)?\b)",
    re.IGNORECASE,
)
# Asymmetric by design (see the comment above): es is caught by pt-exclusive signals, pt by
# es-exclusive signals. `_flipped_language(text, draft_lang)` looks for markers of the OTHER one.
_LANG_MARKERS = {"es": _ES_ONLY, "pt": _PT_ONLY}


def _flipped_language(text: str, draft_lang: str) -> str | None:
    """The matched opposite-language marker when `text` flipped away from `draft_lang`, else None.

    High precision by design (only es/pt, only exclusive signals). Returns None for any language
    outside es/pt and for text with no strong marker of the other language, so a faithful
    same-language rephrase is never blocked. The returned marker is logged for the trace.
    """
    other = {"es": "pt", "pt": "es"}.get(draft_lang)
    if other is None:
        return None
    match = _LANG_MARKERS[other].search(text)
    return match.group(0) if match else None


def _normalize_lang(lang: str) -> str:
    """Collapse `lang` to a `SUPPORTED_LANGUAGES` value (es/pt), falling back to es.

    Normalized ONCE in `generate`, before both the template lookup and the prompt interpolation,
    so the template, the MANDATORY-LANGUAGE prompt instruction and the flip guard all agree on the
    same language. `render_template` already falls back to es for unsupported values; this makes
    that fallback explicit so an unsupported `lang` never reaches the prompt unvalidated (the guard
    returns None outside es/pt, and the English prompt would otherwise obey a mismatched instruction).
    """
    normalized = lang.strip().lower()
    return normalized if normalized in SUPPORTED_LANGUAGES else "es"


@dataclass(frozen=True)
class GeneratedResponse:
    """The rendered customer-facing text plus the trace signals for the turn.

    `text` is what the customer sees. `outcome`/`lang` identify the template used. `polished` is
    True only when the LLM successfully rephrased the draft AND the rephrasing passed grounding;
    `prompt_hash` is the SHA-256 of the polish prompt file when polish ran, else `None` (so a trace
    distinguishes template-only turns). `grounding_blocked` is True when a polished draft was
    discarded for inventing a figure not in the turn's tool results (REQ-08), so the trace records
    the fall back to the grounded template.
    """

    text: str
    outcome: Outcome
    lang: str
    polished: bool
    prompt_hash: str | None
    grounding_blocked: bool = False
    # True when a polished draft was discarded because the LLM changed the draft's language
    # (e.g. translated a pt draft to es), so the trace records the fall back to the template.
    language_blocked: bool = False


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _prompt_hash() -> str:
    """SHA-256 of the versioned polish prompt file bytes (recorded in the trace, REQ-08)."""
    return hashlib.sha256(_PROMPT_PATH.read_bytes()).hexdigest()


def generate(
    outcome: Outcome,
    lang: str,
    *,
    fields: dict[str, str] | None = None,
    tool_results: list[Result] | None = None,
    polish: bool = True,
    client: LLMClient | None = None,
) -> GeneratedResponse:
    """Render the customer-facing response for `outcome` in `lang`, optionally LLM-polished.

    `fields` are the template placeholders, filled from TOOL-RESULT FACTS by the caller (never
    the model) - e.g. the grounded `facts` string for an answer or the masked product number for
    a confirmation. `tool_results` are THIS TURN's tool `Result` objects; a polished draft is
    grounded against them (REQ-08) and discarded if it introduces a figure they do not contain.
    `polish=False` forces the deterministic template (used where polish adds no value, e.g.
    re-auth). The LLM, when used, only rephrases the masked, fact-filled draft.

    Fails closed to the deterministic template on a render error (missing field), an unavailable
    LLM, an empty reply, or an ungrounded polish (REQ-40/REQ-08). The template path is grounded by
    construction, so the fallback never invents a figure.
    """
    fields = fields or {}
    tool_results = tool_results or []
    # Normalize the language ONCE (REQ-18): the template lookup, the MANDATORY-LANGUAGE prompt
    # instruction and the flip guard must all agree, so an unsupported value never reaches the
    # prompt as an emphatic "answer in <X>" instruction the English prompt would obey.
    lang = _normalize_lang(lang)
    try:
        draft = render_template(outcome, lang, **fields)
    except KeyError as exc:  # a required placeholder was not supplied -> cannot render safely
        logger.warning("template render failed for %s/%s: missing field %s", outcome, lang, exc)
        raise

    if not polish:
        return GeneratedResponse(text=draft, outcome=outcome, lang=lang, polished=False, prompt_hash=None)

    llm = client if client is not None else get_llm_client()
    # Mask the (already fact-only) draft BEFORE it reaches the LLM - defence in depth: the figures
    # are grounded, but a product/account value in `facts` must still be masked (REQ-36).
    masked_draft = mask_pii(draft)
    prompt = _load_prompt().format(lang=lang, draft=masked_draft)
    try:
        reply = llm.complete(system="", user=prompt, max_tokens=512)
    except LLMUnavailable:
        logger.info("LLM unavailable; returning deterministic template for %s/%s", outcome, lang)
        return GeneratedResponse(text=draft, outcome=outcome, lang=lang, polished=False, prompt_hash=None)

    polished = reply.strip()
    if not polished:  # empty reply -> honest fallback to the grounded template (REQ-40)
        return GeneratedResponse(text=draft, outcome=outcome, lang=lang, polished=False, prompt_hash=None)

    # Evaluate BOTH safety gates and record BOTH flags, so a reply that is unsafe for more than one
    # reason is counted under each type in the failure-handling metrics (not misattributed to
    # whichever gate happens to run first). Either flag set discards the polish and returns the
    # grounded template. Deterministic code (not the model) makes the block decision.
    #
    # Grounding gate (task 4.3, REQ-08): the LLM may only rephrase; a figure/date/status not in
    # this turn's tool results is ungrounded.
    grounding = check_grounding(polished, tool_results)
    # Language gate (REQ-18, language steering): the LLM must rephrase in the DRAFT's language, not
    # translate it. The prompt says so, but "the LLM proposes, deterministic code decides" - a
    # confident es<->pt flip marker in the polish is caught here as a deterministic backstop.
    flip_marker = _flipped_language(polished, lang)
    language_blocked = flip_marker is not None

    if language_blocked:
        logger.warning(
            "polish changed the language for %s/%s; matched opposite-language marker=%r; "
            "falling back to template",
            outcome,
            lang,
            flip_marker,
        )
    if not grounding.ok:
        logger.warning(
            "grounding blocked polished text for %s/%s; offending=%s; falling back to template",
            outcome,
            lang,
            grounding.offending,
        )

    if language_blocked or not grounding.ok:
        return GeneratedResponse(
            text=draft,
            outcome=outcome,
            lang=lang,
            polished=False,
            prompt_hash=None,
            grounding_blocked=not grounding.ok,
            language_blocked=language_blocked,
        )

    return GeneratedResponse(
        text=polished, outcome=outcome, lang=lang, polished=True, prompt_hash=_prompt_hash()
    )


if __name__ == "__main__":  # self-check: PII masked before the LLM, fallback on unavailable
    from cora.agent.llm import StubLLMClient

    class _Spy(StubLLMClient):
        def __init__(self) -> None:
            super().__init__(default="texto cordial")
            self.seen_user = ""

        def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
            self.seen_user = user
            return super().complete(system=system, user=user, max_tokens=max_tokens)

    spy = _Spy()
    out = generate(Outcome.ANSWER, "es", fields={"facts": "Tu correo es juan@mail.com"}, client=spy)
    assert out.polished and out.prompt_hash is not None
    assert "juan@mail.com" not in spy.seen_user, "PII reached the LLM"

    class _Down(StubLLMClient):
        def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
            raise LLMUnavailable("down")

    fb = generate(Outcome.REFUSE, "pt", client=_Down())
    assert not fb.polished and fb.prompt_hash is None and fb.text.strip()

    # Grounding gate: a polish that invents a figure not in the (empty) tool results is blocked
    # and the grounded template is returned instead (REQ-08).
    blocked = generate(Outcome.CLARIFY, "es", client=StubLLMClient(default="Tu saldo es 9999 USD"))
    assert not blocked.polished and blocked.grounding_blocked and "9999" not in blocked.text

    # Language gate (REQ-18): a pt draft the LLM "polishes" into es is discarded for the template.
    flipped = generate(
        Outcome.REFUSE, "pt", client=StubLLMClient(default="Voy a transferir tu caso a una persona.")
    )
    assert not flipped.polished and flipped.language_blocked and flipped.text.strip()
    # A faithful same-language rephrase (no opposite-language marker) is kept.
    kept = generate(Outcome.REFUSE, "es", client=StubLLMClient(default="No puedo ayudarte con eso, gracias."))
    assert kept.polished and not kept.language_blocked

    # Guard precision/exclusivity against the EXACT phrasings from the review (issues #1, #2):
    # ordinary Spanish that must NOT block (es draft, checked for pt markers) ...
    for ok_es in (
        "Tu tarjeta está activa",
        "esa información no está disponible",
        "tarjeta SIM",
        "lo consigo por este canal",
    ):
        assert _flipped_language(ok_es, "es") is None, f"false positive on {ok_es!r}"
    # ... and the pt->es translations the demo actually produced that MUST block (pt draft).
    for bad_pt in (
        "Su saldo actual es de 1.250,75 COP y tiene 1.100,00 COP disponibles para usar",
        "Voy a transferir tu caso a una persona del equipo",
    ):
        assert _flipped_language(bad_pt, "pt") is not None, f"missed flip on {bad_pt!r}"
    # Faithful Portuguese polish of the same shapes must NOT block (pt draft, checked for es markers).
    assert _flipped_language("Vou encaminhar o seu caso a uma pessoa da equipe", "pt") is None
    # Non-es/pt language is out of scope: the guard stays silent.
    assert _flipped_language("I cannot help with that", "en") is None
    print("generator stub self-check OK")
