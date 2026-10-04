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
3. Fails closed to the deterministic template on ANYTHING unexpected: LLM unavailable, an empty
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
from dataclasses import dataclass
from pathlib import Path

from cora.agent.llm import LLMClient, LLMUnavailable, get_llm_client
from cora.agent.templates import Outcome, render_template
from cora.tools.base import mask_pii

logger = logging.getLogger("cora.agent.generator")

__all__ = ["GeneratedResponse", "generate"]

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "response_polish.txt"


@dataclass(frozen=True)
class GeneratedResponse:
    """The rendered customer-facing text plus the trace signals for the turn.

    `text` is what the customer sees. `outcome`/`lang` identify the template used. `polished` is
    True only when the LLM successfully rephrased the draft; `prompt_hash` is the SHA-256 of the
    polish prompt file when polish ran, else `None` (so a trace distinguishes template-only turns).
    """

    text: str
    outcome: Outcome
    lang: str
    polished: bool
    prompt_hash: str | None


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
    polish: bool = True,
    client: LLMClient | None = None,
) -> GeneratedResponse:
    """Render the customer-facing response for `outcome` in `lang`, optionally LLM-polished.

    `fields` are the template placeholders, filled from TOOL-RESULT FACTS by the caller (never
    the model) - e.g. the grounded `facts` string for an answer or the masked product number for
    a confirmation. `polish=False` forces the deterministic template (used where polish adds no
    value, e.g. re-auth). The LLM, when used, only rephrases the masked, fact-filled draft.

    Fails closed to the deterministic template on a render error (missing field), an unavailable
    LLM, or an empty reply (REQ-40). The template path is grounded by construction, so the
    fallback never invents a figure.
    """
    fields = fields or {}
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
    print("generator stub self-check OK")
