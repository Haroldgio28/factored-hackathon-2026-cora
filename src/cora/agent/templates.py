"""Deterministic customer-facing response templates, one per decision outcome, in es + pt
(task 4.2, design sections 4/5, REQ-18, REQ-36, REQ-40).

The template is the PRIMARY renderer. Every outcome the state machine can land on
(`graph.TurnResult`) has a fixed, grounded template here; the LLM (`generator.py`) only
rephrases the already-filled template text and never originates a figure. If the LLM is
unavailable the template is returned verbatim (REQ-40 fallback), so the system always has a
safe, grounded thing to say.

Keying: templates are indexed by `(Outcome, lang)`. `Outcome` maps one-to-one onto the policy
`Decision` enum (every edge the orchestrator dispatches to) plus `TOOL_UNAVAILABLE` for the
fail-closed "a tool did not answer" path (design section 4, P4). `from_decision` turns a policy
`Decision` into the matching `Outcome`, so the generator never hand-maps the two enums.

Language policy (steering): every outcome exists in BOTH `es` and `pt`; `missing_translations()`
reports any gap and a test fails on it. Spanish copy is a neutral LATAM register that reads
naturally for MX/CO/AR (no country-specific slang, no "vos"/"ustedes" split that would exclude a
variant). Placeholders are filled ONLY from tool-result facts by the generator; the templates
here contain no figures of their own.
"""

from __future__ import annotations

from enum import StrEnum

from cora.policy import Decision
from cora.tools.confirmations import CardAction

__all__ = [
    "Outcome",
    "SUPPORTED_LANGUAGES",
    "action_verb",
    "missing_translations",
    "render_template",
]

# The two customer-facing languages the product ships (language steering). A missing translation
# for either, for any outcome, must fail a test.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("es", "pt")


class Outcome(StrEnum):
    """Every customer-facing outcome the orchestrator can render.

    One value per policy `Decision` (so the dispatch table and the template table stay in
    lockstep) plus `TOOL_UNAVAILABLE` for the fail-closed path when a tool returns non-OK and
    there is nothing grounded to answer with.
    """

    ANSWER = "answer"
    CONFIRM = "confirm"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"
    ABSTAIN_ROUTE = "abstain_route"
    REFUSE = "refuse"
    REAUTH = "reauth"
    TOOL_UNAVAILABLE = "tool_unavailable"
    # The verified outcome of a confirmed card action (task 4.4): reported ONLY after the tool's
    # read-back confirms the post-condition (REQ-09), and the cancelled path when the customer did
    # not give an explicit affirmative (REQ-14). Both are grounded, figure-free copy.
    ACTION_DONE = "action_done"
    ACTION_CANCELLED = "action_cancelled"
    # A greeting for an identity that is NOT a current customer of the bank: welcome them and tell
    # them they will be routed to a human agent (deterministic non-customer branch in graph.py).
    # It is NOT a policy `Decision`, so `from_decision` leaves it untouched; it is rendered
    # directly by the orchestrator's non-customer handoff path. The copy discloses no account data.
    NONCUSTOMER_HANDOFF = "noncustomer_handoff"

    @classmethod
    def from_decision(cls, decision: Decision) -> Outcome:
        """Map a policy `Decision` to its customer-facing `Outcome` (1:1 by value)."""
        return cls(decision.value)


# The template table. `{...}` placeholders are filled by the generator from tool-result facts
# ONLY; no template carries a balance, date, rate, limit or status of its own (REQ-08). `answer`
# is a thin grounded carrier for the fact text the generator assembles from the tool `Result`;
# the deterministic layer owns the sentence, the LLM only polishes it.
_TEMPLATES: dict[tuple[Outcome, str], str] = {
    # ANSWER — a grounded read result. `{facts}` is assembled by the generator from the tool
    # Result (never invented); the template only frames it.
    (Outcome.ANSWER, "es"): "{facts}",
    (Outcome.ANSWER, "pt"): "{facts}",
    # CONFIRM — restate the exact action + masked product number and ask for an explicit yes.
    # The product number is already masked (`mask_product_number`) before it reaches here.
    (Outcome.CONFIRM, "es"): (
        "Para confirmar: voy a {action} la tarjeta {product_number_masked}. "
        "¿Deseas que continúe? Respóndeme sí o no."
    ),
    (Outcome.CONFIRM, "pt"): (
        "Para confirmar: vou {action} o cartão {product_number_masked}. "
        "Você deseja que eu continue? Responda sim ou não."
    ),
    # CLARIFY — ask one focused question; never guess (REQ-18 low-confidence/mixed language).
    (Outcome.CLARIFY, "es"): (
        "Quiero ayudarte bien, pero no estoy seguro de haber entendido. "
        "¿Puedes decirme con otras palabras qué necesitas?"
    ),
    (Outcome.CLARIFY, "pt"): (
        "Quero te ajudar direito, mas não tenho certeza se entendi. "
        "Você pode me dizer com outras palavras o que precisa?"
    ),
    # ESCALATE — hand off to a human; promise nothing about the outcome (REQ-15/REQ-17).
    (Outcome.ESCALATE, "es"): (
        "Voy a transferir tu caso a una persona del equipo para que lo revise. Gracias por tu paciencia."
    ),
    (Outcome.ESCALATE, "pt"): (
        "Vou encaminhar o seu caso a uma pessoa da equipe para que seja analisado. "
        "Obrigado pela sua paciência."
    ),
    # ABSTAIN — disclose nothing, offer a human (fail closed, P4).
    (Outcome.ABSTAIN, "es"): (
        "Lo siento, no puedo ayudarte con eso por este medio. "
        "Si lo deseas, puedo transferirte con una persona del equipo."
    ),
    (Outcome.ABSTAIN, "pt"): (
        "Desculpe, não posso ajudar com isso por este canal. "
        "Se preferir, posso encaminhar você a uma pessoa da equipe."
    ),
    # ABSTAIN_ROUTE — credit eligibility is out of scope (POL-030, REQ-33); route to a human,
    # decide nothing about credit.
    (Outcome.ABSTAIN_ROUTE, "es"): (
        "No puedo evaluar temas de crédito o aumentos de cupo por este canal. "
        "Voy a transferirte con una persona del equipo que pueda revisarlo."
    ),
    (Outcome.ABSTAIN_ROUTE, "pt"): (
        "Não posso avaliar assuntos de crédito ou aumento de limite por este canal. "
        "Vou encaminhar você a uma pessoa da equipe que possa analisar."
    ),
    # REFUSE — money movement is not available here (POL-020, REQ-34).
    (Outcome.REFUSE, "es"): (
        "No puedo mover dinero ni realizar transferencias o pagos por este canal. "
        "¿Hay algo más en lo que pueda ayudarte?"
    ),
    (Outcome.REFUSE, "pt"): (
        "Não posso movimentar dinheiro nem fazer transferências ou pagamentos por este canal. "
        "Posso ajudar com mais alguma coisa?"
    ),
    # REAUTH — fail closed on expiry/tampering; ask to authenticate again, disclose nothing.
    (Outcome.REAUTH, "es"): (
        "Por tu seguridad necesito verificar tu identidad nuevamente antes de continuar. "
        "¿Podemos volver a validar tu acceso?"
    ),
    (Outcome.REAUTH, "pt"): (
        "Por segurança, preciso verificar a sua identidade novamente antes de continuar. "
        "Podemos validar o seu acesso outra vez?"
    ),
    # ACTION_DONE — a confirmed card action verified by read-back (REQ-09). No figure of its own;
    # the verb/target were restated in the preceding CONFIRM turn.
    (Outcome.ACTION_DONE, "es"): ("Listo, ya apliqué el cambio en tu tarjeta y lo verifiqué. ¿Algo más?"),
    (Outcome.ACTION_DONE, "pt"): (
        "Pronto, apliquei a alteração no seu cartão e verifiquei. Mais alguma coisa?"
    ),
    # ACTION_CANCELLED — no explicit affirmative, so nothing was executed (REQ-14).
    (Outcome.ACTION_CANCELLED, "es"): (
        "De acuerdo, no realicé ningún cambio en tu tarjeta. ¿Hay algo más en lo que pueda ayudarte?"
    ),
    (Outcome.ACTION_CANCELLED, "pt"): (
        "Certo, não fiz nenhuma alteração no seu cartão. Posso ajudar com mais alguma coisa?"
    ),
    # NONCUSTOMER_HANDOFF — the verified identity has NO row in the bank's `customers` records, so
    # it is not a current customer (a deterministic customer-record check over the verified
    # customer_id, never a model decision; distinct from a real customer who owns zero products).
    # Greet, then route to a human; disclose no account data. No placeholders, so it renders with
    # no fields.
    (Outcome.NONCUSTOMER_HANDOFF, "es"): (
        "Hola, soy CORA. Como aún no figuras como cliente del banco, "
        "voy a transferirte con una persona del equipo para ayudarte con tus consultas."
    ),
    (Outcome.NONCUSTOMER_HANDOFF, "pt"): (
        "Olá, sou a CORA. Como você ainda não consta como cliente do banco, "
        "vou encaminhar você a uma pessoa da equipe para ajudar com suas dúvidas."
    ),
    # TOOL_UNAVAILABLE — a tool did not answer; be honest, never guess a value (REQ-40).
    (Outcome.TOOL_UNAVAILABLE, "es"): (
        "En este momento no puedo obtener esa información. "
        "Por favor intenta de nuevo en unos minutos o puedo transferirte con una persona."
    ),
    (Outcome.TOOL_UNAVAILABLE, "pt"): (
        "No momento não consigo obter essa informação. "
        "Por favor tente novamente em alguns minutos ou posso encaminhar você a uma pessoa."
    ),
}


def missing_translations() -> list[tuple[Outcome, str]]:
    """Return every `(Outcome, lang)` with no template, so a test fails on a missing es/pt copy.

    Every outcome must exist in every `SUPPORTED_LANGUAGES` entry (language steering). An empty
    list means the table is complete.
    """
    return [
        (outcome, lang)
        for outcome in Outcome
        for lang in SUPPORTED_LANGUAGES
        if not _TEMPLATES.get((outcome, lang), "").strip()
    ]


def render_template(outcome: Outcome, lang: str, **fields: str) -> str:
    """Fill the `(outcome, lang)` template with tool-fact `fields`; fall back to `es` if needed.

    `lang` outside `SUPPORTED_LANGUAGES` falls back to `es` (the dataset language) rather than
    failing a live turn - the completeness is enforced by tests, not by raising here. Only the
    placeholders a template declares are used; extra fields are ignored. A missing REQUIRED field
    raises `KeyError`, which the generator treats as a render failure (fail closed to a human),
    never as a silent blank.
    """
    key = (outcome, lang if lang in SUPPORTED_LANGUAGES else "es")
    template = _TEMPLATES[key]
    return template.format(**fields)


# The infinitive verb that fills the CONFIRM restatement ("voy a {action} la tarjeta ..."), per
# action and language. Kept here with the rest of the customer-facing copy so a missing es/pt verb
# is as visible as a missing template. es and pt happen to share the spelling for both actions.
_ACTION_VERBS: dict[tuple[CardAction, str], str] = {
    (CardAction.FREEZE, "es"): "congelar",
    (CardAction.FREEZE, "pt"): "congelar",
    (CardAction.UNFREEZE, "es"): "descongelar",
    (CardAction.UNFREEZE, "pt"): "descongelar",
}


def action_verb(action: CardAction, lang: str) -> str:
    """The localized infinitive for a card action, used in the CONFIRM restatement (`es` fallback)."""
    return _ACTION_VERBS[(action, lang if lang in SUPPORTED_LANGUAGES else "es")]


if __name__ == "__main__":  # self-check: table is complete and renders without inventing figures
    assert not missing_translations(), f"missing templates: {missing_translations()}"
    assert set(Outcome) >= {Outcome.from_decision(d) for d in Decision}, "a Decision has no Outcome"
    msg = render_template(
        Outcome.CONFIRM, "es", action=action_verb(CardAction.FREEZE, "es"), product_number_masked="****1234"
    )
    assert "****1234" in msg and "congelar" in msg and "{" not in msg
    assert render_template(Outcome.REFUSE, "pt").strip()  # no placeholders, still renders
    print("templates self-check OK")
