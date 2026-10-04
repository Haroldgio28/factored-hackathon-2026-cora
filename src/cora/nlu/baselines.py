"""Three intent-classification baselines the model report compares against (task 3.4, REQ-29).

A learned classifier only earns its place if it beats the cheap alternatives, so Phase 3 ships the
cheap alternatives as first-class, reusable code:

- `MajorityBaseline` (B-maj) predicts the training-set modal intent. One `collections.Counter`
  (ponytail: stdlib, no `sklearn.DummyClassifier`).
- `KeywordBaseline` (B-kw) is an ordered list of `(regex, Intent)` rules per language; first match
  wins, no match -> `OTHER`. It is deterministic and is ALSO the Phase-5 fallback when the learned
  classifier is unavailable (design section 10), so it is reused, not throwaway.
- `ZeroShotBaseline` (B-zs) asks the LLM, via the shared `LLMClient`, to pick one of the 16 labels;
  the reply is untrusted DATA, validated against `Intent`, and an invalid/empty reply -> `OTHER`
  (fail closed, never crash the eval). If Bedrock is unconfigured it degrades to the stub.

All three expose the same `predict(utterances) -> list[Intent]` shape as the learned classifier so
the report tabulates them uniformly.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from cora.agent.llm import LLMClient, LLMUnavailable, get_llm_client
from cora.nlu.labels import Intent

logger = logging.getLogger("cora.nlu.baselines")

__all__ = ["KeywordBaseline", "MajorityBaseline", "ZeroShotBaseline"]

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"


class MajorityBaseline:
    """Predict the single most frequent training intent for every input (B-maj)."""

    def __init__(self, train_intents: Iterable[str]) -> None:
        counts = Counter(train_intents)
        if not counts:
            raise ValueError("MajorityBaseline needs at least one training label")
        # most_common ties break on insertion order; sort the modal set so the pick is deterministic.
        top = max(counts.values())
        self.majority = Intent(sorted(label for label, n in counts.items() if n == top)[0])

    def predict(self, utterances: Sequence[str]) -> list[Intent]:
        return [self.majority] * len(utterances)


# Ordered keyword rules per language: FIRST match wins, so the more specific/urgent intents
# (fraud, dispute, money movement) are listed before the generic read intents they could overlap
# with (e.g. "no reconozco esta compra" is a dispute, not a transaction lookup). No match -> OTHER.
# Spanish rules cover MX/CO/AR variants; Portuguese rules cover the BR set (language steering).
_KEYWORD_RULES: dict[str, list[tuple[str, Intent]]] = {
    "es": [
        (
            r"\bfraude\b|clonaron|clonar|hackearon|sin (mi )?(permiso|autorizaci[oó]n)|usaron mi tarjeta",
            Intent.E3,
        ),
        (
            r"no reconozco|no (lo )?hice|que no compr|cargo que yo no|disputar|desconocer|reclamar un (cargo|movimiento|consumo)",
            Intent.E1,
        ),
        (
            r"\bqueja\b|quejar|reclamo formal|reclamaci[oó]n|mal servicio|p[eé]simo servicio|muy (molesto|enojado)",
            Intent.E2,
        ),
        (
            r"hablar con (una persona|alguien|un (agente|asesor))|agente humano|asesor humano|me atienda alguien|transferir con un asesor|con un bot",
            Intent.E4,
        ),
        (
            r"transfer[ií]|transfiera|env[ií]o de dinero|mandar?\b|manda\b|mand[aeá]\b|pagar? (mi|la) (tarjeta|factura|recibo)",
            Intent.X2,
        ),
        (
            r"sub(an|ir)? el (l[ií]mite|cupo)|aument(en|ar) (la|mi) l[ií]nea|pr[eé]stamo|cr[eé]dito (personal|hipotecario|automotriz|de libre)|elegible|calific",
            Intent.X1,
        ),
        (r"congel|bloque(a|á|e|ar)|desactivar|suspend|inactivar", Intent.A1),
        (r"desbloque|descongel|reactiv|activ(a|á|e|ar).*(de nuevo|otra vez)|habilit", Intent.A2),
        (r"d[oó]lar|euro|tipo de cambio|tasa de cambio|cotizaci[oó]n|convertir|cu[aá]ntos pesos", Intent.I5),
        (
            r"saldo|cu[aá]nto (dinero )?tengo|cu[aá]nta plata|disponible en (la|mi)|caja de ahorro|cu[aá]nto me queda",
            Intent.I1,
        ),
        (r"movimiento|transacci|[uú]ltima compra|mis compras|cu[aá]nto gast|historial", Intent.I2),
        (
            r"pendiente|rechaz|declin|duplicad|dos veces|ya se (proces|acredit)|en proceso|qu[eé] pas[oó] con (el|la)",
            Intent.I3,
        ),
        (
            r"estado de (mi|la) tarjeta|tarjeta est[aá] (activa|bloqueada)|l[ií]mite de (mi|la) tarjeta|d[ií]as de (mora|atraso)|vence mi tarjeta",
            Intent.I4,
        ),
        (
            r"qu[eé] productos|mis cuentas y tarjetas|lista(do)? de (mis|todos)|tarjetas (tengo )?asociad",
            Intent.I6,
        ),
        (
            r"sucursal|oficina m[aá]s cercana|horario de atenci[oó]n|abrir una cuenta|app para el|cambiar? (mi )?(contrase[nñ]a|clave)",
            Intent.X3,
        ),
    ],
    "pt": [
        (
            r"\bfraude\b|clonaram|clonar|hackearam|sem (minha )?(permiss[aã]o|autoriza[cç][aã]o)|usaram meu cart[aã]o",
            Intent.E3,
        ),
        (
            r"n[aã]o reconhe[cç]o|n[aã]o fiz|que n[aã]o comprei|cobran[cç]a que eu n[aã]o|contestar|reclamar (de )?(uma|a) (cobran[cç]a|compra|movimenta)",
            Intent.E1,
        ),
        (
            r"reclama[cç][aã]o|reclamar|mau servi[cç]o|p[eé]ssimo (servi[cç]o|atendimento)|muito (irritado|chateado|bravo)",
            Intent.E2,
        ),
        (
            r"falar com (uma pessoa|algu[eé]m|um atendente)|atendente humano|pessoa de verdade|me atenda algu[eé]m|com um rob[oô]",
            Intent.E4,
        ),
        (
            r"transfer[ií]|transfira|envio de dinheiro|mand(ar|e|a)\b|pagar? (meu|o) (cart[aã]o|conta|boleto)",
            Intent.X2,
        ),
        (
            r"aument(ar|em)? o (limite|cr[eé]dito)|empr[eé]stimo|cr[eé]dito (pessoal|imobili[aá]rio)|eleg[ií]vel|me qualific",
            Intent.X1,
        ),
        (r"congel|bloque(ar|ie)|desativar|suspend|inativar", Intent.A1),
        (r"desbloque|descongel|reativ|ativ(ar|e).*(de novo|novamente)|habilit", Intent.A2),
        (r"d[oó]lar|euro|taxa de c[aâ]mbio|cota[cç][aã]o|converter|quantos reais", Intent.I5),
        (r"saldo|quanto (dinheiro )?tenho|dispon[ií]vel na (minha )?conta|quanto me resta", Intent.I1),
        (r"movimenta|transa[cç]|[uú]ltima compra|minhas compras|quanto gastei|hist[oó]rico", Intent.I2),
        (
            r"pendente|recus|negad|duplicad|duas vezes|j[aá] (foi )?process|em processamento|o que (aconteceu|houve) com",
            Intent.I3,
        ),
        (
            r"estado do (meu )?cart[aã]o|cart[aã]o est[aá] (ativo|bloqueado)|limite do (meu )?cart[aã]o|dias de atraso|vence o meu cart[aã]o",
            Intent.I4,
        ),
        (
            r"quais produtos|minhas contas e cart[oõ]es|lista de (meus|todos)|cart[oõ]es (que tenho )?associad",
            Intent.I6,
        ),
        (
            r"ag[eê]ncia|mais pr[oó]xima|hor[aá]rio de atendimento|abrir uma conta|app (para|do)|trocar (minha )?senha",
            Intent.X3,
        ),
    ],
}


class KeywordBaseline:
    """First-match-wins regex rules per language (B-kw); no match -> OTHER. Also the Phase-5 fallback."""

    def __init__(self) -> None:
        self._rules = {
            lang: [(re.compile(pat, re.IGNORECASE), intent) for pat, intent in rules]
            for lang, rules in _KEYWORD_RULES.items()
        }

    def predict_one(self, utterance: str, language: str) -> Intent:
        for pattern, intent in self._rules.get(language, []):
            if pattern.search(utterance):
                return intent
        return Intent.OTHER

    def predict(self, utterances: Sequence[str], languages: Sequence[str] | None = None) -> list[Intent]:
        """Classify each utterance. `languages` defaults to Spanish for every row if omitted."""
        langs = list(languages) if languages is not None else ["es"] * len(utterances)
        if len(langs) != len(utterances):
            raise ValueError("languages must match utterances in length")
        return [self.predict_one(u, lang) for u, lang in zip(utterances, langs, strict=True)]


def _load_prompt(language: str) -> str:
    path = _PROMPTS_DIR / f"zero_shot_intent.{language}.txt"
    if not path.exists():  # pt and es ship; any other language falls back to the es prompt
        path = _PROMPTS_DIR / "zero_shot_intent.es.txt"
    return path.read_text(encoding="utf-8")


_VALID_LABELS = frozenset(Intent)


def _mask_pii(text: str) -> str:
    """Mask PII before the utterance reaches the LLM (security steering: mask before any LLM call).

    The composed `mask_pii` masker lands in `tools.base` at task 3.7; import it lazily so B-zs picks
    it up automatically once it exists. Until then, fail CLOSED rather than leak: a configured
    Bedrock call on un-masked free text is refused. The gold-set eval runs on the stub (no network),
    so this does not block the 3.4 evidence.

    # ponytail: lazy import of tools.base.mask_pii (arrives 3.7); refuse real calls until then
    #           instead of shipping a weaker inline masker that would duplicate 3.7's one PII home.
    """
    try:
        from cora.tools.base import mask_pii  # type: ignore[attr-defined]
    except ImportError:
        return text  # not yet available; the guard in ZeroShotBaseline handles real-call safety
    return mask_pii(text)


class ZeroShotBaseline:
    """Ask the LLM to pick one of the 16 labels (B-zs); reply is validated, invalid/empty -> OTHER."""

    def __init__(self, client: LLMClient | None = None, *, system: str = "") -> None:
        # Default to the shared factory: stub under CORA_NLU_STUB=1 or when Bedrock is unconfigured.
        self._client = client if client is not None else get_llm_client()
        self._system = system or "Responde solo con el codigo de etiqueta."
        # True once tools.base.mask_pii exists; gates real (non-stub) calls on un-masked text.
        from cora.agent.llm import StubLLMClient

        self._is_stub = isinstance(self._client, StubLLMClient)

    def predict_one(self, utterance: str, language: str = "es") -> Intent:
        masked = _mask_pii(utterance)
        if masked is utterance and not self._is_stub:
            # mask_pii not available yet AND this is a real network client: refuse (fail closed).
            logger.warning("zero-shot refused: PII masker (3.7) absent for a non-stub client")
            return Intent.OTHER
        prompt = _load_prompt(language).format(utterance=masked)
        try:
            reply = self._client.complete(system=self._system, user=prompt, max_tokens=8)
        except LLMUnavailable:
            return Intent.OTHER
        return self._parse(reply)

    def predict(self, utterances: Sequence[str], languages: Sequence[str] | None = None) -> list[Intent]:
        langs = list(languages) if languages is not None else ["es"] * len(utterances)
        if len(langs) != len(utterances):
            raise ValueError("languages must match utterances in length")
        return [self.predict_one(u, lang) for u, lang in zip(utterances, langs, strict=True)]

    @staticmethod
    def _parse(reply: str) -> Intent:
        """Pull a valid label out of the (untrusted) reply; anything else -> OTHER (fail closed)."""
        token = reply.strip().split()[0].strip(".,:;\"'").upper() if reply.strip() else ""
        if token in _VALID_LABELS:
            return Intent(token)
        # Tolerate a reply that embeds the label in a sentence, but still fail closed on no match.
        for label in _VALID_LABELS:
            if re.search(rf"\b{label}\b", reply.upper()):
                return Intent(label)
        return Intent.OTHER
