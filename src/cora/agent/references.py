"""Deterministic cross-turn reference resolution (task 4.1, design section 4, REQ-06).

A customer rarely repeats an id. They say "esa tarjeta", "o ultimo movimento", "la segunda".
Resolving those to a concrete `product_id`/`transaction_id` is a DETERMINISTIC lookup against
`SessionState` - never an LLM call (P1: the model may propose a `product_ref` phrase, but code
decides which concrete id it means). The LLM-proposed phrase is treated strictly as data: it
only selects among ids the session already holds; it can never introduce a new id.

`resolve_reference(state, product_ref)` returns a `Reference`:

- a demonstrative ("esa/esta tarjeta", "esse cartao") -> the last referenced product id;
- "the last / most recent" ("el ultimo", "o ultimo") -> the last entry of the last listed set;
- an ordinal ("la segunda", "a terceira", "the second one") -> that 1-based position of the
  last listed set, bounds-checked;
- no referential phrase -> `kind=none` (the turn carried a concrete id or none at all);
- a referential phrase that matches nothing resolvable in state -> `ambiguous=True`, so the
  policy clarifies (POL-070) rather than guessing (fail closed).

Only the SMALL closed set of referential phrases the dataset uses is matched; anything else is
`none`, leaving a concrete id (if any) untouched. Spanish covers MX/CO/AR; Portuguese is BR.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from cora.agent.state import SessionState

__all__ = ["Reference", "ReferenceKind", "resolve_reference"]


class ReferenceKind(StrEnum):
    """What a resolved reference points at (or that there was nothing to resolve)."""

    NONE = "none"  # no referential phrase; a concrete id (if any) stands as-is
    PRODUCT = "product"
    TRANSACTION = "transaction"


@dataclass(frozen=True)
class Reference:
    """The outcome of resolving a cross-turn reference deterministically from session state."""

    kind: ReferenceKind
    product_id: str | None = None
    transaction_id: str | None = None
    # True when a referential phrase was used but nothing in state resolves it -> clarify.
    ambiguous: bool = False


_NONE = Reference(kind=ReferenceKind.NONE)

# es/pt ordinal words -> 1-based position. Only the few the dataset realistically uses.
_ORDINALS: dict[str, int] = {
    "primer": 1,
    "primero": 1,
    "primera": 1,
    "primeiro": 1,
    "primeira": 1,
    "first": 1,
    "segundo": 2,
    "segunda": 2,
    "second": 2,
    "tercer": 3,
    "tercero": 3,
    "tercera": 3,
    "terceiro": 3,
    "terceira": 3,
    "third": 3,
}

# "the last / most recent" phrasings (es + pt + en). Matched as whole words.
_LAST_RE = re.compile(
    r"\b(ultim[oa]|el ultimo|la ultima|o ultimo|a ultima|mais recente|mas reciente|last|most recent|latest)\b"
)

# Demonstratives pointing at a previously referenced product/transaction ("that card").
_DEMONSTRATIVE_RE = re.compile(r"\b(es[ae]|est[ae]|aquel+a?|ess[ae]|aquel[ea]|that|this)\b")

# Words that disambiguate the referent type when a demonstrative/ordinal is used.
_TXN_RE = re.compile(
    r"\b(movimiento|movimento|transaccion|transacao|transaction|cargo|compra|pago|pagamento)\b"
)
_PRODUCT_RE = re.compile(r"\b(tarjeta|cartao|cuenta|conta|producto|produto|card|account|product)\b")

_ORDINAL_RE = re.compile(r"\b(" + "|".join(_ORDINALS) + r")\b")


def _normalize(text: str) -> str:
    """Lowercase and strip accents so "último"/"ultimo"/"ÚLTIMO" all match one spelling."""
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def _wants_transaction(text: str) -> bool:
    """Prefer the transaction referent only when the turn names a transaction and not a product."""
    return bool(_TXN_RE.search(text)) and not _PRODUCT_RE.search(text)


def resolve_reference(state: SessionState, product_ref: str | None) -> Reference:
    """Resolve a referential phrase to a concrete id using ONLY `state`; never an LLM.

    `product_ref` is the candidate phrase the entity extractor proposed (untrusted data). With
    no phrase, or a phrase carrying no referential marker, the result is `NONE` and the caller
    keeps whatever concrete id the turn already had. A referential marker that resolves to
    nothing in state is `ambiguous` so the policy clarifies instead of guessing.
    """
    if not product_ref or not product_ref.strip():
        return _NONE
    text = _normalize(product_ref)
    want_txn = _wants_transaction(text)
    ids = state.last_listed_transaction_ids if want_txn else state.last_listed_product_ids
    kind = ReferenceKind.TRANSACTION if want_txn else ReferenceKind.PRODUCT

    ordinal_match = _ORDINAL_RE.search(text)
    is_last = bool(_LAST_RE.search(text))
    is_demonstrative = bool(_DEMONSTRATIVE_RE.search(text))

    if not (ordinal_match or is_last or is_demonstrative):
        # No referential marker at all: not a cross-turn reference, leave the concrete id alone.
        return _NONE

    if ordinal_match:
        position = _ORDINALS[ordinal_match.group(1)]
        if 1 <= position <= len(ids):
            return _ref(kind, ids[position - 1])
        return Reference(kind=kind, ambiguous=True)

    if is_last:
        if ids:
            return _ref(kind, ids[-1])
        return Reference(kind=kind, ambiguous=True)

    # Demonstrative "that/this ...": the single last-referenced id of the inferred type.
    referenced = state.referenced_transaction_id if want_txn else state.referenced_product_id
    if referenced is not None:
        return _ref(kind, referenced)
    # A demonstrative pointing at a type we have exactly one listed id for is unambiguous too.
    if len(ids) == 1:
        return _ref(kind, ids[0])
    return Reference(kind=kind, ambiguous=True)


def _ref(kind: ReferenceKind, resolved_id: str) -> Reference:
    if kind is ReferenceKind.TRANSACTION:
        return Reference(kind=kind, transaction_id=resolved_id)
    return Reference(kind=kind, product_id=resolved_id)


if __name__ == "__main__":  # tiny self-check (no network, no LLM)
    st = SessionState(jti="j")
    st.last_listed_product_ids = ["PRD-1", "PRD-2", "PRD-3"]
    st.referenced_product_id = "PRD-2"
    assert resolve_reference(st, "la segunda tarjeta").product_id == "PRD-2"
    assert resolve_reference(st, "o ultimo cartao").product_id == "PRD-3"
    assert resolve_reference(st, "esa tarjeta").product_id == "PRD-2"
    assert resolve_reference(st, None).kind is ReferenceKind.NONE
    assert resolve_reference(SessionState(jti="j"), "la segunda tarjeta").ambiguous  # nothing listed
    st.last_listed_transaction_ids = ["TRX-9"]
    assert resolve_reference(st, "ese movimiento").transaction_id == "TRX-9"
    print("references self-check OK")
