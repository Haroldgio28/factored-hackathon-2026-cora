"""Tests for task 4.1 cross-turn reference resolution (REQ-06).

Reference resolution is DETERMINISTIC and reads only `SessionState` (never an LLM): a
demonstrative, "the last", and an ordinal each resolve to a concrete id from the last listed set,
and a referential phrase that resolves to nothing is `ambiguous` (so the policy clarifies rather
than guessing). A turn with no referential phrase leaves any concrete id untouched (`NONE`).
"""

from __future__ import annotations

from cora.agent.references import ReferenceKind, resolve_reference
from cora.agent.state import SessionState


def _state() -> SessionState:
    st = SessionState(jti="j")
    st.last_listed_product_ids = ["PRD-1", "PRD-2", "PRD-3"]
    st.referenced_product_id = "PRD-2"
    st.last_listed_transaction_ids = ["TRX-9", "TRX-8"]
    st.referenced_transaction_id = "TRX-9"
    return st


def test_demonstrative_resolves_to_last_referenced_product() -> None:
    ref = resolve_reference(_state(), "esa tarjeta")
    assert ref.kind is ReferenceKind.PRODUCT
    assert ref.product_id == "PRD-2"
    assert not ref.ambiguous


def test_portuguese_last_resolves_to_most_recent_listed() -> None:
    ref = resolve_reference(_state(), "o ultimo cartao")
    assert ref.product_id == "PRD-3"


def test_ordinal_resolves_by_position() -> None:
    assert resolve_reference(_state(), "la segunda tarjeta").product_id == "PRD-2"
    assert resolve_reference(_state(), "a primeira conta").product_id == "PRD-1"
    assert resolve_reference(_state(), "the third one").product_id == "PRD-3"


def test_transaction_phrase_resolves_against_transaction_ids() -> None:
    ref = resolve_reference(_state(), "ese movimiento")
    assert ref.kind is ReferenceKind.TRANSACTION
    assert ref.transaction_id == "TRX-9"
    assert resolve_reference(_state(), "o ultimo movimento").transaction_id == "TRX-8"


def test_out_of_range_ordinal_is_ambiguous() -> None:
    st = SessionState(jti="j")
    st.last_listed_product_ids = ["PRD-1"]  # only one listed; "la tercera" is out of range
    ref = resolve_reference(st, "la tercera tarjeta")
    assert ref.ambiguous
    assert ref.product_id is None


def test_referential_phrase_with_no_state_is_ambiguous() -> None:
    empty = SessionState(jti="j")
    assert resolve_reference(empty, "la segunda tarjeta").ambiguous
    assert resolve_reference(empty, "esa tarjeta").ambiguous


def test_no_phrase_or_non_referential_is_none() -> None:
    assert resolve_reference(_state(), None).kind is ReferenceKind.NONE
    assert resolve_reference(_state(), "").kind is ReferenceKind.NONE
    # A concrete id phrase carries no demonstrative/ordinal/last marker -> leave it alone.
    assert resolve_reference(_state(), "PRD-1").kind is ReferenceKind.NONE


def test_accents_and_case_are_normalized() -> None:
    assert resolve_reference(_state(), "la ÚLTIMA tarjeta").product_id == "PRD-3"
