"""Tests for task 3.7: the free-text PII maskers added to tools.base (REQ-36, security steering).

Each masker has at least one ES and one PT example so the es/pt coverage the language policy
requires is actually exercised. `mask_pii` composes them; the composed check asserts no raw PII
survives before an utterance would reach an LLM.
"""

from __future__ import annotations

from cora.tools.base import (
    mask_address,
    mask_document_number,
    mask_email,
    mask_phone,
    mask_pii,
)


def test_mask_email_es_and_pt() -> None:
    assert mask_email("escríbeme a juan.perez@mail.com por favor") == "escríbeme a j***@mail.com por favor"
    assert mask_email("meu email é maria.silva@banco.com.br") == "meu email é m***@banco.com.br"


def test_mask_phone_es_and_pt() -> None:
    masked_es = mask_phone("mi celular es +52 55 1234 5678")
    assert "5678" in masked_es and "1234" not in masked_es and "****" in masked_es
    masked_pt = mask_phone("meu telefone é +55 11 98765 4321")
    assert "4321" in masked_pt and "98765" not in masked_pt


def test_mask_document_number_es_and_pt() -> None:
    masked_es = mask_document_number("mi DNI 20123456 está vencido")
    assert "20123456" not in masked_es and "3456" in masked_es
    masked_pt = mask_document_number("meu CPF 123.456.789-00 nao funciona")
    assert "123.456.789" not in masked_pt and "8900" in masked_pt


def test_mask_address_es_and_pt() -> None:
    assert "[address]" in mask_address("vivo en Calle 5 número 12-34, Bogotá")
    assert "[address]" in mask_address("moro na Rua das Flores 123")


def test_mask_pii_composes_and_leaves_no_raw_pii() -> None:
    text = "Soy juan@mail.com, DNI 20123456, tel +52 55 1234 5678, vivo en Calle 5 #12-34"
    masked = mask_pii(text)
    assert "juan@mail.com" not in masked
    assert "20123456" not in masked
    assert "1234 5678" not in masked
    assert "Calle 5" not in masked
    # The domain survives (visibly an email) but the identifying local part does not.
    assert "@mail.com" in masked


def test_mask_pii_leaves_plain_text_untouched() -> None:
    # A benign balance question carries no PII; masking must not mangle it.
    clean = "¿Cuánto dinero tengo en mi cuenta?"
    assert mask_pii(clean) == clean
