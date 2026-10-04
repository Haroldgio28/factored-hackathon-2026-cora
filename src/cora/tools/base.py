"""Shared tool-layer primitives: the uniform `Result`, status codes, PII masking,
SQL-literal safety and access-attempt logging (task 2.2, design section 6, REQ-12/REQ-36).

Every tool returns a `Result` with a `status` drawn from a fixed vocabulary and never leaks
anything on failure (P4 fail-closed). The tool layer NEVER accepts a `customer_id` as an
input field: it is injected from the verified session by the `ToolLayer` (REQ-12). Helpers
here keep that contract enforceable:

- `mask_product_number` masks a PAN/account number to its last four digits before it reaches
  anything display-bound (REQ-36, security steering).
- `mask_email` / `mask_phone` / `mask_document_number` / `mask_address` and the composing
  `mask_pii` strip free-text PII from a whole utterance before any LLM call (task 3.7). This is
  the one home for PII handling, so the NLU layer imports `mask_pii` here rather than owning a
  second masker.
- `sql_str_literal` / `validate_identifier` make the caller-controlled `where` strings handed
  to the `DataSource` injection-safe, since a merchant substring or a product id can originate
  from user text (treated strictly as data).
- `AccessAttempt` + `AccessLog` record every denied cross-customer access (REQ-12).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("cora.tools.authz")

# Identifiers in the dataset look like `CLI-...`, `PRD-...`, `TRX-...`, `LOAN-...` or plain
# digits; this guard rejects anything carrying SQL metacharacters before it is interpolated.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class Status(StrEnum):
    """The uniform outcome vocabulary every tool returns (design section 6)."""

    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    FORBIDDEN = "FORBIDDEN"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class SourceRef(BaseModel):
    """A pointer to the record a value came from, so answers can be traced to source (P5)."""

    model_config = ConfigDict(frozen=True)

    table: str
    ref: str


class Result[DataT: BaseModel](BaseModel):
    """Uniform tool envelope: a status plus optional typed data, source refs and freshness.

    On any non-OK status `data` is `None` (fail closed: disclose nothing). `message` carries a
    short, non-sensitive explanation safe to log and to summarise; it never contains PII.
    """

    status: Status
    data: DataT | None = None
    source_refs: list[SourceRef] = []
    as_of: datetime | None = None
    message: str | None = None


# -- PII masking (REQ-36, security steering) -------------------------------------------


def mask_product_number(product_number: str | None) -> str | None:
    """Mask a card PAN / account / loan number to its last four characters (`****1234`).

    Full card numbers are PII and must never reach anything display-bound. A short or empty
    value is fully masked so nothing identifying leaks.
    """
    if product_number is None:
        return None
    digits = product_number.strip()
    if len(digits) <= 4:
        return "****"
    return "****" + digits[-4:]


# Free-text PII maskers applied to a whole utterance before it reaches an LLM (security steering:
# "Mask PII ... before any LLM call"; task 3.7, design section 7b). Each is a pinned regex covering
# the es and pt shapes the LATAM dataset carries; `mask_pii` composes them and is the only entry
# point callers use. These intentionally over-mask (a false positive hides a non-PII token) rather
# than under-mask: for a free-text channel, leaking real PII to Bedrock is the unacceptable failure.

# An email local-part@domain. The local part is reduced to its first char so the address is not
# reconstructable but the masking is visibly an email ("j***@mail.com").
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")

# Phone numbers in the LATAM shapes: an optional country code (+52 MX / +57 CO / +54 AR / +55 BR)
# then 8-13 more digits possibly split by spaces/dashes/dots. Requires >= 10 total digits so short
# amounts/dates are not swallowed. Masked to the last four digits.
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d .\-]{8,14}\d)(?!\d)")

# Document numbers: an explicit label (DNI/CURP/RFC/CC/CPF/RG/NIT and es/pt words for "document")
# followed by the identifier. Labelled form only, so plain amounts are never masked as documents.
_DOC_RE = re.compile(
    r"\b(DNI|CURP|RFC|CC|CPF|RG|NIT|CUIL|CUIT|documento|c[eé]dula|identidad|identidade)\b"
    r"\s*(?:n[.ºo]*\s*|:\s*)?([A-Za-z0-9][A-Za-z0-9.\- ]{3,20}[A-Za-z0-9])",
    re.IGNORECASE,
)

# Street addresses: an es/pt street-type lead (Calle/Carrera/Cra/Av./Avenida/Rua) and the words
# that follow up to a comma or sentence end. Replaced wholesale with the "[address]" placeholder.
_ADDRESS_RE = re.compile(
    r"\b(?:calle|carrera|cra\.?|avenida|av\.?|avda\.?|rua|r\.|jr\.?|jir[oó]n|pasaje|psje\.?|diagonal|transversal|tv\.?)\s+[^,.;\n]{1,40}",
    re.IGNORECASE,
)


def _mask_doc_match(match: re.Match[str]) -> str:
    label, number = match.group(1), match.group(2)
    digits = re.sub(r"\D", "", number)
    tail = digits[-4:] if len(digits) >= 4 else ""
    return f"{label} ****{tail}"


def mask_email(text: str) -> str:
    """Mask any email address to `firstchar***@domain` (REQ-36)."""
    return _EMAIL_RE.sub(r"\1***\2", text)


def mask_phone(text: str) -> str:
    """Mask any LATAM-shaped phone number to its last four digits (`****5678`) (REQ-36)."""

    def repl(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(1))
        return "****" + digits[-4:] if len(digits) >= 4 else "****"

    return _PHONE_RE.sub(repl, text)


def mask_document_number(text: str) -> str:
    """Mask a labelled national id (DNI/CURP/RFC/CC/CPF/RG/...) to its last four digits (REQ-36)."""
    return _DOC_RE.sub(_mask_doc_match, text)


def mask_address(text: str) -> str:
    """Replace an es/pt street address with the `[address]` placeholder (REQ-36)."""
    return _ADDRESS_RE.sub("[address]", text)


def mask_pii(text: str) -> str:
    """Mask every supported PII kind in free text before it reaches an LLM (security steering).

    The single entry point the NLU layer calls (task 3.7). Order matters: document numbers are
    masked before phones so a labelled id with >= 10 digits is caught as a document, not a phone.
    Over-masking is deliberate - a free-text channel must never leak real PII to Bedrock.
    """
    text = mask_email(text)
    text = mask_document_number(text)
    text = mask_phone(text)
    text = mask_address(text)
    return text


# -- SQL-literal safety (DataSource `where`/`columns` are raw SQL) ----------------------


def validate_identifier(value: str, *, what: str = "identifier") -> str:
    """Return `value` if it is a safe identifier, else raise `ValueError`.

    Session and resource ids are interpolated into `DataSource` `where` clauses; a product id
    can originate from user/model text, so it is validated as data before it touches SQL.
    """
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        raise ValueError(f"unsafe {what}: {value!r}")
    return value


def sql_str_literal(value: str) -> str:
    """Quote a string as a SQL literal, doubling single quotes (standard-SQL escaping).

    Rejects NUL bytes outright. Used for free-text predicates (e.g. a merchant substring)
    that originate from user text and must be treated strictly as data.
    """
    if "\x00" in value:
        raise ValueError("NUL byte in SQL literal")
    return "'" + value.replace("'", "''") + "'"


# -- Access-attempt logging (REQ-12) ---------------------------------------------------


@dataclass(frozen=True)
class AccessAttempt:
    """A record of a denied cross-customer access, logged whenever a tool returns FORBIDDEN."""

    tool: str
    session_customer_id: str
    resource_type: str
    resource_id: str
    at: datetime


@dataclass
class AccessLog:
    """Collects denied access attempts and mirrors them to the `cora.tools.authz` logger.

    The in-memory list is the test seam and the Phase-5 tracing hook; the logger line is the
    operational signal. No PII is recorded - only ids and the owning session's customer id.
    """

    attempts: list[AccessAttempt] = field(default_factory=list)

    def record(self, attempt: AccessAttempt) -> None:
        self.attempts.append(attempt)
        logger.warning(
            "forbidden access attempt tool=%s session_customer=%s resource=%s:%s",
            attempt.tool,
            attempt.session_customer_id,
            attempt.resource_type,
            attempt.resource_id,
        )
