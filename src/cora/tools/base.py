"""Shared tool-layer primitives: the uniform `Result`, status codes, PII masking,
SQL-literal safety and access-attempt logging (task 2.2, design section 6, REQ-12/REQ-36).

Every tool returns a `Result` with a `status` drawn from a fixed vocabulary and never leaks
anything on failure (P4 fail-closed). The tool layer NEVER accepts a `customer_id` as an
input field: it is injected from the verified session by the `ToolLayer` (REQ-12). Helpers
here keep that contract enforceable:

- `mask_product_number` masks a PAN/account number to its last four digits before it reaches
  anything display-bound (REQ-36, security steering).
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
