"""Handoff store seam for the tool layer (task 2.2).

`create_handoff` needs a place to persist a handoff package and return a case id. The real
store (durable queue the agent console reads) is Phase 4; this module provides the interface
and a deterministic in-memory implementation so the tool's CONTRACT and authorization are
real now and only the persistence is swapped later. The stored record is intentionally the
raw package dict - masking/redaction policy for the package belongs to the Phase-4 builder.
"""

from __future__ import annotations

import uuid
from typing import Protocol


def new_case_id() -> str:
    """Mint an opaque `CASE-...` id (shared so a caller can reserve the id before persisting)."""
    return "CASE-" + uuid.uuid4().hex[:12].upper()


class HandoffStore(Protocol):
    """Persists a handoff package and returns an opaque case id."""

    def create(self, package: dict[str, object], *, case_id: str | None = None) -> str: ...


class InMemoryHandoffStore:
    """In-process stub: assigns a `CASE-...` id and keeps packages in a dict (test/demo seam)."""

    def __init__(self) -> None:
        self.cases: dict[str, dict[str, object]] = {}

    def create(self, package: dict[str, object], *, case_id: str | None = None) -> str:
        case_id = case_id or new_case_id()
        self.cases[case_id] = dict(package)
        return case_id
