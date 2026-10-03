"""Reversible card-freeze overlay for the tool layer (task 2.3, design sections 6/7, REQ-09).

The dataset has NO native card-freeze field (`product_status` is one of
`{Active, Closed, Blocked, Suspended}`) and the source/curated data is READ-ONLY (SRC-84). A
freeze/unfreeze therefore cannot mutate the product record; it is modelled as an OVERLAY: a
small, separate state store keyed by `product_id` that records whether the customer has frozen
the card. The overlay is REVERSIBLE (unfreeze clears the frozen flag) and AUDITABLE (each entry
keeps who/when and the confirmation id that authorised it), exactly as design section 6/7
require ("writes go to a separate state table so they are reversible and auditable").

Two implementations share one `CardOverlay` protocol:
- `JsonCardOverlay` persists to a gitignored runtime JSON file under `data/_state/`
  (consistent with the pipeline's watermark/freshness state in task 1.4), written atomically.
- `InMemoryCardOverlay` keeps the state in a dict for tests and the demo.

Limitation vs a real core-banking API: there is no card-management system to call, so "frozen"
is an overlay flag rather than a field on the card. Documented in TOOL_CONTRACTS.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

# Runtime location, consistent with the pipeline state (`data/_state/`). `data/` is gitignored
# (`/data/` in `.gitignore`), so this file is runtime-only and never committed.
DEFAULT_OVERLAY_PATH = Path("data/_state/card_overlay.json")


@dataclass(frozen=True)
class CardFreezeState:
    """The overlay record for one product: whether it is frozen, plus the audit trail.

    `confirmation_id` is the id of the confirmation that authorised the most recent change, so
    the overlay is auditable; it is never a card number or other PII.
    """

    product_id: str
    customer_id: str
    frozen: bool
    confirmation_id: str
    updated_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "product_id": self.product_id,
            "customer_id": self.customer_id,
            "frozen": self.frozen,
            "confirmation_id": self.confirmation_id,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CardFreezeState:
        return cls(
            product_id=str(data["product_id"]),
            customer_id=str(data["customer_id"]),
            frozen=bool(data["frozen"]),
            confirmation_id=str(data["confirmation_id"]),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
        )


class CardOverlay(Protocol):
    """Reversible per-product freeze overlay: read and write a single product's freeze state."""

    def get(self, product_id: str) -> CardFreezeState | None: ...

    def put(self, state: CardFreezeState) -> None: ...


class InMemoryCardOverlay:
    """In-process overlay (test/demo seam); keeps freeze state in a dict."""

    def __init__(self) -> None:
        self._states: dict[str, CardFreezeState] = {}

    def get(self, product_id: str) -> CardFreezeState | None:
        return self._states.get(product_id)

    def put(self, state: CardFreezeState) -> None:
        self._states[state.product_id] = state


class JsonCardOverlay:
    """Overlay persisted to a gitignored runtime JSON file under `data/_state/`.

    The whole file is a `{product_id: record}` object; it is tiny (one row per card ever
    frozen), so each `get`/`put` reads/writes the whole file. Writes are atomic (tmp file +
    `os.replace`), the same pattern the pipeline uses for its watermark/freshness state so a
    crash mid-write never leaves half-written state.
    """

    def __init__(self, path: Path = DEFAULT_OVERLAY_PATH) -> None:
        self._path = Path(path)

    def _load(self) -> dict[str, CardFreezeState]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return {str(pid): CardFreezeState.from_dict(record) for pid, record in data.items()}

    def _save(self, states: dict[str, CardFreezeState]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {pid: state.to_dict() for pid, state in states.items()}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self._path)

    def get(self, product_id: str) -> CardFreezeState | None:
        return self._load().get(product_id)

    def put(self, state: CardFreezeState) -> None:
        states = self._load()
        states[state.product_id] = state
        self._save(states)
