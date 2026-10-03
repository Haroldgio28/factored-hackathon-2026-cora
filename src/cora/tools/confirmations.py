"""Confirmation seam for state-changing tools (task 2.3, REQ-14).

A card freeze/unfreeze MUST NOT execute without a VALID confirmation (REQ-14: a confirmation
that is absent, mismatched or expired => do not execute). The full confirmation PROTOCOL -
restating the exact action and masked target and waiting for an explicit affirmative in the
same session - is task 4.4; this module is the minimal seam the write needs now: a store of
issued confirmations that the write validates against.

A `Confirmation` binds ONE action (`freeze`/`unfreeze`) to ONE `(customer_id, product_id)` and
expires, so it cannot be replayed against another customer, product or action, nor used after
its TTL. The store is NOT single-use in task 2.3 (a still-valid confirmation re-applies the
SAME action to the SAME target, which is idempotent and safe); single-use consumption belongs
to the confirmation protocol in task 4.4.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol

# Confirmations are short-lived: a stale affirmative must not authorise a later write (REQ-14).
DEFAULT_CONFIRMATION_TTL = timedelta(minutes=5)


class CardAction(StrEnum):
    """The two reversible card actions a confirmation can authorise (A1/A2)."""

    FREEZE = "freeze"
    UNFREEZE = "unfreeze"


@dataclass(frozen=True)
class Confirmation:
    """An issued confirmation bound to one customer, action and product, with an expiry."""

    confirmation_id: str
    customer_id: str
    action: CardAction
    product_id: str
    expires_at: datetime

    def is_valid_for(self, *, customer_id: str, action: CardAction, product_id: str, now: datetime) -> bool:
        """True only if this confirmation authorises exactly this action and is unexpired."""
        return (
            self.customer_id == customer_id
            and self.action == action
            and self.product_id == product_id
            and now < self.expires_at
        )


class ConfirmationStore(Protocol):
    """Looks up an issued confirmation by its id (returns `None` when unknown)."""

    def get(self, confirmation_id: str) -> Confirmation | None: ...


class InMemoryConfirmationStore:
    """In-process confirmation store (test/demo seam).

    `issue` is how the confirmation protocol (task 4.4) will record a customer's affirmative
    before calling the write; here it lets tests and the demo mint a valid confirmation.
    """

    def __init__(self) -> None:
        self._items: dict[str, Confirmation] = {}

    def issue(
        self,
        *,
        customer_id: str,
        action: CardAction,
        product_id: str,
        now: datetime,
        ttl: timedelta = DEFAULT_CONFIRMATION_TTL,
    ) -> str:
        confirmation_id = "CONF-" + uuid.uuid4().hex[:12].upper()
        self._items[confirmation_id] = Confirmation(
            confirmation_id=confirmation_id,
            customer_id=customer_id,
            action=CardAction(action),
            product_id=product_id,
            expires_at=now + ttl,
        )
        return confirmation_id

    def get(self, confirmation_id: str) -> Confirmation | None:
        return self._items.get(confirmation_id)
