"""Tests for task 2.3: the sandbox freeze/unfreeze overlay (REQ-09, REQ-14).

The freeze state has no native field in the dataset, so it lives in a reversible OVERLAY that
is separate from the read-only product data. These tests drive the real `ToolLayer.freeze_card`
/`unfreeze_card` against a tiny self-built Parquet landing (never the real `data/`) with REAL
verified sessions from the task-2.1 `MockIdentityService`, so `customer_id` is injected exactly
as in production.

They cover: a confirmed freeze followed by a read-back that shows frozen; an idempotent
double-freeze that is a no-op success; refusal when the confirmation is missing/expired/for the
wrong target (REQ-14); unfreeze reversing a freeze; a post-condition-fail path that reports the
action was NOT completed and offers escalation (REQ-09); and that `JsonCardOverlay` persists the
overlay to a gitignored runtime file and reloads it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.identity import MockIdentityService, Session
from cora.tools import (
    CardAction,
    CardFreezeState,
    InMemoryCardOverlay,
    InMemoryConfirmationStore,
    JsonCardOverlay,
    Status,
    ToolLayer,
)
from cora.tools.card_overlay import CardOverlay
from cora.tools.models import CardFreezeInput

OWNER = "CLI-OWNER0000001"
OTHER = "CLI-OTHER0000002"
OWNER_CARD = "PRD-OWNERCARD01"
OTHER_CARD = "PRD-OTHERCARD1"

# A fixed clock so confirmation TTLs and read-back timestamps are deterministic.
NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


# -- fixture landing -------------------------------------------------------------------


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.astype("string").to_parquet(path, index=False)


def _build_landing(root: Path) -> None:
    products = pd.DataFrame(
        [
            {
                "product_id": OWNER_CARD,
                "customer_id": OWNER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            },
            {
                "product_id": OTHER_CARD,
                "customer_id": OTHER,
                "product_type": "Tarjeta Crédito",
                "product_number": "4000000000000002",
                "currency": "COP",
                "current_balance": "100.0",
                "credit_limit": "5000.0",
                "days_past_due": "0.0",
                "expiration_date": "2027-01-01",
                "product_status": "Active",
                "last_updated": "2026-06-10 08:00:00",
            },
        ]
    )
    _write(root / "products.parquet", products)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("raw_parquet")
    _build_landing(root)
    return root


@pytest.fixture
def source(landing: Path) -> LocalSource:
    return LocalSource(root=landing)


def _session(customer_id: str) -> Session:
    service = MockIdentityService(signing_key="test-key", session_ttl=timedelta(minutes=15))
    challenge = service.begin_authentication(customer_id)
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


def _layer(
    source: LocalSource,
    *,
    customer_id: str = OWNER,
    overlay: CardOverlay | None = None,
    confirmations: InMemoryConfirmationStore | None = None,
) -> ToolLayer:
    return ToolLayer(
        _session(customer_id),
        source,
        card_overlay=overlay if overlay is not None else InMemoryCardOverlay(),
        confirmations=confirmations if confirmations is not None else InMemoryConfirmationStore(),
        clock=lambda: NOW,
    )


# -- happy path: freeze, read-back shows frozen ----------------------------------------


def test_freeze_then_readback_shows_frozen(source: LocalSource) -> None:
    overlay = InMemoryCardOverlay()
    confirmations = InMemoryConfirmationStore()
    layer = _layer(source, overlay=overlay, confirmations=confirmations)
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id=OWNER_CARD, now=NOW
    )

    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))

    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.product_id == OWNER_CARD
    assert result.data.requested_status == "Blocked"
    assert result.data.applied is True
    assert result.source_refs[0].table == "card_overlay"
    # The overlay itself was written and reads back as frozen for this customer.
    state = overlay.get(OWNER_CARD)
    assert state is not None
    assert state.frozen is True
    assert state.customer_id == OWNER
    assert state.confirmation_id == confirmation_id


# -- idempotent double-freeze is a no-op success ---------------------------------------


def test_double_freeze_is_idempotent_noop_success(source: LocalSource) -> None:
    overlay = InMemoryCardOverlay()
    confirmations = InMemoryConfirmationStore()
    layer = _layer(source, overlay=overlay, confirmations=confirmations)
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id=OWNER_CARD, now=NOW
    )

    first = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))
    second = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))

    assert first.status is Status.OK
    assert second.status is Status.OK
    assert second.data is not None
    assert second.data.applied is True
    # Still exactly one frozen record for the card; the second write changed nothing.
    state = overlay.get(OWNER_CARD)
    assert state is not None
    assert state.frozen is True


# -- unfreeze reverses a freeze --------------------------------------------------------


def test_unfreeze_reverses_a_freeze(source: LocalSource) -> None:
    overlay = InMemoryCardOverlay()
    confirmations = InMemoryConfirmationStore()
    layer = _layer(source, overlay=overlay, confirmations=confirmations)
    freeze_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id=OWNER_CARD, now=NOW
    )
    unfreeze_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.UNFREEZE, product_id=OWNER_CARD, now=NOW
    )

    layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=freeze_id))
    assert overlay.get(OWNER_CARD).frozen is True  # type: ignore[union-attr]

    result = layer.unfreeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=unfreeze_id))
    assert result.status is Status.OK
    assert result.data is not None
    assert result.data.requested_status == "Active"
    state = overlay.get(OWNER_CARD)
    assert state is not None
    assert state.frozen is False


# -- REQ-14: refuse without a valid confirmation ---------------------------------------


def test_freeze_refuses_unknown_confirmation(source: LocalSource) -> None:
    overlay = InMemoryCardOverlay()
    layer = _layer(source, overlay=overlay)
    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id="CONF-NOPE"))
    assert result.status is Status.INVALID
    assert result.data is None
    assert overlay.get(OWNER_CARD) is None  # nothing written


def test_freeze_refuses_expired_confirmation(source: LocalSource) -> None:
    overlay = InMemoryCardOverlay()
    confirmations = InMemoryConfirmationStore()
    layer = _layer(source, overlay=overlay, confirmations=confirmations)
    # Issued one hour ago with a 5-minute TTL -> already expired at NOW.
    confirmation_id = confirmations.issue(
        customer_id=OWNER,
        action=CardAction.FREEZE,
        product_id=OWNER_CARD,
        now=NOW - timedelta(hours=1),
    )
    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))
    assert result.status is Status.INVALID
    assert overlay.get(OWNER_CARD) is None


def test_freeze_refuses_confirmation_for_another_product(source: LocalSource) -> None:
    confirmations = InMemoryConfirmationStore()
    overlay = InMemoryCardOverlay()
    layer = _layer(source, overlay=overlay, confirmations=confirmations)
    # A valid confirmation, but issued for a different product id.
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id="PRD-OTHERPROD9", now=NOW
    )
    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))
    assert result.status is Status.INVALID
    assert overlay.get(OWNER_CARD) is None


def test_freeze_refuses_confirmation_for_the_other_action(source: LocalSource) -> None:
    confirmations = InMemoryConfirmationStore()
    overlay = InMemoryCardOverlay()
    layer = _layer(source, overlay=overlay, confirmations=confirmations)
    # An UNFREEZE confirmation cannot authorise a FREEZE.
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.UNFREEZE, product_id=OWNER_CARD, now=NOW
    )
    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))
    assert result.status is Status.INVALID
    assert overlay.get(OWNER_CARD) is None


def test_freeze_forbidden_on_foreign_card_before_confirmation(source: LocalSource) -> None:
    # Authorization fails closed BEFORE the confirmation is even consulted; the attempt is logged.
    confirmations = InMemoryConfirmationStore()
    overlay = InMemoryCardOverlay()
    layer = _layer(source, overlay=overlay, confirmations=confirmations)
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id=OTHER_CARD, now=NOW
    )
    result = layer.freeze_card(CardFreezeInput(product_id=OTHER_CARD, confirmation_id=confirmation_id))
    assert result.status is Status.FORBIDDEN
    assert overlay.get(OTHER_CARD) is None
    assert layer.access_log.attempts[0].tool == "freeze_card"


# -- REQ-09: post-condition fails -> not completed + escalation ------------------------


class _DroppingOverlay:
    """An overlay whose writes never land, to drive the read-back failure path (REQ-09)."""

    def get(self, product_id: str) -> CardFreezeState | None:
        return None

    def put(self, state: CardFreezeState) -> None:
        return None


def test_post_condition_failure_reports_not_completed_and_offers_escalation(source: LocalSource) -> None:
    confirmations = InMemoryConfirmationStore()
    layer = _layer(source, overlay=_DroppingOverlay(), confirmations=confirmations)
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id=OWNER_CARD, now=NOW
    )
    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))
    assert result.status is Status.UNAVAILABLE
    assert result.data is None  # fail closed: nothing is claimed done
    assert result.message is not None
    assert "not completed" in result.message
    assert "human" in result.message  # offers escalation


# -- REQ-09 "or the tool errors": a raising overlay fails closed, it does not propagate --------


class _RaisingOverlay:
    """An overlay whose write raises, to drive REQ-09's tool-error branch (corrupt/unwritable
    `data/_state/`). The failure must become a fail-closed UNAVAILABLE, not an exception out of
    the tool."""

    def get(self, product_id: str) -> CardFreezeState | None:
        raise OSError("simulated overlay read failure")

    def put(self, state: CardFreezeState) -> None:
        raise OSError("simulated overlay write failure")


def test_overlay_error_reports_not_completed_and_offers_escalation(source: LocalSource) -> None:
    confirmations = InMemoryConfirmationStore()
    layer = _layer(source, overlay=_RaisingOverlay(), confirmations=confirmations)
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id=OWNER_CARD, now=NOW
    )
    # The raising overlay must not propagate: the tool returns a fail-closed UNAVAILABLE.
    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))
    assert result.status is Status.UNAVAILABLE
    assert result.data is None  # fail closed: nothing is claimed done
    assert result.message is not None
    assert "not completed" in result.message
    assert "human" in result.message  # offers escalation


# -- JsonCardOverlay persistence (gitignored runtime file) -----------------------------


def test_json_overlay_persists_and_reloads(source: LocalSource, tmp_path: Path) -> None:
    overlay_path = tmp_path / "_state" / "card_overlay.json"
    confirmations = InMemoryConfirmationStore()
    layer = _layer(source, overlay=JsonCardOverlay(overlay_path), confirmations=confirmations)
    confirmation_id = confirmations.issue(
        customer_id=OWNER, action=CardAction.FREEZE, product_id=OWNER_CARD, now=NOW
    )

    result = layer.freeze_card(CardFreezeInput(product_id=OWNER_CARD, confirmation_id=confirmation_id))
    assert result.status is Status.OK
    assert overlay_path.exists()  # overlay written to the runtime location

    # A fresh overlay over the same file reads the persisted frozen state back.
    reloaded = JsonCardOverlay(overlay_path).get(OWNER_CARD)
    assert reloaded is not None
    assert reloaded.frozen is True
    assert reloaded.customer_id == OWNER


def test_json_overlay_round_trips_state(tmp_path: Path) -> None:
    overlay = JsonCardOverlay(tmp_path / "card_overlay.json")
    state = CardFreezeState(
        product_id=OWNER_CARD,
        customer_id=OWNER,
        frozen=True,
        confirmation_id="CONF-ABC",
        updated_at=NOW,
    )
    overlay.put(state)
    assert overlay.get(OWNER_CARD) == state
