"""Tests for task 6.3 fault injection: the seeded `FaultInjectingToolLayer` (REQ-44).

Asserted over a tiny Parquet landing + a verified session: a faulted read returns
`Status.UNAVAILABLE` without touching the backend; a fixed seed replays the SAME failure pattern
(so a repeat is a true replay, not a reroll); rate=0 is a pass-through; a non-faultable attribute
(`customer_id`, the write tools) delegates straight to the wrapped layer.
"""

from __future__ import annotations

import random
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.eval.faults import FaultInjectingToolLayer
from cora.identity import MockIdentityService, Session
from cora.tools import GetBalanceInput, Status, ToolLayer


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.astype("string").to_parquet(path, index=False)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("faults_raw_parquet")
    _write(
        root / "customers.parquet",
        pd.DataFrame([{"customer_id": "CLI-1", "country": "MX", "customer_segment": "Mass"}]),
    )
    _write(
        root / "products.parquet",
        pd.DataFrame(
            [
                {
                    "product_id": "PRD-1",
                    "customer_id": "CLI-1",
                    "product_type": "Tarjeta Crédito",
                    "product_number": "4111111111111111",
                    "currency": "USD",
                    "current_balance": "100.0",
                    "credit_limit": "1000.0",
                    "days_past_due": "0.0",
                    "expiration_date": "2027-01-01",
                    "product_status": "Active",
                    "last_updated": "2026-06-15 10:00:00",
                }
            ]
        ),
    )
    return root


def _session() -> Session:
    service = MockIdentityService(signing_key="faults-key", session_ttl=timedelta(minutes=15))
    challenge = service.begin_authentication("CLI-1")
    token = service.complete_authentication(challenge.challenge_id, challenge.code)
    return service.verify_token(token)


@pytest.fixture
def inner(landing: Path) -> ToolLayer:
    return ToolLayer(_session(), LocalSource(root=landing))


def test_rate_zero_is_pass_through(inner: ToolLayer) -> None:
    layer = FaultInjectingToolLayer(inner, rate=0.0, rng=random.Random(0))
    result = layer.get_balance(GetBalanceInput(product_id="PRD-1"))
    assert result.status is Status.OK
    assert result.data is not None and result.data.current_balance == 100.0


def test_rate_one_always_faults_as_unavailable(inner: ToolLayer) -> None:
    layer = FaultInjectingToolLayer(inner, rate=1.0, rng=random.Random(0))
    result = layer.get_balance(GetBalanceInput(product_id="PRD-1"))
    assert result.status is Status.UNAVAILABLE
    assert result.data is None  # fail closed: nothing disclosed


def test_same_seed_replays_the_same_pattern(inner: ToolLayer) -> None:
    one = FaultInjectingToolLayer(inner, rate=0.5, rng=random.Random(7))
    two = FaultInjectingToolLayer(inner, rate=0.5, rng=random.Random(7))
    seq_one = [one.get_balance(GetBalanceInput(product_id="PRD-1")).status for _ in range(30)]
    seq_two = [two.get_balance(GetBalanceInput(product_id="PRD-1")).status for _ in range(30)]
    assert seq_one == seq_two
    assert Status.UNAVAILABLE in seq_one and Status.OK in seq_one  # the rate actually mixes both


def test_non_faultable_attribute_delegates(inner: ToolLayer) -> None:
    layer = FaultInjectingToolLayer(inner, rate=1.0, rng=random.Random(0))
    assert layer.customer_id == "CLI-1"  # a property, never faulted


def test_rate_out_of_range_rejected(inner: ToolLayer) -> None:
    with pytest.raises(ValueError):
        FaultInjectingToolLayer(inner, rate=1.5, rng=random.Random(0))
