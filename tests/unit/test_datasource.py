"""Tests for the DataSource interface (CORA task 1.1 remainder, REQ-20/REQ-51).

LocalSource runs against the real local Parquet landing (skipped when it is absent, so
CI without the data still passes). S3Source is checked structurally with boto3
monkeypatched so NO network call happens; a live read is opt-in via an env flag.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from cora.data.datasource import LocalSource, S3Source

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw_parquet"

local_data = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="local Parquet landing (data/raw_parquet) is absent",
)


@pytest.fixture
def source() -> LocalSource:
    return LocalSource(root=DATA_ROOT)


@local_data
def test_local_count_customers(source: LocalSource) -> None:
    assert source.count("customers") == 150000


@local_data
def test_local_scan_returns_relation(source: LocalSource) -> None:
    rel = source.scan("transactions", year=2026, month=6, limit=5)
    assert isinstance(rel, duckdb.DuckDBPyRelation)


@local_data
def test_local_partition_filter_count(source: LocalSource) -> None:
    total = source.count("transactions")
    filtered = source.count("transactions", year=2026, month=6)
    assert 0 < filtered <= total


@local_data
def test_local_fetch_df_respects_limit(source: LocalSource) -> None:
    df = source.fetch_df("transactions", limit=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 5


@local_data
def test_local_fetch_df_default_caps_rows(source: LocalSource) -> None:
    df = source.fetch_df("transactions")
    assert len(df) <= 1000


def test_local_unknown_table_rejected(source: LocalSource) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        source.count("not_a_table")


def test_fetch_df_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive limit"):
        LocalSource(root=DATA_ROOT).fetch_df("customers", limit=0)


def test_s3_constructs_from_settings_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # boto3.Session is a hard failure here: construction must not touch the network.
    import boto3

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("S3Source construction must not create a boto3 Session")

    monkeypatch.setattr(boto3, "Session", _boom)
    src = S3Source(bucket="example-bucket", region="us-east-2", profile="cora-datathon")
    assert src.bucket == "example-bucket"
    assert src.region == "us-east-2"
    assert src.profile == "cora-datathon"
    # The s3 source expression is built without any network call.
    assert "s3://example-bucket/data/customers.parquet" in src._source("customers")
    assert "hive_partitioning=true" in src._source("transactions")


def test_s3_defaults_from_settings() -> None:
    src = S3Source()
    assert src.profile == "cora-datathon"
    assert src.region == "us-east-2"


@pytest.mark.skipif(
    os.getenv("CORA_RUN_S3_LIVE") != "1",
    reason="live AWS read is opt-in (set CORA_RUN_S3_LIVE=1)",
)
def test_s3_live_read() -> None:
    src = S3Source()
    assert src.count("customers") > 0


def test_local_per_partition_schema_isolation(tmp_path: Path) -> None:
    """BUG-001 fix: a fully-specified single-partition fact read is schema-isolated to THAT
    partition's files, so an added column survives and a missing column is not borrowed from a
    sibling partition.

    Builds its own tiny Hive-partitioned landing under tmp_path (never touches real data/):
      - partition A (2026-06-12) carries an EXTRA nullable column `promo_flag`;
      - partition B (2026-06-13) is MISSING `customer_id` that A has.
    Hive value formats mirror the pipeline/fixture convention (year INT, month/day zero-padded).
    """
    table = "transactions"
    root = tmp_path / "raw"

    def _write(year: int, month: int, day: int, df: pd.DataFrame) -> None:
        part = root / table / f"year={year}" / f"month={month:02d}" / f"day={day:02d}"
        part.mkdir(parents=True, exist_ok=True)
        df.to_parquet(part / f"{table}_{year}{month:02d}{day:02d}.parquet", index=False)

    # Partition A: has promo_flag and customer_id.
    _write(
        2026,
        6,
        12,
        pd.DataFrame(
            {
                "transaction_id": ["T1", "T2"],
                "customer_id": ["C1", "C2"],
                "amount": ["10.0", "20.0"],
                "promo_flag": ["spring", "spring"],
            }
        ),
    )
    # Partition B: missing customer_id (and no promo_flag).
    _write(
        2026,
        6,
        13,
        pd.DataFrame(
            {
                "transaction_id": ["T3"],
                "amount": ["30.0"],
            }
        ),
    )

    src = LocalSource(root=root)

    # (a) Partition A returns the extra column.
    df_a = src.fetch_df(table, year=2026, month=6, day=12, limit=10)
    assert "promo_flag" in df_a.columns

    # (b) Partition B does NOT borrow A's schema: neither promo_flag nor customer_id present.
    df_b = src.fetch_df(table, year=2026, month=6, day=13, limit=10)
    assert "promo_flag" not in df_b.columns
    assert "customer_id" not in df_b.columns

    # (c) The read succeeds (no cross-file schema-mismatch IOException) with B's row count.
    assert src.count(table, year=2026, month=6, day=13) == 1
