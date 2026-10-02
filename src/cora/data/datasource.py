"""The `DataSource` abstraction over the LATAM Bank dataset (design sections 6/7).

The same interface serves the local Parquet landing (`LocalSource`) and the AWS
organizer bucket (`S3Source`), so every reader runs identical code locally and on AWS
(REQ-20/REQ-51). All reads are bounded and memory-safe by construction:

- `scan` returns a lazy DuckDB relation; nothing is materialized until the caller
  aggregates or fetches, so a fact table is never pulled whole into memory.
- `count` runs an aggregate on the engine and returns a single integer.
- `fetch_df` *requires* a `limit` (default 1000) so a caller cannot accidentally
  convert a full fact table into a pandas DataFrame.

The dataset layout mirrors the raw landing (`scripts/data/fetch_to_parquet.py`):
dimensions are single files `<table>.parquet`; facts are Hive-partitioned under
`<table>/year=YYYY/month=MM/day=DD/<table>_YYYYMMDD.parquet`. Hive partitioning exposes
`year`, `month`, `day` as INT columns, so partition pruning filters on those columns
(`process_date` is a normal column, NOT the partition key).

Table names are validated against a fixed registry before being interpolated into SQL,
which prevents SQL injection through the `table` argument. The `where`/`columns`
arguments are caller-controlled SQL and must only receive trusted input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    import pandas as pd

# Registry mirroring the raw landing layout (scripts/data/fetch_to_parquet.py).
DIMENSIONS: frozenset[str] = frozenset(
    {
        "customers",
        "products",
        "branches",
        "service_agents",
        "marketing_campaigns",
        "daily_exchange_rates",
    }
)
FACTS: frozenset[str] = frozenset(
    {
        "transactions",
        "call_center_interactions",
        "call_transcripts",
        "satisfaction_surveys",
        "digital_events",
        "complaints",
        "campaign_sends",
    }
)
TABLES: frozenset[str] = DIMENSIONS | FACTS


def _validate_table(table: str) -> None:
    """Reject any table name not in the registry (SQL-injection guard)."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; expected one of {sorted(TABLES)}")


def _build_sql(
    source: str,
    table: str,
    *,
    columns: Sequence[str] | None,
    year: int | None,
    month: int | None,
    day: int | None,
    where: str | None,
    limit: int | None,
) -> str:
    """Shared query builder used by every adapter so Local and S3 cannot drift.

    `source` is the `read_parquet(...)` expression for the given table/engine. Partition
    predicates are emitted as integer comparisons on the Hive `year`/`month`/`day`
    columns (facts only).
    """
    select = ", ".join(columns) if columns else "*"
    predicates: list[str] = []
    if table in FACTS:
        for name, value in (("year", year), ("month", month), ("day", day)):
            if value is not None:
                predicates.append(f"{name} = {int(value)}")
    if where:
        predicates.append(f"({where})")
    sql = f"SELECT {select} FROM {source}"  # noqa: S608 - table validated, source engine-built
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql


class DataSource(ABC):
    """Read-only, memory-safe access to the LATAM Bank dataset (design sections 6/7)."""

    @abstractmethod
    def _source(
        self,
        table: str,
        *,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> str:
        """Return the engine-specific `read_parquet(...)` expression for `table`.

        When a FACT is read with a FULLY-specified single partition (year AND month AND day
        all given), the expression is narrowed to that partition's files so the read is
        per-partition schema-isolated (see BUG-001). A partial spec (year only, year+month,
        or none) keeps the whole-table/subtree glob.
        """

    @abstractmethod
    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Return a configured DuckDB connection (memory limit + threads applied)."""

    def scan(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        where: str | None = None,
        limit: int | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """Return a lazy DuckDB relation; nothing is materialized until fetched."""
        _validate_table(table)
        sql = _build_sql(
            self._source(table, year=year, month=month, day=day),
            table,
            columns=columns,
            year=year,
            month=month,
            day=day,
            where=where,
            limit=limit,
        )
        return self._connect().sql(sql)

    def count(
        self,
        table: str,
        *,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        where: str | None = None,
    ) -> int:
        """Return the row count for `table` under the given filters."""
        _validate_table(table)
        inner = _build_sql(
            self._source(table, year=year, month=month, day=day),
            table,
            columns=None,
            year=year,
            month=month,
            day=day,
            where=where,
            limit=None,
        )
        row = self._connect().sql(f"SELECT count(*) FROM ({inner})").fetchone()
        return int(row[0]) if row else 0

    def fetch_df(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        where: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Materialize at most `limit` rows into a pandas DataFrame.

        `limit` is required (default 1000) so a caller cannot pull a whole fact table
        into memory. A non-positive limit is rejected.
        """
        if limit is None or int(limit) <= 0:
            raise ValueError("fetch_df requires a positive limit")
        return self.scan(
            table,
            columns=columns,
            year=year,
            month=month,
            day=day,
            where=where,
            limit=int(limit),
        ).df()


class LocalSource(DataSource):
    """`DataSource` over the local Parquet landing in `data/raw_parquet` via DuckDB.

    Facts are read with `hive_partitioning=true` over `<table>/**/*.parquet`; dimensions
    are read from the single `<table>.parquet` file. Each call runs on its own DuckDB
    connection configured with a memory limit so large aggregates stay within the host's
    ~4 GB RAM budget (never load a whole fact into pandas). Implements the design §6/§7
    `DataSource` (REQ-20/REQ-51: same interface local and on AWS).
    """

    def __init__(
        self,
        root: Path = Path("data/raw_parquet"),
        memory_limit: str = "2GB",
        threads: int = 4,
    ) -> None:
        self.root = Path(root)
        self.memory_limit = memory_limit
        self.threads = threads
        self._con: duckdb.DuckDBPyConnection | None = None

    def _source(
        self,
        table: str,
        *,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> str:
        if table in FACTS:
            if year is not None and month is not None and day is not None:
                # Per-partition schema isolation (BUG-001): a fully-specified single partition
                # points read_parquet at THAT partition's files, so an added column survives
                # and a missing required column is genuinely absent (not NULL-filled from a
                # sibling). Hive value formats mirror pipeline/fixture convention (D4).
                part = (
                    self.root
                    / table
                    / f"year={int(year)}"
                    / f"month={int(month):02d}"
                    / f"day={int(day):02d}"
                )
                path = str(part / "*.parquet").replace("\\", "/")
            else:
                path = str(self.root / table / "**" / "*.parquet").replace("\\", "/")
            return f"read_parquet('{path}', hive_partitioning=true)"
        path = str(self.root / f"{table}.parquet").replace("\\", "/")
        return f"read_parquet('{path}')"

    def _connect(self) -> duckdb.DuckDBPyConnection:
        # One connection per instance kept alive so returned relations stay usable.
        if self._con is None:
            con = duckdb.connect()
            con.execute(f"SET memory_limit='{self.memory_limit}'; SET threads={int(self.threads)}")
            self._con = con
        return self._con


class S3Source(DataSource):
    """`DataSource` over the organizer bucket, read-only, via DuckDB `httpfs`.

    Defaults are resolved from `cora.settings.get_settings()` (`datathon_bucket`,
    `datathon_region`, `datathon_profile`) so no bucket, region or credential is
    hard-coded. Credentials are resolved from the named SSO profile
    (`boto3.Session(profile_name=...).get_credentials()`) and applied to DuckDB; the
    adapter never writes. An alternative read path for large single objects is
    `aws s3api get-object` (per the AWS steering); this adapter instead reads Parquet in
    place. Construction performs NO network call — credentials and the `httpfs` extension
    are resolved lazily on the first query. Implements the same `scan/count/fetch_df`
    surface as `LocalSource` (REQ-20/REQ-51).
    """

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        profile: str | None = None,
        memory_limit: str = "2GB",
        threads: int = 4,
    ) -> None:
        from cora.settings import get_settings

        settings = get_settings()
        self.bucket = bucket if bucket is not None else settings.datathon_bucket
        self.region = region if region is not None else settings.datathon_region
        self.profile = profile if profile is not None else settings.datathon_profile
        self.memory_limit = memory_limit
        self.threads = threads
        self._con: duckdb.DuckDBPyConnection | None = None

    def _source(
        self,
        table: str,
        *,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> str:
        base = f"s3://{self.bucket}/data"
        if table in FACTS:
            # Mirror LocalSource per-partition narrowing (BUG-001) so Local and S3 do not
            # drift (REQ-20/REQ-51). The path is a pure string; no network call is made here.
            if year is not None and month is not None and day is not None:
                part = f"{base}/{table}/year={int(year)}/month={int(month):02d}/day={int(day):02d}"
                return f"read_parquet('{part}/*.parquet', hive_partitioning=true)"
            return f"read_parquet('{base}/{table}/**/*.parquet', hive_partitioning=true)"
        return f"read_parquet('{base}/{table}.parquet')"

    def _connect(self) -> duckdb.DuckDBPyConnection:
        import boto3

        if self._con is not None:
            return self._con
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{self.memory_limit}'; SET threads={int(self.threads)}")
        con.execute("INSTALL httpfs; LOAD httpfs")
        con.execute(f"SET s3_region='{self.region}'")
        creds = boto3.Session(profile_name=self.profile).get_credentials()
        if creds is not None:
            frozen = creds.get_frozen_credentials()
            con.execute(f"SET s3_access_key_id='{frozen.access_key}'")
            con.execute(f"SET s3_secret_access_key='{frozen.secret_key}'")
            if frozen.token:
                con.execute(f"SET s3_session_token='{frozen.token}'")
        self._con = con
        return con
