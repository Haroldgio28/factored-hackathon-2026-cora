"""Data-access layer for CORA (design sections 6/7, REQ-20/REQ-51).

Exposes the `DataSource` interface and its `LocalSource` / `S3Source` adapters so the
rest of the codebase reads the LATAM Bank dataset through one memory-safe surface,
identically against the local Parquet landing and the AWS organizer bucket.
"""

from __future__ import annotations

from cora.data.datasource import DataSource, LocalSource, S3Source

__all__ = ["DataSource", "LocalSource", "S3Source"]
