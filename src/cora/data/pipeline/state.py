"""Watermark + freshness JSON I/O for the incremental pipeline (task 1.4, REQ-24).

The pipeline's runtime state lives in two small JSON files under `data/_state/`:
- `watermarks.json` — a flat `{table: ISO-date}` object (the high-watermark per table).
- `freshness.json` — a `{table: FreshnessRecord}` object.

`data/` is gitignored (`/data/` + `*.parquet` in `.gitignore`), so these FILES are runtime
state, never committed. The committed artefact is THIS code and the record SCHEMA in
`types.py`. Init-when-absent: `load_watermarks` returns an empty `Watermarks` when the file
does not exist, so the first run of a table processes all available partitions (or from a
configurable start date).

Writes are atomic: the payload is written to a `*.tmp` sibling and then `os.replace`d over
the target, so a crash mid-write never leaves a half-written state file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .types import FreshnessRecord, Watermarks

STATE_DIR = Path("data/_state")
WATERMARKS_FILE = STATE_DIR / "watermarks.json"
FRESHNESS_FILE = STATE_DIR / "freshness.json"


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write `payload` as JSON to `path` atomically (tmp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_watermarks(path: Path = WATERMARKS_FILE) -> Watermarks:
    """Load the watermarks, returning an empty `Watermarks` when the file is absent."""
    path = Path(path)
    if not path.exists():
        return Watermarks()
    data = json.loads(path.read_text(encoding="utf-8"))
    return Watermarks.from_dict(data)


def save_watermarks(watermarks: Watermarks, path: Path = WATERMARKS_FILE) -> None:
    """Persist the watermarks atomically to `path`."""
    _atomic_write_json(Path(path), watermarks.to_dict())


def load_freshness(path: Path = FRESHNESS_FILE) -> dict[str, FreshnessRecord]:
    """Load the per-table freshness records, returning an empty dict when absent."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(table): FreshnessRecord.from_dict(record) for table, record in data.items()}


def save_freshness(records: dict[str, FreshnessRecord], path: Path = FRESHNESS_FILE) -> None:
    """Persist the per-table freshness records atomically to `path`."""
    payload = {table: record.to_dict() for table, record in records.items()}
    _atomic_write_json(Path(path), payload)
