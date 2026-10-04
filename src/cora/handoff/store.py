"""Durable JSON handoff store (task 4.5, design section 4, REQ-16).

Task 2.2 shipped the `HandoffStore` protocol and an in-memory stub so the `create_handoff`
tool's contract and authorization were real while persistence waited for Phase 4. This is that
persistence: `JsonHandoffStore` writes each handoff package to a gitignored runtime JSON file
under `data/_state/`, so the agent console (`ui/agent_console.py`) can list open cases across
process restarts.

It reuses `JsonCardOverlay`'s exact pattern: the whole store is one `{case_id: package}` JSON
object (the queue is tiny - one row per escalation), each `create` reads-modifies-writes the
whole file, and the write is ATOMIC (tmp file + `os.replace`) so a crash mid-write never leaves
half-written state. The in-memory stub (`tools/handoff_store.py`) stays the test/demo seam; this
is the durable implementation the demo and the console use.

The stored record is the raw package dict (masking is the builder's job, task 4.5) - the store
never inspects or re-interprets it, so no instruction-like text in a package is ever acted on.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

# Runtime location, consistent with the card overlay and the pipeline state. `data/` is
# gitignored (`/data/` in `.gitignore`), so this file is runtime-only and never committed.
DEFAULT_HANDOFF_PATH = Path("data/_state/handoffs.json")


class JsonHandoffStore:
    """Handoff store persisted to a gitignored runtime JSON file under `data/_state/`.

    Implements the `HandoffStore` protocol (`create`) plus read helpers the agent console needs
    (`get`, `list_cases`). Writes are atomic (tmp file + `os.replace`), the same pattern
    `JsonCardOverlay` uses, so a crashed write never corrupts the queue.
    """

    def __init__(self, path: Path = DEFAULT_HANDOFF_PATH) -> None:
        self._path = Path(path)

    def _load(self) -> dict[str, dict[str, object]]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return {str(case_id): record for case_id, record in data.items()}

    def _save(self, cases: dict[str, dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(cases, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self._path)

    def create(self, package: dict[str, object]) -> str:
        """Persist a handoff package and return its opaque case id (`CASE-...`)."""
        case_id = "CASE-" + uuid.uuid4().hex[:12].upper()
        cases = self._load()
        cases[case_id] = dict(package)
        self._save(cases)
        return case_id

    def get(self, case_id: str) -> dict[str, object] | None:
        """Return one persisted package by case id (read-only; the console renders it)."""
        return self._load().get(case_id)

    def list_cases(self) -> dict[str, dict[str, object]]:
        """Return every persisted case (the agent console's queue view). Read-only."""
        return self._load()
