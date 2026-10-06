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
from datetime import UTC, datetime
from pathlib import Path

from cora.tools.base import mask_pii
from cora.tools.handoff_store import new_case_id

# Runtime location, consistent with the card overlay and the pipeline state. `data/` is
# gitignored (`/data/` in `.gitignore`), so this file is runtime-only and never committed.
DEFAULT_HANDOFF_PATH = Path("data/_state/handoffs.json")

# The agent-workflow lifecycle a human agent drives on a case, from the console (deliverable 5,
# REQ-16). Canonical English status ids are what the store persists; the console shows the es/pt
# labels below. The order is the intended progression (Open -> In progress -> Resolved, with
# Follow-up required as the "needs more work" terminal state).
CASE_STATUSES: tuple[str, ...] = ("Open", "In progress", "Resolved", "Follow-up required")

# es/pt display labels for each status (agent-facing copy; language steering requires BOTH). A
# missing translation is caught by `test_case_status_labels_complete_in_es_and_pt`, mirroring the
# templates' `missing_translations()` guard.
_STATUS_LABELS: dict[str, dict[str, str]] = {
    "Open": {"es": "Abierto", "pt": "Aberto"},
    "In progress": {"es": "En gestion", "pt": "Em andamento"},
    "Resolved": {"es": "Resuelto", "pt": "Resolvido"},
    "Follow-up required": {"es": "Requiere seguimiento", "pt": "Requer acompanhamento"},
}

CASE_STATUS_LANGUAGES: tuple[str, ...] = ("es", "pt")


def status_label(status: str, lang: str) -> str:
    """Return the es/pt display label for a canonical status id (falls back to `es`).

    Pure so the console's localized status choices are testable without Streamlit. An unknown
    `status` returns the canonical id unchanged rather than raising, so a legacy/unexpected value
    still renders a human-readable chip instead of crashing the view.
    """
    labels = _STATUS_LABELS.get(status)
    if labels is None:
        return status
    return labels.get(lang if lang in CASE_STATUS_LANGUAGES else "es", status)


def default_agent_workflow(now: datetime) -> dict[str, object]:
    """Return the default agent-workflow section for a brand-new or legacy case.

    A fresh case starts `Open`, with no notes, no follow-up, stamped at `now`. Pure so backward
    compatibility (legacy rows reading as `Open`) is testable without the store.
    """
    return {"status": "Open", "notes": "", "follow_up": False, "updated_at": now.isoformat()}


def with_agent_workflow(package: dict[str, object]) -> dict[str, object]:
    """Return `package` guaranteed to carry an `agent_workflow` section (inject defaults if absent).

    Backward compatible: a package stored BEFORE the agent-workflow feature (no `agent_workflow`
    key) reads as the Open/empty defaults, so the console always sees the section without ever
    rewriting the file on read. Pure (returns a shallow copy; does not mutate the input).
    """
    if isinstance(package.get("agent_workflow"), dict):
        return dict(package)
    enriched = dict(package)
    enriched["agent_workflow"] = default_agent_workflow(datetime.now(UTC))
    return enriched


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

    def create(self, package: dict[str, object], *, case_id: str | None = None) -> str:
        """Persist a handoff package and return its opaque case id (`CASE-...`).

        `case_id` lets a caller reserve the id before the durable write (task 5.1: the handoff
        span and case id are finalized on the turn and traced BEFORE this persist runs).
        """
        case_id = case_id or new_case_id()
        cases = self._load()
        cases[case_id] = dict(package)
        self._save(cases)
        return case_id

    def get(self, case_id: str) -> dict[str, object] | None:
        """Return one persisted package by case id (read-only; the console renders it).

        The returned package always carries an `agent_workflow` section (defaults for legacy rows
        that predate the feature), so the console reads a uniform shape without rewriting the file.
        """
        package = self._load().get(case_id)
        return None if package is None else with_agent_workflow(package)

    def list_cases(self) -> dict[str, dict[str, object]]:
        """Return every persisted case (the agent console's queue view). Read-only.

        Each package carries an `agent_workflow` section (defaults injected for legacy rows), so
        the console sees a uniform shape; the file itself is left untouched on read.
        """
        return {case_id: with_agent_workflow(package) for case_id, package in self._load().items()}

    def update_agent_workflow(
        self,
        case_id: str,
        *,
        status: str | None = None,
        notes: str | None = None,
        follow_up: bool | None = None,
        now: datetime | None = None,
    ) -> dict[str, object] | None:
        """Apply a human agent's workflow edits to a case and persist them atomically (REQ-16).

        Only the provided fields are changed; the rest of the `agent_workflow` section (and the
        REQ-16 package itself) is left intact. A legacy case without the section gains the defaults
        first (backward compatible). An invalid `status` raises `ValueError` and nothing is written
        (fail closed); an unknown `case_id` returns `None`. The write reuses the atomic tmp +
        `os.replace` path, so a crash mid-write never corrupts the queue.

        Security spine: agent notes are free text, so they are masked with the shared `mask_pii`
        boundary BEFORE they are written - the durable store never gains a new raw-PII path (an
        agent typing an email/phone/document/address/card into the notes cannot leak it). After the
        atomic save the case is RELOADED and the requested fields are verified against what was
        persisted; the method returns the reloaded case only when the post-condition holds, else it
        returns `None` (fail closed) so a caller never reports a write as done without read-back.
        """
        if status is not None and status not in CASE_STATUSES:
            raise ValueError(f"unknown case status: {status!r}")
        cases = self._load()
        package = cases.get(case_id)
        if package is None:
            return None
        workflow = with_agent_workflow(package)["agent_workflow"]
        assert isinstance(workflow, dict)  # noqa: S101 - with_agent_workflow guarantees a dict
        masked_notes = mask_pii(notes) if notes is not None else None
        if status is not None:
            workflow["status"] = status
        if masked_notes is not None:
            workflow["notes"] = masked_notes
        if follow_up is not None:
            workflow["follow_up"] = follow_up
        workflow["updated_at"] = (now or datetime.now(UTC)).isoformat()
        package["agent_workflow"] = workflow
        cases[case_id] = package
        self._save(cases)
        # Read-back verification: reload from disk and confirm the fields we meant to persist are
        # exactly what landed, before reporting success. Fail closed on any mismatch (missing case,
        # concurrent removal, partial write) by returning None.
        reloaded = self.get(case_id)
        if reloaded is None:
            return None
        persisted = reloaded["agent_workflow"]
        assert isinstance(persisted, dict)  # noqa: S101 - get() routes through with_agent_workflow
        if status is not None and persisted.get("status") != status:
            return None
        if masked_notes is not None and persisted.get("notes") != masked_notes:
            return None
        if follow_up is not None and persisted.get("follow_up") != follow_up:
            return None
        return reloaded
