"""Tests for task 4.5: the durable JSON handoff store (REQ-16).

`JsonHandoffStore` persists a handoff package to a gitignored runtime JSON file so the agent
console can list cases across restarts. These prove it round-trips a package, assigns an opaque
case id, survives a reopen (durable), and writes atomically (no `.tmp` left behind).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cora.handoff.store import (
    CASE_STATUS_LANGUAGES,
    CASE_STATUSES,
    JsonHandoffStore,
    status_label,
)


def _package() -> dict[str, object]:
    return {
        "customer_request": "no reconozco un cargo",
        "language": "es",
        "verified_facts": [{"label": "Current balance", "value": "952.03", "source_refs": []}],
        "reason": "E1",
        "priority": "high",
        "transcript_ref": "JTI-123",
    }


def test_create_returns_case_id_and_round_trips(tmp_path: Path) -> None:
    store = JsonHandoffStore(path=tmp_path / "handoffs.json")
    case_id = store.create(_package())
    assert case_id.startswith("CASE-")
    got = store.get(case_id)
    assert got is not None and got["customer_request"] == "no reconozco un cargo"
    assert got["priority"] == "high"


def test_persisted_across_new_instance(tmp_path: Path) -> None:
    path = tmp_path / "handoffs.json"
    case_id = JsonHandoffStore(path=path).create(_package())
    # A fresh instance pointed at the same file sees the case (durable).
    reopened = JsonHandoffStore(path=path)
    assert case_id in reopened.list_cases()


def test_multiple_cases_each_get_distinct_id(tmp_path: Path) -> None:
    store = JsonHandoffStore(path=tmp_path / "handoffs.json")
    first = store.create(_package())
    second = store.create(_package())
    assert first != second
    assert set(store.list_cases()) == {first, second}


def test_write_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    path = tmp_path / "handoffs.json"
    JsonHandoffStore(path=path).create(_package())
    assert path.exists()
    # The atomic tmp file is renamed into place, never left behind.
    assert not (tmp_path / "handoffs.json.tmp").exists()


def test_get_unknown_case_is_none(tmp_path: Path) -> None:
    store = JsonHandoffStore(path=tmp_path / "handoffs.json")
    assert store.get("CASE-DOESNOTEXIST") is None


def test_empty_store_lists_nothing(tmp_path: Path) -> None:
    store = JsonHandoffStore(path=tmp_path / "handoffs.json")
    assert store.list_cases() == {}


def test_legacy_package_loads_with_default_agent_workflow(tmp_path: Path) -> None:
    # A package stored BEFORE the agent-workflow feature (no `agent_workflow` key) must read as the
    # Open/empty defaults, so the console sees a uniform shape without rewriting the file.
    store = JsonHandoffStore(path=tmp_path / "handoffs.json")
    case_id = store.create(_package())
    workflow = store.get(case_id)["agent_workflow"]
    assert workflow["status"] == "Open"
    assert workflow["notes"] == ""
    assert workflow["follow_up"] is False
    assert workflow["updated_at"]
    # `list_cases` injects the same defaults.
    assert store.list_cases()[case_id]["agent_workflow"]["status"] == "Open"


def test_update_agent_workflow_persists_and_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "handoffs.json"
    case_id = JsonHandoffStore(path=path).create(_package())
    updated = JsonHandoffStore(path=path).update_agent_workflow(
        case_id, status="In progress", notes="called customer", follow_up=True
    )
    assert updated["agent_workflow"]["status"] == "In progress"
    # A fresh instance at the same path reads the persisted values (durable).
    reopened = JsonHandoffStore(path=path).get(case_id)["agent_workflow"]
    assert reopened["status"] == "In progress"
    assert reopened["notes"] == "called customer"
    assert reopened["follow_up"] is True
    # The REQ-16 package itself is untouched by the workflow edit.
    assert JsonHandoffStore(path=path).get(case_id)["customer_request"] == "no reconozco un cargo"
    # No atomic tmp file left behind.
    assert not (tmp_path / "handoffs.json.tmp").exists()


def test_update_agent_workflow_masks_notes_before_durable_write(tmp_path: Path) -> None:
    # Agent notes are free text: raw PII an agent types must be masked at the STORE boundary before
    # it is persisted, so the durable file never gains a new raw-PII path (security steering).
    path = tmp_path / "handoffs.json"
    case_id = JsonHandoffStore(path=path).create(_package())
    raw = (
        "Cliente escribio desde juan.perez@example.com, telefono +52 55 1234 5678, "
        "DNI 12345678, vive en Calle Falsa 123, tarjeta 4111 1111 1111 1111"
    )
    JsonHandoffStore(path=path).update_agent_workflow(case_id, notes=raw)
    # A fresh instance (true durability) must NOT be able to recover the raw PII from the file.
    stored = JsonHandoffStore(path=path).get(case_id)["agent_workflow"]["notes"]
    raw_file = path.read_text(encoding="utf-8")
    for leak in ("juan.perez@example.com", "+52 55 1234 5678", "Calle Falsa 123"):
        assert leak not in stored
        assert leak not in raw_file
    # No 13-16 digit run (full card/account) survives in the persisted note.
    import re

    assert not re.search(r"\d{13,16}", stored)
    # The note is still non-empty (masking replaces, it does not drop the whole note).
    assert stored.strip()


def test_update_agent_workflow_read_back_verifies_before_returning(tmp_path: Path) -> None:
    # The method must RELOAD and verify the persisted post-condition, returning the verified case.
    path = tmp_path / "handoffs.json"
    case_id = JsonHandoffStore(path=path).create(_package())
    result = JsonHandoffStore(path=path).update_agent_workflow(case_id, status="Resolved")
    assert result is not None
    # The returned case is the reloaded, verified one (status matches what landed on disk).
    assert result["agent_workflow"]["status"] == "Resolved"
    assert JsonHandoffStore(path=path).get(case_id)["agent_workflow"]["status"] == "Resolved"


def test_update_agent_workflow_fails_closed_when_case_vanishes(tmp_path: Path) -> None:
    # If the case disappears concurrently (read-modify-write race, manual removal), the read-back
    # verification fails and the method returns None - the caller must NOT report success.
    path = tmp_path / "handoffs.json"
    case_id = JsonHandoffStore(path=path).create(_package())
    store = JsonHandoffStore(path=path)
    # Simulate the file being emptied between load and the verifying reload by monkeypatching
    # _save to drop the case right after the atomic write.
    original_save = store._save

    def _save_then_drop(cases: dict) -> None:
        original_save(cases)
        empty = {k: v for k, v in cases.items() if k != case_id}
        original_save(empty)

    store._save = _save_then_drop  # type: ignore[method-assign]
    result = store.update_agent_workflow(case_id, status="Resolved")
    assert result is None


def test_status_labels_cover_four_statuses(tmp_path: Path) -> None:
    # The four canonical statuses each have a non-empty localized label (the colour-coding now
    # lives in the console/shared CSS via the `.cora-status-*` classes, not a store-side token).
    labels = {status_label(status, "es") for status in CASE_STATUSES}
    assert len(labels) == len(CASE_STATUSES)


def test_update_only_changes_provided_fields(tmp_path: Path) -> None:
    path = tmp_path / "handoffs.json"
    store = JsonHandoffStore(path=path)
    case_id = store.create(_package())
    store.update_agent_workflow(case_id, notes="first note")
    store.update_agent_workflow(case_id, status="Resolved")
    workflow = store.get(case_id)["agent_workflow"]
    # The earlier note survives a status-only update.
    assert workflow["status"] == "Resolved"
    assert workflow["notes"] == "first note"


def test_update_rejects_unknown_status(tmp_path: Path) -> None:
    path = tmp_path / "handoffs.json"
    store = JsonHandoffStore(path=path)
    case_id = store.create(_package())
    with pytest.raises(ValueError, match="unknown case status"):
        store.update_agent_workflow(case_id, status="Closed")
    # Fail closed: nothing was mutated.
    assert store.get(case_id)["agent_workflow"]["status"] == "Open"


def test_update_unknown_case_returns_none(tmp_path: Path) -> None:
    store = JsonHandoffStore(path=tmp_path / "handoffs.json")
    assert store.update_agent_workflow("CASE-DOESNOTEXIST", status="Resolved") is None


def test_case_status_labels_complete_in_es_and_pt() -> None:
    # Every status must carry a NON-EMPTY es AND pt label (language steering); a missing one fails.
    missing = [
        (status, lang)
        for status in CASE_STATUSES
        for lang in CASE_STATUS_LANGUAGES
        if not status_label(status, lang).strip() or status_label(status, lang) == status
    ]
    assert missing == []


def test_status_label_falls_back_to_es_for_unknown_language() -> None:
    assert status_label("Open", "it") == status_label("Open", "es") == "Abierto"
