"""Tests for task 4.5: the durable JSON handoff store (REQ-16).

`JsonHandoffStore` persists a handoff package to a gitignored runtime JSON file so the agent
console can list cases across restarts. These prove it round-trips a package, assigns an opaque
case id, survives a reopen (durable), and writes atomically (no `.tmp` left behind).
"""

from __future__ import annotations

from pathlib import Path

from cora.handoff.store import JsonHandoffStore


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
