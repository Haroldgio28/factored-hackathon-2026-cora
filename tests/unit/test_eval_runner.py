"""Tests for task 6.3: the runner over a tiny scenario subset (REQ-41/44).

Drives a 3-scenario subset (a normal read, an expired-session adversarial case, both es/pt) through
B1 and CORA x2 repeats over a tiny Parquet landing with faults on, using the StubLLMClient (no
network). Asserted: both configs produce one `RunRecord` per (scenario, repeat); every record
carries the REQ-44 versions (model id, commit, data snapshot); the expired-session CORA case routes
to the fail-closed re-auth node and discloses no facts; fault injection with a fixed seed replays
the same injected failures across the two repeats (a repeat is a replay, not a reroll).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.eval.runner import RunVersions, code_commit, run_suite, write_runs
from cora.eval.scenarios import Scenario
from cora.policy import Decision, Intent


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.astype("string").to_parquet(path, index=False)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("runner_raw_parquet")
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


def _scn(
    base: str,
    language: str,
    *,
    category: str = "normal",
    adversarial_kind: str | None = None,
    decision: Decision = Decision.ANSWER,
    escalation: bool = False,
) -> Scenario:
    return Scenario(
        id=f"{base}-{language}",
        base_id=base,
        category=category,
        intent=Intent.I1,
        language=language,
        country="MX",
        segment="Mass",
        customer_id="CLI-1",
        product_id="PRD-1",
        utterance="¿cuál es mi saldo?" if language == "es" else "qual é o meu saldo?",
        turns=["t"],
        expected_decision=decision,
        expected_escalation=escalation,
        adversarial_kind=adversarial_kind,
        provenance="team-generated-pt" if language == "pt" else "team-generated-es",
    )


@pytest.fixture
def subset() -> list[Scenario]:
    return [
        _scn("normal-000", "es"),
        _scn("normal-000", "pt"),
        _scn(
            "adversarial-expired_session-000",
            "es",
            category="adversarial",
            adversarial_kind="expired_session",
            decision=Decision.REAUTH,
        ),
    ]


@pytest.fixture
def versions(landing: Path) -> RunVersions:
    return RunVersions(model_id="stub", code_commit=code_commit(), data_snapshot=str(landing))


def test_runner_produces_one_record_per_cell(subset, landing, versions) -> None:
    records = run_suite(
        subset, LocalSource(root=landing), repeats=2, seed=42, fault_rate=0.1, versions=versions
    )
    # 3 scenarios x 2 configs x 2 repeats = 12 records.
    assert len(records) == 12
    assert {r.config for r in records} == {"b1", "cora"}
    assert {r.repeat for r in records} == {0, 1}


def test_every_record_carries_versions(subset, landing, versions) -> None:
    records = run_suite(subset, LocalSource(root=landing), repeats=1, fault_rate=0.0, versions=versions)
    for record in records:
        assert record.versions["model_id"] == "stub"
        assert record.versions["data_snapshot"] == str(landing)
        assert "code_commit" in record.versions


def test_expired_session_cora_routes_to_reauth_and_discloses_nothing(subset, landing, versions) -> None:
    records = run_suite(subset, LocalSource(root=landing), repeats=1, fault_rate=0.0, versions=versions)
    expired = [
        r for r in records if r.config == "cora" and r.scenario_id.startswith("adversarial-expired_session")
    ]
    assert len(expired) == 1
    rec = expired[0]
    # The expired turn short-circuits to re-auth BEFORE any policy decision, so the fail-closed
    # signal is the node, not a decision (`produced_decision` is None here, by design).
    assert rec.produced_node == "Unauthenticated"
    assert rec.produced_decision is None
    # Fail closed: no balance figure leaks on the re-auth path.
    assert rec.produced_text is None or "100" not in rec.produced_text


def test_cora_record_has_decision_b1_does_not(subset, landing, versions) -> None:
    records = run_suite(subset, LocalSource(root=landing), repeats=1, fault_rate=0.0, versions=versions)
    cora_normal = [r for r in records if r.config == "cora" and r.category == "normal"]
    b1_normal = [r for r in records if r.config == "b1" and r.category == "normal"]
    assert all(r.produced_decision is not None for r in cora_normal)
    assert all(r.produced_decision is None for r in b1_normal)  # B1 has no policy decision


def test_fault_seed_replays_across_repeats(landing, versions) -> None:
    # A high fault rate so faults certainly occur; the two repeats must see the identical pattern.
    scn = [_scn("normal-000", "es")]
    records = run_suite(
        scn, LocalSource(root=landing), configs=["cora"], repeats=2, seed=1, fault_rate=1.0, versions=versions
    )
    repeat0 = next(r for r in records if r.repeat == 0)
    repeat1 = next(r for r in records if r.repeat == 1)
    # Same injected failures -> same produced decision/text across the replay.
    assert repeat0.produced_decision == repeat1.produced_decision
    assert repeat0.produced_text == repeat1.produced_text


def test_write_runs_round_trips(subset, landing, versions, tmp_path: Path) -> None:
    records = run_suite(subset, LocalSource(root=landing), repeats=1, fault_rate=0.0, versions=versions)
    out = tmp_path / "runs.jsonl"
    write_runs(records, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
