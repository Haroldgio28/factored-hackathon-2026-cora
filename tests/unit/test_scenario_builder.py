"""Tests for task 6.1: the held-out scenario builder + suite loader (REQ-42).

The builder is driven over a TINY self-built Parquet landing (never the real `data/`, no network),
exactly like `test_card_overlay.py`: a handful of customers across MX/CO/AR, each with one owned
product. The builder cycles that pool deterministically to fill the full design-section-9 mix, so
a small fixture still exercises the ~400-case count, the 50/50 es/pt split, the country strata and
every adversarial family.

Asserted: the §9 mix (240 normal + 60 escalation + 40 unsupported + 60 adversarial = 400 rows),
50/50 es/pt, all three countries present, every adversarial kind present, seed determinism (two
builds with the same seed serialise byte-identically), the es/pt pairing check fails on a dropped
translation, and the reference `expected_decision` matches a direct `PolicyEngine.decide` call
(the reference is the engine's output, not a hand-written label).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cora.data.datasource import LocalSource
from cora.eval.builder import build_suite
from cora.eval.scenarios import (
    ADVERSARIAL_KINDS,
    ScenarioError,
    check_language_pairing,
    load_scenarios,
    write_scenarios,
)
from cora.policy import Decision, Intent, PolicyEngine, PolicyInput


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.astype("string").to_parquet(path, index=False)


def _build_landing(root: Path) -> None:
    customers = pd.DataFrame(
        [
            {"customer_id": f"CLI-{i:013d}", "country": country, "segment": segment}
            for i, (country, segment) in enumerate(
                [
                    ("MX", "Mass"),
                    ("CO", "Affluent"),
                    ("AR", "SME"),
                    ("MX", "Affluent"),
                    ("CO", "Mass"),
                    ("AR", "Mass"),
                ]
            )
        ]
    )
    products = pd.DataFrame(
        [
            {
                "product_id": f"PRD-{i:011d}",
                "customer_id": f"CLI-{i:013d}",
                "product_type": "Tarjeta Crédito",
                "product_number": "4717188030863827",
                "currency": "USD",
                "current_balance": "952.03",
                "credit_limit": "40451.75",
                "days_past_due": "0.0",
                "expiration_date": "2027-09-12",
                "product_status": "Active",
                "last_updated": "2026-06-15 10:00:00",
            }
            for i in range(6)
        ]
    )
    # A USD->COP rate so the I5 reference derivation can serve a rate (REQ-28). Dated well before
    # the builder's latest-rate "today", so the exact-date path is exercised deterministically.
    rates = pd.DataFrame(
        [
            {
                "date": "2026-06-18",
                "source_currency": "USD",
                "target_currency": "COP",
                "exchange_rate": "4000.0",
                "buy_rate": None,
                "sell_rate": None,
            }
        ]
    )
    _write(root / "customers.parquet", customers)
    _write(root / "products.parquet", products)
    _write(root / "daily_exchange_rates.parquet", rates)


@pytest.fixture(scope="module")
def landing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("eval_raw_parquet")
    _build_landing(root)
    return root


@pytest.fixture
def source(landing: Path) -> LocalSource:
    return LocalSource(root=landing)


@pytest.fixture
def policy() -> PolicyEngine:
    return PolicyEngine.from_yaml()


def test_suite_has_the_section9_mix(source: LocalSource, policy: PolicyEngine) -> None:
    scenarios, manifest = build_suite(source, policy, seed=42)

    assert manifest.total == 400
    assert manifest.counts_by_category == {
        "normal": 240,
        "escalation": 60,
        "unsupported": 40,
        "adversarial": 60,
    }


def test_suite_is_fifty_fifty_es_pt(source: LocalSource, policy: PolicyEngine) -> None:
    scenarios, manifest = build_suite(source, policy, seed=42)
    assert manifest.counts_by_language == {"es": 200, "pt": 200}
    # Both languages are present in every category (not just overall).
    for category in ("normal", "escalation", "unsupported", "adversarial"):
        langs = {s.language for s in scenarios if s.category == category}
        assert langs == {"es", "pt"}, category


def test_all_countries_and_adversarial_kinds_present(source: LocalSource, policy: PolicyEngine) -> None:
    scenarios, manifest = build_suite(source, policy, seed=42)
    assert set(manifest.counts_by_country) == {"MX", "CO", "AR"}
    assert set(manifest.counts_by_adversarial_kind) == set(ADVERSARIAL_KINDS)
    # The two REQ-42 guardrail families are present.
    assert "credit_request" in manifest.counts_by_adversarial_kind
    assert "money_movement_request" in manifest.counts_by_adversarial_kind


def test_seed_determinism_byte_identical(source: LocalSource, policy: PolicyEngine, tmp_path: Path) -> None:
    first, _ = build_suite(source, policy, seed=42)
    second, _ = build_suite(source, policy, seed=42)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_scenarios(first, a)
    write_scenarios(second, b)
    assert a.read_bytes() == b.read_bytes()


def test_round_trip_through_jsonl(source: LocalSource, policy: PolicyEngine, tmp_path: Path) -> None:
    scenarios, _ = build_suite(source, policy, seed=42)
    path = tmp_path / "scenarios.jsonl"
    write_scenarios(scenarios, path)
    reloaded = load_scenarios(path)
    assert len(reloaded) == len(scenarios)
    assert reloaded[0].to_dict() == scenarios[0].to_dict()


def test_language_pairing_fails_on_dropped_translation(source: LocalSource, policy: PolicyEngine) -> None:
    scenarios, _ = build_suite(source, policy, seed=42)
    # Drop one pt row and the pairing check must raise.
    pruned = [s for s in scenarios if not (s.language == "pt" and s.base_id == scenarios[0].base_id)]
    with pytest.raises(ScenarioError):
        check_language_pairing(pruned)


def test_expected_decision_matches_policy_engine(source: LocalSource, policy: PolicyEngine) -> None:
    # A money-movement adversarial case must carry the engine's REFUSE decision for X2 - the
    # reference is the engine's output, never a hand-written label (security steering P1).
    scenarios, _ = build_suite(source, policy, seed=42)
    money = next(s for s in scenarios if s.adversarial_kind == "money_movement_request")
    direct = policy.decide(PolicyInput(session_valid=True, intent=Intent.X2, intent_confidence=1.0))
    assert money.expected_decision == direct.decision
    assert money.expected_decision == Decision.REFUSE

    credit = next(s for s in scenarios if s.adversarial_kind == "credit_request")
    direct_credit = policy.decide(PolicyInput(session_valid=True, intent=Intent.X1, intent_confidence=1.0))
    assert credit.expected_decision == direct_credit.decision


def test_non_referenced_reads_and_fx_reference_is_answer(source: LocalSource, policy: PolicyEngine) -> None:
    # Regression for the live read-ownership + FX fix (REQ-04, REQ-28): the reference outcome for
    # a non-referenced read (I6 list products, bare I1 balance) and a valid FX request (I5) is now
    # ANSWER, not the old POL-999 abstain. The owner in this fixture has a product and a USD->COP
    # rate exists, so the builder's independent ownership derivation resolves resource_owned=True.
    scenarios, _ = build_suite(source, policy, seed=42)
    for intent in (Intent.I6, Intent.I1, Intent.I5):
        case = next(s for s in scenarios if s.category == "normal" and s.intent is intent)
        assert case.expected_decision is Decision.ANSWER, intent
    # The reference stays an INDEPENDENT PolicyEngine.decide of an owned read, not a copied answer.
    direct = policy.decide(
        PolicyInput(session_valid=True, intent=Intent.I6, intent_confidence=1.0, resource_owned=True)
    )
    assert direct.decision is Decision.ANSWER
    # The bare I1 reference additionally carries the single owned product's balance figure.
    bare_i1 = next(s for s in scenarios if s.category == "normal" and s.intent is Intent.I1)
    assert "current_balance" in bare_i1.expected_facts
    # A non-referenced I2-I4 has no account-level tool/renderer, so the reference must NOT flip it
    # to a product-list answer: it stays the fail-closed abstain, matching the orchestrator.
    for intent in (Intent.I2, Intent.I3, Intent.I4):
        case = next(s for s in scenarios if s.category == "normal" and s.intent is intent)
        assert case.expected_decision is Decision.ABSTAIN, intent
        assert "product_count" not in case.expected_facts


def test_adversarial_read_families_carry_concrete_setup(source: LocalSource, policy: PolicyEngine) -> None:
    # Eval-validity regression: unauthorized_access and tool_failure must not force resource_owned
    # by label on an own-account utterance (which now answers). Each carries CONCRETE setup the
    # runner replays, and its reference is still the fail-closed abstain derived from a real Result.
    scenarios, _ = build_suite(source, policy, seed=42)

    unauth = next(s for s in scenarios if s.adversarial_kind == "unauthorized_access")
    # A FOREIGN product reference the runner seeds; it must NOT be one of the session customer's.
    assert unauth.referenced_product_id is not None
    assert unauth.referenced_product_id != unauth.product_id
    assert not unauth.force_tool_fault
    assert unauth.expected_decision is Decision.ABSTAIN

    tool_fail = next(s for s in scenarios if s.adversarial_kind == "tool_failure")
    assert tool_fail.force_tool_fault is True
    assert tool_fail.referenced_product_id is None
    assert tool_fail.expected_decision is Decision.ABSTAIN
