"""Tests for task 2.4: the deterministic policy engine (REQ-13/REQ-15/REQ-33/REQ-34).

The core is a TABLE-DRIVEN suite with one case per rule id. Each case is engineered so that the
NAMED rule is the first one that can match: the inputs satisfy that rule's condition while every
earlier rule's condition is false. That proves two things at once - the rule's condition, and the
top-down FIRST-MATCH-WINS ordering (if ordering were wrong, an earlier rule would steal the case).

Separate cases cover the two non-negotiable guardrails (credit -> abstain_route, money -> refuse)
and the allow-list rejection of an out-of-allow-list LLM proposal (REQ-13).
"""

from __future__ import annotations

import pytest

from cora.policy import (
    Decision,
    Intent,
    PolicyEngine,
    PolicyInput,
    ReferencedTransaction,
    Thresholds,
)

# A real engine loaded from the shipped rules.yaml, used by every case.
ENGINE = PolicyEngine.from_yaml()

# Thresholds mirror the placeholder defaults in rules.yaml so the confidence/fraud cases are
# written against known numbers (final values are chosen in Phase 3, REQ-32).
TAU_FRAUD = ENGINE.thresholds.tau_fraud
TAU_ESCALATE = ENGINE.thresholds.tau_escalate
TAU_CLARIFY = ENGINE.thresholds.tau_clarify


# A fully "clean" authenticated read that resolves to POL-090 (answer); each table case starts
# from a copy of this and perturbs only the fields its target rule needs, so earlier rules stay
# un-triggered and the case isolates one rule.
def _clean_read() -> PolicyInput:
    return PolicyInput(
        session_valid=True,
        intent=Intent.I1,
        intent_confidence=1.0,
        resource_owned=True,
    )


# -- one case per rule id, proving first-match-wins ------------------------------------

# (rule id, input, expected decision). Ordered as in rules.yaml so the table reads like §5.
_CASES: list[tuple[str, PolicyInput, Decision]] = [
    (
        "POL-000",
        # Invalid session beats everything else, even an otherwise-answerable read.
        PolicyInput(session_valid=False, intent=Intent.I1, resource_owned=True),
        Decision.REAUTH,
    ),
    (
        "POL-010",
        # Injection hit AND the LLM proposed an action the intent does not permit (answer is not
        # allowed for a money-movement intent). Session is valid so POL-000 does not fire.
        PolicyInput(
            session_valid=True,
            intent=Intent.X2,
            injection_hit=True,
            proposed_action=Decision.ANSWER,
        ),
        Decision.ABSTAIN,
    ),
    (
        "POL-020",
        # Money movement, no injection: refused by rule (REQ-34).
        PolicyInput(session_valid=True, intent=Intent.X2),
        Decision.REFUSE,
    ),
    (
        "POL-030",
        # Credit: abstain and route, eligibility never evaluated (REQ-33).
        PolicyInput(session_valid=True, intent=Intent.X1),
        Decision.ABSTAIN_ROUTE,
    ),
    (
        "POL-040",
        # Fraud signal on a referenced txn escalates with high priority - even for a read intent
        # that would otherwise be answered.
        PolicyInput(
            session_valid=True,
            intent=Intent.I2,
            resource_owned=True,
            referenced_txn=ReferencedTransaction(is_fraud=True),
        ),
        Decision.ESCALATE,
    ),
    (
        "POL-050",
        # Escalation intent (dispute). No fraud signal, so POL-040 does not pre-empt it.
        PolicyInput(session_valid=True, intent=Intent.E1),
        Decision.ESCALATE,
    ),
    (
        "POL-060",
        # Low confidence (< tau_escalate) on an otherwise-answerable read escalates.
        PolicyInput(
            session_valid=True,
            intent=Intent.I1,
            intent_confidence=TAU_ESCALATE - 0.01,
            resource_owned=True,
        ),
        Decision.ESCALATE,
    ),
    (
        "POL-070",
        # Confidence between tau_escalate and tau_clarify asks a clarifying question. Kept above
        # tau_escalate so POL-060 does not fire first.
        PolicyInput(
            session_valid=True,
            intent=Intent.I1,
            intent_confidence=(TAU_ESCALATE + TAU_CLARIFY) / 2,
            resource_owned=True,
        ),
        Decision.CLARIFY,
    ),
    (
        "POL-080",
        # Card action with the product owned and its state allowing the action: confirm.
        PolicyInput(
            session_valid=True,
            intent=Intent.A1,
            product_owned=True,
            state_allows=True,
        ),
        Decision.CONFIRM,
    ),
    (
        "POL-090",
        # Clean authenticated read of an owned resource: answer.
        _clean_read(),
        Decision.ANSWER,
    ),
    (
        "POL-999",
        # Authenticated but nothing matches: an action on a card that is NOT owned falls through
        # every specific rule to the catch-all abstain (fail closed).
        PolicyInput(session_valid=True, intent=Intent.A1, product_owned=False),
        Decision.ABSTAIN,
    ),
]


@pytest.mark.parametrize(("rule_id", "inp", "expected"), _CASES, ids=[c[0] for c in _CASES])
def test_rule_matches_and_order(rule_id: str, inp: PolicyInput, expected: Decision) -> None:
    decision = ENGINE.decide(inp)
    assert decision.rule_id == rule_id
    assert decision.decision == expected
    # Every decision pins the rules version for traceability (REQ-39).
    assert decision.rules_version == ENGINE.version


def test_every_rule_in_yaml_has_a_table_case() -> None:
    # Guards against adding a rule to rules.yaml without a first-match-wins case for it.
    covered = {rule_id for rule_id, _, _ in _CASES}
    defined = {rule.id for rule in ENGINE.rules}
    assert covered == defined


# -- guardrail specifics ----------------------------------------------------------------


def test_credit_routes_out_of_scope() -> None:
    # REQ-33: credit is never evaluated; it abstains and carries the route reason.
    decision = ENGINE.decide(PolicyInput(session_valid=True, intent=Intent.X1))
    assert decision.rule_id == "POL-030"
    assert decision.decision == Decision.ABSTAIN_ROUTE
    assert decision.route == "CREDIT_OUT_OF_SCOPE"


def test_money_movement_is_refused() -> None:
    # REQ-34: money movement is refused by rule.
    decision = ENGINE.decide(PolicyInput(session_valid=True, intent=Intent.X2))
    assert decision.rule_id == "POL-020"
    assert decision.decision == Decision.REFUSE


def test_fraud_escalation_is_high_priority() -> None:
    decision = ENGINE.decide(
        PolicyInput(
            session_valid=True,
            intent=Intent.I2,
            resource_owned=True,
            referenced_txn=ReferencedTransaction(fraud_score=TAU_FRAUD),
        )
    )
    assert decision.rule_id == "POL-040"
    assert decision.priority == "high"


# -- allow-list rejection (REQ-13: LLM proposes, policy decides) ------------------------


def test_out_of_allow_list_proposal_is_rejected_but_policy_still_decides() -> None:
    # The LLM proposes `refuse` for a read intent, which the allow-list does not permit. The
    # proposal is rejected and the engine decides deterministically from the rules instead,
    # never adopting the proposed step (REQ-13).
    decision = ENGINE.decide(
        PolicyInput(
            session_valid=True,
            intent=Intent.I1,
            resource_owned=True,
            # `refuse` is not a permitted proposal for a read intent -> rejected.
            proposed_action=Decision.REFUSE,
        )
    )
    assert decision.proposal_rejected is True
    # The policy decided from the rules (answer), NOT from the rejected proposal.
    assert decision.decision == Decision.ANSWER
    assert decision.rule_id == "POL-090"


def test_permitted_proposal_is_not_flagged_rejected() -> None:
    decision = ENGINE.decide(
        PolicyInput(
            session_valid=True,
            intent=Intent.I1,
            resource_owned=True,
            proposed_action=Decision.ANSWER,  # permitted for a read
        )
    )
    assert decision.proposal_rejected is False
    assert decision.decision == Decision.ANSWER


def test_allow_list_permits_matrix() -> None:
    allow = ENGINE.allow_list
    # Reads may be answered; actions may not.
    assert allow.permits(Intent.I1, Decision.ANSWER)
    assert not allow.permits(Intent.I1, Decision.CONFIRM)
    # Actions may be confirmed; reads may not.
    assert allow.permits(Intent.A1, Decision.CONFIRM)
    assert not allow.permits(Intent.A1, Decision.ANSWER)
    # Only money movement may be refused.
    assert allow.permits(Intent.X2, Decision.REFUSE)
    assert not allow.permits(Intent.I1, Decision.REFUSE)
    # Decisions outside the allow-list vocabulary are never permitted, even for a known intent.
    assert not allow.permits(Intent.I1, Decision.CLARIFY)
    assert not allow.permits(Intent.X1, Decision.ABSTAIN_ROUTE)
    # A None intent / action is never permitted (fail closed).
    assert not allow.permits(None, Decision.ANSWER)
    assert not allow.permits(Intent.I1, None)


# -- injection text is data, not instructions (REQ-35) ----------------------------------


def test_injection_without_bad_proposal_does_not_abstain_via_pol_010() -> None:
    # An injection hit on a legitimate, permitted read must NOT trigger POL-010: the requested
    # step is inside the allow-list, so the normal rule decides. The screen adds signal; it is
    # not the boundary (design section 2). Instruction-like text never reaches the engine as a
    # rule - only this structured `injection_hit` flag does.
    decision = ENGINE.decide(
        PolicyInput(
            session_valid=True,
            intent=Intent.I1,
            resource_owned=True,
            injection_hit=True,
            proposed_action=Decision.ANSWER,
        )
    )
    assert decision.rule_id == "POL-090"
    assert decision.decision == Decision.ANSWER


# -- loading / validation ---------------------------------------------------------------


def test_injectable_thresholds_override_defaults_but_keep_version() -> None:
    custom = Thresholds(tau_fraud=1.0, tau_escalate=0.9, tau_clarify=0.95)
    engine = PolicyEngine.from_yaml(thresholds=custom)
    assert engine.thresholds == custom
    # Version still comes from the file, pinning the rule table alongside injected thresholds.
    assert engine.version == ENGINE.version
    # A fraud_score of 1.0 now meets the injected tau_fraud and escalates.
    decision = engine.decide(
        PolicyInput(
            session_valid=True,
            intent=Intent.I2,
            resource_owned=True,
            referenced_txn=ReferencedTransaction(fraud_score=1.0),
        )
    )
    assert decision.rule_id == "POL-040"
