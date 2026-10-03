"""Deterministic policy engine (task 2.4, design section 5, REQ-13/REQ-15/REQ-33/REQ-34).

The policy engine is the point where "the LLM proposes, deterministic code decides" (principle
P1). It consumes ONLY typed, structured signals (`PolicyInput`) - never free text - so any
instruction-like string hidden in a user turn, a transcript or a tool result is data, never a
rule (REQ-35, security steering). It loads a versioned YAML rule table (`rules.yaml`), evaluates
the rules TOP-DOWN, and returns the FIRST match together with the matched rule id, so the
decision is explainable from traces and policy ids rather than model reasoning (REQ-39).

Two guardrails are hard-coded by rule, not learned or generated:

- Credit eligibility is NEVER evaluated: an X1 request routes out of scope with reason
  `CREDIT_OUT_OF_SCOPE` (POL-030, REQ-33). The engine computes nothing about eligibility.
- Money movement is refused: an X2 request is refused by rule (POL-020, REQ-34).

The allow-list (REQ-13) maps each intent to the decisions the LLM may propose. An LLM-proposed
next step outside the allow-list is REJECTED: the engine never adopts the proposal, it decides
from the rules regardless, and marks the decision `proposal_rejected=True`. A rejected proposal
combined with an injection-screen hit is itself a rule (POL-010 -> abstain + log).

Thresholds (`tau_fraud`, `tau_escalate`, `tau_clarify`) are placeholder defaults in `rules.yaml`
for Phase 3; they are injectable at construction and the rules `version` is recorded with every
decision so a trace pins exactly which table and thresholds produced it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

logger = logging.getLogger("cora.policy")

_RULES_PATH = Path(__file__).with_name("rules.yaml")

# Read intents (I1..I6) and card-action intents (A1/A2); named once so the predicates and the
# allow-list stay in step with design section 5 / REQ-04.
_READ_INTENTS = frozenset({"I1", "I2", "I3", "I4", "I5", "I6"})
_ACTION_INTENTS = frozenset({"A1", "A2"})
_ESCALATION_INTENTS = frozenset({"E1", "E2", "E3", "E4"})


class Intent(StrEnum):
    """The intents the classifier emits (REQ-29), used as rule inputs. Not a model output."""

    I1 = "I1"  # balance / available credit
    I2 = "I2"  # recent transactions / search
    I3 = "I3"  # explain a transaction's status
    I4 = "I4"  # card status / limit / days past due
    I5 = "I5"  # currency conversion
    I6 = "I6"  # list products
    A1 = "A1"  # freeze a card (confirmation required)
    A2 = "A2"  # unfreeze a card (confirmation required)
    E1 = "E1"  # dispute / unrecognized charge
    E2 = "E2"  # complaint
    E3 = "E3"  # suspected fraud
    E4 = "E4"  # human request / repeated failure
    X1 = "X1"  # credit-limit increase / loan eligibility (out of scope, REQ-33)
    X2 = "X2"  # money movement (refused, REQ-34)
    X3 = "X3"  # anything else
    OTHER = "OTHER"


class Decision(StrEnum):
    """Every outcome the policy table can return (design section 5)."""

    REAUTH = "reauth"
    ABSTAIN = "abstain"
    ABSTAIN_ROUTE = "abstain_route"
    REFUSE = "refuse"
    ESCALATE = "escalate"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    ANSWER = "answer"


# The vocabulary the allow-list is expressed in and the only actions the LLM may propose
# (REQ-13). A proposal must be one of these AND be permitted for the intent.
_ALLOW_LIST_VOCAB = frozenset(
    {Decision.ANSWER, Decision.CONFIRM, Decision.ESCALATE, Decision.ABSTAIN, Decision.REFUSE}
)


@dataclass(frozen=True)
class ReferencedTransaction:
    """Fraud signals for a transaction the turn refers to (POL-040). Values come from a tool,
    never from the model: `is_fraud` and `fraud_score` are columns in the data."""

    is_fraud: bool = False
    fraud_score: float | None = None


@dataclass(frozen=True)
class PolicyInput:
    """The structured signals the policy decides on. No free text: every field is a typed fact
    produced by deterministic code (identity, tools, the classifier), so nothing here can carry
    an instruction (REQ-35)."""

    # Identity (POL-000). False whenever the session is missing, expired or tampered.
    session_valid: bool
    # Classifier output (may be absent before the turn is understood).
    intent: Intent | None = None
    intent_confidence: float = 1.0
    # Injection screen result for this turn (POL-010). The screen adds signal; it is not the
    # security boundary (design section 2).
    injection_hit: bool = False
    # Fraud signal on a referenced transaction (POL-040).
    referenced_txn: ReferencedTransaction | None = None
    # Entity resolution flagged the turn as ambiguous (POL-070).
    ambiguous_entity: bool = False
    # Ownership / state checks resolved deterministically by the tool layer.
    product_owned: bool = False  # the A1/A2 target card belongs to the session customer
    state_allows: bool = False  # the card's current state permits the requested action
    resource_owned: bool = False  # the I1..I6 resource belongs to the session customer
    # The next step the LLM proposes, if any. Validated against the allow-list; never adopted
    # as the decision (the rules decide).
    proposed_action: Decision | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """A policy outcome, carrying the matched rule id for traces and explanations (REQ-39)."""

    rule_id: str
    decision: Decision
    rules_version: str
    priority: str | None = None
    route: str | None = None
    # True when POL-010 fired and this decision should be logged as an injection event.
    log: bool = False
    # True when an LLM-proposed next step was supplied but was outside the allow-list and so
    # was rejected; the decision was taken deterministically regardless (REQ-13).
    proposal_rejected: bool = False


@dataclass(frozen=True)
class _Rule:
    id: str
    when: str
    decision: Decision
    priority: str | None = None
    route: str | None = None
    log: bool = False


@dataclass(frozen=True)
class Thresholds:
    """Injectable decision thresholds (REQ-32). Defaults are placeholders set in Phase 3."""

    tau_fraud: float
    tau_escalate: float
    tau_clarify: float


class AllowList:
    """Maps each intent to the decisions the LLM may propose for it (REQ-13)."""

    def __init__(self, mapping: dict[Intent, frozenset[Decision]]) -> None:
        self._mapping = mapping

    def permits(self, intent: Intent | None, action: Decision | None) -> bool:
        """True iff `action` is a permitted proposal for `intent`.

        An unknown intent (not in the map) or an action outside the allow-list vocabulary is
        never permitted (fail closed).
        """
        if intent is None or action is None:
            return False
        if action not in _ALLOW_LIST_VOCAB:
            return False
        return action in self._mapping.get(intent, frozenset())

    def allowed(self, intent: Intent) -> frozenset[Decision]:
        return self._mapping.get(intent, frozenset())


# -- condition predicates --------------------------------------------------------------
#
# Each predicate maps a rule's `when` name to a pure check over the structured input. Keeping
# the logic in code (not in YAML, not via `eval`) means conditions are typed, testable and
# cannot be influenced by any instruction-like text in the inputs.

_Predicate = Callable[["PolicyEngine", PolicyInput], bool]


def _requested_outside_allow_list(engine: PolicyEngine, inp: PolicyInput) -> bool:
    """True when the requested next step for this intent is outside the allow-list.

    Covers both an unsupported/unknown intent and a proposed action the intent does not permit
    (REQ-13). Used by POL-010 and surfaced as `proposal_rejected`.
    """
    if inp.intent is None or inp.intent not in engine.allow_list_intents:
        return True
    if inp.proposed_action is None:
        return False
    return not engine.allow_list.permits(inp.intent, inp.proposed_action)


def _fraud_signal(engine: PolicyEngine, inp: PolicyInput) -> bool:
    txn = inp.referenced_txn
    if txn is None:
        return False
    if txn.is_fraud:
        return True
    return txn.fraud_score is not None and txn.fraud_score >= engine.thresholds.tau_fraud


_PREDICATES: dict[str, _Predicate] = {
    "session_not_valid": lambda engine, inp: not inp.session_valid,
    "injection_and_requested_outside_allow_list": (
        lambda engine, inp: inp.injection_hit and _requested_outside_allow_list(engine, inp)
    ),
    "intent_money_movement": lambda engine, inp: inp.intent == Intent.X2,
    "intent_credit": lambda engine, inp: inp.intent == Intent.X1,
    "fraud_signal": _fraud_signal,
    "intent_escalation": lambda engine, inp: inp.intent in _ESCALATION_INTENTS,
    "confidence_below_escalate": (lambda engine, inp: inp.intent_confidence < engine.thresholds.tau_escalate),
    "confidence_below_clarify_or_ambiguous_entity": (
        lambda engine, inp: inp.intent_confidence < engine.thresholds.tau_clarify or inp.ambiguous_entity
    ),
    "action_product_owned_and_state_allows": (
        lambda engine, inp: inp.intent in _ACTION_INTENTS and inp.product_owned and inp.state_allows
    ),
    "read_resource_owned": (lambda engine, inp: inp.intent in _READ_INTENTS and inp.resource_owned),
    "otherwise": lambda engine, inp: True,
}


class PolicyError(Exception):
    """The rule table is malformed or incomplete; the engine refuses to operate (fail closed)."""


@dataclass
class PolicyEngine:
    """Loads the versioned rule table and evaluates it top-down, first match wins."""

    version: str
    rules: list[_Rule]
    allow_list: AllowList
    thresholds: Thresholds
    allow_list_intents: frozenset[Intent] = field(default_factory=frozenset)

    @classmethod
    def from_yaml(
        cls, path: Path | str = _RULES_PATH, *, thresholds: Thresholds | None = None
    ) -> PolicyEngine:
        """Build the engine from `rules.yaml`. `thresholds`, if given, overrides the file's
        placeholder defaults while the file's `version` is still recorded on every decision."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PolicyError("rules file is not a mapping")

        version = raw.get("version")
        if not isinstance(version, str) or not version:
            raise PolicyError("rules file is missing a string `version`")

        thresholds = thresholds or _parse_thresholds(raw.get("thresholds"))
        allow_list, allow_intents = _parse_allow_list(raw.get("allow_list"))
        rules = _parse_rules(raw.get("rules"))

        return cls(
            version=version,
            rules=rules,
            allow_list=allow_list,
            thresholds=thresholds,
            allow_list_intents=allow_intents,
        )

    def decide(self, inp: PolicyInput) -> PolicyDecision:
        """Return the first matching rule's decision, with its id, for `inp`.

        The LLM proposal (if any) is validated but never adopted: the decision is always taken
        from the rule table (REQ-13). `proposal_rejected` records whether the proposal was out
        of the allow-list, and a rejected proposal is logged.
        """
        proposal_rejected = inp.proposed_action is not None and not self.allow_list.permits(
            inp.intent, inp.proposed_action
        )
        if proposal_rejected:
            logger.warning(
                "rejected out-of-allow-list proposal intent=%s proposed=%s",
                inp.intent,
                inp.proposed_action,
            )

        for rule in self.rules:
            if _PREDICATES[rule.when](self, inp):
                decision = PolicyDecision(
                    rule_id=rule.id,
                    decision=rule.decision,
                    rules_version=self.version,
                    priority=rule.priority,
                    route=rule.route,
                    log=rule.log,
                    proposal_rejected=proposal_rejected,
                )
                if rule.log:
                    logger.warning(
                        "policy %s -> %s (injection event) intent=%s",
                        rule.id,
                        rule.decision,
                        inp.intent,
                    )
                return decision

        # POL-999 (`otherwise`) always matches, so this is unreachable with a well-formed table.
        # Fail closed if it is ever reached.
        return PolicyDecision(
            rule_id="POL-999",
            decision=Decision.ABSTAIN,
            rules_version=self.version,
            proposal_rejected=proposal_rejected,
        )


def _parse_thresholds(raw: object) -> Thresholds:
    if not isinstance(raw, dict):
        raise PolicyError("rules file is missing a `thresholds` mapping")
    try:
        return Thresholds(
            tau_fraud=float(raw["tau_fraud"]),
            tau_escalate=float(raw["tau_escalate"]),
            tau_clarify=float(raw["tau_clarify"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyError(f"invalid thresholds: {exc}") from exc


def _parse_allow_list(raw: object) -> tuple[AllowList, frozenset[Intent]]:
    if not isinstance(raw, dict):
        raise PolicyError("rules file is missing an `allow_list` mapping")
    mapping: dict[Intent, frozenset[Decision]] = {}
    for intent_name, actions in raw.items():
        try:
            intent = Intent(intent_name)
        except ValueError as exc:
            raise PolicyError(f"unknown intent in allow_list: {intent_name!r}") from exc
        if not isinstance(actions, list) or not actions:
            raise PolicyError(f"allow_list[{intent_name}] must be a non-empty list")
        decisions: set[Decision] = set()
        for action in actions:
            try:
                decision = Decision(action)
            except ValueError as exc:
                raise PolicyError(f"unknown action in allow_list[{intent_name}]: {action!r}") from exc
            if decision not in _ALLOW_LIST_VOCAB:
                raise PolicyError(
                    f"allow_list[{intent_name}] action {action!r} is outside the allowed vocabulary"
                )
            decisions.add(decision)
        mapping[intent] = frozenset(decisions)
    return AllowList(mapping), frozenset(mapping)


def _parse_rules(raw: object) -> list[_Rule]:
    if not isinstance(raw, list) or not raw:
        raise PolicyError("rules file is missing a non-empty `rules` list")
    rules: list[_Rule] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise PolicyError("each rule must be a mapping")
        rule_id = entry.get("id")
        when = entry.get("when")
        decision_name = entry.get("decision")
        if not isinstance(rule_id, str) or not rule_id:
            raise PolicyError("a rule is missing its `id`")
        if rule_id in seen:
            raise PolicyError(f"duplicate rule id: {rule_id}")
        seen.add(rule_id)
        if when not in _PREDICATES:
            raise PolicyError(f"rule {rule_id} has unknown condition {when!r}")
        try:
            decision = Decision(decision_name)
        except ValueError as exc:
            raise PolicyError(f"rule {rule_id} has unknown decision {decision_name!r}") from exc
        rules.append(
            _Rule(
                id=rule_id,
                when=when,
                decision=decision,
                priority=entry.get("priority"),
                route=entry.get("route"),
                log=bool(entry.get("log", False)),
            )
        )
    return rules


__all__ = [
    "AllowList",
    "Decision",
    "Intent",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyError",
    "PolicyInput",
    "ReferencedTransaction",
    "Thresholds",
]
