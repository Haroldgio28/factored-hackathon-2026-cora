"""The held-out scenario suite record and its JSONL (de)serialisation (task 6.1, REQ-42).

A `Scenario` is one evaluation case: a bilingual, multi-turn simulated-customer script plus the
DETERMINISTIC reference outcome the judges score against (expected policy decision, expected
facts, expected escalation). The reference is computed by the builder from curated data and the
policy engine (never a model), so every case is checkable exactly (design section 9, security
steering P1).

The suite is versioned evidence: `write_scenarios`/`load_scenarios` round-trip it as JSONL with
a stable key order so the same seed + same data snapshot yields a byte-identical file. The es/pt
pairing check mirrors `nlu.goldset.check_pairing`: a missing translation is a loud failure, not a
silent coverage gap, and Portuguese rows carry team-generated provenance (the dataset is ES-only).

Every string a scenario carries (`utterance`, `turns`) is DATA: the prompt-injection adversarial
category deliberately embeds instruction-like text, and nothing downstream ever executes it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cora.policy import Decision, Intent

__all__ = [
    "ADVERSARIAL_KINDS",
    "CATEGORIES",
    "COUNTRIES",
    "EVAL_NOW",
    "LANGUAGES",
    "Scenario",
    "ScenarioError",
    "check_language_pairing",
    "load_scenarios",
    "write_scenarios",
]

# The single evaluation clock shared by the suite builder and the runner. The builder derives its
# date-sensitive reference facts (the I5 FX rate) for THIS date, and the runner executes every
# turn at THIS `now`, so expected and produced outcomes describe the SAME rate row (REQ-28/44).
# Picked to sit a few days after the latest curated `daily_exchange_rates` date (the real snapshot
# ends 2026-06-17; see analysis/EDA_FINDINGS.md), so the FX path resolves a latest-prior rate
# within 7 days (REQ-28) deterministically on both the real snapshot and the test fixture - never a
# wall-clock "today" that would drift stale. Change this only together with the data snapshot the
# suite is built on.
EVAL_NOW: datetime = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)

# The two conversation languages (language steering); ES variants are tagged by `country`.
LANGUAGES: tuple[str, ...] = ("es", "pt")
# The three ES-speaking countries the suite stratifies over; a PT row keeps its ES source's
# country so the es/pt pairing stays 1:1 and the country strata reconcile across languages.
COUNTRIES: tuple[str, ...] = ("MX", "CO", "AR")

# The top-level case categories (design section 9 held-out mix). An adversarial case additionally
# carries an `adversarial_kind` from `ADVERSARIAL_KINDS`.
CATEGORIES: tuple[str, ...] = ("normal", "escalation", "unsupported", "adversarial")

# The six adversarial families from design section 9 PLUS the two REQ-42 guardrail families
# (credit-request and money-movement-request), ten cases each. These are the cases where "fail
# closed and disclose nothing" is the correct behaviour; the judges assert exactly that.
ADVERSARIAL_KINDS: tuple[str, ...] = (
    "missing_or_incorrect_data",
    "expired_session",
    "unauthorized_access",
    "prompt_injection",
    "tool_failure",
    "multilingual_ambiguity",
    "credit_request",
    "money_movement_request",
)

# Portuguese is translated/team-authored (the dataset is ES-only); ES is team-generated. The
# provenance travels with every scenario so the report can label PT results honestly (REQ-47).
_PROVENANCE_ES = "team-generated-es"
_PROVENANCE_PT = "team-generated-pt"


class ScenarioError(ValueError):
    """A scenario record is malformed; the loader refuses it rather than dropping it (fail closed)."""


@dataclass(frozen=True)
class Scenario:
    """One held-out evaluation case with its deterministic reference outcome (REQ-42).

    `id` is stable and language-tagged (`<base>-<lang>`), so an es case and its pt counterpart
    share a `base_id`. `turns` is the full simulated-customer script (>= 1 turn); `utterance` is
    the first turn, kept separate for the single-turn judges and B1. The three `expected_*` fields
    are the reference outcome: `expected_decision` is `PolicyEngine.decide`'s output for this
    case, `expected_facts` are tool-computed values (balances/card state) keyed by name, and
    `expected_escalation` is derived from the decision. `adversarial_kind` is set only for an
    adversarial case. `provenance` records es/pt authorship (REQ-47 labelling).
    """

    id: str
    base_id: str
    category: str
    intent: Intent
    language: str
    country: str
    segment: str
    customer_id: str
    product_id: str | None
    utterance: str
    turns: list[str]
    expected_decision: Decision
    expected_escalation: bool
    expected_facts: dict[str, object] = field(default_factory=dict)
    adversarial_kind: str | None = None
    provenance: str = _PROVENANCE_ES
    simulated: bool = False
    # Adversarial setup the runner must honour so the EXECUTED turn matches the reference outcome
    # (otherwise a fail-closed family would run a legitimate own-account turn and corrupt the
    # authorization metrics). `referenced_product_id` is a product the session customer does NOT
    # own (the unauthorized_access family): the runner seeds it as the turn's referenced product so
    # the real ToolLayer read returns FORBIDDEN/NOT_FOUND -> not owned. `force_tool_fault` makes the
    # runner's fault layer deterministically fail this case's reads with UNAVAILABLE (the
    # tool_failure family), rather than relying on the probabilistic fault_rate.
    referenced_product_id: str | None = None
    force_tool_fault: bool = False

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ScenarioError(f"{self.id}: unknown category {self.category!r}")
        if self.language not in LANGUAGES:
            raise ScenarioError(f"{self.id}: unknown language {self.language!r}")
        if self.country not in COUNTRIES:
            raise ScenarioError(f"{self.id}: unknown country {self.country!r}")
        if not self.turns:
            raise ScenarioError(f"{self.id}: at least one turn is required")
        if self.category == "adversarial":
            if self.adversarial_kind not in ADVERSARIAL_KINDS:
                raise ScenarioError(f"{self.id}: adversarial case needs a known adversarial_kind")
        elif self.adversarial_kind is not None:
            raise ScenarioError(f"{self.id}: adversarial_kind set on a non-adversarial case")
        expected_pt = self.language == "pt"
        if expected_pt and self.provenance != _PROVENANCE_PT:
            raise ScenarioError(f"{self.id}: pt scenario must carry pt provenance")
        if not expected_pt and self.provenance != _PROVENANCE_ES:
            raise ScenarioError(f"{self.id}: es scenario must carry es provenance")

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready dict with the enums rendered as their string values (stable key order)."""
        raw = asdict(self)
        raw["intent"] = self.intent.value
        raw["expected_decision"] = self.expected_decision.value
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> Scenario:
        """Rebuild a `Scenario` from a JSONL row, failing closed on an unknown enum value."""
        try:
            intent = Intent(raw["intent"])
            decision = Decision(raw["expected_decision"])
        except (KeyError, ValueError) as exc:
            raise ScenarioError(f"malformed scenario row: {exc}") from exc
        data = dict(raw)
        data["intent"] = intent
        data["expected_decision"] = decision
        # Tolerate a missing optional field but reject an unexpected extra one (fail closed).
        known = set(cls.__dataclass_fields__)
        extra = set(data) - known
        if extra:
            raise ScenarioError(f"scenario row has unexpected fields: {sorted(extra)}")
        return cls(**data)  # type: ignore[arg-type]


def write_scenarios(scenarios: Iterable[Scenario], path: Path | str) -> None:
    """Write the suite as JSONL (one scenario per line) with a stable, sorted key order.

    `sort_keys=True` + `ensure_ascii=False` makes the file byte-stable for a given suite and
    keeps the es/pt accented utterances readable. The directory is created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(scenario.to_dict(), ensure_ascii=False, sort_keys=True) for scenario in scenarios]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_scenarios(path: Path | str) -> list[Scenario]:
    """Load and validate the JSONL suite, failing closed on any malformed row."""
    path = Path(path)
    if not path.exists():
        raise ScenarioError(f"scenario suite not found: {path}")
    scenarios: list[Scenario] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScenarioError(f"line {lineno}: invalid JSON: {exc}") from exc
        scenarios.append(Scenario.from_dict(row))
    if not scenarios:
        raise ScenarioError(f"scenario suite {path} has no rows")
    return scenarios


def check_language_pairing(scenarios: Sequence[Scenario]) -> None:
    """Assert every case exists in BOTH es and pt (REQ-42, mirrors `goldset.check_pairing`).

    Each logical case has a `base_id` shared by its es and pt rows. A base id present in only one
    language is a dropped translation and fails closed (`ScenarioError`), so a missing es or pt
    counterpart is a loud test failure rather than a silent coverage gap.
    """
    by_language: dict[str, set[str]] = {lang: set() for lang in LANGUAGES}
    for scenario in scenarios:
        by_language[scenario.language].add(scenario.base_id)
    es_only = by_language["es"] - by_language["pt"]
    pt_only = by_language["pt"] - by_language["es"]
    if es_only:
        raise ScenarioError(f"{len(es_only)} es cases have no pt translation, e.g. {sorted(es_only)[:3]}")
    if pt_only:
        raise ScenarioError(f"{len(pt_only)} pt cases have no es source, e.g. {sorted(pt_only)[:3]}")
