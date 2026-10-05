"""CORA evaluation harness (Phase 6, design section 9, REQ-41..47).

Everything the held-out evaluation needs lives here: the scenario suite (6.1), the baselines
(6.2), the runner (6.3), the deterministic + LLM judges (6.4/6.5), the metrics/fairness
aggregators (6.6/6.7) and the historical baseline (6.8). The report (6.9) is assembled under
`documentation/reports/`.

The non-negotiable invariant the whole package obeys (security steering P1): reference
outcomes and the deterministic judges are PURE CODE over curated data and the policy engine -
no model output decides a fact, a policy outcome or an escalation. The LLM is consulted only
as a scored subject (the baseline B1 and CORA) and, in 6.5, as a tone/clarity judge that never
touches facts or policy.
"""

from __future__ import annotations

from cora.eval.scenarios import (
    ADVERSARIAL_KINDS,
    CATEGORIES,
    Scenario,
    ScenarioError,
    check_language_pairing,
    load_scenarios,
    write_scenarios,
)

__all__ = [
    "ADVERSARIAL_KINDS",
    "CATEGORIES",
    "Scenario",
    "ScenarioError",
    "check_language_pairing",
    "load_scenarios",
    "write_scenarios",
]
