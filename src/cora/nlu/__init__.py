"""NLU package (Phase 3): gold set, splits, baselines, classifier and screens.

The learned component feeds exactly three `PolicyInput` fields (`intent`, `intent_confidence`,
`injection_hit`); it never decides anything. See `.agents/tasks/phase3-design.md`.
"""

from __future__ import annotations

from cora.nlu.labels import Intent, Provenance, Variant

__all__ = ["Intent", "Provenance", "Variant"]
