"""Intent label set and provenance vocabulary for the gold set (task 3.1, REQ-30/REQ-37).

The intent label set is NOT redefined here: it is `cora.policy.engine.Intent`, the 16-class
StrEnum the policy engine already decides on (one source of truth, design section 1). This module
re-exports it so NLU code reads `cora.nlu.Intent` without a second vocabulary ever drifting from
the policy one.

Two closed vocabularies specific to the gold set live here:

- `Provenance` records where a labeled utterance came from. It is the REQ-30/REQ-37 audit trail:
  only `team-generated` and `translated` rows may enter a split; `dataset-seed` rows (paraphrased
  from the 42 templated transcript strings, EDA F7) are seed material only and never a test case.
- `Variant` records the country/accent of the utterance so the Spanish set actually covers the
  Mexican, Colombian and Argentine variants the language steering requires (BR tags Portuguese).
"""

from __future__ import annotations

from enum import StrEnum

from cora.policy.engine import Intent

__all__ = ["Intent", "Provenance", "Variant"]


class Provenance(StrEnum):
    """Where a gold-set row came from (REQ-30/REQ-37). `dataset-seed` is never a test case."""

    DATASET_SEED = "dataset-seed"  # paraphrased from one of the 42 templated transcripts (seed only)
    TEAM_GENERATED = "team-generated"  # authored from scratch (es MX/CO/AR or native-style pt)
    TRANSLATED = "translated"  # ES->PT translation of a team row (labeled, may enter a split)


class Variant(StrEnum):
    """Country/accent of an utterance. BR tags Portuguese; the rest tag Spanish variants."""

    MX = "MX"  # Mexican Spanish
    CO = "CO"  # Colombian Spanish
    AR = "AR"  # Argentine Spanish
    BR = "BR"  # Brazilian Portuguese
