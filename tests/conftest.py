"""Shared test fixtures.

The NLU unit tests must never touch the network or download the ~470 MB MiniLM weights / the
fastText lid model: project policy is that the suite runs on deterministic stubs with no network
(see data/nlu/model_card.json and the Phase 3 design). The real encoders are exercised only by the
offline script data/nlu/train_intent_model.py, which is not a test.

`cora.nlu.embeddings.encode` and `cora.nlu.language.detect` select their deterministic stub when
CORA_NLU_STUB=1. Forcing it here for the whole suite is the single source of truth, so individual
NLU test modules no longer each have to remember to set it (test_splits.py had not, which made it
fail closed-network instead of using the hash fallback encoder).

A test that deliberately needs the real encoder can opt out with
`monkeypatch.delenv("CORA_NLU_STUB", raising=False)`; none currently do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The `ui/` Streamlit views live at the repo root (not under `src/`, so not installed), but the
# agent-console smoke test imports `ui.agent_console`. Put the repo root on `sys.path` so `ui` is
# importable in the suite exactly as it is under `python -c "import ui.agent_console"` from root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _force_nlu_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test on the offline, network-free NLU stub encoders by default."""
    monkeypatch.setenv("CORA_NLU_STUB", "1")
