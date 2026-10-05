"""JSONL trace exporter (task 5.1, REQ-39): one JSON record per turn, appended to a local file.

Design section 4 says "every node writes a trace span" and the orchestrator already produces an
ordered `TurnResult.spans` list plus a stable per-turn `trace_id`. This module is the laziest
honest exporter over that: `export_turn(result)` assembles ONE record from the existing
`TurnResult`/`TraceSpan` scalar fields and appends it as a single line to `<trace_dir>/traces.jsonl`
(stdlib `json`, append mode, atomic enough for a single-process append). No server, no SDK.

OTel/X-Ray decision (ponytail, recorded in `.agents/tasks/phase5-5.1-notes.md`): the design names
OpenTelemetry, but the hard requirement (REQ-39) is spans per node/tool exported to local JSONL
with zero infra. The project already emits the spans, so a `json.dumps` per turn is a few lines and
zero new dependencies; adding `opentelemetry-sdk` would buy nothing locally. The AWS target
(OTel -> X-Ray) is kept behind one seam: swap the sink inside `export_turn` (or point
`_LANGFUSE_SINK` at a real exporter) without touching callers.

Trace export is BEST-EFFORT observability (security steering P5, "everything is a record"): a
trace is an AUDIT ARTIFACT, not a gate on an already-verified turn. A trace write failure is
logged and swallowed by the caller and NEVER changes the turn outcome - it never reorders, gates
or rolls back a verified card action or a confirmation prompt. Fail-closed (P1/P4) is about doubt
over identity, authorization, policy or a tool outcome; a non-durable local JSONL append is none
of those, so coupling it into fail-closed control flow would be over-engineering. The orchestrator
calls `export_turn` on every happy-path turn to emit one record; if the append raises, the real
turn result is still returned unchanged.

NO raw PII (security steering, fail closed), enforced two ways rather than trusted by convention:
(1) the record never reads `result.response.text` or the raw customer utterance - only the masked
input and output METADATA (template outcome, polished flag, grounding result); (2) every free-form
field that DOES go in (`masked_input`, `message`, span `detail`) is run through `mask_pii` at the
serialization boundary, so an email/document/phone/address/full-card string that slips into any of
them is masked here, not written raw. The test `tests/unit/test_tracing.py` drives all five PII
categories through the serialized fields to prove the invariant holds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cora.settings import get_settings
from cora.tools.base import mask_pii

if TYPE_CHECKING:
    from cora.agent.graph import TurnResult

__all__ = ["TRACE_FILENAME", "build_record", "export_turn"]

TRACE_FILENAME = "traces.jsonl"


def _mask(value: str | None) -> str | None:
    """Mask any free-form string before it enters a trace record (defence in depth, REQ-36).

    The turn-level fields fed here (`masked_input`, `message`, span `detail`) are supposed to be
    PII-free already, but the exporter does NOT trust the producer's convention: it runs the same
    `mask_pii` the LLM boundary uses, so an email/document/phone/address/full-card string that
    slips into any of them is masked at the serialization boundary, not written raw. `None`/empty
    pass through untouched (nothing to mask).
    """
    return mask_pii(value) if value else value


def build_record(result: TurnResult) -> dict[str, object]:
    """Assemble the REQ-39 execution record for one turn from the existing TurnResult/TraceSpan.

    REQ-39 wants, per turn: masked input, language, intent + confidence, policy decision + rule id,
    the per-node spans, the output signal (template outcome / polished / grounding result) and a
    timestamp. All of that is read from the EXISTING `TurnResult`/`TraceSpan`/`GeneratedResponse`
    scalar fields - no parallel span model. Two deliberate boundaries keep it fail-closed:

    - **No raw customer text:** `result.response.text` (what the customer sees) is NEVER read;
      only the output METADATA (`outcome`, `polished`, `prompt_hash`, grounding/language blocks)
      goes in, so a figure or a PII-shaped value in the rendered reply cannot leak into the trace.
    - **Free-form fields are masked:** `masked_input`, `message` and each span `detail` are run
      through `mask_pii` here regardless of upstream masking (see `_mask`).

    # ponytail: tool call args/latency, retries, model id, tokens and cost are the remaining REQ-39
    #   fields; they are NOT captured on the turn yet (the retry/breaker and tool-timing work is
    #   5.2 and the Bedrock usage surfacing is 5.3). No empty `metrics` placeholder is emitted for
    #   them (YAGNI - an empty schema seam buys no compatibility). Upgrade path: when 5.2/5.3 thread
    #   those values onto the existing `TurnResult`/`TraceSpan`, read them here - no new record type.
    """
    response = result.response
    return {
        "trace_id": result.trace_id,
        "ts": datetime.now(UTC).isoformat(),
        "node": result.node,
        "masked_input": _mask(result.masked_input),
        "language": result.language,
        "intent": result.intent.value if result.intent is not None else None,
        "intent_confidence": result.intent_confidence,
        "decision": result.decision.value if result.decision is not None else None,
        "rule_id": result.rule_id,
        "rules_version": result.rules_version,
        "proposal_rejected": result.proposal_rejected,
        "handoff_case_id": result.handoff_case_id,
        "message": _mask(result.message),
        # Output SIGNAL only - never the rendered customer text (`response.text` is not read).
        "output": None
        if response is None
        else {
            "outcome": response.outcome.value,
            "lang": response.lang,
            "polished": response.polished,
            "prompt_hash": response.prompt_hash,
            "grounding_blocked": response.grounding_blocked,
            "language_blocked": response.language_blocked,
        },
        # Per-node spans (every §4 node writes one). `detail` is masked defensively.
        "spans": [
            {
                "node": span.node,
                "decision": span.decision,
                "rule_id": span.rule_id,
                "rules_version": span.rules_version,
                "detail": _mask(span.detail),
            }
            for span in result.spans
        ],
    }


def export_turn(result: TurnResult, *, settings=None, path: Path | None = None) -> Path:  # noqa: ANN001
    """Append one turn's trace record to the JSONL file and return the file path.

    `path` overrides the destination file directly (used by tests); otherwise the file is
    `<settings.trace_dir>/traces.jsonl` (default a gitignored runtime dir). Parent dirs are created;
    the record is one `json.dumps` line (`default=str` so a datetime/enum never raises).
    """
    target = path if path is not None else _default_path(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = build_record(result)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    _emit_to_langfuse(record)
    return target


def _default_path(settings) -> Path:  # noqa: ANN001
    settings = settings or get_settings()
    return Path(settings.trace_dir) / TRACE_FILENAME


def _emit_to_langfuse(record: dict[str, object]) -> None:
    """Optional Langfuse export seam - intentionally a no-op.

    ponytail: wiring the Langfuse SDK is deferred and adds a dependency only when the owner opts
    in; until then this is a documented, uncalled-in-anger stub so the JSONL sink stays the single
    source of truth. Upgrade path: replace this body with a Langfuse client call behind a settings
    flag (e.g. CORA_LANGFUSE_ENABLED) when a hosted trace UI is actually wanted.
    """
    return None
