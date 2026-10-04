"""The handoff package the agent hands a human agent (task 4.5, design section 4, REQ-16).

When the state machine escalates, a human needs everything to continue WITHOUT re-asking the
customer: what they asked (verbatim, in their language), the facts CORA already verified and
where each came from, any action CORA took and whether it was verified, the transactions
involved, the questions still open, why it escalated, how urgent it is, and a pointer back to
the session transcript. REQ-16 enumerates exactly those fields; `HandoffPackage` carries that
set and nothing more.

The package is built from VERIFIED facts only (P1, security steering): every value comes from a
tool `Result` gathered THIS conversation (its `source_refs` are copied verbatim) or from the
session's own masked turn history - never from model output. The verbatim request is the
customer's text AFTER `mask_pii` (the orchestrator masks before anything stores it), so the
package carries no raw PII into the console a human reads.

`build_package(state, tool_results, reason, priority)` assembles it deterministically; the LLM
is never consulted here. Priority is `high` on a fraud escalation (POL-040) and `normal`
otherwise - a deterministic mapping, not a judgement call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from cora.tools.base import Result, SourceRef, mask_pii
from cora.tools.models import HandoffReason

if TYPE_CHECKING:
    from cora.agent.state import SessionState

__all__ = ["HandoffPackage", "Priority", "VerifiedFact", "build_package"]


class Priority:
    """The two priorities a handoff can carry (fraud is urgent, everything else is normal)."""

    HIGH = "high"
    NORMAL = "normal"


class VerifiedFact(BaseModel):
    """One fact CORA verified this turn, with the source refs that back it (REQ-16).

    `value` is a short, display-safe, already-masked string (the tool masks PII before it ever
    reaches a `Result`); `source_refs` are copied verbatim from the tool `Result` so a human can
    trace the fact to the exact record (P5).
    """

    model_config = ConfigDict(frozen=True)

    label: str
    value: str
    source_refs: list[SourceRef] = Field(default_factory=list)


class HandoffPackage(BaseModel):
    """Exactly the REQ-16 fields a human agent needs to take over, from verified facts only.

    Nothing here is model-authored: the request is the masked customer text, the facts and the
    supporting transactions come from tool `Result`s, and the reason/priority come from the
    deterministic escalation path. `transcript_ref` is the session `jti` so the full (masked)
    turn history is retrievable without putting raw PII in the package.
    """

    model_config = ConfigDict(frozen=True)

    customer_request: str  # verbatim customer text, PII-masked
    language: str
    verified_facts: list[VerifiedFact] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    supporting_transactions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    reason: HandoffReason
    priority: str = Priority.NORMAL
    transcript_ref: str
    # Pointer to this turn's JSONL trace (task 5.1, REQ-39): the stable per-turn `trace_id` so a
    # human agent (and the UI) can pull the explainable trace alongside the masked transcript.
    # Optional and opaque; it carries no PII. Empty when no trace id was threaded in.
    trace_ref: str = ""
    created_at: datetime

    def to_store_dict(self) -> dict[str, object]:
        """A JSON-ready dict for the handoff store (the store persists raw dicts, task 2.2)."""
        return self.model_dump(mode="json")


# Facts worth surfacing to a human, pulled from the typed tool `Result.data` by field name. Only
# display-safe fields are read (the tool already masked numbers); the money/limit figures are the
# verified values a human continues from. Kept as a small table so adding a fact is one line, not
# a new code path.
_FACT_FIELDS: tuple[tuple[str, str], ...] = (
    ("product_id", "Product"),
    ("product_number_masked", "Card number"),
    ("product_status", "Card status"),
    ("current_balance", "Current balance"),
    ("credit_limit", "Credit limit"),
    ("available_credit", "Available credit"),
    ("currency", "Currency"),
)


def _facts_from_result(result: Result) -> list[VerifiedFact]:
    """Extract display-safe verified facts from one OK tool `Result`, carrying its source refs."""
    data = result.data
    if data is None:
        return []
    facts: list[VerifiedFact] = []
    for field_name, label in _FACT_FIELDS:
        value = getattr(data, field_name, None)
        if value is None or value == "":
            continue
        facts.append(VerifiedFact(label=label, value=str(value), source_refs=list(result.source_refs)))
    return facts


def _transactions_from_result(result: Result) -> list[str]:
    """Pull transaction ids out of a `search_transactions` Result (the supporting evidence)."""
    data = result.data
    transactions = getattr(data, "transactions", None)
    if not transactions:
        return []
    return [t.transaction_id for t in transactions]


def build_package(
    state: SessionState,
    tool_results: list[Result],
    reason: HandoffReason,
    *,
    priority: str | None = None,
    unresolved_questions: list[str] | None = None,
    actions_taken: list[str] | None = None,
    trace_ref: str = "",
    now: datetime | None = None,
) -> HandoffPackage:
    """Assemble the REQ-16 handoff package deterministically from verified facts (never the LLM).

    The verbatim request is the LAST masked customer turn in the session history; `verified_facts`
    and `supporting_transactions` are read out of THIS turn's OK tool `Result`s (their source refs
    copied verbatim). `priority` defaults to `high` for a fraud escalation (POL-040) and `normal`
    otherwise. `unresolved_questions`/`actions_taken` are passed in by the escalating node (e.g. the
    dispute slot-filler) since they are turn context, not a tool fact. `trace_ref` is the turn's
    opaque `trace_id` (task 5.1) so a human can pull the JSONL trace; it carries no PII.
    """
    now = now or datetime.now(UTC)
    verbatim = state.turn_history[-1].utterance_masked if state.turn_history else ""
    verified_facts: list[VerifiedFact] = []
    supporting_transactions: list[str] = []
    for result in tool_results:
        verified_facts.extend(_facts_from_result(result))
        supporting_transactions.extend(_transactions_from_result(result))

    resolved_priority = priority or (Priority.HIGH if reason is HandoffReason.FRAUD else Priority.NORMAL)
    return HandoffPackage(
        # Defence in depth: the request is already masked upstream, but mask again so the package
        # can never carry raw PII even if a caller passes unmasked text (security steering).
        customer_request=mask_pii(verbatim),
        language=state.language,
        verified_facts=verified_facts,
        actions_taken=actions_taken or [],
        supporting_transactions=supporting_transactions,
        unresolved_questions=unresolved_questions or [],
        reason=reason,
        priority=resolved_priority,
        transcript_ref=state.jti,
        trace_ref=trace_ref,
        created_at=now,
    )
