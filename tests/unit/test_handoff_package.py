"""Tests for task 4.5: the REQ-16 handoff package builder (REQ-15, REQ-16).

`build_package` assembles the handoff from VERIFIED facts only - the OK tool `Result`s gathered
this turn (their source refs copied verbatim) and the session's masked turn history - never from
model output. These prove:

- every REQ-16 field is present and populated from the verified facts;
- the verbatim request is PII-masked (defence in depth, even if a caller passes raw text);
- priority is `high` on a fraud escalation and `normal` otherwise;
- source refs are carried through so a human can trace each fact.
"""

from __future__ import annotations

from datetime import UTC, datetime

from cora.agent.state import SessionState
from cora.handoff.package import Priority, build_package
from cora.tools.base import Result, SourceRef, Status
from cora.tools.models import BalanceData, HandoffReason, TransactionsData, TransactionSummary

_NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


def _state_with_turn(request: str) -> SessionState:
    state = SessionState(jti="JTI-123", customer_id="CLI-1", language="es")
    state.record_turn(request, "E1", "escalate")
    return state


def _balance_result() -> Result[BalanceData]:
    return Result[BalanceData](
        status=Status.OK,
        data=BalanceData(
            product_id="PRD-1",
            currency="USD",
            current_balance=952.03,
            credit_limit=40451.75,
            available_credit=39499.72,
        ),
        source_refs=[SourceRef(table="products", ref="PRD-1")],
    )


def _transactions_result() -> Result[TransactionsData]:
    return Result[TransactionsData](
        status=Status.OK,
        data=TransactionsData(
            transactions=[
                TransactionSummary(
                    transaction_id="TRX-9",
                    transaction_date=_NOW,
                    product_id="PRD-1",
                    amount=100.0,
                    currency="USD",
                    transaction_type="purchase",
                    transaction_status="posted",
                    merchant_name="Oxxo",
                    transaction_category="retail",
                )
            ]
        ),
        source_refs=[SourceRef(table="transactions", ref="TRX-9")],
    )


def test_package_has_all_req16_fields_from_verified_facts() -> None:
    state = _state_with_turn("no reconozco un cargo")
    package = build_package(
        state,
        [_balance_result(), _transactions_result()],
        HandoffReason.DISPUTE,
        unresolved_questions=["reason", "card_in_possession"],
        actions_taken=["offer_freeze(A1)"],
        now=_NOW,
    )

    # Verbatim request + language.
    assert package.customer_request == "no reconozco un cargo"
    assert package.language == "es"
    # Verified facts with source refs (REQ-16): the balance figures are carried with their refs.
    labels = {f.label: f for f in package.verified_facts}
    assert "Current balance" in labels
    assert labels["Current balance"].value == "952.03"
    assert labels["Current balance"].source_refs == [SourceRef(table="products", ref="PRD-1")]
    # Supporting evidence: the transaction involved.
    assert package.supporting_transactions == ["TRX-9"]
    # Actions taken, unresolved questions, reason, transcript ref.
    assert package.actions_taken == ["offer_freeze(A1)"]
    assert package.unresolved_questions == ["reason", "card_in_possession"]
    assert package.reason is HandoffReason.DISPUTE
    assert package.transcript_ref == "JTI-123"
    assert package.created_at == _NOW


def test_priority_high_on_fraud() -> None:
    state = _state_with_turn("creo que es fraude")
    package = build_package(state, [_balance_result()], HandoffReason.FRAUD, now=_NOW)
    assert package.priority == Priority.HIGH


def test_priority_normal_without_fraud() -> None:
    state = _state_with_turn("quiero una queja")
    package = build_package(state, [], HandoffReason.COMPLAINT, now=_NOW)
    assert package.priority == Priority.NORMAL


def test_explicit_priority_overrides_default() -> None:
    state = _state_with_turn("hola")
    package = build_package(state, [], HandoffReason.DISPUTE, priority=Priority.HIGH, now=_NOW)
    assert package.priority == Priority.HIGH


def test_verbatim_request_is_pii_masked() -> None:
    # Defence in depth: a caller passing an unmasked request still yields a masked package.
    state = SessionState(jti="JTI-9", customer_id="CLI-1", language="es")
    state.record_turn("mi correo es juan.perez@mail.com", "E2", "escalate")
    package = build_package(state, [], HandoffReason.COMPLAINT, now=_NOW)
    assert "juan.perez@mail.com" not in package.customer_request
    assert "@mail.com" in package.customer_request  # masked, not deleted


def test_empty_turn_history_yields_empty_request() -> None:
    state = SessionState(jti="JTI-0", customer_id="CLI-1")
    package = build_package(state, [], HandoffReason.HUMAN_REQUEST, now=_NOW)
    assert package.customer_request == ""
    assert package.verified_facts == []
    assert package.supporting_transactions == []


def test_store_dict_is_json_ready() -> None:
    state = _state_with_turn("hola")
    package = build_package(state, [_balance_result()], HandoffReason.DISPUTE, now=_NOW)
    record = package.to_store_dict()
    assert record["reason"] == "E1"
    assert isinstance(record["verified_facts"], list)
    assert isinstance(record["created_at"], str)  # datetime serialised for JSON
