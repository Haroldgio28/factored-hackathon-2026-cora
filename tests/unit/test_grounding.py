"""Tests for task 4.3 grounding checker (REQ-08).

The grounding checker is the deterministic guard that the LLM only rephrased grounded facts and
never originated a figure. These tests are ADVERSARIAL: a generated response that invents a
balance, date, rate, limit or status not present in the current turn's tool results MUST be
blocked; a faithful (grounded) response passes. Number formats are checked under both es/pt
(`1.234,56`) and en (`1,234.56`) grouping, dates across ISO and day-first localized forms, and
empty tool results with a figure in the text fail closed.

They also pin the generator wiring: an ungrounded LLM polish is discarded and the deterministic,
grounded template is returned instead (the block decision is code's, not the model's).
"""

from __future__ import annotations

from datetime import datetime

from cora.agent.generator import generate
from cora.agent.grounding import check_grounding
from cora.agent.llm import StubLLMClient
from cora.agent.templates import Outcome
from cora.tools.base import Result, SourceRef, Status
from cora.tools.models import (
    BalanceData,
    CardDetailsData,
    ConversionData,
    TransactionsData,
    TransactionSummary,
)


def _balance_result() -> Result[BalanceData]:
    """A grounded balance: 1234.56 USD, limit 5000, from products/PRD-1 as of 2026-06-15."""
    return Result[BalanceData](
        status=Status.OK,
        data=BalanceData(
            product_id="PRD-1",
            currency="USD",
            current_balance=1234.56,
            credit_limit=5000.0,
            available_credit=3765.44,
        ),
        source_refs=[SourceRef(table="products", ref="PRD-1")],
        as_of=datetime(2026, 6, 15, 10, 0),
    )


# -- faithful (grounded) responses pass ---------------------------------------------------


def test_grounded_balance_passes_es_format() -> None:
    """A balance stated under es/pt grouping (1.234,56) matches the stored 1234.56."""
    result = check_grounding("Tu saldo actual es 1.234,56 USD.", [_balance_result()])
    assert result.ok
    assert result.offending == []


def test_grounded_balance_passes_en_format() -> None:
    """The same balance under en grouping (1,234.56) also matches the stored 1234.56."""
    assert check_grounding("Your balance is 1,234.56 USD.", [_balance_result()]).ok


def test_grounded_limit_passes() -> None:
    """A credit limit present in the tool result is grounded."""
    assert check_grounding("Tu limite es de 5000 USD.", [_balance_result()]).ok


def test_grounded_date_passes_localized() -> None:
    """The as_of date 2026-06-15 stated day-first (15/06/2026) is grounded."""
    assert check_grounding("Informacion al 15/06/2026.", [_balance_result()]).ok


def test_grounded_date_passes_iso() -> None:
    """The same date in ISO form is grounded."""
    assert check_grounding("Datos al 2026-06-15.", [_balance_result()]).ok


# -- adversarial: invented figures are blocked --------------------------------------------


def test_invented_balance_is_blocked() -> None:
    """REQ-08: a balance the tools never returned is blocked and listed as offending."""
    result = check_grounding("Tu saldo es 9.999,99 USD.", [_balance_result()])
    assert not result.ok
    assert "9.999,99" in result.offending


def test_invented_limit_is_blocked() -> None:
    """An invented credit limit is blocked."""
    result = check_grounding("Tu limite es de 20000 USD.", [_balance_result()])
    assert not result.ok
    assert "20000" in result.offending


def test_invented_date_is_blocked() -> None:
    """An invented date not among the turn's tool dates is blocked."""
    assert not check_grounding("Tu tarjeta vence el 2030-01-01.", [_balance_result()]).ok


def test_invented_rate_is_blocked() -> None:
    """A conversion rate the tools never produced is blocked (rates are figures too)."""
    conv = Result[ConversionData](
        status=Status.OK,
        data=ConversionData(
            original_amount=100.0,
            from_currency="USD",
            to_currency="COP",
            converted_amount=400000.0,
            rate=4000.0,
            rate_date=datetime(2026, 6, 15).date(),
            requested_date=datetime(2026, 6, 15).date(),
            used_prior_rate=False,
        ),
    )
    # 4000 and 400000 are grounded; an invented 4321 rate is not.
    assert check_grounding("La tasa fue 4000 y el total 400000 COP.", [conv]).ok
    assert not check_grounding("La tasa aplicada fue 4321.", [conv]).ok


def test_invented_status_is_blocked() -> None:
    """A status word the tools did not return (card not actually blocked) is blocked."""
    card = Result[CardDetailsData](
        status=Status.OK,
        data=CardDetailsData(
            product_id="PRD-1",
            product_type="Tarjeta Credito",
            product_number_masked="****1234",
            currency="USD",
            product_status="Active",
            credit_limit=5000.0,
            days_past_due=0,
            expiration_date=datetime(2028, 1, 31),
        ),
    )
    # "activa" matches the stored Active status; "bloqueada" was never returned -> blocked.
    assert check_grounding("Tu tarjeta esta activa.", [card]).ok
    blocked = check_grounding("Tu tarjeta esta bloqueada.", [card])
    assert not blocked.ok
    assert "bloqueada" in blocked.offending


def test_swapped_currency_code_is_blocked() -> None:
    """REQ-08: a polish that keeps the number but swaps the currency (USD -> COP) is blocked.

    The amount is grounded, so only the number check is not enough; the currency code must match a
    currency the tools returned this turn or the customer is told the wrong currency.
    """
    swapped = check_grounding("Tu saldo es 1.234,56 COP.", [_balance_result()])
    assert not swapped.ok
    assert "COP" in swapped.offending
    # The correct code passes.
    assert check_grounding("Tu saldo es 1.234,56 USD.", [_balance_result()]).ok


def test_swapped_fx_currency_is_blocked() -> None:
    """An FX answer renders both codes; swapping the target (COP -> ARS) is blocked."""
    conv = Result[ConversionData](
        status=Status.OK,
        data=ConversionData(
            original_amount=100.0,
            from_currency="USD",
            to_currency="COP",
            converted_amount=400000.0,
            rate=4000.0,
            rate_date=datetime(2026, 6, 15).date(),
            requested_date=datetime(2026, 6, 15).date(),
            used_prior_rate=False,
        ),
    )
    # Both USD and COP are grounded by the conversion result; ARS was never returned.
    assert check_grounding("100 USD equivale a 400000 COP.", [conv]).ok
    bad = check_grounding("100 USD equivale a 400000 ARS.", [conv])
    assert not bad.ok
    assert "ARS" in bad.offending


def test_grounded_status_from_transactions_passes() -> None:
    """A transaction status present in the tool result is grounded."""
    txns = Result[TransactionsData](
        status=Status.OK,
        data=TransactionsData(
            transactions=[
                TransactionSummary(
                    transaction_id="TRX-1",
                    transaction_date=datetime(2026, 6, 10, 12, 0),
                    product_id="PRD-1",
                    amount=50.0,
                    currency="USD",
                    transaction_type="purchase",
                    transaction_status="completed",
                    merchant_name="Store",
                    transaction_category="retail",
                )
            ]
        ),
    )
    assert check_grounding("La transaccion por 50 USD esta completed.", [txns]).ok


def test_masked_product_tail_passes() -> None:
    """The last-four tail of a masked product number is grounded by the tool result string."""
    card = Result[CardDetailsData](
        status=Status.OK,
        data=CardDetailsData(
            product_id="PRD-1",
            product_type="Tarjeta Credito",
            product_number_masked="****1234",
            currency="USD",
            product_status="Active",
            credit_limit=None,
            days_past_due=None,
            expiration_date=None,
        ),
    )
    assert check_grounding("Tu tarjeta terminada en 1234.", [card]).ok


# -- fail closed --------------------------------------------------------------------------


def test_figure_with_empty_tool_results_is_blocked() -> None:
    """Fail closed: any figure in the text with no tool results this turn is blocked."""
    assert not check_grounding("El total es 500.", []).ok


def test_text_without_figures_passes_with_empty_results() -> None:
    """A figure-free message (e.g. a clarify question) is grounded even with no tool results."""
    assert check_grounding("No estoy seguro de haber entendido, puedes repetir?", []).ok


# -- generator wiring ---------------------------------------------------------------------


def test_generator_blocks_ungrounded_polish_and_falls_back() -> None:
    """An LLM polish that invents a figure is discarded; the grounded template is returned.

    The block decision is deterministic code's (the grounding checker), never the model's.
    """
    liar = StubLLMClient(default="Tu saldo es 9.999,99 USD y vence el 2099-12-31.")
    out = generate(
        Outcome.ANSWER,
        "es",
        fields={"facts": "Tu saldo es 1.234,56 USD."},
        tool_results=[_balance_result()],
        client=liar,
    )
    assert not out.polished
    assert out.grounding_blocked
    assert out.prompt_hash is None
    assert out.text == "Tu saldo es 1.234,56 USD."  # the grounded template verbatim


def test_generator_blocks_currency_swapping_polish_and_falls_back() -> None:
    """A polish that keeps the figure but swaps the currency code is discarded for the template.

    This is the adversarial currency case: the number survives the number check, but the swapped
    code is not grounded, so the deterministic checker blocks it and the grounded facts are shown.
    """
    swapper = StubLLMClient(default="Tu saldo es 1.234,56 COP.")  # swapped USD -> COP
    out = generate(
        Outcome.ANSWER,
        "es",
        fields={"facts": "Tu saldo es 1.234,56 USD."},
        tool_results=[_balance_result()],
        client=swapper,
    )
    assert not out.polished
    assert out.grounding_blocked
    assert out.text == "Tu saldo es 1.234,56 USD."  # grounded template verbatim, correct currency


def test_generator_keeps_grounded_polish() -> None:
    """A faithful rephrasing that keeps the grounded figures passes grounding and is kept."""
    honest = StubLLMClient(default="Claro, tu saldo disponible es 1.234,56 USD.")
    out = generate(
        Outcome.ANSWER,
        "es",
        fields={"facts": "Tu saldo es 1.234,56 USD."},
        tool_results=[_balance_result()],
        client=honest,
    )
    assert out.polished
    assert not out.grounding_blocked
    assert out.prompt_hash is not None
