"""Pydantic input/output contracts for the tool layer (task 2.2, design section 6).

Inputs carry ONLY what the model/user is allowed to choose; none of them has a `customer_id`
field - that is injected from the verified session by `ToolLayer` (REQ-12). `extra="forbid"`
means a model that smuggles in an unexpected field (e.g. a `customer_id`) is rejected rather
than silently accepted. Output models hold already-masked, display-safe values.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Currencies present in `daily_exchange_rates` (USD/MXN/COP/ARS). Products/transactions never
# carry MXN (ADR-016), but FX pairs do, so conversion must accept it.
_CURRENCIES = frozenset({"USD", "MXN", "COP", "ARS"})


def is_known_currency(code: str) -> bool:
    return code in _CURRENCIES


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -- list_products (I6) ----------------------------------------------------------------


class ListProductsInput(_Input):
    """No customer-chosen fields: the customer is the session owner."""


class ProductSummary(BaseModel):
    product_id: str
    product_type: str
    product_number_masked: str | None
    currency: str
    product_status: str


class ProductsData(BaseModel):
    products: list[ProductSummary]


# -- get_balance (I1) ------------------------------------------------------------------


class GetBalanceInput(_Input):
    product_id: str


class BalanceData(BaseModel):
    product_id: str
    currency: str
    current_balance: float | None
    credit_limit: float | None
    available_credit: float | None


# -- search_transactions (I2, I3) ------------------------------------------------------

MAX_TRANSACTION_LIMIT = 20


class SearchTransactionsInput(_Input):
    product_id: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    merchant: str | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    limit: int = Field(default=MAX_TRANSACTION_LIMIT, ge=1, le=MAX_TRANSACTION_LIMIT)


class TransactionSummary(BaseModel):
    transaction_id: str
    transaction_date: datetime | None
    product_id: str | None
    amount: float | None
    currency: str | None
    transaction_type: str | None
    transaction_status: str | None
    merchant_name: str | None
    transaction_category: str | None


class TransactionsData(BaseModel):
    transactions: list[TransactionSummary]


# -- get_card_details (I4) -------------------------------------------------------------


class GetCardDetailsInput(_Input):
    product_id: str


class CardDetailsData(BaseModel):
    product_id: str
    product_type: str
    product_number_masked: str | None
    currency: str
    product_status: str
    credit_limit: float | None
    days_past_due: int | None
    expiration_date: datetime | None


# -- convert_currency (I5, REQ-28) -----------------------------------------------------


class ConvertCurrencyInput(_Input):
    amount: float = Field(ge=0)
    from_currency: str
    to_currency: str
    on_date: date


class ConversionData(BaseModel):
    original_amount: float
    from_currency: str
    to_currency: str
    converted_amount: float
    rate: float
    rate_date: date
    requested_date: date
    # True when no rate existed for `requested_date` and the latest prior rate was used; the
    # caller MUST surface this (REQ-28 "state it").
    used_prior_rate: bool


# -- freeze_card / unfreeze_card (A1, A2) ----------------------------------------------


class CardFreezeInput(_Input):
    product_id: str
    # A valid confirmation id is required before any state change (design section 6). Presence
    # is enforced here (`min_length=1`); the id is validated against the confirmation store by
    # `ToolLayer._card_action` (task 2.3). The full confirmation protocol is task 4.4.
    confirmation_id: str = Field(min_length=1)


class CardActionData(BaseModel):
    product_id: str
    requested_status: str
    applied: bool


# -- create_handoff (E1-E4) ------------------------------------------------------------


class HandoffReason(StrEnum):
    DISPUTE = "E1"
    COMPLAINT = "E2"
    FRAUD = "E3"
    HUMAN_REQUEST = "E4"
    # A verified identity that is not a current customer of the bank: a welcome + human handoff
    # (E5 is outside the E1-E4 intent map; it is set deterministically by the non-customer branch,
    # never by an intent classifier). Priority is `normal`.
    NON_CUSTOMER = "E5"


class CreateHandoffInput(_Input):
    reason: HandoffReason
    summary: str = Field(min_length=1)
    product_id: str | None = None
    transaction_id: str | None = None


class HandoffData(BaseModel):
    case_id: str
    reason: HandoffReason
