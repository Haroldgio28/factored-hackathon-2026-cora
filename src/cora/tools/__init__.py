"""CORA tool layer (task 2.2, design section 6, REQ-12/REQ-28/REQ-34/REQ-38).

A small, authorization-enforcing surface over the read-only `DataSource`. Every tool is a
method of `ToolLayer`, which is bound to a VERIFIED `Session`; the session's `customer_id` is
injected into every read and never appears as a tool input, so neither the model nor user
text can widen access (REQ-12). Tools return a uniform `Result{status, data, source_refs,
as_of}`; non-OK results disclose nothing (fail closed). Display-bound values are PII-masked
(REQ-36). There is deliberately NO tool that moves money or changes a balance (REQ-34).
"""

from __future__ import annotations

from cora.tools.base import (
    AccessAttempt,
    AccessLog,
    Result,
    SourceRef,
    Status,
    mask_product_number,
)
from cora.tools.card_overlay import (
    CardFreezeState,
    CardOverlay,
    InMemoryCardOverlay,
    JsonCardOverlay,
)
from cora.tools.confirmations import (
    CardAction,
    Confirmation,
    ConfirmationStore,
    InMemoryConfirmationStore,
)
from cora.tools.handoff_store import HandoffStore, InMemoryHandoffStore
from cora.tools.layer import TOOL_NAMES, ToolLayer
from cora.tools.models import (
    BalanceData,
    CardActionData,
    CardDetailsData,
    CardFreezeInput,
    ConversionData,
    ConvertCurrencyInput,
    CreateHandoffInput,
    GetBalanceInput,
    GetCardDetailsInput,
    HandoffData,
    HandoffReason,
    ListProductsInput,
    ProductsData,
    ProductSummary,
    SearchTransactionsInput,
    TransactionsData,
    TransactionSummary,
)

__all__ = [
    "TOOL_NAMES",
    "AccessAttempt",
    "AccessLog",
    "BalanceData",
    "CardAction",
    "CardActionData",
    "CardDetailsData",
    "CardFreezeInput",
    "CardFreezeState",
    "CardOverlay",
    "Confirmation",
    "ConfirmationStore",
    "ConversionData",
    "ConvertCurrencyInput",
    "CreateHandoffInput",
    "GetBalanceInput",
    "GetCardDetailsInput",
    "HandoffData",
    "HandoffReason",
    "HandoffStore",
    "InMemoryCardOverlay",
    "InMemoryConfirmationStore",
    "InMemoryHandoffStore",
    "JsonCardOverlay",
    "ListProductsInput",
    "ProductSummary",
    "ProductsData",
    "Result",
    "SearchTransactionsInput",
    "SourceRef",
    "Status",
    "ToolLayer",
    "TransactionSummary",
    "TransactionsData",
    "mask_product_number",
]
