"""The tool layer with per-customer authorization (task 2.2, design section 6).

`ToolLayer` is constructed from a VERIFIED `Session` and a `DataSource`. The session's
`customer_id` is captured once at construction and is the ONLY source of customer identity
for every tool; no tool input carries a `customer_id`, so neither the LLM nor user text can
widen access (REQ-12, P1). All reads go through the `DataSource` with bounded/limited queries
(host ~4 GB RAM). Values that reach display are masked (REQ-36). Every denied cross-customer
access returns `FORBIDDEN` and is logged as an access attempt (REQ-12).

The raw Parquet landing stores every column as VARCHAR, so reads project `TRY_CAST(...)`
expressions to recover typed values; a bad/garbage string becomes NULL rather than raising.

Deliberately absent: there is NO tool that moves money or changes a balance (REQ-34). A
transfer/payment request has nothing to call here; policy refuses it (POL-020).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from cora.identity import Session
from cora.tools.base import (
    AccessAttempt,
    AccessLog,
    Result,
    SourceRef,
    Status,
    mask_product_number,
    sql_str_literal,
    validate_identifier,
)
from cora.tools.card_overlay import CardFreezeState, CardOverlay, InMemoryCardOverlay
from cora.tools.confirmations import CardAction, ConfirmationStore, InMemoryConfirmationStore
from cora.tools.handoff_store import HandoffStore, InMemoryHandoffStore
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
    ListProductsInput,
    ProductsData,
    ProductSummary,
    SearchTransactionsInput,
    TransactionsData,
    TransactionSummary,
    is_known_currency,
)

if TYPE_CHECKING:
    import pandas as pd

    from cora.data.datasource import DataSource

logger = logging.getLogger("cora.tools.layer")

# The seven design-section-6 contracts this layer exposes as eight callables (freeze_card and
# unfreeze_card share one contract row). Named explicitly so a test can assert the surface and
# prove no money-movement tool exists (REQ-34).
TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_products",
        "get_balance",
        "search_transactions",
        "get_card_details",
        "convert_currency",
        "freeze_card",
        "unfreeze_card",
        "create_handoff",
    }
)

# Typed projection of the one `products` read reused by every product-scoped tool.
_PRODUCT_COLUMNS: tuple[str, ...] = (
    "product_id",
    "customer_id",
    "product_type",
    "product_number",
    "currency",
    "TRY_CAST(current_balance AS DOUBLE) AS current_balance",
    "TRY_CAST(credit_limit AS DOUBLE) AS credit_limit",
    "TRY_CAST(TRY_CAST(days_past_due AS DOUBLE) AS BIGINT) AS days_past_due",
    "TRY_CAST(expiration_date AS TIMESTAMP) AS expiration_date",
    "product_status",
    "TRY_CAST(last_updated AS TIMESTAMP) AS last_updated",
)

_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "TRY_CAST(transaction_date AS TIMESTAMP) AS transaction_date",
    "product_id",
    "TRY_CAST(amount AS DOUBLE) AS amount",
    "currency",
    "transaction_type",
    "transaction_status",
    "merchant_name",
    "transaction_category",
)

# FX conversion only accepts a rate dated within this window before the requested date; a
# latest-prior rate older than this is too stale and the tool abstains (REQ-28).
_MAX_RATE_STALENESS = timedelta(days=7)

# Product families where `credit_limit` is a revolving-credit line, so
# `available_credit = credit_limit - current_balance` is a meaningful figure. In this dataset
# three families carry a `credit_limit` (`Tarjeta Crédito`, `Préstamo Personal`,
# `Préstamo Hipotecario`), but for the loan families `credit_limit` is the original principal
# and `current_balance` is outstanding debt, so the subtraction is meaningless and routinely
# negative (measured over-limit share: mortgages 49.2%, personal loans 2.3%, cards 1.2% over
# the full `products` table). Only revolving credit gets a derived `available_credit`; every
# other product type reports `None` (B1). See TOOL_CONTRACTS.md §get_balance.
_REVOLVING_CREDIT_TYPES: frozenset[str] = frozenset({"Tarjeta Crédito"})

# The effective status a freeze/unfreeze projects onto the card. The dataset has no native
# freeze field, so the overlay carries a boolean `frozen`; a frozen card is surfaced as
# `Blocked` and an unfrozen one as `Active` (design section 6 "new status"). These are display
# labels for the ACTION result, not a write to the read-only `product_status`.
_FROZEN_STATUS = "Blocked"
_ACTIVE_STATUS = "Active"

# Fail-closed message when a card action cannot be verified - the read-back post-condition
# failed OR the overlay raised. Either way: disclose nothing, claim nothing done, offer a human
# (REQ-09 "or the tool errors", security.md "on ... tool failure ... offer a human").
_ACTION_NOT_COMPLETED_MESSAGE = (
    "the action could not be verified and was not completed; offering a human agent"
)


# -- scalar coercion from a pandas record (NaN/NaT -> None) ----------------------------


def _is_null(value: object) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _s(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    return None if _is_null(value) else str(value)


def _f(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    return None if _is_null(value) else float(value)  # type: ignore[arg-type]


def _i(row: dict[str, object], key: str) -> int | None:
    value = row.get(key)
    return None if _is_null(value) else int(float(value))  # type: ignore[arg-type]


def _dt(row: dict[str, object], key: str) -> datetime | None:
    value = row.get(key)
    if _is_null(value):
        return None
    to_py = getattr(value, "to_pydatetime", None)
    return to_py() if to_py is not None else value  # type: ignore[return-value]


def _date_literal(value: date) -> str:
    # `value` is a `date`/`datetime`, never user text, so its ISO form is injection-safe.
    return f"DATE '{value.isoformat()[:10]}'"


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalise an `as_of` to UTC-aware so freshness compares across tools (REQ-26, N2).

    Raw-landing `TRY_CAST(... AS TIMESTAMP)` yields naive datetimes; the FX and handoff tools
    produce UTC-aware ones. Normalising at the layer boundary keeps every `as_of` comparable.
    """
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ToolLayer:
    """Session-bound tools. `customer_id` is injected from the verified session, never input."""

    def __init__(
        self,
        session: Session,
        source: DataSource,
        *,
        access_log: AccessLog | None = None,
        handoff_store: HandoffStore | None = None,
        card_overlay: CardOverlay | None = None,
        confirmations: ConfirmationStore | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session = session
        self.source = source
        self.access_log = access_log if access_log is not None else AccessLog()
        self.handoff_store = handoff_store if handoff_store is not None else InMemoryHandoffStore()
        self.card_overlay = card_overlay if card_overlay is not None else InMemoryCardOverlay()
        self.confirmations = confirmations if confirmations is not None else InMemoryConfirmationStore()
        self._clock = clock

    @property
    def customer_id(self) -> str:
        """Read-only view of the verified session's customer id (N3).

        Exposing identity as a property over the frozen session makes the injection invariant
        structural: there is no writable attribute a caller could repoint at another customer.
        """
        return self._session.customer_id

    def tools(self) -> dict[str, Callable[..., Result]]:
        """Registry of the exposed tools, keyed by name (used by the policy layer and tests)."""
        return {
            "list_products": self.list_products,
            "get_balance": self.get_balance,
            "search_transactions": self.search_transactions,
            "get_card_details": self.get_card_details,
            "convert_currency": self.convert_currency,
            "freeze_card": self.freeze_card,
            "unfreeze_card": self.unfreeze_card,
            "create_handoff": self.create_handoff,
        }

    # -- authorization helper ----------------------------------------------------------

    def _load_owned_product(
        self, tool: str, product_id: str
    ) -> tuple[dict[str, object] | None, Status | None]:
        """Load a product and confirm the session customer owns it.

        Returns `(row, None)` when owned, or `(None, status)` where status is INVALID (unsafe
        id), NOT_FOUND (no such product) or FORBIDDEN (exists but belongs to another customer;
        the attempt is logged per REQ-12).
        """
        try:
            validate_identifier(product_id, what="product_id")
        except ValueError:
            return None, Status.INVALID
        df = self.source.fetch_df(
            "products",
            columns=list(_PRODUCT_COLUMNS),
            where=f"product_id = {sql_str_literal(product_id)}",
            limit=1,
        )
        if df.empty:
            return None, Status.NOT_FOUND
        row = df.to_dict("records")[0]
        if _s(row, "customer_id") != self.customer_id:
            self.access_log.record(
                AccessAttempt(
                    tool=tool,
                    session_customer_id=self.customer_id,
                    resource_type="product",
                    resource_id=product_id,
                    at=self._clock(),
                )
            )
            return None, Status.FORBIDDEN
        return row, None

    def _load_owned_transaction(
        self, tool: str, transaction_id: str
    ) -> tuple[dict[str, object] | None, Status | None]:
        """Load a transaction and confirm the session customer owns it (mirror of products, B1).

        REQ-12 names transactions, like products, as a resource that must fail closed when it
        is not owned. Returns `(row, None)` when owned, or `(None, status)` where status is
        INVALID (unsafe id), NOT_FOUND (no such transaction) or FORBIDDEN (exists but belongs
        to another customer; the attempt is logged). One bounded point lookup, limit 1.
        """
        try:
            validate_identifier(transaction_id, what="transaction_id")
        except ValueError:
            return None, Status.INVALID
        df = self.source.fetch_df(
            "transactions",
            columns=["transaction_id", "customer_id"],
            where=f"transaction_id = {sql_str_literal(transaction_id)}",
            limit=1,
        )
        if df.empty:
            return None, Status.NOT_FOUND
        row = df.to_dict("records")[0]
        if _s(row, "customer_id") != self.customer_id:
            self.access_log.record(
                AccessAttempt(
                    tool=tool,
                    session_customer_id=self.customer_id,
                    resource_type="transaction",
                    resource_id=transaction_id,
                    at=self._clock(),
                )
            )
            return None, Status.FORBIDDEN
        return row, None

    @staticmethod
    def _resource_error_message(status: Status, resource: str = "product") -> str:
        return {
            Status.INVALID: f"invalid {resource} id",
            Status.NOT_FOUND: f"{resource} not found",
            Status.FORBIDDEN: "not authorized for this resource",
        }[status]

    # -- I6: list_products -------------------------------------------------------------

    def list_products(self, _input: ListProductsInput | None = None) -> Result[ProductsData]:
        """List the session customer's products with masked numbers (I6). No side effects."""
        df = self.source.fetch_df(
            "products",
            columns=[
                "product_id",
                "product_type",
                "product_number",
                "currency",
                "product_status",
                "TRY_CAST(last_updated AS TIMESTAMP) AS last_updated",
            ],
            where=f"customer_id = {sql_str_literal(self.customer_id)}",
            limit=100,
        )
        rows = df.to_dict("records")
        products = [
            ProductSummary(
                product_id=str(_s(r, "product_id")),
                product_type=str(_s(r, "product_type")),
                product_number_masked=mask_product_number(_s(r, "product_number")),
                currency=str(_s(r, "currency")),
                product_status=str(_s(r, "product_status")),
            )
            for r in rows
        ]
        as_of = max((d for r in rows if (d := _dt(r, "last_updated"))), default=None)
        return Result[ProductsData](
            status=Status.OK,
            data=ProductsData(products=products),
            source_refs=[SourceRef(table="products", ref=p.product_id) for p in products],
            as_of=_as_utc(as_of),
        )

    # -- I1: get_balance ---------------------------------------------------------------

    def get_balance(self, tool_input: GetBalanceInput) -> Result[BalanceData]:
        """Return a product's balance, credit limit and derived available credit (I1)."""
        row, status = self._load_owned_product("get_balance", tool_input.product_id)
        if status is not None:
            return Result[BalanceData](status=status, message=self._resource_error_message(status))
        assert row is not None
        balance = _f(row, "current_balance")
        credit_limit = _f(row, "credit_limit")
        # `available_credit` is only meaningful for revolving credit; for loans the same
        # subtraction is meaningless (and often negative), so suppress it there (B1). For an
        # over-limit card the raw negative difference is kept and surfaced as-is - it is the
        # truthful over-limit position, not a derived guess.
        is_revolving = _s(row, "product_type") in _REVOLVING_CREDIT_TYPES
        available = (
            credit_limit - balance
            if is_revolving and credit_limit is not None and balance is not None
            else None
        )
        return Result[BalanceData](
            status=Status.OK,
            data=BalanceData(
                product_id=tool_input.product_id,
                currency=str(_s(row, "currency")),
                current_balance=balance,
                credit_limit=credit_limit,
                available_credit=available,
            ),
            source_refs=[SourceRef(table="products", ref=tool_input.product_id)],
            as_of=_as_utc(_dt(row, "last_updated")),
        )

    # -- I4: get_card_details ----------------------------------------------------------

    def get_card_details(self, tool_input: GetCardDetailsInput) -> Result[CardDetailsData]:
        """Return card status, limit, days-past-due and expiry with a masked number (I4)."""
        row, status = self._load_owned_product("get_card_details", tool_input.product_id)
        if status is not None:
            return Result[CardDetailsData](status=status, message=self._resource_error_message(status))
        assert row is not None
        return Result[CardDetailsData](
            status=Status.OK,
            data=CardDetailsData(
                product_id=tool_input.product_id,
                product_type=str(_s(row, "product_type")),
                product_number_masked=mask_product_number(_s(row, "product_number")),
                currency=str(_s(row, "currency")),
                product_status=str(_s(row, "product_status")),
                credit_limit=_f(row, "credit_limit"),
                days_past_due=_i(row, "days_past_due"),
                expiration_date=_dt(row, "expiration_date"),
            ),
            source_refs=[SourceRef(table="products", ref=tool_input.product_id)],
            as_of=_as_utc(_dt(row, "last_updated")),
        )

    # -- I2, I3: search_transactions ---------------------------------------------------

    def search_transactions(self, tool_input: SearchTransactionsInput) -> Result[TransactionsData]:
        """Search the session customer's transactions, newest first, capped at 20 rows (I2/I3)."""
        predicates = [f"customer_id = {sql_str_literal(self.customer_id)}"]

        if tool_input.product_id is not None:
            row, status = self._load_owned_product("search_transactions", tool_input.product_id)
            if status is not None:
                return Result[TransactionsData](status=status, message=self._resource_error_message(status))
            predicates.append(f"product_id = {sql_str_literal(tool_input.product_id)}")

        if tool_input.date_from is not None:
            predicates.append(f"CAST(transaction_date AS DATE) >= {_date_literal(tool_input.date_from)}")
        if tool_input.date_to is not None:
            predicates.append(f"CAST(transaction_date AS DATE) <= {_date_literal(tool_input.date_to)}")
        if tool_input.merchant is not None:
            needle = sql_str_literal(f"%{tool_input.merchant}%")
            predicates.append(f"merchant_name ILIKE {needle}")
        if tool_input.min_amount is not None:
            if not math.isfinite(tool_input.min_amount):
                return Result[TransactionsData](status=Status.INVALID, message="invalid amount range")
            predicates.append(f"TRY_CAST(amount AS DOUBLE) >= {tool_input.min_amount!r}")
        if tool_input.max_amount is not None:
            if not math.isfinite(tool_input.max_amount):
                return Result[TransactionsData](status=Status.INVALID, message="invalid amount range")
            predicates.append(f"TRY_CAST(amount AS DOUBLE) <= {tool_input.max_amount!r}")

        relation = self.source.scan(
            "transactions",
            columns=list(_TRANSACTION_COLUMNS),
            where=" AND ".join(predicates),
        )
        df: pd.DataFrame = relation.order("transaction_date DESC NULLS LAST").limit(tool_input.limit).df()
        rows = df.to_dict("records")
        transactions = [
            TransactionSummary(
                transaction_id=str(_s(r, "transaction_id")),
                transaction_date=_dt(r, "transaction_date"),
                product_id=_s(r, "product_id"),
                amount=_f(r, "amount"),
                currency=_s(r, "currency"),
                transaction_type=_s(r, "transaction_type"),
                transaction_status=_s(r, "transaction_status"),
                merchant_name=_s(r, "merchant_name"),
                transaction_category=_s(r, "transaction_category"),
            )
            for r in rows
        ]
        # NOTE (N2): for this tool `as_of` is the most recent MATCHING transaction's business
        # date, not a data-load/freshness stamp (REQ-26). A customer with no recent activity
        # gets an older `as_of`; the response layer must not render it as a freshness claim.
        # Documented in TOOL_CONTRACTS.md §search_transactions.
        as_of = max((t.transaction_date for t in transactions if t.transaction_date), default=None)
        return Result[TransactionsData](
            status=Status.OK,
            data=TransactionsData(transactions=transactions),
            source_refs=[SourceRef(table="transactions", ref=t.transaction_id) for t in transactions],
            as_of=_as_utc(as_of),
        )

    # -- I5, REQ-28: convert_currency --------------------------------------------------

    def convert_currency(self, tool_input: ConvertCurrencyInput) -> Result[ConversionData]:
        """Convert an amount using `daily_exchange_rates`, citing the rate and its date (I5).

        Uses the rate dated on `on_date`; if none exists, uses the latest prior rate and flags
        `used_prior_rate`; if the latest prior rate is older than 7 days, abstains (REQ-28).
        Currency is never relabelled - the amounts are reported exactly as asked (ADR-016).
        """
        src = tool_input.from_currency.upper()
        dst = tool_input.to_currency.upper()
        if not is_known_currency(src) or not is_known_currency(dst):
            return Result[ConversionData](status=Status.INVALID, message="unknown currency")

        if src == dst:
            return Result[ConversionData](
                status=Status.OK,
                data=ConversionData(
                    original_amount=tool_input.amount,
                    from_currency=src,
                    to_currency=dst,
                    converted_amount=tool_input.amount,
                    rate=1.0,
                    rate_date=tool_input.on_date,
                    requested_date=tool_input.on_date,
                    used_prior_rate=False,
                ),
                message="identity conversion (same currency)",
            )

        earliest = tool_input.on_date - _MAX_RATE_STALENESS
        # Push the ordering and the bound into the engine (like `search_transactions`) so the
        # latest-prior rate is chosen deterministically by construction, not after an unordered
        # fetch (N1). A non-castable `exchange_rate` is excluded in SQL so it cannot win.
        where = (
            f"source_currency = {sql_str_literal(src)} "
            f"AND target_currency = {sql_str_literal(dst)} "
            f"AND CAST(date AS DATE) <= {_date_literal(tool_input.on_date)} "
            f"AND CAST(date AS DATE) >= {_date_literal(earliest)} "
            f"AND TRY_CAST(exchange_rate AS DOUBLE) IS NOT NULL"
        )
        relation = self.source.scan(
            "daily_exchange_rates",
            columns=["CAST(date AS DATE) AS rate_date", "TRY_CAST(exchange_rate AS DOUBLE) AS rate"],
            where=where,
        )
        df = relation.order("rate_date DESC").limit(1).df()
        if df.empty:
            return Result[ConversionData](
                status=Status.UNAVAILABLE,
                message="no exchange rate within 7 days of the requested date",
            )
        best = df.iloc[0]
        rate_date = best["rate_date"]
        rate_date = rate_date.date() if hasattr(rate_date, "date") else rate_date
        rate = float(best["rate"])
        return Result[ConversionData](
            status=Status.OK,
            data=ConversionData(
                original_amount=tool_input.amount,
                from_currency=src,
                to_currency=dst,
                converted_amount=tool_input.amount * rate,
                rate=rate,
                rate_date=rate_date,
                requested_date=tool_input.on_date,
                used_prior_rate=rate_date != tool_input.on_date,
            ),
            source_refs=[SourceRef(table="daily_exchange_rates", ref=f"{src}-{dst}@{rate_date.isoformat()}")],
            as_of=datetime(rate_date.year, rate_date.month, rate_date.day, tzinfo=UTC),
        )

    # -- A1, A2: freeze_card / unfreeze_card (sandbox overlay write, task 2.3) ----------

    def freeze_card(self, tool_input: CardFreezeInput) -> Result[CardActionData]:
        """Freeze a card (A1): confirmed, idempotent overlay write with read-back (REQ-09/14)."""
        return self._card_action("freeze_card", tool_input, CardAction.FREEZE)

    def unfreeze_card(self, tool_input: CardFreezeInput) -> Result[CardActionData]:
        """Unfreeze a card (A2): confirmed, idempotent overlay write with read-back (REQ-09/14)."""
        return self._card_action("unfreeze_card", tool_input, CardAction.UNFREEZE)

    def _card_action(
        self, tool: str, tool_input: CardFreezeInput, action: CardAction
    ) -> Result[CardActionData]:
        """Apply a freeze/unfreeze to the sandbox overlay, then verify it by read-back.

        Steps, each fail-closed (design P3/P4):
        1. Authorize: the product must exist and be owned by the session customer, else
           INVALID/NOT_FOUND/FORBIDDEN (foreign access logged) and nothing is written.
        2. Confirm (REQ-14): a confirmation bound to THIS customer+action+product must exist
           and be unexpired, else INVALID - the action is NOT executed and nothing is disclosed.
        3. Write: set the overlay's `frozen` flag. The write is idempotent, so a double-freeze
           (or double-unfreeze) is a no-op success.
        4. Read back (REQ-09): re-read the overlay and report OK only if the post-condition
           holds; otherwise report the action was NOT completed and offer a human (UNAVAILABLE).
           A read-back post-condition failure AND an overlay I/O error both fail closed the same
           way: the overlay I/O (write + read-back) is wrapped so a raising store (corrupt or
           unwritable `data/_state/`) becomes UNAVAILABLE, not an exception out of the tool.
        """
        row, status = self._load_owned_product(tool, tool_input.product_id)
        if status is not None:
            return Result[CardActionData](status=status, message=self._resource_error_message(status))
        assert row is not None

        now = self._clock()
        confirmation = self.confirmations.get(tool_input.confirmation_id)
        if confirmation is None or not confirmation.is_valid_for(
            customer_id=self.customer_id,
            action=action,
            product_id=tool_input.product_id,
            now=now,
        ):
            # REQ-14: absent/mismatched/expired confirmation => do not execute, disclose nothing.
            return Result[CardActionData](
                status=Status.INVALID,
                message="missing, invalid or expired confirmation; the action was not performed",
            )

        frozen = action is CardAction.FREEZE
        requested_status = _FROZEN_STATUS if frozen else _ACTIVE_STATUS
        try:
            # Idempotent overlay write: writing the same state again is a no-op success.
            self.card_overlay.put(
                CardFreezeState(
                    product_id=tool_input.product_id,
                    customer_id=self.customer_id,
                    frozen=frozen,
                    confirmation_id=tool_input.confirmation_id,
                    updated_at=now,
                )
            )
            # REQ-09: verify before you tell. Read the state back and only report success if the
            # post-condition holds for THIS customer; otherwise the action is not completed.
            readback = self.card_overlay.get(tool_input.product_id)
        except Exception:
            # REQ-09 "or the tool errors" / security.md "on ... tool failure ... offer a human":
            # a raising overlay (corrupt/truncated JSON, unreadable or unwritable `data/_state/`)
            # fails closed exactly like a failed post-condition. Log the failure WITHOUT the
            # overlay contents (ids/state never reach the log line); disclose nothing.
            logger.exception("card overlay I/O failed for tool=%s", tool)
            return Result[CardActionData](
                status=Status.UNAVAILABLE,
                message=_ACTION_NOT_COMPLETED_MESSAGE,
            )

        if readback is None or readback.frozen != frozen or readback.customer_id != self.customer_id:
            return Result[CardActionData](
                status=Status.UNAVAILABLE,
                message=_ACTION_NOT_COMPLETED_MESSAGE,
            )
        return Result[CardActionData](
            status=Status.OK,
            data=CardActionData(
                product_id=tool_input.product_id,
                requested_status=requested_status,
                applied=True,
            ),
            source_refs=[SourceRef(table="card_overlay", ref=tool_input.product_id)],
            as_of=_as_utc(readback.updated_at),
        )

    # -- E1-E4: create_handoff (contract + authZ real; store is a Phase-4 seam) ---------

    def create_handoff(self, tool_input: CreateHandoffInput) -> Result[HandoffData]:
        """Create a handoff case for the session customer (E1-E4). Writes the handoff store."""
        if tool_input.product_id is not None:
            _row, status = self._load_owned_product("create_handoff", tool_input.product_id)
            if status is not None:
                return Result[HandoffData](status=status, message=self._resource_error_message(status))

        if tool_input.transaction_id is not None:
            # REQ-12 (B1): a transaction id arrives from user/model text and is persisted into
            # the package a human agent reads; verify ownership before it crosses that boundary.
            _tx, tx_status = self._load_owned_transaction("create_handoff", tool_input.transaction_id)
            if tx_status is not None:
                return Result[HandoffData](
                    status=tx_status, message=self._resource_error_message(tx_status, "transaction")
                )

        created_at = self._clock()
        package: dict[str, object] = {
            "customer_id": self.customer_id,  # injected from the session, never from input
            "reason": tool_input.reason.value,
            "summary": tool_input.summary,
            "product_id": tool_input.product_id,
            "transaction_id": tool_input.transaction_id,
            "created_at": created_at.isoformat(),
        }
        case_id = self.handoff_store.create(package)
        return Result[HandoffData](
            status=Status.OK,
            data=HandoffData(case_id=case_id, reason=tool_input.reason),
            as_of=_as_utc(created_at),  # N3: normalise like every other tool's as_of
        )
