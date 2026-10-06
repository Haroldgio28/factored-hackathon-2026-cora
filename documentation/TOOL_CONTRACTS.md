# CORA Tool Contracts (task 2.2)

Implements design §6 and REQ-12, REQ-28, REQ-34, REQ-36, REQ-38. Source:
`src/cora/tools/` (`base.py`, `models.py`, `layer.py`, `handoff_store.py`).

## Cross-cutting contract

Every tool is a method of `ToolLayer`, which is constructed from a **verified** `Session`
(task 2.1) and a read-only `DataSource` (task 1.1). The session's `customer_id` is captured
at construction and is the ONLY source of customer identity. No tool input model carries a
`customer_id`; the input models use `extra="forbid"`, so an attempt to smuggle one in is
rejected by validation (REQ-12, design P1). Neither the LLM nor user text can widen access.

All tools return the uniform envelope `Result{status, data, source_refs, as_of, message}`:

| Field | Meaning |
|---|---|
| `status` | `OK \| NOT_FOUND \| FORBIDDEN \| UNAVAILABLE \| INVALID` |
| `data` | typed payload on `OK`; **`None` on every non-OK status** (fail closed, disclose nothing — design P4) |
| `source_refs` | list of `{table, ref}` pointing at the records a value came from (traceability, design P5) |
| `as_of` | freshness timestamp of the data returned (REQ-26) |
| `message` | short, non-sensitive explanation; never contains PII |

Shared behaviours:

- **Authorization.** A product-scoped request first loads the product and checks ownership.
  A product that does not exist → `NOT_FOUND`. A product owned by **another** customer →
  `FORBIDDEN`, and the attempt is recorded in `AccessLog` and logged to the `cora.tools.authz`
  logger (REQ-12). A syntactically unsafe id → `INVALID` (the id never reaches SQL).
  - **Phase-4 note (N4):** the `FORBIDDEN`-vs-`NOT_FOUND` distinction exists so the layer can
    log a foreign-access attempt; it must **not** leak to the customer. If the two statuses ever
    map to different customer-visible wording a caller could enumerate which ids exist, so the
    Phase-4 response templates must collapse `FORBIDDEN` and `NOT_FOUND` into one neutral
    message.
- **PII masking.** Product/card numbers are masked to their last four characters (`****3827`)
  before leaving the layer (REQ-36). No tool returns document numbers, emails, phones or
  addresses.
- **Bounded reads.** All reads go through the `DataSource` with explicit `limit`s or a bounded
  window; raw-landing VARCHAR columns are projected through `TRY_CAST` so a garbage string
  becomes `NULL` rather than raising.
- **Injection safety.** Caller-supplied strings (a product id, a merchant substring) are
  validated as identifiers or escaped as SQL string literals before any interpolation; all
  user/tool text is treated strictly as data.
- **No money movement (REQ-34).** The layer exposes no tool that moves money or changes a
  balance. A transfer/payment request has nothing to call here; policy refuses it (POL-020).
  Enforced two ways: `test_no_money_movement_tool_exists` guards the `TOOL_NAMES` registry by
  name, and `test_datasource_exposes_no_write_methods` asserts the underlying `DataSource` has
  no mutating method at all, so no tool could change a balance even if misnamed.
- **Session lifecycle.** `ToolLayer` holds the verified `Session` (which carries `expires_at`)
  and an injectable clock, but it does **not** itself re-check expiry on each call; session
  lifecycle enforcement (reject an expired/tampered session, fail closed) belongs to the
  task-2.3 session layer that sits in front of the tools. Within 2.2 a `ToolLayer` is assumed
  to be constructed per turn from a freshly verified session.

## Tools

### `list_products` — I6
- **Input:** `ListProductsInput` (no fields).
- **Output:** `ProductsData{products: [ProductSummary{product_id, product_type,
  product_number_masked, currency, product_status}]}`.
- **Errors:** `OK` always for a valid session (an empty list if the customer has no products).
- **Side effects:** none.
- **Latency:** one indexed-free scan of the `products` dimension filtered by `customer_id`,
  capped at 100 rows. Local DuckDB over a single Parquet file: single-digit ms.
- **Limitations vs a real API:** no pagination/cursor; a real core-banking `GET /products`
  would page and expose far more product metadata and relationship links.
- **Overlay freeze flag not reflected (F2).** `product_status` here is the read-only source
  value; the task-2.3 card-freeze **overlay** is **not** merged into this read, so a card
  frozen via `freeze_card` still lists as `Active`. Whether a read tool should merge overlay
  state over `product_status` changes what the customer sees and is a product decision deferred
  to Phase 4 (agent wiring); it must be raised explicitly with the owner there.

### `get_balance` — I1
- **Input:** `GetBalanceInput{product_id}` (any owned product, not only cards).
- **Output:** `BalanceData{product_id, currency, current_balance, credit_limit,
  available_credit}`.
- **`available_credit` derivation (B1).** `available_credit = credit_limit - current_balance`
  is reported **only for revolving-credit products** (`product_type == "Tarjeta Crédito"`); for
  every other product type it is `None`, even when a `credit_limit` is present.
  - Rationale: in this dataset three families carry a `credit_limit` — `Tarjeta Crédito`,
    `Préstamo Personal`, `Préstamo Hipotecario` — but only the card is a revolving line. The
    derivation rests on the assumption that **`current_balance` is outstanding debt drawn
    against the limit**, which holds for a revolving card but not for a loan, where
    `credit_limit` is the original principal. Applying it to loans produces a meaningless and
    frequently negative figure.
  - Measured over-limit share of `products` (rows where `current_balance > credit_limit`, full
    table): `Préstamo Hipotecario` 5,854/11,910 = **49.2%**, `Préstamo Personal` 453/19,960 =
    **2.3%**, `Tarjeta Crédito` 1,203/100,102 = **1.2%**. The loan rates are why the derivation
    is suppressed there.
  - **Over-limit cards:** for the 1.2% of revolving cards whose balance exceeds the limit,
    `available_credit` is the **negative** difference, surfaced as-is (not floored to `0.0`). A
    negative value is the truthful over-limit position; the response layer should present it as
    "over limit", not as spendable credit.
- **Errors:** `INVALID` (bad id), `NOT_FOUND`, `FORBIDDEN` (foreign product, logged).
- **Side effects:** none.
- **Latency:** one `products` point-lookup by `product_id`, limit 1. Single-digit ms locally.
- **Limitations vs a real API:** balance is the stored daily value, not a real-time ledger
  position; no holds/pending authorizations, no available-vs-posted split. `available_credit`
  is derived from the stored balance and limit, not read from a credit-line system. Currency is
  as recorded on the product and never relabelled (ADR-016: Mexican products are USD).

### `search_transactions` — I2, I3
- **Input:** `SearchTransactionsInput{product_id?, date_from?, date_to?, merchant?,
  min_amount?, max_amount?, limit=20}`. `limit` is validated to `1..20` by the contract.
- **Output:** `TransactionsData{transactions: [TransactionSummary{transaction_id,
  transaction_date, product_id, amount, currency, transaction_type, transaction_status,
  merchant_name, transaction_category}]}`, newest first.
- **Errors:** `FORBIDDEN` when `product_id` is a foreign product (logged); `INVALID` on a
  non-finite amount bound. Always scoped to the session customer.
- **Side effects:** none.
- **Latency:** scans the Hive-partitioned `transactions` fact filtered by `customer_id` (+
  optional predicates), ordered by date, materialising at most `limit` rows. A date range
  narrows the partitions read. Without a date range the engine streams all partitions and
  short-circuits at the top-N; this is the slowest read in the layer (sub-second to a few
  seconds on the full local dataset) and is the main candidate for a partition/date default
  in production.
- **`as_of` semantics (N2):** for this tool `as_of` is the most recent **matching
  transaction's** business date, not a data-load/freshness stamp (REQ-26). A customer with no
  recent activity gets an older `as_of`; the response layer must not render it as a freshness
  claim. (Data-load freshness is `process_date`, a separate quantity this tool does not return.)
- **Limitations vs a real API:** `limit ≤ 20` by rule; no free cursor pagination; merchant
  match is a substring `ILIKE`, not a normalized merchant id; no running balance per line.

### `get_card_details` — I4
- **Input:** `GetCardDetailsInput{product_id}`.
- **Output:** `CardDetailsData{product_id, product_type, product_number_masked, currency,
  product_status, credit_limit, days_past_due, expiration_date}`.
- **Errors:** `INVALID`, `NOT_FOUND`, `FORBIDDEN` (logged).
- **Side effects:** none.
- **Latency:** one `products` point-lookup, limit 1. Single-digit ms locally.
- **Limitations vs a real API:** the dataset has no native card freeze/limit-change fields;
  "status" is `product_status` (`Active/Blocked/Suspended/Closed`). The tool returns details
  for any product id, not only cards; the caller/policy decides card-ness from `product_type`.
- **Overlay freeze flag not reflected (F2).** The task-2.3 card-freeze **overlay** is **not**
  merged into `product_status` here, so after a confirmed `freeze_card` this read still reports
  the source status (`Active`), and unfreezing a natively `Blocked` card still reads `Blocked`.
  Per-turn grounding is not violated (the value comes from a tool result), but the agent could
  confirm a freeze in one turn and contradict it in the next. Merging the overlay over
  `product_status` changes what the customer sees and is a product decision deferred to Phase 4;
  it must be raised explicitly with the owner when the agent is wired.

### `convert_currency` — I5, REQ-28
- **Input:** `ConvertCurrencyInput{amount≥0, from_currency, to_currency, on_date}`.
- **Output:** `ConversionData{original_amount, from_currency, to_currency, converted_amount,
  rate, rate_date, requested_date, used_prior_rate}`.
- **Rate rules (REQ-28, ADR-016):**
  - Uses the `daily_exchange_rates` row for `on_date` when present (`used_prior_rate=False`).
  - If none exists, uses the **latest prior** rate within 7 days and sets `used_prior_rate=True`
    with `rate_date` < `requested_date` — the caller MUST state this to the customer.
  - If the latest prior rate is **older than 7 days**, abstains → `UNAVAILABLE`.
  - Same-currency request is an identity conversion (`rate=1.0`).
  - Currency is never relabelled; amounts are reported exactly as asked.
- **Errors:** `INVALID` (currency not in `{USD, MXN, COP, ARS}`), `UNAVAILABLE` (no rate within
  7 days).
- **Side effects:** none.
- **Latency:** one bounded read of `daily_exchange_rates` over a ≤8-day window, limit 64;
  single-digit ms locally.
- **Limitations vs a real API:** uses the official daily rate only (no intraday/bid-ask spread
  beyond the stored `buy_rate`/`sell_rate`, which this tool does not apply); no fee modelling.

### `freeze_card` / `unfreeze_card` — A1, A2 (sandbox overlay write, task 2.3)
- **Input:** `CardFreezeInput{product_id, confirmation_id}`. `confirmation_id` is required
  (`min_length=1`) and must be a confirmation issued for **this** customer, action and product
  (the hook for the task-4.4 confirmation protocol).
- **Output:** `CardActionData{product_id, requested_status, applied}` — `requested_status` is
  `Blocked` for a freeze and `Active` for an unfreeze; `applied` is `True` only after the
  read-back confirms the overlay state.
- **Behaviour (REQ-09, REQ-14):**
  1. **Authorize** — the product must exist and be owned by the session customer, else
     `INVALID`/`NOT_FOUND`/`FORBIDDEN` (foreign access logged); nothing is written.
  2. **Confirm** — a confirmation bound to this `(customer_id, action, product_id)` must exist
     and be unexpired; otherwise `INVALID` and the action is **not executed** (REQ-14).
  3. **Write** — set the overlay's `frozen` flag (`freeze` → `True`, `unfreeze` → `False`). The
     write is **idempotent**: a repeated freeze (or unfreeze) is a no-op success.
  4. **Read-back (REQ-09)** — the overlay is re-read; `OK` is returned only if the post-condition
     holds for this customer. If the read-back fails or the overlay errors, the tool returns
     `UNAVAILABLE` with a message that the action was **not completed** and offers a human
     (fail closed; nothing is claimed done — design P3/P4).
- **Errors:** `INVALID` (unsafe id, or missing/invalid/expired confirmation), `NOT_FOUND`
  (unknown product), `FORBIDDEN` (foreign card, logged), `UNAVAILABLE` (post-condition not
  verified).
- **Side effects:** writes the reversible card-freeze **overlay** (`CardOverlay`). The default
  `InMemoryCardOverlay` is the test/demo seam; `JsonCardOverlay` persists to the gitignored
  runtime file `data/_state/card_overlay.json` (atomic tmp-file + `os.replace`, consistent with
  the pipeline's watermark/freshness state). Source/curated product data is never mutated.
- **Latency:** one `products` point-lookup for authorization, one confirmation lookup, and a
  small overlay read/write/read-back; negligible locally.
- **Limitations vs a real API:** the dataset has **no native card-freeze field** (`product_status`
  ∈ `{Active, Closed, Blocked, Suspended}`), so freeze is modelled as a reversible **overlay**
  flag, not a call to a card-management system; `requested_status` is a display label for the
  action result, not a write to `product_status`. Single-use consumption of a confirmation (vs
  the TTL-bounded, action/target-bound validity enforced here) lands with the confirmation
  protocol in **task 4.4**.

### `create_handoff` — E1-E4
- **Input:** `CreateHandoffInput{reason: E1|E2|E3|E4, summary, product_id?, transaction_id?}`.
- **Output:** `HandoffData{case_id, reason}`.
- **Reason set:** the `HandoffReason` enum carries **E1-E4** (dispute/complaint/fraud/human
  request) as the intent-mapped reasons this tool accepts, plus **E5 `NON_CUSTOMER`** — a
  deterministic, orchestrator-only reason set by the non-customer branch when the verified
  `customer_id` has no row in the bank's `customers` records (deliverable 2). E5 is **never a
  valid `create_handoff` input**: a non-customer handoff is persisted by the orchestrator's shared
  `_persist_handoff` path (`build_package`+store) with reason E5 and `normal` priority, carrying
  empty verified facts (no account data). The model/user can never request an E5 handoff.
- **Behaviour:** the handoff package is built with the `customer_id` **injected from the
  session** (never from input). Any referenced resource is authorized before it is persisted:
  if `product_id` is given it must be owned, and if `transaction_id` is given it must be owned
  (`FORBIDDEN` + logged access attempt otherwise). Only then is the package persisted via a
  `HandoffStore` seam and an opaque `CASE-...` id returned.
- **Errors:** `FORBIDDEN` on a foreign referenced product **or transaction** (logged as an
  access attempt, REQ-12); `NOT_FOUND` on an unknown referenced product/transaction; `INVALID`
  on a syntactically unsafe id. On any non-OK status nothing is written to the store.
- **Side effects:** writes a handoff record to the store **only** after every referenced
  resource is authorized. The default `InMemoryHandoffStore` is a Phase-4 seam (the durable
  queue the agent console reads is task 4.5); the **contract and authorization are real now**,
  only persistence is swapped later.
- **Latency:** up to two `products`/`transactions` point lookups for authorization, then an
  in-memory write; negligible locally.
- **Limitations vs a real API:** no durable queue, SLA timer, assignment or notification yet.
  The package redaction policy for agent display belongs to the task-4.5 builder.

## Evidence

- Tests: `tests/unit/test_tools.py` — 38 cases: per-tool happy paths; `get_balance` of a
  **loan** (no `available_credit`) and of an **over-limit card** (negative `available_credit`,
  B1); `FORBIDDEN` on a foreign resource (+ access-attempt logging) for both products **and** a
  handoff's referenced transaction (with the owned-transaction and unknown-transaction paths);
  `convert_currency` exact-date / latest-prior-within-7-days / exactly-7-days boundary / 7-day
  abstain / identity / unknown-currency; `limit ≤ 20` cap; masking; input rejection of a
  `customer_id` field; the no-money-movement registry check and a direct `DataSource`
  write-absence assertion.
- Verification run: `.agents/tasks/phase2-task-2.2-verification.md`.
