"""Static registries for the quality checks (task 1.3, grounded in the EDA + contracts).

These small maps are the single source for the cross-table relationships, primary keys,
currency allow-sets, dedup ordering columns and the numeric ranges the report needs. They
mirror facts already encoded in the Pandera contracts and the EDA findings; where a bound
also lives on a contract column, `RANGE_CHECKS` is unit-tested to agree with the contract
so the two never drift (see tests/unit/test_quality.py).
"""

from __future__ import annotations

# First column of each table is its PK (per landing_validation.md), except the composite
# daily_exchange_rates key. One source for both the dup check and dedup.
PK_COLUMNS: dict[str, list[str]] = {
    "customers": ["customer_id"],
    "products": ["product_id"],
    "branches": ["branch_id"],
    "service_agents": ["agent_id"],
    "marketing_campaigns": ["campaign_id"],
    "daily_exchange_rates": ["date", "source_currency", "target_currency"],
    "transactions": ["transaction_id"],
    "call_center_interactions": ["interaction_id"],
    "call_transcripts": ["transcript_id"],
    "satisfaction_surveys": ["survey_id"],
    "digital_events": ["event_id"],
    "complaints": ["complaint_id"],
    "campaign_sends": ["send_id"],
}

# Documented, verified-clean relationships (EDA §10.1, all observed 0% orphan). Each tuple
# is (child_column, parent_table, parent_column). Nullable child columns count orphans only
# among non-null values. digital_events is intentionally omitted (24% null customer_id,
# 15.6M rows) and documented as optional in the report.
FK_MAP: dict[str, list[tuple[str, str, str]]] = {
    "products": [("customer_id", "customers", "customer_id")],
    "transactions": [
        ("customer_id", "customers", "customer_id"),
        ("product_id", "products", "product_id"),
    ],
    "call_center_interactions": [
        ("customer_id", "customers", "customer_id"),
        ("agent_id", "service_agents", "agent_id"),  # nullable
    ],
    "call_transcripts": [
        ("interaction_id", "call_center_interactions", "interaction_id"),
    ],
    "complaints": [("customer_id", "customers", "customer_id")],
    "campaign_sends": [
        ("campaign_id", "marketing_campaigns", "campaign_id"),
        ("customer_id", "customers", "customer_id"),
    ],
}

# Per-table currency allow-sets (ADR-016/017): products/transactions have NO MXN (F3);
# complaints DO include MXN; FX pairs cover all four for both source and target.
CURRENCY_SETS: dict[str, dict[str, frozenset[str]]] = {
    "products": {"currency": frozenset({"USD", "COP", "ARS"})},
    "transactions": {"currency": frozenset({"USD", "COP", "ARS"})},
    "complaints": {"currency": frozenset({"USD", "MXN", "COP", "ARS"})},
    "daily_exchange_rates": {
        "source_currency": frozenset({"USD", "MXN", "COP", "ARS"}),
        "target_currency": frozenset({"USD", "MXN", "COP", "ARS"}),
    },
}

# Arrival column used to order dedup "latest wins" (REQ-23). None -> no arrival column, so
# the tie-break is "keep first occurrence in input order" (documented in the result).
ARRIVAL_COLUMN: dict[str, str | None] = {
    "customers": "last_updated",
    "products": "last_updated",
    "branches": None,
    "service_agents": None,
    "marketing_campaigns": None,
    "daily_exchange_rates": None,
    "transactions": "process_date",
    "call_center_interactions": "process_date",
    "call_transcripts": "process_date",
    "satisfaction_surveys": "process_date",
    "digital_events": "process_date",
    "complaints": "process_date",
    "campaign_sends": "process_date",
}

# Numeric ranges the report checks, keyed by (table, column) -> (low, high). These duplicate
# only the few bounds the report needs; a unit test asserts they agree with the contract
# Pandera checks so they never drift.
RANGE_CHECKS: dict[tuple[str, str], tuple[float, float]] = {
    ("customers", "credit_score"): (300, 850),
    ("transactions", "fraud_score"): (0, 100),
    ("service_agents", "avg_csat"): (1, 5),
}
