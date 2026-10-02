"""Pandera contracts for the six dimension / reference tables (task 1.2, REQ-21/REQ-22).

Each schema declares the CANONICAL dtype a column should have AFTER coercion from the
raw-string landing (EDA finding F1), with `coerce=True` so Pandera casts on validation.
Enums encode the OBSERVED vocabulary (Spanish where the data uses Spanish, EDA finding
F2); nullability matches the OBSERVED null pattern (EDA finding F6); primary keys are
declared with `unique=True`. Foreign-key relationships are documented in each schema's
`description` only — cross-table referential checks belong to the quality pipeline (1.3).

See `__init__.py` for the full design rationale and EDA citations.
"""

from __future__ import annotations

import pandera as pa

from ._checks import FACT_WINDOW_END, FACT_WINDOW_START, boolean_column, spanish_enum

# --- Observed vocabularies (from documentation/reports/data_profile.md) ---------------

_COUNTRIES = ["México", "Colombia", "Argentina"]  # dict: Mexico/Colombia/Argentina
_ACCENTS = ["mexican", "colombian", "argentine", "neutral"]

CUSTOMERS = pa.DataFrameSchema(
    name="customers",
    description="Customer master. PK customer_id. FK registration_branch_id -> branches.",
    coerce=True,
    columns={
        "customer_id": pa.Column(str, unique=True, nullable=False),
        "document_number": pa.Column(str, unique=True, nullable=False),
        # dict: DNI/ForeignerId/Passport/CitizenId
        "document_type": pa.Column(str, spanish_enum(["DNI", "CE", "Pasaporte", "CC"])),
        "first_name": pa.Column(str, nullable=True),
        "last_name": pa.Column(str, nullable=True),
        "date_of_birth": pa.Column("datetime64[ns]", nullable=True),
        "gender": pa.Column(str, spanish_enum(["M", "F", "O"]), nullable=True),
        "email": pa.Column(str, nullable=True),
        "mobile_phone": pa.Column(str, nullable=True),
        "landline_phone": pa.Column(str, nullable=True),  # ~50% null (F6)
        "address": pa.Column(str, nullable=True),
        "city": pa.Column(str, nullable=False),
        "state": pa.Column(str, nullable=False),
        "country": pa.Column(str, spanish_enum(_COUNTRIES), nullable=False),
        "postal_code": pa.Column(str, nullable=True),  # ~10% null
        "detected_accent": pa.Column(str, spanish_enum(_ACCENTS), nullable=True),  # ~30% null (F6)
        "segment": pa.Column(str, spanish_enum(["Premium", "Plus", "Basic", "Student"]), nullable=False),
        "credit_score": pa.Column(
            "Int64", pa.Check.in_range(300, 850), nullable=True
        ),  # ~15% null; range 300-850
        "estimated_monthly_income": pa.Column(float, nullable=True),  # ~20% null
        "occupation": pa.Column(str, nullable=True),
        "marital_status": pa.Column(str, nullable=True),
        "education_level": pa.Column(str, nullable=True),
        "registration_date": pa.Column("datetime64[ns]", nullable=False),
        "registration_branch_id": pa.Column(str, nullable=False),  # FK branches
        "customer_status": pa.Column(
            str, spanish_enum(["Active", "Inactive", "Suspended", "Closed"]), nullable=False
        ),
        "last_updated": pa.Column("datetime64[ns]", nullable=False),
        "accepts_marketing": boolean_column(nullable=False),
    },
)

PRODUCTS = pa.DataFrameSchema(
    name="products",
    description=(
        "Customer products. PK product_id. FK customer_id -> customers, "
        "opening_branch_id -> branches. credit_limit/days_past_due are structural-null "
        "off credit products (~69% null)."
    ),
    coerce=True,
    columns={
        "product_id": pa.Column(str, unique=True, nullable=False),
        "customer_id": pa.Column(str, nullable=False),  # FK customers
        "product_type": pa.Column(
            str,
            spanish_enum(
                [
                    "Cuenta Ahorro",
                    "Tarjeta Crédito",
                    "Cuenta Corriente",
                    "Tarjeta Débito",
                    "Préstamo Personal",
                    "Préstamo Hipotecario",
                    "Inversión",
                    "Seguro",
                ]
            ),
            nullable=False,
        ),
        "product_number": pa.Column(str, unique=True, nullable=False),
        "currency": pa.Column(str, spanish_enum(["USD", "COP", "ARS"])),  # no MXN (F3)
        "current_balance": pa.Column(float, nullable=True),  # not range-bounded (reversals)
        "credit_limit": pa.Column(float, nullable=True),  # conditional: null off credit (~69%)
        "interest_rate": pa.Column(float, nullable=True),  # ~10% null
        "opening_date": pa.Column("datetime64[ns]", nullable=True),
        "expiration_date": pa.Column("datetime64[ns]", nullable=True),  # ~67% null
        "opening_branch_id": pa.Column(str, nullable=True),  # FK branches
        "product_status": pa.Column(str, spanish_enum(["Active", "Closed", "Blocked", "Suspended"])),
        "opening_channel": pa.Column(
            str, spanish_enum(["Branch", "Web", "App", "Call Center"]), nullable=True
        ),
        "has_linked_app": boolean_column(nullable=True),
        "days_past_due": pa.Column(
            "Int64", pa.Check.ge(0), nullable=True
        ),  # conditional: null off credit (~69%)
        "last_transaction_date": pa.Column("datetime64[ns]", nullable=True),  # ~24% null
        "last_updated": pa.Column("datetime64[ns]", nullable=True),
    },
)

BRANCHES = pa.DataFrameSchema(
    name="branches",
    description="Branch master. PK branch_id. Profiling shows 0 nulls across the table.",
    coerce=True,
    columns={
        "branch_id": pa.Column(str, unique=True, nullable=False),
        "branch_code": pa.Column(str, unique=True, nullable=False),
        "branch_name": pa.Column(str, nullable=False),
        "branch_type": pa.Column(
            str, spanish_enum(["Main", "Express", "Premium", "Corporate"]), nullable=False
        ),
        "address": pa.Column(str, nullable=False),
        "city": pa.Column(str, nullable=False),
        "state": pa.Column(str, nullable=False),
        "country": pa.Column(str, spanish_enum(_COUNTRIES), nullable=False),
        "postal_code": pa.Column(str, nullable=False),
        # dict: Urban/Suburban/Rural - observed Spanish "Urbana" (F2)
        "geographic_zone": pa.Column(str, spanish_enum(["Urbana", "Suburbana", "Rural"]), nullable=False),
        "phone": pa.Column(str, nullable=True),
        "email": pa.Column(str, nullable=True),
        # TIME-of-day stored as string; kept as str (no datetime cast).
        "opening_time": pa.Column(str, nullable=False),
        "closing_time": pa.Column(str, nullable=False),
        "has_atms": boolean_column(nullable=False),
        "atm_count": pa.Column("Int64", pa.Check.ge(0), nullable=False),
        "has_teller_windows": boolean_column(nullable=False),
        "teller_window_count": pa.Column("Int64", pa.Check.ge(0), nullable=False),
        "latitude": pa.Column(float, nullable=True),
        "longitude": pa.Column(float, nullable=True),
        "branch_opening_date": pa.Column("datetime64[ns]", nullable=False),
        "branch_status": pa.Column(
            str, spanish_enum(["Active", "Temporarily Closed", "Closed"]), nullable=False
        ),
    },
)

SERVICE_AGENTS = pa.DataFrameSchema(
    name="service_agents",
    description="Service-agent master. PK agent_id. FK assigned_branch_id -> branches (~31% null).",
    coerce=True,
    columns={
        "agent_id": pa.Column(str, unique=True, nullable=False),
        # Observed NOT unique (1187 distinct / 1200 rows): PK is agent_id, not employee_code.
        "employee_code": pa.Column(str, nullable=False),
        "first_name": pa.Column(str, nullable=False),
        "last_name": pa.Column(str, nullable=False),
        "email": pa.Column(str, nullable=True),
        "phone": pa.Column(str, nullable=True),  # ~6% null
        "native_accent": pa.Column(str, spanish_enum(["mexican", "colombian", "argentine"])),
        "country_of_origin": pa.Column(str, nullable=False),
        "assigned_branch_id": pa.Column(str, nullable=True),  # FK branches, ~31% null
        "agent_type": pa.Column(str, spanish_enum(["Phone", "In-Person", "Digital", "Hybrid"])),
        "experience_level": pa.Column(str, spanish_enum(["Junior", "Mid-Senior", "Senior", "Specialist"])),
        "languages": pa.Column(str, nullable=False),
        "specialty": pa.Column(str, nullable=True),  # ~40% null
        "hire_date": pa.Column("datetime64[ns]", nullable=False),
        "avg_csat": pa.Column(float, pa.Check.in_range(1, 5), nullable=True),  # ~11% null; 1-5
        "total_monthly_interactions": pa.Column("Int64", pa.Check.ge(0), nullable=True),  # ~9% null
        "agent_status": pa.Column(str, spanish_enum(["Active", "Vacation", "Leave", "Inactive"])),
        "work_shift": pa.Column(str, spanish_enum(["Morning", "Afternoon", "Night", "Rotating"])),
    },
)

MARKETING_CAMPAIGNS = pa.DataFrameSchema(
    name="marketing_campaigns",
    description="Marketing-campaign master. PK campaign_id.",
    coerce=True,
    columns={
        "campaign_id": pa.Column(str, unique=True, nullable=False),
        "campaign_name": pa.Column(str, nullable=False),
        "description": pa.Column(str, nullable=True),  # ~20% null
        "campaign_type": pa.Column(str, spanish_enum(["Email", "SMS", "Push", "WhatsApp", "Voice", "Mix"])),
        "campaign_objective": pa.Column(
            str,
            spanish_enum(["Acquisition", "Retention", "Cross-sell", "Up-sell", "Reactivation"]),
        ),
        "promoted_product": pa.Column(str, nullable=True),  # ~11% null
        "target_segment": pa.Column(str, nullable=True),  # ~40% null
        "target_country": pa.Column(str, nullable=True),  # ~56% null
        "start_date": pa.Column("datetime64[ns]", nullable=True),
        "end_date": pa.Column("datetime64[ns]", nullable=True),
        "budget": pa.Column(float, pa.Check.ge(0), nullable=True),  # ~16% null
        "campaign_status": pa.Column(str, spanish_enum(["Planned", "Active", "Paused", "Completed"])),
        "expected_conversion_rate": pa.Column(float, nullable=True),  # ~7% null
    },
)

# Composite PK (date, source_currency, target_currency) via schema-level `unique`.
# FX pairs cover all four currencies, so source/target DO include MXN (unlike products).
DAILY_EXCHANGE_RATES = pa.DataFrameSchema(
    name="daily_exchange_rates",
    description="Daily FX rates. Composite PK (date, source_currency, target_currency).",
    coerce=True,
    unique=["date", "source_currency", "target_currency"],
    columns={
        "date": pa.Column(
            "datetime64[ns]",
            pa.Check.in_range(FACT_WINDOW_START, FACT_WINDOW_END),
            nullable=False,
        ),
        "source_currency": pa.Column(str, spanish_enum(["USD", "MXN", "COP", "ARS"]), nullable=False),
        "target_currency": pa.Column(str, spanish_enum(["USD", "MXN", "COP", "ARS"]), nullable=False),
        "exchange_rate": pa.Column(float, pa.Check.gt(0), nullable=False),
        "buy_rate": pa.Column(float, nullable=True),
        "sell_rate": pa.Column(float, nullable=True),
        "source": pa.Column(str, nullable=True),
    },
)

DIMENSION_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "customers": CUSTOMERS,
    "products": PRODUCTS,
    "branches": BRANCHES,
    "service_agents": SERVICE_AGENTS,
    "marketing_campaigns": MARKETING_CAMPAIGNS,
    "daily_exchange_rates": DAILY_EXCHANGE_RATES,
}
