"""Pandera contracts for the seven fact tables (task 1.2, REQ-21/REQ-22).

Same canonical-typed, `coerce=True` design as the dimensions (see `__init__.py`). Facts
carry the Hive partition columns `year`/`month`/`day` plus a `process_date`; the landing
stores `month`/`day` as VARCHAR for some tables (EDA finding F1), so they are modeled as
coercible `Int64`. Schemas are `strict=False` so extra Hive/partition columns present in a
read do not fail validation. Foreign keys and partition columns are documented in each
schema's `description` only; cross-table referential checks belong to 1.3.
"""

from __future__ import annotations

import pandera as pa

from ._checks import FACT_WINDOW_END, FACT_WINDOW_START, boolean_column, spanish_enum

_PROCESS_DATE = pa.Column(
    "datetime64[ns]",
    pa.Check.in_range(FACT_WINDOW_START, FACT_WINDOW_END),
    nullable=False,
)
# Hive partition columns; month/day land as VARCHAR for some tables (F1), coerced to Int64.
_YEAR = pa.Column("Int64", nullable=True, required=False)
_MONTH = pa.Column("Int64", nullable=True, required=False)
_DAY = pa.Column("Int64", nullable=True, required=False)

TRANSACTIONS = pa.DataFrameSchema(
    name="transactions",
    description=(
        "Transaction fact. PK transaction_id. FK product_id/customer_id, branch_id. "
        "Hive partitions year/month/day; process_date is a column, not the partition key."
    ),
    coerce=True,
    strict=False,
    columns={
        "transaction_id": pa.Column(str, unique=True, nullable=False),
        "transaction_date": pa.Column("datetime64[ns]", nullable=False),
        "process_date": _PROCESS_DATE,
        "product_id": pa.Column(str, nullable=False),  # FK products
        "customer_id": pa.Column(str, nullable=False),  # FK customers
        "transaction_type": pa.Column(
            str,
            spanish_enum(["Deposit", "Withdrawal", "Transfer", "Payment", "Purchase", "Adjustment"]),
        ),
        "transaction_category": pa.Column(
            str,
            spanish_enum(["Food", "Transport", "Services", "Entertainment", "Health", "Other"]),
            nullable=True,
        ),  # ~61% null
        "amount": pa.Column(float, nullable=True),  # not bounded (reversals/adjustments)
        "currency": pa.Column(str, spanish_enum(["USD", "COP", "ARS"])),  # no MXN (F3)
        "amount_usd": pa.Column(float, nullable=True),  # ~57% null
        "channel": pa.Column(str, spanish_enum(["ATM", "Branch", "Web", "App", "POS", "Transfer"])),
        "branch_id": pa.Column(str, nullable=True),  # FK branches, ~69% null
        "merchant_name": pa.Column(str, nullable=True),  # ~77% null
        "merchant_category": pa.Column(str, nullable=True),  # ~77% null
        "transaction_country": pa.Column(str, nullable=True),
        "transaction_city": pa.Column(str, nullable=True),
        "transaction_status": pa.Column(str, spanish_enum(["Approved", "Declined", "Pending", "Reversed"])),
        "response_code": pa.Column(str, nullable=True),
        "is_fraud": boolean_column(nullable=True),
        "fraud_score": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),  # ~20% null; 0-100
        "latitude": pa.Column(float, nullable=True),  # ~81% null
        "longitude": pa.Column(float, nullable=True),  # ~81% null
        "year": _YEAR,
        "month": _MONTH,
        "day": _DAY,
    },
)

CALL_CENTER_INTERACTIONS = pa.DataFrameSchema(
    name="call_center_interactions",
    description=(
        "Call-center interaction fact. PK interaction_id. FK customer_id, agent_id. "
        "reason_category and detected_sentiment use the OBSERVED Spanish vocabulary (F2)."
    ),
    coerce=True,
    strict=False,
    columns={
        "interaction_id": pa.Column(str, unique=True, nullable=False),
        "interaction_date": pa.Column("datetime64[ns]", nullable=False),
        "process_date": _PROCESS_DATE,
        "customer_id": pa.Column(str, nullable=False),  # FK customers
        "agent_id": pa.Column(str, nullable=True),  # FK service_agents
        "interaction_type": pa.Column(
            str, spanish_enum(["Inbound Call", "Outbound Call", "Chat", "Email", "Video"])
        ),
        "channel": pa.Column(str, spanish_enum(["Phone", "Web Chat", "WhatsApp", "Email", "App", "Web"])),
        "contact_reason": pa.Column(str, nullable=True),
        # Observed Spanish vocabulary (profiler left enums:{}; from EDA §2/§10).
        "reason_category": pa.Column(
            str,
            spanish_enum(["Transaccional", "Producto", "Técnico", "Comercial", "Queja", "Retención"]),
        ),
        "duration_seconds": pa.Column(float, pa.Check.ge(0), nullable=True),  # ~14% null
        "wait_time_seconds": pa.Column(float, pa.Check.ge(0), nullable=True),  # ~30% null
        "was_resolved": boolean_column(nullable=True),
        "requires_followup": boolean_column(nullable=True),
        # Observed Spanish sentiment labels (F2): Muy Positivo/Positivo/Neutral/Negativo/Muy Negativo.
        "detected_sentiment": pa.Column(
            str,
            spanish_enum(["Muy Positivo", "Positivo", "Neutral", "Negativo", "Muy Negativo"]),
            nullable=True,
        ),
        "sentiment_score": pa.Column(float, pa.Check.in_range(-1, 1), nullable=True),
        "customer_detected_accent": pa.Column(str, nullable=True),  # ~30% null
        "agent_used_accent": pa.Column(str, nullable=True),  # ~30% null
        "was_escalated": boolean_column(nullable=True),
        "mentioned_products": pa.Column(str, nullable=True),  # ~60% null
        "has_transcript": boolean_column(nullable=True),
        "has_recording": boolean_column(nullable=True),
        "year": _YEAR,
        "month": _MONTH,
        "day": _DAY,
    },
)

CALL_TRANSCRIPTS = pa.DataFrameSchema(
    name="call_transcripts",
    description=(
        "Call-transcript fact. PK transcript_id. FK interaction_id -> call_center_interactions, "
        "customer_id, agent_id. detected_language is 100% 'es' (no Portuguese data)."
    ),
    coerce=True,
    strict=False,
    columns={
        "transcript_id": pa.Column(str, unique=True, nullable=False),
        "interaction_id": pa.Column(str, nullable=False),  # FK call_center_interactions
        "process_date": _PROCESS_DATE,
        "customer_id": pa.Column(str, nullable=True),  # FK customers
        "agent_id": pa.Column(str, nullable=True),  # FK service_agents
        "full_text": pa.Column(str, nullable=True),
        "customer_text": pa.Column(str, nullable=True),
        "agent_text": pa.Column(str, nullable=True),
        "detected_language": pa.Column(str, spanish_enum(["es"])),  # 100% es (F7)
        "detected_accent": pa.Column(str, nullable=True),  # ~37% null
        "accent_confidence": pa.Column(float, pa.Check.in_range(0, 1), nullable=True),  # ~10% null
        "detected_keywords": pa.Column(str, nullable=True),  # ~5% null
        "mentioned_entities": pa.Column(str, nullable=True),  # ~10% null
        "detected_intents": pa.Column(str, nullable=True),  # ~5% null
        "main_topics": pa.Column(str, nullable=True),
        "transcription_model": pa.Column(
            str, spanish_enum(["AWS Transcribe", "Google STT", "Whisper v3", "Azure Speech"])
        ),
        "audio_quality": pa.Column(str, spanish_enum(["High", "Medium", "Low"]), nullable=True),  # ~5% null
        "duration_seconds": pa.Column(float, pa.Check.ge(0), nullable=True),  # ~14% null
        "year": _YEAR,
        "month": _MONTH,
        "day": _DAY,
    },
)

SATISFACTION_SURVEYS = pa.DataFrameSchema(
    name="satisfaction_surveys",
    description=(
        "Satisfaction-survey fact. PK survey_id. FK interaction_id, customer_id, agent_id. "
        "main_score is a mixed scale (1-5 CSAT / 0-10 NPS), left unbounded; no Promoters observed (F10)."
    ),
    coerce=True,
    strict=False,
    columns={
        "survey_id": pa.Column(str, unique=True, nullable=False),
        "survey_date": pa.Column("datetime64[ns]", nullable=False),
        "process_date": _PROCESS_DATE,
        "interaction_id": pa.Column(str, nullable=True),  # FK call_center_interactions
        "customer_id": pa.Column(str, nullable=True),  # FK customers
        "agent_id": pa.Column(str, nullable=True),  # FK service_agents
        "survey_type": pa.Column(str, spanish_enum(["CSAT", "NPS", "CES"])),
        "send_channel": pa.Column(str, spanish_enum(["Email", "SMS", "IVR", "App", "Web"]), nullable=True),
        "main_score": pa.Column("Int64", nullable=True),  # mixed scale, unbounded
        # Promoter kept in the allowed set though none observed (F10); absence is not a violation.
        "nps_category": pa.Column(
            str, spanish_enum(["Promoter", "Passive", "Detractor"]), nullable=True
        ),  # ~72% null
        "question_1_text": pa.Column(str, nullable=True),
        "question_1_response": pa.Column("Int64", pa.Check.in_range(1, 5), nullable=True),  # ~43% null
        "question_2_text": pa.Column(str, nullable=True),  # ~62% null
        "question_2_response": pa.Column("Int64", pa.Check.in_range(1, 5), nullable=True),  # ~62% null
        "question_3_text": pa.Column(str, nullable=True),  # ~81% null
        "question_3_response": pa.Column("Int64", pa.Check.in_range(1, 5), nullable=True),  # ~81% null
        "open_comments": pa.Column(str, nullable=True),  # ~52% null
        "comment_sentiment": pa.Column(
            str, spanish_enum(["Positive", "Neutral", "Negative"]), nullable=True
        ),  # ~52% null
        "response_time_hours": pa.Column(float, pa.Check.ge(0), nullable=True),
        "campaign_response_rate": pa.Column(float, nullable=True),
        "year": _YEAR,
        "month": _MONTH,
        "day": _DAY,
    },
)

DIGITAL_EVENTS = pa.DataFrameSchema(
    name="digital_events",
    description=(
        "Digital-event fact. PK event_id. FK customer_id (~24% null, anonymous events), "
        "product_id (~91% null)."
    ),
    coerce=True,
    strict=False,
    columns={
        "event_id": pa.Column(str, unique=True, nullable=False),
        "event_date": pa.Column("datetime64[ns]", nullable=False),
        "process_date": _PROCESS_DATE,
        "customer_id": pa.Column(str, nullable=True),  # FK customers; ~24% null (anonymous events)
        "session_id": pa.Column(str, nullable=True),
        "event_type": pa.Column(
            str,
            spanish_enum(["PageView", "Click", "FormSubmit", "Login", "Logout", "Error", "Purchase"]),
        ),
        "event_category": pa.Column(
            str, spanish_enum(["Navigation", "Transaction", "Authentication", "Product"])
        ),
        "channel": pa.Column(str, spanish_enum(["Android App", "iOS App", "Desktop Web", "Mobile Web"])),
        "platform": pa.Column(str, nullable=True),
        "browser": pa.Column(str, nullable=True),  # ~63% null
        "app_version": pa.Column(str, nullable=True),
        "page_url": pa.Column(str, nullable=True),
        "page_title": pa.Column(str, nullable=True),
        "action": pa.Column(str, nullable=True),
        "element_id": pa.Column(str, nullable=True),
        "product_id": pa.Column(str, nullable=True),  # FK products, ~91% null
        "event_value": pa.Column(float, nullable=True),  # ~95% null
        "duration_seconds": pa.Column(float, pa.Check.ge(0), nullable=True),  # ~64% null
        "ip_address": pa.Column(str, nullable=True),
        "ip_country": pa.Column(str, nullable=True),
        "ip_city": pa.Column(str, nullable=True),
        "is_mobile": boolean_column(nullable=True),
        "referrer": pa.Column(str, nullable=True),  # ~93% null
        "utm_source": pa.Column(str, nullable=True),  # ~95% null
        "utm_medium": pa.Column(str, nullable=True),  # ~95% null
        "utm_campaign": pa.Column(str, nullable=True),  # ~95% null
        "year": _YEAR,
        "month": _MONTH,
        "day": _DAY,
    },
)

COMPLAINTS = pa.DataFrameSchema(
    name="complaints",
    description=(
        "Complaint fact. PK complaint_id. FK customer_id, affected_product_id, related_branch_id, "
        "assigned_agent_id. origin_interaction_id is declared but 100% null in the data (F6/§10.1): "
        "the complaints->interaction link is never populated."
    ),
    coerce=True,
    strict=False,
    columns={
        "complaint_id": pa.Column(str, unique=True, nullable=False),
        "creation_date": pa.Column("datetime64[ns]", nullable=False),
        "process_date": _PROCESS_DATE,
        "customer_id": pa.Column(str, nullable=False),  # FK customers
        "case_type": pa.Column(str, spanish_enum(["Complaint", "Claim", "Request", "Suggestion"])),
        "category": pa.Column(str, nullable=True),
        "subcategory": pa.Column(str, nullable=True),
        "reception_channel": pa.Column(
            str,
            spanish_enum(["Call Center", "Email", "Web", "App", "Branch", "Regulator"]),
            nullable=True,
        ),
        "affected_product_id": pa.Column(str, nullable=True),  # FK products
        "related_branch_id": pa.Column(str, nullable=True),  # FK branches, ~72% null
        "origin_interaction_id": pa.Column(str, nullable=True),  # 100% null (F6/§10.1)
        "description": pa.Column(str, nullable=True),
        "claimed_amount": pa.Column(float, nullable=True),
        "currency": pa.Column(str, spanish_enum(["USD", "MXN", "COP", "ARS"]), nullable=True),
        "priority": pa.Column(str, spanish_enum(["Low", "Medium", "High", "Critical"]), nullable=True),
        "status": pa.Column(
            str,
            spanish_enum(["Open", "In Process", "Escalated", "Resolved", "Closed", "Rejected"]),
        ),
        "assigned_agent_id": pa.Column(str, nullable=True),  # FK service_agents
        "assignment_date": pa.Column("datetime64[ns]", nullable=True),
        "first_response_date": pa.Column("datetime64[ns]", nullable=True),
        "resolution_date": pa.Column("datetime64[ns]", nullable=True),  # ~77% null
        "closing_date": pa.Column("datetime64[ns]", nullable=True),  # ~96% null
        "sla_breached": boolean_column(nullable=True),
        "resolution_days": pa.Column("Int64", pa.Check.ge(0), nullable=True),  # ~76% null
        "resolution": pa.Column(str, nullable=True),  # ~77% null
        "compensation_granted": pa.Column(float, nullable=True),  # ~93% null
        "resolution_satisfaction": pa.Column("Int64", pa.Check.in_range(1, 5), nullable=True),  # ~96% null
        "is_repeat_complainer": boolean_column(nullable=True),
        "year": _YEAR,
        "month": _MONTH,
        "day": _DAY,
    },
)

CAMPAIGN_SENDS = pa.DataFrameSchema(
    name="campaign_sends",
    description=(
        "Campaign-send fact. PK send_id. FK campaign_id -> marketing_campaigns, customer_id -> customers."
    ),
    coerce=True,
    strict=False,
    columns={
        "send_id": pa.Column(str, unique=True, nullable=False),
        "send_date": pa.Column("datetime64[ns]", nullable=False),
        "process_date": _PROCESS_DATE,
        "campaign_id": pa.Column(str, nullable=False),  # FK marketing_campaigns
        "customer_id": pa.Column(str, nullable=False),  # FK customers
        "send_channel": pa.Column(str, spanish_enum(["Email", "SMS", "Push", "WhatsApp", "Voice"])),
        "template_used": pa.Column(str, nullable=True),
        "subject": pa.Column(str, nullable=True),  # ~84% null
        "send_status": pa.Column(str, spanish_enum(["Sent", "Failed", "Bounced", "Blocked"])),
        "was_delivered": boolean_column(nullable=True),
        "was_opened": boolean_column(nullable=True),
        "open_date": pa.Column("datetime64[ns]", nullable=True),
        "was_clicked": boolean_column(nullable=True),
        "click_date": pa.Column("datetime64[ns]", nullable=True),  # ~94% null
        "click_count": pa.Column("Int64", pa.Check.ge(0), nullable=True),  # ~94% null
        "had_conversion": boolean_column(nullable=True),
        "conversion_date": pa.Column("datetime64[ns]", nullable=True),  # ~99% null
        "conversion_value": pa.Column(float, nullable=True),  # ~99% null
        "open_device": pa.Column(str, nullable=True),  # ~73% null
        "open_country": pa.Column(str, nullable=True),  # ~73% null
        "failure_reason": pa.Column(str, nullable=True),  # ~94% null
        "send_cost": pa.Column(float, pa.Check.ge(0), nullable=True),
        "year": _YEAR,
        "month": _MONTH,
        "day": _DAY,
    },
)

FACT_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "transactions": TRANSACTIONS,
    "call_center_interactions": CALL_CENTER_INTERACTIONS,
    "call_transcripts": CALL_TRANSCRIPTS,
    "satisfaction_surveys": SATISFACTION_SURVEYS,
    "digital_events": DIGITAL_EVENTS,
    "complaints": COMPLAINTS,
    "campaign_sends": CAMPAIGN_SENDS,
}
