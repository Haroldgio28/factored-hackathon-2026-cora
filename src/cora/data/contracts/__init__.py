"""CORA data contracts: one Pandera schema per table + a non-raising validate helper.

Task 1.2 (REQ-21: data contracts; REQ-22: quality gates build on them). This package
declares a `pandera.DataFrameSchema` for each of the 13 LATAM Bank tables and a
`validate(table, df)` helper that COLLECTS all failures (never raises) so the quarantine
pipeline (task 1.3) can build on top of the returned failure cases.

Design, grounded in the EDA (analysis/EDA_FINDINGS.md, documentation/reports/data_profile.md):

- Raw-string landing vs canonical-typed (EDA finding F1). The raw landing stores EVERY
  column as a VARCHAR string: booleans are the literal strings 'True'/'False', numerics and
  dates are strings, and some Hive partition columns (`month`/`day`) land as VARCHAR while
  `year` is INT. Each schema declares the CANONICAL dtype each column should have AFTER
  casting and sets `coerce=True`, so Pandera casts the raw strings on validation. Booleans
  are the one coercion hazard — `astype(bool)` maps the string 'False' to True — so they are
  modeled as nullable pandas "boolean" with a coercion-tolerant check (see `_checks.py`).
- Observed Spanish vocabulary for enums (EDA finding F2). Business categories are validated
  against the vocabulary actually observed in the data (`Cuenta Ahorro`, `Tarjeta Crédito`,
  `Transaccional`, `Queja`, `Muy Positivo`, ...), NOT the English in the data dictionary.
  `_checks.SPANISH_TO_CANONICAL` records the mapping to canonical English for later curation.
- No MXN in product/transaction currency (EDA finding F3): those enums are {USD, COP, ARS}.
  FX pairs and complaints.currency DO include MXN, so those columns use {USD, MXN, COP, ARS}.
- Nullability matches the OBSERVED structural pattern (EDA finding F6), not the dictionary's
  NOT NULL: `credit_limit`/`days_past_due` are ~69% null (null off non-credit products),
  `landline_phone` ~50%, `detected_accent` ~30%, `complaints.origin_interaction_id` 100% null.
- Primary keys are declared with `unique=True` even though 0 duplicates are observed
  (EDA finding F5); `daily_exchange_rates` uses a composite PK via the schema-level `unique`.
- Ranges only where justified (EDA finding F6): fraud_score 0-100, credit_score 300-850,
  sentiment_score -1..1, accent_confidence 0..1, avg_csat/response scores 1..5, counts/costs
  >= 0; the fact process window 2023-06-17..2026-06-17 bounds fact `process_date` and the FX
  `date` only (dimension dates predate the fact window and are not bounded).

Pandera is a shipped runtime dependency (not analysis-only) because these contracts are
imported by the pipeline; `pandera==0.20.4` and its `pandas==2.2.3` runtime need are pinned
exactly in `[project].dependencies` (same versions already resolved in `uv.lock`).

Public API: `CONTRACTS_VERSION`, `SCHEMAS`, `get_schema(table)`, `validate(table, df)`.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa

from cora.data.datasource import TABLES

from ._checks import normalize_landing
from .dimensions import DIMENSION_SCHEMAS
from .facts import FACT_SCHEMAS

CONTRACTS_VERSION = "1.0"

#: Registry mapping each table name to its contract. Covers all 13 tables.
SCHEMAS: dict[str, pa.DataFrameSchema] = {**DIMENSION_SCHEMAS, **FACT_SCHEMAS}

# Fail loudly at import time if a schema is missing or extra versus the DataSource registry.
if set(SCHEMAS) != set(TABLES):
    missing = sorted(set(TABLES) - set(SCHEMAS))
    extra = sorted(set(SCHEMAS) - set(TABLES))
    raise RuntimeError(f"contract registry mismatch: missing={missing}, extra={extra}")

# Native column shape of Pandera's SchemaErrors.failure_cases, reused for the empty frame.
_FAILURE_CASE_COLUMNS = ["schema_context", "column", "check", "check_number", "failure_case", "index"]


def get_schema(table: str) -> pa.DataFrameSchema:
    """Return the contract for `table`, or raise `KeyError` if there is none."""
    try:
        return SCHEMAS[table]
    except KeyError as exc:
        raise KeyError(f"no contract for table {table!r}; expected one of {sorted(SCHEMAS)}") from exc


def validate(table: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate `df` against `table`'s contract and RETURN (valid_df, failure_cases).

    The raw-string landing is first normalized (schema-driven) so booleans and nullable
    integers can be coerced (see `_checks.normalize_landing`). Validation then uses Pandera
    `lazy=True` so ALL failures are collected rather than stopping at the first. This never
    raises on a validation failure: on success it returns the coerced DataFrame and an empty
    failure-cases frame; on failure it returns the input rows that did NOT fail plus
    Pandera's native `failure_cases` frame (columns: schema_context, column, check,
    check_number, failure_case, index). Task 1.3 builds quarantine on top of these cases.
    """
    schema = get_schema(table)
    normalized = normalize_landing(schema, df)
    try:
        validated = schema.validate(normalized, lazy=True)
        empty = pd.DataFrame(columns=_FAILURE_CASE_COLUMNS)
        return validated, empty
    except pa.errors.SchemaErrors as err:
        failure_cases = err.failure_cases
        failing_index = pd.Index(failure_cases["index"].dropna().unique())
        valid_df = normalized.drop(index=failing_index, errors="ignore")
        return valid_df, failure_cases


__all__ = ["CONTRACTS_VERSION", "SCHEMAS", "get_schema", "validate"]
