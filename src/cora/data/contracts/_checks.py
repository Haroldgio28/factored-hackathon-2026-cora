"""Reusable Pandera primitives for the CORA data contracts (task 1.2, REQ-21/REQ-22).

These helpers keep the 13 table schemas uniform and encode two cross-cutting facts
established by the EDA (see the package docstring in `__init__.py`):

- Enums carry the OBSERVED vocabulary, which is Spanish for most business categories
  (EDA finding F2: `Cuenta Ahorro`, `Tarjeta Crédito`, `Transaccional`, `Queja`, ...),
  not the English listed in the data dictionary. `SPANISH_TO_CANONICAL` is a
  reference-only mapping to canonical English codes; task 1.2 performs no transformation.
- Raw landing stores every column as a VARCHAR string (EDA finding F1). Booleans are the
  literal strings `'True'`/`'False'`. Pandas `astype(bool)` would map the non-empty string
  `'False'` to `True`, so booleans must NOT be coerced with bare `bool`; `boolean_column`
  models them as a nullable pandas `"boolean"` column with a coercion-tolerant check.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pandera as pa

# Dates in the landed history span 2018-06-18..2026-06-17 for dimensions (registration /
# opening dates predate the fact window) and 2023-06-17..2026-06-17 for the facts. The
# fact process window is used only as a lenient documented bound on fact `process_date`
# and `daily_exchange_rates.date` (EDA finding F6/§9).
FACT_WINDOW_START = pd.Timestamp("2023-06-17")
FACT_WINDOW_END = pd.Timestamp("2026-06-17")

# Reference-only mapping from the observed Spanish vocabulary to canonical English codes.
# Task 1.2 does NOT transform values; this documents the intended normalization for later
# curation (EDA finding F2). Only the business categories that differ from English appear.
SPANISH_TO_CANONICAL: dict[str, dict[str, str]] = {
    "product_type": {
        "Cuenta Ahorro": "SavingsAccount",
        "Cuenta Corriente": "CheckingAccount",
        "Tarjeta Crédito": "CreditCard",
        "Tarjeta Débito": "DebitCard",
        "Préstamo Personal": "PersonalLoan",
        "Préstamo Hipotecario": "MortgageLoan",
        "Inversión": "Investment",
        "Seguro": "Insurance",
    },
    "country": {
        "México": "Mexico",
        "Colombia": "Colombia",
        "Argentina": "Argentina",
    },
    "document_type": {
        "DNI": "DNI",
        "CE": "ForeignerId",
        "Pasaporte": "Passport",
        "CC": "CitizenId",
    },
    "reason_category": {
        "Transaccional": "Transactional",
        "Producto": "Product",
        "Técnico": "Technical",
        "Comercial": "Commercial",
        "Queja": "Complaint",
        "Retención": "Retention",
    },
    "detected_sentiment": {
        "Muy Positivo": "VeryPositive",
        "Positivo": "Positive",
        "Neutral": "Neutral",
        "Negativo": "Negative",
        "Muy Negativo": "VeryNegative",
    },
    "geographic_zone": {
        "Urbana": "Urban",
        "Suburbana": "Suburban",
        "Rural": "Rural",
    },
}


def spanish_enum(values: Sequence[str], **kwargs: object) -> pa.Check:
    """Return an `isin` check over the OBSERVED vocabulary for a categorical column.

    Keeps enum definitions uniform and self-documenting across the 13 schemas. The
    vocabulary is what the EDA observed in the landing, not the dictionary's English
    (EDA finding F2); where they differ, `SPANISH_TO_CANONICAL` records the mapping.
    """
    return pa.Check.isin(list(values), **kwargs)


_BOOL_LANDING_MAP: dict[str, bool] = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
}


def _to_landing_bool(value: object) -> object:
    """Map a raw-string landing boolean to a real bool; pass through null/bool/unknown."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or isinstance(value, bool):
        return value
    return _BOOL_LANDING_MAP.get(str(value).strip().lower(), value)


def normalize_landing(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    """Pre-cast the raw-string landing frame so schema coercion can succeed.

    Pandera coerces BEFORE parsers run, and the landing stores booleans as the strings
    `'True'`/`'False'` and nullable integers as float-strings like `'701.0'` (EDA finding
    F1) — neither of which `astype('boolean')` / `astype('Int64')` can parse directly. This
    helper is schema-DRIVEN (it reads each column's declared dtype, so it never duplicates
    the per-table definitions): boolean columns get string->bool mapping, nullable-integer
    columns get `to_numeric` (float path) before the integer cast. Every other dtype
    (str, float, datetime) coerces from the raw strings on its own, so it is left untouched.
    Unknown boolean-like tokens are left as-is so the column's `landing_boolean` check still
    reports them as failures rather than silently dropping the value.
    """
    out = df.copy()
    for name, column in schema.columns.items():
        if name not in out.columns:
            continue
        dtype = str(column.dtype)
        if dtype == "boolean":
            out[name] = out[name].map(_to_landing_bool)
        elif dtype in ("Int64", "int64"):
            out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


def _boolean_landing_check(series: pd.Series) -> pd.Series:
    """Element-wise check that a value is a landing-shaped boolean.

    Accepts real booleans and the raw-string forms the landing uses (`'True'`/`'False'`,
    plus a few tolerant spellings). Nulls are reported as passing here; nullability is
    enforced separately by the column's `nullable` flag.
    """

    def _ok(value: object) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return True
        if isinstance(value, bool):
            return True
        return str(value).strip().lower() in _BOOL_LANDING_MAP

    return series.map(_ok)


def boolean_column(*, nullable: bool = True, **kwargs: object) -> pa.Column:
    """A boolean column that tolerates the raw-string landing shape.

    Modeled as a pandas nullable `"boolean"` column with a coercion-tolerant element
    check instead of a bare `bool` coercion, because `astype(bool)` turns the string
    `'False'` into `True` (EDA finding F1). With `coerce=True` on the schema, Pandera
    casts the accepted string forms to the `"boolean"` dtype.
    """
    return pa.Column(
        "boolean",
        checks=pa.Check(_boolean_landing_check, element_wise=False, name="landing_boolean"),
        nullable=nullable,
        **kwargs,
    )
