"""Grounding checker: block any generated figure not present in the current turn's tool results
(task 4.3, design section 2, REQ-08).

The LLM only REPHRASES a grounded template draft (`generator.py`); it must never originate a
figure. This module is the deterministic guard that proves it did not: a pure number/entity diff
of the generated text against the TOOL RESULTS OF THE CURRENT TURN. No second model is involved
(design section 2 "deterministic code decides"); `check_grounding` returns a verdict and the
GENERATOR, not the model, acts on it (discard the LLM text, fall back to the grounded template).

What counts as a figure (the only things an LLM could fabricate that mislead a customer):

- amounts / limits / rates: any number token (handles es/pt `1.234,56` and en `1,234.56`),
- percentages: a number followed by `%`,
- dates: ISO `YYYY-MM-DD` and localized `DD/MM/YYYY` / `DD-MM-YYYY`,
- statuses: the enumerated product/transaction/card status words from the dataset.

Every candidate must appear in the flattened set of values taken from this turn's `Result`
objects (`data`, `source_refs`, `as_of`). Numbers are compared NUMERICALLY after normalising the
separator convention, so `1.234,56`, `1,234.56` and `1234.56` all match the stored `1234.56`;
dates are compared as `date` objects so `2026-06-15` matches a stored `15/06/2026`. Statuses are
compared case-insensitively. Any candidate with no match → `ok=False` and the offending tokens
are listed (for the trace, REQ-08). Fail closed: empty tool results + a figure in text → blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from cora.tools.base import Result

__all__ = ["GroundingResult", "check_grounding"]

# Status words a tool `Result` can carry (dataset product/transaction/card enums, `tools.models`)
# mapped to a language-neutral CONCEPT. Templates render in es/pt, but the stored enum values are
# English-ish, so a faithful translation ("Active" -> "activa") must be treated as grounded while a
# CONTRADICTING status ("bloqueada" when the card is Active) must be blocked. Grounding therefore
# compares the status CONCEPT, not the surface word: every surface form here maps to its concept,
# the stored value maps the same way, and the check is "is this concept among the turn's statuses".
_STATUS_CONCEPTS: dict[str, str] = {
    # active
    "active": "active",
    "activa": "active",
    "activo": "active",
    "ativo": "active",
    "ativa": "active",
    # blocked
    "blocked": "blocked",
    "bloqueada": "blocked",
    "bloqueado": "blocked",
    # suspended
    "suspended": "suspended",
    "suspendida": "suspended",
    "suspendido": "suspended",
    "suspensa": "suspended",
    # closed
    "closed": "closed",
    "cerrada": "closed",
    "cerrado": "closed",
    "encerrada": "closed",
    "fechada": "closed",
    # inactive
    "inactive": "inactive",
    "inactiva": "inactive",
    "inativa": "inactive",
    # completed
    "completed": "completed",
    "completada": "completed",
    "concluida": "completed",
    "concluída": "completed",
    # pending
    "pending": "pending",
    "pendiente": "pending",
    "pendente": "pending",
    # failed
    "failed": "failed",
    "fallida": "failed",
    "falhou": "failed",
    "falha": "failed",
    # reversed
    "reversed": "reversed",
    "revertida": "reversed",
    "estornada": "reversed",
    # declined
    "declined": "declined",
    "rechazada": "declined",
    "recusada": "declined",
}


def _status_concept(word: str) -> str | None:
    """The language-neutral status concept a word denotes, or `None` if it is not a status word."""
    return _STATUS_CONCEPTS.get(word.strip().lower())


# A number token: an optional sign, digits with optional grouping/decimal separators. Matches
# `1234`, `1.234,56` (es/pt), `1,234.56` (en), `1234.56`, `0,5`. The surrounding `%` is captured
# separately by `_PCT_RE` only to classify a candidate as a rate for the message; the numeric
# value is checked the same way regardless.
_NUMBER_RE = re.compile(r"(?<![\w.,])[+-]?\d[\d.,]*\d|\b\d\b")

# ISO date `YYYY-MM-DD`.
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# Localized date `DD/MM/YYYY` or `DD-MM-YYYY` (day-first; the LATAM dataset convention).
_LOCAL_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")


@dataclass(frozen=True)
class GroundingResult:
    """The verdict of a grounding check plus the offending tokens for the trace (REQ-08).

    `ok` is True when every figure in the text is backed by a current-turn tool value. `offending`
    lists the ungrounded tokens (numbers/dates/statuses) so the trace records WHY a response was
    blocked - never the model's reasoning, only the concrete mismatch (REQ-39).
    """

    ok: bool
    offending: list[str] = field(default_factory=list)


def _normalise_number(token: str) -> float | None:
    """Parse a number token under either es/pt (`1.234,56`) or en (`1,234.56`) grouping.

    The convention is inferred from which separator appears LAST: that one is the decimal point,
    the other is grouping and is stripped. A token with neither separator parses directly. Returns
    `None` if it is not a number (so a false-positive candidate is simply ignored, not crashed on).
    """
    token = token.strip().lstrip("+")
    if not token or not any(c.isdigit() for c in token):
        return None
    last_dot = token.rfind(".")
    last_comma = token.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        # Both present: the rightmost is the decimal separator, the other is grouping.
        if last_comma > last_dot:
            cleaned = token.replace(".", "").replace(",", ".")
        else:
            cleaned = token.replace(",", "")
    elif last_comma >= 0:
        # Only commas: treat a single trailing group of 1-2 digits as a decimal, else grouping.
        tail = token[last_comma + 1 :]
        cleaned = token.replace(",", ".") if len(tail) in (1, 2) else token.replace(",", "")
    else:
        # Only dots (or none): a single trailing group of 1-2 digits is a decimal, else grouping.
        tail = token[last_dot + 1 :] if last_dot >= 0 else ""
        cleaned = token if len(tail) in (1, 2) or last_dot < 0 else token.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_dates(text: str) -> list[date]:
    """Every ISO and day-first localized date in `text`, as `date` objects (invalid ones dropped)."""
    found: list[date] = []
    for y, m, d in _ISO_DATE_RE.findall(text):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            continue
    for d, m, y in _LOCAL_DATE_RE.findall(text):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            continue
    return found


def _allowed_values(tool_results: list[Result]) -> tuple[set[float], set[date], set[str]]:
    """Flatten this turn's tool results into the sets of grounded numbers, dates and statuses.

    Walks each `Result`'s `data` (a Pydantic model) plus its `source_refs` and `as_of`. A float/int
    becomes an allowed number; a `date`/`datetime` an allowed date; a string recognised as a status
    word an allowed status CONCEPT. This is the whitelist every text figure must match against.
    """
    numbers: set[float] = set()
    dates: set[date] = set()
    statuses: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, bool):
            return  # a bool is not a figure
        if isinstance(value, (int, float)):
            numbers.add(float(value))
        elif isinstance(value, datetime):
            dates.add(value.date())
        elif isinstance(value, date):
            dates.add(value)
        elif isinstance(value, str):
            # A string value may itself contain a figure (e.g. a masked "****1234" tail, or an
            # as_of serialized into a ref). Pull any numbers/dates/statuses out of it too.
            concept = _status_concept(value)
            if concept is not None:
                statuses.add(concept)
            for num in _NUMBER_RE.findall(value):
                parsed = _normalise_number(num)
                if parsed is not None:
                    numbers.add(parsed)
            dates.update(_parse_dates(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)

    for result in tool_results:
        if result.data is not None:
            walk(result.data.model_dump())
        for ref in result.source_refs:
            walk(ref.model_dump())
        if result.as_of is not None:
            dates.add(result.as_of.date())

    return numbers, dates, statuses


def check_grounding(text: str, tool_results: list[Result]) -> GroundingResult:
    """Verify every figure in `text` is present in the current turn's `tool_results` (REQ-08).

    Deterministic: extracts candidate dates, statuses and numbers from `text` and requires each to
    match a value flattened from the tool results. A number matches within a tiny epsilon so float
    formatting does not cause a false block; dates are compared as `date` objects across formats;
    statuses case-insensitively. Any unmatched candidate makes the verdict `ok=False` and is listed
    in `offending`. The generator discards the LLM text and falls back to the grounded template on
    a non-ok verdict - the deterministic code, not the model, makes the block decision.
    """
    allowed_numbers, allowed_dates, allowed_statuses = _allowed_values(tool_results)
    offending: list[str] = []

    # Dates first, so a `2026-06-15` is consumed as a date and its digit runs are not re-checked
    # as bare numbers (which would never match a stored number and cause a false block).
    date_spans: list[tuple[int, int]] = []
    for match in list(_ISO_DATE_RE.finditer(text)) + list(_LOCAL_DATE_RE.finditer(text)):
        date_spans.append(match.span())
    for candidate in _parse_dates(text):
        if candidate not in allowed_dates:
            offending.append(candidate.isoformat())

    # Statuses: any status WORD in the text must denote a status CONCEPT the tools returned this
    # turn. A faithful translation ("Active" -> "activa") shares the concept and passes; a
    # contradicting status ("bloqueada" on an Active card) does not and is blocked.
    for word in re.findall(r"[A-Za-zÀ-ÿ]+", text):
        concept = _status_concept(word)
        if concept is not None and concept not in allowed_statuses:
            offending.append(word)

    # Numbers: every number token not inside a matched date span must be a grounded value.
    for match in _NUMBER_RE.finditer(text):
        start, end = match.span()
        if any(ds <= start and end <= de for ds, de in date_spans):
            continue  # part of a date already checked above
        parsed = _normalise_number(match.group())
        if parsed is None:
            continue
        if not any(abs(parsed - allowed) < 1e-6 for allowed in allowed_numbers):
            offending.append(match.group())

    return GroundingResult(ok=not offending, offending=offending)


if __name__ == "__main__":  # self-check: adversarial figure blocked, grounded figure passes
    from cora.tools.base import SourceRef, Status
    from cora.tools.models import BalanceData

    grounded = Result[BalanceData](
        status=Status.OK,
        data=BalanceData(
            product_id="PRD-1",
            currency="USD",
            current_balance=1234.56,
            credit_limit=None,
            available_credit=None,
        ),
        source_refs=[SourceRef(table="products", ref="PRD-1")],
        as_of=datetime(2026, 6, 15, 10, 0),
    )
    # Grounded: the same balance under es formatting passes.
    assert check_grounding("Tu saldo es 1.234,56 USD al 15/06/2026.", [grounded]).ok
    # Adversarial: an invented balance is blocked and listed.
    bad = check_grounding("Tu saldo es 9.999,99 USD.", [grounded])
    assert not bad.ok and "9.999,99" in bad.offending
    # Adversarial: an invented date is blocked.
    assert not check_grounding("Vence el 2030-01-01.", [grounded]).ok
    # Adversarial: an invented status word is blocked.
    assert not check_grounding("Tu tarjeta esta bloqueada.", [grounded]).ok
    # Fail closed: a figure with no tool results is blocked.
    assert not check_grounding("El total es 500.", []).ok
    print("grounding self-check OK")
