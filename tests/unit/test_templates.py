"""Tests for task 4.2 customer-facing templates (REQ-18, REQ-36, REQ-40, language steering).

The template table is the grounded, primary renderer. These tests pin the guarantees the
steering makes non-negotiable:

- EVERY outcome exists in BOTH `es` and `pt`; a missing translation fails here (language steering).
- Every policy `Decision` maps to an `Outcome`, so the dispatch table and the copy stay in lockstep.
- Templates carry no figures of their own - only the placeholders a caller fills from tool facts.
- Rendering fills from supplied fields; a missing required field fails loud (never a silent blank).
"""

from __future__ import annotations

import re

import pytest

from cora.agent.templates import (
    SUPPORTED_LANGUAGES,
    Outcome,
    missing_translations,
    render_template,
)
from cora.policy import Decision


def test_no_outcome_is_missing_a_language() -> None:
    """The completeness guard: a missing es/pt copy for any outcome must fail (language steering)."""
    assert missing_translations() == [], f"missing templates: {missing_translations()}"


# The fields each outcome's template declares, so the completeness test can render every one.
_FIELDS: dict[Outcome, dict[str, str]] = {
    Outcome.ANSWER: {"facts": "x"},
    Outcome.CONFIRM: {"action": "congelar", "product_number_masked": "****1234"},
}


@pytest.mark.parametrize("decision", list(Decision))
def test_every_decision_maps_to_a_rendered_outcome(decision: Decision) -> None:
    """Every policy Decision has an Outcome that renders in both languages (dispatch lockstep)."""
    outcome = Outcome.from_decision(decision)
    for lang in SUPPORTED_LANGUAGES:
        text = render_template(outcome, lang, **_FIELDS.get(outcome, {}))
        assert text.strip()


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_templates_declare_no_figures_of_their_own(lang: str) -> None:
    """A template must not hard-code an amount/date/rate/limit - figures come from tool facts only."""
    # Render every non-answer outcome with the fields it declares and assert no stray digits leak
    # from the template text itself (REQ-08: figures originate from tool results, not copy).
    digit_re = re.compile(r"\d")
    assert not digit_re.search(render_template(Outcome.CLARIFY, lang))
    assert not digit_re.search(render_template(Outcome.ABSTAIN, lang))
    assert not digit_re.search(render_template(Outcome.ABSTAIN_ROUTE, lang))
    assert not digit_re.search(render_template(Outcome.REFUSE, lang))
    assert not digit_re.search(render_template(Outcome.ESCALATE, lang))
    assert not digit_re.search(render_template(Outcome.REAUTH, lang))
    assert not digit_re.search(render_template(Outcome.TOOL_UNAVAILABLE, lang))


def test_confirm_restatement_uses_only_supplied_masked_number() -> None:
    """Confirm restates the exact action + the masked number the caller passes; nothing invented."""
    for lang in SUPPORTED_LANGUAGES:
        text = render_template(Outcome.CONFIRM, lang, action="congelar", product_number_masked="****1234")
        assert "****1234" in text
        assert "{" not in text  # every placeholder filled


def test_unknown_language_falls_back_to_spanish() -> None:
    """A live turn in an unsupported language renders es rather than raising (tests enforce es/pt)."""
    assert render_template(Outcome.REFUSE, "en") == render_template(Outcome.REFUSE, "es")


def test_missing_required_field_raises() -> None:
    """A missing placeholder is a loud failure (the generator fails closed on it), never a blank."""
    with pytest.raises(KeyError):
        render_template(Outcome.CONFIRM, "es")  # no action / product_number_masked
