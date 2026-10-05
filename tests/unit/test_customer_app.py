"""Smoke + pure-helper tests for task 5.4 (the customer chat view is logic-free, ponytail).

The view is a thin urllib client over the 5.3 API with no business logic, so it needs no behavioural
test beyond proving it imports without pulling Streamlit at module load (imported lazily inside
`main()`), that its UI copy is complete in es AND pt, and that the pure helpers parse the `/chat`
contract and format the trace line. No network: the parse helper is exercised with a sample dict.
"""

from __future__ import annotations

import pytest


def test_customer_app_imports() -> None:
    import ui.customer_app as app

    assert hasattr(app, "main")
    assert hasattr(app, "_parse_chat_response")
    assert hasattr(app, "ui_text")


def test_ui_copy_complete_in_both_languages() -> None:
    import ui.customer_app as app

    es_keys = set(app.UI_COPY["es"])
    pt_keys = set(app.UI_COPY["pt"])
    assert es_keys == pt_keys, "es and pt UI copy must carry the same keys"
    assert set(app.LANGUAGES) <= set(app.UI_COPY)


def test_missing_translation_fails() -> None:
    import ui.customer_app as app

    # A key present in es but not pt (or vice versa) must be detectable as a failure, the way the
    # completeness test above would flag a real omission.
    assert all(key in app.UI_COPY["pt"] for key in app.UI_COPY["es"])
    with pytest.raises(KeyError):
        app.ui_text("pt", "a_key_that_does_not_exist")


def test_parse_chat_response_extracts_text_and_trace_id() -> None:
    import ui.customer_app as app

    payload = {
        "response_text": "Tu saldo es 100 USD.",
        "trace_id": "abc123def456",
        "outcome": "RESOLVED",
        "decision": "Answer",
        "rule_id": None,
        "handoff_case_id": None,
    }
    text, trace_id = app._parse_chat_response(payload)
    assert text == "Tu saldo es 100 USD."
    assert trace_id == "abc123def456"


def test_parse_chat_response_degrades_on_missing_fields() -> None:
    import ui.customer_app as app

    text, trace_id = app._parse_chat_response({})
    assert text == ""
    assert trace_id == ""


def test_trace_label_formats_and_handles_empty() -> None:
    import ui.customer_app as app

    assert app.trace_label("abc123") == "trace: abc123"
    assert "none" in app.trace_label("")
    assert "none" in app.trace_label(None)


def test_language_toggle_default_is_spanish() -> None:
    import ui.customer_app as app

    assert app.DEFAULT_LANGUAGE == "es"
    assert app.DEFAULT_LANGUAGE in app.LANGUAGES


def test_chat_body_includes_the_explicit_language() -> None:
    import ui.customer_app as app

    # The request body carries the explicit UI language so the server runs the turn in it; identity
    # still travels only as the token (no customer_id field).
    body = app._chat_body("tok-123", "quiero mi saldo", "pt")
    assert body == {"token": "tok-123", "utterance": "quiero mi saldo", "language": "pt"}


def test_set_page_config_is_the_first_streamlit_command() -> None:
    # Streamlit raises StreamlitSetPageConfigMustBeFirstCommandError if any `st.` command runs
    # before `st.set_page_config`. The smoke test can't catch it (it never runs the Streamlit
    # flow), so assert it statically on main()'s source: the first `st.` call must be
    # `st.set_page_config`. Regression guard for the real-run ordering bug found in the demo.
    import inspect

    import ui.customer_app as app

    body = inspect.getsource(app.main)
    first_st = next(line.strip() for line in body.splitlines() if line.strip().startswith("st."))
    assert first_st.startswith("st.set_page_config("), (
        f"set_page_config must be the first st.* command, found: {first_st}"
    )
