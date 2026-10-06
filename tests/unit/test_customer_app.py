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


def test_customer_app_exposes_login_submission() -> None:
    import ui.customer_app as app

    assert hasattr(app, "login_submission")


def test_login_submission_true_only_when_submitted_and_id_present() -> None:
    import ui.customer_app as app

    # One click = one auth attempt: the predicate is True only when the form was submitted AND a
    # non-blank id was typed. This is the regression guard for the double-click login bug.
    assert app.login_submission(True, "CLI-1") is True
    assert app.login_submission(False, "CLI-1") is False
    assert app.login_submission(True, "   ") is False
    assert app.login_submission(True, "") is False


def test_customer_app_exposes_login_succeeded() -> None:
    import ui.customer_app as app

    assert hasattr(app, "login_succeeded")


def test_login_succeeded_only_with_a_token() -> None:
    import ui.customer_app as app

    # Regression guard for the first-click defect: a successful auth yields a token, which must
    # drive an immediate repaint of the chat view (st.rerun). The error paths leave the token None
    # and must stay on the login screen.
    assert app.login_succeeded("tok-123") is True
    assert app.login_succeeded(None) is False
    assert app.login_succeeded("") is False


def test_main_reruns_after_successful_first_click_login() -> None:
    import inspect

    import ui.customer_app as app

    # The success path must call st.rerun() so one click transitions to the authenticated UI
    # instead of leaving the login screen rendered until another interaction.
    body = inspect.getsource(app.main)
    assert "if login_succeeded(st.session_state.token):" in body
    assert "st.rerun()" in body


def test_parse_chat_response_extracts_text_trace_and_case_id() -> None:
    import ui.customer_app as app

    payload = {
        "response_text": "Tu saldo es 100 USD.",
        "trace_id": "abc123def456",
        "outcome": "RESOLVED",
        "decision": "Answer",
        "rule_id": None,
        "handoff_case_id": "CASE-82118D30",
    }
    text, trace_id, case_id = app._parse_chat_response(payload)
    assert text == "Tu saldo es 100 USD."
    assert trace_id == "abc123def456"
    assert case_id == "CASE-82118D30"


def test_parse_chat_response_degrades_on_missing_fields() -> None:
    import ui.customer_app as app

    # A turn with no handoff carries no case id: the third value is "" and is NEVER fabricated.
    text, trace_id, case_id = app._parse_chat_response({})
    assert text == ""
    assert trace_id == ""
    assert case_id == ""


def test_parse_chat_response_case_id_empty_when_null() -> None:
    import ui.customer_app as app

    _, _, case_id = app._parse_chat_response({"response_text": "ok", "handoff_case_id": None})
    assert case_id == ""


def test_logo_path_returns_existing_asset() -> None:
    import ui.customer_app as app

    path = app.logo_path()
    # The asset is committed, so the helper resolves it; it must point at the brand PNG.
    assert path is not None
    assert path.name == "cora_logo.png"
    assert path.exists()


def test_header_title_is_product_name() -> None:
    import ui.customer_app as app

    title = app.header_title("es")
    assert "CORA" in title
    assert "Customer-Oriented Resolution Agent" in title
    # Language-neutral: the brand name is identical across languages.
    assert app.header_title("pt") == title


def test_brand_css_is_scoped_and_carries_brand_tokens() -> None:
    import ui.customer_app as app

    css = app.brand_css()
    assert css.strip().startswith("<style>")
    assert css.strip().endswith("</style>")
    # Carries the design-system palette so a regression in the tokens is caught.
    assert "#0E2A47" in css  # navy-900
    assert "#2E8C97" in css  # teal-500
    assert "#C8A24B" in css  # gold-500 accent


def test_case_badge_formats_and_handles_empty() -> None:
    import ui.customer_app as app

    assert app.case_badge("CASE-1") == "Caso: CASE-1"
    assert app.case_badge("  CASE-2  ") == "Caso: CASE-2"
    assert app.case_badge("") == ""
    assert app.case_badge(None) == ""


def test_main_persists_case_id_in_session() -> None:
    import inspect

    import ui.customer_app as app

    # Once a turn returns a case id it is stored in session so it stays visible on later turns,
    # and the session value is seeded so the first (no-handoff) render shows no fabricated case.
    body = inspect.getsource(app.main)
    assert 'st.session_state.setdefault("case_id", "")' in body
    assert "st.session_state.case_id = handoff_case_id" in body


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
