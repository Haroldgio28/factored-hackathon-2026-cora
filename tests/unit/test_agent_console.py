"""Smoke test for task 4.5: the agent-console view imports cleanly (ponytail: UI is logic-free).

The console is a thin render over `JsonHandoffStore` with no business logic, so it needs no
behavioural test beyond proving it imports without pulling Streamlit at module load (Streamlit is
imported lazily inside `main()`), and that its pure render helpers tolerate an empty package.
"""

from __future__ import annotations


def test_agent_console_imports() -> None:
    import ui.agent_console as console

    assert hasattr(console, "main")


def test_priority_badge_uses_gold_and_ink_not_emoji() -> None:
    import ui.agent_console as console

    high = console._priority_badge("high")
    normal = console._priority_badge("normal")
    assert "HIGH" in high and "normal" in normal
    # DESIGN_SYSTEM.md section 6/8: a single gold high marker and a neutral ink marker, no emoji.
    assert "cora-prio-high" in high
    assert "cora-prio-normal" in normal
    assert "🔴" not in high and "⚪" not in normal


def test_save_outcome_true_only_for_verified_result() -> None:
    import ui.agent_console as console

    # The store returns the reloaded, read-back-verified case on success and None on any failure.
    assert console.save_outcome({"agent_workflow": {"status": "Resolved"}}) is True
    assert console.save_outcome(None) is False


def test_status_chip_html_colour_codes_each_status() -> None:
    import ui.agent_console as console
    from cora.handoff.store import CASE_STATUSES, status_label

    expected = {
        "Open": "cora-status-open",
        "In progress": "cora-status-progress",
        "Resolved": "cora-status-resolved",
        "Follow-up required": "cora-status-followup",
    }
    for status in CASE_STATUSES:
        chip = console.status_chip_html(status, "es")
        assert expected[status] in chip
        # The localized label is shown, not the raw English id.
        assert status_label(status, "es") in chip


class _FakeSt:
    """Minimal Streamlit recorder: captures markdown/caption calls for pure render-helper tests.

    `container(border=True)` returns a no-op context manager and `table` is a sink, so the panel
    render helpers can be exercised without importing Streamlit; markdown is still recorded so the
    marker/heading markup is assertable.
    """

    def __init__(self) -> None:
        self.markdowns: list[str] = []
        self.captions: list[str] = []
        self.tables: list[object] = []

    def markdown(self, text: str, *_: object, **__: object) -> None:
        self.markdowns.append(text)

    def caption(self, text: str, *_: object, **__: object) -> None:
        self.captions.append(text)

    def table(self, data: object, *_: object, **__: object) -> None:
        self.tables.append(data)

    def container(self, *_: object, **__: object) -> _FakeSt:
        return self

    def __enter__(self) -> _FakeSt:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def test_render_list_renders_dash_placeholder_when_empty() -> None:
    import ui.agent_console as console

    st = _FakeSt()
    console._render_list(st, "Unresolved questions", [])
    # DESIGN_SYSTEM.md section 8: the header still renders and the empty body reads "—".
    assert any("Unresolved questions" in m for m in st.markdowns)
    assert "—" in st.captions


def test_render_list_renders_items_when_present() -> None:
    import ui.agent_console as console

    st = _FakeSt()
    console._render_list(st, "Actions taken", ["froze card"])
    assert any("froze card" in m for m in st.markdowns)
    assert "—" not in st.captions


def test_customer_info_rows_only_safe_fields() -> None:
    import ui.agent_console as console

    package = {
        "language": "es",
        "transcript_ref": "JTI-123",
        "verified_facts": [
            {"label": "Card number", "value": "**** **** **** 1234"},
            {"label": "Current balance", "value": "952.03"},
        ],
    }
    rows = console.customer_info_rows(package)
    labels = {label for label, _ in rows}
    values = " ".join(value for _, value in rows)
    assert ("Language", "es") in rows
    assert "Session (transcript ref)" in labels
    assert "Card number" in labels
    # Only the masked identifier is surfaced; no raw 16-digit card number appears.
    import re

    assert not re.search(r"\d{13,16}", values)
    assert "952.03" in values


def test_customer_info_rows_omits_absent_fields() -> None:
    import ui.agent_console as console

    assert console.customer_info_rows({}) == []


def test_agent_workflow_view_defaults_for_legacy_package() -> None:
    import ui.agent_console as console

    workflow = console.agent_workflow_view({"customer_request": "hola"})
    assert workflow["status"] == "Open"
    assert workflow["notes"] == ""
    assert workflow["follow_up"] is False


def test_agent_workflow_view_reads_existing_section() -> None:
    import ui.agent_console as console

    package = {"agent_workflow": {"status": "Resolved", "notes": "done", "follow_up": True}}
    workflow = console.agent_workflow_view(package)
    assert workflow["status"] == "Resolved"
    assert workflow["notes"] == "done"


def test_trace_label_formats_and_handles_empty() -> None:
    import ui.agent_console as console

    assert console.trace_label("abc123") == "Trace id: abc123"
    assert console.trace_label("") == "Trace id: (none)"


def test_branding_logo_path_and_css() -> None:
    import ui.branding as branding

    path = branding.logo_path()
    assert path is not None and path.name == "cora_logo.png" and path.exists()
    css = branding.brand_css()
    assert css.strip().startswith("<style>") and css.strip().endswith("</style>")
    assert "#1C4E80" in css and "#2E8C97" in css
    assert "CORA" in branding.header_title()
    # The four canonical status colours and the card/priority/status classes must be present, so a
    # regression that drops the console design system fails here (console-review finding 3).
    for token in ("--ok-600", "--warn-600", "--active-600", "--idle-500"):
        assert token in css
    for cls in (
        "cora-card",
        "cora-card-customer",
        "cora-prio-high",
        "cora-status-resolved",
    ):
        assert cls in css


def test_case_option_label_includes_reason_and_localized_status() -> None:
    import ui.agent_console as console

    package = {"reason": "E2", "agent_workflow": {"status": "In progress"}}
    label = console.case_option_label(package, "CASE-82118D30", "es")
    # The selector row must carry the case id, the reason code, and the LOCALIZED status so an
    # agent can triage without opening the case (DESIGN_SYSTEM.md section 6).
    assert "CASE-82118D30" in label
    assert "E2" in label
    assert "En gestion" in label  # es label for "In progress"
    # A legacy package without a reason/status degrades to a bare case id + default Open label.
    legacy = console.case_option_label({}, "CASE-X", "pt")
    assert "CASE-X" in legacy
    assert "Aberto" in legacy  # pt label for the default Open status


def test_customer_card_marker_targets_whole_container() -> None:
    import ui.agent_console as console
    import ui.branding as branding

    # The customer panel emits a hidden marker INSIDE the bordered container, and the shared CSS
    # tints the WHOLE container via a `:has()` parent selector (not just a styled heading div).
    st = _FakeSt()
    console._render_customer_panel(st, {"language": "es"})
    assert any("cora-card-marker-customer" in m for m in st.markdowns)
    css = branding.brand_css()
    assert ":has(.cora-card-marker-customer)" in css
    assert ":has(.cora-card-marker)" in css


def test_left_pane_uses_navy_section_headers_and_mono_ids() -> None:
    import ui.agent_console as console
    import ui.branding as branding

    # Section headers render as navy H3 (`.cora-section`) and opaque ids in monospace (`.cora-id`),
    # not plain bold markdown (console-review blocking finding).
    st = _FakeSt()
    console._section_header(st, "Verified facts")
    console._id_line(st, "CASE-123")
    assert any("cora-section" in m and "Verified facts" in m for m in st.markdowns)
    assert any("cora-id" in m and "CASE-123" in m for m in st.markdowns)
    css = branding.brand_css()
    assert ".cora-section" in css
    assert ".cora-id" in css


def test_customer_app_shares_the_branding_module() -> None:
    # Finding 3: once the shared branding module exists BOTH surfaces must use it, so the palette
    # cannot drift. The customer app must render the SAME css/logo/header as the branding module.
    import ui.branding as branding
    import ui.customer_app as app

    assert app.brand_css() == branding.brand_css()
    assert app.logo_path() == branding.logo_path()
    assert app.header_title("es") == branding.header_title()
