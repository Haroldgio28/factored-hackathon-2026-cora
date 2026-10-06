"""Agent console: an actionable one-screen workspace over the handoff queue (REQ-16).

A human agent must resolve a case WITHOUT hopping between screens, so this view shows everything
in one place: the handoff package's REQ-16 fields (customer request, verified facts with source
refs, actions taken, supporting transactions, unresolved questions, reason, priority, transcript
and trace refs) on the left, and on the right a CUSTOMER INFORMATION panel plus an editable case
management block (status, free-text notes, follow-up flag).

The console still holds NO business logic (ponytail): status validation and the atomic persistence
of the agent-workflow edits live in `JsonHandoffStore.update_agent_workflow`; the pure helpers here
only shape what the view renders. Everything shown comes from the stored package (masked upstream)
- the console reads no raw PII and expands nothing stored.

Run with: `uv run streamlit run ui/agent_console.py` (needs the `ui` dependency group).
"""

from __future__ import annotations

from typing import Any

from cora.handoff.store import (
    CASE_STATUSES,
    JsonHandoffStore,
    status_label,
    with_agent_workflow,
)
from ui.branding import brand_css, header_title, logo_path

# Canonical status id -> the `.cora-status-*` CSS class the shared brand block colour-codes
# (DESIGN_SYSTEM.md section 7). An unknown status falls back to the neutral "open" class.
_STATUS_CSS_CLASS: dict[str, str] = {
    "Open": "cora-status-open",
    "In progress": "cora-status-progress",
    "Resolved": "cora-status-resolved",
    "Follow-up required": "cora-status-followup",
}


def status_chip_html(status: str, lang: str) -> str:
    """Render a colour-coded status chip (coloured dot + localized label) as inline HTML.

    Pure so the colour-coding is testable without Streamlit. The colour comes from the shared
    palette via the `.cora-status-*` class (DESIGN_SYSTEM.md section 7).
    """
    css_class = _STATUS_CSS_CLASS.get(status, "cora-status-open")
    label = status_label(status, lang)
    return f'<span class="cora-status {css_class}"><span class="cora-status-dot">●</span>{label}</span>'


def _priority_badge(priority: str) -> str:
    """Priority marker: a single gold dot for high, a neutral ink dot otherwise (DESIGN_SYSTEM 6).

    Gold is the one loud accent, used at most once per screen on a high-priority case; a normal
    case gets the neutral ink dot. No red/white emoji (section 8 forbids emoji-as-UI).
    """
    if priority == "high":
        return '<span class="cora-prio cora-prio-high">●</span> HIGH'
    return '<span class="cora-prio cora-prio-normal">●</span> normal'


def save_outcome(result: object) -> bool:
    """Decide whether a workflow save succeeded, from the store's verified return value.

    Pure so the success-vs-fail-closed decision is testable without Streamlit. The store returns
    the reloaded, read-back-verified case on success and `None` on any failure (missing case,
    mismatch), so a truthy result is the only signal the console may report completion on.
    """
    return result is not None


def trace_label(trace_ref: object) -> str:
    """Format a package's `trace_ref` (task 5.1) into the read-only line the console shows.

    Pure so the UI retrieval path is testable without Streamlit. An empty/missing id reads as a
    plain "none" rather than a blank line, so a human sees the turn simply had no trace id.
    """
    trace_ref = str(trace_ref or "").strip()
    return f"Trace id: {trace_ref}" if trace_ref else "Trace id: (none)"


def case_option_label(package: dict[str, object], case_id: str, lang: str) -> str:
    """Format a sidebar case-selector row as `CASE-… · reason · <localized status>` (DESIGN 6).

    Pure so the triage label is testable without Streamlit. The agent must distinguish and triage
    open/resolved/follow-up cases WITHOUT opening each one, so the row carries the reason code and
    the localized status alongside the case id. Missing fields degrade to a bare case id.
    """
    reason = str(package.get("reason") or "").strip()
    status = str(agent_workflow_view(package).get("status", "Open"))
    parts = [case_id]
    if reason:
        parts.append(reason)
    parts.append(status_label(status, lang))
    return " · ".join(parts)


def customer_info_rows(package: dict[str, object]) -> list[tuple[str, str]]:
    """Build the CUSTOMER INFORMATION panel from SAFE package fields only (no raw PII, REQ-16).

    Pure so the panel is testable without Streamlit. It reuses only what the package already
    carries (masked upstream): language, the session transcript ref, and any already-masked
    identifiers inside `verified_facts` (e.g. a masked card number / product id). It never reads
    or expands raw PII and never fabricates a value - an absent field is simply omitted.
    """
    rows: list[tuple[str, str]] = []
    language = str(package.get("language") or "").strip()
    if language:
        rows.append(("Language", language))
    transcript_ref = str(package.get("transcript_ref") or "").strip()
    if transcript_ref:
        rows.append(("Session (transcript ref)", transcript_ref))
    facts = package.get("verified_facts") or []
    if isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, dict):
                label = str(fact.get("label") or "").strip()
                value = str(fact.get("value") or "").strip()
                if label and value:
                    rows.append((label, value))
    return rows


def agent_workflow_view(package: dict[str, object]) -> dict[str, object]:
    """Return the case's `agent_workflow` section, with defaults for a legacy package.

    Pure so the console's edit defaults (Open / no notes / no follow-up) are testable without the
    store. Backward compatible: a package stored before the feature reads as the Open defaults.
    """
    workflow = with_agent_workflow(package)["agent_workflow"]
    assert isinstance(workflow, dict)  # noqa: S101 - with_agent_workflow guarantees a dict
    return workflow


def _section_header(st: Any, title: str) -> None:
    """Render a left-pane REQ-16 section header as a navy H3 (DESIGN_SYSTEM.md section 3/6).

    Section headers are the hierarchy on the scannable left pane, so they use the navy H3 style
    (`.cora-section`) rather than bold body markdown.
    """
    st.markdown(f'<div class="cora-section">{title}</div>', unsafe_allow_html=True)


def _id_line(st: Any, text: str) -> None:
    """Render an opaque identifier (case / transcript / trace) in monospace (DESIGN_SYSTEM 3)."""
    st.markdown(f'<div class="cora-id">{text}</div>', unsafe_allow_html=True)


def _render_package(st: Any, case_id: str, package: dict[str, object]) -> None:
    """Render one handoff package's REQ-16 fields read-only (no editing, no actions)."""
    _id_line(st, f"CASE {case_id}")
    cols = st.columns(3)
    cols[0].metric("Reason", str(package.get("reason", "")))
    cols[1].markdown("Priority")
    cols[1].markdown(_priority_badge(str(package.get("priority", "normal"))), unsafe_allow_html=True)
    cols[2].metric("Language", str(package.get("language", "")))

    _section_header(st, "Customer request (verbatim, PII-masked)")
    st.info(str(package.get("customer_request", "")))

    facts = package.get("verified_facts") or []
    _section_header(st, "Verified facts (with source references)")
    if isinstance(facts, list) and facts:
        st.table(
            [
                {
                    "fact": f.get("label", ""),
                    "value": f.get("value", ""),
                    "sources": ", ".join(
                        f"{ref.get('table')}:{ref.get('ref')}" for ref in (f.get("source_refs") or [])
                    ),
                }
                for f in facts
                if isinstance(f, dict)
            ]
        )
    else:
        st.caption("—")

    _render_list(st, "Actions taken (verified outcomes)", package.get("actions_taken"))
    _render_list(st, "Supporting transactions", package.get("supporting_transactions"))
    _render_list(st, "Unresolved questions", package.get("unresolved_questions"))

    _id_line(st, f"Transcript reference (session): {package.get('transcript_ref', '')}")
    _id_line(st, trace_label(package.get("trace_ref")))
    st.caption(f"Created at: {package.get('created_at', '')}")


def _render_list(st: Any, title: str, items: object) -> None:
    """Render a REQ-16 list section; an EMPTY section still renders its header and a `—` placeholder.

    DESIGN_SYSTEM.md section 8: empty sections read "—" so an absent value is visible, not silently
    dropped (a missing section is otherwise indistinguishable from "not rendered" for the agent).
    The header uses the navy H3 section style, matching the rest of the left pane.
    """
    _section_header(st, title)
    if isinstance(items, list) and items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption("—")


def _render_customer_panel(st: Any, package: dict[str, object]) -> None:
    """Render the CUSTOMER INFORMATION card (right pane): safe, masked fields only.

    Rendered inside a teal-tinted `.cora-card .cora-card-customer` container (DESIGN_SYSTEM.md
    section 6) so the customer-info panel reads as a distinct surface.
    """
    with st.container(border=True):
        # The hidden marker lets the shared `:has()` selector tint the WHOLE bordered container as
        # the teal customer card (Streamlit renders the table below as a sibling, so a styled <div>
        # cannot wrap it directly). The title keeps the navy H3 style.
        st.markdown(
            '<span class="cora-card-marker cora-card-marker-customer"></span>'
            '<div class="cora-card-title">Cliente</div>',
            unsafe_allow_html=True,
        )
        rows = customer_info_rows(package)
        if rows:
            st.table([{"campo": label, "valor": value} for label, value in rows])
        else:
            st.caption("—")


def _render_case_management(st: Any, case_id: str, package: dict[str, object], lang: str) -> None:
    """Render the editable case-management card (right pane) and persist edits via the store.

    No business logic here: status validation and the atomic write live in
    `JsonHandoffStore.update_agent_workflow`. The selectbox shows localized labels but stores the
    canonical English status id.
    """
    workflow = agent_workflow_view(package)
    current_status = str(workflow.get("status", "Open"))
    current_index = CASE_STATUSES.index(current_status) if current_status in CASE_STATUSES else 0

    with st.container(border=True):
        # The hidden marker lets the shared `:has()` selector shape the WHOLE bordered container as
        # the case-management card wrapping status, notes, follow-up and Save (not just the title).
        st.markdown(
            '<span class="cora-card-marker"></span><div class="cora-card-title">Gestion del caso</div>',
            unsafe_allow_html=True,
        )
        # Show the current status colour-coded (DESIGN_SYSTEM.md section 7) before the edit form.
        st.markdown(status_chip_html(current_status, lang), unsafe_allow_html=True)
        _render_case_form(st, case_id, current_index, workflow, lang)
    st.caption(f"actualizado: {workflow.get('updated_at', '')}")


def _render_case_form(
    st: Any, case_id: str, current_index: int, workflow: dict[str, object], lang: str
) -> None:
    """Render the editable status/notes/follow-up form and persist a verified save (REQ-16)."""
    with st.form(f"case_mgmt_{case_id}"):
        status = st.selectbox(
            "Estado",
            CASE_STATUSES,
            index=current_index,
            format_func=lambda value: status_label(value, lang),
        )
        notes = st.text_area("Notas del agente", value=str(workflow.get("notes", "")))
        follow_up = st.checkbox("Requiere seguimiento", value=bool(workflow.get("follow_up", False)))
        saved = st.form_submit_button("Guardar")

    if saved:
        # The store masks the notes, writes atomically, then reloads and verifies the persisted
        # post-condition. Report completion ONLY on that verified result; otherwise fail closed
        # (the case may have been removed, or the write did not land) and claim nothing.
        result = JsonHandoffStore().update_agent_workflow(
            case_id, status=status, notes=notes, follow_up=follow_up
        )
        if save_outcome(result):
            st.success("Caso actualizado")
        else:
            st.error("No se pudo actualizar el caso. Intenta de nuevo.")


def main() -> None:
    import streamlit as st

    icon = logo_path()
    st.set_page_config(
        page_title="CORA agent console",
        page_icon=str(icon) if icon is not None else None,
        layout="wide",
    )
    st.markdown(brand_css(), unsafe_allow_html=True)

    header = logo_path()
    if header is not None:
        st.image(str(header), width=160)
    st.markdown(
        f'<div class="cora-header"><span class="cora-product">{header_title()}</span>'
        f'<span class="cora-tagline">Consola del agente</span></div>',
        unsafe_allow_html=True,
    )

    cases = JsonHandoffStore().list_cases()
    if not cases:
        st.write("No open handoff cases.")
        return

    # The selector shows each case with its reason and localized status so the agent can triage
    # open/resolved/follow-up cases without opening each one (DESIGN_SYSTEM.md section 6). The
    # label language follows the case's own language; the sidebar list is sorted by id.
    def _case_label(cid: str) -> str:
        pkg = cases[cid]
        return case_option_label(pkg, cid, str(pkg.get("language") or "es"))

    case_id = st.sidebar.selectbox("Open cases", sorted(cases), format_func=_case_label)
    lang = str(cases[case_id].get("language") or "es")
    package = cases[case_id]

    left, right = st.columns([2, 1])
    with left:
        _render_package(st, case_id, package)
    with right:
        _render_customer_panel(st, package)
        _render_case_management(st, case_id, package, lang)


if __name__ == "__main__":
    main()
