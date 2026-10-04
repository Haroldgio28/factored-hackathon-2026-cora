"""Agent console: a read-only Streamlit view of the handoff queue (task 4.5, REQ-16).

A human agent needs to see the handoff packages CORA created and open one to continue a case.
This view does exactly that and nothing more (ponytail: the UI has no business logic): it reads
packages from the durable `JsonHandoffStore` and renders one package's REQ-16 fields read-only.
All grounding, masking and decisions already happened upstream; the console only displays what
the store holds.

Run with: `uv run streamlit run ui/agent_console.py` (needs the `ui` dependency group).
"""

from __future__ import annotations

from typing import Any

from cora.handoff.store import JsonHandoffStore


def _priority_badge(priority: str) -> str:
    return "🔴 HIGH" if priority == "high" else "⚪ normal"


def trace_label(trace_ref: object) -> str:
    """Format a package's `trace_ref` (task 5.1) into the read-only line the console shows.

    Pure so the UI retrieval path is testable without Streamlit. An empty/missing id reads as a
    plain "none" rather than a blank line, so a human sees the turn simply had no trace id.
    """
    trace_ref = str(trace_ref or "").strip()
    return f"Trace id: {trace_ref}" if trace_ref else "Trace id: (none)"


def _render_package(st: Any, case_id: str, package: dict[str, object]) -> None:
    """Render one handoff package's REQ-16 fields read-only (no editing, no actions)."""
    st.subheader(f"Case {case_id}")
    cols = st.columns(3)
    cols[0].metric("Reason", str(package.get("reason", "")))
    cols[1].metric("Priority", _priority_badge(str(package.get("priority", "normal"))))
    cols[2].metric("Language", str(package.get("language", "")))

    st.markdown("**Customer request (verbatim, PII-masked)**")
    st.info(str(package.get("customer_request", "")))

    facts = package.get("verified_facts") or []
    if isinstance(facts, list) and facts:
        st.markdown("**Verified facts** (with source references)")
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

    _render_list(st, "Actions taken (verified outcomes)", package.get("actions_taken"))
    _render_list(st, "Supporting transactions", package.get("supporting_transactions"))
    _render_list(st, "Unresolved questions", package.get("unresolved_questions"))

    st.caption(f"Transcript reference (session): {package.get('transcript_ref', '')}")
    st.caption(trace_label(package.get("trace_ref")))
    st.caption(f"Created at: {package.get('created_at', '')}")


def _render_list(st: Any, title: str, items: object) -> None:
    if isinstance(items, list) and items:
        st.markdown(f"**{title}**")
        for item in items:
            st.markdown(f"- {item}")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="CORA agent console", layout="wide")
    st.title("CORA agent console - handoff queue")

    cases = JsonHandoffStore().list_cases()
    if not cases:
        st.write("No open handoff cases.")
        return

    case_id = st.sidebar.selectbox("Open cases", sorted(cases))
    _render_package(st, case_id, cases[case_id])


if __name__ == "__main__":
    main()
