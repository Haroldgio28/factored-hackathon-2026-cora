"""Shared CORA brand assets for the Streamlit surfaces (DESIGN_SYSTEM.md, REQ-05).

Both the customer chat view and the agent console must look consistent and professional: the same
logo, palette and typography. These three pure helpers (logo path, scoped CSS, product name) are
the ponytail seam that keeps the two surfaces identical without a frontend dependency or a web
font. They are pure strings/paths so they are testable without importing Streamlit.

Both `ui/customer_app.py` and `ui/agent_console.py` resolve `logo_path`/`brand_css`/`header_title`
through this module (the customer app delegates thin wrappers to it), so a palette change here flows
to both surfaces and they can never drift. Keep the tokens in sync with DESIGN_SYSTEM.md.
"""

from __future__ import annotations

from pathlib import Path

# Language-neutral brand line shown in the header next to the logo (DESIGN_SYSTEM.md section 5).
PRODUCT_NAME = "CORA - Customer-Oriented Resolution Agent"


def logo_path() -> Path | None:
    """Return the committed brand logo path, or `None` when the asset is missing.

    Pure so the logo-vs-text-fallback decision is testable without Streamlit. A missing file must
    degrade to a styled text title rather than crash the UI.
    """
    path = Path(__file__).parent / "assets" / "cora_logo.png"
    return path if path.exists() else None


def header_title() -> str:
    """Return the language-neutral brand name shown in the header (DESIGN_SYSTEM.md section 5)."""
    return PRODUCT_NAME


def brand_css() -> str:
    """Return one scoped `<style>` block applying the DESIGN_SYSTEM.md tokens (section 2/4/5/7).

    Pure string so it is testable (non-empty, carries the brand hex and status tokens) without
    Streamlit, and reuse-the-platform: no CSS framework, no JS, no web font. One shared block for
    BOTH surfaces (customer chat + agent console): header, mono trace/case chips, chat bubbles, the
    two-pane console cards, the gold/ink priority markers and the four colour-coded status chips.
    """
    return """
<style>
  :root {
    --navy-900:#0E2A47; --navy-700:#1C4E80; --teal-500:#2E8C97; --teal-100:#DCEDEF;
    --gold-500:#C8A24B; --ink-900:#14202B; --ink-500:#5B6B78; --line-200:#E3E8EC;
    --bg-50:#F6F8FA; --bg-0:#FFFFFF;
    --ok-600:#2E7D5B; --warn-600:#B4791F; --active-600:#1C4E80; --idle-500:#5B6B78;
  }
  .cora-header { display:flex; flex-direction:column; gap:4px; padding-bottom:8px;
    border-bottom:1px solid var(--line-200); margin-bottom:12px; }
  .cora-product { color:var(--navy-900); font-size:20px; font-weight:600; line-height:1.25; }
  .cora-tagline { color:var(--ink-500); font-size:13px; }
  .cora-chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
  .cora-chip { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px;
    letter-spacing:.02em; color:var(--ink-500); background:var(--bg-0);
    border:1px solid var(--line-200); border-radius:6px; padding:2px 8px; }
  .cora-chip-case { color:var(--navy-900); border-color:var(--gold-500); }
  .cora-card { background:var(--bg-0); border:1px solid var(--line-200); border-radius:8px;
    padding:12px 14px; margin-bottom:12px; }
  .cora-card-customer { background:var(--teal-100); }
  /* Streamlit renders widgets as siblings, so a styled <div> cannot wrap them directly. Instead a
     hidden marker is emitted INSIDE `st.container(border=True)` and a scoped parent selector tints
     and shapes the WHOLE bordered block (DESIGN_SYSTEM.md section 6: the customer/workflow panels
     are real card surfaces, not just styled headings). */
  .cora-card-marker { display:none; }
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.cora-card-marker) {
    border:1px solid var(--line-200); border-radius:8px; }
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.cora-card-marker-customer) {
    background:var(--teal-100); }
  .cora-card h3, .cora-card-title { color:var(--navy-900); font-size:16px; font-weight:600;
    margin:0 0 8px 0; }
  /* H3 section hierarchy for the left REQ-16 pane: navy headings, not bold body markdown. */
  .cora-section { color:var(--navy-900); font-size:16px; font-weight:600; line-height:1.25;
    margin:12px 0 4px 0; }
  /* Opaque identifiers (case / transcript / trace) always in monospace, ink-500 (section 3). */
  .cora-id { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px;
    letter-spacing:.02em; color:var(--ink-500); }
  /* Priority marker: gold used once for high, neutral ink otherwise (section 6, no emoji). */
  .cora-prio { font-size:14px; line-height:1; }
  .cora-prio-high { color:var(--gold-500); }
  .cora-prio-normal { color:var(--ink-500); }
  /* Status chip: the one place colour is loud (section 7); colour comes inline from the token. */
  .cora-status { display:inline-flex; align-items:center; gap:6px; font-size:13px; font-weight:600; }
  .cora-status-dot { font-size:12px; line-height:1; }
  .cora-status-open { color:var(--idle-500); }
  .cora-status-progress { color:var(--active-600); }
  .cora-status-resolved { color:var(--ok-600); }
  .cora-status-followup { color:var(--warn-600); }
  /* Clear bot-vs-customer separation: teal-tinted bot bubbles, hairline-bordered user bubbles. */
  .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background:var(--teal-100); border-radius:10px; }
  .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background:var(--bg-0); border:1px solid var(--line-200); border-radius:10px; }
</style>
"""
