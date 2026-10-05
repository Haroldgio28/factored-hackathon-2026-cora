# CORA Design System

A deliberate visual language for a **banking** product — calm, trustworthy, dense-but-clear. The anti-goal is the generic "AI chatbot" look: no purple gradients, no glassmorphism, no neon, no emoji-as-UI, no center-everything landing page. Think private-banking console, not a toy assistant.

## 1. Design principles

1. **Institutional calm.** Muted navy/teal, generous whitespace, one gold accent used sparingly. Color carries meaning (status, priority), never decoration.
2. **Flat and honest.** No drop shadows beyond a hairline, no gradients except the logo itself. Borders and background tints create hierarchy, not elevation tricks.
3. **Typography does the work.** A real type scale and weight contrast replace boxes and color blocks.
4. **Left-aligned, structured.** Content sits in a constrained column (chat) or a two-pane layout (console). Nothing floats centered like a splash screen.
5. **Dense where it helps the agent.** The console shows everything on one screen; scanning beats clicking.

## 2. Color palette (exact)

Pulled from the logo and extended into a usable UI system.

| Token | Hex | Use |
|---|---|---|
| `--navy-900` | `#0E2A47` | Primary brand, headers, bot name, primary buttons |
| `--navy-700` | `#1C4E80` | Links, active states, secondary headers |
| `--teal-500` | `#2E8C97` | Accent lines, focus ring, info chips |
| `--teal-100` | `#DCEDEF` | Bot message bubble background, info panel tint |
| `--gold-500` | `#C8A24B` | ONE accent only: logo-adjacent divider, "priority: high" marker |
| `--ink-900` | `#14202B` | Primary text |
| `--ink-500` | `#5B6B78` | Secondary text, captions, timestamps |
| `--line-200` | `#E3E8EC` | Hairline borders, dividers |
| `--bg-50` | `#F6F8FA` | App background (NOT pure white) |
| `--bg-0` | `#FFFFFF` | Cards, message area |
| `--ok-600` | `#2E7D5B` | status Resolved |
| `--warn-600` | `#B4791F` | status Follow-up required |
| `--active-600` | `#1C4E80` | status In progress |
| `--idle-500` | `#5B6B78` | status Open |

Rules: navy is the voice of the brand; teal is the only "friendly" accent; gold appears at most once per screen. Backgrounds are off-white (`--bg-50`), never stark white, so cards read as surfaces.

## 3. Typography

- **Family:** `Inter` if available via the system/Streamlit default stack, else the native sans stack (`-apple-system, Segoe UI, Roboto, Helvetica, Arial`). No web-font dependency — ponytail; Inter only if already present.
- **Scale:** H1 28/600, H2 20/600, H3 16/600, body 15/400, caption 13/400, mono 13 (for ids/trace/case).
- **Line-height:** 1.5 body, 1.25 headings.
- IDs (`CASE-...`, `trace: ...`) always in monospace, `--ink-500`, letter-spacing 0.02em.

## 4. Spacing & shape

- 8px base grid: 4, 8, 12, 16, 24, 32.
- Radius: 8px cards, 10px message bubbles, 6px inputs/buttons. No pills, no fully-round.
- Borders: 1px `--line-200`. Shadow: at most `0 1px 2px rgba(14,42,71,.06)` on cards; none on bubbles.
- Content width: chat column max **720px**, left-aligned within the page; console a 2/3 + 1/3 split.

## 5. Customer chat — layout

```
┌───────────────────────────────────────────────┐
│ [logo 40px]  CORA            [ES ▾]  caso ▸ ·   │  header bar, navy text on bg-0, hairline bottom
│              Customer-Oriented Resolution Agent │  tagline, caption, ink-500
├───────────────────────────────────────────────┤
│  trace: a1b2…   ·   Caso: CASE-… (sticky chips) │  mono chips row, only when present
│                                                 │
│   ┌─ bot ─────────────────────┐                 │  bot bubble: teal-100 bg, ink-900, left
│   │ Hola, soy CORA…           │                 │  avatar = small logo mark, 24px
│   └───────────────────────────┘                 │
│                   ┌─ you ───────────────────┐   │  user bubble: bg-0 + 1px line-200, right
│                   │ ¿cuál es mi saldo?      │   │
│                   └─────────────────────────┘   │
├───────────────────────────────────────────────┤
│  [ Escribe tu mensaje…               ] [ → ]    │  input bar, navy send button, pinned bottom
└───────────────────────────────────────────────┘
```

- Header: logo left, product name navy H2, tagline caption. Language toggle right. A sticky chip row under it shows `trace:` and `Caso:` in mono **only when they exist** (no fabricated placeholders).
- Bot bubbles: `--teal-100` background, no border, left-aligned, small logo mark as avatar. User bubbles: white with hairline border, right-aligned. 10px radius, 12px padding, 8px gap, max-width 80%.
- Send button: `--navy-900`, white arrow, focus ring `--teal-500`. One click submits (the login/double-click fix pattern applies to send too).
- Login screen (pre-auth): the logo centered ONCE at ~160px, product name, a single `customer_id` field + "Ingresar" button in a 360px card on `--bg-50`. Spinner on submit. This is the only centered view.

## 6. Agent console — layout (one screen, no hopping)

```
┌──────────────────────────────────────────────────────────────┐
│ [logo] CORA · Consola de agente                 [status ▾]     │  header, navy, hairline
├───────────────────────────────┬──────────────────────────────┤
│ CASE  CASE-82118D30…           │  CLIENTE                       │
│ ● priority  ·  reason E2       │  Nombre / id (masked)          │  right pane 1/3:
│ idioma: es   trace: …          │  Productos (masked ****1234)   │  customer info panel,
│                                │  Segmento / país               │  teal-100 tint card
│ SOLICITUD DEL CLIENTE          │                                │
│ "…masked request…"             │  ─────────────────────────     │
│                                │  GESTIÓN DEL CASO              │  agent workflow card:
│ HECHOS VERIFICADOS             │  Estado: [Open ▾]             │   status select (4 states,
│  • balance … (src: …)          │  Notas del agente:            │   color-coded),
│ ACCIONES / TRANSACCIONES       │  [ textarea … ]               │   notes textarea,
│ PREGUNTAS ABIERTAS             │  ☐ Requiere follow-up         │   follow-up checkbox,
│                                │  [ Guardar ]  actualizado: …  │   Save + updated_at
└───────────────────────────────┴──────────────────────────────┘
```

- Left pane (2/3): the REQ-16 package, scannable — section headers in H3 navy, values in body, source refs + ids in mono caption. Priority shown as a colored dot (gold `--gold-500` for high, ink for normal).
- Right pane (1/3): two stacked cards — **Cliente** (customer info, masked) on a `--teal-100` tint, and **Gestión del caso** (the actionable part): status `selectbox` with the four states color-coded per the palette, a notes `text_area`, a "Requiere follow-up" checkbox, a Save button (navy), and a muted `actualizado: <timestamp>` caption. Saving persists to the store atomically.
- Case list (sidebar or top select): open cases, each row `CASE-… · reason · ● status-color`. Selecting loads the two-pane view.

## 7. Status system (the only place color is loud)

| State | Dot/chip color | es label | pt label |
|---|---|---|---|
| Open | `--idle-500` grey | Abierto | Aberto |
| In progress | `--active-600` blue | En gestión | Em andamento |
| Resolved | `--ok-600` green | Resuelto | Resolvido |
| Follow-up required | `--warn-600` amber | Requiere seguimiento | Requer acompanhamento |

## 8. Hard "don'ts" (to avoid the AI-generic look)

- No purple/violet, no gradient backgrounds, no glassmorphic blur, no neon glow.
- No emoji used as functional UI (a single small check/dot as a status glyph is fine; no decorative emoji).
- No full-width hero banner; no everything-centered landing except the login card.
- No drop-shadow stacking; no rounded-pill buttons; no rainbow of accent colors.
- Don't invent data to fill space; empty sections read "—" in `--ink-500`.

## 9. Implementation notes (ponytail — reuse the platform)

- Streamlit theme via `.streamlit/config.toml`: `primaryColor = "#1C4E80"`, `backgroundColor = "#F6F8FA"`, `secondaryBackgroundColor = "#FFFFFF"`, `textColor = "#14202B"`, `font = "sans serif"`. (Config, not a dependency.)
- One scoped `st.markdown("<style>...</style>", unsafe_allow_html=True)` block per app for bubbles, chips, status dots, header — kept small, tokens inlined from section 2. No CSS framework, no JS.
- Logo via `st.image("ui/assets/cora_logo.png", width=...)` with a styled-text fallback if the file is missing.
- All copy es/pt through the existing `ui_text` map; status labels from section 7.
