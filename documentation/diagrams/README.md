# Diagrams

Interactive, self-contained HTML diagrams generated with [archify](https://github.com/tt-a1i/archify)
from the typed JSON specs in this folder. Each `*.sequence.json` is the source of truth; the
matching `*.html` is the rendered artifact (open it in a browser - pan/zoom, theme toggle,
guided views and PNG/SVG export are built in).

## Phase 2 - Identity, tools & policy

Both diagrams trace the same real example - a customer asking to **freeze a card** - which exercises
all four Phase 2 subsystems in order: identity (2.1), policy (2.4), the authorized tool layer (2.2)
and the sandbox overlay with a verified write (2.3).

| File | Audience | What it shows |
|---|---|---|
| `phase2-confirmation-flow-business.*` | Non-technical / stakeholders | Plain-language story: verify identity -> the rulebook (not the AI) decides to ask for confirmation -> apply only after an explicit yes -> re-check before telling the customer. |
| `phase2-confirmation-flow-technical.*` | Engineers / reviewers | Code-level flow: `PolicyEngine.evaluate` -> `ToolLayer.freeze_card` with session-injected `customer_id`, `_load_owned_product` (FORBIDDEN on foreign), `Confirmation.is_valid_for`, idempotent `CardOverlay.put` + read-back, and the `Result` envelope. Maps to REQ-09/12/13/14. |

## Phase 3 - NLU & learned component

One data-flow diagram tracing how the learned intent component is built and served, from the
team-written es/pt gold set through the leakage-safe split, the calibrated classifier and the
frozen cost-matrix thresholds, to serve time where it fills exactly three `PolicyInput` fields and
the deterministic policy engine decides.

| File | What it shows |
|---|---|
| `phase3-nlu-dataflow.*` | Five stages (gold data -> leakage-safe split -> learn & compare -> freeze evidence -> serve). Gold sets (3.1/3.2) -> `make_splits` (3.3) -> MiniLM+LogReg calibrated head + TF-IDF comparison (3.5) -> `thresholds.json` frozen on validation (3.6) -> runtime `mask / language / injection` screens (3.7) feeding `intent, confidence, injection_hit` into `PolicyEngine`. Maps to REQ-29/31/32/35/47 and the "learned proposes, policy decides" security rule. |

### Regenerating

From a checkout with the archify skill available:

```bash
# validate the spec, then render + verify the HTML
node <archify>/bin/archify.mjs validate sequence phase2-confirmation-flow-business.sequence.json --quality showcase
node <archify>/bin/archify.mjs deliver  sequence phase2-confirmation-flow-business.sequence.json phase2-confirmation-flow-business.html --quality showcase
node <archify>/bin/archify.mjs validate sequence phase2-confirmation-flow-technical.sequence.json --quality showcase
node <archify>/bin/archify.mjs deliver  sequence phase2-confirmation-flow-technical.sequence.json phase2-confirmation-flow-technical.html --quality showcase

# Phase 3 data-flow diagram
node <archify>/bin/archify.mjs validate dataflow phase3-nlu-dataflow.dataflow.json --quality showcase
node <archify>/bin/archify.mjs deliver  dataflow phase3-nlu-dataflow.dataflow.json phase3-nlu-dataflow.html --quality showcase
```

The Phase 2 specs pass `--quality showcase` validation (9/9 artifact checks, 0 composition
errors/warnings) and bounded desktop `visual-check` (containment + readability at 1440x900,
1600x1000, 1920x1080, 2048x1320).

The Phase 3 data-flow spec passes `--quality showcase` validation (0 composition errors/warnings;
node legibility, corridors and labels all clear). Its `visual-check` reports vertical overflow at
the desktop viewports (the five-stage flow plus three summary cards is taller than a 900px window),
so the standalone HTML scrolls vertically on a small desktop; the diagram panel itself is contained
horizontally and legible. Perceptual review of the rendered HTML is still a human step.
