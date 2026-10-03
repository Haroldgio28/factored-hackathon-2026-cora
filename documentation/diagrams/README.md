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

### Regenerating

From a checkout with the archify skill available:

```bash
# validate the spec, then render + verify the HTML
node <archify>/bin/archify.mjs validate sequence phase2-confirmation-flow-business.sequence.json --quality showcase
node <archify>/bin/archify.mjs deliver  sequence phase2-confirmation-flow-business.sequence.json phase2-confirmation-flow-business.html --quality showcase
node <archify>/bin/archify.mjs validate sequence phase2-confirmation-flow-technical.sequence.json --quality showcase
node <archify>/bin/archify.mjs deliver  sequence phase2-confirmation-flow-technical.sequence.json phase2-confirmation-flow-technical.html --quality showcase
```

Both specs pass `--quality showcase` validation (9/9 artifact checks, 0 composition errors/warnings)
and bounded desktop `visual-check` (containment + readability at 1440x900, 1600x1000, 1920x1080, 2048x1320).
Perceptual review of the rendered HTML is still a human step.
