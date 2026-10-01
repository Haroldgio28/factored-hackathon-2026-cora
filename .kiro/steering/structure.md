---
inclusion: always
---

# Repository structure

```
.kiro/specs/cora/   requirements.md, design.md, tasks.md
.kiro/steering/     standing rules (this folder)
documentation/      SOURCE_REQUIREMENTS, TRACEABILITY, DECISIONS, DEVELOPMENT_PLAN, reports/
analysis/           EDA scripts, figures, findings
src/cora/           data/ identity/ tools/ policy/ nlu/ agent/ handoff/ api/ eval/ obs/
ui/                 Streamlit customer + agent views
tests/              unit, contract, fixtures/incremental (TEAM-GENERATED)
infra/              AWS CDK (later)
Docs/               original PDFs - gitignored, never commit
```

- Versioned docs go in `documentation/`, never in `Docs/` (Windows is case-insensitive; `Docs/` is ignored).
- Generated artifacts (figures, reports, metrics JSON) are committed when they are evidence for the submission.
- Prompts are versioned files under `src/cora/agent/prompts/`; their hash is recorded in traces.
- Policy rules live in `src/cora/policy/rules.yaml`, each with a stable rule id.
