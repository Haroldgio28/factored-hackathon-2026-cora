---
inclusion: always
---

# Product: CORA

CORA (*Customer-Oriented Resolution Agent*) is an AI-first banking customer-service assistant for
the synthetic LATAM Bank (Mexico, Colombia, Argentina), built for the Factored AI & Data Hackathon 2026.

- **Workflow (only one):** account & card servicing - balances, transactions, card status/limits,
  FX conversion, product list; card freeze/unfreeze with confirmation; disputes, complaints, fraud
  and human requests escalate with a handoff package. Credit eligibility and money movement are out
  of scope by rule.
- **Users:** bank customers (es/pt) and human agents receiving handoffs.
- **What wins:** depth, demonstrated behavior and engineering judgment across 6 axes - data-backed
  problem, functioning AI, controlled automation, data/ML rigor, measured failure handling, route to operation.
- **Source of truth:** `.kiro/specs/cora/` (requirements -> design -> tasks), `documentation/SOURCE_REQUIREMENTS.md`,
  `documentation/TRACEABILITY.md`, `CORA_CONTEXT.md`. Never add a capability without a REQ and a matrix row.
- Original PDFs live in `Docs/` (gitignored, contain credentials): never read secrets from them into code or docs.
