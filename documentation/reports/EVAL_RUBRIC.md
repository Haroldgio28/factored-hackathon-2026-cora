# LLM-judge rubric — tone, clarity, handoff usefulness (REQ-45)

**Label: `simulation`.** This document defines the rubric the CORA evaluation's *LLM judge* applies.
The LLM judge scores **only the style** of a produced answer. It never scores whether facts, figures
or bank policy are correct — those are decided by the **deterministic judges (task 6.4), which are
pure code** — so a model opinion can never flip a safety-relevant outcome.

## What the judge scores

Each produced answer is scored on three ordinal dimensions, integer **1–3**:

| Field | 1 | 2 | 3 |
|-------|---|---|---|
| `tone` | cold or inappropriate | acceptable | warm and professional |
| `clarity` | confusing | understandable | clear and direct |
| `handoff_usefulness` | useless to a human agent | partially useful | summarizes the case well for the agent |

The reply must be a JSON object carrying **exactly** these three keys, each an integer in `{1,2,3}`.
Anything else (non-JSON, a missing or extra key, an out-of-range value) is treated as **malformed and
fails closed**: no score is recorded for that answer (`available = false`), never a guessed value.

## What the judge must NOT do

- It does **not** judge factual correctness, numeric accuracy, or policy compliance (6.4 owns those).
- It is shown only **PII-masked** text (`mask_pii` runs before the LLM call, security steering / REQ-36).
- Its output is **never fed back into a decision**; it is a reported quality signal only.
- The model id comes from settings/injection (`CORA_BEDROCK_JUDGE_MODEL_ID`), **never a hard-coded id**.

## Validation — silver labels, not human labels

REQ-45 asks for judge agreement against **human** labels. Per the owner decision (2026-10,
`.agents/tasks/phase6-6.5-owner-decision.md`), CORA validates the base judge against **reference
("silver") labels produced by a more robust LLM** — the owner points `CORA_BEDROCK_JUDGE_MODEL_ID` at
a higher-capability model for the reference run. Agreement is reported per field as **Cohen's κ** (the
same counting formula the gold set uses, `nlu.goldset.cohen_kappa_pairs`) plus **percent agreement**,
over the ≥50 samples both judges could score.

> Agreement is judge-vs-robust-LLM silver labels, not human validation; the 50 human-labelled samples
> REQ-45 asks for remain a documented limitation.

Every validation figure is labelled `simulation`. The agreement artifact is written to
`data/eval/judge_agreement.json`. The real base + robust reference run is executed by the owner with
`.env` configured (`CORA_BEDROCK_JUDGE_MODEL_ID` pointing at the robust model); the harness and its
tests run offline against `StubLLMClient`.
