# CORA limitations — honest gaps (task 7.4, REQ-19 / REQ-47)

A single consolidated list of every known gap in the CORA prototype, pulled together from the
phase reports rather than restated. Each item is labelled per REQ-47 as **offline measurement**,
**simulation** or **projection**, and points at the artifact that already carries the detail.
Nothing here is hidden behind an optimistic SLA; no number is re-estimated.

Source artifacts referenced (not duplicated):
[`EVALUATION.md`](EVALUATION.md), [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md),
[`intent_model.md`](intent_model.md), [`DECISIONS.md`](../DECISIONS.md) (ADR-004),
[`model_card.json`](../../data/nlu/model_card.json).

## 1. Data & language limitations (REQ-19)

- **The dataset is Spanish-only.** Portuguese coverage is **team-generated / translated**, not
  organizer data: 192 translated + 48 native rewrites, each gold row tagged with its provenance
  (`cora.nlu.labels.Provenance`). PT therefore carries a **translationese caveat** — translated
  utterances skew toward source-language phrasing and under-represent native BR idiom, so PT
  behaviour is not validated against real PT traffic. *(offline measurement — see `model_card.json`
  baseline/intent results and the SRC-20/SRC-58 traceability rows.)*
- **Transcripts are synthetic and templated.** The scenario suite is built from curated templates,
  not captured conversations; the gold evaluation set is **team-written**. Realistic phrasing drift,
  code-switching and messy real-world input are under-represented. *(simulation — see
  `src/cora/eval/builder.py` and `EVALUATION.md`.)*
- **Silver labels, not ≥50 human labels.** Per the owner decision (task 6.5), the LLM-judge
  reference labels for the subjective tone/clarity/handoff dimensions are **robust-LLM silver
  labels**, not a human-validated set. A human-labelled validation of ≥50 samples **remains a
  documented gap**; subjective quality is therefore unmeasured against human ground truth.
  *(simulation — see `EVALUATION.md` and the SRC-68 row.)*

## 2. Measurement & sample-size limitations (REQ-47)

- **Small samples; reduced evaluation run.** The real B1-vs-CORA run used a **98-scenario
  stratified subset (49 es / 49 pt) at `--repeats=1`**, below the project's **≥3-repeats** rule and
  the full ~400-case suite. With one repeat there is no run-to-run variability (mean ± sd shows
  `n=1`), and the per-type **Wilson 95% CIs are wide** — e.g. CORA `disclosure_leak` 0/15 reports a
  rule-of-three upper bound of **0.200**, and B1 `disclosure_leak` 1/16 spans **[0.011, 0.283]**.
  Treat all reported rates as a reduced offline measurement, not a production baseline.
  *(offline measurement — see `EVALUATION.md`.)*
- **Cost is unmeasured.** `CORA_PRICE_*` is unset and the runner captures no provider token usage,
  so cost per attempted case and per SAR both render as `$0.000000` / "not defined". When priced,
  cost is an **output-tokens-only estimate** from text length against supplied published rates.
  *(projection — see `EVALUATION.md` cost assumptions.)*

## 3. Encoder / model provenance limitation (REQ-47)

- **MiniLM weights were not downloadable in the sandbox**, so some Phase-3 numbers were produced
  with a **stub encoder**. Two distinct runs must not be collapsed:
  - **Stub-era run (commit `457f512`).** The embedding head used `StubEncoder` (HF SSL blocked),
    while the download-free char-ngram **TF-IDF head is real**: macro-F1 **0.9321** on the
    leakage-safe test split. These were the numbers available without the real 384-d encoder.
  - **Later real-MiniLM run (host with Hugging Face access, current `model_card.json`).** The
    **real multilingual MiniLM path was confirmed**: macro-F1 **0.9354**, accuracy 0.942, 69-case
    test split. That rerun also regenerated the TF-IDF comparison on the same 69-case split
    (TF-IDF macro-F1 **0.917**), so the TF-IDF figure differs by run and should not be read back
    onto the stub era.

  Figures produced without the real 384-d encoder are labelled accordingly and cannot yield
  operational-threshold or intent-dependent numbers. *(offline measurement — see
  `model_card.json.intent_model_results` (`encoder_provenance`) and `intent_model.md` §3.)*

## 4. Architecture & scope limitations

- **Orchestration is a stdlib dispatcher, not LangGraph.** ADR-004 named LangGraph, but it was not
  installable in the environment, so transitions are a deterministic stdlib state dispatcher. The
  "each transition is code, not prompt" invariant holds identically; swapping LangGraph in later is
  mechanical. *(offline measurement — see `DECISIONS.md` ADR-004 and `PRODUCTION_READINESS.md` §6.)*
- **freeze / unfreeze is modeled via a sandbox overlay.** The underlying product data has no native
  freeze/unfreeze field, so card status is simulated through an overlay rather than a real product
  mutation. The Confirm→Execute→Verify protocol and read-back are real; the state it toggles is
  overlay-backed. *(simulation — see `src/cora/tools` card overlay and `TOOL_CONTRACTS.md`.)*

## 5. Known agent-correctness gap — live read-ownership for non-referenced reads

Labelled **honest known limitation** *(offline measurement; live-observed in a manual demo run)*.
This is a real Phase-4 agent-correctness gap, scheduled to be fixed **after Phase 7**.

- **Symptom.** A READ intent with **no specific `product_id`** — e.g. **I6 "list my products"**
  (which uses `list_products`), or a bare "mi cuenta" with no resolved reference — never gets a safe
  answer; the turn abstains even though the classifier is correct (I6 confidence ≈ 0.72), Bedrock
  works, and the data exists.
- **Root cause.** In `src/cora/agent/graph.py` `_build_policy_input`, `resource_owned` is set to
  `True` **only** when `intent in _READ_INTENTS and product_id is not None` **and** a
  `get_balance(product_id)` for that **one specific product** returns `Status.OK`. Ownership is thus
  established solely per-product via a successful balance read. A read with no `product_id` leaves
  `resource_owned=False`, so **POL-090 (`read_resource_owned → answer`) never matches** and the turn
  falls through to **POL-999 (abstain)**.
- **Why the eval doesn't surface it.** The suite *does* exercise non-referenced reads — 396/400
  committed scenarios carry no `product_id`, including all 30 I6 cases. The gap is hidden because
  the deterministic reference-outcome builder **encodes the current abstention as the expected
  outcome**: `src/cora/eval/builder._expected_outcome` only sets `resource_owned=True` when
  `product_id is not None and intent in _READ_INTENTS` and the balance read succeeds, so an I6 case
  derives `resource_owned=False`, and its expected decision is `abstain`. The suite therefore scores
  the abstention as *correct* rather than flagging it, so the held-out run never surfaces the defect.
- **Scope.** Purely an ownership-resolution gap for non-referenced reads — not a classifier, LLM or
  data defect. The fix (resolve ownership for product-list / account-level reads without requiring a
  single `product_id`) is a Phase-4 correctness item deferred to after Phase 7.

<!-- ponytail: this file is the consolidation point; new gaps get one labelled line here, not a new doc. -->
