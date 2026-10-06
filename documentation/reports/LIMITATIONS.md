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

## 5. Live read-ownership for non-referenced reads + FX conversion — RESOLVED

**Resolved** on branch `fix/live-read-ownership-fx` *(offline measurement; new unit tests, suite
green)*. This was a real Phase-4 agent-correctness gap; it is fixed as of the live
read-ownership/FX fix.

- **What was wrong.** A READ intent with **no specific `product_id`** — **I6 "list my products"**
  (served by `list_products`), or a bare "mi cuenta" with no resolved reference — abstained even
  though the classifier was correct (I6 confidence ≈ 0.72), Bedrock worked and the data existed.
  Separately, **FX conversion (I5)** abstained because `convert_currency` was **never invoked** by
  the orchestrator graph.
- **Root cause.** In `src/cora/agent/graph.py` `_build_policy_input`, `resource_owned` was set
  `True` **only** when `intent in _READ_INTENTS and product_id is not None` **and** a
  `get_balance(product_id)` for that one product returned `Status.OK`. A read with no `product_id`,
  and every FX turn, left `resource_owned=False`, so **POL-090 (`read_resource_owned → answer`)
  never matched** and the turn fell through to **POL-999 (abstain)**.
- **The fix.** `_build_policy_input` now resolves ownership for the two missing shapes, and the
  orchestrator **renders the grounded answer** from the same-turn tool results so a POL-090 turn
  returns a real customer-visible reply (not an empty response). Identity is still never taken
  from model/user text:
  - **Non-referenced reads** — scoped to the two shapes that have an account-level tool and
    renderer: **I6 "list my products"** and a **bare I1 "mi saldo"**. For I6 **any** successful
    `list_products()` sets `resource_owned=True` and its `Result` feeds grounding and the handoff —
    an **empty but authorized** list is itself a grounded answer ("you have no products", rendered
    in es/pt), per the I6 answer contract, not an ownership failure. For a bare I1 ownership is
    gated on the **balance** read: with a
    **single** owned product its balance is read and shown (and a balance read that comes back
    `UNAVAILABLE` fails closed to tool-unavailable, never a stand-in product list); with
    **several** products the request is ambiguous, so the turn **clarifies (POL-070)**. A
    non-referenced **I2–I4** has no account-level tool/renderer, so it stays not-owned and
    **abstains** rather than being answered with an unrelated product list.
  - **FX conversion (I5):** `convert_currency(...)` is called from the proposed amount + source /
    target currencies (entity extraction, treated as data; `to_currency` is in the entity schema).
    An OK rate sets `resource_owned=True`; **incomplete entities** mark the entity ambiguous so the
    turn **clarifies (POL-070)**. The target currency must be **unambiguous**: an unqualified
    "pesos" (ambiguous across MXN/COP/ARS) is extracted as `to_currency=null` per the extraction
    prompt, so **deterministic** code — not a model guess — routes it to POL-070. A >7-day-stale
    rate fails closed to the honest tool-unavailable path. When no rate exists for the requested
    date and the **latest prior** rate (within 7 days) is used, the answer **states it** to the
    customer in es/pt (REQ-28), driven by the tool's `used_prior_rate` flag.
  - **Answer rendering (deterministic, same-turn results):** the `{facts}` string is assembled
    **deterministically** from the OK `Result`s gathered this turn and a factual answer is rendered
    **template-only** (`polish=False`): the ANSWER template is those grounded facts verbatim, so no
    LLM rewrites a displayed value and every figure comes from a tool result in the same turn
    (REQ-08). This is the stronger control: the grounding checker validates only numbers, dates and
    statuses, so an LLM rephrase could keep every number while swapping a currency code (USD→COP) —
    skipping polish removes that attack surface entirely. The checker still runs on the paths that
    DO attempt LLM polish (non-factual outcomes); it is simply not needed for a template-only
    factual answer, which is grounded by construction.
  Ownership is **not** weakened: a FORBIDDEN/NOT_FOUND product read still denies, and a tool outage
  fails closed to `TOOL_UNAVAILABLE` via `_safe_read`, never a false "not owned".
- **Eval alignment.** `src/cora/eval/builder._expected_outcome` was updated to derive the reference
  outcome with the **same deterministic tool-ownership logic** (list_products for I6 — an OK list,
  even empty, is `answer`; the gated balance read for a bare I1; convert_currency for I5; I2–I4
  stay abstain), still fed through `PolicyEngine.decide` as an independent derivation — so the
  suite now scores the correct `answer` for I5/I6 and an owned single-product I1 while I2–I4 keep
  abstaining. The committed workload (`data/eval/scenarios.jsonl`) was updated so the empty-list
  I6 case (`normal-037`) scores `answer`, matching the corrected builder. The runner now presents
  each **adversarial condition to both compared configurations** (REQ-41 same workload): a
  `force_tool_fault` case forces every read to fail for **B1 and CORA**, and an
  `unauthorized_access` case makes **both** attempt the same foreign read through their
  session-bound tool layers (which still deny it) — the comparison is no longer applied to CORA
  only. The builder's
  date-sensitive FX reference and the runner now share one evaluation clock (`EVAL_NOW` in
  `src/cora/eval/scenarios.py`), so the expected and produced conversions resolve the **same** rate
  row rather than two different dates.
- **Scope.** Pure ownership-resolution / wiring fix in the orchestrator and the matching reference
  builder; no classifier, LLM, policy-rule or data change.

<!-- ponytail: this file is the consolidation point; new gaps get one labelled line here, not a new doc. -->
