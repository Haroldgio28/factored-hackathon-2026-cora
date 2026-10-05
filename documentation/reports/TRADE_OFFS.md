# CORA trade-off register & intended outcomes (Phase 7)

This document makes two things explicit for the jury: (REQ-02) the customer and business
**outcomes CORA is built to improve, with numeric success targets** compared to
baselines; and (REQ-03) the **trade-offs** CORA deliberately takes and **why each component is AI or
deterministic**. It does not restate numbers it does not own: measured figures come from
[`EVALUATION.md`](EVALUATION.md), the design rationale from [`../DECISIONS.md`](../DECISIONS.md) and
[`../../ARCHITECTURE.md`](../../ARCHITECTURE.md), and the demand/FCR evidence from
[`../../analysis/EDA_FINDINGS.md`](../../analysis/EDA_FINDINGS.md).

> **Honesty note on the measured run.** Every CORA/B1 figure below is `offline measurement` from the
> 2026-10-05 held-out run, which executed the **98-scenario stratified subset (49 es / 49 pt) at
> `--repeats 1`** — below the evaluation rule of **≥3 repeats** and the **full 400-case** suite
> (Bedrock quota throttling, documented in `EVALUATION.md`). Targets are therefore compared to a
> *reduced* measurement; a full `--repeats 3` run on higher quota is the pending step, not a changed
> conclusion. B0 is a **historical human-agent baseline per `reason_category`, not 1:1 with CORA's
> SAR** (different population, full range of human-handled contacts).

---

## 1. Intended outcomes & success targets (REQ-02)

REQ-02 asks for the customer and business outcomes CORA aims to improve, with explicit numeric
success targets. The targets below are the operating bar CORA is held to; the "measured (reduced
run)" column is the 2026-10-05 subset, shown for honesty, not as the full-suite verdict.

> **Provenance of the targets (honesty note).** These exact numeric targets are **provisional
> operating bars documented in this Phase-7 artifact (post-evaluation)**, not prospective criteria
> frozen in a versioned spec before the Phase-6 run — no such pre-evaluation artifact exists in the
> repository history. They are therefore calibrated against, and must not be read as independent of,
> the measured Phase-6 numbers. REQ-02's prospective-target cycle (set targets → *then* measure on
> `--repeats 3` / full suite) remains the pending step: SRC-23 is marked **Done** in
> [`../TRACEABILITY.md`](../TRACEABILITY.md) because this artifact records the outcomes and targets,
> but that pending full-suite / pre-registered-target cycle is the honest limitation on it.

### Customer outcomes

| Outcome | Numeric success target (provisional, post-eval) | Baseline | Measured (reduced run) | Reading |
|---|---|---|---|---|
| Correct answer (no unsafe disclosure) | 0 disclosure leaks; unsafe-disclosure rate ≤ 0.05 | B1 naive LLM: 1/16 leaks | CORA **0/15** leaks (Wilson [0.000, 0.204], rule-of-three upper **0.20**) | Target met on the subset; the 0.20 ceiling is a sample-size limit, not a 0-risk claim. Needs the full suite to tighten the CI. |
| Time-to-resolution (automated turn) | Automated answer in a single turn for in-scope read inquiries (vs human AHT) | B0 human AHT: Transaccional ~221s, Producto ~266s (the automation targets) | CORA p50 **7131.5 ms** / p95 **10161.4 ms** per turn | A grounded single-turn answer replaces a multi-minute human contact for the high-FCR categories; latency is model-bound, well under human AHT. |
| Fewer repeat contacts (right action, verified) | 0 actions reported done without read-back; 0 unnecessary escalations that bounce the customer | B1: reports from model text, no verification | CORA **0 unnecessary escalations**; actions reported only after read-back (Confirm→Execute→Verify) | A verified action and a precise handoff reduce "it didn't work, I'm calling again" repeats. |

### Business outcomes

| Outcome | Numeric success target (provisional, post-eval) | Baseline | Measured (reduced run) | Reading |
|---|---|---|---|---|
| Containment **with correctness** | SAR ≥ 0.35 **while** disclosure leaks = 0 (containment alone is not a goal) | B1 SAR 0.98 but **unsafe** (leaks + misses all escalations) | CORA **SAR 0.40 (24/60)**; containment 0.867 (85/98); 0 leaks | Target met: safe automation over blind automation. B1's 0.98 is over-automation — it contains wrong/unsafe answers. |
| Cost per resolution | **Cost per SAR ≤ US$0.05** — i.e. ≤ 5 US cents of model spend per safe automated resolution (assumptions below) | B0 human: fully loaded agent minutes per AHT above (≈ US$0.37–1.50 per contact at US$6–10/agent-hour × ~221–540s AHT) | CORA **$0.00** (shown because `CORA_PRICE_*` are **unset** — an assumption, not a measured zero) | A 5-cent-per-SAR ceiling keeps CORA roughly 7–30× below the human per-contact cost. The current `$0.00` is a projection with price 0, not a measurement; set `CORA_PRICE_*` to score against the target. |
| Agent time saved | Deflect the high-FCR, low-risk demand (Transaccional+Producto ≈ 57% of volume) from human queues | B0 AHT per category (above); EDA demand mix | Demand mix: Transaccional 35% / Producto 22% / Queja 17% / Técnico 15% / Comercial 8% / Retención 3% | CORA targets the ~57% high-FCR demand for automation and routes the low-FCR Queja (43.6% FCR) to humans — agent time is saved on the automatable half, preserved for the hard half. |

**Why these targets.** The automation targets (Transaccional 91.5% FCR / ~221s AHT; Producto 89.6% /
~266s) and the escalation path (Queja 43.6% FCR / ~435s) come straight from the EDA; CORA is designed
to safely absorb the former and cleanly hand off the latter, not to maximize containment.

**Cost-target assumptions (so the ≤US$0.05/SAR bar is checkable).** Cost per SAR = (model spend over
the run) ÷ (count of safe automated resolutions). Model spend uses the runner's cost model from
`EVALUATION.md`: tokens ≈ characters ÷ `chars_per_token` (4.0), **output tokens only**, priced at the
published Bedrock per-1k output rate once `CORA_PRICE_*` are set (`data/eval/metrics.json`
`cost_assumptions`, currently `0.0`). Workload = the in-scope read/servicing turns CORA attempts (SAR
denominator = 60 on the measured subset). The **US$0.37–1.50 human per-contact** reference is a
projection from fully-loaded agent cost (US$6–10/agent-hour) × the B0 AHT range (~221–540s)
(low end ≈ 6 × 221/3600, high end ≈ 10 × 540/3600), **not a measured figure**; the ≤5-cent bar is
deliberately set roughly 7–30× below it so the economic trade stays favourable even under conservative
token/price assumptions.

---

## 2. Trade-off register (REQ-03)

CORA's chosen operating point is **safe automation over blind automation**. The single clearest
expression of this is SAR: CORA's **0.40** is *lower* than B1's **0.98** on purpose — a lower SAR is
the **intended** trade-off, because B1 reaches 0.98 only by answering (and containing) cases it should
have refused or escalated.

| Axis | B1 (naive) operating point | CORA's chosen operating point | Measured evidence (reduced run) | Why the trade |
|---|---|---|---|---|
| **Autonomy** | Answers almost everything (attempted 1.0, SAR 0.98) | Automates in-scope, available, grounded turns; abstains/escalates otherwise (SAR **0.40**) | SAR 0.40 (24/60) vs 0.98 (59/60) | Lower autonomy is accepted to buy safety; a contained wrong answer is still a failure. |
| **Accuracy / safety** | 1/16 disclosure leaks; misses **all 19** escalations | **0/15** disclosure leaks; escalation precision **1.0**, recall **0.68** | leaks 0/15 (rule-of-three ≤ 0.20); P 1.0 / R 0.68; 0 unnecessary | The whole point: fail closed, never leak, never fabricate an escalation. |
| **Latency** | p50 2191.6 ms / p95 5123.1 ms | p50 **7131.5 ms** / p95 **10161.4 ms** | from `EVALUATION.md` | CORA adds grounding, policy, confirmation and language gating around the model — slower per turn, still well under human AHT. The latency is spent on safety. |
| **Cost** | output-tokens projection | output-tokens projection | **$0.00** because `CORA_PRICE_*` unset (assumption, not a measured zero) | Cost is deferred until prices are set; the extra gating tokens are the cost of the accuracy trade. |
| **Human oversight** | None — contains everything | Deterministic escalation + full REQ-16 handoff package on dispute/complaint/fraud/ambiguity/tool-failure | escalation recall 0.68, **0 unnecessary** escalations | Oversight is routed precisely (precision 1.0): humans get the hard cases with context, not noise. |

Operating thresholds backing this point: **`tau_escalate = 0.0`, `tau_clarify = 0.55`**, selected on
the validation split, **frozen in `data/nlu/thresholds.json`** and **promoted into
`src/cora/policy/rules.yaml`** (never tuned on test).

### 2.1 AI vs deterministic, per component — and why

CORA's rule is **the LLM proposes, deterministic code decides**: identity, authorization, policy and
confirmations never depend on model output. The split below is the engineering expression of that rule.

| Component | AI or deterministic | Why |
|---|---|---|
| NLU intent classification | **AI (learned, calibrated)** | Language understanding is the one place a learned model beats rules; calibrated LogReg over multilingual embeddings gives confidence scores the policy engine can threshold. Its output is a *proposal*, never a permission. |
| Identity / session | **Deterministic** | Authentication is a security boundary; HMAC-JWT with TTL/jti is verifiable and fail-closed — never a model judgement. |
| Authorization (per-customer) | **Deterministic** | `customer_id` comes only from the verified session token; access is a first-match allow-list in code, immune to anything in model or user text. |
| Policy engine | **Deterministic** | Which requests are answerable / confirm-required / abstain / refuse is a first-match rule table with stable POL ids — auditable, not inferred. |
| Confirmation protocol | **Deterministic** | Confirm→Execute→Verify restates the exact action and requires an explicit affirmative in the same unexpired session; the decision to execute is code. |
| Grounding checker | **Deterministic** | Every factual value shown must match a tool result in the same turn; a pure-code number/date/status diff fails closed, discarding ungrounded model text. |
| Response phrasing | **AI-proposed, deterministically gated** | The LLM polishes wording for fluency, then code gates it: grounding check + language guard (discard es↔pt flips) + template fallback. Fluency is the model's job; correctness and language are code's. |

Rationale and alternatives for these choices are recorded in
[`../DECISIONS.md`](../DECISIONS.md) (ADR-004 orchestration, ADR-005 Bedrock behind `LLMClient`) and
the component map in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).
