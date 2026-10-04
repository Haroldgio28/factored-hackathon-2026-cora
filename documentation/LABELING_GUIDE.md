# Intent labeling guide (CORA gold set)

**Task:** 3.1 · **Requirements:** REQ-30 (valid labels), REQ-37 (provenance) · **Spec:** `cora`
**Gold set:** `data/nlu/gold.tsv` (Spanish, `build_gold.py`) + `data/nlu/gold_pt.tsv`
(Portuguese, task 3.2, `build_gold_pt.py`).

This guide is the authoring contract for CORA's intent gold set. It defines the 16 classes, the
anchor examples per Spanish variant, the provenance scheme, and the double-labeling protocol used
to report inter-annotator agreement. Code and this guide are in English; the utterances themselves
are customer-facing Spanish (and Portuguese at task 3.2).

## Why the gold set is team-written (not mined from the dataset)

The LATAM Bank transcripts are **templated**: EDA finding F7 counts only **42 distinct** customer
utterance strings across ~26k rows, and `detected_intents` is `consulta_general` in ~95% of them.
Training or testing an intent classifier on that would leak and score a meaningless ~100%. So the
gold set is **authored by the team**. The 42 templated strings may be used as *seed inspiration*
only (provenance `dataset-seed`); by rule a `dataset-seed` row never enters a split and the loader
(`cora.nlu.goldset.load`) rejects any such row outright. Every row in the committed `gold.tsv` is
therefore `team-generated`.

## The 16 intent classes

The label set is **not** defined here — it is `cora.policy.engine.Intent`, the exact vocabulary the
deterministic policy engine already decides on. The classifier emits one of these; it decides
nothing. Each anchor below is tagged with its country variant (MX = Mexican, CO = Colombian,
AR = Argentine Spanish; BR = Brazilian Portuguese, added at task 3.2).

| Class | Meaning | Anchor examples (variant) |
|---|---|---|
| **I1** | Balance / available credit | "¿Cuánto dinero tengo en mi cuenta?" (MX) · "¿Me puede decir cuánta plata tengo disponible?" (CO) · "¿Cuál es mi saldo hoy?" (AR) |
| **I2** | Recent transactions / search | "Quiero ver mis compras de esta semana" (MX) · "Necesito el historial de transacciones del mes" (CO) · "Mostrame los últimos movimientos" (AR) |
| **I3** | Explain a transaction's status | "¿Por qué aparece pendiente este cargo?" (MX) · "¿Por qué me declinaron esta compra?" (CO) · "¿Esta operación ya se acreditó?" (AR) |
| **I4** | Card status / limit / days past due | "¿Cuál es el límite de mi tarjeta?" (MX) · "¿Cuántos días de mora llevo?" (CO) · "¿Mi tarjeta está activa o bloqueada?" (AR) |
| **I5** | Currency conversion | "¿A cuánto está el dólar hoy?" (MX) · "Quiero pasar 200.000 pesos a dólares" (CO) · "¿Cuál es la cotización del euro?" (AR) |
| **I6** | List products | "¿Qué productos tengo contratados?" (MX) · "Quiero ver el listado de mis productos" (CO) · "¿Cuáles son mis cuentas y tarjetas?" (AR) |
| **A1** | Freeze a card (confirmation required) | "Por favor congela mi tarjeta" (MX) · "Bloquee mi tarjeta, creo que la perdí" (CO) · "Suspendé la tarjeta mientras la encuentro" (AR) |
| **A2** | Unfreeze a card (confirmation required) | "Reactiva mi tarjeta de crédito" (MX) · "Descongele mi tarjeta, la necesito usar" (CO) · "Habilitá de nuevo mi tarjeta" (AR) |
| **E1** | Dispute / unrecognized charge | "No reconozco esta compra, quiero disputarla" (MX) · "Hay un cobro que yo no realicé" (CO) · "Aparece un consumo que no es mío" (AR) |
| **E2** | Complaint | "Quiero poner una queja por el pésimo servicio" (MX) · "Deseo presentar un reclamo formal" (CO) · "Me quiero quejar de la atención" (AR) |
| **E3** | Suspected fraud | "Alguien usó mi tarjeta sin permiso, es un fraude" (MX) · "Sospecho que me clonaron la tarjeta" (CO) · "Me llegó un cargo del exterior que no hice" (AR) |
| **E4** | Human request / repeated failure | "Quiero hablar con una persona, no con un bot" (MX) · "¿Me puede comunicar con un agente?" (CO) · "Pasame con un asesor humano" (AR) |
| **X1** | Credit eligibility / limit increase (out of scope) | "Quiero que me suban el límite de mi tarjeta" (MX) · "¿Califico para un crédito?" (CO) · "¿Soy elegible para una hipoteca?" (AR) |
| **X2** | Money movement (refused) | "Transfiere 2000 pesos a la cuenta de mi hermano" (MX) · "Hágame una transferencia a esta cuenta" (CO) · "Pagá mi factura desde mi saldo" (AR) |
| **X3** | Anything else in-domain but unsupported | "¿Dónde queda la sucursal más cercana?" (MX) · "¿Cuál es el horario de atención?" (CO) · "¿Cómo cambio la contraseña de la app?" (AR) |
| **OTHER** | Off-domain / small talk / gibberish | "Hola, ¿cómo estás?" · "Gracias, muy amable" · "asdf qwerty" |

### Boundary rules (how to disambiguate the hard pairs)

- **I1 vs I4** — "cuánto disponible **tengo**" (money I can spend now) is I1; "cuál es el **límite /
  cupo** de la tarjeta" (the card's ceiling / its state) is I4.
- **I3 vs I2** — a question about *one* charge's state ("¿por qué está pendiente?") is I3; a request
  to *list/search* movements is I2.
- **E1 vs E3** — "no reconozco este cargo" with no theft claim is a dispute (E1); any mention of
  theft, cloning, hacking or unauthorized access is fraud (E3). When both read plausibly, prefer the
  safer **E3** (it escalates with higher priority).
- **A1 vs A2** — freeze/block/suspend is A1; unfreeze/unblock/reactivate/enable is A2.
- **X1, X2** are out of scope by rule (credit eligibility, money movement) and **must** be labeled
  as such so the policy engine can route them out — never relabel them as a servicing read.
- **X3 vs OTHER** — a banking question CORA simply does not serve (branch hours, open an account) is
  X3; non-banking chatter or noise is OTHER.

## Provenance scheme (REQ-30 / REQ-37)

Every row carries a `provenance` from a closed vocabulary (`cora.nlu.labels.Provenance`):

| provenance | meaning | may enter a split? |
|---|---|---|
| `dataset-seed` | paraphrased from one of the 42 templated transcripts (EDA F7) | **No — seed only** |
| `team-generated` | authored from scratch by the team (es MX/CO/AR, or native-style pt) | Yes |
| `translated` | ES→PT translation of a team row (task 3.2), labeled as such | Yes (labeled) |

The loader fails closed on any unknown provenance and on any `dataset-seed` row, so the "seed only"
rule is enforced in code, not just documented.

## TSV schema

`data/nlu/gold.tsv` is tab-separated (utterances contain commas). Columns, in exact order:

```
id  scenario_id  utterance  intent  language  variant  provenance  month  double_labeled  label_b  reviewer_note
```

- `scenario_id` groups paraphrases of the *same underlying request* (same intent, imagined
  situation) across variants and translations. It is the synthetic stand-in for `customer_id`: the
  leakage-safe split (task 3.3) holds a whole `scenario_id` out so no paraphrase straddles
  train/test (REQ-31). It is its own column, not parsed from `id`.
- `month` is the **authoring wave** (e.g. `2026-10-W1`), the time-order key the split uses (earlier
  waves → train, later → test). The gold set is synthetic, so there is no real calendar timestamp.
- `language` ∈ {`es`, `pt`}; `variant` ∈ {`MX`, `CO`, `AR`} for `es`, `BR` for `pt`.
- `double_labeled` is a boolean token; `label_b` holds the second annotator's label when (and only
  when) the row is double-labeled.

## Double-labeling protocol and Cohen's kappa

To report inter-annotator agreement (REQ-30), a deterministic **~20%** of rows are independently
labeled by a second annotator (`label_b`). Most agree with the primary label; a controlled subset
of genuinely ambiguous phrasings carry a *disagreeing* second label drawn from the confusable pairs
above, so the agreement figure is real, not a hard-coded 1.0.

Agreement is **Cohen's κ** on the double-labeled subset, computed by `cora.nlu.goldset.cohen_kappa`
(a direct counting formula — no ML dependency at 3.1):

```
κ = (p_o - p_e) / (1 - p_e)
```

where `p_o` is observed agreement and `p_e` is chance agreement from each annotator's marginals.

**Measured on the committed `gold.tsv`:** 240 rows, 16 classes (15 per class, balanced), 48
double-labeled rows (20.0%), **Cohen's κ = 0.733** — "substantial" agreement on the Landis-Koch
scale. Reproduce with `python data/nlu/build_gold.py` (prints row count, double-labeled count and
κ) or `cora.nlu.goldset.cohen_kappa(load("data/nlu/gold.tsv"))`.

## Size and the escalation classes

The gold set targets a balanced ~15 utterances per class (240 total for Spanish). This is a
deliberately small, CPU-friendly set for the ~4 GB dev host: a calibrated linear model over frozen
multilingual embeddings (task 3.5) fits well at this size, and every number is reported as *offline
measurement* (REQ-47). The escalation classes (E1–E4) and the out-of-scope guards (X1, X2) are kept
at the same per-class size as the readable ones — a missed escalation is unsafe, so recall on those
classes is the metric that matters and they are **not** left thin. If a class proves hard at this
size, the model report (task 3.6, `documentation/reports/intent_model.md`) says so rather than
inflating the set.

## Language coverage & limitations

**Spanish (`data/nlu/gold.tsv`)** is the primary set: 240 team-authored rows across MX/CO/AR, 15
per class. **Portuguese (`data/nlu/gold_pt.tsv`, task 3.2)** mirrors it 1:1 — 240 rows, all
`variant=BR` — authored by `data/nlu/build_gold_pt.py`:

- **192 rows (80%)** are an **ES→PT translation** of the Spanish set, `provenance=translated`. Each
  carries `reviewer_note = "ES->PT translation of <source scenario_id>"`.
- **48 rows (20%)** are **native-style BR rewrites** (`provenance=team-generated`) — idiomatic
  phrasings a Brazilian would actually type (e.g. "Faz um Pix de 2000 reais pro meu irmão" for the
  money-movement class), not a literal rendering. Each carries
  `reviewer_note = "native-style BR rewrite of <source scenario_id>"`.

Each PT `scenario_id` is `S-<intent>-BR-<source-variant>-<idx>`, a BR derivative of its Spanish
source `S-<intent>-<source-variant>-<idx>`, so the es↔pt pairing is a checkable column relation.
`cora.nlu.goldset.check_pairing(es, pt)` fails closed if any Spanish scenario has no Portuguese
counterpart. Double-labeling mirrors the Spanish source (same 20% of rows, same `label_b`), so
Cohen's κ is comparable across languages — **measured κ = 0.733 on both sets**.

### Documented limitation (REQ-18 / REQ-19)

The dataset is **Spanish-only** (EDA F7); there is no native Portuguese source data. The Portuguese
gold set is therefore **synthetic**: machine/assisted ES→PT translation reviewed by the team, plus a
20% native-rewrite slice to counter translationese. Two honesty caveats carry forward to the model
report:

1. **Translationese bias.** 80% of PT rows follow Spanish sentence structure. The multilingual
   embedding (task 3.5) is expected to transfer well, but a Portuguese-native speaker would phrase
   some requests differently; the 20% native slice only partially offsets this.
2. **Per-language metrics must be read with their sample size.** PT results are reported **separately
   from ES** in `documentation/reports/intent_model.md` (task 3.6) with the per-language, per-class
   sample sizes, and are labeled *offline measurement on a synthetic translated set* (REQ-47) — not
   as evidence of production Portuguese performance. Closing this gap needs native PT utterances,
   recorded as future work.
