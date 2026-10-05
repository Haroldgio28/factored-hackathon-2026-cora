# CORA demo script — five paths (REQ-05)

This is the **script** for the demo video. All surrounding prose and labels are English; every
customer-facing utterance and CORA reply is **Spanish or Portuguese** (language steering). The
Spanish lines vary across MX / CO / AR so the demo shows the neutral-LATAM register working for
each variant. Both languages are shown for every path.

> **The video recording is owner-executed, not committed.** A live run needs a valid Bedrock
> bearer token (`AWS_BEARER_TOKEN_BEDROCK`, see [`../AWS_SETUP.md`](../AWS_SETUP.md)) and the data
> landed locally; the owner records and uploads the video separately. The committed deliverable is
> **this `.md` only** — no video file is in the repository. Where a live LLM is unavailable the
> generator falls back to the deterministic template verbatim (REQ-40), so every expected reply
> below is grounded whether or not the model polish ran.

## Conventions for every path

- **Session.** Each path assumes an already-authenticated, unexpired session. Auth is the
  two-step mock OTP on `POST /auth`; the issued short-lived JWT (15-min TTL) carries the
  `customer_id`, which the tool layer reads **only** from the verified token, never from the
  utterance (security steering). The customer views are the Streamlit app
  ([`../../ui/customer_app.py`](../../ui/customer_app.py)) with the es/pt toggle; the human-agent
  view is [`../../ui/agent_console.py`](../../ui/agent_console.py).
- **The LLM proposes, deterministic code decides.** The policy decision and its POL rule id
  ([`../../src/cora/policy/rules.yaml`](../../src/cora/policy/rules.yaml)) are shown on screen next
  to the per-turn `trace_id` ([`../../src/cora/obs/tracing.py`](../../src/cora/obs/tracing.py)) and
  the same-turn tool `Result`(s) every shown figure is read from.
- **What to point at (every turn):** the `trace_id`, the policy decision + POL rule id, and the
  tool `Result`(s) this turn's figures came from. **Factual answers** (balance, FX, product list)
  are rendered **template-only** straight from those same-turn `Result`s — grounded by
  construction, with no LLM rewriting a value — so there is no separate polish/grounding-checker
  step on those turns; the grounding checker applies only where LLM polish is attempted. For
  actions, also point at the masked product number and the read-back verification.

---

## Path 1 — Safe account read (balance of a specific card) — I1 → POL-090 `answer`

**Why a specific product reference.** A read decides `answer` only when `read_resource_owned`
holds (POL-090). `_build_policy_input` in
[`../../src/cora/agent/graph.py`](../../src/cora/agent/graph.py) sets `resource_owned=True` either
when the turn resolves to a **specific** owned product (via `get_balance`) **or**, for a
non-referenced read, at the customer level: **I6** answers on **any** authorized `list_products`
result — including an **empty** one (a grounded "you have no products"), since an empty owned list
is not an ownership failure — and a bare **I1** "¿cuál es mi saldo?" answers when the customer owns
exactly **one** product (its balance is read and shown), otherwise it clarifies (several products) — see
[`LIMITATIONS.md`](LIMITATIONS.md) §5 (resolved). The
demo still uses an **explicit product reference** (a card ending in known digits) so the answered
figure is tied to one named card and the grounding is easy to point at on screen.

- **Setup:** authenticated session; the customer owns a card ending in `1234`.
- **es (MX):** `Hola, ¿me pasas el saldo de mi tarjeta terminada en 1234?`
- **pt:** `Oi, pode me informar o saldo do meu cartão terminado em 1234?`
- **Expected decision:** intent `I1`; `read_resource_owned` true → **POL-090 `answer`**.
- **Expected reply (grounded `ANSWER`, figures from the balance tool `Result`):**
  - es: `Tu saldo es 1.250,75 COP y tu saldo disponible es 1.100,00 COP.`
  - pt: `Seu saldo é 1.250,75 COP e seu saldo disponível é 1.100,00 COP.`
- **Point at:** the balance tool `source_refs` (table `products`), POL-090, the `trace_id`, and
  that the reply is rendered **template-only** from the same-turn balance `Result` — the figures
  are the tool's values verbatim, grounded by construction (no LLM rewrites a factual answer, so
  there is no polish/grounding-checker step to show for this turn).

---

## Path 2 — Card freeze: confirm → execute → verify — A1 → POL-080 `confirm`

Exercises the Confirm → Execute → Verify protocol
([`../../src/cora/agent/confirmation.py`](../../src/cora/agent/confirmation.py)): CORA restates the
exact action and the **masked** card number, requires an explicit affirmative in the same unexpired
session, then reports success **only** after the tool's read-back confirms the post-condition
(REQ-09/REQ-14).

- **Setup:** authenticated session; the customer owns a card ending in `5678`, currently active.
- **Turn 1 — request (es, CO):** `Buenas, necesito congelar mi tarjeta terminada en 5678, creo que la perdí.`
  - **pt:** `Boa, preciso congelar o meu cartão terminado em 5678, acho que perdi.`
  - **Decision:** intent `A1`; `action_product_owned_and_state_allows` → **POL-080 `confirm`**.
  - **CORA reply (`CONFIRM`, masked number restated):**
    - es: `Para confirmar: voy a congelar la tarjeta ****5678. ¿Deseas que continúe? Respóndeme sí o no.`
    - pt: `Para confirmar: vou congelar o cartão ****5678. Você deseja que eu continue? Responda sim ou não.`
- **Turn 2 — explicit affirmative:**
  - es: `Sí, congélala.`
  - pt: `Sim, pode congelar.`
  - **Behavior:** the deterministic confirmation gate (not the LLM) reads the affirmative in the
    same session, calls the freeze tool, then reads back the card state.
  - **CORA reply (`ACTION_DONE`, only after read-back verifies the post-condition):**
    - es: `Listo, ya apliqué el cambio en tu tarjeta y lo verifiqué. ¿Algo más?`
    - pt: `Pronto, apliquei a alteração no seu cartão e verifiquei. Mais alguma coisa?`
- **Point at:** the masked `****5678` in the CONFIRM turn, the explicit `sí`/`sim`, the read-back
  verification result (status now frozen), POL-080, and both turns' `trace_id`s. If the read-back
  did **not** verify, CORA reports *not completed* and routes to a human — show this fail-closed
  branch if time allows.

---

## Path 3 — FX conversion — I5 → POL-090 `answer`

**What the current build does with FX.** The `convert_currency` tool (I5, REQ-28; see
[`../../src/cora/tools/layer.py`](../../src/cora/tools/layer.py)) is now **wired into the policy
path**: for an I5 turn, `_build_policy_input` in
[`../../src/cora/agent/graph.py`](../../src/cora/agent/graph.py) reads the proposed amount + source
/ target currencies (entity extraction, treated as data) and calls `convert_currency`. An OK rate
sets `resource_owned=True`, so **POL-090 answers** with the grounded figure. This closes the
former gap documented in [`LIMITATIONS.md`](LIMITATIONS.md) §5 (resolved). Fail-closed branches are
preserved: missing/ambiguous entities clarify (POL-070), and a rate older than 7 days abstains
honestly rather than guessing a rate (REQ-28).

- **Setup:** authenticated session; the `daily_exchange_rates` table landed with a recent rate for
  the demo pair.
- **es (AR):** `Hola, ¿cuánto me quedan 100 dólares pasados a pesos argentinos?`
- **pt:** `Olá, quanto fica 100 dólares convertidos para pesos argentinos?`
- **Expected decision:** intent `I5`; `convert_currency` returns a rate within 7 days →
  `read_resource_owned` true → **POL-090 `answer`**. The figure comes only from the FX tool
  `Result`; CORA never invents a rate or amount.
- **Expected reply (grounded `ANSWER`, figures from the FX tool `Result` — exact numbers depend on
  the landed rate):** the converted amount with the rate date, flagged if a latest-prior rate was
  used (REQ-28). Verify the rendered figure against the tool `Result` before recording.
- **Honest fallback to point at if the rate is stale (>7 days):** the tool abstains and CORA
  surfaces the fail-closed tool-unavailable copy (no figure) rather than a guessed rate — a good
  secondary beat if the demo data has no fresh rate for the pair.
- **Point at:** POL-090 on the trace span, the FX tool `source_refs` (table
  `daily_exchange_rates`), the `trace_id`, and that the converted figure is rendered
  **template-only** from the FX `Result` (the tool's values verbatim, grounded by construction — no
  LLM polish or grounding-checker step runs on a factual answer).

---

## Path 4 — Escalation / handoff (dispute) — E1 → POL-050 `escalate`

Shows CORA building a handoff package and **promising no outcome** (REQ-15/REQ-16/REQ-17). The
dispute intake collects the structured handoff fields and persists the package; the human-agent
console renders it.

- **Setup:** authenticated session.
- **es (MX):** `Quiero reclamar un cargo que no reconozco en mi tarjeta, nunca compré eso.`
- **pt:** `Quero contestar uma cobrança que não reconheço no meu cartão, eu nunca comprei isso.`
- **Expected decision:** intent `E1` (dispute); `intent_escalation` → **POL-050 `escalate`** with a
  built handoff package.
- **Expected reply (`ESCALATE`, no outcome promised):**
  - es: `Voy a transferir tu caso a una persona del equipo para que lo revise. Gracias por tu paciencia.`
  - pt: `Vou encaminhar o seu caso a uma pessoa da equipe para que seja analisado. Obrigado pela sua paciência.`
- **Point at:** the handoff package in the agent console (verbatim + masked request, language,
  verified facts with source refs, actions taken, supporting transactions, unresolved questions,
  reason code, priority, `trace_ref`) — all built from **verified tool results only** — POL-050,
  and that the reply makes **no** promise about the dispute's result.

---

## Path 5 — Guardrail refusal (credit eligibility) — X1 → POL-030 `abstain_route` (CREDIT_OUT_OF_SCOPE)

Credit eligibility is never evaluated (rule POL-030 / REQ-33); money movement is never performed
(POL-020 / REQ-34). This path demonstrates the credit guardrail; the money-movement refusal is
noted as the sibling guard.

- **Setup:** authenticated session.
- **es (CO):** `¿Me pueden aumentar el cupo de mi tarjeta de crédito? ¿Califico para un préstamo?`
- **pt:** `Vocês podem aumentar o limite do meu cartão de crédito? Eu me qualifico para um empréstimo?`
- **Expected decision:** intent `X1` (credit); `intent_credit` →
  **POL-030 `abstain_route`** with route `CREDIT_OUT_OF_SCOPE`. CORA decides **nothing** about
  credit and routes to a human.
- **Expected reply (`ABSTAIN_ROUTE`):**
  - es: `No puedo evaluar temas de crédito o aumentos de cupo por este canal. Voy a transferirte con una persona del equipo que pueda revisarlo.`
  - pt: `Não posso avaliar assuntos de crédito ou aumento de limite por este canal. Vou encaminhar você a uma pessoa da equipe que possa analisar.`
- **Point at:** POL-030, the `CREDIT_OUT_OF_SCOPE` route recorded on the trace span, and that no
  eligibility figure is ever shown. **Sibling guard (money movement):** a request such as
  `Transfiéreme 500 pesos a esta cuenta` / `Transfira 500 pesos para esta conta` hits intent `X2`
  → **POL-020 `refuse`**, replying
  `No puedo mover dinero ni realizar transferencias o pagos por este canal. ¿Hay algo más en lo que pueda ayudarte?`
  (es) / `Não posso movimentar dinheiro nem fazer transferências ou pagamentos por este canal. Posso ajudar com mais alguma coisa?` (pt).

---

## Live-run prerequisites (owner)

A live recording needs: (1) a valid `AWS_BEARER_TOKEN_BEDROCK` short-term key and the configured
`CORA_BEDROCK_*` model ids ([`../AWS_SETUP.md`](../AWS_SETUP.md)); (2) the local data landed so the
balance / FX tools return real figures; (3) `CORA_NLU_STUB` **unset** so the real classifier and
real Bedrock polish run. Without a valid token the generator falls back to the deterministic
templates above (REQ-40) — safe and grounded, but not the polished LLM evidence. The committed
artifact is this script; the video is produced and uploaded by the owner.
