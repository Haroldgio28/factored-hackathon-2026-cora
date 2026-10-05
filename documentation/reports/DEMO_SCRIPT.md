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
  to the grounding result and the per-turn `trace_id`
  ([`../../src/cora/obs/tracing.py`](../../src/cora/obs/tracing.py)).
- **What to point at (every turn):** the `trace_id`, the policy decision + POL rule id, and the
  grounding result (facts matched against the same turn's tool `Result`). For actions, also the
  masked product number and the read-back verification.

---

## Path 1 — Safe account read (balance of a specific card) — I1 → POL-090 `answer`

**Why a specific product reference.** A read decides `answer` only when `read_resource_owned`
holds (POL-090). In the current build, `_build_policy_input` in
[`../../src/cora/agent/graph.py`](../../src/cora/agent/graph.py) sets `resource_owned=True` only
when the turn resolves to a **specific** product the session owns; a bare "¿cuál es mi saldo?" with
no product reference never sets it, so POL-090 does not match and the turn falls through to POL-999
`abstain` (the live non-referenced-read gap documented in
[`LIMITATIONS.md`](LIMITATIONS.md)). The demo therefore uses an **explicit product reference** (a
card ending in known digits) so POL-090 matches and the balance is answered on a live run.

- **Setup:** authenticated session; the customer owns a card ending in `1234`.
- **es (MX):** `Hola, ¿me pasas el saldo de mi tarjeta terminada en 1234?`
- **pt:** `Oi, pode me informar o saldo do meu cartão terminado em 1234?`
- **Expected decision:** intent `I1`; `read_resource_owned` true → **POL-090 `answer`**.
- **Expected reply (grounded `ANSWER`, figures from the balance tool `Result`):**
  - es: `Tu saldo es 1.250,75 COP y tu saldo disponible es 1.100,00 COP.`
  - pt: `Seu saldo é 1.250,75 COP e seu saldo disponível é 1.100,00 COP.`
- **Point at:** the balance tool `source_refs` (table `products`), the grounding result (every
  figure in the reply matched the tool `Result` for this turn), POL-090, and the `trace_id`.

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

## Path 3 — FX conversion — I5 → POL-999 `abstain` (known gap, honest demo)

**What the current build actually does with FX.** The `convert_currency` tool exists and is
unit-tested at the tool layer (I5, REQ-28; see
[`../../src/cora/tools/layer.py`](../../src/cora/tools/layer.py)), but the turn orchestrator does
**not** wire it into the policy path: `_build_policy_input` in
[`../../src/cora/agent/graph.py`](../../src/cora/agent/graph.py) only calls `get_balance` for a read
intent, and only when a specific owned `product_id` is resolved. An FX request carries an
amount + currency pair, not a product reference, so `product_id` stays `None`, `resource_owned`
stays false, **POL-090 does not match**, and the turn falls through to **POL-999 `abstain`**. This
is the **same non-referenced-read root cause** documented as a known Phase-4 correctness gap in
[`LIMITATIONS.md`](LIMITATIONS.md) §5 (and the FX tool simply isn't invoked by the graph yet). The
demo tells the truth about this rather than claiming an answer the build cannot produce today.

- **Setup:** authenticated session.
- **es (AR):** `Hola, ¿cuánto me quedan 100 dólares pasados a pesos argentinos?`
- **pt:** `Olá, quanto fica 100 dólares convertidos para pesos argentinos?`
- **Expected decision:** intent `I5` (currency conversion) classifies correctly, but with no owned
  `product_id` resolved the FX result is never fetched into the policy input →
  **POL-999 `abstain`** (fail closed, disclose nothing, offer a human). CORA does **not** invent a
  rate or amount.
- **Expected reply (`ABSTAIN`, grounded template verbatim — no figures, offers a human):**
  - es: `Lo siento, no puedo ayudarte con eso por este medio. Si lo deseas, puedo transferirte con una persona del equipo.`
  - pt: `Desculpe, não posso ajudar com isso por este canal. Se preferir, posso encaminhar você a uma pessoa da equipe.`
- **Point at:** POL-999 on the trace span (not POL-090), that **no** FX figure appears (nothing was
  grounded because nothing was fetched — fail closed), the `trace_id`, and the LIMITATIONS §5 note
  explaining this is the known non-referenced-read gap (the fix — resolving ownership for FX /
  account-level reads and invoking `convert_currency` — is deferred to after Phase 7). Verify the
  exact abstain template against the response generator before recording.

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
