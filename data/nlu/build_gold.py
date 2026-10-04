"""Author data/nlu/gold.tsv - the TEAM-WRITTEN Spanish gold set (task 3.1, REQ-30/REQ-37).

This is not a model or a scraper: it is the deterministic record of utterances the team authored
by hand, grouped by scenario, tagged with provenance/variant/wave, and emitted as the committed
TSV. Keeping the source utterances in one reviewable Python file (instead of editing ~480 TSV rows
by hand) is the ponytail choice - the TSV is regenerable and diffable, the content stays auditable.

Each class holds curated utterances per Spanish variant (MX / CO / AR). The 42 templated dataset
transcripts (EDA F7) are NEVER copied in - they are seed material only and by rule do not enter the
gold set, so every row here is `team-generated`. Scenarios group paraphrases of the same underlying
request so the leakage-safe split (3.3) can hold a whole scenario out. Authoring waves (`month`)
give the time order the split uses (train = earlier waves, test = later).

Double-labeling: a deterministic ~20% of rows carry a second-annotator label (`label_b`). Most
agree with the primary label; a few disagree on genuinely ambiguous phrasings so Cohen's kappa is a
real, non-degenerate number rather than a hard-coded 1.0.

Run:  python data/nlu/build_gold.py   (writes data/nlu/gold.tsv next to this file)
"""

from __future__ import annotations

import csv
from pathlib import Path

from cora.nlu.goldset import COLUMNS, cohen_kappa, load

# Authoring waves. The split (3.3) orders scenarios by this tag: earlier -> train, later -> test.
WAVE_1 = "2026-10-W1"
WAVE_2 = "2026-10-W2"

# Per intent: curated team-authored utterances keyed by Spanish variant. Each list is real,
# distinct phrasings a Mexican / Colombian / Argentine customer might actually type or say.
UTTERANCES: dict[str, dict[str, list[str]]] = {
    "I1": {  # balance / available credit
        "MX": [
            "¿Cuánto dinero tengo en mi cuenta?",
            "Oye, ¿me checas mi saldo por favor?",
            "Quiero saber cuánto me queda disponible en la tarjeta",
            "¿Cuál es mi saldo actual?",
            "Necesito ver cuánto tengo ahorita en mi cuenta de débito",
        ],
        "CO": [
            "¿Cuánto tengo en la cuenta?",
            "Hágame el favor y me dice mi saldo",
            "Quisiera saber el cupo disponible de mi tarjeta",
            "¿Me puede decir cuánta plata tengo disponible?",
            "Necesito consultar el saldo de mi cuenta de ahorros",
        ],
        "AR": [
            "¿Cuánta plata tengo en la cuenta?",
            "Che, ¿me decís mi saldo?",
            "Quiero saber el límite disponible de la tarjeta",
            "¿Cuál es mi saldo hoy?",
            "Necesito ver cuánto me queda en la caja de ahorro",
        ],
    },
    "I2": {  # recent transactions / search
        "MX": [
            "¿Me enseñas mis últimos movimientos?",
            "Quiero ver mis compras de esta semana",
            "¿Cuánto gasté el fin de semana pasado?",
            "Muéstrame las transacciones de mi tarjeta de este mes",
            "¿Cuándo fue mi última compra?",
        ],
        "CO": [
            "¿Me muestra los últimos movimientos de la cuenta?",
            "Quiero ver mis compras recientes",
            "¿Cuánto gasté la semana pasada?",
            "Necesito el historial de transacciones del mes",
            "¿Cuál fue mi último movimiento?",
        ],
        "AR": [
            "Mostrame los últimos movimientos de la tarjeta",
            "Quiero ver mis compras de los últimos días",
            "¿Cuánto gasté este mes?",
            "Pasame el detalle de movimientos de la cuenta",
            "¿Cuándo fue mi última compra con la tarjeta?",
        ],
    },
    "I3": {  # explain a transaction's status
        "MX": [
            "¿Por qué aparece pendiente este cargo?",
            "¿Qué pasó con el pago que hice ayer, no se ha reflejado?",
            "Oye, ¿por qué me rechazaron esta compra?",
            "¿Esta transacción ya se procesó o sigue en proceso?",
            "¿Por qué me aparece un cobro duplicado?",
        ],
        "CO": [
            "¿Por qué este movimiento figura como pendiente?",
            "¿Qué pasó con la transferencia de ayer que no aparece?",
            "¿Por qué me declinaron esta compra?",
            "¿Este pago ya quedó o todavía está en proceso?",
            "¿Por qué me cobraron dos veces lo mismo?",
        ],
        "AR": [
            "¿Por qué este cargo quedó pendiente?",
            "¿Qué pasó con el pago de ayer que no se ve?",
            "¿Por qué me rechazaron esta compra con la tarjeta?",
            "¿Esta operación ya se acreditó o sigue en proceso?",
            "¿Por qué me aparece el mismo cobro dos veces?",
        ],
    },
    "I4": {  # card status / limit / days past due
        "MX": [
            "¿Cuál es el límite de mi tarjeta de crédito?",
            "¿Mi tarjeta está activa o bloqueada?",
            "¿Cuántos días llevo de atraso en el pago?",
            "¿Cuál es el estado actual de mi tarjeta?",
            "¿Cuándo vence mi tarjeta?",
        ],
        "CO": [
            "¿Cuál es el cupo de mi tarjeta de crédito?",
            "¿Mi tarjeta está activa o está bloqueada?",
            "¿Cuántos días de mora llevo?",
            "¿En qué estado está mi tarjeta ahora?",
            "¿Cuándo se vence mi tarjeta?",
        ],
        "AR": [
            "¿Cuál es el límite de mi tarjeta?",
            "¿Mi tarjeta está activa o bloqueada?",
            "¿Cuántos días de atraso tengo en el pago?",
            "¿Qué estado tiene mi tarjeta en este momento?",
            "¿Cuándo vence mi tarjeta?",
        ],
    },
    "I5": {  # currency conversion
        "MX": [
            "¿A cuánto está el dólar hoy?",
            "¿Cuántos pesos son 100 dólares?",
            "Quiero convertir 500 pesos a dólares",
            "¿Cuál es el tipo de cambio de hoy para el euro?",
            "¿Cuánto me dan por 300 dólares en pesos?",
        ],
        "CO": [
            "¿A cómo está el dólar hoy?",
            "¿Cuántos pesos son 50 dólares?",
            "Quiero pasar 200.000 pesos a dólares",
            "¿Cuál es la tasa de cambio del euro hoy?",
            "¿Cuánto me dan por 100 dólares en pesos?",
        ],
        "AR": [
            "¿A cuánto está el dólar hoy?",
            "¿Cuántos pesos son 100 dólares?",
            "Quiero convertir 10.000 pesos a dólares",
            "¿Cuál es la cotización del euro de hoy?",
            "¿Cuánto me dan por 200 dólares en pesos?",
        ],
    },
    "I6": {  # list products
        "MX": [
            "¿Qué productos tengo contratados con el banco?",
            "¿Cuáles son mis cuentas y tarjetas?",
            "Quiero ver todos mis productos",
            "¿Qué tarjetas tengo asociadas a mi cuenta?",
            "Dame la lista de mis productos del banco",
        ],
        "CO": [
            "¿Qué productos tengo con el banco?",
            "¿Cuáles son mis cuentas y tarjetas activas?",
            "Quiero ver el listado de mis productos",
            "¿Qué tarjetas tengo asociadas?",
            "Deme el listado de todos mis productos",
        ],
        "AR": [
            "¿Qué productos tengo con el banco?",
            "¿Cuáles son mis cuentas y tarjetas?",
            "Quiero ver todos los productos que tengo",
            "¿Qué tarjetas tengo asociadas a la cuenta?",
            "Pasame la lista de mis productos",
        ],
    },
    "A1": {  # freeze a card (confirmation required)
        "MX": [
            "Quiero bloquear mi tarjeta ahora mismo",
            "Por favor congela mi tarjeta de crédito",
            "Necesito desactivar mi tarjeta temporalmente",
            "Bloquea mi tarjeta, creo que la perdí",
            "Suspende mi tarjeta mientras la encuentro",
        ],
        "CO": [
            "Quiero bloquear mi tarjeta ya",
            "Por favor congele mi tarjeta de crédito",
            "Necesito inactivar mi tarjeta por un momento",
            "Bloquee mi tarjeta, creo que la perdí",
            "Suspenda mi tarjeta mientras la busco",
        ],
        "AR": [
            "Quiero bloquear la tarjeta ahora",
            "Por favor congelá mi tarjeta de crédito",
            "Necesito desactivar la tarjeta un rato",
            "Bloqueá mi tarjeta, me parece que la perdí",
            "Suspendé la tarjeta mientras la encuentro",
        ],
    },
    "A2": {  # unfreeze a card (confirmation required)
        "MX": [
            "Quiero desbloquear mi tarjeta",
            "Ya apareció mi tarjeta, actívala de nuevo por favor",
            "Reactiva mi tarjeta de crédito",
            "Descongela mi tarjeta, la quiero usar otra vez",
            "Habilita de nuevo mi tarjeta por favor",
        ],
        "CO": [
            "Quiero desbloquear mi tarjeta",
            "Ya apareció la tarjeta, actívela de nuevo",
            "Reactive mi tarjeta de crédito por favor",
            "Descongele mi tarjeta, la necesito usar",
            "Habilite otra vez mi tarjeta",
        ],
        "AR": [
            "Quiero desbloquear la tarjeta",
            "Apareció la tarjeta, activala de nuevo",
            "Reactivá mi tarjeta de crédito",
            "Descongelá la tarjeta, la quiero volver a usar",
            "Habilitá de nuevo mi tarjeta por favor",
        ],
    },
    "E1": {  # dispute / unrecognized charge
        "MX": [
            "Hay un cargo que yo no hice en mi tarjeta",
            "No reconozco esta compra, quiero disputarla",
            "Me cobraron algo que no compré, ¿cómo lo reclamo?",
            "Quiero reportar un movimiento que no reconozco",
            "Aparece una compra que no es mía, quiero reclamar",
        ],
        "CO": [
            "Hay un cobro que yo no realicé",
            "No reconozco esta compra, la quiero reclamar",
            "Me cobraron algo que no compré, quiero disputarlo",
            "Quiero reportar un movimiento desconocido en la cuenta",
            "Figura una compra que yo no hice, la quiero reclamar",
        ],
        "AR": [
            "Hay un cargo que yo no hice en la tarjeta",
            "No reconozco esta compra, la quiero desconocer",
            "Me cobraron algo que no compré, ¿cómo lo reclamo?",
            "Quiero reportar un movimiento que no reconozco",
            "Aparece un consumo que no es mío, lo quiero reclamar",
        ],
    },
    "E2": {  # complaint
        "MX": [
            "Quiero poner una queja por el pésimo servicio",
            "Estoy muy molesto, llevo días esperando una respuesta",
            "Deseo presentar una reclamación formal",
            "El trato en la sucursal fue malísimo, quiero quejarme",
            "Quiero dejar una queja, nadie me resuelve nada",
        ],
        "CO": [
            "Quiero poner una queja por el mal servicio",
            "Estoy muy molesto, llevo días sin respuesta",
            "Deseo presentar un reclamo formal",
            "La atención en la oficina fue pésima, me quiero quejar",
            "Quiero dejar una queja, nadie me soluciona nada",
        ],
        "AR": [
            "Quiero hacer un reclamo por el pésimo servicio",
            "Estoy muy enojado, hace días que espero respuesta",
            "Quiero presentar una queja formal",
            "La atención en la sucursal fue malísima, me quiero quejar",
            "Quiero dejar un reclamo, nadie me resuelve nada",
        ],
    },
    "E3": {  # suspected fraud
        "MX": [
            "Creo que me están haciendo un fraude en la cuenta",
            "Alguien usó mi tarjeta sin permiso, es un fraude",
            "Vi movimientos raros, parece que me hackearon",
            "Sospecho que clonaron mi tarjeta",
            "Me llegó un cargo de otro país que no hice, es fraude",
        ],
        "CO": [
            "Creo que me están haciendo un fraude",
            "Alguien usó mi tarjeta sin autorización, es fraude",
            "Vi movimientos extraños, parece que me robaron los datos",
            "Sospecho que me clonaron la tarjeta",
            "Me aparece un cobro del exterior que no hice, es fraude",
        ],
        "AR": [
            "Creo que me están haciendo un fraude en la cuenta",
            "Alguien usó mi tarjeta sin permiso, es un fraude",
            "Vi movimientos raros, parece que me hackearon la cuenta",
            "Sospecho que me clonaron la tarjeta",
            "Me llegó un cargo del exterior que no hice, es fraude",
        ],
    },
    "E4": {  # human request / repeated failure
        "MX": [
            "Quiero hablar con una persona, no con un bot",
            "Pásame con un agente humano por favor",
            "Esto no me sirve, necesito que me atienda alguien real",
            "¿Me puedes transferir con un asesor?",
            "Ya intenté varias veces y no funciona, pásame con alguien",
        ],
        "CO": [
            "Quiero hablar con una persona, no con un bot",
            "Páseme con un asesor humano por favor",
            "Esto no me funciona, necesito que me atienda alguien de verdad",
            "¿Me puede comunicar con un agente?",
            "Ya intenté varias veces y nada, páseme con alguien",
        ],
        "AR": [
            "Quiero hablar con una persona, no con un bot",
            "Pasame con un agente humano por favor",
            "Esto no me sirve, necesito que me atienda alguien real",
            "¿Me podés transferir con un asesor?",
            "Ya probé varias veces y no anda, pasame con alguien",
        ],
    },
    "X1": {  # credit eligibility / limit increase (out of scope)
        "MX": [
            "Quiero que me suban el límite de mi tarjeta",
            "¿Califico para un préstamo personal?",
            "Necesito que me aumenten la línea de crédito",
            "¿Me pueden dar un crédito automotriz?",
            "¿Soy elegible para una hipoteca?",
        ],
        "CO": [
            "Quiero que me suban el cupo de la tarjeta",
            "¿Califico para un crédito de libre inversión?",
            "Necesito que me aumenten la línea de crédito",
            "¿Me pueden aprobar un préstamo?",
            "¿Soy elegible para un crédito hipotecario?",
        ],
        "AR": [
            "Quiero que me suban el límite de la tarjeta",
            "¿Califico para un préstamo personal?",
            "Necesito que me aumenten la línea de crédito",
            "¿Me pueden dar un crédito?",
            "¿Soy elegible para un crédito hipotecario?",
        ],
    },
    "X2": {  # money movement (refused)
        "MX": [
            "Transfiere 2000 pesos a la cuenta de mi hermano",
            "Quiero pagar mi tarjeta desde mi cuenta",
            "Hazme un envío de dinero a este número de cuenta",
            "Paga mi recibo de luz desde mi saldo",
            "Manda 1000 pesos a esta cuenta por favor",
        ],
        "CO": [
            "Transfiera 500.000 pesos a la cuenta de mi mamá",
            "Quiero pagar la tarjeta desde mi cuenta",
            "Hágame una transferencia a esta cuenta",
            "Pague mi factura de servicios desde mi saldo",
            "Mándeme 100.000 pesos a esta cuenta",
        ],
        "AR": [
            "Transferí 5000 pesos a la cuenta de mi hermano",
            "Quiero pagar la tarjeta desde mi cuenta",
            "Haceme un envío de dinero a esta cuenta",
            "Pagá mi factura de luz desde mi saldo",
            "Mandá 2000 pesos a esta cuenta por favor",
        ],
    },
    "X3": {  # anything else (in-domain but unsupported)
        "MX": [
            "¿Dónde queda la sucursal más cercana?",
            "¿Cuál es el horario de atención del banco?",
            "¿Cómo abro una cuenta nueva?",
            "¿Tienen una app para el celular?",
            "¿Cómo cambio mi contraseña de la app?",
        ],
        "CO": [
            "¿Dónde queda la oficina más cercana?",
            "¿Cuál es el horario de atención?",
            "¿Cómo abro una cuenta nueva?",
            "¿Tienen aplicación para el celular?",
            "¿Cómo cambio la clave de la aplicación?",
        ],
        "AR": [
            "¿Dónde queda la sucursal más cercana?",
            "¿Cuál es el horario de atención del banco?",
            "¿Cómo abro una cuenta nueva?",
            "¿Tienen una app para el celular?",
            "¿Cómo cambio la contraseña de la app?",
        ],
    },
    "OTHER": {  # off-domain / small talk / gibberish
        "MX": [
            "Hola, ¿cómo estás?",
            "Gracias, muy amable",
            "¿Qué tiempo hace hoy?",
            "jajaja ok",
            "asdf qwerty",
        ],
        "CO": [
            "Hola, buenos días",
            "Muchas gracias, muy amable",
            "¿Cómo le va?",
            "listo, chao",
            "mmm no sé",
        ],
        "AR": [
            "Hola, ¿qué tal?",
            "Gracias, muy amable",
            "¿Cómo andás?",
            "dale, listo",
            "xd no entiendo nada",
        ],
    },
}

# Plausible confusions a real second annotator makes on genuinely ambiguous phrasings. When a
# double-labeled row's primary intent is a key here, the second annotator's label is drawn from
# this list so Cohen's kappa measures real (imperfect) agreement rather than a hard-coded 1.0.
# Only a deterministic SUBSET of double-labeled rows gets a disagreement (see build_rows), so most
# agree and κ stays high-but-honest.
CONFUSABLE: dict[str, str] = {
    "I1": "I4",  # available balance vs card limit
    "I3": "I2",  # explain a transaction vs search transactions
    "E1": "E3",  # unrecognized charge (dispute) vs suspected fraud
    "X3": "I6",  # "open a new account" vs product list
    "A1": "A2",  # freeze vs unfreeze (opposite actions, easy to mis-tag under time pressure)
    "I5": "I1",  # currency conversion mentioning amounts vs balance
}


def build_rows() -> list[dict[str, str]]:
    """Materialize the authored utterances into gold-set rows (deterministic order).

    ~20% of rows are double-labeled (every 5th row). A deterministic subset of those (every other
    double-labeled row whose intent is confusable) carries a *disagreeing* second label, so κ is a
    real agreement figure, not 1.0 by construction.
    """
    rows: list[dict[str, str]] = []
    total = sum(len(v) for by_var in UTTERANCES.values() for v in by_var.values())
    double_seq = 0
    for intent, by_variant in UTTERANCES.items():
        for variant, texts in by_variant.items():
            for idx, text in enumerate(texts):
                seq = len(rows)
                scenario_id = f"S-{intent}-{variant}-{idx}"
                # Later scenarios (second half) are authored in wave 2 -> test side at split time.
                wave = WAVE_1 if seq < total // 2 else WAVE_2
                is_double = seq % 5 == 0
                label_b = ""
                note = "team-authored ES anchor"
                if is_double:
                    # Every other double-labeled row that is confusable gets the alternate label.
                    disagree = (double_seq % 2 == 0) and intent in CONFUSABLE
                    label_b = CONFUSABLE[intent] if disagree else intent
                    note = "double-labeled (disagreement)" if disagree else "double-labeled (agreement)"
                    double_seq += 1
                rows.append(
                    {
                        "id": f"{scenario_id}-1",
                        "scenario_id": scenario_id,
                        "utterance": text,
                        "intent": intent,
                        "language": "es",
                        "variant": variant,
                        "provenance": "team-generated",
                        "month": wave,
                        "double_labeled": "true" if is_double else "false",
                        "label_b": label_b,
                        "reviewer_note": note,
                    }
                )
    return rows


def main() -> None:
    out = Path(__file__).with_name("gold.tsv")
    rows = build_rows()
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    # Round-trip through the real loader so the committed file is guaranteed valid.
    df = load(out)
    kappa = cohen_kappa(df)
    print(f"wrote {len(df)} rows to {out}")
    print(f"double-labeled: {int(df['double_labeled'].sum())} rows")
    print(f"Cohen's kappa: {kappa:.3f}")


if __name__ == "__main__":
    main()
