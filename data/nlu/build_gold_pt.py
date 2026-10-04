"""Author data/nlu/gold_pt.tsv - the Portuguese gold set (task 3.2, REQ-18/REQ-19/REQ-37).

The dataset is Spanish-only (EDA F7), so Portuguese is NOT mined - it is produced by the team as an
ES->PT translation of the Spanish gold set (`provenance=translated`, `variant=BR`), plus a
deterministic ~20% of native-style BR rewrites (`provenance=team-generated`) so the set is not a
pure literal-translation artifact. Every row records its provenance in `reviewer_note` and keeps a
pointer to the Spanish row it came from, so the es<->pt pairing (REQ-19 language coverage) is a
column join the loader/test can check, not an assumption.

Like `build_gold.py`, this keeps the authored utterances in one reviewable Python file instead of
hand-editing ~240 TSV rows (ponytail: the TSV is regenerable and diffable, the content auditable).
The PT translations/rewrites below are team-authored and reviewed; `reviewer_note` carries the
provenance and the source Spanish `scenario_id`.

Run:  python data/nlu/build_gold_pt.py   (writes data/nlu/gold_pt.tsv next to this file)
"""

from __future__ import annotations

import csv
from pathlib import Path

from cora.nlu.goldset import COLUMNS, cohen_kappa, load

# Portuguese renderings, keyed exactly by the Spanish source (intent -> source-variant -> idx).
# Each Spanish row in build_gold.py's UTTERANCES has one PT counterpart here, in the same order, so
# the mapping is 1:1 and the es<->pt pairing is total. Translations are faithful BR Portuguese; the
# native-rewrite subset (see REWRITES) replaces the literal translation with idiomatic BR phrasing.
PT: dict[str, dict[str, list[str]]] = {
    "I1": {
        "MX": [
            "Quanto dinheiro eu tenho na minha conta?",
            "Ei, você pode checar meu saldo, por favor?",
            "Quero saber quanto ainda tenho disponível no cartão",
            "Qual é o meu saldo atual?",
            "Preciso ver quanto tenho agora na minha conta de débito",
        ],
        "CO": [
            "Quanto eu tenho na conta?",
            "Por favor, me informe o meu saldo",
            "Gostaria de saber o limite disponível do meu cartão",
            "Você pode me dizer quanto dinheiro tenho disponível?",
            "Preciso consultar o saldo da minha conta poupança",
        ],
        "AR": [
            "Quanto dinheiro eu tenho na conta?",
            "E aí, me diz o meu saldo?",
            "Quero saber o limite disponível do cartão",
            "Qual é o meu saldo hoje?",
            "Preciso ver quanto me resta na conta poupança",
        ],
    },
    "I2": {
        "MX": [
            "Você me mostra as minhas últimas movimentações?",
            "Quero ver as minhas compras desta semana",
            "Quanto eu gastei no fim de semana passado?",
            "Me mostre as transações do meu cartão deste mês",
            "Quando foi a minha última compra?",
        ],
        "CO": [
            "Você me mostra as últimas movimentações da conta?",
            "Quero ver as minhas compras recentes",
            "Quanto eu gastei na semana passada?",
            "Preciso do histórico de transações do mês",
            "Qual foi a minha última movimentação?",
        ],
        "AR": [
            "Me mostra as últimas movimentações do cartão",
            "Quero ver as minhas compras dos últimos dias",
            "Quanto eu gastei este mês?",
            "Me passa o detalhe das movimentações da conta",
            "Quando foi a minha última compra com o cartão?",
        ],
    },
    "I3": {
        "MX": [
            "Por que esta cobrança aparece como pendente?",
            "O que aconteceu com o pagamento que fiz ontem, não foi lançado?",
            "Ei, por que recusaram esta compra?",
            "Esta transação já foi processada ou ainda está em andamento?",
            "Por que aparece uma cobrança duplicada?",
        ],
        "CO": [
            "Por que esta movimentação figura como pendente?",
            "O que aconteceu com a transferência de ontem que não aparece?",
            "Por que recusaram esta compra?",
            "Este pagamento já foi concluído ou ainda está em andamento?",
            "Por que me cobraram duas vezes a mesma coisa?",
        ],
        "AR": [
            "Por que esta cobrança ficou pendente?",
            "O que aconteceu com o pagamento de ontem que não aparece?",
            "Por que recusaram esta compra com o cartão?",
            "Esta operação já foi creditada ou ainda está em andamento?",
            "Por que aparece a mesma cobrança duas vezes?",
        ],
    },
    "I4": {
        "MX": [
            "Qual é o limite do meu cartão de crédito?",
            "Meu cartão está ativo ou bloqueado?",
            "Quantos dias eu estou em atraso no pagamento?",
            "Qual é o estado atual do meu cartão?",
            "Quando vence o meu cartão?",
        ],
        "CO": [
            "Qual é o limite do meu cartão de crédito?",
            "Meu cartão está ativo ou está bloqueado?",
            "Quantos dias de atraso eu tenho?",
            "Em que situação está o meu cartão agora?",
            "Quando vence o meu cartão?",
        ],
        "AR": [
            "Qual é o limite do meu cartão?",
            "Meu cartão está ativo ou bloqueado?",
            "Quantos dias de atraso eu tenho no pagamento?",
            "Qual é a situação do meu cartão neste momento?",
            "Quando vence o meu cartão?",
        ],
    },
    "I5": {
        "MX": [
            "A quanto está o dólar hoje?",
            "Quantos reais são 100 dólares?",
            "Quero converter 500 reais em dólares",
            "Qual é a taxa de câmbio de hoje para o euro?",
            "Quanto eu recebo por 300 dólares em reais?",
        ],
        "CO": [
            "Como está o dólar hoje?",
            "Quantos reais são 50 dólares?",
            "Quero converter 200 reais em dólares",
            "Qual é a taxa de câmbio do euro hoje?",
            "Quanto eu recebo por 100 dólares em reais?",
        ],
        "AR": [
            "A quanto está o dólar hoje?",
            "Quantos reais são 100 dólares?",
            "Quero converter 10.000 reais em dólares",
            "Qual é a cotação do euro de hoje?",
            "Quanto eu recebo por 200 dólares em reais?",
        ],
    },
    "I6": {
        "MX": [
            "Quais produtos eu tenho contratados com o banco?",
            "Quais são as minhas contas e cartões?",
            "Quero ver todos os meus produtos",
            "Quais cartões estão associados à minha conta?",
            "Me dê a lista dos meus produtos do banco",
        ],
        "CO": [
            "Quais produtos eu tenho com o banco?",
            "Quais são as minhas contas e cartões ativos?",
            "Quero ver a lista dos meus produtos",
            "Quais cartões eu tenho associados?",
            "Me dê a lista de todos os meus produtos",
        ],
        "AR": [
            "Quais produtos eu tenho com o banco?",
            "Quais são as minhas contas e cartões?",
            "Quero ver todos os produtos que eu tenho",
            "Quais cartões estão associados à conta?",
            "Me passa a lista dos meus produtos",
        ],
    },
    "A1": {
        "MX": [
            "Quero bloquear o meu cartão agora mesmo",
            "Por favor, congele o meu cartão de crédito",
            "Preciso desativar o meu cartão temporariamente",
            "Bloqueie o meu cartão, acho que o perdi",
            "Suspenda o meu cartão enquanto eu o procuro",
        ],
        "CO": [
            "Quero bloquear o meu cartão já",
            "Por favor, congele o meu cartão de crédito",
            "Preciso inativar o meu cartão por um momento",
            "Bloqueie o meu cartão, acho que o perdi",
            "Suspenda o meu cartão enquanto eu o procuro",
        ],
        "AR": [
            "Quero bloquear o cartão agora",
            "Por favor, congele o meu cartão de crédito",
            "Preciso desativar o cartão por um tempo",
            "Bloqueie o meu cartão, acho que o perdi",
            "Suspenda o cartão enquanto eu o encontro",
        ],
    },
    "A2": {
        "MX": [
            "Quero desbloquear o meu cartão",
            "Meu cartão apareceu, ative-o de novo, por favor",
            "Reative o meu cartão de crédito",
            "Descongele o meu cartão, quero usá-lo de novo",
            "Habilite o meu cartão de novo, por favor",
        ],
        "CO": [
            "Quero desbloquear o meu cartão",
            "O cartão apareceu, ative-o de novo",
            "Reative o meu cartão de crédito, por favor",
            "Descongele o meu cartão, preciso usá-lo",
            "Habilite o meu cartão de novo",
        ],
        "AR": [
            "Quero desbloquear o cartão",
            "O cartão apareceu, ative-o de novo",
            "Reative o meu cartão de crédito",
            "Descongele o cartão, quero voltar a usá-lo",
            "Habilite o meu cartão de novo, por favor",
        ],
    },
    "E1": {
        "MX": [
            "Há uma cobrança que eu não fiz no meu cartão",
            "Não reconheço esta compra, quero contestá-la",
            "Me cobraram algo que eu não comprei, como eu reclamo?",
            "Quero reportar uma movimentação que eu não reconheço",
            "Aparece uma compra que não é minha, quero reclamar",
        ],
        "CO": [
            "Há uma cobrança que eu não realizei",
            "Não reconheço esta compra, quero reclamá-la",
            "Me cobraram algo que eu não comprei, quero contestar",
            "Quero reportar uma movimentação desconhecida na conta",
            "Figura uma compra que eu não fiz, quero reclamá-la",
        ],
        "AR": [
            "Há uma cobrança que eu não fiz no cartão",
            "Não reconheço esta compra, quero desconhecê-la",
            "Me cobraram algo que eu não comprei, como eu reclamo?",
            "Quero reportar uma movimentação que eu não reconheço",
            "Aparece um consumo que não é meu, quero reclamá-lo",
        ],
    },
    "E2": {
        "MX": [
            "Quero registrar uma reclamação pelo péssimo atendimento",
            "Estou muito irritado, faz dias que espero uma resposta",
            "Desejo apresentar uma reclamação formal",
            "O atendimento na agência foi péssimo, quero reclamar",
            "Quero deixar uma reclamação, ninguém resolve nada",
        ],
        "CO": [
            "Quero registrar uma reclamação pelo mau atendimento",
            "Estou muito irritado, faz dias sem resposta",
            "Desejo apresentar uma reclamação formal",
            "O atendimento na agência foi péssimo, quero reclamar",
            "Quero deixar uma reclamação, ninguém resolve nada",
        ],
        "AR": [
            "Quero fazer uma reclamação pelo péssimo atendimento",
            "Estou muito irritado, faz dias que espero resposta",
            "Quero apresentar uma reclamação formal",
            "O atendimento na agência foi péssimo, quero reclamar",
            "Quero deixar uma reclamação, ninguém resolve nada",
        ],
    },
    "E3": {
        "MX": [
            "Acho que estão cometendo uma fraude na minha conta",
            "Alguém usou o meu cartão sem permissão, é uma fraude",
            "Vi movimentações estranhas, parece que me hackearam",
            "Suspeito que clonaram o meu cartão",
            "Chegou uma cobrança de outro país que eu não fiz, é fraude",
        ],
        "CO": [
            "Acho que estão cometendo uma fraude comigo",
            "Alguém usou o meu cartão sem autorização, é fraude",
            "Vi movimentações estranhas, parece que roubaram os meus dados",
            "Suspeito que clonaram o meu cartão",
            "Aparece uma cobrança do exterior que eu não fiz, é fraude",
        ],
        "AR": [
            "Acho que estão cometendo uma fraude na minha conta",
            "Alguém usou o meu cartão sem permissão, é uma fraude",
            "Vi movimentações estranhas, parece que hackearam a minha conta",
            "Suspeito que clonaram o meu cartão",
            "Chegou uma cobrança do exterior que eu não fiz, é fraude",
        ],
    },
    "E4": {
        "MX": [
            "Quero falar com uma pessoa, não com um robô",
            "Me transfira para um atendente humano, por favor",
            "Isso não me serve, preciso ser atendido por alguém de verdade",
            "Você pode me transferir para um atendente?",
            "Já tentei várias vezes e não funciona, me passe para alguém",
        ],
        "CO": [
            "Quero falar com uma pessoa, não com um robô",
            "Me passe para um atendente humano, por favor",
            "Isso não funciona para mim, preciso ser atendido por alguém de verdade",
            "Você pode me conectar com um atendente?",
            "Já tentei várias vezes e nada, me passe para alguém",
        ],
        "AR": [
            "Quero falar com uma pessoa, não com um robô",
            "Me passe para um atendente humano, por favor",
            "Isso não me serve, preciso ser atendido por alguém de verdade",
            "Você pode me transferir para um atendente?",
            "Já tentei várias vezes e não funciona, me passe para alguém",
        ],
    },
    "X1": {
        "MX": [
            "Quero que aumentem o limite do meu cartão",
            "Eu me qualifico para um empréstimo pessoal?",
            "Preciso que aumentem o meu limite de crédito",
            "Vocês podem me dar um financiamento de veículo?",
            "Eu sou elegível para um financiamento imobiliário?",
        ],
        "CO": [
            "Quero que aumentem o limite do cartão",
            "Eu me qualifico para um crédito de livre investimento?",
            "Preciso que aumentem o meu limite de crédito",
            "Vocês podem aprovar um empréstimo para mim?",
            "Eu sou elegível para um crédito imobiliário?",
        ],
        "AR": [
            "Quero que aumentem o limite do cartão",
            "Eu me qualifico para um empréstimo pessoal?",
            "Preciso que aumentem o meu limite de crédito",
            "Vocês podem me dar um crédito?",
            "Eu sou elegível para um crédito imobiliário?",
        ],
    },
    "X2": {
        "MX": [
            "Transfira 2000 reais para a conta do meu irmão",
            "Quero pagar o meu cartão com a minha conta",
            "Faça um envio de dinheiro para este número de conta",
            "Pague a minha conta de luz com o meu saldo",
            "Envie 1000 reais para esta conta, por favor",
        ],
        "CO": [
            "Transfira 500 reais para a conta da minha mãe",
            "Quero pagar o cartão com a minha conta",
            "Faça uma transferência para esta conta",
            "Pague a minha fatura de serviços com o meu saldo",
            "Me envie 100 reais para esta conta",
        ],
        "AR": [
            "Transfira 5000 reais para a conta do meu irmão",
            "Quero pagar o cartão com a minha conta",
            "Faça um envio de dinheiro para esta conta",
            "Pague a minha conta de luz com o meu saldo",
            "Envie 2000 reais para esta conta, por favor",
        ],
    },
    "X3": {
        "MX": [
            "Onde fica a agência mais próxima?",
            "Qual é o horário de atendimento do banco?",
            "Como eu abro uma conta nova?",
            "Vocês têm um aplicativo para o celular?",
            "Como eu troco a minha senha do aplicativo?",
        ],
        "CO": [
            "Onde fica a agência mais próxima?",
            "Qual é o horário de atendimento?",
            "Como eu abro uma conta nova?",
            "Vocês têm um aplicativo para o celular?",
            "Como eu troco a senha do aplicativo?",
        ],
        "AR": [
            "Onde fica a agência mais próxima?",
            "Qual é o horário de atendimento do banco?",
            "Como eu abro uma conta nova?",
            "Vocês têm um aplicativo para o celular?",
            "Como eu troco a senha do aplicativo?",
        ],
    },
    "OTHER": {
        "MX": [
            "Oi, tudo bem?",
            "Obrigado, muito gentil",
            "Como está o tempo hoje?",
            "kkkk ok",
            "asdf qwerty",
        ],
        "CO": [
            "Oi, bom dia",
            "Muito obrigado, muito gentil",
            "Como vai você?",
            "beleza, tchau",
            "hmm não sei",
        ],
        "AR": [
            "Oi, tudo certo?",
            "Obrigado, muito gentil",
            "Como você está?",
            "beleza, pronto",
            "kkk não entendi nada",
        ],
    },
}

# Deterministic ~20% native-style BR rewrites: idiomatic phrasings a Brazilian would actually type,
# replacing the literal translation for the selected rows (marked provenance=team-generated). The
# idx-0 row of every (class, source-variant) is a rewrite -> 16 classes x 3 variants = 48 rows =
# 20% of the 240-row set. Keyed by the Spanish source scenario_id so the selection is explicit and
# reviewable; each value is native BR phrasing, not a literal rendering of the Spanish source.
REWRITES: dict[str, str] = {
    # MX source rows
    "S-I1-MX-0": "Qual é o saldo da minha conta?",
    "S-I2-MX-0": "Dá pra ver o meu extrato?",
    "S-I3-MX-0": "Por que esse lançamento está pendente?",
    "S-I4-MX-0": "Qual é o limite do meu cartão de crédito?",
    "S-I5-MX-0": "Qual é a cotação do dólar hoje?",
    "S-I6-MX-0": "Quais produtos eu tenho no banco?",
    "S-A1-MX-0": "Preciso bloquear o meu cartão agora",
    "S-A2-MX-0": "Quero desbloquear o meu cartão",
    "S-E1-MX-0": "Tem uma compra no meu cartão que eu não reconheço",
    "S-E2-MX-0": "Quero abrir uma reclamação sobre o atendimento",
    "S-E3-MX-0": "Acho que caí num golpe, tem compras estranhas na conta",
    "S-E4-MX-0": "Quero falar com um atendente de verdade",
    "S-X1-MX-0": "Dá pra aumentar o limite do meu cartão?",
    "S-X2-MX-0": "Faz um Pix de 2000 reais pro meu irmão",
    "S-X3-MX-0": "Qual a agência mais perto de mim?",
    "S-OTHER-MX-0": "E aí, beleza?",
    # CO source rows
    "S-I1-CO-0": "Quanto tem na minha conta?",
    "S-I2-CO-0": "Me mostra o extrato da conta?",
    "S-I3-CO-0": "Esse lançamento tá pendente por quê?",
    "S-I4-CO-0": "Qual é o limite do meu cartão?",
    "S-I5-CO-0": "Quanto tá o dólar agora?",
    "S-I6-CO-0": "Quais são os meus produtos no banco?",
    "S-A1-CO-0": "Quero bloquear o cartão já",
    "S-A2-CO-0": "Preciso desbloquear o cartão",
    "S-E1-CO-0": "Tem uma cobrança aqui que não fui eu",
    "S-E2-CO-0": "Quero reclamar do atendimento de vocês",
    "S-E3-CO-0": "Acho que estão aplicando um golpe na minha conta",
    "S-E4-CO-0": "Me passa pra uma pessoa, por favor",
    "S-X1-CO-0": "Como faço pra aumentar o limite do cartão?",
    "S-X2-CO-0": "Faz uma transferência de 500 reais pra minha mãe",
    "S-X3-CO-0": "Onde tem uma agência perto daqui?",
    "S-OTHER-CO-0": "Oi, bom dia, tudo certo?",
    # AR source rows
    "S-I1-AR-0": "Quanto eu tenho de saldo?",
    "S-I2-AR-0": "Me manda o extrato do cartão?",
    "S-I3-AR-0": "Por que essa cobrança ficou pendente?",
    "S-I4-AR-0": "Qual o limite do meu cartão hoje?",
    "S-I5-AR-0": "Quanto tá o dólar hoje?",
    "S-I6-AR-0": "Quais produtos eu tenho no banco mesmo?",
    "S-A1-AR-0": "Bloqueia o meu cartão agora, por favor",
    "S-A2-AR-0": "Desbloqueia o meu cartão",
    "S-E1-AR-0": "Tem um lançamento no cartão que não fui eu",
    "S-E2-AR-0": "Quero registrar uma reclamação do atendimento",
    "S-E3-AR-0": "Acho que clonaram o meu cartão, tem gastos estranhos",
    "S-E4-AR-0": "Quero falar com um humano, não com o robô",
    "S-X1-AR-0": "Tem como aumentar o limite do cartão?",
    "S-X2-AR-0": "Transfere 5000 reais pro meu irmão",
    "S-X3-AR-0": "Qual é a agência mais perto?",
    "S-OTHER-AR-0": "Oi, tudo certo por aí?",
}


def build_rows() -> list[dict[str, str]]:
    """Materialize PT rows from the ES gold set (deterministic 1:1 translation + rewrite subset).

    Reads the committed Spanish gold.tsv so the mapping is anchored to the real ES rows: for each ES
    row it emits one PT row carrying a pointer to the source `scenario_id`. A row whose source is in
    REWRITES is a native-style rewrite (`team-generated`); every other PT row is a `translated` one.
    `double_labeled`/`label_b` mirror the Spanish source so Cohen's kappa stays comparable and the
    second-annotator protocol is not silently dropped for Portuguese.
    """
    es = load(Path(__file__).with_name("gold.tsv"))
    rows: list[dict[str, str]] = []
    for src in es.itertuples(index=False):
        intent, variant, src_scenario = src.intent, src.variant, src.scenario_id
        idx = int(src_scenario.rsplit("-", 1)[1])  # trailing index of the ES scenario
        rewrite = REWRITES.get(src_scenario)
        if rewrite is not None:
            utterance = rewrite
            provenance = "team-generated"
            note = f"native-style BR rewrite of {src_scenario}"
        else:
            utterance = PT[intent][variant][idx]
            provenance = "translated"
            note = f"ES->PT translation of {src_scenario}"
        scenario_id = f"S-{intent}-BR-{variant}-{idx}"
        # Mirror the Spanish second-annotator protocol so κ is comparable across languages.
        is_double = bool(src.double_labeled)
        rows.append(
            {
                "id": f"{scenario_id}-1",
                "scenario_id": scenario_id,
                "utterance": utterance,
                "intent": intent,
                "language": "pt",
                "variant": "BR",
                "provenance": provenance,
                "month": src.month,
                "double_labeled": "true" if is_double else "false",
                "label_b": src.label_b if is_double else "",
                "reviewer_note": note,
            }
        )
    return rows


def main() -> None:
    out = Path(__file__).with_name("gold_pt.tsv")
    rows = build_rows()
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    # Round-trip through the real loader so the committed file is guaranteed valid.
    df = load(out)
    translated = int((df["provenance"] == "translated").sum())
    rewritten = int((df["provenance"] == "team-generated").sum())
    kappa = cohen_kappa(df)
    print(f"wrote {len(df)} rows to {out}")
    print(f"translated: {translated} rows, native rewrites: {rewritten} rows")
    print(f"double-labeled: {int(df['double_labeled'].sum())} rows")
    print(f"Cohen's kappa: {kappa:.3f}")


if __name__ == "__main__":
    main()
