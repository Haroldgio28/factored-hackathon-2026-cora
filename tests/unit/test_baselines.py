"""Tests for task 3.4: the three intent baselines and the shared LLM client (REQ-29).

No network and no AWS: the zero-shot baseline is driven by `StubLLMClient` with canned replies, and
`get_llm_client` is checked to return the stub under `CORA_NLU_STUB=1`. The keyword rules are pinned
on one ES and one PT utterance per the languages the gold set actually carries, plus the OTHER
default. The majority baseline is checked to be the deterministic modal class.
"""

from __future__ import annotations

from cora.agent.llm import StubLLMClient, get_llm_client
from cora.nlu.baselines import KeywordBaseline, MajorityBaseline, ZeroShotBaseline
from cora.nlu.labels import Intent


def test_majority_predicts_deterministic_modal_class() -> None:
    train = ["I1", "I1", "I1", "I2", "E3"]
    maj = MajorityBaseline(train)
    assert maj.majority == Intent.I1
    assert maj.predict(["anything", "else"]) == [Intent.I1, Intent.I1]


def test_majority_breaks_ties_deterministically() -> None:
    # I2 and I1 tie; the modal set is sorted so the pick is stable regardless of insertion order.
    assert MajorityBaseline(["I2", "I1"]).majority == Intent.I1


def test_keyword_matches_es_and_pt_and_defaults_to_other() -> None:
    kw = KeywordBaseline()
    # Spanish: a balance question and a freeze request.
    assert kw.predict_one("¿Cuánto dinero tengo en mi cuenta?", "es") == Intent.I1
    assert kw.predict_one("Por favor congela mi tarjeta", "es") == Intent.A1
    # Portuguese: a transactions question and a human-agent request.
    assert kw.predict_one("Quero ver minhas compras recentes", "pt") == Intent.I2
    assert kw.predict_one("Quero falar com uma pessoa, não com um robô", "pt") == Intent.E4
    # No keyword hit -> OTHER (fail safe to the catch-all, never a guess).
    assert kw.predict_one("asdf qwerty", "es") == Intent.OTHER


def test_keyword_specific_intent_wins_over_generic() -> None:
    kw = KeywordBaseline()
    # Contains "compra" (could look like I2) but it is a dispute; the E1 rule is ordered first.
    assert kw.predict_one("No reconozco esta compra, quiero disputarla", "es") == Intent.E1
    # Contains "pagar la tarjeta" (money movement), not a card-status read.
    assert kw.predict_one("Quiero pagar mi tarjeta desde mi cuenta", "es") == Intent.X2


def test_keyword_predict_requires_matching_language_length() -> None:
    kw = KeywordBaseline()
    try:
        kw.predict(["a", "b"], ["es"])
    except ValueError:
        pass
    else:
        raise AssertionError("predict must reject a languages/utterances length mismatch")


def test_zero_shot_valid_label_and_fail_closed_on_garbage() -> None:
    # The stub sees the full rendered prompt (which lists every label word), so key the canned
    # replies on sentinels that only appear in the test utterance, not in the prompt body.
    client = StubLLMClient(
        responses={"zzbalance": "I1", "zzfreeze": "A1", "zzgarbage": "NOT_A_LABEL blah"},
        default="",
    )
    zs = ZeroShotBaseline(client=client)
    assert zs.predict_one("zzbalance", "es") == Intent.I1
    assert zs.predict_one("zzfreeze", "es") == Intent.A1
    # Invalid label text and empty reply both fail closed to OTHER.
    assert zs.predict_one("zzgarbage", "es") == Intent.OTHER
    assert zs.predict_one("zznothing", "es") == Intent.OTHER


def test_zero_shot_extracts_label_embedded_in_a_sentence() -> None:
    client = StubLLMClient(responses={"zzbalance": "La etiqueta es I1."})
    assert ZeroShotBaseline(client=client).predict_one("zzbalance", "es") == Intent.I1


def test_get_llm_client_returns_stub_under_env_flag(monkeypatch) -> None:
    monkeypatch.setenv("CORA_NLU_STUB", "1")
    assert isinstance(get_llm_client(), StubLLMClient)
