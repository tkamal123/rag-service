import json
from pathlib import Path

from app.rag import REFUSAL_MESSAGE

EVAL = json.loads((Path(__file__).parent.parent / "eval" / "eval_set.json").read_text())


def test_retrieval_finds_the_right_document(pipeline):
    sources = [p.source for p in pipeline.retrieve("monthly plan refund within 14 days", k=3)]
    assert "refund_policy.md" in sources


def test_retrieval_hit_rate_on_eval_set_is_high(pipeline):
    items = EVAL["answerable"]
    hits = sum(it["source"] in [p.source for p in pipeline.retrieve(it["question"], k=4)] for it in items)
    assert hits / len(items) >= 0.8


def test_answer_returns_sources_and_calls_llm_once(pipeline):
    res = pipeline.answer("How long is the refund window for monthly plans?")
    assert not res.refused and res.sources and pipeline.llm.calls == 1


def test_out_of_domain_question_is_refused_without_calling_llm(pipeline):
    res = pipeline.answer("quantum chromodynamics lattice simulation")
    assert res.refused and res.reason == "low_retrieval_confidence"
    assert res.answer == REFUSAL_MESSAGE and pipeline.llm.calls == 0


def test_pii_in_question_is_redacted_and_reported(pipeline):
    res = pipeline.answer("My email is a.b@example.com, what is the refund window?")
    assert "email" in res.pii_redacted


def test_pii_in_model_output_is_redacted(pipeline):
    pipeline.llm.reply = "Contact billing@acme.com for refunds [1]."
    res = pipeline.answer("How do I request a refund?")
    assert "billing@acme.com" not in res.answer and "email" in res.pii_redacted


def test_model_saying_idk_is_flagged_as_refused(pipeline):
    pipeline.llm.reply = REFUSAL_MESSAGE
    res = pipeline.answer("How do I request a refund?")
    assert res.refused and res.reason == "model_declined"
