"""Evaluate the RAG pipeline.

Always computes (free, no API key):
  - hit_rate@k and MRR of the expected source document in the retrieved chunks
Optionally computes (needs an LLM API key, e.g. GEMINI_API_KEY):
  - refusal behaviour on unanswerable questions
  - RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
    (uses the LLM chosen by LLM_PROVIDER; free tiers work, just slowly)

Usage:
    python -m eval.run_eval --label baseline --retrieval-only
    python -m eval.run_eval --label baseline
    python -m eval.run_eval --label chunk300 --top-k 3
"""
import argparse
import json
import os
import time
from pathlib import Path

from app.config import get_settings
from app.rag import RAGPipeline, VectorStore, make_llm

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS_DIR = Path(__file__).parent / "results"


def retrieval_metrics(pipeline: RAGPipeline, items: list[dict], k: int) -> dict:
    hits, rr = 0, 0.0
    for it in items:
        sources = [p.source for p in pipeline.retrieve(it["question"], k)]
        if it["source"] in sources:
            hits += 1
            rr += 1.0 / (sources.index(it["source"]) + 1)
    n = len(items)
    return {f"hit_rate@{k}": round(hits / n, 3), "mrr": round(rr / n, 3)}


def make_judge(settings):
    """LangChain chat model for RAGAS, using the same provider as the app."""
    model = os.getenv("JUDGE_MODEL") or settings.llm_model
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0, max_tokens=2048)
    from langchain_openai import ChatOpenAI

    key = os.getenv(settings.llm_key_env) if settings.llm_key_env else "ollama"
    if not key:
        raise SystemExit(f"Set the {settings.llm_key_env} environment variable first.")
    return ChatOpenAI(model=model, base_url=settings.llm_base_url, api_key=key,
                      temperature=0, max_tokens=2048, max_retries=5)


def ragas_scores(rows: list[dict], settings) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig

    from eval.embeddings import ChromaDefaultEmbeddings

    judge = LangchainLLMWrapper(make_judge(settings))
    emb = LangchainEmbeddingsWrapper(ChromaDefaultEmbeddings())
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r["contexts"],
                response=r["answer"],
                reference=r["ground_truth"],
            )
            for r in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(strictness=1),  # many providers can't return n>1 completions
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=emb,
        # Low concurrency + generous retries keep free-tier rate limits (RPM) from failing jobs.
        run_config=RunConfig(max_workers=int(os.getenv("EVAL_WORKERS", "2")), timeout=240, max_retries=8),
        show_progress=True,
    )
    scores = {}
    for k in result.scores[0].keys():
        vals = [s[k] for s in result.scores if s[k] == s[k]]  # drop NaN (judge output failed to parse)
        scores[k] = round(sum(vals) / len(vals), 3) if vals else None
        scores[f"{k}_valid_samples"] = len(vals)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="name for this run, e.g. baseline or chunk300")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    k = args.top_k or settings.top_k
    data = json.loads(EVAL_SET.read_text())
    pipeline = RAGPipeline(VectorStore(settings), make_llm(settings), settings)
    meta_path = Path(settings.chroma_dir) / "index_meta.json"
    index_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    out = {"label": args.label, "top_k": k, "llm": f"{settings.llm_provider}:{settings.llm_model}", "max_distance": settings.max_distance, "index": index_meta,
           "n_answerable": len(data["answerable"]), "n_unanswerable": len(data["unanswerable"])}
    out["retrieval"] = retrieval_metrics(pipeline, data["answerable"], k)

    if not args.retrieval_only:
        rows, false_refusals = [], 0
        for it in data["answerable"]:
            res = pipeline.answer(it["question"])
            false_refusals += res.refused
            rows.append({**it, "answer": res.answer, "contexts": [p.text for p in res.sources]})
        refused = sum(pipeline.answer(q).refused for q in data["unanswerable"])
        out["guardrails"] = {
            "correct_refusal_rate": round(refused / len(data["unanswerable"]), 3),
            "false_refusal_rate": round(false_refusals / len(data["answerable"]), 3),
        }
        out["ragas"] = ragas_scores(rows, settings)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved {path}  ({time.strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
