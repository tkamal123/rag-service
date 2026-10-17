"""Print a markdown table comparing every result file. Paste it into your README.

    python -m eval.compare
"""
import json
from pathlib import Path

rows = [json.loads(p.read_text()) for p in sorted(Path(__file__).parent.glob("results/*.json"))]
if not rows:
    raise SystemExit("No results yet. Run: python -m eval.run_eval --label baseline")

cols = ["label", "chunk_size", "top_k", "hit_rate", "mrr", "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "false_refusal", "correct_refusal"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    ragas, g, ret = r.get("ragas", {}), r.get("guardrails", {}), r["retrieval"]
    vals = [
        r["label"], r.get("index", {}).get("chunk_size", "?"), r["top_k"],
        next((v for k, v in ret.items() if k.startswith("hit_rate")), "-"), ret.get("mrr", "-"),
        ragas.get("faithfulness", "-"), ragas.get("answer_relevancy", "-"),
        ragas.get("llm_context_precision_with_reference", "-"), ragas.get("context_recall", "-"),
        g.get("false_refusal_rate", "-"), g.get("correct_refusal_rate", "-"),
    ]
    print("| " + " | ".join(str(v) for v in vals) + " |")
