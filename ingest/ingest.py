"""Chunk every .md/.txt file in data/docs and load it into Chroma.

Usage:
    python -m ingest.ingest                        # defaults
    python -m ingest.ingest --chunk-size 300 --overlap 50
"""
import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.rag import VectorStore, chunk_text


def run(docs_dir: str, chunk_size: int, overlap: int, embedding_function=None) -> dict:
    settings = get_settings()
    store = VectorStore(settings, embedding_function)
    store.reset()
    total, files = 0, 0
    for path in sorted(Path(docs_dir).glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        total += store.add(chunk_text(path.read_text(encoding="utf-8"), chunk_size, overlap), path.name)
        files += 1
    meta = {"files": files, "chunks": total, "chunk_size": chunk_size, "overlap": overlap}
    # The eval script reads this so every result file records how the index was built.
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    (Path(settings.chroma_dir) / "index_meta.json").write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=80)
    a = ap.parse_args()
    print(run(a.docs, a.chunk_size, a.overlap))
