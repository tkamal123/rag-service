"""Shared fixtures. Tests use a hashing embedder and a fake LLM: offline, free and deterministic."""
import hashlib
import math
import re
from pathlib import Path

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from app.config import get_settings
from app.rag import RAGPipeline, VectorStore
from ingest.ingest import run as ingest_run

DOCS = str(Path(__file__).parent.parent / "data" / "docs")
STOP = {"the", "a", "an", "of", "to", "is", "are", "in", "on", "for", "and", "or", "do", "i", "my", "can", "how", "what", "does"}


class HashEmbedding(EmbeddingFunction[Documents]):
    """Bag-of-words hashed into 256 dims. Crude, but similar texts land close together."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            vec = [0.0] * 256
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in STOP:
                    continue
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FakeLLM:
    def __init__(self, reply="The refund window is 14 days [1]."):
        self.reply, self.calls = reply, 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("MAX_DISTANCE", "0.9")
    ingest_run(DOCS, 500, 80, embedding_function=HashEmbedding())
    settings = get_settings()
    return RAGPipeline(VectorStore(settings, HashEmbedding()), FakeLLM(), settings)
