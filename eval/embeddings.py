"""LangChain-compatible embeddings backed by Chroma's built-in ONNX MiniLM model.

RAGAS' answer_relevancy needs an embedding model. Reusing Chroma's default avoids pulling in torch.
"""
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
