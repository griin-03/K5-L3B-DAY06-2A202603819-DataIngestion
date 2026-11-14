"""Embedding provider using fastembed with BAAI/bge-small-en-v1.5."""
from typing import Protocol
from fastembed import TextEmbedding


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class FastEmbedder:
    """Real semantic embedder using BAAI/bge-small-en-v1.5 (384 dimensions)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dimensions = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return dense embeddings for a list of document chunks."""
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_query(self, question: str) -> list[float]:
        """Return dense embedding for a query question."""
        embeddings = list(self._model.embed([question]))
        emb = embeddings[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)
