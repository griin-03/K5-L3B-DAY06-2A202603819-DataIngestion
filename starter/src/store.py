"""Persistent Vector Store implementation using ChromaDB."""
from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings
from .models import Chunk


class ChromaVectorStore:
    """Vector store supporting persistent collections, cosine metric, and metadata filtering."""

    def __init__(self, persist_directory: str = "./results/chromadb", dimensions: int = 384) -> None:
        self.persist_directory = persist_directory
        self.dimensions = dimensions
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

    def create_schema(self, strategy: str) -> None:
        """Create or reset collection for a given strategy."""
        collection_name = f"strategy_{strategy}"
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunks, vectors, and metadata reliably."""
        collection_name = f"strategy_{strategy}"
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            # Flatten metadata for ChromaDB (scalars only)
            meta = {
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.metadata.get("section", ""),
                "strategy": chunk.metadata.get("strategy", strategy),
                "chunk_size": chunk.metadata.get("chunk_size", len(chunk.text))
            }
            metadatas.append(meta)

        # Batch add chunks
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=vectors[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                documents=documents[i:i + batch_size]
            )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query top-k chunks by cosine similarity with optional metadata filter."""
        collection_name = f"strategy_{strategy}"
        collection = self.client.get_collection(name=collection_name)
        
        where_filter = None
        if filters:
            # ChromaDB expects scalar equality in where clause
            where_filter = filters

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        hits: list[dict[str, Any]] = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                doc_text = results["documents"][0][idx] if results["documents"] else ""
                metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                dist = results["distances"][0][idx] if results["distances"] else 0.0
                # Cosine distance to similarity score
                similarity = 1.0 - dist
                hits.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "score": round(similarity, 4),
                    "distance": round(dist, 4),
                    "metadata": metadata
                })

        return hits
