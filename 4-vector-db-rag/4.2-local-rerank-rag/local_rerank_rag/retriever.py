"""Document retriever — embed documents and search by vector similarity."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


class EmbeddingClient:
    """Client for ``/v1/embeddings`` on a llama.cpp server."""

    def __init__(self, server_url: str = "http://localhost:8080", timeout: int = 60):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def embed(self, text: str, model: Optional[str] = None) -> np.ndarray:
        import urllib.error
        import urllib.request

        payload: dict = {"input": text}
        if model is not None:
            payload["model"] = model
        data = json.dumps(payload).encode()
        try:
            req = urllib.request.Request(
                f"{self.server_url}/v1/embeddings",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            raise EmbeddingError(str(exc)) from exc
        try:
            return np.array(body["data"][0]["embedding"], dtype=np.float32)
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(f"Unexpected response: {body}") from exc

    def embed_many(self, texts: List[str], model: Optional[str] = None) -> List[np.ndarray]:
        return [self.embed(t, model=model) for t in texts]


class EmbeddingError(Exception):
    pass


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


@dataclass
class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    documents: List[str] = field(default_factory=list)
    vectors: List[np.ndarray] = field(default_factory=list)

    def add(self, document: str, vector: np.ndarray) -> None:
        self.documents.append(document)
        self.vectors.append(vector)

    def add_many(self, documents: List[str], vectors: List[np.ndarray]) -> None:
        self.documents.extend(documents)
        self.vectors.extend(vectors)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        similarity_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> List[int]:
        """Return indices of top-k most similar documents, descending."""
        if similarity_fn is None:
            similarity_fn = cosine_similarity
        scores = [similarity_fn(query_vec, v) for v in self.vectors]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def __len__(self) -> int:
        return len(self.documents)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v.copy()


@dataclass
class Retriever:
    """High-level retriever that embeds queries and searches the store."""

    store: VectorStore
    client: EmbeddingClient
    embed_model: Optional[str] = None

    def retrieve(
        self, query: str, top_k: int = 10
    ) -> List[Dict]:
        """Embed the query and return top-k results with scores.

        Each result: ``{"index": int, "document": str, "score": float}``.
        """
        qvec = self.client.embed(query, model=self.embed_model)
        indices = self.store.search(qvec, top_k=top_k)
        results = []
        for idx in indices:
            score = cosine_similarity(qvec, self.store.vectors[idx])
            results.append({
                "index": idx,
                "document": self.store.documents[idx],
                "score": score,
            })
        return results
