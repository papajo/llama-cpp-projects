"""Embedding client and tenant-aware retriever."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from multi_tenant_rag.tenant_store import TenantVectorStore, cosine_similarity


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


@dataclass
class TenantRetriever:
    """Retrieves documents for a specific tenant from the tenant store."""

    store: TenantVectorStore
    client: EmbeddingClient
    embed_model: Optional[str] = None

    def retrieve(
        self, tenant_id: str, query: str, top_k: int = 5
    ) -> List[Dict]:
        """Retrieve documents for *tenant_id*.

        Returns list of ``{"index": int, "document": str, "score": float}``.
        """
        qvec = self.client.embed(query, model=self.embed_model)
        indices = self.store.search(tenant_id, qvec, top_k=top_k)
        results = []
        for idx in indices:
            score = cosine_similarity(qvec, self.store._vectors[tenant_id][idx])
            results.append({
                "index": idx,
                "document": self.store._documents[tenant_id][idx],
                "score": score,
            })
        return results

    def retrieve_all_tenants(
        self, query: str, top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """Search across all tenants (no isolation)."""
        qvec = self.client.embed(query, model=self.embed_model)
        all_results = self.store.search_all_tenants(qvec, top_k=top_k)
        output: Dict[str, List[Dict]] = {}
        for tid, indices in all_results.items():
            output[tid] = []
            for idx in indices:
                score = cosine_similarity(qvec, self.store._vectors[tid][idx])
                output[tid].append({
                    "index": idx,
                    "document": self.store._documents[tid][idx],
                    "score": score,
                })
        return output
