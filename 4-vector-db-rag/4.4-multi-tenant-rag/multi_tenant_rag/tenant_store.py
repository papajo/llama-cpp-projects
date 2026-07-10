"""Tenant-aware vector store with isolation guarantees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


@dataclass
class TenantVectorStore:
    """In-memory vector store partitioned by tenant ID.

    Documents added under a tenant ID can only be retrieved when
    querying with that same tenant ID.
    """

    _documents: Dict[str, List[str]] = field(default_factory=dict)
    _vectors: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def add(self, tenant_id: str, document: str, vector: np.ndarray) -> None:
        if tenant_id not in self._documents:
            self._documents[tenant_id] = []
            self._vectors[tenant_id] = []
        self._documents[tenant_id].append(document)
        self._vectors[tenant_id].append(vector)

    def add_many(
        self, tenant_id: str, documents: List[str], vectors: List[np.ndarray]
    ) -> None:
        if tenant_id not in self._documents:
            self._documents[tenant_id] = []
            self._vectors[tenant_id] = []
        self._documents[tenant_id].extend(documents)
        self._vectors[tenant_id].extend(vectors)

    def search(
        self,
        tenant_id: str,
        query_vec: np.ndarray,
        top_k: int = 5,
        similarity_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> List[int]:
        """Search documents belonging to *tenant_id* only.

        Returns up to *top_k* document indices (relative to the tenant's
        document list), sorted descending by similarity.
        """
        if similarity_fn is None:
            similarity_fn = cosine_similarity
        docs = self._documents.get(tenant_id, [])
        vecs = self._vectors.get(tenant_id, [])
        if not vecs:
            return []
        scores = [similarity_fn(query_vec, v) for v in vecs]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def search_all_tenants(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        similarity_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> Dict[str, List[int]]:
        """Search across all tenants (no isolation).

        Used as a baseline to measure cross-tenant leakage.
        """
        result: Dict[str, List[int]] = {}
        for tid in self._documents:
            result[tid] = self.search(tid, query_vec, top_k, similarity_fn)
        return result

    def tenant_count(self) -> int:
        return len(self._documents)

    def doc_count(self, tenant_id: str) -> int:
        return len(self._documents.get(tenant_id, []))

    def total_docs(self) -> int:
        return sum(len(d) for d in self._documents.values())

    def tenants(self) -> List[str]:
        return list(self._documents.keys())
