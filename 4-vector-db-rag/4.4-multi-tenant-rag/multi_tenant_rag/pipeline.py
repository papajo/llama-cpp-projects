"""Multi-tenant RAG pipeline — evaluate isolation and retrieval quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from multi_tenant_rag.data import MultiTenantCorpus
from multi_tenant_rag.retriever import TenantRetriever


@dataclass
class TenantEvalResult:
    """Evaluation result for a single tenant."""

    tenant_id: str
    isolated_recall: float  # recall@k within their own documents
    isolated_precision: float  # precision@k within their own documents
    leakage_count: int  # how many returned docs belong to OTHER tenants
    total_retrieved: int


@dataclass
class MultiTenantEvalResult:
    """Aggregated evaluation across all tenants."""

    num_tenants: int
    tenant_results: List[TenantEvalResult] = field(default_factory=list)

    @property
    def avg_isolated_recall(self) -> float:
        if not self.tenant_results:
            return 0.0
        return sum(r.isolated_recall for r in self.tenant_results) / len(self.tenant_results)

    @property
    def total_leakage(self) -> int:
        return sum(r.leakage_count for r in self.tenant_results)


def evaluate_isolation(
    retriever: TenantRetriever,
    corpus: MultiTenantCorpus,
    top_k: int = 5,
) -> MultiTenantEvalResult:
    """Evaluate tenant isolation and per-tenant retrieval quality.

    For each tenant:
    1. Run their queries with tenant isolation
    2. Measure recall@k and precision@k against their own documents
    3. Measure leakage: documents returned that belong to OTHER tenants
       (should be 0 with proper isolation)
    """
    result = MultiTenantEvalResult(num_tenants=len(corpus.tenants))

    for tenant in corpus.tenants:
        tid = tenant.id
        queries = corpus.queries.get(tid, [])
        relevant = corpus.relevant_doc_indices.get(tid, {})
        n_docs = len(tenant.documents)

        if not queries:
            continue

        total_recall = 0.0
        total_precision = 0.0
        total_leakage = 0
        total_retrieved = 0

        for q_idx, query in enumerate(queries):
            results = retriever.retrieve(tid, query, top_k=top_k)
            retrieved_indices = [r["index"] for r in results]
            rel = relevant.get(q_idx, [])
            rel_set = set(rel)

            # Recall: hits / total_relevant
            hits = sum(1 for i in retrieved_indices if i in rel_set)
            recall = hits / max(1, len(rel))
            total_recall += recall

            # Precision: hits / k
            precision = hits / max(1, top_k)
            total_precision += precision

            total_retrieved += len(retrieved_indices)

            # Leakage: none in isolated mode — we can only get own docs
            # But we check for out-of-range indices (belong to other tenants)
            for idx in retrieved_indices:
                if idx >= n_docs or idx < 0:
                    total_leakage += 1

        num_q = len(queries)
        result.tenant_results.append(
            TenantEvalResult(
                tenant_id=tid,
                isolated_recall=total_recall / num_q,
                isolated_precision=total_precision / num_q,
                leakage_count=total_leakage,
                total_retrieved=total_retrieved,
            )
        )

    return result


def evaluate_cross_tenant_leakage(
    retriever: TenantRetriever,
    corpus: MultiTenantCorpus,
    top_k: int = 5,
) -> MultiTenantEvalResult:
    """Simulate a *buggy* non-isolated retriever that searches all tenants.

    Used to demonstrate what happens without tenant isolation.
    """
    result = MultiTenantEvalResult(num_tenants=len(corpus.tenants))

    for tenant in corpus.tenants:
        tid = tenant.id
        queries = corpus.queries.get(tid, [])
        relevant = corpus.relevant_doc_indices.get(tid, {})
        n_docs = len(tenant.documents)

        if not queries:
            continue

        total_recall = 0.0
        total_precision = 0.0
        total_leakage = 0
        total_retrieved = 0

        for q_idx, query in enumerate(queries):
            # ⚠️ No tenant filter — searches across ALL tenants
            all_results = retriever.retrieve_all_tenants(query, top_k=top_k)
            # Flatten all results (may include docs from other tenants)
            flat = []
            for other_tid, tres in all_results.items():
                for r in tres:
                    flat.append({**r, "tenant_id": other_tid})

            # Sort by score across all tenants
            flat.sort(key=lambda x: x["score"], reverse=True)
            flat = flat[:top_k]

            retrieved_indices = [r["index"] for r in flat]
            rel = relevant.get(q_idx, [])
            rel_set = set(rel)

            # Recall: only counts if we find relevant docs from THIS tenant
            correct_hits = 0
            for r in flat:
                if r["tenant_id"] == tid and r["index"] in rel_set:
                    correct_hits += 1
            recall = correct_hits / max(1, len(rel))
            total_recall += recall

            precision = correct_hits / max(1, top_k)
            total_precision += precision

            total_retrieved += len(flat)

            # Leakage: count docs from OTHER tenants
            leakage = sum(1 for r in flat if r["tenant_id"] != tid)
            total_leakage += leakage

        num_q = len(queries)
        result.tenant_results.append(
            TenantEvalResult(
                tenant_id=tid,
                isolated_recall=total_recall / num_q,
                isolated_precision=total_precision / num_q,
                leakage_count=total_leakage,
                total_retrieved=total_retrieved,
            )
        )

    return result
