"""Retrieve-then-rerank pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from local_rerank_rag.retriever import Retriever
from local_rerank_rag.reranker import Reranker, keyword_overlap_score


@dataclass
class RagResult:
    """Result of a RAG pipeline run for a single query."""

    query: str
    retrieved: List[dict] = field(default_factory=list)
    reranked: List[dict] = field(default_factory=list)
    num_candidates: int = 0
    final_top_k: int = 0


def run_pipeline(
    retriever: Retriever,
    reranker: Reranker,
    query: str,
    retrieve_k: int = 20,
    final_k: int = 5,
) -> RagResult:
    """Run retrieve → rerank for a single query.

    1. Retrieve *retrieve_k* candidates from the vector store.
    2. Re-rank the candidates using the LLM-as-judge reranker.
    3. Take top *final_k*.
    """
    retrieved = retriever.retrieve(query, top_k=retrieve_k)
    reranked = reranker.rerank(query, retrieved)
    return RagResult(
        query=query,
        retrieved=retrieved,
        reranked=reranked,
        num_candidates=len(retrieved),
        final_top_k=final_k,
    )


def run_pipeline_keyword_baseline(
    retriever: Retriever,
    query: str,
    retrieve_k: int = 20,
    final_k: int = 5,
) -> RagResult:
    """Retrieve → rerank using keyword overlap (no LLM needed).

    Useful as a fast baseline for comparison.
    """
    retrieved = retriever.retrieve(query, top_k=retrieve_k)
    scored = []
    for c in retrieved:
        score = keyword_overlap_score(query, c["document"])
        scored.append({**c, "relevance_score": score})
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return RagResult(
        query=query,
        retrieved=retrieved,
        reranked=scored,
        num_candidates=len(retrieved),
        final_top_k=final_k,
    )


def evaluate_pipeline(
    retriever: Retriever,
    reranker: Reranker,
    corpus,
    retrieve_k: int = 20,
    final_k: int = 5,
    include_baseline: bool = False,
) -> Dict:
    """Run the pipeline for all queries in a corpus and compute metrics.

    Returns a dict with keys:
    - ``"reranked_mrr"`` — MRR after reranking
    - ``"retrieval_recall"`` — recall@k after initial retrieval
    - ``"baseline_mrr"`` — MRR after keyword-overlap reranking (if include_baseline)
    - ``"per_query"`` — per-query details
    """
    from local_rerank_rag.retrieval_metrics import (
        mean_reciprocal_rank,
        recall_at_k,
    )

    results = []
    for q_idx, query in enumerate(corpus.queries):
        result = run_pipeline(retriever, reranker, query, retrieve_k, final_k)
        results.append(result)

    rerank_ranked_lists = [[r["index"] for r in res.reranked[:final_k]] for res in results]
    rel_lists = [corpus.relevant_doc_ids.get(i, []) for i in range(len(corpus.queries))]

    reranked_mrr = mean_reciprocal_rank(rerank_ranked_lists, rel_lists)

    ret_ranked_lists = [[r["index"] for r in res.retrieved] for res in results]
    retrieval_recall = recall_at_k(ret_ranked_lists, rel_lists, k=retrieve_k)

    output: Dict = {
        "reranked_mrr": reranked_mrr,
        "retrieval_recall": retrieval_recall,
        "num_queries": len(corpus.queries),
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "per_query": [],
    }

    for q_idx, res in enumerate(results):
        output["per_query"].append({
            "query": res.query,
            "retrieved_indices": [r["index"] for r in res.retrieved],
            "reranked_indices": [r["index"] for r in res.reranked],
        })

    if include_baseline:
        baseline_results = []
        for query in corpus.queries:
            baseline_results.append(
                run_pipeline_keyword_baseline(retriever, query, retrieve_k, final_k)
            )
        baseline_lists = [[r["index"] for r in br.reranked[:final_k]] for br in baseline_results]
        output["baseline_mrr"] = mean_reciprocal_rank(baseline_lists, rel_lists)

    return output
