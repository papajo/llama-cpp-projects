"""Similarity functions and retrieval evaluation metrics."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from embedding_norm_ablation.embeddings import l2_normalize

# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0.0–1.0).

    Values are clamped to [-1, 1] before returning to avoid
    floating-point drift outside that range.
    """
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def dot_product_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot-product similarity between two vectors.

    For L2-normalised vectors this is equivalent to cosine similarity.
    """
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query_vec: np.ndarray,
    doc_vecs: List[np.ndarray],
    similarity_fn: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
    top_k: int = 5,
) -> List[int]:
    """Return indices of the top-*k* most similar documents.

    Results are sorted descending by similarity score.
    """
    scores = [similarity_fn(query_vec, d) for d in doc_vecs]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def precision_at_k(
    retrieved: List[int], relevant: List[int], k: Optional[int] = None
) -> float:
    """Precision@k = (relevant retrieved) / k."""
    if k is None:
        k = len(retrieved)
    if k == 0:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for i in retrieved[:k] if i in rel_set)
    return hits / k


def recall_at_k(
    retrieved: List[int], relevant: List[int], k: Optional[int] = None
) -> float:
    """Recall@k = (relevant retrieved) / (total relevant)."""
    if k is None:
        k = len(retrieved)
    if len(relevant) == 0:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for i in retrieved[:k] if i in rel_set)
    return hits / len(relevant)


def average_precision(retrieved: List[int], relevant: List[int]) -> float:
    """Average precision (AP) for a single query.

    AP = sum(P@k for each relevant doc found) / total_relevant
    """
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    score = 0.0
    hits = 0
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel_set:
            hits += 1
            score += hits / k
    return score / len(relevant)


def mean_reciprocal_rank(
    retrieved_list: List[List[int]], relevant_list: List[List[int]]
) -> float:
    """Mean Reciprocal Rank across multiple queries.

    MRR = (1/N) * sum(1 / rank_of_first_relevant)
    """
    total = 0.0
    n = len(retrieved_list)
    if n == 0:
        return 0.0
    for retrieved, relevant in zip(retrieved_list, relevant_list):
        rel_set = set(relevant)
        rank = 1
        found = False
        for doc_id in retrieved:
            if doc_id in rel_set:
                total += 1.0 / rank
                found = True
                break
            rank += 1
        if not found:
            total += 0.0
    return total / n


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------


EvalResult = Dict[str, float]
"""Per-query evaluation results: ``{"p@1": …, "p@5": …, "r@5": …, "ap": …}``."""


def evaluate_retrieval(
    query_vecs: List[np.ndarray],
    doc_vecs: List[np.ndarray],
    relevant: Dict[int, List[int]],
    similarity_fn: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
    top_k: int = 5,
) -> List[EvalResult]:
    """Run retrieval and produce per-query metrics.

    Returns one dict per query with keys ``p@1``, ``p@5``, ``r@5``, ``ap``.
    """
    results: List[EvalResult] = []
    for q_idx, qvec in enumerate(query_vecs):
        retrieved = retrieve(qvec, doc_vecs, similarity_fn=similarity_fn, top_k=top_k)
        rel = relevant.get(q_idx, [])
        results.append(
            {
                "p@1": precision_at_k(retrieved, rel, k=1),
                "p@5": precision_at_k(retrieved, rel, k=5),
                "r@5": recall_at_k(retrieved, rel, k=5),
                "ap": average_precision(retrieved, rel),
            }
        )
    return results


def aggregate_results(results: List[EvalResult]) -> Dict[str, float]:
    """Average per-query metrics across all queries.

    Returns a single dict with the mean of each metric.
    """
    if not results:
        return {}
    keys = results[0].keys()
    aggregated = {}
    for key in keys:
        values = [r[key] for r in results]
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    return aggregated
