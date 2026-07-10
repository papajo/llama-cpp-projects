"""Retrieval evaluation metrics."""

from typing import List


def mean_reciprocal_rank(
    ranked_lists: List[List[int]],
    relevant_lists: List[List[int]],
) -> float:
    """MRR = (1/N) * sum(1/rank_of_first_relevant)."""
    total = 0.0
    n = len(ranked_lists)
    if n == 0:
        return 0.0
    for ranked, relevant in zip(ranked_lists, relevant_lists):
        rel_set = set(relevant)
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in rel_set:
                total += 1.0 / rank
                break
    return total / n


def recall_at_k(
    ranked_lists: List[List[int]],
    relevant_lists: List[List[int]],
    k: int,
) -> float:
    """Average recall@k across queries."""
    total = 0.0
    n = len(ranked_lists)
    if n == 0:
        return 0.0
    for ranked, relevant in zip(ranked_lists, relevant_lists):
        if not relevant:
            continue
        rel_set = set(relevant)
        hits = sum(1 for d in ranked[:k] if d in rel_set)
        total += hits / len(relevant)
    return total / n
