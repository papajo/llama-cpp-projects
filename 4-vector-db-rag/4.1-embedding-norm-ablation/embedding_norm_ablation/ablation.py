"""Ablation experiment runner — comparing normalised vs. unnormalised embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from embedding_norm_ablation.data import Corpus
from embedding_norm_ablation.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    batch_normalize,
    norm_stats,
)
from embedding_norm_ablation.retrieval import (
    aggregate_results,
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    evaluate_retrieval,
)


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment.

    ``corpus`` is typically set at experiment time; treatments defined at
    module level keep it as ``None`` and ``run_ablation`` assigns it.
    """

    corpus: Optional[Corpus] = None
    server_url: str = "http://localhost:8080"
    model: Optional[str] = None
    top_k: int = 5
    similarity_fn: Callable = cosine_similarity
    label: str = "cosine"


# Pre-defined configurations for the four standard treatments
TREATMENT_UNNORM_COSINE = AblationConfig(
    label="unnorm-cosine",
    similarity_fn=cosine_similarity,
)

TREATMENT_NORM_COSINE = AblationConfig(
    label="norm-cosine",
    similarity_fn=cosine_similarity,
)

TREATMENT_UNNORM_DOT = AblationConfig(
    label="unnorm-dot",
    similarity_fn=dot_product_similarity,
)

TREATMENT_NORM_DOT = AblationConfig(
    label="norm-dot",
    similarity_fn=dot_product_similarity,
)

TREATMENT_UNNORM_EUCLIDEAN = AblationConfig(
    label="unnorm-euclidean",
    similarity_fn=euclidean_distance,
)

TREATMENT_NORM_EUCLIDEAN = AblationConfig(
    label="norm-euclidean",
    similarity_fn=euclidean_distance,
)

STANDARD_TREATMENTS = [
    TREATMENT_UNNORM_COSINE,
    TREATMENT_NORM_COSINE,
    TREATMENT_UNNORM_DOT,
    TREATMENT_NORM_DOT,
]
"""The four core treatments for a standard ablation run."""


@dataclass
class TreatmentResult:
    """Results for a single treatment."""

    label: str
    per_query_metrics: List[Dict[str, float]]
    aggregated: Dict[str, float]
    doc_norm_stats: Dict[str, float]
    query_norm_stats: Dict[str, float]


@dataclass
class AblationResult:
    """Complete ablation experiment result."""

    config: AblationConfig
    treatments: List[TreatmentResult] = field(default_factory=list)
    corpus_size: int = 0
    num_queries: int = 0
    embedding_dim: int = 0

    def get_treatment(self, label: str) -> Optional[TreatmentResult]:
        """Look up a treatment by label."""
        for t in self.treatments:
            if t.label == label:
                return t
        return None

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return a nested dict: treatment_label → {metric → value}."""
        return {t.label: t.aggregated for t in self.treatments}


def run_ablation(
    client: EmbeddingClient,
    config: AblationConfig,
    treatments: Optional[List[AblationConfig]] = None,
) -> AblationResult:
    """Run a full ablation experiment.

    1. Embed all documents and queries via *client*.
    2. For each treatment (similarity function + normalisation), compute
       per-query retrieval metrics.
    3. Return an ``AblationResult`` with all data.

    If *treatments* is ``None``, the four standard treatments are used.
    """
    if treatments is None:
        treatments = STANDARD_TREATMENTS

    corpus = config.corpus

    # --- embed once ---
    try:
        doc_vecs = client.embed_many(corpus.documents, model=config.model)
        query_vecs = client.embed_many(corpus.queries, model=config.model)
    except EmbeddingError as exc:
        raise AblationError(str(exc)) from exc

    if not doc_vecs or not query_vecs:
        raise AblationError("No embeddings returned from server")

    embedding_dim = int(doc_vecs[0].shape[0])

    # --- norm stats for raw vectors ---
    doc_ns = norm_stats(doc_vecs)
    query_ns = norm_stats(query_vecs)

    # --- build result ---
    result = AblationResult(
        config=config,
        corpus_size=len(corpus.documents),
        num_queries=len(corpus.queries),
        embedding_dim=embedding_dim,
    )

    for tx in treatments:
        # determine if this treatment requires normalised vectors
        should_norm = tx.label.startswith("norm-")

        if should_norm:
            tx_docs = batch_normalize(doc_vecs)
            tx_queries = batch_normalize(query_vecs)
            tx_doc_ns = norm_stats(tx_docs)
            tx_query_ns = norm_stats(tx_queries)
        else:
            tx_docs = doc_vecs
            tx_queries = query_vecs
            tx_doc_ns = doc_ns
            tx_query_ns = query_ns

        per_q = evaluate_retrieval(
            tx_queries,
            tx_docs,
            relevant=corpus.relevant_doc_ids,
            similarity_fn=tx.similarity_fn,
            top_k=tx.top_k or config.top_k,
        )
        agg = aggregate_results(per_q)

        result.treatments.append(
            TreatmentResult(
                label=tx.label,
                per_query_metrics=per_q,
                aggregated=agg,
                doc_norm_stats=tx_doc_ns,
                query_norm_stats=tx_query_ns,
            )
        )

    return result


class AblationError(Exception):
    """Raised when an ablation experiment fails."""
