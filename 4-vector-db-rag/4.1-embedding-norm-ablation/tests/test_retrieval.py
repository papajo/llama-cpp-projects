"""Tests for similarity functions and retrieval evaluation."""

import numpy as np
import pytest

from embedding_norm_ablation.retrieval import (
    aggregate_results,
    average_precision,
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    retrieve,
)


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_clamped(self):
        """Near-parallel vectors should not exceed 1.0."""
        a = np.array([1e10, 1e-10])
        b = np.array([1e10, 1e-10])
        sim = cosine_similarity(a, b)
        assert sim <= 1.0 + 1e-12


class TestDotProductSimilarity:
    def test_basic(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert dot_product_similarity(a, b) == pytest.approx(11.0)

    def test_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        assert dot_product_similarity(a, b) == 0.0


class TestEuclideanDistance:
    def test_identical_zero(self):
        a = np.array([1.0, 2.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_basic(self):
        a = np.array([1.0, 0.0])
        b = np.array([4.0, 0.0])
        assert euclidean_distance(a, b) == pytest.approx(3.0)


class TestRetrieve:
    def test_basic_retrieval(self):
        query = np.array([1.0, 0.0])
        docs = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.9, 0.1]),
        ]
        results = retrieve(query, docs, top_k=2)
        assert results[0] == 0
        assert results[1] == 2


class TestPrecisionRecall:
    def test_precision_at_k_perfect(self):
        assert precision_at_k([0, 1, 2], [0, 1], k=2) == 1.0

    def test_precision_at_k_partial(self):
        assert precision_at_k([0, 1, 2], [0], k=2) == 0.5

    def test_precision_at_k_zero_k(self):
        assert precision_at_k([0, 1], [0], k=0) == 0.0

    def test_recall_at_k_perfect(self):
        assert recall_at_k([0, 1, 2], [0, 1], k=3) == 1.0

    def test_recall_at_k_partial(self):
        assert recall_at_k([0, 1], [0, 2], k=2) == 0.5

    def test_recall_at_k_no_relevant(self):
        assert recall_at_k([0, 1], [], k=2) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        """Relevant docs at ranks 1 and 2 → AP = (1/1 + 2/2)/2 = 1.0"""
        ap = average_precision([0, 1, 2], [0, 1])
        assert ap == pytest.approx(1.0)

    def test_partial(self):
        """Relevant doc at rank 3 only → AP = (1/3)/2 = 0.1667"""
        ap = average_precision([2, 1, 0], [0])
        assert ap == pytest.approx(1.0 / 3)

    def test_empty_relevant(self):
        assert average_precision([0, 1], []) == 0.0


class TestMeanReciprocalRank:
    def test_basic(self):
        retrieved = [[0, 1, 2], [0, 1, 2]]
        relevant = [[0], [1]]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        # query0: first relevant at rank 1 → 1/1 = 1.0
        # query1: first relevant at rank 2 → 1/2 = 0.5
        # MRR = (1.0 + 0.5) / 2 = 0.75
        assert mrr == pytest.approx(0.75)

    def test_empty(self):
        assert mean_reciprocal_rank([], []) == 0.0


class TestEvaluateRetrieval:
    def test_full_evaluation(self):
        query_vecs = [np.array([1.0, 0.0])]
        doc_vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        relevant = {0: [0]}
        results = evaluate_retrieval(query_vecs, doc_vecs, relevant, top_k=2)
        assert len(results) == 1
        assert results[0]["p@1"] == 1.0


class TestAggregateResults:
    def test_averages(self):
        results = [
            {"p@1": 1.0, "p@5": 0.8, "r@5": 0.5, "ap": 0.9},
            {"p@1": 0.0, "p@5": 0.4, "r@5": 0.3, "ap": 0.2},
        ]
        agg = aggregate_results(results)
        assert agg["mean_p@1"] == pytest.approx(0.5)
        assert agg["mean_p@5"] == pytest.approx(0.6)

    def test_empty(self):
        assert aggregate_results([]) == {}
