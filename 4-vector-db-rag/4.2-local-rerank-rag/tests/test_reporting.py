"""Tests for reporting."""

from local_rerank_rag.reporting import evaluation_to_json, evaluation_to_markdown


class TestReporting:
    def test_markdown_contains_summary(self):
        ev = {
            "reranked_mrr": 0.75,
            "retrieval_recall": 0.9,
            "num_queries": 2,
            "retrieve_k": 20,
            "final_k": 5,
            "per_query": [
                {"query": "q1", "retrieved_indices": [0, 1], "reranked_indices": [1, 0]},
            ],
        }
        md = evaluation_to_markdown(ev)
        assert "RAG Pipeline Evaluation Report" in md
        assert "0.7500" in md
        assert "0.9000" in md

    def test_markdown_with_baseline(self):
        ev = {
            "reranked_mrr": 0.8,
            "baseline_mrr": 0.5,
            "retrieval_recall": 0.9,
            "num_queries": 2,
            "retrieve_k": 20,
            "final_k": 5,
            "per_query": [],
        }
        md = evaluation_to_markdown(ev)
        assert "LLM Reranker vs Keyword Baseline" in md
        assert "+0.3000" in md

    def test_json_roundtrip(self):
        ev = {"reranked_mrr": 0.75, "num_queries": 5, "per_query": []}
        import json
        data = json.loads(evaluation_to_json(ev))
        assert data["reranked_mrr"] == 0.75
