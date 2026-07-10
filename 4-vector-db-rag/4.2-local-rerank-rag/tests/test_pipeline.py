"""Tests for the pipeline module."""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from local_rerank_rag.data import mini_corpus
from local_rerank_rag.pipeline import (
    evaluate_pipeline,
    run_pipeline,
    run_pipeline_keyword_baseline,
)
from local_rerank_rag.retriever import EmbeddingClient, Retriever, VectorStore
from local_rerank_rag.reranker import LlamaClient, Reranker


def _make_urlopen_side_effect(embed_dim=4, rerank_scores=None):
    """Return a callable that patches ``urllib.request.urlopen``.

    The callable inspects the URL to return the right response for
    ``/v1/embeddings`` vs ``/v1/chat/completions``.
    """
    if rerank_scores is None:
        rerank_scores = []
    rerank_iter = iter(rerank_scores)

    def side_effect(request, *args, **kwargs):
        url = request.get_full_url()
        response_body = b""

        if "/v1/embeddings" in url:
            response_body = json.dumps(
                {"data": [{"embedding": [0.1] * embed_dim}]}
            ).encode()
        elif "/v1/chat/completions" in url:
            try:
                score = next(rerank_iter)
            except StopIteration:
                score = 5
            response_body = json.dumps(
                {
                    "choices": [
                        {"message": {"content": json.dumps({"score": score})}}
                    ]
                }
            ).encode()

        # Build a context manager that returns a response-like object
        resp = MagicMock()
        resp.read.return_value = response_body
        cm = MagicMock()
        cm.__enter__.return_value = resp
        return cm

    return side_effect


class TestRunPipeline:
    def test_full_pipeline(self):
        corpus = mini_corpus()
        store = VectorStore()
        rng = np.random.RandomState(42)
        for doc in corpus.documents:
            store.add(doc, rng.randn(4).astype(np.float32))

        client = EmbeddingClient("http://mock:8080")
        retriever = Retriever(store=store, client=client)

        rerank_scores = [8, 3, 5, 2, 0]
        side_effect = _make_urlopen_side_effect(
            embed_dim=4, rerank_scores=rerank_scores
        )

        with patch("urllib.request.urlopen", side_effect=side_effect):
            llm_client = LlamaClient("http://mock:8080")
            reranker = Reranker(client=llm_client)

            result = run_pipeline(retriever, reranker, corpus.queries[0],
                                  retrieve_k=5, final_k=3)
            assert len(result.retrieved) == 5
            assert len(result.reranked) == 5
            assert "relevance_score" in result.reranked[0]

    def test_keyword_baseline(self):
        corpus = mini_corpus()
        store = VectorStore()
        rng = np.random.RandomState(42)
        for doc in corpus.documents:
            store.add(doc, rng.randn(4).astype(np.float32))

        client = EmbeddingClient("http://mock:8080")
        retriever = Retriever(store=store, client=client)

        with patch("urllib.request.urlopen",
                   side_effect=_make_urlopen_side_effect()):
            result = run_pipeline_keyword_baseline(
                retriever, corpus.queries[0], retrieve_k=5, final_k=3
            )
            assert len(result.reranked) == 5

    def test_pipeline_returns_rag_result_type(self):
        corpus = mini_corpus()
        store = VectorStore()
        rng = np.random.RandomState(0)
        for doc in corpus.documents:
            store.add(doc, rng.randn(4).astype(np.float32))

        client = EmbeddingClient("http://mock:8080")
        retriever = Retriever(store=store, client=client)

        with patch("urllib.request.urlopen",
                   side_effect=_make_urlopen_side_effect(
                       rerank_scores=[5] * 5)):
            llm_client = LlamaClient("http://mock:8080")
            reranker = Reranker(client=llm_client)

            result = run_pipeline(retriever, reranker, corpus.queries[0],
                                  retrieve_k=5, final_k=3)
            assert result.query == corpus.queries[0]
            assert result.num_candidates == 5
            assert result.final_top_k == 3


class TestEvaluatePipeline:
    def test_evaluate(self):
        corpus = mini_corpus()
        store = VectorStore()
        rng = np.random.RandomState(42)
        for doc in corpus.documents:
            store.add(doc, rng.randn(4).astype(np.float32))

        client = EmbeddingClient("http://mock:8080")
        retriever = Retriever(store=store, client=client)

        rerank_scores = [8, 3, 5, 2, 0, 0, 2, 8, 3, 5]

        with patch("urllib.request.urlopen",
                   side_effect=_make_urlopen_side_effect(
                       rerank_scores=rerank_scores)):
            llm_client = LlamaClient("http://mock:8080")
            reranker = Reranker(client=llm_client)

            ev = evaluate_pipeline(retriever, reranker, corpus,
                                   retrieve_k=5, final_k=3)
            assert "reranked_mrr" in ev
            assert "retrieval_recall" in ev
            assert ev["num_queries"] == 2

    def test_with_baseline(self):
        corpus = mini_corpus()
        store = VectorStore()
        rng = np.random.RandomState(42)
        for doc in corpus.documents:
            store.add(doc, rng.randn(4).astype(np.float32))

        client = EmbeddingClient("http://mock:8080")
        retriever = Retriever(store=store, client=client)

        rerank_scores = [8, 3, 5, 2, 0, 0, 2, 8, 3, 5]

        with patch("urllib.request.urlopen",
                   side_effect=_make_urlopen_side_effect(
                       rerank_scores=rerank_scores)):
            llm_client = LlamaClient("http://mock:8080")
            reranker = Reranker(client=llm_client)

            ev = evaluate_pipeline(
                retriever, reranker, corpus, retrieve_k=5, final_k=3,
                include_baseline=True,
            )
            assert "baseline_mrr" in ev
