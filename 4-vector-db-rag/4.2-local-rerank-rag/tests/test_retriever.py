"""Tests for the retriever module."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from local_rerank_rag.retriever import (
    EmbeddingClient,
    EmbeddingError,
    Retriever,
    VectorStore,
    cosine_similarity,
    l2_normalize,
)


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestL2Normalize:
    def test_unit_length(self):
        v = np.array([3.0, 4.0])
        n = l2_normalize(v)
        assert np.allclose(np.linalg.norm(n), 1.0)

    def test_zero(self):
        v = np.array([0.0, 0.0])
        n = l2_normalize(v)
        assert np.allclose(n, [0.0, 0.0])


class TestVectorStore:
    def test_add_and_search(self):
        store = VectorStore()
        store.add("doc a", np.array([1.0, 0.0]))
        store.add("doc b", np.array([0.0, 1.0]))
        store.add("doc c", np.array([0.9, 0.1]))
        assert len(store) == 3
        results = store.search(np.array([1.0, 0.0]), top_k=2)
        assert results[0] == 0
        assert results[1] == 2

    def test_add_many(self):
        store = VectorStore()
        store.add_many(["a", "b"], [np.array([1.0, 0.0]), np.array([0.0, 1.0])])
        assert len(store) == 2

    def test_empty_store(self):
        store = VectorStore()
        assert store.search(np.array([1.0, 0.0]), top_k=5) == []


class TestEmbeddingClient:
    def test_success(self):
        mock_resp = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        with patch("urllib.request.urlopen") as m:
            mc = MagicMock()
            mc.read.return_value = json.dumps(mock_resp).encode()
            m.return_value.__enter__.return_value = mc
            client = EmbeddingClient("http://mock:8080")
            vec = client.embed("hello")
            assert vec.shape == (3,)

    def test_error(self):
        with patch("urllib.request.urlopen") as m:
            from urllib.error import URLError
            m.side_effect = URLError("fail")
            client = EmbeddingClient("http://mock:8080")
            with pytest.raises(EmbeddingError):
                client.embed("hello")


class TestRetriever:
    @patch("urllib.request.urlopen")
    def test_retrieve(self, mock_urlopen):
        mock_resp = {"data": [{"embedding": [1.0, 0.0]}]}
        mc = MagicMock()
        mc.read.return_value = json.dumps(mock_resp).encode()
        mock_urlopen.return_value.__enter__.return_value = mc

        store = VectorStore()
        store.add("doc a", np.array([1.0, 0.0]))
        store.add("doc b", np.array([0.0, 1.0]))

        client = EmbeddingClient("http://mock:8080")
        retriever = Retriever(store=store, client=client)
        results = retriever.retrieve("test query", top_k=2)
        assert len(results) == 2
        assert "index" in results[0]
        assert "document" in results[0]
        assert "score" in results[0]
