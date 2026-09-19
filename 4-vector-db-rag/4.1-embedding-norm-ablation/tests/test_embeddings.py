"""Tests for embedding generation and normalisation."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding_norm_ablation.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    batch_normalize,
    l2_normalize,
    norm_stats,
)


class TestL2Normalize:
    def test_normalize_unit_length(self):
        v = np.array([3.0, 4.0])
        n = l2_normalize(v)
        assert np.allclose(np.linalg.norm(n), 1.0)
        assert np.allclose(n, [0.6, 0.8])

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        n = l2_normalize(v)
        assert np.allclose(n, [0.0, 0.0, 0.0])

    def test_does_not_mutate_original(self):
        v = np.array([1.0, 2.0, 3.0])
        orig = v.copy()
        n = l2_normalize(v)
        assert np.array_equal(v, orig)
        assert n is not v

    def test_negative_values(self):
        v = np.array([-3.0, 4.0])
        n = l2_normalize(v)
        assert np.allclose(np.linalg.norm(n), 1.0)

    def test_single_element(self):
        v = np.array([5.0])
        n = l2_normalize(v)
        assert np.allclose(n, [1.0])


class TestBatchNormalize:
    def test_all_normalized(self):
        vecs = [np.array([3.0, 4.0]), np.array([0.0, 5.0])]
        n = batch_normalize(vecs)
        assert len(n) == 2
        assert np.allclose(np.linalg.norm(n[0]), 1.0)
        assert np.allclose(np.linalg.norm(n[1]), 1.0)

    def test_empty_list(self):
        assert batch_normalize([]) == []


class TestNormStats:
    def test_stats(self):
        vecs = [np.array([3.0, 4.0]), np.array([1.0, 0.0])]
        stats = norm_stats(vecs)
        assert stats["mean"] == pytest.approx((5.0 + 1.0) / 2)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_empty_returns_nan(self):
        stats = norm_stats([])
        assert np.isnan(stats["mean"])


class TestEmbeddingClient:
    def test_embed_success(self):
        # Envelope mirrors a real llama-server /v1/embeddings response: each
        # row carries index/object, and usage has no completion_tokens.
        mock_response = {
            "model": "nomic-embed-text-v1.5",
            "object": "list",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
            "data": [{"index": 0, "object": "embedding",
                      "embedding": [0.1, 0.2, 0.3]}],
        }
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            client = EmbeddingClient(server_url="http://mock:8080")
            vec = client.embed("hello")
            assert isinstance(vec, np.ndarray)
            assert vec.shape == (3,)
            assert np.allclose(vec, [0.1, 0.2, 0.3])

    def test_embed_with_model(self):
        mock_response = {
            "data": [{"embedding": [1.0, 0.0]}]
        }
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            client = EmbeddingClient(server_url="http://mock:8080")
            vec = client.embed("text", model="test-model")
            assert vec.shape == (2,)

    def test_embed_connection_error(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            from urllib.error import URLError
            mock_urlopen.side_effect = URLError("connection refused")

            client = EmbeddingClient(server_url="http://mock:8080")
            with pytest.raises(EmbeddingError):
                client.embed("hello")

    def test_embed_bad_json(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = b"not json"
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            client = EmbeddingClient(server_url="http://mock:8080")
            with pytest.raises(EmbeddingError):
                client.embed("hello")

    def test_embed_missing_embedding_key(self):
        mock_response = {"data": [{"not_embedding": [1, 2, 3]}]}
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            client = EmbeddingClient(server_url="http://mock:8080")
            with pytest.raises(EmbeddingError):
                client.embed("hello")

    def test_embed_many(self):
        mock_response = {
            "data": [{"embedding": [0.1, 0.2]}]
        }
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            client = EmbeddingClient(server_url="http://mock:8080")
            vecs = client.embed_many(["a", "b", "c"])
            assert len(vecs) == 3
            assert all(v.shape == (2,) for v in vecs)

    def test_timeout_reached(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            import socket
            mock_urlopen.side_effect = OSError("timeout")

            client = EmbeddingClient(server_url="http://mock:8080")
            with pytest.raises(EmbeddingError):
                client.embed("hello")
