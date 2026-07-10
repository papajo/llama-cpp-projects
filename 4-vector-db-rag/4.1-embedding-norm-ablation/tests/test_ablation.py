"""Tests for the ablation experiment runner."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding_norm_ablation.ablation import (
    AblationConfig,
    AblationError,
    TREATMENT_NORM_COSINE,
    TREATMENT_UNNORM_COSINE,
    run_ablation,
)
from embedding_norm_ablation.data import mini_corpus
from embedding_norm_ablation.embeddings import EmbeddingClient


def _mock_embedding_response(dim: int = 4):
    """Return a canned embedding response for a single text."""
    return {"data": [{"embedding": [0.1] * dim}]}


class TestRunAblation:
    @patch("urllib.request.urlopen")
    def test_basic_run(self, mock_urlopen):
        """With all mocked HTTP, run_ablation should return structured results."""
        # Return a 4-d embedding for every call
        dim = 4
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(
            _mock_embedding_response(dim)
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        corpus = mini_corpus()
        config = AblationConfig(corpus=corpus, top_k=3)
        client = EmbeddingClient(server_url="http://mock:8080")

        result = run_ablation(client, config)

        assert result.corpus_size == 5
        assert result.num_queries == 2
        assert result.embedding_dim == dim
        assert len(result.treatments) == 4  # standard treatments
        assert result.get_treatment("norm-cosine") is not None
        assert result.get_treatment("unnorm-cosine") is not None

    @patch("urllib.request.urlopen")
    def test_single_treatment(self, mock_urlopen):
        dim = 4
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(
            _mock_embedding_response(dim)
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        corpus = mini_corpus()
        config = AblationConfig(corpus=corpus, top_k=3)
        client = EmbeddingClient(server_url="http://mock:8080")

        result = run_ablation(client, config, treatments=[TREATMENT_NORM_COSINE])
        assert len(result.treatments) == 1
        assert result.treatments[0].label == "norm-cosine"

    @patch("urllib.request.urlopen")
    def test_summary_shape(self, mock_urlopen):
        dim = 4
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(
            _mock_embedding_response(dim)
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        corpus = mini_corpus()
        config = AblationConfig(corpus=corpus, top_k=3)
        client = EmbeddingClient(server_url="http://mock:8080")

        result = run_ablation(client, config)
        summary = result.summary()
        assert set(summary.keys()) == {
            "unnorm-cosine", "norm-cosine",
            "unnorm-dot", "norm-dot",
        }
        # Every treatment should have the same metric keys
        metrics = list(summary.values())[0].keys()
        assert "mean_p@1" in metrics
        assert "mean_p@5" in metrics

    @patch("urllib.request.urlopen")
    def test_norm_stats_present(self, mock_urlopen):
        dim = 4
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(
            _mock_embedding_response(dim)
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        corpus = mini_corpus()
        config = AblationConfig(corpus=corpus, top_k=3)
        client = EmbeddingClient(server_url="http://mock:8080")

        result = run_ablation(client, config)
        for t in result.treatments:
            assert "mean" in t.doc_norm_stats
            assert "mean" in t.query_norm_stats

    @patch("urllib.request.urlopen")
    def test_server_error(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("fail")

        corpus = mini_corpus()
        config = AblationConfig(corpus=corpus, top_k=3)
        client = EmbeddingClient(server_url="http://mock:8080")

        with pytest.raises(AblationError):
            run_ablation(client, config)
