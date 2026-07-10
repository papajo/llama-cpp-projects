"""Tests for experiment runner."""

from prompt_cache_chunking.chunkers import FixedSizeChunker, SentenceChunker
from prompt_cache_chunking.data import default_corpus
from prompt_cache_chunking.experiment import DEFAULT_CHUNKERS, run_experiment


class TestRunExperiment:
    def test_default_chunkers(self):
        corpus = default_corpus()
        result = run_experiment(corpus)
        assert result.num_documents == 10
        assert result.num_queries == 10
        assert len(result.results) == len(DEFAULT_CHUNKERS)

    def test_custom_chunkers(self):
        corpus = default_corpus()
        chunkers = [
            FixedSizeChunker(chunk_size=100, overlap=0),
            SentenceChunker(max_sentences=2, overlap_sentences=0),
        ]
        result = run_experiment(corpus, chunkers=chunkers)
        assert len(result.results) == 2

    def test_summary_shape(self):
        corpus = default_corpus()
        result = run_experiment(corpus, chunkers=[FixedSizeChunker(200)])
        summary = result.summary()
        assert "FixedSizeChunker" in summary
        metrics = summary["FixedSizeChunker"]
        assert "cache_hit_ratio" in metrics
        assert "chunk_count" in metrics

    def test_custom_top_k(self):
        corpus = default_corpus()
        result = run_experiment(corpus, top_k=5)
        for r in result.results:
            assert r.total_prompts == 10
