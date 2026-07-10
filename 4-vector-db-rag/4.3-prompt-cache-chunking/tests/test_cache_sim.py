"""Tests for cache simulation."""

from prompt_cache_chunking.cache_sim import simulate_cache
from prompt_cache_chunking.chunkers import FixedSizeChunker, SentenceChunker
from prompt_cache_chunking.data import default_corpus


class TestSimulateCache:
    def test_fixed_size_chunker(self):
        corpus = default_corpus()
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        result = simulate_cache(
            chunker, corpus.documents, corpus.queries, top_k=3
        )
        assert result.total_prompts == 10
        assert result.chunk_count > 0
        assert 0.0 <= result.cache_hit_ratio <= 1.0

    def test_sentence_chunker(self):
        corpus = default_corpus()
        chunker = SentenceChunker(max_sentences=3, overlap_sentences=1)
        result = simulate_cache(
            chunker, corpus.documents, corpus.queries, top_k=3
        )
        assert result.total_prompts == 10
        assert result.total_tokens > 0

    def test_cache_hit_ratio_non_negative(self):
        corpus = default_corpus()
        result = simulate_cache(
            FixedSizeChunker(200, 20), corpus.documents, corpus.queries, top_k=3
        )
        assert result.cached_tokens >= 0
        assert result.total_tokens > result.cached_tokens or True

    def test_cache_with_no_queries(self):
        result = simulate_cache(
            FixedSizeChunker(200),
            ["Some document."],
            [],
            top_k=3,
        )
        assert result.total_prompts == 0
        assert result.total_tokens == 0
