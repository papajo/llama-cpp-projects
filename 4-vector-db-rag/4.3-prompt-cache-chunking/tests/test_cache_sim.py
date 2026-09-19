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
        # `... or True` here used to make this assertion unfalsifiable, which
        # hid the fact that cached_tokens was structurally always 0.
        assert result.total_tokens >= result.cached_tokens

    def test_cache_with_no_queries(self):
        result = simulate_cache(
            FixedSizeChunker(200),
            ["Some document."],
            [],
            top_k=3,
        )
        assert result.total_prompts == 0
        assert result.total_tokens == 0


class TestChunkLevelCacheHits:
    """Regression tests: a repeated chunk run must register as a cache hit.

    `chunk_prefix` used to be keyed on the query-bearing prefix and was never
    added to `seen_prefixes`, so the chunk-level branch was dead code and
    `cached_tokens` was always 0 for any corpus with unique queries -- i.e.
    always, in practice. The headline `cache_hit_ratio` metric was a constant
    0.0 and no offline assertion could fail.
    """

    def test_repeated_chunk_run_is_cached(self):
        from prompt_cache_chunking.chunkers import ParagraphChunker
        from prompt_cache_chunking.cache_sim import simulate_cache

        # Queries 0-2 all retrieve docs [0, 1, 2], so the chunk run repeats.
        result = simulate_cache(
            chunker=ParagraphChunker(),
            documents=["Alpha document.", "Beta document.", "Gamma document."],
            queries=["first query", "second query", "third query"],
        )
        assert result.cached_tokens > 0, "repeated chunk run produced no hits"
        assert 0.0 < result.cache_hit_ratio <= 1.0

    def test_cached_never_exceeds_total(self):
        from prompt_cache_chunking.chunkers import FixedSizeChunker
        from prompt_cache_chunking.cache_sim import simulate_cache
        from prompt_cache_chunking.data import default_corpus

        corpus = default_corpus()
        result = simulate_cache(
            chunker=FixedSizeChunker(chunk_size=100, overlap=10),
            documents=corpus.documents,
            queries=corpus.queries,
        )
        assert result.cached_tokens <= result.total_tokens
        assert 0.0 <= result.cache_hit_ratio <= 1.0

    def test_single_query_has_no_hits(self):
        """Nothing to reuse on a cold cache with one prompt."""
        from prompt_cache_chunking.chunkers import ParagraphChunker
        from prompt_cache_chunking.cache_sim import simulate_cache

        result = simulate_cache(
            chunker=ParagraphChunker(),
            documents=["Alpha document."],
            queries=["only query"],
        )
        assert result.cached_tokens == 0
        assert result.cache_hit_ratio == 0.0
