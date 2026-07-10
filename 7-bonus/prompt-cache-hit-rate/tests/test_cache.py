"""Tests for prompt cache hit-rate analyzer."""

import pytest

from prompt_cache.analyzer import (
    CacheEvent,
    PromptCache,
    simulate_workload,
)


class TestPromptCache:
    def test_lookup_miss_on_empty(self):
        cache = PromptCache(max_tokens=4096)
        hit, tokens = cache.lookup("hello world")
        assert hit is False
        assert tokens == 0

    def test_store_and_hit(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("system prompt A", 100)
        hit, tokens = cache.lookup("system prompt A")
        assert hit is True
        assert tokens == 100

    def test_eviction_lru(self):
        cache = PromptCache(max_tokens=200)
        cache.store("prompt A" * 10, 150)
        cache.store("prompt B" * 10, 100)  # should evict A
        hit_a, _ = cache.lookup("prompt A" * 10)
        assert hit_a is False  # evicted
        hit_b, _ = cache.lookup("prompt B" * 10)
        assert hit_b is True

    def test_stats_recorded(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("prompt1", 50)
        cache.lookup("prompt1")
        cache.lookup("nonexistent")
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert cache.stats.total_lookups == 2

    def test_hit_rate(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("p1", 10)
        cache.lookup("p1")  # hit
        cache.lookup("not_there")  # miss
        assert cache.stats.hit_rate == 0.5

    def test_clear_resets(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("p1", 50)
        cache.lookup("p1")
        assert cache.stats.hits == 1
        cache.clear()
        assert cache.entry_count == 0
        assert cache.stats.hits == 0

    def test_size_tokens(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("p1", 100)
        cache.store("p2", 200)
        assert cache.size_tokens == 300

    def test_recommendations_low_hit_rate(self):
        cache = PromptCache(max_tokens=4096)
        cache.lookup("nothing")
        recs = cache.stats.recommendations()
        assert any("Low hit rate" in r for r in recs)

    def test_recommendations_good_hit_rate(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("p1", 50)
        cache.lookup("p1")
        cache.lookup("p1")
        cache.lookup("p1")
        recs = cache.stats.recommendations()
        assert any("Good" in r for r in recs)

    def test_summary_contains_fields(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("p1", 50)
        cache.lookup("p1")
        cache.lookup("p1")
        s = cache.stats.summary()
        assert "Hit Rate" in s
        assert "Lookups" in s

    def test_avg_latencies(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("p1", 50)
        for _ in range(10):
            cache.lookup("p1")
        for _ in range(10):
            cache.lookup("not_there")
        assert cache.stats.avg_hit_latency_ms > 0
        assert cache.stats.avg_miss_latency_ms > 0
        assert cache.stats.avg_hit_latency_ms < cache.stats.avg_miss_latency_ms


class TestSimulation:
    def test_simulate_returns_stats(self):
        cache = PromptCache(max_tokens=8192)
        stats = simulate_workload(cache, n_requests=100, reuse_probability=0.5)
        assert stats.total_lookups > 0
        assert stats.hits + stats.misses == stats.total_lookups

    def test_simulate_increases_hit_rate_with_reuse(self):
        cache = PromptCache(max_tokens=16384)
        stats = simulate_workload(cache, n_requests=200, reuse_probability=0.8)
        assert stats.hit_rate > 0.3

    def test_low_reuse_gives_lower_hit_rate(self):
        cache = PromptCache(max_tokens=16384)
        stats = simulate_workload(cache, n_requests=200, reuse_probability=0.05)
        # With 3000 unique prompts and 200 requests, very few coincidental repeats
        assert stats.hit_rate < 0.3

    def test_high_reuse_gives_high_hit_rate(self):
        cache = PromptCache(max_tokens=16384)
        stats = simulate_workload(cache, n_requests=200, reuse_probability=0.9)
        assert stats.hit_rate > 0.5

    def test_latency_savings(self):
        cache = PromptCache(max_tokens=16384)
        stats = simulate_workload(cache, n_requests=100, reuse_probability=0.7)
        assert stats.latency_savings_ms >= 0

    def test_event_records(self):
        cache = PromptCache(max_tokens=4096)
        cache.store("test", 50)
        cache.lookup("test")
        cache.lookup("missing")
        assert len(cache.stats.events) == 3
        event_types = [e.event for e in cache.stats.events]
        assert CacheEvent.STORE in event_types
        assert CacheEvent.HIT in event_types
        assert CacheEvent.MISS in event_types
