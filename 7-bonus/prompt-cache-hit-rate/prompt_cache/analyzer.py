"""Prompt cache hit-rate analyzer for llama.cpp.

Simulates or monitors the key-value cache to measure efficiency:
hit rate, miss penalty, eviction patterns, and actionable
recommendations for improving cache utilization.
"""

from __future__ import annotations

import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class CacheEvent(Enum):
    HIT = "hit"
    MISS = "miss"
    EVICT = "evict"
    STORE = "store"


@dataclass
class CacheEntry:
    """A single cached prompt prefix."""
    prompt_hash: int
    prompt_preview: str  # first 40 chars
    tokens: int
    stored_at: float
    last_access: float
    access_count: int = 0


@dataclass
class CacheEventRecord:
    """A recorded cache event."""
    timestamp: float
    event: CacheEvent
    prompt_hash: int
    prompt_preview: str
    tokens: int
    cache_size_before: int
    cache_size_after: int


@dataclass
class CacheStats:
    """Aggregate cache statistics."""
    total_lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    total_tokens_cached: int = 0
    total_tokens_served_from_cache: int = 0
    sum_miss_latency_ms: float = 0.0
    sum_hit_latency_ms: float = 0.0
    events: List[CacheEventRecord] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.hits / self.total_lookups

    @property
    def avg_miss_latency_ms(self) -> float:
        if self.misses == 0:
            return 0.0
        return self.sum_miss_latency_ms / self.misses

    @property
    def avg_hit_latency_ms(self) -> float:
        if self.hits == 0:
            return 0.0
        return self.sum_hit_latency_ms / self.hits

    @property
    def latency_savings_ms(self) -> float:
        """Total latency saved by cache hits vs recompute."""
        return self.hits * (self.avg_miss_latency_ms - self.avg_hit_latency_ms)

    def record(self, event: CacheEvent, prompt_hash: int, prompt_preview: str,
               tokens: int, before: int, after: int) -> None:
        rec = CacheEventRecord(
            timestamp=time.time(),
            event=event,
            prompt_hash=prompt_hash,
            prompt_preview=prompt_preview[:40],
            tokens=tokens,
            cache_size_before=before,
            cache_size_after=after,
        )
        self.events.append(rec)
        if event == CacheEvent.HIT:
            self.hits += 1
            self.total_lookups += 1
            self.total_tokens_served_from_cache += tokens
            self.sum_hit_latency_ms += random.uniform(0.5, 2.0)  # fast
        elif event == CacheEvent.MISS:
            self.misses += 1
            self.total_lookups += 1
            self.sum_miss_latency_ms += random.uniform(15.0, 60.0)  # slow
        elif event == CacheEvent.STORE:
            self.stores += 1
            self.total_tokens_cached += tokens
        elif event == CacheEvent.EVICT:
            self.evictions += 1

    def summary(self) -> str:
        return (
            f"Prompt Cache Report\n"
            f"  Lookups:     {self.total_lookups}\n"
            f"  Hits:        {self.hits}\n"
            f"  Misses:      {self.misses}\n"
            f"  Hit Rate:    {self.hit_rate:.1%}\n"
            f"  Stores:      {self.stores}\n"
            f"  Evictions:   {self.evictions}\n"
            f"  Cache Size:  {self.total_tokens_cached} tokens\n"
            f"  Avg Hit Lat:   {self.avg_hit_latency_ms:.1f} ms\n"
            f"  Avg Miss Lat:  {self.avg_miss_latency_ms:.1f} ms\n"
            f"  Latency Saved: {self.latency_savings_ms:.0f} ms\n"
        )

    def recommendations(self) -> List[str]:
        recs: List[str] = []
        if self.hit_rate < 0.3:
            recs.append("Low hit rate. Consider increasing cache size or reusing prompt prefixes.")
        if self.hit_rate < 0.5:
            recs.append("Medium hit rate. Ensure common system prompts are identical across requests.")
        if self.hit_rate >= 0.7:
            recs.append("Good hit rate. Cache is working well.")
        if self.evictions > self.stores * 0.5:
            recs.append("High eviction rate. Cache may be too small for workload.")
        if self.total_tokens_served_from_cache > 0:
            recs.append(
                f"Served {self.total_tokens_served_from_cache} tokens from cache "
                f"(saved ~{self.latency_savings_ms:.0f}ms aggregate latency)."
            )
        if not recs:
            recs.append("No data yet. Run some inference requests to build cache stats.")
        return recs


# ---------------------------------------------------------------------------
# Simulated cache
# ---------------------------------------------------------------------------


class PromptCache:
    """Simulates llama.cpp's prompt cache for hit-rate analysis.

    Uses an LRU eviction policy.
    """

    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self._cache: OrderedDict[int, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self._current_tokens = 0

    def lookup(self, prompt: str) -> Tuple[bool, int]:
        """Look up a prompt in the cache. Returns (hit, cached_tokens)."""
        ph = hash(prompt)
        entry = self._cache.get(ph)

        if entry is not None:
            # Hit — move to end (LRU)
            entry.last_access = time.time()
            entry.access_count += 1
            self._cache.move_to_end(ph)
            self.stats.record(
                CacheEvent.HIT, ph, prompt, entry.tokens,
                self._current_tokens, self._current_tokens,
            )
            return True, entry.tokens
        else:
            # Miss
            self.stats.record(
                CacheEvent.MISS, ph, prompt, 0,
                self._current_tokens, self._current_tokens,
            )
            return False, 0

    def store(self, prompt: str, tokens: int) -> None:
        """Store a prompt in the cache."""
        ph = hash(prompt)
        # Evict until space available
        while self._current_tokens + tokens > self.max_tokens and self._cache:
            evicted_ph, evicted_entry = self._cache.popitem(last=False)
            self._current_tokens -= evicted_entry.tokens
            self.stats.record(
                CacheEvent.EVICT, evicted_ph, evicted_entry.prompt_preview,
                evicted_entry.tokens,
                self._current_tokens + evicted_entry.tokens,
                self._current_tokens,
            )

        if self._current_tokens + tokens > self.max_tokens:
            # Can't fit at all
            return

        entry = CacheEntry(
            prompt_hash=ph,
            prompt_preview=prompt[:40],
            tokens=tokens,
            stored_at=time.time(),
            last_access=time.time(),
        )
        self._cache[ph] = entry
        self._current_tokens += tokens
        self.stats.record(
            CacheEvent.STORE, ph, prompt, tokens,
            self._current_tokens - tokens,
            self._current_tokens,
        )

    def clear(self) -> None:
        self._cache.clear()
        self._current_tokens = 0
        self.stats = CacheStats()

    @property
    def size_tokens(self) -> int:
        return self._current_tokens

    @property
    def entry_count(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Workload simulation
# ---------------------------------------------------------------------------

# Large pools so random.choice creates many unique prompts.
_SYSTEM_PROMPTS = [
    f"You are a helpful assistant for domain {i}. Respond in {lang}."
    for i in range(20)
    for lang in ["English", "French", "German"]
]

_USER_PROMPTS = [
    f"Tell me about topic {i} in detail with examples and code."
    for i in range(50)
]


def simulate_workload(
    cache: PromptCache,
    n_requests: int = 200,
    reuse_probability: float = 0.4,
) -> CacheStats:
    """Simulate inference requests with configurable full-prompt reuse.

    When ``reuse_probability`` fires, the exact same full prompt (system +
    user) is reused from an earlier request, producing a cache hit.
    Otherwise a new random prompt is constructed.

    Args:
        cache: The PromptCache instance.
        n_requests: Number of inference requests to simulate.
        reuse_probability: Probability that a request reuses a previous prompt.

    Returns:
        The cache stats after the simulation.
    """
    seen_prompts: List[str] = []

    for _ in range(n_requests):
        # Decide whether to reuse an already-seen prompt
        if seen_prompts and random.random() < reuse_probability:
            full_prompt = random.choice(seen_prompts)
        else:
            system = random.choice(_SYSTEM_PROMPTS)
            user = random.choice(_USER_PROMPTS)
            full_prompt = system + "\n" + user

        tokens = len(full_prompt) // 2 + random.randint(10, 50)

        hit, _ = cache.lookup(full_prompt)
        if not hit:
            cache.store(full_prompt, tokens)
            seen_prompts.append(full_prompt)

    return cache.stats
