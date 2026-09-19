"""Prompt cache simulation — measures KV-cache hit rates for chunked RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prompt_cache_chunking.chunkers import Chunk, Chunker


# ---------------------------------------------------------------------------
# Cache model
# ---------------------------------------------------------------------------


@dataclass
class CacheSimulationResult:
    """Result of simulating cache behaviour for a chunking strategy."""

    strategy_name: str
    total_prompts: int
    total_tokens: int
    cached_tokens: int
    unique_chunks: int
    chunk_count: int
    prompts_per_chunk_avg: float  # average number of prompts that use each chunk
    details: List[dict] = field(default_factory=list)

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of tokens served from cache."""
        if self.total_tokens == 0:
            return 0.0
        return self.cached_tokens / self.total_tokens


def simulate_cache(
    chunker: Chunker,
    documents: List[str],
    queries: List[str],
    system_prompt: str = "You are a helpful assistant. Answer based on the provided context.",
    top_k: int = 3,
) -> CacheSimulationResult:
    """Simulate prompt caching for a retrieve-then-read RAG loop.

    Model:
    - Each prompt = system prompt + query + chunks (each preceded by
      a ``<chunk>`` tag)
    - The cache stores the KV cache prefix up to the start of the chunks.
      When the same query is repeated, the system+query prefix is reused.
    - Across queries, chunks that appear in multiple prompts provide
      additional cache hits (their prefix up to the end of the repeated
      chunk is cached).

    Returns a ``CacheSimulationResult`` with hit-rate statistics.
    """
    chunks = chunker.chunk(documents)

    # Build a map: doc_index -> chunks (for simulating retrieval)
    doc_to_chunks: Dict[int, List[Chunk]] = {}
    for c in chunks:
        doc_to_chunks.setdefault(c.doc_index, []).append(c)

    total_tokens = 0
    cached_tokens = 0
    details: List[dict] = []

    # Every prompt prefix we have already paid to compute. Two kinds go in
    # here: the "system|query" prefix (so a repeated query is a hit) and the
    # cumulative chunk sequence up to and including each chunk (so a repeated
    # chunk run is a hit). A hit requires an exact match, mirroring the way a
    # real KV cache only reuses an identical token prefix.
    seen_prefixes: set = set()

    # System prompt tokens (estimated).
    #
    # chars // 4 is a heuristic, not a tokenizer. Measured against the real
    # server's /tokenize with the SmolLM2 vocab it over-estimates by roughly
    # 20-26% (66 chars -> 16 estimated vs 13 actual; 559 chars -> 139 vs 110).
    # Ratios between strategies stay comparable because every strategy uses the
    # same estimator, but absolute token counts here are not real token counts.
    sys_tokens = max(1, len(system_prompt) // 4)

    for q_idx, query in enumerate(queries):
        # Find chunks for relevant docs (simulate retrieval)
        relevant = [0, 1, 2] if q_idx < 3 else [q_idx - 1, q_idx, (q_idx + 1) % len(queries)]
        relevant = [r % len(documents) for r in relevant]

        selected_chunks: List[Chunk] = []
        for doc_idx in relevant:
            selected_chunks.extend(doc_to_chunks.get(doc_idx, []))

        # Build prompt structure and track cache hits
        prompt_tokens = 0
        prompt_cached = 0

        # System + query prefix (same across queries with same query)
        # In a real cache, the same query would reuse the system+query cache.
        # We treat repeated queries as a cache hit.
        q_tokens = max(1, len(query) // 4)
        prefix = f"{system_prompt}|{query}"

        # For each query, the prefix is fully computed the first time
        for seen in seen_prefixes:
            if seen == prefix:
                prompt_cached += sys_tokens + q_tokens
                break

        prompt_tokens += sys_tokens + q_tokens

        # Process each chunk.
        #
        # The cache key for the chunk region is the cumulative chunk sequence
        # seen so far, NOT the query-bearing prefix. A real KV cache matches on
        # the token prefix, so two prompts that open with the same chunk
        # sequence reuse it regardless of which query follows. Measured against
        # the real server: two prompts sharing a system prompt plus one chunk
        # but differing in the trailing query reported
        # usage.prompt_tokens_details.cached_tokens = 135 of 145 prompt tokens.
        chunk_prefix_parts: List[str] = []
        for chunk in selected_chunks:
            chunk_tag = f"<chunk>{chunk.text}</chunk>"
            chunk_tokens = max(1, len(chunk_tag) // 4)

            # Prefix up to and including this chunk.
            chunk_prefix_parts.append(chunk_tag)
            chunk_prefix = "|".join(chunk_prefix_parts)
            if chunk_prefix in seen_prefixes:
                prompt_cached += chunk_tokens
            seen_prefixes.add(chunk_prefix)

            prompt_tokens += chunk_tokens

        total_tokens += prompt_tokens
        cached_tokens += prompt_cached

        details.append({
            "query_idx": q_idx,
            "query": query,
            "num_chunks": len(selected_chunks),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": prompt_cached,
        })

        # Add the prefix for this query to seen set
        seen_prefixes.add(prefix)

    total_prompts = len(queries)
    chunk_count = len(chunks)
    unique_chunks = len(set(c.doc_index for c in chunks))
    prompts_per_chunk_avg = total_prompts / max(1, chunk_count)

    return CacheSimulationResult(
        strategy_name=type(chunker).__name__,
        total_prompts=total_prompts,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        unique_chunks=unique_chunks,
        chunk_count=chunk_count,
        prompts_per_chunk_avg=prompts_per_chunk_avg,
        details=details,
    )
