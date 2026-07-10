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

    # Track which (start_of_chunk_sequence) prefixes we've seen.
    # For simplicity: the prefix up to the start of each chunk.
    # We track unique strings that represent the prefix of the prompt
    # before each chunk.
    seen_prefixes: set = set()

    # System prompt tokens (estimated)
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

        # Process each chunk
        for idx, chunk in enumerate(selected_chunks):
            chunk_tag = f"<chunk>{chunk.text}</chunk>"
            chunk_tokens = max(1, len(chunk_tag) // 4)

            # Prefix up to this chunk
            chunk_prefix = f"{prefix}|{idx}"
            if chunk_prefix in seen_prefixes:
                prompt_cached += chunk_tokens

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
