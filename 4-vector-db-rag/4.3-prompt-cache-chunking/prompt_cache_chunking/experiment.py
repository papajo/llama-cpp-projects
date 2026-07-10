"""Run comparative experiments across chunking strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prompt_cache_chunking.cache_sim import (
    CacheSimulationResult,
    simulate_cache,
)
from prompt_cache_chunking.chunkers import (
    Chunker,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
)
from prompt_cache_chunking.data import ChunkingCorpus

# Standard set of chunkers to compare
DEFAULT_CHUNKERS: List[Chunker] = [
    FixedSizeChunker(chunk_size=200, overlap=20),
    FixedSizeChunker(chunk_size=100, overlap=10),
    SentenceChunker(max_sentences=3, overlap_sentences=1),
    SentenceChunker(max_sentences=2, overlap_sentences=0),
    ParagraphChunker(),
    RecursiveChunker(max_chars=150),
]


@dataclass
class ExperimentResult:
    """Results from comparing multiple chunking strategies."""

    corpus_name: str
    num_documents: int
    num_queries: int
    results: List[CacheSimulationResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Dict]:
        """Nested dict: strategy_name → {metric → value}."""
        return {
            r.strategy_name: {
                "chunk_count": r.chunk_count,
                "total_tokens": r.total_tokens,
                "cache_hit_ratio": r.cache_hit_ratio,
                "unique_chunks": r.unique_chunks,
                "prompts_per_chunk_avg": r.prompts_per_chunk_avg,
            }
            for r in self.results
        }


def run_experiment(
    corpus: ChunkingCorpus,
    chunkers: Optional[List[Chunker]] = None,
    top_k: int = 3,
    system_prompt: str = "You are a helpful assistant.",
) -> ExperimentResult:
    """Run the cache simulation for each chunker on the given corpus."""
    if chunkers is None:
        chunkers = DEFAULT_CHUNKERS

    result = ExperimentResult(
        corpus_name="default",
        num_documents=len(corpus.documents),
        num_queries=len(corpus.queries),
    )

    for chunker in chunkers:
        sim = simulate_cache(
            chunker=chunker,
            documents=corpus.documents,
            queries=corpus.queries,
            system_prompt=system_prompt,
            top_k=top_k,
        )
        result.results.append(sim)

    return result
