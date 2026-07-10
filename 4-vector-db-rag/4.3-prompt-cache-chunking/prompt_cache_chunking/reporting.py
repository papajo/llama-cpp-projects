"""Reporting for chunking experiments."""

from __future__ import annotations

import json
from typing import Dict

from prompt_cache_chunking.experiment import ExperimentResult


def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def experiment_to_markdown(result: ExperimentResult) -> str:
    """Full markdown experiment report."""
    lines = []
    lines.append("# Prompt Cache Chunking Experiment Report")
    lines.append("")
    lines.append(f"- **Corpus:** {result.corpus_name}")
    lines.append(f"- **Documents:** {result.num_documents}")
    lines.append(f"- **Queries:** {result.num_queries}")
    lines.append("")

    if not result.results:
        lines.append("_(no results)_")
        return "\n".join(lines)

    # Summary comparison table
    lines.append("## Strategy Comparison")
    lines.append("")
    lines.append(
        "| Strategy | Chunks | Unique Docs | Tokens | Cache Hit Ratio | "
        "Prompts/Chunk |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in result.results:
        lines.append(
            f"| {r.strategy_name} | {r.chunk_count} | {r.unique_chunks} | "
            f"{r.total_tokens} | {_fmt(r.cache_hit_ratio)} | "
            f"{_fmt(r.prompts_per_chunk_avg)} |"
        )
    lines.append("")

    # Detail per strategy
    lines.append("## Per-Strategy Detail")
    lines.append("")
    for r in result.results:
        lines.append(f"### {r.strategy_name}")
        lines.append("")
        lines.append(f"- **Total tokens:** {r.total_tokens}")
        lines.append(f"- **Cached tokens:** {r.cached_tokens}")
        lines.append(f"- **Cache hit ratio:** {_fmt(r.cache_hit_ratio)}")
        lines.append(f"- **Chunk count:** {r.chunk_count}")
        lines.append(f"- **Prompts per chunk (avg):** {_fmt(r.prompts_per_chunk_avg)}")
        lines.append("")
        lines.append("| Query | Chunks | Prompt Tokens | Cached Tokens |")
        lines.append("|---|---|---|---|")
        for d in r.details:
            lines.append(
                f"| {d['query_idx']} | {d['num_chunks']} | "
                f"{d['prompt_tokens']} | {d['cached_tokens']} |"
            )
        lines.append("")

    return "\n".join(lines)


def experiment_to_json(result: ExperimentResult) -> str:
    """Serialize experiment result to JSON."""
    data = {
        "corpus_name": result.corpus_name,
        "num_documents": result.num_documents,
        "num_queries": result.num_queries,
        "results": [
            {
                "strategy_name": r.strategy_name,
                "chunk_count": r.chunk_count,
                "unique_chunks": r.unique_chunks,
                "total_tokens": r.total_tokens,
                "cached_tokens": r.cached_tokens,
                "cache_hit_ratio": r.cache_hit_ratio,
                "prompts_per_chunk_avg": r.prompts_per_chunk_avg,
                "details": r.details,
            }
            for r in result.results
        ],
    }
    return json.dumps(data, indent=2)
