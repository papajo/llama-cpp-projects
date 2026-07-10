"""Report generation for RAG pipeline evaluations."""

from typing import Dict


def evaluation_to_markdown(eval_result: Dict) -> str:
    """Generate a markdown report from an evaluation result dict."""
    lines = []
    lines.append("# RAG Pipeline Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Queries:** {eval_result.get('num_queries', '?')}")
    lines.append(f"- **Initial retrieval k:** {eval_result.get('retrieve_k', '?')}")
    lines.append(f"- **Final top-k:** {eval_result.get('final_k', '?')}")
    if "retrieval_recall" in eval_result:
        lines.append(f"- **Retrieval recall@{eval_result['retrieve_k']}:** {eval_result['retrieval_recall']:.4f}")
    if "reranked_mrr" in eval_result:
        lines.append(f"- **Reranked MRR:** {eval_result['reranked_mrr']:.4f}")
    if "baseline_mrr" in eval_result:
        lines.append(f"- **Baseline MRR (keyword):** {eval_result['baseline_mrr']:.4f}")
    lines.append("")

    # Comparison table if both available
    if "reranked_mrr" in eval_result and "baseline_mrr" in eval_result:
        lines.append("## LLM Reranker vs Keyword Baseline")
        lines.append("")
        lines.append("| Method | MRR |")
        lines.append("|---|---|")
        lines.append(f"| LLM-as-judge reranker | {eval_result['reranked_mrr']:.4f} |")
        lines.append(f"| Keyword overlap baseline | {eval_result['baseline_mrr']:.4f} |")
        gap = eval_result["reranked_mrr"] - eval_result["baseline_mrr"]
        direction = "+" if gap >= 0 else ""
        lines.append(f"| **Delta** | **{direction}{gap:.4f}** |")
        lines.append("")

    if "per_query" in eval_result:
        lines.append("## Per-Query Detail")
        lines.append("")
        for pq in eval_result["per_query"]:
            lines.append(f"- **Query:** {pq['query']}")
            lines.append(f"  - Retrieved: {pq['retrieved_indices'][:5]}")
            lines.append(f"  - Reranked:  {pq['reranked_indices'][:5]}")
            lines.append("")

    return "\n".join(lines)


def evaluation_to_json(eval_result: Dict) -> str:
    """Serialize evaluation result to JSON."""
    import json
    return json.dumps(eval_result, indent=2, default=str)
