"""Report generation for ablation experiments — markdown & JSON."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from embedding_norm_ablation.ablation import AblationResult, TreatmentResult


def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def _metric_table(treatments: List[TreatmentResult]) -> str:
    """Build a markdown table of aggregated metrics per treatment."""
    if not treatments:
        return "_(no data)_"

    metrics = list(treatments[0].aggregated.keys())
    header = "| Treatment | " + " | ".join(m for m in metrics) + " |"
    sep = "|" + "|".join("---" for _ in range(len(metrics) + 1)) + "|"
    rows = []
    for t in treatments:
        cells = [_fmt(t.aggregated[m]) for m in metrics]
        rows.append(f"| {t.label} | " + " | ".join(cells) + " |")
    return header + "\n" + sep + "\n" + "\n".join(rows)


def _norm_stats_table(
    treatments: List[TreatmentResult],
    side: str = "doc",
) -> str:
    """Build a norm-statistics table for the given side (doc or query)."""
    header = f"| Treatment | mean | std | min | max |"
    sep = "|---" * 5 + "|"
    rows = []
    for t in treatments:
        stats = t.doc_norm_stats if side == "doc" else t.query_norm_stats
        rows.append(
            f"| {t.label} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} |"
            f" {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
        )
    return header + "\n" + sep + "\n" + "\n".join(rows)


def report_to_markdown(result: AblationResult) -> str:
    """Generate a full markdown ablation report."""
    lines: List[str] = []
    lines.append("# Embedding Norm Ablation Report")
    lines.append("")
    lines.append(f"- **Corpus size:** {result.corpus_size} documents")
    lines.append(f"- **Queries:** {result.num_queries}")
    lines.append(f"- **Embedding dimension:** {result.embedding_dim}")
    lines.append(f"- **Top-K:** {result.config.top_k}")
    lines.append("")

    # --- summary table ---
    lines.append("## Aggregated Metrics")
    lines.append("")
    lines.append(_metric_table(result.treatments))
    lines.append("")

    # --- per-query detail ---
    lines.append("## Per-Query Detail")
    lines.append("")
    for t in result.treatments:
        lines.append(f"### {t.label}")
        lines.append("")
        header_q = "| Query | p@1 | p@5 | r@5 | ap |"
        sep_q = "|---" * 5 + "|"
        rows_q = []
        for qi, pq in enumerate(t.per_query_metrics):
            rows_q.append(
                f"| {qi} | {_fmt(pq['p@1'])} | {_fmt(pq['p@5'])} |"
                f" {_fmt(pq['r@5'])} | {_fmt(pq['ap'])} |"
            )
        lines.append(header_q)
        lines.append(sep_q)
        lines.extend(rows_q)
        lines.append("")

    # --- norm statistics ---
    lines.append("## Document Norm Statistics")
    lines.append("")
    lines.append(_norm_stats_table(result.treatments, side="doc"))
    lines.append("")
    lines.append("## Query Norm Statistics")
    lines.append("")
    lines.append(_norm_stats_table(result.treatments, side="query"))
    lines.append("")

    return "\n".join(lines)


def report_to_json(result: AblationResult) -> str:
    """Serialize the ablation result to pretty-printed JSON."""
    data = {
        "config": {
            "top_k": result.config.top_k,
            "label": result.config.label,
        },
        "corpus_size": result.corpus_size,
        "num_queries": result.num_queries,
        "embedding_dim": result.embedding_dim,
        "treatments": [
            {
                "label": t.label,
                "aggregated": t.aggregated,
                "per_query_metrics": t.per_query_metrics,
                "doc_norm_stats": t.doc_norm_stats,
                "query_norm_stats": t.query_norm_stats,
            }
            for t in result.treatments
        ],
    }
    return json.dumps(data, indent=2)


def write_report(
    result: AblationResult,
    markdown_path: Optional[str] = None,
    json_path: Optional[str] = None,
) -> None:
    """Write ablation report to markdown and/or JSON files."""
    if markdown_path:
        with open(markdown_path, "w") as f:
            f.write(report_to_markdown(result))
    if json_path:
        with open(json_path, "w") as f:
            f.write(report_to_json(result))
