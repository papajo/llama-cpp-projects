"""
Report generation for reasoning budget sweep results.

Produces markdown tables and summaries from ``ConfigSummary`` objects.

Usage::

    from budget_sweep.metrics import compute_metrics
    from budget_sweep.reporter import format_report

    summaries = compute_metrics(results)
    report = format_report(summaries, title="Token Budget Sweep")
    print(report)
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional

from .metrics import ConfigSummary, PerPromptMetrics


def format_report(
    summaries: List[ConfigSummary],
    title: str = "Reasoning Budget Sweep Report",
    notes: Optional[List[str]] = None,
) -> str:
    """
    Format sweep results as a markdown report.

    Args:
        summaries: List of config summaries from ``compute_metrics``.
        title: Report title.
        notes: Optional list of note strings appended at the bottom.

    Returns:
        Markdown string.
    """
    buf = io.StringIO()
    buf.write(f"# {title}\n\n")

    # Overview
    total_runs = sum(s.total_runs for s in summaries)
    total_errors = sum(s.errors for s in summaries)
    n_configs = len(summaries)
    buf.write(f"- **Configurations tested:** {n_configs}\n")
    buf.write(f"- **Total API calls:** {total_runs}\n")
    buf.write(f"- **Errors:** {total_errors}\n\n")

    if not summaries:
        buf.write("*(No results to report)*\n")
        return buf.getvalue()

    # Best token efficiency
    best = max(summaries, key=lambda s: s.token_efficiency)
    buf.write("### Best Token Efficiency\n\n")
    buf.write(
        f"**{_param_str(best.params)}** — "
        f"{best.token_efficiency:.2f} words/token "
        f"(avg {best.avg_word_count:.0f} words / "
        f"{best.avg_completion_tokens:.0f} tokens)\n\n"
    )

    # Leaderboard (sorted by token efficiency desc)
    buf.write("## Leaderboard by Token Efficiency\n\n")
    sorted_by_eff = sorted(
        summaries, key=lambda s: s.token_efficiency, reverse=True
    )
    buf.write("| Rank | Params | Words | Tokens | Efficiency | Diversity | Repetition |\n")
    buf.write("|------|--------|-------|--------|------------|-----------|------------|\n")
    for rank, s in enumerate(sorted_by_eff, 1):
        buf.write(
            f"| {rank} "
            f"| {_param_str(s.params)} "
            f"| {s.avg_word_count:.0f} "
            f"| {s.avg_completion_tokens:.0f} "
            f"| {s.token_efficiency:.2f} "
            f"| {s.avg_diversity:.3f} "
            f"| {s.avg_repetition_rate:.3f} "
            f"|\n"
        )
    buf.write("\n")

    # Leaderboard by diversity
    buf.write("## Leaderboard by Diversity\n\n")
    sorted_by_div = sorted(
        summaries, key=lambda s: s.avg_diversity, reverse=True
    )
    buf.write("| Rank | Params | Diversity | Entropy | Repetition | Tokens |\n")
    buf.write("|------|--------|-----------|---------|------------|--------|\n")
    for rank, s in enumerate(sorted_by_div, 1):
        buf.write(
            f"| {rank} "
            f"| {_param_str(s.params)} "
            f"| {s.avg_diversity:.3f} "
            f"| {s.avg_entropy:.2f} "
            f"| {s.avg_repetition_rate:.3f} "
            f"| {s.avg_completion_tokens:.0f} "
            f"|\n"
        )
    buf.write("\n")

    # Speed table
    buf.write("## Generation Speed\n\n")
    sorted_by_speed = sorted(
        summaries, key=lambda s: s.avg_speed_tok_s, reverse=True
    )
    buf.write("| Rank | Params | Tokens/s | Words/s | Time Eff |\n")
    buf.write("|------|--------|----------|---------|----------|\n")
    for rank, s in enumerate(sorted_by_speed, 1):
        buf.write(
            f"| {rank} "
            f"| {_param_str(s.params)} "
            f"| {s.avg_speed_tok_s:.1f} "
            f"| {s.avg_speed_tok_s * s.token_efficiency:.1f} "
            f"| {s.time_efficiency:.2f} "
            f"|\n"
        )
    buf.write("\n")

    # Detailed configs
    buf.write("## Per-Configuration Details\n\n")
    for s in sorted_by_eff:
        buf.write(f"### Config: {_param_str(s.params)}\n\n")
        buf.write(f"- **Word count:** {s.avg_word_count:.0f} avg\n")
        buf.write(f"- **Completion tokens:** {s.avg_completion_tokens:.0f} avg\n")
        buf.write(f"- **Token efficiency:** {s.token_efficiency:.2f} words/token\n")
        buf.write(f"- **Diversity:** {s.avg_diversity:.3f}\n")
        buf.write(f"- **Entropy:** {s.avg_entropy:.2f}\n")
        buf.write(f"- **Repetition rate:** {s.avg_repetition_rate:.3f}\n")
        buf.write(f"- **Speed:** {s.avg_speed_tok_s:.1f} tok/s\n")
        buf.write(f"- **Errors:** {s.errors}/{s.total_runs}\n\n")

    # Notes
    if notes:
        buf.write("---\n\n")
        buf.write("## Notes\n\n")
        for n in notes:
            buf.write(f"- {n}\n")
        buf.write("\n")

    return buf.getvalue()


def format_per_prompt(summaries: List[ConfigSummary]) -> str:
    """
    Format per-prompt breakdown for debugging or detailed analysis.

    Only useful if the sweep has multiple prompts.
    """
    buf = io.StringIO()
    buf.write("## Per-Prompt Breakdown\n\n")
    for s in summaries:
        buf.write(f"### {_param_str(s.params)}\n\n")
        buf.write(f"- **Prompts:** {s.n_prompts}\n")
        buf.write(f"- **Total runs:** {s.total_runs}\n")
        buf.write(f"- **Errors:** {s.errors}\n\n")
    return buf.getvalue()


def _param_str(params: Dict) -> str:
    """Short inline param string for table cells."""
    parts: List[str] = []
    if "temperature" in params:
        parts.append(f"T={params['temperature']}")
    if "max_tokens" in params:
        parts.append(f"M={params['max_tokens']}")
    if "top_p" in params:
        parts.append(f"p={params['top_p']}")
    if "top_k" in params:
        parts.append(f"k={params['top_k']}")
    if "min_p" in params:
        parts.append(f"mp={params['min_p']}")
    return ", ".join(parts) if parts else "default"
