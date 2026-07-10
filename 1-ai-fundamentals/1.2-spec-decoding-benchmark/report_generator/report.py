"""
generate_report — produce a comprehensive Markdown report from benchmark results.

Also saves JSON data and interactive HTML charts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from .analysis import BenchmarkAnalysis
from .plots import (
    build_dashboard,
    plot_acceptance_rate,
    plot_tokens_per_second,
    plot_strategy_comparison,
    plot_latency_profile,
    plot_draft_efficiency,
    plot_workload_strategy_heatmap,
)


def generate_report(
    analysis: BenchmarkAnalysis,
    output_dir: str = "results",
    title: str = "Speculative Decoding Benchmark",
) -> Path:
    """
    Generate a complete benchmark report package.

    Produces:
      - report.md          — full markdown report
      - report.html        — interactive HTML dashboard
      - summary.csv        — aggregated metrics table
      - summary.json       — raw data

    Returns path to output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Markdown report ──
    md = _build_markdown(analysis, title)
    (out / "report.md").write_text(md)

    # ── HTML dashboard ──
    html = build_dashboard(analysis)
    (out / "report.html").write_text(html)

    # ── CSV summary ──
    summary = analysis.summary_table()
    if summary:
        with open(out / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

    # ── JSON summary ──
    analysis.to_json(out / "summary.json")

    print(f"\n📊 Report generated in {out}/")
    print(f"   📄 report.md")
    print(f"   🌐 report.html")
    print(f"   📊 summary.csv")
    print(f"   📊 summary.json")

    return out


def _build_markdown(analysis: BenchmarkAnalysis, title: str) -> str:
    """Generate the Markdown report content."""
    lines = []
    lines.append(f"# {title}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Strategies tested:** {len(analysis.strategies)}")
    lines.append(f"- **Prompts used:** {', '.join(analysis.prompts_used)}")
    lines.append(f"- **Baseline:** {analysis.baseline.avg('tokens_per_second'):.1f} tok/s (no speculation)" if analysis.baseline else "- **Baseline:** N/A")
    lines.append("")

    # Summary table
    lines.append("## Summary Results")
    lines.append("")
    lines.append("| Strategy | Family | tok/s | Speedup | Accept Rate | Draft % | p95 Latency |")
    lines.append("|----------|--------|-------|---------|-------------|---------|-------------|")

    summary = analysis.summary_table()
    for row in sorted(summary, key=lambda r: r.get("speedup", 0), reverse=True):
        lines.append(
            f"| {row['strategy']} | {row['family']} | {row['avg_tok_s']} | "
            f"{row['speedup']}× | {row['avg_accept_rate']:.1%} | "
            f"{row['avg_draft_proportion']:.1%} | {row['avg_latency_p95_ms']} ms |"
        )
    lines.append("")

    # Per-workload analysis
    lines.append("## Per-Workload Analysis")
    lines.append("")
    for prompt in analysis.prompts_used:
        lines.append(f"### {prompt}")
        lines.append("")
        lines.append("| Strategy | tok/s | Accept Rate | Draft % |")
        lines.append("|----------|-------|-------------|---------|")

        for label, strat in analysis.strategies.items():
            if prompt in strat.prompt_results:
                vals = [m for m in strat.prompt_results[prompt]]
                if vals:
                    avg_tok = sum(m.tokens_per_second for m in vals) / len(vals)
                    avg_acc = sum(m.acceptance_rate for m in vals) / len(vals)
                    avg_draft = sum(m.draft_proportion for m in vals) / len(vals)
                    lines.append(
                        f"| {label} | {avg_tok:.1f} | {avg_acc:.1%} | "
                        f"{avg_draft:.1%} |"
                    )
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    for rec in analysis.recommendations():
        lines.append(f"- {rec}")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("Each strategy was tested against multiple prompt workloads:")
    lines.append("- **code/** — structured code generation (predictable syntax)")
    lines.append("- **prose/** — natural language (variable patterns)")
    lines.append("- **logs/** — repetitive templated text (highly predictable)")
    lines.append("")
    lines.append("Metrics collected:")
    lines.append("- **Acceptance rate**: fraction of draft tokens accepted by the target model")
    lines.append("- **Tokens/second**: generation throughput (wall time)")
    lines.append("- **p95 latency**: 95th percentile per-token generation latency")
    lines.append("- **Draft efficiency**: accepted tokens per draft forward pass")
    lines.append("- **Verification ratio**: target forward passes per generated token")
    lines.append("")
    lines.append("### llama-server Configuration")
    lines.append("")
    lines.append("All runs use the same target model with `--cont-batching` enabled. ")
    lines.append("Draft-model strategies require a compatible draft GGUF. ")
    lines.append("N-gram strategies operate on prompt context only.")
    lines.append("")

    return "\n".join(lines)
