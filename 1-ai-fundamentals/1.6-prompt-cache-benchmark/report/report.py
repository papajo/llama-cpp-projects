"""
Report generator for cache benchmarks.

Produces:
  - Interactive HTML report with Plotly charts
  - Markdown summary
  - JSON export
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..harness.metrics import CacheBenchmarkResult
from .plots import (
    ttft_comparison,
    speedup_chart,
    per_prompt_scatter,
    savings_heatmap,
)


def generate_markdown(
    results: Dict[str, CacheBenchmarkResult],
    output_path: str = "cache-benchmark-report.md",
) -> str:
    """Generate a markdown report."""
    lines = [
        "# Prompt Cache Benchmark Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Scenarios: {len(results)}",
        "",
        "## Summary",
        "",
        "| Scenario | Uncached (ms) | Cached (ms) | Speedup | Savings |",
        "|----------|--------------|-------------|---------|---------|",
    ]

    for name, r in results.items():
        saved = r.avg_ttft_uncached_ms - r.avg_ttft_cached_ms
        lines.append(
            f"| {name} | {r.avg_ttft_uncached_ms:.1f} | "
            f"{r.avg_ttft_cached_ms:.1f} | "
            f"{r.speedup_factor:.1f}x | "
            f"{saved:.0f} ms |"
        )

    lines.extend(["", "## Per-Scenario Details", ""])

    for name, r in results.items():
        lines.extend([
            f"### {name}",
            f"_{r.description}_",
            "",
            f"- **Uncached TTFT**: {r.avg_ttft_uncached_ms:.1f} ms "
            f"(median: {r.median_ttft_uncached_ms:.1f} ms)",
            f"- **Cached TTFT**: {r.avg_ttft_cached_ms:.1f} ms "
            f"(median: {r.median_ttft_cached_ms:.1f} ms)",
            f"- **Speedup**: {r.speedup_factor:.1f}x ({r.speedup_pct:.0f}% reduction)",
            f"- **Prompts measured**: {len(r.runs_cached)} cached, {len(r.runs_uncached)} uncached",
            "",
        ])

        # Per-prompt table
        if r.per_prompt:
            lines.extend([
                "| Prompt | Length | Uncached (ms) | Cached (ms) | Saved (ms) |",
                "|--------|--------|---------------|-------------|------------|",
            ])
            for p in r.per_prompt:
                lines.append(
                    f"| {p['index']+1} | {p['prompt_length']} | "
                    f"{p['ttft_uncached_ms']:.1f} | "
                    f"{p['ttft_cached_ms']:.1f} | "
                    f"{p['saving_ms']:.1f} |"
                )
            lines.append("")

    md = "\n".join(lines)
    Path(output_path).write_text(md)
    return md


def generate_html(
    results: Dict[str, CacheBenchmarkResult],
    output_path: str = "cache-benchmark-report.html",
) -> str:
    """Generate an interactive HTML report with Plotly charts."""

    plot1 = ttft_comparison(results)
    plot2 = speedup_chart(results)
    plot3 = savings_heatmap(results)

    # Summary table rows
    table_rows = ""
    for name, r in results.items():
        saved = r.avg_ttft_uncached_ms - r.avg_ttft_cached_ms
        color = "green" if r.speedup_factor > 1.5 else "amber" if r.speedup_factor > 1.0 else "red"
        table_rows += (
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{r.avg_ttft_uncached_ms:.1f}</td>"
            f"<td>{r.avg_ttft_cached_ms:.1f}</td>"
            f"<td class='{color}'>{r.speedup_factor:.1f}x</td>"
            f"<td>{saved:.0f} ms</td>"
            f"<td>{len(r.per_prompt)}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prompt Cache Benchmark Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          background: #0f1117; color: #e4e6f0; margin: 0; padding: 24px; }}
  h1 {{ font-size: 22px; }}
  .subtitle {{ color: #8b8fa3; font-size: 14px; margin-bottom: 24px; }}
  .section {{ background: #1a1d27; border-radius: 8px; padding: 16px;
              margin-bottom: 20px; border: 1px solid #2a2d3a; }}
  .section h2 {{ font-size: 16px; margin-top: 0; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #2a2d3a; }}
  th {{ color: #8b8fa3; text-align: right; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .green {{ color: #34d399; }} .amber {{ color: #fbbf24; }} .red {{ color: #f87171; }}
</style>
</head>
<body>
<h1>⚡ Prompt Cache Benchmark Report</h1>
<div class="subtitle">{datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(results)} scenarios</div>

<div class="section">
<h2>Summary</h2>
<table><thead><tr>
<th>Scenario</th><th>Uncached (ms)</th><th>Cached (ms)</th><th>Speedup</th><th>Saved</th><th>Prompts</th>
</tr></thead><tbody>
{table_rows}
</tbody></table>
</div>

<div class="section"><h2>TTFT Comparison</h2>{plot1}</div>
<div class="section"><h2>Speedup Factor</h2>{plot2}</div>
<div class="section"><h2>Time Saved Heatmap</h2>{plot3}</div>
"""

    # Per-scenario detail
    for name, r in results.items():
        pplot = per_prompt_scatter(r, f"{name} — Per-Prompt TTFT")
        html += f'<div class="section"><h2>{name}</h2>'
        html += f"<p style='color: #8b8fa3; font-size: 13px;'>{r.description}</p>"
        html += (
            f"<p>Uncached: <strong>{r.avg_ttft_uncached_ms:.1f}</strong> ms &nbsp;|&nbsp; "
            f"Cached: <strong>{r.avg_ttft_cached_ms:.1f}</strong> ms &nbsp;|&nbsp; "
            f"Speedup: <strong class='green'>{r.speedup_factor:.1f}x</strong></p>"
        )
        if pplot:
            html += pplot

        # Per-prompt table
        if r.per_prompt:
            html += """
<table><thead><tr>
<th>#</th><th>Length</th><th>Uncached (ms)</th><th>Cached (ms)</th><th>Saved (ms)</th><th>Speedup</th>
</tr></thead><tbody>"""
            for p in r.per_prompt:
                html += (
                    f"<tr><td>{p['index']+1}</td>"
                    f"<td>{p['prompt_length']}</td>"
                    f"<td>{p['ttft_uncached_ms']:.1f}</td>"
                    f"<td>{p['ttft_cached_ms']:.1f}</td>"
                    f"<td class='green'>{p['saving_ms']:.1f}</td>"
                    f"<td class='green'>{p['speedup']:.1f}x</td>"
                    f"</tr>"
                )
            html += "</tbody></table>"

        html += "</div>"

    html += """
<p style="color: #8b8fa3; font-size: 12px; margin-top: 32px;">
Generated by Prompt Cache Benchmark
</p>
</body>
</html>"""

    Path(output_path).write_text(html)
    return html


def generate_json(
    results: Dict[str, CacheBenchmarkResult],
    output_path: str = "cache-benchmark-results.json",
) -> str:
    """Export results as JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "scenarios": {
            name: r.to_dict() for name, r in results.items()
        },
    }
    Path(output_path).write_text(json.dumps(data, indent=2))
    return output_path
