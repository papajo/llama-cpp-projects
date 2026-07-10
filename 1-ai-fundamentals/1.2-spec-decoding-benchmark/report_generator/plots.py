"""
Plotly chart generators for speculative decoding benchmark visualisations.

Charts:
  1. Acceptance rate by strategy (grouped bar)
  2. Tokens per second by strategy (grouped bar)
  3. Strategy comparison radar / parallel plot
  4. Latency profile (p50/p95/p99)
  5. Draft efficiency scatter
  6. Workload × strategy heatmap
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

import plotly.graph_objects as go

from .analysis import AggregatedStrategy, BenchmarkAnalysis
from harness.metrics import BenchmarkMetrics

# ── Colour palette ────────────────────────────────────────────────

STRATEGY_COLORS = {
    "baseline": "#1f77b4",
    "Draft-Simple": "#2ca02c",
    "Draft-EAGLE3": "#ff7f0e",
    "Draft-MTP": "#d62728",
    "Ngram-Simple": "#9467bd",
    "Ngram-Map-K": "#8c564b",
    "Ngram-Map-K4V": "#e377c2",
    "Ngram-Mod": "#7f7f7f",
    "Ngram-Cache": "#bcbd22",
}

FAMILY_COLORS = {
    "baseline": "#1f77b4",
    "draft-model": "#2ca02c",
    "ngram": "#9467bd",
}

THEME = dict(
    paper_bgcolor="#f8f9fa",
    plot_bgcolor="#f8f9fa",
    font=dict(family="SF Mono, Menlo, Consolas, monospace", size=12),
    hovermode="x unified",
    margin=dict(l=60, r=40, t=60, b=60),
)


def _color_for(label: str) -> str:
    return STRATEGY_COLORS.get(label, "#333333")


# ── Individual charts ────────────────────────────────────────────

def plot_acceptance_rate(
    analysis: BenchmarkAnalysis,
) -> go.Figure:
    """
    Grouped bar chart: acceptance rate by strategy and prompt type.
    Higher is better — indicates draft tokens are accepted more often.
    """
    fig = go.Figure(layout=THEME)

    for label, strat in analysis.strategies.items():
        prompt_rates = strat.by_prompt("acceptance_rate")
        fig.add_trace(go.Bar(
            name=label,
            x=list(prompt_rates.keys()),
            y=list(prompt_rates.values()),
            marker_color=_color_for(label),
            hovertemplate="%{y:.1%}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Draft Token Acceptance Rate by Strategy</b><br>"
                 "<sup>Higher = better draft quality</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(title=dict(text="Prompt")),
        yaxis=dict(
            title=dict(text="Acceptance Rate"),
            tickformat=".0%",
            gridcolor="#e9ecef",
            range=[0, 1],
        ),
        barmode="group",
        legend=dict(title=dict(text="Strategy")),
    )

    return fig


def plot_tokens_per_second(
    analysis: BenchmarkAnalysis,
) -> go.Figure:
    """
    Grouped bar chart: generation throughput by strategy and prompt.
    The primary performance metric.
    """
    fig = go.Figure(layout=THEME)

    for label, strat in analysis.strategies.items():
        prompt_tok = strat.by_prompt("tokens_per_second")
        fig.add_trace(go.Bar(
            name=label,
            x=list(prompt_tok.keys()),
            y=list(prompt_tok.values()),
            marker_color=_color_for(label),
            hovertemplate="%{y:.1f} tok/s<extra></extra>",
        ))

    # Add baseline as a line overlay
    if analysis.baseline:
        base_tok = analysis.baseline.by_prompt("tokens_per_second")
        fig.add_trace(go.Scatter(
            name="Baseline (no speculation)",
            x=list(base_tok.keys()),
            y=list(base_tok.values()),
            mode="lines+markers",
            line=dict(color=STRATEGY_COLORS["baseline"], width=3, dash="dash"),
            marker=dict(size=10),
            hovertemplate="%{y:.1f} tok/s<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Generation Throughput by Strategy</b><br>"
                 "<sup>Higher = faster generation</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(title=dict(text="Prompt")),
        yaxis=dict(
            title=dict(text="Tokens / Second"),
            gridcolor="#e9ecef",
        ),
        barmode="group",
        legend=dict(title=dict(text="Strategy")),
    )

    return fig


def plot_strategy_comparison(
    analysis: BenchmarkAnalysis,
    metrics: Optional[List[str]] = None,
) -> go.Figure:
    """
    Radar chart comparing strategies across multiple metrics.

    Useful for seeing which strategy has the best all-round profile.
    """
    if metrics is None:
        metrics = [
            "tokens_per_second",
            "acceptance_rate",
            "draft_efficiency",
        ]

    # Normalise each metric to [0, 1] across all strategies
    all_vals = {m: [] for m in metrics}
    for label, strat in analysis.strategies.items():
        for m in metrics:
            all_vals[m].append(strat.avg(m))

    # Min-max normalisation
    def normalise(vals):
        mn, mx = min(vals), max(vals)
        if mx - mn == 0:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    fig = go.Figure(layout=THEME)

    for i, (label, strat) in enumerate(analysis.strategies.items()):
        values = [normalise(all_vals[m])[i] for m in metrics]
        # Close the loop
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=list(metrics) + [metrics[0]],
            name=label,
            line=dict(color=_color_for(label), width=2),
            fill="toself",
            opacity=0.3,
        ))

    fig.update_layout(
        title=dict(
            text="<b>Strategy Comparison Radar</b><br>"
                 "<sup>Normalised scores (higher = better)</sup>",
            font=dict(size=16),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
            )),
        legend=dict(title=dict(text="Strategy")),
    )

    return fig


def plot_latency_profile(
    analysis: BenchmarkAnalysis,
) -> go.Figure:
    """
    Grouped bar chart: p50 / p95 / p99 latency by strategy.
    Lower is better — indicates more consistent generation.
    """
    percentiles = ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]
    labels = ["p50", "p95", "p99"]

    fig = go.Figure(layout=THEME)

    for label, strat in analysis.strategies.items():
        vals = [strat.avg(p) for p in percentiles]
        fig.add_trace(go.Bar(
            name=label,
            x=labels,
            y=vals,
            marker_color=_color_for(label),
            hovertemplate="%{y:.1f} ms<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Generation Latency Profile</b><br>"
                 "<sup>Lower = faster per-token generation</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(title=dict(text="Percentile")),
        yaxis=dict(
            title=dict(text="Latency (ms)"),
            gridcolor="#e9ecef",
        ),
        barmode="group",
        legend=dict(title=dict(text="Strategy")),
    )

    return fig


def plot_draft_efficiency(
    analysis: BenchmarkAnalysis,
) -> go.Figure:
    """
    Scatter plot: draft efficiency vs acceptance rate.

    Each point is a strategy; size = tokens/sec.
    Helps identify strategies that balance efficiency with quality.
    """
    fig = go.Figure(layout=THEME)

    for label, strat in analysis.strategies.items():
        accept_rate = strat.avg("acceptance_rate")
        draft_eff = strat.avg("draft_efficiency")
        tok_s = strat.avg("tokens_per_second")

        fig.add_trace(go.Scatter(
            x=[accept_rate],
            y=[draft_eff],
            mode="markers+text",
            name=label,
            marker=dict(
                size=min(30, max(10, tok_s * 2)),
                color=_color_for(label),
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            text=[label],
            textposition="top center",
            hovertemplate=(
                f"Accept: %{{x:.1%}}<br>"
                f"Draft eff: %{{y:.2f}}<br>"
                f"Throughput: {tok_s:.1f} tok/s<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="<b>Draft Efficiency vs Acceptance Rate</b><br>"
                 "<sup>Point size = throughput (tok/s)</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="Acceptance Rate"),
            tickformat=".0%",
            gridcolor="#e9ecef",
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(text="Draft Efficiency (accepted / forward pass)"),
            gridcolor="#e9ecef",
        ),
        legend=dict(title=dict(text="Strategy")),
    )

    return fig


def plot_workload_strategy_heatmap(
    analysis: BenchmarkAnalysis,
) -> go.Figure:
    """
    Heatmap: speedup factor × (strategy, workload).

    Shows which strategy works best for which type of prompt.
    """
    strategies = list(analysis.strategies.keys())
    prompts = analysis.prompts_used

    # Build speedup matrix
    baseline_tok = {
        prompt: statistics.mean([
            m.tokens_per_second
            for m in (analysis.baseline.prompt_results.get(prompt, [])
                      if analysis.baseline else [])
        ])
        for prompt in prompts
    }
    base_avg = statistics.mean(list(baseline_tok.values())) if baseline_tok else 1.0

    z_data = []
    for strat_label in strategies:
        strat = analysis.strategies[strat_label]
        row = []
        for prompt in prompts:
            vals = [m.tokens_per_second for m in strat.prompt_results.get(prompt, [])]
            avg = statistics.mean(vals) if vals else 0
            base = baseline_tok.get(prompt, base_avg)
            speedup = avg / base if base > 0 else 0
            row.append(round(speedup, 2))
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=prompts,
        y=strategies,
        colorscale="Viridis",
        texttemplate="%{z:.2f}×",
        textfont=dict(size=11),
        hovertemplate="Strategy: %{y}<br>Prompt: %{x}<br>Speedup: %{z:.2f}×<extra></extra>",
        colorbar=dict(title=dict(text="Speedup vs Baseline")),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Speedup: Strategy × Workload Heatmap</b><br>"
                 "<sup>Values > 1.0 = faster than no speculation</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(title=dict(text="Prompt"), tickangle=30),
        yaxis=dict(title=dict(text="Strategy")),
    )

    return fig


# ── Full report dashboard ────────────────────────────────────────

def build_dashboard(analysis: BenchmarkAnalysis) -> str:
    """Build a full HTML dashboard with all charts."""
    from plotly.io import to_html

    charts = [
        ("Throughput", plot_tokens_per_second(analysis)),
        ("Acceptance Rate", plot_acceptance_rate(analysis)),
        ("Latency Profile", plot_latency_profile(analysis)),
        ("Strategy Radar", plot_strategy_comparison(analysis)),
        ("Draft Efficiency", plot_draft_efficiency(analysis)),
        ("Workload Heatmap", plot_workload_strategy_heatmap(analysis)),
    ]

    summary = analysis.summary_table()
    recs = analysis.recommendations()

    charts_html = "\n".join([
        f'<div class="chart-card">'
        f'<h2>{title}</h2>'
        f'{to_html(fig, include_plotlyjs=False, full_html=False)}'
        f'</div>'
        for title, fig in charts
    ])

    recs_html = "\n".join([f"<li>{r}</li>" for r in recs])
    summary_html = _summary_table_html(summary)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Speculative Decoding Benchmark Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8f9fa; color: #333; padding: 24px;
}}
h1 {{ font-size: 28px; margin-bottom: 4px; }}
.subtitle {{ color: #666; margin-bottom: 24px; }}
table {{
    border-collapse: collapse; width: 100%; margin-bottom: 24px;
    background: white; border-radius: 8px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #1a1a2e; color: white; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
tr:hover {{ background: #f1f3f5; }}
.speedup-good {{ color: #2ca02c; font-weight: bold; }}
.recs {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.recs li {{ margin: 8px 0; line-height: 1.5; }}
.chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.chart-card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.chart-card h2 {{ font-size: 16px; margin-bottom: 8px; }}
@media (max-width: 1000px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>⚡ Speculative Decoding Benchmark Report</h1>
<p class="subtitle">Comparing all 8 speculative decoding strategies across workload types</p>

<div class="recs">
<h2>📋 Recommendations</h2>
<ol>{recs_html}</ol>
</div>

<h2>📊 Summary Table</h2>
{summary_html}

<div class="chart-grid">
{charts_html}
</div>
</body>
</html>"""
    return html


def _summary_table_html(summary: List[Dict]) -> str:
    """Build HTML table from summary data."""
    rows_html = ""
    for row in sorted(summary, key=lambda r: r.get("speedup", 0), reverse=True):
        speedup = row.get("speedup", 1.0)
        speedup_class = "speedup-good" if speedup > 1.0 else ""
        rows_html += (
            f"<tr>"
            f"<td>{row['strategy']}</td>"
            f"<td>{row['family']}</td>"
            f"<td>{row['avg_tok_s']}</td>"
            f"<td class='{speedup_class}'>{speedup}×</td>"
            f"<td>{row['avg_accept_rate']:.1%}</td>"
            f"<td>{row['avg_draft_proportion']:.1%}</td>"
            f"<td>{row['avg_latency_p95_ms']}</td>"
            f"</tr>"
        )

    return f"""<table>
<thead><tr>
<th>Strategy</th><th>Family</th><th>tok/s</th><th>Speedup</th>
<th>Accept Rate</th><th>Draft %</th><th>p95 Latency (ms)</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""
