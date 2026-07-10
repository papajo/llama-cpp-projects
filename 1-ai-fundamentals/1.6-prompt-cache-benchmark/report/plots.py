"""
Plotly chart generators for cache benchmark reports.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..harness.metrics import CacheBenchmarkResult


def _chart_div(fig: dict) -> str:
    """Convert a Plotly figure dict to an HTML div string."""
    import random
    uid = random.randint(100000, 999999)
    return f"""<div id="chart-{uid}" style="width:100%;height:400px;">
<script>
  try {{
    Plotly.newPlot('chart-{uid}', {json.dumps(fig['data'])},
                   {json.dumps(fig.get('layout', {}))},
                   {{responsive: true, displayModeBar: false}});
  }} catch(e) {{ console.error('Chart error:', e); }}
</script>
</div>"""


def ttft_comparison(
    results: Dict[str, CacheBenchmarkResult],
    title: str = "TTFT: Cached vs Uncached by Scenario",
) -> str:
    """
    Grouped bar chart comparing cached vs uncached TTFT per scenario.
    """
    scenarios = list(results.keys())
    cached = [results[s].avg_ttft_cached_ms for s in scenarios]
    uncached = [results[s].avg_ttft_uncached_ms for s in scenarios]
    speedups = [results[s].speedup_factor for s in scenarios]

    fig = {
        "data": [
            {
                "type": "bar",
                "name": "Uncached",
                "x": scenarios,
                "y": uncached,
                "marker": {"color": "#f87171"},
                "hovertemplate": "%{y:.1f} ms<extra></extra>",
            },
            {
                "type": "bar",
                "name": "Cached",
                "x": scenarios,
                "y": cached,
                "marker": {"color": "#34d399"},
                "hovertemplate": "%{y:.1f} ms<extra></extra>",
            },
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "barmode": "group",
            "xaxis": {"title": "Scenario", "tickangle": -45},
            "yaxis": {"title": "TTFT (ms)"},
            "template": "plotly_dark",
            "margin": {"b": 120, "t": 60, "r": 30, "l": 60},
            "legend": {"orientation": "h", "y": 1.1},
            "hovermode": "x unified",
            "annotations": [
                {
                    "x": s, "y": max(u, c) + 5,
                    "text": f"{s}x",
                    "showarrow": False,
                    "font": {"size": 10, "color": "#fbbf24"},
                    "xref": "x", "yref": "y",
                }
                for s, u, c in zip(scenarios, uncached, cached)
            ],
        },
    }
    return _chart_div(fig)


def speedup_chart(
    results: Dict[str, CacheBenchmarkResult],
    title: str = "Cache Speedup Factor by Scenario",
) -> str:
    """
    Horizontal bar chart of speedup factors.
    """
    scenarios = list(results.keys())
    factors = [results[s].speedup_factor for s in scenarios]
    colors = ["#34d399" if f > 1.5 else "#fbbf24" if f > 1.0 else "#f87171"
              for f in factors]

    fig = {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": factors,
                "y": scenarios,
                "marker": {"color": colors},
                "text": [f"{f:.1f}x" for f in factors],
                "textposition": "outside",
                "hovertemplate": "%{y}<br>Speedup: %{x:.1f}x<extra></extra>",
            }
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": "Speedup Factor (×)", "range": [0, max(factors) * 1.3]},
            "yaxis": {"title": "", "autorange": "reversed"},
            "template": "plotly_dark",
            "margin": {"l": 120, "t": 60, "r": 60, "b": 40},
            "hovermode": "y unified",
        },
    }
    return _chart_div(fig)


def per_prompt_scatter(
    result: CacheBenchmarkResult,
    title: str = "Per-Prompt TTFT: Cached vs Uncached",
) -> str:
    """
    Scatter plot comparing cached vs uncached TTFT per prompt.
    """
    if not result.per_prompt:
        return ""

    prompt_lens = [p["prompt_length"] for p in result.per_prompt]
    cached = [p["ttft_cached_ms"] for p in result.per_prompt]
    uncached = [p["ttft_uncached_ms"] for p in result.per_prompt]

    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Uncached",
                "x": prompt_lens,
                "y": uncached,
                "marker": {"color": "#f87171", "size": 6},
                "line": {"color": "#f87171", "dash": "dot"},
                "hovertemplate": "Len: %{x}<br>TTFT: %{y:.1f} ms<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Cached",
                "x": prompt_lens,
                "y": cached,
                "marker": {"color": "#34d399", "size": 6},
                "line": {"color": "#34d399"},
                "hovertemplate": "Len: %{x}<br>TTFT: %{y:.1f} ms<extra></extra>",
            },
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": "Prompt Length (chars)"},
            "yaxis": {"title": "TTFT (ms)"},
            "template": "plotly_dark",
            "margin": {"t": 60, "r": 30, "l": 60, "b": 60},
            "hovermode": "closest",
            "legend": {"orientation": "h", "y": 1.1},
        },
    }
    return _chart_div(fig)


def savings_heatmap(
    results: Dict[str, CacheBenchmarkResult],
    title: str = "Time Saved by Cache (ms per prompt)",
) -> str:
    """
    Heatmap showing per-prompt time savings across scenarios.
    """
    scenarios = list(results.keys())
    max_prompts = max(len(results[s].per_prompt) for s in scenarios) if results else 0

    if max_prompts == 0:
        return ""

    z_data = []
    for s in scenarios:
        row = [p["saving_ms"] for p in results[s].per_prompt]
        # Pad to uniform length
        row += [0] * (max_prompts - len(row))
        z_data.append(row)

    fig = {
        "data": [
            {
                "type": "heatmap",
                "z": z_data,
                "x": [f"Prompt {i+1}" for i in range(max_prompts)],
                "y": scenarios,
                "colorscale": [
                    [0, "#2a2d3a"],
                    [0.5, "#60a5fa"],
                    [1, "#34d399"],
                ],
                "hovertemplate": "Scenario: %{y}<br>Prompt: %{x}<br>Saved: %{z:.0f} ms<extra></extra>",
            }
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": ""},
            "yaxis": {"title": "", "autorange": "reversed"},
            "template": "plotly_dark",
            "margin": {"l": 120, "t": 60, "r": 30, "b": 80},
        },
    }
    return _chart_div(fig)
