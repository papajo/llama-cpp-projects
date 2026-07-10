"""
Plotly chart generators for quantisation visualisation.

Chart types:
  1. Memory waterfall — stacked breakdown (weights + KV + overhead)
  2. Quant comparison — horizontal bars
  3. OOM frontier — memory vs quality with VRAM thresholds
  4. Model comparison — compare multiple models at same quant
  5. Context scaling — memory vs context length
"""

from __future__ import annotations

from typing import Dict, List, Optional

from gguf_reader.models import GGUFModel, QuantType

from quant_planner.estimator import HardwareProfile, MemoryEstimate, QuantEstimator


def _chart_to_html(chart_json: dict) -> str:
    """Convert a Plotly figure dict to an HTML div string."""
    import json
    return f"""<div id="chart-{id(chart_json)}" style="width:100%;height:500px;">
<script>
  try {{
    Plotly.newPlot('chart-{id(chart_json)}', {json.dumps(chart_json['data'])},
                   {json.dumps(chart_json.get('layout', {}))},
                   {{responsive: true, displayModeBar: true}});
  }} catch(e) {{ console.error('Plotly chart error:', e); }}
</script>
</div>"""


def memory_waterfall(
    estimates: List[MemoryEstimate],
    title: str = "Memory Breakdown by Quant Type",
) -> str:
    """
    Stacked bar chart showing model weights, KV cache, and overhead.
    
    Returns an HTML string with embedded Plotly.js.
    """
    labels = [e.quant_type.name for e in estimates]
    weights = [round(e.model_weights_gb, 2) for e in estimates]
    kv = [round(e.kv_cache_gb, 2) for e in estimates]
    overhead = [round(e.overhead_gb, 2) for e in estimates]

    fig = {
        "data": [
            {
                "type": "bar",
                "name": "Model Weights",
                "x": labels,
                "y": weights,
                "marker": {"color": "#6366f1"},
                "hovertemplate": "%{y} GB<extra></extra>",
            },
            {
                "type": "bar",
                "name": "KV Cache",
                "x": labels,
                "y": kv,
                "marker": {"color": "#34d399"},
                "hovertemplate": "%{y} GB<extra></extra>",
            },
            {
                "type": "bar",
                "name": "Overhead",
                "x": labels,
                "y": overhead,
                "marker": {"color": "#fbbf24"},
                "hovertemplate": "%{y} GB<extra></extra>",
            },
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "barmode": "stack",
            "xaxis": {"title": "Quantisation Type", "tickangle": -45},
            "yaxis": {"title": "Memory (GB)"},
            "template": "plotly_dark",
            "margin": {"b": 120, "t": 60, "r": 30, "l": 60},
            "legend": {"orientation": "h", "y": 1.1},
            "hovermode": "x unified",
        },
    }
    return _chart_to_html(fig)


def quant_comparison(
    estimates: List[MemoryEstimate],
    hardware: Optional[HardwareProfile] = None,
    title: str = "Quantisation Comparison",
) -> str:
    """
    Horizontal bar chart showing total memory per quant type,
    with VRAM threshold line.
    """
    # Sort by total descending
    sorted_ests = sorted(estimates, key=lambda e: e.total_gb, reverse=True)

    labels = [e.quant_type.name for e in sorted_ests]
    totals = [round(e.total_gb, 2) for e in sorted_ests]
    qualities = [round(e.quality_score * 100, 0) for e in sorted_ests]
    fits = [e.fits_in_vram for e in sorted_ests]

    colors = ["#34d399" if f else "#f87171" for f in fits]

    fig = {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": totals,
                "y": labels,
                "marker": {"color": colors},
                "text": [f"{q}%" for q in qualities],
                "textposition": "outside",
                "hovertemplate": (
                    "%{y}<br>Total: %{x} GB<br>Quality: %{text}"
                    "<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": "Total Memory (GB)"},
            "yaxis": {"title": "", "autorange": "reversed"},
            "template": "plotly_dark",
            "margin": {"l": 120, "t": 60, "r": 60, "b": 40},
            "hovermode": "y unified",
        },
    }

    # Add VRAM threshold line
    if hardware is not None:
        vram = hardware.total_vram_gb if not hardware.is_apple_silicon else hardware.total_ram_gb
        fig["data"].append({
            "type": "scatter",
            "mode": "lines",
            "x": [vram, vram],
            "y": [labels[0], labels[-1]],
            "name": f"VRAM ({vram} GB)",
            "line": {"dash": "dash", "color": "#fbbf24", "width": 2},
            "hovertemplate": f"VRAM limit: {vram} GB<extra></extra>",
        })

    return _chart_to_html(fig)


def oom_frontier(
    estimates: List[MemoryEstimate],
    hardware: HardwareProfile,
    title: str = "OOM Frontier — Quality vs Memory",
) -> str:
    """
    Scatter plot: quality score vs memory, with VRAM threshold.
    Helps find the best quant that fits.
    """
    qualities = [e.quality_score * 100 for e in estimates]
    totals = [e.total_gb for e in estimates]
    labels = [e.quant_type.name for e in estimates]
    fits = [e.fits_in_vram for e in estimates]
    colors = ["#34d399" if f else "#f87171" for f in fits]
    sizes = [12 if f else 8 for f in fits]

    vram = hardware.total_vram_gb if not hardware.is_apple_silicon else hardware.total_ram_gb

    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers+text",
                "x": totals,
                "y": qualities,
                "text": labels,
                "textposition": "top center",
                "marker": {
                    "color": colors,
                    "size": sizes,
                    "line": {"width": 1, "color": "white"},
                },
                "hovertemplate": (
                    "%{text}<br>Memory: %{x:.2f} GB<br>Quality: %{y:.1f}%"
                    "<extra></extra>"
                ),
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [vram, vram],
                "y": [min(qualities) - 5, max(qualities) + 5],
                "name": f"VRAM ({vram} GB)",
                "line": {"dash": "dash", "color": "#fbbf24", "width": 2},
                "hovertemplate": f"VRAM limit: {vram} GB<extra></extra>",
            },
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": "Total Memory (GB)", "range": [0, max(totals) * 1.2]},
            "yaxis": {"title": "Relative Quality (%)", "range": [min(qualities) - 5, max(qualities) + 5]},
            "template": "plotly_dark",
            "margin": {"t": 60, "r": 30, "l": 60, "b": 60},
            "hovermode": "closest",
            "annotations": [
                {
                    "x": vram,
                    "y": max(qualities) + 2,
                    "text": "✓ Fits VRAM",
                    "showarrow": False,
                    "font": {"color": "#34d399", "size": 11},
                    "xanchor": "left",
                }
            ],
        },
    }
    return _chart_to_html(fig)


def context_scaling(
    estimator: QuantEstimator,
    quant_types: List[QuantType],
    context_lengths: List[int],
    title: str = "Memory vs Context Length",
) -> str:
    """
    Line chart showing how memory scales with context length for
    selected quant types.
    """
    traces = []
    for qt in quant_types:
        mems = [estimator.estimate(qt, ctx).total_gb for ctx in context_lengths]
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": qt.name,
            "x": context_lengths,
            "y": [round(m, 2) for m in mems],
            "hovertemplate": f"{qt.name}<br>Ctx: %{{x}}<br>Memory: %{{y:.2f}} GB<extra></extra>",
        })

    fig = {
        "data": traces,
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": "Context Length (tokens)"},
            "yaxis": {"title": "Total Memory (GB)"},
            "template": "plotly_dark",
            "margin": {"t": 60, "r": 30, "l": 60, "b": 60},
            "hovermode": "x unified",
            "legend": {"orientation": "h", "y": 1.1},
        },
    }
    return _chart_to_html(fig)


def generate_report_html(
    model: GGUFModel,
    estimates: List[MemoryEstimate],
    hardware: Optional[HardwareProfile] = None,
    output_path: str = "quant-report.html",
) -> str:
    """
    Generate a complete HTML report with all charts.
    Returns the HTML string and writes to output_path.
    """
    waterfall = memory_waterfall(estimates)
    comparison = quant_comparison(estimates, hardware)

    oom_html = ""
    if hardware is not None:
        oom_html = oom_frontier(estimates, hardware)

    # Build the HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quant Report — {model.summary()}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          background: #0f1117; color: #e4e6f0; margin: 0; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #8b8fa3; font-size: 14px; margin-bottom: 24px; }}
  .chart-container {{ background: #1a1d27; border-radius: 8px;
                     padding: 16px; margin-bottom: 20px;
                     border: 1px solid #2a2d3a; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #2a2d3a; }}
  th {{ color: #8b8fa3; text-align: right; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .fit {{ color: #34d399; }} .no-fit {{ color: #f87171; }}
</style>
</head>
<body>
<h1>🔍 Quantisation Report</h1>
<div class="subtitle">{model.summary()}</div>"""

    if hardware is not None:
        html += f'<div class="subtitle">Target: {hardware.name} ({hardware.total_vram_gb} GB VRAM)</div>'

    # Summary table
    html += """<div class="chart-container">
<table><thead><tr>
<th>Quant</th><th>Weights (GB)</th><th>KV Cache (GB)</th><th>Total (GB)</th>
<th>Quality</th><th>Fits</th>
</tr></thead><tbody>"""
    for e in sorted(estimates, key=lambda x: x.total_gb):
        fit_class = "fit" if e.fits_in_vram else "no-fit"
        check = "\u2713" if e.fits_in_vram else "\u2717"
        html += (
            f"<tr><td>{e.quant_type.name}</td>"
            f"<td>{e.model_weights_gb:.2f}</td>"
            f"<td>{e.kv_cache_gb:.2f}</td>"
            f"<td><strong>{e.total_gb:.2f}</strong></td>"
            f"<td>{e.quality_score*100:.0f}%</td>"
            f"<td class='{fit_class}'>{check}</td>"
            f"</tr>"
        )
    html += "</tbody></table></div>"

    # Charts
    html += f'<div class="chart-container">{waterfall}</div>'
    html += f'<div class="chart-container">{comparison}</div>'
    if oom_html:
        html += f'<div class="chart-container">{oom_html}</div>'

    html += """
<p style="color: #8b8fa3; font-size: 12px; margin-top: 32px;">
Generated by GGUF Quant Explorer
</p>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return html
