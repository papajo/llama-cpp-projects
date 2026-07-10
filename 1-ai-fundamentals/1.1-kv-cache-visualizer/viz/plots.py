"""
Plotly chart generators for KV Cache Visualizer.

Each function returns a plotly.graph_objects.Figure ready for
display (fig.show()), HTML export (fig.to_html()), or JSON serialization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from calculator.models import KVQuant, ModelConfig
from calculator.calculator import KVCacheCalculator


# ── Color palette ─────────────────────────────────────────────────

COLORS = {
    "f16": "#1f77b4",     # blue
    "q8_0": "#2ca02c",    # green
    "q4_0": "#d62728",    # red
    "bg": "#f8f9fa",
    "grid": "#e9ecef",
}

QUANT_LABELS = {
    "f16": "f16 (2 B/elem)",
    "q8_0": "q8_0 (1 B/elem)",
    "q4_0": "q4_0 (0.5 B/elem)",
}

THEME_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="SF Mono, Menlo, Consolas, monospace", size=13),
    hovermode="x unified",
    margin=dict(l=60, r=40, t=60, b=60),
)


# ── Individual plots ──────────────────────────────────────────────


def plot_memory_scaling(
    calculator: KVCacheCalculator,
    max_ctx: Optional[int] = None,
    quants: Optional[List[KVQuant]] = None,
) -> go.Figure:
    """
    Line chart: KV cache memory (MiB) vs context length for each quant type.

    The primary chart — shows the linear scaling of KV cache with context.
    """
    if max_ctx is None:
        max_ctx = calculator.config.max_ctx
    if quants is None:
        quants = KVQuant.all()

    result = calculator.calculate(max_ctx=max_ctx, quants=quants)

    fig = go.Figure(layout=THEME_LAYOUT)
    fig.update_layout(
        title=dict(
            text=f"<b>KV Cache Memory vs Context Length</b><br>"
                 f"<sup>{calculator.config.name} · {calculator.n_slots} slot(s) · "
                 f"per-slot allocation</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="Context Length (tokens)", font=dict(size=14)),
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Memory (MiB)", font=dict(size=14)),
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        legend=dict(title=dict(text="KV Quantization")),
    )

    for q in quants:
        xs = [p["ctx"] for p in result.scaling]
        ys = [p[f"{q.label}_MiB"] for p in result.scaling]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=QUANT_LABELS[q.label],
            line=dict(color=COLORS[q.label], width=2.5),
            hovertemplate="ctx=%{x:,} · %{y:,.1f} MiB<extra></extra>",
        ))

    # Annotation for max ctx
    fig.add_vline(
        x=max_ctx,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text=f" max ctx: {max_ctx:,}",
        annotation_position="top right",
    )

    return fig


def plot_quant_comparison(
    calculator: KVCacheCalculator,
    ctx: Optional[int] = None,
    quants: Optional[List[KVQuant]] = None,
) -> go.Figure:
    """
    Bar chart: memory at a specific ctx length, grouped by quant type.

    Breaks down per-layer, total, and all-slots memory.
    """
    if ctx is None:
        ctx = calculator.config.max_ctx
    if quants is None:
        quants = KVQuant.all()

    result = calculator.calculate(max_ctx=ctx, quants=quants)

    categories = ["Per Layer", "Total (1 slot)", f"Total ({calculator.n_slots} slots)"]
    fig = go.Figure(layout=THEME_LAYOUT)

    for q in quants:
        bd = result.breakdowns[q.label]
        values = [bd.miB_per_layer, bd.miB_total, bd.miB_total_all_slots]
        fig.add_trace(go.Bar(
            name=QUANT_LABELS[q.label],
            x=categories,
            y=values,
            marker_color=COLORS[q.label],
            hovertemplate="%{y:,.1f} MiB<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>KV Cache Memory Breakdown at {ctx:,} tokens</b><br>"
                 f"<sup>{calculator.config.name}</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(title=dict(text="Scope", font=dict(size=14))),
        yaxis=dict(
            title=dict(text="Memory (MiB)", font=dict(size=14)),
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        barmode="group",
        legend=dict(title=dict(text="KV Quantization")),
    )

    return fig


def plot_oom_frontier(
    calculator: KVCacheCalculator,
    vram_range: Tuple[float, float, float] = (4, 48, 5),
    quants: Optional[List[KVQuant]] = None,
    model_weights_gb: Optional[float] = None,
) -> go.Figure:
    """
    Line chart: max context length vs available VRAM for each quant.

    Shows "how much context can I fit with my GPU?"
    """
    if quants is None:
        quants = KVQuant.all()

    vram_values = [vram_range[0] + i * vram_range[2]
                   for i in range(int((vram_range[1] - vram_range[0]) / vram_range[2]) + 1)]

    fig = go.Figure(layout=THEME_LAYOUT)

    for q in quants:
        max_ctx_values = []
        for vram in vram_values:
            results = calculator.predict_oom(
                vram_gb=vram,
                quants=[q],
                model_weights_gb=model_weights_gb,
            )
            max_ctx_values.append(results[0]["max_ctx_tokens"])

        fig.add_trace(go.Scatter(
            x=vram_values,
            y=max_ctx_values,
            mode="lines+markers",
            name=QUANT_LABELS[q.label],
            line=dict(color=COLORS[q.label], width=2.5),
            hovertemplate="VRAM=%{x:.1f} GB · Max ctx=%{y:,}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>OOM Frontier: Max Context vs Available VRAM</b><br>"
                 f"<sup>{calculator.config.name} · {calculator.n_slots} slot(s)</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="Available VRAM (GB)", font=dict(size=14)),
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Max Context Length (tokens)", font=dict(size=14)),
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        legend=dict(title=dict(text="KV Quantization")),
    )

    return fig


def plot_model_comparison(
    configs: List[ModelConfig],
    ctx: Optional[int] = None,
    quant: KVQuant = KVQuant.F16,
) -> go.Figure:
    """
    Bar chart: compare KV cache memory across multiple model architectures
    at a fixed context length.
    """
    if ctx is None:
        # Use the smallest max_ctx among all configs
        ctx = min(c.max_ctx for c in configs)

    fig = go.Figure(layout=THEME_LAYOUT)

    model_names = []
    mem_values = []

    for cfg in configs:
        calc = KVCacheCalculator(cfg, n_slots=1)
        # Cap at model's actual max if smaller
        actual_ctx = min(ctx, cfg.max_ctx)
        result = calc.calculate(max_ctx=actual_ctx, quants=[quant])
        bd = result.breakdowns[quant.label]
        model_names.append(cfg.name)
        mem_values.append(bd.gb_total)

    fig.add_trace(go.Bar(
        x=model_names,
        y=mem_values,
        marker_color=[COLORS[quant.label]] * len(model_names),
        hovertemplate="%{y:.2f} GB<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>KV Cache Memory Comparison at {ctx:,} tokens</b><br>"
                 f"<sup>{quant.label.upper()} quantization</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="Model", font=dict(size=14)),
            tickangle=45,
        ),
        yaxis=dict(
            title=dict(text="KV Cache (GB)", font=dict(size=14)),
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
    )

    return fig


def plot_slot_allocation(
    calculator: KVCacheCalculator,
    max_slots: int = 8,
    ctx: Optional[int] = None,
    quant: KVQuant = KVQuant.F16,
) -> go.Figure:
    """
    Line chart: total KV cache memory vs number of slots.

    Compares per-slot vs unified allocation strategies.
    """
    if ctx is None:
        ctx = calculator.config.max_ctx

    fig = go.Figure(layout=THEME_LAYOUT)

    slots_range = list(range(1, max_slots + 1))

    per_slot_mem = []
    unified_mem = []

    for n_slots in slots_range:
        calc = KVCacheCalculator(calculator.config, n_slots=n_slots)
        # Per-slot
        result_ps = calc.calculate(max_ctx=ctx, quants=[quant], allocation="per-slot")
        per_slot_mem.append(result_ps.breakdowns[quant.label].gb_total)

        # Unified (approximated as shared pool)
        result_uni = calc.calculate(max_ctx=ctx, quants=[quant], allocation="unified")
        unified_mem.append(result_uni.breakdowns[quant.label].gb_total)

    fig.add_trace(go.Scatter(
        x=slots_range,
        y=per_slot_mem,
        mode="lines+markers",
        name=f"Per-slot ({quant.label})",
        line=dict(color=COLORS[quant.label], width=2.5),
        hovertemplate="Slots=%{x} · %{y:.2f} GB<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=slots_range,
        y=unified_mem,
        mode="lines+markers",
        name=f"Unified ({quant.label})",
        line=dict(color=COLORS["q8_0"], width=2.5, dash="dash"),
        hovertemplate="Slots=%{x} · %{y:.2f} GB<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>KV Cache Memory vs Number of Slots<br>"
                 f"<sup>{calculator.config.name} at {ctx:,} tokens · "
                 f"Per-slot vs Unified allocation</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="Number of Slots", font=dict(size=14)),
            tickmode="linear",
            tick0=1,
            dtick=1,
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="KV Cache (GB)", font=dict(size=14)),
            gridcolor=COLORS["grid"],
            showgrid=True,
        ),
        legend=dict(title=dict(text="Allocation")),
    )

    return fig


# ── Multi-panel dashboard ─────────────────────────────────────────


def build_dashboard(
    calculator: KVCacheCalculator,
    max_ctx: Optional[int] = None,
) -> str:
    """
    Build an HTML dashboard with multiple charts side-by-side.

    Returns the full HTML string for the dashboard.
    """
    from plotly.io import to_html

    if max_ctx is None:
        max_ctx = calculator.config.max_ctx

    fig_scaling = plot_memory_scaling(calculator, max_ctx)
    fig_breakdown = plot_quant_comparison(calculator, max_ctx)
    fig_oom = plot_oom_frontier(calculator)
    fig_slots = plot_slot_allocation(calculator)

    html_charts = "\n".join([
        to_html(fig_scaling, include_plotlyjs=False, full_html=False),
        to_html(fig_breakdown, include_plotlyjs=False, full_html=False),
        to_html(fig_oom, include_plotlyjs=False, full_html=False),
        to_html(fig_slots, include_plotlyjs=False, full_html=False),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KV Cache Dashboard — {calculator.config.name}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8f9fa;
    color: #333;
    padding: 24px;
}}
h1 {{
    font-size: 28px;
    margin-bottom: 4px;
}}
.subtitle {{
    color: #666;
    margin-bottom: 24px;
    font-size: 14px;
}}
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}}
.stat-card {{
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.stat-card .label {{
    font-size: 12px;
    text-transform: uppercase;
    color: #666;
    letter-spacing: 0.5px;
}}
.stat-card .value {{
    font-size: 24px;
    font-weight: 700;
    margin-top: 4px;
}}
.chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}
.chart-card {{
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.chart-card.full-width {{
    grid-column: 1 / -1;
}}
@media (max-width: 900px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<h1>🔬 KV Cache Dashboard</h1>
<p class="subtitle">{calculator.config.name} · {max_ctx:,} max ctx · {calculator.n_slots} slot(s)</p>

<div class="stats-grid">
    <div class="stat-card">
        <div class="label">Layers</div>
        <div class="value">{calculator.config.n_layers}</div>
    </div>
    <div class="stat-card">
        <div class="label">KV Heads</div>
        <div class="value">{calculator.config.n_kv_heads}</div>
    </div>
    <div class="stat-card">
        <div class="label">Head Dim</div>
        <div class="value">{calculator.config.head_dim}</div>
    </div>
    <div class="stat-card">
        <div class="label">Max Context</div>
        <div class="value">{max_ctx:,}</div>
    </div>
    <div class="stat-card">
        <div class="label">KV f16 (1 slot)</div>
        <div class="value">{calculator.bytes_at_ctx(KVQuant.F16, max_ctx) / (1024**3):.1f} GB</div>
    </div>
    <div class="stat-card">
        <div class="label">KV q4_0 (1 slot)</div>
        <div class="value">{calculator.bytes_at_ctx(KVQuant.Q4_0, max_ctx) / (1024**3):.1f} GB</div>
    </div>
</div>

<div class="chart-grid">
{html_charts}
</div>
</body>
</html>"""
    return html
