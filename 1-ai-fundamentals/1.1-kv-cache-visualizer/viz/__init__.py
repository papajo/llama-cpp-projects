"""
viz — Plotly-based chart generators for KV cache visualizations.

Produces standalone HTML charts that can be embedded in a web UI
or exported as PNG/PDF.
"""

from .plots import (
    plot_memory_scaling,
    plot_quant_comparison,
    plot_oom_frontier,
    plot_model_comparison,
    plot_slot_allocation,
)

__all__ = [
    "plot_memory_scaling",
    "plot_quant_comparison",
    "plot_oom_frontier",
    "plot_model_comparison",
    "plot_slot_allocation",
]
