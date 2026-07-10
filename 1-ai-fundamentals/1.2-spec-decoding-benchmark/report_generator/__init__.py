"""
Report generator for speculative decoding benchmark results.

Produces:
  - Comparative charts (Plotly HTML)
  - Summary tables (CSV)
  - Full benchmark report (Markdown)
  - Strategy recommendation based on workload type
"""

from .analysis import BenchmarkAnalysis
from .plots import (
    plot_acceptance_rate,
    plot_tokens_per_second,
    plot_strategy_comparison,
    plot_latency_profile,
    plot_draft_efficiency,
    plot_workload_strategy_heatmap,
)
from .report import generate_report

__all__ = [
    "BenchmarkAnalysis",
    "plot_acceptance_rate",
    "plot_tokens_per_second",
    "plot_strategy_comparison",
    "plot_latency_profile",
    "plot_draft_efficiency",
    "plot_workload_strategy_heatmap",
    "generate_report",
]
