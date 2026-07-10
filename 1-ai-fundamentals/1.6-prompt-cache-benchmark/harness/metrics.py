"""
Metrics for cache benchmarking.

Measures:
  - Time-to-first-token (TTFT) with and without cache
  - Prompt processing speed (tokens/s)
  - Cache hit ratio (estimated)
  - Speedup factor
  - Memory savings estimate
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptResult:
    """Results for a single prompt."""
    prompt_index: int
    prompt_length_chars: int
    ttft_ms: float               # Time-to-first-token in ms
    prompt_tokens: int = 0       # Approximate prompt token count
    prompt_processing_ms: float = 0.0
    predicted_per_second: float = 0.0  # Generation speed
    cached: bool = False         # Whether this run used cache


@dataclass
class CacheBenchmarkResult:
    """Aggregated results from a cache benchmark run."""
    scenario_name: str
    description: str = ""

    # All runs
    runs_cached: List[PromptResult] = field(default_factory=list)
    runs_uncached: List[PromptResult] = field(default_factory=list)

    # Aggregated metrics
    avg_ttft_cached_ms: float = 0.0
    avg_ttft_uncached_ms: float = 0.0
    median_ttft_cached_ms: float = 0.0
    median_ttft_uncached_ms: float = 0.0
    speedup_factor: float = 0.0
    speedup_pct: float = 0.0

    # Per-prompt analysis
    per_prompt: List[Dict] = field(default_factory=list)

    # Timing breakdown
    cached_timing: Dict[str, float] = field(default_factory=dict)
    uncached_timing: Dict[str, float] = field(default_factory=dict)

    def compute(self):
        """Compute aggregated metrics from raw results."""
        if self.runs_cached:
            cached_ttfts = [r.ttft_ms for r in self.runs_cached]
            self.avg_ttft_cached_ms = statistics.mean(cached_ttfts)
            self.median_ttft_cached_ms = statistics.median(cached_ttfts)

        if self.runs_uncached:
            uncached_ttfts = [r.ttft_ms for r in self.runs_uncached]
            self.avg_ttft_uncached_ms = statistics.mean(uncached_ttfts)
            self.median_ttft_uncached_ms = statistics.median(uncached_ttfts)

        if self.avg_ttft_uncached_ms > 0:
            self.speedup_factor = (
                self.avg_ttft_uncached_ms / max(self.avg_ttft_cached_ms, 0.001)
            )
            self.speedup_pct = (
                (self.avg_ttft_uncached_ms - self.avg_ttft_cached_ms)
                / self.avg_ttft_uncached_ms * 100
            )

        # Per-prompt comparison
        self.per_prompt = []
        min_len = min(len(self.runs_cached), len(self.runs_uncached))
        for i in range(min_len):
            c = self.runs_cached[i]
            u = self.runs_uncached[i]
            saving_ms = u.ttft_ms - c.ttft_ms
            self.per_prompt.append({
                "index": i,
                "prompt_length": c.prompt_length_chars,
                "ttft_cached_ms": round(c.ttft_ms, 2),
                "ttft_uncached_ms": round(u.ttft_ms, 2),
                "saving_ms": round(saving_ms, 2),
                "speedup": round(u.ttft_ms / max(c.ttft_ms, 0.001), 2),
            })

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "description": self.description,
            "num_runs_cached": len(self.runs_cached),
            "num_runs_uncached": len(self.runs_uncached),
            "avg_ttft_cached_ms": round(self.avg_ttft_cached_ms, 2),
            "avg_ttft_uncached_ms": round(self.avg_ttft_uncached_ms, 2),
            "median_ttft_cached_ms": round(self.median_ttft_cached_ms, 2),
            "median_ttft_uncached_ms": round(self.median_ttft_uncached_ms, 2),
            "speedup_factor": round(self.speedup_factor, 2),
            "speedup_pct": round(self.speedup_pct, 1),
            "per_prompt": self.per_prompt,
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)
