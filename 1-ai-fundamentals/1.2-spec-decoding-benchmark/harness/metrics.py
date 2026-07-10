"""
BenchmarkMetrics — statistical analysis of speculative decoding runs.

Metrics calculated:
  - Acceptance rate:        fraction of draft tokens accepted
  - Tokens per second:      generation throughput
  - Latency variance:       p50/p95/p99 generation latency
  - Draft efficiency:       accepted tokens per draft model forward pass
  - Verification ratio:     target model forward passes vs total tokens
  - Speedup over baseline:  tok/s relative to no-speculation baseline
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkMetrics:
    """Computed metrics for a single strategy run."""

    strategy_label: str
    prompt_name: str

    # ── Raw counts (set from completion data) ──
    total_tokens_generated: int = 0
    total_draft_tokens: int = 0
    total_accepted_draft_tokens: int = 0
    total_verified_tokens: int = 0
    num_target_forward_passes: int = 0
    num_draft_forward_passes: int = 0

    # ── Timing (seconds) ──
    wall_time_s: float = 0.0
    prompt_eval_time_s: float = 0.0
    token_timestamps_s: List[float] = field(default_factory=list)  # per-token

    # ── Config ──
    draft_n_max: int = 5
    config_flags: str = ""

    # ── Derived metrics ──

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted by the target model."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_draft_tokens / self.total_draft_tokens

    @property
    def tokens_per_second(self) -> float:
        """Overall generation throughput (including prompt eval)."""
        if self.wall_time_s == 0:
            return 0.0
        return self.total_tokens_generated / self.wall_time_s

    @property
    def tokens_per_second_excluding_prompt(self) -> float:
        """Generation throughput excluding prompt evaluation time."""
        gen_time = self.wall_time_s - self.prompt_eval_time_s
        if gen_time <= 0:
            return self.tokens_per_second
        return self.total_tokens_generated / gen_time

    @property
    def draft_efficiency(self) -> float:
        """Accepted draft tokens per draft model forward pass."""
        if self.num_draft_forward_passes == 0:
            return 0.0
        return self.total_accepted_draft_tokens / self.num_draft_forward_passes

    @property
    def verification_ratio(self) -> float:
        """Target forward passes per generated token (lower is better)."""
        if self.total_tokens_generated == 0:
            return 0.0
        return self.num_target_forward_passes / self.total_tokens_generated

    @property
    def draft_proportion(self) -> float:
        """Fraction of generated tokens that came from draft acceptance."""
        if self.total_tokens_generated == 0:
            return 0.0
        return self.total_accepted_draft_tokens / self.total_tokens_generated

    @property
    def avg_draft_length(self) -> float:
        """Average number of draft tokens per verification step."""
        if self.num_target_forward_passes == 0:
            return 0.0
        return self.total_draft_tokens / self.num_target_forward_passes

    # ── Latency percentiles ──

    def latency_percentile(self, p: float) -> float:
        """Compute the p-th percentile of per-token latencies."""
        if len(self.token_timestamps_s) < 2:
            return 0.0
        # Convert timestamps to inter-token latencies
        intervals = [
            self.token_timestamps_s[i + 1] - self.token_timestamps_s[i]
            for i in range(len(self.token_timestamps_s) - 1)
        ]
        if not intervals:
            return 0.0
        sorted_ints = sorted(intervals)
        idx = int(math.ceil(p / 100.0 * len(sorted_ints))) - 1
        idx = max(0, min(idx, len(sorted_ints) - 1))
        return sorted_ints[idx]

    @property
    def latency_p50_ms(self) -> float:
        return self.latency_percentile(50) * 1000

    @property
    def latency_p95_ms(self) -> float:
        return self.latency_percentile(95) * 1000

    @property
    def latency_p99_ms(self) -> float:
        return self.latency_percentile(99) * 1000

    @property
    def latency_jitter(self) -> float:
        """Coefficient of variation of inter-token latencies."""
        intervals = [
            self.token_timestamps_s[i + 1] - self.token_timestamps_s[i]
            for i in range(len(self.token_timestamps_s) - 1)
        ]
        if len(intervals) < 2:
            return 0.0
        mean = statistics.mean(intervals)
        if mean == 0:
            return 0.0
        stdev = statistics.stdev(intervals)
        return stdev / mean

    # ── Serialisation ──

    def as_dict(self) -> Dict:
        return {
            "strategy": self.strategy_label,
            "prompt": self.prompt_name,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "tokens_per_second_gen": round(self.tokens_per_second_excluding_prompt, 2),
            "draft_efficiency": round(self.draft_efficiency, 4),
            "verification_ratio": round(self.verification_ratio, 4),
            "draft_proportion": round(self.draft_proportion, 4),
            "avg_draft_length": round(self.avg_draft_length, 2),
            "total_tokens": self.total_tokens_generated,
            "total_draft": self.total_draft_tokens,
            "total_accepted": self.total_accepted_draft_tokens,
            "wall_time_s": round(self.wall_time_s, 3),
            "prompt_eval_time_s": round(self.prompt_eval_time_s, 3),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_p99_ms": round(self.latency_p99_ms, 2),
            "latency_jitter": round(self.latency_jitter, 4),
            "draft_n_max": self.draft_n_max,
            "config_flags": self.config_flags,
        }
