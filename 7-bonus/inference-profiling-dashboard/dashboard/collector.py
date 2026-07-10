"""Inference profiling data collector — simulates real-time metrics.

In production this would scrape llama.cpp's ``/metrics`` endpoint or
read from Prometheus.  Here we generate realistic synthetic data for
the dashboard demo.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Snapshot:
    """A single point-in-time measurement."""
    timestamp: float
    tokens_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_percent: float
    memory_mb: float
    batch_size: int
    prompt_processing_speed: float  # tokens/s for prompt eval phase
    gpu_util_percent: float = 0.0
    kv_cache_usage_percent: float = 0.0


@dataclass
class ProfileRun:
    """A collection of snapshots from a single inference run."""
    label: str  # e.g. "compact-4", "spread-4"
    snapshots: List[Snapshot] = field(default_factory=list)

    def add(self, snap: Snapshot) -> None:
        self.snapshots.append(snap)

    @property
    def avg_throughput(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s.tokens_per_second for s in self.snapshots) / len(self.snapshots)

    @property
    def p95_latency(self) -> float:
        if not self.snapshots:
            return 0.0
        sorted_lat = sorted(s.latency_p95_ms for s in self.snapshots)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _gen_profile_run(
    label: str,
    n_points: int,
    *,
    base_tps: float = 50.0,
    base_latency: float = 20.0,
    jitter: float = 0.15,
    trend: str = "stable",
) -> ProfileRun:
    """Generate a profile run with synthetic data.

    Args:
        label: Name for this run.
        n_points: Number of data points.
        base_tps: Baseline tokens/second.
        base_latency: Baseline p50 latency in ms.
        jitter: Random noise fraction (0-1).
        trend: ``stable``, ``degrading`` (memory leak sim), or ``improving``.
    """
    run = ProfileRun(label=label)
    for i in range(n_points):
        t = time.time() + i * 2  # 2-second intervals

        if trend == "degrading":
            decay = 1.0 - 0.3 * (i / n_points)
            tps = base_tps * decay * random.gauss(1.0, jitter)
            lat = base_latency / decay * random.gauss(1.0, jitter)
        elif trend == "improving":
            gain = 1.0 + 0.3 * (i / n_points)
            tps = base_tps * gain * random.gauss(1.0, jitter)
            lat = base_latency / gain * random.gauss(1.0, jitter)
        else:
            tps = base_tps * random.gauss(1.0, jitter)
            lat = base_latency * random.gauss(1.0, jitter)

        tps = max(tps, 1.0)
        lat = max(lat, 1.0)

        snap = Snapshot(
            timestamp=t,
            tokens_per_second=round(tps, 1),
            latency_p50_ms=round(lat, 1),
            latency_p95_ms=round(lat * random.uniform(1.5, 3.0), 1),
            latency_p99_ms=round(lat * random.uniform(2.0, 5.0), 1),
            cpu_percent=round(random.uniform(30, 95), 1),
            memory_mb=round(random.uniform(2000, 8000), 1),
            batch_size=random.choice([1, 1, 1, 2, 4, 8]),
            prompt_processing_speed=round(base_tps * 3 * random.gauss(1.0, 0.1), 1),
            gpu_util_percent=round(random.uniform(40, 99), 1),
            kv_cache_usage_percent=round(random.uniform(20, 80), 1),
        )
        run.add(snap)

    return run


def generate_sample_data() -> Dict[str, ProfileRun]:
    """Generate a realistic set of profile runs for the dashboard."""
    return {
        "compact-4": _gen_profile_run("compact-4", 60, base_tps=45.0, base_latency=22.0, trend="stable"),
        "spread-4": _gen_profile_run("spread-4", 60, base_tps=58.0, base_latency=18.0, trend="stable"),
        "hybrid-4": _gen_profile_run("hybrid-4", 60, base_tps=52.0, base_latency=20.0, trend="stable"),
        "no-pin-4": _gen_profile_run("no-pin-4", 60, base_tps=38.0, base_latency=28.0, trend="degrading"),
    }


def run_to_dict(run: ProfileRun) -> dict:
    """Convert a ProfileRun to a JSON-serializable dict."""
    return {
        "label": run.label,
        "avg_throughput": round(run.avg_throughput, 1),
        "p95_latency": round(run.p95_latency, 1),
        "snapshots": [
            {
                "timestamp": s.timestamp,
                "tokens_per_second": s.tokens_per_second,
                "latency_p50_ms": s.latency_p50_ms,
                "latency_p95_ms": s.latency_p95_ms,
                "latency_p99_ms": s.latency_p99_ms,
                "cpu_percent": s.cpu_percent,
                "memory_mb": s.memory_mb,
                "batch_size": s.batch_size,
                "prompt_processing_speed": s.prompt_processing_speed,
                "gpu_util_percent": s.gpu_util_percent,
                "kv_cache_usage_percent": s.kv_cache_usage_percent,
            }
            for s in run.snapshots
        ],
    }


def all_runs_to_dict(runs: Dict[str, ProfileRun]) -> dict:
    """Convert all runs to a JSON-serializable dict keyed by name."""
    return {name: run_to_dict(r) for name, r in runs.items()}
