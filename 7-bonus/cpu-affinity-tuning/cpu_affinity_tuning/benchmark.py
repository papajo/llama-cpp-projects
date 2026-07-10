"""Simulated benchmark harness for comparing pinning strategies.

On a real Linux deployment you would point this at a llama.cpp executable.
On macOS (or without a real binary) it runs a synthetic CPU-bound workload
to demonstrate the relative throughput of each pinning strategy.
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .strategies import PinningPlan
from .topology import CpuTopology


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    strategy: str
    worker_count: int
    duration: float  # seconds
    throughput: float  # tokens / second (simulated)
    cv: float = 0.0  # coefficient of variation among workers

    def summary(self) -> str:
        return (
            f"  {self.strategy:12s}  {self.worker_count:3d} workers  "
            f"{self.duration:8.3f}s  {self.throughput:10.1f} t/s  "
            f"CV={self.cv:.3f}"
        )


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results across strategies."""
    topology: CpuTopology
    results: List[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def best(self) -> BenchmarkResult:
        """Return the result with highest throughput."""
        return max(self.results, key=lambda r: r.throughput)

    def report(self) -> str:
        lines = [
            "=" * 72,
            "CPU Affinity Benchmark Report",
            "=" * 72,
            f"\nTopology:\n{self.topology.summary()}",
            f"\n{'Strategy':12s}  {'Workers':>4s}  {'Duration':>8s}  {'Throughput':>10s}  {'CV':>6s}",
            "-" * 72,
        ]
        for r in sorted(self.results, key=lambda x: -x.throughput):
            lines.append(r.summary())
        lines.append("-" * 72)
        best = self.best()
        lines.append(f"\n🏆 Best: {best.strategy} @ {best.throughput:.1f} t/s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulated benchmark runner
# ---------------------------------------------------------------------------

def _simulate_inference(
    plan: PinningPlan,
    *,
    total_tokens: int = 500,
    base_ms_per_token: float = 50.0,
    noise: float = 0.05,
) -> List[float]:
    """Simulate inference throughput per worker given a pinning plan.

    Compact workers (same package) share cache → lower latency but
    may contend for memory bandwidth.  Spread workers have higher
    single-worker latency but better aggregate throughput.

    Returns list of duration estimates (in seconds) for each worker.
    """
    durations: List[float] = []
    num_workers = len(plan.logical_cpus)

    for cpus in plan.logical_cpus:
        # Compact → fewer packages involved → higher contention
        unique_packages = _count_packages(cpus)
        if unique_packages <= 1:
            # Same package: cache friendly, but BW contention
            latency = base_ms_per_token * (1.0 + 0.10 * math.sqrt(num_workers))
        else:
            # Spread across packages: less contention, higher latency
            latency = base_ms_per_token * (1.0 + 0.02 * num_workers)

        durations.append(total_tokens * latency / 1000.0)

    return durations


def _count_packages(cpus: List[int]) -> int:
    """Dummy — in a real benchmark we'd look up topology."""
    return len(set(c // 4 for c in cpus))  # crude heuristic


def _coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def benchmark_plans(
    plans: Dict[str, PinningPlan],
    topology: CpuTopology,
    *,
    total_tokens: int = 500,
    base_ms_per_token: float = 50.0,
) -> BenchmarkSuite:
    """Run simulated benchmarks for a set of pinning plans.

    Args:
        plans: ``{name: PinningPlan}`` returned by ``generate_plans()``.
        topology: CPU topology (used for report only).
        total_tokens: Number of tokens in the simulated workload.
        base_ms_per_token: Base latency per token in ms.

    Returns:
        A BenchmarkSuite with results for every plan.
    """
    suite = BenchmarkSuite(topology=topology)

    for sname, plan in plans.items():
        worker_durations = _simulate_inference(
            plan,
            total_tokens=total_tokens,
            base_ms_per_token=base_ms_per_token,
        )
        avg_duration = sum(worker_durations) / len(worker_durations)
        cv = _coefficient_of_variation(worker_durations)
        throughput = (total_tokens * len(worker_durations)) / avg_duration

        suite.add(BenchmarkResult(
            strategy=sname,
            worker_count=len(plan.logical_cpus),
            duration=avg_duration,
            throughput=throughput,
            cv=cv,
        ))

    return suite
