"""
Speculative Decoding Benchmark Suite — harness for benchmarking
all 8 speculative decoding strategies in llama.cpp.

Exposes:
  - SpecStrategy, StrategyConfig — strategy definitions
  - BenchmarkRunner — subprocess manager
  - BenchmarkMetrics — accept rate, tok/s, latency
  - run_benchmark() — single-entry convenience
"""
from .configs import SpecStrategy, StrategyConfig, ALL_STRATEGIES
from .runner import BenchmarkRunner, BenchmarkResult
from .metrics import BenchmarkMetrics

__all__ = [
    "SpecStrategy",
    "StrategyConfig",
    "ALL_STRATEGIES",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkMetrics",
]
