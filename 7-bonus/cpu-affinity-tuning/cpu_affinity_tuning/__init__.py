"""CPU affinity tuning tools for llama.cpp inference workers."""

from .benchmark import BenchmarkResult, BenchmarkSuite, benchmark_plans
from .launcher import generate_launch_command, generate_launch_script
from .strategies import PinningPlan, generate_plans
from .topology import CpuTopology, detect_topology, topology_to_json

__all__ = [
    "CpuTopology",
    "PinningPlan",
    "BenchmarkResult",
    "BenchmarkSuite",
    "detect_topology",
    "topology_to_json",
    "generate_plans",
    "benchmark_plans",
    "generate_launch_command",
    "generate_launch_script",
]
