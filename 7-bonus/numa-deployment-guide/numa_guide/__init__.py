"""NUMA-aware deployment guide for llama.cpp."""

from .deployment import (
    MemoryPolicy,
    NumaDeploymentPlan,
    recommend_plan,
)
from .topology import NumaNode, NumaTopology, detect_numa_topology

__all__ = [
    "NumaNode",
    "NumaTopology",
    "NumaDeploymentPlan",
    "MemoryPolicy",
    "detect_numa_topology",
    "recommend_plan",
]
