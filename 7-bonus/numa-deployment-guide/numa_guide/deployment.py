"""Memory policies and numactl command generation for NUMA-aware deployment."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .topology import NumaTopology


class MemoryPolicy(Enum):
    LOCAL = "localalloc"
    PREFERRED = "preferred"
    BIND = "bind"
    INTERLEAVE = "interleave"


@dataclass
class NumaDeploymentPlan:
    """A NUMA-aware deployment plan for llama.cpp inference workers."""
    worker_count: int
    memory_policy: MemoryPolicy
    policy_args: List[int]  # node IDs for bind/preferred/interleave
    node_assignments: Dict[int, List[int]]  # node_id → [worker indices]
    cpu_assignments: Dict[int, List[int]]  # worker_idx → [cpu_ids]
    description: str = ""

    def to_numactl_cmds(
        self,
        binary: str = "./llama-cli",
        model: str = "models/llama-7b.gguf",
        extra_args: str = "",
    ) -> List[str]:
        """Generate numactl commands for each worker."""
        system = platform.system()
        cmds: List[str] = []

        for wid in range(self.worker_count):
            cpus = self.cpu_assignments.get(wid, [])
            cpu_list = ",".join(map(str, cpus)) if cpus else ""

            if system == "Linux":
                policy_flag = self.memory_policy.value
                if self.memory_policy == MemoryPolicy.LOCAL or not self.policy_args:
                    policy_str = f"--{policy_flag}"
                elif self.memory_policy == MemoryPolicy.INTERLEAVE:
                    nodes = ",".join(map(str, self.policy_args))
                    policy_str = f"--interleave={nodes}"
                else:
                    nodes = ",".join(map(str, self.policy_args))
                    policy_str = f"--{policy_flag}={nodes}"

                affinity = f"--physcpubind={cpu_list}" if cpu_list else ""
                cmd = f"numactl {affinity} {policy_str} {binary} --model {model} --id {wid} {extra_args}"
            elif system == "Darwin":
                cmd = (
                    f"# macOS ({platform.machine()}): no numactl available\n"
                    f"# Suggested: pin worker {wid} to CPUs [{cpu_list}] via thread_policy\n"
                    f"{binary} --model {model} --id {wid} {extra_args}"
                )
            else:
                cmd = (
                    f"# Worker {wid} — NUMA policy: {self.memory_policy.value} "
                    f"nodes={self.policy_args}\n"
                    f"{binary} --model {model} --id {wid} {extra_args}"
                )
            cmds.append(cmd)

        return cmds

    def summary(self) -> str:
        lines = [
            f"NUMA Deployment Plan",
            f"  Workers: {self.worker_count}",
            f"  Memory policy: {self.memory_policy.value}",
            f"  Target nodes: {self.policy_args}",
            f"  Description: {self.description}",
        ]
        for node_id, workers in self.node_assignments.items():
            lines.append(f"  Node {node_id} → workers {workers}")
        for wid, cpus in self.cpu_assignments.items():
            lines.append(f"  Worker {wid} → CPUs {cpus}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

def recommend_plan(
    topology: NumaTopology,
    worker_count: int,
    policy: MemoryPolicy = MemoryPolicy.BIND,
) -> NumaDeploymentPlan:
    """Automatically recommend a NUMA deployment plan.

    Args:
        topology: Detected NUMA topology.
        worker_count: Number of inference workers.
        policy: Memory allocation policy.

    Strategy:
        - If 1 NUMA node: bind to all CPUs, no special NUMA handling.
        - If multiple nodes: distribute workers round-robin across nodes,
          bind each worker's memory to its assigned node.
    """
    if not topology.has_numa or topology.node_count <= 1:
        cpus = topology.nodes[0].cpu_ids if topology.nodes else []
        return NumaDeploymentPlan(
            worker_count=worker_count,
            memory_policy=MemoryPolicy.LOCAL,
            policy_args=[],
            node_assignments={0: list(range(worker_count))},
            cpu_assignments={i: cpus for i in range(worker_count)},
            description="Single NUMA node — local allocation.",
        )

    node_ids = sorted(n.node_id for n in topology.nodes)
    node_assignments: Dict[int, List[int]] = {n: [] for n in node_ids}
    cpu_assignments: Dict[int, List[int]] = {}

    for wi in range(worker_count):
        target_node = node_ids[wi % len(node_ids)]
        node_assignments[target_node].append(wi)
        # Assign all CPUs from this node to the worker
        node = next(n for n in topology.nodes if n.node_id == target_node)
        cpu_assignments[wi] = node.cpu_ids

    return NumaDeploymentPlan(
        worker_count=worker_count,
        memory_policy=policy,
        policy_args=node_ids if policy != MemoryPolicy.LOCAL else [],
        node_assignments=node_assignments,
        cpu_assignments=cpu_assignments,
        description=f"Distribute {worker_count} workers across {len(node_ids)} NUMA nodes with {policy.value} memory policy.",
    )
