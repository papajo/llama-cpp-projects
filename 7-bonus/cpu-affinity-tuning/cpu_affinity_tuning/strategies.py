"""CPU pinning strategies for llama.cpp inference workers.

The module is cross-platform: on Linux it produces `taskset` commands;
on macOS it prints advisory config (macOS does not expose a taskset equivalent
to userland, but the topology report helps the user pin via thread_policy or
launchd config).  All strategies work on the *logical* CPU IDs reported by
:func:`~cpu_affinity_tuning.topology.detect_topology`.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .topology import CpuTopology


@dataclass
class PinningPlan:
    """A suggested CPU pinning plan for a given worker count."""
    worker_count: int
    strategy_name: str
    logical_cpus: List[List[int]]  # per-worker list of logical CPU IDs
    description: str = ""

    def to_taskset_cmds(self, binary: str = "./llama-cli", args: str = "") -> List[str]:
        """Generate taskset commands for Linux deployment."""
        system = platform.system()
        if system == "Linux":
            cmds = []
            for i, cpus in enumerate(self.logical_cpus):
                mask = ",".join(map(str, cpus))
                cmds.append(f"taskset -c {mask} {binary} --id {i} {args}")
            return cmds
        else:
            return self._advisory_cmds(binary, args)

    def _advisory_cmds(self, binary: str = "./llama-cli", args: str = "") -> List[str]:
        """Advisory output for non-Linux systems."""
        cmds = []
        for i, cpus in enumerate(self.logical_cpus):
            cmds.append(
                f"# Worker {i}: pin to CPUs {cpus}  (platform does not support taskset)"
            )
            cmds.append(f"{binary} --id {i} {args}")
        return cmds

    def summary(self) -> str:
        lines = [
            f"Strategy: {self.strategy_name}",
            f"Workers:  {self.worker_count}",
            f"Description: {self.description}",
            "CPU assignments per worker:",
        ]
        for i, cpus in enumerate(self.logical_cpus):
            lines.append(f"  Worker {i}: CPUs {cpus}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

PinningFn = Callable[[CpuTopology, int], PinningPlan]


def strategy_compact(topology: CpuTopology, worker_count: int) -> PinningPlan:
    """Pack workers onto as few physical cores as possible.

    Each worker gets a contiguous block of CPUs.  Useful when you want
    to maximise L1/L2 cache sharing between workers (e.g. prompt processing).
    """
    cpus = [c.core_id for c in sorted(topology.cores, key=lambda c: c.core_id)]
    assignments = _chunk_round_robin(cpus, worker_count)
    return PinningPlan(
        worker_count=worker_count,
        strategy_name="compact",
        logical_cpus=assignments,
        description="Pack workers onto contiguous CPUs to share cache.",
    )


def strategy_spread(topology: CpuTopology, worker_count: int) -> PinningPlan:
    """Spread workers across packages and NUMA nodes.

    Each worker gets CPUs from different physical packages.  Useful for
    memory-bandwidth-bound workloads like batch inference.
    """
    # Group by package, then interleave across packages
    by_pkg = topology.cores_by_package()
    pkg_ids = sorted(by_pkg.keys())

    # Build a round-robin list across packages
    pkg_cpus: Dict[int, List[int]] = {
        pkg: sorted(c.core_id for c in by_pkg[pkg])
        for pkg in pkg_ids
    }

    assignments: List[List[int]] = [[] for _ in range(worker_count)]
    wi = 0
    max_len = max(len(v) for v in pkg_cpus.values())
    for idx in range(max_len):
        for pkg in pkg_ids:
            if idx < len(pkg_cpus[pkg]):
                assignments[wi % worker_count].append(pkg_cpus[pkg][idx])
                wi += 1

    return PinningPlan(
        worker_count=worker_count,
        strategy_name="spread",
        logical_cpus=assignments,
        description="Spread workers across packages for bandwidth-bound workloads.",
    )


def strategy_hybrid(topology: CpuTopology, worker_count: int) -> PinningPlan:
    """Reserve one package for prompt processing and spread elsewhere.

    Requires at least 2 packages.  The first worker gets the first package;
    remaining workers are spread across the rest.  If only one package
    exists, falls back to the compact strategy.
    """
    by_pkg = topology.cores_by_package()
    pkg_ids = sorted(by_pkg.keys())

    if len(pkg_ids) < 2:
        return strategy_compact(topology, worker_count)

    prompt_pkg = pkg_ids[0]
    gen_pkgs = pkg_ids[1:]

    prompt_cpus = sorted(c.core_id for c in by_pkg[prompt_pkg])
    gen_cpus = [
        sorted(c.core_id for c in by_pkg[pkg])
        for pkg in gen_pkgs
    ]

    assignments: List[List[int]] = [[] for _ in range(worker_count)]

    # First worker gets all of prompt_pkg
    if worker_count >= 1:
        assignments[0] = prompt_cpus

    # Remaining workers round-robin across gen_pkgs
    rem_count = worker_count - 1
    if rem_count > 0:
        # Flatten gen CPUs interleaving across packages
        interleaved: List[int] = []
        max_len = max(len(c) for c in gen_cpus)
        for i in range(max_len):
            for g in gen_cpus:
                if i < len(g):
                    interleaved.append(g[i])

        chunks = _chunk_round_robin(interleaved, rem_count)
        for i, chunk in enumerate(chunks):
            assignments[i + 1] = chunk

    return PinningPlan(
        worker_count=worker_count,
        strategy_name="hybrid",
        logical_cpus=assignments,
        description="Reserve package 0 for prompt workers; spread generation workers across remaining packages.",
    )


def _chunk_round_robin(cpus: List[int], num_chunks: int) -> List[List[int]]:
    """Distribute CPUs round-robin into num_chunks groups."""
    chunks: List[List[int]] = [[] for _ in range(num_chunks)]
    for idx, cpu in enumerate(cpus):
        chunks[idx % num_chunks].append(cpu)
    return chunks


# ---------------------------------------------------------------------------
# Apply strategies / helpers
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, PinningFn] = {
    "compact": strategy_compact,
    "spread": strategy_spread,
    "hybrid": strategy_hybrid,
}


def generate_plans(
    topology: CpuTopology,
    worker_count: int,
    strategies: Optional[List[str]] = None,
) -> Dict[str, PinningPlan]:
    """Generate pinning plans for the given topology and worker count.

    Args:
        topology: CPU topology from ``detect_topology()``.
        worker_count: Number of inference workers to plan for.
        strategies: Strategy names to generate (default: all).

    Returns:
        ``{strategy_name: PinningPlan}``.
    """
    selected = strategies or list(STRATEGIES.keys())
    plans: Dict[str, PinningPlan] = {}
    for name in selected:
        if name in STRATEGIES:
            plans[name] = STRATEGIES[name](topology, worker_count)
    return plans
