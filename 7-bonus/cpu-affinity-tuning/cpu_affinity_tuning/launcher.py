"""Generate llama.cpp launch commands with CPU affinity settings.

On Linux this produces ``taskset``-based invocations.  On macOS
it prints the topology and recommended ``thread_policy`` settings
(which require a small C helper or the ``affinity`` PyPI package).
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import List, Optional

from .strategies import PinningPlan, generate_plans
from .topology import CpuTopology, detect_topology, topology_to_json


def generate_launch_command(
    plan: PinningPlan,
    worker_id: int,
    *,
    binary: str = "./llama-cli",
    model: str = "models/llama-7b.gguf",
    prompt: str = "Hello, world",
    n_threads: Optional[int] = None,
    extra_args: str = "",
) -> List[str]:
    """Generate launch commands for a specific worker.

    Args:
        plan: The pinning plan to use.
        worker_id: Which worker index to generate for.
        binary: Path to the llama.cpp executable.
        model: Path to the model file.
        prompt: Input prompt text.
        n_threads: Threads per worker (default: number of CPUs assigned).
        extra_args: Any additional llama.cpp arguments.

    Returns:
        List of command strings (may include setup commands before the main binary).
    """
    cpus = plan.logical_cpus[worker_id]
    if n_threads is None:
        n_threads = len(cpus)

    system = platform.system()
    cpu_list = ",".join(map(str, cpus))

    if system == "Linux":
        return [
            f"taskset -c {cpu_list} {binary} "
            f"--model {model} "
            f"--prompt \"{prompt}\" "
            f"--threads {n_threads} "
            f"--id {worker_id} "
            f"{extra_args}"
        ]
    elif system == "Darwin":
        return [
            f"# macOS: set thread affinity via thread_policy_set()",
            f"# CPUs for worker {worker_id}: {cpu_list}",
            f"# Install: pip install affinity  (if available)",
            f"export AFFINITY_CPUS=\"{cpu_list}\"",
            f"{binary} "
            f"--model {model} "
            f"--prompt \"{prompt}\" "
            f"--threads {n_threads} "
            f"--id {worker_id} "
            f"{extra_args}",
        ]
    else:
        return [
            f"# Pinning CPUs {cpu_list} for worker {worker_id}",
            f"{binary} "
            f"--model {model} "
            f"--prompt \"{prompt}\" "
            f"--threads {n_threads} "
            f"--id {worker_id} "
            f"{extra_args}",
        ]


def generate_launch_script(
    plans: List[PinningPlan],
    *,
    binary: str = "./llama-cli",
    model: str = "models/llama-7b.gguf",
    prompt: str = "Hello, world",
    extra_args: str = "",
) -> str:
    """Generate a complete shell script launching all workers.

    Args:
        plans: List of PinningPlan objects (one per strategy to compare).
        prompt: Input prompt.
        extra_args: Additional llama.cpp arguments.

    Returns:
        A shell script string.
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    env_lines: List[str] = []

    for plan in plans:
        for wid in range(plan.worker_count):
            cmds = generate_launch_command(
                plan, wid,
                binary=binary, model=model, prompt=prompt,
                extra_args=extra_args,
            )
            for cmd in cmds:
                if cmd.startswith("#"):
                    env_lines.append(cmd)
                else:
                    env_lines.append(f"  {cmd} &")
        env_lines.append("wait")
        lines.extend(env_lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Print topology, strategies, and sample commands."""
    import platform

    topo = detect_topology()
    print("=" * 60)
    print("CPU Topology")
    print("=" * 60)
    print(topo.summary())

    print("\nStrategies for 4 workers:")
    plans = generate_plans(topo, worker_count=4)

    for name, plan in plans.items():
        print(f"\n--- {name.upper()} ---")
        print(plan.summary())
        print("\nCommands (worker 0):")
        for cmd in generate_launch_command(plan, 0):
            print(f"  {cmd}")

    print("\n" + "=" * 60)
    print("JSON Topology")
    print("=" * 60)
    print(topology_to_json(topo))


if __name__ == "__main__":
    main()
