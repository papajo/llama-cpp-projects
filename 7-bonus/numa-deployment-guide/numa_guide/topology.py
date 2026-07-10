"""NUMA topology detection — cross-platform with Linux and macOS support."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class NumaNode:
    """A single NUMA node."""
    node_id: int
    cpu_ids: List[int]
    total_memory_kb: int = 0
    free_memory_kb: int = 0
    distance_map: Dict[int, int] = field(default_factory=dict)  # target_node → distance


@dataclass
class NumaTopology:
    """Full NUMA topology."""
    nodes: List[NumaNode]
    has_numa: bool

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def summary(self) -> str:
        lines = [f"NUMA nodes: {self.node_count}", ""]
        for n in self.nodes:
            lines.append(
                f"  Node {n.node_id}: CPUs={n.cpu_ids}, "
                f"Mem={n.total_memory_kb // 1024}MB "
                f"(free: {n.free_memory_kb // 1024}MB)"
            )
            if n.distance_map:
                distances = ", ".join(
                    f"N{d}={dist}" for d, dist in sorted(n.distance_map.items())
                )
                lines.append(f"         Distances: {distances}")
        return "\n".join(lines)


def _detect_linux_numa() -> Optional[NumaTopology]:
    """Detect NUMA topology on Linux via /sys and numactl."""
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        return None

    node_dirs = sorted(base.glob("node[0-9]*"))
    if not node_dirs:
        return None

    nodes: List[NumaNode] = []
    for nd in node_dirs:
        try:
            node_id = int(nd.name.replace("node", ""))
        except ValueError:
            continue

        # Read CPU list
        cpulist_path = nd / "cpulist"
        cpu_ids: List[int] = []
        if cpulist_path.exists():
            raw = cpulist_path.read_text().strip()
            for part in raw.split(","):
                if "-" in part:
                    lo, hi = part.split("-")
                    cpu_ids.extend(range(int(lo), int(hi) + 1))
                else:
                    part = part.strip()
                    if part:
                        cpu_ids.append(int(part))

        # Read memory info
        meminfo_path = nd / "meminfo"
        total_kb = 0
        free_kb = 0
        if meminfo_path.exists():
            for line in meminfo_path.read_text().splitlines():
                if "MemTotal" in line:
                    total_kb = int(line.split()[1])
                elif "MemFree" in line:
                    free_kb = int(line.split()[1])

        # Distance map
        dist_path = nd / "distance"
        distance_map: Dict[int, int] = {}
        if dist_path.exists():
            dists = dist_path.read_text().strip().split()
            for i, d in enumerate(dists):
                distance_map[i] = int(d)

        nodes.append(NumaNode(
            node_id=node_id,
            cpu_ids=cpu_ids,
            total_memory_kb=total_kb,
            free_memory_kb=free_kb,
            distance_map=distance_map,
        ))

    return NumaTopology(nodes=nodes, has_numa=len(nodes) > 1) if nodes else None


def _detect_macos_numa() -> Optional[NumaTopology]:
    """macOS does not expose NUMA; return single-node topology."""
    try:
        logical = int(
            subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"]).strip()
        )
        mem_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None

    node = NumaNode(
        node_id=0,
        cpu_ids=list(range(logical)),
        total_memory_kb=mem_bytes // 1024,
    )
    return NumaTopology(nodes=[node], has_numa=False)


def _detect_psutil_numa() -> NumaTopology:
    """Fallback: single-node topology via psutil."""
    import psutil

    logical = psutil.cpu_count(logical=True) or 1
    mem = psutil.virtual_memory()

    node = NumaNode(
        node_id=0,
        cpu_ids=list(range(logical)),
        total_memory_kb=mem.total // 1024,
        free_memory_kb=mem.available // 1024,
    )
    return NumaTopology(nodes=[node], has_numa=False)


def detect_numa_topology() -> NumaTopology:
    """Detect NUMA topology using platform-specific methods."""
    system = platform.system()
    if system == "Linux":
        result = _detect_linux_numa()
        if result is not None:
            return result
    elif system == "Darwin":
        result = _detect_macos_numa()
        if result is not None:
            return result
    return _detect_psutil_numa()
