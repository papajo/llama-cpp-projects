"""CPU topology detection — cross-platform with macOS/Linux/fallback."""

from __future__ import annotations

import json
import os
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CoreInfo:
    """Info about a single logical CPU core."""
    core_id: int
    package_id: int
    numa_node: int = 0
    smt_id: int = 0  # hyperthread sibling index


@dataclass
class CpuTopology:
    """Full CPU topology for the system."""
    logical_cpus: int
    physical_cores: int
    packages: int
    numa_nodes: int
    cores: List[CoreInfo] = field(default_factory=list)

    def cores_by_package(self) -> Dict[int, List[CoreInfo]]:
        result: Dict[int, List[CoreInfo]] = {}
        for c in self.cores:
            result.setdefault(c.package_id, []).append(c)
        return result

    def cores_by_numa(self) -> Dict[int, List[CoreInfo]]:
        result: Dict[int, List[CoreInfo]] = {}
        for c in self.cores:
            result.setdefault(c.numa_node, []).append(c)
        return result

    def summary(self) -> str:
        return (
            f"  Logical CPUs: {self.logical_cpus}\n"
            f"  Physical cores: {self.physical_cores}\n"
            f"  Packages (sockets): {self.packages}\n"
            f"  NUMA nodes: {self.numa_nodes}\n"
        )


# ---------------------------------------------------------------------------
# topology probing
# ---------------------------------------------------------------------------

def _detect_macos() -> Optional[CpuTopology]:
    """Detect CPU topology on macOS via sysctl."""
    try:
        logical = int(
            subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"]).strip()
        )
        physical = int(
            subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip()
        )
        packages = int(
            subprocess.check_output(["sysctl", "-n", "hw.packages"]).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None

    # macOS has no NUMA (< 1 means not available)
    numa_nodes = 1

    cores_per_package = physical // packages if packages else physical
    smt_on = logical > physical
    smt_factor = logical // physical if physical else 1

    cores: List[CoreInfo] = []
    for cpu in range(logical):
        pkg = cpu // (cores_per_package * smt_factor) if smt_on else cpu // cores_per_package
        if pkg >= packages:
            pkg = packages - 1
        phys_core_within_pkg = (cpu % (cores_per_package * smt_factor)) // smt_factor if smt_on else cpu % cores_per_package
        smt = (cpu % (cores_per_package * smt_factor)) % smt_factor if smt_on else 0
        cores.append(CoreInfo(
            core_id=cpu,
            package_id=pkg,
            numa_node=0,
            smt_id=smt,
        ))

    return CpuTopology(
        logical_cpus=logical,
        physical_cores=physical,
        packages=packages,
        numa_nodes=numa_nodes,
        cores=cores,
    )


def _detect_linux() -> Optional[CpuTopology]:
    """Detect CPU topology on Linux by reading /proc/cpuinfo and /sys."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except FileNotFoundError:
        return None

    processors: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in cpuinfo.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            current[key.strip()] = val.strip()
        elif line.strip() == "" and current:
            processors.append(current)
            current = {}
    if current:
        processors.append(current)

    if not processors:
        return None

    cores: List[CoreInfo] = []
    for p in processors:
        try:
            cpu = int(p.get("processor", -1))
            core_id = int(p.get("core id", 0))
            phys_id = int(p.get("physical id", 0))
            numa = int(p.get("numa_node", -1) or -1)
            # On some kernels numa_node may be -1; default to 0
            if numa < 0:
                numa = 0
        except (ValueError, KeyError):
            continue
        if cpu < 0:
            continue
        cores.append(CoreInfo(
            core_id=cpu,
            package_id=phys_id,
            numa_node=numa,
            smt_id=core_id,
        ))

    if not cores:
        return None

    logical_cpus = len(cores)
    physical_cores = len({(c.package_id, c.smt_id) for c in cores})
    packages = len({c.package_id for c in cores})
    numa_nodes = len({c.numa_node for c in cores})

    return CpuTopology(
        logical_cpus=logical_cpus,
        physical_cores=physical_cores,
        packages=packages,
        numa_nodes=numa_nodes,
        cores=cores,
    )


def _detect_psutil() -> CpuTopology:
    """Fallback: use psutil for basic topology."""
    import psutil

    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or 1

    packages = 1
    numa_nodes = 1

    cores = [
        CoreInfo(core_id=i, package_id=0, numa_node=0, smt_id=0)
        for i in range(logical)
    ]
    return CpuTopology(
        logical_cpus=logical,
        physical_cores=physical,
        packages=packages,
        numa_nodes=numa_nodes,
        cores=cores,
    )


def detect_topology() -> CpuTopology:
    """Detect CPU topology, trying platform-specific methods then psutil fallback."""
    system = platform.system()
    if system == "Darwin":
        result = _detect_macos()
        if result is not None:
            return result
    elif system == "Linux":
        result = _detect_linux()
        if result is not None:
            return result
    return _detect_psutil()


def topology_to_json(topology: CpuTopology) -> str:
    """Serialize topology to JSON string."""
    data = {
        "logical_cpus": topology.logical_cpus,
        "physical_cores": topology.physical_cores,
        "packages": topology.packages,
        "numa_nodes": topology.numa_nodes,
        "cores": [
            {
                "core_id": c.core_id,
                "package_id": c.package_id,
                "numa_node": c.numa_node,
                "smt_id": c.smt_id,
            }
            for c in topology.cores
        ],
    }
    return json.dumps(data, indent=2)
