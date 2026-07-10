"""System metrics collector — CPU, memory, disk utilities."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CpuMetrics:
    percent: float
    cores: int
    load_avg: List[float]


@dataclass
class MemoryMetrics:
    total_gb: float
    used_gb: float
    percent: float


@dataclass
class DiskMetrics:
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float


@dataclass
class SystemMetrics:
    cpu: CpuMetrics = field(default_factory=lambda: CpuMetrics(0, 0, []))
    memory: MemoryMetrics = field(default_factory=lambda: MemoryMetrics(0, 0, 0))
    disk: DiskMetrics = field(default_factory=lambda: DiskMetrics(0, 0, 0, 0))
    timestamp: float = 0.0


class MetricsCollector:
    """Collect system metrics.

    Uses ``psutil`` if available, otherwise returns simulated/fallback values.
    """

    def __init__(self, use_simulated: bool = False):
        self.use_simulated = use_simulated
        self._psutil = None
        if not use_simulated:
            try:
                import psutil
                self._psutil = psutil
            except ImportError:
                self.use_simulated = True

    def collect(self) -> SystemMetrics:
        now = time.time()
        if self._psutil is not None:
            cpu = CpuMetrics(
                percent=self._psutil.cpu_percent(interval=0.1),
                cores=self._psutil.cpu_count() or 0,
                load_avg=list(self._psutil.getloadavg()),
            )
            mem = self._psutil.virtual_memory()
            memory = MemoryMetrics(
                total_gb=mem.total / (1024 ** 3),
                used_gb=mem.used / (1024 ** 3),
                percent=mem.percent,
            )
            disk = self._psutil.disk_usage("/")
            d = DiskMetrics(
                total_gb=disk.total / (1024 ** 3),
                used_gb=disk.used / (1024 ** 3),
                free_gb=disk.free / (1024 ** 3),
                percent=disk.percent,
            )
            return SystemMetrics(cpu=cpu, memory=memory, disk=d, timestamp=now)
        else:
            return self._simulated()

    def _simulated(self) -> SystemMetrics:
        import random
        return SystemMetrics(
            cpu=CpuMetrics(
                percent=random.uniform(10, 90),
                cores=os.cpu_count() or 4,
                load_avg=[random.uniform(0, 4) for _ in range(3)],
            ),
            memory=MemoryMetrics(
                total_gb=16.0,
                used_gb=random.uniform(4, 12),
                percent=random.uniform(25, 75),
            ),
            disk=DiskMetrics(
                total_gb=256.0,
                used_gb=random.uniform(50, 200),
                free_gb=random.uniform(50, 200),
                percent=random.uniform(20, 80),
            ),
            timestamp=time.time(),
        )

    def get_cpu(self) -> CpuMetrics:
        return self.collect().cpu

    def get_memory(self) -> MemoryMetrics:
        return self.collect().memory

    def get_disk(self) -> DiskMetrics:
        return self.collect().disk


def format_metrics(metrics: SystemMetrics) -> str:
    lines = [
        f"## System Metrics ({time.strftime('%H:%M:%S', time.localtime(metrics.timestamp))})",
        "",
        f"### CPU",
        f"- Usage: {metrics.cpu.percent:.1f}%",
        f"- Cores: {metrics.cpu.cores}",
        f"- Load avg: {', '.join(f'{x:.2f}' for x in metrics.cpu.load_avg)}",
        "",
        f"### Memory ({metrics.memory.percent:.1f}%)",
        f"- Total: {metrics.memory.total_gb:.1f} GB",
        f"- Used: {metrics.memory.used_gb:.1f} GB",
        "",
        f"### Disk ({metrics.disk.percent:.1f}%)",
        f"- Total: {metrics.disk.total_gb:.1f} GB",
        f"- Used: {metrics.disk.used_gb:.1f} GB",
        f"- Free: {metrics.disk.free_gb:.1f} GB",
    ]
    return "\n".join(lines)
