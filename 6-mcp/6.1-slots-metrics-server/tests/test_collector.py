"""Tests for metrics collector and formatter."""

import pytest

from metrics_server.collector import (
    CpuMetrics,
    DiskMetrics,
    MemoryMetrics,
    MetricsCollector,
    SystemMetrics,
    format_metrics,
)


class TestMetricsCollector:
    def test_simulated_cpu(self):
        collector = MetricsCollector(use_simulated=True)
        cpu = collector.get_cpu()
        assert 10 <= cpu.percent <= 90
        assert cpu.cores >= 1
        assert len(cpu.load_avg) == 3

    def test_simulated_memory(self):
        collector = MetricsCollector(use_simulated=True)
        mem = collector.get_memory()
        assert mem.total_gb == 16.0
        assert 4 <= mem.used_gb <= 12

    def test_simulated_disk(self):
        collector = MetricsCollector(use_simulated=True)
        disk = collector.get_disk()
        assert disk.total_gb == 256.0
        assert 50 <= disk.used_gb <= 200

    def test_collect_returns_all(self):
        collector = MetricsCollector(use_simulated=True)
        metrics = collector.collect()
        assert isinstance(metrics.cpu, CpuMetrics)
        assert isinstance(metrics.memory, MemoryMetrics)
        assert isinstance(metrics.disk, DiskMetrics)
        assert metrics.timestamp > 0


class TestFormatting:
    def test_format_metrics_includes_all_sections(self):
        metrics = SystemMetrics(
            cpu=CpuMetrics(percent=50.0, cores=8, load_avg=[1.0, 2.0, 3.0]),
            memory=MemoryMetrics(total_gb=16.0, used_gb=8.0, percent=50.0),
            disk=DiskMetrics(total_gb=256.0, used_gb=128.0, free_gb=128.0, percent=50.0),
            timestamp=1000000,
        )
        output = format_metrics(metrics)
        assert "CPU" in output
        assert "Memory" in output
        assert "Disk" in output
        assert "50.0%" in output
        assert "8" in output  # cores
        assert "16.0 GB" in output

    def test_dataclass_defaults(self):
        cpu = CpuMetrics(percent=0, cores=0, load_avg=[])
        assert cpu.percent == 0
        mem = MemoryMetrics(total_gb=0, used_gb=0, percent=0)
        assert mem.total_gb == 0
        disk = DiskMetrics(total_gb=0, used_gb=0, free_gb=0, percent=0)
        assert disk.used_gb == 0
