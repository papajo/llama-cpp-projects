"""Tests for MCP server tools."""

import pytest

from metrics_server.server import get_cpu_usage, get_disk_usage, get_memory_usage, get_system_metrics


class TestServerTools:
    def test_get_system_metrics(self):
        result = get_system_metrics()
        assert "CPU" in result
        assert "Memory" in result
        assert "Disk" in result

    def test_get_cpu_usage(self):
        result = get_cpu_usage()
        assert "CPU" in result
        assert "%" in result

    def test_get_memory_usage(self):
        result = get_memory_usage()
        assert "Memory" in result
        assert "GB" in result

    def test_get_disk_usage(self):
        result = get_disk_usage("/")
        assert "Disk" in result
        assert "GB" in result
