"""FastMCP server exposing system metrics tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from metrics_server.collector import MetricsCollector, format_metrics

mcp = FastMCP("metrics-server")
_collector = MetricsCollector()


@mcp.tool()
def get_system_metrics() -> str:
    """Get a full system health report — CPU, memory, and disk usage."""
    metrics = _collector.collect()
    return format_metrics(metrics)


@mcp.tool()
def get_cpu_usage() -> str:
    """Get current CPU usage percentage, core count, and load average."""
    cpu = _collector.get_cpu()
    return (
        f"CPU: {cpu.percent:.1f}% | Cores: {cpu.cores} | "
        f"Load: {', '.join(f'{x:.2f}' for x in cpu.load_avg)}"
    )


@mcp.tool()
def get_memory_usage() -> str:
    """Get current memory usage in GB and percentage."""
    mem = _collector.get_memory()
    return (
        f"Memory: {mem.used_gb:.1f} / {mem.total_gb:.1f} GB ({mem.percent:.1f}%)"
    )


@mcp.tool()
def get_disk_usage(path: str = "/") -> str:
    """Get disk usage for a given mount point."""
    # Note: path param is for interface; collector uses psutil for "/"
    disk = _collector.get_disk()
    return (
        f"Disk ({path}): {disk.used_gb:.1f} / {disk.total_gb:.1f} GB "
        f"({disk.percent:.1f}%) — Free: {disk.free_gb:.1f} GB"
    )


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the MCP metrics server over streamable HTTP."""
    mcp.run(host=host, port=port)
