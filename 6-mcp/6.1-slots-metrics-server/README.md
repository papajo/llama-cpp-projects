# 6.1 MCP Metrics Server

A FastMCP server that exposes CPU, memory, and disk metrics as MCP tools — usable by any MCP client (Claude, Copilot, etc.)

## Tools

| Tool | Returns |
|------|---------|
| `get_system_metrics()` | Full CPU + memory + disk report |
| `get_cpu_usage()` | CPU%, core count, load average |
| `get_memory_usage()` | Used/total GB and percentage |
| `get_disk_usage(path)` | Disk usage for a mount point |

## Architecture

```
  MCP Client ──▶ FastMCP Server ──▶ MetricsCollector
    (Claude)       (tools)             │
                                       ├── psutil (real)
                                       └── simulated (fallback)
```

## Usage

```python
from metrics_server.server import mcp
mcp.run()  # starts streamable HTTP server on 127.0.0.1:8000
```

Or via CLI after `pip install -e .`:

```bash
python -c "from metrics_server.server import run_server; run_server()"
```

## Files

| File | Purpose |
|------|---------|
| `metrics_server/collector.py` | `MetricsCollector`, dataclasses, `format_metrics()` |
| `metrics_server/server.py` | FastMCP server with 4 tools |
| `tests/` | 10 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
