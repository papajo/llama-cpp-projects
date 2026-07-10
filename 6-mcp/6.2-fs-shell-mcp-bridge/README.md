# 6.2 FS-Shell MCP Bridge

A FastMCP server exposing filesystem and shell tools with configurable safety guards — path validation, command allowlist/blocklist, size limits, and timeouts.

## Tools

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read a file's contents (respects `allowed_root`, `blocked_paths`, `max_file_size`) |
| `write_file(path, content)` | Write to a file (only if `allow_write=True`) |
| `list_directory(path)` | List entries in a directory |
| `file_info(path)` | Get metadata about a file or directory |
| `run_command(command)` | Execute a shell command (allowlist/blocklist enforced) |

## Safety Configuration

```python
from fs_shell_bridge.bridge import FsSafetyConfig, ShellSafetyConfig

fs_cfg = FsSafetyConfig(
    allowed_root="/home/user/projects",
    allow_write=True,
    max_file_size=1_000_000,
    blocked_paths={"/home/user/projects/secrets"},
)

shell_cfg = ShellSafetyConfig(
    allowed_commands={"ls", "cat", "echo", "pwd", "whoami"},
    blocked_commands={"rm", "sudo", "curl"},
    timeout=10,
)
```

## Files

| File | Purpose |
|------|---------|
| `fs_shell_bridge/bridge.py` | `FsSafetyConfig`, `ShellSafetyConfig`, `FsShellBridge` |
| `fs_shell_bridge/server.py` | FastMCP server with 5 tools |
| `tests/` | 14 tests, all passing |

## Running Tests

```bash
pip install -e .[test]
pytest tests/ -v
```
