"""FastMCP server exposing filesystem and shell tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from fs_shell_bridge.bridge import FsShellBridge

mcp = FastMCP("fs-shell-bridge")
_bridge = FsShellBridge()


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file. Path must be within allowed root."""
    return _bridge.read_file(path)


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file. Only enabled if allow_write is True."""
    return _bridge.write_file(path, content)


@mcp.tool()
def list_directory(path: str) -> str:
    """List files and directories in a directory."""
    entries = _bridge.list_directory(path)
    lines = [f"{e['type']:10s} {e['size']:>8d}  {e['name']}" for e in entries]
    return "\n".join(lines)


@mcp.tool()
def file_info(path: str) -> str:
    """Get metadata about a file or directory."""
    info = _bridge.file_info(path)
    return "\n".join(f"{k}: {v}" for k, v in info.items())


@mcp.tool()
def run_command(command: str) -> str:
    """Execute a shell command (allowlist enforced)."""
    return _bridge.run_command(command)
