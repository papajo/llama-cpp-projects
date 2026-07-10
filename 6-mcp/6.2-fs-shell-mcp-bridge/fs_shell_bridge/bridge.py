"""Filesystem operations and shell execution with safety guards."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class FsSafetyConfig:
    """Safety configuration for filesystem access."""

    allowed_root: str = "/"
    allow_write: bool = False
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    max_dir_entries: int = 1000
    blocked_paths: Set[str] = field(default_factory=lambda: {
        "/etc/shadow", "/etc/sudoers", "/etc/passwd",
    })


@dataclass
class ShellSafetyConfig:
    """Safety configuration for shell execution."""

    allowed_commands: Set[str] = field(default_factory=lambda: {
        "ls", "cat", "head", "tail", "wc", "echo", "pwd",
        "date", "whoami", "uname", "df", "du", "ps", "which",
        "find", "grep", "sort", "cut", "tr", "diff",
    })
    blocked_commands: Set[str] = field(default_factory=lambda: {
        "rm", "mv", "cp", "chmod", "chown", "mkfs", "dd",
        "sudo", "su", "passwd", "kill", "reboot", "shutdown",
        "wget", "curl", "nc", "nmap", "ssh", "scp",
    })
    timeout: int = 30
    max_output_bytes: int = 100_000


class FsShellBridge:
    """Safe filesystem and shell operations.

    All paths are validated against the allowed root.
    All commands are checked against the allowlist/blocklist.
    """

    def __init__(
        self,
        fs_config: Optional[FsSafetyConfig] = None,
        shell_config: Optional[ShellSafetyConfig] = None,
    ):
        self.fs_config = fs_config or FsSafetyConfig()
        self.shell_config = shell_config or ShellSafetyConfig()

    # ── Filesystem ──────────────────────────────────────────

    def _resolve_path(self, path: str) -> Path:
        """Resolve and validate a path is within allowed root."""
        p = Path(path).resolve()
        root = Path(self.fs_config.allowed_root).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise PermissionError(
                f"Path '{path}' is outside allowed root "
                f"'{self.fs_config.allowed_root}'"
            )
        for blocked in self.fs_config.blocked_paths:
            if str(p) == str(Path(blocked).resolve()):
                raise PermissionError(f"Access to '{blocked}' is blocked")
        return p

    def read_file(self, path: str) -> str:
        """Read and return the contents of a file."""
        p = self._resolve_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        size = p.stat().st_size
        if size > self.fs_config.max_file_size:
            raise ValueError(
                f"File too large: {size} bytes (max {self.fs_config.max_file_size})"
            )
        return p.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file (if writes are allowed)."""
        if not self.fs_config.allow_write:
            raise PermissionError("File writes are disabled")
        p = self._resolve_path(path)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {p}"

    def list_directory(self, path: str) -> List[Dict[str, object]]:
        """List the contents of a directory."""
        p = self._resolve_path(path)
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        entries = []
        for entry in p.iterdir():
            if len(entries) >= self.fs_config.max_dir_entries:
                break
            try:
                stat = entry.stat()
                entries.append({
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
            except OSError:
                continue
        return entries

    def file_info(self, path: str) -> Dict[str, object]:
        """Get metadata about a file or directory."""
        p = self._resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        stat = p.stat()
        return {
            "name": p.name,
            "type": "directory" if p.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "permissions": oct(stat.st_mode),
        }

    # ── Shell ───────────────────────────────────────────────

    def _validate_command(self, command: str) -> List[str]:
        """Parse and validate a shell command."""
        parts = shlex.split(command)
        if not parts:
            raise ValueError("Empty command")
        cmd = parts[0]
        if cmd in self.shell_config.blocked_commands:
            raise PermissionError(f"Command '{cmd}' is blocked")
        if cmd not in self.shell_config.allowed_commands:
            raise PermissionError(
                f"Command '{cmd}' not in allowed list. "
                f"Allowed: {sorted(self.shell_config.allowed_commands)}"
            )
        return parts

    def run_command(self, command: str) -> str:
        """Execute a shell command and return its output."""
        parts = self._validate_command(command)
        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=self.shell_config.timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Command timed out after {self.shell_config.timeout}s"
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Command not found: {parts[0]}")

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        if len(output) > self.shell_config.max_output_bytes:
            output = output[:self.shell_config.max_output_bytes] + "\n... (truncated)"

        return output
