"""Tests for filesystem and shell bridge."""

import os
import tempfile

import pytest

from fs_shell_bridge.bridge import (
    FsSafetyConfig,
    FsShellBridge,
    ShellSafetyConfig,
)


@pytest.fixture
def bridge():
    return FsShellBridge()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


class TestFileSystem:
    def test_read_file_not_found(self, bridge):
        with pytest.raises(FileNotFoundError):
            bridge.read_file("/nonexistent/path.txt")

    def test_read_file_allowed(self, bridge, temp_dir):
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello world")
        bridge.fs_config.allowed_root = temp_dir
        content = bridge.read_file(test_file)
        assert content == "hello world"

    def test_write_disabled_by_default(self, bridge, temp_dir):
        bridge.fs_config.allowed_root = temp_dir
        with pytest.raises(PermissionError, match="writes are disabled"):
            bridge.write_file(os.path.join(temp_dir, "out.txt"), "data")

    def test_write_enabled(self, temp_dir):
        cfg = FsSafetyConfig(allowed_root=temp_dir, allow_write=True)
        b = FsShellBridge(fs_config=cfg)
        path = os.path.join(temp_dir, "out.txt")
        result = b.write_file(path, "hello")
        assert "Wrote 5 bytes" in result
        with open(path) as f:
            assert f.read() == "hello"

    def test_file_info(self, bridge, temp_dir):
        test_file = os.path.join(temp_dir, "info.txt")
        with open(test_file, "w") as f:
            f.write("data")
        bridge.fs_config.allowed_root = temp_dir
        info = bridge.file_info(test_file)
        assert info["name"] == "info.txt"
        assert info["size"] == 4

    def test_list_directory(self, bridge, temp_dir):
        for name in ["a.txt", "b.txt"]:
            with open(os.path.join(temp_dir, name), "w") as f:
                f.write(name)
        os.mkdir(os.path.join(temp_dir, "sub"))
        bridge.fs_config.allowed_root = temp_dir
        entries = bridge.list_directory(temp_dir)
        names = {e["name"] for e in entries}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "sub" in names

    def test_blocked_path(self, bridge):
        bridge.fs_config.blocked_paths.add("/etc/hosts")
        # Default allowed_root is "/" so /etc/hosts passes root check
        with pytest.raises(PermissionError, match="blocked"):
            bridge.read_file("/etc/hosts")

    def test_file_too_large(self, bridge, temp_dir):
        big = os.path.join(temp_dir, "big.txt")
        with open(big, "w") as f:
            f.write("x" * 100)
        bridge.fs_config.max_file_size = 50
        bridge.fs_config.allowed_root = temp_dir
        with pytest.raises(ValueError, match="File too large"):
            bridge.read_file(big)


class TestShell:
    def test_allowed_command(self, bridge):
        result = bridge.run_command("echo hello")
        assert "hello" in result

    def test_blocked_command(self, bridge):
        with pytest.raises(PermissionError, match="blocked"):
            bridge.run_command("rm test.txt")

    def test_not_in_allowlist(self, bridge):
        with pytest.raises(PermissionError, match="not in allowed"):
            bridge.run_command("pip install foo")

    def test_timeout(self, bridge):
        bridge.shell_config.allowed_commands.add("sleep")
        bridge.shell_config.timeout = 1
        with pytest.raises(TimeoutError):
            bridge.run_command("sleep 10")

    def test_command_not_found(self, bridge):
        bridge.shell_config.allowed_commands.add("nonexistent_cmd_xyz")
        with pytest.raises(FileNotFoundError):
            bridge.run_command("nonexistent_cmd_xyz")

    def test_empty_command(self, bridge):
        with pytest.raises(ValueError, match="Empty"):
            bridge.run_command("")
