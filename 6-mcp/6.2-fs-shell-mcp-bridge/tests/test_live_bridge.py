"""Live integration tests for the fs/shell MCP bridge.

Scope note: this project has NO llama-server dependency, and its offline tests
are not mocked — they already drive the real filesystem (via tmp_path) and real
subprocesses. So there is no in-process interception here for a live layer to
replace.

What these tests add is the one real integration that the offline suite cannot
cover: round-tripping genuine model output through the bridge's safety guards.
Model text is unpredictable in length and encoding (SmolLM2 emits unicode
punctuation and unbalanced quotes), which is exactly what a path/size guard
has to survive. The shell guards are also re-checked against real binaries.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from fs_shell_bridge.bridge import (
    FsSafetyConfig,
    FsShellBridge,
    ShellSafetyConfig,
)


@pytest.fixture
def model_text(chat_base_url, chat_model):
    """A short, real completion from llama-server. One call per test that asks."""
    payload = {
        "model": chat_model,
        "messages": [{"role": "user", "content": "Write one short sentence."}],
        "max_tokens": 32,
    }
    req = urllib.request.Request(
        f"{chat_base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    text = body["choices"][0]["message"]["content"]
    assert text.strip(), "model returned empty content"
    return text


@pytest.fixture
def writable_bridge(tmp_path):
    return FsShellBridge(
        fs_config=FsSafetyConfig(allowed_root=str(tmp_path), allow_write=True)
    )


# ---------------------------------------------------------------------------
# Real model output through the filesystem guards
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_roundtrip_real_model_output(writable_bridge, tmp_path, model_text):
    """write_file → read_file preserves real (unicode, messy) model text."""
    target = tmp_path / "completion.txt"

    msg = writable_bridge.write_file(str(target), model_text)
    assert str(len(model_text)) in msg

    assert writable_bridge.read_file(str(target)) == model_text


@pytest.mark.live
def test_file_info_on_real_model_output(writable_bridge, tmp_path, model_text):
    target = tmp_path / "completion.txt"
    writable_bridge.write_file(str(target), model_text)

    info = writable_bridge.file_info(str(target))
    assert info["name"] == "completion.txt"
    assert info["type"] == "file"
    # Bytes, not characters: any non-ASCII in the model output makes size larger.
    assert info["size"] >= len(model_text)
    assert info["modified"] > 0


@pytest.mark.live
def test_model_output_still_subject_to_size_guard(tmp_path, model_text):
    """The max_file_size guard applies to real content like any other."""
    bridge = FsShellBridge(
        fs_config=FsSafetyConfig(
            allowed_root=str(tmp_path), allow_write=True, max_file_size=8
        )
    )
    target = tmp_path / "completion.txt"
    bridge.write_file(str(target), model_text)  # write is not size-guarded

    with pytest.raises(ValueError, match="File too large"):
        bridge.read_file(str(target))


@pytest.mark.live
def test_model_output_cannot_escape_allowed_root(writable_bridge, tmp_path, model_text):
    """A path outside the root is refused no matter what the content is."""
    with pytest.raises(PermissionError, match="outside allowed root"):
        writable_bridge.write_file(str(tmp_path.parent / "escaped.txt"), model_text)


@pytest.mark.live
def test_list_directory_sees_written_completion(writable_bridge, tmp_path, model_text):
    writable_bridge.write_file(str(tmp_path / "completion.txt"), model_text)

    entries = writable_bridge.list_directory(str(tmp_path))
    names = {e["name"] for e in entries}
    assert "completion.txt" in names


# ---------------------------------------------------------------------------
# Shell guards against real binaries
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_real_shell_roundtrip_of_model_output(writable_bridge, tmp_path, model_text):
    """`cat` on a real file returns what was written.

    subprocess captures text with a trailing newline policy of its own, so
    compare on stripped content.
    """
    target = tmp_path / "completion.txt"
    writable_bridge.write_file(str(target), model_text)

    out = writable_bridge.run_command(f"cat {target}")
    assert model_text.strip() in out


@pytest.mark.live
def test_real_allowed_and_blocked_commands():
    bridge = FsShellBridge()

    assert "hello" in bridge.run_command("echo hello")

    with pytest.raises(PermissionError, match="is blocked"):
        bridge.run_command("rm -rf /tmp/nothing")

    with pytest.raises(PermissionError, match="not in allowed list"):
        bridge.run_command("python --version")


@pytest.mark.live
def test_real_command_output_truncation(tmp_path, model_text):
    """max_output_bytes truncates real subprocess output."""
    bridge = FsShellBridge(
        fs_config=FsSafetyConfig(allowed_root=str(tmp_path), allow_write=True),
        shell_config=ShellSafetyConfig(max_output_bytes=16),
    )
    target = tmp_path / "completion.txt"
    bridge.write_file(str(target), model_text * 4)

    out = bridge.run_command(f"cat {target}")
    assert out.endswith("... (truncated)")
