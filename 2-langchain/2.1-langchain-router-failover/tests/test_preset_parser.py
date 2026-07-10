"""Tests for the INI preset parser."""

import tempfile
from pathlib import Path

import pytest

from langchain_llamacpp_router import PresetConfig, load_presets_ini
from langchain_llamacpp_router.exceptions import PresetParseError


def test_load_basic_ini():
    """Parse a valid INI with two models."""
    ini_content = """
[coder:Qwen2.5-7B-Q4_K_M]
model = /models/qwen2.5-7b-q4_k_m.gguf
tags = code, generation, fast
description = Lightweight coder

[chat:Llama-3.1-70B]
model = /models/llama-3.1-70b.gguf
tags = chat, creative, slow
description = Full 70B chat model
cold_start_seconds = 15
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write(ini_content)
        tmp_path = f.name

    try:
        presets = load_presets_ini(Path(tmp_path))
        assert len(presets) == 2

        # Check coder
        assert "coder" in presets
        coder = presets["coder"]
        assert coder.name == "coder"
        assert coder.model_path == "/models/qwen2.5-7b-q4_k_m.gguf"
        assert coder.tags == ["code", "generation", "fast"]
        assert coder.description == "Lightweight coder"
        assert coder.cold_start_seconds == 0.0

        # Check chat
        assert "chat" in presets
        chat = presets["chat"]
        assert chat.name == "chat"
        assert chat.model_path == "/models/llama-3.1-70b.gguf"
        assert chat.tags == ["chat", "creative", "slow"]
        assert chat.description == "Full 70B chat model"
        assert chat.cold_start_seconds == 15.0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_ini_missing_model_key():
    """INI section without 'model' key should raise PresetParseError."""
    ini_content = """
[coder]
tags = code

[chat]
model = /models/chat.gguf
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write(ini_content)
        tmp_path = f.name

    try:
        with pytest.raises(PresetParseError, match="missing required 'model' key"):
            load_presets_ini(Path(tmp_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_ini_file_not_found():
    """Non-existent file should raise PresetParseError."""
    with pytest.raises(PresetParseError, match="File not found"):
        load_presets_ini(Path("/nonexistent/path.ini"))


def test_load_ini_empty_tags():
    """Sections with empty or missing tags should produce empty tag list."""
    ini_content = """
[coder]
model = /models/coder.gguf
tags =
description = No tags
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write(ini_content)
        tmp_path = f.name

    try:
        presets = load_presets_ini(Path(tmp_path))
        assert presets["coder"].tags == []
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_preset_config_section_id():
    """PresetConfig.section_id returns the original section header."""
    config = PresetConfig(
        name="coder",
        alias="coder:Qwen2.5-7B",
        model_path="/models/qwen.gguf",
        tags=["code"],
    )
    assert config.section_id == "coder:coder:Qwen2.5-7B"

    config2 = PresetConfig(
        name="coder",
        alias="coder",
        model_path="/models/qwen.gguf",
        tags=["code"],
    )
    assert config2.section_id == "coder"
