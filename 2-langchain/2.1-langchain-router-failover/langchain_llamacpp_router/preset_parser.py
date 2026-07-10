"""
INI preset parser for llama.cpp ``--models-preset`` files.

llama.cpp's router server accepts a ``--models-preset`` INI file that
defines model aliases and their backing GGUFs.  This module loads those
files and enriches them with routing metadata (tags, description,
cold-start timeout).

Example INI::

    [coder:Qwen2.5-7B-Q4_K_M]
    model = /models/qwen2.5-7b-q4_k_m.gguf
    tags = code, generation, fast
    description = Lightweight 7B coder — fast on CPU

    [chat:Llama-3.1-70B-Q3_K_L]
    model = /models/llama-3.1-70b-q3_k_l.gguf
    tags = chat, creative, slow
    description = Full 70B creative writing model — needs GPU
    cold_start_seconds = 15

    [embedder:all-MiniLM-L6-v2]
    model = /models/all-MiniLM-L6-v2.Q4_0.gguf
    tags = embedding, fast
    description = Sentence embedding — no generation; for RAG
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .exceptions import PresetParseError


@dataclass
class PresetConfig:
    """A single model entry parsed from a ``--models-preset`` INI file."""

    name: str
    """Short alias used as the model identifier (e.g. ``coder``)."""

    alias: str
    """Full ``[section]`` header — includes both alias and optional model name."""

    model_path: str
    """Path to the GGUF file on disk."""

    tags: List[str] = field(default_factory=list)
    """User-defined routing tags (e.g. ``["code", "generation", "fast"]``)."""

    description: str = ""
    """Human-readable description."""

    cold_start_seconds: float = 0.0
    """Expected cold-start latency in seconds (0 means unknown / live probe)."""

    @property
    def section_id(self) -> str:
        """Return the full section header as it appeared in the INI (e.g. ``coder:Qwen2.5-7B-Q4_K_M``)."""
        return f"{self.name}:{self.alias}" if self.name != self.alias else self.name


def load_presets_ini(path: Path | str) -> Dict[str, PresetConfig]:
    """Parse a ``--models-preset`` INI file into ``PresetConfig`` dict keyed by **alias**.

    The INI section header format is ``[alias:ModelName]`` or just ``[alias]``.

    Raises:
        PresetParseError: If the file cannot be read or parsed.
    """
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parsed = parser.read(str(path))
        if not parsed:
            raise PresetParseError(f"File not found or empty: {path}")
    except configparser.Error as e:
        raise PresetParseError(f"INI parse error in {path}: {e}") from e

    presets: Dict[str, PresetConfig] = {}
    for section in parser.sections():
        # Section header: "alias" or "alias:ModelName"
        name = section.split(":")[0].strip()
        alias = section  # keep the raw header

        model_path = parser.get(section, "model", fallback="")
        if not model_path:
            raise PresetParseError(
                f"Section [{section}] is missing required 'model' key"
            )

        raw_tags = parser.get(section, "tags", fallback="")
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        description = parser.get(section, "description", fallback="")
        cold_start = parser.getfloat(section, "cold_start_seconds", fallback=0.0)

        config = PresetConfig(
            name=name,
            alias=alias,
            model_path=model_path,
            tags=tags,
            description=description,
            cold_start_seconds=cold_start,
        )
        presets[name] = config

    return presets
