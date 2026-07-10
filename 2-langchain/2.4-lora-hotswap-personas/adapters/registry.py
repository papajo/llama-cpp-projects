"""
LoRA adapter registry — defines available adapters and their metadata.

Each adapter corresponds to a persona/task.  When the agent selects a
persona, it calls ``POST /lora-adapters`` with the adapter ID and scale.

Usage::

    from adapters import AdapterRegistry, load_adapter_registry

    registry = load_adapter_registry("adapters/registry.json")
    for name, defn in registry.items():
        print(f"{name}: {defn.description} (scale={defn.scale})")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AdapterDef:
    """Definition of a single LoRA adapter."""

    name: str
    """Short identifier (e.g. ``legal-tone``)."""

    lora_id: int
    """
    LoRA slot ID on the llama.cpp server.
    Corresponds to the order in ``--lora`` / ``--lora-scaled`` flags.
    """

    gguf_path: str
    """Path to the LoRA GGUF file on disk."""

    scale: float = 1.0
    """Adapter scale (``--lora-scaled FNAME:SCALE``)."""

    description: str = ""
    """Human-readable description of the persona."""

    tags: List[str] = field(default_factory=list)
    """Tags for routing (e.g. ``["legal", "formal"]``)."""

    keywords: List[str] = field(default_factory=list)
    """Keywords that trigger this persona (e.g. ``["contract", "lawsuit"]``)."""

    is_default: bool = False
    """If True, this adapter is applied at startup as the default persona."""


class AdapterRegistry(Dict[str, AdapterDef]):
    """
    Dict of adapter name → ``AdapterDef``.

    Provides lookup methods by tag and keyword.
    """

    def find_by_tag(self, tag: str) -> List[AdapterDef]:
        """Return adapters whose tags contain *tag* (case-insensitive)."""
        tag_lower = tag.lower()
        return [a for a in self.values() if tag_lower in [t.lower() for t in a.tags]]

    def find_by_keyword(self, text: str) -> List[AdapterDef]:
        """
        Return adapters whose keywords appear in *text* (case-insensitive).

        The more keywords match, the earlier the adapter appears in the list.
        """
        text_lower = text.lower()
        scored = []
        for adapter in self.values():
            matches = sum(1 for kw in adapter.keywords if kw.lower() in text_lower)
            if matches > 0:
                scored.append((matches, adapter))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored]

    def get_default(self) -> Optional[AdapterDef]:
        """Return the adapter marked as default, or None."""
        for a in self.values():
            if a.is_default:
                return a
        return None


def load_adapter_registry(path: str | Path) -> AdapterRegistry:
    """
    Load adapter definitions from a JSON file.

    JSON format::

        {
            "legal-tone": {
                "lora_id": 0,
                "gguf_path": "/adapters/legal-tone.gguf",
                "scale": 1.0,
                "description": "Formal legal writing style",
                "tags": ["legal", "formal"],
                "keywords": ["contract", "lawsuit", "clause", "legal"],
                "is_default": false
            },
            ...
        }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter registry not found: {path}")

    with open(path) as f:
        data = json.load(f)

    registry = AdapterRegistry()
    for name, defn in data.items():
        registry[name] = AdapterDef(
            name=name,
            lora_id=defn.get("lora_id", 0),
            gguf_path=defn.get("gguf_path", ""),
            scale=defn.get("scale", 1.0),
            description=defn.get("description", ""),
            tags=defn.get("tags", []),
            keywords=defn.get("keywords", []),
            is_default=defn.get("is_default", False),
        )

    return registry
