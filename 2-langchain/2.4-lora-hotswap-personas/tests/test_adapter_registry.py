"""Tests for AdapterRegistry."""

import json
import tempfile
from pathlib import Path

import pytest

from adapters import AdapterRegistry, AdapterDef, load_adapter_registry


SAMPLE_REGISTRY = {
    "legal-tone": {
        "lora_id": 0,
        "gguf_path": "/adapters/legal-tone.gguf",
        "scale": 1.0,
        "description": "Formal legal writing",
        "tags": ["legal", "formal"],
        "keywords": ["contract", "lawsuit", "clause"],
        "is_default": False,
    },
    "concise-summarizer": {
        "lora_id": 1,
        "gguf_path": "/adapters/concise.gguf",
        "scale": 0.8,
        "description": "Concise summarizer",
        "tags": ["summarization", "concise"],
        "keywords": ["summarize", "tl;dr"],
        "is_default": True,
    },
}


class TestAdapterRegistry:
    def test_load_from_json(self):
        """Load adapter registry from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_REGISTRY, f)
            tmp_path = f.name

        try:
            registry = load_adapter_registry(tmp_path)
            assert len(registry) == 2
            assert "legal-tone" in registry
            assert "concise-summarizer" in registry
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_find_by_tag(self):
        """Find adapters by tag."""
        registry = AdapterRegistry()
        registry["legal"] = AdapterDef(
            name="legal", lora_id=0, gguf_path="/test.gguf",
            tags=["legal", "formal"],
        )
        registry["chat"] = AdapterDef(
            name="chat", lora_id=1, gguf_path="/test2.gguf",
            tags=["chat", "general"],
        )

        matches = registry.find_by_tag("legal")
        assert len(matches) == 1
        assert matches[0].name == "legal"

    def test_find_by_keyword(self):
        """Find adapters by keyword matching."""
        registry = AdapterRegistry()
        registry["legal"] = AdapterDef(
            name="legal", lora_id=0, gguf_path="/test.gguf",
            keywords=["contract", "lawsuit"],
        )

        matches = registry.find_by_keyword("I need to review this contract")
        assert len(matches) == 1
        assert matches[0].name == "legal"

    def test_find_by_keyword_scored(self):
        """Adapters with more keyword matches appear first."""
        registry = AdapterRegistry()
        registry["chat"] = AdapterDef(
            name="chat", lora_id=0, gguf_path="/chat.gguf",
            keywords=["hello"],
        )
        registry["legal"] = AdapterDef(
            name="legal", lora_id=1, gguf_path="/legal.gguf",
            keywords=["contract", "lawsuit", "liability"],
        )

        matches = registry.find_by_keyword(
            "This contract has liability issues, hello world"
        )
        assert len(matches) == 2
        assert matches[0].name == "legal"  # 2 keyword matches (contract, liability)
        assert matches[1].name == "chat"   # 1 keyword match (hello)

    def test_get_default(self):
        """get_default returns the adapter marked as default."""
        registry = AdapterRegistry()
        registry["a"] = AdapterDef(
            name="a", lora_id=0, gguf_path="/a.gguf",
            is_default=False,
        )
        registry["b"] = AdapterDef(
            name="b", lora_id=1, gguf_path="/b.gguf",
            is_default=True,
        )

        default = registry.get_default()
        assert default is not None
        assert default.name == "b"

    def test_get_default_none(self):
        """get_default returns None if no adapter is marked default."""
        registry = AdapterRegistry()
        registry["a"] = AdapterDef(
            name="a", lora_id=0, gguf_path="/a.gguf",
        )
        assert registry.get_default() is None

    def test_load_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_adapter_registry("/nonexistent/registry.json")
