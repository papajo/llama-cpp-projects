"""Tests for model registry and routing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from model_router.registry import (
    ModelInfo,
    ModelRegistry,
    RoutingStrategy,
)


def _mock_models_response(models_data: list) -> MagicMock:
    resp = {"data": models_data}
    mc = MagicMock()
    mc.read.return_value = json.dumps(resp).encode()
    return mc


SAMPLE_MODELS = [
    {"id": "gpt-4", "object": "model", "owned_by": "openai"},
    {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
    {"id": "text-embedding-ada-002", "object": "model", "owned_by": "openai"},
    {"id": "llama-3-70b-instruct", "object": "model", "owned_by": "meta"},
]


class TestModelInfo:
    def test_from_openai_response(self):
        data = {"id": "gpt-4", "object": "model", "owned_by": "openai"}
        info = ModelInfo.from_openai_response(data)
        assert info.id == "gpt-4"
        assert info.owned_by == "openai"


class TestModelRegistry:
    def test_discover(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_models_response(SAMPLE_MODELS)
            registry = ModelRegistry()
            models = registry.discover()
            assert len(models) == 4
            assert registry.get_model("gpt-4") is not None
            assert registry.get_model("nonexistent") is None

    def test_list_models(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_models_response(SAMPLE_MODELS)
            registry = ModelRegistry()
            registry.discover()
            models = registry.list_models()
            assert len(models) == 4

    def test_route_first_available(self):
        registry = ModelRegistry()
        registry.models = {"a": ModelInfo("a"), "b": ModelInfo("b")}
        model_id = registry.route(RoutingStrategy.FIRST_AVAILABLE)
        assert model_id == "a"

    def test_route_round_robin(self):
        registry = ModelRegistry()
        registry.models = {"a": ModelInfo("a"), "b": ModelInfo("b")}
        first = registry.route(RoutingStrategy.ROUND_ROBIN)
        second = registry.route(RoutingStrategy.ROUND_ROBIN)
        third = registry.route(RoutingStrategy.ROUND_ROBIN)
        assert first == "a"
        assert second == "b"
        assert third == "a"  # wraps around

    def test_route_by_name(self):
        registry = ModelRegistry()
        registry.models = {"gpt-4": ModelInfo("gpt-4")}
        model_id = registry.route(RoutingStrategy.BY_NAME, preferred_model="gpt-4")
        assert model_id == "gpt-4"
        # Nonexistent preferred
        model_id = registry.route(RoutingStrategy.BY_NAME, preferred_model="nonexistent")
        assert model_id is None

    def test_route_empty(self):
        registry = ModelRegistry()
        assert registry.route() is None

    def test_infer_type(self):
        assert ModelRegistry.infer_type("text-embedding-ada-002") == "embedding"
        assert ModelRegistry.infer_type("llama-3-70b-instruct") == "chat"
        assert ModelRegistry.infer_type("gpt-4") == "completion"
        assert ModelRegistry.infer_type("code-llama") == "completion"

    def test_discover_if_empty(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_models_response(SAMPLE_MODELS)
            registry = ModelRegistry()
            models = registry.discover_if_empty()
            assert len(models) == 4
            # Second call should return cached
            models2 = registry.discover_if_empty()
            assert len(models2) == 4

    def test_discover_error(self):
        with patch("urllib.request.urlopen") as m:
            from urllib.error import URLError
            m.side_effect = URLError("connection refused")
            registry = ModelRegistry()
            with pytest.raises(Exception, match="Failed to discover"):
                registry.discover()
