"""Tests for model registry and routing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from model_router.registry import (
    ModelInfo,
    ModelRegistry,
    RoutingStrategy,
)



# ---------------------------------------------------------------------------
# Real llama-server /v1/models payload (build b11046-60081bb2b).
#
# These tests used to mock a four-model OpenAI cloud catalogue (gpt-4,
# gpt-3.5-turbo, text-embedding-ada-002, llama-3-70b-instruct) with
# owned_by "openai"/"meta". A llama-server process serves exactly ONE model,
# reports owned_by "llamacpp", ships a per-model "meta" block carrying n_ctx,
# and includes an Ollama-style "models" list beside the OpenAI "data" list.
# See drift-graph.md entry 4.
# ---------------------------------------------------------------------------

CHAT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0"
EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"


def _model_entry(model_id: str, n_ctx: int = 2048, n_embd: int = 960) -> dict:
    """One entry of the OpenAI-style "data" list, as llama-server sends it."""
    return {
        "id": model_id,
        "aliases": [model_id],
        "tags": [],
        "object": "model",
        "created": 1789829232,
        "owned_by": "llamacpp",
        "meta": {
            "vocab_type": True,
            "n_vocab": 49152,
            "n_ctx": n_ctx,
            "n_ctx_train": 8192,
            "n_embd": n_embd,
            "n_params": 361821120,
            "size": 384618240,
            "ftype": "Q8_0",
        },
    }


def _models_payload(*model_ids: str) -> dict:
    """A full /v1/models body, including the Ollama-style sibling list."""
    ids = model_ids or (CHAT_MODEL_ID,)
    return {
        "models": [
            {
                "name": mid, "model": mid, "modified_at": "", "size": "",
                "digest": "", "type": "model", "description": "", "tags": [""],
                "capabilities": ["completion"], "parameters": "",
                "details": {
                    "parent_model": "", "format": "gguf", "family": "",
                    "families": [""], "parameter_size": "",
                    "quantization_level": "",
                },
            }
            for mid in ids
        ],
        "object": "list",
        "data": [_model_entry(mid) for mid in ids],
    }



def _mock_models_response(payload: dict) -> MagicMock:
    mc = MagicMock()
    mc.read.return_value = json.dumps(payload).encode()
    return mc


# The real chat server: a single model. Routing strategies that need more than
# one model build their registries directly, without HTTP.
SAMPLE_PAYLOAD = _models_payload(CHAT_MODEL_ID)


class TestModelInfo:
    def test_from_openai_response(self):
        info = ModelInfo.from_openai_response(_model_entry(CHAT_MODEL_ID))
        assert info.id == CHAT_MODEL_ID
        assert info.owned_by == "llamacpp"
        assert info.context_length == 2048

    def test_from_openai_response_without_meta(self):
        """A payload with no meta block (plain OpenAI) leaves context_length 0."""
        info = ModelInfo.from_openai_response(
            {"id": "gpt-4", "object": "model", "owned_by": "openai"}
        )
        assert info.id == "gpt-4"
        assert info.owned_by == "openai"
        assert info.context_length == 0


class TestModelRegistry:
    def test_discover(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_models_response(SAMPLE_PAYLOAD)
            registry = ModelRegistry()
            models = registry.discover()
            assert len(models) == 1
            info = registry.get_model(CHAT_MODEL_ID)
            assert info is not None
            assert info.owned_by == "llamacpp"
            assert info.context_length == 2048  # from meta.n_ctx
            assert registry.get_model("gpt-4") is None
            assert registry.get_model("nonexistent") is None

    def test_list_models(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_models_response(SAMPLE_PAYLOAD)
            registry = ModelRegistry()
            registry.discover()
            models = registry.list_models()
            assert len(models) == 1

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
        # The real, HF-style ids the live servers report.
        assert ModelRegistry.infer_type(CHAT_MODEL_ID) == "chat"
        assert ModelRegistry.infer_type(EMBED_MODEL_ID) == "embedding"

    def test_discover_if_empty(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_models_response(SAMPLE_PAYLOAD)
            registry = ModelRegistry()
            models = registry.discover_if_empty()
            assert len(models) == 1
            # Second call should return cached
            models2 = registry.discover_if_empty()
            assert len(models2) == 1

    def test_discover_error(self):
        with patch("urllib.request.urlopen") as m:
            from urllib.error import URLError
            m.side_effect = URLError("connection refused")
            registry = ModelRegistry()
            with pytest.raises(Exception, match="Failed to discover"):
                registry.discover()
