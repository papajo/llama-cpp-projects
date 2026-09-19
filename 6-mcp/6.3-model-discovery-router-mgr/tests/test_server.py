"""Tests for MCP server tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from model_router import server as server_mod
from model_router.registry import ModelRegistry
from model_router.server import get_model_info, list_models, route_request

# The real chat server's /v1/models payload shape: exactly one model, owned_by
# "llamacpp", with a meta block. These tests previously mocked a "gpt-4" /
# owned_by "openai" entry, which no llama-server ever returns.
# See drift-graph.md entry 4.
CHAT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0"

MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {
            "id": CHAT_MODEL_ID,
            "aliases": [CHAT_MODEL_ID],
            "tags": [],
            "object": "model",
            "created": 1789829232,
            "owned_by": "llamacpp",
            "meta": {
                "vocab_type": True, "n_vocab": 49152, "n_ctx": 2048,
                "n_ctx_train": 8192, "n_embd": 960, "n_params": 361821120,
                "size": 384618240, "ftype": "Q8_0",
            },
        }
    ],
    "models": [
        {
            "name": CHAT_MODEL_ID, "model": CHAT_MODEL_ID, "modified_at": "",
            "size": "", "digest": "", "type": "model", "description": "",
            "tags": [""], "capabilities": ["completion"], "parameters": "",
            "details": {
                "parent_model": "", "format": "gguf", "family": "",
                "families": [""], "parameter_size": "", "quantization_level": "",
            },
        }
    ],
}


@pytest.fixture(autouse=True)
def reset_registry():
    """Give every test a fresh registry.

    model_router.server holds a module-global `_registry` and the tools call
    discover_if_empty(). Without this reset the first test to run populates the
    cache and every later test is served from it, ignoring its own mock.
    """
    original = server_mod._registry
    server_mod._registry = ModelRegistry()
    yield
    server_mod._registry = original


def _mock_urlopen(body: dict):
    mc = MagicMock()
    mc.read.return_value = json.dumps(body).encode()
    cm = MagicMock()
    cm.__enter__.return_value = mc
    return cm


class TestServerTools:
    def test_list_models(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(
                MODELS_RESPONSE
            ).__enter__.return_value
            result = list_models()
            assert CHAT_MODEL_ID in result
            # "...-Instruct-GGUF:Q8_0" contains "instruct", so infer_type -> chat
            assert "type: chat" in result

    def test_get_model_info_found(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(
                MODELS_RESPONSE
            ).__enter__.return_value
            result = get_model_info(CHAT_MODEL_ID)
            assert CHAT_MODEL_ID in result
            assert "llamacpp" in result
            assert "Type: chat" in result

    def test_get_model_info_not_found(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(
                MODELS_RESPONSE
            ).__enter__.return_value
            result = get_model_info("gpt-4")
            assert "not found" in result

    def test_route_request(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(
                MODELS_RESPONSE
            ).__enter__.return_value
            result = route_request(task_type="chat")
            assert "Routed" in result
            assert CHAT_MODEL_ID in result
