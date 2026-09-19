"""Live integration tests for model discovery and routing.

These exercise the same code paths as the mocked tests in test_registry.py /
test_server.py, but against the real llama-server instances on
$LLM_CHAT_BASE_URL and $LLM_EMBED_BASE_URL.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

Key reality the offline mocks did not reflect (see drift-graph.md):
  * Each llama-server process serves exactly ONE model. There is no
    multi-model endpoint to discover, so `discover()` returns a 1-element list.
  * `owned_by` is always "llamacpp", never "openai"/"meta".
  * The payload carries an Ollama-style "models" list alongside the
    OpenAI-style "data" list, plus a per-model "meta" block.
"""

from __future__ import annotations

import pytest

from model_router import server as server_mod
from model_router.registry import (
    ModelDiscoveryError,
    ModelInfo,
    ModelRegistry,
    RoutingStrategy,
)


@pytest.fixture
def chat_registry(chat_base_url):
    return ModelRegistry(server_url=chat_base_url)


@pytest.fixture
def embed_registry(embed_base_url):
    return ModelRegistry(server_url=embed_base_url)


@pytest.fixture
def fresh_server_registry(chat_base_url):
    """Point the module-level singleton at the real chat server.

    model_router.server keeps a module-global `_registry` that caches via
    discover_if_empty(), so each test must reset it or it leaks across tests.
    """
    original = server_mod._registry
    server_mod._registry = ModelRegistry(server_url=chat_base_url)
    yield server_mod._registry
    server_mod._registry = original


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_discover_chat_server(chat_registry, chat_model):
    """discover() parses the real /v1/models payload."""
    models = chat_registry.discover()

    assert len(models) == 1, "llama-server serves exactly one model per process"
    info = models[0]
    assert isinstance(info, ModelInfo)
    assert info.id == chat_model
    assert info.object == "model"
    assert info.owned_by == "llamacpp"
    assert info.permission == [], "llama-server sends no OpenAI permission list"
    assert chat_registry.get_model(chat_model) is info
    assert chat_registry.get_model("nonexistent") is None


@pytest.mark.live
def test_discover_embed_server(embed_registry, embed_model):
    """The embed server is a separate process with its own single model."""
    models = embed_registry.discover()

    assert len(models) == 1
    assert models[0].id == embed_model
    assert models[0].owned_by == "llamacpp"


@pytest.mark.live
def test_discover_populates_context_length(chat_registry):
    """context_length comes from the real payload's meta.n_ctx."""
    info = chat_registry.discover()[0]
    assert info.context_length == 2048, "chat server is started with --ctx-size 2048"


@pytest.mark.live
def test_real_payload_carries_meta_and_ollama_list(chat_base_url, live_server_meta):
    """Document the dual-shaped payload the mocks flattened away."""
    payload = live_server_meta["chat"]

    # OpenAI-style half, which discover() reads.
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    entry = payload["data"][0]
    assert entry["owned_by"] == "llamacpp"
    assert set(entry["meta"]) >= {"n_ctx", "n_embd", "n_params", "n_vocab", "ftype"}
    assert entry["meta"]["n_ctx"] == 2048

    # Ollama-style half, which has no OpenAI equivalent at all.
    assert isinstance(payload["models"], list)
    assert payload["models"][0]["capabilities"] == ["completion"]


@pytest.mark.live
def test_embed_model_reports_768_dims(live_server_meta):
    """n_embd distinguishes the two servers; the chat model is 960."""
    assert live_server_meta["embed"]["data"][0]["meta"]["n_embd"] == 768
    assert live_server_meta["chat"]["data"][0]["meta"]["n_embd"] == 960


@pytest.mark.live
def test_discover_if_empty_caches(chat_registry):
    """Second call must not re-fetch; it returns the cached list."""
    first = chat_registry.discover_if_empty()
    second = chat_registry.discover_if_empty()
    assert len(first) == len(second) == 1
    assert [m.id for m in first] == [m.id for m in second]


@pytest.mark.live
def test_discover_error_on_unreachable_server():
    """The failure branch of discover(), against a closed port."""
    registry = ModelRegistry(server_url="http://127.0.0.1:1")
    with pytest.raises(ModelDiscoveryError, match="Failed to discover"):
        registry.discover()


# ---------------------------------------------------------------------------
# Routing over the real (single-model) registry
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_route_real_model(chat_registry, chat_model):
    chat_registry.discover()

    assert chat_registry.route(RoutingStrategy.FIRST_AVAILABLE) == chat_model
    assert (
        chat_registry.route(RoutingStrategy.BY_NAME, preferred_model=chat_model)
        == chat_model
    )
    assert chat_registry.route(RoutingStrategy.BY_NAME, preferred_model="gpt-4") is None


@pytest.mark.live
def test_round_robin_over_one_model_is_stable(chat_registry, chat_model):
    """With a single real model, round-robin must keep returning it."""
    chat_registry.discover()
    assert [chat_registry.route(RoutingStrategy.ROUND_ROBIN) for _ in range(3)] == [
        chat_model
    ] * 3


@pytest.mark.live
def test_infer_type_on_real_model_ids(chat_model, embed_model):
    """The real ids are long HF-style paths, not bare names like 'gpt-4'."""
    assert ModelRegistry.infer_type(chat_model) == "chat"
    assert ModelRegistry.infer_type(embed_model) == "embedding"


# ---------------------------------------------------------------------------
# MCP server tools
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_tool_list_models(fresh_server_registry, chat_model):
    out = server_mod.list_models()
    assert chat_model in out
    assert "type: chat" in out


@pytest.mark.live
def test_tool_get_model_info(fresh_server_registry, chat_model):
    out = server_mod.get_model_info(chat_model)
    assert chat_model in out
    assert "llamacpp" in out
    assert "Type: chat" in out


@pytest.mark.live
def test_tool_get_model_info_not_found(fresh_server_registry):
    assert "not found" in server_mod.get_model_info("gpt-4")


@pytest.mark.live
def test_tool_route_request(fresh_server_registry, chat_model):
    out = server_mod.route_request(task_type="chat")
    assert "Routed" in out
    assert chat_model in out
