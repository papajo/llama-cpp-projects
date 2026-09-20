"""Shared live-server pytest fixtures, loaded as a global plugin.

Registered via PYTEST_PLUGINS=_shared.live_fixtures (see env.sh) so it applies
to every project regardless of that project's own pytest rootdir.

The offline unit tests intercept HTTP in-process (unittest.mock on
urllib.request.urlopen, or pytest-httpx's httpx_mock) and never open a socket.
This file adds the opposite layer: fixtures that talk to the REAL llama-server
instances, so drift between the canned responses and actual server behaviour
becomes visible.

Live tests are opt-in. They run only when LLM_LIVE=1 is set; otherwise they are
skipped, keeping the default suite fast and offline.

Endpoints come from env (see env.sh), never hardcoded per project:
    LLM_CHAT_BASE_URL    default http://127.0.0.1:8090
    LLM_EMBED_BASE_URL   default http://127.0.0.1:8081
    LLM_RERANK_BASE_URL  default http://127.0.0.1:8082 (optional)

Port 8080 is Open WebUI, not llama.cpp. Nothing here may point at it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

CHAT_BASE_URL = os.environ.get("LLM_CHAT_BASE_URL", "http://127.0.0.1:8090")
EMBED_BASE_URL = os.environ.get("LLM_EMBED_BASE_URL", "http://127.0.0.1:8081")
CHAT_MODEL = os.environ.get(
    "LLM_CHAT_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0"
)
EMBED_MODEL = os.environ.get(
    "LLM_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"
)
# Cross-encoder reranker. It needs its own server because --reranking forces
# pooling to "rank", which corrupts /v1/embeddings on the same process.
RERANK_BASE_URL = os.environ.get("LLM_RERANK_BASE_URL", "http://127.0.0.1:8082")
RERANK_MODEL = os.environ.get(
    "LLM_RERANK_MODEL", "gpustack/bge-reranker-v2-m3-GGUF:Q8_0"
)

_FORBIDDEN_PORT = "8080"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test hits a real llama-server; requires LLM_LIVE=1",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LLM_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live server tests need LLM_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    """POST JSON and return the decoded response."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _require_llamacpp(base_url: str, label: str) -> None:
    """Fail loudly if the URL is not a real llama-server."""
    if _FORBIDDEN_PORT in base_url:
        pytest.fail(
            f"{label} base url {base_url} points at port {_FORBIDDEN_PORT}, "
            "which is Open WebUI, not llama.cpp. Results would be meaningless."
        )
    try:
        health = _get(f"{base_url}/health", timeout=5.0)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(f"{label} server at {base_url} is not reachable: {exc}")
    # Real llama-server reports {"status": "ok"}; Open WebUI reports
    # {"status": true}. Guard against pointing at the wrong service.
    if health.get("status") != "ok":
        pytest.fail(
            f"{label} server at {base_url} returned health {health!r}, "
            "which is not a llama-server response."
        )


@pytest.fixture(scope="session")
def chat_base_url() -> str:
    _require_llamacpp(CHAT_BASE_URL, "chat")
    return CHAT_BASE_URL


@pytest.fixture(scope="session")
def embed_base_url() -> str:
    _require_llamacpp(EMBED_BASE_URL, "embed")
    return EMBED_BASE_URL


@pytest.fixture(scope="session")
def rerank_base_url() -> str:
    """The cross-encoder rerank server; skips when it is not running."""
    _require_llamacpp(RERANK_BASE_URL, "rerank")
    return RERANK_BASE_URL


@pytest.fixture(scope="session")
def rerank_model() -> str:
    return RERANK_MODEL


@pytest.fixture(scope="session")
def chat_model() -> str:
    return CHAT_MODEL


@pytest.fixture(scope="session")
def embed_model() -> str:
    return EMBED_MODEL


@pytest.fixture
def live_chat(chat_base_url, chat_model):
    """Call the real chat server. Returns the raw decoded JSON response."""

    def _call(messages, **kw):
        payload = {"model": chat_model, "messages": messages, **kw}
        payload.setdefault("max_tokens", 64)
        return _post(f"{chat_base_url}/v1/chat/completions", payload)

    return _call


@pytest.fixture
def live_embed(embed_base_url, embed_model):
    """Call the real embeddings server. Returns the raw decoded JSON response."""

    def _call(text, **kw):
        payload = {"model": embed_model, "input": text, **kw}
        return _post(f"{embed_base_url}/v1/embeddings", payload)

    return _call


@pytest.fixture
def live_rerank(rerank_base_url, rerank_model):
    """Call the real cross-encoder. Returns the raw decoded JSON response.

    Scores are the model's raw logits, NOT normalised to 0-1 the way hosted
    rerank APIs (Cohere, Jina) return them.
    """

    def _call(query, documents, **kw):
        payload = {"model": rerank_model, "query": query,
                   "documents": list(documents), **kw}
        return _post(f"{rerank_base_url}/v1/rerank", payload)

    return _call


@pytest.fixture(scope="session")
def live_server_meta(chat_base_url, embed_base_url):
    """/v1/models metadata for both servers, for capability assertions."""
    return {
        "chat": _get(f"{chat_base_url}/v1/models"),
        "embed": _get(f"{embed_base_url}/v1/models"),
    }
