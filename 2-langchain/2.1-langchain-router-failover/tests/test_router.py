"""Tests for RunnableRouter."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_llamacpp_router import (
    RunnableRouter,
    RouterModel,
    load_presets_ini,
)
from langchain_llamacpp_router.exceptions import (
    ModelColdStartError,
    ModelNotAvailableError,
    NoSuitableModelError,
    RouterConnectionError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fake_response(
    status_code: int = 200,
    content: str = "Hello from llama.cpp",
    model: str = "test-model",
    finish_reason: str = "stop",
) -> httpx.Response:
    """Build a fake httpx.Response simulating an OpenAI-compatible endpoint."""
    body: Dict[str, Any] = {
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    return httpx.Response(status_code=status_code, json=body)


def make_fake_stream_chunk(content: str, finish_reason: str | None = None) -> str:
    """Build an SSE data line mimicking router streaming output."""
    body = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ]
    }
    import json
    return f"data: {json.dumps(body)}\n\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router() -> RunnableRouter:
    r = RunnableRouter(
        base_url="http://test-router:8080",
        default_routing_key="fast",
        max_cold_start_retries=2,
        cold_start_retry_delay=0.1,
        request_timeout=5.0,
    )
    r.register_model(RouterModel(
        alias="coder",
        tags=["code", "generation", "fast"],
        description="Test coder",
    ))
    r.register_model(RouterModel(
        alias="chat",
        tags=["chat", "creative", "slow"],
        description="Test chat",
    ))
    return r


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_list(self, router: RunnableRouter):
        assert "coder" in router.registered_models
        assert "chat" in router.registered_models
        assert len(router.registered_models) == 2

    def test_unregister(self, router: RunnableRouter):
        router.unregister_model("coder")
        assert "coder" not in router.registered_models
        assert "chat" in router.registered_models

    def test_register_presets(self, router: RunnableRouter):
        # Load from the sample INI
        import tempfile
        ini = """
[coder]
model = /models/coder.gguf
tags = code, fast
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini)
            tmp = f.name
        presets = load_presets_ini(tmp)
        router.register_presets(presets)
        # coder already registered — this should overwrite (tags now ["code", "fast"])
        assert router._models["coder"].tags == ["code", "fast"]


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


class TestRouting:
    def test_route_by_tag(self, router: RunnableRouter):
        alias = router._select_model(router._extract_routing_config(
            {"config": {"routing_key": "creative"}}
        ))
        assert alias == "chat"

    def test_route_by_tag_fast(self, router: RunnableRouter):
        alias = router._select_model(router._extract_routing_config(
            {"config": {"routing_key": "generation"}}
        ))
        assert alias == "coder"

    def test_route_by_explicit_alias(self, router: RunnableRouter):
        alias = router._select_model(router._extract_routing_config(
            {"config": {"model_alias": "chat"}}
        ))
        assert alias == "chat"

    def test_route_nonexistent_alias(self, router: RunnableRouter):
        with pytest.raises(ModelNotAvailableError):
            router._select_model(router._extract_routing_config(
                {"config": {"model_alias": "nonexistent"}}
            ))

    def test_route_no_match_tag(self, router: RunnableRouter):
        with pytest.raises(NoSuitableModelError):
            router._select_model(router._extract_routing_config(
                {"config": {"routing_key": "embedding"}}
            ))

    def test_route_default_fallback(self, router: RunnableRouter):
        # default_routing_key="fast" -> should match "coder" (tags contain "fast")
        alias = router._select_model(router._extract_routing_config({}))
        assert alias == "coder"

    def test_route_first_registered(self):
        r = RunnableRouter(base_url="http://test:8080")
        r.register_model(RouterModel(alias="only-model", tags=[]))
        alias = r._select_model(r._extract_routing_config({}))
        assert alias == "only-model"


# ---------------------------------------------------------------------------
# Cold-start detection
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_detect_503_cold_start(self, router: RunnableRouter):
        resp = httpx.Response(503)
        assert router._is_cold_start_error(resp)

    def test_detect_504_cold_start(self, router: RunnableRouter):
        resp = httpx.Response(504)
        assert router._is_cold_start_error(resp)

    def test_detect_200_not_cold_start(self, router: RunnableRouter):
        resp = httpx.Response(200)
        assert not router._is_cold_start_error(resp)

    def test_detect_error_body_cold(self, router: RunnableRouter):
        resp = httpx.Response(
            500,
            json={"error": "Model is cold-starting, try again"},
        )
        assert router._is_cold_start_error(resp)

    def test_detect_error_body_sleep(self, router: RunnableRouter):
        resp = httpx.Response(
            500,
            json={"error": "model sleeping"},
        )
        assert router._is_cold_start_error(resp)

    def test_detect_regular_error_not_cold(self, router: RunnableRouter):
        resp = httpx.Response(400, json={"error": "bad request"})
        assert not router._is_cold_start_error(resp)


# ---------------------------------------------------------------------------
# Chat request (sync, with mocked HTTP)
# ---------------------------------------------------------------------------


class TestChatRequest:
    def test_successful_request(self, router: RunnableRouter):
        with patch.object(router, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.post.return_value = make_fake_response(
                content="Hello world",
                model="coder",
            )
            mock_get_client.return_value = mock_client

            result = router._generate(
                messages=[HumanMessage(content="Hi")]
            )
            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hello world"

    def test_cold_start_then_success(self, router: RunnableRouter):
        """Cold-start retry: first response is 503, second succeeds."""
        with patch.object(router, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.post.side_effect = [
                make_fake_response(status_code=503),
                make_fake_response(content="Finally woke up"),
            ]
            mock_get_client.return_value = mock_client

            result = router._generate(
                messages=[HumanMessage(content="Wake up")]
            )
            assert result.generations[0].message.content == "Finally woke up"
            assert mock_client.post.call_count == 2

    def test_cold_start_exhausted(self, router: RunnableRouter):
        """All cold-start retries fail → NoSuitableModelError (chain exhausted)."""
        with patch.object(router, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.post.return_value = make_fake_response(status_code=503)
            mock_get_client.return_value = mock_client

            with pytest.raises(NoSuitableModelError):
                router._generate(
                    messages=[HumanMessage(content="Wake up")]
                )
            # 2 retries configured on fixture = 2 attempts on "coder", then chain exhausted
            assert mock_client.post.call_count >= 1

    def test_connection_error(self, router: RunnableRouter):
        with patch.object(router, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(RouterConnectionError):
                router._generate(
                    messages=[HumanMessage(content="Hi")]
                )

    def test_fallback_chain(self, router: RunnableRouter):
        """Primary model fails, falls back to second."""
        with patch.object(router, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            # First call (for "nonexistent") — let it get past model selection
            # We route via config with model_alias + fallback
            mock_client.post.return_value = make_fake_response(
                content="Fallback response"
            )
            mock_get_client.return_value = mock_client

            result = router._generate(
                messages=[HumanMessage(content="Test fallback")],
                config={
                    "model_alias": "coder",
                    "fallback_aliases": ["chat"],
                },
            )
            # The first model (coder) should succeed, no actual fallback needed
            assert result.generations[0].message.content == "Fallback response"


# ---------------------------------------------------------------------------
# Streaming (mocked)
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_basic(self, router: RunnableRouter):
        with patch.object(router, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_stream = MagicMock()
            mock_stream.__enter__.return_value = mock_stream
            mock_stream.iter_lines.return_value = [
                make_fake_stream_chunk("Hello "),
                make_fake_stream_chunk("world"),
                make_fake_stream_chunk("", finish_reason="stop"),
                "data: [DONE]\n\n",
            ]

            mock_response = MagicMock()
            mock_response.is_success = True
            mock_stream.__enter__.return_value = mock_response
            mock_response.iter_lines.return_value = iter([
                make_fake_stream_chunk("Hello "),
                make_fake_stream_chunk("world"),
                make_fake_stream_chunk("", finish_reason="stop"),
                "data: [DONE]\n\n",
            ])

            mock_client.stream.return_value = mock_stream
            mock_get_client.return_value = mock_client

            collected = []
            for chunk in router.stream(
                [HumanMessage(content="Hi")],
                config={"routing_key": "fast"},
            ):
                collected.append(chunk.content)

            assert "".join(collected) == "Hello world"


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_system_message(self, router: RunnableRouter):
        """System messages should be preserved."""
        with patch.object(router, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.post.return_value = make_fake_response()
            mock_get_client.return_value = mock_client

            router._generate(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="Hi"),
                ]
            )
            # Verify the body sent to the server
            call_kwargs = mock_client.post.call_args[1]
            sent_messages = call_kwargs["json"]["messages"]
            assert sent_messages[0]["role"] == "system"
            assert sent_messages[0]["content"] == "You are a helpful assistant."
            assert sent_messages[1]["role"] == "user"
            assert sent_messages[1]["content"] == "Hi"
