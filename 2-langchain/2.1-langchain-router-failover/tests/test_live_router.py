"""Live integration tests for 2.1 against the real chat server.

Mirrors the httpx-mocked paths in test_router.py, but over a real socket to a
real llama-server: invoke, stream, their async twins, message conversion,
response parsing and the fallback chain.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

Two limits of this environment shape the file:

* There is ONE model behind the endpoint, and llama-server silently serves it
  whatever `model` you ask for. So registering several aliases exercises the
  routing *logic* faithfully, but every alias lands on SmolLM2 -- the real
  multi-model router (llama-swap et al.) is not what is running here.
* Nothing ever returns 503/504, so the cold-start retry path cannot be
  triggered live. It stays covered by the offline tests, and the reason is
  documented in test_cold_start_path_is_not_reproducible_live.

max_tokens is kept at 8-16 throughout, and no assertion touches what the model
actually says.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_llamacpp_router.exceptions import (
    ModelNotAvailableError,
    NoSuitableModelError,
)
from langchain_llamacpp_router.router import RouterModel, RunnableRouter


@pytest.fixture
def router(chat_base_url):
    """A router with two aliases registered, pointed at the real server."""
    r = RunnableRouter(base_url=chat_base_url, default_max_tokens=16)
    r.register_model(RouterModel(alias="coder", tags=["code", "fast"]))
    r.register_model(RouterModel(alias="chat", tags=["chat", "creative"]))
    yield r
    r.close()


# ---------------------------------------------------------------------------
# Routing logic, with real registrations
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_routes_by_tag_and_alias(router):
    from langchain_llamacpp_router.router import RoutingConfig

    assert router._select_model(RoutingConfig(routing_key="creative")) == "chat"
    assert router._select_model(RoutingConfig(routing_key="fast")) == "coder"
    assert router._select_model(RoutingConfig(model_alias="chat")) == "chat"


@pytest.mark.live
def test_unregistered_alias_is_rejected_before_any_request(router):
    """The router must fail locally, not send a doomed request.

    Worth asserting live: llama-server would happily answer a request naming
    an unknown model (see test_server_ignores_the_model_field), so this guard
    is the only thing that catches the mistake.
    """
    from langchain_llamacpp_router.router import RoutingConfig

    with pytest.raises(ModelNotAvailableError):
        router._select_model(RoutingConfig(model_alias="nope"))


@pytest.mark.live
def test_unmatched_tag_is_rejected(router):
    from langchain_llamacpp_router.router import RoutingConfig

    with pytest.raises(NoSuitableModelError):
        router._select_model(RoutingConfig(routing_key="no-such-tag"))


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_invoke_returns_ai_message(router):
    msg = router.invoke(
        [HumanMessage(content="Say hi")], config={"routing_key": "fast"}
    )
    assert isinstance(msg, AIMessage)
    assert isinstance(msg.content, str)
    assert msg.content.strip()


@pytest.mark.live
def test_response_metadata_is_populated(router):
    msg = router.invoke(
        [HumanMessage(content="Say hi")], config={"model_alias": "coder"}
    )
    meta = msg.response_metadata
    assert meta["finish_reason"] in {"stop", "length"}
    usage = meta["usage"]
    assert set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.live
def test_server_ignores_the_model_field(router, chat_model):
    """The response reports the LOADED model, not the alias that was sent.

    The router asks for `model: "coder"`; llama-server answers as
    SmolLM2. A mocked test that echoed the alias back in `model` would be
    asserting behaviour the real server does not have. See drift-rag.md.
    """
    msg = router.invoke(
        [HumanMessage(content="Say hi")], config={"model_alias": "coder"}
    )
    assert msg.response_metadata["model"] == chat_model
    assert msg.response_metadata["model"] != "coder"


@pytest.mark.live
def test_system_message_is_accepted(router):
    """SystemMessage -> role "system" survives the real chat template."""
    msg = router.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say hi"),
        ],
        config={"routing_key": "fast"},
    )
    assert isinstance(msg.content, str)
    assert msg.content.strip()


@pytest.mark.live
def test_max_tokens_is_honoured(router):
    msg = router.invoke(
        [HumanMessage(content="Say hi")],
        config={"model_alias": "coder"},
        max_tokens=8,
    )
    assert msg.response_metadata["usage"]["completion_tokens"] <= 8


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_fallback_chain_uses_primary_when_healthy(router):
    """With a healthy primary, the fallback is never reached."""
    from langchain_llamacpp_router.router import RoutingConfig

    chain = router._get_fallback_chain(
        RoutingConfig(model_alias="coder", fallback_aliases=["chat"]), "coder"
    )
    assert chain == ["coder", "chat"]

    msg = router.invoke(
        [HumanMessage(content="Say hi")],
        config={"model_alias": "coder", "fallback_aliases": ["chat"]},
    )
    assert msg.content.strip()


@pytest.mark.live
def test_unregistered_fallbacks_are_dropped_from_the_chain(router):
    from langchain_llamacpp_router.router import RoutingConfig

    chain = router._get_fallback_chain(
        RoutingConfig(model_alias="coder", fallback_aliases=["ghost", "chat"]),
        "coder",
    )
    assert chain == ["coder", "chat"]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_stream_yields_text_chunks(router):
    """Real SSE, parsed by _parse_stream_chunk.

    The real stream both opens and closes with chunks that carry no text: the
    first is {"role": "assistant", "content": null} and the last is
    {"delta": {}} with the finish_reason. Both are correctly skipped, so the
    collected text is the message and nothing else. See drift-rag.md.
    """
    chunks = list(
        router.stream([HumanMessage(content="Say hi")], config={"routing_key": "fast"})
    )
    assert chunks, "stream produced no chunks"
    for c in chunks:
        assert isinstance(c.content, str)
    # Any empty chunk here is LangChain's own terminal sentinel
    # (chunk_position="last"), appended by BaseChatModel.stream -- not
    # something _parse_stream_chunk emitted.
    for c in chunks:
        if c.content == "":
            assert c.chunk_position == "last"
    assert "".join(c.content for c in chunks).strip()


@pytest.mark.live
def test_streamed_text_is_comparable_to_invoke(router):
    """Both paths return non-empty text for the same prompt.

    Only shape is compared. At the server's default sampling the two calls
    will not produce identical strings, and this suite does not pretend
    otherwise.
    """
    streamed = "".join(
        c.content
        for c in router.stream(
            [HumanMessage(content="Say hi")], config={"model_alias": "coder"}
        )
    )
    invoked = router.invoke(
        [HumanMessage(content="Say hi")], config={"model_alias": "coder"}
    ).content
    assert streamed.strip()
    assert invoked.strip()


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


# pytest-asyncio is not installed in the shared venv (and requirements are
# shared with the other workers), so these drive the coroutines with
# asyncio.run() from ordinary sync tests rather than adding a plugin. The
# offline suite covers no async path at all, so _agenerate/_astream are only
# exercised here.


@pytest.mark.live
def test_ainvoke_returns_ai_message(router):
    async def go():
        try:
            return await router.ainvoke(
                [HumanMessage(content="Say hi")], config={"routing_key": "fast"}
            )
        finally:
            await router.aclose()

    msg = asyncio.run(go())
    assert isinstance(msg, AIMessage)
    assert msg.content.strip()


@pytest.mark.live
def test_astream_yields_text_chunks(router):
    async def go():
        collected = []
        try:
            async for c in router.astream(
                [HumanMessage(content="Say hi")], config={"routing_key": "fast"}
            ):
                collected.append(c.content)
        finally:
            await router.aclose()
        return collected

    collected = asyncio.run(go())
    assert collected
    assert "".join(collected).strip()


# ---------------------------------------------------------------------------
# Error paths against the real server
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_real_400_is_raised_not_treated_as_cold_start(router):
    """A malformed request must surface, not be silently retried.

    llama-server answers a bad request with
    {"error": {"code": 400, "message": ..., "type": "invalid_request_error"}}.
    _is_cold_start_error looks for "cold"/"sleep"/"unavailable" in the error
    body, none of which appear, so this correctly raises instead of burning
    three retries.
    """
    resp = router._get_client().post(
        "/v1/chat/completions", json={"model": "coder", "messages": "not-a-list"}
    )
    assert resp.status_code == 400
    assert router._is_cold_start_error(resp) is False
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == 400


@pytest.mark.live
def test_unreachable_router_raises_connection_error(chat_base_url):
    """A closed port must become RouterConnectionError, not a bare httpx error."""
    from langchain_llamacpp_router.exceptions import RouterConnectionError

    # Port 1 is reserved and nothing listens on it.
    r = RunnableRouter(base_url="http://127.0.0.1:1", request_timeout=5.0)
    r.register_model(RouterModel(alias="coder", tags=["fast"]))
    try:
        with pytest.raises(RouterConnectionError):
            r.invoke([HumanMessage(content="Say hi")], config={"model_alias": "coder"})
    finally:
        r.close()


@pytest.mark.live
def test_cold_start_path_is_not_reproducible_live(router):
    """Documents why cold-start stays mock-only.

    The retry logic triggers on HTTP 503/504, which a plain single-model
    llama-server never emits -- there is no model to wake, because it is
    already resident. Reproducing it needs a routing front-end that unloads
    idle models (llama-swap, or llama-server's own router mode with
    --sleep-idle-seconds). Unsupported here, so the offline tests remain the
    only coverage; this test just pins the fact that the server is healthy.
    """
    resp = router._get_client().post(
        "/v1/chat/completions",
        json={
            "model": "coder",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 4,
        },
    )
    assert resp.status_code == 200
    assert router._is_cold_start_error(resp) is False
