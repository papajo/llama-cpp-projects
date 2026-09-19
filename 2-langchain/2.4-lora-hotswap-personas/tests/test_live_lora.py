"""Live tests for 2.4 against the real chat server.

No LoRA adapters are loaded on this build: `GET /lora-adapters` returns `[]`
because llama-server was started without `--lora` / `--lora-scaled`. So persona
hot-swapping cannot actually do anything here.

READ THIS BEFORE TRUSTING A GREEN RUN OF THIS FILE:

`POST /lora-adapters` answers `{"success": true}` for *every* body -- id 0, id
99, id -5, a string id, even an entry with no id at all -- while `GET` keeps
returning `[]`. With zero adapters loaded the endpoint is a silent no-op that
always reports success. So `apply_adapter()` "succeeding" against this server
proves nothing whatsoever, and a live test asserting only "apply_adapter did
not raise" would be worse than no test: it would look like passing evidence
that hot-swapping works. The tests below assert that no-op *as the no-op it
is*, and the tests that would need a real adapter are xfailed.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from adapters import load_adapter_registry
from agent import LoraAgent, LoraManager
from persona_router import PersonaRouter

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "adapters" / "registry.json"


@pytest.fixture
def manager(chat_base_url):
    m = LoraManager(base_url=chat_base_url)
    yield m
    m.close()


@pytest.fixture(scope="module")
def registry():
    return load_adapter_registry(REGISTRY_PATH)


@pytest.fixture
def agent(chat_base_url, registry):
    a = LoraAgent(
        router=PersonaRouter(registry),
        base_url=chat_base_url,
        max_tokens=16,
    )
    yield a
    a.close()


# ---------------------------------------------------------------------------
# What is actually loaded
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_no_adapters_are_loaded(manager):
    """GET /lora-adapters returns a JSON ARRAY, and here it is empty.

    The mocked test used to return `{"adapters": []}` -- an object. The real
    endpoint returns a bare array, and `list_adapters` was annotated
    `Dict[str, Any]`. Both corrected; see drift-rag.md.
    """
    adapters = manager.list_adapters()
    assert isinstance(adapters, list)
    assert adapters == [], (
        "adapters are loaded now -- the xfails in this file should be revisited"
    )


# ---------------------------------------------------------------------------
# The silent no-op
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_apply_adapter_reports_success_but_does_nothing(manager):
    """Pins the no-op explicitly so it cannot be mistaken for working hotswap."""
    result = manager.apply_adapter(lora_id=0, scale=1.0)
    assert result == {"success": True}
    # ...and yet nothing was loaded.
    assert manager.list_adapters() == []


@pytest.mark.live
@pytest.mark.parametrize(
    "adapters",
    [
        [{"id": 0, "scale": 1.0}],
        [{"id": 99, "scale": 1.0}],
        [{"id": -5, "scale": 2.0}],
        [{"id": 0}],
        [{"scale": 1.0}],
        [{"id": "notanint", "scale": 1.0}],
        [],
    ],
)
def test_post_lora_adapters_always_succeeds(manager, adapters):
    """Every body is accepted, valid or not. There is no validation to rely on.

    This is why `apply_adapter` returning successfully can never be treated as
    confirmation that a persona was applied.
    """
    assert manager.set_adapters(adapters) == {"success": True}


@pytest.mark.live
def test_non_array_body_is_the_only_rejection(manager, chat_base_url):
    """The one thing the endpoint does validate is that the body is an array."""
    client = httpx.Client(timeout=30.0)
    try:
        resp = client.post(
            f"{chat_base_url}/lora-adapters", json={"id": 0, "scale": 1.0}
        )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "must be an array" in err["message"]
    finally:
        client.close()


@pytest.mark.live
def test_delete_endpoint_does_not_exist_and_falls_back(manager, chat_base_url):
    """DELETE /lora-adapters/<id> is not a route on this build.

    It falls through to the static-file handler and returns 404 "File Not
    Found". `remove_adapter` handles that correctly: it raises internally, then
    catches its own HTTPStatusError and falls back to disable_adapter (POST
    with scale 0), so the caller still gets a result.
    """
    client = httpx.Client(timeout=30.0)
    try:
        raw = client.delete(f"{chat_base_url}/lora-adapters/0")
        assert raw.status_code == 404
        assert raw.json()["error"]["type"] == "not_found_error"
    finally:
        client.close()

    # The fallback path still returns a result rather than propagating the 404.
    assert manager.remove_adapter(lora_id=0) == {"success": True}


@pytest.mark.live
def test_disable_adapter_uses_scale_zero(manager):
    assert manager.disable_adapter(lora_id=0) == {"success": True}


# ---------------------------------------------------------------------------
# Routing logic and the text path, which do work
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_registry_loads_from_disk(registry):
    assert len(registry) > 0
    for name, adapter in registry.items():
        assert adapter.name == name
        assert isinstance(adapter.lora_id, int)
        assert isinstance(adapter.scale, float)


@pytest.mark.live
def test_router_selects_personas(registry):
    router = PersonaRouter(registry)
    personas = router.list_personas()
    assert len(personas) == len(registry)
    # Keyword routing must always resolve to something in the registry.
    for text in ["review this contract", "write me a poem", "hello there"]:
        assert router.route_by_keywords(text).name in registry


@pytest.mark.live
def test_agent_generates_text_against_the_real_server(agent):
    """LoraAgent._generate is a normal chat call once the (no-op) POST is done.

    So the agent does produce real output -- it is just output from the base
    model, with no persona applied. Nothing here asserts tone or style: with no
    adapter loaded there is no persona to detect, and SmolLM2-360M could not
    reliably demonstrate one anyway.
    """
    msg = agent.invoke([HumanMessage(content="Say hi")])
    assert isinstance(msg, AIMessage)
    assert isinstance(msg.content, str)
    assert msg.content.strip()


@pytest.mark.live
def test_switch_persona_updates_local_state(agent, registry):
    """switch_persona tracks state locally; the server side is the no-op."""
    name = next(iter(registry))
    agent.switch_persona(name)
    assert agent.current_persona == name


@pytest.mark.live
def test_switch_to_unknown_persona_raises(agent):
    with pytest.raises(KeyError):
        agent.switch_persona("no-such-persona")


# ---------------------------------------------------------------------------
# What cannot be verified here
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.xfail(
    reason="unsupported: no --lora adapters loaded, so GET /lora-adapters "
    "stays [] no matter what is applied. Needs llama-server started with "
    "--lora <adapter.gguf>.",
    strict=True,
)
def test_applied_adapter_appears_in_the_listing(manager, registry):
    """The only real proof a hotswap happened -- and it cannot pass here."""
    adapter = next(iter(registry.values()))
    manager.apply_adapter(lora_id=adapter.lora_id, scale=adapter.scale)
    loaded = manager.list_adapters()
    assert loaded, "no adapters reported after applying one"
    assert any(a.get("id") == adapter.lora_id for a in loaded)
