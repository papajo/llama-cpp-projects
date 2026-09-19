"""Canonical smoke test for the live-server harness.

Owned by no project, so the per-category workers never collide on it.
Run with:  source env.sh && LLM_LIVE=1 pytest tests/test_live_smoke.py
"""

import pytest


@pytest.mark.live
def test_chat_server_responds(live_chat):
    r = live_chat([{"role": "user", "content": "Say hi"}], max_tokens=8)
    assert r["choices"][0]["message"]["content"]
    assert r["object"] == "chat.completion"


@pytest.mark.live
def test_embed_server_responds(live_embed):
    r = live_embed("hello")
    assert len(r["data"][0]["embedding"]) == 768


@pytest.mark.live
def test_servers_are_llamacpp(live_server_meta):
    assert "SmolLM2" in live_server_meta["chat"]["data"][0]["id"]
    assert "nomic" in live_server_meta["embed"]["data"][0]["id"]
