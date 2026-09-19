"""Live integration tests for the supervisor agent.

`SupervisorAgent` routes a task to sub-agents by keyword-matching the task
string against each sub-agent's `capability`. The offline tests use synthetic
sub-agent functions; here the sub-agents really call llama-server — the chat
server for text, the embed server for vectors — so delegation is verified with
genuine work behind it.

Note: routing is a pure string match on `capability` keywords. The model plays
no part in choosing a sub-agent, and it cannot: `/props` reports
`supports_tools: false` for SmolLM2-360M, so there is no tool/function-calling
path for model-driven routing (see drift-graph.md entry 9).

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import pytest

from supervisor_agent.core import (
    DelegationStrategy,
    SubAgent,
    SupervisorAgent,
)


@pytest.fixture
def agents(live_chat, live_embed):
    """Two real sub-agents plus a call log."""
    calls: list[str] = []

    def writer(state):
        calls.append("writer")
        resp = live_chat(
            [{"role": "user", "content": "Name a colour."}], max_tokens=16
        )
        return {"text": resp["choices"][0]["message"]["content"]}

    def embedder(state):
        calls.append("embedder")
        resp = live_embed(state.get("text", "placeholder"))
        return {"dims": len(resp["data"][0]["embedding"])}

    made = [
        SubAgent(name="writer", capability="write draft text", fn=writer),
        SubAgent(name="embedder", capability="embed vector text", fn=embedder),
    ]
    return made, calls


# ---------------------------------------------------------------------------
# FIRST_MATCH
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_first_match_delegates_to_one_real_agent(agents):
    """"draft" matches only the writer, so only it is invoked."""
    subs, calls = agents
    supervisor = SupervisorAgent(
        sub_agents=subs, strategy=DelegationStrategy.FIRST_MATCH
    )

    result = supervisor.delegate("please draft something")

    assert result.error is None
    assert result.selected_agents == ["writer"]
    assert calls == ["writer"], "the embedder must not be called"
    assert result.num_selected == 1
    assert result.all_succeeded
    assert result.final_state["text"].strip()

    # Every sub-agent is recorded for transparency, selected or not.
    assert len(result.records) == 2
    by_name = {r.sub_agent: r for r in result.records}
    assert by_name["writer"].selected is True
    assert by_name["embedder"].selected is False
    assert by_name["embedder"].output == {}


@pytest.mark.live
def test_no_match_calls_nothing(agents):
    subs, calls = agents
    supervisor = SupervisorAgent(
        sub_agents=subs, strategy=DelegationStrategy.FIRST_MATCH
    )

    result = supervisor.delegate("completely unrelated request")

    assert result.selected_agents == []
    assert calls == [], "no server contacted when nothing matches"
    assert result.num_selected == 0
    assert all(not r.selected for r in result.records)


# ---------------------------------------------------------------------------
# ALL_MATCH across both real servers
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_all_match_runs_both_real_agents(agents):
    """"text" matches both capabilities, so both servers get used."""
    subs, calls = agents
    supervisor = SupervisorAgent(
        sub_agents=subs, strategy=DelegationStrategy.ALL_MATCH
    )

    result = supervisor.delegate("work with text")

    assert result.error is None
    assert set(result.selected_agents) == {"writer", "embedder"}
    assert sorted(calls) == ["embedder", "writer"]
    assert result.num_succeeded == 2
    assert result.final_state["text"].strip()
    assert result.final_state["dims"] == 768


# ---------------------------------------------------------------------------
# CUSTOM router
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_custom_router_overrides_capability_matching(agents):
    """A custom router can pick an agent whose capability does not match."""
    subs, calls = agents

    def always_embedder(task, state, available):
        return [a for a in available if a.name == "embedder"]

    supervisor = SupervisorAgent(
        sub_agents=subs,
        strategy=DelegationStrategy.CUSTOM,
        router=always_embedder,
    )

    result = supervisor.delegate("please draft something")

    assert result.selected_agents == ["embedder"]
    assert calls == ["embedder"]
    assert result.final_state["dims"] == 768


# ---------------------------------------------------------------------------
# Real failure
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_real_failure_in_sub_agent_is_recorded(agents):
    import urllib.request

    def broken(state):
        urllib.request.urlopen("http://127.0.0.1:1/v1/models", timeout=5)
        return {}

    subs, _ = agents
    supervisor = SupervisorAgent(
        sub_agents=[SubAgent(name="broken", capability="draft text", fn=broken)],
        strategy=DelegationStrategy.FIRST_MATCH,
    )

    result = supervisor.delegate("please draft something")

    assert result.selected_agents == ["broken"]
    assert result.num_failed == 1
    assert result.all_succeeded is False
    record = next(r for r in result.records if r.sub_agent == "broken")
    assert record.error is not None
    assert record.succeeded is False
