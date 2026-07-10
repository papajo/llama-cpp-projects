"""Tests for supervisor agent."""

import pytest

from supervisor_agent.core import (
    DelegationStrategy,
    SubAgent,
    SupervisorAgent,
    SupervisorResult,
)


def inc(state: dict) -> dict:
    return {"count": state.get("count", 0) + 1}


def double(state: dict) -> dict:
    return {"count": state.get("count", 0) * 2}


def add_label(state: dict) -> dict:
    return {"label": f"v{state.get('count', 0)}"}


def fail(state: dict) -> dict:
    raise ValueError("sub-agent crash")


class TestSubAgent:
    def test_can_handle_match(self):
        agent = SubAgent(name="coder", capability="code generation", fn=inc)
        assert agent.can_handle("write some code") is True
        assert agent.can_handle("just review") is False

    def test_can_handle_case_insensitive(self):
        agent = SubAgent(name="coder", capability="Python", fn=inc)
        assert agent.can_handle("write python code") is True


class TestSupervisorAgent:
    def test_first_match_routes_to_first(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("coder", "code", inc),
                SubAgent("reviewer", "review", double),
            ],
            strategy=DelegationStrategy.FIRST_MATCH,
        )
        result = agent.delegate("write some code and review", {"count": 0})
        # "code" matches both "code" and "review", but FIRST_MATCH picks "coder"
        assert result.selected_agents == ["coder"]
        assert result.final_state == {"count": 1}
        assert result.all_succeeded is True

    def test_first_match_prioritises_order(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("reviewer", "review", double),
                SubAgent("coder", "code", inc),
            ],
            strategy=DelegationStrategy.FIRST_MATCH,
        )
        result = agent.delegate("write code", {"count": 0})
        # "coder" capability "code" matches "write code"
        assert result.selected_agents == ["coder"]
        assert result.final_state == {"count": 1}

    def test_no_keyword_match(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("reviewer", "review", double),
                SubAgent("coder", "code", inc),
            ],
        )
        result = agent.delegate("eat sleep", {"count": 0})
        # Neither "review" nor "code" appears in "eat sleep"
        assert result.selected_agents == []
        assert result.final_state == {"count": 0}

    def test_all_match_routes_to_all(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("coder", "code", inc),
                SubAgent("reviewer", "review", double),
                SubAgent("labeler", "label", add_label),
            ],
            strategy=DelegationStrategy.ALL_MATCH,
        )
        result = agent.delegate("code review with label", {"count": 5})
        # All three match: "code", "review", "label"
        assert len(result.selected_agents) == 3
        assert "coder" in result.selected_agents
        assert "reviewer" in result.selected_agents
        assert "labeler" in result.selected_agents
        # inc: 5→6, double: 6→12 (state is shared sequentially), label: "v12"
        assert result.final_state["count"] == 12
        assert result.final_state["label"] == "v12"

    def test_no_match_returns_empty(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("coder", "code", inc),
            ],
            strategy=DelegationStrategy.FIRST_MATCH,
        )
        result = agent.delegate("just eat sleep", {"count": 0})
        assert result.selected_agents == []
        assert result.final_state == {"count": 0}
        assert result.all_succeeded is True  # vacuously true

    def test_custom_router(self):
        def always_second(task, state, agents):
            return [agents[1]]

        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("first", "anything", inc),
                SubAgent("second", "anything", double),
            ],
            strategy=DelegationStrategy.CUSTOM,
            router=always_second,
        )
        result = agent.delegate("anything", {"count": 3})
        assert result.selected_agents == ["second"]
        assert result.final_state["count"] == 6

    def test_custom_router_multiple(self):
        def all_except_first(task, state, agents):
            return agents[1:]

        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("skip", "anything", inc),
                SubAgent("a", "anything", lambda s: {"x": 1}),
                SubAgent("b", "anything", lambda s: {"y": 2}),
            ],
            strategy=DelegationStrategy.CUSTOM,
            router=all_except_first,
        )
        result = agent.delegate("anything")
        assert result.selected_agents == ["a", "b"]
        assert result.final_state == {"x": 1, "y": 2}

    def test_sub_agent_failure(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("coder", "code", fail),
            ],
            strategy=DelegationStrategy.FIRST_MATCH,
        )
        result = agent.delegate("write code", {"count": 0})
        assert result.error is not None
        assert "sub-agent crash" in result.error
        assert result.all_succeeded is False
        assert result.num_failed == 1

    def test_records_all_agents(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("coder", "code", inc),
                SubAgent("reviewer", "review", double),
            ],
        )
        result = agent.delegate("review the code", {"count": 1})
        assert len(result.records) == 2
        coder_rec = [r for r in result.records if r.sub_agent == "coder"][0]
        reviewer_rec = [r for r in result.records if r.sub_agent == "reviewer"][0]
        assert coder_rec.selected is True   # FIRST_MATCH picks "coder" (keyword "code" before "review")
        assert reviewer_rec.selected is False

    def test_state_snapshot_per_agent(self):
        agent = SupervisorAgent(
            sub_agents=[
                SubAgent("a", "a", lambda s: {"val": s.get("val", 0) + 1}),
                SubAgent("b", "b", lambda s: {"val": s.get("val", 0) + 10}),
            ],
            strategy=DelegationStrategy.ALL_MATCH,
        )
        result = agent.delegate("a and b", {"val": 0})
        # a: 0→1, b: 1→11 (b sees a's update because state is shared between agents)
        assert result.final_state["val"] == 11

    def test_empty_sub_agents(self):
        agent = SupervisorAgent(
            sub_agents=[],
            strategy=DelegationStrategy.FIRST_MATCH,
        )
        result = agent.delegate("anything")
        assert result.selected_agents == []
        assert result.final_state == {}

    def test_supervisor_result_properties(self):
        r = SupervisorResult(task="test", strategy=DelegationStrategy.FIRST_MATCH)
        assert r.num_selected == 0
        assert r.all_succeeded is True
        assert r.num_succeeded == 0
        assert r.num_failed == 0
