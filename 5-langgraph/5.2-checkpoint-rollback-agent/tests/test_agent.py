"""Tests for checkpoint agent workflow."""

import pytest

from checkpoint_rollback_agent.agent import AgentStep, CheckpointAgent


def add_one(state: dict) -> dict:
    return {"count": state.get("count", 0) + 1}


def add_label(state: dict) -> dict:
    return {"label": f"v{state.get('count', 0)}"}


class TestCheckpointAgent:
    def test_run_all_steps(self):
        agent = CheckpointAgent([
            AgentStep(name="increment", fn=add_one),
            AgentStep(name="label", fn=add_label),
        ])
        result = agent.run({"count": 0})
        assert result.error is None
        assert result.rolled_back is False
        assert result.num_steps_executed == 2
        assert result.final_state == {"count": 1, "label": "v1"}

    def test_checkpoints_created(self):
        agent = CheckpointAgent([
            AgentStep(name="increment", fn=add_one),
        ])
        agent.run()
        cps = agent.list_checkpoints()
        assert len(cps) == 1
        assert cps[0].step_name == "increment"

    def test_rollback_and_rerun(self):
        agent = CheckpointAgent([
            AgentStep(name="a", fn=add_one),
            AgentStep(name="b", fn=lambda s: {"count": s.get("count", 0) * 10}),
            AgentStep(name="c", fn=add_one),
        ])
        agent.run({"count": 0})
        assert agent.state["count"] == 11  # 0→1→10→11

        # Rollback to step 1 (after "a")
        state = agent.rollback_to(1)
        assert state is not None
        assert state["count"] == 1  # back to after step "a"
        assert len(agent.list_checkpoints()) == 2  # only checkpoints 0 and 1 remain

    def test_step_failure(self):
        def fail(state):
            raise ValueError("boom")

        agent = CheckpointAgent([
            AgentStep(name="good", fn=add_one),
            AgentStep(name="bad", fn=fail),
            AgentStep(name="never", fn=add_one),
        ])
        result = agent.run({"count": 0})
        assert result.error == "boom"
        assert result.num_steps_executed == 2  # stopped at step 2
        assert result.steps[1].error == "boom"

    def test_initial_state_empty(self):
        agent = CheckpointAgent([AgentStep(name="add", fn=add_one)])
        result = agent.run()
        assert result.final_state == {"count": 1}

    def test_latest_checkpoint(self):
        agent = CheckpointAgent([
            AgentStep(name="a", fn=add_one),
            AgentStep(name="b", fn=add_one),
        ])
        agent.run()
        cp = agent.latest_checkpoint
        assert cp is not None
        assert cp.step_name == "b"

    def test_rollback_to_invalid(self):
        agent = CheckpointAgent([AgentStep(name="a", fn=add_one)])
        agent.run()
        assert agent.rollback_to(99) is None
