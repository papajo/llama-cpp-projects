"""Live integration tests for the checkpoint/rollback agent.

This project is pure orchestration: `CheckpointAgent` runs a list of
`AgentStep`s, each holding an arbitrary callable, and snapshots state around
them. The offline tests use synthetic lambdas. These tests use step functions
that make REAL llama-server calls, so the checkpointed state contains genuine
model output and a genuine mid-workflow failure is a real network/HTTP error
rather than a hand-raised exception.

Assertions are structural. SmolLM2-360M's actual words are never asserted on.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import pytest

from checkpoint_rollback_agent.agent import AgentStep, CheckpointAgent
from checkpoint_rollback_agent.store import Checkpoint


@pytest.fixture
def llm_step(live_chat):
    """Build an AgentStep whose fn really calls the chat server."""

    def make(name: str, prompt: str, out_key: str) -> AgentStep:
        def fn(state):
            resp = live_chat(
                [{"role": "user", "content": prompt}], max_tokens=16
            )
            return {out_key: resp["choices"][0]["message"]["content"]}

        return AgentStep(name=name, fn=fn, description=f"real call: {prompt!r}")

    return make


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_run_checkpoints_real_llm_steps(llm_step):
    """Two real LLM steps, each checkpointed before execution."""
    agent = CheckpointAgent([
        llm_step("draft", "Name a colour.", "colour"),
        llm_step("expand", "Name an animal.", "animal"),
    ])

    result = agent.run({"seed": 1})

    assert result.error is None
    assert result.num_steps_executed == 2
    assert result.final_state["seed"] == 1
    assert result.final_state["colour"].strip()
    assert result.final_state["animal"].strip()

    # A checkpoint is taken BEFORE each step, so it holds pre-step state.
    checkpoints = agent.list_checkpoints()
    assert len(checkpoints) == 2
    assert [c.step_index for c in checkpoints] == [0, 1]
    assert [c.step_name for c in checkpoints] == ["draft", "expand"]

    assert checkpoints[0].state == {"seed": 1}, "first snapshot predates any output"
    assert "colour" in checkpoints[1].state, "second snapshot has step 1's output"
    assert "animal" not in checkpoints[1].state


@pytest.mark.live
def test_checkpoint_state_is_isolated_from_later_mutation(llm_step):
    """Snapshots are copies: real output added later must not leak backwards."""
    agent = CheckpointAgent([llm_step("draft", "Name a colour.", "colour")])
    agent.run({"seed": 1})

    cp = agent.list_checkpoints()[0]
    assert cp.state == {"seed": 1}
    assert "colour" in agent.state


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rollback_discards_real_output(llm_step):
    """Rolling back to step 0 throws away everything the model produced."""
    agent = CheckpointAgent([
        llm_step("draft", "Name a colour.", "colour"),
        llm_step("expand", "Name an animal.", "animal"),
    ])
    agent.run({"seed": 1})
    assert "animal" in agent.state

    restored = agent.rollback_to(0)

    assert restored == {"seed": 1}
    assert agent.state == {"seed": 1}
    assert "colour" not in agent.state
    assert "animal" not in agent.state
    assert len(agent.list_checkpoints()) == 1, "later checkpoints are discarded"


@pytest.mark.live
def test_rollback_to_unknown_index_returns_none(llm_step):
    agent = CheckpointAgent([llm_step("draft", "Name a colour.", "colour")])
    agent.run({"seed": 1})
    assert agent.rollback_to(99) is None


# ---------------------------------------------------------------------------
# Real failure mid-workflow
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_real_connection_failure_halts_and_preserves_checkpoint(llm_step, live_chat):
    """A genuine network failure in step 2 stops the run with state intact.

    Step 2 points at a closed port, so urlopen raises URLError — an uncaught
    exception from the step fn, which is exactly what the agent's except branch
    is for.
    """
    import urllib.error
    import urllib.request

    def failing_fn(state):
        urllib.request.urlopen("http://127.0.0.1:1/v1/chat/completions", timeout=5)
        return {"never": "reached"}

    agent = CheckpointAgent([
        llm_step("draft", "Name a colour.", "colour"),
        AgentStep(name="unreachable", fn=failing_fn),
        llm_step("after", "Name an animal.", "animal"),
    ])

    result = agent.run({"seed": 1})

    assert result.error is not None
    assert result.num_steps_executed == 2, "stops at the failing step"
    assert result.steps[-1].step_name == "unreachable"
    assert result.steps[-1].error is not None
    assert "animal" not in result.final_state, "step 3 never ran"
    assert "colour" in result.final_state, "step 1's real output survives"

    assert isinstance(result.steps[-1].checkpoint, Checkpoint)
    assert agent.rollback_to(0) == {"seed": 1}
