"""Live integration tests for parallel execution.

`ParallelAgent` fans a `ParallelGroup` out across a ThreadPoolExecutor, then
merges the task outputs back into state. The offline tests fan out synthetic
lambdas; these fan out REAL concurrent llama-server requests, which is the only
way to verify the group actually overlaps work rather than serialising it.

Concurrency is capped at 3: `/props` reports `total_slots: 3` (llama-server
started with `--parallel 3`), so a fourth simultaneous request would queue
behind a slot rather than run in parallel. Keeping the fan-out at the slot count
keeps the timing assertion honest.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import time

import pytest

from parallel_execution.core import (
    ParallelAgent,
    ParallelGroup,
    ParallelTask,
    SerialStep,
)

SLOTS = 3


@pytest.fixture
def chat_task(live_chat):
    """A ParallelTask that really calls the chat server."""

    def make(name: str, prompt: str, max_tokens: int = 24) -> ParallelTask:
        def fn(state):
            started = time.monotonic()
            resp = live_chat(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens
            )
            return {
                name: resp["choices"][0]["message"]["content"],
                f"{name}_elapsed": time.monotonic() - started,
            }

        return ParallelTask(name=name, fn=fn)

    return make


# ---------------------------------------------------------------------------
# Real concurrent fan-out
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_parallel_group_merges_real_outputs(chat_task):
    """Three concurrent real calls, all merged into state."""
    group = ParallelGroup(
        name="ask_three",
        tasks=[
            chat_task("colour", "Name a colour."),
            chat_task("animal", "Name an animal."),
            chat_task("city", "Name a city."),
        ],
    )
    agent = ParallelAgent(parallel_groups=[group], max_workers=SLOTS)

    result = agent.run({"seed": 1})

    assert result.error is None
    assert len(result.group_results) == 1

    gr = result.group_results[0]
    assert gr.all_succeeded
    assert gr.num_tasks == 3
    assert {tr.task_name for tr in gr.task_results} == {"colour", "animal", "city"}

    for key in ("colour", "animal", "city"):
        assert result.final_state[key].strip(), f"{key} produced no text"
    assert result.final_state["seed"] == 1, "pre-existing state survives the merge"


@pytest.mark.live
def test_parallel_is_faster_than_serial_would_be(chat_task):
    """Wall-clock proof the group really overlaps real requests.

    Each task measures its own duration. If the group ran serially the total
    would be about the sum; concurrently it is about the max. Asserting
    total < sum * 0.9 is loose enough to survive a loaded CPU while still
    failing outright if the executor serialises.
    """
    group = ParallelGroup(
        name="ask_three",
        tasks=[
            chat_task("a", "Name a colour.", max_tokens=32),
            chat_task("b", "Name an animal.", max_tokens=32),
            chat_task("c", "Name a city.", max_tokens=32),
        ],
    )
    agent = ParallelAgent(parallel_groups=[group], max_workers=SLOTS)

    started = time.monotonic()
    result = agent.run({})
    total = time.monotonic() - started

    per_task = [result.final_state[f"{k}_elapsed"] for k in ("a", "b", "c")]
    assert result.error is None
    assert total < sum(per_task) * 0.9, (
        f"group took {total:.2f}s but tasks summed to {sum(per_task):.2f}s — "
        "looks serialised, not parallel"
    )


@pytest.mark.live
def test_serial_steps_bracket_the_parallel_group(chat_task):
    """pre-steps → group → post-steps, with real calls in the middle."""
    agent = ParallelAgent(
        pre_steps=[SerialStep(name="prepare", fn=lambda s: {"prepared": True})],
        parallel_groups=[
            ParallelGroup(
                name="ask_two",
                tasks=[
                    chat_task("colour", "Name a colour."),
                    chat_task("animal", "Name an animal."),
                ],
            )
        ],
        post_steps=[
            SerialStep(
                name="summarise",
                fn=lambda s: {"n_answers": sum(
                    1 for k in ("colour", "animal") if s.get(k)
                )},
            )
        ],
        max_workers=SLOTS,
    )

    result = agent.run({})

    assert result.error is None
    assert result.final_state["prepared"] is True
    assert result.final_state["n_answers"] == 2, "post-step saw both merged outputs"
    assert [s.step_name for s in result.steps] == ["prepare", "ask_two", "summarise"]


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_one_failing_task_fails_the_group(chat_task):
    """A real connection error in one task marks the whole group failed."""
    import urllib.request

    def unreachable(state):
        urllib.request.urlopen("http://127.0.0.1:1/v1/models", timeout=5)
        return {}

    group = ParallelGroup(
        name="mixed",
        tasks=[
            chat_task("colour", "Name a colour."),
            ParallelTask(name="broken", fn=unreachable),
        ],
    )
    agent = ParallelAgent(parallel_groups=[group], max_workers=2)

    result = agent.run({})

    assert result.error is not None
    assert "mixed" in result.error

    gr = result.group_results[0]
    assert gr.all_succeeded is False
    failed = [tr for tr in gr.task_results if not tr.succeeded]
    assert [tr.task_name for tr in failed] == ["broken"]

    # The group merges nothing when any task fails, so the successful
    # sibling's real output is discarded too.
    assert gr.merged_output == {}
    assert "colour" not in result.final_state
