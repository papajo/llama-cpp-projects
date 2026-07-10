"""Tests for parallel execution agent."""

import time

import pytest

from parallel_execution.core import (
    ParallelAgent,
    ParallelGroup,
    ParallelTask,
    SerialStep,
)


def inc(state: dict) -> dict:
    return {"count": state.get("count", 0) + 1}


def slow_double(state: dict) -> dict:
    time.sleep(0.05)
    return {"count": state.get("count", 0) * 2}


def add_label(state: dict) -> dict:
    return {"label": f"v{state.get('count', 0)}"}


def fail(state: dict) -> dict:
    raise ValueError("fail task")


def slow_inc(state: dict) -> dict:
    time.sleep(0.05)
    return {"count": state.get("count", 0) + 1}


class TestParallelAgent:
    def test_serial_pre_steps(self):
        agent = ParallelAgent(
            pre_steps=[SerialStep("inc1", inc), SerialStep("inc2", inc)],
        )
        result = agent.run({"count": 0})
        assert result.final_state == {"count": 2}
        assert result.num_steps == 2

    def test_single_parallel_group(self):
        agent = ParallelAgent(
            parallel_groups=[
                ParallelGroup("group1", tasks=[
                    ParallelTask("inc", inc),
                    ParallelTask("double", slow_double),
                ]),
            ],
        )
        result = agent.run({"count": 3})
        # inc: 3→4, double: 3→6, merged: {"count": 6, ...} (double wins merge order)
        assert result.final_state["count"] == 6
        assert result.num_steps == 1  # one group

    def test_parallel_tasks_execute_concurrently(self):
        """Parallel tasks should complete faster than sequential."""
        agent = ParallelAgent(
            parallel_groups=[
                ParallelGroup("slow", tasks=[
                    ParallelTask("a", slow_inc),
                    ParallelTask("b", slow_inc),
                    ParallelTask("c", slow_inc),
                ]),
            ],
        )
        start = time.time()
        result = agent.run({"count": 0})
        elapsed = time.time() - start
        # 3 tasks × 50ms each in parallel should take ~50-100ms, not 150ms+
        assert elapsed < 0.15  # well under 150ms
        assert result.final_state["count"] == 1  # all inc from 0 → 1

    def test_pre_parallel_post(self):
        agent = ParallelAgent(
            pre_steps=[SerialStep("pre_inc", inc)],
            parallel_groups=[
                ParallelGroup("g1", tasks=[
                    ParallelTask("double", slow_double),
                    ParallelTask("label", add_label),
                ]),
            ],
            post_steps=[SerialStep("post_inc", inc)],
        )
        result = agent.run({"count": 2})
        # pre: 2→3, parallel: double=6, label="v3", merged: {count:6, label:"v3"}
        # post: 6→7
        assert result.final_state == {"count": 7, "label": "v3"}

    def test_custom_merge(self):
        def merge(outputs):
            total = sum(o.get("count", 0) for o in outputs)
            return {"total": total}

        agent = ParallelAgent(
            parallel_groups=[
                ParallelGroup("sum", tasks=[
                    ParallelTask("a", lambda s: {"count": 10}),
                    ParallelTask("b", lambda s: {"count": 20}),
                ], merge_fn=merge),
            ],
        )
        result = agent.run()
        assert result.final_state == {"total": 30}

    def test_task_failure_in_group(self):
        agent = ParallelAgent(
            parallel_groups=[
                ParallelGroup("g1", tasks=[
                    ParallelTask("good", inc),
                    ParallelTask("bad", fail),
                ]),
            ],
        )
        result = agent.run({"count": 0})
        assert result.error is not None
        assert "fail task" in result.error
        gr = result.group_results[0]
        assert gr.all_succeeded is False

    def test_multiple_parallel_groups_sequentially(self):
        agent = ParallelAgent(
            parallel_groups=[
                ParallelGroup("first", tasks=[
                    ParallelTask("inc", inc),
                ]),
                ParallelGroup("second", tasks=[
                    ParallelTask("double", slow_double),
                ]),
            ],
        )
        result = agent.run({"count": 0})
        # first: 0→1, second: 1→2
        assert result.final_state == {"count": 2}
        assert len(result.group_results) == 2

    def test_post_steps_after_parallel(self):
        agent = ParallelAgent(
            parallel_groups=[
                ParallelGroup("g1", tasks=[
                    ParallelTask("inc", inc),
                ]),
            ],
            post_steps=[SerialStep("post_label", add_label)],
        )
        result = agent.run({"count": 5})
        assert result.final_state == {"count": 6, "label": "v6"}

    def test_serial_failure(self):
        def fail_serial(state):
            raise ValueError("serial fail")

        agent = ParallelAgent(
            pre_steps=[SerialStep("fail", fail_serial)],
        )
        result = agent.run({"count": 0})
        assert result.error == "serial fail"

    def test_empty_workflow(self):
        agent = ParallelAgent()
        result = agent.run()
        assert result.num_steps == 0
        assert result.final_state == {}

    def test_empty_parallel_group(self):
        agent = ParallelAgent(
            parallel_groups=[ParallelGroup("empty", tasks=[])],
        )
        result = agent.run({"count": 0})
        assert result.final_state == {"count": 0}
        assert result.group_results[0].all_succeeded is True

    def test_task_naming(self):
        pt = ParallelTask(name="extract", fn=lambda s: {"x": 1})
        assert pt.name == "extract"
