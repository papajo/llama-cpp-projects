"""Tests for multi-model task decomposition."""

import pytest

from task_decomp.core import (
    DecompositionPlan,
    HandlerRegistry,
    MultiModelOrchestrator,
    OrchestrationResult,
    Subtask,
    TaskDecomposer,
)


def handler_summarize(st: Subtask, state: dict) -> dict:
    return {"summary": f"Summary of: {st.description}"}


def handler_codegen(st: Subtask, state: dict) -> dict:
    return {"code": f"# Code for {st.description}"}


def handler_extract(st: Subtask, state: dict) -> dict:
    return {"extracted": st.params.get("field", "unknown")}


class TestSubtask:
    def test_basic(self):
        st = Subtask(name="parse", description="Parse input", handler_key="parser")
        assert st.name == "parse"
        assert st.handler_key == "parser"


class TestHandlerRegistry:
    def test_register_and_get(self):
        reg = HandlerRegistry()
        reg.register("code", handler_codegen)
        assert reg.has("code")
        assert reg.get("code") is handler_codegen

    def test_missing(self):
        reg = HandlerRegistry()
        assert not reg.has("missing")
        assert reg.get("missing") is None


class TestDecompositionPlan:
    def test_add_and_count(self):
        plan = DecompositionPlan()
        plan.add(Subtask(name="a", description="A", handler_key="h"))
        assert plan.count == 1

    def test_topological_order_no_deps(self):
        plan = DecompositionPlan()
        plan.add(Subtask(name="a", description="A", handler_key="h"))
        plan.add(Subtask(name="b", description="B", handler_key="h"))
        ordered = plan.topological_order()
        assert [s.name for s in ordered] == ["a", "b"]

    def test_topological_order_with_deps(self):
        plan = DecompositionPlan()
        plan.add(Subtask(name="a", description="A", handler_key="h"))
        plan.add(Subtask(name="c", description="C", handler_key="h", dependencies=["a"]))
        plan.add(Subtask(name="b", description="B", handler_key="h", dependencies=["a"]))
        ordered = plan.topological_order()
        assert ordered[0].name == "a"
        # b and c can be in either order, but both after a
        assert ordered[1].name in ("b", "c")
        assert ordered[2].name in ("b", "c")
        assert ordered[1].name != ordered[2].name


class TestTaskDecomposer:
    def test_default_decompose(self):
        d = TaskDecomposer()
        plan = d.decompose("Write code")
        assert plan.count == 1
        assert plan.subtasks[0].name == "default"
        assert plan.subtasks[0].handler_key == "general"


class TestMultiModelOrchestrator:
    def test_simple_execution(self):
        class SimpleDecomposer(TaskDecomposer):
            def decompose(self, task):
                return DecompositionPlan(
                    subtasks=[Subtask(name="sum", description=task, handler_key="summary")]
                )

        reg = HandlerRegistry()
        reg.register("summary", handler_summarize)
        orch = MultiModelOrchestrator(decomposer=SimpleDecomposer(), registry=reg)
        result = orch.execute("Hello world")
        assert result.all_succeeded
        assert result.num_executed == 1
        assert result.get_output("sum", "summary") == "Summary of: Hello world"

    def test_missing_handler(self):
        class SimpleDecomposer(TaskDecomposer):
            def decompose(self, task):
                return DecompositionPlan(
                    subtasks=[Subtask(name="x", description="X", handler_key="missing")]
                )

        reg = HandlerRegistry()
        orch = MultiModelOrchestrator(decomposer=SimpleDecomposer(), registry=reg)
        result = orch.execute("task")
        assert not result.all_succeeded
        assert result.error == "Missing handler: missing"

    def test_multi_model_routing(self):
        """Different subtasks route to different handlers."""

        class MultiDecomposer(TaskDecomposer):
            def decompose(self, task):
                return DecompositionPlan(subtasks=[
                    Subtask(name="summarize", description=task, handler_key="summary"),
                    Subtask(name="codegen", description="build fn", handler_key="code"),
                    Subtask(name="extract", description="get field", handler_key="extract",
                            params={"field": "name"}),
                ])

        reg = HandlerRegistry()
        reg.register("summary", handler_summarize)
        reg.register("code", handler_codegen)
        reg.register("extract", handler_extract)
        orch = MultiModelOrchestrator(decomposer=MultiDecomposer(), registry=reg)
        result = orch.execute("Build a parser")
        assert result.all_succeeded
        assert result.num_executed == 3
        assert "summary" in str(result.get_output("summarize"))
        assert "build fn" in result.get_output("codegen", "code")
        assert result.get_output("extract", "extracted") == "name"

    def test_dependency_ordering(self):
        """Subtask B depends on A — B sees A's output in shared state."""

        def handler_a(st, state):
            return {"a_done": True, "value": 42}

        def handler_b(st, state):
            return {"b_used_value": state.get("value")}

        class DepDecomposer(TaskDecomposer):
            def decompose(self, task):
                return DecompositionPlan(subtasks=[
                    Subtask(name="step_a", description="A", handler_key="a"),
                    Subtask(name="step_b", description="B", handler_key="b",
                            dependencies=["step_a"]),
                ])

        reg = HandlerRegistry()
        reg.register("a", handler_a)
        reg.register("b", handler_b)
        orch = MultiModelOrchestrator(decomposer=DepDecomposer(), registry=reg)
        result = orch.execute("task")
        assert result.all_succeeded
        assert result.shared_state["b_used_value"] == 42
        assert result.shared_state["a_done"] is True

    def test_handler_failure(self):
        def failing(st, state):
            raise ValueError("Handler crash")

        class FailDecomposer(TaskDecomposer):
            def decompose(self, task):
                return DecompositionPlan(
                    subtasks=[Subtask(name="fail", description="F", handler_key="f")]
                )

        reg = HandlerRegistry()
        reg.register("f", failing)
        orch = MultiModelOrchestrator(decomposer=FailDecomposer(), registry=reg)
        result = orch.execute("task")
        assert not result.all_succeeded
        assert result.error == "Handler crash"

    def test_empty_plan(self):
        class EmptyDecomposer(TaskDecomposer):
            def decompose(self, task):
                return DecompositionPlan()

        reg = HandlerRegistry()
        orch = MultiModelOrchestrator(decomposer=EmptyDecomposer(), registry=reg)
        result = orch.execute("task")
        assert result.all_succeeded
        assert result.num_executed == 0

    def test_get_result_nonexistent(self):
        result = OrchestrationResult(
            task_description="test", plan=DecompositionPlan()
        )
        assert result.get_result("nope") is None
        assert result.get_output("nope") is None
