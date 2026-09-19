"""Live integration tests for multi-model task decomposition.

`MultiModelOrchestrator` routes subtasks to handlers registered in a
`HandlerRegistry`. The offline tests register synthetic handlers; these
register handlers that make REAL llama-server calls, so dependency ordering is
verified over genuine model output flowing through shared state.

IMPORTANT (see drift-graph.md): "multi-model" is aspirational on this host.
Each llama-server process serves exactly ONE model, so the chat handler and the
embedding handler are the only two distinct "models" available, and they live on
different ports. There is no single endpoint to route between models.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import pytest

from task_decomp.core import (
    DecompositionPlan,
    HandlerRegistry,
    MultiModelOrchestrator,
    Subtask,
    TaskDecomposer,
)


class FixedPlanDecomposer(TaskDecomposer):
    """A decomposer returning a preset plan, so ordering is deterministic."""

    def __init__(self, plan: DecompositionPlan):
        self._plan = plan

    def decompose(self, task: str) -> DecompositionPlan:
        return self._plan


@pytest.fixture
def registry(live_chat, live_embed):
    """A registry whose handlers hit the two real servers."""
    reg = HandlerRegistry()
    calls: list[str] = []

    def chat_handler(subtask, state):
        calls.append(subtask.name)
        resp = live_chat(
            [{"role": "user", "content": subtask.description}], max_tokens=16
        )
        return {f"{subtask.name}_text": resp["choices"][0]["message"]["content"]}

    def embed_handler(subtask, state):
        calls.append(subtask.name)
        # Embed whatever the upstream chat subtask produced, if anything.
        text = next(
            (v for k, v in state.items() if k.endswith("_text")),
            subtask.description,
        )
        resp = live_embed(text)
        return {f"{subtask.name}_dims": len(resp["data"][0]["embedding"])}

    reg.register("chat", chat_handler)
    reg.register("embed", embed_handler)
    reg.calls = calls
    return reg


# ---------------------------------------------------------------------------
# Execution across the two real servers
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_chat_then_embed_respects_dependency(registry):
    """The embed subtask depends on the chat subtask and runs after it."""
    plan = DecompositionPlan(description="describe then embed")
    plan.add(Subtask(
        name="embed_it", description="unused", handler_key="embed",
        dependencies=["write"],
    ))
    plan.add(Subtask(
        name="write", description="Name a colour.", handler_key="chat",
    ))

    orchestrator = MultiModelOrchestrator(
        decomposer=FixedPlanDecomposer(plan), registry=registry
    )
    result = orchestrator.execute("describe then embed")

    assert result.error is None
    assert result.all_succeeded
    assert result.num_executed == 2
    assert registry.calls == ["write", "embed_it"], "topological order, not list order"

    assert result.get_output("write", "write_text").strip()
    assert result.get_output("embed_it", "embed_it_dims") == 768
    assert "write_text" in result.shared_state


@pytest.mark.live
def test_independent_subtasks_both_run(registry):
    plan = DecompositionPlan(description="two colours")
    plan.add(Subtask(name="a", description="Name a colour.", handler_key="chat"))
    plan.add(Subtask(name="b", description="Name an animal.", handler_key="chat"))

    result = MultiModelOrchestrator(
        decomposer=FixedPlanDecomposer(plan), registry=registry
    ).execute("two colours")

    assert result.all_succeeded
    assert result.num_executed == 2
    assert result.get_output("a", "a_text").strip()
    assert result.get_output("b", "b_text").strip()


@pytest.mark.live
def test_default_decomposer_single_subtask(registry):
    """TaskDecomposer's default plan uses handler_key 'general'."""
    registry.register("general", registry.get("chat"))

    result = MultiModelOrchestrator(
        decomposer=TaskDecomposer(), registry=registry
    ).execute("Name a colour.")

    assert result.error is None
    assert result.plan.count == 1
    assert result.num_executed == 1
    assert result.get_output("default", "default_text").strip()


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_missing_handler_stops_before_any_call(registry):
    plan = DecompositionPlan()
    plan.add(Subtask(name="nope", description="x", handler_key="does_not_exist"))
    plan.add(Subtask(name="after", description="Name a colour.", handler_key="chat"))

    result = MultiModelOrchestrator(
        decomposer=FixedPlanDecomposer(plan), registry=registry
    ).execute("x")

    assert result.error is not None
    assert "does_not_exist" in result.error
    assert result.num_executed == 0
    assert registry.calls == [], "no server was contacted"


@pytest.mark.live
def test_real_server_error_is_captured_and_halts(registry, live_chat):
    """A handler raising a real HTTP error aborts the plan after its predecessor."""
    import urllib.request

    def broken(subtask, state):
        urllib.request.urlopen("http://127.0.0.1:1/v1/models", timeout=5)
        return {}

    registry.register("broken", broken)

    plan = DecompositionPlan()
    plan.add(Subtask(name="first", description="Name a colour.", handler_key="chat"))
    plan.add(Subtask(
        name="boom", description="x", handler_key="broken", dependencies=["first"]
    ))
    plan.add(Subtask(
        name="never", description="Name an animal.", handler_key="chat",
        dependencies=["boom"],
    ))

    result = MultiModelOrchestrator(
        decomposer=FixedPlanDecomposer(plan), registry=registry
    ).execute("x")

    assert result.error is not None
    assert result.num_executed == 1, "only 'first' completed"
    assert result.num_failed == 1
    assert registry.calls == ["first"], "'never' was not reached"
    assert result.get_result("boom").succeeded is False
