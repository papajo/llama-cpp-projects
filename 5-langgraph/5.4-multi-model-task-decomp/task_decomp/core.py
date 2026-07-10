"""Task decomposition into subtasks with specialised handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Subtask:
    """A single unit of decomposed work."""

    name: str
    description: str
    handler_key: str
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubtaskResult:
    subtask: Subtask
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    executed: bool = False

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.executed


TaskHandler = Callable[[Subtask, Dict[str, Any]], Dict[str, Any]]


@dataclass
class HandlerRegistry:
    """Maps handler keys to callable handlers.

    A handler takes (subtask, shared_state) and returns output dict.
    """

    handlers: Dict[str, TaskHandler] = field(default_factory=dict)

    def register(self, key: str, handler: TaskHandler) -> None:
        self.handlers[key] = handler

    def get(self, key: str) -> Optional[TaskHandler]:
        return self.handlers.get(key)

    def has(self, key: str) -> bool:
        return key in self.handlers


@dataclass
class DecompositionPlan:
    """A list of subtasks with dependency metadata."""

    subtasks: List[Subtask] = field(default_factory=list)
    description: str = ""

    def add(self, subtask: Subtask) -> None:
        self.subtasks.append(subtask)

    @property
    def count(self) -> int:
        return len(self.subtasks)

    def topological_order(self) -> List[Subtask]:
        """Return subtasks in dependency-respecting order (simple Kahn's)."""
        # Build in-degree map
        names = {s.name for s in self.subtasks}
        in_degree: Dict[str, int] = {s.name: 0 for s in self.subtasks}
        dep_map: Dict[str, List[str]] = {s.name: [] for s in self.subtasks}

        for s in self.subtasks:
            for dep in s.dependencies:
                if dep in dep_map:
                    dep_map[dep].append(s.name)
                    in_degree[s.name] += 1

        queue = [s for s in self.subtasks if in_degree[s.name] == 0]
        ordered = []

        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for dependent in dep_map[node.name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(
                        next(s for s in self.subtasks if s.name == dependent)
                    )

        # If cycle or missing deps, just return original order
        if len(ordered) != len(self.subtasks):
            return list(self.subtasks)
        return ordered


@dataclass
class OrchestrationResult:
    task_description: str
    plan: DecompositionPlan
    results: List[SubtaskResult] = field(default_factory=list)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def all_succeeded(self) -> bool:
        return all(r.succeeded for r in self.results)

    @property
    def num_executed(self) -> int:
        return sum(1 for r in self.results if r.executed)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    def get_result(self, subtask_name: str) -> Optional[SubtaskResult]:
        for r in self.results:
            if r.subtask.name == subtask_name:
                return r
        return None

    def get_output(self, subtask_name: str, key: str = "") -> Any:
        r = self.get_result(subtask_name)
        if r is None:
            return None
        if key:
            return r.output.get(key)
        return r.output


@dataclass
class TaskDecomposer:
    """Splits a complex task into a plan of subtasks.

    Override ``decompose()`` in subclasses for different strategies.
    """

    def decompose(self, task: str) -> DecompositionPlan:
        """Default: single subtask that does everything."""
        return DecompositionPlan(
            subtasks=[Subtask(name="default", description=task, handler_key="general")],
            description=task,
        )


@dataclass
class MultiModelOrchestrator:
    """Executes a decomposition plan by routing subtasks to handlers.

    Handles dependency ordering and shared state accumulation.
    """

    decomposer: TaskDecomposer
    registry: HandlerRegistry

    def execute(self, task: str) -> OrchestrationResult:
        plan = self.decomposer.decompose(task)
        result = OrchestrationResult(
            task_description=task,
            plan=plan,
        )
        state: Dict[str, Any] = {}

        ordered = plan.topological_order()

        for subtask in ordered:
            handler = self.registry.get(subtask.handler_key)
            if handler is None:
                result.results.append(
                    SubtaskResult(
                        subtask=subtask,
                        error=f"No handler registered for '{subtask.handler_key}'",
                    )
                )
                result.error = f"Missing handler: {subtask.handler_key}"
                break

            try:
                output = handler(subtask, dict(state))
                state.update(output)
                result.results.append(
                    SubtaskResult(subtask=subtask, output=output, executed=True)
                )
            except Exception as e:
                sr = SubtaskResult(
                    subtask=subtask, error=str(e), executed=False
                )
                result.results.append(sr)
                result.error = str(e)
                break

        result.shared_state = state
        return result
