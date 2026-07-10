"""Agent workflow with checkpoint/rollback support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from checkpoint_rollback_agent.store import Checkpoint, CheckpointStore


@dataclass
class AgentStep:
    """A single step in an agent workflow."""

    name: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    description: str = ""


@dataclass
class StepResult:
    step_name: str
    step_index: int
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    checkpoint: Checkpoint
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    steps: List[StepResult] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    rolled_back: bool = False
    rollback_to: Optional[int] = None
    error: Optional[str] = None

    @property
    def num_steps_executed(self) -> int:
        return len(self.steps)


class CheckpointAgent:
    """An agent that checkpoints state before/after each step.

    Supports rollback: re-run from a previous checkpoint, discarding
    everything after it.
    """

    def __init__(self, steps: List[AgentStep]):
        self.steps = steps
        self.store = CheckpointStore()
        self._state: Dict[str, Any] = {}

    @property
    def state(self) -> Dict[str, Any]:
        """Read-only view of current state."""
        return dict(self._state)

    @property
    def latest_checkpoint(self) -> Optional[Checkpoint]:
        return self.store.latest

    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Execute all steps, checkpointing before each one."""
        self._state = dict(initial_state or {})
        result = WorkflowResult()

        for idx, step in enumerate(self.steps):
            try:
                # Snapshot before executing
                cp = self.store.save(
                    step_name=step.name,
                    state=dict(self._state),
                    metadata={"step_index": idx, "description": step.description},
                )

                # Execute
                output = step.fn(dict(self._state))
                self._state.update(output)

                step_result = StepResult(
                    step_name=step.name,
                    step_index=idx,
                    input_state=dict(self._state),
                    output_state=output,
                    checkpoint=cp,
                )
            except Exception as e:
                step_result = StepResult(
                    step_name=step.name,
                    step_index=idx,
                    input_state=dict(self._state),
                    output_state={},
                    checkpoint=self.store.latest or Checkpoint(
                        step_index=idx, step_name=step.name,
                        state=dict(self._state)
                    ),
                    error=str(e),
                )
                result.final_state = dict(self._state)
                result.steps.append(step_result)
                result.error = str(e)
                return result

            result.steps.append(step_result)

        result.final_state = dict(self._state)
        return result

    def rollback_to(self, step_index: int) -> Optional[Dict[str, Any]]:
        """Rollback to a previous checkpoint, restoring and discarding later state."""
        cp = self.store.rollback(step_index)
        if cp is None:
            return None
        self._state = dict(cp.state)
        return dict(self._state)

    def list_checkpoints(self) -> List[Checkpoint]:
        return self.store.list_checkpoints()
