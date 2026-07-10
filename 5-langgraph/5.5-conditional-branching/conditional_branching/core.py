"""Conditional branching agent — route execution along different paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

StatePredicate = Callable[[Dict[str, Any]], bool]
StepFunction = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Condition:
    """A named predicate that checks workflow state."""

    name: str
    predicate: StatePredicate
    description: str = ""

    def evaluate(self, state: Dict[str, Any]) -> bool:
        return self.predicate(state)


@dataclass
class ActionStep:
    """A single action that mutates state."""

    name: str
    fn: StepFunction
    description: str = ""


@dataclass
class Branch:
    """A named branch with a condition and a sequence of steps."""

    name: str
    condition: Condition
    steps: List[ActionStep] = field(default_factory=list)
    is_default: bool = False

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class BranchResult:
    branch_name: str
    condition_met: bool
    was_default: bool = False
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class StepRecord:
    step_name: str
    step_index: int
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    branch: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ConditionalWorkflowResult:
    steps: List[StepRecord] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    branch_taken: Optional[str] = None
    branch_results: List[BranchResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass
class ConditionalAgent:
    """Agent that evaluates conditions and routes execution to matching branches.

    Steps are executed in order until a ``BranchPoint`` is encountered.
    At a branch point, conditions are evaluated (in order) and the first
    matching branch's steps are run. If no condition matches, the default
    branch is used.
    """

    pre_steps: List[ActionStep] = field(default_factory=list)
    branches: List[Branch] = field(default_factory=list)
    post_steps: List[ActionStep] = field(default_factory=list)

    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> ConditionalWorkflowResult:
        state = dict(initial_state or {})
        result = ConditionalWorkflowResult()
        idx = [0]  # mutable counter for step_index

        def _exec_step(step: ActionStep, branch_name: Optional[str] = None) -> Optional[str]:
            before = dict(state)
            try:
                output = step.fn(before)
                state.update(output)
            except Exception as e:
                result.steps.append(StepRecord(
                    step_name=step.name,
                    step_index=idx[0],
                    state_before=before,
                    state_after=dict(state),
                    branch=branch_name,
                    error=str(e),
                ))
                result.error = str(e)
                return str(e)
            result.steps.append(StepRecord(
                step_name=step.name,
                step_index=idx[0],
                state_before=before,
                state_after=dict(state),
                branch=branch_name,
            ))
            idx[0] += 1
            return None

        # Pre-steps
        for step in self.pre_steps:
            err = _exec_step(step)
            if err:
                result.final_state = dict(state)
                return result

        # Branch point — find the first matching branch
        taken_branch: Optional[Branch] = None
        for branch in self.branches:
            br = BranchResult(
                branch_name=branch.name,
                condition_met=branch.condition.evaluate(state),
                was_default=branch.is_default,
            )
            if br.condition_met and taken_branch is None:
                taken_branch = branch
                result.branch_taken = branch.name
                for bstep in branch.steps:
                    err = _exec_step(bstep, branch.name)
                    if err:
                        br.error = err
                        result.branch_results.append(br)
                        result.final_state = dict(state)
                        return result
            result.branch_results.append(br)

        if taken_branch is None:
            # No branch matched — check for default
            for branch in self.branches:
                if branch.is_default:
                    result.branch_taken = branch.name
                    taken_branch = branch
                    br = BranchResult(
                        branch_name=branch.name,
                        condition_met=True,
                        was_default=True,
                    )
                    for bstep in branch.steps:
                        err = _exec_step(bstep, branch.name)
                        if err:
                            br.error = err
                            result.branch_results.append(br)
                            result.final_state = dict(state)
                            return result
                    result.branch_results.append(br)
                    break

        # Post-steps
        for step in self.post_steps:
            err = _exec_step(step)
            if err:
                result.final_state = dict(state)
                return result

        result.final_state = dict(state)
        return result
