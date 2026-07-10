"""Parallel execution agent — fan-out, concurrent subtasks, fan-in merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

StepFunction = Callable[[Dict[str, Any]], Dict[str, Any]]
MergeFunction = Callable[[List[Dict[str, Any]]], Dict[str, Any]]


@dataclass
class SerialStep:
    """A single sequential step."""

    name: str
    fn: StepFunction
    description: str = ""


@dataclass
class ParallelTask:
    """A single task within a parallel group."""

    name: str
    fn: StepFunction
    description: str = ""


@dataclass
class ParallelGroup:
    """A set of tasks to run concurrently, merged afterward."""

    name: str
    tasks: List[ParallelTask]
    merge_fn: MergeFunction = field(default=lambda outputs: {k: v for d in outputs for k, v in d.items()})

    @property
    def task_count(self) -> int:
        return len(self.tasks)


@dataclass
class TaskResult:
    task_name: str
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    succeeded: bool = True


@dataclass
class ParallelGroupResult:
    group_name: str
    task_results: List[TaskResult] = field(default_factory=list)
    merged_output: Dict[str, Any] = field(default_factory=dict)
    all_succeeded: bool = True

    @property
    def num_tasks(self) -> int:
        return len(self.task_results)


@dataclass
class StepRecord:
    step_name: str
    step_index: int
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    group_result: Optional[ParallelGroupResult] = None
    error: Optional[str] = None


@dataclass
class ParallelWorkflowResult:
    steps: List[StepRecord] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    group_results: List[ParallelGroupResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def num_steps(self) -> int:
        return len(self.steps)


def _default_merge(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge by simple dict update — later keys overwrite earlier."""
    merged: Dict[str, Any] = {}
    for d in outputs:
        merged.update(d)
    return merged


@dataclass
class ParallelAgent:
    """Agent that executes serial steps and parallel groups.

    Workflow: serial pre-steps → parallel group(s) → serial post-steps.
    Each parallel group fans out to concurrent tasks, then merges their
    outputs back into state.
    """

    pre_steps: List[SerialStep] = field(default_factory=list)
    parallel_groups: List[ParallelGroup] = field(default_factory=list)
    post_steps: List[SerialStep] = field(default_factory=list)
    max_workers: int = 4

    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> ParallelWorkflowResult:
        state = dict(initial_state or {})
        result = ParallelWorkflowResult()
        idx = [0]

        def _exec_serial(step: SerialStep) -> Optional[str]:
            before = dict(state)
            try:
                output = step.fn(before)
                state.update(output)
            except Exception as e:
                result.steps.append(StepRecord(
                    step_name=step.name, step_index=idx[0],
                    state_before=before, state_after=dict(state), error=str(e),
                ))
                result.error = str(e)
                return str(e)
            result.steps.append(StepRecord(
                step_name=step.name, step_index=idx[0],
                state_before=before, state_after=dict(state),
            ))
            idx[0] += 1
            return None

        def _exec_parallel_group(group: ParallelGroup) -> Optional[str]:
            before = dict(state)

            gr = ParallelGroupResult(group_name=group.name)

            if not group.tasks:
                result.group_results.append(gr)
                result.steps.append(StepRecord(
                    step_name=group.name, step_index=idx[0],
                    state_before=before, state_after=dict(state),
                    group_result=gr,
                ))
                idx[0] += 1
                return None

            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=min(self.max_workers, group.task_count)) as pool:
                fut_map = {}
                for task in group.tasks:
                    fut = pool.submit(task.fn, before)
                    fut_map[fut] = task

                for fut in as_completed(fut_map):
                    task = fut_map[fut]
                    try:
                        output = fut.result()
                        gr.task_results.append(TaskResult(
                            task_name=task.name, output=output, succeeded=True
                        ))
                    except Exception as e:
                        gr.task_results.append(TaskResult(
                            task_name=task.name, error=str(e), succeeded=False
                        ))
                        gr.all_succeeded = False

            if gr.all_succeeded:
                outputs = [tr.output for tr in gr.task_results if tr.succeeded]
                gr.merged_output = group.merge_fn(outputs)
                state.update(gr.merged_output)
            else:
                errors = [tr.error for tr in gr.task_results if tr.error]
                result.group_results.append(gr)
                result.error = f"Parallel group '{group.name}' failed: {'; '.join(errors)}"
                result.steps.append(StepRecord(
                    step_name=group.name, step_index=idx[0],
                    state_before=before, state_after=dict(state),
                    group_result=gr, error=result.error,
                ))
                idx[0] += 1
                return result.error

            result.group_results.append(gr)
            result.steps.append(StepRecord(
                step_name=group.name, step_index=idx[0],
                state_before=before, state_after=dict(state),
                group_result=gr,
            ))
            idx[0] += 1
            return None

        # Pre-steps
        for step in self.pre_steps:
            err = _exec_serial(step)
            if err:
                result.final_state = dict(state)
                return result

        # Parallel groups
        for group in self.parallel_groups:
            err = _exec_parallel_group(group)
            if err:
                result.final_state = dict(state)
                return result

        # Post-steps
        for step in self.post_steps:
            err = _exec_serial(step)
            if err:
                result.final_state = dict(state)
                return result

        result.final_state = dict(state)
        return result
