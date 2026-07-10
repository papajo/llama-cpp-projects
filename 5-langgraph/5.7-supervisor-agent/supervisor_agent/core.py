"""Supervisor agent — hierarchical delegation to sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

SubAgentFn = Callable[[Dict[str, Any]], Dict[str, Any]]
RouterFn = Callable[[str, Dict[str, Any], List["SubAgent"]], List["SubAgent"]]


class DelegationStrategy(Enum):
    FIRST_MATCH = "first_match"     # route to first matching sub-agent
    ALL_MATCH = "all_match"         # route to all matching sub-agents, merge results
    CUSTOM = "custom"               # use custom router function


@dataclass
class SubAgent:
    """A named sub-agent with a capability and execution function."""

    name: str
    capability: str
    fn: SubAgentFn
    description: str = ""

    def can_handle(self, task: str) -> bool:
        """Check if this sub-agent's capability matches the task."""
        keywords = self.capability.lower().split()
        task_lower = task.lower()
        return any(kw in task_lower for kw in keywords)


@dataclass
class DelegationRecord:
    sub_agent: str
    capability: str
    input_state: Dict[str, Any]
    output: Dict[str, Any] = field(default_factory=dict)
    selected: bool = False
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class SupervisorResult:
    task: str
    strategy: DelegationStrategy
    records: List[DelegationRecord] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    selected_agents: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def num_selected(self) -> int:
        return len(self.selected_agents)

    @property
    def all_succeeded(self) -> bool:
        return all(r.succeeded for r in self.records if r.selected)

    @property
    def num_succeeded(self) -> int:
        return sum(1 for r in self.records if r.selected and r.succeeded)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.records if r.selected and r.error)


def _default_router(
    task: str, state: Dict[str, Any], agents: List[SubAgent]
) -> List[SubAgent]:
    """Default FIRST_MATCH router — return first agent whose capability matches."""
    for agent in agents:
        if agent.can_handle(task):
            return [agent]
    return []


def _all_match_router(
    task: str, state: Dict[str, Any], agents: List[SubAgent]
) -> List[SubAgent]:
    """Return ALL agents whose capability matches."""
    return [a for a in agents if a.can_handle(task)]


@dataclass
class SupervisorAgent:
    """Supervisor that routes tasks to sub-agents based on capability.

    Strategies:
      - FIRST_MATCH: delegate to the first sub-agent whose capability matches
      - ALL_MATCH: delegate to all matching sub-agents, merging outputs
      - CUSTOM: use a provided router function
    """

    sub_agents: List[SubAgent]
    strategy: DelegationStrategy = DelegationStrategy.FIRST_MATCH
    router: Optional[RouterFn] = None

    def delegate(self, task: str, state: Optional[Dict[str, Any]] = None) -> SupervisorResult:
        current_state = dict(state or {})
        result = SupervisorResult(
            task=task,
            strategy=self.strategy,
        )

        # Select sub-agents
        selected: List[SubAgent] = []
        if self.strategy == DelegationStrategy.CUSTOM and self.router is not None:
            selected = self.router(task, current_state, self.sub_agents)
        elif self.strategy == DelegationStrategy.FIRST_MATCH:
            selected = _default_router(task, current_state, self.sub_agents)
        elif self.strategy == DelegationStrategy.ALL_MATCH:
            selected = _all_match_router(task, current_state, self.sub_agents)
        else:
            selected = _default_router(task, current_state, self.sub_agents)

        result.selected_agents = [a.name for a in selected]

        # Record all agents (selected or not) for transparency
        for agent in self.sub_agents:
            rec = DelegationRecord(
                sub_agent=agent.name,
                capability=agent.capability,
                input_state=dict(current_state),
                selected=agent in selected,
            )
            result.records.append(rec)

        if not selected:
            result.final_state = dict(current_state)
            return result

        # Execute selected sub-agents
        if self.strategy == DelegationStrategy.FIRST_MATCH:
            agent = selected[0]
            try:
                output = agent.fn(dict(current_state))
                current_state.update(output)
                # Update the record
                for rec in result.records:
                    if rec.sub_agent == agent.name:
                        rec.output = output
                        break
            except Exception as e:
                for rec in result.records:
                    if rec.sub_agent == agent.name:
                        rec.error = str(e)
                        break
                result.error = f"Sub-agent '{agent.name}' failed: {e}"
                result.final_state = dict(current_state)
                return result

        else:  # ALL_MATCH or CUSTOM with multiple
            for agent in selected:
                try:
                    output = agent.fn(dict(current_state))
                    current_state.update(output)
                    for rec in result.records:
                        if rec.sub_agent == agent.name:
                            rec.output = output
                            break
                except Exception as e:
                    for rec in result.records:
                        if rec.sub_agent == agent.name:
                            rec.error = str(e)
                            break
                    result.error = f"Sub-agent '{agent.name}' failed: {e}"
                    result.final_state = dict(current_state)
                    return result

        result.final_state = dict(current_state)
        return result
