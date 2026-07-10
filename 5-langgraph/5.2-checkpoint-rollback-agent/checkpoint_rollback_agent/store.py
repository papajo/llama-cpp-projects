"""Immutable state snapshots for agent workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Checkpoint:
    """An immutable snapshot of agent state at a point in time."""

    step_index: int
    step_name: str
    state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"[{self.step_index}] {self.step_name}"


class CheckpointStore:
    """An append-only store of checkpoints.

    Checkpoints can be retrieved by index for rollback.
    """

    def __init__(self):
        self._checkpoints: Dict[int, Checkpoint] = {}
        self._next_index: int = 0

    @property
    def latest(self) -> Optional[Checkpoint]:
        if not self._checkpoints:
            return None
        return self._checkpoints[max(self._checkpoints)]

    @property
    def count(self) -> int:
        return len(self._checkpoints)

    def save(self, step_name: str, state: Dict[str, Any],
             metadata: Optional[Dict[str, Any]] = None) -> Checkpoint:
        cp = Checkpoint(
            step_index=self._next_index,
            step_name=step_name,
            state=state,
            metadata=metadata or {},
        )
        self._checkpoints[cp.step_index] = cp
        self._next_index += 1
        return cp

    def get(self, step_index: int) -> Optional[Checkpoint]:
        return self._checkpoints.get(step_index)

    def rollback(self, step_index: int) -> Optional[Checkpoint]:
        """Return checkpoint at step_index and discard all later ones."""
        target = self.get(step_index)
        if target is None:
            return None
        # Discard everything after target
        keys_to_discard = [k for k in self._checkpoints if k > step_index]
        for k in keys_to_discard:
            del self._checkpoints[k]
        self._next_index = step_index + 1
        return target

    def list_checkpoints(self) -> list[Checkpoint]:
        return [self._checkpoints[i] for i in sorted(self._checkpoints)]

    def clear(self) -> None:
        self._checkpoints.clear()
        self._next_index = 0
