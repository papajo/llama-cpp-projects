"""Draft-verify graph — the reflexion loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from draft_verify_graph.nodes import (
    DraftNode,
    ImproveNode,
    VerificationResult,
    VerifyNode,
)


@dataclass
class IterationRecord:
    iteration: int
    draft: str
    verification: VerificationResult
    improved: Optional[str] = None


@dataclass
class DraftVerifyResult:
    task: str
    final_output: str = ""
    passed: bool = False
    iterations: List[IterationRecord] = field(default_factory=list)
    total_calls: int = 0

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)


@dataclass
class DraftVerifyGraph:
    """Reflexion loop: draft → verify → (improve → verify) × N.

    Stops when verification passes or max_iterations reached.
    """

    draft_node: DraftNode
    verify_node: VerifyNode
    improve_node: ImproveNode
    max_iterations: int = 5
    min_score: int = 7

    def run(self, task: str) -> DraftVerifyResult:
        result = DraftVerifyResult(task=task)
        calls = 0

        # 1. Initial draft
        draft = self.draft_node.run(task)
        calls += 1

        for iteration in range(1, self.max_iterations + 1):
            # 2. Verify
            verification = self.verify_node.run(task, draft)
            calls += 1

            record = IterationRecord(
                iteration=iteration,
                draft=draft,
                verification=verification,
            )

            if verification.passed or iteration == self.max_iterations:
                result.final_output = draft
                result.passed = verification.passed
                result.iterations.append(record)
                result.total_calls = calls
                return result

            # 3. Improve
            improved = self.improve_node.run(
                task, draft, verification.issues
            )
            calls += 1
            record.improved = improved
            result.iterations.append(record)
            draft = improved

        # Should not reach here due to loop logic, but safety:
        result.final_output = draft
        result.passed = False
        result.total_calls = calls
        return result
