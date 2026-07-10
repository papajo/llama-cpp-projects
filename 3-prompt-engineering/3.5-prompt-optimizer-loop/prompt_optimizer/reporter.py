"""
reporter.py — Iteration history tracking and markdown report generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .optimizer import IterationSnapshot


@dataclass
class OptimizationReport:
    """Full report from an optimization run."""

    initial_prompt: str
    iterations: list[IterationSnapshot]
    best_prompt: str
    best_score: float
    final_score: float
    total_iterations: int
    target_score: float
    reached_target: bool
    stopped_early: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """Render the report as GitHub-flavored markdown."""
        lines: list[str] = [
            "# Prompt Optimization Report",
            "",
            f"- **Date:** {self.timestamp}",
            f"- **Total iterations:** {self.total_iterations}",
            f"- **Target score:** {self.target_score:.2f}",
            f"- **Reached target:** {'✅ Yes' if self.reached_target else '❌ No'}",
            f"- **Stopped early:** {'Yes (plateau)' if self.stopped_early else 'No'}",
            f"- **Best score:** {self.best_score:.3f}",
            f"- **Final score:** {self.final_score:.3f}",
            "",
            "## Score Trajectory",
            "",
            "| Iteration | Score | Improvement |",
            "|-----------|-------|-------------|",
        ]

        for i, snap in enumerate(self.iterations):
            delta = ""
            if i > 0:
                d = snap.overall_score - self.iterations[i - 1].overall_score
                delta = f"+{d:.3f}" if d > 0 else (f"{d:.3f}" if d < 0 else "—")

            emoji = "🟢" if snap.overall_score >= self.target_score else (
                "🟡" if snap.overall_score > self.target_score * 0.7 else "🔴")
            lines.append(
                f"| {snap.iteration} | {emoji} {snap.overall_score:.3f} | {delta} |"
            )

        lines.extend([
            "",
            "## Initial Prompt",
            "",
            "```",
            self.initial_prompt,
            "```",
            "",
            "## Best Prompt",
            "",
            "```",
            self.best_prompt,
            "```",
            "",
            "## Iteration Details",
            "",
        ])

        for snap in self.iterations:
            lines.append(f"### Iteration {snap.iteration} — Score {snap.overall_score:.3f}")
            lines.append("")
            lines.append("```")
            lines.append(snap.prompt_text)
            lines.append("```")
            lines.append("")

            # Per-test breakdown
            for idx in sorted(snap.scorecards):
                sc = snap.scorecards[idx]
                status = "✅ PASS" if sc.passed else "❌ FAIL"
                lines.append(f"- **Test {idx}:** {status} (overall: {sc.overall:.2f})")
                for s in sc.scores:
                    lines.append(f"  - {s.scorer}: {s.score:.2f} — {s.detail}")

            if snap.improvement_proposal:
                lines.extend([
                    "",
                    "**Improvement proposal:**",
                    "",
                    "```",
                    snap.improvement_proposal,
                    "```",
                ])

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON for programmatic consumption."""
        return json.dumps({
            "initial_prompt": self.initial_prompt,
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "final_score": self.final_score,
            "total_iterations": self.total_iterations,
            "target_score": self.target_score,
            "reached_target": self.reached_target,
            "stopped_early": self.stopped_early,
            "timestamp": self.timestamp,
            "iterations": [
                {
                    "iteration": s.iteration,
                    "score": s.overall_score,
                    "prompt_text": s.prompt_text,
                    "improvement_proposal": s.improvement_proposal,
                }
                for s in self.iterations
            ],
        }, indent=indent, ensure_ascii=False)

    def write_markdown(self, path: str) -> None:
        """Write the markdown report to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

    def write_json(self, path: str) -> None:
        """Write the JSON report to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def build_report(history: list[IterationSnapshot],
                 initial_prompt: str,
                 target_score: float = 0.9) -> OptimizationReport:
    """Build a report from the optimization loop history."""
    if not history:
        return OptimizationReport(
            initial_prompt=initial_prompt,
            iterations=[],
            best_prompt=initial_prompt,
            best_score=0.0,
            final_score=0.0,
            total_iterations=0,
            target_score=target_score,
            reached_target=False,
            stopped_early=False,
        )

    last = history[-1]
    best = max(history, key=lambda s: s.overall_score)
    return OptimizationReport(
        initial_prompt=initial_prompt,
        iterations=history,
        best_prompt=best.prompt_text,
        best_score=best.overall_score,
        final_score=last.overall_score,
        total_iterations=len(history),
        target_score=target_score,
        reached_target=last.overall_score >= target_score,
        stopped_early=last.stopped_early,
    )
