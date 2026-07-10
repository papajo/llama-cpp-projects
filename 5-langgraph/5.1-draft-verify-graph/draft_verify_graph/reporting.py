"""Reporting for draft-verify graph runs."""

from __future__ import annotations

from draft_verify_graph.graph import DraftVerifyResult


def result_to_markdown(result: DraftVerifyResult) -> str:
    """Generate a markdown report of a draft-verify run."""
    lines = []
    lines.append("# Draft-Verify Graph Report")
    lines.append("")
    lines.append(f"- **Task:** {result.task}")
    lines.append(f"- **Passed:** {'✅' if result.passed else '❌'}")
    lines.append(f"- **Iterations:** {result.num_iterations}")
    lines.append(f"- **Total LLM calls:** {result.total_calls}")
    lines.append("")

    for i, rec in enumerate(result.iterations):
        lines.append(f"## Iteration {rec.iteration}")
        lines.append("")
        lines.append(f"### Draft")
        lines.append("")
        lines.append(f"```\n{rec.draft}\n```")
        lines.append("")
        lines.append(f"### Verification")
        lines.append("")
        lines.append(f"- **Score:** {rec.verification.score}/10")
        lines.append(f"- **Verdict:** {rec.verification.verdict}")
        if rec.verification.issues:
            lines.append("- **Issues:**")
            for issue in rec.verification.issues:
                lines.append(f"  - {issue}")
        lines.append("")
        if rec.improved is not None:
            lines.append(f"### Improved")
            lines.append("")
            lines.append(f"```\n{rec.improved}\n```")
            lines.append("")

    lines.append("## Final Output")
    lines.append("")
    lines.append(f"```\n{result.final_output}\n```")
    lines.append("")

    return "\n".join(lines)
