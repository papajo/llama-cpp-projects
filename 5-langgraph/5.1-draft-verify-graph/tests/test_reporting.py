"""Tests for reporting."""

from draft_verify_graph.graph import DraftVerifyResult, IterationRecord
from draft_verify_graph.nodes import VerificationResult
from draft_verify_graph.reporting import result_to_markdown


def _make_result(passed: bool = True, num_iterations: int = 2):
    result = DraftVerifyResult(
        task="Write a poem",
        final_output="Final answer",
        passed=passed,
    )
    for i in range(num_iterations):
        rec = IterationRecord(
            iteration=i + 1,
            draft=f"Draft {i+1}",
            verification=VerificationResult(
                score=8 if passed else 3,
                issues=[] if passed else ["Issue"],
                verdict="pass" if passed else "fail",
            ),
            improved=f"Improved {i+1}" if i < num_iterations - 1 else None,
        )
        result.iterations.append(rec)
    result.total_calls = num_iterations * 3 - 1
    return result


class TestReporting:
    def test_markdown_contains_task(self):
        md = result_to_markdown(_make_result())
        assert "Write a poem" in md
        assert "Draft-Verify Graph Report" in md

    def test_markdown_passed(self):
        md = result_to_markdown(_make_result(passed=True))
        assert "✅" in md

    def test_markdown_failed(self):
        md = result_to_markdown(_make_result(passed=False))
        assert "❌" in md

    def test_markdown_includes_iterations(self):
        md = result_to_markdown(_make_result(passed=True, num_iterations=3))
        assert "Iteration 1" in md
        assert "Iteration 2" in md
        assert "Iteration 3" in md
