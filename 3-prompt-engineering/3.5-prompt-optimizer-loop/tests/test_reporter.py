"""Tests for prompt_optimizer.reporter."""

import json
from prompt_optimizer.reporter import build_report, OptimizationReport
from prompt_optimizer.optimizer import IterationSnapshot
from prompt_optimizer.evaluator import Scorecard


def _make_snap(iteration, score, prompt_text="test prompt",
               improvement=None, stopped=False):
    return IterationSnapshot(
        iteration=iteration,
        prompt_text=prompt_text,
        outputs={0: "output"},
        scorecards={0: Scorecard("output")},
        overall_score=score,
        improvement_proposal=improvement,
        stopped_early=stopped,
    )


class TestBuildReport:
    def test_empty_history(self):
        report = build_report([], initial_prompt="orig", target_score=0.9)
        assert report.total_iterations == 0
        assert report.best_score == 0.0
        assert report.reached_target is False
        assert report.stopped_early is False

    def test_single_iteration(self):
        report = build_report(
            [_make_snap(1, 0.95, "hello")],
            initial_prompt="orig",
            target_score=0.9,
        )
        assert report.total_iterations == 1
        assert report.best_score == 0.95
        assert report.reached_target is True
        assert report.best_prompt == "hello"

    def test_reached_target(self):
        report = build_report(
            [_make_snap(1, 0.8), _make_snap(2, 0.95)],
            initial_prompt="orig",
            target_score=0.9,
        )
        assert report.reached_target is True
        assert report.best_score == 0.95

    def test_not_reached_target(self):
        report = build_report(
            [_make_snap(1, 0.5), _make_snap(2, 0.6)],
            initial_prompt="orig",
            target_score=0.9,
        )
        assert report.reached_target is False
        assert report.final_score == 0.6

    def test_stopped_early(self):
        report = build_report(
            [_make_snap(1, 0.5, stopped=True)],
            initial_prompt="orig",
        )
        assert report.stopped_early is True


class TestOptimizationReport:
    def test_to_markdown_contains_sections(self):
        report = build_report(
            [_make_snap(1, 0.8, "prompt A"),
             _make_snap(2, 0.95, "prompt B", improvement="try adding few-shot")],
            initial_prompt="initial",
            target_score=0.9,
        )
        md = report.to_markdown()
        assert "# Prompt Optimization Report" in md
        assert "Score Trajectory" in md
        assert "## Initial Prompt" in md
        assert "## Best Prompt" in md
        assert "## Iteration Details" in md
        assert "prompt A" in md
        assert "prompt B" in md
        assert "try adding few-shot" in md
        assert "✅" in md or "🟢" in md

    def test_to_json_roundtrip(self):
        report = build_report(
            [_make_snap(1, 0.75, "p1"),
             _make_snap(2, 0.92, "p2")],
            initial_prompt="p0",
            target_score=0.9,
        )
        data = json.loads(report.to_json())
        assert data["initial_prompt"] == "p0"
        assert data["best_prompt"] == "p2"
        assert data["best_score"] == 0.92
        assert len(data["iterations"]) == 2

    def test_write_markdown(self, tmp_path):
        report = build_report(
            [_make_snap(1, 0.8)],
            initial_prompt="orig",
        )
        path = tmp_path / "report.md"
        report.write_markdown(str(path))
        assert path.read_text().startswith("# Prompt Optimization Report")

    def test_write_json(self, tmp_path):
        report = build_report(
            [_make_snap(1, 0.8)],
            initial_prompt="orig",
        )
        path = tmp_path / "report.json"
        report.write_json(str(path))
        data = json.loads(path.read_text())
        assert data["initial_prompt"] == "orig"
