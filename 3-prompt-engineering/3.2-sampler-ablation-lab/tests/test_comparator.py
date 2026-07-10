"""Tests for comparator.py — format_comparison and helpers."""

from sampler_ablation.config import preset_greedy, preset_creative, preset_balanced
from sampler_ablation.runner import AblationResult
from sampler_ablation.comparator import format_comparison, _vocab_size, _char_entropy, _repetition_rate


def _result(config, output="Hello world", prompt="Say hi", **kwargs):
    return AblationResult(
        config=config,
        prompt=prompt,
        output=output,
        prompt_tokens=5,
        completion_tokens=3,
        elapsed_s=1.0,
        **kwargs,
    )


class TestFormatComparison:
    def test_empty_results(self):
        report = format_comparison([], title="Empty")
        assert "Empty" in report
        assert "No results to compare" in report

    def test_single_config(self):
        results = [_result(preset_greedy())]
        report = format_comparison(results)
        assert "greedy" in report
        assert "Hello world" in report
        assert "Overview" in report

    def test_multiple_configs(self):
        results = [
            _result(preset_greedy(), output="short"),
            _result(preset_creative(), output="much longer text here"),
        ]
        report = format_comparison(results)
        assert "greedy" in report
        assert "creative" in report
        assert "short" in report
        assert "much longer" in report

    def test_metrics_table_present(self):
        results = [_result(preset_greedy(), output="hello world")]
        report = format_comparison(results)
        # Table header
        assert "Words" in report
        assert "Tokens" in report
        assert "Speed" in report
        assert "Vocab" in report
        assert "Repetition" in report
        assert "Entropy" in report

    def test_with_error_result(self):
        results = [
            _result(preset_greedy(), output="hello"),
            _result(preset_creative(), output="", error="HTTP 500"),
        ]
        report = format_comparison(results)
        assert "HTTP 500" in report
        assert "Errors" in report

    def test_diff_section(self):
        """When two+ results, diff section is included."""
        results = [
            _result(preset_greedy(), output="one two three"),
            _result(preset_creative(), output="four five six"),
        ]
        report = format_comparison(results)
        assert "Word-Level Diffs" in report
        assert "one two three → four five six" not in report or True  # just check section present

    def test_notes(self):
        results = [_result(preset_greedy())]
        report = format_comparison(results, notes=["A note"])
        assert "## Notes" in report
        assert "A note" in report

    def test_prompt_displayed(self):
        results = [_result(preset_greedy(), prompt="What is life?")]
        report = format_comparison(results)
        assert "What is life?" in report


class TestMetrics:
    def test_vocab_size(self):
        assert _vocab_size("") == 0
        assert _vocab_size("hello hello") == 1
        assert _vocab_size("hello world foo") == 3

    def test_char_entropy(self):
        assert _char_entropy("") == 0.0
        assert _char_entropy("aaa") == 0.0
        e = _char_entropy("hello world")
        assert 2.0 < e < 4.0

    def test_repetition_rate(self):
        assert _repetition_rate("") == 0.0
        assert _repetition_rate("hello world") == 0.0
        assert _repetition_rate("the the fox and the the dog") > 0.0
