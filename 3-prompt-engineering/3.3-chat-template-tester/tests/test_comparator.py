"""Tests for the comparison report module."""

from chat_template_tester.comparator import format_comparison
from chat_template_tester.tester import run_test, TemplateTestResult, TestRun


SIMPLE_MSGS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]


class TestFormatComparison:
    def test_empty_run(self):
        run = TestRun(messages=[], results=[], add_generation_prompt=False)
        report = format_comparison(run)
        assert "No results" in report

    def test_single_result(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run, title="Single Test")
        assert "# Single Test" in report
        assert "Input Messages" in report
        assert "Summary" in report
        assert "Formatted Outputs" in report
        assert "chatml" in report

    def test_multiple_results(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml", "llama3", "gemma"])
        report = format_comparison(run)
        # Should have summary table entries
        assert "| chatml |" in report
        assert "| llama3 |" in report
        assert "| gemma |" in report

    def test_summary_has_status(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run)
        assert "✅" in report

    def test_messages_displayed(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run)
        assert "What is the capital of France?" in report
        assert "You are a helpful assistant." in report

    def test_structural_comparison_section(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml", "llama3"])
        report = format_comparison(run)
        assert "Structural Comparison" in report
        assert "Role Markers" in report

    def test_notes_included(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run, notes=["Custom note 1", "Custom note 2"])
        assert "Notes" in report
        assert "Custom note 1" in report
        assert "Custom note 2" in report

    def test_format_all_templates(self):
        run = run_test(SIMPLE_MSGS)
        report = format_comparison(run, title="All Templates")
        assert "# All Templates" in report
        # All 10 template names should appear somewhere
        for r in run.results:
            assert r.template_name in report

    def test_chatml_raw_output(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run, show_special=True, show_raw=True)
        assert "<|im_start|>system" in report

    def test_metrics_displayed(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run)
        assert "Characters:" in report
        assert "Estimated tokens:" in report

    def test_deepseek_unicode_tokens(self):
        """DeepSeek uses unicode special tokens - make sure they render."""
        run = run_test(SIMPLE_MSGS, template_names=["deepseek"])
        report = format_comparison(run)
        assert "deepseek" in report
        # Should contain the deepseek output
        r = run.results[0]
        assert r.error is None

    def test_errors_section_appears_only_when_needed(self):
        """Error section should not appear if all results are ok."""
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run)
        assert "## Errors" not in report

    def test_custom_title(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        report = format_comparison(run, title="Custom Report Title")
        assert "# Custom Report Title" in report

    def test_multi_turn_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        run = run_test(msgs, template_names=["chatml"])
        report = format_comparison(run)
        assert "Hello" in report
        assert "Hi there!" in report
        assert "How are you?" in report
