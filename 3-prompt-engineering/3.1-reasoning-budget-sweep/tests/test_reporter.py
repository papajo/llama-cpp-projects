"""Tests for reporter.py — format_report and helpers."""

from budget_sweep.metrics import ConfigSummary
from budget_sweep.reporter import format_report, format_per_prompt


def _make_summary(
    params,
    token_efficiency=1.0,
    avg_word_count=50.0,
    avg_completion_tokens=50.0,
    avg_diversity=0.5,
    avg_entropy=3.0,
    avg_repetition_rate=0.1,
    avg_speed_tok_s=10.0,
    time_efficiency=5.0,
    total_runs=5,
    errors=0,
    n_prompts=1,
) -> ConfigSummary:
    return ConfigSummary(
        params=params,
        n_prompts=n_prompts,
        total_runs=total_runs,
        errors=errors,
        avg_word_count=avg_word_count,
        avg_completion_tokens=avg_completion_tokens,
        avg_speed_tok_s=avg_speed_tok_s,
        avg_diversity=avg_diversity,
        avg_entropy=avg_entropy,
        avg_repetition_rate=avg_repetition_rate,
        token_efficiency=token_efficiency,
        time_efficiency=time_efficiency,
    )


class TestFormatReport:
    def test_empty_report(self):
        report = format_report([], title="Empty Test")
        assert "Empty Test" in report
        assert "No results to report" in report

    def test_single_config_report(self):
        summaries = [
            _make_summary({"temperature": 0.0, "max_tokens": 128}),
        ]
        report = format_report(summaries, title="Single Config")
        assert "Single Config" in report
        assert "T=0.0" in report
        assert "Efficiency" in report

    def test_leaderboard_ordering(self):
        """Configs should be sorted by token efficiency desc."""
        summaries = [
            _make_summary({"temperature": 0.0}, token_efficiency=0.5),
            _make_summary({"temperature": 1.0}, token_efficiency=2.0),
            _make_summary({"temperature": 0.7}, token_efficiency=1.0),
        ]
        report = format_report(summaries)
        # The efficiency leaderboard section is between the "Leaderboard by Token Efficiency"
        # header and the "Leaderboard by Diversity" header
        start = report.index("Leaderboard by Token Efficiency")
        end = report.index("Leaderboard by Diversity")
        eff_section = report[start:end]
        # Count the data rows in the efficiency table (starts with | followed by a number)
        lines = eff_section.splitlines()
        data_rows = [l for l in lines if l.strip().startswith("| ") and not l.strip().startswith("| Rank") and not l.strip().startswith("|--")]
        assert len(data_rows) == 3
        # First row should have efficiency 2.00 (T=1.0)
        assert "2.00" in data_rows[0]
        # Check all three are present
        assert any("T=1.0" in l for l in data_rows)
        assert any("T=0.7" in l for l in data_rows)
        assert any("T=0.0" in l for l in data_rows)

    def test_diversity_leaderboard_has_header(self):
        """Diversity leaderboard table is present."""
        summaries = [
            _make_summary({"temperature": 0.0}, avg_diversity=0.9),
            _make_summary({"temperature": 0.7}, avg_diversity=0.3),
        ]
        report = format_report(summaries)
        # Diversity header should be after the efficiency section
        assert "## Leaderboard by Diversity" in report

    def test_speed_table(self):
        """Speed table present with Tokens/s column."""
        summaries = [
            _make_summary({"temperature": 0.0}, avg_speed_tok_s=5.0),
            _make_summary({"temperature": 1.0}, avg_speed_tok_s=50.0),
        ]
        report = format_report(summaries)
        assert "Tokens/s" in report
        assert "Generation Speed" in report

    def test_notes_appended(self):
        summaries = [_make_summary({})]
        report = format_report(summaries, notes=["First note", "Second note"])
        assert "## Notes" in report
        assert "First note" in report
        assert "Second note" in report

    def test_best_token_efficiency_section(self):
        """Best efficiency section highlights the top config."""
        summaries = [
            _make_summary({"temperature": 0.0}, token_efficiency=5.0),
            _make_summary({"temperature": 1.0}, token_efficiency=0.1),
        ]
        report = format_report(summaries)
        assert "Best Token Efficiency" in report
        assert "5.00" in report  # formatted efficiency


class TestFormatPerPrompt:
    def test_basic_output(self):
        summaries = [_make_summary({"max_tokens": 128}, n_prompts=3)]
        output = format_per_prompt(summaries)
        assert "Per-Prompt Breakdown" in output
        assert "3" in output  # n_prompts
