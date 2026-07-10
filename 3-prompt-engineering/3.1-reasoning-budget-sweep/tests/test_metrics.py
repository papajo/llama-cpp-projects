"""Tests for metrics.py — SweepResult, compute_metrics, helpers."""

import math
from budget_sweep.metrics import (
    SweepResult,
    compute_metrics,
    _char_entropy,
    _repetition_rate,
)


class TestSweepResult:
    def test_is_ok_default(self):
        r = SweepResult(params={}, prompt="hi", run_id=0, output="hello")
        assert r.is_ok

    def test_not_ok_with_error(self):
        r = SweepResult(params={}, prompt="hi", run_id=0, output="", error="fail")
        assert not r.is_ok

    def test_output_words(self):
        r = SweepResult(params={}, prompt="", run_id=0, output="hello world foo")
        assert r.output_words == ["hello", "world", "foo"]
        assert r.output_word_count == 3

    def test_output_char_count(self):
        r = SweepResult(params={}, prompt="", run_id=0, output="hi")
        assert r.output_char_count == 2


class TestCharEntropy:
    def test_entropy_uniform(self):
        # All same char -> entropy 0
        assert _char_entropy("aaa") == 0.0

    def test_entropy_empty(self):
        assert _char_entropy("") == 0.0

    def test_entropy_two_chars(self):
        # "ab" -> 2 chars, each p=0.5 -> -2 * (0.5 * log2(0.5)) = 1.0
        assert _char_entropy("ab") == 1.0

    def test_entropy_non_power_of_two(self):
        e = _char_entropy("hello world")
        assert 2.0 < e < 4.0  # sensible range for ASCII


class TestRepetitionRate:
    def test_empty_or_short(self):
        assert _repetition_rate("") == 0.0
        assert _repetition_rate("hello") == 0.0
        assert _repetition_rate("hello world") == 0.0

    def test_no_repetition(self):
        assert _repetition_rate("the quick brown fox") == 0.0

    def test_with_repetition(self):
        # "the the" is a repeated bigram
        rate = _repetition_rate("the the fox and the the dog")
        assert rate > 0.0


class TestComputeMetrics:
    def test_empty_results(self):
        summaries = compute_metrics([])
        assert summaries == []

    def test_single_result(self):
        results = [
            SweepResult(
                params={"max_tokens": 128, "temperature": 0.0},
                prompt="Say hello",
                run_id=0,
                output="hello world",
                prompt_tokens=5,
                completion_tokens=10,
                elapsed_s=1.0,
            ),
        ]
        summaries = compute_metrics(results)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.avg_word_count == 2  # "hello world"
        assert s.avg_completion_tokens == 10
        assert s.token_efficiency == 0.2  # 2 / 10

    def test_multiple_runs_diversity(self):
        results = [
            SweepResult(
                params={"max_tokens": 256, "temperature": 0.7},
                prompt="Write a sentence",
                run_id=i,
                output=text,
                prompt_tokens=5,
                completion_tokens=20,
                elapsed_s=1.0,
            )
            for i, text in enumerate([
                "one two three four five",
                "six seven eight nine ten",
            ])
        ]
        summaries = compute_metrics(results)
        assert len(summaries) == 1
        s = summaries[0]
        # Two completely different outputs -> diversity ~= 1.0
        assert s.avg_diversity > 0.9
        assert s.avg_word_count == 5
        assert s.avg_completion_tokens == 20

    def test_multiple_configs(self):
        results = [
            SweepResult(
                params={"max_tokens": 64},
                prompt="test",
                run_id=0,
                output="a b c",
                completion_tokens=5,
                elapsed_s=0.5,
            ),
            SweepResult(
                params={"max_tokens": 128},
                prompt="test",
                run_id=0,
                output="a b c d e",
                completion_tokens=10,
                elapsed_s=1.0,
            ),
        ]
        summaries = compute_metrics(results)
        assert len(summaries) == 2

    def test_with_errors(self):
        results = [
            SweepResult(
                params={"max_tokens": 64},
                prompt="test",
                run_id=0,
                output="abc",
                completion_tokens=5,
                elapsed_s=0.5,
            ),
            SweepResult(
                params={"max_tokens": 64},
                prompt="test",
                run_id=1,
                output="",
                error="HTTP 500",
                completion_tokens=None,
                elapsed_s=0.5,
            ),
        ]
        summaries = compute_metrics(results)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.errors == 1
        assert s.total_runs == 2

    def test_token_efficiency_ranking(self):
        """Results with higher words-per-token rank higher."""
        results = [
            SweepResult(
                params={"cfg": "A"},
                prompt="test",
                run_id=0,
                output="one two three",
                completion_tokens=3,  # 3/3 = 1.0
                elapsed_s=1.0,
            ),
            SweepResult(
                params={"cfg": "B"},
                prompt="test",
                run_id=0,
                output="one",
                completion_tokens=5,  # 1/5 = 0.2
                elapsed_s=1.0,
            ),
        ]
        summaries = compute_metrics(results)
        assert len(summaries) == 2
        a = [s for s in summaries if s.params["cfg"] == "A"][0]
        b = [s for s in summaries if s.params["cfg"] == "B"][0]
        assert a.token_efficiency > b.token_efficiency
