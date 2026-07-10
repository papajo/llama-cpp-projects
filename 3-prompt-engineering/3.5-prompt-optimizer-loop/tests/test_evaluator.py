"""Tests for prompt_optimizer.evaluator."""

import json
from unittest.mock import patch, MagicMock

import pytest
from prompt_optimizer.evaluator import (
    Score,
    Scorecard,
    BaseEvaluator,
    ExactMatchEvaluator,
    ContainsEvaluator,
    RubricEvaluator,
    CompositeEvaluator,
)


class TestScore:
    def test_defaults(self):
        s = Score(scorer="test", score=0.5)
        assert s.scorer == "test"
        assert s.score == 0.5
        assert s.passed is True
        assert s.detail == ""

    def test_score_bounds(self):
        s = Score("test", 0.0)
        assert s.score == 0.0
        s = Score("test", 1.0)
        assert s.score == 1.0


class TestScorecard:
    def test_empty(self):
        sc = Scorecard(actual="hello")
        assert sc.overall == 0.0
        assert sc.passed is False

    def test_add_and_overall(self):
        sc = Scorecard(actual="hello", expected="world")
        sc.add("exact", 0.0)
        sc.add("contains", 1.0)
        assert sc.overall == 0.5
        assert sc.passed is False  # exact failed

    def test_all_pass(self):
        sc = Scorecard(actual="hi", expected="hi")
        sc.add("exact", 1.0, threshold=0.5)
        assert sc.passed is True
        assert sc.overall == 1.0

    def test_add_clamps_score(self):
        sc = Scorecard(actual="x")
        sc.add("test", 1.5)
        assert sc.scores[0].score == 1.0
        sc.add("test2", -0.5)
        assert sc.scores[1].score == 0.0

    def test_summary(self):
        sc = Scorecard(actual="a")
        sc.add("exact", 0.5)
        summary = sc.summary()
        assert "score" in summary or "overall" in summary or "exact" in summary


class TestExactMatchEvaluator:
    def test_exact_match(self):
        ev = ExactMatchEvaluator()
        r = ev("hello", "hello")
        assert r.score == 1.0
        assert r.passed is True

    def test_no_match(self):
        ev = ExactMatchEvaluator()
        r = ev("hello", "world")
        assert r.score == 0.0
        assert r.passed is False

    def test_strip_default(self):
        ev = ExactMatchEvaluator()
        r = ev("  hello  ", "hello")
        assert r.score == 1.0

    def test_strip_disabled(self):
        ev = ExactMatchEvaluator(strip=False)
        r = ev("  hello  ", "hello")
        assert r.score == 0.0

    def test_no_expected(self):
        ev = ExactMatchEvaluator()
        r = ev("hello", None)
        assert r.score == 0.0
        assert r.passed is False

    def test_callable(self):
        ev = ExactMatchEvaluator()
        assert ev("abc", "abc").passed is True


class TestContainsEvaluator:
    def test_contains(self):
        ev = ContainsEvaluator()
        r = ev("The quick brown fox", "brown")
        assert r.score == 1.0
        assert r.passed is True

    def test_does_not_contain(self):
        ev = ContainsEvaluator()
        r = ev("The quick brown fox", "cat")
        assert r.score == 0.0
        assert r.passed is False

    def test_case_sensitive(self):
        ev = ContainsEvaluator(case_sensitive=True)
        assert ev("Hello World", "world").score == 0.0

    def test_case_insensitive(self):
        ev = ContainsEvaluator(case_sensitive=False)
        assert ev("Hello World", "world").score == 1.0

    def test_no_expected(self):
        ev = ContainsEvaluator()
        r = ev("hello", None)
        assert r.score == 0.0


class TestRubricEvaluator:
    def test_successful_judge_call(self):
        """Mock a successful LLM judge response."""
        ev = RubricEvaluator(server_url="http://mock:8080")

        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"score": 9, "rationale": "Excellent output"}'
                }
            }]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            result = ev("Good translation", "Expected translation")
            assert result.score == 0.9
            assert result.passed is True
            assert "9/10" in result.detail

    def test_low_score_fails(self):
        ev = RubricEvaluator(threshold=0.7)
        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"score": 3, "rationale": "Poor quality"}'
                }
            }]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            result = ev("Bad output", "Expected good output")
            assert result.score == 0.3
            assert result.passed is False

    def test_no_expected(self):
        ev = RubricEvaluator()
        r = ev("actual", None)
        assert r.score == 0.0
        assert r.passed is False

    def test_judge_failure(self):
        ev = RubricEvaluator(server_url="http://mock:8080")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = ConnectionError("Refused")

            result = ev("actual", "expected")
            assert result.score == 0.0
            assert result.passed is False
            assert "failed" in result.detail.lower()

    def test_fallback_score_from_code_fence(self):
        """Response with ```json wrapper."""
        ev = RubricEvaluator()
        mock_response = {
            "choices": [{
                "message": {
                    "content": "```json\n{\"score\": 8, \"rationale\": \"Good\"}\n```"
                }
            }]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_cm = MagicMock()
            mock_cm.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_cm

            result = ev("actual", "expected")
            assert result.score == 0.8
            assert result.passed is True


class TestCompositeEvaluator:
    def test_min_strategy(self):
        comp = CompositeEvaluator(
            [ExactMatchEvaluator(), ContainsEvaluator()],
            strategy="min",
        )
        r = comp("hello world", "hello world")
        assert r.score == 1.0

        r = comp("hello universe", "hello world")
        assert r.score == 0.0  # min: exact=0, contains=0 (doesn't contain "hello world" exactly)

    def test_avg_strategy(self):
        comp = CompositeEvaluator(
            [ExactMatchEvaluator(), ContainsEvaluator(case_sensitive=False)],
            strategy="avg",
        )
        r = comp("Hello world", "hello")
        assert r.score == 0.5  # exact=0, contains=1

    def test_max_strategy(self):
        comp = CompositeEvaluator(
            [ExactMatchEvaluator(), ContainsEvaluator(case_sensitive=False)],
            strategy="max",
        )
        r = comp("Hello world", "hello")
        assert r.score == 1.0  # max(0, 1)
