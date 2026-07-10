"""
evaluator.py — Scoring functions for prompt outputs.

Evaluators examine (actual, expected) pairs and return a ``Score``.
A ``Scorecard`` collects multiple scores for aggregation.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─── Data Structures ───────────────────────────────────────────────


@dataclass(frozen=True)
class Score:
    """A single evaluation score.

    Attributes:
        scorer: Name of the evaluator that produced this score.
        score:  Float in [0.0, 1.0] indicating quality.
        passed: Whether this score meets the pass threshold.
        detail: Human-readable justification.
    """

    scorer: str
    score: float
    passed: bool = True
    detail: str = ""


@dataclass
class Scorecard:
    """Collection of scores for one output evaluation."""

    actual: str
    expected: str | None = None
    scores: list[Score] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Weighted or average score across all evaluators."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def passed(self) -> bool:
        """All individual scores passed."""
        return all(s.passed for s in self.scores) if self.scores else False

    def add(self, scorer: str, score: float, *, detail: str = "",
            threshold: float = 0.5) -> None:
        """Append a scored result."""
        self.scores.append(Score(
            scorer=scorer,
            score=max(0.0, min(1.0, score)),
            passed=score >= threshold,
            detail=detail,
        ))

    def summary(self) -> str:
        """Short summary line."""
        parts = [f"{s.scorer}={s.score:.2f}" for s in self.scores]
        return f"overall={self.overall:.2f} passed={self.passed} [{', '.join(parts)}]"


# ─── Base Evaluator ────────────────────────────────────────────────


class BaseEvaluator(ABC):
    """Abstract evaluator. Subclasses implement ``evaluate``."""

    def __init__(self, name: str, threshold: float = 0.5) -> None:
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, actual: str, expected: str | None = None,
                 **kwargs: Any) -> Score:
        """Score an output against the expected reference."""

    def __call__(self, actual: str, expected: str | None = None,
                 **kwargs: Any) -> Score:
        return self.evaluate(actual, expected, **kwargs)


# ─── Built-in Evaluators ───────────────────────────────────────────


class ExactMatchEvaluator(BaseEvaluator):
    """Score 1.0 if actual == expected, else 0.0."""

    def __init__(self, threshold: float = 1.0, strip: bool = True) -> None:
        super().__init__(name="exact_match", threshold=threshold)
        self.strip = strip

    def evaluate(self, actual: str, expected: str | None = None,
                 **kwargs: Any) -> Score:
        if expected is None:
            return Score("exact_match", 0.0, False,
                         "No expected output provided")
        a = actual.strip() if self.strip else actual
        e = expected.strip() if self.strip else expected
        ok = a == e
        return Score(
            self.name, 1.0 if ok else 0.0, ok,
            "Exact match" if ok else
            f"Expected {e!r}, got {a!r}",
        )


class ContainsEvaluator(BaseEvaluator):
    """Score 1.0 if expected substring is in actual, else 0.0."""

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = True) -> None:
        super().__init__(name="contains", threshold=threshold)
        self.case_sensitive = case_sensitive

    def evaluate(self, actual: str, expected: str | None = None,
                 **kwargs: Any) -> Score:
        if expected is None:
            return Score("contains", 0.0, False,
                         "No expected output provided")
        a = actual if self.case_sensitive else actual.lower()
        e = expected if self.case_sensitive else expected.lower()
        ok = e in a
        return Score(
            self.name, 1.0 if ok else 0.0, ok,
            f"Contains {expected!r}" if ok else
            f"Expected to contain {expected!r}",
        )


class RubricEvaluator(BaseEvaluator):
    """LLM-as-judge scorer using a rubric.

    Calls a local llama.cpp server (or any OpenAI-compatible endpoint)
    to score the generated output against a rubric and expected answer.

    The rubric prompt asks the judge model to rate the output on
    correctness, faithfulness, and completeness — returning a JSON
    block with keys ``score`` (int 0-10) and ``rationale`` (str).
    """

    RUBRIC_TEMPLATE = """\
You are evaluating the quality of an LLM-generated output.

## Expected Answer
{expected}

## Actual Output
{actual}

## Rubric
Score the actual output from 0 (terrible) to 10 (perfect) based on:
- **Correctness**: Does it match the expected answer factually?
- **Completeness**: Does it cover all key points?
- **Clarity**: Is it well-written and easy to understand?

## Output Format
Respond with ONLY a JSON object:
{{"score": <int 0-10>, "rationale": "<brief explanation>"}}
"""

    def __init__(
        self,
        threshold: float = 0.7,
        server_url: str = "http://localhost:8080",
        model: str = "",
        rubric_template: str | None = None,
    ) -> None:
        super().__init__(name="rubric", threshold=threshold)
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.rubric_template = rubric_template or self.RUBRIC_TEMPLATE

    def evaluate(self, actual: str, expected: str | None = None,
                 **kwargs: Any) -> Score:
        if expected is None:
            return Score("rubric", 0.0, False,
                         "No expected output provided for rubric")

        prompt = self.rubric_template.format(
            expected=expected.strip(),
            actual=actual.strip(),
        )

        try:
            score_val, rationale = self._call_judge(prompt)
        except Exception as exc:
            logger.warning("Rubric evaluation failed: %s", exc)
            return Score("rubric", 0.0, False,
                         f"Judge call failed: {exc}")

        normalized = score_val / 10.0
        passed = normalized >= self.threshold
        return Score(self.name, normalized, passed,
                     f"Score {score_val}/10 — {rationale}")

    def _call_judge(self, prompt: str) -> tuple[int, str]:
        """Call the local LLM for judgment. Returns (score_int, rationale)."""
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 256,
        }
        if self.model:
            payload["model"] = self.model

        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"].strip()

        # Try to parse JSON from the response
        # The model might wrap in ```json ... ``` or output raw
        json_block = content
        if "```" in content:
            # Extract first JSON block
            for part in content.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    json_block = part
                    break

        try:
            parsed = json.loads(json_block)
            score_val = int(parsed["score"])
            rationale = parsed.get("rationale", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract score via regex
            import re
            m = re.search(r'"score"\s*:\s*(\d+)', content)
            if m:
                score_val = int(m.group(1))
                rationale = content[:200]
            else:
                logger.warning("Could not parse judge output: %s",
                               content[:150])
                score_val = 5  # neutral fallback
                rationale = "Could not parse judge output"

        return max(0, min(10, score_val)), rationale


# ─── Composite Evaluator ───────────────────────────────────────────


class CompositeEvaluator(BaseEvaluator):
    """Runs multiple evaluators and combines scores.

    By default returns the **minimum** score (pessimistic).
    """

    def __init__(self, evaluators: list[BaseEvaluator],
                 strategy: str = "min") -> None:
        super().__init__(name="composite", threshold=0.5)
        self.evaluators = evaluators
        self.strategy = strategy

    def evaluate(self, actual: str, expected: str | None = None,
                 **kwargs: Any) -> Score:
        results = [ev(actual, expected, **kwargs) for ev in self.evaluators]
        scores = [r.score for r in results]

        if self.strategy == "min":
            combined = min(scores)
        elif self.strategy == "max":
            combined = max(scores)
        elif self.strategy == "avg":
            combined = sum(scores) / len(scores)
        else:
            combined = min(scores)

        details = "; ".join(f"{r.scorer}={r.score:.2f}" for r in results)
        return Score(
            self.name, combined,
            combined >= self.threshold,
            details,
        )
