"""
optimizer.py — The prompt optimization loop.

Orchestrates: render prompt → call LLM → evaluate → improve → repeat.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .prompt import PromptTemplate, PromptTemplateError, TestCase
from .evaluator import BaseEvaluator, Scorecard

logger = logging.getLogger(__name__)

# ─── Snapshot ──────────────────────────────────────────────────────


@dataclass
class IterationSnapshot:
    """State captured at one iteration of the optimization loop."""

    iteration: int
    prompt_text: str
    outputs: dict[int, str]  # test-case index → LLM output
    scorecards: dict[int, Scorecard]  # test-case index → scorecard
    overall_score: float
    improvement_proposal: str | None = None
    stopped_early: bool = False


# ─── Meta-Prompt Templates ─────────────────────────────────────────

DEFAULT_META_PROMPT = """\
You are optimizing a system prompt for an LLM. Your job is to suggest an \
improved version based on test results.

## Current Prompt

```
{current_prompt}
```

## Evaluation Results

Overall score: {overall_score:.2f} / 1.00

{test_results}

## Instructions

Suggest an improved version of the prompt. Focus on:
1. Fixing any misunderstandings revealed by failing tests.
2. Adding clarifying instructions for edge cases.
3. Making the prompt more precise and directive.

Output ONLY the new prompt text with no preamble, no explanation, \
and no markdown formatting outside the prompt itself.
"""

QUICK_META_PROMPT = """\
Rewrite this prompt to fix the issues shown below.

CURRENT:
{current_prompt}

SCORE: {overall_score:.2f}

FAILURES:
{test_results}

Output ONLY the improved prompt, nothing else.
"""


# ─── LLM Client ────────────────────────────────────────────────────


class LlamaClient:
    """Minimal client for a local llama.cpp OpenAI-compatible endpoint."""

    def __init__(self, server_url: str = "http://localhost:8080",
                 model: str = "", temperature: float = 0.7,
                 max_tokens: int = 512) -> None:
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        """Send a chat completion request and return the response text."""
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None
                           else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None
                          else self.max_tokens,
        }
        if self.model:
            payload["model"] = self.model

        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        return body["choices"][0]["message"]["content"].strip()


# ─── Optimization Loop ─────────────────────────────────────────────


@dataclass
class OptimizationLoop:
    """The core optimization engine.

    Args:
        template: The PromptTemplate to optimize.
        test_cases: List of TestCase to evaluate against.
        evaluator: An evaluator (or CompositeEvaluator).
        llm_client: Client for generation and improvement.
        min_score: Stop when overall score >= this (default 0.9).
        max_iterations: Hard stop after this many iterations (default 10).
        plateau_window: Stop if no improvement for N iterations (default 3).
        meta_prompt: Template for the improvement prompt. Available
                     placeholders: ``current_prompt``, ``overall_score``,
                     ``test_results``.
        improvement_temperature: Temperature for the meta-prompt call.
        verbose: If True, log per-iteration details.
        progress_callback: Called after each iteration with the snapshot.
    """

    template: PromptTemplate
    test_cases: list[TestCase]
    evaluator: BaseEvaluator
    llm_client: LlamaClient
    min_score: float = 0.9
    max_iterations: int = 10
    plateau_window: int = 3
    meta_prompt: str = DEFAULT_META_PROMPT
    improvement_temperature: float = 0.7
    verbose: bool = True
    progress_callback: Callable[[IterationSnapshot], None] | None = None

    # Internal state
    _history: list[IterationSnapshot] = field(default_factory=list, repr=False)
    _current_prompt: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._current_prompt = self.template.template

    # ── Public API ──────────────────────────────────────────────

    @property
    def history(self) -> list[IterationSnapshot]:
        """Read-only view of iteration history."""
        return list(self._history)

    @property
    def best_score(self) -> float:
        """Best overall score achieved so far."""
        if not self._history:
            return 0.0
        return max(s.overall_score for s in self._history)

    @property
    def best_prompt(self) -> str:
        """Prompt text from the best-scoring iteration."""
        if not self._history:
            return self._current_prompt
        return max(self._history, key=lambda s: s.overall_score).prompt_text

    def run(self) -> list[IterationSnapshot]:
        """Execute the optimization loop. Returns the full history."""
        self._history = []
        self._current_prompt = self.template.template

        for iteration in range(1, self.max_iterations + 1):
            snapshot = self._run_iteration(iteration)
            self._history.append(snapshot)

            if self.progress_callback:
                self.progress_callback(snapshot)

            if self.verbose:
                logger.info(
                    "Iteration %d: score=%.3f  best=%.3f  target=%.3f%s",
                    iteration,
                    snapshot.overall_score,
                    self.best_score,
                    self.min_score,
                    "  ✗" if snapshot.stopped_early else "",
                )

            # Check stop conditions
            if snapshot.overall_score >= self.min_score:
                if self.verbose:
                    logger.info("Target score reached — stopping")
                break

            if snapshot.stopped_early:
                if self.verbose:
                    logger.info("Plateau detected — stopping")
                break

        return self._history

    # ── Internal ────────────────────────────────────────────────

    def _run_iteration(self, iteration: int) -> IterationSnapshot:
        """Run one iteration: generate outputs, evaluate, improve."""
        outputs: dict[int, str] = {}
        scorecards: dict[int, Scorecard] = {}

        for i, tc in enumerate(self.test_cases):
            # Render the prompt with the test case variables
            try:
                rendered = self.template.render(tc.input_vars)
            except PromptTemplateError:
                # Fallback: render with current_prompt if template changed
                rendered = self._format_prompt(tc.input_vars)
                if rendered is None:
                    scorecards[i] = Scorecard(
                        actual="", expected=tc.expected)
                    scorecards[i].add(
                        "render", 0.0,
                        detail=f"Missing vars: {set(tc.input_vars)}")
                    continue

            # Call LLM
            try:
                output = self.llm_client.complete(rendered)
            except Exception as exc:
                logger.warning("LLM call failed for test %d: %s", i, exc)
                output = ""
                scorecards[i] = Scorecard(
                    actual="", expected=tc.expected)
                scorecards[i].add(
                    "generation", 0.0,
                    detail=f"LLM call failed: {exc}")
                outputs[i] = ""
                continue

            outputs[i] = output

            # Evaluate
            sc = Scorecard(actual=output, expected=tc.expected)
            score = self.evaluator(output, tc.expected)
            sc.scores.append(score)
            scorecards[i] = sc

        # Compute overall score
        all_scores = [sc.overall for sc in scorecards.values()]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Check plateau
        stopped_early = False
        if len(self._history) >= self.plateau_window:
            recent = [h.overall_score for h in self._history[-self.plateau_window:]]
            if all(abs(s - recent[0]) < 0.01 for s in recent):
                stopped_early = True

        # Improvement step (skip if already at target)
        improvement_proposal: str | None = None
        if overall < self.min_score and not stopped_early:
            improvement_proposal = self._suggest_improvement(
                iteration, overall, scorecards)

        return IterationSnapshot(
            iteration=iteration,
            prompt_text=self._current_prompt,
            outputs=outputs,
            scorecards=scorecards,
            overall_score=overall,
            improvement_proposal=improvement_proposal,
            stopped_early=stopped_early,
        )

    def _format_prompt(self, variables: dict[str, str]) -> str | None:
        """Try to render the current prompt with variables."""
        missing = self.template.variables - set(variables)
        if missing:
            return None
        return self.template.render(variables)

    def _suggest_improvement(
        self,
        iteration: int,
        overall_score: float,
        scorecards: dict[int, Scorecard],
    ) -> str:
        """Use the meta-prompt to ask the LLM for a better prompt."""
        # Build test results summary focusing on failures
        lines: list[str] = []
        for i, (idx, sc) in enumerate(sorted(scorecards.items())):
            tc = self.test_cases[idx]
            status = "✅ PASS" if sc.passed else "❌ FAIL"
            lines.append(f"Test {idx}: {status}")
            if tc.input_vars:
                lines.append(f"  Input: {json.dumps(tc.input_vars)}")
            if tc.expected is not None:
                lines.append(f"  Expected: {tc.expected!r}")
            if not sc.passed:
                actual_preview = sc.actual[:200]
                lines.append(f"  Got: {actual_preview!r}")
                for s in sc.scores:
                    lines.append(f"  {s.scorer}: {s.score:.2f} — {s.detail}")
            if i >= 4:  # limit to 5 test cases
                lines.append("  ... (remaining tests omitted)")
                break

        test_results = "\n".join(lines)

        meta = self.meta_prompt.format(
            current_prompt=self._current_prompt,
            overall_score=overall_score,
            test_results=test_results,
        )

        try:
            new_prompt = self.llm_client.complete(
                meta,
                temperature=self.improvement_temperature,
                max_tokens=1024,
            )
            # Strip code fences if present
            new_prompt = self._strip_fences(new_prompt)

            if new_prompt.strip():
                self._current_prompt = new_prompt.strip()
                # Update the template
                self.template = PromptTemplate(self._current_prompt)
                return new_prompt.strip()
            return "(empty improvement proposal)"
        except Exception as exc:
            logger.warning("Improvement step failed: %s", exc)
            return f"(improvement failed: {exc})"

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences around the prompt text."""
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
