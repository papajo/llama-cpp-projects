"""Graph nodes — draft, verify, improve."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from draft_verify_graph.llm import LlamaClient

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = "You are a helpful assistant. Produce a clear, accurate response."

VERIFY_SYSTEM = """You are a quality reviewer. Evaluate the response against the task.

Return a JSON object with:
- "score": integer 0-10
- "issues": list of specific problems found
- "verdict": "pass" if score >= 7, otherwise "fail"

Criteria:
- 0-3: Incorrect, off-topic, or incoherent
- 4-6: Partially correct but has significant issues
- 7-8: Mostly correct with minor issues
- 9-10: Fully correct, complete, and clear"""

IMPROVE_SYSTEM = "You are an editor tasked with improving a response based on reviewer feedback."


@dataclass
class DraftNode:
    """Generate an initial draft."""

    client: LlamaClient
    model: Optional[str] = None
    system_prompt: str = DRAFT_SYSTEM
    temperature: float = 0.7

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        return self.client.complete(
            messages, model=self.model, temperature=self.temperature
        )


@dataclass
class VerificationResult:
    score: int       # 0-10
    issues: List[str]
    verdict: str     # "pass" or "fail"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass" and self.score >= 7


@dataclass
class VerifyNode:
    """Critique a draft and return a structured assessment."""

    client: LlamaClient
    model: Optional[str] = None
    system_prompt: str = VERIFY_SYSTEM

    def run(self, task: str, draft: str) -> VerificationResult:
        prompt = (
            f"Task: {task}\n\n"
            f"Response to evaluate:\n{draft}\n\n"
            f"Evaluation:"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = self.client.complete(messages, model=self.model, max_tokens=256)
        return self._parse(raw)

    def _parse(self, raw: str) -> VerificationResult:
        cleaned = re.sub(
            r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE
        )
        try:
            parsed = json.loads(cleaned)
            score = int(parsed.get("score", 0))
            issues = parsed.get("issues", [])
            if isinstance(issues, str):
                issues = [issues]
            verdict = parsed.get("verdict", "fail")
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: find score in text
            match = re.search(r"\b(10|[0-9])\b", cleaned)
            score = int(match.group(1)) if match else 0
            issues = ["Could not parse structured feedback"]
            verdict = "pass" if score >= 7 else "fail"
        return VerificationResult(
            score=max(0, min(10, score)),
            issues=issues if isinstance(issues, list) else [str(issues)],
            verdict=verdict if verdict in ("pass", "fail") else "fail",
        )


@dataclass
class ImproveNode:
    """Improve the draft based on verification feedback."""

    client: LlamaClient
    model: Optional[str] = None
    system_prompt: str = IMPROVE_SYSTEM
    temperature: float = 0.5

    def run(self, task: str, draft: str, issues: List[str]) -> str:
        issues_text = "\n".join(f"- {i}" for i in issues)
        prompt = (
            f"Task: {task}\n\n"
            f"Current response:\n{draft}\n\n"
            f"Reviewer issues to address:\n{issues_text}\n\n"
            f"Please provide an improved response that addresses all the "
            f"issues above while staying accurate and clear."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.client.complete(
            messages, model=self.model, temperature=self.temperature
        )
