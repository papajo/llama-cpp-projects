"""LLM-as-judge reranker — scores query-document relevance via llama.cpp."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional


class LlamaClient:
    """Lightweight client for llama.cpp chat completions."""

    def __init__(self, server_url: str = "http://localhost:8080", timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        import urllib.error
        import urllib.request

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model is not None:
            payload["model"] = model
        data = json.dumps(payload).encode()
        try:
            req = urllib.request.Request(
                f"{self.server_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            raise LlamaError(str(exc)) from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaError(f"Unexpected response: {body}") from exc


class LlamaError(Exception):
    pass


# ---------------------------------------------------------------------------
# Relevance scorer
# ---------------------------------------------------------------------------

RELEVANCE_SYSTEM_PROMPT = """You are a relevance judge. Evaluate how relevant a document is to a given query.

Return a JSON object with two fields:
- "score": an integer from 0 to 10
- "rationale": a one-sentence explanation

Criteria:
- 0-3: Not relevant or only tangentially related
- 4-6: Somewhat relevant, mentions related topics
- 7-8: Clearly relevant, directly addresses the query
- 9-10: Highly relevant, directly answers or contains key information

Output ONLY valid JSON with no additional text."""


@dataclass
class Reranker:
    """Reranks documents by asking an LLM to score query-document relevance."""

    client: LlamaClient
    model: Optional[str] = None
    system_prompt: str = RELEVANCE_SYSTEM_PROMPT

    def score(self, query: str, document: str) -> float:
        """Score relevance of *document* to *query* on 0.0–1.0."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument: {document}\n\nRelevance score:",
            },
        ]
        raw = self.client.complete(messages, model=self.model, max_tokens=128)
        return self._parse_score(raw)

    def _parse_score(self, raw: str) -> float:
        """Extract a normalised 0.0–1.0 score from LLM output."""
        # Try JSON first
        # Strip code fences
        cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)

        try:
            parsed = json.loads(cleaned)
            score = int(parsed.get("score", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: find any integer 0-10 in the text
            match = re.search(r"\b(10|[0-9])\b", cleaned)
            score = int(match.group(1)) if match else 0

        return max(0.0, min(1.0, score / 10.0))

    def rerank(
        self, query: str, candidates: List[dict]
    ) -> List[dict]:
        """Score each candidate and re-sort descending by score.

        Each candidate is a dict with at least ``"document"``.
        Returns a new list with a ``"relevance_score"`` field added.
        """
        results = []
        for c in candidates:
            score = self.score(query, c["document"])
            results.append({**c, "relevance_score": score})
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Fallback: simple keyword overlap baseline
# ---------------------------------------------------------------------------


def keyword_overlap_score(query: str, document: str) -> float:
    """Simple token-overlap relevance baseline (0.0–1.0)."""
    q_tokens = set(query.lower().split())
    d_tokens = set(document.lower().split())
    if not q_tokens:
        return 0.0
    intersection = q_tokens & d_tokens
    return len(intersection) / len(q_tokens)
