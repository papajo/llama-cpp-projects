"""Tests for graph nodes."""

import json
from unittest.mock import MagicMock, patch

import pytest

from draft_verify_graph.nodes import (
    DraftNode,
    ImproveNode,
    VerificationResult,
    VerifyNode,
)


def _mock_chat_response(content: str):
    """Return a MagicMock whose .read() returns the encoded chat-completion JSON."""
    resp = {"choices": [{"message": {"content": content}}]}
    mc = MagicMock()
    mc.read.return_value = json.dumps(resp).encode()
    return mc


class TestDraftNode:
    def test_run_returns_draft(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response("Draft answer")
            from draft_verify_graph.llm import LlamaClient
            node = DraftNode(client=LlamaClient("http://mock:8080"))
            result = node.run("Write a poem")
            assert result == "Draft answer"


class TestVerifyNode:
    def test_run_parses_json(self):
        content = json.dumps({"score": 8, "issues": ["Minor"], "verdict": "pass"})
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response(content)
            from draft_verify_graph.llm import LlamaClient
            node = VerifyNode(client=LlamaClient("http://mock:8080"))
            result = node.run("task", "draft")
            assert result.score == 8
            assert result.passed is True
            assert result.verdict == "pass"

    def test_run_fail(self):
        content = json.dumps({"score": 3, "issues": ["Wrong", "Incomplete"], "verdict": "fail"})
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response(content)
            from draft_verify_graph.llm import LlamaClient
            node = VerifyNode(client=LlamaClient("http://mock:8080"))
            result = node.run("task", "draft")
            assert result.score == 3
            assert result.passed is False
            assert len(result.issues) == 2

    def test_code_fence(self):
        content = "```json\n{\"score\": 9, \"issues\": [], \"verdict\": \"pass\"}\n```"
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response(content)
            from draft_verify_graph.llm import LlamaClient
            node = VerifyNode(client=LlamaClient("http://mock:8080"))
            result = node.run("task", "draft")
            assert result.score == 9
            assert result.passed is True

    def test_fallback_to_integer(self):
        content = "Score: 6 out of 10. Needs improvement."
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response(content)
            from draft_verify_graph.llm import LlamaClient
            node = VerifyNode(client=LlamaClient("http://mock:8080"))
            result = node.run("task", "draft")
            assert result.score == 6
            assert result.passed is False

    def test_clamps_score_range(self):
        content = json.dumps({"score": 15, "issues": [], "verdict": "pass"})
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response(content)
            from draft_verify_graph.llm import LlamaClient
            node = VerifyNode(client=LlamaClient("http://mock:8080"))
            result = node.run("task", "draft")
            assert result.score == 10  # clamped

    def test_issues_as_string(self):
        content = json.dumps({"score": 5, "issues": "Single issue text", "verdict": "fail"})
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response(content)
            from draft_verify_graph.llm import LlamaClient
            node = VerifyNode(client=LlamaClient("http://mock:8080"))
            result = node.run("task", "draft")
            assert isinstance(result.issues, list)
            assert result.issues == ["Single issue text"]


class TestImproveNode:
    def test_run_returns_improvement(self):
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_chat_response("Improved answer")
            from draft_verify_graph.llm import LlamaClient
            node = ImproveNode(client=LlamaClient("http://mock:8080"))
            result = node.run("task", "draft", ["issue 1"])
            assert result == "Improved answer"
