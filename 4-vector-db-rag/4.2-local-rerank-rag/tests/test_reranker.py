"""Tests for the reranker module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from local_rerank_rag.reranker import (
    LlamaClient,
    LlamaError,
    Reranker,
    keyword_overlap_score,
)


def _mock_llm_response(content: str):
    """Build a canned chat completions response dict."""
    return {
        "choices": [{"message": {"content": content}}]
    }


class TestKeywordOverlap:
    def test_full_overlap(self):
        assert keyword_overlap_score("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        s = keyword_overlap_score("hello world", "hello there")
        assert s == 0.5

    def test_no_overlap(self):
        assert keyword_overlap_score("hello world", "foo bar") == 0.0

    def test_empty_query(self):
        assert keyword_overlap_score("", "hello world") == 0.0


class TestLlamaClient:
    def test_success(self):
        mock_resp = _mock_llm_response("hello world")
        with patch("urllib.request.urlopen") as m:
            mc = MagicMock()
            mc.read.return_value = json.dumps(mock_resp).encode()
            m.return_value.__enter__.return_value = mc
            client = LlamaClient("http://mock:8080")
            result = client.complete([{"role": "user", "content": "hi"}])
            assert result == "hello world"

    def test_error(self):
        with patch("urllib.request.urlopen") as m:
            from urllib.error import URLError
            m.side_effect = URLError("fail")
            client = LlamaClient("http://mock:8080")
            with pytest.raises(LlamaError):
                client.complete([{"role": "user", "content": "hi"}])


class TestReranker:
    def _reranker_with_mock(self, mock_urlopen, response_content: str):
        mock_resp = _mock_llm_response(response_content)
        mc = MagicMock()
        mc.read.return_value = json.dumps(mock_resp).encode()
        mock_urlopen.return_value.__enter__.return_value = mc
        client = LlamaClient("http://mock:8080")
        return Reranker(client=client)

    def test_score_parses_json(self):
        with patch("urllib.request.urlopen") as m:
            reranker = self._reranker_with_mock(
                m, '{"score": 8, "rationale": "Directly relevant."}'
            )
            score = reranker.score("test query", "test doc")
            assert score == 0.8

    def test_score_low(self):
        with patch("urllib.request.urlopen") as m:
            reranker = self._reranker_with_mock(
                m, '{"score": 2, "rationale": "Not relevant."}'
            )
            score = reranker.score("test query", "test doc")
            assert score == 0.2

    def test_score_code_fence(self):
        with patch("urllib.request.urlopen") as m:
            reranker = self._reranker_with_mock(
                m, '```json\n{"score": 9, "rationale": "Very relevant."}\n```'
            )
            score = reranker.score("test query", "test doc")
            assert score == 0.9

    def test_score_fallback_to_integer(self):
        with patch("urllib.request.urlopen") as m:
            reranker = self._reranker_with_mock(
                m, "The relevance is 7 out of 10."
            )
            score = reranker.score("test query", "test doc")
            assert score == 0.7

    def test_score_no_number_fallback_zero(self):
        with patch("urllib.request.urlopen") as m:
            reranker = self._reranker_with_mock(
                m, "I have no opinion on relevance."
            )
            score = reranker.score("test query", "test doc")
            assert score == 0.0

    def test_rerank_reorders(self):
        with patch("urllib.request.urlopen") as m:
            # Return decreasing scores so reranking reverses the order
            responses = [
                '{"score": 3, "rationale": "Low."}',
                '{"score": 8, "rationale": "High."}',
                '{"score": 5, "rationale": "Medium."}',
            ]
            mock_cm = MagicMock()
            mock_cm.read.side_effect = [
                json.dumps(_mock_llm_response(r)).encode() for r in responses
            ]
            m.return_value.__enter__.return_value = mock_cm

            client = LlamaClient("http://mock:8080")
            reranker = Reranker(client=client)
            candidates = [
                {"index": 0, "document": "doc low"},
                {"index": 1, "document": "doc high"},
                {"index": 2, "document": "doc medium"},
            ]
            reranked = reranker.rerank("test query", candidates)
            # Should be reordered: high (8), medium (5), low (3)
            assert reranked[0]["index"] == 1
            assert reranked[1]["index"] == 2
            assert reranked[2]["index"] == 0
            assert reranked[0]["relevance_score"] == 0.8
