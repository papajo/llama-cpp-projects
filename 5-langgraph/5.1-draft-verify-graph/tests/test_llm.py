"""Tests for LLM client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from draft_verify_graph.llm import LlamaClient, LlamaError


class TestLlamaClient:
    def test_complete_success(self):
        mock_resp = {"choices": [{"message": {"content": "Hello!"}}]}
        with patch("urllib.request.urlopen") as m:
            mc = MagicMock()
            mc.read.return_value = json.dumps(mock_resp).encode()
            m.return_value.__enter__.return_value = mc
            client = LlamaClient("http://mock:8080")
            assert client.complete([{"role": "user", "content": "hi"}]) == "Hello!"

    def test_complete_error(self):
        with patch("urllib.request.urlopen") as m:
            from urllib.error import URLError
            m.side_effect = URLError("fail")
            client = LlamaClient("http://mock:8080")
            with pytest.raises(LlamaError):
                client.complete([{"role": "user", "content": "hi"}])

    def test_complete_bad_response(self):
        mock_resp = {"wrong": "format"}
        with patch("urllib.request.urlopen") as m:
            mc = MagicMock()
            mc.read.return_value = json.dumps(mock_resp).encode()
            m.return_value.__enter__.return_value = mc
            client = LlamaClient("http://mock:8080")
            with pytest.raises(LlamaError):
                client.complete([{"role": "user", "content": "hi"}])
