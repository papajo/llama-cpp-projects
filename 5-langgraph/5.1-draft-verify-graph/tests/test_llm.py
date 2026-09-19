"""Tests for LLM client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from draft_verify_graph.llm import LlamaClient, LlamaError

# Mirrors a real llama-server /v1/chat/completions body (build b11046-60081bb2b).
# The thin {"choices":[{"message":{"content":...}}]} shape these tests used to
# assert omitted finish_reason, usage and llama.cpp's timings block; see
# drift-graph.md entry 1. max_tokens-bounded calls really do return
# finish_reason "length", not "stop".
def _chat_envelope(content: str, finish_reason: str = "length") -> dict:
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "index": 0,
            "message": {"role": "assistant", "content": content},
        }],
        "created": 1789829233,
        "model": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0",
        "system_fingerprint": "b11046-60081bb2b",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 8,
            "prompt_tokens": 32,
            "total_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 31},
        },
        "id": "chatcmpl-aKdOb19SjynKzSDgbeclm4DEyuq5XX9z",
        "timings": {
            "cache_n": 31, "prompt_n": 1, "prompt_ms": 72.875,
            "predicted_n": 8, "predicted_ms": 240.826,
            "predicted_per_second": 29.066629018461462,
        },
    }



class TestLlamaClient:
    def test_complete_success(self):
        mock_resp = _chat_envelope("Hello!")
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
