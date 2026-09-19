"""Tests for the draft-verify graph."""

import json
from unittest.mock import MagicMock, patch

import pytest

from draft_verify_graph.graph import DraftVerifyGraph
from draft_verify_graph.llm import LlamaClient
from draft_verify_graph.nodes import DraftNode, ImproveNode, VerifyNode

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



def _make_side_effect():
    """URL-aware mock: chat completions return canned JSON."""
    # Track call count to rotate through responses
    call_count = [0]

    def side_effect(request, *args, **kwargs):
        url = request.get_full_url()
        nonlocal call_count
        idx = call_count[0]
        call_count[0] += 1

        if "/v1/chat/completions" in url:
            body = request.data
            import json as _json
            payload = _json.loads(body)

            # Check the system prompt to determine which node is calling
            sys_content = ""
            for m in payload.get("messages", []):
                if m.get("role") == "system":
                    sys_content = m.get("content", "")

            if "quality reviewer" in sys_content:
                # Verify node: pass on first call, fail otherwise
                if idx < 2:  # first verify call
                    response_text = _json.dumps(
                        {"score": 8, "issues": [], "verdict": "pass"}
                    )
                elif idx < 3:  # second verify call
                    response_text = _json.dumps(
                        {"score": 3, "issues": ["Too short"], "verdict": "fail"}
                    )
                else:
                    response_text = _json.dumps(
                        {"score": 9, "issues": [], "verdict": "pass"}
                    )
            elif "editor" in sys_content:
                # Improve node
                response_text = "Improved response text here."
            else:
                # Draft node
                response_text = "This is a draft response."

            resp = _chat_envelope(response_text)
            mc = MagicMock()
            mc.read.return_value = _json.dumps(resp).encode()
            cm = MagicMock()
            cm.__enter__.return_value = mc
            return cm

        mc = MagicMock()
        mc.read.return_value = json.dumps({}).encode()
        cm = MagicMock()
        cm.__enter__.return_value = mc
        return cm

    return side_effect


class TestDraftVerifyGraph:
    def test_passes_first_attempt(self):
        """If verification passes on first try, no improvement needed."""
        side_effect = _make_side_effect()

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = LlamaClient("http://mock:8080")
            graph = DraftVerifyGraph(
                draft_node=DraftNode(client=client),
                verify_node=VerifyNode(client=client),
                improve_node=ImproveNode(client=client),
                max_iterations=5,
            )
            result = graph.run("Write a short poem")
            assert result.passed is True
            assert result.num_iterations >= 1  # at least one iteration
            assert result.total_calls >= 2  # draft + verify

    def test_improves_on_failure(self):
        """If verification fails, the graph should improve and retry."""
        # Make first verify fail, second pass
        call_count = [0]

        def failing_then_passing(req, *args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            body = req.data
            import json as _json
            payload = _json.loads(body)
            sys_content = ""
            for m in payload.get("messages", []):
                if m.get("role") == "system":
                    sys_content = m.get("content", "")

            if "quality reviewer" in sys_content:
                if idx == 0 or idx == 2:  # draft or improve message
                    response_text = "Some draft text"
                elif idx == 1:  # first verify → fail
                    response_text = _json.dumps(
                        {"score": 3, "issues": ["Too short", "Inaccurate"], "verdict": "fail"}
                    )
                else:  # second verify → pass
                    response_text = _json.dumps(
                        {"score": 8, "issues": [], "verdict": "pass"}
                    )
                resp = _chat_envelope(response_text)
            elif "editor" in sys_content:
                resp = {"choices": [{"message": {"content": "Improved draft here."}}]}
            else:
                resp = {"choices": [{"message": {"content": "Initial draft."}}]}

            mc = MagicMock()
            mc.read.return_value = _json.dumps(resp).encode()
            cm = MagicMock()
            cm.__enter__.return_value = mc
            return cm

        with patch("urllib.request.urlopen", side_effect=failing_then_passing):
            client = LlamaClient("http://mock:8080")
            graph = DraftVerifyGraph(
                draft_node=DraftNode(client=client),
                verify_node=VerifyNode(client=client),
                improve_node=ImproveNode(client=client),
                max_iterations=5,
            )
            result = graph.run("Write a story")
            assert result.passed is True
            assert result.num_iterations == 2  # first fail, second pass
            assert result.total_calls == 4  # draft + verify(fail) + improve + verify(pass)

    def test_max_iterations(self):
        """Stops when max_iterations reached even without passing."""
        call_count = [0]

        def always_fail(req, *args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            body = req.data
            import json as _json
            payload = _json.loads(body)
            sys_content = ""
            for m in payload.get("messages", []):
                if m.get("role") == "system":
                    sys_content = m.get("content", "")

            if "quality reviewer" in sys_content:
                response_text = _json.dumps(
                    {"score": 2, "issues": ["Always wrong"], "verdict": "fail"}
                )
            elif "editor" in sys_content:
                response_text = "Still wrong."
            else:
                response_text = "Initial draft."

            resp = _chat_envelope(response_text)
            mc = MagicMock()
            mc.read.return_value = _json.dumps(resp).encode()
            cm = MagicMock()
            cm.__enter__.return_value = mc
            return cm

        with patch("urllib.request.urlopen", side_effect=always_fail):
            client = LlamaClient("http://mock:8080")
            graph = DraftVerifyGraph(
                draft_node=DraftNode(client=client),
                verify_node=VerifyNode(client=client),
                improve_node=ImproveNode(client=client),
                max_iterations=3,
            )
            result = graph.run("Task")
            assert result.passed is False
            assert result.num_iterations == 3
