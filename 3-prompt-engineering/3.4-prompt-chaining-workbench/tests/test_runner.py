"""Tests for the chain runner."""

import json

import pytest
import httpx
from prompt_chaining.chain import ChainStep, PromptChain
from prompt_chaining.runner import run_chain, StepResult, ChainResult


SIMPLE_CHAIN = PromptChain(
    name="simple",
    steps=[
        ChainStep(name="step1", user_prompt="Echo: {{input}}"),
        ChainStep(name="step2", user_prompt="Again: {{step_1}}"),
    ],
)


def _mock_response(content: str, status: int = 200):
    """Build a mock httpx Response."""
    json_data = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return httpx.Response(status_code=status, json=json_data, request=httpx.Request("POST", "http://test/"))


class TestRunChain:
    def test_successful_chain(self, httpx_mock):
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "Again: hello world"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        })

        result = run_chain(SIMPLE_CHAIN, input_text="hello", base_url="http://test")
        assert result.chain_name == "simple"
        assert result.input_text == "hello"
        assert len(result.steps) == 2
        assert result.steps[0].output == "hello world"
        assert result.steps[1].output == "Again: hello world"
        assert result.all_ok
        assert result.last_output == "Again: hello world"

    def test_context_passing(self, httpx_mock):
        """Step 2 should receive step 1 output in context."""
        def check_body(request):
            body = json.loads(request.content)
            msgs = body["messages"]
            return _mock_response(f"Got: {msgs[-1]['content']}")

        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "first"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        })
        httpx_mock.add_callback(check_body)

        result = run_chain(SIMPLE_CHAIN, input_text="test", base_url="http://test")
        assert result.steps[1].output == "Got: Again: first"

    def test_stop_on_error(self, httpx_mock):
        # Step1: 3 retry attempts (max_retries=2) all 503
        for _ in range(3):
            httpx_mock.add_response(status_code=503)
        # Step2 should NOT be called — stop_on_error=True
        result = run_chain(SIMPLE_CHAIN, input_text="x", base_url="http://test")
        assert len(result.steps) == 1
        assert not result.all_ok
        assert not result.steps[0].is_ok
        assert "503" in result.steps[0].error

    def test_retry_then_succeed(self, httpx_mock):
        # Step1: 502 then 200
        httpx_mock.add_response(status_code=502)
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "retried"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        })
        # Step2: 200
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "second"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        })

        result = run_chain(SIMPLE_CHAIN, input_text="x", base_url="http://test")
        assert len(result.steps) == 2
        assert result.steps[0].output == "retried"
        assert result.steps[1].output == "second"
        assert result.all_ok

    def test_continue_on_error(self, httpx_mock):
        """When stop_on_error=False, chain continues after a step failure."""
        # Step1: 3 retry attempts, all 503
        for _ in range(3):
            httpx_mock.add_response(status_code=503)
        # Step2: 200
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "second step"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        })

        result = run_chain(SIMPLE_CHAIN, input_text="x", base_url="http://test",
                           stop_on_error=False)
        assert len(result.steps) == 2
        assert not result.steps[0].is_ok
        assert result.steps[1].is_ok

    def test_progress_callback(self, httpx_mock):
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "a"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "b"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

        calls = []
        def cb(idx, total, name, status):
            calls.append((idx, total, name, status))

        run_chain(SIMPLE_CHAIN, input_text="x", base_url="http://test", progress_cb=cb)
        assert len(calls) >= 4  # running, ok, running, ok

    def test_last_output_with_mixed_results(self, httpx_mock):
        # Step1: 200
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "first ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        # Step2: 3 retry attempts, all 500
        for _ in range(3):
            httpx_mock.add_response(status_code=500)

        result = run_chain(SIMPLE_CHAIN, input_text="x", base_url="http://test",
                           stop_on_error=False)
        assert result.last_output == "first ok"

    def test_empty_output(self, httpx_mock):
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        })
        # Step2 needs a response too
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        })

        result = run_chain(SIMPLE_CHAIN, input_text="x", base_url="http://test")
        assert result.steps[0].output == ""
        assert result.steps[0].is_ok

    def test_no_choices(self, httpx_mock):
        httpx_mock.add_response(json={
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        })
        httpx_mock.add_response(json={
            "choices": [{"message": {"content": "step2"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        })

        result = run_chain(SIMPLE_CHAIN, input_text="x", base_url="http://test")
        # Should handle gracefully
        assert result.steps[0].output == ""
        assert result.steps[0].is_ok


class TestStepResult:
    def test_is_ok(self):
        ok = StepResult(step_name="a", step_index=1, output="hi")
        assert ok.is_ok
        err = StepResult(step_name="b", step_index=2, error="fail")
        assert not err.is_ok

    def test_rendered_fields(self):
        sr = StepResult(
            step_name="s1", step_index=1, output="out",
            rendered_system="sys", rendered_user="usr",
        )
        assert sr.rendered_system == "sys"
        assert sr.rendered_user == "usr"


class TestChainResult:
    def test_ok_steps(self):
        r = ChainResult(
            chain_name="t", input_text="in",
            steps=[
                StepResult(step_name="a", step_index=1, output="ok"),
                StepResult(step_name="b", step_index=2, error="fail"),
                StepResult(step_name="c", step_index=3, output="ok2"),
            ],
        )
        assert len(r.ok_steps) == 2
        assert len(r.failed_steps) == 1

    def test_all_ok(self):
        r = ChainResult(
            chain_name="t", input_text="in",
            steps=[
                StepResult(step_name="a", step_index=1, output="ok"),
            ],
        )
        assert r.all_ok

    def test_not_all_ok(self):
        r = ChainResult(
            chain_name="t", input_text="in",
            steps=[
                StepResult(step_name="a", step_index=1, error="fail"),
            ],
        )
        assert not r.all_ok

    def test_context_build(self):
        r = ChainResult(
            chain_name="t", input_text="hello",
            steps=[
                StepResult(step_name="first", step_index=1, output="out1"),
                StepResult(step_name="second", step_index=2, error="fail"),
                StepResult(step_name="third", step_index=3, output="out3"),
            ],
        )
        ctx = r.context()
        assert ctx["input"] == "hello"
        assert ctx["step_1"] == "out1"
        assert ctx["first"] == "out1"
        assert "step_2" not in ctx  # failed step excluded
        assert ctx["step_3"] == "out3"
        assert ctx["third"] == "out3"
