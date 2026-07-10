"""Tests for sweep.py — run_sweep and helpers."""

import json

import httpx
import pytest

from budget_sweep.budget import SweepConfig
from budget_sweep.sweep import run_sweep


def _make_response(content="Hello world", prompt_tokens=10, completion_tokens=5):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


class TestRunSweep:
    def test_successful_run(self, httpx_mock):
        """All calls succeed, results are collected."""
        n_expected = 2  # 2 configs × 1 prompt × 1 run
        for _ in range(n_expected):
            httpx_mock.add_response(
                method="POST",
                url="http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "choices": [{"message": {"content": "Hello world"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                status_code=200,
            )

        cfg = SweepConfig(
            base_url="http://127.0.0.1:8080",
            param_grid={"max_tokens": [64, 128]},
            prompts=["Say hi"],
            n_runs=1,
        )
        results = run_sweep(cfg, max_retries=0)
        assert len(results) == n_expected
        assert all(r.is_ok for r in results)
        assert results[0].output == "Hello world"
        assert results[0].completion_tokens == 5
        assert results[0].prompt_tokens == 10
        assert results[0].elapsed_s > 0

    def test_with_multiple_prompts_and_runs(self, httpx_mock):
        """Correct total count with multiplier dimensions."""
        n_configs = 3   # max_tokens: [64, 128, 256]
        n_prompts = 2
        n_runs = 3
        total = n_configs * n_prompts * n_runs  # 18

        for _ in range(total):
            httpx_mock.add_response(
                method="POST",
                url="http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
                status_code=200,
            )

        cfg = SweepConfig(
            base_url="http://127.0.0.1:8080",
            param_grid={"max_tokens": [64, 128, 256]},
            prompts=["p1", "p2"],
            n_runs=3,
        )
        results = run_sweep(cfg, max_retries=0)
        assert len(results) == total

    def test_http_error_captured(self, httpx_mock):
        """Non-500 HTTP errors produce SweepResult with error field."""
        httpx_mock.add_response(
            method="POST",
            url="http://127.0.0.1:8080/v1/chat/completions",
            json={"error": "bad request"},
            status_code=400,
        )

        cfg = SweepConfig(
            param_grid={"max_tokens": [64]},
            prompts=["hi"],
            n_runs=1,
        )
        results = run_sweep(cfg, max_retries=0)
        assert len(results) == 1
        assert not results[0].is_ok
        assert "HTTP 400" in results[0].error

    def test_server_error_then_recovers(self, httpx_mock):
        """Server 500 on first attempt, succeeds on retry."""
        # First attempt: 500
        httpx_mock.add_response(
            method="POST",
            url="http://127.0.0.1:8080/v1/chat/completions",
            status_code=500,
        )
        # Second attempt (retry 1): 200
        httpx_mock.add_response(
            method="POST",
            url="http://127.0.0.1:8080/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
            status_code=200,
        )

        cfg = SweepConfig(
            param_grid={"max_tokens": [64]},
            prompts=["hi"],
            n_runs=1,
        )
        results = run_sweep(cfg, max_retries=1)
        assert len(results) == 1
        assert results[0].is_ok
        assert results[0].output == "ok"

    def test_server_error_all_fail(self, httpx_mock):
        """All attempts return 500, eventually fails with error."""
        max_retries = 2
        n_calls = max_retries + 1  # initial + retries
        for _ in range(n_calls):
            httpx_mock.add_response(
                method="POST",
                url="http://127.0.0.1:8080/v1/chat/completions",
                status_code=500,
            )

        cfg = SweepConfig(
            param_grid={"max_tokens": [64]},
            prompts=["hi"],
            n_runs=1,
        )
        results = run_sweep(cfg, max_retries=max_retries)
        assert len(results) == 1
        assert not results[0].is_ok
        assert "500" in results[0].error

    def test_timeout_retries_then_fails(self, httpx_mock):
        """All attempts time out, eventually returns error."""
        max_retries = 1
        n_calls = max_retries + 1
        for _ in range(n_calls):
            httpx_mock.add_exception(
                httpx.TimeoutException("timeout", request=None),
                method="POST",
                url="http://127.0.0.1:8080/v1/chat/completions",
            )

        cfg = SweepConfig(
            param_grid={"max_tokens": [64]},
            prompts=["hi"],
            n_runs=1,
        )
        results = run_sweep(cfg, max_retries=max_retries)
        assert len(results) == 1
        assert not results[0].is_ok
        assert "timeout" in results[0].error.lower()

    def test_progress_callback(self, httpx_mock):
        """Progress callback receives (current, total, label)."""
        n_expected = 2
        for _ in range(n_expected):
            httpx_mock.add_response(
                method="POST",
                url="http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
                status_code=200,
            )

        calls = []

        def cb(current, total, label):
            calls.append((current, total))

        cfg = SweepConfig(
            param_grid={"max_tokens": [64]},
            prompts=["hi"],
            n_runs=2,
        )
        run_sweep(cfg, progress_cb=cb, max_retries=0)
        # Should have been called for each step + final done
        assert len(calls) >= 2

    def test_system_prompt_included(self, httpx_mock):
        """System prompt is sent in the request body."""
        request_history = []

        def store_request(request):
            request_history.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )

        httpx_mock.add_callback(store_request, method="POST")

        cfg = SweepConfig(
            param_grid={"max_tokens": [64]},
            prompts=["hello"],
            n_runs=1,
            system_prompt="You are a helpful assistant.",
        )
        run_sweep(cfg, max_retries=0)
        assert len(request_history) == 1
        body = request_history[0]
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are a helpful assistant."
        assert body["messages"][1]["role"] == "user"

    def test_no_system_prompt(self, httpx_mock):
        """Without system prompt, only user message is sent."""
        request_history = []

        def store_request(request):
            request_history.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            )

        httpx_mock.add_callback(store_request, method="POST")

        cfg = SweepConfig(
            param_grid={"max_tokens": [64]},
            prompts=["hello"],
            n_runs=1,
        )
        run_sweep(cfg, max_retries=0)
        body = request_history[0]
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
