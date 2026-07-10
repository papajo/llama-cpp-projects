"""Tests for runner.py — run_ablation."""

import json
import httpx

from sampler_ablation.config import preset_greedy, preset_creative
from sampler_ablation.runner import run_ablation


class TestRunAblation:
    def test_single_config(self, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url="http://127.0.0.1:8080/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            status_code=200,
        )

        results = run_ablation(
            configs=[preset_greedy()],
            prompt="Say hi",
            max_retries=0,
        )
        assert len(results) == 1
        assert results[0].is_ok
        assert results[0].output == "hello world"
        assert results[0].completion_tokens == 2

    def test_multiple_configs(self, httpx_mock):
        n = 2
        for _ in range(n):
            httpx_mock.add_response(
                method="POST",
                url="http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
                status_code=200,
            )

        results = run_ablation(
            configs=[preset_greedy(), preset_creative()],
            prompt="test",
            max_retries=0,
        )
        assert len(results) == 2
        assert all(r.is_ok for r in results)
        assert results[0].config.label == "greedy"
        assert results[1].config.label == "creative"

    def test_http_error(self, httpx_mock):
        httpx_mock.add_response(
            method="POST",
            url="http://127.0.0.1:8080/v1/chat/completions",
            status_code=400,
        )

        results = run_ablation(
            configs=[preset_greedy()],
            prompt="hi",
            max_retries=0,
        )
        assert len(results) == 1
        assert not results[0].is_ok
        assert "HTTP 400" in results[0].error

    def test_server_error_then_recovers(self, httpx_mock):
        # First attempt fails, retry succeeds
        httpx_mock.add_response(
            method="POST",
            url="http://127.0.0.1:8080/v1/chat/completions",
            status_code=500,
        )
        httpx_mock.add_response(
            method="POST",
            url="http://127.0.0.1:8080/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "recovered"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            status_code=200,
        )

        results = run_ablation(
            configs=[preset_greedy()],
            prompt="hi",
            max_retries=1,
        )
        assert len(results) == 1
        assert results[0].is_ok
        assert results[0].output == "recovered"

    def test_timeout_retries_then_fails(self, httpx_mock):
        for _ in range(2):  # initial + 1 retry
            httpx_mock.add_exception(
                httpx.TimeoutException("timeout", request=None),
                method="POST",
                url="http://127.0.0.1:8080/v1/chat/completions",
            )

        results = run_ablation(
            configs=[preset_greedy()],
            prompt="hi",
            max_retries=1,
        )
        assert len(results) == 1
        assert not results[0].is_ok
        assert "timeout" in results[0].error.lower()

    def test_system_prompt_included(self, httpx_mock):
        requests = []

        def store(req):
            requests.append(json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )

        httpx_mock.add_callback(store, method="POST")

        run_ablation(
            configs=[preset_greedy()],
            prompt="hello",
            system_prompt="Be concise.",
            max_retries=0,
        )
        assert len(requests) == 1
        msgs = requests[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be concise."

    def test_request_body_has_sampler_params(self, httpx_mock):
        requests = []

        def store(req):
            requests.append(json.loads(req.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            )

        httpx_mock.add_callback(store, method="POST")

        run_ablation(
            configs=[preset_greedy()],
            prompt="hello",
            max_retries=0,
        )
        body = requests[0]
        # greedy sets temperature=0.0 and top_k=1
        assert body.get("temperature") == 0.0
        assert body.get("top_k") == 1

    def test_progress_callback(self, httpx_mock):
        for _ in range(2):
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

        def cb(done, total, label):
            calls.append((done, total, label))

        run_ablation(
            configs=[preset_greedy(), preset_creative()],
            prompt="hi",
            progress_cb=cb,
            max_retries=0,
        )
        assert len(calls) >= 2  # at least one per config + "Done!"
