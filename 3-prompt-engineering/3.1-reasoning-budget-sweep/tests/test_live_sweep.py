"""Live integration tests for the reasoning budget sweep.

The offline suite drives run_sweep through pytest-httpx with canned
OpenAI-shaped bodies. These run the same code path against the real
chat server.

Sweeps are kept tiny on purpose: SmolLM2-360M on CPU is slow, so grids
are 1-2 configs with max_tokens in the 8-32 range. Assertions are
structural - output *shape*, token accounting, error handling - never
output quality.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import pytest

from budget_sweep.budget import SweepConfig
from budget_sweep.metrics import compute_metrics
from budget_sweep.sweep import run_sweep


@pytest.mark.live
def test_single_run_against_real_server(chat_base_url):
    """One config, one prompt, one run - the minimal real sweep."""
    cfg = SweepConfig(
        base_url=chat_base_url,
        param_grid={"max_tokens": [16]},
        prompts=["Say hi."],
        n_runs=1,
    )
    results = run_sweep(cfg, max_retries=0)

    assert len(results) == 1
    r = results[0]
    assert r.is_ok, r.error
    assert r.output, "real server returned empty content"
    assert r.elapsed_s > 0
    assert r.prompt_tokens and r.prompt_tokens > 0
    assert r.completion_tokens and r.completion_tokens > 0


@pytest.mark.live
def test_max_tokens_actually_caps_the_completion(chat_base_url):
    """The budget knob the whole project exists to sweep must bind."""
    cfg = SweepConfig(
        base_url=chat_base_url,
        param_grid={"max_tokens": [8]},
        prompts=["Write a long essay about the sea."],
        n_runs=1,
    )
    r = run_sweep(cfg, max_retries=0)[0]
    assert r.is_ok, r.error
    assert r.completion_tokens <= 8


@pytest.mark.live
def test_grid_expansion_produces_one_result_per_cell(chat_base_url):
    """configs x prompts x runs, against the real server."""
    cfg = SweepConfig(
        base_url=chat_base_url,
        param_grid={"max_tokens": [8, 16]},
        prompts=["Say hi."],
        n_runs=1,
    )
    results = run_sweep(cfg, max_retries=0)
    assert len(results) == cfg.n_total_runs == 2
    assert all(r.is_ok for r in results), [r.error for r in results]
    # Each result carries back the params that produced it.
    assert {r.params["max_tokens"] for r in results} == {8, 16}


@pytest.mark.live
def test_system_prompt_reaches_the_real_model(chat_base_url):
    """A system turn is accepted - SmolLM2's template supports it."""
    cfg = SweepConfig(
        base_url=chat_base_url,
        param_grid={"max_tokens": [16]},
        prompts=["Hello."],
        n_runs=1,
        system_prompt="You are a terse assistant.",
    )
    r = run_sweep(cfg, max_retries=0)[0]
    assert r.is_ok, r.error
    assert r.output


@pytest.mark.live
def test_metrics_compute_over_real_results(chat_base_url):
    """compute_metrics produces finite figures from real outputs."""
    cfg = SweepConfig(
        base_url=chat_base_url,
        param_grid={"max_tokens": [16]},
        prompts=["Name a colour."],
        n_runs=2,
    )
    results = run_sweep(cfg, max_retries=0)
    summaries = compute_metrics(results)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.total_runs == 2
    assert s.errors == 0
    assert s.avg_completion_tokens > 0
    assert s.avg_speed_tok_s > 0
    assert 0.0 <= s.avg_diversity <= 1.0
    assert 0.0 <= s.avg_repetition_rate <= 1.0
    assert s.avg_entropy >= 0.0
    assert s.token_efficiency >= 0.0


@pytest.mark.live
def test_real_400_is_captured_as_a_result_not_an_exception(chat_base_url):
    """A malformed request must land in SweepResult.error.

    Real error body is {"error":{"code":400,"message":...,"type":...}},
    which is a different shape from the flat {"error": "bad request"} the
    offline test feeds in - but run_sweep only formats response.text, so
    both work.
    """
    cfg = SweepConfig(
        base_url=chat_base_url,
        # temperature must be a number; a string is rejected by the server.
        param_grid={"temperature": ["hot"], "max_tokens": [8]},
        prompts=["Say hi."],
        n_runs=1,
    )
    r = run_sweep(cfg, max_retries=0)[0]
    assert not r.is_ok
    assert "HTTP 400" in r.error
    assert "invalid_request_error" in r.error


@pytest.mark.live
def test_context_overflow_is_reported_as_a_400(chat_base_url, live_server_meta):
    """Exceeding n_ctx (2048) yields a typed error, not a hang."""
    n_ctx = live_server_meta["chat"]["data"][0]["meta"]["n_ctx"]
    cfg = SweepConfig(
        base_url=chat_base_url,
        param_grid={"max_tokens": [8]},
        prompts=["word " * (n_ctx * 3)],
        n_runs=1,
    )
    r = run_sweep(cfg, max_retries=0)[0]
    assert not r.is_ok
    assert "HTTP 400" in r.error
    assert "exceed_context_size_error" in r.error
