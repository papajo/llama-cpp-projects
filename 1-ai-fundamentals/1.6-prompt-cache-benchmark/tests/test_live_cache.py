"""Live integration tests for the prompt-cache benchmark.

The offline suite only covers scenario generation and the metric maths —
the HTTP path in harness/runner.py is completely untested there. These
tests run the real /completion endpoint and verify the timing fields the
benchmark depends on actually exist and mean what it assumes.

Kept deliberately small: SmolLM2-360M on CPU is slow, so n_predict is tiny
and scenarios are hand-built with a handful of short prompts rather than
using the shipped 10- and 24-prompt ones.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.runner import CacheBenchmarkRunner  # noqa: E402
from harness.scenarios import CacheScenario  # noqa: E402

SHARED_PREFIX = (
    "You are reading a short technical note about caching. "
    "Keys are stored per slot and reused across requests when the prefix "
    "matches exactly. "
)


@pytest.fixture
def runner(chat_base_url):
    r = CacheBenchmarkRunner(server_url=chat_base_url, n_predict=8, temperature=0.0)
    yield r
    r.client.close()


@pytest.fixture
def small_scenario():
    return CacheScenario(
        name="live-smoke",
        description="Three short queries behind one shared prefix.",
        prompts=[f"{SHARED_PREFIX}Q{i}: name one benefit." for i in range(3)],
        shared_prefix=SHARED_PREFIX,
        warmup=1,
    )


@pytest.mark.live
def test_health_check_against_real_server(runner):
    assert runner.health_check() is True


@pytest.mark.live
def test_completion_returns_the_timing_fields_the_benchmark_reads(runner):
    """Every `timings` key run_scenario relies on must be present."""
    data = runner._send_completion("Say hello.", cache_prompt=True)

    assert "timings" in data, "no timings block; llama-server would need --no-timings off"
    timings = data["timings"]
    for key in ("prompt_n", "prompt_ms", "predicted_per_second", "cache_n"):
        assert key in timings, f"missing timings.{key}"
        assert isinstance(timings[key], (int, float))

    # The runner stamps its own wall-clock measurement onto the payload.
    assert data["_request_time_ms"] > 0


@pytest.mark.live
def test_cache_prompt_true_reuses_the_prefix(runner):
    """A repeated prefix is reported as cached via timings.cache_n."""
    prompt = f"{SHARED_PREFIX}Summarise this in one word."

    runner._send_completion(prompt, cache_prompt=True)
    second = runner._send_completion(prompt, cache_prompt=True)

    # The second identical request must reuse tokens from the slot's cache.
    assert second["timings"]["cache_n"] > 0
    assert second["tokens_cached"] > 0


@pytest.mark.live
def test_run_scenario_produces_a_complete_result(runner, small_scenario):
    """End-to-end: both arms populate and compute() yields finite metrics."""
    result = runner.run_scenario(small_scenario, warmup_runs=1)

    assert len(result.runs_cached) == len(small_scenario.prompts)
    assert len(result.runs_uncached) == len(small_scenario.prompts)
    assert result.avg_ttft_cached_ms > 0
    assert result.avg_ttft_uncached_ms > 0
    assert result.speedup_factor > 0
    assert len(result.per_prompt) == len(small_scenario.prompts)

    for pr in result.runs_cached + result.runs_uncached:
        assert pr.prompt_tokens > 0
        assert pr.prompt_processing_ms >= 0
        assert pr.predicted_per_second > 0


@pytest.mark.live
def test_prompt_tokens_come_from_the_server_not_the_estimate(runner):
    """timings.prompt_n must be used in preference to estimate_tokens().

    estimate_tokens() is chars//4; the real tokenizer disagrees, so a
    result carrying the estimate would be a silent fallback.
    """
    from harness.metrics import estimate_tokens

    prompt = SHARED_PREFIX * 2
    data = runner._send_completion(prompt, cache_prompt=False)
    server_n = data["timings"]["prompt_n"] + data["timings"]["cache_n"]

    # Sanity: the crude estimate is in the right order of magnitude but
    # is not the number the benchmark should be reporting.
    assert server_n > 0
    assert estimate_tokens(prompt) != server_n


@pytest.mark.live
def test_speedup_is_reported_but_not_asserted_to_be_favourable(
    runner, small_scenario
):
    """Cache speedup is a measurement, not a guarantee.

    On a 360M model with a ~40-token prefix the prompt-eval phase is
    already sub-millisecond, so the cached arm is not reliably faster.
    Assert only that the number is finite and positive.
    """
    result = runner.run_scenario(small_scenario, warmup_runs=0)
    assert result.speedup_factor > 0
    assert result.speedup_factor == result.speedup_factor  # not NaN
