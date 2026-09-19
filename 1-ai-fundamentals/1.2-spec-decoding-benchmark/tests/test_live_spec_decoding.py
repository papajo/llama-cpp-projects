"""Live integration tests for the spec-decoding harness.

`BenchmarkRunner` normally spawns its own llama-server with a draft
model. These tests deliberately do NOT spawn anything: they point the
harness's HTTP client at the already-running shared chat server and
exercise `send_completion`, which is where the SSE stream is parsed.

The speculative-decoding parts genuinely cannot run here (no draft model
is loaded), so those are skipped with a reason rather than faked.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.runner import BenchmarkRunner  # noqa: E402


@pytest.fixture
def bench(chat_base_url):
    """A harness wired to the shared server, with no subprocess started."""
    b = BenchmarkRunner(model_path="/nonexistent/not-used.gguf")
    b._server_url = chat_base_url
    b._client.close()
    b._client = httpx.Client(base_url=chat_base_url, timeout=120.0)
    yield b
    b._client.close()
    # Guard: nothing may have been spawned.
    assert b._server_proc is None


@pytest.mark.live
def test_send_completion_parses_real_stream(bench):
    """The SSE parser collects tokens, timestamps and final timings."""
    r = bench.send_completion("Count to three:", n_predict=12, seed=42)

    assert r["tokens"], "no tokens collected from the stream"
    assert len(r["timestamps"]) == len(r["tokens"])
    assert r["wall_time_s"] > 0
    assert r["prompt_eval_ms"] > 0

    timing = r["timing"]
    assert {"prompt_n", "prompt_ms", "predicted_n", "predicted_per_second"} <= set(timing)


@pytest.mark.live
def test_full_text_is_the_accumulated_stream(bench):
    """full_text must be the concatenated tokens, not the stop chunk.

    Regression guard: the stop chunk's `content` is empty on a real
    server, so reading it left full_text == "" for every run.
    """
    r = bench.send_completion("The capital of France is", n_predict=12, seed=42)
    assert r["full_text"], "full_text came back empty"
    assert r["full_text"] == "".join(r["tokens"])


@pytest.mark.live
def test_token_count_matches_server_accounting(bench):
    """total_tokens must agree with the server's own predicted_n.

    Regression guard: the empty final chunk used to be counted as a token,
    inflating every count by exactly one.
    """
    r = bench.send_completion("List three colours:", n_predict=12, seed=42)
    assert r["total_tokens"] == r["timing"]["predicted_n"]
    assert r["total_tokens"] == len(r["tokens"])


@pytest.mark.live
def test_draft_acceptance_telemetry_absent_without_draft_model(bench):
    """No draft model is loaded, so acceptance counters stay at zero.

    This is the honest result for this deployment, not a passing
    measurement of speculative decoding.
    """
    r = bench.send_completion("Say hello", n_predict=8, seed=42)
    assert r["draft_accepted_count"] == 0
    assert r["draft_total_count"] == 0


@pytest.mark.live
def test_server_reports_no_speculative_slot(bench):
    """/slots confirms speculative decoding is inactive on this server."""
    slots = bench._client.get("/slots").json()
    assert slots
    assert all(s["speculative"] is False for s in slots)


@pytest.mark.live
@pytest.mark.skip(
    reason="unsupported: no draft model is loaded on the shared server, and "
    "spawning a second llama-server with --model-draft is out of scope for "
    "the test suite (CPU-only build, and the harness would need two GGUFs)."
)
def test_speculative_strategy_end_to_end():
    """Would require start_server(--model-draft ...); not run here."""
