"""Live tests for 4.3 against the real chat server.

This project is a pure simulation -- it opens no sockets, so there is no
mocked-HTTP path to mirror. What it *does* have is a model of how llama.cpp's
KV prompt cache behaves, and that model is checkable: the real server reports
`usage.prompt_tokens_details.cached_tokens` and `timings.cache_n`, and
`/tokenize` gives real token counts to check the chars//4 estimator against.

So these tests validate the simulator's assumptions against the real thing:
  * the server really does reuse a shared prompt prefix across different
    queries (the behaviour simulate_cache claims to model);
  * more shared prefix means more cached tokens (the direction the experiment
    ranks strategies by);
  * chars//4 is a biased estimator, and by how much.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

Prompts are deliberately tiny and max_tokens is 4: this measures prompt
processing, not generation quality. Nothing here asserts anything about what
the model says.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from prompt_cache_chunking.cache_sim import simulate_cache
from prompt_cache_chunking.chunkers import (
    FixedSizeChunker,
    ParagraphChunker,
)
from prompt_cache_chunking.data import default_corpus
from prompt_cache_chunking.experiment import run_experiment

SYSTEM_PROMPT = "You are a helpful assistant. Answer based on the provided context."


@pytest.fixture
def tokenize(chat_base_url):
    """Real token counts from the chat server's /tokenize endpoint."""

    def _call(text: str) -> list:
        req = urllib.request.Request(
            f"{chat_base_url}/tokenize",
            data=json.dumps({"content": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())["tokens"]

    return _call


def _chunk_block(chunker, documents) -> str:
    """Render chunks the way simulate_cache models the prompt."""
    return "".join(f"<chunk>{c.text}</chunk>" for c in chunker.chunk(documents))


# ---------------------------------------------------------------------------
# Does the real server actually reuse a shared prefix?
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_server_reports_cache_telemetry(live_chat):
    """cached_tokens and cache_n are present and mutually consistent."""
    r = live_chat([{"role": "user", "content": "Say hi"}], max_tokens=4)
    cached = r["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert isinstance(cached, int)
    assert 0 <= cached <= r["usage"]["prompt_tokens"]
    # llama.cpp reports the same number twice, under two different keys.
    assert r["timings"]["cache_n"] == cached


@pytest.mark.live
def test_shared_prefix_is_reused_across_different_queries(live_chat):
    """The core assumption of simulate_cache, measured.

    Three prompts share a system prompt and an identical chunk block and differ
    only in the trailing question. The first is a cold miss; the rest must be
    served largely from cache. This is what makes the project's premise -- that
    chunking strategy changes cache reuse -- real rather than theoretical.
    """
    block = "<chunk>" + ("The Roman Empire was a large ancient empire. " * 8) + "</chunk>"
    cached = []
    for question in ["What is this about?", "Name one fact.", "Summarise briefly."]:
        r = live_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{block}\n\nQuestion: {question}"},
            ],
            max_tokens=4,
        )
        cached.append(
            (
                r["usage"]["prompt_tokens_details"]["cached_tokens"],
                r["usage"]["prompt_tokens"],
            )
        )

    # Every request stays within its own prompt length.
    for c, total in cached:
        assert 0 <= c <= total

    # Once the prefix is warm, most of the prompt is reused. The exact number
    # depends on slot scheduling, so this asserts "most", not an exact count.
    warm = cached[1:]
    for c, total in warm:
        assert c > total // 2, f"expected majority cached, got {c}/{total}"


@pytest.mark.live
def test_more_shared_prefix_means_more_cached_tokens(live_chat):
    """The direction the experiment ranks strategies by, verified.

    A prompt that repeats a long warm prefix must reuse more than one that
    diverges early. Only the ordering is asserted, never absolute counts.
    """
    long_block = "<chunk>" + ("Shared context sentence. " * 20) + "</chunk>"

    # Warm the long prefix.
    live_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{long_block}\n\nQuestion: warm up"},
        ],
        max_tokens=4,
    )
    # Same long prefix, different tail -> should reuse nearly all of it.
    same = live_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{long_block}\n\nQuestion: different tail"},
        ],
        max_tokens=4,
    )
    # Diverges right after the system prompt -> little to reuse.
    different = live_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "<chunk>Totally unrelated opening text.</chunk>"
                "\n\nQuestion: different tail",
            },
        ],
        max_tokens=4,
    )

    same_cached = same["usage"]["prompt_tokens_details"]["cached_tokens"]
    diff_cached = different["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert same_cached > diff_cached, (
        f"repeating a long prefix cached {same_cached} tokens but diverging "
        f"early cached {diff_cached}; prefix reuse is not behaving as modelled"
    )


# ---------------------------------------------------------------------------
# Is the chars//4 token estimator honest?
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_chars_over_four_overestimates_real_tokens(tokenize):
    """simulate_cache counts tokens as len(text)//4. Reality disagrees.

    Measured with the SmolLM2 vocab, chars//4 runs ~20-26% high. Recorded in
    drift-rag.md: cross-strategy *ratios* stay usable because every strategy
    shares the estimator, but absolute token counts in a report are not real
    token counts.
    """
    for text in [
        SYSTEM_PROMPT,
        "<chunk>" + ("The Roman Empire was a large ancient empire. " * 8) + "</chunk>",
    ]:
        estimated = len(text) // 4
        actual = len(tokenize(text))
        assert actual > 0
        assert estimated > actual, (
            f"chars//4 estimated {estimated} but real tokenizer gave {actual}"
        )
        # Biased, but within a bounded factor -- not wildly wrong.
        assert estimated < actual * 1.6


@pytest.mark.live
def test_estimator_is_monotonic_in_real_tokens(tokenize):
    """A longer chunk block must estimate higher AND tokenize higher."""
    corpus = default_corpus()
    short = _chunk_block(ParagraphChunker(), corpus.documents[:1])
    long = _chunk_block(ParagraphChunker(), corpus.documents[:4])

    assert len(short) // 4 < len(long) // 4
    assert len(tokenize(short)) < len(tokenize(long))


# ---------------------------------------------------------------------------
# The simulator's own output, on real corpus text
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_simulated_hit_ratio_is_non_degenerate(tokenize):
    """The simulation must report real reuse, not a constant 0.0.

    Guards the bug this live pass uncovered: the chunk-level cache branch was
    unreachable, so cache_hit_ratio was structurally 0.0 for every strategy.
    """
    corpus = default_corpus()
    result = simulate_cache(
        chunker=ParagraphChunker(),
        documents=corpus.documents,
        queries=corpus.queries,
        system_prompt=SYSTEM_PROMPT,
    )
    assert result.cached_tokens > 0
    assert 0.0 < result.cache_hit_ratio < 1.0
    assert result.total_tokens > result.cached_tokens

    # Every prompt's own accounting must be internally consistent.
    for d in result.details:
        assert d["cached_tokens"] <= d["prompt_tokens"]

    # And the simulated prompts must be small enough to actually run on this
    # server: n_ctx is 2048, and chars//4 over-estimates, so the real prompt is
    # smaller still.
    worst = max(d["prompt_tokens"] for d in result.details)
    assert worst < 2048, f"simulated prompt of {worst} tokens exceeds n_ctx"


@pytest.mark.live
def test_experiment_ranks_all_strategies(tokenize):
    """Every default strategy produces a usable, non-degenerate row."""
    result = run_experiment(default_corpus(), system_prompt=SYSTEM_PROMPT)
    summary = result.summary()
    assert len(summary) >= 4
    for name, metrics in summary.items():
        assert metrics["chunk_count"] > 0, name
        assert metrics["total_tokens"] > 0, name
        assert 0.0 <= metrics["cache_hit_ratio"] <= 1.0, name


@pytest.mark.live
def test_simulated_prompt_actually_fits_and_runs(live_chat):
    """Feed a real simulated prompt to the real server; it must be accepted.

    n_ctx here is 2048, so a chunking strategy that builds an over-long prompt
    would fail in production while the offline simulation stayed happy.
    """
    corpus = default_corpus()
    block = _chunk_block(FixedSizeChunker(chunk_size=200, overlap=20),
                         corpus.documents[:2])
    r = live_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{block}\n\nQuestion: {corpus.queries[0]}"},
        ],
        max_tokens=4,
    )
    assert r["object"] == "chat.completion"
    assert r["usage"]["prompt_tokens"] < 2048
    assert r["choices"][0]["message"]["role"] == "assistant"
