"""Live integration tests for map-reduce.

`MapReduceAgent` splits input, maps chunks concurrently, and reduces. The
offline tests map synthetic functions; these map REAL llama-server calls — the
embed server for a numeric map-reduce (768-dim vectors are deterministic and
cheap) and the chat server for a text one.

Fan-out is capped at 3 to match `/props` `total_slots: 3`. The embed server is
used for the wider fan-outs because embeddings are far cheaper on CPU than
generation.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import pytest

from map_reduce.core import (
    MapReduceAgent,
    reduce_concat,
    reduce_list,
    reduce_sum,
    split_by_delimiter,
    split_lines,
)

SLOTS = 3


# ---------------------------------------------------------------------------
# Embeddings map-reduce — deterministic, so real values can be asserted
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_map_embeddings_and_sum_dims(live_embed):
    """Map each line to its embedding width, reduce by sum."""

    def embed_dims(data, index):
        resp = live_embed(data)
        return len(resp["data"][0]["embedding"])

    agent = MapReduceAgent(
        chunk_fn=embed_dims,
        reduce_fn=reduce_sum,
        split_fn=split_lines,
        max_workers=SLOTS,
    )

    result = agent.run("red\ngreen\nblue")

    assert result.error is None
    assert result.num_chunks == 3
    assert result.all_succeeded
    assert result.num_succeeded == 3
    assert result.num_failed == 0
    assert result.map_success_rate == 1.0
    assert result.reduced_output == 768 * 3

    # Results are re-sorted into chunk order after concurrent completion.
    assert [m.chunk_index for m in result.map_results] == [0, 1, 2]
    assert [m.input_data for m in result.map_results] == ["red", "green", "blue"]


@pytest.mark.live
def test_map_embeddings_to_list_preserves_order(live_embed):
    """reduce_list keeps chunk order despite out-of-order completion."""

    def first_component(data, index):
        resp = live_embed(data)
        return (index, resp["data"][0]["index"])

    agent = MapReduceAgent(
        chunk_fn=first_component,
        reduce_fn=reduce_list,
        split_fn=split_lines,
        max_workers=SLOTS,
    )

    result = agent.run("red\ngreen\nblue")

    assert result.all_succeeded
    # Each single-input embed request reports data[0].index == 0.
    assert result.reduced_output == [(0, 0), (1, 0), (2, 0)]


@pytest.mark.live
def test_embeddings_are_deterministic_across_chunks(live_embed):
    """The same text embeds identically, so map-reduce is reproducible."""

    def embed_first(data, index):
        resp = live_embed(data)
        return resp["data"][0]["embedding"][0]

    agent = MapReduceAgent(
        chunk_fn=embed_first,
        reduce_fn=reduce_list,
        split_fn=split_lines,
        max_workers=SLOTS,
    )

    result = agent.run("red\nred\nred")

    assert result.all_succeeded
    values = result.reduced_output
    assert len(values) == 3
    assert values[0] == values[1] == values[2], "identical input must embed identically"


# ---------------------------------------------------------------------------
# Chat map-reduce — structure only
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_map_chat_over_chunks_and_concat(live_chat):
    """Two concurrent real generations, concatenated."""

    def summarise(data, index):
        resp = live_chat(
            [{"role": "user", "content": f"Reply with one word about: {data}"}],
            max_tokens=12,
        )
        return resp["choices"][0]["message"]["content"]

    agent = MapReduceAgent(
        chunk_fn=summarise,
        reduce_fn=reduce_concat,
        split_fn=lambda t: split_by_delimiter(t, "|"),
        max_workers=2,
    )

    result = agent.run("the sea|the sky")

    assert result.error is None
    assert result.num_chunks == 2
    assert result.all_succeeded
    assert isinstance(result.reduced_output, str)
    assert result.reduced_output.strip(), "concatenated output is empty"
    # Structure only: never assert what a 360M model says about the sea.
    for m in result.map_results:
        assert isinstance(m.output, str)


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_partial_failure_is_accounted(live_embed):
    """One chunk hits a closed port; the rest still succeed."""
    import urllib.request

    def flaky(data, index):
        if data == "green":
            urllib.request.urlopen("http://127.0.0.1:1/v1/embeddings", timeout=5)
        resp = live_embed(data)
        return len(resp["data"][0]["embedding"])

    agent = MapReduceAgent(
        chunk_fn=flaky,
        reduce_fn=reduce_list,
        split_fn=split_lines,
        max_workers=SLOTS,
    )

    result = agent.run("red\ngreen\nblue")

    assert result.num_chunks == 3
    assert result.num_succeeded == 2
    assert result.num_failed == 1
    assert result.all_succeeded is False
    assert result.map_success_rate == pytest.approx(2 / 3)

    failed = [m for m in result.map_results if not m.succeeded]
    assert [m.input_data for m in failed] == ["green"]
    assert failed[0].error is not None
