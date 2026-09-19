"""Live integration tests for 4.1 against the real embedding server.

Mirrors the code paths covered by the mocked tests in test_embeddings.py and
test_ablation.py, but against a real llama-server /v1/embeddings endpoint.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

Assertions are structural (shape, dtype, keys, invariants) — never semantic.
SmolLM2/nomic output quality is not under test here.
"""

from __future__ import annotations

import numpy as np
import pytest

from embedding_norm_ablation.ablation import (
    STANDARD_TREATMENTS,
    AblationConfig,
    run_ablation,
)
from embedding_norm_ablation.data import mini_corpus
from embedding_norm_ablation.embeddings import (
    EmbeddingClient,
    batch_normalize,
    l2_normalize,
    norm_stats,
)
from embedding_norm_ablation.reporting import report_to_json, report_to_markdown


@pytest.fixture
def client(embed_base_url):
    """An EmbeddingClient pointed at the real embedding server."""
    return EmbeddingClient(server_url=embed_base_url)


# ---------------------------------------------------------------------------
# Raw response shape — what the mocks stand in for
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_raw_response_envelope(live_embed):
    """The real /v1/embeddings envelope, which the canned mocks abbreviate.

    The mocked tests supply only {"data": [{"embedding": [...]}]}. The real
    server also returns per-item `index`/`object` and a top-level
    `model`/`object`/`usage`. See drift-rag.md.
    """
    r = live_embed("hello")
    assert r["object"] == "list"
    assert set(r["usage"]) == {"prompt_tokens", "total_tokens"}
    assert r["usage"]["total_tokens"] > 0
    item = r["data"][0]
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert isinstance(item["embedding"], list)
    # No completion_tokens on an embedding request.
    assert "completion_tokens" not in r["usage"]


@pytest.mark.live
def test_batch_input_returns_indexed_rows(live_embed):
    """A list input yields one indexed row per text in a single call."""
    r = live_embed(["alpha", "beta", "gamma"])
    assert len(r["data"]) == 3
    assert [d["index"] for d in r["data"]] == [0, 1, 2]
    assert all(len(d["embedding"]) == 768 for d in r["data"])


# ---------------------------------------------------------------------------
# EmbeddingClient against the real server
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_embed_returns_768_dim_float32(client):
    vec = client.embed("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (768,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()


@pytest.mark.live
def test_embed_with_explicit_model(client, embed_model):
    vec = client.embed("hello", model=embed_model)
    assert vec.shape == (768,)


@pytest.mark.live
def test_embed_unknown_model_is_served_anyway(client):
    """llama-server ignores an unknown model id and serves the loaded model.

    There is no 404 here — a mocked test asserting an error for a bad model id
    would be asserting behaviour the real server does not have.
    """
    vec = client.embed("hello", model="no-such-model-xyz")
    assert vec.shape == (768,)


@pytest.mark.live
def test_embed_many_preserves_order_and_count(client):
    texts = ["the first text", "an unrelated second text", "a third"]
    vecs = client.embed_many(texts)
    assert len(vecs) == 3
    assert all(v.shape == (768,) for v in vecs)
    # Same text must embed deterministically to the same vector.
    again = client.embed(texts[0])
    assert np.allclose(vecs[0], again, atol=1e-6)


# ---------------------------------------------------------------------------
# The normalisation premise of this project
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_server_returns_l2_normalised_vectors(client):
    """Real /v1/embeddings output is ALREADY unit-norm.

    llama-server normalises embeddings by default (--embd-normalize 2). The
    mocked tests feed non-unit vectors such as [0.1, 0.2, 0.3], so the offline
    suite never sees this. Recorded in drift-rag.md: it makes the
    norm-vs-unnorm ablation degenerate on this server.
    """
    vecs = client.embed_many(["a short doc", "another short doc"])
    for v in vecs:
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.live
def test_l2_normalize_is_idempotent_on_real_vectors(client):
    """Normalising an already-normalised real vector changes nothing."""
    vec = client.embed("hello")
    assert np.allclose(vec, l2_normalize(vec), atol=1e-6)


@pytest.mark.live
def test_norm_stats_on_real_vectors(client):
    stats = norm_stats(client.embed_many(["one", "two", "three"]))
    assert set(stats) == {"mean", "std", "min", "max"}
    assert stats["mean"] == pytest.approx(1.0, abs=1e-5)
    assert stats["std"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.live
def test_batch_normalize_preserves_direction(client):
    vecs = client.embed_many(["alpha", "beta"])
    normed = batch_normalize(vecs)
    assert len(normed) == 2
    for raw, unit in zip(vecs, normed):
        # Already unit-norm, so direction is unchanged.
        assert float(np.dot(raw, unit)) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Full ablation run end to end
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_run_ablation_end_to_end(client, embed_model):
    """The whole experiment path on real embeddings (mini corpus: 5 docs, 2 q)."""
    config = AblationConfig(
        corpus=mini_corpus(), model=embed_model, top_k=3, label="live"
    )
    result = run_ablation(client, config)

    assert result.embedding_dim == 768
    assert result.corpus_size == 5
    assert result.num_queries == 2
    assert len(result.treatments) == len(STANDARD_TREATMENTS)

    for t in result.treatments:
        assert len(t.per_query_metrics) == 2
        for pq in t.per_query_metrics:
            assert set(pq) == {"p@1", "p@5", "r@5", "ap"}
            assert all(0.0 <= v <= 1.0 for v in pq.values())
        for metric in ("mean_p@1", "mean_ap", "std_ap"):
            assert metric in t.aggregated


@pytest.mark.live
def test_normalisation_treatments_coincide_on_real_vectors(client, embed_model):
    """norm-* and unnorm-* agree because the server pre-normalises.

    This is a property of the real server, not of the project code. Asserted
    so the day someone starts llama-server with --embd-normalize -1, this
    test tells them the ablation has become meaningful again.
    """
    config = AblationConfig(
        corpus=mini_corpus(), model=embed_model, top_k=3, label="live"
    )
    result = run_ablation(client, config)

    for base in ("cosine", "dot"):
        unnorm = result.get_treatment(f"unnorm-{base}").aggregated
        norm = result.get_treatment(f"norm-{base}").aggregated
        assert unnorm == pytest.approx(norm, abs=1e-6)


@pytest.mark.live
def test_reports_render_from_a_real_run(client, embed_model):
    """Markdown and JSON reporting over a real result object."""
    import json

    config = AblationConfig(
        corpus=mini_corpus(), model=embed_model, top_k=3, label="live"
    )
    result = run_ablation(client, config)

    md = report_to_markdown(result)
    assert "# Embedding Norm Ablation Report" in md
    assert "**Embedding dimension:** 768" in md

    payload = json.loads(report_to_json(result))
    assert payload["embedding_dim"] == 768
    assert len(payload["treatments"]) == len(STANDARD_TREATMENTS)
