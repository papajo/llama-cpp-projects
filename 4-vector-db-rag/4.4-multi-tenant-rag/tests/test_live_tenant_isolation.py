"""Live integration tests for 4.4 against the real embedding server.

Mirrors the mocked paths in test_tenant_store.py, test_pipeline.py and
test_reporting.py, but with real 768-dim nomic embeddings instead of canned
vectors.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

The property under test is tenant isolation, which is a *structural* guarantee
(documents are partitioned by dict key and indices are tenant-relative), so it
must hold no matter what the embeddings actually are. These tests confirm it
still holds once the vectors are real, high-dimensional and near-collinear --
the case the canned orthogonal test vectors never exercise.

Retrieval *quality* is never asserted. Only this project's embedding server is
used; no chat model is involved, so the whole file is fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from multi_tenant_rag.data import default_corpus
from multi_tenant_rag.pipeline import (
    evaluate_cross_tenant_leakage,
    evaluate_isolation,
)
from multi_tenant_rag.reporting import evaluation_to_json, evaluation_to_markdown
from multi_tenant_rag.retriever import EmbeddingClient, TenantRetriever
from multi_tenant_rag.tenant_store import TenantVectorStore


@pytest.fixture(scope="module")
def client(embed_base_url):
    return EmbeddingClient(server_url=embed_base_url)


@pytest.fixture(scope="module")
def corpus():
    return default_corpus()


@pytest.fixture(scope="module")
def live_store(client, corpus, embed_model):
    """The full 3-tenant corpus indexed with real embeddings (24 documents).

    Module-scoped on purpose: indexing costs 24 real embed calls, and nothing
    in this file mutates the store, so re-embedding per test would only make
    the suite slower.
    """
    store = TenantVectorStore()
    for tenant in corpus.tenants:
        store.add_many(
            tenant.id,
            tenant.documents,
            client.embed_many(tenant.documents, model=embed_model),
        )
    return store


@pytest.fixture(scope="module")
def retriever(live_store, client, embed_model):
    return TenantRetriever(
        store=live_store, client=client, embed_model=embed_model
    )


# ---------------------------------------------------------------------------
# The store, built from real vectors
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_store_indexed_all_tenants(live_store, corpus):
    assert live_store.tenant_count() == 3
    assert live_store.total_docs() == 24
    assert sorted(live_store.tenants()) == sorted(t.id for t in corpus.tenants)
    for tenant in corpus.tenants:
        assert live_store.doc_count(tenant.id) == len(tenant.documents)


@pytest.mark.live
def test_all_vectors_are_real_embeddings(live_store):
    for tid in live_store.tenants():
        for vec in live_store._vectors[tid]:
            assert vec.shape == (768,)
            assert vec.dtype == np.float32
            assert np.isfinite(vec).all()


@pytest.mark.live
def test_real_vectors_are_not_orthogonal(client, embed_model):
    """Why running this live matters.

    The canned test vectors are clean unit basis vectors, so every non-matching
    similarity is exactly 0.0. Real sentence embeddings are crowded into a
    narrow cone -- even unrelated documents score well above 0. Isolation must
    hold in that regime too, which is what the tests below check.
    """
    a = client.embed("Acme Corp quarterly revenue report", model=embed_model)
    b = client.embed("Healthplus patient intake policy", model=embed_model)
    sim = float(np.dot(a, b))  # both unit-norm, so dot == cosine
    assert sim > 0.05, f"expected a crowded embedding space, got cosine {sim}"
    assert sim < 0.99


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_search_only_returns_own_tenant_documents(retriever, corpus):
    """Cross-checked by document identity, not just by index.

    An index is meaningless on its own -- index 3 exists for every tenant. This
    asserts the returned *text* belongs to the querying tenant.
    """
    by_tenant = {t.id: set(t.documents) for t in corpus.tenants}

    for tenant in corpus.tenants:
        others = set()
        for other_id, docs in by_tenant.items():
            if other_id != tenant.id:
                others |= docs

        for query in corpus.queries.get(tenant.id, []):
            results = retriever.retrieve(tenant.id, query, top_k=5)
            assert results, "real embeddings should retrieve something"
            for r in results:
                assert r["document"] in by_tenant[tenant.id]
                assert r["document"] not in others


@pytest.mark.live
def test_indices_are_tenant_relative_and_in_range(retriever, corpus):
    for tenant in corpus.tenants:
        n_docs = len(tenant.documents)
        for query in corpus.queries.get(tenant.id, []):
            for r in retriever.retrieve(tenant.id, query, top_k=5):
                assert 0 <= r["index"] < n_docs


@pytest.mark.live
def test_results_are_score_sorted(retriever, corpus):
    tenant = corpus.tenants[0]
    results = retriever.retrieve(
        tenant.id, corpus.queries[tenant.id][0], top_k=5
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


@pytest.mark.live
def test_unknown_tenant_retrieves_nothing(retriever):
    """A tenant id that was never indexed must yield no documents at all."""
    assert retriever.retrieve("no-such-tenant", "any query", top_k=5) == []


@pytest.mark.live
def test_top_k_capped_by_tenant_doc_count(retriever, corpus):
    tenant = corpus.tenants[0]
    results = retriever.retrieve(tenant.id, "anything", top_k=99)
    assert len(results) == len(tenant.documents)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_evaluate_isolation_reports_zero_leakage(retriever, corpus):
    """The security property, end to end on real vectors."""
    result = evaluate_isolation(retriever, corpus, top_k=3)

    assert result.num_tenants == 3
    assert len(result.tenant_results) == 3
    assert result.total_leakage == 0, "isolated retrieval must never leak"
    for tr in result.tenant_results:
        assert tr.leakage_count == 0
        assert 0.0 <= tr.isolated_recall <= 1.0
        assert 0.0 <= tr.isolated_precision <= 1.0
        assert tr.total_retrieved == 3 * len(corpus.queries[tr.tenant_id])
    assert 0.0 <= result.avg_isolated_recall <= 1.0


@pytest.mark.live
def test_non_isolated_baseline_does_leak(retriever, corpus):
    """The contrast the project exists to demonstrate.

    evaluate_cross_tenant_leakage deliberately drops the tenant filter. With
    three tenants competing for the same top_k slots, foreign documents surface
    -- which is the whole point of the baseline. Asserted in aggregate across
    all 9 queries, not per tenant, since which specific documents win is a
    ranking outcome and not something this suite pins down.
    """
    result = evaluate_cross_tenant_leakage(retriever, corpus, top_k=3)

    assert result.num_tenants == 3
    assert result.total_leakage > 0, (
        "the non-isolated baseline returned no foreign documents; it is "
        "supposed to demonstrate leakage"
    )
    for tr in result.tenant_results:
        assert 0 <= tr.leakage_count <= tr.total_retrieved


@pytest.mark.live
def test_isolated_beats_baseline_on_leakage(retriever, corpus):
    """Isolation must strictly reduce leakage versus no isolation."""
    isolated = evaluate_isolation(retriever, corpus, top_k=3)
    baseline = evaluate_cross_tenant_leakage(retriever, corpus, top_k=3)
    assert isolated.total_leakage < baseline.total_leakage


@pytest.mark.live
def test_retrieve_all_tenants_spans_every_tenant(retriever, corpus):
    """The unfiltered path really does reach every partition."""
    out = retriever.retrieve_all_tenants("quarterly financial report", top_k=2)
    assert set(out) == {t.id for t in corpus.tenants}
    for tid, results in out.items():
        assert len(results) == 2
        for r in results:
            assert set(r) == {"index", "document", "score"}


# ---------------------------------------------------------------------------
# Reporting over a real evaluation
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_reports_render_from_a_real_evaluation(retriever, corpus):
    import json

    result = evaluate_isolation(retriever, corpus, top_k=3)

    md = evaluation_to_markdown(result)
    assert isinstance(md, str) and md.strip()
    for tenant in corpus.tenants:
        assert tenant.id in md

    # evaluation_to_json nests under "isolated", and only adds
    # "non_isolated" when a second result is passed.
    payload = json.loads(evaluation_to_json(result))
    assert payload["isolated"]["num_tenants"] == 3
    assert payload["isolated"]["total_leakage"] == 0
    assert len(payload["isolated"]["per_tenant"]) == 3
    assert "non_isolated" not in payload

    both = json.loads(
        evaluation_to_json(result, evaluate_cross_tenant_leakage(retriever, corpus, top_k=3))
    )
    assert both["non_isolated"]["total_leakage"] > 0
