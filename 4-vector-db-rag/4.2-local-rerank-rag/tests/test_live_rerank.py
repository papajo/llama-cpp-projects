"""Live integration tests for 4.2 against the real servers.

Covers the same paths as the mocked tests in test_reranker.py,
test_retriever.py and test_pipeline.py, but end to end:
embeddings from the real embed server, LLM-as-judge scores from the real
chat server.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q

Note on scope: this project's "reranker" is LLM-as-judge over
/v1/chat/completions. llama.cpp's native /v1/rerank is covered separately
against a real cross-encoder (bge-reranker-v2-m3) on its own server -- see
test_native_rerank_*. It runs separately because --reranking forces pooling to
"rank", which corrupts /v1/embeddings on the same process.

Every assertion is structural. SmolLM2-360M scores documents poorly and
inconsistently; ranking *quality* is deliberately not asserted anywhere.
The chat model is slow on CPU, so candidate counts are kept small.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from local_rerank_rag.data import mini_corpus
from local_rerank_rag.pipeline import (
    evaluate_pipeline,
    run_pipeline,
    run_pipeline_keyword_baseline,
)
from local_rerank_rag.reranker import (
    RELEVANCE_SYSTEM_PROMPT,
    LlamaClient,
    Reranker,
    keyword_overlap_score,
)
from local_rerank_rag.retriever import EmbeddingClient, Retriever, VectorStore


@pytest.fixture
def embed_client(embed_base_url):
    return EmbeddingClient(server_url=embed_base_url)


@pytest.fixture
def llama_client(chat_base_url):
    return LlamaClient(server_url=chat_base_url)


@pytest.fixture
def reranker(llama_client, chat_model):
    return Reranker(client=llama_client, model=chat_model)


@pytest.fixture
def live_store(embed_client, embed_model):
    """A VectorStore populated with real embeddings of the mini corpus."""
    corpus = mini_corpus()
    store = VectorStore()
    store.add_many(
        corpus.documents,
        embed_client.embed_many(corpus.documents, model=embed_model),
    )
    return store


# ---------------------------------------------------------------------------
# LlamaClient against the real chat server
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_llama_client_complete_returns_text(llama_client, chat_model):
    out = llama_client.complete(
        [{"role": "user", "content": "Say hi"}], model=chat_model, max_tokens=8
    )
    assert isinstance(out, str)
    assert out.strip()


@pytest.mark.live
def test_chat_response_has_llamacpp_only_keys(live_chat):
    """The canned _mock_llm_response() is a bare minimum of the real envelope.

    The mock is {"choices": [{"message": {"content": ...}}]}. The real server
    also returns id/created/model/object/system_fingerprint/usage, a
    per-choice finish_reason and index, plus llama.cpp-specific `timings` and
    usage.prompt_tokens_details.cached_tokens. See drift-rag.md.
    """
    r = live_chat([{"role": "user", "content": "Say hi"}], max_tokens=8)
    assert r["object"] == "chat.completion"
    assert r["id"].startswith("chatcmpl-")
    # Not pinned to "stop": at max_tokens=8 and the server's default
    # temperature, a short reply may or may not truncate. The dedicated
    # truncation test below asserts "length" deterministically.
    assert r["choices"][0]["finish_reason"] in {"stop", "length"}
    assert r["choices"][0]["index"] == 0
    assert set(r["usage"]) >= {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
    }
    assert "cached_tokens" in r["usage"]["prompt_tokens_details"]
    assert "timings" in r
    assert r["timings"]["predicted_n"] >= 1


@pytest.mark.live
def test_finish_reason_length_when_truncated(live_chat):
    """Hitting max_tokens yields finish_reason 'length', not 'stop'.

    `ignore_eos` (a llama.cpp extension, not OpenAI) is what makes this
    deterministic. Without it SmolLM2-360M often emits EOS within a few
    tokens even for "count to 100", so finish_reason flips between runs --
    exactly the kind of output-quality variance this suite must not depend on.
    """
    r = live_chat(
        [{"role": "user", "content": "Say hi"}], max_tokens=8, ignore_eos=True
    )
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] == 8


# ---------------------------------------------------------------------------
# Native /v1/rerank against a real cross-encoder
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_native_rerank_absent_from_chat_and_embed_servers(
    chat_base_url, embed_base_url
):
    """/v1/rerank stays 501 on the chat and embeddings servers, by design.

    --reranking forces pooling to "rank", which corrupts /v1/embeddings on the
    same process (verified: vectors come back as denormals and garbage). So the
    cross-encoder runs on its own server and these two must NOT offer the
    endpoint. This asserts that separation holds.
    """
    for base in (chat_base_url, embed_base_url):
        req = urllib.request.Request(
            f"{base}/v1/rerank",
            data=json.dumps(
                {"model": "m", "query": "q", "documents": ["d1", "d2"]}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 501
        err = json.loads(exc.value.read().decode())["error"]
        assert err["type"] == "not_supported_error"
        assert "--reranking" in err["message"]


@pytest.mark.live
def test_native_rerank_response_shape(live_rerank):
    """The real /v1/rerank envelope, pinned against a cross-encoder."""
    r = live_rerank("capital of France", ["Paris is the capital of France.",
                                          "Bananas are rich in potassium."])
    assert r["object"] == "list"
    assert {"model", "object", "results", "usage"} <= set(r)
    assert len(r["results"]) == 2
    for item in r["results"]:
        assert set(item) >= {"index", "relevance_score"}
        assert isinstance(item["index"], int)
        assert isinstance(item["relevance_score"], float)
    # One row per input document, indices covering the input positions.
    assert sorted(item["index"] for item in r["results"]) == [0, 1]
    # Embedding-style usage: prompt/total only, no completion_tokens.
    assert set(r["usage"]) == {"prompt_tokens", "total_tokens"}


@pytest.mark.live
def test_native_rerank_scores_are_raw_logits_not_0_to_1(live_rerank):
    """llama.cpp returns raw cross-encoder logits, unlike hosted rerank APIs.

    Cohere and Jina normalise relevance_score to 0.0-1.0. llama.cpp does not:
    a strong match scores well above 1 and a poor one goes negative. Any code
    assuming a 0-1 range silently misreads these.
    """
    r = live_rerank(
        "What is the capital of France?",
        [
            "Paris is the capital and largest city of France.",
            "The Great Barrier Reef is off the coast of Australia.",
        ],
    )
    scores = {item["index"]: item["relevance_score"] for item in r["results"]}
    assert scores[0] > 1.0, "expected a logit above the 0-1 band for a match"
    assert scores[1] < 0.0, "expected a negative logit for an unrelated doc"


@pytest.mark.live
def test_native_rerank_discriminates_relevant_from_irrelevant(live_rerank):
    """The cross-encoder ranks the answer above the distractors.

    Unlike the LLM-as-judge path, a real cross-encoder is strong enough that
    ranking *quality* can be asserted here - the gap is orders of magnitude,
    not a coin flip.
    """
    docs = [
        "The Great Barrier Reef is off the coast of Australia.",
        "Bananas are a good source of potassium.",
        "Paris is the capital and largest city of France.",
    ]
    r = live_rerank("What is the capital of France?", docs)
    ranked = sorted(r["results"], key=lambda i: -i["relevance_score"])
    assert ranked[0]["index"] == 2, "the Paris document should rank first"
    assert ranked[0]["relevance_score"] > ranked[1]["relevance_score"] + 5.0


@pytest.mark.live
def test_native_rerank_preserves_input_order_in_indices(live_rerank):
    """index refers to the caller's document order, whatever the ranking."""
    docs = ["Paris is the capital of France.", "Potassium is in bananas.",
            "Australia has a large reef."]
    r = live_rerank("France", docs)
    assert sorted(i["index"] for i in r["results"]) == [0, 1, 2]
    # The results array is not required to be pre-sorted by score; callers sort.
    best = max(r["results"], key=lambda i: i["relevance_score"])
    assert best["index"] == 0


@pytest.mark.live
def test_native_rerank_single_document(live_rerank):
    """A one-document request still returns a well-formed single row."""
    r = live_rerank("France", ["Paris is the capital of France."])
    assert len(r["results"]) == 1
    assert r["results"][0]["index"] == 0


# ---------------------------------------------------------------------------
# Retrieval over real embeddings
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_embed_client_dimension(embed_client, embed_model):
    vec = embed_client.embed("hello", model=embed_model)
    assert vec.shape == (768,)
    assert vec.dtype == np.float32


@pytest.mark.live
def test_store_populated_with_real_vectors(live_store):
    assert len(live_store) == 5
    assert all(v.shape == (768,) for v in live_store.vectors)


@pytest.mark.live
def test_retriever_returns_well_formed_results(live_store, embed_client, embed_model):
    retriever = Retriever(
        store=live_store, client=embed_client, embed_model=embed_model
    )
    results = retriever.retrieve("gene editing and genetics", top_k=3)

    assert len(results) == 3
    indices = [r["index"] for r in results]
    assert len(set(indices)) == 3, "retrieve() must not return duplicates"
    for r in results:
        assert set(r) == {"index", "document", "score"}
        assert 0 <= r["index"] < len(live_store)
        assert r["document"] == live_store.documents[r["index"]]
        # Real embeddings are unit-norm, so cosine stays inside [-1, 1].
        assert -1.0 <= r["score"] <= 1.0
    # search() must return descending scores.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.live
def test_retriever_top_k_caps_at_corpus_size(
    live_store, embed_client, embed_model
):
    retriever = Retriever(
        store=live_store, client=embed_client, embed_model=embed_model
    )
    assert len(retriever.retrieve("anything", top_k=99)) == 5


# ---------------------------------------------------------------------------
# LLM-as-judge scoring — structure only
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_score_returns_normalised_float(reranker):
    """A real score is a float in [0, 1]. Its *value* is not asserted.

    SmolLM2-360M ignores the JSON instruction and replies with a bare number
    like "0", regardless of how relevant the document is. That is a
    model-capacity limitation, not a bug -- but it did expose a real crash in
    _parse_score (fixed; see drift-rag.md and the regression tests in
    test_reranker.py).
    """
    score = reranker.score(
        "global sporting competitions",
        "The Olympic Games are a major international sporting event.",
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.live
def test_score_never_raises_on_real_model_output(reranker):
    """The parser must survive whatever the small model actually emits."""
    for query, doc in [
        ("gene editing", "CRISPR is a gene-editing tool."),
        ("", "An empty query is still a valid request."),
    ]:
        score = reranker.score(query, doc)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


@pytest.mark.live
def test_system_prompt_is_the_shipped_one(reranker):
    assert reranker.system_prompt == RELEVANCE_SYSTEM_PROMPT


@pytest.mark.live
def test_rerank_returns_sorted_scored_candidates(reranker):
    candidates = [
        {"index": 0, "document": "Einstein developed the theory of relativity."},
        {"index": 1, "document": "CRISPR is a gene-editing tool."},
    ]
    out = reranker.rerank("gene editing and genetics", candidates)

    assert len(out) == 2
    # Original fields are preserved and a score is added.
    assert {c["index"] for c in out} == {0, 1}
    for c in out:
        assert set(c) == {"index", "document", "relevance_score"}
        assert 0.0 <= c["relevance_score"] <= 1.0
    scores = [c["relevance_score"] for c in out]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_run_pipeline_end_to_end(live_store, embed_client, embed_model, reranker):
    retriever = Retriever(
        store=live_store, client=embed_client, embed_model=embed_model
    )
    result = run_pipeline(
        retriever, reranker, "gene editing and genetics", retrieve_k=2, final_k=1
    )

    assert result.query == "gene editing and genetics"
    assert result.num_candidates == 2
    assert result.final_top_k == 1
    assert len(result.retrieved) == 2
    assert len(result.reranked) == 2
    assert all("relevance_score" in c for c in result.reranked)


@pytest.mark.live
def test_keyword_baseline_needs_no_llm(live_store, embed_client, embed_model):
    """The keyword baseline path touches only the embedding server."""
    retriever = Retriever(
        store=live_store, client=embed_client, embed_model=embed_model
    )
    result = run_pipeline_keyword_baseline(
        retriever, "gene editing and genetics", retrieve_k=3, final_k=2
    )
    assert result.num_candidates == 3
    scores = [c["relevance_score"] for c in result.reranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.live
def test_keyword_overlap_matches_real_documents(live_store):
    """Pure-python helper, checked against the real corpus strings."""
    doc = live_store.documents[1]  # CRISPR document
    assert keyword_overlap_score("CRISPR gene-editing tool", doc) > 0.0
    assert keyword_overlap_score("zzz nonexistent", doc) == 0.0


@pytest.mark.live
def test_evaluate_pipeline_metrics_shape(
    live_store, embed_client, embed_model, reranker
):
    """Metrics dict over the real stack: 2 queries x 2 candidates = 4 chat calls."""
    retriever = Retriever(
        store=live_store, client=embed_client, embed_model=embed_model
    )
    out = evaluate_pipeline(
        retriever,
        reranker,
        mini_corpus(),
        retrieve_k=2,
        final_k=1,
        include_baseline=True,
    )

    assert out["num_queries"] == 2
    assert out["retrieve_k"] == 2
    assert out["final_k"] == 1
    for key in ("reranked_mrr", "retrieval_recall", "baseline_mrr"):
        assert 0.0 <= out[key] <= 1.0
    assert len(out["per_query"]) == 2
    for pq in out["per_query"]:
        assert set(pq) == {"query", "retrieved_indices", "reranked_indices"}
        assert len(pq["retrieved_indices"]) == 2
        assert sorted(pq["reranked_indices"]) == sorted(pq["retrieved_indices"])
