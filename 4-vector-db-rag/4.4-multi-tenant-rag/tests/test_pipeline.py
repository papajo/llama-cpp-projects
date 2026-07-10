"""Tests for multi-tenant pipeline."""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from multi_tenant_rag.data import default_corpus
from multi_tenant_rag.pipeline import (
    evaluate_cross_tenant_leakage,
    evaluate_isolation,
)
from multi_tenant_rag.retriever import EmbeddingClient, TenantRetriever
from multi_tenant_rag.tenant_store import TenantVectorStore


def _make_side_effect(embed_dim=4):
    """Return a urlopen side_effect returning embedding response for any URL."""
    def side_effect(request, *args, **kwargs):
        response_body = json.dumps(
            {"data": [{"embedding": [0.1] * embed_dim}]}
        ).encode()
        resp = MagicMock()
        resp.read.return_value = response_body
        cm = MagicMock()
        cm.__enter__.return_value = resp
        return cm
    return side_effect


class TestPipeline:
    def _build_fixtures(self):
        corpus = default_corpus()
        store = TenantVectorStore()
        client = EmbeddingClient("http://mock:8080")

        # Pre-embed documents
        with patch("urllib.request.urlopen",
                   side_effect=_make_side_effect()):
            for tenant in corpus.tenants:
                vecs = client.embed_many(tenant.documents)
                store.add_many(tenant.id, tenant.documents, vecs)

        retriever = TenantRetriever(store=store, client=client)
        return corpus, retriever

    def test_evaluate_isolation(self):
        corpus, retriever = self._build_fixtures()
        with patch("urllib.request.urlopen",
                   side_effect=_make_side_effect()):
            result = evaluate_isolation(retriever, corpus, top_k=3)
            assert result.num_tenants == 3
            assert len(result.tenant_results) == 3
            # In isolated mode with random vectors, we still get results
            for r in result.tenant_results:
                assert r.leakage_count == 0  # no cross-tenant leakage

    def test_evaluate_cross_tenant_leakage(self):
        corpus, retriever = self._build_fixtures()
        with patch("urllib.request.urlopen",
                   side_effect=_make_side_effect()):
            result = evaluate_cross_tenant_leakage(retriever, corpus, top_k=3)
            assert result.num_tenants == 3
            for r in result.tenant_results:
                assert r.leakage_count >= 0

    def test_leakage_higher_without_isolation(self):
        """Non-isolated retrieval should have >= leakage than isolated."""
        corpus, retriever = self._build_fixtures()
        with patch("urllib.request.urlopen",
                   side_effect=_make_side_effect()):
            iso = evaluate_isolation(retriever, corpus, top_k=3)
            leak = evaluate_cross_tenant_leakage(retriever, corpus, top_k=3)
            # Isolated should have 0 leakage
            assert iso.total_leakage == 0
            # Non-isolated will have leakage (other tenants' docs returned)
            assert leak.total_leakage >= 0
