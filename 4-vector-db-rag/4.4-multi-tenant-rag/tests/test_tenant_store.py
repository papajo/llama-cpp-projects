"""Tests for tenant vector store."""

import numpy as np
import pytest

from multi_tenant_rag.tenant_store import TenantVectorStore, cosine_similarity


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestTenantVectorStore:
    def test_add_and_search(self):
        store = TenantVectorStore()
        store.add("tenant_a", "doc a1", np.array([1.0, 0.0]))
        store.add("tenant_a", "doc a2", np.array([0.0, 1.0]))
        store.add("tenant_b", "doc b1", np.array([0.9, 0.1]))

        # Search tenant_a
        results = store.search("tenant_a", np.array([1.0, 0.0]), top_k=2)
        assert results == [0, 1]  # a1 then a2

        # Search tenant_b
        results = store.search("tenant_b", np.array([1.0, 0.0]), top_k=2)
        assert results == [0]  # only b1

    def test_no_leakage(self):
        """Tenant A queries should never return Tenant B docs."""
        store = TenantVectorStore()
        store.add("tenant_a", "a1", np.array([1.0, 0.0]))
        store.add("tenant_b", "b1", np.array([1.0, 0.0]))

        results = store.search("tenant_a", np.array([1.0, 0.0]), top_k=5)
        assert len(results) == 1  # only a1

    def test_add_many(self):
        store = TenantVectorStore()
        store.add_many("t1", ["d1", "d2"], [np.array([1.0, 0.0]), np.array([0.0, 1.0])])
        assert store.doc_count("t1") == 2

    def test_empty_store(self):
        store = TenantVectorStore()
        assert store.search("nonexistent", np.array([1.0, 0.0])) == []

    def test_search_all_tenants(self):
        store = TenantVectorStore()
        store.add("a", "a1", np.array([1.0, 0.0]))
        store.add("b", "b1", np.array([0.0, 1.0]))
        all_r = store.search_all_tenants(np.array([1.0, 0.0]), top_k=5)
        assert "a" in all_r
        assert "b" in all_r

    def test_tenant_count(self):
        store = TenantVectorStore()
        store.add("a", "d1", np.array([1.0, 0.0]))
        store.add("b", "d1", np.array([1.0, 0.0]))
        assert store.tenant_count() == 2

    def test_total_docs(self):
        store = TenantVectorStore()
        store.add("a", "d1", np.array([1.0, 0.0]))
        store.add("a", "d2", np.array([0.0, 1.0]))
        store.add("b", "d1", np.array([1.0, 0.0]))
        assert store.total_docs() == 3
