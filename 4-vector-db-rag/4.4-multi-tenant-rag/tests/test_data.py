"""Tests for multi-tenant data."""

from multi_tenant_rag.data import default_corpus


class TestData:
    def test_three_tenants(self):
        c = default_corpus()
        assert len(c.tenants) == 3

    def test_each_tenant_has_docs(self):
        c = default_corpus()
        for t in c.tenants:
            assert len(t.documents) == 8

    def test_each_tenant_has_queries(self):
        c = default_corpus()
        for t in c.tenants:
            assert len(c.queries[t.id]) == 3

    def test_each_tenant_has_relevance(self):
        c = default_corpus()
        for t in c.tenants:
            assert len(c.relevant_doc_indices[t.id]) == 3
