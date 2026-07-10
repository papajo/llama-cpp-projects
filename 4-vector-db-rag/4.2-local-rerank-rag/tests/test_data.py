"""Tests for data module."""

from local_rerank_rag.data import RagCorpus, default_corpus, mini_corpus


class TestData:
    def test_default_corpus(self):
        c = default_corpus()
        assert len(c.documents) == 25
        assert len(c.queries) == 10
        assert len(c.relevant_doc_ids) == 10

    def test_mini_corpus(self):
        c = mini_corpus()
        assert len(c.documents) == 5
        assert len(c.queries) == 2

    def test_all_indices_valid(self):
        c = default_corpus()
        n = len(c.documents)
        for indices in c.relevant_doc_ids.values():
            for idx in indices:
                assert 0 <= idx < n
