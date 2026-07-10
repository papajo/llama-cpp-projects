"""Tests for the built-in corpus data."""

from embedding_norm_ablation.data import Corpus, default_corpus, mini_corpus


class TestCorpus:
    def test_default_corpus_has_documents(self):
        c = default_corpus()
        assert isinstance(c, Corpus)
        assert len(c.documents) == 30
        assert len(c.queries) == 10

    def test_default_corpus_has_relevance(self):
        c = default_corpus()
        assert len(c.relevant_doc_ids) == 10

    def test_all_relevance_indices_in_range(self):
        c = default_corpus()
        n = len(c.documents)
        for q_idx, doc_indices in c.relevant_doc_ids.items():
            assert 0 <= q_idx < len(c.queries)
            for d_idx in doc_indices:
                assert 0 <= d_idx < n, f"doc {d_idx} out of range (0-{n-1})"

    def test_mini_corpus_small(self):
        c = mini_corpus()
        assert len(c.documents) == 5
        assert len(c.queries) == 2

    def test_mini_corpus_has_relevance(self):
        c = mini_corpus()
        assert len(c.relevant_doc_ids) == 2

    def test_corpus_is_frozen(self):
        c = default_corpus()
        import dataclasses
        assert dataclasses.is_dataclass(c) and isinstance(c, Corpus)
