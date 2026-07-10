"""Tests for chunkers."""

import pytest

from prompt_cache_chunking.chunkers import (
    Chunk,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
)


class TestChunk:
    def test_token_estimate(self):
        c = Chunk(doc_index=0, text="hello world")
        assert c.token_estimate > 0


class TestFixedSizeChunker:
    def test_basic(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        docs = ["a" * 120]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 3  # 120/50 = 3
        assert all(c.doc_index == 0 for c in chunks)

    def test_with_overlap(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        docs = ["a" * 120]
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 3
        # Overlap means the second chunk starts at 40 not 50
        assert len(chunks[0].text) == 50

    def test_multiple_docs(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        docs = ["a" * 50, "b" * 50]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 2
        assert chunks[0].doc_index == 0
        assert chunks[1].doc_index == 1

    def test_empty_doc(self):
        chunker = FixedSizeChunker(chunk_size=50)
        assert chunker.chunk([""]) == []


class TestSentenceChunker:
    def test_basic(self):
        chunker = SentenceChunker(max_sentences=2, overlap_sentences=0)
        docs = ["First sentence. Second sentence. Third sentence. Fourth sentence."]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 2

    def test_overlap(self):
        chunker = SentenceChunker(max_sentences=2, overlap_sentences=1)
        docs = ["S1. S2. S3. S4."]
        chunks = chunker.chunk(docs)
        # With max=2, overlap=1: [S1,S2], [S2,S3], [S3,S4]
        assert len(chunks) == 3

    def test_sentence_splitter(self):
        sentences = SentenceChunker._split_sentences("Hello world. How are you? I'm fine.")
        assert len(sentences) == 3

    def test_no_documents(self):
        chunker = SentenceChunker()
        assert chunker.chunk([]) == []


class TestParagraphChunker:
    def test_splits_at_midpoint(self):
        chunker = ParagraphChunker()
        # A doc long enough that a ". " falls within the first 30-70% range
        docs = [
            "This is the first paragraph of a longer document. It discusses the "
            "initial topic and sets up the context. Now here is the second half "
            "of the document that covers a different aspect entirely."
        ]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 2

    def test_short_doc_stays_whole(self):
        chunker = ParagraphChunker()
        docs = ["Short doc."]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1


class TestRecursiveChunker:
    def test_basic(self):
        chunker = RecursiveChunker(max_chars=50)
        docs = ["a" * 120]
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 2

    def test_respects_sentence_boundaries(self):
        chunker = RecursiveChunker(max_chars=60)
        docs = ["Short part. " * 20]
        chunks = chunker.chunk(docs)
        for c in chunks:
            assert len(c.text) <= 60

    def test_small_doc(self):
        chunker = RecursiveChunker(max_chars=100)
        docs = ["Short doc."]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1
