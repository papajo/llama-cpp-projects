"""Chunking strategies for document splitting."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """A single chunk with its source document index and text."""

    doc_index: int
    text: str

    @property
    def token_estimate(self) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(self.text) // 4)


class Chunker(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, documents: List[str]) -> List[Chunk]:
        """Split documents into chunks."""
        ...


# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------


class FixedSizeChunker(Chunker):
    """Split each document into chunks of *chunk_size* characters.

    Uses *overlap* characters of overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[str]) -> List[Chunk]:
        result: List[Chunk] = []
        for doc_idx, doc in enumerate(documents):
            start = 0
            while start < len(doc):
                end = min(start + self.chunk_size, len(doc))
                result.append(Chunk(doc_index=doc_idx, text=doc[start:end]))
                if end == len(doc):
                    break
                start += self.chunk_size - self.overlap
        return result


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------


class SentenceChunker(Chunker):
    """Split each document by sentences, grouping into chunks of up to
    *max_sentences* sentences with *overlap_sentences* overlap."""

    def __init__(self, max_sentences: int = 3, overlap_sentences: int = 1):
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, documents: List[str]) -> List[Chunk]:
        result: List[Chunk] = []
        for doc_idx, doc in enumerate(documents):
            sentences = self._split_sentences(doc)
            start = 0
            while start < len(sentences):
                end = min(start + self.max_sentences, len(sentences))
                text = " ".join(sentences[start:end])
                result.append(Chunk(doc_index=doc_idx, text=text))
                if end == len(sentences):
                    break
                start += self.max_sentences - self.overlap_sentences
        return result


# ---------------------------------------------------------------------------
# Paragraph chunker
# ---------------------------------------------------------------------------


class ParagraphChunker(Chunker):
    """Each paragraph (split by double newline or single newline for
    these dense texts) becomes a chunk."""

    def chunk(self, documents: List[str]) -> List[Chunk]:
        result: List[Chunk] = []
        for doc_idx, doc in enumerate(documents):
            # Split on sentence boundaries as paragraphs (these are
            # dense single-paragraph docs so we split into ~2 halves)
            mid = len(doc) // 2
            # Find the nearest sentence boundary
            first_half = doc[:mid]
            second_half = doc[mid:]
            # Try to split at a `. ` near the midpoint
            split_at = first_half.rfind(". ")
            if split_at > len(first_half) * 0.3:  # only if reasonably placed
                para1 = doc[: split_at + 1]
                para2 = doc[split_at + 1 :].strip()
                if para2:
                    result.append(Chunk(doc_index=doc_idx, text=para1))
                    result.append(Chunk(doc_index=doc_idx, text=para2))
                    continue
            result.append(Chunk(doc_index=doc_idx, text=doc))
        return result


# ---------------------------------------------------------------------------
# Recursive chunker — splits on progressively finer boundaries
# ---------------------------------------------------------------------------


class RecursiveChunker(Chunker):
    """Recursively split on paragraph → sentence boundaries until
    each chunk is under *max_chars*."""

    def __init__(self, max_chars: int = 150):
        self.max_chars = max_chars

    def chunk(self, documents: List[str]) -> List[Chunk]:
        result: List[Chunk] = []
        for doc_idx, doc in enumerate(documents):
            self._recursive_split(doc, doc_idx, result)
        return result

    def _recursive_split(
        self, text: str, doc_idx: int, result: List[Chunk]
    ) -> None:
        if len(text) <= self.max_chars:
            result.append(Chunk(doc_index=doc_idx, text=text))
            return

        # Try splitting at the last sentence boundary within max_chars
        candidate = text[: self.max_chars]
        split_at = candidate.rfind(". ")
        if split_at > self.max_chars * 0.3:
            result.append(Chunk(doc_index=doc_idx, text=text[: split_at + 1]))
            remainder = text[split_at + 1 :].strip()
            if remainder:
                self._recursive_split(remainder, doc_idx, result)
        else:
            # Hard split at max_chars
            result.append(Chunk(doc_index=doc_idx, text=candidate))
            remainder = text[self.max_chars :].strip()
            if remainder:
                self._recursive_split(remainder, doc_idx, result)
