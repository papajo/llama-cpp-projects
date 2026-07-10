"""Map-reduce agent — split, process chunks in parallel, aggregate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

ChunkFn = Callable[[Any, int], Any]       # (chunk_data, chunk_index) → processed
ReduceFn = Callable[[List[Any]], Any]       # [processed_chunks] → final result
SplitFn = Callable[[Any], List[Any]]        # raw input → [chunks]


@dataclass
class Chunk:
    """A single chunk of data from the split phase."""

    index: int
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MapResult:
    """Result of processing a single chunk."""

    chunk_index: int
    input_data: Any
    output: Any = None
    error: Optional[str] = None
    succeeded: bool = True


@dataclass
class MapReduceResult:
    """Overall result of a map-reduce operation."""

    input_data: Any
    chunks: List[Chunk] = field(default_factory=list)
    map_results: List[MapResult] = field(default_factory=list)
    reduced_output: Any = None
    error: Optional[str] = None

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_succeeded(self) -> int:
        return sum(1 for r in self.map_results if r.succeeded)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.map_results if r.error is not None)

    @property
    def all_succeeded(self) -> bool:
        return self.num_failed == 0 and self.num_chunks > 0

    @property
    def map_success_rate(self) -> float:
        if not self.map_results:
            return 0.0
        return self.num_succeeded / len(self.map_results)


def split_fixed_size(text: str, size: int) -> List[str]:
    """Split text into fixed-size character chunks."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def split_lines(text: str) -> List[str]:
    """Split text into lines (stripped, non-empty)."""
    return [line for line in text.split("\n") if line.strip()]


def split_by_delimiter(text: str, delimiter: str = ".") -> List[str]:
    """Split text by a delimiter, preserving delimiter in output."""
    parts = text.split(delimiter)
    return [p.strip() + delimiter for p in parts if p.strip()]


def reduce_concat(results: List[str]) -> str:
    """Concatenate all string results."""
    return "".join(results)


def reduce_sum(results: List[float]) -> float:
    """Sum all numeric results."""
    return sum(results)


def reduce_list(results: List[Any]) -> list:
    """Flatten results into a single list."""
    flattened = []
    for r in results:
        if isinstance(r, list):
            flattened.extend(r)
        else:
            flattened.append(r)
    return flattened


@dataclass
class MapReduceAgent:
    """Agent that splits input, maps chunks in parallel, and reduces.

    Example workflow:
      1. Split: "abc def ghi" → ["abc", "def", "ghi"]
      2. Map (parallel): count words in each → [1, 1, 1]
      3. Reduce: sum → 3
    """

    chunk_fn: ChunkFn
    reduce_fn: ReduceFn
    split_fn: Optional[SplitFn] = None
    max_workers: int = 4

    def run(self, input_data: Any) -> MapReduceResult:
        result = MapReduceResult(input_data=input_data)

        # 1. Split
        if self.split_fn is not None:
            raw_chunks = self.split_fn(input_data)
        else:
            raw_chunks = [input_data]  # single chunk if no splitter

        chunks = [
            Chunk(index=i, data=data)
            for i, data in enumerate(raw_chunks)
        ]
        result.chunks = chunks

        if not chunks:
            return result

        # 2. Map (parallel)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        map_results: Dict[int, MapResult] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as pool:
            fut_map = {}
            for chunk in chunks:
                fut = pool.submit(self.chunk_fn, chunk.data, chunk.index)
                fut_map[fut] = chunk

            for fut in as_completed(fut_map):
                chunk = fut_map[fut]
                try:
                    output = fut.result()
                    map_results[chunk.index] = MapResult(
                        chunk_index=chunk.index,
                        input_data=chunk.data,
                        output=output,
                        succeeded=True,
                    )
                except Exception as e:
                    map_results[chunk.index] = MapResult(
                        chunk_index=chunk.index,
                        input_data=chunk.data,
                        error=str(e),
                        succeeded=False,
                    )

        result.map_results = [map_results[i] for i in sorted(map_results)]

        if not result.all_succeeded:
            failed = [r for r in result.map_results if r.error]
            errors = "; ".join(f"chunk {r.chunk_index}: {r.error}" for r in failed)
            result.error = f"Map phase failed: {errors}"
            return result

        # 3. Reduce
        try:
            outputs = [r.output for r in result.map_results if r.succeeded]
            result.reduced_output = self.reduce_fn(outputs)
        except Exception as e:
            result.error = f"Reduce phase failed: {e}"

        return result
