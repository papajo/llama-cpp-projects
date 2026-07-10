"""Tests for map-reduce agent."""

import time

import pytest

from map_reduce.core import (
    Chunk,
    MapReduceAgent,
    MapReduceResult,
    MapResult,
    reduce_concat,
    reduce_list,
    reduce_sum,
    split_by_delimiter,
    split_fixed_size,
    split_lines,
)


def count_words(chunk: str, idx: int) -> int:
    return len(chunk.split())


def slow_upper(chunk: str, idx: int) -> str:
    time.sleep(0.05)
    return chunk.upper()


class TestChunk:
    def test_basic(self):
        c = Chunk(index=0, data="hello")
        assert c.data == "hello"
        assert c.index == 0


class TestSplitFunctions:
    def test_split_fixed_size(self):
        result = split_fixed_size("abcdefghij", 3)
        assert result == ["abc", "def", "ghi", "j"]

    def test_split_lines(self):
        result = split_lines("abc\ndef\n\nghi")
        assert result == ["abc", "def", "ghi"]

    def test_split_by_delimiter(self):
        result = split_by_delimiter("Hello.World.Test.", ".")
        assert result == ["Hello.", "World.", "Test."]


class TestReduceFunctions:
    def test_reduce_concat(self):
        assert reduce_concat(["a", "b", "c"]) == "abc"

    def test_reduce_sum(self):
        assert reduce_sum([1, 2, 3]) == 6

    def test_reduce_list(self):
        assert reduce_list([[1, 2], [3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
        assert reduce_list(["a", "b"]) == ["a", "b"]


class TestMapReduceAgent:
    def test_no_splitter_single_chunk(self):
        agent = MapReduceAgent(
            chunk_fn=lambda d, i: d.upper(),
            reduce_fn=reduce_concat,
        )
        result = agent.run("hello")
        assert result.num_chunks == 1
        assert result.reduced_output == "HELLO"
        assert result.all_succeeded

    def test_split_and_count_words(self):
        agent = MapReduceAgent(
            split_fn=lambda t: split_fixed_size(t, 10),
            chunk_fn=count_words,
            reduce_fn=reduce_sum,
        )
        result = agent.run("hello world foo bar baz qux")
        assert result.num_chunks > 1
        assert result.reduced_output > 0

    def test_map_concurrency(self):
        """Multiple chunks should process faster than sequential."""
        text = "\n".join([f"line {i}" for i in range(10)])
        agent = MapReduceAgent(
            split_fn=split_lines,
            chunk_fn=slow_upper,
            reduce_fn=reduce_concat,
            max_workers=10,
        )
        start = time.time()
        result = agent.run(text)
        elapsed = time.time() - start
        # 10 chunks × 50ms each in parallel = ~50-100ms
        assert elapsed < 0.15
        assert result.all_succeeded
        # Each line should be uppercased
        assert "LINE" in result.reduced_output

    def test_split_lines(self):
        agent = MapReduceAgent(
            split_fn=split_lines,
            chunk_fn=lambda d, i: d.strip(),
            reduce_fn=reduce_concat,
        )
        result = agent.run("  hello  \n  world  \n")
        assert result.num_chunks == 2
        assert result.reduced_output == "helloworld"

    def test_chunk_failure(self):
        def fail(chunk, idx):
            if idx == 1:
                raise ValueError("chunk error")
            return chunk

        agent = MapReduceAgent(
            split_fn=split_lines,
            chunk_fn=fail,
            reduce_fn=reduce_concat,
        )
        result = agent.run("a\nb\nc")
        assert result.all_succeeded is False
        assert result.num_failed == 1
        assert "chunk error" in result.error

    def test_empty_input(self):
        agent = MapReduceAgent(
            split_fn=lambda t: [],
            chunk_fn=lambda d, i: d,
            reduce_fn=reduce_concat,
        )
        result = agent.run("")
        assert result.num_chunks == 0
        assert result.reduced_output is None
        assert result.all_succeeded is False  # no chunks

    def test_no_split_fn_passthrough(self):
        agent = MapReduceAgent(
            chunk_fn=lambda d, i: d * 2,
            reduce_fn=reduce_concat,
        )
        result = agent.run("x")
        assert result.num_chunks == 1
        assert result.reduced_output == "xx"

    def test_reduce_failure(self):
        def bad_reduce(outputs):
            raise ValueError("reduce crash")

        agent = MapReduceAgent(
            chunk_fn=lambda d, i: d,
            reduce_fn=bad_reduce,
        )
        result = agent.run("test")
        assert result.error is not None
        assert "reduce crash" in result.error

    def test_map_reduce_result_properties(self):
        r = MapReduceResult(input_data="test")
        assert r.num_chunks == 0
        assert r.num_succeeded == 0
        assert r.num_failed == 0
        assert r.all_succeeded is False
        assert r.map_success_rate == 0.0

        r.chunks = [Chunk(0, "a"), Chunk(1, "b")]
        r.map_results = [
            MapResult(0, "a", "A", succeeded=True),
            MapResult(1, "b", "B", succeeded=True),
        ]
        assert r.num_chunks == 2
        assert r.num_succeeded == 2
        assert r.all_succeeded is True
        assert r.map_success_rate == 1.0

    def test_reduce_list_of_dicts(self):
        def extract_words(chunk, idx):
            return chunk.split()

        agent = MapReduceAgent(
            split_fn=split_lines,
            chunk_fn=extract_words,
            reduce_fn=reduce_list,
        )
        result = agent.run("hello world\nfoo bar")
        assert result.all_succeeded
        assert "hello" in result.reduced_output
        assert "world" in result.reduced_output
