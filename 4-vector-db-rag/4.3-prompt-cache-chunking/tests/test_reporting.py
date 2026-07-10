"""Tests for reporting."""

import json

from prompt_cache_chunking.data import default_corpus
from prompt_cache_chunking.experiment import run_experiment
from prompt_cache_chunking.chunkers import FixedSizeChunker
from prompt_cache_chunking.reporting import experiment_to_json, experiment_to_markdown


class TestReporting:
    def test_markdown_contains_sections(self):
        corpus = default_corpus()
        result = run_experiment(corpus, chunkers=[FixedSizeChunker(200)])
        md = experiment_to_markdown(result)
        assert "Prompt Cache Chunking Experiment Report" in md
        assert "Strategy Comparison" in md
        assert "Per-Strategy Detail" in md
        assert "FixedSizeChunker" in md

    def test_json_roundtrip(self):
        corpus = default_corpus()
        result = run_experiment(corpus, chunkers=[FixedSizeChunker(200)])
        js = experiment_to_json(result)
        data = json.loads(js)
        assert data["num_documents"] == 10
        assert len(data["results"]) == 1
