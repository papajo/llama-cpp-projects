"""Tests for report generation."""

import json

import pytest

from embedding_norm_ablation.ablation import (
    AblationConfig,
    AblationResult,
    TreatmentResult,
)
from embedding_norm_ablation.data import mini_corpus
from embedding_norm_ablation.reporting import (
    report_to_json,
    report_to_markdown,
    write_report,
)


def _make_result(num_treatments: int = 2) -> AblationResult:
    corpus = mini_corpus()
    result = AblationResult(
        config=AblationConfig(corpus=corpus, top_k=3),
        corpus_size=len(corpus.documents),
        num_queries=len(corpus.queries),
        embedding_dim=4,
    )
    for i in range(num_treatments):
        result.treatments.append(
            TreatmentResult(
                label=f"treatment-{i}",
                per_query_metrics=[
                    {"p@1": 1.0, "p@5": 0.8, "r@5": 0.6, "ap": 0.9},
                    {"p@1": 0.0, "p@5": 0.4, "r@5": 0.3, "ap": 0.2},
                ],
                aggregated={
                    "mean_p@1": 0.5,
                    "mean_p@5": 0.6,
                    "mean_r@5": 0.45,
                    "mean_ap": 0.55,
                },
                doc_norm_stats={"mean": 1.0, "std": 0.1, "min": 0.8, "max": 1.2},
                query_norm_stats={"mean": 1.0, "std": 0.1, "min": 0.9, "max": 1.1},
            )
        )
    return result


class TestReportToMarkdown:
    def test_contains_sections(self):
        md = report_to_markdown(_make_result())
        assert "Embedding Norm Ablation Report" in md
        assert "Aggregated Metrics" in md
        assert "Per-Query Detail" in md
        assert "Document Norm Statistics" in md
        assert "Query Norm Statistics" in md

    def test_treatment_labels_in_output(self):
        md = report_to_markdown(_make_result(2))
        assert "treatment-0" in md
        assert "treatment-1" in md

    def test_no_data(self):
        result = _make_result(0)
        md = report_to_markdown(result)
        assert "_(no data)_" in md


class TestReportToJson:
    def test_roundtrip(self):
        result = _make_result()
        js = report_to_json(result)
        data = json.loads(js)
        assert data["corpus_size"] == 5
        assert data["embedding_dim"] == 4
        assert len(data["treatments"]) == 2

    def test_metric_values_preserved(self):
        result = _make_result(1)
        js = report_to_json(result)
        data = json.loads(js)
        assert data["treatments"][0]["aggregated"]["mean_p@1"] == 0.5


class TestWriteReport:
    def test_writes_markdown(self, tmp_path):
        result = _make_result()
        md_path = tmp_path / "report.md"
        write_report(result, markdown_path=str(md_path))
        assert md_path.read_text().startswith("# Embedding")

    def test_writes_json(self, tmp_path):
        result = _make_result()
        json_path = tmp_path / "report.json"
        write_report(result, json_path=str(json_path))
        data = json.loads(json_path.read_text())
        assert data["corpus_size"] == 5
