"""Tests for reporting."""

import json

from multi_tenant_rag.pipeline import MultiTenantEvalResult, TenantEvalResult
from multi_tenant_rag.reporting import evaluation_to_json, evaluation_to_markdown


def _make_result():
    return MultiTenantEvalResult(
        num_tenants=2,
        tenant_results=[
            TenantEvalResult(
                tenant_id="acme",
                isolated_recall=0.85,
                isolated_precision=0.80,
                leakage_count=0,
                total_retrieved=15,
            ),
            TenantEvalResult(
                tenant_id="globebank",
                isolated_recall=0.75,
                isolated_precision=0.70,
                leakage_count=0,
                total_retrieved=15,
            ),
        ],
    )


def _make_leakage_result():
    return MultiTenantEvalResult(
        num_tenants=2,
        tenant_results=[
            TenantEvalResult(
                tenant_id="acme",
                isolated_recall=0.60,
                isolated_precision=0.40,
                leakage_count=5,
                total_retrieved=15,
            ),
            TenantEvalResult(
                tenant_id="globebank",
                isolated_recall=0.55,
                isolated_precision=0.35,
                leakage_count=6,
                total_retrieved=15,
            ),
        ],
    )


class TestReporting:
    def test_markdown_isolated(self):
        md = evaluation_to_markdown(_make_result())
        assert "Multi-Tenant RAG Evaluation Report" in md
        assert "Isolated Retrieval" in md
        assert "acme" in md
        assert "globebank" in md

    def test_markdown_with_leakage_comparison(self):
        md = evaluation_to_markdown(_make_result(), _make_leakage_result())
        assert "Cross-Tenant Retrieval" in md
        assert "Isolated vs Non-Isolated" in md

    def test_json_roundtrip(self):
        js = evaluation_to_json(_make_result(), _make_leakage_result())
        data = json.loads(js)
        assert "isolated" in data
        assert "non_isolated" in data
        assert data["isolated"]["num_tenants"] == 2
