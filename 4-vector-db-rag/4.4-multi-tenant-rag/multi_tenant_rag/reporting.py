"""Reporting for multi-tenant RAG evaluation."""

from __future__ import annotations

import json
from typing import Dict

from multi_tenant_rag.pipeline import MultiTenantEvalResult


def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def evaluation_to_markdown(
    isolated_result: MultiTenantEvalResult,
    leakage_result: MultiTenantEvalResult | None = None,
) -> str:
    """Generate a markdown report comparing isolated vs non-isolated retrieval."""
    lines = []
    lines.append("# Multi-Tenant RAG Evaluation Report")
    lines.append("")
    lines.append(f"- **Tenants:** {isolated_result.num_tenants}")
    lines.append("")

    # Isolated results table
    lines.append("## Isolated Retrieval (Tenant-Aware)")
    lines.append("")
    lines.append("| Tenant | Recall | Precision | Leakage | Retrieved |")
    lines.append("|---|---|---|---|---|")
    for r in isolated_result.tenant_results:
        lines.append(
            f"| {r.tenant_id} | {_fmt(r.isolated_recall)} | "
            f"{_fmt(r.isolated_precision)} | {r.leakage_count} | "
            f"{r.total_retrieved} |"
        )
    lines.append("")
    lines.append(
        f"**Average recall:** {_fmt(isolated_result.avg_isolated_recall)}  |  "
        f"**Total leakage:** {isolated_result.total_leakage}"
    )
    lines.append("")

    # Cross-tenant leakage results (if provided)
    if leakage_result is not None:
        lines.append("## Cross-Tenant Retrieval (No Isolation — Buggy)")
        lines.append("")
        lines.append("| Tenant | Recall | Precision | Leakage | Retrieved |")
        lines.append("|---|---|---|---|---|")
        for r in leakage_result.tenant_results:
            lines.append(
                f"| {r.tenant_id} | {_fmt(r.isolated_recall)} | "
                f"{_fmt(r.isolated_precision)} | {r.leakage_count} | "
                f"{r.total_retrieved} |"
            )
        lines.append("")
        lines.append(
            f"**Average recall:** {_fmt(leakage_result.avg_isolated_recall)}  |  "
            f"**Total leakage:** {leakage_result.total_leakage}"
        )
        lines.append("")

        # Comparison
        lines.append("## Isolated vs Non-Isolated Comparison")
        lines.append("")
        lines.append("| Metric | Isolated | Non-Isolated | Delta |")
        lines.append("|---|---|---|---|")
        iso_recall = isolated_result.avg_isolated_recall
        leak_recall = leakage_result.avg_isolated_recall
        iso_leak = isolated_result.total_leakage
        leak_leak = leakage_result.total_leakage
        lines.append(
            f"| Recall | {_fmt(iso_recall)} | {_fmt(leak_recall)} | "
            f"{_fmt(iso_recall - leak_recall, decimals=4)} |"
        )
        lines.append(
            f"| Leakage | {iso_leak} | {leak_leak} | "
            f"{iso_leak - leak_leak:+d} |"
        )
        lines.append("")

    return "\n".join(lines)


def evaluation_to_json(
    isolated_result: MultiTenantEvalResult,
    leakage_result: MultiTenantEvalResult | None = None,
) -> str:
    """Serialize evaluation results to JSON."""
    data: Dict = {}
    data["isolated"] = {
        "num_tenants": isolated_result.num_tenants,
        "avg_recall": isolated_result.avg_isolated_recall,
        "total_leakage": isolated_result.total_leakage,
        "per_tenant": [
            {
                "tenant_id": r.tenant_id,
                "recall": r.isolated_recall,
                "precision": r.isolated_precision,
                "leakage": r.leakage_count,
                "retrieved": r.total_retrieved,
            }
            for r in isolated_result.tenant_results
        ],
    }
    if leakage_result is not None:
        data["non_isolated"] = {
            "num_tenants": leakage_result.num_tenants,
            "avg_recall": leakage_result.avg_isolated_recall,
            "total_leakage": leakage_result.total_leakage,
            "per_tenant": [
                {
                    "tenant_id": r.tenant_id,
                    "recall": r.isolated_recall,
                    "precision": r.isolated_precision,
                    "leakage": r.leakage_count,
                    "retrieved": r.total_retrieved,
                }
                for r in leakage_result.tenant_results
            ],
        }
    return json.dumps(data, indent=2)
