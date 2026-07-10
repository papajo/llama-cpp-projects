"""
BenchmarkAnalysis — statistical analysis of benchmark results.

Aggregates across multiple runs and strategies to produce
comparative insights and recommendations.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harness.metrics import BenchmarkMetrics


@dataclass
class AggregatedStrategy:
    """Aggregated results for one strategy across all prompts and runs."""
    label: str
    prompt_results: Dict[str, List[BenchmarkMetrics]] = field(default_factory=dict)

    @property
    def all_metrics(self) -> List[BenchmarkMetrics]:
        """Flatten all runs across all prompts."""
        return [
            m for prompt_results in self.prompt_results.values()
            for m in prompt_results
        ]

    def avg(self, attr: str) -> float:
        """Average of a metric attribute across all runs."""
        vals = [getattr(m, attr) for m in self.all_metrics]
        if not vals:
            return 0.0
        return statistics.mean(vals)

    def stdev(self, attr: str) -> float:
        """Standard deviation of a metric attribute."""
        vals = [getattr(m, attr) for m in self.all_metrics]
        if len(vals) < 2:
            return 0.0
        return statistics.stdev(vals)

    def by_prompt(self, attr: str) -> Dict[str, float]:
        """Average of a metric attribute per prompt."""
        result = {}
        for prompt_name, metrics_list in self.prompt_results.items():
            vals = [getattr(m, attr) for m in metrics_list]
            result[prompt_name] = statistics.mean(vals) if vals else 0.0
        return result

    def speedup_vs_baseline(self, baseline: "AggregatedStrategy") -> float:
        """Speedup factor vs baseline (tok/s)."""
        my_tok = self.avg("tokens_per_second")
        base_tok = baseline.avg("tokens_per_second")
        if base_tok == 0:
            return 0.0
        return my_tok / base_tok


@dataclass
class BenchmarkAnalysis:
    """
    Full analysis of a benchmark run, with aggregate statistics
    and strategy recommendations.
    """
    strategies: Dict[str, AggregatedStrategy] = field(default_factory=dict)
    baseline: Optional[AggregatedStrategy] = None
    prompts_used: List[str] = field(default_factory=list)

    @classmethod
    def from_results(cls, results: list) -> "BenchmarkAnalysis":
        """Build analysis from a list of BenchmarkResult objects."""
        analysis = cls()
        prompts_used = set()

        for result in results:
            label = result.strategy
            prompt_name = result.prompt_name
            metrics = result.metrics
            prompts_used.add(prompt_name)

            if label == "baseline":
                if analysis.baseline is None:
                    analysis.baseline = AggregatedStrategy(label="baseline")
                analysis.baseline.prompt_results.setdefault(prompt_name, []).append(metrics)
            else:
                if label not in analysis.strategies:
                    analysis.strategies[label] = AggregatedStrategy(label=label)
                analysis.strategies[label].prompt_results.setdefault(prompt_name, []).append(metrics)

        analysis.prompts_used = sorted(prompts_used)
        return analysis

    def summary_table(self) -> List[Dict]:
        """Generate a summary table as list of dicts."""
        rows = []

        # Baseline row
        if self.baseline:
            rows.append({
                "strategy": "baseline",
                "family": "-",
                "avg_tok_s": round(self.baseline.avg("tokens_per_second"), 2),
                "avg_accept_rate": 1.0,
                "avg_draft_proportion": 0.0,
                "avg_latency_p95_ms": round(self.baseline.avg("latency_p95_ms"), 2),
                "speedup": 1.0,
            })

        # Strategy rows
        baseline_tok = self.baseline.avg("tokens_per_second") if self.baseline else 1.0

        for label, strat in self.strategies.items():
            speedup = strat.avg("tokens_per_second") / baseline_tok if baseline_tok > 0 else 0.0
            rows.append({
                "strategy": label,
                "family": "draft-model" if any(
                    "draft" in label.lower()
                    for d in ["draft"]
                ) else "ngram",
                "avg_tok_s": round(strat.avg("tokens_per_second"), 2),
                "avg_accept_rate": round(strat.avg("acceptance_rate"), 4),
                "avg_draft_proportion": round(strat.avg("draft_proportion"), 4),
                "avg_latency_p95_ms": round(strat.avg("latency_p95_ms"), 2),
                "speedup": round(speedup, 3),
            })

        return rows

    def best_strategy_for_workload(self, workload: str) -> Optional[str]:
        """Recommend the best strategy for a given workload type."""
        best_label = None
        best_tok_s = 0.0

        for label, strat in self.strategies.items():
            # Get tok/s for this specific workload
            if workload in strat.prompt_results:
                vals = [m.tokens_per_second for m in strat.prompt_results[workload]]
                avg_tok = statistics.mean(vals) if vals else 0.0
                if avg_tok > best_tok_s:
                    best_tok_s = avg_tok
                    best_label = label

        return best_label

    def recommendations(self) -> List[str]:
        """Generate human-readable recommendations."""
        recs = []

        if not self.strategies:
            return ["No strategy data to analyse."]

        # Find overall fastest
        summary = self.summary_table()
        non_baseline = [r for r in summary if r["strategy"] != "baseline"]
        if non_baseline:
            fastest = max(non_baseline, key=lambda r: r["avg_tok_s"])
            recs.append(
                f"🏆 Fastest overall: {fastest['strategy']} "
                f"({fastest['avg_tok_s']} tok/s, "
                f"{fastest['speedup']}× baseline)"
            )

        # Find best for each workload
        for prompt in self.prompts_used:
            best = self.best_strategy_for_workload(prompt)
            if best:
                strat = self.strategies[best]
                tok_s = strat.avg("tokens_per_second")
                recs.append(
                    f"  For '{prompt}': best is {best} ({tok_s:.1f} tok/s)"
                )

        # Draft-model vs n-gram comparison
        draft_strats = {k: v for k, v in self.strategies.items()
                        if any(d in k.lower() for d in ["draft"])}
        ngram_strats = {k: v for k, v in self.strategies.items()
                        if any(n in k.lower() for n in ["ngram"])}

        if draft_strats and ngram_strats:
            best_draft = max(draft_strats.values(),
                             key=lambda s: s.avg("tokens_per_second"))
            best_ngram = max(ngram_strats.values(),
                             key=lambda s: s.avg("tokens_per_second"))
            draft_tok = best_draft.avg("tokens_per_second")
            ngram_tok = best_ngram.avg("tokens_per_second")

            if draft_tok > ngram_tok:
                recs.append(
                    f"💡 Draft-model strategies outperformed n-gram "
                    f"({best_draft.label}: {draft_tok:.1f} vs "
                    f"{best_ngram.label}: {ngram_tok:.1f} tok/s)"
                )
            else:
                recs.append(
                    f"💡 N-gram strategies outperformed draft-model "
                    f"({best_ngram.label}: {ngram_tok:.1f} vs "
                    f"{best_draft.label}: {draft_tok:.1f} tok/s — "
                    f"your workload may be highly repetitive)"
                )

        return recs

    def to_json(self, path: Optional[Path] = None) -> str:
        """Export analysis to JSON."""
        data = {
            "summary": self.summary_table(),
            "recommendations": self.recommendations(),
            "prompts": self.prompts_used,
            "strategies_tested": list(self.strategies.keys()),
        }
        text = json.dumps(data, indent=2)
        if path:
            path.write_text(text)
        return text
