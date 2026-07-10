"""
Metrics for reasoning budget sweep results.

Computes output length, repetition, diversity, and token efficiency
from per-run data collected during a sweep.

Usage::

    from budget_sweep.metrics import SweepResult, compute_metrics

    results = [SweepResult(...), ...]
    metrics = compute_metrics(results)
    print(metrics.token_efficiency)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SweepResult:
    """
    A single API call result from the sweep.

    Args:
        params: The parameter dict used for this call.
        prompt: The input prompt string.
        run_id: Which repeat run (0-indexed).
        output: The generated text.
        prompt_tokens: Token count from server response (or None).
        completion_tokens: Token count from server response (or None).
        elapsed_s: Wall-clock time for this request.
        error: Error message if the request failed.
    """

    params: Dict[str, Any]
    prompt: str
    run_id: int
    output: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @property
    def output_words(self) -> List[str]:
        return self.output.split()

    @property
    def output_word_count(self) -> int:
        return len(self.output_words)

    @property
    def output_char_count(self) -> int:
        return len(self.output)


@dataclass
class PerPromptMetrics:
    """Aggregated metrics for one prompt under one config."""

    params: Dict[str, Any]
    prompt: str
    n_runs: int          # total attempts (including errors)
    ok_count: int        # successful runs only
    avg_word_count: float
    std_word_count: float
    avg_char_count: float
    avg_completion_tokens: float
    avg_speed_tok_s: float
    pairwise_diversity: float          # mean pairwise Jaccard distance
    entropy: float                     # token-type entropy
    repetition_rate: float             # fraction of repeated bigrams
    error_count: int


@dataclass
class ConfigSummary:
    """Aggregated summary across all prompts for one config."""

    params: Dict[str, Any]
    n_prompts: int
    total_runs: int
    errors: int
    avg_word_count: float
    avg_completion_tokens: float
    avg_speed_tok_s: float
    avg_diversity: float
    avg_entropy: float
    avg_repetition_rate: float
    token_efficiency: float            # avg_word_count / avg_completion_tokens
    time_efficiency: float             # avg_word_count / avg_speed_tok_s


# ── Compute ──────────────────────────────────────────────────────


def compute_metrics(results: List[SweepResult]) -> List[ConfigSummary]:
    """
    Group results by (param_config, prompt), compute per-prompt metrics,
    then roll up per-config summaries.
    """
    # Group by serialised params
    groups: Dict[str, List[SweepResult]] = {}
    for r in results:
        key = _params_key(r.params)
        groups.setdefault(key, []).append(r)

    summaries: List[ConfigSummary] = []
    for key, group in groups.items():
        params = group[0].params

        # Sub-group by prompt
        prompt_groups: Dict[str, List[SweepResult]] = {}
        for r in group:
            prompt_groups.setdefault(r.prompt, []).append(r)

        per_prompt: List[PerPromptMetrics] = []
        for prompt, rg in prompt_groups.items():
            per_prompt.append(_per_prompt(prompt, params, rg))

        # Roll up
        n_prompts = len(per_prompt)
        total_runs = sum(pp.n_runs for pp in per_prompt)
        errors = sum(pp.error_count for pp in per_prompt)
        avg_wc = _avg(pp.avg_word_count for pp in per_prompt)
        avg_ct = _avg(pp.avg_completion_tokens for pp in per_prompt)
        avg_sp = _avg(pp.avg_speed_tok_s for pp in per_prompt) if any(
            pp.avg_speed_tok_s > 0 for pp in per_prompt
        ) else 0.0
        avg_div = _avg(pp.pairwise_diversity for pp in per_prompt)
        avg_ent = _avg(pp.entropy for pp in per_prompt)
        avg_rep = _avg(pp.repetition_rate for pp in per_prompt)

        token_eff = avg_wc / avg_ct if avg_ct > 0 else 0.0
        time_eff = avg_wc / avg_sp if avg_sp > 0 else 0.0

        summaries.append(ConfigSummary(
            params=params,
            n_prompts=n_prompts,
            total_runs=total_runs,
            errors=errors,
            avg_word_count=avg_wc,
            avg_completion_tokens=avg_ct,
            avg_speed_tok_s=avg_sp,
            avg_diversity=avg_div,
            avg_entropy=avg_ent,
            avg_repetition_rate=avg_rep,
            token_efficiency=token_eff,
            time_efficiency=time_eff,
        ))

    return summaries


# ── Internal ─────────────────────────────────────────────────────


def _per_prompt(
    prompt: str,
    params: Dict[str, Any],
    results: List[SweepResult],
) -> PerPromptMetrics:
    """Compute metrics for runs of the same prompt under the same config."""
    ok = [r for r in results if r.is_ok]
    n = len(ok)
    n_total = len(results)
    errors = n_total - n

    if n == 0:
        return PerPromptMetrics(
            params=params, prompt=prompt, n_runs=n_total, ok_count=0,
            avg_word_count=0, std_word_count=0, avg_char_count=0,
            avg_completion_tokens=0, avg_speed_tok_s=0,
            pairwise_diversity=0, entropy=0, repetition_rate=0,
            error_count=errors,
        )

    word_counts = [r.output_word_count for r in ok]
    char_counts = [r.output_char_count for r in ok]
    comp_tokens = [r.completion_tokens or 0 for r in ok]
    speeds = []
    for r in ok:
        ct = r.completion_tokens or 0
        sp = ct / r.elapsed_s if r.elapsed_s > 0 else 0
        speeds.append(sp)

    avg_wc = sum(word_counts) / n
    std_wc = _std(word_counts)
    avg_cc = sum(char_counts) / n
    avg_ct = sum(comp_tokens) / n
    avg_sp = sum(speeds) / n if speeds else 0

    # Pairwise diversity (Jaccard distance between word sets)
    divs: List[float] = []
    for i in range(n):
        set_i = set(ok[i].output_words)
        for j in range(i + 1, n):
            set_j = set(ok[j].output_words)
            intersection = set_i & set_j
            union = set_i | set_j
            if union:
                divs.append(1.0 - len(intersection) / len(union))
    pairwise_div = sum(divs) / len(divs) if divs else 0.0

    # Entropy (character-level)
    all_text = " ".join(r.output for r in ok)
    entropy = _char_entropy(all_text)

    # Repetition rate (bigram-level)
    rep_rate = _repetition_rate(ok[0].output)

    return PerPromptMetrics(
        params=params, prompt=prompt, n_runs=n_total, ok_count=n,
        avg_word_count=avg_wc, std_word_count=std_wc,
        avg_char_count=avg_cc,
        avg_completion_tokens=avg_ct, avg_speed_tok_s=avg_sp,
        pairwise_diversity=pairwise_div,
        entropy=entropy, repetition_rate=rep_rate,
        error_count=errors,
    )


def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution."""
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _repetition_rate(text: str) -> float:
    """Fraction of overlapping bigrams that are repeats."""
    words = text.split()
    if len(words) < 3:
        return 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if not bigrams:
        return 0.0
    seen: set = set()
    repeats = 0
    total = 0
    for bg in bigrams:
        total += 1
        if bg in seen:
            repeats += 1
        seen.add(bg)
    return repeats / total


def _params_key(params: Dict[str, Any]) -> str:
    """Stable string key for a params dict."""
    return str(sorted(params.items()))


def _avg(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _std(values):
    vals = list(values)
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(variance)
