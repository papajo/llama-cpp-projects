"""
Comparator — side-by-side comparison of ablation results.

Produces markdown reports with metrics tables, per-config output
sections, and word-level diff highlighting against a reference.

Usage::

    from sampler_ablation.comparator import format_comparison

    report = format_comparison(results, title="Sampler Showdown")
    print(report)
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Optional, Set

from .runner import AblationResult


def format_comparison(
    results: List[AblationResult],
    title: str = "Sampler Ablation Comparison",
    reference_label: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> str:
    """
    Format ablation results as a markdown report.

    Includes:
    - Overview table with per-config metrics
    - Full output for each config
    - Word-level diff against a reference config

    Args:
        results: Results from ``run_ablation()``.
        title: Report title.
        reference_label: Label of the config to diff against.
            Defaults to the first config.
        notes: Optional notes at the bottom.

    Returns:
        Markdown string.
    """
    buf = io.StringIO()
    buf.write(f"# {title}\n\n")

    if not results:
        buf.write("*(No results to compare)*\n")
        return buf.getvalue()

    # Prompt
    prompt = results[0].prompt
    buf.write(f"**Prompt:** {prompt}\n\n")

    # Overview table
    ok = [r for r in results if r.is_ok]
    errors = [r for r in results if not r.is_ok]

    buf.write("## Overview\n\n")
    buf.write(
        "| Config | Words | Chars | Tokens | Speed | Vocab | Repetition | Entropy |\n"
    )
    buf.write(
        "|--------|-------|-------|--------|-------|-------|------------|---------|\n"
    )
    for r in ok:
        words = r.output_word_count
        chars = len(r.output)
        tokens = r.completion_tokens or 0
        speed = f"{tokens / r.elapsed_s:.1f} t/s" if r.elapsed_s > 0 else "—"
        vocab = _vocab_size(r.output)
        rep = f"{_repetition_rate(r.output):.3f}"
        ent = f"{_char_entropy(r.output):.2f}"
        buf.write(
            f"| {r.config.label} "
            f"| {words} "
            f"| {chars} "
            f"| {tokens} "
            f"| {speed} "
            f"| {vocab} "
            f"| {rep} "
            f"| {ent} "
            f"|\n"
        )
    for r in errors:
        buf.write(f"| {r.config.label} | *error: {r.error}* |\n")
    buf.write("\n")

    # Parameter details
    buf.write("## Sampler Configurations\n\n")
    for r in ok:
        buf.write(f"### {r.config.label}\n\n")
        buf.write(f"*{r.config.description}*\n\n")
        detail = r.config.param_summary()
        buf.write(f"```\n{detail}\n```\n\n")

    # Full outputs
    buf.write("## Outputs\n\n")
    for r in ok:
        buf.write(f"### {r.config.label}\n\n")
        buf.write(f"```\n{r.output}\n```\n\n")

    # Diff against reference
    if reference_label is None and ok:
        reference_label = ok[0].config.label

    if reference_label and len(ok) >= 2:
        reference = _find_by_label(ok, reference_label)
        if reference is not None:
            buf.write("## Word-Level Diffs\n\n")
            for r in ok:
                if r.config.label == reference_label:
                    continue
                buf.write(f"### {reference.config.label} → {r.config.label}\n\n")
                buf.write("```diff\n")
                buf.write(_word_diff(reference.output, r.output))
                buf.write("```\n\n")

    # Errors
    if errors:
        buf.write("## Errors\n\n")
        for r in errors:
            buf.write(f"- **{r.config.label}:** {r.error}\n")
        buf.write("\n")

    # Notes
    if notes:
        buf.write("---\n\n")
        buf.write("## Notes\n\n")
        for n in notes:
            buf.write(f"- {n}\n")
        buf.write("\n")

    return buf.getvalue()


# ── Metrics ──────────────────────────────────────────────────────


def _vocab_size(text: str) -> int:
    """Number of unique whitespace-separated tokens (case-sensitive)."""
    return len(set(text.split()))


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
    seen: Set[str] = set()
    repeats = 0
    for bg in bigrams:
        if bg in seen:
            repeats += 1
        seen.add(bg)
    return repeats / len(bigrams)


def _find_by_label(
    results: List[AblationResult], label: str
) -> Optional[AblationResult]:
    for r in results:
        if r.config.label == label:
            return r
    return None


# ── Diff ─────────────────────────────────────────────────────────


def _word_diff(text_a: str, text_b: str) -> str:
    """
    Simple word-level diff showing additions and removals.

    Lines prefixed with ``+ `` are in ``text_b`` but not ``text_a``.
    Lines prefixed with ``- `` are in ``text_a`` but not ``text_b``.

    This is a simplified diff based on word-set comparison, not a
    full Levenshtein or LCS-based diff.
    """
    words_a = set(text_a.split())
    words_b = set(text_b.split())

    removed = words_a - words_b
    added = words_b - words_a

    lines: List[str] = []

    if removed:
        lines.append(f"# {len(removed)} word(s) removed:")
        for w in sorted(removed)[:30]:  # cap display
            lines.append(f"- {w}")
        if len(removed) > 30:
            lines.append(f"- ... and {len(removed) - 30} more")

    if added:
        lines.append(f"# {len(added)} word(s) added:")
        for w in sorted(added)[:30]:
            lines.append(f"+ {w}")
        if len(added) > 30:
            lines.append(f"+ ... and {len(added) - 30} more")

    if not removed and not added:
        lines.append("# (identical word sets)")

    return "\n".join(lines) + "\n"
