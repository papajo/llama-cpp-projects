"""
Comparison report generator — formats ``TestRun`` results as a markdown
report with summary table, raw formatted output sections, and
template-by-template breakdown.

Usage::

    from chat_template_tester.comparator import format_comparison
    from chat_template_tester.tester import run_test

    run = run_test(messages, template_names=["llama3", "chatml", "gemma"])
    report = format_comparison(run)
    print(report)
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional

from .formatter import find_special_tokens, strip_special_tokens
from .tester import TemplateTestResult, TestRun


def format_comparison(
    run: TestRun,
    title: str = "Chat Template Comparison",
    show_special: bool = True,
    show_raw: bool = True,
    notes: Optional[List[str]] = None,
) -> str:
    """Generate a markdown report comparing template outputs.

    Args:
        run: The ``TestRun`` from ``run_test()``.
        title: Report title.
        show_special: Include the special-tokens breakdown section.
        show_raw: Show raw (with special tokens) and clean versions.
        notes: Optional notes at the bottom.

    Returns:
        Markdown string.
    """
    buf = io.StringIO()
    buf.write(f"# {title}\n\n")

    if not run.results:
        buf.write("*(No results to compare)*\n")
        return buf.getvalue()

    # ── Messages ──────────────────────────────────────────────
    buf.write("## Input Messages\n\n")
    for i, msg in enumerate(run.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        buf.write(f"### [{i + 1}] {role}\n\n")
        buf.write(f"```\n{content}\n```\n\n")

    # ── Overview table ────────────────────────────────────────
    buf.write("## Summary\n\n")
    buf.write("| Template | Chars | Est. Tokens | Status |\n")
    buf.write("|----------|-------|-------------|--------|\n")

    for r in run.results:
        if r.error:
            status = f"❌ {r.error}"
            chars = "—"
            tokens = "—"
        else:
            status = "✅"
            chars = str(r.char_count)
            tokens = str(r.estimated_tokens)
        buf.write(f"| {r.template_name} | {chars} | {tokens} | {status} |\n")

    buf.write("\n")

    # ── Per-template outputs ──────────────────────────────────
    buf.write("## Formatted Outputs\n\n")

    for r in run.results:
        buf.write(f"### {r.template_name}\n\n")
        buf.write(f"*{r.template_description}*\n\n")

        if r.error:
            buf.write(f"```\nERROR: {r.error}\n```\n\n")
            continue

        if show_raw:
            buf.write("#### Raw\n\n")
            buf.write("```\n")
            buf.write(r.formatted)
            buf.write("\n```\n\n")

        # Clean version
        stop_tokens = _get_stop_tokens(r.template_name)
        cleaned = strip_special_tokens(r.formatted, stop_tokens)
        # Also strip bos/eos tokens by common names
        cleaned = _clean_common_special(cleaned)

        if show_raw and cleaned != r.formatted:
            buf.write("#### Clean\n\n")
            buf.write("```\n")
            buf.write(cleaned)
            buf.write("\n```\n\n")

        # Metrics
        buf.write("#### Metrics\n\n")
        buf.write(f"- Characters: **{r.char_count}**\n")
        buf.write(f"- Estimated tokens: **{r.estimated_tokens}**\n")
        if r.special_token_counts:
            buf.write(f"- Special tokens found: `{r.special_token_counts}`\n")
        buf.write("\n")

    # ── Comparison notes ──────────────────────────────────────
    buf.write("## Structural Comparison\n\n")
    ok_results = [r for r in run.results if r.error is None]
    if len(ok_results) >= 2:
        # Show which templates add BOS by checking first char
        bos_templates = []
        no_bos_templates = []
        for r in ok_results:
            first_word = r.formatted[:20] if r.formatted else ""
            if any(
                first_word.startswith(tok)
                for tok in ("<s>", "<bos>", "<|begin", "<BOS", "｜begin")
            ):
                bos_templates.append(r.template_name)
            else:
                no_bos_templates.append(r.template_name)

        if bos_templates:
            buf.write(f"**Adds BOS token:** {', '.join(bos_templates)}\n\n")
        if no_bos_templates:
            buf.write(f"**No BOS token:** {', '.join(no_bos_templates)}\n\n")

        # Role markers used
        buf.write("### Role Markers\n\n")
        for r in ok_results:
            markers = _detect_role_markers(r.formatted)
            buf.write(f"- **{r.template_name}:** `{', '.join(markers)}`\n")
        buf.write("\n")
    else:
        buf.write("*(Need at least 2 successful results for comparison)*\n\n")

    # ── Errors ────────────────────────────────────────────────
    errors = run.error_results()
    if errors:
        buf.write("## Errors\n\n")
        for r in errors:
            buf.write(f"- **{r.template_name}:** {r.error}\n")
        buf.write("\n")

    # ── Notes ─────────────────────────────────────────────────
    if notes:
        buf.write("---\n\n")
        buf.write("## Notes\n\n")
        for n in notes:
            buf.write(f"- {n}\n")
        buf.write("\n")

    return buf.getvalue()


# ── Helpers ────────────────────────────────────────────────────────

# Known stop tokens by template name
_KNOWN_STOP_TOKENS: Dict[str, List[str]] = {
    "llama3": ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
    "llama2": ["</s>"],
    "chatml": ["<|im_end|>", "<|im_start|>"],
    "mistral": ["</s>"],
    "vicuna": ["</s>"],
    "gemma": ["<eos>", "<end_of_turn>", "<start_of_turn>"],
    "phi3": ["<|end|>", "<|user|>", "<|assistant|>"],
    "deepseek": ["<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"],
    "command-r": ["<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>",
                   "<|SYSTEM_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>",
                   "<BOS_TOKEN>"],
    "qwen2.5": ["<|im_end|>", "<|im_start|>"],
}

_COMMON_SPECIAL = [
    "<s>", "</s>",
    "<bos>", "<eos>",
    "<BOS_TOKEN>",
]


def _get_stop_tokens(name: str) -> List[str]:
    return _KNOWN_STOP_TOKENS.get(name, [])


def _clean_common_special(text: str) -> str:
    result = text
    for tok in _COMMON_SPECIAL:
        result = result.replace(tok, "")
    return result


def _detect_role_markers(text: str) -> List[str]:
    """Detect which role markers appear in the formatted text."""
    markers = []
    patterns = [
        ("<|start_header_id|>",),           # Llama 3
        ("<|im_start|>",),                  # ChatML / Qwen
        ("[INST]",),                        # Llama 2 / Mistral
        ("USER:",),                         # Vicuna
        ("<start_of_turn>",),               # Gemma
        ("<|user|>",),                      # Phi-3
        ("User:",),                         # DeepSeek
        ("<|START_OF_TURN_TOKEN|>",),       # Command R
        ("<|SYSTEM_TOKEN|>",),              # Command R
        ("<BOS_TOKEN>",),                   # Command R
    ]
    for pattern in patterns:
        if pattern[0] in text:
            markers.append(pattern[0])
    return markers
