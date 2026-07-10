"""
Test runner — formats a message sequence through multiple templates
and collects structured results.

Usage::

    from chat_template_tester.tester import run_test
    from chat_template_tester.templates import registry

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    results = run_test(messages, template_names=["llama3", "chatml", "mistral"])
    for r in results:
        print(f"[{r.template_name}] {r.char_count} chars, ~{r.estimated_tokens} tokens")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .formatter import (
    ChatFormatter,
    estimate_tokens,
    find_special_tokens,
)
from .templates import registry as default_registry, TemplateInfo, TemplateRegistry


@dataclass
class TemplateTestResult:
    """Result of formatting messages through one template."""

    template_name: str
    template_description: str
    formatted: str
    char_count: int
    estimated_tokens: int
    special_token_counts: Dict[str, int]
    error: Optional[str] = None


@dataclass
class TestRun:
    __test__ = False  # pytest: not a test class
    """Collection of results for a single message sequence."""

    messages: List[Dict[str, str]]
    results: List[TemplateTestResult]
    add_generation_prompt: bool = False

    def ok_results(self) -> List[TemplateTestResult]:
        """Return only results without errors."""
        return [r for r in self.results if r.error is None]

    def error_results(self) -> List[TemplateTestResult]:
        """Return only results with errors."""
        return [r for r in self.results if r.error is not None]

    @property
    def template_names(self) -> List[str]:
        """Names of all templates that were tested."""
        return [r.template_name for r in self.results]


def run_test(
    messages: List[Dict[str, str]],
    template_names: Optional[List[str]] = None,
    add_generation_prompt: bool = True,
    registry: Optional[TemplateRegistry] = None,
    custom_templates: Optional[List[TemplateInfo]] = None,
) -> TestRun:
    """Format a message sequence through one or more templates.

    Args:
        messages: Conversation turns as ``{"role": …, "content": …}``.
        template_names: Subset of templates to test.  ``None`` means all.
        add_generation_prompt: Append assistant turn starter.
        registry: Template registry to use (defaults to built-in).
        custom_templates: Additional templates to test (not in registry).

    Returns:
        A ``TestRun`` with one ``TemplateTestResult`` per template.
    """
    if registry is None:
        registry = default_registry

    # Determine which templates to use
    infos: List[TemplateInfo] = []
    seen_names: Set[str] = set()

    if template_names:
        for name in template_names:
            info = registry.get(name)
            if info.name not in seen_names:
                infos.append(info)
                seen_names.add(info.name)
    else:
        for info in registry.list():
            if info.name not in seen_names:
                infos.append(info)
                seen_names.add(info.name)

    if custom_templates:
        for info in custom_templates:
            if info.name not in seen_names:
                infos.append(info)
                seen_names.add(info.name)

    # Render
    formatter = ChatFormatter()
    results: List[TemplateTestResult] = []

    for info in infos:
        try:
            formatted = formatter.format(
                info, messages, add_generation_prompt=add_generation_prompt
            )
            token_counts = find_special_tokens(formatted, info.stop_tokens)
            results.append(TemplateTestResult(
                template_name=info.name,
                template_description=info.description,
                formatted=formatted,
                char_count=len(formatted),
                estimated_tokens=estimate_tokens(formatted),
                special_token_counts=token_counts,
                error=None,
            ))
        except ValueError as e:
            results.append(TemplateTestResult(
                template_name=info.name,
                template_description=info.description,
                formatted="",
                char_count=0,
                estimated_tokens=0,
                special_token_counts={},
                error=str(e),
            ))

    return TestRun(
        messages=messages,
        results=results,
        add_generation_prompt=add_generation_prompt,
    )
