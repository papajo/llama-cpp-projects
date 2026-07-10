"""
Jinja2-based chat template formatter.

Renders a list of messages through a ``TemplateInfo`` template, injecting
``bos_token``, ``eos_token``, and ``add_generation_prompt`` variables.

Usage::

    from chat_template_tester.templates import registry
    from chat_template_tester.formatter import ChatFormatter

    formatter = ChatFormatter()
    tmpl = registry.get("llama3")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    formatted = formatter.format(tmpl, messages, add_generation_prompt=True)
    print(formatted)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from jinja2 import Environment, StrictUndefined, TemplateError, UndefinedError
from jinja2.exceptions import TemplateSyntaxError

from .templates import TemplateInfo

# Regex to estimate token count (~4 chars per token for English text)
_CHARS_PER_TOKEN: float = 4.0


def _raise_exception(msg: str) -> None:
    """Raise a ValueError (used by templates that validate role alternation).

    This is made available as a Jinja2 global so templates like Llama 2
    and Mistral that call ``raise_exception(...)`` work correctly.
    """
    raise ValueError(msg)


# Shared Jinja2 environment with common globals
_JINJA_ENV = Environment(
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)
_JINJA_ENV.globals["raise_exception"] = _raise_exception


class ChatFormatter:
    """Renders messages through a chat template."""

    def __init__(self) -> None:
        self._env = _JINJA_ENV

    def format(
        self,
        template: TemplateInfo,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Render ``messages`` through the given template.

        Args:
            template: The template definition.
            messages: List of ``{"role": …, "content": …}`` dicts.
            add_generation_prompt: If ``True``, append the assistant
                turn starter (e.g. ``"<|im_start|>assistant\\n"``).

        Returns:
            The formatted prompt string.

        Raises:
            ValueError: If the template's role-alternation check fails.
            jinja2.TemplateError: On other Jinja2 errors.
        """
        try:
            jinja_template = self._env.from_string(template.template)
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error in {template.name!r}: {e}") from e

        try:
            rendered = jinja_template.render(
                messages=messages,
                bos_token=template.bos_token or "",
                eos_token=template.eos_token or "",
                add_generation_prompt=add_generation_prompt,
            )
        except ValueError:
            # Re-raise role alternation errors with a clearer message
            raise
        except UndefinedError as e:
            raise ValueError(
                f"Template {template.name!r} references undefined variable: {e}"
            ) from e
        except TemplateError as e:
            raise ValueError(
                f"Template rendering error for {template.name!r}: {e}"
            ) from e

        return rendered

    def format_all(
        self,
        templates: List[TemplateInfo],
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> Dict[str, str]:
        """Render messages through multiple templates.

        Returns:
            ``{template_name: formatted_string}``
        """
        results: Dict[str, str] = {}
        for tmpl in templates:
            try:
                results[tmpl.name] = self.format(
                    tmpl, messages, add_generation_prompt=add_generation_prompt
                )
            except ValueError as e:
                results[tmpl.name] = f"<ERROR: {e}>"
        return results


# ── Utility functions ──────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough token estimate (characters / 4, rounded up).

    This is a heuristic.  Real token counts depend on the model's
    tokenizer and the specific text.
    """
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def find_special_tokens(text: str, tokens: List[str]) -> Dict[str, int]:
    """Count occurrences of special tokens in ``text``.

    Returns:
        ``{token: count}``
    """
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = text.count(token)
    return counts


def strip_special_tokens(text: str, tokens: List[str]) -> str:
    """Remove all occurrences of ``tokens`` from ``text``."""
    result = text
    for token in tokens:
        result = result.replace(token, "")
    return result
