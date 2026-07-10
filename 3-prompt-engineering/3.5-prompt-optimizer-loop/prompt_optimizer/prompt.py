"""
prompt.py — PromptTemplate and TestCase definitions.

A PromptTemplate is a text string with ``{{variable}}`` placeholders.
A TestCase pairs an input dict with an optional expected output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


class PromptTemplateError(Exception):
    """Raised when prompt template operations fail (missing vars, etc.)."""


@dataclass(frozen=True)
class TestCase:
    """A single test case for prompt evaluation.

    Attributes:
        input_vars: Variables to substitute into the template.
        expected: Optional reference output for scoring. If ``None``,
                  only rubric (LLM-as-judge) evaluation can score it.
        metadata: Optional free-form metadata (e.g. category, difficulty).
    """

    input_vars: dict[str, str] = field(default_factory=dict)
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """A prompt template with ``{{variable}}`` placeholders.

    Examples:
        >>> pt = PromptTemplate("Translate to French: {{text}}")
        >>> pt.render({"text": "Hello"})
        'Translate to French: Hello'
    """

    template: str
    variables: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", self._extract_variables())

    def _extract_variables(self) -> frozenset[str]:
        return frozenset(_VAR_RE.findall(self.template))

    def render(self, variables: dict[str, str]) -> str:
        """Substitute variables into the template.

        Raises:
            PromptTemplateError: If a required variable is missing.
        """
        missing = self.variables - set(variables)
        if missing:
            raise PromptTemplateError(
                f"Missing template variables: {', '.join(sorted(missing))}"
            )

        def _replacer(m: re.Match[str]) -> str:
            return variables[m.group(1)]

        return _VAR_RE.sub(_replacer, self.template)

    def has_variable(self, name: str) -> bool:
        """Check if the template uses the given variable name."""
        return name in self.variables

    @classmethod
    def from_file(cls, path: str) -> PromptTemplate:
        """Load a template from a text file."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def __str__(self) -> str:
        return self.template
