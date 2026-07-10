"""
Chain definitions for prompt chaining.

Provides ``ChainStep`` (a single LLM call in a sequence) and
``PromptChain`` (the full pipeline).  Steps reference prior outputs
via template variables like ``{{step_1.output}}`` or ``{{input}}``.

Usage::

    from prompt_chaining.chain import PromptChain, ChainStep

    chain = PromptChain(
        name="my-chain",
        steps=[
            ChainStep(
                name="summarize",
                system_prompt="You are a summarizer.",
                user_prompt="Summarize: {{input}}",
            ),
            ChainStep(
                name="translate",
                system_prompt="You are a translator.",
                user_prompt="Translate to French: {{step_1.output}}",
            ),
        ],
    )
    print(chain.to_dict())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Regex to find template variables like {{input}}, {{step_1.output}}
_TEMPLATE_VAR_RE = re.compile(r"\{\{(\w+(?:\.\w+)?)\}\}")


def resolve_template(text: str, context: Dict[str, str]) -> str:
    """Replace ``{{var}}`` or ``{{step_N.field}}`` placeholders.

    Args:
        text: Template string with ``{{variable}}`` placeholders.
        context: Flat dict of available values (e.g. ``{"input": …,
            "step_1": …, "step_2": …}``).

    Returns:
        Text with all placeholders resolved.  Unknown variables are
        left as-is.
    """
    def _replacer(m: re.Match) -> str:
        key = m.group(1)
        if key in context:
            return context[key]
        # Support dotted keys: step_1.output → context["step_1"]
        if "." in key:
            base = key.split(".")[0]
            if base in context:
                return context[base]
        return m.group(0)  # leave unresolved
    return _TEMPLATE_VAR_RE.sub(_replacer, text)


@dataclass
class ChainStep:
    """A single step in a prompt chain.

    Each step is one LLM call.  The ``user_prompt`` (and optionally
    ``system_prompt``) can reference prior outputs with ``{{input}}``,
    ``{{step_1}}``, ``{{step_2}}``, etc.
    """

    name: str
    user_prompt: str
    system_prompt: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def render(self, context: Dict[str, str]) -> Dict[str, str]:
        """Resolve template variables and return the final messages."""
        system = resolve_template(self.system_prompt, context) if self.system_prompt else ""
        user = resolve_template(self.user_prompt, context)
        return {"system": system, "user": user}


@dataclass
class PromptChain:
    """A sequence of chained LLM calls.

    Attributes:
        name: Human-readable chain name.
        description: What this chain does.
        steps: Ordered list of ``ChainStep``.
        temperature: Default temperature for all steps (overridable per-step).
        max_tokens: Default max_tokens for all steps.
    """

    name: str
    steps: List[ChainStep]
    description: str = ""
    temperature: float = 0.7
    max_tokens: int = 512

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "name": self.name,
            "description": self.description,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "steps": [
                {
                    "name": s.name,
                    "user_prompt": s.user_prompt,
                    "system_prompt": s.system_prompt,
                    "temperature": s.temperature,
                    "max_tokens": s.max_tokens,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptChain:
        """Deserialise from a dict."""
        steps = [
            ChainStep(
                name=s["name"],
                user_prompt=s["user_prompt"],
                system_prompt=s.get("system_prompt", ""),
                temperature=s.get("temperature"),
                max_tokens=s.get("max_tokens"),
            )
            for s in data["steps"]
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 512),
            steps=steps,
        )

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> ChainStep:
        return self.steps[index]
