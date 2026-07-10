"""
Persona router — classifies user input and selects the best LoRA adapter.

Two routing strategies:

1. **Keyword matching** — matches input text against adapter keywords
   (fast, no extra inference).
2. **LLM classification** — uses a lightweight model to classify the
   intent and pick the right persona (more accurate).

Usage::

    from adapters import AdapterRegistry, load_adapter_registry
    from persona_router import PersonaRouter

    registry = load_adapter_registry("adapters/registry.json")
    router = PersonaRouter(registry)

    # Quick keyword match
    persona = router.route_by_keywords("I need to review this contract")
    print(persona.name)  # -> "legal-tone"
"""

from __future__ import annotations

from typing import Dict, List, Optional

from adapters import AdapterDef, AdapterRegistry


class PersonaRouter:
    """
    Routes user input to the most appropriate LoRA adapter/persona.

    Supports keyword-based routing (fast, local) and LLM-based
    classification (more accurate, requires an inference call).
    """

    def __init__(self, registry: AdapterRegistry):
        """
        Args:
            registry: ``AdapterRegistry`` with all available personas.
        """
        self.registry = registry

    def route_by_keywords(self, text: str) -> AdapterDef:
        """
        Select an adapter by matching keywords in the input text.

        Falls back to the default adapter (or the first registered)
        if no keywords match.

        Args:
            text: User input text.

        Returns:
            The best-matching ``AdapterDef``.
        """
        matches = self.registry.find_by_keyword(text)
        if matches:
            return matches[0]

        default = self.registry.get_default()
        if default:
            return default

        # First registered adapter
        if self.registry:
            return next(iter(self.registry.values()))

        raise ValueError(
            "No adapters registered and no default persona configured."
        )

    def route_by_tags(self, tag: str) -> AdapterDef:
        """
        Select an adapter by tag name.

        Args:
            tag: Tag to match (e.g. ``"legal"``, ``"creative"``).

        Returns:
            The best-matching ``AdapterDef``.

        Raises:
            ValueError: If no adapter has the given tag.
        """
        matches = self.registry.find_by_tag(tag)
        if matches:
            return matches[0]

        raise ValueError(f"No adapter found with tag {tag!r}")

    def list_personas(self) -> List[Dict[str, str]]:
        """List all available personas with summary info."""
        return [
            {
                "name": a.name,
                "description": a.description,
                "tags": ", ".join(a.tags),
                "keywords": ", ".join(a.keywords),
                "scale": a.scale,
                "is_default": a.is_default,
            }
            for a in self.registry.values()
        ]
