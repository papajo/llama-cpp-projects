"""
LangChain ``Runnable`` that wraps a prompt template + llama.cpp grammar
constraint into a structured-output chain.

Usage::

    from pydantic import BaseModel, Field
    from parser import GrammarStructuredOutput

    class Person(BaseModel):
        name: str = Field(description="The person's full name")
        age: int = Field(description="Age in years")

    chain = GrammarStructuredOutput(
        base_url="http://127.0.0.1:8080",
        schema=Person,
        prompt_template="Extract info: {input}",
    )

    result = chain.invoke({"input": "John Smith is 30 years old"})
    # -> Person(name="John Smith", age=30)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, Optional, Type

from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, create_model

from .parser import GrammarOutputParser


class GrammarStructuredOutput(Runnable[Dict[str, Any], BaseModel]):
    """
    LangChain ``Runnable`` that applies a prompt template and sends the
    result to llama.cpp with a JSON-schema grammar constraint.

    Combines:
    1. Prompt template (string with ``{input}`` or other variables)
    2. JSON schema compilation from a Pydantic model
    3. llama.cpp ``/completion`` call with ``--json-schema``
    4. Parsed, validated Pydantic output — zero retries
    """

    def __init__(
        self,
        schema: Type[BaseModel],
        prompt_template: str = "{input}",
        base_url: str = "http://127.0.0.1:8080",
        request_timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        self.schema = schema
        self.prompt_template = prompt_template
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._parser = GrammarOutputParser(
            base_url=base_url,
            request_timeout=request_timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Run the chain: format prompt → constrain with grammar → parse.

        Args:
            input: Dict with keys matching ``prompt_template`` format
                variables (e.g. ``{"input": "..."}``).
            config: Optional LangChain ``RunnableConfig``.

        Returns:
            An instance of ``self.schema``.
        """
        prompt = self.prompt_template.format(**input)

        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        return self._parser.invoke(
            prompt=messages,
            schema=self.schema,
            config=config,
        )

    def _get_template_variables(self) -> list[str]:
        """Extract ``{var}`` placeholders from the template."""
        return re.findall(r"\{(\w+)\}", self.prompt_template)

    def close(self) -> None:
        if self._parser:
            self._parser.close()

    def __del__(self) -> None:
        self.close()
