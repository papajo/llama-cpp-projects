"""
LangChain output parser that uses llama.cpp's ``--json-schema`` grammar
constraint to produce guaranteed schema-conformant output — zero retries.

Unlike ``with_structured_output`` (which relies on tool-calling fine-tunes
and prompt-based retries), this parser sends the JSON schema as a
per-request grammar constraint to the server-level ``/completion``
endpoint.  The sampler **cannot** produce tokens outside the grammar,
so the output is always valid.

Usage::

    from pydantic import BaseModel
    from parser import GrammarOutputParser

    class Person(BaseModel):
        name: str
        age: int

    parser = GrammarOutputParser(base_url="http://127.0.0.1:8080")
    person = parser.invoke("John is 30 years old", schema=Person)
    # -> Person(name="John", age=30)  — guaranteed, no retries
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import httpx
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

from schema_compiler import SchemaCompiler


class GrammarOutputParser(BaseOutputParser[BaseModel]):
    """
    Output parser that constrains generation via llama.cpp's ``-j``
    JSON-schema grammar parameter.

    The generated text is parsed into the target Pydantic model.
    Because the grammar enforces the schema at the sampler level,
    parsing **never** fails on syntax — no retries needed.
    """

    base_url: str = "http://127.0.0.1:8080"
    """llama.cpp server URL (the ``/completion`` endpoint)."""

    request_timeout: float = 120.0
    """HTTP request timeout for generation."""

    temperature: float = 0.0
    """Sampling temperature (0 = greedy, best for structured output)."""

    max_tokens: int = 4096
    """Max tokens for the completion."""

    _compiler: SchemaCompiler = None
    _client: Optional[httpx.Client] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._compiler = SchemaCompiler()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.request_timeout,
            )
        return self._client

    def compile_schema(self, model: Type[BaseModel]) -> Dict[str, Any]:
        """Compile a Pydantic model to JSON Schema (for inspection)."""
        return self._compiler.compile(model)

    def invoke(
        self,
        prompt: str | List[Dict[str, str]],
        schema: Type[BaseModel],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Generate a structured output constrained by the given Pydantic schema.

        Args:
            prompt: A plain-text prompt or list of message dicts
                (``[{"role": "user", "content": "..."}]``).
            schema: The Pydantic model class to constrain output.
            config: Optional LangChain ``RunnableConfig``.
            **kwargs: Overrides for temperature, max_tokens, etc.

        Returns:
            An instance of ``schema`` with the parsed output.

        Raises:
            httpx.HTTPError: If the server is unreachable.
            ValidationError: If the output doesn't match the schema
                (should not happen with grammar constraints, but guards
                against server bugs).
        """
        client = self._get_client()
        json_schema = self._compiler.compile(schema)

        # Build the prompt
        if isinstance(prompt, str):
            request_prompt = prompt
        else:
            # Convert message list to a prompt string
            request_prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in prompt
            )

        body: Dict[str, Any] = {
            "prompt": request_prompt,
            # Must be the schema OBJECT, not a JSON string. llama-server
            # rejects a string with HTTP 400 "Field 'json_schema': ... schema
            # must be an object".
            "json_schema": json_schema,
            "temperature": kwargs.get("temperature", self.temperature),
            "n_predict": kwargs.get("max_tokens", self.max_tokens),
            "cache_prompt": True,
        }

        response = client.post("/completion", json=body)
        if response.status_code >= 400:
            # httpx 0.28+: request property raises if _request is not set
            try:
                req = response.request
            except RuntimeError:
                req = None
            raise httpx.HTTPStatusError(
                f"llama.cpp server returned {response.status_code}",
                request=req,
                response=response,
            )
        data = response.json()

        raw_output = data.get("content", "")
        # Strip any markdown code fences the model might add
        raw_output = self._clean_output(raw_output)

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            # This should never happen with grammar constraints, but
            # guard against edge cases
            raise ValueError(
                f"Grammar-constrained output was not valid JSON. "
                f"Output: {raw_output[:500]!r}"
            ) from e

        return schema.model_validate(parsed)

    def parse(self, text: str) -> BaseModel:
        """Parse plain text (required by BaseOutputParser).

        Note: For grammar-constrained parsing, use :meth:`invoke` instead.
        This method raises if called directly.
        """
        raise NotImplementedError(
            "GrammarOutputParser uses invoke(text, schema) for "
            "grammar-constrained generation. Call invoke() instead."
        )

    def parse_result(
        self, result: List[str], *, partial: bool = False
    ) -> BaseModel:
        # Not used for grammar-based parsing; use invoke() instead.
        raise NotImplementedError(
            "GrammarOutputParser uses invoke() for grammar-constrained "
            "generation, not parse_result()."
        )

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip markdown code fences from generated output."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @property
    def _type(self) -> str:
        return "llamacpp_grammar_output_parser"

    def close(self) -> None:
        if self._client:
            self._client.close()

    def __del__(self) -> None:
        self.close()
