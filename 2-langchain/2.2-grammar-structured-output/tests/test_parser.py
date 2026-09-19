"""Tests for GrammarOutputParser (mocked HTTP)."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field


# We need to add the parent to sys.path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import GrammarOutputParser


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------


class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150)
    email: str | None = None


class Review(BaseModel):
    rating: int = Field(ge=1, le=5)
    summary: str
    verified: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_completion_response(content: str) -> httpx.Response:
    """Build a fake /completion response."""
    return httpx.Response(
        status_code=200,
        json={
            "content": content,
            "tokens_predicted": 50,
            "tokens_evaluated": 100,
            "truncated": False,
            "model": "test-model",
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_invoke_text_prompt(self):
        """Invoke with a plain-text prompt returns a valid Pydantic object."""
        parser = GrammarOutputParser(base_url="http://test:8080")
        with patch.object(parser, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_completion_response(
                json.dumps({"name": "Alice", "age": 30, "email": None})
            )
            mock_get.return_value = mock_client

            result = parser.invoke(
                prompt="Extract: Alice is 30 years old",
                schema=Person,
            )
            assert isinstance(result, Person)
            assert result.name == "Alice"
            assert result.age == 30
            assert result.email is None

    def test_invoke_message_prompt(self):
        """Invoke with a message list works too."""
        parser = GrammarOutputParser(base_url="http://test:8080")
        with patch.object(parser, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_completion_response(
                json.dumps({"name": "Bob", "age": 25})
            )
            mock_get.return_value = mock_client

            result = parser.invoke(
                prompt=[{"role": "user", "content": "Bob is 25"}],
                schema=Person,
            )
            assert result.name == "Bob"
            assert result.age == 25

    def test_clean_output_strips_code_fences(self):
        """Markdown code fences are stripped before JSON parsing."""
        parser = GrammarOutputParser(base_url="http://test:8080")
        with patch.object(parser, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_completion_response(
                "```json\n{\"rating\": 4, \"summary\": \"Great\"}\n```"
            )
            mock_get.return_value = mock_client

            result = parser.invoke(
                prompt="Review this: great product",
                schema=Review,
            )
            assert result.rating == 4
            assert result.summary == "Great"

    def test_compile_schema(self):
        """compile_schema returns a JSON schema from the Pydantic model."""
        parser = GrammarOutputParser(base_url="http://test:8080")
        schema = parser.compile_schema(Person)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_server_error(self):
        """HTTP errors are propagated."""
        parser = GrammarOutputParser(base_url="http://test:8080")
        with patch.object(parser, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = httpx.Response(
                status_code=503,
                json={"error": "Model loading"},
            )
            mock_get.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                parser.invoke(prompt="test", schema=Person)

    def test_parse_result_raises(self):
        """parse_result() is not supported (use invoke())."""
        parser = GrammarOutputParser(base_url="http://test:8080")
        with pytest.raises(NotImplementedError):
            parser.parse_result(["test"])


class TestChain:
    """Tests for GrammarStructuredOutput chain."""

    def test_chain_invoke(self):
        """GrammarStructuredOutput.invoke() formats and parses."""
        from parser import GrammarStructuredOutput

        chain = GrammarStructuredOutput(
            base_url="http://test:8080",
            schema=Person,
            prompt_template="Extract from: {input}",
        )
        with patch.object(chain._parser, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_completion_response(
                json.dumps({"name": "Charlie", "age": 35})
            )
            mock_get.return_value = mock_client

            result = chain.invoke({"input": "Charlie is 35"})
            assert isinstance(result, Person)
            assert result.name == "Charlie"
            assert result.age == 35

    def test_chain_with_system_prompt(self):
        """System prompt is prepended when configured."""
        from parser import GrammarStructuredOutput

        chain = GrammarStructuredOutput(
            base_url="http://test:8080",
            schema=Review,
            prompt_template="Product: {product}",
            system_prompt="Extract product reviews as JSON.",
        )
        assert chain.system_prompt == "Extract product reviews as JSON."


class TestRequestBody:
    """Regression tests: what actually goes over the wire.

    No offline test used to inspect the request body, so the fact that
    `json_schema` was being sent as a JSON *string* went unnoticed. A real
    llama-server rejects that with HTTP 400:
    "Field 'json_schema': ... schema must be an object".
    """

    def _captured_body(self, schema=None) -> dict:
        parser = GrammarOutputParser(base_url="http://test:8080")
        with patch.object(parser, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_completion_response(
                json.dumps({"name": "Alice", "age": 30, "email": None})
            )
            mock_get.return_value = mock_client
            parser.invoke(prompt="Extract: Alice is 30", schema=schema or Person)
            _, kwargs = mock_client.post.call_args
            return kwargs["json"]

    def test_json_schema_is_sent_as_an_object(self):
        body = self._captured_body()
        assert isinstance(body["json_schema"], dict), (
            "json_schema must be the schema object; llama-server returns 400 "
            "for a JSON string"
        )
        assert body["json_schema"]["type"] == "object"
        assert "name" in body["json_schema"]["properties"]

    def test_posts_to_the_completion_endpoint(self):
        parser = GrammarOutputParser(base_url="http://test:8080")
        with patch.object(parser, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_completion_response(
                json.dumps({"name": "Alice", "age": 30, "email": None})
            )
            mock_get.return_value = mock_client
            parser.invoke(prompt="Extract: Alice is 30", schema=Person)
            args, _ = mock_client.post.call_args
            assert args[0] == "/completion"

    def test_native_completion_param_names(self):
        """/completion takes n_predict, not max_tokens."""
        body = self._captured_body()
        assert "n_predict" in body
        assert "max_tokens" not in body
        assert body["cache_prompt"] is True
        assert body["temperature"] == 0.0
