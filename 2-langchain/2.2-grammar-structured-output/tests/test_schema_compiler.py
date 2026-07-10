"""Tests for SchemaCompiler."""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field
import pytest

from schema_compiler import SchemaCompiler, json_schema_from_pydantic


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Address(BaseModel):
    street: str
    city: str
    zip_code: str = Field(pattern=r"\d{5}")


class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150)
    email: Optional[str] = None
    address: Address


class Review(BaseModel):
    sentiment: Sentiment
    score: int = Field(ge=1, le=5, description="Rating 1-5")
    summary: str = Field(min_length=1, max_length=200)
    keywords: List[str]
    reviewer_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompiler:
    def test_compile_basic(self):
        """Compile a model with required and optional fields."""
        compiler = SchemaCompiler()
        schema = compiler.compile(Person)
        assert schema["type"] == "object"
        assert set(schema["properties"].keys()) == {
            "name", "age", "email", "address"
        }
        assert "name" in schema["required"]
        assert "age" in schema["required"]
        assert "address" in schema["required"]
        assert "email" not in schema["required"]  # Optional[str]

    def test_field_constraints(self):
        """Numeric and string constraints are propagated."""
        compiler = SchemaCompiler()
        schema = compiler.compile(Review)
        score = schema["properties"]["score"]
        assert score["minimum"] == 1
        assert score["maximum"] == 5

        summary = schema["properties"]["summary"]
        assert summary["minLength"] == 1
        assert summary["maxLength"] == 200

    def test_enum(self):
        """Enum fields become string + enum values."""
        compiler = SchemaCompiler()
        schema = compiler.compile(Review)
        sentiment = schema["properties"]["sentiment"]
        assert sentiment["type"] == "string"
        assert set(sentiment["enum"]) == {"positive", "negative", "neutral"}

    def test_nested_model(self):
        """Nested Pydantic models produce nested JSON schemas."""
        compiler = SchemaCompiler()
        schema = compiler.compile(Person)
        address = schema["properties"]["address"]
        assert address["type"] == "object"
        assert "street" in address["properties"]
        assert "city" in address["properties"]
        assert "zip_code" in address["properties"]

    def test_list_type(self):
        """list[str] becomes array of strings."""
        compiler = SchemaCompiler()
        schema = compiler.compile(Review)
        keywords = schema["properties"]["keywords"]
        assert keywords["type"] == "array"
        assert keywords["items"]["type"] == "string"

    def test_optionals_not_required(self):
        """Optional[X] fields are excluded from the required list."""

        class WithOptional(BaseModel):
            a: str
            b: Optional[int] = None

        compiler = SchemaCompiler()
        schema = compiler.compile(WithOptional)
        assert schema["required"] == ["a"]

    def test_field_description(self):
        """Field descriptions appear in the schema."""

        class Described(BaseModel):
            name: str = Field(description="The person's name")

        compiler = SchemaCompiler()
        schema = compiler.compile(Described)
        assert schema["properties"]["name"]["description"] == "The person's name"

    def test_empty_model(self):
        """A model with no fields produces an empty object schema."""

        class Empty(BaseModel):
            pass

        compiler = SchemaCompiler()
        schema = compiler.compile(Empty)
        assert schema == {"type": "object", "properties": {}}

    def test_convenience_function(self):
        """json_schema_from_pydantic() works as a one-shot."""
        schema = json_schema_from_pydantic(Review)
        assert schema["type"] == "object"
        assert "sentiment" in schema["properties"]

    def test_compile_to_json(self):
        """compile_to_json returns a valid JSON string."""
        compiler = SchemaCompiler()
        json_str = compiler.compile_to_json(Review)
        import json
        parsed = json.loads(json_str)
        assert parsed["type"] == "object"

    def test_dict_type(self):
        """dict[str, X] produces additionalProperties schema."""

        class WithDict(BaseModel):
            metadata: Dict[str, str]

        compiler = SchemaCompiler()
        schema = compiler.compile(WithDict)
        props = schema["properties"]["metadata"]
        assert props["type"] == "object"

    def test_literal_enum(self):
        """Literal types become enums in the schema."""

        class WithLiteral(BaseModel):
            status: Literal["active", "inactive", "pending"]

        compiler = SchemaCompiler()
        schema = compiler.compile(WithLiteral)
        status = schema["properties"]["status"]
        assert status["type"] == "string"
        assert set(status["enum"]) == {"active", "inactive", "pending"}

    def test_pattern_constraint(self):
        """Regex pattern from Field(pattern=...) is included."""

        class WithPattern(BaseModel):
            code: str = Field(pattern=r"^[A-Z]{3}\d{4}$")

        compiler = SchemaCompiler()
        schema = compiler.compile(WithPattern)
        code = schema["properties"]["code"]
        assert code["pattern"] == r"^[A-Z]{3}\d{4}$"
