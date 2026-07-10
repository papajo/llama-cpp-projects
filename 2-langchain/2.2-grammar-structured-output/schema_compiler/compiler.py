"""
Schema compiler — converts Pydantic models into JSON Schema for
llama.cpp's ``-j`` / ``--json-schema`` parameter.

llama.cpp's grammar-based JSON schema support ensures the output is
**guaranteed** to conform to the schema at the sampler level — no
retries needed, unlike prompt-based structured output.

Usage::

    from pydantic import BaseModel, Field
    from schema_compiler import SchemaCompiler

    class Person(BaseModel):
        name: str = Field(description="Full name")
        age: int = Field(ge=0, le=150)
        email: str | None = Field(default=None)

    compiler = SchemaCompiler()
    schema = compiler.compile(Person)
    # -> {"type": "object", "properties": {"name": ..., "age": ...}, ...}
    # Ready to pass as -j '{"type": "object", ...}' to llama.cpp /completion
"""

from __future__ import annotations

import json
from typing import Any, Dict, Type, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic._internal._model_construction import ModelMetaclass


class SchemaCompiler:
    """
    Compiles Pydantic models to JSON Schema dictionaries suitable for
    llama.cpp's ``-j`` / ``--json-schema`` grammar constraint.

    Handles:
    - Required / optional fields
    - Nested models (recursive)
    - Typed lists (``list[X]``)
    - Typed dicts (``dict[str, X]``)
    - Union types (``X | Y``)
    - Enums (via ``Literal`` or ``enum.Enum`` subclasses)
    - Field constraints (``ge``, ``le``, ``min_length``, etc.)
    - Field descriptions
    """

    def compile(self, model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Convert a Pydantic model class to a JSON Schema dict.

        Args:
            model: A Pydantic ``BaseModel`` subclass.

        Returns:
            A JSON Schema dict (compatible with llama.cpp ``-j``).
        """
        return self._model_to_schema(model)

    def compile_to_json(self, model: Type[BaseModel], indent: int = 2) -> str:
        """
        Convert a Pydantic model to a JSON Schema string.

        Args:
            model: A Pydantic ``BaseModel`` subclass.
            indent: JSON indentation level.

        Returns:
            A JSON Schema string ready to pass as ``-j`` to llama.cpp.
        """
        schema = self.compile(model)
        # llama.cpp expects the schema as a compact inline JSON string
        return json.dumps(schema, indent=indent)

    # ------------------------------------------------------------------
    # Internal schema builders
    # ------------------------------------------------------------------

    def _model_to_schema(self, model: Type[BaseModel]) -> Dict[str, Any]:
        """Convert a Pydantic model to a JSON Schema object definition."""
        properties: Dict[str, Any] = {}
        required: list[str] = []

        for field_name, field_info in model.model_fields.items():
            field_schema = self._field_to_schema(field_name, field_info)
            if field_schema is not None:
                properties[field_name] = field_schema
                # Required if field has no default and is not Optional[X]
                is_optional = self._is_optional(field_info.annotation)
                if not is_optional and field_info.is_required():
                    required.append(field_name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        # Add model description if available
        if hasattr(model, "__doc__") and model.__doc__:
            schema["description"] = model.__doc__.strip()

        return schema

    def _field_to_schema(
        self, name: str, field_info: Any
    ) -> Dict[str, Any] | None:
        """Convert a Pydantic field info to a JSON Schema property definition."""
        schema: Dict[str, Any] = {}

        # Add description
        if field_info.description:
            schema["description"] = field_info.description

        # Get type info
        annotation = field_info.annotation
        if annotation is None:
            return None

        # Resolve the type schema
        type_schema = self._resolve_type(annotation)
        if type_schema:
            schema.update(type_schema)

        # Add constraints from Field()
        constraints = self._extract_constraints(field_info)
        schema.update(constraints)

        return schema if schema else None

    def _resolve_type(self, tp: Any) -> Dict[str, Any]:
        """Resolve a Python type annotation to a JSON Schema fragment."""
        origin = get_origin(tp)
        args = get_args(tp)

        # Handle Optional[X] / X | None
        if origin is UnionType or origin is Union:
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                # Optional[X] -> just X (null handled by missing from required)
                return self._resolve_type(non_none_args[0])
            # Union[X, Y] -> anyOf
            return {
                "anyOf": [self._resolve_type(a) for a in non_none_args],
            }

        # Handle list[X]
        if origin is list:
            if args:
                return {
                    "type": "array",
                    "items": self._resolve_type(args[0]),
                }
            return {"type": "array"}

        # Handle dict[str, X]
        if origin is dict:
            if args:
                return {
                    "type": "object",
                    "additionalProperties": self._resolve_type(args[1]),
                }
            return {"type": "object"}

        # Handle Literal[X, Y, Z]
        if origin is Literal:
            values = [self._literal_to_json(a) for a in args]
            if all(isinstance(v, str) for v in values):
                return {"type": "string", "enum": values}
            if all(isinstance(v, (int, float)) for v in values):
                return {"type": "number", "enum": values}
            return {"enum": values}

        # Handle nested Pydantic models
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            return self._model_to_schema(tp)

        # Handle Enum subclasses
        if isinstance(tp, type) and issubclass(tp, Enum):
            values = [e.value for e in tp]
            if all(isinstance(v, str) for v in values):
                return {"type": "string", "enum": values}
            return {"enum": values}

        # Primitive types
        type_map: Dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        if tp in type_map:
            return {"type": type_map[tp]}

        # Fallback: generic object
        return {}

    def _is_optional(self, tp: Any) -> bool:
        """Check if a type is Optional[X] (i.e. Union[X, None])."""
        origin = get_origin(tp)
        args = get_args(tp)
        if origin is UnionType or origin is Union:
            return type(None) in args
        return False

    def _extract_constraints(self, field_info: Any) -> Dict[str, Any]:
        """
        Extract numeric/string constraints from a Pydantic Field.
        Returns JSON Schema constraint keywords.

        In Pydantic v2, constraints are stored in ``field_info.metadata``
        as ``annotated_types`` objects (Ge, Le, MinLen, MaxLen, etc.).
        """
        constraints: Dict[str, Any] = {}
        extra = getattr(field_info, "json_schema_extra", None) or {}

        # Pydantic v2: constraints live in metadata list
        for m in field_info.metadata:
            # Numeric constraints
            if hasattr(m, "ge") and m.ge is not None:
                constraints["minimum"] = m.ge
            if hasattr(m, "le") and m.le is not None:
                constraints["maximum"] = m.le
            if hasattr(m, "gt") and m.gt is not None:
                constraints["exclusiveMinimum"] = m.gt
            if hasattr(m, "lt") and m.lt is not None:
                constraints["exclusiveMaximum"] = m.lt
            if hasattr(m, "multiple_of") and m.multiple_of is not None:
                constraints["multipleOf"] = m.multiple_of

            # String constraints
            if hasattr(m, "min_length") and m.min_length is not None:
                constraints["minLength"] = m.min_length
            if hasattr(m, "max_length") and m.max_length is not None:
                constraints["maxLength"] = m.max_length
            if hasattr(m, "pattern") and m.pattern is not None:
                constraints["pattern"] = m.pattern

        return constraints

    def _literal_to_json(self, value: Any) -> Any:
        """Convert a Literal value to a JSON-compatible value."""
        if isinstance(value, Enum):
            return value.value
        return value


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def json_schema_from_pydantic(model: Type[BaseModel]) -> Dict[str, Any]:
    """One-shot: convert a Pydantic model to JSON Schema."""
    return SchemaCompiler().compile(model)


# Late imports for type hinting
from enum import Enum
from types import UnionType
from typing import Literal, Union
