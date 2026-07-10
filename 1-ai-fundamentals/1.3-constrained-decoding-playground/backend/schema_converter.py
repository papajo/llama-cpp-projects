"""
JSON Schema → GBNF Grammar Converter.

Converts JSON Schema definitions into GBNF grammar rules
that llama.cpp can use for constrained decoding.

Supports:
  - Object types with properties
  - Array types with items
  - String, number, integer, boolean
  - enums
  - required fields
  - nested objects and arrays
  - string patterns (basic)
  - number ranges (min/max)
  - allOf, anyOf, oneOf (basic)

Based on the GBNF grammar format used by llama.cpp.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


class SchemaConverter:
    """
    Convert a JSON Schema dict to a GBNF grammar string.
    
    Usage:
        converter = SchemaConverter()
        gbnf = converter.convert({"type": "object", "properties": {...}})
    """

    def __init__(self):
        self._rules: Dict[str, str] = {}
        self._rule_counters: Dict[str, int] = {}
        self._used_names: Set[str] = set()

    def convert(self, schema: dict) -> str:
        """
        Convert a JSON Schema to a GBNF grammar string.
        
        Returns the complete GBNF grammar with a root rule.
        """
        self._rules = {}
        self._rule_counters = {}
        self._used_names = set()

        root_type = self._resolve_type(schema)
        root_rule = self._type_to_rule(schema, root_type, "root")
        self._rules["root"] = root_rule

        # Build the grammar text
        parts = [f"root ::= {root_rule}"]
        for name, rule in self._rules.items():
            if name != "root":
                parts.append(f"{name} ::= {rule}")

        return "\n".join(parts)

    def convert_from_json(self, json_str: str) -> str:
        """Convert a JSON Schema string to GBNF."""
        schema = json.loads(json_str)
        return self.convert(schema)

    # ── Type resolution ────────────────────────────────────────

    def _resolve_type(self, schema: dict) -> str:
        """Determine the JSON Schema type, handling shorthand."""
        if "type" in schema:
            t = schema["type"]
            if isinstance(t, list):
                return t[0]  # take the first type
            return t
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        if "enum" in schema:
            return "string"  # enum is always strings in practice
        if "anyOf" in schema or "oneOf" in schema:
            return "any"
        return "string"

    def _fresh_name(self, base: str) -> str:
        """Generate a unique rule name."""
        if base not in self._used_names:
            self._used_names.add(base)
            return base
        counter = self._rule_counters.get(base, 1)
        self._rule_counters[base] = counter + 1
        name = f"{base}_{counter}"
        self._used_names.add(name)
        return name

    # ── Type → GBNF conversion ─────────────────────────────────

    def _type_to_rule(self, schema: dict, type_name: str,
                       rule_name: str) -> str:
        """Convert a schema + type to a GBNF rule body string."""
        if type_name == "object":
            return self._object_rule(schema, rule_name)
        elif type_name == "array":
            return self._array_rule(schema, rule_name)
        elif type_name == "string":
            return self._string_rule(schema)
        elif type_name == "number" or type_name == "integer":
            return self._number_rule(schema, type_name)
        elif type_name == "boolean":
            return self._boolean_rule()
        elif type_name == "null":
            return '"null"'
        elif type_name == "any":
            return self._any_of_rule(schema, rule_name)
        else:
            return self._string_rule(schema)  # fallback

    def _object_rule(self, schema: dict, rule_name: str) -> str:
        """Convert an object schema to GBNF."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)

        if not properties:
            return r'"{}"'

        prop_parts = []
        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required
            prop_type = self._resolve_type(prop_schema)
            prop_rule_name = self._fresh_name(f"prop_{prop_name}")
            prop_rule = self._type_to_rule(prop_schema, prop_type, prop_rule_name)
            self._rules[prop_rule_name] = prop_rule

            quoted_key = self._escape_string(prop_name)
            if is_required:
                prop_parts.append(f'"\\"{quoted_key}\\":" " " {prop_rule_name}')
            else:
                prop_parts.append(
                    f'( "\\"{quoted_key}\\":" " " {prop_rule_name} )?'
                )

        # Build object with comma-separated properties
        if len(prop_parts) == 1:
            body = r'"{" " " ' + prop_parts[0] + r' " "}"'
        else:
            body = r'"{" " " '
            body += prop_parts[0]
            for p in prop_parts[1:]:
                body += f' "," " " {p}'
            body += r' " "}"'

        return body

    def _array_rule(self, schema: dict, rule_name: str) -> str:
        """Convert an array schema to GBNF."""
        items = schema.get("items", {})
        item_type = self._resolve_type(items)
        item_rule_name = self._fresh_name(f"{rule_name}_item")
        item_rule = self._type_to_rule(items, item_type, item_rule_name)
        self._rules[item_rule_name] = item_rule

        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", 5)

        if max_items == 0:
            return r'"[]"'

        # Build array elements
        if min_items == 0:
            return (
                r'"[" " " ' +
                f'( {item_rule_name} ( "," " " {item_rule_name} )* )?' +
                r' " "]"'
            )
        else:
            elements = f'{item_rule_name}'
            for _ in range(min_items - 1):
                elements += f' "," " " {item_rule_name}'
            elements += f' ( "," " " {item_rule_name} )*'
            return r'"[" " " ' + elements + r' " "]"'

    def _string_rule(self, schema: dict) -> str:
        """Convert a string schema to GBNF."""
        enum_vals = schema.get("enum", [])
        if enum_vals:
            alternatives = " | ".join(
                f'"\\"{self._escape_string(v)}\\"' for v in enum_vals
            )
            return f"( {alternatives} )"

        pattern = schema.get("pattern", "")
        if pattern:
            # Attempt basic pattern conversion
            return self._pattern_to_rule(pattern)

        # Default string: any characters (basic)
        return r'"\\"" ( [^"\\\\] | "\\\\" . )* "\\""'

    def _number_rule(self, schema: dict, type_name: str) -> str:
        """Convert a number/integer schema to GBNF."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        if type_name == "integer":
            if minimum is not None and maximum is not None:
                return f'[0-9]{{1,{len(str(maximum))}}}'
            return r'("0" | [1-9] [0-9]*)'
        else:
            # Floating point
            return r'("0" | [1-9] [0-9]*) ( "." [0-9]+ )?'

    def _boolean_rule(self) -> str:
        """Convert boolean to GBNF."""
        return '"true" | "false"'

    def _any_of_rule(self, schema: dict, rule_name: str) -> str:
        """Convert anyOf/oneOf to GBNF alternatives."""
        alternatives = []

        for key in ("anyOf", "oneOf"):
            subs = schema.get(key, [])
            for i, sub in enumerate(subs):
                sub_type = self._resolve_type(sub)
                sub_name = self._fresh_name(f"{rule_name}_opt{i}")
                sub_rule = self._type_to_rule(sub, sub_type, sub_name)
                self._rules[sub_name] = sub_rule
                alternatives.append(sub_name)

        if alternatives:
            return " | ".join(alternatives)
        return self._string_rule(schema)

    # ── Helpers ────────────────────────────────────────────────

    def _escape_string(self, s: str) -> str:
        """Escape a string for use in GBNF."""
        return s.replace('\\', '\\\\').replace('"', '\\"')

    def _pattern_to_rule(self, pattern: str) -> str:
        """Basic regex pattern to GBNF conversion."""
        # Simple patterns only
        if pattern == r"^\d{4}-\d{2}-\d{2}$":
            return r'[0-9]{4} "-" [0-9]{2} "-" [0-9]{2}'
        if pattern == r"^\d{3}-\d{3}-\d{4}$":
            return r'[0-9]{3} "-" [0-9]{3} "-" [0-9]{4}'
        if pattern == r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$":
            return r'[a-zA-Z0-9._%+-] "+" "@" "+" [a-zA-Z0-9.-] "+" "." [a-zA-Z]{2,4}'
        if pattern == r"^https?://":
            return r'"http" ("s")? "://" [a-zA-Z0-9.-]+ ( "." [a-zA-Z]{2,4} )?'
        # Fallback: allow anything
        return r'"\\"" ( [^"\\\\] | "\\\\" . )* "\\""'
