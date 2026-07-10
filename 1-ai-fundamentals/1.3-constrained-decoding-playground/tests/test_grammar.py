"""Tests for the GBNF grammar utilities."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.grammar import GBNFParser, GrammarDebugger, GrammarMaskStep
from backend.schema_converter import SchemaConverter


def test_parse_simple_grammar():
    """Parse a basic GBNF grammar."""
    text = 'root ::= "hello" " " "world"'
    rules = GBNFParser.parse(text)
    assert len(rules) == 1
    assert rules[0].name == "root"
    assert rules[0].is_root
    print(f"  ✓ Parsed root rule: {rules[0].summary()}")


def test_parse_multi_rule_grammar():
    """Parse a grammar with multiple rules."""
    text = """
root ::= expr
expr ::= term (("+" | "-") term)*
term ::= factor (("*" | "/") factor)*
factor ::= number | "(" expr ")"
number ::= "0" | [1-9] [0-9]*
    """
    rules = GBNFParser.parse(text)
    assert len(rules) == 5
    assert any(r.name == "root" for r in rules)
    assert any(r.name == "number" for r in rules)
    print(f"  ✓ Parsed {len(rules)} rules ({', '.join(r.name for r in rules)})")


def test_validate_empty():
    """Empty grammar should fail validation."""
    errors = GBNFParser.validate("")
    assert len(errors) > 0
    print(f"  ✓ Empty grammar: {errors[0]}")


def test_validate_no_root():
    """Grammar without root rule should warn."""
    text = 'greeting ::= "hello"'
    errors = GBNFParser.validate(text)
    assert any("root" in e for e in errors)
    print(f"  ✓ No-root grammar: {errors[0]}")


def test_schema_to_grammar_simple():
    """Convert a simple JSON schema to GBNF."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    converter = SchemaConverter()
    gbnf = converter.convert(schema)
    assert "root ::=" in gbnf
    assert len(gbnf) > 0
    print(f"  ✓ Converted schema to GBNF ({len(gbnf)} chars)")
    print(f"    {gbnf[:80]}...")


def test_schema_to_grammar_enum():
    """Convert schema with enum values."""
    schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "pending"],
            },
        },
        "required": ["status"],
    }
    converter = SchemaConverter()
    gbnf = converter.convert(schema)
    assert "active" in gbnf
    assert "inactive" in gbnf
    print(f"  ✓ Enum values present in grammar")


def test_schema_to_grammar_nested():
    """Convert schema with nested objects and arrays."""
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["id"],
            },
        },
        "required": ["user"],
    }
    converter = SchemaConverter()
    gbnf = converter.convert(schema)
    assert "root ::=" in gbnf
    print(f"  ✓ Nested schema converted ({len(gbnf)} chars)")


def test_trace_analysis():
    """Analyse a completion with n_probs for grammar masking."""
    completion_data = {
        "content": '{"name": "Alice"}',
        "completion_probabilities": [
            {
                "text": '{"',
                "id": 1,
                "logprob": -0.1,
                "top_logprobs": [
                    {"id": 1, "logprob": -0.1},
                    {"id": 2, "logprob": -1.5},
                    {"id": 3, "logprob": -2.0},
                ],
            },
            {
                "text": "name",
                "id": 42,
                "logprob": -0.3,
                "top_logprobs": [
                    {"id": 42, "logprob": -0.3},
                    {"id": 43, "logprob": -0.8},
                    {"id": 44, "logprob": -1.2},
                ],
            },
        ],
    }
    result = GrammarDebugger.analyse_completion(completion_data)
    assert result.total_tokens == 2
    assert result.full_text == '{"name": "Alice"}'
    print(f"  ✓ Trace analysis: {result.total_tokens} tokens, "
          f"{result.tokens_masked} masked, "
          f"{result.mask_rate*100:.1f}% mask rate")


def test_weather_schema_from_file():
    """Convert the weather sample schema."""
    schema_path = Path(__file__).parent.parent / "sample-schemas" / "weather.json"
    schema = json.loads(schema_path.read_text())
    converter = SchemaConverter()
    gbnf = converter.convert(schema)
    assert "root ::=" in gbnf
    assert "location" in gbnf or "prop_location" in gbnf
    print(f"  ✓ Weather schema converted ({len(gbnf)} chars)")


if __name__ == "__main__":
    print("GBNF Grammar Tests")
    print("=" * 50)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except Exception as e:
                print(f"  ✗ FAILED: {name}: {e}")
                import traceback
                traceback.print_exc()
    print("=" * 50)
    print("Done!")
