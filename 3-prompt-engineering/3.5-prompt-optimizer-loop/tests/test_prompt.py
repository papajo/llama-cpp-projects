"""Tests for prompt_optimizer.prompt."""

import pytest
from prompt_optimizer.prompt import PromptTemplate, PromptTemplateError, TestCase


class TestPromptTemplate:
    def test_render_basic(self):
        pt = PromptTemplate("Hello {{name}}!")
        assert pt.render({"name": "World"}) == "Hello World!"

    def test_render_multiple_variables(self):
        pt = PromptTemplate("{{greeting}} {{name}}, {{emotion}} day!")
        result = pt.render({"greeting": "Hi", "name": "Alice", "emotion": "great"})
        assert result == "Hi Alice, great day!"

    def test_render_no_variables(self):
        pt = PromptTemplate("Hello world")
        assert pt.render({}) == "Hello world"

    def test_render_missing_variable(self):
        pt = PromptTemplate("Hello {{name}}!")
        with pytest.raises(PromptTemplateError, match="Missing template variables"):
            pt.render({})

    def test_render_extra_variables_ignored(self):
        pt = PromptTemplate("Hello {{name}}!")
        assert pt.render({"name": "Bob", "extra": "ignored"}) == "Hello Bob!"

    def test_has_variable(self):
        pt = PromptTemplate("{{a}} and {{b}}")
        assert pt.has_variable("a")
        assert pt.has_variable("b")
        assert not pt.has_variable("c")

    def test_variables_property(self):
        pt = PromptTemplate("{{x}} {{y}} {{x}}")
        assert pt.variables == frozenset({"x", "y"})

    def test_str(self):
        t = "Test {{var}} template"
        pt = PromptTemplate(t)
        assert str(pt) == t

    def test_render_empty_var(self):
        """Empty string is a valid value."""
        pt = PromptTemplate("Hello {{name}}!")
        assert pt.render({"name": ""}) == "Hello !"

    def test_special_chars_in_value(self):
        pt = PromptTemplate("Reply: {{msg}}")
        result = pt.render({"msg": "Hello, {world}! {{escaped}}"})
        assert result == "Reply: Hello, {world}! {{escaped}}"

    def test_from_file(self, tmp_path):
        f = tmp_path / "template.txt"
        f.write_text("File template {{var}}")
        pt = PromptTemplate.from_file(str(f))
        assert pt.render({"var": "test"}) == "File template test"


class TestTestCase:
    def test_defaults(self):
        tc = TestCase()
        assert tc.input_vars == {}
        assert tc.expected is None
        assert tc.metadata == {}

    def test_with_values(self):
        tc = TestCase(
            input_vars={"text": "Hello"},
            expected="Bonjour",
            metadata={"difficulty": "easy"},
        )
        assert tc.input_vars == {"text": "Hello"}
        assert tc.expected == "Bonjour"
        assert tc.metadata["difficulty"] == "easy"
