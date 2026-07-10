"""Tests for the chat formatter module."""

import pytest
from chat_template_tester.formatter import (
    ChatFormatter,
    estimate_tokens,
    find_special_tokens,
    strip_special_tokens,
    _raise_exception,
)
from chat_template_tester.templates import TemplateInfo


# ── Sample messages ───────────────────────────────────────────────

SIMPLE_MSGS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

MULTI_TURN_MSGS = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
]


class TestChatFormatter:
    def test_chatml_no_gen_prompt(self):
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="chatml",
            template=(
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
                "{% endfor %}"
            ),
        )
        result = formatter.format(tmpl, SIMPLE_MSGS, add_generation_prompt=False)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_end|>" in result
        assert "You are a helpful assistant." in result
        assert "What is the capital of France?" in result
        # Should NOT have assistant prompt
        assert "<|im_start|>assistant" not in result

    def test_chatml_with_gen_prompt(self):
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="chatml",
            template=(
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\\n' }}"
                "{% endif %}"
            ),
        )
        result = formatter.format(tmpl, SIMPLE_MSGS, add_generation_prompt=True)
        assert "<|im_start|>assistant" in result

    def test_bos_token_injection(self):
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="bos-test",
            template="{{ bos_token }}Hello",
            bos_token="<s>",
        )
        result = formatter.format(tmpl, [])
        assert result == "<s>Hello"

    def test_eos_token_injection(self):
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="eos-test",
            template="Hello{{ eos_token }}",
            eos_token="</s>",
        )
        result = formatter.format(tmpl, [])
        assert result == "Hello</s>"

    def test_multiple_formats_preserve_state(self):
        """Formatter should be reusable across calls."""
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="chatml",
            template=(
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
                "{% endfor %}"
            ),
        )
        r1 = formatter.format(tmpl, SIMPLE_MSGS)
        r2 = formatter.format(tmpl, SIMPLE_MSGS)
        assert r1 == r2

    def test_multi_turn(self):
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="chatml",
            template=(
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
                "{% endfor %}"
            ),
        )
        result = formatter.format(tmpl, MULTI_TURN_MSGS)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert result.count("<|im_start|>") == 4  # sys + user + asst + user

    def test_format_all(self):
        formatter = ChatFormatter()
        tmpl_a = TemplateInfo(name="a", template="Hello")
        tmpl_b = TemplateInfo(name="b", template="World")
        results = formatter.format_all([tmpl_a, tmpl_b], [])
        assert results["a"] == "Hello"
        assert results["b"] == "World"

    def test_raise_exception(self):
        """The raise_exception global should actually raise."""
        with pytest.raises(ValueError, match="test error"):
            _raise_exception("test error")

    def test_template_error_propagates(self):
        """A template that calls raise_exception should propagate."""
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="bad",
            template="{{ raise_exception('boom') }}",
        )
        with pytest.raises(ValueError, match="boom"):
            formatter.format(tmpl, [])

    def test_bad_jinja_syntax(self):
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="broken",
            template="{{ unclosed",
        )
        with pytest.raises(ValueError, match="syntax error|broken"):
            formatter.format(tmpl, [])

    def test_undefined_variable(self):
        formatter = ChatFormatter()
        tmpl = TemplateInfo(
            name="undef",
            template="{{ undefined_var }}",
        )
        with pytest.raises(ValueError, match="undefined variable|undefined_var"):
            formatter.format(tmpl, [])

    def test_literal_template_no_messages(self):
        """Template that doesn't reference messages."""
        formatter = ChatFormatter()
        tmpl = TemplateInfo(name="literal", template="static text")
        result = formatter.format(tmpl, [{"role": "user", "content": "hi"}])
        assert result == "static text"


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("Hello") == 1  # round(5/4) = 1

    def test_typical(self):
        text = "The quick brown fox jumps over the lazy dog."
        # 44 chars / 4 = 11
        assert estimate_tokens(text) == 11

    def test_ten_chars(self):
        assert estimate_tokens("1234567890") == 2  # round(10/4) = 2 (bankers)

    def test_four_chars(self):
        assert estimate_tokens("test") == 1  # 4/4 = 1


class TestFindSpecialTokens:
    def test_none_found(self):
        assert find_special_tokens("hello world", ["<s>", "</s>"]) == {
            "<s>": 0, "</s>": 0
        }

    def test_some_found(self):
        text = "<s>Hello</s>World</s>"
        assert find_special_tokens(text, ["<s>", "</s>"]) == {
            "<s>": 1, "</s>": 2
        }

    def test_empty_tokens_list(self):
        assert find_special_tokens("hello", []) == {}

    def test_empty_text(self):
        assert find_special_tokens("", ["<s>"]) == {"<s>": 0}


class TestStripSpecialTokens:
    def test_basic(self):
        result = strip_special_tokens("<s>Hello</s>", ["<s>", "</s>"])
        assert result == "Hello"

    def test_no_tokens(self):
        result = strip_special_tokens("Hello World", ["<s>"])
        assert result == "Hello World"

    def test_multiple_occurrences(self):
        result = strip_special_tokens(
            "<s><s>Hello</s></s>", ["<s>", "</s>"]
        )
        assert result == "Hello"
