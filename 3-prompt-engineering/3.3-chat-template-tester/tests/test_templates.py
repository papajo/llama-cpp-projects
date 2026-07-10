"""Tests for the template registry module."""

import pytest
from chat_template_tester.templates import (
    TemplateInfo,
    TemplateRegistry,
    registry,
    _BUILTINS,
)


class TestTemplateInfo:
    def test_minimal(self):
        info = TemplateInfo(name="test", template="Hello {{ name }}")
        assert info.name == "test"
        assert info.bos_token is None
        assert info.eos_token is None
        assert info.description == ""
        assert info.stop_tokens == []

    def test_full(self):
        info = TemplateInfo(
            name="full",
            template="{{ bos_token }}{{ messages }}",
            bos_token="<s>",
            eos_token="</s>",
            description="Full test template",
            stop_tokens=["</s>"],
        )
        assert info.bos_token == "<s>"
        assert info.eos_token == "</s>"


class TestTemplateRegistry:
    def test_register_and_get(self):
        reg = TemplateRegistry()
        info = TemplateInfo(name="a", template="x")
        reg.register(info)
        assert reg.get("a") is info

    def test_get_unknown(self):
        reg = TemplateRegistry()
        with pytest.raises(KeyError, match="Unknown template"):
            reg.get("nope")

    def test_contains(self):
        reg = TemplateRegistry()
        reg.register(TemplateInfo(name="b", template="y"))
        assert "b" in reg
        assert "c" not in reg

    def test_list_sorted(self):
        reg = TemplateRegistry()
        reg.register(TemplateInfo(name="z", template="1"))
        reg.register(TemplateInfo(name="a", template="2"))
        names = [t.name for t in reg.list()]
        assert names == ["a", "z"]

    def test_register_replaces(self):
        reg = TemplateRegistry()
        reg.register(TemplateInfo(name="x", template="old"))
        reg.register(TemplateInfo(name="x", template="new"))
        assert reg.get("x").template == "new"


class TestBuiltins:
    def test_all_builtins_registered(self):
        assert len(_BUILTINS) == 10

    def test_all_builtins_have_required_fields(self):
        for info in _BUILTINS:
            assert info.name, f"Missing name in {info}"
            assert info.template, f"Missing template in {info.name}"
            assert info.description, f"Missing description in {info.name}"

    def test_names_are_unique(self):
        names = [t.name for t in _BUILTINS]
        assert len(names) == len(set(names))

    def test_registry_has_all_builtins(self):
        assert len(registry.list()) == len(_BUILTINS)
        for info in _BUILTINS:
            assert info.name in registry

    def test_specific_builtins_exist(self):
        expected = {"llama3", "llama2", "chatml", "mistral", "vicuna",
                    "gemma", "phi3", "deepseek", "command-r", "qwen2.5"}
        actual = {t.name for t in registry.list()}
        assert actual == expected

    def test_all_templates_have_stop_tokens(self):
        for info in _BUILTINS:
            assert len(info.stop_tokens) > 0, f"{info.name} has no stop tokens"

    def test_template_strings_contain_jinja2_syntax(self):
        for info in _BUILTINS:
            assert "{" in info.template, f"{info.name} lacks Jinja2 syntax"
            assert "messages" in info.template, f"{info.name} missing messages var"

    def test_llama3_template_structure(self):
        tmpl = registry.get("llama3")
        assert "<|start_header_id|>" in tmpl.template
        assert "<|eot_id|>" in tmpl.template
        assert tmpl.bos_token == "<|begin_of_text|>"

    def test_chatml_template_structure(self):
        tmpl = registry.get("chatml")
        assert "<|im_start|>" in tmpl.template
        assert "<|im_end|>" in tmpl.template
