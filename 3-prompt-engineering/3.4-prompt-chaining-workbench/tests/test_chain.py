"""Tests for chain definitions."""

import pytest
from prompt_chaining.chain import (
    ChainStep,
    PromptChain,
    resolve_template,
)


class TestResolveTemplate:
    def test_simple_var(self):
        assert resolve_template("Hello {{input}}", {"input": "world"}) == "Hello world"

    def test_multiple_vars(self):
        ctx = {"input": "hello", "step_1": "world"}
        assert resolve_template("{{input}} {{step_1}}", ctx) == "hello world"

    def test_unknown_var_left_asis(self):
        assert resolve_template("{{unknown}}", {}) == "{{unknown}}"

    def test_dotted_key(self):
        assert resolve_template("{{step_1.output}}", {"step_1": "value"}) == "value"

    def test_no_vars(self):
        assert resolve_template("plain text", {}) == "plain text"

    def test_empty_string(self):
        assert resolve_template("", {}) == ""

    def test_mixed_known_unknown(self):
        ctx = {"input": "hi"}
        result = resolve_template("{{input}} {{missing}}", ctx)
        assert result == "hi {{missing}}"

    def test_multiple_occurrences(self):
        ctx = {"input": "yes"}
        assert resolve_template("{{input}} or {{input}}?", ctx) == "yes or yes?"


class TestChainStep:
    def test_minimal(self):
        step = ChainStep(name="s1", user_prompt="Hello")
        assert step.name == "s1"
        assert step.system_prompt == ""
        assert step.temperature is None
        assert step.max_tokens is None

    def test_render_no_system(self):
        step = ChainStep(name="s1", user_prompt="Say {{input}}")
        result = step.render({"input": "hi"})
        assert result["system"] == ""
        assert result["user"] == "Say hi"

    def test_render_with_system(self):
        step = ChainStep(
            name="s1",
            system_prompt="You are {{role}}",
            user_prompt="Do: {{input}}",
        )
        result = step.render({"role": "bot", "input": "task"})
        assert result["system"] == "You are bot"
        assert result["user"] == "Do: task"

    def test_render_with_step_ref(self):
        step = ChainStep(name="s2", user_prompt="Refine: {{step_1}}")
        result = step.render({"step_1": "draft"})
        assert result["user"] == "Refine: draft"

    def test_render_unknown_var(self):
        step = ChainStep(name="s1", user_prompt="{{missing}}")
        result = step.render({})
        assert "{{missing}}" in result["user"]


class TestPromptChain:
    def test_minimal(self):
        chain = PromptChain(name="test", steps=[ChainStep(name="s1", user_prompt="hi")])
        assert chain.name == "test"
        assert chain.step_count == 1
        assert chain.temperature == 0.7
        assert chain.max_tokens == 512

    def test_step_count(self):
        chain = PromptChain(
            name="test",
            steps=[
                ChainStep(name="a", user_prompt="1"),
                ChainStep(name="b", user_prompt="2"),
            ],
        )
        assert chain.step_count == 2

    def test_getitem(self):
        s1 = ChainStep(name="a", user_prompt="1")
        s2 = ChainStep(name="b", user_prompt="2")
        chain = PromptChain(name="test", steps=[s1, s2])
        assert chain[0] is s1
        assert chain[1] is s2

    def test_to_dict(self):
        chain = PromptChain(
            name="test-chain",
            description="A test",
            temperature=0.5,
            max_tokens=256,
            steps=[
                ChainStep(name="s1", user_prompt="Hello {{input}}", system_prompt="Be nice.", temperature=0.3),
                ChainStep(name="s2", user_prompt="Next: {{step_1}}", max_tokens=128),
            ],
        )
        d = chain.to_dict()
        assert d["name"] == "test-chain"
        assert d["description"] == "A test"
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 256
        assert len(d["steps"]) == 2
        assert d["steps"][0]["name"] == "s1"
        assert d["steps"][0]["temperature"] == 0.3
        assert d["steps"][1]["max_tokens"] == 128

    def test_from_dict(self):
        data = {
            "name": "restored",
            "description": "Round trip",
            "temperature": 0.3,
            "max_tokens": 1024,
            "steps": [
                {"name": "a", "user_prompt": "{{input}}", "system_prompt": "", "temperature": None, "max_tokens": None},
                {"name": "b", "user_prompt": "{{step_1}}", "system_prompt": "Review", "temperature": 0.1, "max_tokens": 512},
            ],
        }
        chain = PromptChain.from_dict(data)
        assert chain.name == "restored"
        assert chain.step_count == 2
        assert chain[0].name == "a"
        assert chain[1].system_prompt == "Review"
        assert chain[1].temperature == 0.1

    def test_round_trip(self):
        original = PromptChain(
            name="round-trip",
            description="Test round trip serialisation",
            temperature=0.7,
            max_tokens=2048,
            steps=[
                ChainStep(name="step1", user_prompt="First: {{input}}"),
                ChainStep(name="step2", user_prompt="Then: {{step_1}}", temperature=0.5),
            ],
        )
        restored = PromptChain.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.step_count == original.step_count
        for orig_step, rest_step in zip(original.steps, restored.steps):
            assert orig_step.name == rest_step.name
            assert orig_step.user_prompt == rest_step.user_prompt
            assert orig_step.temperature == rest_step.temperature
