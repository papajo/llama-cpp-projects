"""Tests for the test runner module."""

import pytest
from chat_template_tester.tester import run_test, TemplateTestResult, TestRun
from chat_template_tester.templates import TemplateInfo, TemplateRegistry


SIMPLE_MSGS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]


class TestRunTest:
    def test_all_templates(self):
        run = run_test(SIMPLE_MSGS)
        assert len(run.results) >= 10
        assert run.messages == SIMPLE_MSGS
        assert run.add_generation_prompt is True

    def test_single_template(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        assert len(run.results) == 1
        assert run.results[0].template_name == "chatml"
        assert run.results[0].error is None

    def test_multiple_templates(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml", "llama3", "mistral"])
        names = [r.template_name for r in run.results]
        assert names == ["chatml", "llama3", "mistral"]

    def test_no_gen_prompt(self):
        run = run_test(SIMPLE_MSGS, add_generation_prompt=False)
        assert run.add_generation_prompt is False
        # ChatML should not have assistant starter
        chatml_result = next(r for r in run.results if r.template_name == "chatml")
        assert "<|im_start|>assistant" not in chatml_result.formatted

    def test_with_gen_prompt(self):
        run = run_test(SIMPLE_MSGS, add_generation_prompt=True)
        chatml_result = next(r for r in run.results if r.template_name == "chatml")
        assert "<|im_start|>assistant" in chatml_result.formatted

    def test_custom_templates(self):
        custom = TemplateInfo(
            name="custom-test",
            template="CUSTOM: {{ messages[0]['content'] }}",
        )
        run = run_test(SIMPLE_MSGS, custom_templates=[custom])
        custom_result = next(r for r in run.results if r.template_name == "custom-test")
        assert custom_result.error is None
        assert "CUSTOM:" in custom_result.formatted

    def test_unknown_template_name(self):
        with pytest.raises(KeyError):
            run_test(SIMPLE_MSGS, template_names=["does-not-exist"])

    def test_custom_templates_are_included_with_all(self):
        custom = TemplateInfo(name="extra", template="Hello")
        run = run_test(SIMPLE_MSGS, custom_templates=[custom])
        names = [r.template_name for r in run.results]
        assert "extra" in names

    def test_empty_messages_no_gen_prompt(self):
        run = run_test([], template_names=["chatml"], add_generation_prompt=False)
        assert len(run.results) == 1
        assert run.results[0].error is None
        # ChatML with empty messages and no gen prompt should be empty
        assert run.results[0].formatted == ""

    def test_empty_messages_with_gen_prompt(self):
        run = run_test([], template_names=["chatml"], add_generation_prompt=True)
        assert run.results[0].error is None
        assert "<|im_start|>assistant" in run.results[0].formatted


class TestTemplateTestResult:
    def test_error_vs_ok(self):
        ok = TemplateTestResult(
            template_name="a", template_description="", formatted="hi",
            char_count=2, estimated_tokens=1, special_token_counts={},
        )
        err = TemplateTestResult(
            template_name="b", template_description="", formatted="",
            char_count=0, estimated_tokens=0, special_token_counts={},
            error="Something broke",
        )
        assert ok.error is None
        assert err.error == "Something broke"

    def test_special_token_counts(self):
        # Use deepseek which has unicode special tokens
        run = run_test(SIMPLE_MSGS, template_names=["deepseek"])
        r = run.results[0]
        assert r.error is None
        assert "User:" in r.formatted
        assert r.char_count > 0
        assert r.estimated_tokens > 0


class TestTestRun:
    def test_ok_results(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml"])
        assert len(run.ok_results()) == 1
        assert len(run.error_results()) == 0

    def test_template_names_property(self):
        run = run_test(SIMPLE_MSGS, template_names=["chatml", "llama3"])
        assert run.template_names == ["chatml", "llama3"]
