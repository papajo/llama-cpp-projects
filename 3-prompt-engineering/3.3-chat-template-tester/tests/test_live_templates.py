"""Live integration tests for the chat template tester.

This project is pure client-side string formatting - it never speaks HTTP,
so the offline suite has no server to drift from. The live layer supplies
the one thing it cannot check on its own: whether its rendering of a
template matches what llama-server actually builds from the model's own
Jinja chat template.

SmolLM2-360M-Instruct uses ChatML, so `run_test(..., ["chatml"])` and the
server's POST /apply-template must agree byte for byte. That is an exact
assertion about formatting, not about output quality.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from chat_template_tester.tester import default_registry, run_test

SYSTEM = "You are helpful."
USER = "Hi"


def _apply_template(base_url: str, messages: list[dict]) -> str:
    """Ask the server to render messages with the model's own template."""
    req = urllib.request.Request(
        f"{base_url}/apply-template",
        data=json.dumps({"messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["prompt"]


@pytest.fixture(scope="module")
def props(chat_base_url):
    with urllib.request.urlopen(f"{chat_base_url}/props", timeout=10) as r:
        return json.loads(r.read().decode())


@pytest.mark.live
def test_server_model_really_uses_chatml(props):
    """Guard the premise of the byte-equality tests below."""
    assert props["bos_token"] == "<|im_start|>"
    assert props["eos_token"] == "<|im_end|>"
    assert "<|im_start|>" in props["chat_template"]


@pytest.mark.live
@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER}],
            id="system+user",
        ),
        pytest.param(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "One"},
             {"role": "assistant", "content": "Two"},
             {"role": "user", "content": "Three"}],
            id="multi-turn",
        ),
        pytest.param(
            [{"role": "system", "content": "Line one.\nLine two."},
             {"role": "user", "content": "  padded  "}],
            id="whitespace-and-newlines",
        ),
    ],
)
def test_chatml_render_matches_server_apply_template(chat_base_url, messages):
    """Our chatml rendering is byte-identical to the server's."""
    ours = run_test(messages, template_names=["chatml"]).results[0]
    assert ours.error is None
    theirs = _apply_template(chat_base_url, messages)
    assert ours.formatted == theirs


@pytest.mark.live
def test_generation_prompt_is_the_only_difference_when_disabled(chat_base_url):
    """add_generation_prompt=False drops exactly the assistant header."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER}]
    theirs = _apply_template(chat_base_url, messages)
    ours = run_test(
        messages, template_names=["chatml"], add_generation_prompt=False
    ).results[0]
    assert theirs == ours.formatted + "<|im_start|>assistant\n"


DEFAULT_SYSTEM = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
)


@pytest.mark.live
def test_server_injects_a_default_system_turn_when_none_is_given(chat_base_url):
    """SmolLM2's Jinja template supplies its own system message.

    When the first message is not a system turn, the model's template
    prepends a fixed system block. A client-side ChatML renderer cannot
    know that string, so byte-equality with /apply-template only holds
    when the caller supplies a system message of their own.

    This is model-template behaviour, not a formatter defect - asserted
    here so the difference is pinned rather than mistaken for drift.
    """
    messages = [{"role": "user", "content": USER}]
    theirs = _apply_template(chat_base_url, messages)
    ours = run_test(messages, template_names=["chatml"]).results[0]

    assert DEFAULT_SYSTEM in theirs
    assert DEFAULT_SYSTEM not in ours.formatted
    # The server's output is ours with the default system block prepended.
    injected = f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
    assert theirs == injected + ours.formatted

    # Supplying any system message suppresses the injection.
    with_system = [{"role": "system", "content": SYSTEM}] + messages
    assert DEFAULT_SYSTEM not in _apply_template(chat_base_url, with_system)


@pytest.mark.live
def test_rendered_prompt_is_accepted_verbatim_by_the_server(chat_base_url):
    """Feed our own rendering to /completion as a raw prompt.

    If the formatting were wrong the model would still emit something, so
    this asserts only that the round trip is accepted and that the model
    stops on the template's own stop token rather than running to the cap.
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Say hi."}]
    rendered = run_test(messages, template_names=["chatml"]).results[0].formatted

    body = {
        "prompt": rendered,
        "n_predict": 24,
        "temperature": 0.0,
        "stop": ["<|im_end|>"],
    }
    req = urllib.request.Request(
        f"{chat_base_url}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())

    assert data["content"] is not None
    # "eos" or "word" both mean the template's stop condition fired;
    # "limit" would mean the model never found a turn boundary.
    assert data["stop_type"] in {"eos", "word"}, data["stop_type"]


@pytest.mark.live
def test_token_estimate_is_only_an_estimate(chat_base_url):
    """estimate_tokens() is chars//4 and does not match the real tokenizer.

    Recorded so the estimate is never mistaken for a measurement.
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER}]
    result = run_test(messages, template_names=["chatml"]).results[0]

    req = urllib.request.Request(
        f"{chat_base_url}/tokenize",
        data=json.dumps({"content": result.formatted}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        real = len(json.loads(r.read().decode())["tokens"])

    assert real > 0
    assert result.estimated_tokens > 0
    # Same order of magnitude, but not equal - it is a heuristic.
    assert 0.25 * real <= result.estimated_tokens <= 4 * real


@pytest.mark.live
def test_non_chatml_templates_differ_from_this_model(chat_base_url):
    """Templates for other model families must NOT match SmolLM2's.

    Guards against a formatter that silently renders everything the same.
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER}]
    theirs = _apply_template(chat_base_url, messages)

    run = run_test(messages)
    by_name = {r.template_name: r for r in run.results}
    assert by_name["chatml"].formatted == theirs

    for name in ("llama3", "llama2", "mistral", "gemma", "vicuna"):
        assert by_name[name].formatted != theirs, f"{name} collided with chatml"


@pytest.mark.live
def test_qwen25_is_chatml_compatible(chat_base_url):
    """qwen2.5 is a ChatML variant, so it should also match.

    If it ever stops matching, the registry has diverged from reality.
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER}]
    theirs = _apply_template(chat_base_url, messages)
    ours = run_test(messages, template_names=["qwen2.5"]).results[0]
    assert ours.formatted == theirs


@pytest.mark.live
def test_tool_calling_is_unsupported_by_this_chat_template(props):
    """SmolLM2's template has no tool/function-calling support.

    The registry exposes templates that describe tool use, but this model
    cannot exercise them, so any tool-calling assertion here would be
    meaningless. Pinned rather than skipped silently.
    """
    caps = props["chat_template_caps"]
    assert caps["supports_tools"] is False
    assert caps["supports_tool_calls"] is False
    # System role, which the byte-equality tests rely on, IS supported.
    assert caps["supports_system_role"] is True
