"""Live integration tests for the prompt chaining workbench.

The offline suite drives run_chain through pytest-httpx with canned
bodies. These run the same code path against the real chat server, where
the thing that actually matters is that step N's real output lands in
step N+1's rendered prompt.

max_tokens is kept at 8-24 throughout: PromptChain defaults to 512, which
is far too slow for a CPU-only 360M model across multiple steps.

Assertions are structural - context plumbing, token accounting, error
propagation. What SmolLM2 actually says is never asserted.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live
"""

from __future__ import annotations

import pytest

from prompt_chaining.chain import ChainStep, PromptChain
from prompt_chaining.runner import run_chain


def _chain(*steps: ChainStep, max_tokens: int = 16, **kw) -> PromptChain:
    return PromptChain(
        name=kw.pop("name", "live"),
        steps=list(steps),
        temperature=kw.pop("temperature", 0.0),
        max_tokens=max_tokens,
        **kw,
    )


@pytest.mark.live
def test_single_step_chain(chat_base_url):
    chain = _chain(ChainStep(name="greet", user_prompt="Say hi."))
    result = run_chain(chain, input_text="", base_url=chat_base_url, max_retries=0)

    assert result.all_ok, [s.error for s in result.steps]
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.step_name == "greet"
    assert step.step_index == 1
    assert step.output
    assert step.prompt_tokens > 0
    assert step.completion_tokens > 0
    assert result.total_elapsed_s > 0
    assert result.last_output == step.output


@pytest.mark.live
def test_step_output_feeds_the_next_step(chat_base_url):
    """The core promise of the project, against a real model.

    Step 2's rendered prompt must literally contain step 1's real output.
    """
    chain = _chain(
        ChainStep(name="pick", user_prompt="Name one colour. Answer with one word."),
        ChainStep(name="echo", user_prompt="Repeat this word back: {{step_1}}"),
    )
    result = run_chain(chain, input_text="", base_url=chat_base_url, max_retries=0)

    assert result.all_ok, [s.error for s in result.steps]
    first, second = result.steps
    assert first.output
    assert first.output in second.rendered_user
    assert "{{step_1}}" not in second.rendered_user


@pytest.mark.live
def test_named_reference_resolves_as_well_as_positional(chat_base_url):
    """{{<step name>}} works alongside {{step_N}}."""
    chain = _chain(
        ChainStep(name="pick", user_prompt="Name one colour. One word only."),
        ChainStep(name="echo", user_prompt="Echo: {{pick}}"),
    )
    result = run_chain(chain, input_text="", base_url=chat_base_url, max_retries=0)

    assert result.all_ok, [s.error for s in result.steps]
    assert result.steps[0].output in result.steps[1].rendered_user


@pytest.mark.live
def test_input_text_is_available_to_every_step(chat_base_url):
    chain = _chain(
        ChainStep(name="one", user_prompt="Repeat: {{input}}"),
        ChainStep(name="two", user_prompt="The original was {{input}}. Say ok."),
    )
    result = run_chain(
        chain, input_text="banana", base_url=chat_base_url, max_retries=0
    )
    assert result.all_ok, [s.error for s in result.steps]
    assert "banana" in result.steps[0].rendered_user
    assert "banana" in result.steps[1].rendered_user


@pytest.mark.live
def test_per_step_overrides_beat_chain_defaults(chat_base_url):
    """A step's own max_tokens wins over the chain's."""
    chain = _chain(
        ChainStep(
            name="tight",
            user_prompt="Write a long essay about the sea.",
            max_tokens=8,
        ),
        max_tokens=64,
    )
    result = run_chain(chain, input_text="", base_url=chat_base_url, max_retries=0)
    assert result.all_ok, [s.error for s in result.steps]
    assert result.steps[0].completion_tokens <= 8


@pytest.mark.live
def test_system_prompt_is_rendered_and_sent(chat_base_url):
    chain = _chain(
        ChainStep(
            name="terse",
            system_prompt="You are terse. The topic is {{input}}.",
            user_prompt="Say one word about it.",
        )
    )
    result = run_chain(
        chain, input_text="rain", base_url=chat_base_url, max_retries=0
    )
    assert result.all_ok, [s.error for s in result.steps]
    step = result.steps[0]
    assert step.rendered_system == "You are terse. The topic is rain."
    assert step.output


@pytest.mark.live
def test_three_step_chain_accumulates_context(chat_base_url):
    chain = _chain(
        ChainStep(name="a", user_prompt="Say the word red."),
        ChainStep(name="b", user_prompt="Given {{step_1}}, say the word blue."),
        ChainStep(name="c", user_prompt="You said {{step_1}} then {{step_2}}. Say ok."),
        max_tokens=12,
    )
    result = run_chain(chain, input_text="", base_url=chat_base_url, max_retries=0)

    assert result.all_ok, [s.error for s in result.steps]
    assert len(result.ok_steps) == 3
    assert result.failed_steps == []
    third = result.steps[2]
    assert result.steps[0].output in third.rendered_user
    assert result.steps[1].output in third.rendered_user

    ctx = result.context()
    assert ctx["step_1"] == result.steps[0].output
    assert ctx["a"] == result.steps[0].output
    assert ctx["input"] == ""


@pytest.mark.live
def test_real_error_stops_the_chain(chat_base_url):
    """A step that overflows n_ctx fails, and stop_on_error halts."""
    chain = _chain(
        ChainStep(name="huge", user_prompt="word " * 6000),
        ChainStep(name="never", user_prompt="Say ok."),
        max_tokens=8,
    )
    result = run_chain(
        chain, input_text="", base_url=chat_base_url, max_retries=0,
        stop_on_error=True,
    )

    assert not result.all_ok
    assert len(result.steps) == 1, "chain should have stopped after the failure"
    err = result.steps[0].error
    assert "HTTP 400" in err
    assert "exceed_context_size_error" in err
    assert result.last_output == ""


@pytest.mark.live
def test_stop_on_error_false_continues_past_a_failure(chat_base_url):
    chain = _chain(
        ChainStep(name="huge", user_prompt="word " * 6000),
        ChainStep(name="still_runs", user_prompt="Say ok."),
        max_tokens=8,
    )
    result = run_chain(
        chain, input_text="", base_url=chat_base_url, max_retries=0,
        stop_on_error=False,
    )

    assert len(result.steps) == 2
    assert len(result.failed_steps) == 1
    assert len(result.ok_steps) == 1
    assert result.ok_steps[0].step_name == "still_runs"
    # A failed step contributes nothing to the context.
    assert "huge" not in result.context()


@pytest.mark.live
def test_progress_callback_reports_each_step(chat_base_url):
    seen = []

    def cb(index, total, name, status):
        seen.append((index, total, name, status))

    chain = _chain(
        ChainStep(name="one", user_prompt="Say a."),
        ChainStep(name="two", user_prompt="Say b."),
        max_tokens=8,
    )
    run_chain(
        chain, input_text="", base_url=chat_base_url,
        max_retries=0, progress_cb=cb,
    )

    assert [s[2] for s in seen if s[3] == "running"] == ["one", "two"]
    assert [s[2] for s in seen if s[3] == "ok"] == ["one", "two"]
    assert all(total == 2 for _, total, _, _ in seen)


@pytest.mark.live
def test_real_usage_block_is_richer_than_the_canned_one(chat_base_url):
    """Pin the response envelope the OpenAI-shaped mocks omit.

    The offline fixtures send only
    {"choices":[{"message":{"content":...}}],
     "usage":{"prompt_tokens":N,"completion_tokens":M}}.
    A real llama-server response additionally carries finish_reason, index,
    message.role, id, model, created, system_fingerprint, a llama.cpp-only
    `timings` block, and usage.total_tokens plus
    usage.prompt_tokens_details.cached_tokens.

    run_chain only reads prompt_tokens/completion_tokens, so the thin mock
    is not a bug - but nothing offline pins the real contract, so this does.
    """
    import json
    import urllib.request

    body = {
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{chat_base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())

    assert {"choices", "usage", "id", "model", "created",
            "object", "system_fingerprint", "timings"} <= set(data)
    assert data["object"] == "chat.completion"

    choice = data["choices"][0]
    assert choice["finish_reason"] in {"stop", "length"}
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"

    usage = data["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert "cached_tokens" in usage["prompt_tokens_details"]

    # llama.cpp-only timing block, absent from every canned fixture.
    assert {"prompt_n", "predicted_n", "predicted_per_second"} <= set(data["timings"])
