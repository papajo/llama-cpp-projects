"""Live integration tests for conditional branching.

`ConditionalAgent` runs pre-steps, evaluates branch conditions against state,
runs the first matching branch, then post-steps. The offline tests drive it with
synthetic state; here the branch condition is evaluated against state produced
by a REAL llama-server call.

The conditions deliberately test STRUCTURAL properties of the model output
(length, type, presence) rather than meaning. A 360M model's content is not
predictable, so a semantic condition would make the test flaky — that is a
property of the model, not of the branching code.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import pytest

from conditional_branching.core import (
    ActionStep,
    Branch,
    Condition,
    ConditionalAgent,
)


@pytest.fixture
def chat_step(live_chat):
    """An ActionStep that really calls the chat server."""

    def make(
        name: str,
        prompt: str,
        out_key: str,
        max_tokens: int = 16,
        ignore_eos: bool = False,
    ):
        def fn(state):
            kw = {"max_tokens": max_tokens}
            if ignore_eos:
                # Suppress EOS so the token cap is the only stop condition and
                # finish_reason is "length" by construction rather than by luck.
                kw["ignore_eos"] = True
            resp = live_chat([{"role": "user", "content": prompt}], **kw)
            content = resp["choices"][0]["message"]["content"]
            return {
                out_key: content,
                f"{out_key}_len": len(content),
                f"{out_key}_finish": resp["choices"][0]["finish_reason"],
            }

        return ActionStep(name=name, fn=fn)

    return make


def _marker(name: str, key: str, value):
    return ActionStep(name=name, fn=lambda state: {key: value})


# ---------------------------------------------------------------------------
# Branch chosen from real model output
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_branch_selected_from_real_output_length(chat_step):
    """A condition over real output length picks exactly one branch."""
    agent = ConditionalAgent(
        pre_steps=[chat_step("generate", "Name a colour.", "answer")],
        branches=[
            Branch(
                name="nonempty",
                condition=Condition(
                    name="has_text",
                    predicate=lambda s: s.get("answer_len", 0) > 0,
                ),
                steps=[_marker("handle_nonempty", "route", "nonempty")],
            ),
            Branch(
                name="empty",
                condition=Condition(
                    name="no_text",
                    predicate=lambda s: s.get("answer_len", 0) == 0,
                ),
                steps=[_marker("handle_empty", "route", "empty")],
                is_default=True,
            ),
        ],
        post_steps=[_marker("finish", "done", True)],
    )

    result = agent.run({})

    assert result.error is None
    assert result.final_state["answer"].strip()
    # llama-server always returns content for these prompts, so:
    assert result.branch_taken == "nonempty"
    assert result.final_state["route"] == "nonempty"
    assert result.final_state["done"] is True

    # The unmatched branch was evaluated but not executed.
    by_name = {br.branch_name: br for br in result.branch_results}
    assert by_name["nonempty"].condition_met is True
    assert "handle_empty" not in [s.step_name for s in result.steps]


@pytest.mark.live
def test_branch_on_real_finish_reason(chat_step):
    """finish_reason is a real server field; branch on it.

    max_tokens=8 with ignore_eos=True makes the token cap the only stop
    condition, so finish_reason == "length" deterministically. Without
    ignore_eos SmolLM2 sometimes emits EOS inside 8 tokens and returns "stop"
    instead, which made this test flaky (see drift-rag.md entry 6). The offline
    mocks omitted the field entirely (drift-graph.md entry 1).
    """
    agent = ConditionalAgent(
        pre_steps=[
            chat_step(
                "generate",
                "Count from one to one hundred slowly.",
                "answer",
                max_tokens=8,
                ignore_eos=True,
            )
        ],
        branches=[
            Branch(
                name="truncated",
                condition=Condition(
                    name="hit_token_cap",
                    predicate=lambda s: s.get("answer_finish") == "length",
                ),
                steps=[_marker("note_truncation", "truncated", True)],
            ),
            Branch(
                name="complete",
                condition=Condition(
                    name="stopped_naturally",
                    predicate=lambda s: s.get("answer_finish") == "stop",
                ),
                steps=[_marker("note_complete", "truncated", False)],
                is_default=True,
            ),
        ],
    )

    result = agent.run({})

    assert result.final_state["answer_finish"] == "length"
    assert result.branch_taken == "truncated"
    assert result.final_state["truncated"] is True


# ---------------------------------------------------------------------------
# Default branch
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_default_branch_when_nothing_matches(chat_step):
    agent = ConditionalAgent(
        pre_steps=[chat_step("generate", "Name a colour.", "answer")],
        branches=[
            Branch(
                name="impossible",
                condition=Condition(
                    name="never", predicate=lambda s: False
                ),
                steps=[_marker("unreachable", "route", "impossible")],
            ),
            Branch(
                name="fallback",
                condition=Condition(
                    name="default", predicate=lambda s: False
                ),
                steps=[_marker("handle_default", "route", "fallback")],
                is_default=True,
            ),
        ],
    )

    result = agent.run({})

    assert result.branch_taken == "fallback"
    assert result.final_state["route"] == "fallback"


# ---------------------------------------------------------------------------
# Real failure
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_real_failure_in_pre_step_skips_branching():
    import urllib.request

    def unreachable(state):
        urllib.request.urlopen("http://127.0.0.1:1/v1/models", timeout=5)
        return {}

    agent = ConditionalAgent(
        pre_steps=[ActionStep(name="generate", fn=unreachable)],
        branches=[
            Branch(
                name="any",
                condition=Condition(name="always", predicate=lambda s: True),
                steps=[_marker("should_not_run", "route", "any")],
                is_default=True,
            )
        ],
    )

    result = agent.run({})

    assert result.error is not None
    assert result.branch_taken is None
    assert "route" not in result.final_state
    assert result.steps[-1].error is not None
