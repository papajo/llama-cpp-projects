"""Tests for conditional branching agent."""

import pytest

from conditional_branching.core import (
    ActionStep,
    Branch,
    Condition,
    ConditionalAgent,
    ConditionalWorkflowResult,
)


def inc(state: dict) -> dict:
    return {"count": state.get("count", 0) + 1}


def double(state: dict) -> dict:
    return {"count": state.get("count", 0) * 2}


def add_label(state: dict) -> dict:
    return {"label": f"v{state.get('count', 0)}"}


class TestCondition:
    def test_evaluate_true(self):
        c = Condition("is_high", predicate=lambda s: s.get("count", 0) > 10)
        assert c.evaluate({"count": 20}) is True
        assert c.evaluate({"count": 5}) is False

    def test_name(self):
        c = Condition("test_cond", predicate=lambda s: True)
        assert c.name == "test_cond"


class TestBranch:
    def test_default_flag(self):
        c = Condition("always", predicate=lambda s: True)
        b = Branch(name="main", condition=c, is_default=True)
        assert b.is_default is True

    def test_step_count(self):
        c = Condition("always", predicate=lambda s: True)
        b = Branch(name="b", condition=c, steps=[ActionStep("a", inc)])
        assert b.step_count == 1


class TestConditionalAgent:
    def test_pre_steps_only(self):
        agent = ConditionalAgent(pre_steps=[ActionStep("inc", inc)])
        result = agent.run({"count": 0})
        assert result.final_state == {"count": 1}
        assert result.num_steps == 1

    def test_branch_taken(self):
        high_cond = Condition("high", predicate=lambda s: s.get("count", 0) > 5)
        low_cond = Condition("low", predicate=lambda s: s.get("count", 0) <= 5)

        agent = ConditionalAgent(
            pre_steps=[ActionStep("inc_x3", lambda s: {"count": s.get("count", 0) + 3})],
            branches=[
                Branch(name="high_branch", condition=high_cond,
                       steps=[ActionStep("double", double)]),
                Branch(name="low_branch", condition=low_cond,
                       steps=[ActionStep("triple", lambda s: {"count": s["count"] * 3})]),
            ],
        )
        result = agent.run({"count": 0})
        # pre: 0 → 3, low branch matches (3 <= 5): 3 * 3 = 9
        assert result.final_state == {"count": 9}
        assert result.branch_taken == "low_branch"

    def test_no_match_uses_default(self):
        high_cond = Condition("high", predicate=lambda s: s.get("count", 0) > 100)
        default_cond = Condition("default", predicate=lambda s: True)

        agent = ConditionalAgent(
            pre_steps=[ActionStep("start", inc)],
            branches=[
                Branch(name="high", condition=high_cond,
                       steps=[ActionStep("double", double)]),
                Branch(name="fallback", condition=default_cond, is_default=True,
                       steps=[ActionStep("add_ten", lambda s: {"count": s["count"] + 10})]),
            ],
        )
        result = agent.run({"count": 0})
        # pre: 0 → 1, high doesn't match (1 < 100), default taken: 1 + 10 = 11
        assert result.final_state == {"count": 11}
        assert result.branch_taken == "fallback"

    def test_no_match_no_default(self):
        high_cond = Condition("high", predicate=lambda s: s.get("count", 0) > 100)

        agent = ConditionalAgent(
            pre_steps=[ActionStep("start", inc)],
            branches=[
                Branch(name="high", condition=high_cond,
                       steps=[ActionStep("double", double)]),
            ],
        )
        result = agent.run({"count": 0})
        # No branch matches, no default — just pre-steps execute
        assert result.final_state == {"count": 1}
        assert result.branch_taken is None

    def test_post_steps(self):
        cond = Condition("always", predicate=lambda s: True)

        agent = ConditionalAgent(
            pre_steps=[ActionStep("pre", inc)],
            branches=[
                Branch(name="main", condition=cond,
                       steps=[ActionStep("double", double)]),
            ],
            post_steps=[ActionStep("label", add_label)],
        )
        result = agent.run({"count": 1})
        # pre: 1 → 2, branch: 2 → 4, post: label = "v4"
        assert result.final_state == {"count": 4, "label": "v4"}
        assert result.num_steps == 3

    def test_first_matching_branch_wins(self):
        c1 = Condition("c1", predicate=lambda s: True)
        c2 = Condition("c2", predicate=lambda s: True)

        agent = ConditionalAgent(
            branches=[
                Branch(name="first", condition=c1,
                       steps=[ActionStep("inc", inc)]),
                Branch(name="second", condition=c2,
                       steps=[ActionStep("double", double)]),
            ],
        )
        result = agent.run({"count": 0})
        assert result.branch_taken == "first"
        assert result.final_state == {"count": 1}  # inc, not double

    def test_step_failure_in_pre(self):
        def fail(state):
            raise ValueError("boom")

        agent = ConditionalAgent(pre_steps=[ActionStep("fail", fail)])
        result = agent.run()
        assert result.error == "boom"
        assert result.steps[0].error == "boom"

    def test_step_failure_in_branch(self):
        def fail(state):
            raise ValueError("branch fail")

        cond = Condition("always", predicate=lambda s: True)
        agent = ConditionalAgent(
            branches=[
                Branch(name="main", condition=cond,
                       steps=[ActionStep("fail", fail)]),
            ],
        )
        result = agent.run()
        assert result.error == "branch fail"

    def test_step_failure_in_post(self):
        def fail(state):
            raise ValueError("post fail")

        cond = Condition("always", predicate=lambda s: True)
        agent = ConditionalAgent(
            branches=[Branch(name="main", condition=cond)],
            post_steps=[ActionStep("fail", fail)],
        )
        result = agent.run()
        assert result.error == "post fail"

    def test_branch_step_results_tracking(self):
        c = Condition("always", predicate=lambda s: True)
        agent = ConditionalAgent(
            branches=[
                Branch(name="b1", condition=c,
                       steps=[ActionStep("s1", inc), ActionStep("s2", inc)]),
            ],
        )
        result = agent.run({"count": 0})
        assert result.num_steps == 2
        assert result.steps[0].branch == "b1"
        assert result.steps[1].branch == "b1"

    def test_empty_workflow(self):
        agent = ConditionalAgent()
        result = agent.run()
        assert result.num_steps == 0
        assert result.final_state == {}

    def test_empty_initial_state(self):
        cond = Condition("always", predicate=lambda s: True)
        agent = ConditionalAgent(
            branches=[Branch(name="b", condition=cond,
                             steps=[ActionStep("inc", inc)])],
        )
        result = agent.run()
        assert result.final_state == {"count": 1}
