"""Live integration tests for the human-approval loop.

`HumanApprovalAgent` gates arbitrary callables behind an `Approver`. The
offline tests gate synthetic lambdas; these gate step functions that make REAL
llama-server calls. That makes the REJECT path meaningful: a rejected gate must
not call the model at all, which a lambda cannot demonstrate.

Run with:  source env.sh && LLM_LIVE=1 pytest -m live -q
"""

from __future__ import annotations

import pytest

from human_approval_loop.gates import (
    ApprovalDecision,
    ApprovalGate,
    Approver,
    AutoStep,
    Decision,
    HumanApprovalAgent,
)


class ScriptedApprover(Approver):
    """Returns a preset decision per gate id, recording what it was asked."""

    def __init__(self, decisions: dict[str, ApprovalDecision]):
        self.decisions = decisions
        self.seen: list[str] = []

    def review(self, request):
        self.seen.append(request.gate_id)
        return self.decisions.get(
            request.gate_id, ApprovalDecision(decision=Decision.APPROVE)
        )


@pytest.fixture
def llm_call(live_chat):
    """A step fn that really calls the chat server, counting invocations."""
    calls: list[str] = []

    def make(prompt: str, out_key: str):
        def fn(state):
            calls.append(prompt)
            resp = live_chat([{"role": "user", "content": prompt}], max_tokens=16)
            return {out_key: resp["choices"][0]["message"]["content"]}

        return fn

    make.calls = calls
    return make


# ---------------------------------------------------------------------------
# APPROVE
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_approved_gate_calls_the_model(llm_call):
    gate = ApprovalGate(
        step_name="generate",
        function=llm_call("Name a colour.", "colour"),
        description="Ask the model for a colour",
    )
    agent = HumanApprovalAgent([("gate", gate)], approver=Approver())

    result = agent.run({"seed": 1})

    assert result.error is None
    assert result.all_approved is True
    assert (result.num_gates, result.num_approved, result.num_rejected) == (1, 1, 0)
    assert len(llm_call.calls) == 1
    assert result.final_state["colour"].strip()

    rec = result.steps[0]
    assert rec.gate_result is not None
    assert rec.gate_result.decision == Decision.APPROVE
    assert rec.gate_result.gate_id == "gate-generate"


# ---------------------------------------------------------------------------
# REJECT — the model must not be called
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rejected_gate_never_calls_the_model(llm_call):
    gate = ApprovalGate(
        step_name="generate",
        function=llm_call("Name a colour.", "colour"),
        description="Ask the model for a colour",
    )
    approver = ScriptedApprover({
        "gate-generate": ApprovalDecision(
            decision=Decision.REJECT, comment="not now"
        )
    })
    agent = HumanApprovalAgent([("gate", gate)], approver=approver)

    result = agent.run({"seed": 1})

    assert llm_call.calls == [], "a rejected gate must not reach llama-server"
    assert result.num_rejected == 1
    assert result.all_approved is False
    assert result.final_state == {"seed": 1}, "state unchanged"
    assert approver.seen == ["gate-generate"]


# ---------------------------------------------------------------------------
# MODIFY — human output replaces the model's
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_modified_gate_substitutes_human_state(llm_call):
    gate = ApprovalGate(
        step_name="generate",
        function=llm_call("Name a colour.", "colour"),
        description="Ask the model for a colour",
    )
    approver = ScriptedApprover({
        "gate-generate": ApprovalDecision(
            decision=Decision.MODIFY,
            comment="use our brand colour instead",
            modified_state={"colour": "corporate blue"},
        )
    })
    agent = HumanApprovalAgent([("gate", gate)], approver=approver)

    result = agent.run({"seed": 1})

    assert result.num_modified == 1
    assert result.final_state["colour"] == "corporate blue"
    assert llm_call.calls == [], "MODIFY replaces the call, it does not follow it"


# ---------------------------------------------------------------------------
# Mixed workflow
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_auto_step_then_gated_real_call(llm_call):
    """An ungated auto step runs freely; only the gate is reviewed."""
    auto = AutoStep(
        step_name="prepare",
        function=lambda state: {"prepared": True},
    )
    gate = ApprovalGate(
        step_name="generate",
        function=llm_call("Name a colour.", "colour"),
    )
    approver = ScriptedApprover({})
    agent = HumanApprovalAgent(
        [("auto", auto), ("gate", gate)], approver=approver
    )

    result = agent.run({"seed": 1})

    assert result.num_steps == 2
    assert result.num_gates == 1, "the auto step is not gated"
    assert approver.seen == ["gate-generate"]
    assert result.final_state["prepared"] is True
    assert result.final_state["colour"].strip()
    assert len(llm_call.calls) == 1


@pytest.mark.live
def test_real_failure_inside_approved_gate_halts(live_chat):
    """A genuine network error inside an approved gate stops the workflow."""
    import urllib.request

    def unreachable(state):
        urllib.request.urlopen("http://127.0.0.1:1/v1/chat/completions", timeout=5)
        return {"never": "reached"}

    agent = HumanApprovalAgent(
        [("gate", ApprovalGate(step_name="generate", function=unreachable))],
        approver=Approver(),
    )
    result = agent.run({"seed": 1})

    assert result.error is not None
    assert result.num_approved == 1, "it was approved, then it failed"
    assert result.steps[-1].error is not None
    assert "never" not in result.final_state
