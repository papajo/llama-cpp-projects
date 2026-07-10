"""Tests for approval gates and agent."""

import pytest

from human_approval_loop.gates import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    Approver,
    AutoStep,
    Decision,
    HumanApprovalAgent,
    WorkflowItem,
)


def inc(state: dict) -> dict:
    return {"count": state.get("count", 0) + 1}


class TestApprovalGate:
    def test_prepare_request(self):
        gate = ApprovalGate(step_name="review", function=inc, description="Increment count")
        req = gate.prepare_request({"count": 5})
        assert req.gate_id == "gate-review"
        assert req.step_name == "review"
        assert req.context == {"count": 5}
        assert "Increment count" in req.proposed_action


class TestHumanApprovalAgent:
    def test_auto_steps_run(self):
        agent = HumanApprovalAgent([
            ("auto", AutoStep("inc1", inc)),
            ("auto", AutoStep("inc2", inc)),
        ])
        result = agent.run({"count": 0})
        assert result.num_steps == 2
        assert result.final_state == {"count": 2}
        assert result.all_approved is True

    def test_gate_approved_runs(self):
        agent = HumanApprovalAgent([
            ("gate", ApprovalGate("approve-inc", inc, "Increment")),
        ])
        result = agent.run({"count": 0})
        assert result.final_state == {"count": 1}
        assert result.num_gates == 1
        assert result.num_approved == 1
        assert result.all_approved is True

    def test_gate_rejected_skips(self):
        class Rejector(Approver):
            def review(self, request):
                return ApprovalDecision(decision=Decision.REJECT, comment="Not now")

        agent = HumanApprovalAgent(
            [("gate", ApprovalGate("reject-inc", inc, "Increment"))],
            approver=Rejector(),
        )
        result = agent.run({"count": 0})
        assert result.final_state == {"count": 0}  # unchanged
        assert result.num_gates == 1
        assert result.num_rejected == 1
        assert result.all_approved is False
        assert result.steps[0].gate_result is not None
        assert result.steps[0].gate_result.comment == "Not now"

    def test_gate_modified(self):
        class Modifier(Approver):
            def review(self, request):
                return ApprovalDecision(
                    decision=Decision.MODIFY,
                    comment="Use 42 instead",
                    modified_state={"count": 42},
                )

        agent = HumanApprovalAgent(
            [("gate", ApprovalGate("mod-inc", inc, "Increment"))],
            approver=Modifier(),
        )
        result = agent.run({"count": 0})
        assert result.final_state == {"count": 42}
        assert result.num_modified == 1
        assert result.all_approved is False
        assert result.steps[0].gate_result.modified is True

    def test_mixed_auto_and_gates(self):
        agent = HumanApprovalAgent([
            ("auto", AutoStep("inc1", inc)),
            ("gate", ApprovalGate("approve-inc", inc, "Increment")),
            ("auto", AutoStep("inc3", inc)),
        ])
        result = agent.run({"count": 0})
        assert result.final_state == {"count": 3}
        assert result.num_steps == 3

    def test_rejected_gate_skips_execution(self):
        """When rejected, the gate's function should NOT run."""
        executed = [False]

        def tracked_inc(state):
            executed[0] = True
            return {"count": state.get("count", 0) + 1}

        class Rejector(Approver):
            def review(self, request):
                return ApprovalDecision(decision=Decision.REJECT)

        agent = HumanApprovalAgent(
            [("gate", ApprovalGate("rej", tracked_inc, "Tracked"))],
            approver=Rejector(),
        )
        agent.run()
        assert executed[0] is False

    def test_gate_timeout_property(self):
        gate = ApprovalGate(step_name="slow", function=inc, timeout_seconds=30.0)
        assert gate.timeout_seconds == 30.0

    def test_step_failure(self):
        def fail(state):
            raise ValueError("oops")

        agent = HumanApprovalAgent([
            ("auto", AutoStep("good", inc)),
            ("auto", AutoStep("bad", fail)),
        ])
        result = agent.run({"count": 0})
        assert result.error == "oops"
        assert result.num_steps == 2
        assert result.steps[1].error == "oops"

    def test_gate_failure(self):
        def fail(state):
            raise ValueError("gate fail")

        agent = HumanApprovalAgent([
            ("gate", ApprovalGate("fail-gate", fail, "Will fail")),
        ])
        result = agent.run()
        assert result.error == "gate fail"
        assert result.steps[0].error == "gate fail"

    def test_empty_steps(self):
        agent = HumanApprovalAgent([])
        result = agent.run()
        assert result.num_steps == 0
        assert result.final_state == {}

    def test_initial_state_empty(self):
        agent = HumanApprovalAgent([
            ("auto", AutoStep("inc", inc)),
        ])
        result = agent.run()
        assert result.final_state == {"count": 1}
