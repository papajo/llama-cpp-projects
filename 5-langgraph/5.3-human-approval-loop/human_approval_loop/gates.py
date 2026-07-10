"""Human-in-the-loop approval gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class Decision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


@dataclass
class ApprovalRequest:
    """Represents a pending human approval."""

    gate_id: str
    step_name: str
    context: Dict[str, Any] = field(default_factory=dict)
    proposed_action: str = ""
    details: str = ""


@dataclass
class ApprovalDecision:
    """The human's decision on an approval request."""

    decision: Decision
    comment: str = ""
    modified_action: str = ""      # used when decision == MODIFY
    modified_state: Dict[str, Any] = field(default_factory=dict)


class Approver:
    """Interface for making approval decisions.

    The default implementation always approves. Subclass or mock for
    testing.
    """

    def review(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(decision=Decision.APPROVE)


@dataclass
class ApprovalGate:
    """A workflow step that requires human approval before proceeding."""

    step_name: str
    function: Callable[[Dict[str, Any]], Dict[str, Any]]
    description: str = ""
    timeout_seconds: float = 0     # 0 = no timeout

    def prepare_request(self, current_state: Dict[str, Any]) -> ApprovalRequest:
        return ApprovalRequest(
            gate_id=f"gate-{self.step_name}",
            step_name=self.step_name,
            context=dict(current_state),
            proposed_action=self.description,
            details=f"Step: {self.step_name}\nDescription: {self.description}",
        )


@dataclass
class AutoStep:
    """A normal (non-gated) auto-executing step."""

    step_name: str
    function: Callable[[Dict[str, Any]], Dict[str, Any]]
    description: str = ""


WorkflowItem = Tuple[str, Any]  # ("auto", AutoStep) | ("gate", ApprovalGate)


@dataclass
class GateResult:
    gate_id: str
    step_name: str
    decision: Decision
    comment: str = ""
    modified: bool = False


@dataclass
class StepRecord:
    step_name: str
    step_index: int
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    gate_result: Optional[GateResult] = None
    error: Optional[str] = None


@dataclass
class ApprovalWorkflowResult:
    steps: List[StepRecord] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    all_approved: bool = True
    num_gates: int = 0
    num_approved: int = 0
    num_rejected: int = 0
    num_modified: int = 0
    error: Optional[str] = None

    @property
    def num_steps(self) -> int:
        return len(self.steps)


class HumanApprovalAgent:
    """Agent that executes steps but pauses at approval gates.

    When the agent encounters an ``ApprovalGate``, it creates an
    ``ApprovalRequest`` and calls the ``Approver.review()`` method.
    Based on the decision:
      - APPROVE: continue with proposed action
      - REJECT: skip the step (keep state unchanged)
      - MODIFY: apply the human's alternative state
    """

    def __init__(self, steps: List[WorkflowItem], approver: Optional[Approver] = None):
        self.steps = steps
        self.approver = approver or Approver()
        self._state: Dict[str, Any] = {}

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)

    def run(self, initial_state: Optional[Dict[str, Any]] = None) -> ApprovalWorkflowResult:
        self._state = dict(initial_state or {})
        result = ApprovalWorkflowResult()

        for idx, (kind, item) in enumerate(self.steps):
            before = dict(self._state)

            if kind == "auto":
                step: AutoStep = item
                try:
                    output = step.function(before)
                    self._state.update(output)
                except Exception as e:
                    rec = StepRecord(
                        step_name=step.step_name,
                        step_index=idx,
                        state_before=before,
                        state_after=dict(self._state),
                        error=str(e),
                    )
                    result.steps.append(rec)
                    result.error = str(e)
                    result.final_state = dict(self._state)
                    return result

                rec = StepRecord(
                    step_name=step.step_name,
                    step_index=idx,
                    state_before=before,
                    state_after=dict(self._state),
                )

            elif kind == "gate":
                gate: ApprovalGate = item
                request = gate.prepare_request(before)
                decision = self.approver.review(request)
                result.num_gates += 1

                if decision.decision == Decision.APPROVE:
                    result.num_approved += 1
                    try:
                        output = gate.function(before)
                        self._state.update(output)
                    except Exception as e:
                        rec = StepRecord(
                            step_name=gate.step_name,
                            step_index=idx,
                            state_before=before,
                            state_after=dict(self._state),
                            gate_result=GateResult(
                                gate_id=request.gate_id,
                                step_name=gate.step_name,
                                decision=Decision.APPROVE,
                            ),
                            error=str(e),
                        )
                        result.steps.append(rec)
                        result.error = str(e)
                        result.final_state = dict(self._state)
                        return result

                    rec = StepRecord(
                        step_name=gate.step_name,
                        step_index=idx,
                        state_before=before,
                        state_after=dict(self._state),
                        gate_result=GateResult(
                            gate_id=request.gate_id,
                            step_name=gate.step_name,
                            decision=Decision.APPROVE,
                        ),
                    )

                elif decision.decision == Decision.REJECT:
                    result.num_rejected += 1
                    result.all_approved = False
                    rec = StepRecord(
                        step_name=gate.step_name,
                        step_index=idx,
                        state_before=before,
                        state_after=before,  # unchanged
                        gate_result=GateResult(
                            gate_id=request.gate_id,
                            step_name=gate.step_name,
                            decision=Decision.REJECT,
                            comment=decision.comment,
                        ),
                    )

                else:  # MODIFY
                    result.num_modified += 1
                    result.all_approved = False
                    if decision.modified_state:
                        self._state.update(decision.modified_state)
                    rec = StepRecord(
                        step_name=gate.step_name,
                        step_index=idx,
                        state_before=before,
                        state_after=dict(self._state),
                        gate_result=GateResult(
                            gate_id=request.gate_id,
                            step_name=gate.step_name,
                            decision=Decision.MODIFY,
                            comment=decision.comment,
                            modified=True,
                        ),
                    )

            else:
                raise ValueError(f"Unknown step kind: {kind}")

            result.steps.append(rec)

        result.final_state = dict(self._state)
        return result
