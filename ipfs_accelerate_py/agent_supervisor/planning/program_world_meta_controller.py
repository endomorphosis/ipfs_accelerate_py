"""SAWM-032 Autonomous Meta-Controller adapter for program-world actions.

Cascade is exact reuse → proof/receipt → procedure → static/symbolic →
retrieval → specialist → local model → remote model → human. Validation
reserves cannot be spent. Actions are proposal-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class CognitiveAction(str, Enum):
    EXACT_REUSE = "exact_reuse"
    PROOF_RECEIPT = "proof_receipt"
    PROCEDURE = "procedure"
    STATIC_SYMBOLIC = "static_symbolic"
    RETRIEVAL = "retrieval"
    SPECIALIST = "specialist"
    LOCAL_MODEL = "local_model"
    REMOTE_MODEL = "remote_model"
    HUMAN_REVIEW = "human_review"


CASCADE: tuple[CognitiveAction, ...] = tuple(CognitiveAction)


@dataclass(frozen=True, slots=True)
class CognitiveBudget:
    remaining_validation_reserve: float
    model_calls_remaining: int


@dataclass(frozen=True, slots=True)
class InferenceCascadeReceipt:
    action: CognitiveAction
    residual_question: str
    proposal_only: bool = True
    completion_authority: bool = False


class ProgramWorldMetaControllerError(ValueError):
    """Closed meta-controller contract violation."""


@dataclass
class ProgramWorldMetaControllerAdapter:
    def select_program_world_cognitive_action(
        self, request: Mapping[str, Any]
    ) -> InferenceCascadeReceipt:
        reserve = float(request.get("remaining_validation_reserve") or 0.0)
        if reserve < 0:
            raise ProgramWorldMetaControllerError("validation reserve cannot be negative")
        available = {
            CognitiveAction(name)
            for name in request.get("available_actions") or [item.value for item in CASCADE]
        }
        exhausted = set(request.get("exhausted_actions") or ())
        for action in CASCADE:
            if action.value in exhausted or action not in available:
                continue
            if action in {CognitiveAction.LOCAL_MODEL, CognitiveAction.REMOTE_MODEL}:
                if int(request.get("model_calls_remaining") or 0) <= 0:
                    continue
                if reserve <= 0:
                    continue
            return InferenceCascadeReceipt(
                action=action,
                residual_question=str(request.get("residual_question") or "unnamed residual"),
            )
        return InferenceCascadeReceipt(
            action=CognitiveAction.HUMAN_REVIEW,
            residual_question=str(request.get("residual_question") or "human review required"),
        )

    def explain_program_world_escalation(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        receipt = self.select_program_world_cognitive_action(request)
        return {
            "action": receipt.action.value,
            "residual_question": receipt.residual_question,
            "cascade": [item.value for item in CASCADE],
            "proposal_only": True,
            "completion_authority": False,
        }


def select_program_world_cognitive_action(
    request: Mapping[str, Any],
) -> InferenceCascadeReceipt:
    return ProgramWorldMetaControllerAdapter().select_program_world_cognitive_action(request)


def explain_program_world_escalation(request: Mapping[str, Any]) -> dict[str, Any]:
    return ProgramWorldMetaControllerAdapter().explain_program_world_escalation(request)
