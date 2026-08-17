"""Campaign-facing coordinator for BoundedExpertIteration@1.

This adapter binds the generate-check-retain-refill-train-qualify loop to
``IRLearningCampaign@1`` operations.  It does not import the self-improvement
package (DAG-forbidden) and does not invent a second refill controller.
Refill decisions are either the loop's local bound policy or an injected
``refill_decide`` port — tests wire ``CampaignRefillController.decide``.

Promote is never invoked from this surface.  Qualification is compare-only.
Curriculum revisions are the only result identity the coordinator emits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..proof.bounded_expert_iteration import (
    BOUNDED_EXPERT_ITERATION_INTERFACE,
    BoundedExpertIteration,
    CurriculumRevision,
    ExpertIterationBounds,
    ExpertIterationExample,
    ExpertIterationRefillCandidate,
    ExpertIterationResult,
    ExpertIterationStopReason,
    HiddenTestFeedbackError,
    CheckpointSelfPromotionError,
)
from ..proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)
from .ir_learning_campaign import (
    CampaignOperationKind,
    CampaignOperationReceipt,
    CampaignOperationStatus,
    IRLearningCampaign,
    compare_campaign,
    execute_campaign_operation,
    proof_replay_campaign,
    refill_campaign,
)
from .ir_learning_campaign_contracts import IRLearningCampaignValidationError


EXPERT_ITERATION_CAMPAIGN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-campaign@1"
)
EXPERT_ITERATION_CAMPAIGN_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-campaign-receipt@1"
)


class ExpertIterationCampaignError(IRLearningCampaignValidationError):
    """Campaign-facing expert-iteration request was refused."""


def _campaign(value: IRLearningCampaign | Mapping[str, Any]) -> IRLearningCampaign:
    if isinstance(value, IRLearningCampaign):
        return value
    if isinstance(value, Mapping):
        return IRLearningCampaign.from_dict(value)
    raise ContractValidationError("campaign must be an IRLearningCampaign or mapping")


def refill_candidates_from_result(
    result: ExpertIterationResult,
) -> tuple[dict[str, Any], ...]:
    """Project loop residuals into campaign-refill candidate mappings."""

    if result.refill_candidates:
        return tuple(item.to_dict() for item in result.refill_candidates)
    latest = result.latest_curriculum_revision
    if latest is None:
        return ()
    return tuple(
        ExpertIterationRefillCandidate(
            candidate_id=example_id,
            trigger="proof_residual",
            residual_count=1,
            curriculum_key=example_id,
        ).to_dict()
        for example_id in latest.residual_example_ids
    )


def refill_history_from_result(result: ExpertIterationResult) -> dict[str, Any]:
    """Deterministic refill history projection used by the injected port."""

    latest = result.latest_curriculum_revision
    return {
        "refill_rounds": len(result.receipts),
        "open_work": len(latest.residual_example_ids) if latest is not None else 0,
        "no_progress_streak": result.no_progress_streak,
        "last_progress_identity": (
            result.retained_attempts[-1].example_id if result.retained_attempts else ""
        ),
        "curriculum_repetitions": {
            item.example_id: 1 for item in result.attempts if not item.retained
        },
    }


@dataclass(frozen=True)
class ExpertIterationCampaignReceipt:
    """Campaign-bound expert-iteration result. Never a promotion pointer."""

    campaign_id: str
    campaign_revision: str
    result: ExpertIterationResult
    refill_operation: CampaignOperationReceipt | None = None
    proof_replay_operation: CampaignOperationReceipt | None = None
    compare_operation: CampaignOperationReceipt | None = None
    promote_refused: bool = True

    @property
    def curriculum_revision(self) -> CurriculumRevision | None:
        return self.result.latest_curriculum_revision

    @property
    def result_identity(self) -> str:
        latest = self.curriculum_revision
        return latest.revision_id if latest is not None else ""

    @property
    def promotion_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": EXPERT_ITERATION_CAMPAIGN_RECEIPT_SCHEMA,
            "interface": BOUNDED_EXPERT_ITERATION_INTERFACE,
            "campaign_id": self.campaign_id,
            "campaign_revision": self.campaign_revision,
            "result": self.result.to_dict(),
            "curriculum_revision_id": self.result_identity,
            "result_identity": self.result_identity,
            "refill_operation": (
                self.refill_operation.to_dict()
                if self.refill_operation is not None
                else None
            ),
            "proof_replay_operation": (
                self.proof_replay_operation.to_dict()
                if self.proof_replay_operation is not None
                else None
            ),
            "compare_operation": (
                self.compare_operation.to_dict()
                if self.compare_operation is not None
                else None
            ),
            "promote_refused": True,
            "promotion_authority": False,
            "stop_reason": self.result.stop_reason.value,
            "complete": self.result.complete,
        }
        payload["receipt_id"] = content_identity(
            {key: value for key, value in payload.items() if key != "receipt_id"}
        )
        return payload


def run_expert_iteration_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    examples: Sequence[ExpertIterationExample | Mapping[str, Any]],
    *,
    bounds: ExpertIterationBounds | Mapping[str, Any] | None = None,
    state_path: Path | str | None = None,
    stage_runner: Callable[..., Mapping[str, Any] | None] | None = None,
    refill_decide: Callable[..., Any] | None = None,
    checkpoint_id: str = "",
    caller: str = "operator:expert-iteration",
    halt_after_stage: str | None = None,
    halt_after_round: int | None = None,
) -> ExpertIterationCampaignReceipt:
    """Run the bounded loop and bind refill/replay/compare campaign operations.

    ``CampaignOperationKind.PROMOTE`` is never executed.  A caller that asks
    this surface to promote is refused with ``CheckpointSelfPromotionError``.
    """

    selected = _campaign(campaign)
    if selected.metadata.get("self_promote") or selected.metadata.get("promotion"):
        raise CheckpointSelfPromotionError(
            "expert-iteration campaign metadata cannot carry promotion authority"
        )
    loop = BoundedExpertIteration(
        campaign_id=selected.campaign_id,
        bounds=bounds,
        state_path=state_path,
        stage_runner=stage_runner,
        refill_decide=refill_decide,
        checkpoint_id=checkpoint_id or selected.campaign_revision,
    )
    try:
        result = loop.run(
            examples,
            halt_after_stage=halt_after_stage,
            halt_after_round=halt_after_round,
        )
    except HiddenTestFeedbackError:
        raise
    refill_op = refill_campaign(selected, caller=caller)
    replay_op = proof_replay_campaign(selected, caller=caller)
    compare_op = compare_campaign(selected, caller=caller)
    if result.stop_reason is ExpertIterationStopReason.COMPLETED:
        # Qualification is compare-only.  Promote remains a separate,
        # independently gated campaign operation that this coordinator
        # will not invoke.
        execute_campaign_operation(
            CampaignOperationKind.REPORT,
            selected,
            caller=caller,
        )
    return ExpertIterationCampaignReceipt(
        campaign_id=selected.campaign_id,
        campaign_revision=selected.campaign_revision,
        result=result,
        refill_operation=refill_op,
        proof_replay_operation=replay_op,
        compare_operation=compare_op,
        promote_refused=True,
    )


def refuse_expert_iteration_promotion(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:expert-iteration",
    task_id: str = "",
) -> CampaignOperationReceipt:
    """Fail closed if a caller tries to treat the loop as a promoter."""

    raise CheckpointSelfPromotionError(
        "bounded expert iteration cannot self-promote a checkpoint"
    )


def campaign_operation_status_for_stop(
    stop: ExpertIterationStopReason,
) -> CampaignOperationStatus:
    if stop is ExpertIterationStopReason.COMPLETED:
        return CampaignOperationStatus.SUCCEEDED
    if stop in {
        ExpertIterationStopReason.NO_PROGRESS,
        ExpertIterationStopReason.REPETITION_BOUNDED,
        ExpertIterationStopReason.ROUND_BOUNDED,
        ExpertIterationStopReason.EXHAUSTED,
        ExpertIterationStopReason.CALL_BOUNDED,
        ExpertIterationStopReason.CANDIDATE_BOUNDED,
        ExpertIterationStopReason.DEPTH_BOUNDED,
        ExpertIterationStopReason.SOLVER_TIME_BOUNDED,
    }:
        return CampaignOperationStatus.BLOCKED
    return CampaignOperationStatus.REJECTED


__all__ = (
    "EXPERT_ITERATION_CAMPAIGN_RECEIPT_SCHEMA",
    "EXPERT_ITERATION_CAMPAIGN_SCHEMA",
    "ExpertIterationCampaignError",
    "ExpertIterationCampaignReceipt",
    "campaign_operation_status_for_stop",
    "refill_candidates_from_result",
    "refill_history_from_result",
    "refuse_expert_iteration_promotion",
    "run_expert_iteration_campaign",
)
