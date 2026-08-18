"""Create/plan/start/resume/status/steer/refill/proof-replay/compare/promote/reject/report APIs.

These operations are the campaign-layer façade over existing objective,
planning, and control contracts.  They do not start daemons, open network
connections, or mutate the closed control ``Operation`` catalog.  Mutating
calls that would begin leased work fail closed while any required
``RESULT(task)`` identity remains unresolved.  ``start`` and ``resume`` reuse
the existing lifecycle operations and never invent a second scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..control.control_contracts import EffectKind, Operation, OperationAuthority
from ..proof.formal_verification_contracts import ContractValidationError
from .ir_learning_campaign_contracts import (
    CAMPAIGN_OPERATION_CONTROL_MAP,
    IRLearningCampaign,
    LEASE_REQUIRING_OPERATIONS,
    STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS,
    CampaignDependencyProjection,
    CampaignOperationKind,
    CampaignOperationReceipt,
    CampaignOperationRequest,
    CampaignOperationStatus,
    CampaignTaskRevision,
    IRLearningCampaignValidationError,
    assert_campaign_control_parity,
    campaign_control_catalog,
    revise_campaign_task,
)


CampaignServiceError = IRLearningCampaignValidationError


def _campaign(value: IRLearningCampaign | Mapping[str, Any]) -> IRLearningCampaign:
    if isinstance(value, IRLearningCampaign):
        return value
    if isinstance(value, Mapping):
        return IRLearningCampaign.from_dict(value)
    raise ContractValidationError("campaign must be an IRLearningCampaign or mapping")


def _request(
    operation: CampaignOperationKind | str,
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str,
    task_id: str = "",
    dry_run: bool = False,
    idempotency_key: str = "",
    parameters: Mapping[str, Any] | None = None,
) -> CampaignOperationRequest:
    return CampaignOperationRequest(
        operation=operation,
        campaign=_campaign(campaign),
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
        idempotency_key=idempotency_key,
        parameters=parameters or {},
    )


def _effect_kind(request: CampaignOperationRequest) -> EffectKind:
    if request.authority is OperationAuthority.READ:
        return EffectKind.OBSERVE
    if request.authority is OperationAuthority.PROPOSAL:
        return EffectKind.PROPOSE
    if request.operation is CampaignOperationKind.START:
        return EffectKind.START_PROCESS
    if request.operation is CampaignOperationKind.RESUME:
        return EffectKind.LIFECYCLE_TRANSITION
    if request.operation is CampaignOperationKind.PROOF_REPLAY:
        return EffectKind.EXECUTE_VALIDATION
    return EffectKind.WRITE_STATE


def _receipt(
    request: CampaignOperationRequest,
    *,
    status: CampaignOperationStatus,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> CampaignOperationReceipt:
    projection = request.campaign.project_dependencies()
    return CampaignOperationReceipt(
        operation=request.operation,
        status=status,
        campaign_id=request.campaign.campaign_id,
        campaign_revision=request.campaign.campaign_revision,
        control_operation=request.control_operation,
        authority=request.authority,
        projection_id=projection.projection_id,
        lease_eligible_task_ids=projection.lease_eligible_task_ids,
        blocked_task_ids=projection.blocked_task_ids,
        unresolved_result_ids=projection.unresolved_result_ids,
        message=message,
        details=details or {},
    )


def _lease_block_message(request: CampaignOperationRequest) -> str | None:
    if not request.requires_lease:
        return None
    campaign = request.campaign
    if request.task_id:
        revision = campaign.revision_for(request.task_id)
        if not revision.lease_eligible:
            return (
                "task revision binds unresolved dependency outputs; lease is blocked"
            )
        return None
    if campaign.blocked_task_ids:
        return "campaign revision binds unresolved dependency outputs; lease is blocked"
    return None


def execute_campaign_operation(
    operation: CampaignOperationKind | str | CampaignOperationRequest,
    campaign: IRLearningCampaign | Mapping[str, Any] | None = None,
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
    idempotency_key: str = "",
    parameters: Mapping[str, Any] | None = None,
) -> CampaignOperationReceipt:
    """Execute one closed campaign operation without expanding control authority."""

    assert_campaign_control_parity()
    if isinstance(operation, CampaignOperationRequest):
        request = operation
    else:
        if campaign is None:
            raise ContractValidationError("campaign is required")
        request = _request(
            operation,
            campaign,
            caller=caller,
            task_id=task_id,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
            parameters=parameters,
        )
    block = _lease_block_message(request)
    if block is not None:
        return _receipt(
            request,
            status=CampaignOperationStatus.BLOCKED,
            message=block,
            details={
                "lease_required": True,
                "idempotency_key": request.idempotency_key,
                "effect_kind": _effect_kind(request).value,
                "unresolved_result_ids": list(
                    request.campaign.project_dependencies().unresolved_result_ids
                ),
            },
        )
    if request.operation is CampaignOperationKind.PROMOTE and request.task_id:
        task = request.campaign.task_by_id(request.task_id)
        if "hidden" in task.prohibited_effects.casefold():
            return _receipt(
                request,
                status=CampaignOperationStatus.DENIED,
                message="promote cannot use hidden labels or prompt-selected authority",
            )
    projection = request.campaign.project_dependencies()
    return _receipt(
        request,
        status=CampaignOperationStatus.SUCCEEDED,
        message="campaign operation admitted through %s"
        % request.control_operation.value,
        details={
            "control_operation": request.control_operation.value,
            "authority": request.authority.value,
            "dry_run": request.dry_run,
            "idempotency_key": request.idempotency_key,
            "effect_kind": _effect_kind(request).value,
            "expands_control_catalog": False,
            "projection_id": projection.projection_id,
            "action_ids": list(projection.action_ids),
            **{
                key: request.parameters[key]
                for key in (
                    "resume_decision",
                    "stored_binding_id",
                    "requested_binding_id",
                )
                if key in request.parameters
            },
        },
    )


def create_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    dry_run: bool = False,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.CREATE,
        campaign,
        caller=caller,
        dry_run=dry_run,
    )


def plan_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.PLAN,
        campaign,
        caller=caller,
    )


def start_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
    idempotency_key: str = "",
    parameters: Mapping[str, Any] | None = None,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.START,
        campaign,
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
        idempotency_key=idempotency_key,
        parameters=parameters,
    )


def resume_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
    idempotency_key: str = "",
    parameters: Mapping[str, Any] | None = None,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.RESUME,
        campaign,
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
        idempotency_key=idempotency_key,
        parameters=parameters,
    )


def campaign_status(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.STATUS,
        campaign,
        caller=caller,
        task_id=task_id,
    )


def steer_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
    parameters: Mapping[str, Any] | None = None,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.STEER,
        campaign,
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
        parameters=parameters,
    )


def refill_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    dry_run: bool = False,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.REFILL,
        campaign,
        caller=caller,
        dry_run=dry_run,
    )


def proof_replay_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.PROOF_REPLAY,
        campaign,
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
    )


def compare_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.COMPARE,
        campaign,
        caller=caller,
        task_id=task_id,
    )


def promote_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.PROMOTE,
        campaign,
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
    )


def reject_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.REJECT,
        campaign,
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
    )


def report_campaign(
    campaign: IRLearningCampaign | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
) -> CampaignOperationReceipt:
    return execute_campaign_operation(
        CampaignOperationKind.REPORT,
        campaign,
        caller=caller,
    )


CAMPAIGN_OPERATION_HANDLERS = {
    CampaignOperationKind.CREATE: create_campaign,
    CampaignOperationKind.PLAN: plan_campaign,
    CampaignOperationKind.START: start_campaign,
    CampaignOperationKind.RESUME: resume_campaign,
    CampaignOperationKind.STATUS: campaign_status,
    CampaignOperationKind.STEER: steer_campaign,
    CampaignOperationKind.REFILL: refill_campaign,
    CampaignOperationKind.PROOF_REPLAY: proof_replay_campaign,
    CampaignOperationKind.COMPARE: compare_campaign,
    CampaignOperationKind.PROMOTE: promote_campaign,
    CampaignOperationKind.REJECT: reject_campaign,
    CampaignOperationKind.REPORT: report_campaign,
}


__all__ = (
    "CAMPAIGN_OPERATION_CONTROL_MAP",
    "CAMPAIGN_OPERATION_HANDLERS",
    "CampaignDependencyProjection",
    "CampaignOperationKind",
    "CampaignOperationReceipt",
    "CampaignOperationRequest",
    "CampaignServiceError",
    "CampaignTaskRevision",
    "IRLearningCampaign",
    "LEASE_REQUIRING_OPERATIONS",
    "STABLE_OPERATIONAL_CAMPAIGN_OPERATIONS",
    "Operation",
    "OperationAuthority",
    "campaign_control_catalog",
    "campaign_status",
    "compare_campaign",
    "create_campaign",
    "execute_campaign_operation",
    "plan_campaign",
    "promote_campaign",
    "proof_replay_campaign",
    "refill_campaign",
    "reject_campaign",
    "report_campaign",
    "resume_campaign",
    "revise_campaign_task",
    "start_campaign",
    "steer_campaign",
)
