"""Execute the O2 operational campaign surface with exact resume.

This adapter is the durable owner of start/resume side effects.  It delegates
campaign verbs to ``IRLearningCampaign@1`` and binds resume to the existing
learning-checkpoint contract.  Compatible resume is exact: lineage must match
and progress may only advance.  The adapter never grants promotion authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..control.campaign_public_api import (
    OPERATIONAL_CAMPAIGN_API_INTERFACE,
    operational_campaign_control_operation,
)
from ..control.control_contracts import Operation
from ..objectives.ir_learning_campaign import (
    CampaignOperationKind,
    CampaignOperationReceipt,
    execute_campaign_operation,
    resume_campaign,
    start_campaign,
)
from ..proof.campaign_proof_replay import assert_campaign_proof_replay_authority
from .learning_checkpoint import (
    IncompatibleResumeError,
    LearningCheckpointBinding,
    LearningCheckpointError,
    resume_decision,
)


OPERATIONAL_CAMPAIGN_RUNTIME_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/operational-campaign-runtime@1"
)


def _binding(value: LearningCheckpointBinding | Mapping[str, Any]) -> LearningCheckpointBinding:
    if isinstance(value, LearningCheckpointBinding):
        return value
    if isinstance(value, Mapping):
        return LearningCheckpointBinding.from_dict(value)
    raise LearningCheckpointError("checkpoint binding must be an object")


def execute_operational_campaign(
    operation: CampaignOperationKind | str,
    campaign: Mapping[str, Any] | Any,
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
    idempotency_key: str = "",
    parameters: Mapping[str, Any] | None = None,
    stored_checkpoint: LearningCheckpointBinding | Mapping[str, Any] | None = None,
    requested_checkpoint: LearningCheckpointBinding | Mapping[str, Any] | None = None,
) -> CampaignOperationReceipt:
    """Execute one O2 campaign operation with auth, lease, and resume gates."""

    kind = (
        operation
        if isinstance(operation, CampaignOperationKind)
        else CampaignOperationKind(str(operation))
    )
    control = operational_campaign_control_operation(kind.value)
    merged: dict[str, Any] = dict(parameters or {})
    if kind is CampaignOperationKind.PROOF_REPLAY:
        assert_campaign_proof_replay_authority(parameters=merged)
    if kind is CampaignOperationKind.RESUME:
        if stored_checkpoint is None or requested_checkpoint is None:
            raise IncompatibleResumeError(
                "resume requires stored and requested checkpoint bindings"
            )
        stored = _binding(stored_checkpoint)
        requested = _binding(requested_checkpoint)
        decision = resume_decision(stored, requested)
        merged = {
            **merged,
            "resume_decision": decision,
            "stored_binding_id": stored.binding_id,
            "requested_binding_id": requested.binding_id,
        }
        receipt = resume_campaign(
            campaign,
            caller=caller,
            task_id=task_id,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
            parameters=merged,
        )
    elif kind is CampaignOperationKind.START:
        receipt = start_campaign(
            campaign,
            caller=caller,
            task_id=task_id,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
            parameters=merged,
        )
    else:
        receipt = execute_campaign_operation(
            kind,
            campaign,
            caller=caller,
            task_id=task_id,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
            parameters=merged,
        )
    if receipt.control_operation is not control:
        raise LearningCheckpointError(
            "operational campaign receipt control operation drifted from %s"
            % control.value
        )
    return receipt


def exact_resume_operational_campaign(
    campaign: Mapping[str, Any] | Any,
    stored_checkpoint: LearningCheckpointBinding | Mapping[str, Any],
    requested_checkpoint: LearningCheckpointBinding | Mapping[str, Any],
    *,
    caller: str = "operator:campaign",
    task_id: str = "",
    dry_run: bool = False,
    idempotency_key: str = "",
) -> CampaignOperationReceipt:
    """Resume only when the requested binding is compatible and exact."""

    return execute_operational_campaign(
        CampaignOperationKind.RESUME,
        campaign,
        caller=caller,
        task_id=task_id,
        dry_run=dry_run,
        idempotency_key=idempotency_key,
        stored_checkpoint=stored_checkpoint,
        requested_checkpoint=requested_checkpoint,
    )


def operational_campaign_runtime_manifest() -> dict[str, Any]:
    """Static runtime publication; does not start a supervisor."""

    return {
        "schema": OPERATIONAL_CAMPAIGN_RUNTIME_SCHEMA,
        "api": OPERATIONAL_CAMPAIGN_API_INTERFACE,
        "start_control_operation": Operation.START.value,
        "resume_control_operation": Operation.RESUME.value,
        "promotion_authority": False,
        "exact_resume": True,
        "expands_control_catalog": False,
    }


__all__ = (
    "OPERATIONAL_CAMPAIGN_RUNTIME_SCHEMA",
    "exact_resume_operational_campaign",
    "execute_operational_campaign",
    "operational_campaign_runtime_manifest",
)
