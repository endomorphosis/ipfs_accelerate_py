"""O3 prompt handoff into the operational campaign surface.

Prompts may be admitted through the existing prompt-plan gates and then bound
to read/proposal campaign verbs.  They cannot select start/resume/promote/
reject authority, hidden labels, or a new control operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..control.campaign_public_api import (
    OPERATIONAL_CAMPAIGN_API_INTERFACE,
    operational_campaign_control_operation,
    prompt_may_select_campaign_operation,
)
from ..control.control_contracts import Operation
from ..proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)
from ..prompt.prompt_plan_admission import (
    PromptPlanAdmissionPolicy,
    PromptPlanAdmissionResult,
    admit_prompt_plan,
)


CAMPAIGN_PROMPT_HANDOFF_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-prompt-handoff@1"
)
CAMPAIGN_PROMPT_HANDOFF_VERSION: Final = 1

_FORBIDDEN = frozenset(
    {
        "hidden_labels",
        "hidden_label",
        "prompt_authority",
        "prompt_selected_authority",
        "self_promote",
        "self_promotion",
        "secret",
        "secrets",
    }
)


class CampaignPromptHandoffError(ContractValidationError):
    """Prompt handoff tried to expand authority or skip admission."""


def _reject_forbidden(payload: Mapping[str, Any] | None, *, noun: str) -> None:
    if not payload:
        return
    keys = {str(key).strip().lower().replace("-", "_") for key in payload}
    if keys.intersection(_FORBIDDEN):
        raise CampaignPromptHandoffError(
            "%s cannot carry prompt-selected authority or hidden labels" % noun
        )
    nested = payload.get("metadata") if isinstance(payload, Mapping) else None
    if isinstance(nested, Mapping):
        _reject_forbidden(nested, noun="%s metadata" % noun)


def admit_campaign_prompt_handoff(
    graph: Any,
    *,
    repository_tree_id: str,
    ir_request: Any = None,
    workflow_request: Any = None,
    scan_receipt: Any = None,
    policy: PromptPlanAdmissionPolicy | None = None,
    requested_operation: str = "plan",
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit an O3 prompt and bind it to a non-expanding campaign verb.

    Admission never authorizes execution, promotion, or a new ``Operation``.
    The returned record is deterministic for identical admitted graphs.
    """

    _reject_forbidden(parameters, noun="prompt handoff")
    operation_name = str(requested_operation or "plan").strip() or "plan"
    if not prompt_may_select_campaign_operation(operation_name):
        raise CampaignPromptHandoffError(
            "prompt handoff cannot select campaign operation %s" % operation_name
        )
    control = operational_campaign_control_operation(operation_name)
    if control.mutating:
        raise CampaignPromptHandoffError(
            "prompt handoff cannot select a mutating control operation"
        )

    admitted: PromptPlanAdmissionResult = admit_prompt_plan(
        graph,
        repository_tree_id=repository_tree_id,
        ir_request=ir_request,
        workflow_request=workflow_request,
        scan_receipt=scan_receipt,
        policy=policy,
    )
    receipt = admitted.receipt
    payload = {
        "schema": CAMPAIGN_PROMPT_HANDOFF_SCHEMA,
        "version": CAMPAIGN_PROMPT_HANDOFF_VERSION,
        "api": OPERATIONAL_CAMPAIGN_API_INTERFACE,
        "requested_operation": operation_name,
        "control_operation": control.value,
        "admitted": bool(admitted.admitted),
        "authorizes_execution": False,
        "authorizes_promotion": False,
        "expands_control_catalog": False,
        "prompt_selected_authority": False,
        "reason_codes": list(admitted.reason_codes),
        "final_plan_cid": receipt.final_plan_cid if admitted.admitted else "",
        "admission_receipt_id": receipt.receipt_id,
    }
    payload["handoff_id"] = content_identity(payload)
    return payload


def campaign_prompt_handoff_policy() -> dict[str, Any]:
    """Return the closed O3 prompt policy without running admission."""

    return {
        "schema": CAMPAIGN_PROMPT_HANDOFF_SCHEMA,
        "selectable_operations": ("plan", "status", "compare", "report"),
        "forbidden_operations": (
            "create",
            "start",
            "resume",
            "steer",
            "refill",
            "proof-replay",
            "promote",
            "reject",
        ),
        "authorizes_execution": False,
        "authorizes_promotion": False,
        "expands_control_catalog": False,
        "prompt_selected_authority": False,
        "control_plan_operation": Operation.PLAN.value,
    }


__all__ = (
    "CAMPAIGN_PROMPT_HANDOFF_SCHEMA",
    "CAMPAIGN_PROMPT_HANDOFF_VERSION",
    "CampaignPromptHandoffError",
    "admit_campaign_prompt_handoff",
    "campaign_prompt_handoff_policy",
)
