"""DCR-084 bounded deterministic supervisor self-improvement proposals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..objectives.deterministic_repair_selection import (
    RepairSelectionEvidence,
    RepairSelectionResult,
    select_and_refill_repairs,
)
from ..objectives.repair_authority_projection import (
    RepairAuthorityProjection,
    RepairAuthorityStatus,
)
from ..proof.formal_verification_contracts import content_identity
from ..todo_daemon.deterministic_repair_recovery import RecoveryDecision, RecoveryDisposition
from .operators.registry import OperatorDescriptor, OperatorRegistry

DCR_IMPROVEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-improvement-proposal@1"
)
_FORBIDDEN: Final[frozenset[str]] = frozenset(
    {
        "policy_root",
        "validator",
        "authority",
        "lifecycle",
        "no_llm",
        "logic",
        "profile",
        "safety_floor",
        "provider",
        "source",
        "prompt",
    }
)


class ImprovementDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    PROPOSAL_PENDING = "proposal_pending"
    NO_OP = "stable_no_op"
    REJECTED = "rejected"


@dataclass(frozen=True)
class BoundedParameterPolicy:
    """Closed reviewed integer bounds; no policy or authority fields appear."""

    parameters: Mapping[str, tuple[int, int, int]]

    def __post_init__(self) -> None:
        values = dict(self.parameters)
        if not values:
            raise ValueError("bounded parameter policy is required")
        for name, bounds in values.items():
            if not isinstance(name, str) or not name or name in _FORBIDDEN:
                raise ValueError("parameter name is not reviewed")
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 3
                or any(type(item) is not int for item in bounds)
            ):
                raise ValueError("parameter bounds must be integer triples")
            minimum, maximum, current = bounds
            if minimum > maximum or not minimum <= current <= maximum:
                raise ValueError("parameter bounds/current are invalid")
        object.__setattr__(self, "parameters", dict(sorted(values.items())))


@dataclass(frozen=True)
class ImprovementMetrics:
    safety: int
    correctness: int
    cost: int
    latency: int

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in self.__dataclass_fields__
        ):
            raise ValueError("metrics must be non-negative integers")

    @property
    def score(self) -> tuple[int, int, int, int]:
        return (-self.safety, -self.correctness, self.cost, self.latency)


@dataclass(frozen=True)
class ShadowReceipt:
    baseline_cid: str
    candidate_cid: str
    receipt_cid: str
    passed: bool


@dataclass(frozen=True)
class ImprovementProposal:
    selection: RepairSelectionResult
    selection_evidence: tuple[RepairSelectionEvidence, ...]
    recovery: RecoveryDecision
    authority: RepairAuthorityProjection
    roots_cid: str
    baseline: ImprovementMetrics
    candidate: ImprovementMetrics
    safety_floors: Mapping[str, int]
    parameter_changes: Mapping[str, int]
    parameter_policy: BoundedParameterPolicy
    shadow: ShadowReceipt
    inverse_cid: str
    approval_class: str
    registry: OperatorRegistry | None = None
    descriptor: OperatorDescriptor | None = None
    pinned_registry_cid: str = ""


@dataclass(frozen=True)
class ImprovementResult:
    disposition: ImprovementDisposition
    reason_codes: tuple[str, ...]
    proposal_cid: str = ""
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def evaluate_improvement_proposal(value: Any) -> ImprovementResult:
    if not isinstance(value, ImprovementProposal):
        return ImprovementResult(ImprovementDisposition.REJECTED, ("typed_proposal_required",))
    reasons: list[str] = []
    if (
        not isinstance(value.selection, RepairSelectionResult)
        or not isinstance(value.recovery, RecoveryDecision)
        or not isinstance(value.authority, RepairAuthorityProjection)
    ):
        reasons.append("typed_selection_recovery_authority_required")
    if value.approval_class not in {
        "manual",
        "policy_pinned_review",
    }:
        reasons.append("self_admission_or_approval_class_forbidden")
    if (
        not value.roots_cid
        or value.roots_cid != value.authority.authority_roots.content_id
        or not value.inverse_cid
        or not value.shadow.passed
    ):
        reasons.append("roots_inverse_or_shadow_invalid")
    if (
        value.recovery.decision_cid != content_identity(value.recovery.to_dict())
        or value.recovery.disposition is not RecoveryDisposition.INTEGRATION_PENDING
        or value.authority.projection_cid != content_identity(value.authority.to_dict())
        or value.authority.status is RepairAuthorityStatus.COMPLETED
        or not value.recovery.task_id
        or value.recovery.task_id != value.selection.selected_key
        or value.authority.task_id != value.selection.selected_key
        or value.recovery.roots_cid != value.roots_cid
        or value.recovery.dcr080_receipt_cid != value.authority.dcr080_transition_cid
    ):
        reasons.append("forged_or_authoritative_recovery_projection")
    recomputed_selection = select_and_refill_repairs(value.selection_evidence)
    if (
        not isinstance(value.selection, RepairSelectionResult)
        or value.selection != recomputed_selection
        or not value.selection.selection_cid
    ):
        reasons.append("forged_or_unrecomputed_dcr081_selection")
    elif any(
        item.key != value.selection.selected_key
        or item.dependencies.roots.content_id != value.roots_cid
        or item.transition.receipt_cid != value.authority.dcr080_transition_cid
        for item in value.selection_evidence
    ):
        reasons.append("dcr081_dcr080_task_or_root_cross_binding_mismatch")
    if (
        value.shadow.baseline_cid != content_identity(value.baseline.__dict__)
        or value.shadow.candidate_cid != content_identity(value.candidate.__dict__)
        or value.shadow.receipt_cid
        != content_identity(
            {
                "baseline": value.shadow.baseline_cid,
                "candidate": value.shadow.candidate_cid,
                "passed": True,
            }
        )
    ):
        reasons.append("forged_shadow_receipt")
    if not isinstance(value.parameter_policy, BoundedParameterPolicy):
        reasons.append("typed_bounded_parameter_policy_required")
    elif (
        set(value.parameter_changes).difference(value.parameter_policy.parameters)
        or any(type(item) is not int for item in value.parameter_changes.values())
        or any(
            not bounds[0] <= value.parameter_changes[name] <= bounds[1]
            for name, bounds in value.parameter_policy.parameters.items()
            if name in value.parameter_changes
        )
    ):
        reasons.append("unknown_or_out_of_bound_parameter_change")
    if (
        set(value.safety_floors) != {"safety", "correctness"}
        or any(type(item) is not int for item in value.safety_floors.values())
        or any(
            value.baseline.__dict__[name] < floor or value.candidate.__dict__[name] < floor
            for name, floor in value.safety_floors.items()
        )
    ):
        reasons.append("safety_floor_regression")
    if value.descriptor is not None or value.registry is not None:
        if (
            not isinstance(value.descriptor, OperatorDescriptor)
            or not isinstance(value.registry, OperatorRegistry)
            or value.pinned_registry_cid != value.registry.report().get("registry_cid")
            or value.descriptor not in value.registry.enumerate()
        ):
            reasons.append("reviewed_operator_registry_pin_invalid")
    elif not value.parameter_changes:
        reasons.append("bounded_parameter_change_or_reviewed_operator_required")
    if reasons:
        return ImprovementResult(ImprovementDisposition.REJECTED, tuple(sorted(set(reasons))))
    if value.candidate.score >= value.baseline.score:
        return ImprovementResult(ImprovementDisposition.NO_OP, ("non_improving_fixed_point",))
    body = {
        "schema": DCR_IMPROVEMENT_SCHEMA,
        "roots_cid": value.roots_cid,
        "baseline": value.baseline.__dict__,
        "candidate": value.candidate.__dict__,
        "parameters": dict(sorted(value.parameter_changes.items())),
        "shadow": value.shadow.receipt_cid,
        "inverse": value.inverse_cid,
        "approval": value.approval_class,
    }
    return ImprovementResult(
        ImprovementDisposition.PROPOSAL_PENDING,
        ("integration_pending_live_dcr080_dcr083",),
        content_identity(body),
    )


__all__ = [
    "BoundedParameterPolicy",
    "DCR_IMPROVEMENT_SCHEMA",
    "ImprovementDisposition",
    "ImprovementMetrics",
    "ImprovementProposal",
    "ImprovementResult",
    "ShadowReceipt",
    "evaluate_improvement_proposal",
]
