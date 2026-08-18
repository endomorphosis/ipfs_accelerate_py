"""M3 policy admission for deterministic promotion decisions.

Admission consumes a comparison receipt plus current control, lease, and
optional human-approval evidence.  It may keep or downgrade a comparison
decision.  It never upgrades reject, regressed, or inconclusive to
promote, never lowers semantic or proof minima, and never grants a model
or evaluator promotion authority.

The closed M3 population is non-compensable.  A passing authorization
gate cannot waive missing human approval, a stale fence, or stale proof.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from ..validation.promotion_comparison import (
    DEFAULT_PROOF_MINIMUM_MILLIONTHS,
    DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS,
    FORBIDDEN_PROMOTION_ROLES,
    M2_GATES,
    PromotionComparisonPolicy,
    PromotionComparisonReceipt,
    PromotionDecision,
    compare_promotion,
)
from .control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    Operation,
    OperationAuthority,
)


PROMOTION_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-admission@1"
)
PROMOTION_ADMISSION_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-admission-policy@1"
)
PROMOTION_ADMISSION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-admission-receipt@1"
)
HUMAN_APPROVAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-human-approval@1"
)

M3_GATES: Final[tuple[str, ...]] = (
    "authorization",
    "human_approval",
    "fresh_proof",
    "lease_fence",
    "policy_identity",
)

# Campaign promote/reject reuse the existing closed control catalog.
PROMOTE_CONTROL_OPERATION: Final = Operation.OBJECTIVE_RECONCILE
REJECT_CONTROL_OPERATION: Final = Operation.QUARANTINE


class PromotionAdmissionError(ValueError):
    """Malformed admission policy, approval, or request."""


class M3GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise PromotionAdmissionError(f"{name} must be a string")
    else:
        text = value.strip()
    if "\x00" in text:
        raise PromotionAdmissionError(f"{name} must not contain NUL")
    if required and not text:
        raise PromotionAdmissionError(f"{name} must be a non-empty string")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PromotionAdmissionError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromotionAdmissionError(f"{name} must be an integer")
    if value < minimum:
        raise PromotionAdmissionError(f"{name} must be at least {minimum}")
    return value


def _strings(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise PromotionAdmissionError(f"{name} must be a sequence of strings")
    items = tuple(_text(item, name) for item in values)
    if len(items) != len(set(items)):
        raise PromotionAdmissionError(f"{name} values must be unique")
    return items


@dataclass(frozen=True)
class HumanApprovalRecord:
    """Optional human approval bound to one exact comparison receipt."""

    approval_identity: str
    actor_identity: str
    comparison_receipt_id: str
    candidate_checkpoint_id: str
    policy_identity: str
    approved: bool = True
    schema: str = HUMAN_APPROVAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "approval_identity", _text(self.approval_identity, "approval_identity")
        )
        object.__setattr__(self, "actor_identity", _text(self.actor_identity, "actor_identity"))
        object.__setattr__(
            self,
            "comparison_receipt_id",
            _text(self.comparison_receipt_id, "comparison_receipt_id"),
        )
        object.__setattr__(
            self,
            "candidate_checkpoint_id",
            _text(self.candidate_checkpoint_id, "candidate_checkpoint_id"),
        )
        object.__setattr__(
            self, "policy_identity", _text(self.policy_identity, "policy_identity")
        )
        object.__setattr__(self, "approved", _bool(self.approved, "approved"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != HUMAN_APPROVAL_SCHEMA:
            raise PromotionAdmissionError("unsupported human approval schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_identity": self.actor_identity,
            "approval_identity": self.approval_identity,
            "approved": self.approved,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "comparison_receipt_id": self.comparison_receipt_id,
            "policy_identity": self.policy_identity,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanApprovalRecord":
        if not isinstance(payload, Mapping):
            raise PromotionAdmissionError("human approval must be an object")
        return cls(
            approval_identity=payload.get("approval_identity", ""),
            actor_identity=payload.get("actor_identity", ""),
            comparison_receipt_id=payload.get("comparison_receipt_id", ""),
            candidate_checkpoint_id=payload.get("candidate_checkpoint_id", ""),
            policy_identity=payload.get("policy_identity", ""),
            approved=payload.get("approved", True),
            schema=payload.get("schema", HUMAN_APPROVAL_SCHEMA),
        )


@dataclass(frozen=True)
class PromotionAdmissionPolicy:
    """Independent promotion authority.  Evaluator/model cannot hold it."""

    policy_id: str
    policy_revision: str
    authorized_actors: tuple[str, ...]
    active_lease_fences: Mapping[str, int]
    require_human_approval: bool = False
    require_fresh_proof: bool = True
    semantic_minimum_millionths: int = DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS
    proof_minimum_millionths: int = DEFAULT_PROOF_MINIMUM_MILLIONTHS
    allowed_roles: tuple[str, ...] = ("operator", "qualifier")
    schema: str = PROMOTION_ADMISSION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )
        object.__setattr__(
            self, "authorized_actors", _strings(self.authorized_actors, "authorized_actors")
        )
        if not self.authorized_actors:
            raise PromotionAdmissionError("authorized_actors must not be empty")
        if not isinstance(self.active_lease_fences, Mapping):
            raise PromotionAdmissionError("active_lease_fences must be a mapping")
        fences: dict[str, int] = {}
        for key, value in self.active_lease_fences.items():
            fences[_text(key, "lease_id")] = _int(value, "fencing_epoch", minimum=1)
        object.__setattr__(self, "active_lease_fences", MappingProxyType(dict(sorted(fences.items()))))
        object.__setattr__(
            self,
            "require_human_approval",
            _bool(self.require_human_approval, "require_human_approval"),
        )
        object.__setattr__(
            self, "require_fresh_proof", _bool(self.require_fresh_proof, "require_fresh_proof")
        )
        object.__setattr__(
            self,
            "semantic_minimum_millionths",
            _int(self.semantic_minimum_millionths, "semantic_minimum_millionths"),
        )
        object.__setattr__(
            self,
            "proof_minimum_millionths",
            _int(self.proof_minimum_millionths, "proof_minimum_millionths"),
        )
        if self.semantic_minimum_millionths < DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS:
            raise PromotionAdmissionError("semantic minimum cannot be lowered")
        if self.proof_minimum_millionths < DEFAULT_PROOF_MINIMUM_MILLIONTHS:
            raise PromotionAdmissionError("proof minimum cannot be lowered")
        roles = tuple(_text(item, "role").casefold() for item in self.allowed_roles)
        if not roles:
            raise PromotionAdmissionError("allowed_roles must not be empty")
        forbidden = sorted(set(roles) & FORBIDDEN_PROMOTION_ROLES)
        if forbidden:
            raise PromotionAdmissionError(
                "allowed_roles cannot include evaluator/model authority: "
                + ", ".join(forbidden)
            )
        object.__setattr__(self, "allowed_roles", roles)
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_ADMISSION_POLICY_SCHEMA:
            raise PromotionAdmissionError("unsupported promotion admission policy schema")

    @property
    def policy_identity(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_lease_fences": dict(self.active_lease_fences),
            "allowed_roles": list(self.allowed_roles),
            "authorized_actors": list(self.authorized_actors),
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "proof_minimum_millionths": self.proof_minimum_millionths,
            "require_fresh_proof": self.require_fresh_proof,
            "require_human_approval": self.require_human_approval,
            "schema": self.schema,
            "semantic_minimum_millionths": self.semantic_minimum_millionths,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionAdmissionPolicy":
        if not isinstance(payload, Mapping):
            raise PromotionAdmissionError("promotion admission policy must be an object")
        return cls(
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            authorized_actors=tuple(payload.get("authorized_actors") or ()),
            active_lease_fences=payload.get("active_lease_fences") or {},
            require_human_approval=payload.get("require_human_approval", False),
            require_fresh_proof=payload.get("require_fresh_proof", True),
            semantic_minimum_millionths=payload.get(
                "semantic_minimum_millionths", DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS
            ),
            proof_minimum_millionths=payload.get(
                "proof_minimum_millionths", DEFAULT_PROOF_MINIMUM_MILLIONTHS
            ),
            allowed_roles=tuple(payload.get("allowed_roles") or ("operator", "qualifier")),
            schema=payload.get("schema", PROMOTION_ADMISSION_POLICY_SCHEMA),
        )


@dataclass(frozen=True)
class PromotionAdmissionRequest:
    """One admission attempt over a comparison receipt and current policy."""

    comparison: PromotionComparisonReceipt
    policy: PromotionAdmissionPolicy
    actor_identity: str
    actor_role: str
    lease_id: str
    fencing_epoch: int
    authorization: AuthorizationDecision | None = None
    human_approval: HumanApprovalRecord | None = None
    comparison_policy: PromotionComparisonPolicy | None = None
    proof_fresh: bool = True
    schema: str = PROMOTION_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        comparison = (
            self.comparison
            if isinstance(self.comparison, PromotionComparisonReceipt)
            else PromotionComparisonReceipt.from_dict(self.comparison)
        )
        object.__setattr__(self, "comparison", comparison)
        policy = (
            self.policy
            if isinstance(self.policy, PromotionAdmissionPolicy)
            else PromotionAdmissionPolicy.from_dict(self.policy)
        )
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "actor_identity", _text(self.actor_identity, "actor_identity"))
        object.__setattr__(self, "actor_role", _text(self.actor_role, "actor_role").casefold())
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "fencing_epoch", _int(self.fencing_epoch, "fencing_epoch", minimum=1)
        )
        if self.authorization is not None and not isinstance(
            self.authorization, AuthorizationDecision
        ):
            raise PromotionAdmissionError("authorization must be an AuthorizationDecision")
        if self.human_approval is not None and not isinstance(
            self.human_approval, HumanApprovalRecord
        ):
            if isinstance(self.human_approval, Mapping):
                object.__setattr__(
                    self, "human_approval", HumanApprovalRecord.from_dict(self.human_approval)
                )
            else:
                raise PromotionAdmissionError("human_approval must be a HumanApprovalRecord")
        if self.comparison_policy is not None and not isinstance(
            self.comparison_policy, PromotionComparisonPolicy
        ):
            object.__setattr__(
                self,
                "comparison_policy",
                PromotionComparisonPolicy.from_dict(self.comparison_policy),
            )
        object.__setattr__(self, "proof_fresh", _bool(self.proof_fresh, "proof_fresh"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_ADMISSION_SCHEMA:
            raise PromotionAdmissionError("unsupported promotion admission schema")


@dataclass(frozen=True)
class PromotionAdmissionReceipt:
    """Audit receipt.  Pointer CAS may proceed only when ``admitted``."""

    decision: PromotionDecision
    admitted: bool
    comparison_receipt_id: str
    candidate_checkpoint_id: str
    baseline_checkpoint_id: str
    expected_current_pointer: str
    policy_identity: str
    actor_identity: str
    lease_id: str
    fencing_epoch: int
    control_operation: str
    m3_results: Mapping[str, str]
    admitted_gates: tuple[str, ...]
    reasons: tuple[str, ...]
    human_approval_identity: str = ""
    cas_authorized: bool = False
    schema: str = PROMOTION_ADMISSION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        decision = (
            self.decision
            if isinstance(self.decision, PromotionDecision)
            else PromotionDecision(str(self.decision))
        )
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(
            self,
            "comparison_receipt_id",
            _text(self.comparison_receipt_id, "comparison_receipt_id"),
        )
        object.__setattr__(
            self,
            "candidate_checkpoint_id",
            _text(self.candidate_checkpoint_id, "candidate_checkpoint_id"),
        )
        object.__setattr__(
            self,
            "baseline_checkpoint_id",
            _text(self.baseline_checkpoint_id, "baseline_checkpoint_id"),
        )
        object.__setattr__(
            self,
            "expected_current_pointer",
            _text(self.expected_current_pointer, "expected_current_pointer", required=False),
        )
        object.__setattr__(
            self, "policy_identity", _text(self.policy_identity, "policy_identity")
        )
        object.__setattr__(self, "actor_identity", _text(self.actor_identity, "actor_identity"))
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "fencing_epoch", _int(self.fencing_epoch, "fencing_epoch", minimum=1)
        )
        object.__setattr__(
            self, "control_operation", _text(self.control_operation, "control_operation")
        )
        if not isinstance(self.m3_results, Mapping):
            raise PromotionAdmissionError("m3_results must be a mapping")
        results = {
            _text(key, "m3_gate"): _text(value, "m3_status")
            for key, value in self.m3_results.items()
        }
        if tuple(sorted(results)) != tuple(sorted(M3_GATES)):
            raise PromotionAdmissionError(
                "m3_results must contain the exact closed non-compensable M3 population"
            )
        object.__setattr__(self, "m3_results", MappingProxyType(dict(sorted(results.items()))))
        object.__setattr__(
            self, "admitted_gates", tuple(_text(item, "admitted_gate") for item in self.admitted_gates)
        )
        object.__setattr__(
            self, "reasons", tuple(_text(item, "reason") for item in self.reasons)
        )
        object.__setattr__(
            self,
            "human_approval_identity",
            _text(self.human_approval_identity, "human_approval_identity", required=False),
        )
        object.__setattr__(self, "cas_authorized", _bool(self.cas_authorized, "cas_authorized"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_ADMISSION_RECEIPT_SCHEMA:
            raise PromotionAdmissionError("unsupported promotion admission receipt schema")
        if self.admitted != (
            self.decision is PromotionDecision.PROMOTE and self.cas_authorized
        ):
            raise PromotionAdmissionError(
                "admitted is derived from promote plus authorized CAS"
            )
        if self.admitted and set(self.admitted_gates) != set(M2_GATES):
            raise PromotionAdmissionError("admitted promotion must retain every M2 gate")
        if not self.admitted and self.cas_authorized:
            raise PromotionAdmissionError("CAS cannot be authorized unless admitted")

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_identity": self.actor_identity,
            "admitted": self.admitted,
            "admitted_gates": list(self.admitted_gates),
            "baseline_checkpoint_id": self.baseline_checkpoint_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "cas_authorized": self.cas_authorized,
            "comparison_receipt_id": self.comparison_receipt_id,
            "control_operation": self.control_operation,
            "decision": self.decision.value,
            "expected_current_pointer": self.expected_current_pointer,
            "fencing_epoch": self.fencing_epoch,
            "human_approval_identity": self.human_approval_identity,
            "lease_id": self.lease_id,
            "m3_results": dict(self.m3_results),
            "policy_identity": self.policy_identity,
            "reasons": list(self.reasons),
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionAdmissionReceipt":
        if not isinstance(payload, Mapping):
            raise PromotionAdmissionError("promotion admission receipt must be an object")
        claimed = payload.get("receipt_id")
        result = cls(
            decision=payload.get("decision", ""),
            admitted=payload.get("admitted", False),
            comparison_receipt_id=payload.get("comparison_receipt_id", ""),
            candidate_checkpoint_id=payload.get("candidate_checkpoint_id", ""),
            baseline_checkpoint_id=payload.get("baseline_checkpoint_id", ""),
            expected_current_pointer=payload.get("expected_current_pointer", ""),
            policy_identity=payload.get("policy_identity", ""),
            actor_identity=payload.get("actor_identity", ""),
            lease_id=payload.get("lease_id", ""),
            fencing_epoch=payload.get("fencing_epoch", 0),
            control_operation=payload.get("control_operation", ""),
            m3_results=payload.get("m3_results") or {},
            admitted_gates=tuple(payload.get("admitted_gates") or ()),
            reasons=tuple(payload.get("reasons") or ()),
            human_approval_identity=payload.get("human_approval_identity", ""),
            cas_authorized=payload.get("cas_authorized", False),
            schema=payload.get("schema", PROMOTION_ADMISSION_RECEIPT_SCHEMA),
        )
        if claimed is not None and claimed != result.receipt_id:
            raise PromotionAdmissionError("forged promotion admission receipt_id")
        return result


def _authorization_status(request: PromotionAdmissionRequest) -> tuple[M3GateStatus, str]:
    if request.actor_identity not in request.policy.authorized_actors:
        return M3GateStatus.FAIL, "authorization:actor_not_permitted"
    if request.actor_role not in request.policy.allowed_roles:
        return M3GateStatus.FAIL, "authorization:role_not_permitted"
    if request.actor_role in FORBIDDEN_PROMOTION_ROLES:
        return M3GateStatus.FAIL, "authorization:evaluator_or_model_denied"
    decision = request.authorization
    if decision is None:
        return M3GateStatus.FAIL, "authorization:missing_permit"
    if decision.verdict is not AuthorizationVerdict.PERMIT:
        return M3GateStatus.FAIL, "authorization:denied"
    if decision.granted_authority is not OperationAuthority.MUTATION:
        return M3GateStatus.FAIL, "authorization:insufficient_authority"
    if decision.lease_id != request.lease_id or decision.fencing_epoch != request.fencing_epoch:
        return M3GateStatus.FAIL, "authorization:lease_mismatch"
    return M3GateStatus.PASS, "authorization:pass"


def _human_approval_status(request: PromotionAdmissionRequest) -> tuple[M3GateStatus, str]:
    approval = request.human_approval
    if not request.policy.require_human_approval:
        if approval is None:
            return M3GateStatus.PASS, "human_approval:not_required"
        # A present approval must still bind the exact comparison.
    elif approval is None:
        return M3GateStatus.FAIL, "human_approval:required"
    assert approval is not None
    if not approval.approved:
        return M3GateStatus.FAIL, "human_approval:not_approved"
    if approval.comparison_receipt_id != request.comparison.receipt_id:
        return M3GateStatus.FAIL, "human_approval:comparison_mismatch"
    if approval.candidate_checkpoint_id != request.comparison.candidate_checkpoint_id:
        return M3GateStatus.FAIL, "human_approval:candidate_mismatch"
    if approval.policy_identity != request.policy.policy_identity:
        return M3GateStatus.FAIL, "human_approval:policy_mismatch"
    return M3GateStatus.PASS, "human_approval:pass"


def _fresh_proof_status(request: PromotionAdmissionRequest) -> tuple[M3GateStatus, str]:
    if not request.policy.require_fresh_proof:
        return M3GateStatus.PASS, "fresh_proof:not_required"
    if not request.comparison.proof_evidence_identity:
        return M3GateStatus.FAIL, "fresh_proof:missing"
    if not request.proof_fresh:
        return M3GateStatus.FAIL, "fresh_proof:stale"
    return M3GateStatus.PASS, "fresh_proof:pass"


def _lease_status(request: PromotionAdmissionRequest) -> tuple[M3GateStatus, str]:
    expected = request.policy.active_lease_fences.get(request.lease_id)
    if expected is None:
        return M3GateStatus.FAIL, "lease_fence:unknown_lease"
    if expected != request.fencing_epoch:
        return M3GateStatus.FAIL, "lease_fence:stale"
    return M3GateStatus.PASS, "lease_fence:pass"


def _policy_identity_status(request: PromotionAdmissionRequest) -> tuple[M3GateStatus, str]:
    comparison_policy = request.comparison_policy
    if comparison_policy is None:
        if request.comparison.policy_identity:
            return M3GateStatus.PASS, "policy_identity:comparison_bound"
        return M3GateStatus.FAIL, "policy_identity:missing"
    if comparison_policy.policy_identity != request.comparison.policy_identity:
        return M3GateStatus.FAIL, "policy_identity:comparison_mismatch"
    if comparison_policy.semantic_minimum_millionths < request.policy.semantic_minimum_millionths:
        return M3GateStatus.FAIL, "policy_identity:semantic_minimum_lowered"
    if comparison_policy.proof_minimum_millionths < request.policy.proof_minimum_millionths:
        return M3GateStatus.FAIL, "policy_identity:proof_minimum_lowered"
    return M3GateStatus.PASS, "policy_identity:pass"


_M3_EVALUATORS = {
    "authorization": _authorization_status,
    "human_approval": _human_approval_status,
    "fresh_proof": _fresh_proof_status,
    "lease_fence": _lease_status,
    "policy_identity": _policy_identity_status,
}


def admit_promotion(
    request: PromotionAdmissionRequest | Mapping[str, Any],
) -> PromotionAdmissionReceipt:
    """Admit or downgrade one comparison under current control policy."""

    admission = (
        request
        if isinstance(request, PromotionAdmissionRequest)
        else PromotionAdmissionRequest(
            comparison=request.get("comparison", {}),
            policy=request.get("policy", {}),
            actor_identity=request.get("actor_identity", ""),
            actor_role=request.get("actor_role", "operator"),
            lease_id=request.get("lease_id", ""),
            fencing_epoch=request.get("fencing_epoch", 0),
            authorization=request.get("authorization"),
            human_approval=request.get("human_approval"),
            comparison_policy=request.get("comparison_policy"),
            proof_fresh=request.get("proof_fresh", True),
            schema=request.get("schema", PROMOTION_ADMISSION_SCHEMA),
        )
    )
    comparison = admission.comparison
    m3_results: dict[str, str] = {}
    reasons: list[str] = []
    for gate_id in M3_GATES:
        status, reason = _M3_EVALUATORS[gate_id](admission)
        m3_results[gate_id] = status.value
        if status is M3GateStatus.FAIL:
            reasons.append(reason)
    m3_passed = all(status == M3GateStatus.PASS.value for status in m3_results.values())
    decision = comparison.decision
    if decision is PromotionDecision.PROMOTE and not m3_passed:
        decision = PromotionDecision.REJECT
        reasons.append("m3_policy_admission_failed")
    elif decision is not PromotionDecision.PROMOTE:
        reasons.extend(comparison.reasons)
        reasons.append("comparison_not_promotable")
    admitted = decision is PromotionDecision.PROMOTE and m3_passed
    control = (
        PROMOTE_CONTROL_OPERATION.value if admitted else REJECT_CONTROL_OPERATION.value
    )
    return PromotionAdmissionReceipt(
        decision=decision,
        admitted=admitted,
        comparison_receipt_id=comparison.receipt_id,
        candidate_checkpoint_id=comparison.candidate_checkpoint_id,
        baseline_checkpoint_id=comparison.baseline_checkpoint_id,
        expected_current_pointer=comparison.expected_current_pointer,
        policy_identity=admission.policy.policy_identity,
        actor_identity=admission.actor_identity,
        lease_id=admission.lease_id,
        fencing_epoch=admission.fencing_epoch,
        control_operation=control,
        m3_results=m3_results,
        admitted_gates=comparison.admitted_gates if admitted else (),
        reasons=tuple(dict.fromkeys(reasons or comparison.reasons)),
        human_approval_identity=(
            admission.human_approval.approval_identity if admission.human_approval else ""
        ),
        cas_authorized=admitted,
    )


def compare_and_admit_promotion(
    comparison_request: Mapping[str, Any] | Any,
    admission_kwargs: Mapping[str, Any],
) -> tuple[PromotionComparisonReceipt, PromotionAdmissionReceipt]:
    """Run comparison then admission as one fail-closed sequence."""

    comparison = compare_promotion(comparison_request)
    receipt = admit_promotion(
        PromotionAdmissionRequest(
            comparison=comparison,
            **dict(admission_kwargs),
        )
    )
    return comparison, receipt


__all__ = (
    "HUMAN_APPROVAL_SCHEMA",
    "M3_GATES",
    "PROMOTE_CONTROL_OPERATION",
    "PROMOTION_ADMISSION_POLICY_SCHEMA",
    "PROMOTION_ADMISSION_RECEIPT_SCHEMA",
    "PROMOTION_ADMISSION_SCHEMA",
    "REJECT_CONTROL_OPERATION",
    "HumanApprovalRecord",
    "M3GateStatus",
    "PromotionAdmissionError",
    "PromotionAdmissionPolicy",
    "PromotionAdmissionReceipt",
    "PromotionAdmissionRequest",
    "admit_promotion",
    "compare_and_admit_promotion",
)
