"""Evidence-bound adaptive selection for plans targeting one frozen goal.

Candidate generation is deliberately outside this module.  The adaptive
planner accepts candidate declarations from deterministic or optional
providers, binds them to the same immutable goal/tree/policy snapshot, applies
typed non-compensable admission receipts, and only then delegates quality and
cost ranking to :mod:`plan_evaluator`.

The selection receipt is suitable for objective evidence.  In particular it
only claims :data:`AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID` when a selected
admissible plan is more expensive than a rejected candidate carrying a failed
authority receipt.  A model assertion, a cheap score, or a failure in another
dimension cannot manufacture that evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from .formal_replanner import RepairTransition
from .formal_verification_contracts import canonical_json, content_identity
from .plan_evaluator import (
    EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION,
    EvidenceAwarePlanCandidate,
    EvidenceAwarePlanEvaluation,
    EvidenceAwarePlanPolicy,
    EvaluatedEvidenceAwarePlan,
    PlanBranchValidationError,
    PlanDimensionAssessment,
    PlanEvaluationDimension,
    evaluate_evidence_aware_plans,
    validate_evidence_aware_plan_evaluation,
)


ADAPTIVE_PLANNER_VERSION: Final = 2
ADAPTIVE_PLAN_SELECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-plan-selection@2"
)
HARD_CONSTRAINT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plan-hard-constraint-receipt@2"
)
REQUIREMENT_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plan-requirement-evidence@1"
)
ADAPTIVE_PLAN_CANDIDATE_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-plan-candidate-snapshot@1"
)

# ASI-G097: a cheaper authority-violating plan is rejected.
AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID: Final = (
    "173075880069453142914839090434430341799"
)


class AdaptivePlannerValidationError(ValueError):
    """Raised when a candidate or receipt crosses the planning boundary badly."""


class HardPlanConstraint(str, Enum):
    """Plan properties that weighted scoring may never compensate for."""

    AUTHORITY = "authority"
    SCOPE = "scope"
    SAFETY = "safety"
    PROOF = "proof"


class GateProducerKind(str, Enum):
    """Trusted boundary classes allowed to produce hard-gate observations."""

    AUTHORIZATION_ENGINE = "authorization_engine"
    POLICY_ENGINE = "policy_engine"
    FORMAL_VALIDATOR = "formal_validator"
    PROOF_VERIFIER = "proof_verifier"


_ALLOWED_GATE_PRODUCERS: Final[Mapping[HardPlanConstraint, frozenset[GateProducerKind]]] = {
    HardPlanConstraint.AUTHORITY: frozenset(
        {GateProducerKind.AUTHORIZATION_ENGINE, GateProducerKind.POLICY_ENGINE}
    ),
    HardPlanConstraint.SCOPE: frozenset(
        {GateProducerKind.AUTHORIZATION_ENGINE, GateProducerKind.POLICY_ENGINE}
    ),
    HardPlanConstraint.SAFETY: frozenset(
        {GateProducerKind.FORMAL_VALIDATOR, GateProducerKind.POLICY_ENGINE}
    ),
    HardPlanConstraint.PROOF: frozenset(
        {GateProducerKind.PROOF_VERIFIER, GateProducerKind.FORMAL_VALIDATOR}
    ),
}


def _text(value: Any, field_name: str) -> str:
    result = str(value or "").strip()
    if not result or "\x00" in result:
        raise AdaptivePlannerValidationError(
            f"{field_name} must be a non-empty string without NUL bytes"
        )
    return result


def _strings(
    value: Sequence[Any] | Iterable[Any],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise AdaptivePlannerValidationError(f"{field_name} must be an array")
    result = tuple(
        sorted({_text(item, field_name) for item in value})
    )
    if not result and not allow_empty:
        raise AdaptivePlannerValidationError(f"{field_name} must not be empty")
    return result


def _ordered_strings(
    value: Sequence[Any] | Iterable[Any],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise AdaptivePlannerValidationError(f"{field_name} must be an array")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _text(item, field_name)
        if normalized in seen:
            raise AdaptivePlannerValidationError(
                f"{field_name} must not contain duplicates"
            )
        seen.add(normalized)
        result.append(normalized)
    if not result and not allow_empty:
        raise AdaptivePlannerValidationError(f"{field_name} must not be empty")
    return tuple(result)


def _integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AdaptivePlannerValidationError(
            f"{field_name} must be an integer of at least {minimum}"
        )
    return value


def adaptive_plan_candidate_snapshot_id(
    plan: EvidenceAwarePlanCandidate,
    *,
    goal_content_id: str,
    repository_tree_id: str,
    policy_digest: str,
    formal_plan_id: str = "",
    repair_transition: RepairTransition | None = None,
) -> str:
    """Return the canonical snapshot independently inspected by hard gates.

    A branch identifier is a human/scheduler key and is not a content
    identity.  Gate receipts therefore bind this digest, which includes the
    full evaluator declaration, frozen inputs, and optional formal-repair
    provenance.  Reusing a receipt after changing cost, scope, evidence, or a
    repair transition consequently fails closed.
    """

    resolved_plan = (
        plan
        if isinstance(plan, EvidenceAwarePlanCandidate)
        else EvidenceAwarePlanCandidate.from_dict(plan)
    )
    resolved_goal = _text(goal_content_id, "goal_content_id")
    resolved_tree = _text(repository_tree_id, "repository_tree_id")
    resolved_policy = _text(policy_digest, "policy_digest")
    resolved_formal_plan_id = str(formal_plan_id or "").strip()
    if repair_transition is not None:
        if not isinstance(repair_transition, RepairTransition):
            raise AdaptivePlannerValidationError(
                "repair_transition must be a formal RepairTransition"
            )
        if not resolved_formal_plan_id:
            raise AdaptivePlannerValidationError(
                "a repair transition requires formal_plan_id"
            )
        if repair_transition.repaired_plan_id != resolved_formal_plan_id:
            raise AdaptivePlannerValidationError(
                "repair transition does not produce formal_plan_id"
            )
    return content_identity(
        {
            "schema": ADAPTIVE_PLAN_CANDIDATE_SNAPSHOT_SCHEMA,
            "plan": resolved_plan.to_dict(profile_g=True),
            "goal_content_id": resolved_goal,
            "repository_tree_id": resolved_tree,
            "policy_digest": resolved_policy,
            "formal_plan_id": resolved_formal_plan_id,
            "repair_transition": (
                repair_transition.to_dict()
                if repair_transition is not None
                else None
            ),
        }
    )


@dataclass(frozen=True)
class FrozenPlanningGoal:
    """Immutable goal, repository and evaluator policy used by every branch."""

    goal_id: str
    goal_content_id: str
    repository_tree_id: str
    policy: EvidenceAwarePlanPolicy

    def __post_init__(self) -> None:
        for name in ("goal_id", "goal_content_id", "repository_tree_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        policy = (
            self.policy
            if isinstance(self.policy, EvidenceAwarePlanPolicy)
            else EvidenceAwarePlanPolicy.from_dict(self.policy)
        )
        object.__setattr__(self, "policy", policy)

    @property
    def policy_digest(self) -> str:
        """Content identity of the integer-only frozen evaluation policy."""

        return content_identity(self.policy.to_dict(profile_g=True))

    @property
    def frozen_goal_id(self) -> str:
        """Compatibility spelling for integrations that name the content ID."""

        return self.goal_content_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_content_id": self.goal_content_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_digest": self.policy_digest,
            "policy": self.policy.to_dict(profile_g=True),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenPlanningGoal":
        allowed = {
            "goal_id",
            "goal_content_id",
            "repository_tree_id",
            "policy_digest",
            "policy",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise AdaptivePlannerValidationError(
                "unknown frozen-goal fields: " + ", ".join(unknown)
            )
        result = cls(
            goal_id=payload.get("goal_id", ""),
            goal_content_id=payload.get("goal_content_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            policy=_decode_policy(payload.get("policy") or {}),
        )
        if payload.get("policy_digest") != result.policy_digest:
            raise AdaptivePlannerValidationError(
                "frozen policy digest is inconsistent"
            )
        return result


@dataclass(frozen=True)
class HardConstraintReceipt:
    """One independently produced hard-gate result with exact input bindings."""

    constraint: HardPlanConstraint
    candidate_id: str
    candidate_snapshot_id: str
    goal_content_id: str
    repository_tree_id: str
    policy_digest: str
    passed: bool
    producer_kind: GateProducerKind
    producer_id: str
    evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint", HardPlanConstraint(self.constraint))
        object.__setattr__(self, "producer_kind", GateProducerKind(self.producer_kind))
        for name in (
            "candidate_id",
            "candidate_snapshot_id",
            "goal_content_id",
            "repository_tree_id",
            "policy_digest",
            "producer_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.passed, bool):
            raise AdaptivePlannerValidationError("passed must be boolean")
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(self.evidence_ids, "evidence_ids"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _strings(
                self.reason_codes,
                "reason_codes",
                allow_empty=self.passed,
            ),
        )
        if self.producer_kind not in _ALLOWED_GATE_PRODUCERS[self.constraint]:
            raise AdaptivePlannerValidationError(
                f"{self.producer_kind.value} cannot decide {self.constraint.value}"
            )
        if self.passed and self.reason_codes:
            raise AdaptivePlannerValidationError(
                "a passing hard-gate receipt cannot contain rejection reason codes"
            )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": HARD_CONSTRAINT_RECEIPT_SCHEMA,
            "constraint": self.constraint.value,
            "candidate_id": self.candidate_id,
            "candidate_snapshot_id": self.candidate_snapshot_id,
            "goal_content_id": self.goal_content_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_digest": self.policy_digest,
            "passed": self.passed,
            "producer_kind": self.producer_kind.value,
            "producer_id": self.producer_id,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardConstraintReceipt":
        allowed = {
            "schema",
            "receipt_id",
            "constraint",
            "candidate_id",
            "candidate_snapshot_id",
            "goal_content_id",
            "repository_tree_id",
            "policy_digest",
            "passed",
            "producer_kind",
            "producer_id",
            "evidence_ids",
            "reason_codes",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise AdaptivePlannerValidationError(
                "unknown hard-constraint receipt fields: " + ", ".join(unknown)
            )
        if payload.get("schema") != HARD_CONSTRAINT_RECEIPT_SCHEMA:
            raise AdaptivePlannerValidationError(
                "unsupported hard-constraint receipt schema"
            )
        result = cls(
            constraint=payload.get("constraint", ""),
            candidate_id=payload.get("candidate_id", ""),
            candidate_snapshot_id=payload.get("candidate_snapshot_id", ""),
            goal_content_id=payload.get("goal_content_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            policy_digest=payload.get("policy_digest", ""),
            passed=payload.get("passed"),
            producer_kind=payload.get("producer_kind", ""),
            producer_id=payload.get("producer_id", ""),
            evidence_ids=payload.get("evidence_ids") or (),
            reason_codes=payload.get("reason_codes") or (),
        )
        claimed = str(payload.get("receipt_id") or "")
        if claimed and claimed != result.receipt_id:
            raise AdaptivePlannerValidationError(
                "hard-constraint receipt identity does not match content"
            )
        return result


@dataclass(frozen=True)
class AdaptivePlanCandidate:
    """Evaluator candidate plus frozen bindings and authoritative gate receipts."""

    plan: EvidenceAwarePlanCandidate
    goal_content_id: str
    repository_tree_id: str
    policy_digest: str
    hard_constraint_receipts: tuple[HardConstraintReceipt, ...]
    formal_plan_id: str = ""
    repair_transition: RepairTransition | None = None

    def __post_init__(self) -> None:
        plan = (
            self.plan
            if isinstance(self.plan, EvidenceAwarePlanCandidate)
            else EvidenceAwarePlanCandidate.from_dict(self.plan)
        )
        object.__setattr__(self, "plan", plan)
        for name in ("goal_content_id", "repository_tree_id", "policy_digest"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        receipts = tuple(
            item
            if isinstance(item, HardConstraintReceipt)
            else HardConstraintReceipt.from_dict(item)
            for item in self.hard_constraint_receipts
        )
        by_constraint = {item.constraint: item for item in receipts}
        if len(receipts) != len(by_constraint) or set(by_constraint) != set(
            HardPlanConstraint
        ):
            raise AdaptivePlannerValidationError(
                "each candidate requires exactly one authority, scope, safety, "
                "and proof receipt"
            )
        for receipt in receipts:
            expected = (
                plan.candidate_id,
                self.goal_content_id,
                self.repository_tree_id,
                self.policy_digest,
            )
            actual = (
                receipt.candidate_id,
                receipt.goal_content_id,
                receipt.repository_tree_id,
                receipt.policy_digest,
            )
            if actual != expected:
                raise AdaptivePlannerValidationError(
                    "hard-constraint receipt is not bound to its candidate snapshot"
                )
        formal_plan_id = str(self.formal_plan_id or "").strip()
        object.__setattr__(self, "formal_plan_id", formal_plan_id)
        if self.repair_transition is not None:
            if not isinstance(self.repair_transition, RepairTransition):
                raise AdaptivePlannerValidationError(
                    "repair_transition must be a formal RepairTransition"
                )
            if not formal_plan_id:
                raise AdaptivePlannerValidationError(
                    "a repair transition requires formal_plan_id"
                )
            if self.repair_transition.repaired_plan_id != formal_plan_id:
                raise AdaptivePlannerValidationError(
                    "repair transition does not produce formal_plan_id"
                )
        snapshot_id = adaptive_plan_candidate_snapshot_id(
            plan,
            goal_content_id=self.goal_content_id,
            repository_tree_id=self.repository_tree_id,
            policy_digest=self.policy_digest,
            formal_plan_id=formal_plan_id,
            repair_transition=self.repair_transition,
        )
        if any(
            receipt.candidate_snapshot_id != snapshot_id
            for receipt in receipts
        ):
            raise AdaptivePlannerValidationError(
                "hard-constraint receipt is not bound to the candidate content"
            )
        object.__setattr__(
            self,
            "hard_constraint_receipts",
            tuple(sorted(receipts, key=lambda item: item.constraint.value)),
        )

    @property
    def candidate_id(self) -> str:
        return self.plan.candidate_id

    @property
    def snapshot_id(self) -> str:
        """Canonical content identity shared by all gate receipts."""

        return adaptive_plan_candidate_snapshot_id(
            self.plan,
            goal_content_id=self.goal_content_id,
            repository_tree_id=self.repository_tree_id,
            policy_digest=self.policy_digest,
            formal_plan_id=self.formal_plan_id,
            repair_transition=self.repair_transition,
        )

    def receipt_for(self, constraint: HardPlanConstraint) -> HardConstraintReceipt:
        return next(
            item
            for item in self.hard_constraint_receipts
            if item.constraint is constraint
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_snapshot_id": self.snapshot_id,
            "plan": self.plan.to_dict(profile_g=True),
            "goal_content_id": self.goal_content_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_digest": self.policy_digest,
            "hard_constraint_receipts": [
                item.to_dict() for item in self.hard_constraint_receipts
            ],
            "formal_plan_id": self.formal_plan_id,
            "repair_transition": (
                self.repair_transition.to_dict()
                if self.repair_transition is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptivePlanCandidate":
        allowed = {
            "candidate_id",
            "candidate_snapshot_id",
            "plan",
            "goal_content_id",
            "repository_tree_id",
            "policy_digest",
            "hard_constraint_receipts",
            "formal_plan_id",
            "repair_transition",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise AdaptivePlannerValidationError(
                "unknown adaptive candidate fields: " + ", ".join(unknown)
            )
        transition_payload = payload.get("repair_transition")
        result = cls(
            plan=_decode_profile_candidate(payload.get("plan") or {}),
            goal_content_id=payload.get("goal_content_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            policy_digest=payload.get("policy_digest", ""),
            hard_constraint_receipts=tuple(
                HardConstraintReceipt.from_dict(item)
                for item in payload.get("hard_constraint_receipts") or ()
            ),
            formal_plan_id=str(payload.get("formal_plan_id") or ""),
            repair_transition=(
                RepairTransition.from_dict(transition_payload)
                if transition_payload is not None
                else None
            ),
        )
        claimed = str(payload.get("candidate_id") or "")
        if claimed and claimed != result.candidate_id:
            raise AdaptivePlannerValidationError(
                "adaptive candidate identity does not match plan"
            )
        claimed_snapshot = str(payload.get("candidate_snapshot_id") or "")
        if claimed_snapshot and claimed_snapshot != result.snapshot_id:
            raise AdaptivePlannerValidationError(
                "adaptive candidate snapshot identity does not match content"
            )
        return result


def _decode_profile_candidate(payload: Mapping[str, Any]) -> EvidenceAwarePlanCandidate:
    """Decode the evaluator's integer-only receipt projection."""

    values = dict(payload)
    branch = dict(values.pop("branch"))
    values.pop("candidate_id", None)
    branch["estimated_cost"] = branch.pop("estimated_cost_millionths") / 1_000_000
    branch["risk"] = branch.pop("risk_millionths") / 1_000_000
    branch["expected_objective_delta"] = (
        branch.pop("expected_objective_delta_millionths") / 1_000_000
    )
    values["novelty"] = values.pop("novelty_millionths") / 1_000_000
    values["estimated_resource_cost"] = (
        values.pop("estimated_resource_cost_millionths") / 1_000_000
    )
    return EvidenceAwarePlanCandidate.from_dict({"branch": branch, **values})


def _decode_evaluated(payload: Mapping[str, Any]) -> EvaluatedEvidenceAwarePlan:
    dimensions = tuple(
        PlanDimensionAssessment(
            dimension=item["dimension"],
            passed=item["passed"],
            hard_gate=item["hard_gate"],
            score_millionths=item["score_millionths"],
            reasons=tuple(item["reasons"]),
        )
        for item in payload["dimensions"]
    )
    return EvaluatedEvidenceAwarePlan(
        candidate=_decode_profile_candidate(payload["candidate"]),
        score_millionths=payload["score_millionths"],
        dimensions=dimensions,
        hard_gate_failures=tuple(payload["hard_gate_failures"]),
    )


def _decode_policy(payload: Mapping[str, Any]) -> EvidenceAwarePlanPolicy:
    values = dict(payload)
    values["min_novelty"] = values.pop("min_novelty_millionths") / 1_000_000
    values["max_estimated_resource_cost"] = (
        values.pop("max_estimated_resource_cost_millionths") / 1_000_000
    )
    return EvidenceAwarePlanPolicy.from_dict(values)


@dataclass(frozen=True)
class AuthorityNonCompensationEvidence:
    """Concrete witness that cost did not compensate for invalid authority."""

    goal_content_id: str
    repository_tree_id: str
    policy_digest: str
    selected_candidate_id: str
    selected_cost_millionths: int
    rejected_candidate_ids: tuple[str, ...]
    rejected_cost_millionths: tuple[int, ...]
    authority_receipt_ids: tuple[str, ...]
    requirement_id: str = AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID
    producer_kind: str = "adaptive_plan_selection"

    def __post_init__(self) -> None:
        for name in (
            "goal_content_id",
            "repository_tree_id",
            "policy_digest",
            "selected_candidate_id",
            "requirement_id",
            "producer_kind",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID:
            raise AdaptivePlannerValidationError("unsupported requirement evidence id")
        object.__setattr__(
            self,
            "rejected_candidate_ids",
            _ordered_strings(
                self.rejected_candidate_ids, "rejected_candidate_ids"
            ),
        )
        object.__setattr__(
            self,
            "authority_receipt_ids",
            _ordered_strings(
                self.authority_receipt_ids, "authority_receipt_ids"
            ),
        )
        _integer(self.selected_cost_millionths, "selected_cost_millionths")
        costs = tuple(
            _integer(item, "rejected_cost_millionths")
            for item in self.rejected_cost_millionths
        )
        if len(costs) != len(self.rejected_candidate_ids):
            raise AdaptivePlannerValidationError(
                "rejected candidate and cost evidence must have equal length"
            )
        if len(self.authority_receipt_ids) != len(self.rejected_candidate_ids):
            raise AdaptivePlannerValidationError(
                "every rejected candidate requires an authority receipt"
            )
        if any(item >= self.selected_cost_millionths for item in costs):
            raise AdaptivePlannerValidationError(
                "authority non-compensation evidence requires cheaper rejected plans"
            )
        object.__setattr__(self, "rejected_cost_millionths", costs)

    @property
    def evidence_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REQUIREMENT_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "producer_kind": self.producer_kind,
            "goal_content_id": self.goal_content_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_digest": self.policy_digest,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_cost_millionths": self.selected_cost_millionths,
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "rejected_cost_millionths": list(self.rejected_cost_millionths),
            "authority_receipt_ids": list(self.authority_receipt_ids),
        }
        if include_identity:
            payload["evidence_id"] = self.evidence_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AuthorityNonCompensationEvidence":
        allowed = {
            "schema",
            "evidence_id",
            "requirement_id",
            "producer_kind",
            "goal_content_id",
            "repository_tree_id",
            "policy_digest",
            "selected_candidate_id",
            "selected_cost_millionths",
            "rejected_candidate_ids",
            "rejected_cost_millionths",
            "authority_receipt_ids",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise AdaptivePlannerValidationError(
                "unknown requirement evidence fields: " + ", ".join(unknown)
            )
        if payload.get("schema") != REQUIREMENT_EVIDENCE_SCHEMA:
            raise AdaptivePlannerValidationError(
                "unsupported requirement evidence schema"
            )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            producer_kind=payload.get("producer_kind", ""),
            goal_content_id=payload.get("goal_content_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            policy_digest=payload.get("policy_digest", ""),
            selected_candidate_id=payload.get("selected_candidate_id", ""),
            selected_cost_millionths=payload.get("selected_cost_millionths"),
            rejected_candidate_ids=payload.get("rejected_candidate_ids") or (),
            rejected_cost_millionths=payload.get("rejected_cost_millionths") or (),
            authority_receipt_ids=payload.get("authority_receipt_ids") or (),
        )
        claimed = str(payload.get("evidence_id") or "")
        if claimed and claimed != result.evidence_id:
            raise AdaptivePlannerValidationError(
                "requirement evidence identity does not match content"
            )
        return result


@dataclass(frozen=True)
class AdaptivePlanSelectionReceipt:
    """Complete deterministic decision trace for one frozen-goal selection."""

    frozen_goal: FrozenPlanningGoal
    evaluation: EvidenceAwarePlanEvaluation
    hard_constraint_receipts: tuple[HardConstraintReceipt, ...]
    authority_non_compensation_evidence: (
        AuthorityNonCompensationEvidence | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_goal, FrozenPlanningGoal):
            raise AdaptivePlannerValidationError(
                "frozen_goal must be FrozenPlanningGoal"
            )
        if not isinstance(self.evaluation, EvidenceAwarePlanEvaluation):
            raise AdaptivePlannerValidationError(
                "evaluation must be EvidenceAwarePlanEvaluation"
            )
        if self.evaluation.policy != self.frozen_goal.policy:
            raise AdaptivePlannerValidationError(
                "evaluation policy does not match frozen goal"
            )
        try:
            validate_evidence_aware_plan_evaluation(self.evaluation)
        except PlanBranchValidationError as exc:
            raise AdaptivePlannerValidationError(str(exc)) from exc
        receipts = tuple(self.hard_constraint_receipts)
        evaluated_ids = {item.candidate_id for item in self.evaluation.ranked}
        if {item.candidate_id for item in receipts} != evaluated_ids:
            raise AdaptivePlannerValidationError(
                "hard receipts must cover every evaluated candidate"
            )
        if len(receipts) != len(evaluated_ids) * len(HardPlanConstraint):
            raise AdaptivePlannerValidationError(
                "selection requires four hard receipts per candidate"
            )
        receipt_matrix = {
            (item.candidate_id, item.constraint): item for item in receipts
        }
        expected_matrix = {
            (candidate_id, constraint)
            for candidate_id in evaluated_ids
            for constraint in HardPlanConstraint
        }
        if len(receipt_matrix) != len(receipts) or set(receipt_matrix) != expected_matrix:
            raise AdaptivePlannerValidationError(
                "selection requires exactly one receipt for every "
                "candidate/constraint pair"
            )
        for candidate_id in evaluated_ids:
            snapshot_ids = {
                receipt_matrix[
                    (candidate_id, constraint)
                ].candidate_snapshot_id
                for constraint in HardPlanConstraint
            }
            if len(snapshot_ids) != 1:
                raise AdaptivePlannerValidationError(
                    "hard receipts for a candidate must bind one content snapshot"
                )
        for receipt in receipts:
            if (
                receipt.goal_content_id != self.frozen_goal.goal_content_id
                or receipt.repository_tree_id
                != self.frozen_goal.repository_tree_id
                or receipt.policy_digest != self.frozen_goal.policy_digest
            ):
                raise AdaptivePlannerValidationError(
                    "hard receipt does not match the frozen goal bindings"
                )
        evaluated_by_id = {
            item.candidate_id: item for item in self.evaluation.ranked
        }
        expected_failure_dimension = {
            HardPlanConstraint.AUTHORITY: (
                PlanEvaluationDimension.CONFLICT_SCOPE_AND_AUTHORITY.value
            ),
            HardPlanConstraint.SCOPE: (
                PlanEvaluationDimension.CONFLICT_SCOPE_AND_AUTHORITY.value
            ),
            HardPlanConstraint.SAFETY: (
                PlanEvaluationDimension.CONFLICT_SCOPE_AND_AUTHORITY.value
            ),
            HardPlanConstraint.PROOF: (
                PlanEvaluationDimension.VALIDATION_AND_PROOF.value
            ),
        }
        for receipt in receipts:
            if (
                not receipt.passed
                and expected_failure_dimension[receipt.constraint]
                not in evaluated_by_id[receipt.candidate_id].hard_gate_failures
            ):
                raise AdaptivePlannerValidationError(
                    f"failed {receipt.constraint.value} receipt is not reflected "
                    "in the evaluator hard-gate result"
                )
        object.__setattr__(
            self,
            "hard_constraint_receipts",
            tuple(
                sorted(
                    receipts,
                    key=lambda item: (item.candidate_id, item.constraint.value),
                )
            ),
        )
        evidence = self.authority_non_compensation_evidence
        expected_witnesses: tuple[tuple[str, int, str], ...] = ()
        selected = self.evaluation.selected
        if selected is not None:
            selected_cost = _cost_millionths(selected.candidate)
            expected_witnesses = tuple(
                (
                    rejected.candidate_id,
                    _cost_millionths(rejected.candidate),
                    receipt_matrix[
                        (
                            rejected.candidate_id,
                            HardPlanConstraint.AUTHORITY,
                        )
                    ].receipt_id,
                )
                for rejected in sorted(
                    self.evaluation.rejected,
                    key=lambda item: item.candidate_id,
                )
                if (
                    not receipt_matrix[
                        (
                            rejected.candidate_id,
                            HardPlanConstraint.AUTHORITY,
                        )
                    ].passed
                    and _cost_millionths(rejected.candidate) < selected_cost
                )
            )
        if bool(evidence) != bool(expected_witnesses):
            raise AdaptivePlannerValidationError(
                "authority non-compensation evidence must exactly cover "
                "all qualifying rejected candidates"
            )
        if evidence is not None:
            if not isinstance(evidence, AuthorityNonCompensationEvidence):
                raise AdaptivePlannerValidationError(
                    "invalid authority non-compensation evidence"
                )
            if (
                evidence.goal_content_id != self.frozen_goal.goal_content_id
                or evidence.repository_tree_id
                != self.frozen_goal.repository_tree_id
                or evidence.policy_digest != self.frozen_goal.policy_digest
            ):
                raise AdaptivePlannerValidationError(
                    "requirement evidence does not match frozen bindings"
                )
            if selected is None or (
                evidence.selected_candidate_id != selected.candidate_id
            ):
                raise AdaptivePlannerValidationError(
                    "requirement evidence does not name the selected candidate"
                )
            selected_authority = receipt_matrix[
                (selected.candidate_id, HardPlanConstraint.AUTHORITY)
            ]
            if not selected_authority.passed:
                raise AdaptivePlannerValidationError(
                    "authority evidence cannot select a candidate that failed authority"
                )
            selected_cost = _cost_millionths(selected.candidate)
            if evidence.selected_cost_millionths != selected_cost:
                raise AdaptivePlannerValidationError(
                    "requirement evidence selected cost is inconsistent"
                )
            actual_witnesses = tuple(
                zip(
                    evidence.rejected_candidate_ids,
                    evidence.rejected_cost_millionths,
                    evidence.authority_receipt_ids,
                )
            )
            if actual_witnesses != expected_witnesses:
                raise AdaptivePlannerValidationError(
                    "authority non-compensation evidence is incomplete or inconsistent"
                )
            rejected = {
                item.candidate_id: item for item in self.evaluation.rejected
            }
            for candidate_id, claimed_cost, claimed_receipt_id in zip(
                evidence.rejected_candidate_ids,
                evidence.rejected_cost_millionths,
                evidence.authority_receipt_ids,
            ):
                evaluated = rejected.get(candidate_id)
                if evaluated is None:
                    raise AdaptivePlannerValidationError(
                        "requirement evidence names a candidate that was not rejected"
                    )
                authority_receipt = receipt_matrix[
                    (candidate_id, HardPlanConstraint.AUTHORITY)
                ]
                if authority_receipt.passed:
                    raise AdaptivePlannerValidationError(
                        "requirement evidence requires a failed authority receipt"
                    )
                if claimed_receipt_id != authority_receipt.receipt_id:
                    raise AdaptivePlannerValidationError(
                        "requirement evidence authority receipt is inconsistent"
                    )
                if claimed_cost != _cost_millionths(evaluated.candidate):
                    raise AdaptivePlannerValidationError(
                        "requirement evidence rejected cost is inconsistent"
                    )

    @property
    def selected(self) -> EvidenceAwarePlanCandidate | None:
        return (
            self.evaluation.selected.candidate
            if self.evaluation.selected is not None
            else None
        )

    @property
    def selected_candidate_id(self) -> str | None:
        return self.selected.candidate_id if self.selected is not None else None

    @property
    def proves_authority_non_compensation(self) -> bool:
        return self.authority_non_compensation_evidence is not None

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (
            (AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID,)
            if self.proves_authority_non_compensation
            else ()
        )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ADAPTIVE_PLAN_SELECTION_SCHEMA,
            "planner_version": ADAPTIVE_PLANNER_VERSION,
            "evaluator_version": EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION,
            "frozen_goal": self.frozen_goal.to_dict(),
            "evaluation": self.evaluation.to_profile_g_dict(),
            "hard_constraint_receipts": [
                item.to_dict() for item in self.hard_constraint_receipts
            ],
            "proved_requirement_ids": list(self.proved_requirement_ids),
            "authority_non_compensation_evidence": (
                self.authority_non_compensation_evidence.to_dict()
                if self.authority_non_compensation_evidence is not None
                else None
            ),
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AdaptivePlanSelectionReceipt":
        allowed = {
            "schema",
            "receipt_id",
            "planner_version",
            "evaluator_version",
            "frozen_goal",
            "evaluation",
            "hard_constraint_receipts",
            "proved_requirement_ids",
            "authority_non_compensation_evidence",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise AdaptivePlannerValidationError(
                "unknown adaptive-plan receipt fields: " + ", ".join(unknown)
            )
        if payload.get("schema") != ADAPTIVE_PLAN_SELECTION_SCHEMA:
            raise AdaptivePlannerValidationError(
                "unsupported adaptive-plan selection schema"
            )
        if payload.get("planner_version") != ADAPTIVE_PLANNER_VERSION:
            raise AdaptivePlannerValidationError(
                "unsupported adaptive planner version"
            )
        if (
            payload.get("evaluator_version")
            != EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION
        ):
            raise AdaptivePlannerValidationError(
                "unsupported adaptive receipt evaluator version"
            )
        frozen_goal = FrozenPlanningGoal.from_dict(payload["frozen_goal"])
        policy = frozen_goal.policy
        evaluation_payload = payload["evaluation"]
        try:
            evaluation = EvidenceAwarePlanEvaluation(
                selected=(
                    _decode_evaluated(evaluation_payload["selected"])
                    if evaluation_payload.get("selected") is not None
                    else None
                ),
                admissible=tuple(
                    _decode_evaluated(item)
                    for item in evaluation_payload.get("admissible") or ()
                ),
                rejected=tuple(
                    _decode_evaluated(item)
                    for item in evaluation_payload.get("rejected") or ()
                ),
                policy=policy,
                evaluator_version=evaluation_payload.get(
                    "evaluator_version",
                    EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION,
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptivePlannerValidationError(
                f"invalid persisted plan evaluation: {exc}"
            ) from exc
        evidence_payload = payload.get("authority_non_compensation_evidence")
        result = cls(
            frozen_goal=frozen_goal,
            evaluation=evaluation,
            hard_constraint_receipts=tuple(
                HardConstraintReceipt.from_dict(item)
                for item in payload.get("hard_constraint_receipts") or ()
            ),
            authority_non_compensation_evidence=(
                AuthorityNonCompensationEvidence.from_dict(evidence_payload)
                if evidence_payload is not None
                else None
            ),
        )
        if tuple(payload.get("proved_requirement_ids") or ()) != (
            result.proved_requirement_ids
        ):
            raise AdaptivePlannerValidationError(
                "proved requirement projection is inconsistent"
            )
        claimed = str(payload.get("receipt_id") or "")
        if claimed and claimed != result.receipt_id:
            raise AdaptivePlannerValidationError(
                "adaptive-plan receipt identity does not match content"
            )
        return result


def _cost_millionths(candidate: EvidenceAwarePlanCandidate) -> int:
    """One deterministic total-cost projection used only for the criterion."""

    # Resource cost is the evaluator's normalized aggregate; tokens remain an
    # explicit positive tie component, and branch cost captures legacy callers.
    return (
        round(candidate.estimated_resource_cost * 1_000_000)
        + candidate.estimated_tokens
        + round(candidate.branch.estimated_cost * 1_000_000)
    )


class AdaptivePlanner:
    """Select an admissible branch for one frozen goal and emit its evidence."""

    def __init__(self, *, max_candidates: int = 32) -> None:
        self.max_candidates = _integer(
            max_candidates, "max_candidates", minimum=1
        )

    def select(
        self,
        frozen_goal: FrozenPlanningGoal,
        candidates: Iterable[AdaptivePlanCandidate],
    ) -> AdaptivePlanSelectionReceipt:
        if not isinstance(frozen_goal, FrozenPlanningGoal):
            raise AdaptivePlannerValidationError(
                "frozen_goal must be FrozenPlanningGoal"
            )
        normalized = tuple(candidates)
        if not normalized:
            raise AdaptivePlannerValidationError(
                "at least one adaptive plan candidate is required"
            )
        if len(normalized) > self.max_candidates:
            raise AdaptivePlannerValidationError(
                "adaptive plan candidate budget exceeded"
            )
        if any(not isinstance(item, AdaptivePlanCandidate) for item in normalized):
            raise AdaptivePlannerValidationError(
                "candidates must be AdaptivePlanCandidate instances"
            )
        ids = [item.candidate_id for item in normalized]
        duplicates = sorted(
            item for item in set(ids) if ids.count(item) > 1
        )
        if duplicates:
            raise AdaptivePlannerValidationError(
                "adaptive candidate ids must be unique: " + ", ".join(duplicates)
            )

        evaluator_candidates: list[EvidenceAwarePlanCandidate] = []
        for candidate in normalized:
            if (
                candidate.goal_content_id != frozen_goal.goal_content_id
                or candidate.repository_tree_id != frozen_goal.repository_tree_id
                or candidate.policy_digest != frozen_goal.policy_digest
            ):
                raise AdaptivePlannerValidationError(
                    "candidate hard receipts do not match the frozen goal bindings"
                )
            plan = candidate.plan
            authority_violations = list(plan.authority_violations)
            unresolved_conflicts = list(plan.unresolved_conflicts)
            authorized_scopes = plan.authorized_scopes
            proof_feasible = plan.proof_feasible

            if candidate.repair_transition is not None and (
                frozen_goal.goal_id not in candidate.repair_transition.goal_ids
            ):
                unresolved_conflicts.append("repair_goal_binding_mismatch")

            for receipt in candidate.hard_constraint_receipts:
                if receipt.passed:
                    continue
                reason = ",".join(receipt.reason_codes)
                if receipt.constraint is HardPlanConstraint.AUTHORITY:
                    authority_violations.append(f"receipt:{reason}")
                elif receipt.constraint is HardPlanConstraint.SCOPE:
                    authorized_scopes = ()
                elif receipt.constraint is HardPlanConstraint.SAFETY:
                    unresolved_conflicts.append(f"safety_receipt:{reason}")
                elif receipt.constraint is HardPlanConstraint.PROOF:
                    proof_feasible = False

            evaluator_candidates.append(
                replace(
                    plan,
                    authority_violations=tuple(authority_violations),
                    unresolved_conflicts=tuple(unresolved_conflicts),
                    authorized_scopes=authorized_scopes,
                    proof_feasible=proof_feasible,
                )
            )

        evaluation = evaluate_evidence_aware_plans(
            evaluator_candidates,
            policy=frozen_goal.policy,
        )
        by_id = {item.candidate_id: item for item in normalized}
        requirement_evidence: AuthorityNonCompensationEvidence | None = None
        if evaluation.selected is not None:
            selected = evaluation.selected.candidate
            selected_cost = _cost_millionths(selected)
            witnesses: list[
                tuple[str, int, str]
            ] = []
            rejected_ids = {item.candidate_id for item in evaluation.rejected}
            for candidate_id in sorted(rejected_ids):
                source = by_id[candidate_id]
                authority_receipt = source.receipt_for(
                    HardPlanConstraint.AUTHORITY
                )
                cost = _cost_millionths(source.plan)
                if not authority_receipt.passed and cost < selected_cost:
                    witnesses.append(
                        (candidate_id, cost, authority_receipt.receipt_id)
                    )
            if witnesses:
                requirement_evidence = AuthorityNonCompensationEvidence(
                    goal_content_id=frozen_goal.goal_content_id,
                    repository_tree_id=frozen_goal.repository_tree_id,
                    policy_digest=frozen_goal.policy_digest,
                    selected_candidate_id=selected.candidate_id,
                    selected_cost_millionths=selected_cost,
                    rejected_candidate_ids=tuple(item[0] for item in witnesses),
                    rejected_cost_millionths=tuple(item[1] for item in witnesses),
                    authority_receipt_ids=tuple(item[2] for item in witnesses),
                )

        return AdaptivePlanSelectionReceipt(
            frozen_goal=frozen_goal,
            evaluation=evaluation,
            hard_constraint_receipts=tuple(
                receipt
                for candidate in normalized
                for receipt in candidate.hard_constraint_receipts
            ),
            authority_non_compensation_evidence=requirement_evidence,
        )

    evaluate = select
    select_plan = select


class AdaptivePlanReceiptStore:
    """Append-only local persistence with content and path integrity checks."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def persist(self, receipt: AdaptivePlanSelectionReceipt) -> Path:
        if not isinstance(receipt, AdaptivePlanSelectionReceipt):
            raise AdaptivePlannerValidationError(
                "receipt must be AdaptivePlanSelectionReceipt"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{receipt.receipt_id}.json"
        encoded = (canonical_json(receipt.to_dict()) + "\n").encode("utf-8")
        try:
            with destination.open("xb") as handle:
                handle.write(encoded)
        except FileExistsError:
            if self.load(receipt.receipt_id) != receipt:
                raise AdaptivePlannerValidationError(
                    "existing adaptive-plan receipt has different content"
                )
        return destination

    def load(self, receipt_id: str) -> AdaptivePlanSelectionReceipt:
        identity = _text(receipt_id, "receipt_id")
        if "/" in identity or "\\" in identity or identity in {".", ".."}:
            raise AdaptivePlannerValidationError("unsafe adaptive receipt identity")
        path = self.directory / f"{identity}.json"
        if path.is_symlink():
            raise AdaptivePlannerValidationError(
                "adaptive receipt cannot be a symlink"
            )
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AdaptivePlannerValidationError(
                "adaptive receipt is unavailable or malformed"
            ) from exc
        receipt = AdaptivePlanSelectionReceipt.from_dict(payload)
        if receipt.receipt_id != identity:
            raise AdaptivePlannerValidationError(
                "adaptive receipt filename does not match content"
            )
        return receipt


def select_adaptive_plan(
    frozen_goal: FrozenPlanningGoal,
    candidates: Iterable[AdaptivePlanCandidate],
    *,
    max_candidates: int = 32,
) -> AdaptivePlanSelectionReceipt:
    """Functional convenience wrapper around :class:`AdaptivePlanner`."""

    return AdaptivePlanner(max_candidates=max_candidates).select(
        frozen_goal, candidates
    )


__all__ = [
    "ADAPTIVE_PLANNER_VERSION",
    "ADAPTIVE_PLAN_SELECTION_SCHEMA",
    "AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID",
    "AdaptivePlanCandidate",
    "AdaptivePlanReceiptStore",
    "AdaptivePlanSelectionReceipt",
    "AdaptivePlanner",
    "AdaptivePlannerValidationError",
    "AuthorityNonCompensationEvidence",
    "FrozenPlanningGoal",
    "GateProducerKind",
    "HardConstraintReceipt",
    "HardPlanConstraint",
    "adaptive_plan_candidate_snapshot_id",
    "select_adaptive_plan",
]
