"""Evidence-bound adaptive generation and selection for one frozen goal.

Candidate routing remains isolated in :mod:`task_proposal_router`, while this
module exposes the cohesive orchestration boundary.  The adaptive planner
always creates a deterministic baseline, accepts bounded declarations from
optional providers over the exact same immutable goal/context/tree/policy
snapshot, applies typed non-compensable admission receipts, and only then
delegates quality and cost ranking to :mod:`plan_evaluator`.

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
import json
import queue
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from .formal_replanner import RepairTransition
from .formal_verification_contracts import canonical_json, content_identity
from .adaptive_goal_refiner import (
    NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID,
    UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID,
    AdaptiveRefinementReceipt,
)
from .plan_evaluator import (
    AND_OR_PLAN_EVALUATOR_VERSION,
    EVIDENCE_AWARE_PLANNING_ACCEPTANCE_CRITERIA,
    EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION,
    AndOrPlanBranch,
    AndOrPlanEvaluation,
    EvidenceAwarePlanCandidate,
    EvidenceAwarePlanEvaluation,
    EvidenceAwarePlanPolicy,
    EvaluatedEvidenceAwarePlan,
    PlanBranchValidationError,
    PlanDimensionAssessment,
    PlanEvaluationDimension,
    PlanSearchHardConstraint,
    PlanSearchHardFailure,
    evaluate_and_or_plan_branches,
    evaluate_evidence_aware_plans,
    validate_and_or_plan_evaluation,
    validate_evidence_aware_plan_evaluation,
)
from .goal_quality import (
    TypedGoal,
    UncertaintyDisposition,
    validate_goal,
)
from .task_proposal_router import (
    AdaptiveCandidateProviderKind,
    AdaptiveCandidateRoutingResult,
    CandidateGenerationBounds,
    FrozenCandidateGenerationRequest,
    route_adaptive_plan_candidates,
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
ADAPTIVE_PLANNING_RUN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-planning-run@1"
)
HARD_CONSTRAINED_PLAN_SELECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/hard-constrained-plan-selection@1"
)
EVIDENCE_AWARE_PLANNING_COMPLETION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "evidence-aware-planning-completion-evidence@1"
)
EVIDENCE_AWARE_PLANNING_OBJECTIVE_ID: Final = "ASI-G030"
EVIDENCE_AWARE_PLANNING_OBJECTIVE_REVISION: Final = "ASI-G030@asi-080"
EVIDENCE_AWARE_PLANNING_REQUIRED_EXHAUSTIVE_RECEIPTS: Final = 2
EVIDENCE_AWARE_PLANNING_PRODUCING_TASK_IDS: Final[tuple[str, ...]] = (
    "ASI-008",
    "ASI-009",
)
EVIDENCE_AWARE_PLANNING_CHILD_GOAL_IDS: Final[tuple[str, ...]] = (
    "ASI-G097",
    "ASI-G098",
    "ASI-G115",
)

# ASI-G097: a cheaper authority-violating plan is rejected.
AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID: Final = (
    "173075880069453142914839090434430341799"
)

# Closed mandatory population used by the ASI-G097 objective-completion
# bridge. Callers may supply proof records, but may not narrow the objective
# to a convenient subset of these clauses.
AUTHORITY_NON_COMPENSATION_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    (
        "Deterministic evaluation covers acceptance evidence, assumptions, "
        "semantics, dependencies, conflicts, validation and proof feasibility, "
        "novelty, and bounded resource/token cost"
    ),
    "authority is a non-compensable gate",
    (
        "an authority-safe branch defeats every cheaper authority-violating "
        "branch"
    ),
    (
        "hard-gate receipts cannot be replayed after any candidate, formal-plan, "
        "or repair-transition change"
    ),
    (
        "and the selected-plan receipt binds the exact requirement ID, frozen "
        "goal/tree/policy identities, canonical candidate snapshots, a "
        "recomputed evaluation, the complete cheaper rejection set, result, "
        "and digest. Unsupported persisted planner, evaluator, and "
        "formal-replanner versions fail closed."
    ),
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
    values["estimated_runtime_seconds"] = (
        values.pop("estimated_runtime_milliseconds", 0) / 1_000
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
    values["max_estimated_runtime_seconds"] = (
        values.pop("max_estimated_runtime_milliseconds", 1_000_000_000)
        / 1_000
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

    def evaluate_objective_completion(
        self,
        *,
        current_state: Any = "active",
        evidence: Sequence[Any] = (),
        tasks_complete: bool = False,
        coverage: Any = None,
        analyzer_health: Any = None,
        exhaustion_quorum: Any = None,
        child_goals: Sequence[Any] = (),
        now: Any = None,
        freshness_seconds: float | None = None,
        clock_skew_seconds: float | None = None,
        analysis_inconclusive: bool = False,
        blocked_reason: str = "",
    ) -> "GoalCompletionDecision":
        """Evaluate ASI-G097 without allowing runtime output to self-certify.

        This selection receipt fixes the repository-tree boundary and may
        carry the objective's runtime witness. It is not, however, a fresh
        validation run, a criterion-to-implementation coverage map, analyzer
        health, or an independent exhaustive scan receipt. Those four proof
        classes must be supplied explicitly and are checked by the canonical
        two-phase completion gate.

        The acceptance population and repository tree are intentionally not
        caller arguments. Selection output, evaluator diagnostics, hard-gate
        receipts, and formal-replanner routing metadata are also never
        forwarded as analysis or completion authority.
        """

        from .goal_completion import evaluate_goal_completion

        def payload(value: Any) -> dict[str, Any]:
            if isinstance(value, Mapping):
                return dict(value)
            converter = getattr(value, "to_dict", None)
            if callable(converter):
                converted = converter()
                if isinstance(converted, Mapping):
                    return dict(converted)
            return {}

        # ASI-G097 requires both health and safety to be explicit, never
        # inferred from a legacy status-only record.
        health_value = payload(analyzer_health)
        if not (
            str(health_value.get("status") or "").strip().lower() == "healthy"
            and health_value.get("healthy") is True
            and health_value.get("safe_for_completion_reasoning") is True
        ):
            health_value = {
                **health_value,
                "healthy": False,
                "safe_for_completion_reasoning": False,
            }

        # Each coverage row must name both the implementation and its
        # validation proof. The canonical gate additionally checks the exact
        # criterion population, verified status, current tree, and freshness.
        coverage_value = payload(coverage)
        coverage_rows_value = coverage_value.get("criteria")
        coverage_rows = (
            coverage_rows_value
            if isinstance(coverage_rows_value, list)
            else []
        )
        coverage_bindings_complete = bool(coverage_rows) and all(
            isinstance(row, Mapping)
            and bool(str(row.get("implementation") or "").strip())
            and bool(str(row.get("validation") or "").strip())
            for row in coverage_rows
        )
        if not coverage_bindings_complete:
            reasons_value = coverage_value.get("reason_codes")
            reasons = (
                list(reasons_value)
                if isinstance(reasons_value, (list, tuple))
                else []
            )
            coverage_value = {
                **coverage_value,
                "verified": False,
                "reason_codes": [
                    *reasons,
                    "coverage_missing_implementation_validation_binding",
                ],
            }

        # Count, independence, binding, and timestamp freshness remain
        # canonical-gate responsibilities. This boundary tightens member
        # semantics: every member is explicitly healthy, completion-safe, and
        # exhaustive.
        quorum_value = payload(exhaustion_quorum)
        quorum_members_value = quorum_value.get("members")
        quorum_members = (
            quorum_members_value
            if isinstance(quorum_members_value, list)
            else []
        )
        quorum_members_healthy = bool(quorum_members) and all(
            isinstance(member, Mapping)
            and member.get("healthy") is True
            and member.get("safe_for_completion_reasoning") is True
            and str(member.get("scan_mode") or "").strip().lower()
            == "exhaustive"
            for member in quorum_members
        )
        required_members = quorum_value.get("required_members")
        member_count = quorum_value.get("member_count")
        configured_count_met = (
            isinstance(required_members, int)
            and not isinstance(required_members, bool)
            and required_members > 0
            and isinstance(member_count, int)
            and not isinstance(member_count, bool)
            and member_count == len(quorum_members)
            and member_count >= required_members
        )
        member_ids = [
            str(member.get("member_id") or "").strip()
            for member in quorum_members
            if isinstance(member, Mapping)
        ]
        receipt_ids = [
            str(member.get("receipt_cid") or "").strip()
            for member in quorum_members
            if isinstance(member, Mapping)
        ]
        channels = [
            str(member.get("evidence_channel") or "").strip()
            for member in quorum_members
            if isinstance(member, Mapping)
        ]

        def independent(values: Sequence[str]) -> bool:
            return (
                len(values) == len(quorum_members)
                and all(values)
                and len(values) == len(set(values))
            )

        binding_value = quorum_value.get("binding")
        binding = (
            dict(binding_value)
            if isinstance(binding_value, Mapping)
            else {}
        )
        binding_is_current = (
            binding.get("tree_id") == self.frozen_goal.repository_tree_id
            and all(
                isinstance(member, Mapping)
                and isinstance(member.get("binding"), Mapping)
                and dict(member["binding"]) == binding
                for member in quorum_members
            )
        )
        if not (
            quorum_members_healthy
            and configured_count_met
            and independent(member_ids)
            and independent(receipt_ids)
            and independent(channels)
            and binding_is_current
        ):
            quorum_value = {
                **quorum_value,
                "satisfied": False,
                "quorum_met": False,
            }

        values: dict[str, Any] = {
            "current_state": current_state,
            "acceptance_criteria": (
                AUTHORITY_NON_COMPENSATION_ACCEPTANCE_CRITERIA
            ),
            "evidence": evidence,
            "tasks_complete": tasks_complete,
            "repository_tree": self.frozen_goal.repository_tree_id,
            "now": now,
            "analysis_inconclusive": analysis_inconclusive,
            "blocked_reason": blocked_reason,
            "coverage": coverage_value,
            "analyzer_health": health_value,
            "exhaustion_quorum": quorum_value,
            "child_goals": child_goals,
            "analysis_result": None,
            "require_completion_gate": True,
        }
        if freshness_seconds is not None:
            values["freshness_seconds"] = freshness_seconds
        if clock_skew_seconds is not None:
            values["clock_skew_seconds"] = clock_skew_seconds
        return evaluate_goal_completion(**values)

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


HardGateEvaluator = Callable[
    [EvidenceAwarePlanCandidate, FrozenPlanningGoal, FrozenCandidateGenerationRequest],
    Iterable[HardConstraintReceipt] | Mapping[HardPlanConstraint | str, Any],
]


def deterministic_hard_gate_receipts(
    plan: EvidenceAwarePlanCandidate,
    frozen_goal: FrozenPlanningGoal,
    request: FrozenCandidateGenerationRequest,
) -> tuple[HardConstraintReceipt, ...]:
    """Apply the local typed policy boundaries to one untrusted declaration.

    Deployments can inject stronger authorization/formal/proof adapters through
    :meth:`AdaptivePlanner.plan`.  This deterministic implementation is the
    safe baseline: it derives failures from frozen policy and candidate
    declarations, never from provider confidence or claimed assurance.
    """

    if (
        request.goal_content_id != frozen_goal.goal_content_id
        or request.repository_tree_id != frozen_goal.repository_tree_id
        or request.policy_digest != frozen_goal.policy_digest
    ):
        raise AdaptivePlannerValidationError(
            "candidate-generation request does not match the frozen goal"
        )
    policy = frozen_goal.policy
    normalized = lambda values: {" ".join(item.casefold().split()) for item in values}
    changed = normalized(plan.changed_scopes)
    candidate_authority = normalized(plan.authorized_scopes)
    policy_authority = normalized(policy.allowed_scopes)
    decisions: Mapping[HardPlanConstraint, tuple[bool, GateProducerKind, tuple[str, ...]]] = {
        HardPlanConstraint.AUTHORITY: (
            not plan.authority_violations,
            GateProducerKind.AUTHORIZATION_ENGINE,
            tuple(plan.authority_violations) or ("authority_policy_satisfied",),
        ),
        HardPlanConstraint.SCOPE: (
            changed <= candidate_authority and changed <= policy_authority,
            GateProducerKind.AUTHORIZATION_ENGINE,
            (
                ("scope_policy_satisfied",)
                if changed <= candidate_authority and changed <= policy_authority
                else ("scope_not_authorized",)
            ),
        ),
        HardPlanConstraint.SAFETY: (
            not plan.unresolved_conflicts,
            GateProducerKind.FORMAL_VALIDATOR,
            tuple(plan.unresolved_conflicts) or ("safety_checks_satisfied",),
        ),
        HardPlanConstraint.PROOF: (
            plan.proof_feasible or not policy.require_proof,
            GateProducerKind.PROOF_VERIFIER,
            (
                ("proof_feasibility_satisfied",)
                if plan.proof_feasible or not policy.require_proof
                else ("required_proof_infeasible",)
            ),
        ),
    }
    snapshot_id = adaptive_plan_candidate_snapshot_id(
        plan,
        goal_content_id=frozen_goal.goal_content_id,
        repository_tree_id=frozen_goal.repository_tree_id,
        policy_digest=frozen_goal.policy_digest,
    )
    receipts: list[HardConstraintReceipt] = []
    for constraint in HardPlanConstraint:
        passed, producer, observations = decisions[constraint]
        reasons = () if passed else tuple(
            re.sub(r"[^a-z0-9_:-]+", "_", item.casefold()).strip("_")
            or f"{constraint.value}_failed"
            for item in observations
        )
        evidence_id = content_identity(
            {
                "kind": "deterministic_hard_gate",
                "constraint": constraint.value,
                "candidate_snapshot_id": snapshot_id,
                "context_id": request.context_id,
                "passed": passed,
                "observations": list(observations),
            }
        )
        receipts.append(
            HardConstraintReceipt(
                constraint=constraint,
                candidate_id=plan.candidate_id,
                candidate_snapshot_id=snapshot_id,
                goal_content_id=frozen_goal.goal_content_id,
                repository_tree_id=frozen_goal.repository_tree_id,
                policy_digest=frozen_goal.policy_digest,
                passed=passed,
                producer_kind=producer,
                producer_id=f"adaptive-planner:{producer.value}:v1",
                evidence_ids=(evidence_id,),
                reason_codes=reasons,
            )
        )
    return tuple(receipts)


def _normalize_gate_receipts(
    value: Iterable[HardConstraintReceipt] | Mapping[HardPlanConstraint | str, Any],
    *,
    plan: EvidenceAwarePlanCandidate,
    frozen_goal: FrozenPlanningGoal,
    request: FrozenCandidateGenerationRequest,
) -> tuple[HardConstraintReceipt, ...]:
    if not isinstance(value, Mapping):
        return tuple(
            item
            if isinstance(item, HardConstraintReceipt)
            else HardConstraintReceipt.from_dict(item)
            for item in value
        )
    default = {
        item.constraint: item
        for item in deterministic_hard_gate_receipts(plan, frozen_goal, request)
    }
    receipts: list[HardConstraintReceipt] = []
    for constraint in HardPlanConstraint:
        observation = value.get(constraint, value.get(constraint.value))
        if isinstance(observation, HardConstraintReceipt):
            receipts.append(observation)
            continue
        if isinstance(observation, Mapping):
            passed = observation.get("passed")
            reason_codes = tuple(observation.get("reason_codes") or ())
            evidence_ids = tuple(observation.get("evidence_ids") or ())
            producer_kind = observation.get(
                "producer_kind", default[constraint].producer_kind
            )
            producer_id = observation.get(
                "producer_id", default[constraint].producer_id
            )
        else:
            passed = observation
            reason_codes = () if passed is True else (f"{constraint.value}_failed",)
            evidence_ids = default[constraint].evidence_ids
            producer_kind = default[constraint].producer_kind
            producer_id = default[constraint].producer_id
        if not isinstance(passed, bool):
            raise AdaptivePlannerValidationError(
                f"hard gate {constraint.value} must return a boolean decision"
            )
        receipts.append(
            replace(
                default[constraint],
                passed=passed,
                reason_codes=tuple(reason_codes),
                evidence_ids=tuple(evidence_ids),
                producer_kind=GateProducerKind(producer_kind),
                producer_id=str(producer_id),
            )
        )
    return tuple(receipts)


@dataclass(frozen=True)
class AdaptivePlanningRunReceipt:
    """Durable orchestration, degradation, evaluation, and cost record."""

    routing: AdaptiveCandidateRoutingResult
    selection: AdaptivePlanSelectionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.routing, AdaptiveCandidateRoutingResult):
            raise AdaptivePlannerValidationError(
                "routing must be AdaptiveCandidateRoutingResult"
            )
        if not isinstance(self.selection, AdaptivePlanSelectionReceipt):
            raise AdaptivePlannerValidationError(
                "selection must be AdaptivePlanSelectionReceipt"
            )
        request = self.routing.request
        goal = self.selection.frozen_goal
        if (
            request.goal_id != goal.goal_id
            or request.goal_content_id != goal.goal_content_id
            or request.repository_tree_id != goal.repository_tree_id
            or request.policy_digest != goal.policy_digest
        ):
            raise AdaptivePlannerValidationError(
                "routing and selection do not share frozen goal bindings"
            )
        routed_ids = {item.candidate_id for item in self.routing.candidates}
        evaluated_ids = {item.candidate_id for item in self.selection.evaluation.ranked}
        if routed_ids != evaluated_ids:
            raise AdaptivePlannerValidationError(
                "selection must evaluate the complete routed candidate population"
            )
        receipts_by_candidate: dict[str, list[HardConstraintReceipt]] = {}
        for receipt in self.selection.hard_constraint_receipts:
            receipts_by_candidate.setdefault(receipt.candidate_id, []).append(receipt)
        for candidate in self.routing.candidates:
            expected_snapshot = adaptive_plan_candidate_snapshot_id(
                candidate,
                goal_content_id=goal.goal_content_id,
                repository_tree_id=goal.repository_tree_id,
                policy_digest=goal.policy_digest,
            )
            if any(
                receipt.candidate_snapshot_id != expected_snapshot
                for receipt in receipts_by_candidate[candidate.candidate_id]
            ):
                raise AdaptivePlannerValidationError(
                    "selection hard gates do not bind the routed candidate content"
                )

    @property
    def selected_candidate_id(self) -> str | None:
        return self.selection.selected_candidate_id

    @property
    def fallback_used(self) -> bool:
        return self.routing.used_fallback

    @property
    def non_selection_reasons(self) -> Mapping[str, tuple[str, ...]]:
        return self.selection.evaluation.non_selection_reasons

    @property
    def run_id(self) -> str:
        # Both nested receipts are independently content-addressed.  Hashing
        # their identities avoids relaxing Profile-G's prohibition on floats
        # merely because a frozen provider context contains a JSON number.
        return content_identity(
            {
                "schema": ADAPTIVE_PLANNING_RUN_SCHEMA,
                "routing_id": self.routing.routing_id,
                "selection_receipt_id": self.selection.receipt_id,
            }
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ADAPTIVE_PLANNING_RUN_SCHEMA,
            "routing": self.routing.to_dict(),
            "selection": self.selection.to_dict(),
            "selected_candidate_id": self.selected_candidate_id,
            "fallback_used": self.fallback_used,
            "selected_reason": (
                [
                    "highest_admissible_deterministic_quality_cost_score",
                    "stable_candidate_id_tie_break",
                ]
                if self.selected_candidate_id is not None
                else ["no_admissible_candidate"]
            ),
            "non_selection_reasons": {
                candidate_id: list(reasons)
                for candidate_id, reasons in sorted(
                    self.non_selection_reasons.items()
                )
            },
            "paired_quality_cost_metrics": [
                item.quality_cost_metrics.to_dict()
                for item in self.selection.evaluation.ranked
            ],
        }
        if include_identity:
            payload["run_id"] = self.run_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptivePlanningRunReceipt":
        allowed = {
            "schema",
            "run_id",
            "routing",
            "selection",
            "selected_candidate_id",
            "fallback_used",
            "selected_reason",
            "non_selection_reasons",
            "paired_quality_cost_metrics",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise AdaptivePlannerValidationError(
                "unknown adaptive-planning run fields: " + ", ".join(unknown)
            )
        if payload.get("schema") != ADAPTIVE_PLANNING_RUN_SCHEMA:
            raise AdaptivePlannerValidationError(
                "unsupported adaptive-planning run schema"
            )
        try:
            result = cls(
                routing=AdaptiveCandidateRoutingResult.from_dict(payload["routing"]),
                selection=AdaptivePlanSelectionReceipt.from_dict(payload["selection"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptivePlannerValidationError(
                f"invalid adaptive-planning run: {exc}"
            ) from exc
        expected = result.to_dict(include_identity=False)
        for field in (
            "selected_candidate_id",
            "fallback_used",
            "selected_reason",
            "non_selection_reasons",
            "paired_quality_cost_metrics",
        ):
            if payload.get(field) != expected[field]:
                raise AdaptivePlannerValidationError(
                    f"adaptive-planning {field} projection is inconsistent"
                )
        claimed = str(payload.get("run_id") or "")
        if claimed and claimed != result.run_id:
            raise AdaptivePlannerValidationError(
                "adaptive-planning run identity does not match content"
            )
        return result


@dataclass(frozen=True)
class EvidenceAwarePlanningCompletionEvidence:
    """Typed ASI-G030 runtime cohort, never completion authority by itself."""

    planning_run: AdaptivePlanningRunReceipt
    changed_refinement_receipt: AdaptiveRefinementReceipt
    backoff_source_receipt: AdaptiveRefinementReceipt
    unchanged_backoff_receipt: AdaptiveRefinementReceipt
    objective_id: str = EVIDENCE_AWARE_PLANNING_OBJECTIVE_ID
    objective_revision: str = EVIDENCE_AWARE_PLANNING_OBJECTIVE_REVISION
    producing_task_ids: tuple[str, ...] = (
        EVIDENCE_AWARE_PLANNING_PRODUCING_TASK_IDS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.planning_run, AdaptivePlanningRunReceipt):
            raise AdaptivePlannerValidationError(
                "planning completion evidence requires a typed planning run"
            )
        for name in (
            "changed_refinement_receipt",
            "backoff_source_receipt",
            "unchanged_backoff_receipt",
        ):
            if not isinstance(getattr(self, name), AdaptiveRefinementReceipt):
                raise AdaptivePlannerValidationError(
                    f"{name} must be an adaptive refinement receipt"
                )
        if self.objective_id != EVIDENCE_AWARE_PLANNING_OBJECTIVE_ID:
            raise AdaptivePlannerValidationError(
                "planning completion evidence objective is not ASI-G030"
            )
        if self.objective_revision != EVIDENCE_AWARE_PLANNING_OBJECTIVE_REVISION:
            raise AdaptivePlannerValidationError(
                "unsupported ASI-G030 objective revision"
            )
        task_ids = tuple(sorted(_strings(self.producing_task_ids, "producing_task_ids")))
        if task_ids != tuple(sorted(EVIDENCE_AWARE_PLANNING_PRODUCING_TASK_IDS)):
            raise AdaptivePlannerValidationError(
                "planning completion evidence must bind every producing task"
            )
        object.__setattr__(self, "producing_task_ids", task_ids)

        evaluation = self.planning_run.selection.evaluation
        if not evaluation.covers_every_planning_dimension:
            raise AdaptivePlannerValidationError(
                "planning run does not evaluate every candidate in every dimension"
            )
        if self.planning_run.selection.proved_requirement_ids != (
            AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID,
        ):
            raise AdaptivePlannerValidationError(
                "planning run lacks the bound hard-safety requirement witness"
            )
        changed = self.changed_refinement_receipt
        if changed.proved_requirement_ids != (
            NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID,
        ) or changed.new_counterexample_evidence is None:
            raise AdaptivePlannerValidationError(
                "changed refinement receipt lacks its counterexample witness"
            )
        backed_off = self.unchanged_backoff_receipt
        if backed_off.proved_requirement_ids != (
            UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID,
        ) or backed_off.unchanged_failure_backoff_evidence is None:
            raise AdaptivePlannerValidationError(
                "backoff receipt lacks its unchanged-failure witness"
            )
        try:
            backed_off.unchanged_failure_backoff_evidence.validate_source(
                self.backoff_source_receipt
            )
        except ValueError as exc:
            raise AdaptivePlannerValidationError(
                f"backoff source receipt is not authoritative: {exc}"
            ) from exc
        tree_ids = {
            self.planning_run.selection.frozen_goal.repository_tree_id,
            changed.repository_tree_id,
            self.backoff_source_receipt.repository_tree_id,
            backed_off.repository_tree_id,
        }
        if len(tree_ids) != 1:
            raise AdaptivePlannerValidationError(
                "planning completion cohort must bind one repository tree"
            )

    @property
    def repository_tree_id(self) -> str:
        return self.planning_run.selection.frozen_goal.repository_tree_id

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID,
                    NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID,
                    UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID,
                )
            )
        )

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def evidence_id(self) -> str:
        # Nested records are independently content-addressed. Hash their
        # identities so provider context floats never cross Profile-G's proof
        # boundary while any nested mutation still changes this cohort.
        return content_identity(
            {
                "schema": EVIDENCE_AWARE_PLANNING_COMPLETION_EVIDENCE_SCHEMA,
                "objective_id": self.objective_id,
                "objective_revision": self.objective_revision,
                "repository_tree_id": self.repository_tree_id,
                "requirement_ids": list(self.requirement_ids),
                "producing_task_ids": list(self.producing_task_ids),
                "planning_run_id": self.planning_run.run_id,
                "changed_refinement_receipt_id": (
                    self.changed_refinement_receipt.receipt_id
                ),
                "backoff_source_receipt_id": (
                    self.backoff_source_receipt.receipt_id
                ),
                "unchanged_backoff_receipt_id": (
                    self.unchanged_backoff_receipt.receipt_id
                ),
            }
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": EVIDENCE_AWARE_PLANNING_COMPLETION_EVIDENCE_SCHEMA,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "repository_tree_id": self.repository_tree_id,
            "requirement_ids": list(self.requirement_ids),
            "producing_task_ids": list(self.producing_task_ids),
            "planning_run": self.planning_run.to_dict(),
            "changed_refinement_receipt": (
                self.changed_refinement_receipt.to_dict()
            ),
            "backoff_source_receipt": self.backoff_source_receipt.to_dict(),
            "unchanged_backoff_receipt": (
                self.unchanged_backoff_receipt.to_dict()
            ),
            "completion_authority": False,
            "safe_for_completion_reasoning": False,
        }
        if include_identity:
            payload["evidence_id"] = self.evidence_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EvidenceAwarePlanningCompletionEvidence":
        allowed = {
            "schema",
            "objective_id",
            "objective_revision",
            "repository_tree_id",
            "requirement_ids",
            "producing_task_ids",
            "planning_run",
            "changed_refinement_receipt",
            "backoff_source_receipt",
            "unchanged_backoff_receipt",
            "completion_authority",
            "safe_for_completion_reasoning",
            "evidence_id",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise AdaptivePlannerValidationError(
                "unknown planning completion evidence fields: "
                + ", ".join(unknown)
            )
        if payload.get("schema") != (
            EVIDENCE_AWARE_PLANNING_COMPLETION_EVIDENCE_SCHEMA
        ):
            raise AdaptivePlannerValidationError(
                "unsupported planning completion evidence schema"
            )
        if (
            payload.get("completion_authority") is not False
            or payload.get("safe_for_completion_reasoning") is not False
        ):
            raise AdaptivePlannerValidationError(
                "operational planning cohort cannot claim completion authority"
            )
        try:
            result = cls(
                planning_run=AdaptivePlanningRunReceipt.from_dict(
                    payload["planning_run"]
                ),
                changed_refinement_receipt=AdaptiveRefinementReceipt.from_dict(
                    payload["changed_refinement_receipt"]
                ),
                backoff_source_receipt=AdaptiveRefinementReceipt.from_dict(
                    payload["backoff_source_receipt"]
                ),
                unchanged_backoff_receipt=AdaptiveRefinementReceipt.from_dict(
                    payload["unchanged_backoff_receipt"]
                ),
                objective_id=payload.get("objective_id", ""),
                objective_revision=payload.get("objective_revision", ""),
                producing_task_ids=tuple(
                    payload.get("producing_task_ids") or ()
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptivePlannerValidationError(
                f"invalid planning completion evidence: {exc}"
            ) from exc
        expected = result.to_dict(include_identity=False)
        for name in ("repository_tree_id", "requirement_ids"):
            if payload.get(name) != expected[name]:
                raise AdaptivePlannerValidationError(
                    f"planning completion {name} projection is inconsistent"
                )
        claimed = str(payload.get("evidence_id") or "")
        if claimed and claimed != result.evidence_id:
            raise AdaptivePlannerValidationError(
                "planning completion evidence identity does not match content"
            )
        return result

    def evaluate_evidence_aware_planning_completion(
        self,
        *,
        producing_tasks: Sequence[Any] = (),
        current_state: Any = "active",
        evidence: Sequence[Any] = (),
        tasks_complete: bool = False,
        coverage: Any = None,
        analyzer_health: Any = None,
        exhaustion_quorum: Any = None,
        required_exhaustive_receipts: int = (
            EVIDENCE_AWARE_PLANNING_REQUIRED_EXHAUSTIVE_RECEIPTS
        ),
        child_goals: Sequence[Any] = (),
        now: Any = None,
        freshness_seconds: float | None = None,
        clock_skew_seconds: float | None = None,
        analysis_inconclusive: bool = False,
        blocked_reason: str = "",
    ) -> "GoalCompletionDecision":
        """Evaluate the closed ASI-G030 parent completion boundary."""

        from .goal_completion import evaluate_goal_completion
        from .scan_receipts import ExhaustionQuorumResult

        if (
            isinstance(required_exhaustive_receipts, bool)
            or not isinstance(required_exhaustive_receipts, int)
            or required_exhaustive_receipts
            != EVIDENCE_AWARE_PLANNING_REQUIRED_EXHAUSTIVE_RECEIPTS
        ):
            raise ValueError(
                "required_exhaustive_receipts must equal the configured "
                f"ASI-G030 count "
                f"{EVIDENCE_AWARE_PLANNING_REQUIRED_EXHAUSTIVE_RECEIPTS}"
            )

        def payload(value: Any) -> dict[str, Any]:
            if isinstance(value, Mapping):
                return dict(value)
            converter = getattr(value, "to_dict", None)
            if callable(converter):
                converted = converter()
                if isinstance(converted, Mapping):
                    return dict(converted)
            return {}

        task_values = [payload(item) for item in producing_tasks]
        task_ids = [
            str(item.get("task_id") or item.get("id") or "").strip()
            for item in task_values
        ]
        successful = {
            "completed",
            "complete",
            "verified",
            "verified_complete",
            "passed",
            "success",
            "succeeded",
        }
        producing_tasks_complete = (
            len(task_ids) == len(set(task_ids))
            and tuple(sorted(task_ids))
            == tuple(sorted(EVIDENCE_AWARE_PLANNING_PRODUCING_TASK_IDS))
            and all(
                str(item.get("status") or item.get("state") or "")
                .strip()
                .lower()
                in successful
                for item in task_values
            )
        )

        evidence_values = [payload(item) for item in evidence]
        receipts_by_criterion: dict[str, set[str]] = {}
        for item in evidence_values:
            source = item.get("evidence", item)
            source = source if isinstance(source, Mapping) else item
            criterion = " ".join(
                str(source.get("acceptance_criterion") or "")
                .strip()
                .lower()
                .split()
            )
            receipt_id = str(
                source.get(
                    "provenance_cid",
                    source.get("receipt_id", source.get("evidence_id", "")),
                )
                or ""
            ).strip()
            if criterion and receipt_id:
                receipts_by_criterion.setdefault(criterion, set()).add(
                    receipt_id
                )

        coverage_projection = getattr(coverage, "completion_gate_evidence", None)
        if callable(coverage_projection):
            try:
                projected = coverage_projection(
                    EVIDENCE_AWARE_PLANNING_OBJECTIVE_ID
                )
            except (TypeError, ValueError):
                projected = {}
            coverage_value = (
                dict(projected) if isinstance(projected, Mapping) else {}
            )
        else:
            coverage_value = payload(coverage)
        rows_value = coverage_value.get("criteria")
        rows = rows_value if isinstance(rows_value, list) else []
        expected_criteria = {
            " ".join(item.lower().split())
            for item in EVIDENCE_AWARE_PLANNING_ACCEPTANCE_CRITERIA
        }
        row_keys = [
            " ".join(
                str(
                    row.get(
                        "criterion",
                        row.get("acceptance_criterion", ""),
                    )
                    or ""
                )
                .lower()
                .split()
            )
            for row in rows
            if isinstance(row, Mapping)
        ]

        def implementation_bound(row: Mapping[str, Any]) -> bool:
            for name in (
                "implementation",
                "changed_files",
                "predicted_files",
                "ast_symbols",
                "interfaces",
            ):
                value = row.get(name)
                if isinstance(value, str) and value.strip():
                    return True
                if (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                    and any(str(item or "").strip() for item in value)
                ):
                    return True
            return False

        def validation_ids(row: Mapping[str, Any]) -> set[str]:
            raw: Any = row.get(
                "validation_receipt_ids",
                row.get("validation_receipt_id", ()),
            )
            if isinstance(raw, str):
                raw = (raw,)
            if not isinstance(raw, Sequence):
                return set()
            return {
                str(item or "").strip()
                for item in raw
                if str(item or "").strip()
            }

        coverage_bound = (
            len(row_keys) == len(expected_criteria)
            and len(row_keys) == len(set(row_keys))
            and set(row_keys) == expected_criteria
            and all(
                isinstance(row, Mapping)
                and implementation_bound(row)
                and bool(
                    validation_ids(row).intersection(
                        receipts_by_criterion.get(
                            " ".join(
                                str(
                                    row.get(
                                        "criterion",
                                        row.get(
                                            "acceptance_criterion",
                                            "",
                                        ),
                                    )
                                    or ""
                                )
                                .lower()
                                .split()
                            ),
                            set(),
                        )
                    )
                )
                for row in rows
            )
        )
        if not coverage_bound:
            reasons = coverage_value.get("reason_codes")
            reasons = list(reasons) if isinstance(reasons, (list, tuple)) else []
            coverage_value = {
                **coverage_value,
                "verified": False,
                "reason_codes": list(
                    dict.fromkeys(
                        [
                            *reasons,
                            "coverage_validation_receipt_unbound",
                        ]
                    )
                ),
            }

        expected_binding_fields = {
            "tree_id": self.repository_tree_id,
            "objective_id": EVIDENCE_AWARE_PLANNING_OBJECTIVE_ID,
            "objective_revision": EVIDENCE_AWARE_PLANNING_OBJECTIVE_REVISION,
        }
        health_value = payload(analyzer_health)
        health_binding = health_value.get("binding")
        health_binding = (
            dict(health_binding)
            if isinstance(health_binding, Mapping)
            else {}
        )
        binding_complete = all(
            health_binding.get(name) == value
            for name, value in expected_binding_fields.items()
        ) and all(
            str(health_binding.get(name) or "").strip()
            for name in (
                "repository_id",
                "analyzer_version",
                "configuration_revision",
            )
        )
        health_valid = (
            str(health_value.get("status") or "").strip().lower()
            == "healthy"
            and health_value.get("healthy") is True
            and health_value.get("safe_for_completion_reasoning") is True
            and binding_complete
        )
        if not health_valid:
            health_value = {
                **health_value,
                "healthy": False,
                "safe_for_completion_reasoning": False,
            }
        else:
            # ASI-G030 checks the exhaustive scan population through the
            # bound quorum below; expose that objective-specific contract in
            # the generic completion gate's health vocabulary.
            health_value = {**health_value, "exhaustive": True}

        evaluated_quorum = isinstance(
            exhaustion_quorum, ExhaustionQuorumResult
        )
        quorum_value = payload(exhaustion_quorum)
        members_value = quorum_value.get("members")
        members = members_value if isinstance(members_value, list) else []
        quorum_binding = quorum_value.get("binding")
        quorum_binding = (
            dict(quorum_binding)
            if isinstance(quorum_binding, Mapping)
            else {}
        )
        identifiers = tuple(
            [
                str(member.get("member_id") or "").strip()
                for member in members
                if isinstance(member, Mapping)
            ],
        )
        receipt_ids = tuple(
            str(member.get("receipt_cid") or "").strip()
            for member in members
            if isinstance(member, Mapping)
        )
        channels = tuple(
            str(member.get("evidence_channel") or "").strip()
            for member in members
            if isinstance(member, Mapping)
        )

        def independent(values: Sequence[str]) -> bool:
            return (
                len(values) == len(members)
                and all(values)
                and len(values) == len(set(values))
            )

        member_semantics = bool(members) and (
            evaluated_quorum
            or all(
                isinstance(member, Mapping)
                and member.get("healthy") is True
                and member.get("safe_for_completion_reasoning") is True
                and str(member.get("scan_mode") or "").lower()
                == "exhaustive"
                for member in members
            )
        )
        quorum_valid = (
            quorum_value.get("required_members")
            == EVIDENCE_AWARE_PLANNING_REQUIRED_EXHAUSTIVE_RECEIPTS
            and quorum_value.get("member_count") == len(members)
            and len(members)
            >= EVIDENCE_AWARE_PLANNING_REQUIRED_EXHAUSTIVE_RECEIPTS
            and quorum_value.get("satisfied") is True
            and member_semantics
            and independent(identifiers)
            and independent(receipt_ids)
            and independent(channels)
            and quorum_binding == health_binding
            and all(
                isinstance(member, Mapping)
                and isinstance(member.get("binding"), Mapping)
                and dict(member["binding"]) == health_binding
                for member in members
            )
        )
        if not quorum_valid:
            quorum_value = {
                **quorum_value,
                "satisfied": False,
                "quorum_met": False,
            }
        else:
            translated_members = []
            for member in members:
                channel = str(
                    member.get("evidence_channel") or ""
                ).strip().lower()
                scan_mode = str(member.get("scan_mode") or "").strip().lower()
                is_audit = (
                    "audit" in channel
                    or scan_mode == "audit"
                    or scan_mode.endswith("_audit")
                )
                translated_members.append(
                    {
                        **member,
                        "status": "passed",
                        "passed": True,
                        "healthy": True,
                        "safe_for_completion_reasoning": True,
                        "exhaustive": True,
                        "conclusive": True,
                        "uncontradicted": True,
                        "analyzer_version": (
                            str(member.get("analyzer_version") or "").strip()
                            or health_binding["analyzer_version"]
                        ),
                        "scan_mode": "audit" if is_audit else "exhaustive",
                    }
                )
            quorum_value = {
                **quorum_value,
                "members": translated_members,
            }

        child_values = [payload(item) for item in child_goals]
        child_ids = [
            str(item.get("goal_id") or item.get("id") or "").strip()
            for item in child_values
        ]
        child_population_complete = (
            len(child_ids) == len(set(child_ids))
            and tuple(sorted(child_ids))
            == tuple(sorted(EVIDENCE_AWARE_PLANNING_CHILD_GOAL_IDS))
        )
        child_bindings_complete = child_population_complete and all(
            isinstance(item.get("completion_gate"), Mapping)
            and isinstance(
                item["completion_gate"].get("evaluated_evidence"), Mapping
            )
            and item["completion_gate"]["evaluated_evidence"].get(
                "repository_tree"
            )
            == self.repository_tree_id
            and bool(item.get("proof_requirements"))
            for item in child_values
        )
        if not child_bindings_complete:
            child_values.append(
                {
                    "goal_id": "ASI-G030-required-descendant-population",
                    "state": "active",
                    "verified": False,
                    "completion_gate": {
                        "passed": False,
                        "reason_code": (
                            "required_descendant_population_or_binding_incomplete"
                        ),
                    },
                }
            )

        values: dict[str, Any] = {
            "current_state": current_state,
            "acceptance_criteria": (
                EVIDENCE_AWARE_PLANNING_ACCEPTANCE_CRITERIA
            ),
            "evidence": evidence,
            "tasks_complete": bool(
                tasks_complete and producing_tasks_complete
            ),
            "repository_tree": self.repository_tree_id,
            "now": now,
            "analysis_inconclusive": analysis_inconclusive,
            "blocked_reason": blocked_reason,
            "coverage": coverage_value,
            "analyzer_health": health_value,
            "exhaustion_quorum": quorum_value,
            "child_goals": child_values,
            "analysis_result": None,
            "require_completion_gate": True,
        }
        if freshness_seconds is not None:
            values["freshness_seconds"] = freshness_seconds
        if clock_skew_seconds is not None:
            values["clock_skew_seconds"] = clock_skew_seconds
        return evaluate_goal_completion(**values)


def _plan_admission_receipt_type() -> type:
    """Resolve the IR admission type lazily to keep compiler imports acyclic."""

    from .ir_constraint_compiler import PlanAdmissionReceipt

    return PlanAdmissionReceipt


def _admission_candidate_plan_id(receipt: Any) -> str:
    return _text(
        getattr(
            receipt,
            "candidate_plan_id",
            getattr(receipt, "candidate_id", ""),
        ),
        "admission receipt candidate_plan_id",
    )


def _admission_root_bindings(receipt: Any) -> tuple[tuple[str, str], ...]:
    """Return the semantic roots carried by a plan-admission receipt."""

    for name in (
        "semantic_roots",
        "semantic_root_ids",
        "ir_root_ids",
        "root_bindings",
    ):
        value = getattr(receipt, name, None)
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (
                        _text(key, f"{name} key"),
                        _text(item, f"{name}[{key}]"),
                    )
                    for key, item in value.items()
                )
            )
    roots: list[tuple[str, str]] = []
    for name in (
        "intent_root_id",
        "legal_root_id",
        "security_root_id",
        "program_root_id",
    ):
        value = str(getattr(receipt, name, "") or "").strip()
        if value:
            roots.append((name.removesuffix("_id"), _text(value, name)))
    return tuple(sorted(roots))


def _receipt_projection(receipt: Any) -> dict[str, Any]:
    converter = getattr(receipt, "to_dict", None)
    if not callable(converter):
        raise AdaptivePlannerValidationError(
            "plan admission receipts must provide canonical to_dict()"
        )
    payload = converter()
    if not isinstance(payload, Mapping):
        raise AdaptivePlannerValidationError(
            "plan admission receipt to_dict() must return a mapping"
        )
    return dict(payload)


@dataclass(frozen=True)
class HardConstrainedPlanSelectionReceipt:
    """Admission-first adaptive selection over one exact candidate population.

    IR admission and adaptive scoring intentionally remain separate receipts.
    This wrapper records their exact join: a failed IR candidate never enters
    the soft evaluator, while its complete diagnostics remain available to
    dependency-local replanning.
    """

    frozen_goal: FrozenPlanningGoal
    candidate_bindings: tuple[tuple[str, str], ...]
    admission_receipts: tuple[Any, ...]
    selection: AdaptivePlanSelectionReceipt | None
    semantic_roots: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_goal, FrozenPlanningGoal):
            raise AdaptivePlannerValidationError(
                "frozen_goal must be FrozenPlanningGoal"
            )
        bindings = tuple(
            (
                _text(item[0], "candidate_id"),
                _text(item[1], "candidate_plan_id"),
            )
            for item in self.candidate_bindings
        )
        if len({item[0] for item in bindings}) != len(bindings):
            raise AdaptivePlannerValidationError(
                "hard-constrained candidate ids must be unique"
            )
        if len({item[1] for item in bindings}) != len(bindings):
            raise AdaptivePlannerValidationError(
                "hard-constrained candidate plan ids must be unique"
            )
        bindings = tuple(sorted(bindings))
        object.__setattr__(self, "candidate_bindings", bindings)

        receipt_type = _plan_admission_receipt_type()
        receipts = tuple(self.admission_receipts)
        if any(not isinstance(item, receipt_type) for item in receipts):
            raise AdaptivePlannerValidationError(
                "admission_receipts must contain typed PlanAdmissionReceipt "
                "instances"
            )
        by_plan_id = {
            _admission_candidate_plan_id(item): item for item in receipts
        }
        expected_plan_ids = {item[1] for item in bindings}
        if len(by_plan_id) != len(receipts) or set(by_plan_id) != expected_plan_ids:
            raise AdaptivePlannerValidationError(
                "plan admission receipts must exactly cover the candidate "
                "plan ids"
            )
        for receipt in receipts:
            if (
                _text(
                    getattr(receipt, "repository_tree_id", ""),
                    "admission receipt repository_tree_id",
                )
                != self.frozen_goal.repository_tree_id
            ):
                raise AdaptivePlannerValidationError(
                    "plan admission receipt repository tree is stale"
                )
            if bool(getattr(receipt, "authorizes_execution", False)):
                raise AdaptivePlannerValidationError(
                    "plan admission cannot grant execution authority"
                )
        object.__setattr__(
            self,
            "admission_receipts",
            tuple(
                sorted(receipts, key=_admission_candidate_plan_id)
            ),
        )

        roots = tuple(
            sorted(
                (
                    _text(item[0], "semantic root domain"),
                    _text(item[1], "semantic root id"),
                )
                for item in self.semantic_roots
            )
        )
        object.__setattr__(self, "semantic_roots", roots)
        for receipt in receipts:
            receipt_roots = _admission_root_bindings(receipt)
            if roots and receipt_roots != roots:
                raise AdaptivePlannerValidationError(
                    "plan admission receipt semantic roots are stale"
                )

        adaptive_by_plan = {
            plan_id: candidate_id for candidate_id, plan_id in bindings
        }
        admitted_ids = {
            adaptive_by_plan[_admission_candidate_plan_id(item)]
            for item in receipts
            if bool(getattr(item, "admitted", False))
        }
        if self.selection is None:
            if admitted_ids:
                raise AdaptivePlannerValidationError(
                    "an admitted candidate requires an adaptive selection"
                )
        else:
            if not isinstance(self.selection, AdaptivePlanSelectionReceipt):
                raise AdaptivePlannerValidationError(
                    "selection must be AdaptivePlanSelectionReceipt"
                )
            if self.selection.frozen_goal != self.frozen_goal:
                raise AdaptivePlannerValidationError(
                    "adaptive selection does not match the frozen goal"
                )
            evaluated_ids = {
                item.candidate_id for item in self.selection.evaluation.ranked
            }
            if evaluated_ids != admitted_ids:
                raise AdaptivePlannerValidationError(
                    "adaptive selection must evaluate exactly the admitted "
                    "candidate population"
                )

    @property
    def admitted_candidate_ids(self) -> tuple[str, ...]:
        adaptive_by_plan = {
            plan_id: candidate_id
            for candidate_id, plan_id in self.candidate_bindings
        }
        return tuple(
            sorted(
                adaptive_by_plan[_admission_candidate_plan_id(item)]
                for item in self.admission_receipts
                if bool(getattr(item, "admitted", False))
            )
        )

    @property
    def rejected_candidate_ids(self) -> tuple[str, ...]:
        admitted = set(self.admitted_candidate_ids)
        return tuple(
            item[0] for item in self.candidate_bindings if item[0] not in admitted
        )

    @property
    def selected_candidate_id(self) -> str | None:
        return (
            self.selection.selected_candidate_id
            if self.selection is not None
            else None
        )

    @property
    def selected_plan_id(self) -> str | None:
        selected = self.selected_candidate_id
        if selected is None:
            return None
        return dict(self.candidate_bindings)[selected]

    @property
    def used_no_model_fallback(self) -> bool:
        """The admission-first path is entirely deterministic and model-free."""

        return True

    @property
    def selection_source(self) -> str:
        return "deterministic_no_model"

    @property
    def admission_receipt_ids(self) -> Mapping[str, str]:
        adaptive_by_plan = {
            plan_id: candidate_id
            for candidate_id, plan_id in self.candidate_bindings
        }
        return MappingProxyType(
            {
                adaptive_by_plan[_admission_candidate_plan_id(item)]: _text(
                    getattr(item, "receipt_id", ""),
                    "plan admission receipt_id",
                )
                for item in self.admission_receipts
            }
        )

    def _rejected_receipts_by_candidate(self) -> dict[str, Any]:
        adaptive_by_plan = {
            plan_id: candidate_id
            for candidate_id, plan_id in self.candidate_bindings
        }
        return {
            adaptive_by_plan[_admission_candidate_plan_id(item)]: item
            for item in self.admission_receipts
            if not bool(getattr(item, "admitted", False))
        }

    @property
    def rejection_reasons(self) -> Mapping[str, tuple[Any, ...]]:
        return MappingProxyType(
            {
                candidate_id: tuple(
                    getattr(
                        receipt,
                        "rejection_reasons",
                        getattr(receipt, "reason_codes", ()),
                    )
                )
                for candidate_id, receipt in sorted(
                    self._rejected_receipts_by_candidate().items()
                )
            }
        )

    @property
    def counterexamples(self) -> Mapping[str, tuple[Any, ...]]:
        return MappingProxyType(
            {
                candidate_id: tuple(getattr(receipt, "counterexamples", ()))
                for candidate_id, receipt in sorted(
                    self._rejected_receipts_by_candidate().items()
                )
            }
        )

    @property
    def local_replan_action_ids(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                candidate_id: tuple(
                    getattr(
                        receipt,
                        "local_replan_action_ids",
                        getattr(receipt, "replan_action_ids", ()),
                    )
                )
                for candidate_id, receipt in sorted(
                    self._rejected_receipts_by_candidate().items()
                )
            }
        )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": HARD_CONSTRAINED_PLAN_SELECTION_SCHEMA,
            "planner_version": ADAPTIVE_PLANNER_VERSION,
            "frozen_goal": self.frozen_goal.to_dict(),
            "candidate_bindings": [
                {
                    "candidate_id": candidate_id,
                    "candidate_plan_id": plan_id,
                }
                for candidate_id, plan_id in self.candidate_bindings
            ],
            "semantic_roots": dict(self.semantic_roots),
            "admission_receipts": [
                _receipt_projection(item) for item in self.admission_receipts
            ],
            "admitted_candidate_ids": list(self.admitted_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "rejection_reasons": {
                key: list(value) for key, value in self.rejection_reasons.items()
            },
            "counterexamples": {
                key: [
                    (
                        item.to_dict()
                        if callable(getattr(item, "to_dict", None))
                        else item
                    )
                    for item in value
                ]
                for key, value in self.counterexamples.items()
            },
            "local_replan_action_ids": {
                key: list(value)
                for key, value in self.local_replan_action_ids.items()
            },
            "admission_receipt_ids": dict(self.admission_receipt_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_plan_id": self.selected_plan_id,
            "used_no_model_fallback": self.used_no_model_fallback,
            "selection_source": self.selection_source,
            "selection": (
                self.selection.to_dict()
                if self.selection is not None
                else None
            ),
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload


class AdaptivePlanner:
    """Select an admissible branch for one frozen goal and emit its evidence."""

    def __init__(self, *, max_candidates: int = 32) -> None:
        self.max_candidates = _integer(
            max_candidates, "max_candidates", minimum=1
        )

    def select_hard_constrained(
        self,
        frozen_goal: FrozenPlanningGoal,
        candidates: Iterable[AdaptivePlanCandidate],
        admission_receipts: Mapping[str, Any],
        *,
        semantic_roots: Mapping[str, str] | None = None,
    ) -> HardConstrainedPlanSelectionReceipt:
        """Prune IR-rejected candidates before deterministic soft selection.

        ``admission_receipts`` is keyed by the exact formal plan identity (or
        by the adaptive candidate identity when no formal plan is attached).
        Every candidate must have exactly one typed receipt and extra receipts
        fail closed.  This keeps the admitted population independent of input
        order and prevents a model or a soft score from compensating for an IR
        domain failure.
        """

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
        candidate_ids = [item.candidate_id for item in normalized]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise AdaptivePlannerValidationError(
                "adaptive candidate ids must be unique"
            )

        bindings = tuple(
            (
                candidate.candidate_id,
                candidate.formal_plan_id or candidate.candidate_id,
            )
            for candidate in normalized
        )
        plan_ids = [item[1] for item in bindings]
        if len(set(plan_ids)) != len(plan_ids):
            raise AdaptivePlannerValidationError(
                "adaptive candidate formal plan ids must be unique"
            )
        if not isinstance(admission_receipts, Mapping):
            raise AdaptivePlannerValidationError(
                "admission_receipts must be a mapping"
            )
        normalized_receipts = {
            _text(key, "admission_receipts key"): value
            for key, value in admission_receipts.items()
        }
        expected_plan_ids = set(plan_ids)
        if set(normalized_receipts) != expected_plan_ids:
            missing = sorted(expected_plan_ids - set(normalized_receipts))
            extra = sorted(set(normalized_receipts) - expected_plan_ids)
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("extra=" + ",".join(extra))
            raise AdaptivePlannerValidationError(
                "plan admission receipt mapping must exactly cover candidate "
                "plan ids"
                + (": " + "; ".join(details) if details else "")
            )

        receipt_type = _plan_admission_receipt_type()
        for plan_id, receipt in normalized_receipts.items():
            if not isinstance(receipt, receipt_type):
                raise AdaptivePlannerValidationError(
                    "admission_receipts must contain typed "
                    "PlanAdmissionReceipt instances"
                )
            if _admission_candidate_plan_id(receipt) != plan_id:
                raise AdaptivePlannerValidationError(
                    "plan admission receipt mapping key does not match its "
                    "candidate plan id"
                )

        if semantic_roots is not None and not isinstance(semantic_roots, Mapping):
            raise AdaptivePlannerValidationError(
                "semantic_roots must be a mapping"
            )
        expected_roots = (
            tuple(
                sorted(
                    (
                        _text(key, "semantic_roots key"),
                        _text(value, f"semantic_roots[{key}]"),
                    )
                    for key, value in semantic_roots.items()
                )
            )
            if semantic_roots is not None
            else ()
        )
        receipt_roots = tuple(
            _admission_root_bindings(item)
            for item in normalized_receipts.values()
        )
        observed_roots = {item for item in receipt_roots if item}
        if observed_roots and any(not item for item in receipt_roots):
            raise AdaptivePlannerValidationError(
                "plan admission receipts incompletely bind semantic roots"
            )
        if expected_roots:
            if observed_roots != {expected_roots}:
                raise AdaptivePlannerValidationError(
                    "plan admission receipts do not bind the expected semantic "
                    "roots"
                )
        elif len(observed_roots) > 1:
            raise AdaptivePlannerValidationError(
                "plan admission receipts bind inconsistent semantic roots"
            )
        elif observed_roots:
            expected_roots = next(iter(observed_roots))

        adaptive_by_plan_id = {
            plan_id: candidate
            for candidate, (_, plan_id) in zip(normalized, bindings)
        }
        admitted = tuple(
            sorted(
                (
                    adaptive_by_plan_id[plan_id]
                    for plan_id, receipt in normalized_receipts.items()
                    if bool(getattr(receipt, "admitted", False))
                ),
                key=lambda item: item.candidate_id,
            )
        )
        selection = self.select(frozen_goal, admitted) if admitted else None
        return HardConstrainedPlanSelectionReceipt(
            frozen_goal=frozen_goal,
            candidate_bindings=bindings,
            admission_receipts=tuple(normalized_receipts.values()),
            selection=selection,
            semantic_roots=expected_roots,
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

    def plan(
        self,
        frozen_goal: FrozenPlanningGoal,
        context: Mapping[str, Any],
        *,
        providers: Mapping[
            AdaptiveCandidateProviderKind | str,
            Callable[[FrozenCandidateGenerationRequest], Any] | None,
        ] | None = None,
        bounds: CandidateGenerationBounds | None = None,
        baseline_factory: Callable[
            [object, Mapping[str, Any]],
            EvidenceAwarePlanCandidate | Mapping[str, Any],
        ] | None = None,
        hard_gate_evaluator: HardGateEvaluator = deterministic_hard_gate_receipts,
    ) -> AdaptivePlanningRunReceipt:
        """Generate, independently gate, evaluate, and select one bounded plan."""

        if not isinstance(frozen_goal, FrozenPlanningGoal):
            raise AdaptivePlannerValidationError(
                "frozen_goal must be FrozenPlanningGoal"
            )
        routing_kwargs: dict[str, Any] = {
            "providers": providers,
            "bounds": bounds
            or CandidateGenerationBounds(max_total_candidates=self.max_candidates),
        }
        if baseline_factory is not None:
            routing_kwargs["baseline_factory"] = baseline_factory
        routing = route_adaptive_plan_candidates(
            frozen_goal,
            context,
            **routing_kwargs,
        )
        gated: list[AdaptivePlanCandidate] = []
        for plan in routing.candidates:
            raw_receipts = hard_gate_evaluator(
                plan, frozen_goal, routing.request
            )
            receipts = _normalize_gate_receipts(
                raw_receipts,
                plan=plan,
                frozen_goal=frozen_goal,
                request=routing.request,
            )
            gated.append(
                AdaptivePlanCandidate(
                    plan=plan,
                    goal_content_id=frozen_goal.goal_content_id,
                    repository_tree_id=frozen_goal.repository_tree_id,
                    policy_digest=frozen_goal.policy_digest,
                    hard_constraint_receipts=receipts,
                )
            )
        selection = self.select(frozen_goal, gated)
        return AdaptivePlanningRunReceipt(routing=routing, selection=selection)


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


class AdaptivePlanningRunStore:
    """Append-only persistence for complete generation and selection runs."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def persist(self, receipt: AdaptivePlanningRunReceipt) -> Path:
        if not isinstance(receipt, AdaptivePlanningRunReceipt):
            raise AdaptivePlannerValidationError(
                "receipt must be AdaptivePlanningRunReceipt"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{receipt.run_id}.json"
        import json

        encoded = (
            json.dumps(
                receipt.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        try:
            with destination.open("xb") as handle:
                handle.write(encoded)
        except FileExistsError:
            if self.load(receipt.run_id) != receipt:
                raise AdaptivePlannerValidationError(
                    "existing adaptive-planning run has different content"
                )
        return destination

    def load(self, run_id: str) -> AdaptivePlanningRunReceipt:
        identity = _text(run_id, "run_id")
        if "/" in identity or "\\" in identity or identity in {".", ".."}:
            raise AdaptivePlannerValidationError(
                "unsafe adaptive-planning run identity"
            )
        path = self.directory / f"{identity}.json"
        if path.is_symlink():
            raise AdaptivePlannerValidationError(
                "adaptive-planning run cannot be a symlink"
            )
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AdaptivePlannerValidationError(
                "adaptive-planning run is unavailable or malformed"
            ) from exc
        receipt = AdaptivePlanningRunReceipt.from_dict(payload)
        if receipt.run_id != identity:
            raise AdaptivePlannerValidationError(
                "adaptive-planning filename does not match content"
            )
        return receipt


# ---------------------------------------------------------------------------
# Typed-goal bounded AND/OR planning (ASI-104)
# ---------------------------------------------------------------------------

AND_OR_SEARCH_REQUIREMENT_ID: Final[str] = (
    "194688567293502311990437777137664011508"
)
AND_OR_PLANNER_VERSION: Final[int] = 1
AND_OR_GRAPH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/and-or-plan-graph@1"
)
AND_OR_SEARCH_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/and-or-plan-search@1"
)


class AndOrNodeKind(str, Enum):
    AND = "and"
    OR = "or"
    PRODUCER = "producer"


class AndOrProducerKind(str, Enum):
    DETERMINISTIC_BASELINE = "deterministic_baseline"
    LLM = "llm"
    LEANSTRAL = "leanstral"
    ANALYSIS_PROVIDER = "analysis_provider"
    # The existing router's spelling remains accepted at this boundary.
    IPFS_DATASETS = "analysis_provider"


@dataclass(frozen=True)
class AndOrSearchBounds:
    """Finite traversal and provider-use limits for one search."""

    max_depth: int = 4
    max_nodes: int = 128
    max_branches: int = 64
    max_alternatives_per_or: int = 8
    max_tokens: int = 32_768
    max_time_milliseconds: int = 30_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        if self.max_depth < 3:
            raise AdaptivePlannerValidationError(
                "max_depth must permit AND, OR, and producer levels"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AndOrSearchBounds":
        if set(payload) != set(cls.__dataclass_fields__):
            raise AdaptivePlannerValidationError(
                "AND/OR search bounds must use the closed schema"
            )
        return cls(**dict(payload))


def _plain_json(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise AdaptivePlannerValidationError(
            "AND/OR planning context must be finite JSON data"
        ) from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class FrozenAndOrPlanningContext:
    """One immutable root/tree/policy/context capsule shared by all producers."""

    goal: TypedGoal
    repository_tree_id: str
    policy_revision: str
    context: Mapping[str, Any]
    context_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.goal, TypedGoal):
            raise AdaptivePlannerValidationError("goal must be a TypedGoal")
        validate_goal(self.goal)
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        plain = _plain_json(self.context)
        if not isinstance(plain, dict):
            raise AdaptivePlannerValidationError(
                "AND/OR planning context must be an object"
            )
        expected = content_identity(
            {
                "goal_content_id": self.goal.content_id,
                "root_content_id": self.goal.root.content_id,
                "repository_tree_id": self.repository_tree_id,
                "policy_revision": self.policy_revision,
                "context": plain,
            }
        )
        if self.context_id and self.context_id != expected:
            raise AdaptivePlannerValidationError(
                "context_id does not match the frozen AND/OR context"
            )
        object.__setattr__(self, "context_id", expected)
        object.__setattr__(self, "context", _freeze_json(plain))

    @property
    def goal_content_id(self) -> str:
        return self.goal.content_id

    @property
    def root_content_id(self) -> str:
        return self.goal.root.content_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.to_record(),
            "goal_content_id": self.goal_content_id,
            "root_content_id": self.root_content_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_revision": self.policy_revision,
            "context": _thaw_json(self.context),
            "context_id": self.context_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "FrozenAndOrPlanningContext":
        expected = {
            "goal",
            "goal_content_id",
            "root_content_id",
            "repository_tree_id",
            "policy_revision",
            "context",
            "context_id",
        }
        if set(payload) != expected:
            raise AdaptivePlannerValidationError(
                "frozen AND/OR context must use the closed schema"
            )
        result = cls(
            goal=TypedGoal.from_dict(payload.get("goal") or {}),
            repository_tree_id=payload.get("repository_tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            context=payload.get("context") or {},
            context_id=payload.get("context_id", ""),
        )
        if (
            payload.get("goal_content_id") != result.goal_content_id
            or payload.get("root_content_id") != result.root_content_id
        ):
            raise AdaptivePlannerValidationError(
                "frozen AND/OR goal identity is inconsistent"
            )
        return result


@dataclass(frozen=True)
class AndOrPlanAlternative:
    """Typed producer output for exactly one required obligation."""

    alternative_id: str
    obligation_id: str
    producer_kind: AndOrProducerKind
    goal_content_id: str
    repository_tree_id: str
    context_id: str
    evidence_ids: tuple[str, ...]
    reduced_uncertainty_ids: tuple[str, ...] = ()
    changed_scopes: tuple[str, ...] = ()
    authorized_scopes: tuple[str, ...] = ()
    authority_granted: bool = True
    dependencies: tuple[str, ...] = ()
    satisfied_dependencies: tuple[str, ...] = ()
    resource_available: bool = True
    fresh: bool = True
    validation_feasible: bool = True
    proof_feasible: bool = True
    critical_path_length: int = 1
    conflict_risk_millionths: int = 0
    estimated_cost_microunits: int = 0
    estimated_tokens: int = 0
    estimated_time_milliseconds: int = 0
    historical_failure_millionths: int = 0

    def __post_init__(self) -> None:
        for name in (
            "alternative_id",
            "obligation_id",
            "goal_content_id",
            "repository_tree_id",
            "context_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "producer_kind", AndOrProducerKind(self.producer_kind)
        )
        for name in (
            "evidence_ids",
            "reduced_uncertainty_ids",
            "changed_scopes",
            "authorized_scopes",
            "dependencies",
            "satisfied_dependencies",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name,
                    allow_empty=name != "evidence_ids",
                ),
            )
        for name in (
            "authority_granted",
            "resource_available",
            "fresh",
            "validation_feasible",
            "proof_feasible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise AdaptivePlannerValidationError(f"{name} must be boolean")
        for name in (
            "critical_path_length",
            "conflict_risk_millionths",
            "estimated_cost_microunits",
            "estimated_tokens",
            "estimated_time_milliseconds",
            "historical_failure_millionths",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        for name in (
            "conflict_risk_millionths",
            "historical_failure_millionths",
        ):
            if getattr(self, name) > 1_000_000:
                raise AdaptivePlannerValidationError(
                    f"{name} must be at most 1000000"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                getattr(self, name).value
                if name == "producer_kind"
                else list(getattr(self, name))
                if isinstance(getattr(self, name), tuple)
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AndOrPlanAlternative":
        if set(payload) != set(cls.__dataclass_fields__):
            raise AdaptivePlannerValidationError(
                "AND/OR alternative must use the closed schema"
            )
        return cls(**dict(payload))


@dataclass(frozen=True)
class AndOrPlanNode:
    node_id: str
    kind: AndOrNodeKind
    child_ids: tuple[str, ...] = ()
    obligation_id: str = ""
    alternative_id: str = ""
    producer_kind: AndOrProducerKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(self, "kind", AndOrNodeKind(self.kind))
        object.__setattr__(
            self,
            "child_ids",
            _ordered_strings(
                self.child_ids, "child_ids", allow_empty=self.kind is AndOrNodeKind.PRODUCER
            ),
        )
        for name in ("obligation_id", "alternative_id"):
            value = str(getattr(self, name) or "").strip()
            object.__setattr__(self, name, value)
        producer = (
            AndOrProducerKind(self.producer_kind)
            if self.producer_kind is not None
            else None
        )
        object.__setattr__(self, "producer_kind", producer)
        if self.kind is AndOrNodeKind.PRODUCER:
            if self.child_ids or not self.obligation_id or not self.alternative_id:
                raise AdaptivePlannerValidationError(
                    "producer nodes require obligation/alternative and no children"
                )
            if producer is None:
                raise AdaptivePlannerValidationError(
                    "producer node requires producer_kind"
                )
        else:
            if not self.child_ids:
                raise AdaptivePlannerValidationError(
                    "AND/OR nodes require at least one child"
                )
            if self.alternative_id or producer is not None:
                raise AdaptivePlannerValidationError(
                    "AND/OR nodes cannot carry producer metadata"
                )
            if (
                self.kind is AndOrNodeKind.AND
                and self.obligation_id
            ) or (
                self.kind is AndOrNodeKind.OR
                and not self.obligation_id
            ):
                raise AdaptivePlannerValidationError(
                    "only OR and producer nodes require an obligation"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "child_ids": list(self.child_ids),
            "obligation_id": self.obligation_id,
            "alternative_id": self.alternative_id,
            "producer_kind": (
                self.producer_kind.value if self.producer_kind else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AndOrPlanNode":
        if set(payload) != set(cls.__dataclass_fields__):
            raise AdaptivePlannerValidationError(
                "AND/OR node must use the closed schema"
            )
        return cls(**dict(payload))


@dataclass(frozen=True)
class AndOrPlanGraph:
    frozen_context_id: str
    goal_content_id: str
    root_node_id: str
    nodes: tuple[AndOrPlanNode, ...]
    alternatives: tuple[AndOrPlanAlternative, ...]
    max_depth: int
    truncated: bool = False

    def __post_init__(self) -> None:
        for name in ("frozen_context_id", "goal_content_id", "root_node_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        nodes = tuple(self.nodes)
        alternatives = tuple(self.alternatives)
        node_ids = [item.node_id for item in nodes]
        alternative_ids = [item.alternative_id for item in alternatives]
        if len(node_ids) != len(set(node_ids)) or self.root_node_id not in node_ids:
            raise AdaptivePlannerValidationError(
                "AND/OR graph node identities are invalid"
            )
        if len(alternative_ids) != len(set(alternative_ids)):
            raise AdaptivePlannerValidationError(
                "AND/OR graph alternative identities must be unique"
            )
        known = set(node_ids)
        if any(set(node.child_ids).difference(known) for node in nodes):
            raise AdaptivePlannerValidationError(
                "AND/OR graph contains a dangling child"
            )
        by_node_id = {item.node_id: item for item in nodes}
        if by_node_id[self.root_node_id].kind is not AndOrNodeKind.AND:
            raise AdaptivePlannerValidationError(
                "AND/OR graph root must be an AND node"
            )
        if any(
            by_node_id[item].kind is not AndOrNodeKind.OR
            for item in by_node_id[self.root_node_id].child_ids
        ):
            raise AdaptivePlannerValidationError(
                "AND root may contain only required OR obligations"
            )
        root_obligations = tuple(
            by_node_id[item].obligation_id
            for item in by_node_id[self.root_node_id].child_ids
        )
        if len(root_obligations) != len(set(root_obligations)):
            raise AdaptivePlannerValidationError(
                "AND root obligations must be unique"
            )
        for node in nodes:
            if node.kind is not AndOrNodeKind.OR:
                continue
            children = [by_node_id[item] for item in node.child_ids]
            if any(item.kind is not AndOrNodeKind.PRODUCER for item in children):
                raise AdaptivePlannerValidationError(
                    "OR nodes may contain only producer alternatives"
                )
            if any(item.obligation_id != node.obligation_id for item in children):
                raise AdaptivePlannerValidationError(
                    "OR producer alternatives must target their parent obligation"
                )
            if not any(
                item.producer_kind
                is AndOrProducerKind.DETERMINISTIC_BASELINE
                for item in children
            ):
                raise AdaptivePlannerValidationError(
                    "every OR node requires a deterministic baseline"
                )
        if any(
            item.goal_content_id != self.goal_content_id
            or item.context_id != self.frozen_context_id
            for item in alternatives
        ):
            raise AdaptivePlannerValidationError(
                "AND/OR alternatives are detached from the graph bindings"
            )
        producer_alternative_ids = {
            item.alternative_id
            for item in nodes
            if item.kind is AndOrNodeKind.PRODUCER
        }
        if producer_alternative_ids != set(alternative_ids):
            raise AdaptivePlannerValidationError(
                "producer nodes must exactly cover graph alternatives"
            )
        alternatives_by_id = {
            item.alternative_id: item for item in alternatives
        }
        if any(
            (
                alternatives_by_id[item.alternative_id].obligation_id
                != item.obligation_id
                or alternatives_by_id[item.alternative_id].producer_kind
                is not item.producer_kind
            )
            for item in nodes
            if item.kind is AndOrNodeKind.PRODUCER
        ):
            raise AdaptivePlannerValidationError(
                "producer nodes must match their alternatives"
            )
        visiting: set[str] = set()
        visited: set[str] = set()

        def depth(node_id: str) -> int:
            if node_id in visiting:
                raise AdaptivePlannerValidationError(
                    "AND/OR graph contains a cycle"
                )
            if node_id in visited:
                return 0
            visiting.add(node_id)
            node = by_node_id[node_id]
            child_depth = max((depth(item) for item in node.child_ids), default=0)
            visiting.remove(node_id)
            visited.add(node_id)
            return 1 + child_depth

        actual_depth = depth(self.root_node_id)
        if visited != known:
            raise AdaptivePlannerValidationError(
                "AND/OR graph contains nodes unreachable from its root"
            )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "alternatives", alternatives)
        object.__setattr__(
            self, "max_depth", _integer(self.max_depth, "max_depth", minimum=1)
        )
        if actual_depth > self.max_depth:
            raise AdaptivePlannerValidationError(
                "AND/OR graph exceeds its declared depth"
            )
        if not isinstance(self.truncated, bool):
            raise AdaptivePlannerValidationError("truncated must be boolean")

    @property
    def graph_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": AND_OR_GRAPH_SCHEMA,
            "planner_version": AND_OR_PLANNER_VERSION,
            "frozen_context_id": self.frozen_context_id,
            "goal_content_id": self.goal_content_id,
            "root_node_id": self.root_node_id,
            "nodes": [item.to_dict() for item in self.nodes],
            "alternatives": [item.to_dict() for item in self.alternatives],
            "max_depth": self.max_depth,
            "truncated": self.truncated,
        }
        if include_identity:
            payload["graph_id"] = self.graph_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AndOrPlanGraph":
        expected = {
            "schema",
            "planner_version",
            "graph_id",
            "frozen_context_id",
            "goal_content_id",
            "root_node_id",
            "nodes",
            "alternatives",
            "max_depth",
            "truncated",
        }
        if set(payload) != expected:
            raise AdaptivePlannerValidationError(
                "AND/OR graph must use the closed schema"
            )
        if (
            payload.get("schema") != AND_OR_GRAPH_SCHEMA
            or payload.get("planner_version") != AND_OR_PLANNER_VERSION
        ):
            raise AdaptivePlannerValidationError(
                "unsupported AND/OR graph version"
            )
        result = cls(
            frozen_context_id=payload.get("frozen_context_id", ""),
            goal_content_id=payload.get("goal_content_id", ""),
            root_node_id=payload.get("root_node_id", ""),
            nodes=tuple(
                AndOrPlanNode.from_dict(item)
                for item in payload.get("nodes") or ()
            ),
            alternatives=tuple(
                AndOrPlanAlternative.from_dict(item)
                for item in payload.get("alternatives") or ()
            ),
            max_depth=payload.get("max_depth", 0),
            truncated=payload.get("truncated"),
        )
        if payload.get("graph_id") != result.graph_id:
            raise AdaptivePlannerValidationError(
                "AND/OR graph identity does not match content"
            )
        return result


AndOrProvider = Callable[
    [FrozenAndOrPlanningContext],
    Iterable[AndOrPlanAlternative | Mapping[str, Any]],
]


def _call_bounded_and_or_provider(
    provider: AndOrProvider,
    frozen: FrozenAndOrPlanningContext,
    timeout_seconds: float,
    max_items: int,
) -> tuple[bool, Any]:
    """Isolate an optional provider behind the remaining wall-clock budget."""

    output: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            produced: list[Any] = []
            for item in provider(frozen):
                produced.append(item)
                if len(produced) > max_items:
                    raise AdaptivePlannerValidationError(
                        "provider alternative budget exceeded"
                    )
            output.put_nowait((True, tuple(produced)))
        except BaseException as exc:  # provider isolation boundary
            output.put_nowait((False, exc))

    worker = threading.Thread(
        target=invoke,
        name="and-or-plan-provider",
        daemon=True,
    )
    worker.start()
    worker.join(max(0.0, timeout_seconds))
    if worker.is_alive():
        return False, TimeoutError("AND/OR provider exceeded time budget")
    try:
        return output.get_nowait()
    except queue.Empty:
        return False, RuntimeError("AND/OR provider returned no result")


def _baseline_alternative(
    frozen: FrozenAndOrPlanningContext,
    obligation_id: str,
    evidence_ids: tuple[str, ...],
) -> AndOrPlanAlternative:
    goal = frozen.goal
    return AndOrPlanAlternative(
        alternative_id=f"baseline:{obligation_id}",
        obligation_id=obligation_id,
        producer_kind=AndOrProducerKind.DETERMINISTIC_BASELINE,
        goal_content_id=goal.content_id,
        repository_tree_id=frozen.repository_tree_id,
        context_id=frozen.context_id,
        evidence_ids=evidence_ids or (f"deterministic:{obligation_id}",),
        changed_scopes=goal.scope.include,
        authorized_scopes=goal.scope.include,
        dependencies=goal.scope.dependency_goal_ids,
        satisfied_dependencies=goal.scope.dependency_goal_ids,
        estimated_time_milliseconds=min(
            1_000, goal.resources.max_wall_seconds * 1_000
        ),
    )


def compile_typed_goal_to_and_or_graph(
    frozen: FrozenAndOrPlanningContext,
    alternatives: Iterable[AndOrPlanAlternative] = (),
    *,
    bounds: AndOrSearchBounds | None = None,
) -> AndOrPlanGraph:
    """Compile joint criteria to an AND root and producer choices to OR nodes."""

    if not isinstance(frozen, FrozenAndOrPlanningContext):
        raise AdaptivePlannerValidationError(
            "frozen must be FrozenAndOrPlanningContext"
        )
    resolved_bounds = bounds or AndOrSearchBounds()
    goal = frozen.goal
    criteria = tuple(goal.acceptance_criteria)
    supplied = tuple(alternatives)
    for item in supplied:
        if not isinstance(item, AndOrPlanAlternative):
            raise AdaptivePlannerValidationError(
                "alternatives must be AndOrPlanAlternative values"
            )
    criterion_ids = {item.criterion_id for item in criteria}
    if any(item.obligation_id not in criterion_ids for item in supplied):
        raise AdaptivePlannerValidationError(
            "alternative targets an unknown goal obligation"
        )

    all_alternatives: list[AndOrPlanAlternative] = []
    nodes: list[AndOrPlanNode] = []
    or_ids: list[str] = []
    truncated = False
    mandatory_nodes = 1 + 2 * len(criteria)
    if mandatory_nodes > resolved_bounds.max_nodes:
        raise AdaptivePlannerValidationError(
            "node budget cannot represent mandatory baselines"
        )
    optional_node_budget = resolved_bounds.max_nodes - mandatory_nodes
    by_obligation: dict[str, list[AndOrPlanAlternative]] = {
        item.criterion_id: [] for item in criteria
    }
    for item in sorted(supplied, key=lambda value: value.alternative_id):
        by_obligation[item.obligation_id].append(item)

    for criterion in criteria:
        choices = [
            _baseline_alternative(
                frozen,
                criterion.criterion_id,
                criterion.evidence_producer_ids,
            ),
            *by_obligation[criterion.criterion_id],
        ]
        # Canonical deduplication prevents a provider from shadowing baseline.
        unique: dict[str, AndOrPlanAlternative] = {}
        for item in choices:
            if item.alternative_id in unique:
                if (
                    unique[item.alternative_id].producer_kind
                    is AndOrProducerKind.DETERMINISTIC_BASELINE
                ):
                    truncated = True
                    continue
                raise AdaptivePlannerValidationError(
                    "duplicate AND/OR alternative identity"
                )
            unique[item.alternative_id] = item
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                item.producer_kind is not AndOrProducerKind.DETERMINISTIC_BASELINE,
                item.producer_kind.value,
                item.alternative_id,
            ),
        )
        if len(ordered) > resolved_bounds.max_alternatives_per_or:
            ordered = ordered[: resolved_bounds.max_alternatives_per_or]
            truncated = True
        leaf_ids: list[str] = []
        for alternative in ordered:
            if (
                alternative.producer_kind
                is not AndOrProducerKind.DETERMINISTIC_BASELINE
                and optional_node_budget <= 0
            ):
                truncated = True
                continue
            if alternative.producer_kind is not AndOrProducerKind.DETERMINISTIC_BASELINE:
                optional_node_budget -= 1
            leaf_id = f"producer:{alternative.alternative_id}"
            leaf_ids.append(leaf_id)
            all_alternatives.append(alternative)
            nodes.append(
                AndOrPlanNode(
                    node_id=leaf_id,
                    kind=AndOrNodeKind.PRODUCER,
                    obligation_id=criterion.criterion_id,
                    alternative_id=alternative.alternative_id,
                    producer_kind=alternative.producer_kind,
                )
            )
        or_id = f"or:{criterion.criterion_id}"
        or_ids.append(or_id)
        nodes.append(
            AndOrPlanNode(
                node_id=or_id,
                kind=AndOrNodeKind.OR,
                child_ids=tuple(leaf_ids),
                obligation_id=criterion.criterion_id,
            )
        )
    root_id = f"and:{goal.goal_id}"
    nodes.append(
        AndOrPlanNode(
            node_id=root_id,
            kind=AndOrNodeKind.AND,
            child_ids=tuple(or_ids),
        )
    )
    if len(nodes) > resolved_bounds.max_nodes:
        raise AdaptivePlannerValidationError("AND/OR graph node budget exceeded")
    return AndOrPlanGraph(
        frozen_context_id=frozen.context_id,
        goal_content_id=goal.content_id,
        root_node_id=root_id,
        nodes=tuple(nodes),
        alternatives=tuple(all_alternatives),
        max_depth=3,
        truncated=truncated,
    )


# Short spelling used by integration code and tests.
compile_typed_goal = compile_typed_goal_to_and_or_graph
compile_and_or_plan_graph = compile_typed_goal_to_and_or_graph


def _alternative_failures(
    alternative: AndOrPlanAlternative,
    frozen: FrozenAndOrPlanningContext,
    bounds: AndOrSearchBounds,
) -> tuple[PlanSearchHardFailure, ...]:
    goal = frozen.goal
    failures: dict[PlanSearchHardConstraint, list[str]] = {}

    def fail(constraint: PlanSearchHardConstraint, code: str) -> None:
        failures.setdefault(constraint, []).append(code)

    if not alternative.authority_granted:
        fail(PlanSearchHardConstraint.AUTHORITY, "authority_not_granted")
    allowed = {item.casefold() for item in goal.scope.include}
    changed = {item.casefold() for item in alternative.changed_scopes}
    authorized = {item.casefold() for item in alternative.authorized_scopes}
    if not changed <= allowed or not changed <= authorized:
        fail(PlanSearchHardConstraint.SCOPE, "scope_not_authorized")
    if not set(alternative.dependencies) <= set(
        alternative.satisfied_dependencies
    ):
        fail(PlanSearchHardConstraint.DEPENDENCY, "dependency_not_satisfied")
    resources = goal.resources
    if (
        not alternative.resource_available
        or alternative.estimated_tokens > min(resources.max_tokens, bounds.max_tokens)
        or alternative.estimated_time_milliseconds
        > resources.max_wall_seconds * 1_000
        or alternative.estimated_cost_microunits > resources.max_cost_microunits
    ):
        fail(PlanSearchHardConstraint.RESOURCE, "resource_envelope_exceeded")
    if (
        not alternative.fresh
        or alternative.goal_content_id != goal.content_id
        or alternative.repository_tree_id != frozen.repository_tree_id
        or alternative.context_id != frozen.context_id
    ):
        fail(PlanSearchHardConstraint.FRESHNESS, "frozen_context_mismatch")
    if not alternative.validation_feasible:
        fail(PlanSearchHardConstraint.VALIDATION, "validation_infeasible")
    if not alternative.proof_feasible:
        fail(PlanSearchHardConstraint.PROOF, "proof_infeasible")
    return tuple(
        PlanSearchHardFailure(constraint, tuple(codes))
        for constraint, codes in sorted(
            failures.items(), key=lambda item: item[0].value
        )
    )


def _combine_alternatives(
    alternatives: Sequence[AndOrPlanAlternative],
    frozen: FrozenAndOrPlanningContext,
    bounds: AndOrSearchBounds,
) -> AndOrPlanBranch:
    required = tuple(
        item.criterion_id for item in frozen.goal.acceptance_criteria
    )
    hard: dict[PlanSearchHardConstraint, set[str]] = {}
    for item in alternatives:
        for failure in _alternative_failures(item, frozen, bounds):
            hard.setdefault(failure.constraint, set()).update(
                failure.reason_codes
            )
    tokens = sum(item.estimated_tokens for item in alternatives)
    elapsed = sum(item.estimated_time_milliseconds for item in alternatives)
    cost = sum(item.estimated_cost_microunits for item in alternatives)
    resources = frozen.goal.resources
    if (
        tokens > min(bounds.max_tokens, resources.max_tokens)
        or elapsed > resources.max_wall_seconds * 1_000
        or cost > resources.max_cost_microunits
    ):
        hard.setdefault(PlanSearchHardConstraint.RESOURCE, set()).add(
            "aggregate_resource_envelope_exceeded"
        )
    alternative_ids = tuple(item.alternative_id for item in alternatives)
    branch_id = content_identity(
        {
            "goal_content_id": frozen.goal.content_id,
            "context_id": frozen.context_id,
            "alternatives": alternative_ids,
        }
    )
    return AndOrPlanBranch(
        branch_id=branch_id,
        goal_content_id=frozen.goal.content_id,
        repository_tree_id=frozen.repository_tree_id,
        context_id=frozen.context_id,
        alternative_ids=alternative_ids,
        producer_kinds=tuple(item.producer_kind.value for item in alternatives),
        required_obligation_ids=required,
        covered_obligation_ids=tuple(
            item.obligation_id for item in alternatives
        ),
        required_uncertainty_ids=tuple(
            item.uncertainty_id
            for item in frozen.goal.uncertainties
            if item.disposition
            in {
                UncertaintyDisposition.OPEN,
                UncertaintyDisposition.BLOCKING,
            }
        ),
        reduced_uncertainty_ids=tuple(
            uncertainty_id
            for item in alternatives
            for uncertainty_id in item.reduced_uncertainty_ids
        ),
        critical_path_length=max(
            (item.critical_path_length for item in alternatives), default=0
        ),
        conflict_risk_millionths=max(
            (item.conflict_risk_millionths for item in alternatives), default=0
        ),
        estimated_cost_microunits=cost,
        estimated_tokens=tokens,
        estimated_time_milliseconds=elapsed,
        historical_failure_millionths=max(
            (
                item.historical_failure_millionths
                for item in alternatives
            ),
            default=0,
        ),
        hard_failures=tuple(
            PlanSearchHardFailure(constraint, tuple(sorted(codes)))
            for constraint, codes in sorted(
                hard.items(), key=lambda item: item[0].value
            )
        ),
    )


@dataclass(frozen=True)
class AndOrSearchReceipt:
    frozen_context: FrozenAndOrPlanningContext
    graph: AndOrPlanGraph
    evaluation: AndOrPlanEvaluation
    bounds: AndOrSearchBounds
    visited_nodes: int
    generated_branches: int
    consumed_tokens: int
    elapsed_milliseconds: int
    termination_reason: str
    provider_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.graph.frozen_context_id != self.frozen_context.context_id:
            raise AdaptivePlannerValidationError(
                "graph is detached from frozen context"
            )
        if self.graph.goal_content_id != self.frozen_context.goal_content_id:
            raise AdaptivePlannerValidationError(
                "graph is detached from frozen goal"
            )
        if any(
            item.repository_tree_id
            != self.frozen_context.repository_tree_id
            for item in self.graph.alternatives
        ):
            raise AdaptivePlannerValidationError(
                "graph alternative is detached from repository tree"
            )
        if self.evaluation.evaluator_version != AND_OR_PLAN_EVALUATOR_VERSION:
            raise AdaptivePlannerValidationError(
                "unsupported AND/OR evaluation"
            )
        if any(
            branch.branch.goal_content_id
            != self.frozen_context.goal_content_id
            or branch.branch.repository_tree_id
            != self.frozen_context.repository_tree_id
            or branch.branch.context_id != self.frozen_context.context_id
            for branch in (
                *self.evaluation.ranked,
                *self.evaluation.pruned,
            )
        ):
            raise AdaptivePlannerValidationError(
                "evaluated branch is detached from frozen context"
            )
        for name in (
            "visited_nodes",
            "generated_branches",
            "consumed_tokens",
            "elapsed_milliseconds",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        if self.visited_nodes > self.bounds.max_nodes:
            raise AdaptivePlannerValidationError("node budget was exceeded")
        if self.generated_branches > self.bounds.max_branches:
            raise AdaptivePlannerValidationError("branch budget was exceeded")
        if self.generated_branches != len(
            (*self.evaluation.ranked, *self.evaluation.pruned)
        ):
            raise AdaptivePlannerValidationError(
                "generated branch count is inconsistent"
            )
        if self.consumed_tokens > self.bounds.max_tokens:
            raise AdaptivePlannerValidationError("token budget was exceeded")
        object.__setattr__(
            self,
            "termination_reason",
            _text(self.termination_reason, "termination_reason"),
        )
        object.__setattr__(
            self,
            "provider_failures",
            _strings(
                self.provider_failures,
                "provider_failures",
                allow_empty=True,
            ),
        )

    @property
    def selected(self) -> AndOrPlanBranch | None:
        return (
            self.evaluation.selected.branch
            if self.evaluation.selected is not None
            else None
        )

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return (AND_OR_SEARCH_REQUIREMENT_ID,)

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": AND_OR_SEARCH_RECEIPT_SCHEMA,
            "planner_version": AND_OR_PLANNER_VERSION,
            "requirement_ids": list(self.requirement_ids),
            "frozen_context": self.frozen_context.to_dict(),
            "graph": self.graph.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "bounds": self.bounds.to_dict(),
            "visited_nodes": self.visited_nodes,
            "generated_branches": self.generated_branches,
            "consumed_tokens": self.consumed_tokens,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "termination_reason": self.termination_reason,
            "provider_failures": list(self.provider_failures),
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AndOrSearchReceipt":
        expected = {
            "schema",
            "planner_version",
            "requirement_ids",
            "receipt_id",
            "frozen_context",
            "graph",
            "evaluation",
            "bounds",
            "visited_nodes",
            "generated_branches",
            "consumed_tokens",
            "elapsed_milliseconds",
            "termination_reason",
            "provider_failures",
        }
        if set(payload) != expected:
            raise AdaptivePlannerValidationError(
                "AND/OR search receipt must use the closed schema"
            )
        if (
            payload.get("schema") != AND_OR_SEARCH_RECEIPT_SCHEMA
            or payload.get("planner_version") != AND_OR_PLANNER_VERSION
            or tuple(payload.get("requirement_ids") or ())
            != (AND_OR_SEARCH_REQUIREMENT_ID,)
        ):
            raise AdaptivePlannerValidationError(
                "unsupported AND/OR search receipt version"
            )
        result = cls(
            frozen_context=FrozenAndOrPlanningContext.from_dict(
                payload.get("frozen_context") or {}
            ),
            graph=AndOrPlanGraph.from_dict(payload.get("graph") or {}),
            evaluation=AndOrPlanEvaluation.from_dict(
                payload.get("evaluation") or {}
            ),
            bounds=AndOrSearchBounds.from_dict(payload.get("bounds") or {}),
            visited_nodes=payload.get("visited_nodes", 0),
            generated_branches=payload.get("generated_branches", 0),
            consumed_tokens=payload.get("consumed_tokens", 0),
            elapsed_milliseconds=payload.get("elapsed_milliseconds", 0),
            termination_reason=payload.get("termination_reason", ""),
            provider_failures=payload.get("provider_failures") or (),
        )
        validate_and_or_plan_evaluation(result.evaluation)
        if payload.get("receipt_id") != result.receipt_id:
            raise AdaptivePlannerValidationError(
                "AND/OR search receipt identity does not match content"
            )
        return result


def search_typed_goal_plans(
    goal: TypedGoal,
    *,
    repository_tree_id: str,
    policy_revision: str,
    context: Mapping[str, Any],
    providers: Mapping[AndOrProducerKind | str, AndOrProvider | None] | None = None,
    alternatives: Iterable[AndOrPlanAlternative] = (),
    bounds: AndOrSearchBounds | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> AndOrSearchReceipt:
    """Run a deterministic, budgeted AND/OR search over one frozen goal."""

    resolved_bounds = bounds or AndOrSearchBounds()
    frozen = FrozenAndOrPlanningContext(
        goal=goal,
        repository_tree_id=repository_tree_id,
        policy_revision=policy_revision,
        context=context,
    )
    started = monotonic_clock()
    provider_failures: list[str] = []
    supplied = list(alternatives)
    if any(not isinstance(item, AndOrPlanAlternative) for item in supplied):
        raise AdaptivePlannerValidationError(
            "alternatives must be AndOrPlanAlternative values"
        )
    consumed_tokens = sum(item.estimated_tokens for item in supplied)
    if consumed_tokens > resolved_bounds.max_tokens:
        raise AdaptivePlannerValidationError(
            "caller alternatives exceed the search token budget"
        )
    normalized_providers = {
        AndOrProducerKind(key): value
        for key, value in (providers or {}).items()
    }
    for kind in (
        AndOrProducerKind.LLM,
        AndOrProducerKind.LEANSTRAL,
        AndOrProducerKind.ANALYSIS_PROVIDER,
    ):
        provider = normalized_providers.get(kind)
        if provider is None:
            continue
        if (monotonic_clock() - started) * 1_000 >= resolved_bounds.max_time_milliseconds:
            provider_failures.append(f"{kind.value}:search_time_budget_exhausted")
            break
        try:
            remaining_seconds = max(
                0.0,
                (
                    resolved_bounds.max_time_milliseconds
                    - (monotonic_clock() - started) * 1_000
                )
                / 1_000,
            )
            succeeded, result = _call_bounded_and_or_provider(
                provider,
                frozen,
                remaining_seconds,
                resolved_bounds.max_nodes,
            )
            if not succeeded:
                raise result
            produced = tuple(result)
            normalized = tuple(
                item
                if isinstance(item, AndOrPlanAlternative)
                else AndOrPlanAlternative.from_dict(item)
                for item in produced
            )
            if any(item.producer_kind is not kind for item in normalized):
                raise AdaptivePlannerValidationError(
                    "provider returned a branch with a different producer kind"
                )
            next_tokens = consumed_tokens + sum(
                item.estimated_tokens for item in normalized
            )
            if next_tokens > resolved_bounds.max_tokens:
                provider_failures.append(f"{kind.value}:token_budget_exhausted")
                continue
            consumed_tokens = next_tokens
            supplied.extend(normalized)
        except BaseException as exc:
            reason = (
                "search_time_budget_exhausted"
                if isinstance(exc, TimeoutError)
                else f"provider_failed:{type(exc).__name__}"
            )
            provider_failures.append(f"{kind.value}:{reason}")

    graph = compile_typed_goal_to_and_or_graph(
        frozen, supplied, bounds=resolved_bounds
    )
    by_obligation: dict[str, list[AndOrPlanAlternative]] = {}
    for item in graph.alternatives:
        by_obligation.setdefault(item.obligation_id, []).append(item)
    obligation_ids = tuple(
        item.criterion_id for item in goal.acceptance_criteria
    )
    generated: list[AndOrPlanBranch] = []
    termination_reason = "search_complete"

    def walk(index: int, selected: tuple[AndOrPlanAlternative, ...]) -> None:
        nonlocal termination_reason
        if len(generated) >= resolved_bounds.max_branches:
            termination_reason = "branch_budget_exhausted"
            return
        if (monotonic_clock() - started) * 1_000 >= resolved_bounds.max_time_milliseconds:
            termination_reason = "time_budget_exhausted"
            return
        if index >= len(obligation_ids):
            generated.append(
                _combine_alternatives(selected, frozen, resolved_bounds)
            )
            return
        for item in by_obligation.get(obligation_ids[index], ()):
            # A failed leaf is retained as an unscored complete branch; walking
            # below it cannot repair a non-compensable property.
            failures = _alternative_failures(item, frozen, resolved_bounds)
            if failures:
                remaining = tuple(
                    next(
                        choice
                        for choice in by_obligation[obligation_id]
                        if choice.producer_kind
                        is AndOrProducerKind.DETERMINISTIC_BASELINE
                    )
                    for obligation_id in obligation_ids[index + 1 :]
                )
                generated.append(
                    _combine_alternatives(
                        (*selected, item, *remaining),
                        frozen,
                        resolved_bounds,
                    )
                )
                if len(generated) >= resolved_bounds.max_branches:
                    termination_reason = "branch_budget_exhausted"
                    return
                continue
            walk(index + 1, (*selected, item))
            if termination_reason != "search_complete":
                return

    walk(0, ())
    if not generated:
        # Optional producers may consume their wall-clock allowance, but the
        # mandatory deterministic baseline is already compiled and requires no
        # additional provider work.
        baseline = tuple(
            next(
                item
                for item in by_obligation[obligation_id]
                if item.producer_kind
                is AndOrProducerKind.DETERMINISTIC_BASELINE
            )
            for obligation_id in obligation_ids
        )
        generated.append(
            _combine_alternatives(baseline, frozen, resolved_bounds)
        )
        termination_reason = "time_budget_exhausted_baseline_fallback"
    # Multiple failed prefixes can converge on the same canonical branch.
    unique = {item.branch_id: item for item in generated}
    evaluation = evaluate_and_or_plan_branches(unique.values())
    elapsed = min(
        resolved_bounds.max_time_milliseconds,
        max(0, int((monotonic_clock() - started) * 1_000)),
    )
    if graph.truncated and termination_reason == "search_complete":
        termination_reason = "node_or_alternative_budget_exhausted"
    return AndOrSearchReceipt(
        frozen_context=frozen,
        graph=graph,
        evaluation=evaluation,
        bounds=resolved_bounds,
        visited_nodes=len(graph.nodes),
        generated_branches=len(unique),
        consumed_tokens=consumed_tokens,
        elapsed_milliseconds=elapsed,
        termination_reason=termination_reason,
        provider_failures=tuple(provider_failures),
    )


search_and_or_plans = search_typed_goal_plans
plan_typed_goal = search_typed_goal_plans


@dataclass(frozen=True)
class AndOrPlannerBenchmark:
    valid_first_plans: int
    total_plans: int
    invalid_branches_promoted: int
    total_invalid_branches: int
    hard_constraint_violations: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        if self.valid_first_plans > self.total_plans:
            raise AdaptivePlannerValidationError(
                "valid_first_plans cannot exceed total_plans"
            )
        if self.invalid_branches_promoted > self.total_invalid_branches:
            raise AdaptivePlannerValidationError(
                "invalid promotion count cannot exceed invalid branch count"
            )

    @property
    def valid_first_rate_millionths(self) -> int:
        return (
            self.valid_first_plans * 1_000_000 // self.total_plans
            if self.total_plans
            else 0
        )

    @property
    def invalid_promotion_rate_millionths(self) -> int:
        return (
            self.invalid_branches_promoted
            * 1_000_000
            // self.total_invalid_branches
            if self.total_invalid_branches
            else 0
        )

    def to_dict(self) -> dict[str, int]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            "valid_first_rate_millionths": self.valid_first_rate_millionths,
            "invalid_promotion_rate_millionths": (
                self.invalid_promotion_rate_millionths
            ),
        }


@dataclass(frozen=True)
class AndOrPlannerPromotionGate:
    baseline: AndOrPlannerBenchmark
    candidate: AndOrPlannerBenchmark
    valid_first_improvement_millionths: int
    invalid_branch_reduction_millionths: int
    passed: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": AND_OR_SEARCH_REQUIREMENT_ID,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "valid_first_improvement_millionths": (
                self.valid_first_improvement_millionths
            ),
            "invalid_branch_reduction_millionths": (
                self.invalid_branch_reduction_millionths
            ),
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }


def evaluate_and_or_planner_promotion(
    baseline: AndOrPlannerBenchmark,
    candidate: AndOrPlannerBenchmark,
) -> AndOrPlannerPromotionGate:
    """Enforce the v2 +15pp valid-first or 25% invalid-branch gate."""

    valid_delta = (
        candidate.valid_first_rate_millionths
        - baseline.valid_first_rate_millionths
    )
    baseline_invalid = baseline.invalid_promotion_rate_millionths
    invalid_reduction = (
        (baseline_invalid - candidate.invalid_promotion_rate_millionths)
        * 1_000_000
        // baseline_invalid
        if baseline_invalid > 0
        else 0
    )
    reasons: list[str] = []
    if baseline.hard_constraint_violations or candidate.hard_constraint_violations:
        reasons.append("hard_constraint_violation")
    if valid_delta < 150_000 and invalid_reduction < 250_000:
        reasons.append("quality_threshold_not_met")
    return AndOrPlannerPromotionGate(
        baseline=baseline,
        candidate=candidate,
        valid_first_improvement_millionths=valid_delta,
        invalid_branch_reduction_millionths=invalid_reduction,
        passed=not reasons,
        reason_codes=tuple(reasons),
    )


evaluate_and_or_plan_promotion = evaluate_and_or_planner_promotion


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


def select_hard_constrained_plan(
    frozen_goal: FrozenPlanningGoal,
    candidates: Iterable[AdaptivePlanCandidate],
    admission_receipts: Mapping[str, Any],
    *,
    semantic_roots: Mapping[str, str] | None = None,
    max_candidates: int = 32,
) -> HardConstrainedPlanSelectionReceipt:
    """Functional admission-first wrapper around :class:`AdaptivePlanner`."""

    return AdaptivePlanner(
        max_candidates=max_candidates
    ).select_hard_constrained(
        frozen_goal,
        candidates,
        admission_receipts,
        semantic_roots=semantic_roots,
    )


def plan_adaptively(
    frozen_goal: FrozenPlanningGoal,
    context: Mapping[str, Any],
    *,
    providers: Mapping[
        AdaptiveCandidateProviderKind | str,
        Callable[[FrozenCandidateGenerationRequest], Any] | None,
    ] | None = None,
    bounds: CandidateGenerationBounds | None = None,
    max_candidates: int = 32,
    hard_gate_evaluator: HardGateEvaluator = deterministic_hard_gate_receipts,
) -> AdaptivePlanningRunReceipt:
    """Functional full-pipeline wrapper around :meth:`AdaptivePlanner.plan`."""

    return AdaptivePlanner(max_candidates=max_candidates).plan(
        frozen_goal,
        context,
        providers=providers,
        bounds=bounds,
        hard_gate_evaluator=hard_gate_evaluator,
    )


__all__ = [
    "ADAPTIVE_PLANNER_VERSION",
    "ADAPTIVE_PLAN_SELECTION_SCHEMA",
    "ADAPTIVE_PLANNING_RUN_SCHEMA",
    "HARD_CONSTRAINED_PLAN_SELECTION_SCHEMA",
    "AND_OR_GRAPH_SCHEMA",
    "AND_OR_PLANNER_VERSION",
    "AND_OR_SEARCH_RECEIPT_SCHEMA",
    "AND_OR_SEARCH_REQUIREMENT_ID",
    "AUTHORITY_NON_COMPENSATION_ACCEPTANCE_CRITERIA",
    "AUTHORITY_NON_COMPENSATION_REQUIREMENT_ID",
    "EVIDENCE_AWARE_PLANNING_ACCEPTANCE_CRITERIA",
    "EVIDENCE_AWARE_PLANNING_CHILD_GOAL_IDS",
    "EVIDENCE_AWARE_PLANNING_COMPLETION_EVIDENCE_SCHEMA",
    "EVIDENCE_AWARE_PLANNING_OBJECTIVE_ID",
    "EVIDENCE_AWARE_PLANNING_OBJECTIVE_REVISION",
    "EVIDENCE_AWARE_PLANNING_PRODUCING_TASK_IDS",
    "EVIDENCE_AWARE_PLANNING_REQUIRED_EXHAUSTIVE_RECEIPTS",
    "AdaptivePlanCandidate",
    "AdaptivePlanReceiptStore",
    "AdaptivePlanSelectionReceipt",
    "AdaptivePlanningRunReceipt",
    "AdaptivePlanningRunStore",
    "AdaptivePlanner",
    "AdaptivePlannerValidationError",
    "AndOrNodeKind",
    "AndOrPlanAlternative",
    "AndOrPlanGraph",
    "AndOrPlanNode",
    "AndOrPlannerBenchmark",
    "AndOrPlannerPromotionGate",
    "AndOrProducerKind",
    "AndOrProvider",
    "AndOrSearchBounds",
    "AndOrSearchReceipt",
    "AuthorityNonCompensationEvidence",
    "EvidenceAwarePlanningCompletionEvidence",
    "FrozenPlanningGoal",
    "GateProducerKind",
    "HardConstraintReceipt",
    "HardConstrainedPlanSelectionReceipt",
    "HardGateEvaluator",
    "HardPlanConstraint",
    "adaptive_plan_candidate_snapshot_id",
    "compile_and_or_plan_graph",
    "compile_typed_goal",
    "compile_typed_goal_to_and_or_graph",
    "deterministic_hard_gate_receipts",
    "evaluate_and_or_plan_promotion",
    "evaluate_and_or_planner_promotion",
    "plan_adaptively",
    "plan_typed_goal",
    "search_and_or_plans",
    "search_typed_goal_plans",
    "select_adaptive_plan",
    "select_hard_constrained_plan",
]
