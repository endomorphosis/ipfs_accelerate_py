"""Evidence-responsive, root-preserving goal refinement.

This module owns the policy boundary between runtime evidence and a formal
goal-refinement proposal.  Runtime observations are normalized into a closed
signal vocabulary and content-addressed without their wall-clock delivery
time.  Consequently, replaying the same failure cannot consume another model
call during its backoff window, while genuinely changed evidence is eligible
in the next supervisor cycle.

The controller deliberately admits at most one refinement per invocation.  A
candidate is admitted only after all of the following hold:

* the root goal and complete assumption set match the frozen request;
* the proposal changes a bounded number of non-root goals;
* an independent verifier proves the proposed child refinement; and
* the content-addressed receipt is durably committed.

The returned plan is the transaction result.  Callers must not apply a model
proposal directly.  Objective revision (changing the root or assumptions) is
outside this API and requires a separate operator-authorized workflow.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager, nullcontext
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Protocol

from .formal_planning_contracts import FormalWorkPlan
from .formal_verification_contracts import (
    ContractValidationError,
    canonical_json,
    content_identity,
)
from .goal_refinement_verification import (
    FrozenRefinementContext,
    RefinementVerificationResult,
)


ADAPTIVE_GOAL_REFINER_VERSION: Final = 2
NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID: Final = (
    "003778425160038348524906247302938706902"
)
UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID: Final = (
    "312819945606360295782005228058369235550"
)

SIGNAL_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/refinement-signal@1"
QUALITY_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/goal-quality@1"
REQUEST_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/adaptive-refinement-request@1"
CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-refinement-candidate@1"
)
RECEIPT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/adaptive-refinement-receipt@1"
REQUIREMENT_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/new-counterexample-refinement-evidence@1"
)

# This is the closed mandatory population for the objective completion bridge
# below.  Keeping it beside the evidence producer prevents a caller from
# accidentally narrowing the objective's prose acceptance contract while
# asking the generic goal-completion gate for a verdict.
NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA: Final = (
    (
        "A changed typed counterexample can generate and admit at most one "
        "bounded refinement in the next cycle"
    ),
    "The frozen root is never mutated",
    "The request and candidate remain on the frozen repository tree",
    (
        "Admission is policy gated, the candidate declares the exact bounded "
        "changed-goal set, and verification binds the exact candidate plan "
        "with a boolean proof result"
    ),
    (
        "The witness binds the exact requirement ID, trigger signal, request "
        "and evidence fingerprint, frozen root/tree/policy identities, "
        "previous and candidate plans, producer, verification receipt, "
        "refinement index, and content digest"
    ),
    (
        "Non-counterexample admissions remain non-authoritative for this "
        "requirement, and restored objective receipts reject unsupported "
        "versions, missing identities, and unknown fields"
    ),
)


class AdaptiveGoalRefinementError(ValueError):
    """An adaptive-refinement input violates the reviewed contract."""


class RefinementPersistenceError(RuntimeError):
    """The refinement receipt could not be durably committed."""


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise AdaptiveGoalRefinementError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise AdaptiveGoalRefinementError(f"{name} is required")
    if "\x00" in value:
        raise AdaptiveGoalRefinementError(f"{name} must not contain NUL bytes")
    return value


def _strings(
    value: Iterable[Any] | None, name: str, *, required: bool = False
) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, (str, bytes, bytearray, memoryview)):
        raise AdaptiveGoalRefinementError(f"{name} must be a sequence")
    else:
        result = tuple(sorted({_text(item, name) for item in value}))
    if required and not result:
        raise AdaptiveGoalRefinementError(f"{name} must not be empty")
    return result


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise AdaptiveGoalRefinementError(f"{name} must be an object with string keys")
    try:
        # Round-tripping canonical JSON gives us a defensive, JSON-only copy.
        result = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise AdaptiveGoalRefinementError(f"{name} is not canonical JSON: {exc}") from exc
    if not isinstance(result, dict):  # pragma: no cover - mapping invariant
        raise AdaptiveGoalRefinementError(f"{name} must be an object")
    return result


def _positive(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AdaptiveGoalRefinementError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdaptiveGoalRefinementError(f"{name} must be a non-negative integer")
    return value


def _enum(value: Any, cls: type[Enum], name: str) -> Any:
    if isinstance(value, cls):
        return value
    try:
        return cls(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise AdaptiveGoalRefinementError(f"{name} is unsupported") from exc


def _claimed(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id") or payload.get("receipt_id")
    if claimed not in (None, "", actual):
        raise AdaptiveGoalRefinementError(f"{noun} content identity does not match")


def _restored_record(
    payload: Mapping[str, Any],
    *,
    noun: str,
    schema: str,
    allowed_fields: frozenset[str],
    version: int | None = None,
    identity_field: str,
) -> None:
    """Fail closed before restoring an authoritative persisted record."""

    if not isinstance(payload, Mapping) or any(
        not isinstance(key, str) for key in payload
    ):
        raise AdaptiveGoalRefinementError(f"{noun} must be an object")
    if payload.get("schema") != schema:
        raise AdaptiveGoalRefinementError(f"unsupported {noun} schema")
    if version is not None and payload.get("version") != version:
        raise AdaptiveGoalRefinementError(f"unsupported {noun} version")
    unknown = sorted(set(payload) - allowed_fields)
    if unknown:
        raise AdaptiveGoalRefinementError(
            f"unknown {noun} fields: {', '.join(unknown)}"
        )
    identity = payload.get(identity_field)
    if not isinstance(identity, str) or not identity.strip():
        raise AdaptiveGoalRefinementError(f"{noun} identity is required")


class RefinementSignalKind(str, Enum):
    """Closed set of runtime changes that may justify goal refinement."""

    COUNTEREXAMPLE = "counterexample"
    STALE_EVIDENCE = "stale_evidence"
    REPEATED_FAILURE = "repeated_failure"
    CAPABILITY_CHANGE = "capability_change"
    INTERFACE_CHANGE = "interface_change"
    SCOPE_CHANGE = "scope_change"
    SCOPE_CONFLICT = "scope_conflict"
    RESOURCE_CHANGE = "resource_change"
    RESOURCE_INFEASIBLE = "resource_infeasible"

    # Compatibility spellings for callers that use the task language.
    UNAVAILABLE_CAPABILITY = "capability_change"
    INFEASIBLE_RESOURCES = "resource_infeasible"


class GoalDebtKind(str, Enum):
    MISSING_OUTCOME = "missing_outcome"
    MISSING_SCOPE = "missing_scope"
    MISSING_ASSUMPTIONS = "missing_assumptions"
    MISSING_NON_GOALS = "missing_non_goals"
    MISSING_ACCEPTANCE = "missing_acceptance"
    MISSING_EVIDENCE_PRODUCER = "missing_evidence_producer"
    MISSING_VALIDATION = "missing_validation"
    MISSING_FRESHNESS = "missing_freshness"
    MISSING_RESOURCE_ENVELOPE = "missing_resource_envelope"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    EXCESSIVE_BREADTH = "excessive_breadth"


class RefinementDecision(str, Enum):
    ADMITTED = "admitted"
    DUPLICATE = "duplicate"
    BACKED_OFF = "backed_off"
    BUDGET_EXHAUSTED = "budget_exhausted"
    GENERATION_FAILED = "generation_failed"
    CANDIDATE_REJECTED = "candidate_rejected"
    VERIFICATION_FAILED = "verification_failed"
    COMMIT_FAILED = "commit_failed"


class RefinementProducerKind(str, Enum):
    """Auditable proposal origin; no producer kind conveys proof authority."""

    DETERMINISTIC = "deterministic"
    FORMAL_REPLANNER = "formal_replanner"
    LEANSTRAL = "leanstral"
    LANGUAGE_MODEL = "language_model"
    OPERATOR = "operator"


@dataclass(frozen=True)
class RefinementSignal:
    """One typed runtime observation.

    ``observed_at`` is retained for audit but excluded from ``evidence_id``.
    Delivery-time changes therefore cannot bypass semantic deduplication.
    ``evidence_revision`` or ``details`` must change for evidence to be new.
    """

    kind: RefinementSignalKind
    subject_id: str
    evidence_revision: str
    observed_at: int
    failure_signature: str = ""
    occurrence_count: int = 1
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, RefinementSignalKind, "kind"))
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(
            self,
            "evidence_revision",
            _text(self.evidence_revision, "evidence_revision"),
        )
        object.__setattr__(self, "observed_at", _nonnegative(self.observed_at, "observed_at"))
        object.__setattr__(
            self,
            "failure_signature",
            _text(self.failure_signature, "failure_signature", required=False),
        )
        object.__setattr__(
            self,
            "occurrence_count",
            _positive(self.occurrence_count, "occurrence_count"),
        )
        object.__setattr__(self, "details", _mapping(self.details, "details"))
        if (
            self.kind is RefinementSignalKind.REPEATED_FAILURE
            and not self.failure_signature
        ):
            raise AdaptiveGoalRefinementError(
                "repeated_failure requires failure_signature"
            )

    @property
    def evidence_id(self) -> str:
        """Semantic fingerprint used for idempotency and backoff."""

        return content_identity(
            {
                "schema": SIGNAL_SCHEMA,
                "version": ADAPTIVE_GOAL_REFINER_VERSION,
                "kind": self.kind.value,
                "subject_id": self.subject_id,
                "evidence_revision": self.evidence_revision,
                "failure_signature": self.failure_signature,
                "details": self.details,
            }
        )

    @property
    def content_id(self) -> str:
        return self.evidence_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SIGNAL_SCHEMA,
            "version": ADAPTIVE_GOAL_REFINER_VERSION,
            "content_id": self.evidence_id,
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "evidence_revision": self.evidence_revision,
            "observed_at": self.observed_at,
            "failure_signature": self.failure_signature,
            "occurrence_count": self.occurrence_count,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementSignal":
        if payload.get("schema") not in (None, "", SIGNAL_SCHEMA):
            raise AdaptiveGoalRefinementError("unsupported refinement signal schema")
        result = cls(
            kind=payload.get("kind", ""),
            subject_id=payload.get("subject_id", ""),
            evidence_revision=payload.get(
                "evidence_revision", payload.get("revision", "")
            ),
            observed_at=payload.get("observed_at", 0),
            failure_signature=payload.get("failure_signature", ""),
            occurrence_count=payload.get("occurrence_count", 1),
            details=payload.get("details") or {},
        )
        _claimed(payload, result.content_id, "refinement signal")
        return result


GoalRefinementSignal = RefinementSignal


@dataclass(frozen=True)
class GoalQualityRecord:
    """Explicit quality envelope and deterministic goal-debt assessment."""

    goal_id: str
    outcome: str
    scope_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    non_goals: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    evidence_producer_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    freshness_horizon_seconds: int
    resource_envelope: Mapping[str, Any]
    unsupported_semantics: tuple[str, ...] = ()
    breadth: int = 1
    max_breadth: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "outcome", _text(self.outcome, "outcome", required=False)
        )
        for name in (
            "scope_ids",
            "assumption_ids",
            "non_goals",
            "acceptance_criteria",
            "evidence_producer_ids",
            "validation_ids",
            "unsupported_semantics",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        object.__setattr__(
            self,
            "freshness_horizon_seconds",
            _nonnegative(self.freshness_horizon_seconds, "freshness_horizon_seconds"),
        )
        object.__setattr__(
            self,
            "resource_envelope",
            _mapping(self.resource_envelope, "resource_envelope"),
        )
        object.__setattr__(self, "breadth", _positive(self.breadth, "breadth"))
        object.__setattr__(
            self, "max_breadth", _positive(self.max_breadth, "max_breadth")
        )

    @property
    def debt(self) -> tuple[GoalDebtKind, ...]:
        findings: list[GoalDebtKind] = []
        checks = (
            (not self.outcome, GoalDebtKind.MISSING_OUTCOME),
            (not self.scope_ids, GoalDebtKind.MISSING_SCOPE),
            (not self.assumption_ids, GoalDebtKind.MISSING_ASSUMPTIONS),
            (not self.non_goals, GoalDebtKind.MISSING_NON_GOALS),
            (not self.acceptance_criteria, GoalDebtKind.MISSING_ACCEPTANCE),
            (
                not self.evidence_producer_ids,
                GoalDebtKind.MISSING_EVIDENCE_PRODUCER,
            ),
            (not self.validation_ids, GoalDebtKind.MISSING_VALIDATION),
            (
                self.freshness_horizon_seconds == 0,
                GoalDebtKind.MISSING_FRESHNESS,
            ),
            (
                not self.resource_envelope,
                GoalDebtKind.MISSING_RESOURCE_ENVELOPE,
            ),
            (bool(self.unsupported_semantics), GoalDebtKind.UNSUPPORTED_SEMANTICS),
            (self.breadth > self.max_breadth, GoalDebtKind.EXCESSIVE_BREADTH),
        )
        for present, kind in checks:
            if present:
                findings.append(kind)
        return tuple(findings)

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": QUALITY_SCHEMA,
            "version": ADAPTIVE_GOAL_REFINER_VERSION,
            "goal_id": self.goal_id,
            "outcome": self.outcome,
            "scope_ids": self.scope_ids,
            "assumption_ids": self.assumption_ids,
            "non_goals": self.non_goals,
            "acceptance_criteria": self.acceptance_criteria,
            "evidence_producer_ids": self.evidence_producer_ids,
            "validation_ids": self.validation_ids,
            "freshness_horizon_seconds": self.freshness_horizon_seconds,
            "resource_envelope": self.resource_envelope,
            "unsupported_semantics": self.unsupported_semantics,
            "breadth": self.breadth,
            "max_breadth": self.max_breadth,
            "debt": tuple(item.value for item in self.debt),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}


GoalQuality = GoalQualityRecord


@dataclass(frozen=True)
class AdaptiveRefinementPolicy:
    """Finite refinement, generation, change, and retry budgets."""

    max_refinements_per_root: int = 3
    max_refinement_depth: int = 3
    max_model_calls_per_cycle: int = 1
    max_signals_per_cycle: int = 16
    max_changed_goals: int = 4
    initial_backoff_seconds: int = 60
    max_backoff_seconds: int = 3600

    def __post_init__(self) -> None:
        for name in (
            "max_refinements_per_root",
            "max_refinement_depth",
            "max_model_calls_per_cycle",
            "max_signals_per_cycle",
            "max_changed_goals",
            "initial_backoff_seconds",
            "max_backoff_seconds",
        ):
            _positive(getattr(self, name), name)
        if self.max_model_calls_per_cycle != 1:
            raise AdaptiveGoalRefinementError(
                "max_model_calls_per_cycle must be exactly one"
            )
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise AdaptiveGoalRefinementError(
                "initial_backoff_seconds cannot exceed max_backoff_seconds"
            )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


GoalRefinementPolicy = AdaptiveRefinementPolicy
RefinementLimits = AdaptiveRefinementPolicy


@dataclass(frozen=True)
class AdaptiveRefinementRequest:
    """One frozen goal/context plus the changed evidence for this cycle."""

    plan: FormalWorkPlan
    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    signals: tuple[RefinementSignal, ...]
    cycle_id: str
    refinement_depth: int = 0
    repository_tree_id: str = ""
    quality: GoalQualityRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FormalWorkPlan):
            raise AdaptiveGoalRefinementError("plan must be a FormalWorkPlan")
        object.__setattr__(
            self, "root_goal_id", _text(self.root_goal_id, "root_goal_id")
        )
        object.__setattr__(
            self,
            "root_goal_content_id",
            _text(self.root_goal_content_id, "root_goal_content_id"),
        )
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )
        object.__setattr__(self, "cycle_id", _text(self.cycle_id, "cycle_id"))
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id", required=False)
            or self.plan.repository_tree_id,
        )
        if not self.repository_tree_id:
            raise AdaptiveGoalRefinementError(
                "repository_tree_id is required for refinement"
            )
        if self.repository_tree_id != self.plan.repository_tree_id:
            raise AdaptiveGoalRefinementError(
                "request repository tree does not match the frozen plan"
            )
        object.__setattr__(
            self,
            "refinement_depth",
            _nonnegative(self.refinement_depth, "refinement_depth"),
        )
        signals = tuple(self.signals)
        if not signals or any(not isinstance(item, RefinementSignal) for item in signals):
            raise AdaptiveGoalRefinementError(
                "signals must contain at least one RefinementSignal"
            )
        # Evidence ordering and duplicates must not change request identity.
        object.__setattr__(
            self,
            "signals",
            tuple({item.evidence_id: item for item in signals}[key]
                  for key in sorted({item.evidence_id for item in signals})),
        )
        if self.quality is not None and not isinstance(
            self.quality, GoalQualityRecord
        ):
            raise AdaptiveGoalRefinementError("quality must be GoalQualityRecord")
        roots = [item for item in self.plan.goals if item.goal_id == self.root_goal_id]
        if len(roots) != 1 or roots[0].content_id != self.root_goal_content_id:
            raise AdaptiveGoalRefinementError(
                "request root does not match the frozen plan root"
            )

    @property
    def frozen_context(self) -> FrozenRefinementContext:
        return FrozenRefinementContext(
            root_goal_id=self.root_goal_id,
            root_goal_content_id=self.root_goal_content_id,
            assumption_ids=self.assumption_ids,
        )

    @property
    def evidence_fingerprint(self) -> str:
        return content_identity(
            {
                "schema": REQUEST_SCHEMA,
                "root_goal_content_id": self.root_goal_content_id,
                "assumption_ids": self.assumption_ids,
                "signal_evidence_ids": tuple(
                    item.evidence_id for item in self.signals
                ),
                "repository_tree_id": self.repository_tree_id,
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_SCHEMA,
            "version": ADAPTIVE_GOAL_REFINER_VERSION,
            "plan_id": self.plan.content_id,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "signals": tuple(item.to_dict() for item in self.signals),
            "cycle_id": self.cycle_id,
            "refinement_depth": self.refinement_depth,
            "repository_tree_id": self.repository_tree_id,
            "quality_id": self.quality.content_id if self.quality else "",
            "evidence_fingerprint": self.evidence_fingerprint,
        }


GoalRefinementRequest = AdaptiveRefinementRequest


@dataclass(frozen=True)
class AdaptiveRefinementCandidate:
    """Untrusted proposed child-plan refinement."""

    plan: FormalWorkPlan
    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    changed_goal_ids: tuple[str, ...]
    signal_kind: RefinementSignalKind
    producer_id: str
    producer_kind: RefinementProducerKind = RefinementProducerKind.DETERMINISTIC
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FormalWorkPlan):
            raise AdaptiveGoalRefinementError("candidate plan must be FormalWorkPlan")
        for name in ("root_goal_id", "root_goal_content_id", "producer_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )
        object.__setattr__(
            self,
            "changed_goal_ids",
            _strings(self.changed_goal_ids, "changed_goal_ids", required=True),
        )
        object.__setattr__(
            self,
            "signal_kind",
            _enum(self.signal_kind, RefinementSignalKind, "signal_kind"),
        )
        object.__setattr__(
            self,
            "producer_kind",
            _enum(self.producer_kind, RefinementProducerKind, "producer_kind"),
        )
        object.__setattr__(
            self, "rationale", _text(self.rationale, "rationale", required=False)
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_SCHEMA,
            "version": ADAPTIVE_GOAL_REFINER_VERSION,
            "plan_id": self.plan.content_id,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "changed_goal_ids": self.changed_goal_ids,
            "signal_kind": self.signal_kind.value,
            "producer_id": self.producer_id,
            "producer_kind": self.producer_kind.value,
            "rationale": self.rationale,
        }


GoalRefinementCandidate = AdaptiveRefinementCandidate
GoalRefinementProposal = AdaptiveRefinementCandidate


@dataclass(frozen=True)
class NewCounterexampleRefinementEvidence:
    """Concrete witness for the ASI-G098 changed-counterexample criterion.

    The witness is deliberately narrower than an admission receipt.  Other
    reviewed signal kinds may still produce useful refinements, but only one
    counterexample signal whose exact candidate plan was independently
    verified can carry this objective evidence.
    """

    counterexample_signal_id: str
    request_id: str
    evidence_fingerprint: str
    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    policy_id: str
    repository_tree_id: str
    previous_plan_id: str
    candidate_plan_id: str
    verification_receipt_id: str
    producer_id: str
    producer_kind: str
    refinement_index: int
    requirement_id: str = NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID
    evidence_producer_kind: str = "adaptive_goal_refinement"

    def __post_init__(self) -> None:
        for name in (
            "counterexample_signal_id",
            "request_id",
            "evidence_fingerprint",
            "root_goal_id",
            "root_goal_content_id",
            "policy_id",
            "repository_tree_id",
            "previous_plan_id",
            "candidate_plan_id",
            "verification_receipt_id",
            "producer_id",
            "producer_kind",
            "requirement_id",
            "evidence_producer_kind",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )
        object.__setattr__(
            self,
            "refinement_index",
            _positive(self.refinement_index, "refinement_index"),
        )
        if self.requirement_id != NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID:
            raise AdaptiveGoalRefinementError(
                "unsupported counterexample-refinement requirement id"
            )
        if self.evidence_producer_kind != "adaptive_goal_refinement":
            raise AdaptiveGoalRefinementError(
                "unsupported counterexample-refinement evidence producer"
            )

    @property
    def evidence_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def content_id(self) -> str:
        return self.evidence_id

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REQUIREMENT_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "evidence_producer_kind": self.evidence_producer_kind,
            "counterexample_signal_id": self.counterexample_signal_id,
            "request_id": self.request_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "previous_plan_id": self.previous_plan_id,
            "candidate_plan_id": self.candidate_plan_id,
            "verification_receipt_id": self.verification_receipt_id,
            "producer_id": self.producer_id,
            "producer_kind": self.producer_kind,
            "refinement_index": self.refinement_index,
        }
        if include_identity:
            payload["evidence_id"] = self.evidence_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "NewCounterexampleRefinementEvidence":
        _restored_record(
            payload,
            noun="counterexample-refinement evidence",
            schema=REQUIREMENT_EVIDENCE_SCHEMA,
            allowed_fields=frozenset(
                {
                    "schema",
                    "requirement_id",
                    "evidence_producer_kind",
                    "counterexample_signal_id",
                    "request_id",
                    "evidence_fingerprint",
                    "root_goal_id",
                    "root_goal_content_id",
                    "assumption_ids",
                    "policy_id",
                    "repository_tree_id",
                    "previous_plan_id",
                    "candidate_plan_id",
                    "verification_receipt_id",
                    "producer_id",
                    "producer_kind",
                    "refinement_index",
                    "evidence_id",
                }
            ),
            identity_field="evidence_id",
        )
        result = cls(
            counterexample_signal_id=payload.get("counterexample_signal_id", ""),
            request_id=payload.get("request_id", ""),
            evidence_fingerprint=payload.get("evidence_fingerprint", ""),
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            policy_id=payload.get("policy_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            previous_plan_id=payload.get("previous_plan_id", ""),
            candidate_plan_id=payload.get("candidate_plan_id", ""),
            verification_receipt_id=payload.get("verification_receipt_id", ""),
            producer_id=payload.get("producer_id", ""),
            producer_kind=payload.get("producer_kind", ""),
            refinement_index=payload.get("refinement_index", 0),
            requirement_id=payload.get("requirement_id", ""),
            evidence_producer_kind=payload.get("evidence_producer_kind", ""),
        )
        claimed = str(payload["evidence_id"])
        if claimed != result.evidence_id:
            raise AdaptiveGoalRefinementError(
                "counterexample-refinement evidence identity does not match"
            )
        return result


@dataclass(frozen=True)
class AdaptiveRefinementReceipt:
    """Durable evidence for one decision at the adaptive trust boundary."""

    decision: RefinementDecision
    request_id: str
    cycle_id: str
    evidence_fingerprint: str
    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    policy_id: str
    repository_tree_id: str
    producer_id: str
    producer_kind: str
    previous_plan_id: str
    candidate_plan_id: str
    verification_receipt_id: str
    model_called: bool
    attempted_at: int
    retry_after: int
    attempt_index: int
    refinement_index: int
    reason: str
    signal_ids: tuple[str, ...]
    signal_kinds: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    new_counterexample_evidence: NewCounterexampleRefinementEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decision", _enum(self.decision, RefinementDecision, "decision")
        )
        for name in (
            "request_id",
            "cycle_id",
            "evidence_fingerprint",
            "root_goal_id",
            "root_goal_content_id",
            "policy_id",
            "previous_plan_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "repository_tree_id",
            "producer_id",
            "producer_kind",
            "candidate_plan_id",
            "verification_receipt_id",
            "reason",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )
        object.__setattr__(
            self, "signal_ids", _strings(self.signal_ids, "signal_ids", required=True)
        )
        signal_kinds = tuple(
            _enum(item, RefinementSignalKind, "signal_kinds").value
            for item in self.signal_kinds
        )
        if len(signal_kinds) != len(self.signal_ids):
            raise AdaptiveGoalRefinementError(
                "signal kinds must correspond to every signal identity"
            )
        object.__setattr__(self, "signal_kinds", signal_kinds)
        object.__setattr__(
            self,
            "requirement_ids",
            _strings(self.requirement_ids, "requirement_ids"),
        )
        evidence = self.new_counterexample_evidence
        if evidence is not None:
            if not isinstance(evidence, NewCounterexampleRefinementEvidence):
                raise AdaptiveGoalRefinementError(
                    "invalid new-counterexample refinement evidence"
                )
            expected = {
                "request_id": self.request_id,
                "evidence_fingerprint": self.evidence_fingerprint,
                "root_goal_id": self.root_goal_id,
                "root_goal_content_id": self.root_goal_content_id,
                "assumption_ids": self.assumption_ids,
                "policy_id": self.policy_id,
                "repository_tree_id": self.repository_tree_id,
                "previous_plan_id": self.previous_plan_id,
                "candidate_plan_id": self.candidate_plan_id,
                "verification_receipt_id": self.verification_receipt_id,
                "producer_id": self.producer_id,
                "producer_kind": self.producer_kind,
                "refinement_index": self.refinement_index,
            }
            mismatched = [
                name
                for name, value in expected.items()
                if getattr(evidence, name) != value
            ]
            if mismatched:
                raise AdaptiveGoalRefinementError(
                    "counterexample evidence does not match receipt bindings: "
                    + ", ".join(mismatched)
                )
            if (
                len(self.signal_ids) != 1
                or self.signal_kinds
                != (RefinementSignalKind.COUNTEREXAMPLE.value,)
                or evidence.counterexample_signal_id != self.signal_ids[0]
            ):
                raise AdaptiveGoalRefinementError(
                    "counterexample evidence requires exactly one bound "
                    "counterexample signal"
                )
        if not isinstance(self.model_called, bool):
            raise AdaptiveGoalRefinementError("model_called must be boolean")
        for name in (
            "attempted_at",
            "retry_after",
            "attempt_index",
            "refinement_index",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.decision is RefinementDecision.ADMITTED:
            if not (
                self.model_called
                and self.candidate_plan_id
                and self.verification_receipt_id
                and self.producer_kind
            ):
                raise AdaptiveGoalRefinementError(
                    "admitted receipt requires generation, verification, and evidence binding"
                )
            if UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID in self.requirement_ids:
                raise AdaptiveGoalRefinementError(
                    "admitted receipt cannot claim unchanged-failure backoff evidence"
                )
        if self.decision is RefinementDecision.BACKED_OFF:
            if self.model_called or self.retry_after <= self.attempted_at:
                raise AdaptiveGoalRefinementError(
                    "backoff receipt must suppress generation until a future time"
                )
            if (
                self.signal_kinds
                == (RefinementSignalKind.REPEATED_FAILURE.value,)
                and UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID
                not in self.requirement_ids
            ):
                raise AdaptiveGoalRefinementError(
                    "backoff receipt is missing its objective evidence binding"
                )
            if NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID in self.requirement_ids:
                raise AdaptiveGoalRefinementError(
                    "backoff receipt cannot claim admitted-refinement evidence"
                )
        if self.decision not in {
            RefinementDecision.ADMITTED,
            RefinementDecision.BACKED_OFF,
        } and self.requirement_ids:
            raise AdaptiveGoalRefinementError(
                "non-evidentiary decisions cannot claim objective requirement coverage"
            )
        expected_requirements: list[str] = []
        if evidence is not None:
            if self.decision is not RefinementDecision.ADMITTED:
                raise AdaptiveGoalRefinementError(
                    "only an admitted receipt may carry counterexample evidence"
                )
            expected_requirements.append(NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID)
        if (
            self.decision is RefinementDecision.BACKED_OFF
            and self.signal_kinds
            == (RefinementSignalKind.REPEATED_FAILURE.value,)
        ):
            expected_requirements.append(UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID)
        if self.requirement_ids != tuple(sorted(expected_requirements)):
            raise AdaptiveGoalRefinementError(
                "receipt requirement projection is inconsistent"
            )

    @property
    def receipt_id(self) -> str:
        return content_identity(self._payload())

    @property
    def content_id(self) -> str:
        return self.receipt_id

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        """Requirements backed by a concrete witness in this receipt."""

        return self.requirement_ids

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Content identities of concrete objective evidence witnesses."""

        return (
            (self.new_counterexample_evidence.evidence_id,)
            if self.new_counterexample_evidence is not None
            else ()
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "version": ADAPTIVE_GOAL_REFINER_VERSION,
            "decision": self.decision.value,
            "request_id": self.request_id,
            "cycle_id": self.cycle_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "producer_id": self.producer_id,
            "producer_kind": self.producer_kind,
            "previous_plan_id": self.previous_plan_id,
            "candidate_plan_id": self.candidate_plan_id,
            "verification_receipt_id": self.verification_receipt_id,
            "model_called": self.model_called,
            "attempted_at": self.attempted_at,
            "retry_after": self.retry_after,
            "attempt_index": self.attempt_index,
            "refinement_index": self.refinement_index,
            "reason": self.reason,
            "signal_ids": self.signal_ids,
            "signal_kinds": self.signal_kinds,
            "requirement_ids": self.requirement_ids,
            "new_counterexample_evidence": (
                self.new_counterexample_evidence.to_dict()
                if self.new_counterexample_evidence is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptiveRefinementReceipt":
        _restored_record(
            payload,
            noun="refinement receipt",
            schema=RECEIPT_SCHEMA,
            version=ADAPTIVE_GOAL_REFINER_VERSION,
            allowed_fields=frozenset(
                {
                    "schema",
                    "version",
                    "decision",
                    "request_id",
                    "cycle_id",
                    "evidence_fingerprint",
                    "root_goal_id",
                    "root_goal_content_id",
                    "assumption_ids",
                    "policy_id",
                    "repository_tree_id",
                    "producer_id",
                    "producer_kind",
                    "previous_plan_id",
                    "candidate_plan_id",
                    "verification_receipt_id",
                    "model_called",
                    "attempted_at",
                    "retry_after",
                    "attempt_index",
                    "refinement_index",
                    "reason",
                    "signal_ids",
                    "signal_kinds",
                    "requirement_ids",
                    "new_counterexample_evidence",
                    "receipt_id",
                }
            ),
            identity_field="receipt_id",
        )
        result = cls(
            decision=payload.get("decision", ""),
            request_id=payload.get("request_id", ""),
            cycle_id=payload.get("cycle_id", ""),
            evidence_fingerprint=payload.get("evidence_fingerprint", ""),
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            policy_id=payload.get("policy_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            producer_id=payload.get("producer_id", ""),
            producer_kind=payload.get("producer_kind", ""),
            previous_plan_id=payload.get("previous_plan_id", ""),
            candidate_plan_id=payload.get("candidate_plan_id", ""),
            verification_receipt_id=payload.get("verification_receipt_id", ""),
            model_called=payload.get("model_called", False),
            attempted_at=payload.get("attempted_at", 0),
            retry_after=payload.get("retry_after", 0),
            attempt_index=payload.get("attempt_index", 0),
            refinement_index=payload.get("refinement_index", 0),
            reason=payload.get("reason", ""),
            signal_ids=tuple(payload.get("signal_ids") or ()),
            signal_kinds=tuple(payload.get("signal_kinds") or ()),
            requirement_ids=tuple(payload.get("requirement_ids") or ()),
            new_counterexample_evidence=(
                NewCounterexampleRefinementEvidence.from_dict(
                    payload["new_counterexample_evidence"]
                )
                if payload.get("new_counterexample_evidence") is not None
                else None
            ),
        )
        if payload["receipt_id"] != result.receipt_id:
            raise AdaptiveGoalRefinementError(
                "refinement receipt content identity does not match"
            )
        return result


GoalRefinementReceipt = AdaptiveRefinementReceipt


@dataclass(frozen=True)
class AdaptiveRefinementResult:
    receipt: AdaptiveRefinementReceipt
    admitted_plan: FormalWorkPlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, AdaptiveRefinementReceipt):
            raise AdaptiveGoalRefinementError(
                "result receipt must be AdaptiveRefinementReceipt"
            )
        expected = self.receipt.decision is RefinementDecision.ADMITTED
        if expected != (self.admitted_plan is not None):
            raise AdaptiveGoalRefinementError(
                "only an admitted receipt may carry an admitted plan"
            )
        if (
            self.admitted_plan is not None
            and self.admitted_plan.content_id != self.receipt.candidate_plan_id
        ):
            raise AdaptiveGoalRefinementError(
                "admitted plan does not match its transaction receipt"
            )

    @property
    def admitted(self) -> bool:
        return self.receipt.decision is RefinementDecision.ADMITTED

    @property
    def model_called(self) -> bool:
        return self.receipt.model_called

    @property
    def decision(self) -> RefinementDecision:
        return self.receipt.decision

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
        """Evaluate ASI-G098 without promoting runtime output into proof.

        The adaptive receipt fixes the repository-tree boundary, but it is not
        itself a validation run, criterion-coverage map, analyzer-health
        declaration, or independent exhaustive quorum.  Those records must be
        submitted explicitly and are checked by the canonical completion gate.

        The mandatory criterion population is intentionally not caller
        configurable.  A caller therefore cannot obtain a positive decision
        by omitting a difficult clause from the objective.  Likewise, this
        bridge never forwards an optional analysis result: a bounded
        refinement and the formal replanner's routing metadata cannot stand in
        for an explicitly healthy completion analyzer or exhaustion receipts.
        """

        from .goal_completion import evaluate_goal_completion

        def payload(value: Any) -> dict[str, Any]:
            if isinstance(value, Mapping):
                return dict(value)
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                result = to_dict()
                if isinstance(result, Mapping):
                    return dict(result)
            return {}

        # The generic completion gate retains compatibility with early health
        # records where "healthy" implied safety.  This objective's contract
        # is stricter: both facts must be explicit.
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

        # A coverage summary must identify both sides of every criterion
        # mapping, not merely claim a verified status.
        coverage_value = payload(coverage)
        coverage_rows = coverage_value.get("criteria")
        coverage_rows = coverage_rows if isinstance(coverage_rows, list) else []
        coverage_bindings_complete = bool(coverage_rows) and all(
            isinstance(row, Mapping)
            and bool(str(row.get("implementation") or "").strip())
            and bool(str(row.get("validation") or "").strip())
            for row in coverage_rows
        )
        if not coverage_bindings_complete:
            coverage_value = {
                **coverage_value,
                "verified": False,
                "reason_codes": [
                    *(
                        coverage_value.get("reason_codes")
                        if isinstance(
                            coverage_value.get("reason_codes"), list
                        )
                        else []
                    ),
                    "coverage_missing_implementation_validation_binding",
                ],
            }

        # Every configured quorum member must explicitly be a healthy,
        # completion-safe exhaustive receipt.  Independence, binding, count,
        # and timestamp freshness remain canonical gate responsibilities.
        quorum_value = payload(exhaustion_quorum)
        quorum_members = quorum_value.get("members")
        quorum_members = (
            quorum_members if isinstance(quorum_members, list) else []
        )
        quorum_members_healthy = bool(quorum_members) and all(
            isinstance(member, Mapping)
            and member.get("healthy") is True
            and member.get("safe_for_completion_reasoning") is True
            and str(member.get("scan_mode") or "").strip().lower()
            == "exhaustive"
            for member in quorum_members
        )
        if not quorum_members_healthy:
            quorum_value = {
                **quorum_value,
                "satisfied": False,
                "quorum_met": False,
            }

        values: dict[str, Any] = {
            "current_state": current_state,
            "acceptance_criteria": (
                NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA
            ),
            "evidence": evidence,
            "tasks_complete": tasks_complete,
            "repository_tree": self.receipt.repository_tree_id,
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


GoalRefinementResult = AdaptiveRefinementResult


class RefinementReceiptStore(Protocol):
    def receipts(self) -> tuple[AdaptiveRefinementReceipt, ...]: ...

    def append(self, receipt: AdaptiveRefinementReceipt) -> None: ...


class InMemoryRefinementStore:
    """Thread-safe receipt store useful for an embedded supervisor."""

    def __init__(self, receipts: Iterable[AdaptiveRefinementReceipt] = ()) -> None:
        self._lock = threading.RLock()
        self._receipts = list(receipts)

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def transaction_key(self) -> str:
        return f"memory:{id(self)}"

    def transaction(self):
        return self._lock

    def receipts(self) -> tuple[AdaptiveRefinementReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def append(self, receipt: AdaptiveRefinementReceipt) -> None:
        if not isinstance(receipt, AdaptiveRefinementReceipt):
            raise RefinementPersistenceError(
                "store accepts only AdaptiveRefinementReceipt"
            )
        with self._lock:
            if all(item.receipt_id != receipt.receipt_id for item in self._receipts):
                self._receipts.append(receipt)


class JsonlRefinementStore:
    """Append-only restart-safe receipt journal.

    Writes are flushed and fsynced before a plan can be returned as admitted.
    A malformed historical row fails closed rather than silently forgetting
    deduplication or budget state.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def transaction_key(self) -> str:
        return f"jsonl:{self.path.resolve()}"

    @contextmanager
    def transaction(self):
        """Serialize a complete lookup/generation/commit across processes."""

        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self._lock:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                with lock_path.open("a+", encoding="utf-8") as handle:
                    try:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                raise RefinementPersistenceError(
                    f"could not lock refinement receipt journal: {exc}"
                ) from exc

    def receipts(self) -> tuple[AdaptiveRefinementReceipt, ...]:
        with self._lock:
            if not self.path.exists():
                return ()
            result: list[AdaptiveRefinementReceipt] = []
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        if not isinstance(payload, Mapping):
                            raise ValueError("record is not an object")
                        result.append(AdaptiveRefinementReceipt.from_dict(payload))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise RefinementPersistenceError(
                    f"could not read refinement receipt journal: {exc}"
                ) from exc
            return tuple(result)

    def append(self, receipt: AdaptiveRefinementReceipt) -> None:
        if not isinstance(receipt, AdaptiveRefinementReceipt):
            raise RefinementPersistenceError(
                "store accepts only AdaptiveRefinementReceipt"
            )
        with self._lock:
            existing = self.receipts()
            if any(item.receipt_id == receipt.receipt_id for item in existing):
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(canonical_json(receipt.to_dict()) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise RefinementPersistenceError(
                    f"could not persist refinement receipt: {exc}"
                ) from exc


CandidateGenerator = Callable[
    [AdaptiveRefinementRequest],
    AdaptiveRefinementCandidate | FormalWorkPlan | Mapping[str, Any],
]
CandidateVerifier = Callable[
    [AdaptiveRefinementCandidate, AdaptiveRefinementRequest], Any
]


_GLOBAL_LOCK_GUARD = threading.Lock()
_GLOBAL_EVIDENCE_LOCKS: dict[tuple[str, str], threading.RLock] = {}


def _evidence_lock(store: RefinementReceiptStore, fingerprint: str) -> threading.RLock:
    key = (
        str(getattr(store, "transaction_key", f"store:{id(store)}")),
        fingerprint,
    )
    with _GLOBAL_LOCK_GUARD:
        return _GLOBAL_EVIDENCE_LOCKS.setdefault(key, threading.RLock())


class AdaptiveGoalRefiner:
    """Bounded, exactly-one adaptive refinement controller."""

    def __init__(
        self,
        generator: CandidateGenerator,
        verifier: CandidateVerifier,
        *,
        policy: AdaptiveRefinementPolicy | None = None,
        store: RefinementReceiptStore | None = None,
        clock: Callable[[], int | float] | None = None,
    ) -> None:
        if not callable(generator):
            raise AdaptiveGoalRefinementError("generator must be callable")
        if not callable(verifier):
            raise AdaptiveGoalRefinementError(
                "an independent candidate verifier is required"
            )
        self.generator = generator
        self.verifier = verifier
        self.policy = policy or AdaptiveRefinementPolicy()
        self.store = store or InMemoryRefinementStore()
        self.clock = clock or __import__("time").time

    def refine(self, request: AdaptiveRefinementRequest) -> AdaptiveRefinementResult:
        """Process changed evidence and admit zero or one verified refinement."""

        if not isinstance(request, AdaptiveRefinementRequest):
            raise AdaptiveGoalRefinementError(
                "request must be AdaptiveRefinementRequest"
            )
        if len(request.signals) > self.policy.max_signals_per_cycle:
            return self._terminal(
                request,
                RefinementDecision.BUDGET_EXHAUSTED,
                "signal budget exhausted",
                model_called=False,
            )
        # The lock covers the whole root/cycle budget, not only one evidence
        # fingerprint.  Distinct changed counterexamples arriving concurrently
        # in the same cycle must not each consume the single generation slot.
        cycle_lock_id = content_identity(
            {
                "root_goal_content_id": request.root_goal_content_id,
                "cycle_id": request.cycle_id,
                "policy_id": self.policy.content_id,
            }
        )
        lock = _evidence_lock(self.store, cycle_lock_id)
        transaction = getattr(self.store, "transaction", None)
        transaction_context = (
            transaction() if callable(transaction) else nullcontext()
        )
        with lock, transaction_context:
            now = _nonnegative(int(self.clock()), "clock")
            history = self.store.receipts()
            matching = tuple(
                item
                for item in history
                if item.evidence_fingerprint == request.evidence_fingerprint
                and item.policy_id == self.policy.content_id
            )
            admitted = next(
                (
                    item
                    for item in reversed(matching)
                    if item.decision is RefinementDecision.ADMITTED
                ),
                None,
            )
            if admitted is not None:
                return AdaptiveRefinementResult(
                    self._receipt(
                        request,
                        RefinementDecision.DUPLICATE,
                        now,
                        model_called=False,
                        reason=f"evidence already admitted by {admitted.receipt_id}",
                        attempt_index=len(matching) + 1,
                        refinement_index=admitted.refinement_index,
                    )
                )
            latest_failure = next(
                (
                    item
                    for item in reversed(matching)
                    if item.decision
                    in {
                        RefinementDecision.GENERATION_FAILED,
                        RefinementDecision.CANDIDATE_REJECTED,
                        RefinementDecision.VERIFICATION_FAILED,
                        RefinementDecision.COMMIT_FAILED,
                    }
                ),
                None,
            )
            if latest_failure is not None and now < latest_failure.retry_after:
                receipt = self._receipt(
                    request,
                    RefinementDecision.BACKED_OFF,
                    now,
                    model_called=False,
                    reason=(
                        "unchanged evidence is in backoff after "
                        f"{latest_failure.decision.value}"
                    ),
                    retry_after=latest_failure.retry_after,
                    attempt_index=len(matching) + 1,
                    refinement_index=latest_failure.refinement_index,
                )
                self._persist_nonadmission(receipt)
                return AdaptiveRefinementResult(receipt)

            cycle_model_calls = tuple(
                item
                for item in history
                if item.root_goal_content_id == request.root_goal_content_id
                and item.cycle_id == request.cycle_id
                and item.policy_id == self.policy.content_id
                and item.model_called
            )
            if len(cycle_model_calls) >= self.policy.max_model_calls_per_cycle:
                return self._terminal(
                    request,
                    RefinementDecision.BUDGET_EXHAUSTED,
                    "cycle refinement generation budget exhausted",
                    model_called=False,
                    attempt_index=len(matching) + 1,
                    refinement_index=max(
                        (item.refinement_index for item in cycle_model_calls),
                        default=0,
                    ),
                )

            root_admissions = tuple(
                item
                for item in history
                if item.root_goal_content_id == request.root_goal_content_id
                and item.decision is RefinementDecision.ADMITTED
            )
            if (
                request.refinement_depth >= self.policy.max_refinement_depth
                or len(root_admissions) >= self.policy.max_refinements_per_root
            ):
                return self._terminal(
                    request,
                    RefinementDecision.BUDGET_EXHAUSTED,
                    "root refinement budget exhausted",
                    model_called=False,
                    attempt_index=len(matching) + 1,
                    refinement_index=len(root_admissions),
                )

            attempt_index = len(matching) + 1
            refinement_index = len(root_admissions) + 1
            try:
                raw_candidate = self.generator(request)
                candidate = self._candidate(raw_candidate, request)
            except BaseException as exc:
                return self._failure(
                    request,
                    RefinementDecision.GENERATION_FAILED,
                    now,
                    attempt_index,
                    refinement_index,
                    f"candidate generation failed closed: {type(exc).__name__}: {exc}",
                )

            invalid = self._candidate_violation(candidate, request)
            if invalid:
                return self._failure(
                    request,
                    RefinementDecision.CANDIDATE_REJECTED,
                    now,
                    attempt_index,
                    refinement_index,
                    invalid,
                    producer_id=candidate.producer_id,
                    producer_kind=candidate.producer_kind.value,
                    candidate_plan_id=candidate.plan.content_id,
                )

            try:
                verification = self.verifier(candidate, request)
                verified, verification_id, verification_reason = (
                    self._verification(verification, candidate, request)
                )
            except BaseException as exc:
                verified, verification_id = False, ""
                verification_reason = (
                    "independent verification failed closed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if not verified:
                return self._failure(
                    request,
                    RefinementDecision.VERIFICATION_FAILED,
                    now,
                    attempt_index,
                    refinement_index,
                    verification_reason or "child sufficiency was not verified",
                    producer_id=candidate.producer_id,
                    producer_kind=candidate.producer_kind.value,
                    candidate_plan_id=candidate.plan.content_id,
                    verification_receipt_id=verification_id,
                )

            receipt = self._receipt(
                request,
                RefinementDecision.ADMITTED,
                now,
                model_called=True,
                reason="new typed evidence produced one bounded verified refinement",
                producer_id=candidate.producer_id,
                producer_kind=candidate.producer_kind.value,
                candidate_plan_id=candidate.plan.content_id,
                verification_receipt_id=verification_id,
                attempt_index=attempt_index,
                refinement_index=refinement_index,
            )
            try:
                self.store.append(receipt)
            except BaseException as exc:
                # The caller never sees the candidate plan if the transaction
                # receipt did not become durable.
                failed = self._receipt(
                    request,
                    RefinementDecision.COMMIT_FAILED,
                    now,
                    model_called=True,
                    reason=f"receipt commit failed closed: {type(exc).__name__}: {exc}",
                    producer_id=candidate.producer_id,
                    producer_kind=candidate.producer_kind.value,
                    candidate_plan_id=candidate.plan.content_id,
                    verification_receipt_id=verification_id,
                    retry_after=self._retry_after(now, len(matching) + 1),
                    attempt_index=attempt_index,
                    refinement_index=refinement_index,
                )
                return AdaptiveRefinementResult(failed)
            return AdaptiveRefinementResult(receipt, candidate.plan)

    refine_goal = refine

    def _candidate(
        self,
        value: AdaptiveRefinementCandidate | FormalWorkPlan | Mapping[str, Any],
        request: AdaptiveRefinementRequest,
    ) -> AdaptiveRefinementCandidate:
        if isinstance(value, AdaptiveRefinementCandidate):
            return value
        if isinstance(value, FormalWorkPlan):
            before = {
                **{item.goal_id: item.content_id for item in request.plan.goals},
                **{
                    item.subgoal_id: item.content_id
                    for item in request.plan.subgoals
                },
            }
            after = {
                **{item.goal_id: item.content_id for item in value.goals},
                **{item.subgoal_id: item.content_id for item in value.subgoals},
            }
            changed = tuple(
                sorted(
                    goal_id
                    for goal_id in set(before) | set(after)
                    if before.get(goal_id) != after.get(goal_id)
                    and goal_id != request.root_goal_id
                )
            )
            return AdaptiveRefinementCandidate(
                plan=value,
                root_goal_id=request.root_goal_id,
                root_goal_content_id=request.root_goal_content_id,
                assumption_ids=request.assumption_ids,
                changed_goal_ids=changed,
                signal_kind=request.signals[0].kind,
                producer_id="adaptive-goal-refiner",
                producer_kind=RefinementProducerKind.DETERMINISTIC,
            )
        if isinstance(value, Mapping):
            plan = value.get("plan")
            if not isinstance(plan, FormalWorkPlan):
                raise AdaptiveGoalRefinementError(
                    "candidate mapping requires a FormalWorkPlan under plan"
                )
            return AdaptiveRefinementCandidate(
                plan=plan,
                root_goal_id=value.get("root_goal_id", request.root_goal_id),
                root_goal_content_id=value.get(
                    "root_goal_content_id", request.root_goal_content_id
                ),
                assumption_ids=tuple(
                    value.get("assumption_ids", request.assumption_ids)
                ),
                changed_goal_ids=tuple(value.get("changed_goal_ids") or ()),
                signal_kind=value.get("signal_kind", request.signals[0].kind),
                producer_id=value.get("producer_id", "adaptive-goal-refiner"),
                producer_kind=value.get(
                    "producer_kind", RefinementProducerKind.DETERMINISTIC
                ),
                rationale=value.get("rationale", ""),
            )
        raise AdaptiveGoalRefinementError(
            "generator must return AdaptiveRefinementCandidate or FormalWorkPlan"
        )

    def _candidate_violation(
        self,
        candidate: AdaptiveRefinementCandidate,
        request: AdaptiveRefinementRequest,
    ) -> str:
        if candidate.root_goal_id != request.root_goal_id:
            return "candidate changed the frozen root goal identifier"
        if candidate.root_goal_content_id != request.root_goal_content_id:
            return "candidate changed the frozen root goal content identity"
        if candidate.assumption_ids != request.assumption_ids:
            return "candidate changed the frozen assumptions"
        if candidate.plan.repository_tree_id != request.repository_tree_id:
            return "candidate plan does not match the frozen repository tree"
        if candidate.signal_kind not in {item.kind for item in request.signals}:
            return "candidate signal kind is not present in the refinement request"
        roots = [
            item
            for item in candidate.plan.goals
            if item.goal_id == request.root_goal_id
        ]
        if len(roots) != 1 or roots[0].content_id != request.root_goal_content_id:
            return "candidate plan mutated or removed the frozen root"
        if candidate.plan.content_id == request.plan.content_id:
            return "candidate does not make a semantic plan change"
        if request.root_goal_id in candidate.changed_goal_ids:
            return "changed_goal_ids may not include the frozen root"
        if len(candidate.changed_goal_ids) > self.policy.max_changed_goals:
            return "candidate exceeds the changed-goal budget"
        before = {
            **{item.goal_id: item.content_id for item in request.plan.goals},
            **{
                item.subgoal_id: item.content_id
                for item in request.plan.subgoals
            },
        }
        after = {
            **{item.goal_id: item.content_id for item in candidate.plan.goals},
            **{
                item.subgoal_id: item.content_id
                for item in candidate.plan.subgoals
            },
        }
        actual = {
            goal_id
            for goal_id in set(before) | set(after)
            if before.get(goal_id) != after.get(goal_id)
            and goal_id != request.root_goal_id
        }
        declared = set(candidate.changed_goal_ids)
        if not actual.issubset(declared):
            return "candidate omitted changed goals from its bounded change declaration"
        if declared != actual:
            return "candidate declared unchanged or unknown goals as changed"
        return ""

    @staticmethod
    def _verification(
        value: Any,
        candidate: AdaptiveRefinementCandidate,
        request: AdaptiveRefinementRequest,
    ) -> tuple[bool, str, str]:
        if isinstance(value, bool):
            # A bare boolean has no independently auditable receipt.
            return False, "", "verifier returned a boolean instead of a receipt"
        raw_verified = getattr(value, "verified", None)
        if not isinstance(raw_verified, bool):
            return False, "", "verification status must be boolean"
        verified = raw_verified
        verification_id = str(
            getattr(value, "content_id", "")
            or getattr(value, "receipt_id", "")
        ).strip()
        reason = str(getattr(value, "reason", "") or "").strip()
        frozen = getattr(value, "frozen_context", None)
        if isinstance(value, RefinementVerificationResult):
            frozen = value.frozen_context
            verified_plan_id = value.rounds[-1].plan_id
        else:
            verified_plan_id = str(
                getattr(value, "candidate_plan_id", "")
                or getattr(value, "plan_id", "")
            ).strip()
        if frozen is None:
            return False, verification_id, "verification omitted frozen context"
        if (
            getattr(frozen, "root_goal_id", None) != request.root_goal_id
            or getattr(frozen, "root_goal_content_id", None)
            != request.root_goal_content_id
            or tuple(getattr(frozen, "assumption_ids", ()))
            != request.assumption_ids
        ):
            return False, verification_id, "verification changed the frozen context"
        if not verified_plan_id:
            return False, verification_id, "verification omitted candidate plan identity"
        if verified_plan_id != candidate.plan.content_id:
            return False, verification_id, "verification was produced for another plan"
        if verified and not verification_id:
            return False, "", "verification omitted its content identity"
        return verified, verification_id, reason

    def _failure(
        self,
        request: AdaptiveRefinementRequest,
        decision: RefinementDecision,
        now: int,
        attempt_index: int,
        refinement_index: int,
        reason: str,
        **fields: Any,
    ) -> AdaptiveRefinementResult:
        receipt = self._receipt(
            request,
            decision,
            now,
            model_called=True,
            reason=reason,
            retry_after=self._retry_after(now, attempt_index),
            attempt_index=attempt_index,
            refinement_index=refinement_index,
            **fields,
        )
        self._persist_nonadmission(receipt)
        return AdaptiveRefinementResult(receipt)

    def _terminal(
        self,
        request: AdaptiveRefinementRequest,
        decision: RefinementDecision,
        reason: str,
        *,
        model_called: bool,
        attempt_index: int = 1,
        refinement_index: int = 0,
    ) -> AdaptiveRefinementResult:
        now = _nonnegative(int(self.clock()), "clock")
        receipt = self._receipt(
            request,
            decision,
            now,
            model_called=model_called,
            reason=reason,
            attempt_index=attempt_index,
            refinement_index=refinement_index,
        )
        self._persist_nonadmission(receipt)
        return AdaptiveRefinementResult(receipt)

    def _persist_nonadmission(self, receipt: AdaptiveRefinementReceipt) -> None:
        try:
            self.store.append(receipt)
        except BaseException as exc:
            raise RefinementPersistenceError(
                f"could not persist refinement decision: {exc}"
            ) from exc

    def _retry_after(self, now: int, failures: int) -> int:
        exponent = max(0, min(failures - 1, 30))
        delay = min(
            self.policy.max_backoff_seconds,
            self.policy.initial_backoff_seconds * (2**exponent),
        )
        return now + delay

    def _receipt(
        self,
        request: AdaptiveRefinementRequest,
        decision: RefinementDecision,
        now: int,
        *,
        model_called: bool,
        reason: str,
        producer_id: str = "",
        producer_kind: str = "",
        candidate_plan_id: str = "",
        verification_receipt_id: str = "",
        retry_after: int = 0,
        attempt_index: int = 1,
        refinement_index: int = 0,
    ) -> AdaptiveRefinementReceipt:
        requirement_ids: list[str] = []
        counterexample_evidence: NewCounterexampleRefinementEvidence | None = None
        if (
            decision is RefinementDecision.ADMITTED
            and len(request.signals) == 1
            and request.signals[0].kind is RefinementSignalKind.COUNTEREXAMPLE
        ):
            counterexample_evidence = NewCounterexampleRefinementEvidence(
                counterexample_signal_id=request.signals[0].evidence_id,
                request_id=request.content_id,
                evidence_fingerprint=request.evidence_fingerprint,
                root_goal_id=request.root_goal_id,
                root_goal_content_id=request.root_goal_content_id,
                assumption_ids=request.assumption_ids,
                policy_id=self.policy.content_id,
                repository_tree_id=request.repository_tree_id,
                previous_plan_id=request.plan.content_id,
                candidate_plan_id=candidate_plan_id,
                verification_receipt_id=verification_receipt_id,
                producer_id=producer_id,
                producer_kind=producer_kind,
                refinement_index=refinement_index,
            )
            requirement_ids.append(NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID)
        elif (
            decision is RefinementDecision.BACKED_OFF
            and len(request.signals) == 1
            and request.signals[0].kind is RefinementSignalKind.REPEATED_FAILURE
        ):
            requirement_ids.append(UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID)
        return AdaptiveRefinementReceipt(
            decision=decision,
            request_id=request.content_id,
            cycle_id=request.cycle_id,
            evidence_fingerprint=request.evidence_fingerprint,
            root_goal_id=request.root_goal_id,
            root_goal_content_id=request.root_goal_content_id,
            assumption_ids=request.assumption_ids,
            policy_id=self.policy.content_id,
            repository_tree_id=request.repository_tree_id,
            producer_id=producer_id,
            producer_kind=producer_kind,
            previous_plan_id=request.plan.content_id,
            candidate_plan_id=candidate_plan_id,
            verification_receipt_id=verification_receipt_id,
            model_called=model_called,
            attempted_at=now,
            retry_after=retry_after,
            attempt_index=attempt_index,
            refinement_index=refinement_index,
            reason=reason,
            signal_ids=tuple(item.evidence_id for item in request.signals),
            signal_kinds=tuple(item.kind.value for item in request.signals),
            requirement_ids=tuple(requirement_ids),
            new_counterexample_evidence=counterexample_evidence,
        )


def refine_goal_from_evidence(
    request: AdaptiveRefinementRequest,
    generator: CandidateGenerator,
    verifier: CandidateVerifier,
    *,
    policy: AdaptiveRefinementPolicy | None = None,
    store: RefinementReceiptStore | None = None,
    clock: Callable[[], int | float] | None = None,
) -> AdaptiveRefinementResult:
    """Functional entry point for one bounded adaptive-refinement cycle."""

    return AdaptiveGoalRefiner(
        generator,
        verifier,
        policy=policy,
        store=store,
        clock=clock,
    ).refine(request)


__all__ = [
    "ADAPTIVE_GOAL_REFINER_VERSION",
    "NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID",
    "UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID",
    "NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA",
    "AdaptiveGoalRefinementError",
    "RefinementPersistenceError",
    "RefinementSignalKind",
    "GoalDebtKind",
    "RefinementDecision",
    "RefinementProducerKind",
    "RefinementSignal",
    "GoalRefinementSignal",
    "GoalQualityRecord",
    "GoalQuality",
    "AdaptiveRefinementPolicy",
    "GoalRefinementPolicy",
    "RefinementLimits",
    "AdaptiveRefinementRequest",
    "GoalRefinementRequest",
    "AdaptiveRefinementCandidate",
    "GoalRefinementCandidate",
    "GoalRefinementProposal",
    "NewCounterexampleRefinementEvidence",
    "AdaptiveRefinementReceipt",
    "GoalRefinementReceipt",
    "AdaptiveRefinementResult",
    "GoalRefinementResult",
    "RefinementReceiptStore",
    "InMemoryRefinementStore",
    "JsonlRefinementStore",
    "AdaptiveGoalRefiner",
    "refine_goal_from_evidence",
]
