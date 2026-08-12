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
from types import MappingProxyType
from typing import Any, Final, Protocol

from ..planning.formal_planning_contracts import FormalWorkPlan
from ..proof.formal_verification_contracts import (
    ContractValidationError,
    canonical_json,
    content_identity,
)
from .goal_refinement_verification import (
    FrozenRefinementContext,
    RefinementVerificationResult,
)


ADAPTIVE_GOAL_REFINER_VERSION: Final = 3
ADAPTIVE_REFINEMENT_RECEIPT_VERSION: Final = 4
NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID: Final = (
    "003778425160038348524906247302938706902"
)
NEW_EVIDENCE_REFINEMENT_GOAL_ID: Final = "ASI-G098"
UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID: Final = (
    "312819945606360295782005228058369235550"
)
UNCHANGED_FAILURE_BACKOFF_GOAL_ID: Final = "ASI-G115"

SIGNAL_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/refinement-signal@1"
QUALITY_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/goal-quality@1"
GOAL_DEBT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/goal-debt@1"
REQUEST_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/adaptive-refinement-request@1"
CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-refinement-candidate@1"
)
RECEIPT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/adaptive-refinement-receipt@2"
REQUIREMENT_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/new-counterexample-refinement-evidence@1"
)
UNCHANGED_FAILURE_BACKOFF_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/unchanged-failure-backoff-evidence@1"
)
REFINEMENT_VALUE_ESTIMATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/refinement-value-estimate@1"
)
REFINEMENT_DELTA_QUALITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/refinement-delta-quality@1"
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

UNCHANGED_FAILURE_BACKOFF_ACCEPTANCE_CRITERIA: Final = (
    (
        "A persisted failed refinement attempt starts a finite policy-bounded "
        "retry window"
    ),
    (
        "A semantically unchanged typed repeated failure observed before the "
        "deadline suppresses candidate generation and independent verification"
    ),
    (
        "Observation time and occurrence-count changes do not disguise the "
        "same failure, while changed failure evidence or plan state bypasses "
        "the old backoff"
    ),
    (
        "The backoff witness binds the exact requirement, repeated-failure "
        "signal and signature, request and evidence fingerprint, frozen root, "
        "assumptions, repository tree, policy, and previous plan"
    ),
    (
        "The witness causally binds the source failure receipt, decision, "
        "model call, attempts, timestamps, retry deadline, and suppressed "
        "no-model-call decision"
    ),
    (
        "Only a source-bound in-window backoff is authoritative, and restored "
        "receipts reject unsupported schemas, versions, identities, fields, "
        "or detached and tampered witnesses"
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


def _millionths(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if result > 1_000_000:
        raise AdaptiveGoalRefinementError(f"{name} must not exceed 1000000")
    return result


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
    UNCOVERED_CRITERION = "uncovered_criterion"
    UNCERTAINTY_CHANGE = "uncertainty_change"
    OPERATOR_REVISION = "operator_revision"

    # Compatibility spellings for callers that use the task language.
    STALE_RECEIPT = "stale_evidence"
    REPEATED_VALIDATION_SIGNATURE = "repeated_failure"
    UNAVAILABLE_CAPABILITY = "capability_change"
    UNAVAILABLE_PROVIDER = "capability_change"
    CHANGED_INTERFACE = "interface_change"
    CONFLICT = "scope_conflict"
    INFEASIBLE_RESOURCES = "resource_infeasible"
    UNCOVERED_ACCEPTANCE = "uncovered_criterion"
    UNCERTAINTY = "uncertainty_change"
    CAPABILITY = "capability_change"
    INTERFACE = "interface_change"
    RESOURCE = "resource_change"


_SIGNAL_KIND_ALIASES: Final[Mapping[str, RefinementSignalKind]] = {
    "stale_receipt": RefinementSignalKind.STALE_EVIDENCE,
    "repeated_validation_signature": RefinementSignalKind.REPEATED_FAILURE,
    "unavailable_capability": RefinementSignalKind.CAPABILITY_CHANGE,
    "unavailable_provider": RefinementSignalKind.CAPABILITY_CHANGE,
    "changed_interface": RefinementSignalKind.INTERFACE_CHANGE,
    "conflict": RefinementSignalKind.SCOPE_CONFLICT,
    "infeasible_resources": RefinementSignalKind.RESOURCE_INFEASIBLE,
    "uncovered_acceptance": RefinementSignalKind.UNCOVERED_CRITERION,
    "uncovered_acceptance_criterion": RefinementSignalKind.UNCOVERED_CRITERION,
    "uncertainty": RefinementSignalKind.UNCERTAINTY_CHANGE,
    "operator_change": RefinementSignalKind.OPERATOR_REVISION,
    "operator-revision": RefinementSignalKind.OPERATOR_REVISION,
    "uncovered criterion": RefinementSignalKind.UNCOVERED_CRITERION,
    "capability": RefinementSignalKind.CAPABILITY_CHANGE,
    "interface": RefinementSignalKind.INTERFACE_CHANGE,
    "resource": RefinementSignalKind.RESOURCE_CHANGE,
}


def _signal_kind(value: Any) -> RefinementSignalKind:
    if isinstance(value, RefinementSignalKind):
        return value
    normalized = str(getattr(value, "value", value)).strip().lower()
    if normalized in _SIGNAL_KIND_ALIASES:
        return _SIGNAL_KIND_ALIASES[normalized]
    return _enum(value, RefinementSignalKind, "kind")


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
    MISSING_REFINEMENT_BUDGET = "missing_refinement_budget"
    AMBIGUOUS = "ambiguous"
    STALE_EVIDENCE = "stale_evidence"
    UNCOVERED_ACCEPTANCE = "uncovered_acceptance"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    EXCESSIVE_BREADTH = "excessive_breadth"


_GOAL_DEBT_DIMENSIONS: Final[Mapping[GoalDebtKind, str]] = {
    GoalDebtKind.MISSING_OUTCOME: "outcome",
    GoalDebtKind.MISSING_SCOPE: "scope",
    GoalDebtKind.MISSING_ASSUMPTIONS: "assumptions",
    GoalDebtKind.MISSING_NON_GOALS: "non_goals",
    GoalDebtKind.MISSING_ACCEPTANCE: "acceptance",
    GoalDebtKind.MISSING_EVIDENCE_PRODUCER: "evidence_producers",
    GoalDebtKind.MISSING_VALIDATION: "validation",
    GoalDebtKind.MISSING_FRESHNESS: "freshness",
    GoalDebtKind.MISSING_RESOURCE_ENVELOPE: "resource_envelope",
    GoalDebtKind.MISSING_REFINEMENT_BUDGET: "refinement_budget",
    GoalDebtKind.AMBIGUOUS: "ambiguity",
    GoalDebtKind.STALE_EVIDENCE: "freshness",
    GoalDebtKind.UNCOVERED_ACCEPTANCE: "acceptance",
    GoalDebtKind.UNSUPPORTED_SEMANTICS: "unsupported_semantics",
    GoalDebtKind.EXCESSIVE_BREADTH: "breadth",
}

_GOAL_DEBT_MESSAGES: Final[Mapping[GoalDebtKind, str]] = {
    GoalDebtKind.MISSING_OUTCOME: "Goal has no explicit outcome.",
    GoalDebtKind.MISSING_SCOPE: "Goal has no bounded scope.",
    GoalDebtKind.MISSING_ASSUMPTIONS: "Goal has no explicit assumption set.",
    GoalDebtKind.MISSING_NON_GOALS: "Goal has no explicit non-goals.",
    GoalDebtKind.MISSING_ACCEPTANCE: "Goal has no acceptance criteria.",
    GoalDebtKind.MISSING_EVIDENCE_PRODUCER: (
        "Goal has no bound evidence producer."
    ),
    GoalDebtKind.MISSING_VALIDATION: "Goal has no validation policy.",
    GoalDebtKind.MISSING_FRESHNESS: "Goal has no evidence freshness horizon.",
    GoalDebtKind.MISSING_RESOURCE_ENVELOPE: (
        "Goal has no finite resource envelope."
    ),
    GoalDebtKind.MISSING_REFINEMENT_BUDGET: (
        "Goal has no finite refinement budget."
    ),
    GoalDebtKind.AMBIGUOUS: "Goal contains unresolved ambiguity.",
    GoalDebtKind.STALE_EVIDENCE: "Goal depends on stale evidence.",
    GoalDebtKind.UNCOVERED_ACCEPTANCE: (
        "Goal has acceptance criteria without evidence coverage."
    ),
    GoalDebtKind.UNSUPPORTED_SEMANTICS: (
        "Goal relies on unsupported semantics."
    ),
    GoalDebtKind.EXCESSIVE_BREADTH: "Goal exceeds its reviewed breadth bound.",
}


class RefinementDecision(str, Enum):
    ADMITTED = "admitted"
    DUPLICATE = "duplicate"
    BACKED_OFF = "backed_off"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INSUFFICIENT_INFORMATION_GAIN = "insufficient_information_gain"
    GENERATION_FAILED = "generation_failed"
    CANDIDATE_REJECTED = "candidate_rejected"
    VERIFICATION_FAILED = "verification_failed"
    COMMIT_FAILED = "commit_failed"


REFINEMENT_FAILURE_DECISIONS: Final[frozenset[RefinementDecision]] = frozenset(
    {
        RefinementDecision.GENERATION_FAILED,
        RefinementDecision.CANDIDATE_REJECTED,
        RefinementDecision.VERIFICATION_FAILED,
        RefinementDecision.COMMIT_FAILED,
    }
)


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
        object.__setattr__(self, "kind", _signal_kind(self.kind))
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
class GoalDebtRecord:
    """One content-addressed, reviewed goal-quality finding.

    Debt is diagnostic rather than completion authority.  Each record is
    nevertheless bound to the exact quality snapshot that produced it so a
    later objective revision cannot silently reuse an old finding.
    """

    goal_id: str
    quality_id: str
    kind: GoalDebtKind
    related_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "quality_id", _text(self.quality_id, "quality_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, GoalDebtKind, "kind"))
        object.__setattr__(
            self,
            "related_ids",
            _strings(self.related_ids, "related_ids"),
        )

    @property
    def dimension(self) -> str:
        return _GOAL_DEBT_DIMENSIONS[self.kind]

    @property
    def message(self) -> str:
        return _GOAL_DEBT_MESSAGES[self.kind]

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": GOAL_DEBT_SCHEMA,
            "version": ADAPTIVE_GOAL_REFINER_VERSION,
            "goal_id": self.goal_id,
            "quality_id": self.quality_id,
            "kind": self.kind.value,
            "dimension": self.dimension,
            "message": self.message,
            "related_ids": self.related_ids,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDebtRecord":
        _restored_record(
            payload,
            noun="goal-debt record",
            schema=GOAL_DEBT_SCHEMA,
            version=ADAPTIVE_GOAL_REFINER_VERSION,
            allowed_fields=frozenset(
                {
                    "schema",
                    "version",
                    "content_id",
                    "goal_id",
                    "quality_id",
                    "kind",
                    "dimension",
                    "message",
                    "related_ids",
                }
            ),
            identity_field="content_id",
        )
        result = cls(
            goal_id=payload.get("goal_id", ""),
            quality_id=payload.get("quality_id", ""),
            kind=payload.get("kind", ""),
            related_ids=payload.get("related_ids") or (),
        )
        if payload.get("dimension") != result.dimension:
            raise AdaptiveGoalRefinementError(
                "goal-debt dimension does not match its reviewed kind"
            )
        if payload.get("message") != result.message:
            raise AdaptiveGoalRefinementError(
                "goal-debt message does not match its reviewed kind"
            )
        _claimed(payload, result.content_id, "goal-debt record")
        return result


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
    refinement_budget: Mapping[str, Any] = field(default_factory=dict)
    ambiguities: tuple[str, ...] = ()
    stale_evidence_ids: tuple[str, ...] = ()
    uncovered_acceptance_criteria: tuple[str, ...] = ()

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
            "ambiguities",
            "stale_evidence_ids",
            "uncovered_acceptance_criteria",
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
        object.__setattr__(
            self,
            "refinement_budget",
            _mapping(self.refinement_budget, "refinement_budget"),
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
            (
                not self.refinement_budget,
                GoalDebtKind.MISSING_REFINEMENT_BUDGET,
            ),
            (
                bool(self.ambiguities) or not self.outcome,
                GoalDebtKind.AMBIGUOUS,
            ),
            (
                bool(self.stale_evidence_ids)
                or self.freshness_horizon_seconds == 0,
                GoalDebtKind.STALE_EVIDENCE,
            ),
            (
                bool(self.uncovered_acceptance_criteria)
                or not self.acceptance_criteria,
                GoalDebtKind.UNCOVERED_ACCEPTANCE,
            ),
            (bool(self.unsupported_semantics), GoalDebtKind.UNSUPPORTED_SEMANTICS),
            (self.breadth > self.max_breadth, GoalDebtKind.EXCESSIVE_BREADTH),
        )
        for present, kind in checks:
            if present:
                findings.append(kind)
        return tuple(findings)

    @property
    def debt_records(self) -> tuple[GoalDebtRecord, ...]:
        """Return stable typed findings bound to this exact quality snapshot."""

        related: Mapping[GoalDebtKind, tuple[str, ...]] = {
            GoalDebtKind.AMBIGUOUS: self.ambiguities,
            GoalDebtKind.STALE_EVIDENCE: self.stale_evidence_ids,
            GoalDebtKind.UNCOVERED_ACCEPTANCE: (
                self.uncovered_acceptance_criteria
            ),
            GoalDebtKind.UNSUPPORTED_SEMANTICS: self.unsupported_semantics,
        }
        return tuple(
            GoalDebtRecord(
                goal_id=self.goal_id,
                quality_id=self.content_id,
                kind=kind,
                related_ids=related.get(kind, ()),
            )
            for kind in self.debt
        )

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
            "refinement_budget": self.refinement_budget,
            "ambiguities": self.ambiguities,
            "stale_evidence_ids": self.stale_evidence_ids,
            "uncovered_acceptance_criteria": (
                self.uncovered_acceptance_criteria
            ),
            "unsupported_semantics": self.unsupported_semantics,
            "breadth": self.breadth,
            "max_breadth": self.max_breadth,
            "debt": tuple(item.value for item in self.debt),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "content_id": self.content_id,
            "debt_records": tuple(item.to_dict() for item in self.debt_records),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalQualityRecord":
        _restored_record(
            payload,
            noun="goal-quality record",
            schema=QUALITY_SCHEMA,
            version=ADAPTIVE_GOAL_REFINER_VERSION,
            allowed_fields=frozenset(
                {
                    "schema",
                    "version",
                    "content_id",
                    "goal_id",
                    "outcome",
                    "scope_ids",
                    "assumption_ids",
                    "non_goals",
                    "acceptance_criteria",
                    "evidence_producer_ids",
                    "validation_ids",
                    "freshness_horizon_seconds",
                    "resource_envelope",
                    "refinement_budget",
                    "ambiguities",
                    "stale_evidence_ids",
                    "uncovered_acceptance_criteria",
                    "unsupported_semantics",
                    "breadth",
                    "max_breadth",
                    "debt",
                    "debt_records",
                }
            ),
            identity_field="content_id",
        )
        result = cls(
            goal_id=payload.get("goal_id", ""),
            outcome=payload.get("outcome", ""),
            scope_ids=payload.get("scope_ids") or (),
            assumption_ids=payload.get("assumption_ids") or (),
            non_goals=payload.get("non_goals") or (),
            acceptance_criteria=payload.get("acceptance_criteria") or (),
            evidence_producer_ids=payload.get("evidence_producer_ids") or (),
            validation_ids=payload.get("validation_ids") or (),
            freshness_horizon_seconds=payload.get(
                "freshness_horizon_seconds", 0
            ),
            resource_envelope=payload.get("resource_envelope") or {},
            refinement_budget=payload.get("refinement_budget") or {},
            ambiguities=payload.get("ambiguities") or (),
            stale_evidence_ids=payload.get("stale_evidence_ids") or (),
            uncovered_acceptance_criteria=payload.get(
                "uncovered_acceptance_criteria"
            )
            or (),
            unsupported_semantics=payload.get("unsupported_semantics") or (),
            breadth=payload.get("breadth", 1),
            max_breadth=payload.get("max_breadth", 8),
        )
        expected_debt = tuple(item.value for item in result.debt)
        if tuple(payload.get("debt") or ()) != expected_debt:
            raise AdaptiveGoalRefinementError(
                "goal-quality debt projection does not match its fields"
            )
        records = payload.get("debt_records")
        if not isinstance(records, Sequence) or isinstance(
            records, (str, bytes, bytearray, memoryview)
        ):
            raise AdaptiveGoalRefinementError(
                "goal-quality debt_records must be a sequence"
            )
        restored_record_values: list[GoalDebtRecord] = []
        for item in records:
            if not isinstance(item, Mapping):
                raise AdaptiveGoalRefinementError(
                    "goal-quality debt_records must contain objects"
                )
            restored_record_values.append(GoalDebtRecord.from_dict(item))
        restored_records = tuple(restored_record_values)
        if restored_records != result.debt_records:
            raise AdaptiveGoalRefinementError(
                "goal-quality debt records do not match its fields"
            )
        _claimed(payload, result.content_id, "goal-quality record")
        return result


GoalQuality = GoalQualityRecord


@dataclass(frozen=True)
class RefinementValueEstimate:
    """Deterministic pre-generation value and blast-radius estimate."""

    information_gain_millionths: int
    expected_downstream_cost_millionths: int
    affected_subject_ids: tuple[str, ...]
    signal_ids: tuple[str, ...]
    rationale_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "information_gain_millionths",
            _millionths(
                self.information_gain_millionths,
                "information_gain_millionths",
            ),
        )
        object.__setattr__(
            self,
            "expected_downstream_cost_millionths",
            _millionths(
                self.expected_downstream_cost_millionths,
                "expected_downstream_cost_millionths",
            ),
        )
        object.__setattr__(
            self,
            "affected_subject_ids",
            _strings(
                self.affected_subject_ids,
                "affected_subject_ids",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "signal_ids",
            _strings(self.signal_ids, "signal_ids", required=True),
        )
        object.__setattr__(
            self,
            "rationale_codes",
            _strings(self.rationale_codes, "rationale_codes"),
        )

    @property
    def net_value_millionths(self) -> int:
        return (
            self.information_gain_millionths
            - self.expected_downstream_cost_millionths
        )

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": REFINEMENT_VALUE_ESTIMATE_SCHEMA,
            "information_gain_millionths": self.information_gain_millionths,
            "expected_downstream_cost_millionths": (
                self.expected_downstream_cost_millionths
            ),
            "affected_subject_ids": self.affected_subject_ids,
            "signal_ids": self.signal_ids,
            "rationale_codes": self.rationale_codes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinementValueEstimate":
        fields = {
            "schema",
            "information_gain_millionths",
            "expected_downstream_cost_millionths",
            "affected_subject_ids",
            "signal_ids",
            "rationale_codes",
            "content_id",
        }
        _restored_record(
            payload,
            noun="refinement value estimate",
            schema=REFINEMENT_VALUE_ESTIMATE_SCHEMA,
            allowed_fields=frozenset(fields),
            identity_field="content_id",
        )
        result = cls(
            information_gain_millionths=payload.get(
                "information_gain_millionths", 0
            ),
            expected_downstream_cost_millionths=payload.get(
                "expected_downstream_cost_millionths", 0
            ),
            affected_subject_ids=tuple(
                payload.get("affected_subject_ids") or ()
            ),
            signal_ids=tuple(payload.get("signal_ids") or ()),
            rationale_codes=tuple(payload.get("rationale_codes") or ()),
        )
        _claimed(payload, result.content_id, "refinement value estimate")
        return result


@dataclass(frozen=True)
class RefinementDeltaQualityReport:
    """Content-bound lint result for the exact generated plan delta."""

    previous_plan_id: str
    candidate_plan_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    changed_goal_ids: tuple[str, ...]
    accepted: bool
    debt_codes: tuple[str, ...] = ()
    linter_id: str = "adaptive-goal-refiner/delta-linter@1"

    def __post_init__(self) -> None:
        for name in (
            "previous_plan_id",
            "candidate_plan_id",
            "root_goal_content_id",
            "linter_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )
        object.__setattr__(
            self,
            "changed_goal_ids",
            _strings(self.changed_goal_ids, "changed_goal_ids", required=True),
        )
        if not isinstance(self.accepted, bool):
            raise AdaptiveGoalRefinementError("accepted must be boolean")
        object.__setattr__(
            self, "debt_codes", _strings(self.debt_codes, "debt_codes")
        )
        if self.accepted and self.debt_codes:
            raise AdaptiveGoalRefinementError(
                "accepted delta quality reports must have no debt"
            )
        if not self.accepted and not self.debt_codes:
            raise AdaptiveGoalRefinementError(
                "rejected delta quality reports must identify typed debt"
            )

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": REFINEMENT_DELTA_QUALITY_SCHEMA,
            "previous_plan_id": self.previous_plan_id,
            "candidate_plan_id": self.candidate_plan_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "changed_goal_ids": self.changed_goal_ids,
            "accepted": self.accepted,
            "debt_codes": self.debt_codes,
            "linter_id": self.linter_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RefinementDeltaQualityReport":
        fields = {
            "schema",
            "previous_plan_id",
            "candidate_plan_id",
            "root_goal_content_id",
            "assumption_ids",
            "changed_goal_ids",
            "accepted",
            "debt_codes",
            "linter_id",
            "content_id",
        }
        _restored_record(
            payload,
            noun="refinement delta quality report",
            schema=REFINEMENT_DELTA_QUALITY_SCHEMA,
            allowed_fields=frozenset(fields),
            identity_field="content_id",
        )
        result = cls(
            previous_plan_id=payload.get("previous_plan_id", ""),
            candidate_plan_id=payload.get("candidate_plan_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            changed_goal_ids=tuple(payload.get("changed_goal_ids") or ()),
            accepted=payload.get("accepted", False),
            debt_codes=tuple(payload.get("debt_codes") or ()),
            linter_id=payload.get("linter_id", ""),
        )
        _claimed(payload, result.content_id, "refinement delta quality report")
        return result


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
    min_information_gain_millionths: int = 100_000
    max_expected_downstream_cost_millionths: int = 900_000

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
        for name in (
            "min_information_gain_millionths",
            "max_expected_downstream_cost_millionths",
        ):
            _millionths(getattr(self, name), name)
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
        if self.quality is not None:
            if self.quality.goal_id != self.root_goal_id:
                raise AdaptiveGoalRefinementError(
                    "quality record must describe the frozen root goal"
                )
            if (
                self.quality.assumption_ids
                and self.quality.assumption_ids != self.assumption_ids
            ):
                raise AdaptiveGoalRefinementError(
                    "quality assumptions do not match the frozen assumptions"
                )
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
                "quality_id": self.quality.content_id if self.quality else "",
                "goal_debt_ids": tuple(
                    item.content_id for item in self.quality.debt_records
                )
                if self.quality
                else (),
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
            "goal_debt_ids": tuple(
                item.content_id for item in self.quality.debt_records
            )
            if self.quality
            else (),
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
            _signal_kind(self.signal_kind),
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
class UnchangedFailureBackoffEvidence:
    """Concrete causal witness for the ASI-G115 no-second-call criterion.

    A backoff decision is authoritative only when it references the exact
    persisted failed attempt that opened the retry window.  Delivery metadata
    may change, but the repeated-failure signal's semantic identity, frozen
    planning context, policy, repository tree, and previous plan must not.
    """

    repeated_failure_signal_id: str
    failure_signature: str
    request_id: str
    cycle_id: str
    evidence_fingerprint: str
    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    policy_id: str
    repository_tree_id: str
    previous_plan_id: str
    source_failure_receipt_id: str
    source_failure_decision: str
    source_failure_model_called: bool
    source_failure_attempted_at: int
    source_failure_retry_after: int
    source_failure_attempt_index: int
    source_failure_refinement_index: int
    suppressed_at: int
    suppressed_attempt_index: int
    retry_after: int
    model_call_suppressed: bool = True
    requirement_id: str = UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID
    evidence_producer_kind: str = "adaptive_goal_refinement"

    def __post_init__(self) -> None:
        for name in (
            "repeated_failure_signal_id",
            "failure_signature",
            "request_id",
            "cycle_id",
            "evidence_fingerprint",
            "root_goal_id",
            "root_goal_content_id",
            "policy_id",
            "repository_tree_id",
            "previous_plan_id",
            "source_failure_receipt_id",
            "source_failure_decision",
            "requirement_id",
            "evidence_producer_kind",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "assumption_ids", _strings(self.assumption_ids, "assumption_ids")
        )
        for name in (
            "source_failure_attempted_at",
            "source_failure_retry_after",
            "source_failure_attempt_index",
            "source_failure_refinement_index",
            "suppressed_at",
            "suppressed_attempt_index",
            "retry_after",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        try:
            source_decision = RefinementDecision(self.source_failure_decision)
        except ValueError as exc:
            raise AdaptiveGoalRefinementError(
                "unsupported backoff source failure decision"
            ) from exc
        if source_decision not in REFINEMENT_FAILURE_DECISIONS:
            raise AdaptiveGoalRefinementError(
                "backoff source must be a failed refinement decision"
            )
        if self.source_failure_model_called is not True:
            raise AdaptiveGoalRefinementError(
                "backoff source must record the preceding model call"
            )
        if self.model_call_suppressed is not True:
            raise AdaptiveGoalRefinementError(
                "backoff evidence must record a suppressed model call"
            )
        if not (
            self.source_failure_attempted_at
            <= self.suppressed_at
            < self.source_failure_retry_after
        ):
            raise AdaptiveGoalRefinementError(
                "backoff suppression must occur inside the source retry window"
            )
        if self.retry_after != self.source_failure_retry_after:
            raise AdaptiveGoalRefinementError(
                "backoff deadline must match the source failure deadline"
            )
        if self.suppressed_attempt_index <= self.source_failure_attempt_index:
            raise AdaptiveGoalRefinementError(
                "backoff attempt must follow the source failure attempt"
            )
        if self.requirement_id != UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID:
            raise AdaptiveGoalRefinementError(
                "unsupported unchanged-failure backoff requirement id"
            )
        if self.evidence_producer_kind != "adaptive_goal_refinement":
            raise AdaptiveGoalRefinementError(
                "unsupported unchanged-failure backoff evidence producer"
            )

    @property
    def evidence_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def content_id(self) -> str:
        return self.evidence_id

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": UNCHANGED_FAILURE_BACKOFF_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "evidence_producer_kind": self.evidence_producer_kind,
            "repeated_failure_signal_id": self.repeated_failure_signal_id,
            "failure_signature": self.failure_signature,
            "request_id": self.request_id,
            "cycle_id": self.cycle_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "previous_plan_id": self.previous_plan_id,
            "source_failure_receipt_id": self.source_failure_receipt_id,
            "source_failure_decision": self.source_failure_decision,
            "source_failure_model_called": self.source_failure_model_called,
            "source_failure_attempted_at": self.source_failure_attempted_at,
            "source_failure_retry_after": self.source_failure_retry_after,
            "source_failure_attempt_index": self.source_failure_attempt_index,
            "source_failure_refinement_index": (
                self.source_failure_refinement_index
            ),
            "suppressed_at": self.suppressed_at,
            "suppressed_attempt_index": self.suppressed_attempt_index,
            "retry_after": self.retry_after,
            "model_call_suppressed": self.model_call_suppressed,
        }
        if include_identity:
            payload["evidence_id"] = self.evidence_id
        return payload

    def validate_source(self, source: "AdaptiveRefinementReceipt") -> None:
        """Reconcile this witness with the persisted attempt it references."""

        if not isinstance(source, AdaptiveRefinementReceipt):
            raise AdaptiveGoalRefinementError(
                "backoff source must be an adaptive refinement receipt"
            )
        expected = {
            "source_failure_receipt_id": source.receipt_id,
            "source_failure_decision": source.decision.value,
            "source_failure_model_called": source.model_called,
            "source_failure_attempted_at": source.attempted_at,
            "source_failure_retry_after": source.retry_after,
            "source_failure_attempt_index": source.attempt_index,
            "source_failure_refinement_index": source.refinement_index,
            "evidence_fingerprint": source.evidence_fingerprint,
            "root_goal_id": source.root_goal_id,
            "root_goal_content_id": source.root_goal_content_id,
            "assumption_ids": source.assumption_ids,
            "policy_id": source.policy_id,
            "repository_tree_id": source.repository_tree_id,
            "previous_plan_id": source.previous_plan_id,
        }
        mismatched = [
            name for name, value in expected.items() if getattr(self, name) != value
        ]
        if mismatched:
            raise AdaptiveGoalRefinementError(
                "backoff evidence does not match its source failure: "
                + ", ".join(mismatched)
            )
        if (
            source.decision not in REFINEMENT_FAILURE_DECISIONS
            or source.requirement_ids
            or source.signal_ids != (self.repeated_failure_signal_id,)
            or source.signal_kinds
            != (RefinementSignalKind.REPEATED_FAILURE.value,)
        ):
            raise AdaptiveGoalRefinementError(
                "backoff source is not the exact non-authoritative repeated "
                "failure attempt"
            )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "UnchangedFailureBackoffEvidence":
        fields = {
            "schema",
            "requirement_id",
            "evidence_producer_kind",
            "repeated_failure_signal_id",
            "failure_signature",
            "request_id",
            "cycle_id",
            "evidence_fingerprint",
            "root_goal_id",
            "root_goal_content_id",
            "assumption_ids",
            "policy_id",
            "repository_tree_id",
            "previous_plan_id",
            "source_failure_receipt_id",
            "source_failure_decision",
            "source_failure_model_called",
            "source_failure_attempted_at",
            "source_failure_retry_after",
            "source_failure_attempt_index",
            "source_failure_refinement_index",
            "suppressed_at",
            "suppressed_attempt_index",
            "retry_after",
            "model_call_suppressed",
            "evidence_id",
        }
        _restored_record(
            payload,
            noun="unchanged-failure backoff evidence",
            schema=UNCHANGED_FAILURE_BACKOFF_EVIDENCE_SCHEMA,
            allowed_fields=frozenset(fields),
            identity_field="evidence_id",
        )
        values = {
            name: payload.get(name)
            for name in fields
            if name
            not in {
                "schema",
                "evidence_id",
            }
        }
        values["assumption_ids"] = tuple(payload.get("assumption_ids") or ())
        result = cls(**values)
        if payload["evidence_id"] != result.evidence_id:
            raise AdaptiveGoalRefinementError(
                "unchanged-failure backoff evidence identity does not match"
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
    value_estimate: RefinementValueEstimate
    quality_lint_report: RefinementDeltaQualityReport | None = None
    new_counterexample_evidence: NewCounterexampleRefinementEvidence | None = None
    unchanged_failure_backoff_evidence: (
        UnchangedFailureBackoffEvidence | None
    ) = None

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
        if not isinstance(self.value_estimate, RefinementValueEstimate):
            raise AdaptiveGoalRefinementError(
                "receipt requires a refinement value estimate"
            )
        if self.value_estimate.signal_ids != self.signal_ids:
            raise AdaptiveGoalRefinementError(
                "value estimate does not bind the receipt signals"
            )
        quality_report = self.quality_lint_report
        if quality_report is not None:
            if not isinstance(quality_report, RefinementDeltaQualityReport):
                raise AdaptiveGoalRefinementError(
                    "invalid refinement delta quality report"
                )
            expected_quality = {
                "previous_plan_id": self.previous_plan_id,
                "candidate_plan_id": self.candidate_plan_id,
                "root_goal_content_id": self.root_goal_content_id,
                "assumption_ids": self.assumption_ids,
            }
            mismatched_quality = [
                name
                for name, value in expected_quality.items()
                if getattr(quality_report, name) != value
            ]
            if mismatched_quality:
                raise AdaptiveGoalRefinementError(
                    "delta quality report does not match receipt bindings: "
                    + ", ".join(mismatched_quality)
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
        backoff_evidence = self.unchanged_failure_backoff_evidence
        if backoff_evidence is not None:
            if not isinstance(
                backoff_evidence, UnchangedFailureBackoffEvidence
            ):
                raise AdaptiveGoalRefinementError(
                    "invalid unchanged-failure backoff evidence"
                )
            expected = {
                "request_id": self.request_id,
                "cycle_id": self.cycle_id,
                "evidence_fingerprint": self.evidence_fingerprint,
                "root_goal_id": self.root_goal_id,
                "root_goal_content_id": self.root_goal_content_id,
                "assumption_ids": self.assumption_ids,
                "policy_id": self.policy_id,
                "repository_tree_id": self.repository_tree_id,
                "previous_plan_id": self.previous_plan_id,
                "suppressed_at": self.attempted_at,
                "suppressed_attempt_index": self.attempt_index,
                "retry_after": self.retry_after,
                "model_call_suppressed": not self.model_called,
            }
            mismatched = [
                name
                for name, value in expected.items()
                if getattr(backoff_evidence, name) != value
            ]
            if mismatched:
                raise AdaptiveGoalRefinementError(
                    "backoff evidence does not match receipt bindings: "
                    + ", ".join(mismatched)
                )
            if (
                len(self.signal_ids) != 1
                or self.signal_kinds
                != (RefinementSignalKind.REPEATED_FAILURE.value,)
                or backoff_evidence.repeated_failure_signal_id
                != self.signal_ids[0]
            ):
                raise AdaptiveGoalRefinementError(
                    "backoff evidence requires exactly one bound repeated-failure "
                    "signal"
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
                and quality_report is not None
                and quality_report.accepted
            ):
                raise AdaptiveGoalRefinementError(
                    "admitted receipt requires generation, quality lint, "
                    "verification, and evidence binding"
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
                and backoff_evidence is None
            ):
                raise AdaptiveGoalRefinementError(
                    "repeated-failure backoff receipt is missing its causal witness"
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
        if backoff_evidence is not None:
            if self.decision is not RefinementDecision.BACKED_OFF:
                raise AdaptiveGoalRefinementError(
                    "only a backoff receipt may carry unchanged-failure evidence"
                )
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

        result: list[str] = []
        if self.new_counterexample_evidence is not None:
            result.append(self.new_counterexample_evidence.requirement_id)
        if self.unchanged_failure_backoff_evidence is not None:
            result.append(self.unchanged_failure_backoff_evidence.requirement_id)
        return tuple(sorted(result))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Content identities of concrete objective evidence witnesses."""

        result: list[str] = []
        if self.new_counterexample_evidence is not None:
            result.append(self.new_counterexample_evidence.evidence_id)
        if self.unchanged_failure_backoff_evidence is not None:
            result.append(self.unchanged_failure_backoff_evidence.evidence_id)
        return tuple(sorted(result))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "version": ADAPTIVE_REFINEMENT_RECEIPT_VERSION,
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
            "value_estimate": self.value_estimate.to_dict(),
            "quality_lint_report": (
                self.quality_lint_report.to_dict()
                if self.quality_lint_report is not None
                else None
            ),
            "new_counterexample_evidence": (
                self.new_counterexample_evidence.to_dict()
                if self.new_counterexample_evidence is not None
                else None
            ),
            "unchanged_failure_backoff_evidence": (
                self.unchanged_failure_backoff_evidence.to_dict()
                if self.unchanged_failure_backoff_evidence is not None
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
            version=ADAPTIVE_REFINEMENT_RECEIPT_VERSION,
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
                    "value_estimate",
                    "quality_lint_report",
                    "new_counterexample_evidence",
                    "unchanged_failure_backoff_evidence",
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
            value_estimate=RefinementValueEstimate.from_dict(
                payload.get("value_estimate") or {}
            ),
            quality_lint_report=(
                RefinementDeltaQualityReport.from_dict(
                    payload["quality_lint_report"]
                )
                if payload.get("quality_lint_report") is not None
                else None
            ),
            new_counterexample_evidence=(
                NewCounterexampleRefinementEvidence.from_dict(
                    payload["new_counterexample_evidence"]
                )
                if payload.get("new_counterexample_evidence") is not None
                else None
            ),
            unchanged_failure_backoff_evidence=(
                UnchangedFailureBackoffEvidence.from_dict(
                    payload["unchanged_failure_backoff_evidence"]
                )
                if payload.get("unchanged_failure_backoff_evidence") is not None
                else None
            ),
        )
        if payload["receipt_id"] != result.receipt_id:
            raise AdaptiveGoalRefinementError(
                "refinement receipt content identity does not match"
            )
        return result


GoalRefinementReceipt = AdaptiveRefinementReceipt


def _validate_receipt_history(
    receipts: Iterable[AdaptiveRefinementReceipt],
) -> tuple[AdaptiveRefinementReceipt, ...]:
    """Require every causal backoff witness to reference an earlier journal row."""

    result: list[AdaptiveRefinementReceipt] = []
    by_id: dict[str, AdaptiveRefinementReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, AdaptiveRefinementReceipt):
            raise AdaptiveGoalRefinementError(
                "refinement history contains a non-receipt value"
            )
        witness = receipt.unchanged_failure_backoff_evidence
        if witness is not None:
            source = by_id.get(witness.source_failure_receipt_id)
            if source is None:
                raise AdaptiveGoalRefinementError(
                    "backoff evidence source failure is absent or not earlier "
                    "in the receipt journal"
                )
            witness.validate_source(source)
        result.append(receipt)
        by_id[receipt.receipt_id] = receipt
    return tuple(result)


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

    @property
    def planning_completion_witness(self) -> Mapping[str, Any]:
        """Project a qualifying ASI-G030 producer without granting authority.

        The parent planning cohort persists the complete receipt and validates
        it again.  This small projection is useful for routing only: it cannot
        substitute for fresh criterion validation, analyzer health,
        descendant proof, or an exhaustion receipt.
        """

        return {
            "objective_id": "ASI-G030",
            "receipt_id": self.receipt.receipt_id,
            "repository_tree_id": self.receipt.repository_tree_id,
            "requirement_ids": list(self.receipt.proved_requirement_ids),
            "evidence_ids": list(self.receipt.evidence_ids),
            "completion_authority": False,
            "safe_for_completion_reasoning": False,
        }

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
        """Evaluate the receipt's refinement objective without self-promotion.

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

        if self.receipt.unchanged_failure_backoff_evidence is not None:
            objective_goal_id = UNCHANGED_FAILURE_BACKOFF_GOAL_ID
            acceptance_criteria = UNCHANGED_FAILURE_BACKOFF_ACCEPTANCE_CRITERIA
        else:
            objective_goal_id = NEW_EVIDENCE_REFINEMENT_GOAL_ID
            acceptance_criteria = NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA

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
        else:
            # This objective validates analyzer health independently from its
            # exhaustive quorum. Translate that reviewed contract into the
            # stricter generic completion-gate vocabulary.
            health_value = {**health_value, "exhaustive": True}

        # GoalCoverageMap is the canonical repository-wide producer.  Ask it
        # for this objective's narrow projection rather than inspecting every
        # unrelated goal row in its full serialization.
        coverage_projection = getattr(coverage, "completion_gate_evidence", None)
        if callable(coverage_projection):
            try:
                projected = coverage_projection(objective_goal_id)
            except (TypeError, ValueError):
                projected = {}
            coverage_value = (
                dict(projected) if isinstance(projected, Mapping) else {}
            )
        else:
            coverage_value = payload(coverage)
        coverage_rows = coverage_value.get("criteria")
        coverage_rows = coverage_rows if isinstance(coverage_rows, list) else []
        criterion_keys = {
            " ".join(item.strip().lower().split())
            for item in acceptance_criteria
        }
        relevant_coverage_rows = [
            row
            for row in coverage_rows
            if isinstance(row, Mapping)
            and " ".join(
                str(
                    row.get(
                        "criterion",
                        row.get("acceptance_criterion", row.get("acceptance", "")),
                    )
                    or ""
                )
                .strip()
                .lower()
                .split()
            )
            in criterion_keys
        ]

        def populated(row: Mapping[str, Any], *names: str) -> bool:
            for name in names:
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

        submitted_validation_ids: set[str] = set()
        for item in evidence:
            source: Any = item
            if isinstance(source, Mapping) and isinstance(
                source.get("evidence"), Mapping
            ):
                source = source["evidence"]
            identity = (
                source.get("provenance_cid")
                if isinstance(source, Mapping)
                else getattr(source, "provenance_cid", "")
            )
            identity = str(identity or "").strip()
            if identity:
                submitted_validation_ids.add(identity)

        def validation_bound(row: Mapping[str, Any]) -> bool:
            receipt_ids = row.get("validation_receipt_ids")
            if isinstance(receipt_ids, Sequence) and not isinstance(
                receipt_ids, (str, bytes, bytearray)
            ):
                normalized = {
                    str(item or "").strip()
                    for item in receipt_ids
                    if str(item or "").strip()
                }
                return bool(
                    normalized
                    and normalized.intersection(submitted_validation_ids)
                )
            # Preserve the reviewed ASI-058 mapping spelling for persisted
            # compatibility. Canonical GoalCoverageMap projections always use
            # validation_receipt_ids and therefore take the bound path above.
            return populated(row, "validation")

        coverage_bindings_complete = (
            len(relevant_coverage_rows) >= len(criterion_keys)
            and {
                " ".join(
                    str(
                        row.get(
                            "criterion",
                            row.get(
                                "acceptance_criterion",
                                row.get("acceptance", ""),
                            ),
                        )
                        or ""
                    )
                    .strip()
                    .lower()
                    .split()
                )
                for row in relevant_coverage_rows
            }
            == criterion_keys
            and all(
                populated(
                    row,
                    "implementation",
                    "changed_files",
                    "predicted_files",
                    "ast_symbols",
                    "interfaces",
                )
                and validation_bound(row)
                for row in relevant_coverage_rows
            )
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
        # completion-safe exhaustive receipt.  Enforce the configured count,
        # unique member/receipt/channel identities, and exact binding here as
        # well as in the generic gate.  This objective boundary must fail
        # closed even if a future generic projection becomes more permissive.
        from .scan_receipts import ExhaustionQuorumResult

        evaluated_quorum = isinstance(
            exhaustion_quorum,
            ExhaustionQuorumResult,
        )
        quorum_value = payload(exhaustion_quorum)
        quorum_members = quorum_value.get("members")
        quorum_members = (
            quorum_members if isinstance(quorum_members, list) else []
        )
        quorum_members_healthy = bool(quorum_members) and (
            # ExhaustionQuorumResult members have already passed the
            # canonical evaluator's terminal, exhaustive-mode, analyzer
            # health, coverage, actionability, and binding filters. Its
            # persisted member projection intentionally does not duplicate
            # health fields from the underlying scan receipts.
            evaluated_quorum
            or all(
                isinstance(member, Mapping)
                and member.get("healthy") is True
                and member.get("safe_for_completion_reasoning") is True
                and str(member.get("scan_mode") or "").strip().lower()
                == "exhaustive"
                for member in quorum_members
            )
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
            binding.get("tree_id") == self.receipt.repository_tree_id
            and all(
                isinstance(member, Mapping)
                and isinstance(member.get("binding"), Mapping)
                and dict(member["binding"]) == binding
                for member in quorum_members
            )
        )
        quorum_valid = (
            quorum_members_healthy
            and configured_count_met
            and independent(member_ids)
            and independent(receipt_ids)
            and independent(channels)
            and binding_is_current
            and quorum_value.get("satisfied") is True
            and quorum_value.get("quorum_met") is True
        )
        if not quorum_valid:
            quorum_value = {
                **quorum_value,
                "satisfied": False,
                "quorum_met": False,
            }
        else:
            translated_members = []
            for member in quorum_members:
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
                            or binding["analyzer_version"]
                        ),
                        "scan_mode": "audit" if is_audit else "exhaustive",
                    }
                )
            quorum_value = {
                **quorum_value,
                "members": translated_members,
            }

        values: dict[str, Any] = {
            "current_state": current_state,
            "acceptance_criteria": acceptance_criteria,
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
        self._receipts = list(_validate_receipt_history(receipts))

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
                _validate_receipt_history((*self._receipts, receipt))
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
            return _validate_receipt_history(result)

    def append(self, receipt: AdaptiveRefinementReceipt) -> None:
        if not isinstance(receipt, AdaptiveRefinementReceipt):
            raise RefinementPersistenceError(
                "store accepts only AdaptiveRefinementReceipt"
            )
        with self._lock:
            existing = self.receipts()
            if any(item.receipt_id == receipt.receipt_id for item in existing):
                return
            _validate_receipt_history((*existing, receipt))
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
InformationGainEstimator = Callable[
    [AdaptiveRefinementRequest], RefinementValueEstimate
]
CandidateQualityLinter = Callable[
    [AdaptiveRefinementCandidate, AdaptiveRefinementRequest],
    RefinementDeltaQualityReport,
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
        value_estimator: InformationGainEstimator | None = None,
        quality_linter: CandidateQualityLinter | None = None,
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
        self.value_estimator = value_estimator or self._default_value_estimate
        self.quality_linter = quality_linter or self._default_quality_lint
        if not callable(self.value_estimator):
            raise AdaptiveGoalRefinementError("value_estimator must be callable")
        if not callable(self.quality_linter):
            raise AdaptiveGoalRefinementError("quality_linter must be callable")

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
                and item.previous_plan_id == request.plan.content_id
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
                    if item.decision in REFINEMENT_FAILURE_DECISIONS
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
                    backoff_source=latest_failure,
                )
                self._persist_nonadmission(receipt)
                return AdaptiveRefinementResult(receipt)

            try:
                value_estimate = self.value_estimator(request)
            except BaseException as exc:
                return self._terminal(
                    request,
                    RefinementDecision.INSUFFICIENT_INFORMATION_GAIN,
                    "information-gain estimation failed closed: "
                    f"{type(exc).__name__}: {exc}",
                    model_called=False,
                )
            if not isinstance(value_estimate, RefinementValueEstimate):
                return self._terminal(
                    request,
                    RefinementDecision.INSUFFICIENT_INFORMATION_GAIN,
                    "information-gain estimator returned an unauditable value",
                    model_called=False,
                )
            if value_estimate.signal_ids != tuple(
                item.evidence_id for item in request.signals
            ):
                return self._terminal(
                    request,
                    RefinementDecision.INSUFFICIENT_INFORMATION_GAIN,
                    "information-gain estimate does not bind the request signals",
                    model_called=False,
                )
            if (
                value_estimate.information_gain_millionths
                < self.policy.min_information_gain_millionths
                or value_estimate.expected_downstream_cost_millionths
                > self.policy.max_expected_downstream_cost_millionths
                or value_estimate.net_value_millionths <= 0
            ):
                return self._terminal(
                    request,
                    RefinementDecision.INSUFFICIENT_INFORMATION_GAIN,
                    "expected information gain does not justify downstream cost",
                    model_called=False,
                    value_estimate=value_estimate,
                )

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
                    value_estimate=value_estimate,
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
                    value_estimate=value_estimate,
                )

            attempt_index = len(matching) + 1
            failure_index = (
                sum(
                    item.decision in REFINEMENT_FAILURE_DECISIONS
                    for item in matching
                )
                + 1
            )
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
                    failure_index=failure_index,
                    value_estimate=value_estimate,
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
                    failure_index=failure_index,
                    producer_id=candidate.producer_id,
                    producer_kind=candidate.producer_kind.value,
                    candidate_plan_id=candidate.plan.content_id,
                    value_estimate=value_estimate,
                )

            try:
                quality_report = self.quality_linter(candidate, request)
            except BaseException as exc:
                return self._failure(
                    request,
                    RefinementDecision.CANDIDATE_REJECTED,
                    now,
                    attempt_index,
                    refinement_index,
                    "delta quality lint failed closed: "
                    f"{type(exc).__name__}: {exc}",
                    failure_index=failure_index,
                    producer_id=candidate.producer_id,
                    producer_kind=candidate.producer_kind.value,
                    candidate_plan_id=candidate.plan.content_id,
                    value_estimate=value_estimate,
                )
            quality_violation = self._quality_violation(
                quality_report, candidate, request
            )
            if quality_violation:
                return self._failure(
                    request,
                    RefinementDecision.CANDIDATE_REJECTED,
                    now,
                    attempt_index,
                    refinement_index,
                    quality_violation,
                    failure_index=failure_index,
                    producer_id=candidate.producer_id,
                    producer_kind=candidate.producer_kind.value,
                    candidate_plan_id=candidate.plan.content_id,
                    value_estimate=value_estimate,
                    quality_lint_report=(
                        quality_report
                        if isinstance(
                            quality_report, RefinementDeltaQualityReport
                        )
                        else None
                    ),
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
                    failure_index=failure_index,
                    producer_id=candidate.producer_id,
                    producer_kind=candidate.producer_kind.value,
                    candidate_plan_id=candidate.plan.content_id,
                    verification_receipt_id=verification_id,
                    value_estimate=value_estimate,
                    quality_lint_report=quality_report,
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
                value_estimate=value_estimate,
                quality_lint_report=quality_report,
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
                    value_estimate=value_estimate,
                    quality_lint_report=quality_report,
                    retry_after=self._retry_after(now, failure_index),
                    attempt_index=attempt_index,
                    refinement_index=refinement_index,
                )
                return AdaptiveRefinementResult(failed)
            return AdaptiveRefinementResult(receipt, candidate.plan)

    refine_goal = refine

    @staticmethod
    def _default_value_estimate(
        request: AdaptiveRefinementRequest,
    ) -> RefinementValueEstimate:
        """Estimate novelty and blast radius without invoking a provider."""

        gain_by_kind = {
            RefinementSignalKind.COUNTEREXAMPLE: 900_000,
            RefinementSignalKind.UNCOVERED_CRITERION: 900_000,
            RefinementSignalKind.OPERATOR_REVISION: 850_000,
            RefinementSignalKind.STALE_EVIDENCE: 750_000,
            RefinementSignalKind.INTERFACE_CHANGE: 750_000,
            RefinementSignalKind.CAPABILITY_CHANGE: 700_000,
            RefinementSignalKind.UNCERTAINTY_CHANGE: 650_000,
            RefinementSignalKind.SCOPE_CONFLICT: 800_000,
            RefinementSignalKind.RESOURCE_INFEASIBLE: 800_000,
            RefinementSignalKind.RESOURCE_CHANGE: 650_000,
            RefinementSignalKind.SCOPE_CHANGE: 650_000,
            RefinementSignalKind.REPEATED_FAILURE: 600_000,
        }
        gains = [gain_by_kind[item.kind] for item in request.signals]
        information_gain = min(
            1_000_000,
            max(gains) + 25_000 * (len(request.signals) - 1),
        )
        affected = tuple(
            sorted({item.subject_id for item in request.signals})
        )
        # More affected subjects and a larger existing plan imply a broader,
        # costlier revalidation suffix.  The estimate is deliberately bounded
        # and independent of provider-authored rationale text.
        plan_nodes = len(request.plan.goals) + len(request.plan.subgoals)
        downstream_cost = min(
            1_000_000,
            50_000
            + 50_000 * max(0, len(affected) - 1)
            + 10_000 * max(0, plan_nodes - 1),
        )
        return RefinementValueEstimate(
            information_gain_millionths=information_gain,
            expected_downstream_cost_millionths=downstream_cost,
            affected_subject_ids=affected,
            signal_ids=tuple(item.evidence_id for item in request.signals),
            rationale_codes=tuple(
                sorted(
                    {
                        f"semantic_event:{item.kind.value}"
                        for item in request.signals
                    }
                )
            ),
        )

    @staticmethod
    def _default_quality_lint(
        candidate: AdaptiveRefinementCandidate,
        request: AdaptiveRefinementRequest,
    ) -> RefinementDeltaQualityReport:
        """Lint the exact canonical delta after formal-plan validation."""

        return RefinementDeltaQualityReport(
            previous_plan_id=request.plan.content_id,
            candidate_plan_id=candidate.plan.content_id,
            root_goal_content_id=request.root_goal_content_id,
            assumption_ids=request.assumption_ids,
            changed_goal_ids=candidate.changed_goal_ids,
            accepted=True,
        )

    @staticmethod
    def _quality_violation(
        value: Any,
        candidate: AdaptiveRefinementCandidate,
        request: AdaptiveRefinementRequest,
    ) -> str:
        if not isinstance(value, RefinementDeltaQualityReport):
            return "quality linter returned an unauditable report"
        expected = {
            "previous_plan_id": request.plan.content_id,
            "candidate_plan_id": candidate.plan.content_id,
            "root_goal_content_id": request.root_goal_content_id,
            "assumption_ids": request.assumption_ids,
            "changed_goal_ids": candidate.changed_goal_ids,
        }
        mismatched = [
            name
            for name, expected_value in expected.items()
            if getattr(value, name) != expected_value
        ]
        if mismatched:
            return (
                "delta quality report does not bind the exact candidate: "
                + ", ".join(mismatched)
            )
        if not value.accepted:
            return "candidate delta failed quality lint: " + ", ".join(
                value.debt_codes
            )
        return ""

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
        *,
        failure_index: int,
        **fields: Any,
    ) -> AdaptiveRefinementResult:
        receipt = self._receipt(
            request,
            decision,
            now,
            model_called=True,
            reason=reason,
            retry_after=self._retry_after(now, failure_index),
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
        value_estimate: RefinementValueEstimate | None = None,
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
            value_estimate=value_estimate,
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
        backoff_source: AdaptiveRefinementReceipt | None = None,
        value_estimate: RefinementValueEstimate | None = None,
        quality_lint_report: RefinementDeltaQualityReport | None = None,
    ) -> AdaptiveRefinementReceipt:
        requirement_ids: list[str] = []
        counterexample_evidence: NewCounterexampleRefinementEvidence | None = None
        backoff_evidence: UnchangedFailureBackoffEvidence | None = None
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
            if backoff_source is None:
                raise AdaptiveGoalRefinementError(
                    "repeated-failure backoff requires its persisted source failure"
                )
            signal = request.signals[0]
            backoff_evidence = UnchangedFailureBackoffEvidence(
                repeated_failure_signal_id=signal.evidence_id,
                failure_signature=signal.failure_signature,
                request_id=request.content_id,
                cycle_id=request.cycle_id,
                evidence_fingerprint=request.evidence_fingerprint,
                root_goal_id=request.root_goal_id,
                root_goal_content_id=request.root_goal_content_id,
                assumption_ids=request.assumption_ids,
                policy_id=self.policy.content_id,
                repository_tree_id=request.repository_tree_id,
                previous_plan_id=request.plan.content_id,
                source_failure_receipt_id=backoff_source.receipt_id,
                source_failure_decision=backoff_source.decision.value,
                source_failure_model_called=backoff_source.model_called,
                source_failure_attempted_at=backoff_source.attempted_at,
                source_failure_retry_after=backoff_source.retry_after,
                source_failure_attempt_index=backoff_source.attempt_index,
                source_failure_refinement_index=(
                    backoff_source.refinement_index
                ),
                suppressed_at=now,
                suppressed_attempt_index=attempt_index,
                retry_after=retry_after,
            )
            backoff_evidence.validate_source(backoff_source)
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
            value_estimate=(
                value_estimate or self._default_value_estimate(request)
            ),
            quality_lint_report=quality_lint_report,
            new_counterexample_evidence=counterexample_evidence,
            unchanged_failure_backoff_evidence=backoff_evidence,
        )


def refine_goal_from_evidence(
    request: AdaptiveRefinementRequest,
    generator: CandidateGenerator,
    verifier: CandidateVerifier,
    *,
    policy: AdaptiveRefinementPolicy | None = None,
    store: RefinementReceiptStore | None = None,
    clock: Callable[[], int | float] | None = None,
    value_estimator: InformationGainEstimator | None = None,
    quality_linter: CandidateQualityLinter | None = None,
) -> AdaptiveRefinementResult:
    """Functional entry point for one bounded adaptive-refinement cycle."""

    return AdaptiveGoalRefiner(
        generator,
        verifier,
        policy=policy,
        store=store,
        clock=clock,
        value_estimator=value_estimator,
        quality_linter=quality_linter,
    ).refine(request)


# ---------------------------------------------------------------------------
# WPD-042 / RefillResidualGuard@1
# ---------------------------------------------------------------------------
# Guards only: refilled / backlog-generated tasks inherit residual LLM rules
# and doctor preconditions.  This surface never mutates the objective heap or
# protected WPD control anchors.  Generated tasks must carry the
# pre-implementation kernel flag, and residual_llm_authorized is admitted only
# when the residual packet schema is declared.

REFILL_RESIDUAL_GUARD_INTERFACE: Final[str] = "RefillResidualGuard@1"
REFILL_RESIDUAL_GUARD_VERSION: Final[int] = 1
REFILL_RESIDUAL_GUARD_EVIDENCE: Final[str] = "wpd/refill-guard@1"
REFILL_RESIDUAL_GUARD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refill-residual-guard@1"
)
REFILL_RESIDUAL_GUARD_PRODUCER: Final[str] = "refill-residual-guard@1"

# Canonical residual packet schema that residual_llm_authorized must declare.
# Kept as a literal so this guard remains a leaf over residual packet identity
# without importing the full packet constructor graph.
REFILL_RESIDUAL_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-llm-packet@1"
)

# Wire keys stamped onto / required of generated refill tasks.
PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: Final[str] = (
    "requires_pre_implementation_kernel"
)
RESIDUAL_PACKET_SCHEMA_KEY: Final[str] = "residual_packet_schema"
IMPLEMENTATION_DISPOSITION_KEY: Final[str] = "implementation_disposition"
DOCTOR_PRECONDITIONS_KEY: Final[str] = "doctor_preconditions"
REFILL_RESIDUAL_RULES_KEY: Final[str] = "refill_residual_rules"

RESIDUAL_LLM_AUTHORIZED_DISPOSITION: Final[str] = "residual_llm_authorized"

# Doctor preconditions that refilled tasks inherit and cannot drop.
REQUIRED_DOCTOR_PRECONDITIONS: Final[tuple[str, ...]] = (
    "doctor_inspect_on_typed_failure",
    "formal_replan_on_typed_failure",
    "residual_packet_before_provider_retry",
    "pre_implementation_kernel_before_provider",
)

# Operator-protected WPD control anchors.  Refill guards reject any generated
# task that lists these as write outputs (guards only; no heap mutation).
DEFAULT_WPD_PROTECTED_CONTROL_PATHS: Final[tuple[str, ...]] = (
    "implementation_plan/docs/47-supervisor-worker-planner-doctor-integration-plan-2026-08-06.md",
    "implementation_plan/docs/47-supervisor-worker-planner-doctor-integration.objectives.md",
    "implementation_plan/docs/47-supervisor-worker-planner-doctor-integration.todo.md",
    "config/supervisor_worker_planner_doctor_integration_scheduler.json",
    "config/supervisor_worker_planner_doctor_supervisor.json",
    "config/supervisor_worker_planner_doctor_authority_policy.json",
    "scripts/validate_supervisor_worker_planner_doctor_board.py",
    "scripts/supervisor_worker_planner_doctor_supervisor.sh",
)

# Stable rejection reason codes (fail closed).
REASON_MISSING_PRE_IMPLEMENTATION_KERNEL_FLAG: Final[str] = (
    "missing_pre_implementation_kernel_flag"
)
REASON_PRE_IMPLEMENTATION_KERNEL_FLAG_FALSE: Final[str] = (
    "pre_implementation_kernel_flag_false"
)
REASON_RESIDUAL_LLM_WITHOUT_PACKET_SCHEMA: Final[str] = (
    "residual_llm_authorized_without_packet_schema"
)
REASON_UNKNOWN_PACKET_SCHEMA: Final[str] = "unknown_residual_packet_schema"
REASON_DROPPED_DOCTOR_PRECONDITION: Final[str] = "dropped_doctor_precondition"
REASON_PROTECTED_CONTROL_PATH: Final[str] = "protected_control_path_write"
REASON_OBJECTIVE_HEAP_MUTATION: Final[str] = (
    "objective_heap_mutation_forbidden"
)
REASON_MALFORMED_TASK: Final[str] = "malformed_refill_task"

_WRITE_PATH_KEYS: Final[tuple[str, ...]] = (
    "predicted_files",
    "output_paths",
    "write_paths",
    "expected_outputs",
    "outputs",
)

_HEAP_MUTATION_TRUE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "mutates_objective_heap",
        "objective_heap_mutation",
        "authorizes_objective_heap_mutation",
        "authorize_objective_heap_mutation",
        "completion_authority",
        "mutation_authority",
    }
)


class RefillResidualGuardError(AdaptiveGoalRefinementError):
    """A generated refill task violates residual / doctor guard rules."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        codes = tuple(
            sorted(
                {
                    _text(code, "reason_code")
                    for code in reason_codes
                    if str(code or "").strip()
                }
            )
        )
        self.reason_codes = codes


class RefillResidualGuardVerdict(str, Enum):
    """Closed guard outcomes for one generated refill task."""

    ADMITTED = "admitted"
    REJECTED = "rejected"


def _truthy_flag(value: Any) -> bool | None:
    """Return True/False for explicit boolean-like flags, else None."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "required", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", "forbidden"}:
            return False
    return None


def _task_mapping(task: Any, *, name: str = "task") -> dict[str, Any]:
    if task is None:
        raise RefillResidualGuardError(
            f"{name} is required",
            reason_codes=(REASON_MALFORMED_TASK,),
        )
    if isinstance(task, Mapping):
        if any(not isinstance(key, str) for key in task):
            raise RefillResidualGuardError(
                f"{name} keys must be strings",
                reason_codes=(REASON_MALFORMED_TASK,),
            )
        return dict(task)
    to_dict = getattr(task, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping) and all(
            isinstance(key, str) for key in payload
        ):
            return dict(payload)
    raise RefillResidualGuardError(
        f"{name} must be a mapping or expose to_dict()",
        reason_codes=(REASON_MALFORMED_TASK,),
    )


def _disposition_token(task: Mapping[str, Any]) -> str:
    for key in (
        IMPLEMENTATION_DISPOSITION_KEY,
        "disposition",
        "implementation_disposition_value",
    ):
        raw = task.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, Enum):
            return str(raw.value).strip().lower()
        return str(raw).strip().lower()
    metadata = task.get("metadata")
    if isinstance(metadata, Mapping):
        for key in (IMPLEMENTATION_DISPOSITION_KEY, "disposition"):
            raw = metadata.get(key)
            if raw is None or raw == "":
                continue
            if isinstance(raw, Enum):
                return str(raw.value).strip().lower()
            return str(raw).strip().lower()
    return ""


def _packet_schema_token(task: Mapping[str, Any]) -> str:
    for key in (
        RESIDUAL_PACKET_SCHEMA_KEY,
        "packet_schema",
        "residual_llm_packet_schema",
        "residual_packet_schema_id",
    ):
        raw = task.get(key)
        if raw is None or raw == "":
            continue
        return str(raw).strip()
    metadata = task.get("metadata")
    if isinstance(metadata, Mapping):
        for key in (
            RESIDUAL_PACKET_SCHEMA_KEY,
            "packet_schema",
            "residual_llm_packet_schema",
        ):
            raw = metadata.get(key)
            if raw is None or raw == "":
                continue
            return str(raw).strip()
    residual_rules = task.get(REFILL_RESIDUAL_RULES_KEY)
    if isinstance(residual_rules, Mapping):
        raw = residual_rules.get(RESIDUAL_PACKET_SCHEMA_KEY)
        if raw not in (None, ""):
            return str(raw).strip()
    return ""


def _kernel_flag_value(task: Mapping[str, Any]) -> bool | None:
    for key in (
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY,
        "pre_implementation_kernel",
        "pre_implementation_kernel_required",
        "require_pre_implementation_kernel",
    ):
        if key in task:
            return _truthy_flag(task.get(key))
    metadata = task.get("metadata")
    if isinstance(metadata, Mapping):
        for key in (
            PRE_IMPLEMENTATION_KERNEL_FLAG_KEY,
            "pre_implementation_kernel",
            "pre_implementation_kernel_required",
        ):
            if key in metadata:
                return _truthy_flag(metadata.get(key))
    residual_rules = task.get(REFILL_RESIDUAL_RULES_KEY)
    if isinstance(residual_rules, Mapping):
        if PRE_IMPLEMENTATION_KERNEL_FLAG_KEY in residual_rules:
            return _truthy_flag(
                residual_rules.get(PRE_IMPLEMENTATION_KERNEL_FLAG_KEY)
            )
    return None


def _doctor_preconditions(task: Mapping[str, Any]) -> tuple[str, ...]:
    collected: list[str] = []
    for container in (
        task,
        task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {},
        (
            task.get(REFILL_RESIDUAL_RULES_KEY)
            if isinstance(task.get(REFILL_RESIDUAL_RULES_KEY), Mapping)
            else {}
        ),
    ):
        if not isinstance(container, Mapping):
            continue
        raw = container.get(DOCTOR_PRECONDITIONS_KEY)
        if raw is None:
            raw = container.get("preconditions")
        if raw is None:
            continue
        if isinstance(raw, (str, bytes, bytearray)):
            text = str(raw).strip()
            if text:
                collected.append(text)
            continue
        if isinstance(raw, Iterable):
            for item in raw:
                text = str(item or "").strip()
                if text:
                    collected.append(text)
    # Preserve first-seen order while deduplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in collected:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _write_paths(task: Mapping[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    containers: list[Mapping[str, Any]] = [task]
    metadata = task.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)
    for container in containers:
        for key in _WRITE_PATH_KEYS:
            raw = container.get(key)
            if raw is None:
                continue
            if isinstance(raw, (str, bytes, bytearray)):
                text = str(raw).strip().replace("\\", "/")
                if text:
                    paths.append(text)
                continue
            if isinstance(raw, Iterable):
                for item in raw:
                    text = str(item or "").strip().replace("\\", "/")
                    if text:
                        paths.append(text)
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return tuple(ordered)


def _path_is_protected(path: str, protected: Sequence[str]) -> bool:
    normalized = path.replace("\\", "/").strip().lstrip("./")
    for anchor in protected:
        target = str(anchor).replace("\\", "/").strip().lstrip("./")
        if not target:
            continue
        if normalized == target or normalized.startswith(target + "/"):
            return True
    return False


def _claims_heap_mutation(task: Mapping[str, Any]) -> bool:
    containers: list[Mapping[str, Any]] = [task]
    metadata = task.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)
    residual_rules = task.get(REFILL_RESIDUAL_RULES_KEY)
    if isinstance(residual_rules, Mapping):
        containers.append(residual_rules)
    for container in containers:
        for key in _HEAP_MUTATION_TRUE_KEYS:
            if key not in container:
                continue
            flag = _truthy_flag(container.get(key))
            if flag is True:
                return True
    return False


def default_refill_residual_rules() -> dict[str, Any]:
    """Return the residual/LLM rules inherited by every guarded refill task."""

    return {
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        RESIDUAL_PACKET_SCHEMA_KEY: REFILL_RESIDUAL_PACKET_SCHEMA,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        "residual_llm_authorized_requires_packet_schema": True,
        "free_reprompt_after_typed_failure_forbidden": True,
        "objective_heap_mutation": False,
        "completion_authority": False,
        "mutation_authority": False,
        "interface": REFILL_RESIDUAL_GUARD_INTERFACE,
        "evidence": REFILL_RESIDUAL_GUARD_EVIDENCE,
        "version": REFILL_RESIDUAL_GUARD_VERSION,
    }


def stamp_refill_residual_rules(
    task: Mapping[str, Any] | Any,
    *,
    disposition: str | None = None,
    residual_packet_schema: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``task`` with mandatory residual/doctor rules stamped.

    Generation path: every refill task inherits the pre-implementation kernel
    flag and doctor preconditions.  When disposition is residual_llm_authorized
    the residual packet schema is also stamped.  This never grants completion
    or objective-heap mutation authority.
    """

    payload = _task_mapping(task)
    rules = default_refill_residual_rules()
    existing_rules = payload.get(REFILL_RESIDUAL_RULES_KEY)
    if isinstance(existing_rules, Mapping):
        # Caller-supplied rules may only tighten, never drop required fields.
        merged = dict(existing_rules)
        merged.update(rules)
        # Preserve any extra doctor preconditions the caller already listed.
        prior = _doctor_preconditions(payload)
        merged[DOCTOR_PRECONDITIONS_KEY] = list(
            dict.fromkeys([*REQUIRED_DOCTOR_PRECONDITIONS, *prior])
        )
        rules = merged

    disposition_token = (
        str(disposition).strip().lower()
        if disposition not in (None, "")
        else _disposition_token(payload)
    )
    schema_token = (
        str(residual_packet_schema).strip()
        if residual_packet_schema not in (None, "")
        else _packet_schema_token(payload)
    )
    if disposition_token == RESIDUAL_LLM_AUTHORIZED_DISPOSITION:
        if not schema_token:
            schema_token = REFILL_RESIDUAL_PACKET_SCHEMA
        if schema_token != REFILL_RESIDUAL_PACKET_SCHEMA:
            raise RefillResidualGuardError(
                "residual_llm_authorized requires the residual-llm-packet@1 schema",
                reason_codes=(REASON_UNKNOWN_PACKET_SCHEMA,),
            )
        payload[IMPLEMENTATION_DISPOSITION_KEY] = (
            RESIDUAL_LLM_AUTHORIZED_DISPOSITION
        )
        payload[RESIDUAL_PACKET_SCHEMA_KEY] = schema_token
        rules[RESIDUAL_PACKET_SCHEMA_KEY] = schema_token
        rules[IMPLEMENTATION_DISPOSITION_KEY] = (
            RESIDUAL_LLM_AUTHORIZED_DISPOSITION
        )
    elif disposition_token:
        payload[IMPLEMENTATION_DISPOSITION_KEY] = disposition_token
        # Non-residual dispositions must not carry residual packet schema.
        payload.pop(RESIDUAL_PACKET_SCHEMA_KEY, None)
        rules.pop(IMPLEMENTATION_DISPOSITION_KEY, None)
        # Keep schema identity as documentation of the residual rule set.
        rules[RESIDUAL_PACKET_SCHEMA_KEY] = REFILL_RESIDUAL_PACKET_SCHEMA

    payload[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] = True
    payload[DOCTOR_PRECONDITIONS_KEY] = list(
        dict.fromkeys(
            [*REQUIRED_DOCTOR_PRECONDITIONS, *_doctor_preconditions(payload)]
        )
    )
    rules[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] = True
    rules[DOCTOR_PRECONDITIONS_KEY] = list(payload[DOCTOR_PRECONDITIONS_KEY])
    rules["objective_heap_mutation"] = False
    rules["completion_authority"] = False
    rules["mutation_authority"] = False
    payload[REFILL_RESIDUAL_RULES_KEY] = rules
    payload["mutates_objective_heap"] = False
    payload["completion_authority"] = False
    payload["mutation_authority"] = False
    return payload


@dataclass(frozen=True)
class RefillResidualGuardResult:
    """Content-addressed outcome of one refill residual guard evaluation."""

    verdict: RefillResidualGuardVerdict
    reason_codes: tuple[str, ...]
    task: Mapping[str, Any]
    guarded_task: Mapping[str, Any] | None = None
    requires_pre_implementation_kernel: bool = False
    disposition: str = ""
    residual_packet_schema: str = ""
    doctor_preconditions: tuple[str, ...] = ()
    interface: str = REFILL_RESIDUAL_GUARD_INTERFACE
    version: int = REFILL_RESIDUAL_GUARD_VERSION
    evidence: str = REFILL_RESIDUAL_GUARD_EVIDENCE
    producer_id: str = REFILL_RESIDUAL_GUARD_PRODUCER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verdict",
            (
                self.verdict
                if isinstance(self.verdict, RefillResidualGuardVerdict)
                else RefillResidualGuardVerdict(str(self.verdict))
            ),
        )
        codes = tuple(
            sorted(
                {
                    _text(code, "reason_code")
                    for code in self.reason_codes
                    if str(code or "").strip()
                }
            )
        )
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(
            self, "task", MappingProxyType(dict(self.task))
        )
        if self.guarded_task is None:
            object.__setattr__(self, "guarded_task", None)
        else:
            object.__setattr__(
                self,
                "guarded_task",
                MappingProxyType(dict(self.guarded_task)),
            )
        object.__setattr__(
            self,
            "requires_pre_implementation_kernel",
            bool(self.requires_pre_implementation_kernel),
        )
        object.__setattr__(
            self,
            "disposition",
            _text(self.disposition, "disposition", required=False),
        )
        object.__setattr__(
            self,
            "residual_packet_schema",
            _text(
                self.residual_packet_schema,
                "residual_packet_schema",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "doctor_preconditions",
            tuple(
                _text(item, "doctor_precondition")
                for item in self.doctor_preconditions
            ),
        )
        object.__setattr__(
            self,
            "interface",
            _text(self.interface, "interface") or REFILL_RESIDUAL_GUARD_INTERFACE,
        )
        object.__setattr__(
            self,
            "version",
            _positive(self.version, "version"),
        )
        object.__setattr__(
            self,
            "evidence",
            _text(self.evidence, "evidence") or REFILL_RESIDUAL_GUARD_EVIDENCE,
        )
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id, "producer_id")
            or REFILL_RESIDUAL_GUARD_PRODUCER,
        )
        if self.verdict is RefillResidualGuardVerdict.ADMITTED and codes:
            raise RefillResidualGuardError(
                "admitted guard results cannot carry rejection reason codes",
                reason_codes=codes,
            )
        if (
            self.verdict is RefillResidualGuardVerdict.ADMITTED
            and self.guarded_task is None
        ):
            raise RefillResidualGuardError(
                "admitted guard results require a guarded_task",
                reason_codes=(REASON_MALFORMED_TASK,),
            )

    @property
    def admitted(self) -> bool:
        return self.verdict is RefillResidualGuardVerdict.ADMITTED

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": REFILL_RESIDUAL_GUARD_SCHEMA,
            "interface": self.interface,
            "version": self.version,
            "evidence": self.evidence,
            "producer_id": self.producer_id,
            "verdict": self.verdict.value,
            "reason_codes": list(self.reason_codes),
            "requires_pre_implementation_kernel": (
                self.requires_pre_implementation_kernel
            ),
            "disposition": self.disposition,
            "residual_packet_schema": self.residual_packet_schema,
            "doctor_preconditions": list(self.doctor_preconditions),
            "task": dict(self.task),
        }
        if self.guarded_task is not None:
            payload["guarded_task"] = dict(self.guarded_task)
        return payload


def evaluate_refill_residual_guard(
    task: Mapping[str, Any] | Any,
    *,
    protected_paths: Sequence[str] | None = None,
    require_all_doctor_preconditions: bool = True,
    stamp_on_admit: bool = True,
) -> RefillResidualGuardResult:
    """Evaluate residual/LLM rules on a generated refill task (fail closed).

    Acceptance (WPD-042):

    * generated refill tasks require the pre-implementation kernel flag;
    * residual_llm_authorized cannot be marked without the residual packet
      schema;
    * doctor preconditions cannot be dropped; and
    * the guard never admits objective-heap mutation of WPD control files.
    """

    try:
        payload = _task_mapping(task)
    except RefillResidualGuardError as exc:
        return RefillResidualGuardResult(
            verdict=RefillResidualGuardVerdict.REJECTED,
            reason_codes=exc.reason_codes or (REASON_MALFORMED_TASK,),
            task={},
            requires_pre_implementation_kernel=False,
        )

    reasons: list[str] = []
    kernel_flag = _kernel_flag_value(payload)
    if kernel_flag is None:
        reasons.append(REASON_MISSING_PRE_IMPLEMENTATION_KERNEL_FLAG)
    elif kernel_flag is False:
        reasons.append(REASON_PRE_IMPLEMENTATION_KERNEL_FLAG_FALSE)

    disposition = _disposition_token(payload)
    packet_schema = _packet_schema_token(payload)
    if disposition == RESIDUAL_LLM_AUTHORIZED_DISPOSITION:
        if not packet_schema:
            reasons.append(REASON_RESIDUAL_LLM_WITHOUT_PACKET_SCHEMA)
        elif packet_schema != REFILL_RESIDUAL_PACKET_SCHEMA:
            reasons.append(REASON_UNKNOWN_PACKET_SCHEMA)

    preconditions = _doctor_preconditions(payload)
    if require_all_doctor_preconditions:
        missing = [
            item
            for item in REQUIRED_DOCTOR_PRECONDITIONS
            if item not in preconditions
        ]
        if missing:
            reasons.append(REASON_DROPPED_DOCTOR_PRECONDITION)

    anchors = (
        tuple(protected_paths)
        if protected_paths is not None
        else DEFAULT_WPD_PROTECTED_CONTROL_PATHS
    )
    for path in _write_paths(payload):
        if _path_is_protected(path, anchors):
            reasons.append(REASON_PROTECTED_CONTROL_PATH)
            break

    if _claims_heap_mutation(payload):
        reasons.append(REASON_OBJECTIVE_HEAP_MUTATION)

    unique_reasons = tuple(sorted(set(reasons)))
    if unique_reasons:
        return RefillResidualGuardResult(
            verdict=RefillResidualGuardVerdict.REJECTED,
            reason_codes=unique_reasons,
            task=payload,
            requires_pre_implementation_kernel=bool(kernel_flag),
            disposition=disposition,
            residual_packet_schema=packet_schema,
            doctor_preconditions=preconditions,
        )

    guarded = (
        stamp_refill_residual_rules(payload)
        if stamp_on_admit
        else dict(payload)
    )
    return RefillResidualGuardResult(
        verdict=RefillResidualGuardVerdict.ADMITTED,
        reason_codes=(),
        task=payload,
        guarded_task=guarded,
        requires_pre_implementation_kernel=True,
        disposition=_disposition_token(guarded),
        residual_packet_schema=_packet_schema_token(guarded),
        doctor_preconditions=_doctor_preconditions(guarded),
    )


def guard_refill_task(
    task: Mapping[str, Any] | Any,
    *,
    protected_paths: Sequence[str] | None = None,
    require_all_doctor_preconditions: bool = True,
) -> dict[str, Any]:
    """Admit a guarded refill task or raise :class:`RefillResidualGuardError`."""

    result = evaluate_refill_residual_guard(
        task,
        protected_paths=protected_paths,
        require_all_doctor_preconditions=require_all_doctor_preconditions,
        stamp_on_admit=True,
    )
    if not result.admitted or result.guarded_task is None:
        raise RefillResidualGuardError(
            "refill residual guard rejected task: "
            + ",".join(result.reason_codes),
            reason_codes=result.reason_codes,
        )
    return dict(result.guarded_task)


def build_refill_task_with_residual_guard(
    *,
    task_id: str,
    title: str,
    predicted_files: Sequence[str] = (),
    validation_commands: Sequence[str] = (),
    disposition: str = "closed_deterministic",
    residual_packet_schema: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    protected_paths: Sequence[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a generated refill task that already satisfies residual rules.

    The pre-implementation kernel flag and doctor preconditions are always
    stamped.  residual_llm_authorized is only accepted with the residual
    packet schema.
    """

    task_id_text = _text(task_id, "task_id")
    title_text = _text(title, "title")
    disposition_token = _text(disposition, "disposition").lower()
    files = tuple(
        _text(path, "predicted_files")
        for path in predicted_files
        if str(path or "").strip()
    )
    commands = tuple(
        _text(command, "validation_commands")
        for command in validation_commands
        if str(command or "").strip()
    )
    base: dict[str, Any] = {
        "task_id": task_id_text,
        "title": title_text,
        "predicted_files": list(files),
        "validation_commands": list(commands),
        IMPLEMENTATION_DISPOSITION_KEY: disposition_token,
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        "mutates_objective_heap": False,
        "completion_authority": False,
        "mutation_authority": False,
        "source": "refill",
    }
    if metadata:
        if not isinstance(metadata, Mapping) or any(
            not isinstance(key, str) for key in metadata
        ):
            raise RefillResidualGuardError(
                "metadata must be an object with string keys",
                reason_codes=(REASON_MALFORMED_TASK,),
            )
        base["metadata"] = dict(metadata)
    for key, value in extra.items():
        if key in base and key not in {
            "metadata",
            "predicted_files",
            "validation_commands",
        }:
            # Explicit constructor fields win over extras for identity keys.
            continue
        base[key] = value

    if disposition_token == RESIDUAL_LLM_AUTHORIZED_DISPOSITION:
        schema = (
            _text(residual_packet_schema, RESIDUAL_PACKET_SCHEMA_KEY)
            if residual_packet_schema not in (None, "")
            else REFILL_RESIDUAL_PACKET_SCHEMA
        )
        base[RESIDUAL_PACKET_SCHEMA_KEY] = schema
    elif residual_packet_schema not in (None, ""):
        # Declaring a packet schema without residual authorization is ignored
        # at stamp time for non-residual dispositions.
        pass

    stamped = stamp_refill_residual_rules(
        base,
        disposition=disposition_token,
        residual_packet_schema=residual_packet_schema,
    )
    return guard_refill_task(
        stamped,
        protected_paths=protected_paths,
        require_all_doctor_preconditions=True,
    )


@dataclass(frozen=True)
class RefillResidualGuard:
    """Production guard binding residual/LLM rules onto refill generation."""

    protected_paths: tuple[str, ...] = DEFAULT_WPD_PROTECTED_CONTROL_PATHS
    require_all_doctor_preconditions: bool = True
    interface: str = REFILL_RESIDUAL_GUARD_INTERFACE
    version: int = REFILL_RESIDUAL_GUARD_VERSION
    evidence: str = REFILL_RESIDUAL_GUARD_EVIDENCE

    def __post_init__(self) -> None:
        paths = tuple(
            _text(path, "protected_paths")
            for path in self.protected_paths
            if str(path or "").strip()
        )
        object.__setattr__(
            self,
            "protected_paths",
            paths or DEFAULT_WPD_PROTECTED_CONTROL_PATHS,
        )
        object.__setattr__(
            self,
            "require_all_doctor_preconditions",
            bool(self.require_all_doctor_preconditions),
        )
        object.__setattr__(
            self,
            "interface",
            _text(self.interface, "interface") or REFILL_RESIDUAL_GUARD_INTERFACE,
        )
        object.__setattr__(self, "version", _positive(self.version, "version"))
        object.__setattr__(
            self,
            "evidence",
            _text(self.evidence, "evidence") or REFILL_RESIDUAL_GUARD_EVIDENCE,
        )

    def evaluate(
        self, task: Mapping[str, Any] | Any
    ) -> RefillResidualGuardResult:
        """Evaluate residual rules without raising."""

        return evaluate_refill_residual_guard(
            task,
            protected_paths=self.protected_paths,
            require_all_doctor_preconditions=(
                self.require_all_doctor_preconditions
            ),
            stamp_on_admit=True,
        )

    def guard(self, task: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Admit a guarded task or raise."""

        return guard_refill_task(
            task,
            protected_paths=self.protected_paths,
            require_all_doctor_preconditions=(
                self.require_all_doctor_preconditions
            ),
        )

    def build_task(
        self,
        *,
        task_id: str,
        title: str,
        predicted_files: Sequence[str] = (),
        validation_commands: Sequence[str] = (),
        disposition: str = "closed_deterministic",
        residual_packet_schema: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build a residual-rule compliant generated refill task."""

        return build_refill_task_with_residual_guard(
            task_id=task_id,
            title=title,
            predicted_files=predicted_files,
            validation_commands=validation_commands,
            disposition=disposition,
            residual_packet_schema=residual_packet_schema,
            metadata=metadata,
            protected_paths=self.protected_paths,
            **extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "version": self.version,
            "evidence": self.evidence,
            "protected_paths": list(self.protected_paths),
            "require_all_doctor_preconditions": (
                self.require_all_doctor_preconditions
            ),
            "required_doctor_preconditions": list(REQUIRED_DOCTOR_PRECONDITIONS),
            "residual_packet_schema": REFILL_RESIDUAL_PACKET_SCHEMA,
            "pre_implementation_kernel_flag_key": (
                PRE_IMPLEMENTATION_KERNEL_FLAG_KEY
            ),
            "objective_heap_mutation": False,
        }


def create_refill_residual_guard(
    *,
    protected_paths: Sequence[str] | None = None,
    require_all_doctor_preconditions: bool = True,
) -> RefillResidualGuard:
    """Construct the production-default :class:`RefillResidualGuard`."""

    return RefillResidualGuard(
        protected_paths=(
            tuple(protected_paths)
            if protected_paths is not None
            else DEFAULT_WPD_PROTECTED_CONTROL_PATHS
        ),
        require_all_doctor_preconditions=require_all_doctor_preconditions,
    )


__all__ = [
    "ADAPTIVE_GOAL_REFINER_VERSION",
    "ADAPTIVE_REFINEMENT_RECEIPT_VERSION",
    "NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID",
    "NEW_EVIDENCE_REFINEMENT_GOAL_ID",
    "UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID",
    "UNCHANGED_FAILURE_BACKOFF_GOAL_ID",
    "UNCHANGED_FAILURE_BACKOFF_EVIDENCE_SCHEMA",
    "REFINEMENT_VALUE_ESTIMATE_SCHEMA",
    "REFINEMENT_DELTA_QUALITY_SCHEMA",
    "QUALITY_SCHEMA",
    "GOAL_DEBT_SCHEMA",
    "NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA",
    "UNCHANGED_FAILURE_BACKOFF_ACCEPTANCE_CRITERIA",
    "AdaptiveGoalRefinementError",
    "RefinementPersistenceError",
    "RefinementSignalKind",
    "GoalDebtKind",
    "GoalDebtRecord",
    "RefinementDecision",
    "RefinementProducerKind",
    "RefinementSignal",
    "GoalRefinementSignal",
    "GoalQualityRecord",
    "GoalQuality",
    "RefinementValueEstimate",
    "RefinementDeltaQualityReport",
    "AdaptiveRefinementPolicy",
    "GoalRefinementPolicy",
    "RefinementLimits",
    "AdaptiveRefinementRequest",
    "GoalRefinementRequest",
    "AdaptiveRefinementCandidate",
    "GoalRefinementCandidate",
    "GoalRefinementProposal",
    "NewCounterexampleRefinementEvidence",
    "UnchangedFailureBackoffEvidence",
    "AdaptiveRefinementReceipt",
    "GoalRefinementReceipt",
    "AdaptiveRefinementResult",
    "GoalRefinementResult",
    "RefinementReceiptStore",
    "InformationGainEstimator",
    "CandidateQualityLinter",
    "InMemoryRefinementStore",
    "JsonlRefinementStore",
    "AdaptiveGoalRefiner",
    "refine_goal_from_evidence",
    # WPD-042 / RefillResidualGuard@1
    "REFILL_RESIDUAL_GUARD_INTERFACE",
    "REFILL_RESIDUAL_GUARD_VERSION",
    "REFILL_RESIDUAL_GUARD_EVIDENCE",
    "REFILL_RESIDUAL_GUARD_SCHEMA",
    "REFILL_RESIDUAL_GUARD_PRODUCER",
    "REFILL_RESIDUAL_PACKET_SCHEMA",
    "PRE_IMPLEMENTATION_KERNEL_FLAG_KEY",
    "RESIDUAL_PACKET_SCHEMA_KEY",
    "IMPLEMENTATION_DISPOSITION_KEY",
    "DOCTOR_PRECONDITIONS_KEY",
    "REFILL_RESIDUAL_RULES_KEY",
    "RESIDUAL_LLM_AUTHORIZED_DISPOSITION",
    "REQUIRED_DOCTOR_PRECONDITIONS",
    "DEFAULT_WPD_PROTECTED_CONTROL_PATHS",
    "REASON_MISSING_PRE_IMPLEMENTATION_KERNEL_FLAG",
    "REASON_PRE_IMPLEMENTATION_KERNEL_FLAG_FALSE",
    "REASON_RESIDUAL_LLM_WITHOUT_PACKET_SCHEMA",
    "REASON_UNKNOWN_PACKET_SCHEMA",
    "REASON_DROPPED_DOCTOR_PRECONDITION",
    "REASON_PROTECTED_CONTROL_PATH",
    "REASON_OBJECTIVE_HEAP_MUTATION",
    "REASON_MALFORMED_TASK",
    "RefillResidualGuardError",
    "RefillResidualGuardVerdict",
    "RefillResidualGuardResult",
    "RefillResidualGuard",
    "default_refill_residual_rules",
    "stamp_refill_residual_rules",
    "evaluate_refill_residual_guard",
    "guard_refill_task",
    "build_refill_task_with_residual_guard",
    "create_refill_residual_guard",
]
