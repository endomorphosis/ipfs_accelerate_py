"""Deterministic bounded consistency checks for canonical formal work plans.

The validator in this module is deliberately a small reference model rather
than an adapter around an optional theorem prover.  It checks the supervisor's
reviewed DCEC/TDFOL planning subset over an explicit finite trace and keeps
static contradictions, trace countermodels, unsupported semantics, exhausted
search, timeout, and cancellation as different outcomes.

A positive result is evidence about ``FormalWorkPlan`` consistency only.  It
does not attest that work happened or that generated source code is correct.
Native DCEC/TDFOL provider evidence is retained as non-authoritative plan-check
evidence.  Promotion to ``kernel_verified`` requires evidence which binds the
exact plan, formula set, assumptions, and bounds and was reconstructed by an
accepted kernel or checked by an accepted model checker.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from typing import Any, Final

from .formal_logic_vocabulary import (
    DCEC_VOCABULARY_VERSION,
    LOGIC_VOCABULARY_VERSION,
    TDFOL_VOCABULARY_VERSION,
    FiniteTrace,
    Formula,
    FormulaOperator,
    ReviewedPredicate,
    TermSort,
    TraceFact,
    TraceStep,
    constant,
    evaluate_formula,
)
from .formal_planning_contracts import (
    ActorKind,
    EventKind,
    EvidenceRequirement,
    EvidenceRequirementKind,
    FormalWorkPlan,
    NormKind,
    PlanConsistencyLevel,
    PlanEvent,
)
from .formal_verification_contracts import canonical_json, content_identity
from .formal_verification_provider import CancellationToken


FORMAL_PLAN_VALIDATOR_VERSION: Final = 1
FORMAL_PLAN_VALIDATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-validation@1"
)
FORMAL_PLAN_VALIDATION_BOUNDS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-validation-bounds@1"
)
FORMAL_PLAN_FINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-finding@1"
)
FORMAL_PLAN_COUNTERMODEL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-countermodel@1"
)


class PlanValidationStatus(str, Enum):
    """Finite validation status from the reviewed architecture vocabulary."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    VIOLATED = "violated"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    INCOMPLETE = "incomplete"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


ValidationStatus = PlanValidationStatus
PlanValidationVerdict = PlanValidationStatus


class PlanValidationOutcome(str, Enum):
    """More precise primary outcome; it is never collapsed into a boolean."""

    CONSISTENT = "consistent"
    CONTRADICTION = "contradiction"
    COUNTERMODEL = "countermodel"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    INCONCLUSIVE = "inconclusive"
    INCOMPLETE_SEARCH = "incomplete_search"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


ValidationOutcome = PlanValidationOutcome


class PlanCheckKind(str, Enum):
    DEPENDENCY_READINESS = "dependency_readiness"
    ACTOR_AUTHORITY = "actor_authority"
    UNIQUE_LEASE = "unique_lease"
    FENCING = "fencing"
    REQUIRED_EVIDENCE = "required_evidence"
    LEGAL_TRANSITION = "legal_transition"
    EVENTUAL_TERMINAL = "eventual_terminal"
    FORBIDDEN_MERGE_STATE = "forbidden_merge_state"
    DEONTIC_CONSISTENCY = "deontic_consistency"
    TEMPORAL_FORMULA = "temporal_formula"
    FORMULA_SUPPORT = "formula_support"
    RESOURCE_BOUND = "resource_bound"


class FindingDisposition(str, Enum):
    CONTRADICTION = "contradiction"
    COUNTERMODEL = "countermodel"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class PlanFindingCode(str, Enum):
    DEPENDENCY_NOT_READY = "dependency_not_ready"
    ACTOR_NOT_ASSIGNED = "actor_not_assigned"
    ACTOR_NOT_AUTHORIZED = "actor_not_authorized"
    ACTOR_CAPABILITY_MISSING = "actor_capability_missing"
    CONFLICTING_NORMS = "conflicting_norms"
    OBLIGATION_BEARER_MISMATCH = "obligation_bearer_mismatch"
    MULTIPLE_ACTIVE_LEASES = "multiple_active_leases"
    LEASE_REUSED = "lease_reused"
    MISSING_FENCING_TOKEN = "missing_fencing_token"
    INVALID_FENCING_TOKEN = "invalid_fencing_token"
    STALE_FENCING_TOKEN = "stale_fencing_token"
    FENCING_TOKEN_CHANGED = "fencing_token_changed"
    REQUIRED_EVIDENCE_UNAVAILABLE = "required_evidence_unavailable"
    ILLEGAL_TRANSITION = "illegal_transition"
    TERMINAL_STATE_NOT_ALLOWED = "terminal_state_not_allowed"
    EVENT_AFTER_TERMINAL = "event_after_terminal"
    TERMINAL_OUTCOME_MISSING = "terminal_outcome_missing"
    FORBIDDEN_MERGE_STATE = "forbidden_merge_state"
    TEMPORAL_CONSTRAINT_FALSE = "temporal_constraint_false"
    PRECONDITION_FALSE = "precondition_false"
    GOAL_NOT_SATISFIED = "goal_not_satisfied"
    MISSING_FORMULA = "missing_formula"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_PROFILE = "unsupported_profile"
    FORMULA_DEPTH_EXCEEDED = "formula_depth_exceeded"
    TRACE_BOUND_TRUNCATED = "trace_bound_truncated"
    SEARCH_BOUND_EXHAUSTED = "search_bound_exhausted"
    DOMAIN_BOUND_EXCEEDED = "domain_bound_exceeded"


@dataclass(frozen=True)
class ValidationBounds:
    """All finite resource and semantic bounds used by the reference checker."""

    max_trace_steps: int = 256
    max_actors: int = 128
    max_goals: int = 128
    max_tasks: int = 512
    max_events: int = 4096
    max_fluents: int = 4096
    max_norms: int = 4096
    max_evidence_requirements: int = 4096
    max_temporal_constraints: int = 4096
    max_formulas: int = 8192
    max_formula_records: int = 16_384
    max_provider_evidence: int = 256
    max_formula_depth: int = 64
    max_search_nodes: int = 100_000
    max_countermodel_steps: int = 64
    timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def bounds_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_VALIDATION_BOUNDS_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationBounds":
        return cls(
            **{
                name: payload[name]
                for name in cls.__dataclass_fields__
                if name in payload
            }
        )


FormalPlanValidationBounds = ValidationBounds


@dataclass(frozen=True)
class AppliedValidationBounds:
    """Requested limits plus the exact finite domains explored."""

    configured: ValidationBounds
    plan_trace_bound: int
    effective_trace_bound: int
    domain_sizes: Mapping[str, int]
    search_nodes_explored: int
    truncated_dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sizes = {
            str(key): int(value) for key, value in sorted(self.domain_sizes.items())
        }
        if any(value < 0 for value in sizes.values()):
            raise ValueError("domain sizes must be non-negative")
        object.__setattr__(self, "domain_sizes", sizes)
        object.__setattr__(
            self, "truncated_dimensions", tuple(sorted(set(self.truncated_dimensions)))
        )

    @property
    def bounds_id(self) -> str:
        # Identity of the checked obligation, not incidental execution
        # accounting. Provider-evidence count and explored-node usage remain
        # in ``to_dict`` but do not prevent a receipt from binding the exact
        # obligation it was asked to reconstruct.
        return content_identity(
            {
                "schema": FORMAL_PLAN_VALIDATION_BOUNDS_SCHEMA,
                "configured": self.configured.to_dict(),
                "plan_trace_bound": self.plan_trace_bound,
                "effective_trace_bound": self.effective_trace_bound,
                "domain_sizes": {
                    key: value
                    for key, value in self.domain_sizes.items()
                    if key != "provider_evidence"
                },
                "truncated_dimensions": list(self.truncated_dimensions),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_VALIDATION_BOUNDS_SCHEMA,
            "configured": self.configured.to_dict(),
            "plan_trace_bound": self.plan_trace_bound,
            "effective_trace_bound": self.effective_trace_bound,
            "domain_sizes": dict(self.domain_sizes),
            "search_nodes_explored": self.search_nodes_explored,
            "truncated_dimensions": list(self.truncated_dimensions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AppliedValidationBounds":
        configured = payload.get("configured") or {}
        return cls(
            configured=(
                configured
                if isinstance(configured, ValidationBounds)
                else ValidationBounds.from_dict(configured)
            ),
            plan_trace_bound=payload.get("plan_trace_bound", 0),
            effective_trace_bound=payload.get("effective_trace_bound", 0),
            domain_sizes=payload.get("domain_sizes") or {},
            search_nodes_explored=payload.get("search_nodes_explored", 0),
            truncated_dimensions=tuple(payload.get("truncated_dimensions") or ()),
        )


@dataclass(frozen=True)
class PlanValidationAssumption:
    assumption_id: str
    kind: str
    statement: str
    subject_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("assumption_id", "kind", "statement"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "subject_ids",
            tuple(
                sorted(
                    {
                        str(item).strip()
                        for item in self.subject_ids
                        if str(item).strip()
                    }
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "kind": self.kind,
            "statement": self.statement,
            "subject_ids": list(self.subject_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanValidationAssumption":
        return cls(
            assumption_id=payload.get("assumption_id", ""),
            kind=payload.get("kind", ""),
            statement=payload.get("statement", ""),
            subject_ids=tuple(payload.get("subject_ids") or ()),
        )


PlanAssumption = PlanValidationAssumption


def _assumption(
    kind: str, statement: str, *subject_ids: str
) -> PlanValidationAssumption:
    subjects = tuple(sorted({item for item in subject_ids if item}))
    return PlanValidationAssumption(
        assumption_id=content_identity(
            {"kind": kind, "statement": statement, "subject_ids": subjects}
        ),
        kind=kind,
        statement=statement,
        subject_ids=subjects,
    )


@dataclass(frozen=True)
class PlanValidationFinding:
    code: PlanFindingCode
    disposition: FindingDisposition
    check: PlanCheckKind
    message: str
    subject_ids: tuple[str, ...] = ()
    logical_time: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", PlanFindingCode(self.code))
        object.__setattr__(self, "disposition", FindingDisposition(self.disposition))
        object.__setattr__(self, "check", PlanCheckKind(self.check))
        message = str(self.message or "").strip()
        if not message:
            raise ValueError("finding message is required")
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "subject_ids",
            tuple(
                sorted(
                    {
                        str(item).strip()
                        for item in self.subject_ids
                        if str(item).strip()
                    }
                )
            ),
        )
        if self.logical_time is not None and (
            isinstance(self.logical_time, bool)
            or not isinstance(self.logical_time, int)
            or self.logical_time < 0
        ):
            raise ValueError("logical_time must be a non-negative integer or null")
        # Strict canonical JSON validation and defensive copying.
        object.__setattr__(self, "details", _canonical_mapping(self.details))

    @property
    def finding_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def issue_id(self) -> str:
        return self.finding_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_FINDING_SCHEMA,
            "code": self.code.value,
            "disposition": self.disposition.value,
            "check": self.check.value,
            "message": self.message,
            "subject_ids": list(self.subject_ids),
            "logical_time": self.logical_time,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanValidationFinding":
        result = cls(
            code=payload.get("code", PlanFindingCode.UNSUPPORTED_OPERATOR),
            disposition=payload.get("disposition", FindingDisposition.UNSUPPORTED),
            check=payload.get("check", PlanCheckKind.FORMULA_SUPPORT),
            message=payload.get("message", ""),
            subject_ids=tuple(payload.get("subject_ids") or ()),
            logical_time=payload.get("logical_time"),
            details=payload.get("details") or {},
        )
        claimed = payload.get("finding_id") or payload.get("issue_id")
        if claimed and claimed != result.finding_id:
            raise ValueError("formal-plan finding identity does not match payload")
        return result


ValidationFinding = PlanValidationFinding


@dataclass(frozen=True)
class CountermodelState:
    logical_time: int
    event_ids: tuple[str, ...] = ()
    task_states: Mapping[str, str] = field(default_factory=dict)
    available_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_time, bool)
            or not isinstance(self.logical_time, int)
            or self.logical_time < 0
        ):
            raise ValueError("logical_time must be a non-negative integer")
        object.__setattr__(self, "event_ids", tuple(sorted(set(self.event_ids))))
        object.__setattr__(
            self,
            "task_states",
            {str(key): str(value) for key, value in sorted(self.task_states.items())},
        )
        object.__setattr__(
            self,
            "available_evidence_ids",
            tuple(sorted(set(self.available_evidence_ids))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_time": self.logical_time,
            "event_ids": list(self.event_ids),
            "task_states": dict(self.task_states),
            "available_evidence_ids": list(self.available_evidence_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CountermodelState":
        return cls(
            logical_time=payload.get("logical_time", -1),
            event_ids=tuple(payload.get("event_ids") or ()),
            task_states=payload.get("task_states") or {},
            available_evidence_ids=tuple(payload.get("available_evidence_ids") or ()),
        )


@dataclass(frozen=True)
class PlanCountermodel:
    violation_finding_id: str
    states: tuple[CountermodelState, ...]
    complete_prefix: bool = True

    def __post_init__(self) -> None:
        finding_id = str(self.violation_finding_id or "").strip()
        if not finding_id:
            raise ValueError("violation_finding_id is required")
        object.__setattr__(self, "violation_finding_id", finding_id)
        states = tuple(
            item
            if isinstance(item, CountermodelState)
            else CountermodelState.from_dict(item)
            for item in self.states
        )
        if tuple(item.logical_time for item in states) != tuple(
            sorted(item.logical_time for item in states)
        ):
            raise ValueError("countermodel states must be in logical-time order")
        object.__setattr__(self, "states", states)
        if not isinstance(self.complete_prefix, bool):
            raise ValueError("complete_prefix must be a boolean")

    @property
    def countermodel_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_COUNTERMODEL_SCHEMA,
            "violation_finding_id": self.violation_finding_id,
            "states": [item.to_dict() for item in self.states],
            "complete_prefix": self.complete_prefix,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCountermodel":
        result = cls(
            violation_finding_id=str(payload.get("violation_finding_id") or ""),
            states=tuple(
                item
                if isinstance(item, CountermodelState)
                else CountermodelState.from_dict(item)
                for item in (payload.get("states") or ())
            ),
            complete_prefix=payload.get("complete_prefix", True),
        )
        claimed = payload.get("countermodel_id")
        if claimed and claimed != result.countermodel_id:
            raise ValueError("countermodel identity does not match payload")
        return result


@dataclass(frozen=True)
class PlanCheckEvidence:
    """Evidence supplied by an optional native prover/kernel/model checker."""

    evidence_id: str
    backend_id: str
    backend_role: str = "native"
    outcome: str = "success"
    plan_id: str = ""
    formula_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    bounds_id: str = ""
    reconstructed: bool = False
    accepted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "backend_id", "backend_role", "outcome"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "plan_id", str(self.plan_id or "").strip())
        object.__setattr__(self, "bounds_id", str(self.bounds_id or "").strip())
        object.__setattr__(self, "formula_ids", tuple(sorted(set(self.formula_ids))))
        object.__setattr__(
            self, "assumption_ids", tuple(sorted(set(self.assumption_ids)))
        )
        if not isinstance(self.reconstructed, bool):
            raise ValueError("reconstructed must be a boolean")
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata))

    @property
    def native_plan_check_only(self) -> bool:
        role = self.backend_role.lower()
        return role in {"native", "dcec", "tdfol", "domain_reasoner"}

    def exactly_binds(
        self,
        *,
        plan_id: str,
        formula_ids: tuple[str, ...],
        assumption_ids: tuple[str, ...],
        bounds_id: str,
    ) -> bool:
        return (
            self.accepted
            and self.outcome.lower() in {"success", "proved", "verified", "satisfied"}
            and self.plan_id == plan_id
            and self.formula_ids == formula_ids
            and self.assumption_ids == assumption_ids
            and self.bounds_id == bounds_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "backend_id": self.backend_id,
            "backend_role": self.backend_role,
            "outcome": self.outcome,
            "plan_id": self.plan_id,
            "formula_ids": list(self.formula_ids),
            "assumption_ids": list(self.assumption_ids),
            "bounds_id": self.bounds_id,
            "reconstructed": self.reconstructed,
            "accepted": self.accepted,
            "native_plan_check_only": self.native_plan_check_only,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCheckEvidence":
        backend_id = str(
            payload.get("backend_id")
            or payload.get("provider_id")
            or payload.get("prover_id")
            or "unknown"
        )
        return cls(
            evidence_id=str(
                payload.get("evidence_id")
                or payload.get("receipt_id")
                or content_identity(dict(payload))
            ),
            backend_id=backend_id,
            backend_role=str(
                payload.get("backend_role") or payload.get("role") or "native"
            ),
            outcome=str(payload.get("outcome") or payload.get("status") or "success"),
            plan_id=str(payload.get("plan_id") or ""),
            formula_ids=tuple(payload.get("formula_ids") or ()),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            bounds_id=str(payload.get("bounds_id") or ""),
            reconstructed=(
                payload.get("reconstructed")
                if "reconstructed" in payload
                else payload.get("kernel_reconstructed", False)
            ),
            accepted=payload.get("accepted", False),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class PlanValidationResult:
    status: PlanValidationStatus
    outcome: PlanValidationOutcome
    plan_id: str
    bounds: AppliedValidationBounds
    assumptions: tuple[PlanValidationAssumption, ...]
    formula_ids: tuple[str, ...]
    findings: tuple[PlanValidationFinding, ...] = ()
    countermodel: PlanCountermodel | None = None
    evidence: tuple[PlanCheckEvidence, ...] = ()
    consistency_level: PlanConsistencyLevel = PlanConsistencyLevel.INCONCLUSIVE
    checks_performed: tuple[PlanCheckKind, ...] = ()
    validator_version: int = FORMAL_PLAN_VALIDATOR_VERSION

    def __post_init__(self) -> None:
        plan_id = str(self.plan_id or "").strip()
        if not plan_id:
            raise ValueError("plan_id is required")
        object.__setattr__(self, "plan_id", plan_id)
        if (
            isinstance(self.validator_version, bool)
            or not isinstance(self.validator_version, int)
            or self.validator_version <= 0
        ):
            raise ValueError("validator_version must be a positive integer")
        object.__setattr__(self, "status", PlanValidationStatus(self.status))
        object.__setattr__(self, "outcome", PlanValidationOutcome(self.outcome))
        object.__setattr__(
            self, "consistency_level", PlanConsistencyLevel(self.consistency_level)
        )
        object.__setattr__(
            self,
            "assumptions",
            tuple(sorted(self.assumptions, key=lambda item: item.assumption_id)),
        )
        object.__setattr__(self, "formula_ids", tuple(sorted(set(self.formula_ids))))
        object.__setattr__(
            self,
            "findings",
            tuple(
                sorted(
                    self.findings,
                    key=lambda item: (
                        -1 if item.logical_time is None else item.logical_time,
                        item.code.value,
                        item.finding_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "evidence",
            tuple(sorted(self.evidence, key=lambda item: item.evidence_id)),
        )
        object.__setattr__(
            self,
            "checks_performed",
            tuple(
                sorted(
                    {PlanCheckKind(item) for item in self.checks_performed},
                    key=lambda item: item.value,
                )
            ),
        )
        if (
            self.outcome is PlanValidationOutcome.COUNTERMODEL
            and self.countermodel is None
        ):
            raise ValueError("countermodel outcomes require a countermodel")
        if self.status is PlanValidationStatus.CONSISTENT and self.findings:
            raise ValueError("consistent results cannot contain findings")
        if (self.status is PlanValidationStatus.CONSISTENT) != (
            self.outcome is PlanValidationOutcome.CONSISTENT
        ):
            raise ValueError("consistent status and outcome must agree")

    @property
    def validation_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def result_id(self) -> str:
        return self.validation_id

    @property
    def verdict(self) -> PlanValidationStatus:
        return self.status

    @property
    def consistent(self) -> bool:
        return self.status is PlanValidationStatus.CONSISTENT

    @property
    def is_consistent(self) -> bool:
        return self.consistent

    @property
    def succeeded(self) -> bool:
        return self.consistent

    @property
    def valid(self) -> bool:
        return self.consistent

    @property
    def issues(self) -> tuple[PlanValidationFinding, ...]:
        return self.findings

    @property
    def counterexample(self) -> PlanCountermodel | None:
        return self.countermodel

    @property
    def assurance(self) -> PlanConsistencyLevel:
        return self.consistency_level

    @property
    def finite_bounds(self) -> Mapping[str, Any]:
        return self.bounds.to_dict()

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        return tuple(item.assumption_id for item in self.assumptions)

    @property
    def plan_check_only(self) -> bool:
        return self.consistency_level is not PlanConsistencyLevel.KERNEL_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_VALIDATION_SCHEMA,
            "validator_version": self.validator_version,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "plan_id": self.plan_id,
            "bounds": self.bounds.to_dict(),
            "assumptions": [item.to_dict() for item in self.assumptions],
            "formula_ids": list(self.formula_ids),
            "findings": [
                {**item.to_dict(), "finding_id": item.finding_id}
                for item in self.findings
            ],
            "countermodel": (
                self.countermodel.to_dict() if self.countermodel is not None else None
            ),
            "evidence": [item.to_dict() for item in self.evidence],
            "consistency_level": self.consistency_level.value,
            "plan_check_only": self.plan_check_only,
            "checks_performed": [item.value for item in self.checks_performed],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "validation_id": self.validation_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanValidationResult":
        bounds = payload.get("bounds") or {}
        countermodel = payload.get("countermodel")
        result = cls(
            status=payload.get("status", PlanValidationStatus.ERROR),
            outcome=payload.get("outcome", PlanValidationOutcome.ERROR),
            plan_id=str(payload.get("plan_id") or ""),
            bounds=(
                bounds
                if isinstance(bounds, AppliedValidationBounds)
                else AppliedValidationBounds.from_dict(bounds)
            ),
            assumptions=tuple(
                item
                if isinstance(item, PlanValidationAssumption)
                else PlanValidationAssumption.from_dict(item)
                for item in (payload.get("assumptions") or ())
            ),
            formula_ids=tuple(payload.get("formula_ids") or ()),
            findings=tuple(
                item
                if isinstance(item, PlanValidationFinding)
                else PlanValidationFinding.from_dict(item)
                for item in (payload.get("findings") or ())
            ),
            countermodel=(
                countermodel
                if isinstance(countermodel, PlanCountermodel)
                else (
                    PlanCountermodel.from_dict(countermodel)
                    if countermodel is not None
                    else None
                )
            ),
            evidence=tuple(
                item
                if isinstance(item, PlanCheckEvidence)
                else PlanCheckEvidence.from_dict(item)
                for item in (payload.get("evidence") or ())
            ),
            consistency_level=payload.get(
                "consistency_level", PlanConsistencyLevel.INCONCLUSIVE
            ),
            checks_performed=tuple(payload.get("checks_performed") or ()),
            validator_version=payload.get(
                "validator_version", FORMAL_PLAN_VALIDATOR_VERSION
            ),
        )
        claimed = payload.get("validation_id") or payload.get("result_id")
        if claimed and claimed != result.validation_id:
            raise ValueError("formal-plan validation identity does not match payload")
        return result

    @classmethod
    def from_json(cls, payload: str) -> "PlanValidationResult":
        import json

        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("formal-plan validation JSON must contain an object")
        return cls.from_dict(decoded)


FormalPlanValidationResult = PlanValidationResult


class _Cancelled(Exception):
    pass


class _TimedOut(Exception):
    pass


class _SearchExhausted(Exception):
    pass


@dataclass
class _TraceModel:
    task_states: list[dict[str, str]]
    event_ids: list[tuple[str, ...]]
    available_evidence: list[set[str]]
    completed_at: dict[str, int]
    terminal_at: dict[str, int]
    facts: FiniteTrace
    assumptions: list[PlanValidationAssumption]


class _BudgetGuard:
    def __init__(
        self,
        bounds: ValidationBounds,
        cancellation_token: CancellationToken | Any | None,
        clock: Callable[[], float],
    ) -> None:
        self.bounds = bounds
        self.cancellation_token = cancellation_token
        self.clock = clock
        self.started = clock()
        self.nodes = 0

    def checkpoint(self, amount: int = 1) -> None:
        self.nodes += amount
        token = self.cancellation_token
        if token is not None:
            cancelled = getattr(token, "cancelled", False)
            if callable(cancelled):
                cancelled = cancelled()
            if not cancelled:
                is_cancelled = getattr(token, "is_cancelled", None)
                cancelled = bool(is_cancelled()) if callable(is_cancelled) else False
            if cancelled:
                raise _Cancelled
        if self.bounds.timeout_ms == 0:
            raise _TimedOut
        if (self.clock() - self.started) * 1000 >= self.bounds.timeout_ms:
            raise _TimedOut
        if self.nodes > self.bounds.max_search_nodes:
            raise _SearchExhausted


_SUPPORTED_PROFILES: Final = {
    ("supervisor-reviewed", LOGIC_VOCABULARY_VERSION),
    ("supervisor-dcec", DCEC_VOCABULARY_VERSION),
    ("supervisor-tdfol", TDFOL_VOCABULARY_VERSION),
}
_SUPPORTED_OPERATORS: Final = {
    FormulaOperator.ATOM,
    FormulaOperator.NOT,
    FormulaOperator.AND,
    FormulaOperator.OR,
    FormulaOperator.IMPLIES,
    FormulaOperator.IFF,
    FormulaOperator.OBLIGATION,
    FormulaOperator.PERMISSION,
    FormulaOperator.PROHIBITION,
    FormulaOperator.DELEGATION,
    FormulaOperator.EXECUTION_EVENT,
    FormulaOperator.DEPENDENCY_ORDER,
    FormulaOperator.DEADLINE,
    FormulaOperator.LIVENESS,
    FormulaOperator.SAFETY,
    FormulaOperator.GOAL_SATISFACTION,
}
_TERMINAL_EVENT_STATES: Final = {
    EventKind.COMPLETED: "completed",
    EventKind.FAILED: "failed",
    EventKind.CANCELLED: "cancelled",
}
_DEFAULT_FORBIDDEN_MERGE_STATES: Final = {
    "forbidden_merge",
    "merge_conflict",
    "merged_with_conflicts",
    "merge_without_evidence",
    "merged_after_failure",
    "merged_cancelled",
}


class FormalPlanValidator:
    """Bounded reference validator for the reviewed planning profiles."""

    def __init__(
        self,
        bounds: ValidationBounds | Mapping[str, Any] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if bounds is None:
            bounds = ValidationBounds()
        elif isinstance(bounds, Mapping):
            bounds = ValidationBounds.from_dict(bounds)
        if not isinstance(bounds, ValidationBounds):
            raise TypeError("bounds must be ValidationBounds, an object, or null")
        self.bounds = bounds
        self._clock = clock

    def validate(
        self,
        plan: FormalWorkPlan | Mapping[str, Any],
        formulas: Iterable[Formula | Mapping[str, Any]] | None = None,
        *,
        cancellation_token: CancellationToken | Any | None = None,
        proof_evidence: Iterable[PlanCheckEvidence | Mapping[str, Any]] = (),
        native_evidence: Iterable[PlanCheckEvidence | Mapping[str, Any]] = (),
    ) -> PlanValidationResult:
        """Validate one plan without mutating it or invoking optional providers."""

        if isinstance(plan, Mapping):
            plan = FormalWorkPlan.from_dict(plan)
        if not isinstance(plan, FormalWorkPlan):
            raise TypeError("plan must be a FormalWorkPlan or canonical plan object")

        guard = _BudgetGuard(self.bounds, cancellation_token, self._clock)
        effective_bound = min(plan.trace_bound, self.bounds.max_trace_steps)
        truncated: list[str] = []
        if effective_bound < plan.trace_bound:
            truncated.append("trace_steps")
        domain_sizes = {
            "actors": len(plan.actors),
            "goals": len(plan.goals),
            "tasks": len(plan.tasks),
            "events": len(plan.events),
            "fluents": len(plan.fluents),
            "norms": len(plan.norms),
            "evidence_requirements": len(plan.evidence_requirements),
            "temporal_constraints": len(plan.temporal_constraints),
            "formulas": 0,
            "provider_evidence": 0,
        }
        assumptions = self._base_assumptions(plan, effective_bound)
        checks: set[PlanCheckKind] = {PlanCheckKind.FORMULA_SUPPORT}
        formula_map: dict[str, Formula] = {}
        evidence: tuple[PlanCheckEvidence, ...] = ()
        try:
            guard.checkpoint(0)
            formula_map, formula_findings = self._load_formulas(plan, formulas, guard)
            evidence, evidence_exceeded = _evidence_records(
                chain(proof_evidence, native_evidence),
                guard=guard,
                limit=self.bounds.max_provider_evidence,
            )
            domain_sizes["provider_evidence"] = (
                self.bounds.max_provider_evidence + 1
                if evidence_exceeded
                else len(evidence)
            )
        except _Cancelled:
            return self._result(
                plan,
                formula_map,
                evidence,
                assumptions,
                (),
                checks,
                guard,
                effective_bound,
                domain_sizes,
                truncated,
                PlanValidationStatus.CANCELLED,
                PlanValidationOutcome.CANCELLED,
            )
        except _TimedOut:
            return self._result(
                plan,
                formula_map,
                evidence,
                assumptions,
                (),
                checks,
                guard,
                effective_bound,
                domain_sizes,
                truncated,
                PlanValidationStatus.TIMED_OUT,
                PlanValidationOutcome.TIMEOUT,
            )
        except _SearchExhausted:
            exhausted = PlanValidationFinding(
                PlanFindingCode.SEARCH_BOUND_EXHAUSTED,
                FindingDisposition.INCOMPLETE,
                PlanCheckKind.RESOURCE_BOUND,
                "deterministic search-node bound was exhausted while loading formulas or evidence",
                details={"max_search_nodes": self.bounds.max_search_nodes},
            )
            return self._result(
                plan,
                formula_map,
                evidence,
                assumptions,
                (exhausted,),
                checks,
                guard,
                effective_bound,
                domain_sizes,
                truncated,
                PlanValidationStatus.INCOMPLETE,
                PlanValidationOutcome.INCOMPLETE_SEARCH,
            )
        if (
            plan.vocabulary_profile_id != "supervisor-reviewed"
            or plan.vocabulary_version != LOGIC_VOCABULARY_VERSION
        ):
            formula_findings.append(
                PlanValidationFinding(
                    PlanFindingCode.UNSUPPORTED_PROFILE,
                    FindingDisposition.UNSUPPORTED,
                    PlanCheckKind.FORMULA_SUPPORT,
                    f"unsupported plan vocabulary profile {plan.vocabulary_profile_id}@{plan.vocabulary_version}",
                    (plan.plan_id,),
                )
            )
        domain_sizes["formulas"] = len(formula_map)
        if not domain_sizes["provider_evidence"]:
            domain_sizes["provider_evidence"] = len(evidence)
        findings: list[PlanValidationFinding] = list(formula_findings)
        trace: _TraceModel | None = None

        try:
            guard.checkpoint(0)
            over = self._domain_overflows(domain_sizes)
            if over:
                findings.extend(
                    PlanValidationFinding(
                        PlanFindingCode.DOMAIN_BOUND_EXCEEDED,
                        FindingDisposition.RESOURCE_EXHAUSTED,
                        PlanCheckKind.RESOURCE_BOUND,
                        f"{name} domain size {actual} exceeds configured limit {limit}",
                        details={"dimension": name, "actual": actual, "limit": limit},
                    )
                    for name, actual, limit in over
                )
                return self._result(
                    plan,
                    formula_map,
                    evidence,
                    assumptions,
                    findings,
                    checks,
                    guard,
                    effective_bound,
                    domain_sizes,
                    truncated,
                    PlanValidationStatus.RESOURCE_EXHAUSTED,
                    PlanValidationOutcome.RESOURCE_EXHAUSTED,
                )
            if findings:
                resource_finding = any(
                    item.disposition is FindingDisposition.RESOURCE_EXHAUSTED
                    for item in findings
                )
                return self._result(
                    plan,
                    formula_map,
                    evidence,
                    assumptions,
                    findings,
                    checks,
                    guard,
                    effective_bound,
                    domain_sizes,
                    truncated,
                    (
                        PlanValidationStatus.RESOURCE_EXHAUSTED
                        if resource_finding
                        else PlanValidationStatus.UNSUPPORTED
                    ),
                    (
                        PlanValidationOutcome.RESOURCE_EXHAUSTED
                        if resource_finding
                        else PlanValidationOutcome.UNSUPPORTED_OPERATOR
                    ),
                )

            static_findings = self._static_checks(plan, formula_map, guard)
            checks.update(
                {
                    PlanCheckKind.DEONTIC_CONSISTENCY,
                    PlanCheckKind.UNIQUE_LEASE,
                    PlanCheckKind.FENCING,
                    PlanCheckKind.ACTOR_AUTHORITY,
                }
            )
            if static_findings:
                findings.extend(static_findings)
                return self._result(
                    plan,
                    formula_map,
                    evidence,
                    assumptions,
                    findings,
                    checks,
                    guard,
                    effective_bound,
                    domain_sizes,
                    truncated,
                    PlanValidationStatus.INCONSISTENT,
                    PlanValidationOutcome.CONTRADICTION,
                )

            trace, dynamic_findings = self._build_and_check_trace(
                plan, formula_map, effective_bound, guard
            )
            assumptions.extend(trace.assumptions)
            findings.extend(dynamic_findings)
            checks.update(
                {
                    PlanCheckKind.DEPENDENCY_READINESS,
                    PlanCheckKind.ACTOR_AUTHORITY,
                    PlanCheckKind.FENCING,
                    PlanCheckKind.REQUIRED_EVIDENCE,
                    PlanCheckKind.LEGAL_TRANSITION,
                    PlanCheckKind.EVENTUAL_TERMINAL,
                    PlanCheckKind.FORBIDDEN_MERGE_STATE,
                }
            )
            if findings:
                countermodel = self._countermodel(findings[0], trace)
                return self._result(
                    plan,
                    formula_map,
                    evidence,
                    assumptions,
                    findings,
                    checks,
                    guard,
                    effective_bound,
                    domain_sizes,
                    truncated,
                    PlanValidationStatus.VIOLATED,
                    PlanValidationOutcome.COUNTERMODEL,
                    countermodel=countermodel,
                )

            if truncated:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.TRACE_BOUND_TRUNCATED,
                        FindingDisposition.INCOMPLETE,
                        PlanCheckKind.RESOURCE_BOUND,
                        "configured trace bound truncates the plan horizon",
                        details={
                            "plan_trace_bound": plan.trace_bound,
                            "effective_trace_bound": effective_bound,
                        },
                    )
                )
                return self._result(
                    plan,
                    formula_map,
                    evidence,
                    assumptions,
                    findings,
                    checks,
                    guard,
                    effective_bound,
                    domain_sizes,
                    truncated,
                    PlanValidationStatus.INCOMPLETE,
                    PlanValidationOutcome.INCOMPLETE_SEARCH,
                )

            formula_violations = self._check_formulas(plan, formula_map, trace, guard)
            checks.add(PlanCheckKind.TEMPORAL_FORMULA)
            if formula_violations:
                findings.extend(formula_violations)
                countermodel = self._countermodel(findings[0], trace)
                return self._result(
                    plan,
                    formula_map,
                    evidence,
                    assumptions,
                    findings,
                    checks,
                    guard,
                    effective_bound,
                    domain_sizes,
                    truncated,
                    PlanValidationStatus.VIOLATED,
                    PlanValidationOutcome.COUNTERMODEL,
                    countermodel=countermodel,
                )

            return self._result(
                plan,
                formula_map,
                evidence,
                assumptions,
                (),
                checks,
                guard,
                effective_bound,
                domain_sizes,
                truncated,
                PlanValidationStatus.CONSISTENT,
                PlanValidationOutcome.CONSISTENT,
            )
        except _Cancelled:
            return self._result(
                plan,
                formula_map,
                evidence,
                assumptions,
                (),
                checks,
                guard,
                effective_bound,
                domain_sizes,
                truncated,
                PlanValidationStatus.CANCELLED,
                PlanValidationOutcome.CANCELLED,
            )
        except _TimedOut:
            return self._result(
                plan,
                formula_map,
                evidence,
                assumptions,
                (),
                checks,
                guard,
                effective_bound,
                domain_sizes,
                truncated,
                PlanValidationStatus.TIMED_OUT,
                PlanValidationOutcome.TIMEOUT,
            )
        except _SearchExhausted:
            findings.append(
                PlanValidationFinding(
                    PlanFindingCode.SEARCH_BOUND_EXHAUSTED,
                    FindingDisposition.INCOMPLETE,
                    PlanCheckKind.RESOURCE_BOUND,
                    "deterministic search-node bound was exhausted",
                    details={"max_search_nodes": self.bounds.max_search_nodes},
                )
            )
            return self._result(
                plan,
                formula_map,
                evidence,
                assumptions,
                findings,
                checks,
                guard,
                effective_bound,
                domain_sizes,
                truncated,
                PlanValidationStatus.INCOMPLETE,
                PlanValidationOutcome.INCOMPLETE_SEARCH,
            )

    def check(
        self,
        plan: FormalWorkPlan | Mapping[str, Any],
        formulas: Iterable[Formula | Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> PlanValidationResult:
        """Compatibility spelling for :meth:`validate`."""

        return self.validate(plan, formulas, **kwargs)

    def _load_formulas(
        self,
        plan: FormalWorkPlan,
        supplied: Iterable[Formula | Mapping[str, Any]] | None,
        guard: _BudgetGuard,
    ) -> tuple[dict[str, Formula], list[PlanValidationFinding]]:
        records = plan.metadata.get("formula_records", ())
        metadata_values: Iterable[Formula | Mapping[str, Any]] = ()
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            metadata_values = records
        values = chain(metadata_values, supplied or ())
        formulas: dict[str, Formula] = {}
        findings: list[PlanValidationFinding] = []
        for index, value in enumerate(values):
            guard.checkpoint()
            if index >= self.bounds.max_formula_records:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.DOMAIN_BOUND_EXCEEDED,
                        FindingDisposition.RESOURCE_EXHAUSTED,
                        PlanCheckKind.RESOURCE_BOUND,
                        "formula record count exceeds the configured finite bound",
                        details={
                            "actual_at_least": index + 1,
                            "limit": self.bounds.max_formula_records,
                        },
                    )
                )
                break
            raw_depth = _raw_formula_depth(value, guard)
            if raw_depth > self.bounds.max_formula_depth:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.FORMULA_DEPTH_EXCEEDED,
                        FindingDisposition.RESOURCE_EXHAUSTED,
                        PlanCheckKind.RESOURCE_BOUND,
                        f"formula depth exceeds {self.bounds.max_formula_depth}",
                        details={
                            "actual_at_least": raw_depth,
                            "limit": self.bounds.max_formula_depth,
                            "record_index": index,
                        },
                    )
                )
                continue
            try:
                formula = (
                    value if isinstance(value, Formula) else Formula.from_dict(value)
                )
            except (TypeError, ValueError) as exc:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.UNSUPPORTED_OPERATOR,
                        FindingDisposition.UNSUPPORTED,
                        PlanCheckKind.FORMULA_SUPPORT,
                        f"formula record {index} is unsupported: {exc}",
                        details={"record_index": index},
                    )
                )
                continue
            formulas[formula.formula_id] = formula

        referenced = {
            *(goal.satisfaction_formula_id for goal in plan.goals),
            *(subgoal.satisfaction_formula_id for subgoal in plan.subgoals),
            *(item.formula_id for item in plan.preconditions),
            *(item.formula_id for item in plan.temporal_constraints),
            *(
                item.activation_formula_id
                for item in plan.norms
                if item.activation_formula_id
            ),
        }
        for formula_id in sorted(referenced):
            if formula_id not in formulas:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.MISSING_FORMULA,
                        FindingDisposition.UNSUPPORTED,
                        PlanCheckKind.FORMULA_SUPPORT,
                        f"referenced formula {formula_id} is unavailable",
                        (formula_id,),
                    )
                )
        for formula in formulas.values():
            if (formula.profile_id, formula.profile_version) not in _SUPPORTED_PROFILES:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.UNSUPPORTED_PROFILE,
                        FindingDisposition.UNSUPPORTED,
                        PlanCheckKind.FORMULA_SUPPORT,
                        f"unsupported formula profile {formula.profile_id}@{formula.profile_version}",
                        (formula.formula_id,),
                    )
                )
            unsupported = _unsupported_operator(formula)
            if unsupported is not None:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.UNSUPPORTED_OPERATOR,
                        FindingDisposition.UNSUPPORTED,
                        PlanCheckKind.FORMULA_SUPPORT,
                        f"operator {unsupported.value} has no accepted bounded semantics",
                        (formula.formula_id,),
                        details={"operator": unsupported.value},
                    )
                )
            depth = _formula_depth(formula)
            if depth > self.bounds.max_formula_depth:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.FORMULA_DEPTH_EXCEEDED,
                        FindingDisposition.RESOURCE_EXHAUSTED,
                        PlanCheckKind.RESOURCE_BOUND,
                        f"formula depth {depth} exceeds {self.bounds.max_formula_depth}",
                        (formula.formula_id,),
                        details={
                            "actual": depth,
                            "limit": self.bounds.max_formula_depth,
                        },
                    )
                )
        return formulas, findings

    def _domain_overflows(self, sizes: Mapping[str, int]) -> list[tuple[str, int, int]]:
        limits = {
            "actors": self.bounds.max_actors,
            "goals": self.bounds.max_goals,
            "tasks": self.bounds.max_tasks,
            "events": self.bounds.max_events,
            "fluents": self.bounds.max_fluents,
            "norms": self.bounds.max_norms,
            "evidence_requirements": self.bounds.max_evidence_requirements,
            "temporal_constraints": self.bounds.max_temporal_constraints,
            "formulas": self.bounds.max_formulas,
            "provider_evidence": self.bounds.max_provider_evidence,
        }
        return [
            (name, sizes[name], limit)
            for name, limit in sorted(limits.items())
            if sizes[name] > limit
        ]

    def _base_assumptions(
        self, plan: FormalWorkPlan, effective_bound: int
    ) -> list[PlanValidationAssumption]:
        return [
            _assumption(
                "finite_domain",
                "Only actors, goals, tasks, events, fluents, norms, and evidence requirements declared by the plan are enumerated.",
                plan.plan_id,
            ),
            _assumption(
                "finite_trace",
                f"Logical time is the inclusive integer interval 0..{effective_bound}.",
                plan.plan_id,
            ),
            _assumption(
                "event_completeness",
                "Canonical planned events are the complete event set for this bounded plan check.",
                plan.plan_id,
            ),
            _assumption(
                "inertia",
                "Declared inertial fluent and task-state values persist until a canonical event or effect changes them.",
                plan.plan_id,
            ),
            _assumption(
                "plan_only",
                "DCEC/TDFOL finite success establishes plan-check evidence, not execution, conformance, or generated-code assurance.",
                plan.plan_id,
            ),
        ]

    def _static_checks(
        self,
        plan: FormalWorkPlan,
        formulas: Mapping[str, Formula],
        guard: _BudgetGuard,
    ) -> list[PlanValidationFinding]:
        findings: list[PlanValidationFinding] = []
        tasks = {item.task_id: item for item in plan.tasks}

        grouped_norms: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for norm in plan.norms:
            guard.checkpoint()
            grouped_norms[(norm.bearer_actor_id, norm.action_id)].append(norm)
            issuer = (
                next(
                    item
                    for item in plan.actors
                    if item.actor_id == norm.issuer_actor_id
                )
                if norm.issuer_actor_id
                else None
            )
            if issuer is not None and not (
                issuer.kind in {ActorKind.SUPERVISOR, ActorKind.HUMAN}
                or norm.action_id in issuer.authority_ids
                or "*" in issuer.authority_ids
                or "enforce_policy" in issuer.capabilities
            ):
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.ACTOR_NOT_AUTHORIZED,
                        FindingDisposition.CONTRADICTION,
                        PlanCheckKind.ACTOR_AUTHORITY,
                        "norm issuer lacks authority to impose the action",
                        (norm.norm_id, norm.issuer_actor_id, norm.action_id),
                    )
                )
            task = tasks.get(norm.action_id)
            if (
                task is not None
                and norm.kind is NormKind.OBLIGATION
                and norm.bearer_actor_id not in task.actor_ids
            ):
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.OBLIGATION_BEARER_MISMATCH,
                        FindingDisposition.CONTRADICTION,
                        PlanCheckKind.DEONTIC_CONSISTENCY,
                        "obligation bearer is not an eligible actor for its task",
                        (norm.norm_id, norm.bearer_actor_id, norm.action_id),
                    )
                )
        for (actor_id, action_id), norms in sorted(grouped_norms.items()):
            for left_index, left in enumerate(norms):
                for right in norms[left_index + 1 :]:
                    guard.checkpoint()
                    if not _intervals_overlap(
                        left.valid_from,
                        left.valid_until,
                        right.valid_from,
                        right.valid_until,
                    ):
                        continue
                    kinds = {left.kind, right.kind}
                    if NormKind.PROHIBITION in kinds and (
                        NormKind.OBLIGATION in kinds or NormKind.PERMISSION in kinds
                    ):
                        if (
                            left.activation_formula_id or right.activation_formula_id
                        ) and (
                            left.activation_formula_id != right.activation_formula_id
                        ):
                            # Distinct activation conditions are not assumed
                            # simultaneously true under this finite profile.
                            continue
                        findings.append(
                            PlanValidationFinding(
                                PlanFindingCode.CONFLICTING_NORMS,
                                FindingDisposition.CONTRADICTION,
                                PlanCheckKind.DEONTIC_CONSISTENCY,
                                "overlapping prohibition conflicts with an obligation or permission",
                                (left.norm_id, right.norm_id, actor_id, action_id),
                            )
                        )

        lease_owner: dict[str, tuple[str, str]] = {}
        lease_fence: dict[str, int] = {}
        for event in sorted(
            plan.events, key=lambda item: (item.logical_time, item.event_id)
        ):
            guard.checkpoint()
            lease_ids = _string_values(event.metadata.get("lease_ids"))
            tokens = _fencing_values(event.metadata)
            if len(lease_ids) > 1:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.MULTIPLE_ACTIVE_LEASES,
                        FindingDisposition.CONTRADICTION,
                        PlanCheckKind.UNIQUE_LEASE,
                        "an event declares more than one live lease for a task",
                        (event.event_id, event.task_id, *lease_ids),
                        event.logical_time,
                    )
                )
            for lease_id in lease_ids:
                owner = (event.task_id, event.actor_id)
                prior = lease_owner.setdefault(lease_id, owner)
                if prior != owner:
                    findings.append(
                        PlanValidationFinding(
                            PlanFindingCode.LEASE_REUSED,
                            FindingDisposition.CONTRADICTION,
                            PlanCheckKind.UNIQUE_LEASE,
                            "one live lease is assigned to multiple task/actor owners",
                            (lease_id, event.event_id, event.task_id, event.actor_id),
                            event.logical_time,
                            {"prior_task_id": prior[0], "prior_actor_id": prior[1]},
                        )
                    )
            if lease_ids and not tokens:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.MISSING_FENCING_TOKEN,
                        FindingDisposition.CONTRADICTION,
                        PlanCheckKind.FENCING,
                        "leased event has no fencing token",
                        (event.event_id, *lease_ids),
                        event.logical_time,
                    )
                )
            if len(tokens) > 1 or (lease_ids and len(tokens) != len(lease_ids)):
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.INVALID_FENCING_TOKEN,
                        FindingDisposition.CONTRADICTION,
                        PlanCheckKind.FENCING,
                        "lease and fencing-token cardinalities are inconsistent",
                        (event.event_id, *lease_ids),
                        event.logical_time,
                        {"fencing_tokens": list(tokens)},
                    )
                )
            for lease_id, token in zip(lease_ids, tokens):
                if token <= 0:
                    findings.append(
                        PlanValidationFinding(
                            PlanFindingCode.INVALID_FENCING_TOKEN,
                            FindingDisposition.CONTRADICTION,
                            PlanCheckKind.FENCING,
                            "fencing tokens must be positive integers",
                            (event.event_id, lease_id),
                            event.logical_time,
                            {"fencing_token": token},
                        )
                    )
                prior = lease_fence.setdefault(lease_id, token)
                if prior != token:
                    findings.append(
                        PlanValidationFinding(
                            PlanFindingCode.FENCING_TOKEN_CHANGED,
                            FindingDisposition.CONTRADICTION,
                            PlanCheckKind.FENCING,
                            "a live lease changes fencing token within one planned execution",
                            (event.event_id, lease_id),
                            event.logical_time,
                            {"prior_token": prior, "observed_token": token},
                        )
                    )
        return findings

    def _build_and_check_trace(
        self,
        plan: FormalWorkPlan,
        formulas: Mapping[str, Formula],
        bound: int,
        guard: _BudgetGuard,
    ) -> tuple[_TraceModel, list[PlanValidationFinding]]:
        tasks = {item.task_id: item for item in plan.tasks}
        actors = {item.actor_id: item for item in plan.actors}
        requirements = {
            item.requirement_id: item for item in plan.evidence_requirements
        }
        events_by_time: dict[int, list[PlanEvent]] = defaultdict(list)
        for event in plan.events:
            if event.logical_time <= bound:
                events_by_time[event.logical_time].append(event)
        for values in events_by_time.values():
            values.sort(key=lambda item: item.event_id)

        authorization = _authorization_map(plan, formulas)
        state = {task_id: "pending" for task_id in tasks}
        available = set(_string_values(plan.metadata.get("available_evidence_ids")))
        available.update(_string_values(plan.metadata.get("evidence_cids")))
        task_states: list[dict[str, str]] = []
        step_events: list[tuple[str, ...]] = []
        evidence_steps: list[set[str]] = []
        completed_at: dict[str, int] = {}
        terminal_at: dict[str, int] = {}
        findings: list[PlanValidationFinding] = []
        assumptions: list[PlanValidationAssumption] = []
        reported_goal_evidence: set[str] = set()
        effect_by_event: dict[str, list[Any]] = defaultdict(list)
        effect_by_task_terminal: dict[str, list[Any]] = defaultdict(list)
        for effect in plan.effects:
            if effect.event_id:
                effect_by_event[effect.event_id].append(effect)
            else:
                effect_by_task_terminal[effect.task_id].append(effect)

        for logical_time in range(bound + 1):
            guard.checkpoint()
            current_events = events_by_time.get(logical_time, [])
            for event in current_events:
                guard.checkpoint()
                task = tasks[event.task_id]
                prior_state = state[event.task_id]
                if event.event_id not in task.event_ids:
                    findings.append(
                        _finding(
                            PlanFindingCode.ILLEGAL_TRANSITION,
                            PlanCheckKind.LEGAL_TRANSITION,
                            "event is not declared by the task transition set",
                            event,
                        )
                    )
                if event.actor_id not in task.actor_ids:
                    findings.append(
                        _finding(
                            PlanFindingCode.ACTOR_NOT_ASSIGNED,
                            PlanCheckKind.ACTOR_AUTHORITY,
                            "event actor is not assigned to the task",
                            event,
                            event.actor_id,
                        )
                    )
                elif event.kind not in {EventKind.ASSIGNED, EventKind.DELEGATED}:
                    allowed = authorization.get((event.actor_id, event.task_id), False)
                    actor = actors[event.actor_id]
                    if actor.authority_ids and not (
                        event.task_id in actor.authority_ids
                        or "*" in actor.authority_ids
                    ):
                        allowed = False
                    if not allowed:
                        findings.append(
                            _finding(
                                PlanFindingCode.ACTOR_NOT_AUTHORIZED,
                                PlanCheckKind.ACTOR_AUTHORITY,
                                "event actor lacks a reviewed authorization premise",
                                event,
                                event.actor_id,
                            )
                        )
                    missing = sorted(
                        set(_string_values(task.metadata.get("resource_needs")))
                        - set(actor.capabilities)
                    )
                    if missing:
                        findings.append(
                            _finding(
                                PlanFindingCode.ACTOR_CAPABILITY_MISSING,
                                PlanCheckKind.ACTOR_AUTHORITY,
                                "event actor lacks required task capabilities",
                                event,
                                event.actor_id,
                                details={"missing_capabilities": missing},
                            )
                        )

                if event.kind in {EventKind.STARTED, EventKind.EXECUTED}:
                    for dependency in task.depends_on:
                        dependency_task = tasks[dependency]
                        dependency_evidence = set(
                            dependency_task.evidence_requirement_ids
                        )
                        if (
                            dependency not in completed_at
                            or not dependency_evidence <= available
                        ):
                            findings.append(
                                _finding(
                                    PlanFindingCode.DEPENDENCY_NOT_READY,
                                    PlanCheckKind.DEPENDENCY_READINESS,
                                    "task starts or executes before a dependency is completed with evidence",
                                    event,
                                    dependency,
                                    details={
                                        "dependency_state": state[dependency],
                                        "missing_evidence_ids": sorted(
                                            dependency_evidence - available
                                        ),
                                    },
                                )
                            )

                for norm in plan.norms:
                    if (
                        norm.kind is NormKind.PROHIBITION
                        and not norm.activation_formula_id
                        and norm.bearer_actor_id == event.actor_id
                        and norm.action_id in {event.task_id, event.event_id}
                        and (norm.valid_from is None or norm.valid_from <= logical_time)
                        and (
                            norm.valid_until is None or logical_time <= norm.valid_until
                        )
                    ):
                        findings.append(
                            _finding(
                                PlanFindingCode.ACTOR_NOT_AUTHORIZED,
                                PlanCheckKind.DEONTIC_CONSISTENCY,
                                "event executes an active prohibition",
                                event,
                                norm.norm_id,
                            )
                        )

                legal = _legal_transition(prior_state, event.kind)
                if not legal:
                    code = (
                        PlanFindingCode.EVENT_AFTER_TERMINAL
                        if prior_state in task.terminal_states
                        else PlanFindingCode.ILLEGAL_TRANSITION
                    )
                    findings.append(
                        _finding(
                            code,
                            PlanCheckKind.LEGAL_TRANSITION,
                            f"event {event.kind.value} is illegal from state {prior_state}",
                            event,
                            details={"prior_state": prior_state},
                        )
                    )

                produced = set(_string_values(event.metadata.get("requirement_ids")))
                produced.update(_string_values(event.metadata.get("evidence_ids")))
                produced.update(
                    item for item in event.provenance_ids if item in requirements
                )
                if event.kind is EventKind.EVIDENCE_PRODUCED:
                    available.update(produced)

                if event.kind in _TERMINAL_EVENT_STATES:
                    terminal_state = _TERMINAL_EVENT_STATES[event.kind]
                    if terminal_state not in task.terminal_states:
                        findings.append(
                            _finding(
                                PlanFindingCode.TERMINAL_STATE_NOT_ALLOWED,
                                PlanCheckKind.LEGAL_TRANSITION,
                                f"terminal state {terminal_state} is not allowed by the task",
                                event,
                                details={"allowed_states": list(task.terminal_states)},
                            )
                        )
                    required = (
                        set(task.evidence_requirement_ids)
                        if event.kind is EventKind.COMPLETED
                        else set()
                    )
                    missing = required - available
                    synthesizable = {
                        item
                        for item in missing
                        if _evidence_is_realisable(requirements[item])
                    }
                    if synthesizable:
                        available.update(synthesizable)
                        for requirement_id in sorted(synthesizable):
                            assumptions.append(
                                _assumption(
                                    "evidence_realisability",
                                    "A declared accepted check can produce this required evidence before the planned terminal transition.",
                                    requirement_id,
                                    task.task_id,
                                )
                            )
                    missing = required - available
                    if missing:
                        findings.append(
                            _finding(
                                PlanFindingCode.REQUIRED_EVIDENCE_UNAVAILABLE,
                                PlanCheckKind.REQUIRED_EVIDENCE,
                                "terminal transition lacks mandatory evidence and no accepted production path is declared",
                                event,
                                *sorted(missing),
                                details={"missing_evidence_ids": sorted(missing)},
                            )
                        )
                    terminal_at[event.task_id] = logical_time
                    if event.kind is EventKind.COMPLETED:
                        completed_at[event.task_id] = logical_time
                    state[event.task_id] = terminal_state
                elif event.kind is EventKind.ASSIGNED:
                    state[event.task_id] = "assigned"
                elif event.kind is EventKind.DELEGATED:
                    state[event.task_id] = "delegated"
                elif event.kind is EventKind.STARTED:
                    state[event.task_id] = "started"
                elif event.kind is EventKind.EXECUTED:
                    state[event.task_id] = "executed"

                _check_fencing_observation(event, findings)
                applicable_effects = list(effect_by_event.get(event.event_id, ()))
                if event.kind in _TERMINAL_EVENT_STATES:
                    applicable_effects.extend(effect_by_task_terminal[event.task_id])
                for effect in applicable_effects:
                    guard.checkpoint()
                    value = (
                        str(effect.value).strip().lower()
                        if effect.value is not None
                        else ""
                    )
                    forbidden = set(_DEFAULT_FORBIDDEN_MERGE_STATES)
                    forbidden.update(
                        value.lower()
                        for value in _string_values(
                            task.metadata.get("forbidden_merge_states")
                            or task.metadata.get("forbidden_states")
                        )
                    )
                    if value in forbidden:
                        findings.append(
                            _finding(
                                PlanFindingCode.FORBIDDEN_MERGE_STATE,
                                PlanCheckKind.FORBIDDEN_MERGE_STATE,
                                f"effect reaches forbidden merge state {value}",
                                event,
                                effect.effect_id,
                            )
                        )
                    if "merge" in value or value == "merged":
                        missing_dependencies = [
                            dep for dep in task.depends_on if dep not in completed_at
                        ]
                        missing_evidence = sorted(
                            set(task.evidence_requirement_ids) - available
                        )
                        if (
                            missing_dependencies
                            or missing_evidence
                            or state[event.task_id] in {"failed", "cancelled"}
                        ):
                            findings.append(
                                _finding(
                                    PlanFindingCode.FORBIDDEN_MERGE_STATE,
                                    PlanCheckKind.FORBIDDEN_MERGE_STATE,
                                    "merge effect occurs before dependencies/evidence or after a forbidden terminal state",
                                    event,
                                    effect.effect_id,
                                    details={
                                        "missing_dependency_ids": missing_dependencies,
                                        "missing_evidence_ids": missing_evidence,
                                        "task_state": state[event.task_id],
                                    },
                                )
                            )

            # Goal-level review/check evidence is existentially scheduled at
            # the first state where every task in the goal is completed.  The
            # assumption is explicit and is only allowed when the requirement
            # declares a reviewed production path.
            for goal in plan.goals:
                goal_tasks = [
                    item for item in plan.tasks if item.goal_id == goal.goal_id
                ]
                if not goal_tasks or not all(
                    item.task_id in completed_at for item in goal_tasks
                ):
                    continue
                missing_goal_evidence = set(goal.evidence_requirement_ids) - available
                for requirement_id in sorted(missing_goal_evidence):
                    requirement = requirements[requirement_id]
                    if not _evidence_is_realisable(requirement):
                        continue
                    available.add(requirement_id)
                    assumptions.append(
                        _assumption(
                            "evidence_realisability",
                            "A declared accepted check can produce this required goal evidence before bounded goal satisfaction.",
                            requirement_id,
                            goal.goal_id,
                        )
                    )
                unavailable = set(goal.evidence_requirement_ids) - available
                if unavailable and goal.goal_id not in reported_goal_evidence:
                    reported_goal_evidence.add(goal.goal_id)
                    findings.append(
                        PlanValidationFinding(
                            PlanFindingCode.REQUIRED_EVIDENCE_UNAVAILABLE,
                            FindingDisposition.COUNTERMODEL,
                            PlanCheckKind.REQUIRED_EVIDENCE,
                            "goal completion lacks mandatory evidence and no accepted production path is declared",
                            (goal.goal_id, *sorted(unavailable)),
                            logical_time,
                            {"missing_evidence_ids": sorted(unavailable)},
                        )
                    )

            task_states.append(dict(state))
            step_events.append(tuple(item.event_id for item in current_events))
            evidence_steps.append(set(available))

        for task in plan.tasks:
            guard.checkpoint()
            if task.task_id not in terminal_at and bound >= plan.trace_bound:
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.TERMINAL_OUTCOME_MISSING,
                        FindingDisposition.COUNTERMODEL,
                        PlanCheckKind.EVENTUAL_TERMINAL,
                        "task has no allowed terminal outcome within the complete plan horizon",
                        (task.task_id,),
                        bound,
                        {"terminal_states": list(task.terminal_states)},
                    )
                )

        finite_trace = _finite_trace(
            plan,
            bound,
            task_states,
            step_events,
            evidence_steps,
            completed_at,
            terminal_at,
            authorization,
        )
        return (
            _TraceModel(
                task_states,
                step_events,
                evidence_steps,
                completed_at,
                terminal_at,
                finite_trace,
                assumptions,
            ),
            findings,
        )

    def _check_formulas(
        self,
        plan: FormalWorkPlan,
        formulas: Mapping[str, Formula],
        trace: _TraceModel,
        guard: _BudgetGuard,
    ) -> list[PlanValidationFinding]:
        findings: list[PlanValidationFinding] = []
        preconditions = {item.precondition_id: item for item in plan.preconditions}
        events = {item.event_id: item for item in plan.events}
        for task in plan.tasks:
            for precondition_id in task.precondition_ids:
                guard.checkpoint()
                precondition = preconditions[precondition_id]
                event = events.get(precondition.event_id)
                at = event.logical_time if event is not None else 0
                formula = formulas[precondition.formula_id]
                if not _evaluate_supported(
                    formula, trace.facts, plan, index=min(at, trace.facts.bound)
                ):
                    findings.append(
                        PlanValidationFinding(
                            PlanFindingCode.PRECONDITION_FALSE,
                            FindingDisposition.COUNTERMODEL,
                            PlanCheckKind.TEMPORAL_FORMULA,
                            "reviewed transition precondition is false in the bounded trace",
                            (
                                precondition.precondition_id,
                                formula.formula_id,
                                task.task_id,
                            ),
                            min(at, trace.facts.bound),
                        )
                    )
        for constraint in plan.temporal_constraints:
            guard.checkpoint()
            formula = formulas[constraint.formula_id]
            if not _evaluate_supported(formula, trace.facts, plan):
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.TEMPORAL_CONSTRAINT_FALSE,
                        FindingDisposition.COUNTERMODEL,
                        PlanCheckKind.TEMPORAL_FORMULA,
                        "reviewed temporal constraint is false in the bounded trace",
                        (
                            constraint.constraint_id,
                            formula.formula_id,
                            *constraint.subject_ids,
                        ),
                        trace.facts.bound,
                    )
                )
        for goal in plan.goals:
            guard.checkpoint()
            formula = formulas[goal.satisfaction_formula_id]
            if not _evaluate_supported(formula, trace.facts, plan):
                findings.append(
                    PlanValidationFinding(
                        PlanFindingCode.GOAL_NOT_SATISFIED,
                        FindingDisposition.COUNTERMODEL,
                        PlanCheckKind.TEMPORAL_FORMULA,
                        "goal satisfaction formula has no witness in the bounded trace",
                        (goal.goal_id, formula.formula_id),
                        trace.facts.bound,
                    )
                )
        return findings

    def _countermodel(
        self, finding: PlanValidationFinding, trace: _TraceModel
    ) -> PlanCountermodel:
        end = (
            finding.logical_time
            if finding.logical_time is not None
            else len(trace.task_states) - 1
        )
        start = max(0, end - self.bounds.max_countermodel_steps + 1)
        states = tuple(
            CountermodelState(
                index,
                trace.event_ids[index],
                trace.task_states[index],
                tuple(trace.available_evidence[index]),
            )
            for index in range(start, end + 1)
        )
        return PlanCountermodel(
            finding.finding_id,
            states,
            complete_prefix=start == 0,
        )

    def _result(
        self,
        plan: FormalWorkPlan,
        formulas: Mapping[str, Formula],
        evidence: tuple[PlanCheckEvidence, ...],
        assumptions: Iterable[PlanValidationAssumption],
        findings: Iterable[PlanValidationFinding],
        checks: Iterable[PlanCheckKind],
        guard: _BudgetGuard,
        effective_bound: int,
        domain_sizes: Mapping[str, int],
        truncated: Iterable[str],
        status: PlanValidationStatus,
        outcome: PlanValidationOutcome,
        *,
        countermodel: PlanCountermodel | None = None,
    ) -> PlanValidationResult:
        unique_assumptions = {item.assumption_id: item for item in assumptions}
        applied = AppliedValidationBounds(
            self.bounds,
            plan.trace_bound,
            effective_bound,
            domain_sizes,
            guard.nodes,
            tuple(truncated),
        )
        formula_ids = tuple(sorted(formulas))
        assumption_ids = tuple(sorted(unique_assumptions))
        consistency = PlanConsistencyLevel.INCONCLUSIVE
        if outcome in {
            PlanValidationOutcome.COUNTERMODEL,
            PlanValidationOutcome.CONTRADICTION,
        }:
            consistency = PlanConsistencyLevel.COUNTEREXAMPLE
        elif status is PlanValidationStatus.CONSISTENT:
            consistency = PlanConsistencyLevel.BOUNDED_CONSISTENT
            for item in evidence:
                role = item.backend_role.lower()
                authoritative_role = role in {
                    "kernel",
                    "accepted_kernel",
                    "model_checker",
                    "accepted_model_checker",
                }
                if (
                    authoritative_role
                    and (item.reconstructed or "model_checker" in role)
                    and item.exactly_binds(
                        plan_id=plan.plan_id,
                        formula_ids=formula_ids,
                        assumption_ids=assumption_ids,
                        bounds_id=applied.bounds_id,
                    )
                ):
                    consistency = PlanConsistencyLevel.KERNEL_VERIFIED
                    break
        return PlanValidationResult(
            status=status,
            outcome=outcome,
            plan_id=plan.plan_id,
            bounds=applied,
            assumptions=tuple(unique_assumptions.values()),
            formula_ids=formula_ids,
            findings=tuple(findings),
            countermodel=countermodel,
            evidence=evidence,
            consistency_level=consistency,
            checks_performed=tuple(checks),
        )


PlanValidator = FormalPlanValidator
BoundedFormalPlanValidator = FormalPlanValidator
PlanValidationConfig = ValidationBounds
ValidationResult = PlanValidationResult
Countermodel = PlanCountermodel


def validate_formal_plan(
    plan: FormalWorkPlan | Mapping[str, Any],
    formulas: Iterable[Formula | Mapping[str, Any]] | None = None,
    *,
    bounds: ValidationBounds | Mapping[str, Any] | None = None,
    cancellation_token: CancellationToken | Any | None = None,
    proof_evidence: Iterable[PlanCheckEvidence | Mapping[str, Any]] = (),
) -> PlanValidationResult:
    """Convenience entry point for one deterministic bounded validation."""

    return FormalPlanValidator(bounds).validate(
        plan,
        formulas,
        cancellation_token=cancellation_token,
        proof_evidence=proof_evidence,
    )


check_formal_plan = validate_formal_plan


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("details/metadata must be an object")
    # canonical_json rejects unsupported non-JSON values.
    import json

    return json.loads(canonical_json(dict(value)))


def _string_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, Sequence):
        return ()
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _fencing_values(metadata: Mapping[str, Any]) -> tuple[int, ...]:
    raw: Any = metadata.get("fencing_tokens")
    if raw in (None, (), []):
        raw = metadata.get("fencing_token")
    if raw is None:
        return ()
    values = (
        raw
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes))
        else (raw,)
    )
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            result.append(-1)
            continue
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            result.append(-1)
    return tuple(sorted(result))


def _unsupported_operator(formula: Formula) -> FormulaOperator | None:
    stack = [formula]
    while stack:
        current = stack.pop()
        if current.operator not in _SUPPORTED_OPERATORS:
            return current.operator
        stack.extend(reversed(current.operands))
    return None


def _formula_depth(formula: Formula) -> int:
    maximum = 0
    stack = [(formula, 1)]
    while stack:
        current, depth = stack.pop()
        maximum = max(maximum, depth)
        stack.extend((item, depth + 1) for item in current.operands)
    return maximum


def _raw_formula_depth(value: Formula | Mapping[str, Any], guard: _BudgetGuard) -> int:
    """Measure untrusted formula records iteratively before recursive decoding."""

    maximum = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        guard.checkpoint()
        current, depth = stack.pop()
        maximum = max(maximum, depth)
        if isinstance(current, Formula):
            stack.extend((item, depth + 1) for item in current.operands)
            continue
        if not isinstance(current, Mapping):
            continue
        operands = current.get("operands") or ()
        if isinstance(operands, Sequence) and not isinstance(operands, (str, bytes)):
            stack.extend((item, depth + 1) for item in operands)
    return maximum


def _intervals_overlap(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> bool:
    left_start = 0 if left_start is None else left_start
    right_start = 0 if right_start is None else right_start
    left_end = 2**63 - 1 if left_end is None else left_end
    right_end = 2**63 - 1 if right_end is None else right_end
    return max(left_start, right_start) <= min(left_end, right_end)


def _authorization_map(
    plan: FormalWorkPlan, formulas: Mapping[str, Formula]
) -> dict[tuple[str, str], bool]:
    result: dict[tuple[str, str], bool] = {}
    task_map = {item.task_id: item for item in plan.tasks}
    actor_map = {item.actor_id: item for item in plan.actors}
    for task in plan.tasks:
        for actor_id in task.actor_ids:
            actor = actor_map[actor_id]
            result[(actor_id, task.task_id)] = bool(
                task.task_id in actor.authority_ids
                or "*" in actor.authority_ids
                or (
                    actor.kind is ActorKind.SUPERVISOR
                    and (
                        "execute" in actor.capabilities
                        or "enforce_policy" in actor.capabilities
                    )
                )
            )
    for precondition in plan.preconditions:
        formula = formulas.get(precondition.formula_id)
        if (
            formula is not None
            and formula.operator is FormulaOperator.ATOM
            and formula.predicate is ReviewedPredicate.AUTHORIZED
        ):
            actor_id = str(formula.terms[0].value)
            task_id = str(formula.terms[1].value)
            if task_id in task_map and actor_id in task_map[task_id].actor_ids:
                result[(actor_id, task_id)] = True
    return result


def _legal_transition(state: str, kind: EventKind) -> bool:
    allowed = {
        "pending": {EventKind.ASSIGNED, EventKind.DELEGATED, EventKind.CANCELLED},
        "assigned": {EventKind.DELEGATED, EventKind.STARTED, EventKind.CANCELLED},
        "delegated": {EventKind.STARTED, EventKind.CANCELLED},
        "started": {
            EventKind.EXECUTED,
            EventKind.EVIDENCE_PRODUCED,
            EventKind.COMPLETED,
            EventKind.FAILED,
            EventKind.CANCELLED,
        },
        "executed": {
            EventKind.EVIDENCE_PRODUCED,
            EventKind.COMPLETED,
            EventKind.FAILED,
            EventKind.CANCELLED,
        },
    }
    return kind in allowed.get(state, set())


def _evidence_is_realisable(requirement: EvidenceRequirement) -> bool:
    metadata_available = requirement.metadata.get("available")
    if metadata_available is False:
        return False
    return bool(
        requirement.fallback_check_ids
        or requirement.source_scope_ids
        or requirement.kind is EvidenceRequirementKind.PLAN_CHECK
        or requirement.metadata.get("production_event_id")
    )


def _finding(
    code: PlanFindingCode,
    check: PlanCheckKind,
    message: str,
    event: PlanEvent,
    *subject_ids: str,
    details: Mapping[str, Any] | None = None,
) -> PlanValidationFinding:
    return PlanValidationFinding(
        code,
        FindingDisposition.COUNTERMODEL,
        check,
        message,
        (event.event_id, event.task_id, *subject_ids),
        event.logical_time,
        details or {},
    )


def _check_fencing_observation(
    event: PlanEvent, findings: list[PlanValidationFinding]
) -> None:
    observed = _fencing_values(event.metadata)
    current = event.metadata.get("current_fencing_token")
    mutation = event.metadata.get("mutation_fencing_token")
    if current is None and mutation is None:
        return
    try:
        expected = int(current)
        actual = int(mutation if mutation is not None else observed[0])
    except (TypeError, ValueError, IndexError):
        findings.append(
            _finding(
                PlanFindingCode.INVALID_FENCING_TOKEN,
                PlanCheckKind.FENCING,
                "fencing observation is not an integer",
                event,
            )
        )
        return
    if actual != expected:
        findings.append(
            _finding(
                PlanFindingCode.STALE_FENCING_TOKEN,
                PlanCheckKind.FENCING,
                "mutation fencing token does not match the current lease fence",
                event,
                details={
                    "mutation_fencing_token": actual,
                    "current_fencing_token": expected,
                },
            )
        )


def _finite_trace(
    plan: FormalWorkPlan,
    bound: int,
    task_states: Sequence[Mapping[str, str]],
    event_ids: Sequence[tuple[str, ...]],
    evidence: Sequence[set[str]],
    completed_at: Mapping[str, int],
    terminal_at: Mapping[str, int],
    authorization: Mapping[tuple[str, str], bool],
) -> FiniteTrace:
    events = {item.event_id: item for item in plan.events}
    tasks = {item.task_id: item for item in plan.tasks}
    facts_by_step: list[list[TraceFact]] = [[] for _ in range(bound + 1)]
    for index in range(bound + 1):
        facts = facts_by_step[index]
        for event_id in event_ids[index]:
            facts.append(
                TraceFact(
                    ReviewedPredicate.EVENT_OCCURRED,
                    (constant(TermSort.EVENT, event_id),),
                )
            )
            event = events[event_id]
            predicate = {
                EventKind.STARTED: ReviewedPredicate.TASK_STARTED,
                EventKind.COMPLETED: ReviewedPredicate.TASK_COMPLETED,
            }.get(event.kind)
            if predicate is not None:
                facts.append(
                    TraceFact(predicate, (constant(TermSort.TASK, event.task_id),))
                )
        for task_id in task_states[index]:
            # TASK_STARTED is an occurrence predicate above, not an
            # indefinitely persistent state.
            if task_id in completed_at and completed_at[task_id] <= index:
                facts.append(
                    TraceFact(
                        ReviewedPredicate.TASK_COMPLETED,
                        (constant(TermSort.TASK, task_id),),
                    )
                )
            if task_id in terminal_at and terminal_at[task_id] <= index:
                facts.append(
                    TraceFact(
                        ReviewedPredicate.TASK_TERMINAL,
                        (constant(TermSort.TASK, task_id),),
                    )
                )
            ready = all(
                dependency in completed_at and completed_at[dependency] <= index
                for dependency in tasks[task_id].depends_on
            )
            if ready:
                facts.append(
                    TraceFact(
                        ReviewedPredicate.TASK_READY,
                        (constant(TermSort.TASK, task_id),),
                    )
                )
            for dependency in tasks[task_id].depends_on:
                if dependency in completed_at and completed_at[dependency] <= index:
                    facts.append(
                        TraceFact(
                            ReviewedPredicate.DEPENDENCY_SATISFIED,
                            (
                                constant(TermSort.TASK, dependency),
                                constant(TermSort.TASK, task_id),
                            ),
                        )
                    )
        for (actor_id, task_id), allowed in authorization.items():
            if allowed:
                facts.append(
                    TraceFact(
                        ReviewedPredicate.AUTHORIZED,
                        (
                            constant(TermSort.ACTOR, actor_id),
                            constant(TermSort.TASK, task_id),
                        ),
                    )
                )
        for evidence_id in evidence[index]:
            facts.append(
                TraceFact(
                    ReviewedPredicate.EVIDENCE_AVAILABLE,
                    (constant(TermSort.EVIDENCE, evidence_id),),
                )
            )
        for goal in plan.goals:
            goal_tasks = [item for item in plan.tasks if item.goal_id == goal.goal_id]
            if (
                all(
                    item.task_id in completed_at
                    and completed_at[item.task_id] <= index
                    and set(item.evidence_requirement_ids) <= evidence[index]
                    for item in goal_tasks
                )
                and set(goal.evidence_requirement_ids) <= evidence[index]
            ):
                facts.append(
                    TraceFact(
                        ReviewedPredicate.GOAL_SATISFIED,
                        (constant(TermSort.GOAL, goal.goal_id),),
                    )
                )
        # Deduplicate since occurrence and persistent facts can coincide.
        unique = {item.fact_id: item for item in facts}
        facts_by_step[index] = list(unique.values())
    return FiniteTrace(
        tuple(
            TraceStep(
                index, tuple(facts_by_step[index]), tuple(sorted(evidence[index]))
            )
            for index in range(bound + 1)
        ),
        bound,
        plan.plan_id,
    )


def _evaluate_supported(
    formula: Formula,
    trace: FiniteTrace,
    plan: FormalWorkPlan,
    *,
    index: int = 0,
) -> bool:
    operator = formula.operator
    if operator is FormulaOperator.ATOM:
        return evaluate_formula(formula, trace, index=index)
    if operator is FormulaOperator.NOT:
        return not _evaluate_supported(formula.operands[0], trace, plan, index=index)
    if operator is FormulaOperator.AND:
        return all(
            _evaluate_supported(item, trace, plan, index=index)
            for item in formula.operands
        )
    if operator is FormulaOperator.OR:
        return any(
            _evaluate_supported(item, trace, plan, index=index)
            for item in formula.operands
        )
    if operator is FormulaOperator.IMPLIES:
        return not _evaluate_supported(
            formula.operands[0], trace, plan, index=index
        ) or _evaluate_supported(formula.operands[1], trace, plan, index=index)
    if operator is FormulaOperator.IFF:
        return _evaluate_supported(
            formula.operands[0], trace, plan, index=index
        ) == _evaluate_supported(formula.operands[1], trace, plan, index=index)
    if operator in {FormulaOperator.DEPENDENCY_ORDER, FormulaOperator.DEADLINE}:
        return evaluate_formula(formula, trace, index=index)
    if operator in {FormulaOperator.LIVENESS, FormulaOperator.GOAL_SATISFACTION}:
        lower = formula.lower_bound or 0
        upper = min(
            formula.upper_bound if formula.upper_bound is not None else trace.bound,
            trace.bound,
        )
        return any(
            _evaluate_supported(formula.operands[0], trace, plan, index=step)
            for step in range(lower, upper + 1)
        )
    if operator is FormulaOperator.SAFETY:
        lower = formula.lower_bound or 0
        upper = min(
            formula.upper_bound if formula.upper_bound is not None else trace.bound,
            trace.bound,
        )
        return all(
            _evaluate_supported(formula.operands[0], trace, plan, index=step)
            for step in range(lower, upper + 1)
        )
    if operator in {
        FormulaOperator.OBLIGATION,
        FormulaOperator.PERMISSION,
        FormulaOperator.PROHIBITION,
    }:
        at = min(int(formula.terms[1].value), trace.bound)
        value = evaluate_formula(formula.operands[0], trace, index=at)
        return not value if operator is FormulaOperator.PROHIBITION else value
    if operator is FormulaOperator.EXECUTION_EVENT:
        actor_id, task_id, event_id, at = (item.value for item in formula.terms)
        return any(
            event.event_id == event_id
            and event.actor_id == actor_id
            and event.task_id == task_id
            and event.logical_time == at
            for event in plan.events
        )
    if operator is FormulaOperator.DELEGATION:
        delegator, delegatee, task_id, at = (item.value for item in formula.terms)
        return any(
            event.kind is EventKind.DELEGATED
            and event.actor_id == delegatee
            and event.task_id == task_id
            and event.logical_time == at
            and event.metadata.get("delegator_actor_id") == delegator
            for event in plan.events
        )
    return False


def _evidence_records(
    values: Iterable[PlanCheckEvidence | Mapping[str, Any]],
    *,
    guard: _BudgetGuard | None = None,
    limit: int | None = None,
) -> tuple[tuple[PlanCheckEvidence, ...], bool]:
    result: dict[str, PlanCheckEvidence] = {}
    exceeded = False
    for index, value in enumerate(values):
        if guard is not None:
            guard.checkpoint()
        if limit is not None and index >= limit:
            exceeded = True
            break
        item = (
            value
            if isinstance(value, PlanCheckEvidence)
            else PlanCheckEvidence.from_dict(value)
        )
        result[item.evidence_id] = item
    return tuple(result[key] for key in sorted(result)), exceeded


__all__ = [
    "AppliedValidationBounds",
    "BoundedFormalPlanValidator",
    "CancellationToken",
    "Countermodel",
    "CountermodelState",
    "FindingDisposition",
    "FORMAL_PLAN_COUNTERMODEL_SCHEMA",
    "FORMAL_PLAN_FINDING_SCHEMA",
    "FORMAL_PLAN_VALIDATION_BOUNDS_SCHEMA",
    "FORMAL_PLAN_VALIDATION_SCHEMA",
    "FORMAL_PLAN_VALIDATOR_VERSION",
    "FormalPlanValidationBounds",
    "FormalPlanValidationResult",
    "FormalPlanValidator",
    "PlanAssumption",
    "PlanCheckEvidence",
    "PlanCheckKind",
    "PlanConsistencyLevel",
    "PlanCountermodel",
    "PlanFindingCode",
    "PlanValidationAssumption",
    "PlanValidationConfig",
    "PlanValidationFinding",
    "PlanValidationOutcome",
    "PlanValidationResult",
    "PlanValidationStatus",
    "PlanValidationVerdict",
    "PlanValidator",
    "ValidationBounds",
    "ValidationFinding",
    "ValidationOutcome",
    "ValidationResult",
    "ValidationStatus",
    "check_formal_plan",
    "validate_formal_plan",
]
