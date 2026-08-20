# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Receding-horizon plan-suffix controller.

``RecedingHorizonController`` freezes one objective revision and one accepted
plan identity, selects the nearest executable prefix-safe segment, and adapts
``FormalDeltaReplanner`` / ``DeltaReplanDecision`` results.  It is not a second
planner, plan-identity system, executor, or effect authority.

``PlanSuffixInvalidationReceipt`` is an autonomy-facing adapter over an existing
delta-replan decision.  When a delta decision is present, the adapter's receipt
identity is exactly that decision's identity.  Completed and unaffected steps
keep their current proof and validation receipts; only the dependency-minimal
evidenced suffix is reopened.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..planning.formal_replanner import (
    FORMAL_REPLANNER_VERSION,
    DeltaPlan,
    DeltaReplanDecision,
    DeltaReplanStopReason,
    FormalDeltaReplanner,
    ReplannerValidationError,
)
from ..planning.plan_failure_memory import (
    BranchFailureKind,
    BranchFailureObservation,
    PlanFailureMemory,
    PlanFailureMemoryError,
    TypedBranchFailure,
)
from ..proof.formal_verification_contracts import canonical_json, content_identity
from .contracts import (
    AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
    MAX_CANONICAL_RECORD_BYTES,
    MAX_IDENTIFIER_BYTES,
    MAX_MAPPING_ITEMS,
    MAX_SEQUENCE_ITEMS,
)

RECEDING_HORIZON_CONTROLLER_INTERFACE: Final[str] = "RecedingHorizonController@1"
PLAN_SUFFIX_INVALIDATION_RECEIPT_INTERFACE: Final[str] = (
    "PlanSuffixInvalidationReceipt@1"
)
PLAN_SUFFIX_INVALIDATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/plan-suffix-invalidation-receipt@1"
)
RECEDING_HORIZON_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/receding-horizon-evidence@1"
)
RECEDING_HORIZON_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/receding-horizon-snapshot@1"
)
NEAREST_SAFE_SEGMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/nearest-safe-segment@1"
)
MAX_RECEDING_HORIZON_SNAPSHOT_BYTES: Final[int] = 4 * MAX_CANONICAL_RECORD_BYTES

_UNBOUND_BRANCH_ID: Final[str] = "branch:unbound"
_FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "chain_of_thought",
        "cookie",
        "credential",
        "decoded_source",
        "executable_code",
        "hidden_reasoning",
        "model_transcript",
        "password",
        "private_key",
        "prompt",
        "raw_prompt",
        "refresh_token",
        "secret",
        "shell_command",
        "source_body",
        "transcript",
    }
)


class RecedingHorizonError(ValueError):
    """Raised when suffix control inputs violate the frozen adapter contract."""


class RecedingHorizonEvidenceKind(str, Enum):
    """Closed locality vocabulary for one receding-horizon observation."""

    CHANGED_FILE = "changed_file"
    FAILED_TEST = "failed_test"
    PROVIDER_REROUTE = "provider_reroute"
    COUNTEREXAMPLE = "counterexample"
    HUMAN_ANSWER = "human_answer"


class RecedingHorizonDisposition(str, Enum):
    """Closed outcome of one suffix-control step."""

    SUFFIX_REOPENED = "suffix_reopened"
    PREFIX_PRESERVED = "prefix_preserved"
    PROVIDER_REROUTED = "provider_rerouted"
    UNCHANGED_BACKOFF = "unchanged_backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    FAILURE_MEMORY_BOUND_REACHED = "failure_memory_bound_reached"
    REPAIR_BOUND_EXCEEDED = "repair_bound_exceeded"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    OBJECTIVE_REVISED = "objective_revised"


_STOP_REASON_DISPOSITION: Final[Mapping[DeltaReplanStopReason, RecedingHorizonDisposition]] = {
    DeltaReplanStopReason.REPLAN_REQUIRED: RecedingHorizonDisposition.SUFFIX_REOPENED,
    DeltaReplanStopReason.UNCHANGED_FAILURE_BACKOFF: (
        RecedingHorizonDisposition.UNCHANGED_BACKOFF
    ),
    DeltaReplanStopReason.IDENTICAL_FAILURE_EXHAUSTED: (
        RecedingHorizonDisposition.IDENTICAL_FAILURE_EXHAUSTED
    ),
    DeltaReplanStopReason.RETRY_BUDGET_EXHAUSTED: (
        RecedingHorizonDisposition.RETRY_BUDGET_EXHAUSTED
    ),
    DeltaReplanStopReason.FAILURE_MEMORY_BOUND_REACHED: (
        RecedingHorizonDisposition.FAILURE_MEMORY_BOUND_REACHED
    ),
    DeltaReplanStopReason.UNBOUND_FAILURE: RecedingHorizonDisposition.PREFIX_PRESERVED,
    DeltaReplanStopReason.REPAIR_BOUND_EXCEEDED: (
        RecedingHorizonDisposition.REPAIR_BOUND_EXCEEDED
    ),
    DeltaReplanStopReason.DEADLINE_EXCEEDED: RecedingHorizonDisposition.DEADLINE_EXCEEDED,
    DeltaReplanStopReason.CANCELLED: RecedingHorizonDisposition.CANCELLED,
}


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise RecedingHorizonError(f"{name} must be a compact identifier")
    if required and not result:
        raise RecedingHorizonError(f"{name} is required")
    if (
        len(result.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
        or any(ord(char) < 32 for char in result)
    ):
        raise RecedingHorizonError(f"{name} must be a compact bounded identifier")
    return result


def _identifiers(
    value: Any,
    name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
    maximum: int = MAX_SEQUENCE_ITEMS,
) -> tuple[str, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise RecedingHorizonError(f"{name} must be a sequence of identifiers")
    if len(raw) > maximum:
        raise RecedingHorizonError(f"{name} contains too many items")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        identifier = _identifier(item, name)
        if identifier not in seen:
            seen.add(identifier)
            normalized.append(identifier)
    if required and not normalized:
        raise RecedingHorizonError(f"{name} must not be empty")
    return tuple(normalized if preserve_order else sorted(normalized))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise RecedingHorizonError(f"{name} must be one of: {allowed}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RecedingHorizonError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecedingHorizonError(f"{name} must be an integer of at least {minimum}")
    return value


def _reject_forbidden_keys(payload: Mapping[str, Any], name: str) -> None:
    for key in payload:
        if not isinstance(key, str):
            raise RecedingHorizonError(f"{name} keys must be strings")
        normalized = key.strip().lower().replace("-", "_")
        if any(
            normalized == marker or normalized.endswith("_" + marker)
            for marker in _FORBIDDEN_FIELD_MARKERS
        ):
            raise RecedingHorizonError(
                f"{name} contains forbidden private or executable data"
            )


def _receipt_map(
    value: Any,
    name: str = "current_receipts",
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RecedingHorizonError(f"{name} must be a mapping of step receipts")
    if len(value) > MAX_MAPPING_ITEMS:
        raise RecedingHorizonError(f"{name} contains too many entries")
    _reject_forbidden_keys(value, name)
    result: dict[str, tuple[str, ...]] = {}
    for key, raw in value.items():
        step_id = _identifier(key, name)
        result[step_id] = _identifiers(raw, name)
    return result


def _coerce_plan(plan: DeltaPlan | Mapping[str, Any]) -> DeltaPlan:
    if isinstance(plan, DeltaPlan):
        return plan
    if isinstance(plan, Mapping):
        try:
            return DeltaPlan.from_dict(plan)
        except ReplannerValidationError as exc:
            raise RecedingHorizonError(str(exc)) from exc
    raise RecedingHorizonError("plan must be a DeltaPlan")


def _coerce_decision(
    decision: DeltaReplanDecision | Mapping[str, Any],
) -> DeltaReplanDecision:
    if isinstance(decision, DeltaReplanDecision):
        return decision
    if isinstance(decision, Mapping):
        try:
            return DeltaReplanDecision.from_dict(decision)
        except ReplannerValidationError as exc:
            raise RecedingHorizonError(str(exc)) from exc
    raise RecedingHorizonError("delta decision must be a DeltaReplanDecision")


def nearest_safe_segment_ids(plan: DeltaPlan) -> tuple[str, ...]:
    """Return the executable frontier whose dependencies are still accepted."""

    accepted = {item.step_id for item in plan.steps if item.accepted}
    return tuple(
        sorted(
            item.step_id
            for item in plan.steps
            if not item.accepted and set(item.dependency_ids).issubset(accepted)
        )
    )


def preserved_step_receipt_ids(
    plan: DeltaPlan,
    current_receipts: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Union current proof/validation receipts of accepted, unaffected steps."""

    extra = _receipt_map(current_receipts)
    collected: list[str] = []
    seen: set[str] = set()
    for item in plan.steps:
        if not item.accepted:
            continue
        for receipt_id in (*item.evidence_ids, *extra.get(item.step_id, ())):
            if receipt_id not in seen:
                seen.add(receipt_id)
                collected.append(receipt_id)
    return tuple(sorted(collected))


def _matching_step_ids(plan: DeltaPlan, evidence: RecedingHorizonEvidence) -> tuple[str, ...]:
    known = {item.step_id: item for item in plan.steps}
    if evidence.step_ids:
        return tuple(sorted(step_id for step_id in evidence.step_ids if step_id in known))
    path_set = set(evidence.path_ids)
    test_set = set(evidence.test_ids)
    capability_set = set(evidence.capability_ids)
    obligation_set = set(evidence.obligation_ids)
    alternative_set = set(evidence.alternative_ids)
    evidence_set = set(evidence.locality_evidence_ids)
    matched: list[str] = []
    for item in plan.steps:
        if (
            path_set.intersection(item.conflict_scope_ids)
            or test_set.intersection(item.validation_signature_ids)
            or capability_set.intersection(item.capability_ids)
            or obligation_set.intersection(item.obligation_ids)
            or alternative_set.intersection(item.alternative_ids)
            or evidence_set.intersection(item.evidence_ids)
        ):
            matched.append(item.step_id)
    return tuple(sorted(matched))


def _branch_for(plan: DeltaPlan, step_ids: Sequence[str]) -> str:
    by_id = {item.step_id: item for item in plan.steps}
    for step_id in step_ids:
        step = by_id.get(step_id)
        if step is not None:
            return step.branch_id
    return _UNBOUND_BRANCH_ID


@dataclass(frozen=True)
class FrozenObjectivePlan:
    """Immutable binding of objective semantics to one accepted plan identity."""

    objective_id: str
    objective_revision: str
    plan_id: str
    admitted_revision: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _identifier(self.objective_id, "objective_id"))
        object.__setattr__(
            self,
            "objective_revision",
            _identifier(self.objective_revision, "objective_revision"),
        )
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self,
            "admitted_revision",
            _identifier(self.admitted_revision, "admitted_revision", required=False),
        )


@dataclass(frozen=True)
class RecedingHorizonEvidence:
    """One typed, locality-bound observation.  Never a prompt or transcript."""

    kind: RecedingHorizonEvidenceKind
    evidence_id: str
    step_ids: tuple[str, ...] = ()
    path_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    alternative_ids: tuple[str, ...] = ()
    locality_evidence_ids: tuple[str, ...] = ()
    delivery_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, RecedingHorizonEvidenceKind, "kind")
        )
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        for name in (
            "step_ids",
            "path_ids",
            "test_ids",
            "capability_ids",
            "obligation_ids",
            "alternative_ids",
            "locality_evidence_ids",
        ):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        object.__setattr__(
            self,
            "delivery_id",
            _identifier(self.delivery_id, "delivery_id", required=False),
        )
        if self.kind is RecedingHorizonEvidenceKind.CHANGED_FILE and not self.path_ids:
            raise RecedingHorizonError("changed_file evidence requires path_ids")
        if self.kind is RecedingHorizonEvidenceKind.FAILED_TEST and not self.test_ids:
            raise RecedingHorizonError("failed_test evidence requires test_ids")
        if (
            self.kind is RecedingHorizonEvidenceKind.PROVIDER_REROUTE
            and not self.capability_ids
        ):
            raise RecedingHorizonError("provider_reroute evidence requires capability_ids")
        if self.kind is RecedingHorizonEvidenceKind.HUMAN_ANSWER and not (
            self.locality_evidence_ids or self.step_ids or self.obligation_ids
        ):
            raise RecedingHorizonError("human_answer evidence requires a locality binding")
        if self.kind is RecedingHorizonEvidenceKind.COUNTEREXAMPLE and not (
            self.step_ids
            or self.obligation_ids
            or self.alternative_ids
            or self.path_ids
        ):
            raise RecedingHorizonError("counterexample evidence requires a locality binding")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RECEDING_HORIZON_EVIDENCE_SCHEMA,
            "kind": self.kind.value,
            "evidence_id": self.evidence_id,
            "step_ids": list(self.step_ids),
            "path_ids": list(self.path_ids),
            "test_ids": list(self.test_ids),
            "capability_ids": list(self.capability_ids),
            "obligation_ids": list(self.obligation_ids),
            "alternative_ids": list(self.alternative_ids),
            "locality_evidence_ids": list(self.locality_evidence_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | RecedingHorizonEvidence) -> RecedingHorizonEvidence:
        if isinstance(payload, RecedingHorizonEvidence):
            return payload
        if not isinstance(payload, Mapping):
            raise RecedingHorizonError("receding-horizon evidence must be an object")
        expected = {
            "schema",
            "kind",
            "evidence_id",
            "step_ids",
            "path_ids",
            "test_ids",
            "capability_ids",
            "obligation_ids",
            "alternative_ids",
            "locality_evidence_ids",
            "delivery_id",
        }
        extra = set(payload).difference(expected)
        if extra:
            raise RecedingHorizonError("receding-horizon evidence contains unsupported fields")
        _reject_forbidden_keys(payload, "evidence")
        if payload.get("schema") not in (None, "", RECEDING_HORIZON_EVIDENCE_SCHEMA):
            raise RecedingHorizonError("unsupported receding-horizon evidence schema")
        return cls(
            kind=payload.get("kind", ""),
            evidence_id=payload.get("evidence_id", ""),
            step_ids=tuple(payload.get("step_ids") or ()),
            path_ids=tuple(payload.get("path_ids") or ()),
            test_ids=tuple(payload.get("test_ids") or ()),
            capability_ids=tuple(payload.get("capability_ids") or ()),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            alternative_ids=tuple(payload.get("alternative_ids") or ()),
            locality_evidence_ids=tuple(payload.get("locality_evidence_ids") or ()),
            delivery_id=str(payload.get("delivery_id") or ""),
        )


@dataclass(frozen=True)
class NearestSafeSegment:
    """Executable frontier of the current frozen plan."""

    step_ids: tuple[str, ...]
    branch_ids: tuple[str, ...]
    prefix_step_ids: tuple[str, ...]
    suffix_step_ids: tuple[str, ...]
    idle: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_ids", _identifiers(self.step_ids, "step_ids"))
        object.__setattr__(self, "branch_ids", _identifiers(self.branch_ids, "branch_ids"))
        object.__setattr__(
            self, "prefix_step_ids", _identifiers(self.prefix_step_ids, "prefix_step_ids")
        )
        object.__setattr__(
            self, "suffix_step_ids", _identifiers(self.suffix_step_ids, "suffix_step_ids")
        )
        object.__setattr__(self, "idle", _bool(self.idle, "idle"))
        if self.idle and (self.step_ids or self.suffix_step_ids):
            raise RecedingHorizonError("an idle plan cannot expose an executable suffix")
        if set(self.step_ids).difference(self.suffix_step_ids):
            raise RecedingHorizonError("nearest safe segment must be inside the open suffix")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NEAREST_SAFE_SEGMENT_SCHEMA,
            "step_ids": list(self.step_ids),
            "branch_ids": list(self.branch_ids),
            "prefix_step_ids": list(self.prefix_step_ids),
            "suffix_step_ids": list(self.suffix_step_ids),
            "idle": self.idle,
        }


def select_nearest_safe_segment(plan: DeltaPlan) -> NearestSafeSegment:
    prefix = tuple(sorted(item.step_id for item in plan.steps if item.accepted))
    suffix = tuple(sorted(item.step_id for item in plan.steps if not item.accepted))
    frontier = nearest_safe_segment_ids(plan)
    by_id = {item.step_id: item for item in plan.steps}
    branches = tuple(sorted({by_id[step_id].branch_id for step_id in frontier}))
    return NearestSafeSegment(
        step_ids=frontier,
        branch_ids=branches,
        prefix_step_ids=prefix,
        suffix_step_ids=suffix,
        idle=not suffix,
    )


def _drop_invalidated_receipts(
    current_receipts: Mapping[str, tuple[str, ...]],
    invalidated_step_ids: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    invalidated = set(invalidated_step_ids)
    return {
        step_id: tuple(receipt_ids)
        for step_id, receipt_ids in current_receipts.items()
        if step_id not in invalidated
    }


@dataclass(frozen=True)
class PlanSuffixInvalidationReceipt:
    """Autonomy-facing adapter over ``DeltaReplanDecision``.

    The adapter never mints a second plan identity.  When a delta decision is
    present, ``receipt_id`` is exactly ``delta_decision.decision_id``.
    """

    objective_id: str
    objective_revision: str
    frozen_plan_id: str
    disposition: RecedingHorizonDisposition
    evidence_kind: RecedingHorizonEvidenceKind | None = None
    evidence_id: str = ""
    delta_decision: DeltaReplanDecision | None = None
    direct_failure_step_ids: tuple[str, ...] = ()
    invalidated_step_ids: tuple[str, ...] = ()
    stale_dependency_step_ids: tuple[str, ...] = ()
    preserved_step_ids: tuple[str, ...] = ()
    preserved_receipt_ids: tuple[str, ...] = ()
    nearest_safe_segment_ids: tuple[str, ...] = ()
    rerouted_step_ids: tuple[str, ...] = ()
    stop_reason: str = ""
    diagnostic_reused: bool = False
    backoff_milliseconds: int = 0
    objective_semantics_changed: bool = False
    admitted_revision: str = ""
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _identifier(self.objective_id, "objective_id"))
        object.__setattr__(
            self,
            "objective_revision",
            _identifier(self.objective_revision, "objective_revision"),
        )
        object.__setattr__(
            self, "frozen_plan_id", _identifier(self.frozen_plan_id, "frozen_plan_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RecedingHorizonDisposition, "disposition"),
        )
        if self.evidence_kind is not None:
            object.__setattr__(
                self,
                "evidence_kind",
                _enum(self.evidence_kind, RecedingHorizonEvidenceKind, "evidence_kind"),
            )
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id", required=False)
        )
        if self.delta_decision is not None and not isinstance(
            self.delta_decision, DeltaReplanDecision
        ):
            object.__setattr__(self, "delta_decision", _coerce_decision(self.delta_decision))
        for name in (
            "direct_failure_step_ids",
            "invalidated_step_ids",
            "stale_dependency_step_ids",
            "preserved_step_ids",
            "preserved_receipt_ids",
            "nearest_safe_segment_ids",
            "rerouted_step_ids",
        ):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        object.__setattr__(
            self,
            "reason_codes",
            _identifiers(self.reason_codes, "reason_codes", preserve_order=True),
        )
        object.__setattr__(
            self, "stop_reason", _identifier(self.stop_reason, "stop_reason", required=False)
        )
        object.__setattr__(
            self, "diagnostic_reused", _bool(self.diagnostic_reused, "diagnostic_reused")
        )
        object.__setattr__(
            self,
            "backoff_milliseconds",
            _int(self.backoff_milliseconds, "backoff_milliseconds"),
        )
        object.__setattr__(
            self,
            "objective_semantics_changed",
            _bool(self.objective_semantics_changed, "objective_semantics_changed"),
        )
        object.__setattr__(
            self,
            "admitted_revision",
            _identifier(self.admitted_revision, "admitted_revision", required=False),
        )
        if self.objective_semantics_changed and not self.admitted_revision:
            raise RecedingHorizonError(
                "objective semantics never change without an admitted revision"
            )
        if set(self.invalidated_step_ids).intersection(self.preserved_step_ids):
            raise RecedingHorizonError("invalidated and preserved step sets must be disjoint")
        if self.delta_decision is not None:
            decision = self.delta_decision
            if self.direct_failure_step_ids != decision.direct_failure_step_ids:
                raise RecedingHorizonError("adapter direct-failure projection is inconsistent")
            if self.invalidated_step_ids != decision.invalidated_step_ids:
                raise RecedingHorizonError("adapter invalidated suffix is inconsistent")
            if self.stale_dependency_step_ids != decision.stale_dependency_step_ids:
                raise RecedingHorizonError("adapter stale-dependency projection is inconsistent")
            if self.preserved_step_ids != decision.preserved_step_ids:
                raise RecedingHorizonError("adapter preserved-prefix projection is inconsistent")
            if self.stop_reason != decision.stop_reason.value:
                raise RecedingHorizonError("adapter stop reason does not match the delta decision")
        encoded = canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        if len(encoded) > MAX_CANONICAL_RECORD_BYTES:
            raise RecedingHorizonError("plan-suffix receipt exceeds its bounded size")

    @property
    def receipt_id(self) -> str:
        if self.delta_decision is not None:
            return self.delta_decision.decision_id
        return content_identity(self.to_dict(include_identity=False))

    @property
    def decision_id(self) -> str:
        return self.delta_decision.decision_id if self.delta_decision is not None else ""

    @property
    def original_plan_id(self) -> str:
        if self.delta_decision is not None:
            return self.delta_decision.original_plan_id
        return self.frozen_plan_id

    @property
    def resulting_plan_id(self) -> str:
        if self.delta_decision is not None:
            return self.delta_decision.resulting_plan.plan_id
        return self.frozen_plan_id

    @property
    def changed(self) -> bool:
        return self.disposition is RecedingHorizonDisposition.SUFFIX_REOPENED

    @property
    def authorizes_effect(self) -> bool:
        return False

    @property
    def authorizes_full_replan(self) -> bool:
        return False

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLAN_SUFFIX_INVALIDATION_RECEIPT_SCHEMA,
            "interface": PLAN_SUFFIX_INVALIDATION_RECEIPT_INTERFACE,
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "frozen_plan_id": self.frozen_plan_id,
            "disposition": self.disposition.value,
            "evidence_kind": None if self.evidence_kind is None else self.evidence_kind.value,
            "evidence_id": self.evidence_id,
            "delta_decision": (
                None if self.delta_decision is None else self.delta_decision.to_dict()
            ),
            "delta_decision_id": self.decision_id,
            "direct_failure_step_ids": list(self.direct_failure_step_ids),
            "invalidated_step_ids": list(self.invalidated_step_ids),
            "stale_dependency_step_ids": list(self.stale_dependency_step_ids),
            "preserved_step_ids": list(self.preserved_step_ids),
            "preserved_receipt_ids": list(self.preserved_receipt_ids),
            "nearest_safe_segment_ids": list(self.nearest_safe_segment_ids),
            "rerouted_step_ids": list(self.rerouted_step_ids),
            "stop_reason": self.stop_reason,
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_milliseconds": self.backoff_milliseconds,
            "objective_semantics_changed": self.objective_semantics_changed,
            "admitted_revision": self.admitted_revision,
            "reason_codes": list(self.reason_codes),
            "authorizes_effect": False,
            "authorizes_full_replan": False,
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlanSuffixInvalidationReceipt:
        expected = {
            "schema",
            "interface",
            "replanner_version",
            "objective_id",
            "objective_revision",
            "frozen_plan_id",
            "disposition",
            "evidence_kind",
            "evidence_id",
            "delta_decision",
            "delta_decision_id",
            "direct_failure_step_ids",
            "invalidated_step_ids",
            "stale_dependency_step_ids",
            "preserved_step_ids",
            "preserved_receipt_ids",
            "nearest_safe_segment_ids",
            "rerouted_step_ids",
            "stop_reason",
            "diagnostic_reused",
            "backoff_milliseconds",
            "objective_semantics_changed",
            "admitted_revision",
            "reason_codes",
            "authorizes_effect",
            "authorizes_full_replan",
            "receipt_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise RecedingHorizonError("plan-suffix receipt must use the closed schema")
        _reject_forbidden_keys(payload, "plan-suffix receipt")
        if payload.get("schema") != PLAN_SUFFIX_INVALIDATION_RECEIPT_SCHEMA:
            raise RecedingHorizonError("unsupported plan-suffix receipt schema")
        if payload.get("interface") != PLAN_SUFFIX_INVALIDATION_RECEIPT_INTERFACE:
            raise RecedingHorizonError("unsupported plan-suffix receipt interface")
        if payload.get("replanner_version") != FORMAL_REPLANNER_VERSION:
            raise RecedingHorizonError("plan-suffix receipt replanner version is unsupported")
        if payload.get("authorizes_effect") is not False:
            raise RecedingHorizonError("plan-suffix receipts cannot authorize effects")
        if payload.get("authorizes_full_replan") is not False:
            raise RecedingHorizonError("plan-suffix receipts cannot authorize a full replan")
        raw_decision = payload.get("delta_decision")
        decision = None if raw_decision is None else _coerce_decision(raw_decision)
        result = cls(
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            frozen_plan_id=payload.get("frozen_plan_id", ""),
            disposition=payload.get("disposition", ""),
            evidence_kind=payload.get("evidence_kind"),
            evidence_id=payload.get("evidence_id", ""),
            delta_decision=decision,
            direct_failure_step_ids=tuple(payload.get("direct_failure_step_ids") or ()),
            invalidated_step_ids=tuple(payload.get("invalidated_step_ids") or ()),
            stale_dependency_step_ids=tuple(payload.get("stale_dependency_step_ids") or ()),
            preserved_step_ids=tuple(payload.get("preserved_step_ids") or ()),
            preserved_receipt_ids=tuple(payload.get("preserved_receipt_ids") or ()),
            nearest_safe_segment_ids=tuple(payload.get("nearest_safe_segment_ids") or ()),
            rerouted_step_ids=tuple(payload.get("rerouted_step_ids") or ()),
            stop_reason=payload.get("stop_reason", ""),
            diagnostic_reused=payload.get("diagnostic_reused"),
            backoff_milliseconds=payload.get("backoff_milliseconds", -1),
            objective_semantics_changed=payload.get("objective_semantics_changed"),
            admitted_revision=payload.get("admitted_revision", ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        if payload.get("delta_decision_id") != result.decision_id:
            raise RecedingHorizonError("adapter delta-decision identity is inconsistent")
        if payload.get("receipt_id") != result.receipt_id:
            raise RecedingHorizonError(
                "plan-suffix adapter identity does not preserve the delta decision"
            )
        return result

    @classmethod
    def from_delta_decision(
        cls,
        decision: DeltaReplanDecision,
        *,
        objective_id: str,
        objective_revision: str,
        frozen_plan_id: str,
        evidence: RecedingHorizonEvidence | None = None,
        preserved_receipt_ids: Sequence[str] = (),
        nearest_safe_ids: Sequence[str] | None = None,
        admitted_revision: str = "",
        extra_reason_codes: Sequence[str] = (),
    ) -> PlanSuffixInvalidationReceipt:
        disposition = _STOP_REASON_DISPOSITION[decision.stop_reason]
        reasons = [disposition.value, decision.stop_reason.value]
        if decision.changed:
            reasons.append("smallest_evidenced_suffix")
        else:
            reasons.append("prefix_and_receipts_preserved")
        reasons.extend(extra_reason_codes)
        return cls(
            objective_id=objective_id,
            objective_revision=objective_revision,
            frozen_plan_id=frozen_plan_id,
            disposition=disposition,
            evidence_kind=None if evidence is None else evidence.kind,
            evidence_id="" if evidence is None else evidence.evidence_id,
            delta_decision=decision,
            direct_failure_step_ids=decision.direct_failure_step_ids,
            invalidated_step_ids=decision.invalidated_step_ids,
            stale_dependency_step_ids=decision.stale_dependency_step_ids,
            preserved_step_ids=decision.preserved_step_ids,
            preserved_receipt_ids=tuple(preserved_receipt_ids),
            nearest_safe_segment_ids=(
                nearest_safe_segment_ids(decision.resulting_plan)
                if nearest_safe_ids is None
                else tuple(nearest_safe_ids)
            ),
            stop_reason=decision.stop_reason.value,
            diagnostic_reused=decision.diagnostic_reused,
            backoff_milliseconds=decision.backoff_milliseconds,
            admitted_revision=admitted_revision,
            reason_codes=tuple(reasons),
        )


class RecedingHorizonController:
    """Freeze one objective/plan and reopen only the evidenced suffix.

    Effect execution, merge, and completion remain outside this module.  The
    controller never regenerates an unaffected step, never invents a plan
    identity, and never changes objective semantics without an admitted
    revision.
    """

    def __init__(
        self,
        *,
        objective_id: str,
        objective_revision: str,
        plan: DeltaPlan | Mapping[str, Any],
        current_receipts: Mapping[str, Sequence[str]] | None = None,
        replanner: FormalDeltaReplanner | None = None,
        failure_memory: PlanFailureMemory | None = None,
        admitted_revision: str = "",
        frozen_plan_id: str | None = None,
    ) -> None:
        value = _coerce_plan(plan)
        self._freeze = FrozenObjectivePlan(
            objective_id=objective_id,
            objective_revision=objective_revision,
            plan_id=frozen_plan_id or value.plan_id,
            admitted_revision=admitted_revision,
        )
        self._plan = value
        self._current_receipts = _receipt_map(current_receipts)
        if replanner is not None and not isinstance(replanner, FormalDeltaReplanner):
            raise RecedingHorizonError("replanner must be FormalDeltaReplanner")
        if failure_memory is not None and replanner is not None:
            raise RecedingHorizonError(
                "inject FormalDeltaReplanner or PlanFailureMemory, not both"
            )
        self._replanner = replanner or FormalDeltaReplanner(failure_memory=failure_memory)

    @property
    def interface(self) -> str:
        return RECEDING_HORIZON_CONTROLLER_INTERFACE

    @property
    def objective_id(self) -> str:
        return self._freeze.objective_id

    @property
    def objective_revision(self) -> str:
        return self._freeze.objective_revision

    @property
    def frozen_plan_id(self) -> str:
        return self._freeze.plan_id

    @property
    def admitted_revision(self) -> str:
        return self._freeze.admitted_revision

    @property
    def plan(self) -> DeltaPlan:
        return self._plan

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    @property
    def current_receipts(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {step_id: tuple(ids) for step_id, ids in sorted(self._current_receipts.items())}
        )

    @property
    def idle(self) -> bool:
        return all(item.accepted for item in self._plan.steps)

    @property
    def replanner(self) -> FormalDeltaReplanner:
        return self._replanner

    def select_nearest_safe_segment(self) -> NearestSafeSegment:
        return select_nearest_safe_segment(self._plan)

    def _to_observation(
        self,
        evidence: RecedingHorizonEvidence,
        anchors: Sequence[str],
    ) -> BranchFailureObservation:
        if evidence.kind is RecedingHorizonEvidenceKind.CHANGED_FILE:
            kind = BranchFailureKind.CONFLICT
        elif evidence.kind is RecedingHorizonEvidenceKind.FAILED_TEST:
            kind = BranchFailureKind.VALIDATION_SIGNATURE
        elif evidence.kind is RecedingHorizonEvidenceKind.COUNTEREXAMPLE:
            kind = BranchFailureKind.COUNTEREXAMPLE
        elif evidence.kind is RecedingHorizonEvidenceKind.HUMAN_ANSWER:
            kind = BranchFailureKind.COUNTEREXAMPLE
        else:
            raise RecedingHorizonError("evidence kind cannot be submitted as a delta failure")
        try:
            features = TypedBranchFailure(
                scope=self._plan.scope,
                kind=kind,
                failure_code=f"failure:{evidence.kind.value}",
                branch_id=_branch_for(self._plan, anchors),
                step_ids=tuple(anchors) or evidence.step_ids,
                obligation_ids=evidence.obligation_ids,
                alternative_ids=evidence.alternative_ids,
                constraint_ids=(),
                validation_signature_ids=evidence.test_ids,
                capability_ids=evidence.capability_ids,
                conflict_scope_ids=evidence.path_ids,
                resource_ids=(),
            )
            return BranchFailureObservation(
                features=features,
                evidence_id=evidence.evidence_id,
                delivery_id=evidence.delivery_id,
            )
        except PlanFailureMemoryError as exc:
            raise RecedingHorizonError(str(exc)) from exc

    def _apply_decision(
        self,
        decision: DeltaReplanDecision,
        *,
        evidence: RecedingHorizonEvidence | None = None,
        extra_reason_codes: Sequence[str] = (),
    ) -> PlanSuffixInvalidationReceipt:
        if decision.original_plan_id != self._plan.plan_id:
            raise RecedingHorizonError("delta decision is bound to a different plan identity")
        if decision.changed:
            self._plan = decision.resulting_plan
            self._current_receipts = _drop_invalidated_receipts(
                self._current_receipts, decision.invalidated_step_ids
            )
        preserved_receipts = preserved_step_receipt_ids(self._plan, self._current_receipts)
        return PlanSuffixInvalidationReceipt.from_delta_decision(
            decision,
            objective_id=self._freeze.objective_id,
            objective_revision=self._freeze.objective_revision,
            frozen_plan_id=self._freeze.plan_id,
            evidence=evidence,
            preserved_receipt_ids=preserved_receipts,
            admitted_revision=self._freeze.admitted_revision,
            extra_reason_codes=extra_reason_codes,
        )

    def adapt_delta(
        self,
        decision: DeltaReplanDecision | Mapping[str, Any],
        *,
        evidence: RecedingHorizonEvidence | Mapping[str, Any] | None = None,
    ) -> PlanSuffixInvalidationReceipt:
        """Adapt one already-computed FormalDeltaReplanner result."""

        bound_evidence = (
            None if evidence is None else RecedingHorizonEvidence.from_dict(evidence)
        )
        return self._apply_decision(_coerce_decision(decision), evidence=bound_evidence)

    def _prefix_preserved(
        self,
        evidence: RecedingHorizonEvidence,
        *,
        stop_reason: str,
        extra_reason_codes: Sequence[str] = (),
    ) -> PlanSuffixInvalidationReceipt:
        preserved = tuple(sorted(item.step_id for item in self._plan.steps if item.accepted))
        segment = select_nearest_safe_segment(self._plan)
        return PlanSuffixInvalidationReceipt(
            objective_id=self._freeze.objective_id,
            objective_revision=self._freeze.objective_revision,
            frozen_plan_id=self._freeze.plan_id,
            disposition=RecedingHorizonDisposition.PREFIX_PRESERVED,
            evidence_kind=evidence.kind,
            evidence_id=evidence.evidence_id,
            preserved_step_ids=preserved,
            preserved_receipt_ids=preserved_step_receipt_ids(
                self._plan, self._current_receipts
            ),
            nearest_safe_segment_ids=segment.step_ids,
            stop_reason=stop_reason,
            reason_codes=(
                RecedingHorizonDisposition.PREFIX_PRESERVED.value,
                stop_reason,
                "prefix_and_receipts_preserved",
                *extra_reason_codes,
            ),
        )

    def _provider_reroute(
        self, evidence: RecedingHorizonEvidence
    ) -> PlanSuffixInvalidationReceipt:
        capability_set = set(evidence.capability_ids)
        rerouted = tuple(
            sorted(
                item.step_id
                for item in self._plan.steps
                if not item.accepted and capability_set.intersection(item.capability_ids)
            )
        )
        preserved = tuple(sorted(item.step_id for item in self._plan.steps if item.accepted))
        segment = select_nearest_safe_segment(self._plan)
        return PlanSuffixInvalidationReceipt(
            objective_id=self._freeze.objective_id,
            objective_revision=self._freeze.objective_revision,
            frozen_plan_id=self._freeze.plan_id,
            disposition=RecedingHorizonDisposition.PROVIDER_REROUTED,
            evidence_kind=evidence.kind,
            evidence_id=evidence.evidence_id,
            preserved_step_ids=preserved,
            preserved_receipt_ids=preserved_step_receipt_ids(
                self._plan, self._current_receipts
            ),
            nearest_safe_segment_ids=segment.step_ids,
            rerouted_step_ids=rerouted,
            stop_reason="provider_reroute",
            reason_codes=(
                RecedingHorizonDisposition.PROVIDER_REROUTED.value,
                "eligible_questions_only",
                "prefix_and_receipts_preserved",
            ),
        )

    def observe(
        self,
        evidence: RecedingHorizonEvidence | Mapping[str, Any],
        *,
        observed_at_milliseconds: int = 1,
        now_milliseconds: int | None = None,
        deadline_milliseconds: int | None = None,
        cancelled: Any = None,
    ) -> PlanSuffixInvalidationReceipt:
        """Reopen only the smallest evidenced suffix, or reroute without replan."""

        bound = RecedingHorizonEvidence.from_dict(evidence)
        if bound.kind is RecedingHorizonEvidenceKind.PROVIDER_REROUTE:
            return self._provider_reroute(bound)
        anchors = _matching_step_ids(self._plan, bound)
        try:
            observation = self._to_observation(bound, anchors)
        except RecedingHorizonError:
            return self._prefix_preserved(
                bound,
                stop_reason="unbound_failure",
                extra_reason_codes=("no_dependency_evidence",),
            )
        try:
            decision = self._replanner.replan(
                self._plan,
                observation,
                observed_at_milliseconds=observed_at_milliseconds,
                now_milliseconds=now_milliseconds,
                deadline_milliseconds=deadline_milliseconds,
                cancelled=cancelled,
            )
        except (ReplannerValidationError, PlanFailureMemoryError) as exc:
            raise RecedingHorizonError(str(exc)) from exc
        extra: list[str] = []
        if bound.kind is RecedingHorizonEvidenceKind.CHANGED_FILE:
            extra.append("changed_file_locality")
        elif bound.kind is RecedingHorizonEvidenceKind.FAILED_TEST:
            extra.append("failed_test_dependency")
        elif bound.kind is RecedingHorizonEvidenceKind.HUMAN_ANSWER:
            extra.append("human_answer_locality")
        elif bound.kind is RecedingHorizonEvidenceKind.COUNTEREXAMPLE:
            extra.append("counterexample_locality")
        return self._apply_decision(decision, evidence=bound, extra_reason_codes=extra)

    def revise_objective(
        self,
        *,
        admitted_revision: str,
        objective_revision: str,
        objective_id: str | None = None,
        plan: DeltaPlan | Mapping[str, Any] | None = None,
        current_receipts: Mapping[str, Sequence[str]] | None = None,
    ) -> PlanSuffixInvalidationReceipt:
        """Admit a new objective revision.  Semantics never change without this."""

        admitted = _identifier(admitted_revision, "admitted_revision")
        new_revision = _identifier(objective_revision, "objective_revision")
        new_objective = (
            self._freeze.objective_id
            if objective_id is None
            else _identifier(objective_id, "objective_id")
        )
        semantics_changed = (
            new_revision != self._freeze.objective_revision
            or new_objective != self._freeze.objective_id
        )
        if semantics_changed and plan is None:
            raise RecedingHorizonError(
                "admitted objective revision requires a rebound plan"
            )
        new_plan = self._plan if plan is None else _coerce_plan(plan)
        self._freeze = FrozenObjectivePlan(
            objective_id=new_objective,
            objective_revision=new_revision,
            plan_id=new_plan.plan_id,
            admitted_revision=admitted,
        )
        self._plan = new_plan
        if current_receipts is not None:
            self._current_receipts = _receipt_map(current_receipts)
        segment = select_nearest_safe_segment(self._plan)
        preserved = tuple(sorted(item.step_id for item in self._plan.steps if item.accepted))
        return PlanSuffixInvalidationReceipt(
            objective_id=self._freeze.objective_id,
            objective_revision=self._freeze.objective_revision,
            frozen_plan_id=self._freeze.plan_id,
            disposition=RecedingHorizonDisposition.OBJECTIVE_REVISED,
            preserved_step_ids=preserved,
            preserved_receipt_ids=preserved_step_receipt_ids(
                self._plan, self._current_receipts
            ),
            nearest_safe_segment_ids=segment.step_ids,
            stop_reason="objective_revised",
            objective_semantics_changed=True,
            admitted_revision=admitted,
            reason_codes=(
                RecedingHorizonDisposition.OBJECTIVE_REVISED.value,
                "admitted_revision",
            ),
        )

    def snapshot(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": RECEDING_HORIZON_SNAPSHOT_SCHEMA,
            "program_id": AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
            "interface": RECEDING_HORIZON_CONTROLLER_INTERFACE,
            "objective_id": self._freeze.objective_id,
            "objective_revision": self._freeze.objective_revision,
            "frozen_plan_id": self._freeze.plan_id,
            "admitted_revision": self._freeze.admitted_revision,
            "current_plan": self._plan.to_dict(),
            "current_receipts": {
                step_id: list(ids)
                for step_id, ids in sorted(self._current_receipts.items())
            },
        }
        payload["snapshot_id"] = content_identity(payload)
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_RECEDING_HORIZON_SNAPSHOT_BYTES:
            raise RecedingHorizonError("receding-horizon snapshot exceeds its bounded size")
        return MappingProxyType(payload)

    def snapshot_json(self) -> str:
        return canonical_json(dict(self.snapshot()))

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any] | str | bytes,
        *,
        replanner: FormalDeltaReplanner | None = None,
        failure_memory: PlanFailureMemory | None = None,
    ) -> RecedingHorizonController:
        if isinstance(snapshot, (bytes, str)):
            encoded = snapshot if isinstance(snapshot, bytes) else snapshot.encode("utf-8")
            if len(encoded) > MAX_RECEDING_HORIZON_SNAPSHOT_BYTES:
                raise RecedingHorizonError("receding-horizon snapshot exceeds its bounded size")
            duplicates: set[str] = set()

            def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                value: dict[str, Any] = {}
                for key, item in pairs:
                    if key in value:
                        duplicates.add(key)
                    value[key] = item
                return value

            try:
                raw = json.loads(encoded.decode("utf-8"), object_pairs_hook=pairs_hook)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RecedingHorizonError("receding-horizon snapshot is malformed") from exc
            if duplicates:
                raise RecedingHorizonError("receding-horizon snapshot contains duplicate fields")
        elif isinstance(snapshot, Mapping):
            raw = dict(snapshot)
        else:
            raise RecedingHorizonError("unsupported receding-horizon snapshot")
        if not isinstance(raw, Mapping):
            raise RecedingHorizonError("receding-horizon snapshot must contain an object")
        expected = {
            "schema",
            "program_id",
            "interface",
            "objective_id",
            "objective_revision",
            "frozen_plan_id",
            "admitted_revision",
            "current_plan",
            "current_receipts",
            "snapshot_id",
        }
        if set(raw) != expected:
            raise RecedingHorizonError("receding-horizon snapshot contains missing or unknown fields")
        if raw["schema"] != RECEDING_HORIZON_SNAPSHOT_SCHEMA:
            raise RecedingHorizonError("receding-horizon snapshot schema mismatch")
        if raw["program_id"] != AUTONOMOUS_META_CONTROLLER_PROGRAM_ID:
            raise RecedingHorizonError("receding-horizon program identity mismatch")
        if raw["interface"] != RECEDING_HORIZON_CONTROLLER_INTERFACE:
            raise RecedingHorizonError("receding-horizon interface mismatch")
        claimed = raw.pop("snapshot_id")
        if claimed != content_identity(raw):
            raise RecedingHorizonError("receding-horizon snapshot identity mismatch")
        return cls(
            objective_id=raw["objective_id"],
            objective_revision=raw["objective_revision"],
            plan=raw["current_plan"],
            current_receipts=raw["current_receipts"],
            replanner=replanner,
            failure_memory=failure_memory,
            admitted_revision=raw["admitted_revision"],
            frozen_plan_id=raw["frozen_plan_id"],
        )


__all__ = [
    "MAX_RECEDING_HORIZON_SNAPSHOT_BYTES",
    "NEAREST_SAFE_SEGMENT_SCHEMA",
    "PLAN_SUFFIX_INVALIDATION_RECEIPT_INTERFACE",
    "PLAN_SUFFIX_INVALIDATION_RECEIPT_SCHEMA",
    "RECEDING_HORIZON_CONTROLLER_INTERFACE",
    "RECEDING_HORIZON_EVIDENCE_SCHEMA",
    "RECEDING_HORIZON_SNAPSHOT_SCHEMA",
    "FrozenObjectivePlan",
    "NearestSafeSegment",
    "PlanSuffixInvalidationReceipt",
    "RecedingHorizonController",
    "RecedingHorizonDisposition",
    "RecedingHorizonError",
    "RecedingHorizonEvidence",
    "RecedingHorizonEvidenceKind",
    "nearest_safe_segment_ids",
    "preserved_step_receipt_ids",
    "select_nearest_safe_segment",
]
