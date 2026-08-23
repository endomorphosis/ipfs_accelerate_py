"""Adapter that exposes verified procedures as AdaptivePlanner operators.

This module owns matching, composition, and planner-order ranking.  It does
not replace AdaptivePlanner, reinterpret its receipts, or grant authority.
Procedure dispatch is attempted only after a live AdaptivePlanner import
qualifies.  An unresolved import incompatibility is a typed-unavailable
result; matching, composition, and the rest of the procedure-compiler
runtime remain usable.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .contracts import (
    ArtifactBindings,
    ConditionOperator,
    EffectClass,
    ProcedureContractError,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    ProcedureSpec,
    ProcedureValidationPlan,
    RiskClass,
    _enum,
    _identifier,
    _nested,
    _strings,
)


ADAPTER_REVISION: Final[str] = "ProcedurePlannerAdapter@1"
ADAPTER_INTERFACE: Final[str] = "ProcedurePlannerAdapter@1"
OPERATOR_INTERFACE: Final[str] = "ProcedureOperator@1"
COMPOSITION_VALIDATOR_INTERFACE: Final[str] = "ProcedureCompositionValidator@1"
ADAPTIVE_PLANNER_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner"
)
ADAPTIVE_PLANNER_SYMBOL: Final[str] = "AdaptivePlanner"
HAMMER_TRACE_SCHEMA_NAME: Final[str] = "HAMMER_TRACE_SCHEMA"
ADAPTIVE_PLANNER_HAMMER_TRACE_REASON_CODE: Final[str] = (
    "adaptive_planner_import_undefined_hammer_trace_schema"
)
MAX_COMPOSITION_OPERATORS: Final[int] = 8
MAX_PLAN_CANDIDATES: Final[int] = 32

_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}
_EFFECT_RANK: Final[dict[EffectClass, int]] = {
    EffectClass.OBSERVE: 0,
    EffectClass.VALIDATION: 1,
    EffectClass.PROOF: 1,
    EffectClass.ARTIFACT_PERSIST: 1,
    EffectClass.RECEIPT_EMIT: 1,
    EffectClass.WORKTREE_CREATE: 2,
    EffectClass.MODEL_REQUEST: 2,
    EffectClass.ROLLBACK: 3,
    EffectClass.REPOSITORY_WRITE: 4,
    EffectClass.MERGE_PREPARE: 5,
    EffectClass.MERGE: 6,
    EffectClass.ESCALATION: 7,
}
_EFFECT_MIN_RISK: Final[dict[EffectClass, RiskClass]] = {
    EffectClass.OBSERVE: RiskClass.OBSERVATION_ONLY,
    EffectClass.VALIDATION: RiskClass.OBSERVATION_ONLY,
    EffectClass.PROOF: RiskClass.OBSERVATION_ONLY,
    EffectClass.ARTIFACT_PERSIST: RiskClass.OBSERVATION_ONLY,
    EffectClass.RECEIPT_EMIT: RiskClass.OBSERVATION_ONLY,
    EffectClass.WORKTREE_CREATE: RiskClass.REVERSIBLE_LOCAL,
    EffectClass.MODEL_REQUEST: RiskClass.REVERSIBLE_LOCAL,
    EffectClass.ROLLBACK: RiskClass.REVERSIBLE_LOCAL,
    EffectClass.REPOSITORY_WRITE: RiskClass.REPOSITORY_WRITE,
    EffectClass.MERGE_PREPARE: RiskClass.PUBLIC_CONTRACT,
    EffectClass.MERGE: RiskClass.PUBLIC_CONTRACT,
    EffectClass.ESCALATION: RiskClass.AUTHORITY_OR_SECURITY,
}
_WRITE_EFFECTS: Final[frozenset[EffectClass]] = frozenset(
    {
        EffectClass.WORKTREE_CREATE,
        EffectClass.REPOSITORY_WRITE,
        EffectClass.MERGE_PREPARE,
        EffectClass.MERGE,
    }
)
_EXACT_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "repository_commit",
    "tree_id",
    "contract_revision",
    "policy_revision",
    "environment_id",
)


class PlannerAdapterError(ProcedureContractError):
    """A planner-adapter request, operator, or composition is unsafe."""


class PlannerCompatibilityStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    QUALIFIED = "qualified"
    TYPED_UNAVAILABLE = "typed_unavailable"
    INCOMPATIBLE = "incompatible"


class PlannerDispatchStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    CANDIDATES = "candidates"
    NO_MATCH = "no-match"
    REFUSED = "refused"
    TYPED_UNAVAILABLE = "typed_unavailable"


class PlannerStage(str, Enum):  # noqa: UP042 - package supports Python 3.8
    EXACT_VERIFIED_PROCEDURE = "exact_verified_procedure"
    COMPOSABLE_VERIFIED_PROCEDURES = "composable_verified_procedures"
    DETERMINISTIC_BASELINE = "deterministic_baseline"
    BOUNDED_LOCAL_SYNTHESIS = "bounded_local_synthesis"
    SMALL_LOCAL_MODEL = "small_local_model"
    STANDARD_REMOTE_MODEL = "standard_remote_model"
    STRONG_REMOTE_MODEL = "strong_remote_model"
    HUMAN_ESCALATION = "human_escalation"


REQUIRED_PLANNER_ORDER: Final[tuple[PlannerStage, ...]] = (
    PlannerStage.EXACT_VERIFIED_PROCEDURE,
    PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES,
    PlannerStage.DETERMINISTIC_BASELINE,
    PlannerStage.BOUNDED_LOCAL_SYNTHESIS,
    PlannerStage.SMALL_LOCAL_MODEL,
    PlannerStage.STANDARD_REMOTE_MODEL,
    PlannerStage.STRONG_REMOTE_MODEL,
    PlannerStage.HUMAN_ESCALATION,
)
PROCEDURE_STAGES: Final[frozenset[PlannerStage]] = frozenset(
    {
        PlannerStage.EXACT_VERIFIED_PROCEDURE,
        PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES,
    }
)


class PlanningNeedKind(str, Enum):  # noqa: UP042 - package supports Python 3.8
    TASK = "task"
    CRITERION = "criterion"
    SUBGOAL = "subgoal"
    REPAIR_SUFFIX = "repair_suffix"
    VALIDATION_STAGE = "validation_stage"


class MatchDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    EXACT = "exact"
    PARTIAL_CRITERION = "partial-criterion"
    INCOMPATIBLE = "incompatible"
    OVERCLAIM = "overclaim"


class CompositionReason(str, Enum):  # noqa: UP042 - package supports Python 3.8
    ACCEPTED = "accepted"
    EMPTY = "empty"
    BOUND_EXCEEDED = "bound-exceeded"
    UNVERIFIED = "unverified"
    ENTAILMENT_FAILURE = "entailment-failure"
    EFFECT_INCOMPATIBLE = "effect-incompatible"
    AUTHORITY_INCOMPATIBLE = "authority-incompatible"
    ENVIRONMENT_INCOMPATIBLE = "environment-incompatible"
    BUDGET_OVERFLOW = "budget-overflow"
    ROLLBACK_UNDEFINED = "rollback-undefined"
    VALIDATION_INCOMPLETE = "validation-incomplete"
    CYCLE = "cycle"
    HIDDEN_EFFECT_ESCALATION = "hidden-effect-escalation"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise PlannerAdapterError("{} must be a boolean".format(field_name))
    return value


def _need_kind(value: Any) -> PlanningNeedKind:
    return _enum(value, PlanningNeedKind, "need_kind")


def _risk_rank(value: RiskClass) -> int:
    return _RISK_RANK[value]


def _effect_rank(value: EffectClass) -> int:
    return _EFFECT_RANK[value]


def _max_risk(values: Sequence[RiskClass]) -> RiskClass:
    return max(values, key=_risk_rank)


def _operand_key(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _operand_key(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_operand_key(item) for item in value)
    return value


def _condition_key(condition: ProcedurePrecondition | ProcedurePostcondition) -> tuple[Any, ...]:
    return (
        condition.binding,
        condition.operator.value,
        _operand_key(condition.operand),
    )


def _exact_bindings(left: ArtifactBindings, right: ArtifactBindings) -> bool:
    return all(
        getattr(left, name) == getattr(right, name) for name in _EXACT_BINDING_FIELDS
    )


def _effect_classes(spec: ProcedureSpec) -> tuple[EffectClass, ...]:
    return tuple(dict.fromkeys(item.effect_class for item in spec.declared_effects))


def _condition_entails(
    postcondition: ProcedurePostcondition,
    precondition: ProcedurePrecondition,
) -> bool:
    if _condition_key(postcondition) != _condition_key(precondition):
        return False
    if precondition.required and not (
        postcondition.evidence_producer and postcondition.evidence_type
    ):
        return False
    if (
        precondition.operator
        in {
            ConditionOperator.EQUALS,
            ConditionOperator.CID_EQUALS,
            ConditionOperator.CURRENT,
            ConditionOperator.ADMITTED,
        }
        and postcondition.operator != precondition.operator
    ):
        return False
    return True


def _post_entails_pre(
    predecessor: ProcedureSpec,
    successor: ProcedureSpec,
) -> tuple[bool, tuple[str, ...]]:
    missing: list[str] = []
    for precondition in successor.preconditions:
        if not precondition.required:
            continue
        if not any(
            _condition_entails(postcondition, precondition)
            for postcondition in predecessor.postconditions
        ):
            missing.append(precondition.condition_id)
    return (not missing, tuple(missing))


_RESOURCE_FIELDS: Final[tuple[str, ...]] = (
    "wall_time_ms",
    "cpu_time_ms",
    "memory_bytes",
    "disk_bytes",
    "model_token_limit",
    "model_call_limit",
    "subprocess_limit",
    "network_request_limit",
)


def _add_resources(
    total: Mapping[str, int], envelope: ProcedureResourceEnvelope
) -> dict[str, int]:
    return {name: total[name] + getattr(envelope, name) for name in _RESOURCE_FIELDS}


def _resource_exceeds(total: Mapping[str, int], budget: ProcedureResourceEnvelope) -> bool:
    return any(total[name] > getattr(budget, name) for name in _RESOURCE_FIELDS)


def _exception_diagnostic(exc: BaseException) -> str:
    return "{}: {}".format(type(exc).__name__, exc)


def _exception_chain_diagnostic(exc: BaseException) -> str:
    parts = [_exception_diagnostic(exc)]
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.__cause__ or current.__context__
        if current is not None:
            parts.append(_exception_diagnostic(current))
    return " | ".join(parts)


@dataclass(frozen=True)
class AdaptivePlannerCompatibility:
    """Live AdaptivePlanner import disposition.  Never simulated."""

    status: PlannerCompatibilityStatus
    reason_code: str
    diagnostic: str
    module_name: str = ADAPTIVE_PLANNER_MODULE
    symbol: str = ADAPTIVE_PLANNER_SYMBOL
    planner_version: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum(self.status, PlannerCompatibilityStatus, "status"),
        )
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "diagnostic", str(self.diagnostic or ""))
        object.__setattr__(
            self, "module_name", _identifier(self.module_name, "module_name")
        )
        object.__setattr__(self, "symbol", _identifier(self.symbol, "symbol"))
        if self.qualified and self.planner_version is None:
            raise PlannerAdapterError("qualified AdaptivePlanner must bind a version")
        if self.qualified and self.status is PlannerCompatibilityStatus.TYPED_UNAVAILABLE:
            raise PlannerAdapterError("qualified compatibility cannot be typed unavailable")

    @property
    def qualified(self) -> bool:
        return self.status is PlannerCompatibilityStatus.QUALIFIED

    @property
    def typed_unavailable(self) -> bool:
        return self.status is PlannerCompatibilityStatus.TYPED_UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostic": self.diagnostic,
            "module_name": self.module_name,
            "planner_version": self.planner_version,
            "reason_code": self.reason_code,
            "status": self.status.value,
            "symbol": self.symbol,
            "typed_unavailable": self.typed_unavailable,
        }


def probe_adaptive_planner() -> AdaptivePlannerCompatibility:
    """Re-evaluate the committed AdaptivePlanner import on this tree.

    A HAMMER_TRACE_SCHEMA NameError is the declared typed-unavailable path.
    The adapter never recreates AdaptivePlanner and never treats a similar
    name as a passing qualification.
    """

    try:
        module = importlib.import_module(ADAPTIVE_PLANNER_MODULE)
    except Exception as exc:
        diagnostic = _exception_chain_diagnostic(exc)
        if HAMMER_TRACE_SCHEMA_NAME in diagnostic:
            reason = ADAPTIVE_PLANNER_HAMMER_TRACE_REASON_CODE
        elif isinstance(exc, NameError):
            reason = "adaptive_planner_import_name_error"
        else:
            reason = "adaptive_planner_import_failed"
        return AdaptivePlannerCompatibility(
            status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
            reason_code=reason,
            diagnostic=diagnostic,
        )

    planner_cls = getattr(module, ADAPTIVE_PLANNER_SYMBOL, None)
    version = getattr(module, "ADAPTIVE_PLANNER_VERSION", None)
    if planner_cls is None or not isinstance(planner_cls, type):
        return AdaptivePlannerCompatibility(
            status=PlannerCompatibilityStatus.INCOMPATIBLE,
            reason_code="adaptive_planner_symbol_missing",
            diagnostic="{} does not export {}".format(
                ADAPTIVE_PLANNER_MODULE, ADAPTIVE_PLANNER_SYMBOL
            ),
        )
    if type(version) is not int:
        return AdaptivePlannerCompatibility(
            status=PlannerCompatibilityStatus.INCOMPATIBLE,
            reason_code="adaptive_planner_version_missing",
            diagnostic="ADAPTIVE_PLANNER_VERSION is not an integer",
        )
    try:
        planner_cls()
    except Exception as exc:
        return AdaptivePlannerCompatibility(
            status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
            reason_code="adaptive_planner_construct_failed",
            diagnostic=_exception_diagnostic(exc),
        )
    return AdaptivePlannerCompatibility(
        status=PlannerCompatibilityStatus.QUALIFIED,
        reason_code="adaptive_planner_qualified",
        diagnostic="",
        planner_version=version,
    )


@dataclass(frozen=True)
class ProcedureOperator:
    """Verified procedure offered as a planning operator, never as authority."""

    procedure: ProcedureSpec
    need_kind: PlanningNeedKind
    satisfied_ids: tuple[str, ...]
    verified: bool
    promoted: bool
    certificate_cid: str = ""
    language_classes: tuple[str, ...] = ()
    framework_classes: tuple[str, ...] = ()
    repository_families: tuple[str, ...] = ()
    interface: str = OPERATOR_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "procedure", _nested(self.procedure, ProcedureSpec, "procedure")
        )
        object.__setattr__(self, "need_kind", _need_kind(self.need_kind))
        object.__setattr__(
            self,
            "satisfied_ids",
            _strings(self.satisfied_ids, "satisfied_ids", identifiers=True, required=True),
        )
        object.__setattr__(self, "verified", _bool(self.verified, "verified"))
        object.__setattr__(self, "promoted", _bool(self.promoted, "promoted"))
        object.__setattr__(
            self,
            "certificate_cid",
            _identifier(self.certificate_cid, "certificate_cid", required=False),
        )
        for name in ("language_classes", "framework_classes", "repository_families"):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        object.__setattr__(self, "interface", _identifier(self.interface, "interface"))
        if self.interface != OPERATOR_INTERFACE:
            raise PlannerAdapterError("unsupported procedure operator interface")
        if self.promoted and not self.verified:
            raise PlannerAdapterError("a promoted operator must already be verified")

    @property
    def procedure_id(self) -> str:
        return self.procedure.name

    @property
    def procedure_cid(self) -> str:
        return self.procedure.content_id

    @property
    def usable(self) -> bool:
        return self.verified and self.promoted

    @property
    def claims_task(self) -> bool:
        return self.need_kind is PlanningNeedKind.TASK

    @property
    def effect_classes(self) -> tuple[EffectClass, ...]:
        return _effect_classes(self.procedure)

    @property
    def max_effect_rank(self) -> int:
        classes = self.effect_classes
        if not classes:
            return 0
        return max(_effect_rank(item) for item in classes)

    def claims(self, need_kind: PlanningNeedKind, need_id: str) -> bool:
        if self.need_kind is not need_kind:
            return False
        return need_id in self.satisfied_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_cid": self.certificate_cid,
            "claims_task": self.claims_task,
            "framework_classes": self.framework_classes,
            "interface": self.interface,
            "language_classes": self.language_classes,
            "need_kind": self.need_kind.value,
            "procedure_cid": self.procedure_cid,
            "procedure_id": self.procedure_id,
            "promoted": self.promoted,
            "repository_families": self.repository_families,
            "satisfied_ids": self.satisfied_ids,
            "usable": self.usable,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class PlanningBoundary:
    """Exact compatibility surface a procedure may occupy."""

    bindings: ArtifactBindings
    task_family_id: str
    need_kind: PlanningNeedKind
    need_id: str
    criterion_ids: tuple[str, ...] = ()
    allowed_effect_classes: tuple[EffectClass, ...] = ()
    risk_ceiling: RiskClass = RiskClass.OBSERVATION_ONLY
    required_capability_ids: tuple[str, ...] = ()
    language_classes: tuple[str, ...] = ()
    framework_classes: tuple[str, ...] = ()
    repository_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings")
        )
        object.__setattr__(
            self, "task_family_id", _identifier(self.task_family_id, "task_family_id")
        )
        object.__setattr__(self, "need_kind", _need_kind(self.need_kind))
        object.__setattr__(self, "need_id", _identifier(self.need_id, "need_id"))
        object.__setattr__(
            self,
            "criterion_ids",
            _strings(self.criterion_ids, "criterion_ids", identifiers=True),
        )
        if self.allowed_effect_classes:
            object.__setattr__(
                self,
                "allowed_effect_classes",
                tuple(
                    _enum(item, EffectClass, "allowed_effect_classes")
                    for item in self.allowed_effect_classes
                ),
            )
        object.__setattr__(
            self, "risk_ceiling", _enum(self.risk_ceiling, RiskClass, "risk_ceiling")
        )
        object.__setattr__(
            self,
            "required_capability_ids",
            _strings(
                self.required_capability_ids,
                "required_capability_ids",
                identifiers=True,
            ),
        )
        for name in ("language_classes", "framework_classes", "repository_families"):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )


@dataclass(frozen=True)
class PlanningRequest:
    """One planning need.  Procedures may satisfy it only on this boundary."""

    boundary: PlanningBoundary
    resource_budget: ProcedureResourceEnvelope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "boundary", _nested(self.boundary, PlanningBoundary, "boundary")
        )
        if self.resource_budget is not None:
            object.__setattr__(
                self,
                "resource_budget",
                _nested(self.resource_budget, ProcedureResourceEnvelope, "resource_budget"),
            )


@dataclass(frozen=True)
class MatchResult:
    disposition: MatchDisposition
    operator: ProcedureOperator
    reason_code: str
    satisfied_ids: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return self.disposition is MatchDisposition.EXACT

    @property
    def accepted(self) -> bool:
        return self.disposition in {
            MatchDisposition.EXACT,
            MatchDisposition.PARTIAL_CRITERION,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "operator": self.operator.to_dict(),
            "reason_code": self.reason_code,
            "satisfied_ids": self.satisfied_ids,
        }


@dataclass(frozen=True)
class EntailmentEvidence:
    predecessor_id: str
    successor_id: str
    matched_condition_ids: tuple[str, ...]
    missing_condition_ids: tuple[str, ...]

    @property
    def exact(self) -> bool:
        return not self.missing_condition_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact": self.exact,
            "matched_condition_ids": self.matched_condition_ids,
            "missing_condition_ids": self.missing_condition_ids,
            "predecessor_id": self.predecessor_id,
            "successor_id": self.successor_id,
        }


@dataclass(frozen=True)
class CompositionDecision:
    accepted: bool
    reason: CompositionReason
    operators: tuple[ProcedureOperator, ...]
    entailment: tuple[EntailmentEvidence, ...] = ()
    composed_effects: tuple[EffectClass, ...] = ()
    composed_risk_ceiling: RiskClass | None = None
    composed_resources: Mapping[str, int] | None = None
    composed_rollback: tuple[ProcedureRollback, ...] = ()
    composed_validation: ProcedureValidationPlan | None = None
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "composed_effects": tuple(item.value for item in self.composed_effects),
            "composed_resources": (
                dict(self.composed_resources) if self.composed_resources else None
            ),
            "composed_risk_ceiling": (
                self.composed_risk_ceiling.value if self.composed_risk_ceiling else None
            ),
            "diagnostic": self.diagnostic,
            "entailment": tuple(item.to_dict() for item in self.entailment),
            "operator_ids": tuple(item.procedure_id for item in self.operators),
            "reason": self.reason.value,
        }


@dataclass(frozen=True)
class ProcedurePlanCandidate:
    stage: PlannerStage
    operators: tuple[ProcedureOperator, ...]
    composition: CompositionDecision | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _enum(self.stage, PlannerStage, "stage"))
        if self.stage not in PROCEDURE_STAGES and self.operators:
            raise PlannerAdapterError(
                "non-procedure planner stages cannot carry procedure operators"
            )
        if self.stage is PlannerStage.EXACT_VERIFIED_PROCEDURE and len(self.operators) != 1:
            raise PlannerAdapterError("exact verified procedure candidates carry one operator")

    @property
    def claimed_need_kinds(self) -> tuple[PlanningNeedKind, ...]:
        return tuple(item.need_kind for item in self.operators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claimed_need_kinds": tuple(item.value for item in self.claimed_need_kinds),
            "operator_ids": tuple(item.procedure_id for item in self.operators),
            "stage": self.stage.value,
        }


@dataclass(frozen=True)
class PlannerAdapterResult:
    status: PlannerDispatchStatus
    reason_code: str
    compatibility: AdaptivePlannerCompatibility
    candidates: tuple[ProcedurePlanCandidate, ...] = ()
    matches: tuple[MatchResult, ...] = ()
    selected_stage: PlannerStage | None = None
    planner_order: tuple[PlannerStage, ...] = REQUIRED_PLANNER_ORDER

    @property
    def typed_unavailable(self) -> bool:
        return self.status is PlannerDispatchStatus.TYPED_UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": tuple(item.to_dict() for item in self.candidates),
            "compatibility": self.compatibility.to_dict(),
            "planner_order": tuple(item.value for item in self.planner_order),
            "reason_code": self.reason_code,
            "selected_stage": None if self.selected_stage is None else self.selected_stage.value,
            "status": self.status.value,
            "typed_unavailable": self.typed_unavailable,
        }


class ProcedureCompositionValidator:
    """Fail-closed composition of verified procedure operators."""

    interface: Final[str] = COMPOSITION_VALIDATOR_INTERFACE

    def validate(
        self,
        operators: Sequence[ProcedureOperator],
        *,
        resource_budget: ProcedureResourceEnvelope | None = None,
        claimed_risk_ceiling: RiskClass | None = None,
        claimed_effect_classes: Sequence[EffectClass] | None = None,
    ) -> CompositionDecision:
        if isinstance(operators, (str, bytes, bytearray)):
            raise PlannerAdapterError("operators must be a sequence of ProcedureOperator")
        records = tuple(operators)
        if any(not isinstance(item, ProcedureOperator) for item in records):
            raise PlannerAdapterError("operators must be ProcedureOperator instances")
        if not records:
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.EMPTY,
                operators=(),
                diagnostic="composition requires at least one operator",
            )
        if len(records) > MAX_COMPOSITION_OPERATORS:
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.BOUND_EXCEEDED,
                operators=records,
                diagnostic="composition operator bound exhausted",
            )
        if any(not item.usable for item in records):
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.UNVERIFIED,
                operators=records,
                diagnostic="composition requires verified promoted operators",
            )

        ids = tuple(item.procedure_id for item in records)
        cids = tuple(item.procedure_cid for item in records)
        if len(set(ids)) != len(ids) or len(set(cids)) != len(cids):
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.CYCLE,
                operators=records,
                diagnostic="composition repeats a procedure and would cycle",
            )

        baseline = records[0].procedure.bindings
        if any(not _exact_bindings(item.procedure.bindings, baseline) for item in records[1:]):
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.ENVIRONMENT_INCOMPATIBLE,
                operators=records,
                diagnostic="composed operators require exact environment and tree bindings",
            )

        authority_decision = self._authority(records)
        if authority_decision is not None:
            return authority_decision

        hidden = self._hidden_escalation(
            records,
            claimed_risk_ceiling=claimed_risk_ceiling,
            claimed_effect_classes=claimed_effect_classes,
        )
        if hidden is not None:
            return hidden

        effect_decision = self._effects(records, claimed_effect_classes=claimed_effect_classes)
        if effect_decision is not None:
            return effect_decision

        entailment: list[EntailmentEvidence] = []
        for predecessor, successor in zip(records, records[1:]):
            matched, missing = _post_entails_pre(predecessor.procedure, successor.procedure)
            evidence = EntailmentEvidence(
                predecessor_id=predecessor.procedure_id,
                successor_id=successor.procedure_id,
                matched_condition_ids=tuple(
                    item.condition_id
                    for item in successor.procedure.preconditions
                    if item.condition_id not in missing
                ),
                missing_condition_ids=missing,
            )
            entailment.append(evidence)
            if not matched:
                return CompositionDecision(
                    accepted=False,
                    reason=CompositionReason.ENTAILMENT_FAILURE,
                    operators=records,
                    entailment=tuple(entailment),
                    diagnostic="postconditions of {} do not entail preconditions of {}".format(
                        predecessor.procedure_id, successor.procedure_id
                    ),
                )

        resources = self._resources(records, resource_budget=resource_budget)
        if isinstance(resources, CompositionDecision):
            return resources

        rollback = self._rollback(records)
        if isinstance(rollback, CompositionDecision):
            return rollback

        validation = self._validation(records)
        if isinstance(validation, CompositionDecision):
            return validation

        composed_effects = tuple(
            sorted(
                {item for operator in records for item in operator.effect_classes},
                key=lambda item: ( _effect_rank(item), item.value),
            )
        )
        return CompositionDecision(
            accepted=True,
            reason=CompositionReason.ACCEPTED,
            operators=records,
            entailment=tuple(entailment),
            composed_effects=composed_effects,
            composed_risk_ceiling=_max_risk(
                tuple(item.procedure.authority.risk_ceiling for item in records)
            ),
            composed_resources=MappingProxyType(resources),
            composed_rollback=rollback,
            composed_validation=validation,
        )

    def _authority(
        self, records: tuple[ProcedureOperator, ...]
    ) -> CompositionDecision | None:
        policies = {item.procedure.authority.authority_policy_revision for item in records}
        if len(policies) != 1:
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.AUTHORITY_INCOMPATIBLE,
                operators=records,
                diagnostic="composed operators require one authority policy revision",
            )
        return None

    def _hidden_escalation(
        self,
        records: tuple[ProcedureOperator, ...],
        *,
        claimed_risk_ceiling: RiskClass | None,
        claimed_effect_classes: Sequence[EffectClass] | None,
    ) -> CompositionDecision | None:
        honest_risk = _max_risk(
            tuple(item.procedure.authority.risk_ceiling for item in records)
        )
        honest_effects = {
            effect
            for operator in records
            for effect in operator.effect_classes
        }
        for operator in records:
            for effect in operator.effect_classes:
                required = _EFFECT_MIN_RISK[effect]
                if _risk_rank(operator.procedure.authority.risk_ceiling) < _risk_rank(required):
                    return CompositionDecision(
                        accepted=False,
                        reason=CompositionReason.HIDDEN_EFFECT_ESCALATION,
                        operators=records,
                        diagnostic=(
                            "{} declares {} under a weaker risk ceiling".format(
                                operator.procedure_id, effect.value
                            )
                        ),
                    )
        if claimed_risk_ceiling is not None and _risk_rank(claimed_risk_ceiling) < _risk_rank(
            honest_risk
        ):
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.HIDDEN_EFFECT_ESCALATION,
                operators=records,
                diagnostic="claimed risk ceiling hides composed effect authority",
            )
        if claimed_effect_classes is not None:
            claimed = {
                _enum(item, EffectClass, "claimed_effect_classes")
                for item in claimed_effect_classes
            }
            hidden = honest_effects - claimed
            if hidden:
                return CompositionDecision(
                    accepted=False,
                    reason=CompositionReason.HIDDEN_EFFECT_ESCALATION,
                    operators=records,
                    diagnostic="claimed effects omit {}".format(
                        ",".join(sorted(item.value for item in hidden))
                    ),
                )
        prefix_rank = records[0].max_effect_rank
        for operator in records[1:]:
            if operator.max_effect_rank > prefix_rank and claimed_effect_classes is not None:
                claimed_rank = max(
                    (_effect_rank(item) for item in claimed_effect_classes),
                    default=0,
                )
                if claimed_rank < operator.max_effect_rank:
                    return CompositionDecision(
                        accepted=False,
                        reason=CompositionReason.HIDDEN_EFFECT_ESCALATION,
                        operators=records,
                        diagnostic="later operator escalates effects without declaration",
                    )
            prefix_rank = max(prefix_rank, operator.max_effect_rank)
        return None

    def _effects(
        self,
        records: tuple[ProcedureOperator, ...],
        *,
        claimed_effect_classes: Sequence[EffectClass] | None,
    ) -> CompositionDecision | None:
        if claimed_effect_classes is None:
            return None
        claimed = {
            _enum(item, EffectClass, "claimed_effect_classes") for item in claimed_effect_classes
        }
        honest = {item for operator in records for item in operator.effect_classes}
        extra = claimed - honest
        if extra:
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.EFFECT_INCOMPATIBLE,
                operators=records,
                diagnostic="claimed effects are not declared by composed operators",
            )
        return None

    def _resources(
        self,
        records: tuple[ProcedureOperator, ...],
        *,
        resource_budget: ProcedureResourceEnvelope | None,
    ) -> dict[str, int] | CompositionDecision:
        total = {name: 0 for name in _RESOURCE_FIELDS}
        for operator in records:
            total = _add_resources(total, operator.procedure.resources)
        if resource_budget is not None and _resource_exceeds(total, resource_budget):
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.BUDGET_OVERFLOW,
                operators=records,
                composed_resources=MappingProxyType(total),
                diagnostic="composed resources exceed the declared budget",
            )
        return total

    def _rollback(
        self, records: tuple[ProcedureOperator, ...]
    ) -> tuple[ProcedureRollback, ...] | CompositionDecision:
        write_effects = [
            effect
            for operator in records
            for effect in operator.procedure.declared_effects
            if effect.effect_class in _WRITE_EFFECTS
        ]
        rollbacks = tuple(
            item for operator in records for item in operator.procedure.rollback
        )
        if write_effects and not rollbacks:
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.ROLLBACK_UNDEFINED,
                operators=records,
                diagnostic="write effects require a defined composed rollback",
            )
        covered = {effect_id for item in rollbacks for effect_id in item.trigger_effect_ids}
        missing = [
            effect.effect_id
            for effect in write_effects
            if effect.effect_id not in covered
        ]
        if missing:
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.ROLLBACK_UNDEFINED,
                operators=records,
                diagnostic="composed rollback omits {}".format(",".join(missing)),
            )
        if any(not item.exact_target_cid for item in rollbacks):
            return CompositionDecision(
                accepted=False,
                reason=CompositionReason.ROLLBACK_UNDEFINED,
                operators=records,
                diagnostic="composed rollback requires an exact target",
            )
        return rollbacks

    def _validation(
        self, records: tuple[ProcedureOperator, ...]
    ) -> ProcedureValidationPlan | CompositionDecision:
        step_ids: list[str] = []
        observation_ids: list[str] = []
        test_contracts: list[str] = []
        proof_contracts: list[str] = []
        post_merge: list[str] = []
        fallback = ""
        for operator in records:
            plan = operator.procedure.validation
            if plan is None:
                return CompositionDecision(
                    accepted=False,
                    reason=CompositionReason.VALIDATION_INCOMPLETE,
                    operators=records,
                    diagnostic="composed operators must retain complete validation",
                )
            if not plan.required_step_ids or not plan.required_observation_ids:
                return CompositionDecision(
                    accepted=False,
                    reason=CompositionReason.VALIDATION_INCOMPLETE,
                    operators=records,
                    diagnostic="composed validation dropped required observations or steps",
                )
            step_ids.extend(plan.required_step_ids)
            observation_ids.extend(plan.required_observation_ids)
            test_contracts.extend(plan.required_test_contracts)
            proof_contracts.extend(plan.required_proof_contracts)
            post_merge.extend(plan.post_merge_validation_contracts)
            if plan.full_test_fallback_contract:
                fallback = plan.full_test_fallback_contract
        return ProcedureValidationPlan(
            required_step_ids=tuple(dict.fromkeys(step_ids)),
            required_observation_ids=tuple(dict.fromkeys(observation_ids)),
            required_test_contracts=tuple(dict.fromkeys(test_contracts)),
            required_proof_contracts=tuple(dict.fromkeys(proof_contracts)),
            post_merge_validation_contracts=tuple(dict.fromkeys(post_merge)),
            full_test_fallback_contract=fallback,
        )


class ProcedurePlannerAdapter:
    """Produce procedure plan candidates without owning AdaptivePlanner."""

    revision: Final[str] = ADAPTER_REVISION
    interface: Final[str] = ADAPTER_INTERFACE

    def __init__(
        self,
        operators: Sequence[ProcedureOperator] = (),
        *,
        validator: ProcedureCompositionValidator | None = None,
        planner_probe: Callable[[], AdaptivePlannerCompatibility] = probe_adaptive_planner,
    ) -> None:
        if isinstance(operators, (str, bytes, bytearray)):
            raise PlannerAdapterError("operators must be a sequence of ProcedureOperator")
        records = tuple(operators)
        if any(not isinstance(item, ProcedureOperator) for item in records):
            raise PlannerAdapterError("operators must be ProcedureOperator instances")
        if len(records) > MAX_PLAN_CANDIDATES:
            raise PlannerAdapterError("operator catalog bound exhausted")
        ids = [item.procedure_id for item in records]
        if len(set(ids)) != len(ids):
            raise PlannerAdapterError("operator procedure ids must be unique")
        if not callable(planner_probe):
            raise PlannerAdapterError("planner_probe must be callable")
        self._operators = records
        self._validator = validator or ProcedureCompositionValidator()
        self._planner_probe = planner_probe
        self._compatibility: AdaptivePlannerCompatibility | None = None

    @property
    def planner_order(self) -> tuple[PlannerStage, ...]:
        return REQUIRED_PLANNER_ORDER

    def compatibility(self, *, refresh: bool = False) -> AdaptivePlannerCompatibility:
        if self._compatibility is None or refresh:
            self._compatibility = self._planner_probe()
            if not isinstance(self._compatibility, AdaptivePlannerCompatibility):
                raise PlannerAdapterError(
                    "planner_probe must return AdaptivePlannerCompatibility"
                )
        return self._compatibility

    def match(self, request: PlanningRequest) -> tuple[MatchResult, ...]:
        if not isinstance(request, PlanningRequest):
            raise PlannerAdapterError("request must be a PlanningRequest")
        results: list[MatchResult] = []
        for operator in self._operators:
            results.append(self._match_one(operator, request))
        return tuple(results)

    def compose(
        self,
        operators: Sequence[ProcedureOperator],
        *,
        resource_budget: ProcedureResourceEnvelope | None = None,
        claimed_risk_ceiling: RiskClass | None = None,
        claimed_effect_classes: Sequence[EffectClass] | None = None,
    ) -> CompositionDecision:
        return self._validator.validate(
            operators,
            resource_budget=resource_budget,
            claimed_risk_ceiling=claimed_risk_ceiling,
            claimed_effect_classes=claimed_effect_classes,
        )

    def plan_candidates(self, request: PlanningRequest) -> PlannerAdapterResult:
        compatibility = self.compatibility()
        matches = self.match(request)
        candidates: list[ProcedurePlanCandidate] = []
        exact = tuple(
            item
            for item in matches
            if item.disposition is MatchDisposition.EXACT and item.operator.usable
        )
        for item in exact:
            candidates.append(
                ProcedurePlanCandidate(
                    stage=PlannerStage.EXACT_VERIFIED_PROCEDURE,
                    operators=(item.operator,),
                )
            )
        if not exact:
            composition = self._compose_covering(request)
            if composition is not None and composition.accepted:
                candidates.append(
                    ProcedurePlanCandidate(
                        stage=PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES,
                        operators=composition.operators,
                        composition=composition,
                    )
                )
        if len(candidates) > MAX_PLAN_CANDIDATES:
            raise PlannerAdapterError("plan candidate bound exhausted")
        if not candidates:
            return PlannerAdapterResult(
                status=PlannerDispatchStatus.NO_MATCH,
                reason_code="no-compatible-procedure",
                compatibility=compatibility,
                matches=matches,
            )
        selected = candidates[0].stage
        return PlannerAdapterResult(
            status=PlannerDispatchStatus.CANDIDATES,
            reason_code="procedure-candidates",
            compatibility=compatibility,
            candidates=tuple(candidates),
            matches=matches,
            selected_stage=selected,
        )

    def dispatch(self, request: PlanningRequest) -> PlannerAdapterResult:
        """Gate procedure use on a live AdaptivePlanner qualification.

        Matching and composition stay available through :meth:`match` and
        :meth:`plan_candidates`.  Dispatch itself refuses when the planner
        import is typed unavailable so other runtime can keep running.
        """

        compatibility = self.compatibility(refresh=True)
        if not compatibility.qualified:
            return PlannerAdapterResult(
                status=PlannerDispatchStatus.TYPED_UNAVAILABLE,
                reason_code=compatibility.reason_code,
                compatibility=compatibility,
            )
        return self.plan_candidates(request)

    def _match_one(self, operator: ProcedureOperator, request: PlanningRequest) -> MatchResult:
        boundary = request.boundary
        if not operator.usable:
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="operator-not-usable",
            )
        if operator.procedure.task_family_id != boundary.task_family_id:
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="task-family-mismatch",
            )
        if not _exact_bindings(operator.procedure.bindings, boundary.bindings):
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="incompatible-boundary",
            )
        if boundary.allowed_effect_classes:
            extra = set(operator.effect_classes) - set(boundary.allowed_effect_classes)
            if extra:
                return MatchResult(
                    disposition=MatchDisposition.INCOMPATIBLE,
                    operator=operator,
                    reason_code="effect-not-permitted",
                )
        if _risk_rank(operator.procedure.authority.risk_ceiling) > _risk_rank(
            boundary.risk_ceiling
        ):
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="risk-ceiling",
            )
        required_caps = set(boundary.required_capability_ids)
        if required_caps and not required_caps.issubset(
            set(operator.procedure.authority.required_capability_ids)
        ):
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="capability-mismatch",
            )
        if boundary.language_classes and not set(boundary.language_classes).issubset(
            set(operator.language_classes)
        ):
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="language-mismatch",
            )
        if boundary.framework_classes and not set(boundary.framework_classes).issubset(
            set(operator.framework_classes)
        ):
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="framework-mismatch",
            )
        if boundary.repository_families and not set(boundary.repository_families).issubset(
            set(operator.repository_families)
        ):
            return MatchResult(
                disposition=MatchDisposition.INCOMPATIBLE,
                operator=operator,
                reason_code="repository-family-mismatch",
            )
        if operator.need_kind is boundary.need_kind and boundary.need_id in operator.satisfied_ids:
            return MatchResult(
                disposition=MatchDisposition.EXACT,
                operator=operator,
                reason_code="exact-compatible-boundary",
                satisfied_ids=operator.satisfied_ids,
            )
        if (
            operator.need_kind is PlanningNeedKind.CRITERION
            and boundary.need_kind is not PlanningNeedKind.CRITERION
            and any(item in operator.satisfied_ids for item in boundary.criterion_ids)
        ):
            return MatchResult(
                disposition=MatchDisposition.PARTIAL_CRITERION,
                operator=operator,
                reason_code="partial-criterion",
                satisfied_ids=tuple(
                    item for item in operator.satisfied_ids if item in boundary.criterion_ids
                ),
            )
        if _need_rank(operator.need_kind) > _need_rank(boundary.need_kind):
            return MatchResult(
                disposition=MatchDisposition.OVERCLAIM,
                operator=operator,
                reason_code="operator-overclaim",
            )
        return MatchResult(
            disposition=MatchDisposition.INCOMPATIBLE,
            operator=operator,
            reason_code="need-mismatch",
        )

    def _compose_covering(self, request: PlanningRequest) -> CompositionDecision | None:
        usable = tuple(item for item in self._operators if item.usable)
        if len(usable) < 2:
            return None
        boundary = request.boundary
        for index, left in enumerate(usable):
            for right in usable[index + 1 :]:
                if left.procedure.task_family_id != boundary.task_family_id:
                    continue
                if right.procedure.task_family_id != boundary.task_family_id:
                    continue
                if not _exact_bindings(left.procedure.bindings, boundary.bindings):
                    continue
                if not _exact_bindings(right.procedure.bindings, boundary.bindings):
                    continue
                covered = set(left.satisfied_ids) | set(right.satisfied_ids)
                if boundary.need_id not in covered and not (
                    boundary.criterion_ids and set(boundary.criterion_ids) <= covered
                ):
                    if not (
                        left.need_kind is PlanningNeedKind.CRITERION
                        and right.need_kind is PlanningNeedKind.CRITERION
                        and boundary.need_kind is PlanningNeedKind.TASK
                        and set(boundary.criterion_ids) <= covered
                    ):
                        continue
                decision = self._validator.validate(
                    (left, right),
                    resource_budget=request.resource_budget,
                )
                if decision.accepted:
                    return decision
                reversed_decision = self._validator.validate(
                    (right, left),
                    resource_budget=request.resource_budget,
                )
                if reversed_decision.accepted:
                    return reversed_decision
        return None


def _need_rank(kind: PlanningNeedKind) -> int:
    return {
        PlanningNeedKind.CRITERION: 0,
        PlanningNeedKind.VALIDATION_STAGE: 1,
        PlanningNeedKind.REPAIR_SUFFIX: 2,
        PlanningNeedKind.SUBGOAL: 3,
        PlanningNeedKind.TASK: 4,
    }[kind]


__all__ = [
    "ADAPTER_INTERFACE",
    "ADAPTER_REVISION",
    "ADAPTIVE_PLANNER_HAMMER_TRACE_REASON_CODE",
    "ADAPTIVE_PLANNER_MODULE",
    "COMPOSITION_VALIDATOR_INTERFACE",
    "HAMMER_TRACE_SCHEMA_NAME",
    "OPERATOR_INTERFACE",
    "PROCEDURE_STAGES",
    "REQUIRED_PLANNER_ORDER",
    "AdaptivePlannerCompatibility",
    "CompositionDecision",
    "CompositionReason",
    "EntailmentEvidence",
    "MatchDisposition",
    "MatchResult",
    "PlannerAdapterError",
    "PlannerAdapterResult",
    "PlannerCompatibilityStatus",
    "PlannerDispatchStatus",
    "PlannerStage",
    "PlanningBoundary",
    "PlanningNeedKind",
    "PlanningRequest",
    "ProcedureCompositionValidator",
    "ProcedureOperator",
    "ProcedurePlanCandidate",
    "ProcedurePlannerAdapter",
    "probe_adaptive_planner",
]
