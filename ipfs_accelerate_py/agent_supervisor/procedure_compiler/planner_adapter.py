"""Procedure operators for adaptive planning without planner authority.

The adapter exposes verified registry procedures as planning operators.  It
does not replace, wrap-as-authority, or reinterpret ``AdaptivePlanner``.  A
current AdaptivePlanner import is re-probed before any procedure dispatch:
an unresolved incompatibility is a typed-unavailable result, and the rest of
the procedure-compiler runtime stays usable.

Qualified planning follows the required order and may apply a procedure only
on exact compatible bindings.  Composition requires exact post-to-pre
entailment evidence, compatible effects/authority/environment, additive
bounded resources, a defined composed rollback, and complete validation.
Cycles and hidden effect escalation are rejected.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Protocol

from .contracts import (
    ArtifactBindings,
    EffectClass,
    ProcedureAuthorityEnvelope,
    ProcedureContractError,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureSpec,
    ProcedureValidationPlan,
    RiskClass,
    _enum,
    _enums,
    _identifier,
    _nested,
    _strings,
    _text,
)
from .registry import ProcedureRegistry, ProcedureRegistryRevision, RegistryFilter


PLANNER_ADAPTER_REVISION: Final[str] = "ProcedurePlannerAdapter@1"
PLANNER_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/planner-adapter@1"
)
ADAPTIVE_PLANNER_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner"
)
ADAPTIVE_PLANNER_SYMBOL: Final[str] = "AdaptivePlanner"
HAMMER_TRACE_SCHEMA_NAME: Final[str] = "HAMMER_TRACE_SCHEMA"
HAMMER_TRACE_IMPORT_SIGNATURE: Final[str] = (
    "NameError: name 'HAMMER_TRACE_SCHEMA' is not defined"
)
HAMMER_TRACE_REASON_CODE: Final[str] = (
    "adaptive_planner_import_undefined_hammer_trace_schema"
)
MAX_COMPOSITION_LENGTH: Final[int] = 4
MAX_PLAN_CANDIDATES: Final[int] = 32
MAX_OPERATORS: Final[int] = 64
MAX_ENTAILMENT_EVIDENCE: Final[int] = 128

REQUIRED_PLANNER_ORDER: Final[tuple[str, ...]] = (
    "exact_verified_procedure",
    "composable_verified_procedures",
    "deterministic_baseline",
    "bounded_local_synthesis",
    "small_local_model",
    "standard_remote_model",
    "strong_remote_model",
    "human_escalation",
)

COMPATIBLE_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "repository_commit",
    "tree_id",
    "contract_revision",
    "policy_revision",
    "environment_id",
)

RESOURCE_FIELDS: Final[tuple[str, ...]] = (
    "wall_time_ms",
    "cpu_time_ms",
    "memory_bytes",
    "disk_bytes",
    "model_token_limit",
    "model_call_limit",
    "subprocess_limit",
    "network_request_limit",
)

_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}

_EFFECT_RANK: Final[dict[EffectClass, int]] = {
    EffectClass.OBSERVE: 0,
    EffectClass.RECEIPT_EMIT: 1,
    EffectClass.VALIDATION: 2,
    EffectClass.PROOF: 2,
    EffectClass.WORKTREE_CREATE: 3,
    EffectClass.ARTIFACT_PERSIST: 4,
    EffectClass.REPOSITORY_WRITE: 5,
    EffectClass.MODEL_REQUEST: 5,
    EffectClass.MERGE_PREPARE: 6,
    EffectClass.MERGE: 7,
    EffectClass.ROLLBACK: 8,
    EffectClass.ESCALATION: 9,
}

_HIGH_ESCALATION_EFFECTS: Final[frozenset[EffectClass]] = frozenset(
    {
        EffectClass.REPOSITORY_WRITE,
        EffectClass.MODEL_REQUEST,
        EffectClass.MERGE_PREPARE,
        EffectClass.MERGE,
        EffectClass.ESCALATION,
    }
)


class PlannerAdapterError(ProcedureContractError):
    """A planner-adapter request, operator, or composition is unsafe."""


class PlannerAdapterStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    AVAILABLE = "available"
    TYPED_UNAVAILABLE = "typed_unavailable"
    INCOMPATIBLE = "incompatible"


class AdaptivePlannerCompatibilityStatus(str, Enum):  # noqa: UP042
    AVAILABLE = "available"
    TYPED_UNAVAILABLE = "typed_unavailable"
    INCOMPATIBLE = "incompatible"


class PlannerStage(str, Enum):  # noqa: UP042
    EXACT_VERIFIED_PROCEDURE = "exact_verified_procedure"
    COMPOSABLE_VERIFIED_PROCEDURES = "composable_verified_procedures"
    DETERMINISTIC_BASELINE = "deterministic_baseline"
    BOUNDED_LOCAL_SYNTHESIS = "bounded_local_synthesis"
    SMALL_LOCAL_MODEL = "small_local_model"
    STANDARD_REMOTE_MODEL = "standard_remote_model"
    STRONG_REMOTE_MODEL = "strong_remote_model"
    HUMAN_ESCALATION = "human_escalation"


class SatisfactionKind(str, Enum):  # noqa: UP042
    TASK = "task"
    CRITERION = "criterion"
    SUBGOAL = "subgoal"
    REPAIR_SUFFIX = "repair_suffix"
    VALIDATION_STAGE = "validation_stage"


_SATISFACTION_RANK: Final[dict[SatisfactionKind, int]] = {
    SatisfactionKind.CRITERION: 0,
    SatisfactionKind.SUBGOAL: 0,
    SatisfactionKind.REPAIR_SUFFIX: 0,
    SatisfactionKind.VALIDATION_STAGE: 0,
    SatisfactionKind.TASK: 1,
}


class CompositionReason(str, Enum):  # noqa: UP042
    ACCEPTED = "accepted"
    MISSING_SPEC = "missing_spec"
    MISSING_ENTAILMENT = "missing_entailment"
    INCOMPATIBLE_EFFECTS = "incompatible_effects"
    INCOMPATIBLE_AUTHORITY = "incompatible_authority"
    INCOMPATIBLE_ENVIRONMENT = "incompatible_environment"
    BUDGET_OVERFLOW = "budget_overflow"
    MISSING_COMPOSED_ROLLBACK = "missing_composed_rollback"
    INCOMPLETE_VALIDATION = "incomplete_validation"
    COMPOSITION_CYCLE = "composition_cycle"
    HIDDEN_EFFECT_ESCALATION = "hidden_effect_escalation"
    INCOMPATIBLE_BOUNDARY = "incompatible_boundary"
    EMPTY_COMPOSITION = "empty_composition"
    COMPOSITION_BOUND = "composition_bound"


class RejectionReason(str, Enum):  # noqa: UP042
    INCOMPATIBLE_BOUNDARY = "incompatible_boundary"
    SATISFACTION_KIND_MISMATCH = "satisfaction_kind_mismatch"
    CLAIMS_MORE_THAN_SATISFIED = "claims_more_than_satisfied"
    NOT_USABLE = "not_usable"
    MISSING_SPEC = "missing_spec"
    CAPABILITY_MISMATCH = "capability_mismatch"
    RISK_CEILING = "risk_ceiling"
    FAMILY_MISMATCH = "family_mismatch"
    LANGUAGE_MISMATCH = "language_mismatch"
    FRAMEWORK_MISMATCH = "framework_mismatch"
    REPOSITORY_FAMILY_MISMATCH = "repository_family_mismatch"
    COMPOSITION_REJECTED = "composition_rejected"


def _status(value: Any, enum_cls: type[Enum], field_name: str) -> Any:
    return _enum(value, enum_cls, field_name)


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise PlannerAdapterError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _risk(value: Any, field_name: str = "risk_ceiling") -> RiskClass:
    return _enum(value, RiskClass, field_name)


def _condition_key(condition: ProcedurePrecondition | ProcedurePostcondition) -> tuple[Any, ...]:
    return (condition.binding, condition.operator.value, condition.operand)


def _effect_rank(effect: EffectClass) -> int:
    return _EFFECT_RANK[effect]


def _risk_rank(risk: RiskClass) -> int:
    return _RISK_RANK[risk]


def _max_effect_rank(effects: Sequence[EffectClass]) -> int:
    if not effects:
        return _EFFECT_RANK[EffectClass.OBSERVE]
    return max(_effect_rank(item) for item in effects)


def _exception_signature(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_hammer_trace_failure(exc: BaseException) -> bool:
    if isinstance(exc, NameError) and HAMMER_TRACE_SCHEMA_NAME in str(exc):
        return True
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, NameError) and HAMMER_TRACE_SCHEMA_NAME in str(cause):
            return True
        text = _exception_signature(cause)
        if HAMMER_TRACE_IMPORT_SIGNATURE in text or HAMMER_TRACE_SCHEMA_NAME in text:
            return True
        cause = cause.__cause__ or cause.__context__
    return HAMMER_TRACE_IMPORT_SIGNATURE in _exception_signature(exc)


def probe_adaptive_planner(
    *,
    importer: Callable[[str], Any] | None = None,
) -> "AdaptivePlannerCompatibility":
    """Re-evaluate the committed AdaptivePlanner import without dispatching."""

    load = importer or importlib.import_module
    try:
        module = load(ADAPTIVE_PLANNER_MODULE)
        planner_cls = getattr(module, ADAPTIVE_PLANNER_SYMBOL, None)
        if planner_cls is None or not isinstance(planner_cls, type):
            return AdaptivePlannerCompatibility(
                status=AdaptivePlannerCompatibilityStatus.TYPED_UNAVAILABLE,
                reason_code="adaptive_planner_symbol_missing",
                signature=f"{ADAPTIVE_PLANNER_SYMBOL} is not a class",
                planner_cls=None,
            )
        plan = getattr(planner_cls, "plan", None)
        if not callable(plan):
            return AdaptivePlannerCompatibility(
                status=AdaptivePlannerCompatibilityStatus.TYPED_UNAVAILABLE,
                reason_code="adaptive_planner_plan_missing",
                signature="AdaptivePlanner.plan is not callable",
                planner_cls=None,
            )
        planner_cls()
    except Exception as exc:
        if _is_hammer_trace_failure(exc):
            return AdaptivePlannerCompatibility(
                status=AdaptivePlannerCompatibilityStatus.INCOMPATIBLE,
                reason_code=HAMMER_TRACE_REASON_CODE,
                signature=HAMMER_TRACE_IMPORT_SIGNATURE,
                planner_cls=None,
            )
        return AdaptivePlannerCompatibility(
            status=AdaptivePlannerCompatibilityStatus.TYPED_UNAVAILABLE,
            reason_code="adaptive_planner_import_failed",
            signature=_exception_signature(exc),
            planner_cls=None,
        )
    return AdaptivePlannerCompatibility(
        status=AdaptivePlannerCompatibilityStatus.AVAILABLE,
        reason_code="adaptive_planner_available",
        signature=f"{ADAPTIVE_PLANNER_MODULE}.{ADAPTIVE_PLANNER_SYMBOL}",
        planner_cls=planner_cls,
    )


@dataclass(frozen=True)
class AdaptivePlannerCompatibility:
    status: AdaptivePlannerCompatibilityStatus
    reason_code: str
    signature: str
    planner_cls: type[Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _status(self.status, AdaptivePlannerCompatibilityStatus, "status"),
        )
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "signature", _text(self.signature, "signature"))
        if self.planner_cls is not None and not isinstance(self.planner_cls, type):
            raise PlannerAdapterError("planner_cls must be a class or None")

    @property
    def available(self) -> bool:
        return self.status is AdaptivePlannerCompatibilityStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "signature": self.signature,
            "available": self.available,
            "planner_symbol": ADAPTIVE_PLANNER_SYMBOL if self.available else "",
        }


@dataclass(frozen=True)
class EntailmentEvidence:
    """Exact admitted evidence that post(A) entails pre(B)."""

    predecessor_cid: str
    successor_cid: str
    postcondition_id: str
    precondition_id: str
    evidence_cid: str
    producer: str
    evidence_type: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "predecessor_cid", _identifier(self.predecessor_cid, "predecessor_cid")
        )
        object.__setattr__(
            self, "successor_cid", _identifier(self.successor_cid, "successor_cid")
        )
        object.__setattr__(
            self, "postcondition_id", _identifier(self.postcondition_id, "postcondition_id")
        )
        object.__setattr__(
            self, "precondition_id", _identifier(self.precondition_id, "precondition_id")
        )
        object.__setattr__(self, "evidence_cid", _identifier(self.evidence_cid, "evidence_cid"))
        object.__setattr__(self, "producer", _identifier(self.producer, "producer"))
        object.__setattr__(
            self, "evidence_type", _identifier(self.evidence_type, "evidence_type")
        )
        if self.predecessor_cid == self.successor_cid:
            raise PlannerAdapterError("entailment evidence cannot be reflexive")

    def to_dict(self) -> dict[str, str]:
        return {
            "predecessor_cid": self.predecessor_cid,
            "successor_cid": self.successor_cid,
            "postcondition_id": self.postcondition_id,
            "precondition_id": self.precondition_id,
            "evidence_cid": self.evidence_cid,
            "producer": self.producer,
            "evidence_type": self.evidence_type,
        }


@dataclass(frozen=True)
class CatalogEntry:
    spec: ProcedureSpec
    satisfaction_kind: SatisfactionKind = SatisfactionKind.TASK
    satisfaction_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec", _nested(self.spec, ProcedureSpec, "spec"))
        object.__setattr__(
            self,
            "satisfaction_kind",
            _status(self.satisfaction_kind, SatisfactionKind, "satisfaction_kind"),
        )
        identity = self.satisfaction_id or self.spec.name
        object.__setattr__(
            self, "satisfaction_id", _identifier(identity, "satisfaction_id")
        )


class ProcedureSpecCatalog(Protocol):
    def get(self, procedure_cid: str) -> CatalogEntry | None:
        ...


class InMemoryProcedureCatalog:
    """CID-addressed procedure bodies used only for matching and composition."""

    def __init__(self, entries: Mapping[str, CatalogEntry | ProcedureSpec] | None = None) -> None:
        stored: dict[str, CatalogEntry] = {}
        for cid, entry in dict(entries or {}).items():
            key = _identifier(cid, "procedure_cid")
            stored[key] = entry if isinstance(entry, CatalogEntry) else CatalogEntry(spec=entry)
            if stored[key].spec.content_id != key:
                raise PlannerAdapterError("catalog CID must equal the procedure content id")
        self._entries = stored

    def add(self, spec: ProcedureSpec, **changes: Any) -> CatalogEntry:
        entry = CatalogEntry(spec=spec, **changes)
        self._entries[spec.content_id] = entry
        return entry

    def get(self, procedure_cid: str) -> CatalogEntry | None:
        return self._entries.get(_identifier(procedure_cid, "procedure_cid"))

    def __contains__(self, procedure_cid: object) -> bool:
        return isinstance(procedure_cid, str) and procedure_cid in self._entries


@dataclass(frozen=True)
class ProcedureOperator:
    """A verified procedure used as a planning operator, never as authority."""

    procedure_id: str
    procedure_cid: str
    revision_id: str
    certificate_cid: str
    task_family_id: str
    satisfaction_kind: SatisfactionKind
    satisfaction_id: str
    bindings: ArtifactBindings
    capability_ids: tuple[str, ...] = ()
    risk_ceiling: RiskClass = RiskClass.OBSERVATION_ONLY
    language_classes: tuple[str, ...] = ()
    framework_classes: tuple[str, ...] = ()
    repository_families: tuple[str, ...] = ()
    spec: ProcedureSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "procedure_id", _identifier(self.procedure_id, "procedure_id"))
        object.__setattr__(
            self, "procedure_cid", _identifier(self.procedure_cid, "procedure_cid")
        )
        object.__setattr__(self, "revision_id", _identifier(self.revision_id, "revision_id"))
        object.__setattr__(
            self, "certificate_cid", _identifier(self.certificate_cid, "certificate_cid")
        )
        object.__setattr__(
            self, "task_family_id", _identifier(self.task_family_id, "task_family_id")
        )
        object.__setattr__(
            self,
            "satisfaction_kind",
            _status(self.satisfaction_kind, SatisfactionKind, "satisfaction_kind"),
        )
        object.__setattr__(
            self, "satisfaction_id", _identifier(self.satisfaction_id, "satisfaction_id")
        )
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self,
            "capability_ids",
            _strings(self.capability_ids, "capability_ids", identifiers=True),
        )
        object.__setattr__(self, "risk_ceiling", _risk(self.risk_ceiling))
        for name in ("language_classes", "framework_classes", "repository_families"):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        if self.spec is not None:
            spec = _nested(self.spec, ProcedureSpec, "spec")
            object.__setattr__(self, "spec", spec)
            if spec.content_id != self.procedure_cid:
                raise PlannerAdapterError("operator procedure CID must match the spec identity")

    @classmethod
    def from_revision(
        cls,
        revision: ProcedureRegistryRevision,
        entry: CatalogEntry | None = None,
    ) -> "ProcedureOperator":
        spec = None if entry is None else entry.spec
        kind = SatisfactionKind.TASK if entry is None else entry.satisfaction_kind
        satisfaction_id = revision.procedure_id if entry is None else entry.satisfaction_id
        return cls(
            procedure_id=revision.procedure_id,
            procedure_cid=revision.procedure_cid,
            revision_id=revision.revision_id,
            certificate_cid=revision.certificate_cid,
            task_family_id=revision.task_family_cid,
            satisfaction_kind=kind,
            satisfaction_id=satisfaction_id,
            bindings=revision.bindings,
            capability_ids=revision.capability_ids,
            risk_ceiling=revision.risk_ceiling,
            language_classes=revision.supported_language_classes,
            framework_classes=revision.supported_framework_classes,
            repository_families=revision.repository_families,
            spec=spec,
        )

    @classmethod
    def from_spec(
        cls,
        spec: ProcedureSpec,
        *,
        revision_id: str,
        certificate_cid: str,
        satisfaction_kind: SatisfactionKind = SatisfactionKind.TASK,
        satisfaction_id: str = "",
        capability_ids: Sequence[str] = (),
        language_classes: Sequence[str] = ("python",),
        framework_classes: Sequence[str] = ("stdlib",),
        repository_families: Sequence[str] = (),
    ) -> "ProcedureOperator":
        return cls(
            procedure_id=spec.name,
            procedure_cid=spec.content_id,
            revision_id=revision_id,
            certificate_cid=certificate_cid,
            task_family_id=spec.task_family_id,
            satisfaction_kind=satisfaction_kind,
            satisfaction_id=satisfaction_id or spec.name,
            bindings=spec.bindings,
            capability_ids=tuple(capability_ids or spec.authority.required_capability_ids),
            risk_ceiling=spec.authority.risk_ceiling,
            language_classes=tuple(language_classes),
            framework_classes=tuple(framework_classes),
            repository_families=tuple(repository_families or (spec.bindings.repository_id,)),
            spec=spec,
        )

    @property
    def claims_task(self) -> bool:
        return self.satisfaction_kind is SatisfactionKind.TASK

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "procedure_cid": self.procedure_cid,
            "revision_id": self.revision_id,
            "certificate_cid": self.certificate_cid,
            "task_family_id": self.task_family_id,
            "satisfaction_kind": self.satisfaction_kind.value,
            "satisfaction_id": self.satisfaction_id,
            "bindings": self.bindings.to_dict(),
            "capability_ids": self.capability_ids,
            "risk_ceiling": self.risk_ceiling.value,
            "language_classes": self.language_classes,
            "framework_classes": self.framework_classes,
            "repository_families": self.repository_families,
            "claims_task": self.claims_task,
            "grants_authority": False,
        }


@dataclass(frozen=True)
class PlanningRequest:
    bindings: ArtifactBindings
    task_family_id: str = ""
    procedure_cid: str = ""
    task_id: str = ""
    criterion_id: str = ""
    subgoal_id: str = ""
    repair_suffix_id: str = ""
    validation_stage_id: str = ""
    capability_ids: tuple[str, ...] = ()
    max_risk: RiskClass = RiskClass.AUTHORITY_OR_SECURITY
    language_classes: tuple[str, ...] = ()
    framework_classes: tuple[str, ...] = ()
    repository_families: tuple[str, ...] = ()
    allowed_effect_classes: tuple[EffectClass, ...] = ()
    resource_budget: ProcedureResourceEnvelope | None = None
    entailment_evidence: tuple[EntailmentEvidence, ...] = ()
    composed_rollback_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "task_family_id",
            "procedure_cid",
            "task_id",
            "criterion_id",
            "subgoal_id",
            "repair_suffix_id",
            "validation_stage_id",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name, required=False)
            )
        for name in (
            "capability_ids",
            "language_classes",
            "framework_classes",
            "repository_families",
            "composed_rollback_ids",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        object.__setattr__(self, "max_risk", _risk(self.max_risk, "max_risk"))
        object.__setattr__(
            self,
            "allowed_effect_classes",
            _enums(
                self.allowed_effect_classes,
                EffectClass,
                "allowed_effect_classes",
                limit=len(EffectClass),
            ),
        )
        if self.resource_budget is not None:
            object.__setattr__(
                self,
                "resource_budget",
                _nested(self.resource_budget, ProcedureResourceEnvelope, "resource_budget"),
            )
        evidence = tuple(self.entailment_evidence or ())
        if len(evidence) > MAX_ENTAILMENT_EVIDENCE:
            raise PlannerAdapterError("entailment evidence bound exhausted")
        normalized: list[EntailmentEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in evidence:
            if not isinstance(item, EntailmentEvidence):
                raise PlannerAdapterError("entailment_evidence must contain EntailmentEvidence")
            key = (
                item.predecessor_cid,
                item.successor_cid,
                item.postcondition_id,
                item.precondition_id,
            )
            if key in seen:
                raise PlannerAdapterError("entailment evidence must not contain duplicates")
            seen.add(key)
            normalized.append(item)
        object.__setattr__(self, "entailment_evidence", tuple(normalized))

    @property
    def requested_need(self) -> tuple[SatisfactionKind, str]:
        if self.criterion_id and not self.task_id:
            return SatisfactionKind.CRITERION, self.criterion_id
        if self.subgoal_id and not self.task_id:
            return SatisfactionKind.SUBGOAL, self.subgoal_id
        if self.repair_suffix_id and not self.task_id:
            return SatisfactionKind.REPAIR_SUFFIX, self.repair_suffix_id
        if self.validation_stage_id and not self.task_id:
            return SatisfactionKind.VALIDATION_STAGE, self.validation_stage_id
        task_id = self.task_id or self.bindings.task_id
        return SatisfactionKind.TASK, task_id


@dataclass(frozen=True)
class OperatorRejection:
    procedure_cid: str
    reason_code: RejectionReason | CompositionReason
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "procedure_cid", _identifier(self.procedure_cid, "procedure_cid")
        )
        if isinstance(self.reason_code, RejectionReason):
            code: Enum = self.reason_code
        elif isinstance(self.reason_code, CompositionReason):
            code = self.reason_code
        else:
            raise PlannerAdapterError("reason_code must be a closed rejection class")
        object.__setattr__(self, "reason_code", code)
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))

    def to_dict(self) -> dict[str, str]:
        return {
            "procedure_cid": self.procedure_cid,
            "reason_code": self.reason_code.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CompositionDecision:
    accepted: bool
    reason_code: CompositionReason
    operators: tuple[ProcedureOperator, ...] = ()
    composed_rollback_ids: tuple[str, ...] = ()
    composed_effect_classes: tuple[EffectClass, ...] = ()
    composed_capability_ids: tuple[str, ...] = ()
    composed_risk: RiskClass = RiskClass.OBSERVATION_ONLY
    composed_resources: ProcedureResourceEnvelope | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(
            self, "reason_code", _status(self.reason_code, CompositionReason, "reason_code")
        )
        operators = tuple(self.operators or ())
        if any(not isinstance(item, ProcedureOperator) for item in operators):
            raise PlannerAdapterError("operators must be ProcedureOperator instances")
        object.__setattr__(self, "operators", operators)
        object.__setattr__(
            self,
            "composed_rollback_ids",
            _strings(self.composed_rollback_ids, "composed_rollback_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "composed_effect_classes",
            _enums(
                self.composed_effect_classes,
                EffectClass,
                "composed_effect_classes",
                limit=len(EffectClass),
            ),
        )
        object.__setattr__(
            self,
            "composed_capability_ids",
            _strings(
                self.composed_capability_ids, "composed_capability_ids", identifiers=True
            ),
        )
        object.__setattr__(self, "composed_risk", _risk(self.composed_risk, "composed_risk"))
        if self.composed_resources is not None:
            object.__setattr__(
                self,
                "composed_resources",
                _nested(
                    self.composed_resources,
                    ProcedureResourceEnvelope,
                    "composed_resources",
                ),
            )
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))
        if self.accepted and self.reason_code is not CompositionReason.ACCEPTED:
            raise PlannerAdapterError("accepted composition must use the accepted reason")
        if self.accepted and len(self.operators) < 2:
            raise PlannerAdapterError("accepted composition requires at least two operators")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code.value,
            "procedure_cids": tuple(item.procedure_cid for item in self.operators),
            "composed_rollback_ids": self.composed_rollback_ids,
            "composed_effect_classes": tuple(
                item.value for item in self.composed_effect_classes
            ),
            "composed_capability_ids": self.composed_capability_ids,
            "composed_risk": self.composed_risk.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PlanCandidate:
    stage: PlannerStage
    applicable: bool
    operators: tuple[ProcedureOperator, ...] = ()
    composition: CompositionDecision | None = None
    claims_task: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _status(self.stage, PlannerStage, "stage"))
        object.__setattr__(self, "applicable", _bool(self.applicable, "applicable"))
        operators = tuple(self.operators or ())
        if any(not isinstance(item, ProcedureOperator) for item in operators):
            raise PlannerAdapterError("operators must be ProcedureOperator instances")
        object.__setattr__(self, "operators", operators)
        if self.composition is not None and not isinstance(
            self.composition, CompositionDecision
        ):
            raise PlannerAdapterError("composition must be a CompositionDecision")
        object.__setattr__(self, "claims_task", _bool(self.claims_task, "claims_task"))
        if self.applicable and self.stage is PlannerStage.EXACT_VERIFIED_PROCEDURE:
            if len(self.operators) != 1:
                raise PlannerAdapterError("exact stage requires exactly one operator")
        if (
            self.applicable
            and self.stage is PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES
            and len(self.operators) < 2
        ):
            raise PlannerAdapterError("composable stage requires at least two operators")

    @property
    def uses_procedure(self) -> bool:
        return bool(self.operators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "applicable": self.applicable,
            "procedure_cids": tuple(item.procedure_cid for item in self.operators),
            "uses_procedure": self.uses_procedure,
            "claims_task": self.claims_task,
        }


@dataclass(frozen=True)
class PlannerDecision:
    status: PlannerAdapterStatus
    reason_code: str
    compatibility: AdaptivePlannerCompatibility
    selected: PlanCandidate | None
    candidates: tuple[PlanCandidate, ...]
    rejected: tuple[OperatorRejection, ...]
    planner_order: tuple[str, ...] = REQUIRED_PLANNER_ORDER
    registry_consulted: bool = False
    procedures_dispatched: bool = False
    adapter_revision: str = PLANNER_ADAPTER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status, PlannerAdapterStatus, "status"))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        if not isinstance(self.compatibility, AdaptivePlannerCompatibility):
            raise PlannerAdapterError("compatibility must be AdaptivePlannerCompatibility")
        if self.selected is not None and not isinstance(self.selected, PlanCandidate):
            raise PlannerAdapterError("selected must be a PlanCandidate")
        candidates = tuple(self.candidates or ())
        if any(not isinstance(item, PlanCandidate) for item in candidates):
            raise PlannerAdapterError("candidates must be PlanCandidate instances")
        if len(candidates) > MAX_PLAN_CANDIDATES:
            raise PlannerAdapterError("plan candidate bound exhausted")
        object.__setattr__(self, "candidates", candidates)
        rejected = tuple(self.rejected or ())
        if any(not isinstance(item, OperatorRejection) for item in rejected):
            raise PlannerAdapterError("rejected must contain OperatorRejection rows")
        object.__setattr__(self, "rejected", rejected)
        object.__setattr__(
            self,
            "planner_order",
            _strings(self.planner_order, "planner_order", identifiers=True, required=True),
        )
        if tuple(self.planner_order) != REQUIRED_PLANNER_ORDER:
            raise PlannerAdapterError("planner_order must be the required closed order")
        object.__setattr__(
            self, "registry_consulted", _bool(self.registry_consulted, "registry_consulted")
        )
        object.__setattr__(
            self,
            "procedures_dispatched",
            _bool(self.procedures_dispatched, "procedures_dispatched"),
        )
        object.__setattr__(
            self, "adapter_revision", _identifier(self.adapter_revision, "adapter_revision")
        )
        if self.procedures_dispatched and not self.compatibility.available:
            raise PlannerAdapterError(
                "procedures cannot be dispatched while AdaptivePlanner is unavailable"
            )
        if self.procedures_dispatched and not self.registry_consulted:
            raise PlannerAdapterError("procedure dispatch requires a registry consultation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_ADAPTER_SCHEMA,
            "adapter_revision": self.adapter_revision,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "compatibility": self.compatibility.to_dict(),
            "selected_stage": "" if self.selected is None else self.selected.stage.value,
            "planner_order": self.planner_order,
            "registry_consulted": self.registry_consulted,
            "procedures_dispatched": self.procedures_dispatched,
            "candidates": tuple(item.to_dict() for item in self.candidates),
            "rejected": tuple(item.to_dict() for item in self.rejected),
        }


def _bindings_compatible(left: ArtifactBindings, right: ArtifactBindings) -> bool:
    return all(getattr(left, name) == getattr(right, name) for name in COMPATIBLE_BINDING_FIELDS)


def _subset(required: Sequence[str], available: Sequence[str]) -> bool:
    if not required:
        return True
    return set(required).issubset(set(available))


def _authority_compatible(left: ProcedureAuthorityEnvelope, right: ProcedureAuthorityEnvelope) -> bool:
    return left.authority_policy_revision == right.authority_policy_revision


def _add_resources(
    left: ProcedureResourceEnvelope, right: ProcedureResourceEnvelope
) -> ProcedureResourceEnvelope:
    values = {name: getattr(left, name) + getattr(right, name) for name in RESOURCE_FIELDS}
    return ProcedureResourceEnvelope(**values)


def _exceeds_budget(
    used: ProcedureResourceEnvelope, budget: ProcedureResourceEnvelope
) -> bool:
    return any(getattr(used, name) > getattr(budget, name) for name in RESOURCE_FIELDS)


def _validation_complete(plan: ProcedureValidationPlan | None) -> bool:
    if plan is None:
        return False
    if not plan.required_step_ids or not plan.required_observation_ids:
        return False
    return bool(plan.required_test_contracts) or bool(plan.required_proof_contracts)


def _posts_by_id(spec: ProcedureSpec) -> dict[str, ProcedurePostcondition]:
    return {item.condition_id: item for item in spec.postconditions}


def _pres_by_id(spec: ProcedureSpec) -> dict[str, ProcedurePrecondition]:
    return {item.condition_id: item for item in spec.preconditions}


def _exact_entailment(
    predecessor: ProcedureSpec, successor: ProcedureSpec
) -> dict[str, str]:
    remaining = {
        item.condition_id: item for item in successor.preconditions if item.required
    }
    covered: dict[str, str] = {}
    posts = list(predecessor.postconditions)
    for post in posts:
        for pre_id, pre in list(remaining.items()):
            if _condition_key(post) == _condition_key(pre):
                covered[pre_id] = post.condition_id
                remaining.pop(pre_id, None)
                break
    return covered if not remaining else {}


def _evidence_entailment(
    predecessor: ProcedureSpec,
    successor: ProcedureSpec,
    evidence: Sequence[EntailmentEvidence],
) -> dict[str, str]:
    posts = _posts_by_id(predecessor)
    pres = _pres_by_id(successor)
    covered: dict[str, str] = {}
    for item in evidence:
        if item.predecessor_cid != predecessor.content_id:
            continue
        if item.successor_cid != successor.content_id:
            continue
        post = posts.get(item.postcondition_id)
        pre = pres.get(item.precondition_id)
        if post is None or pre is None:
            continue
        if not item.producer or not item.evidence_type or not item.evidence_cid:
            continue
        covered[pre.condition_id] = post.condition_id
    required = {item.condition_id for item in successor.preconditions if item.required}
    return covered if required <= set(covered) else {}


class ProcedureCompositionValidator:
    """Fail-closed composition checks for verified procedure operators."""

    revision: Final[str] = "ProcedureCompositionValidator@1"

    def validate(
        self,
        operators: Sequence[ProcedureOperator],
        request: PlanningRequest,
    ) -> CompositionDecision:
        sequence = tuple(operators)
        if not sequence:
            return CompositionDecision(
                accepted=False,
                reason_code=CompositionReason.EMPTY_COMPOSITION,
                detail="composition requires operators",
            )
        if len(sequence) > MAX_COMPOSITION_LENGTH:
            return CompositionDecision(
                accepted=False,
                reason_code=CompositionReason.COMPOSITION_BOUND,
                operators=sequence,
                detail="composition length bound exhausted",
            )
        cids = tuple(item.procedure_cid for item in sequence)
        if len(set(cids)) != len(cids):
            return CompositionDecision(
                accepted=False,
                reason_code=CompositionReason.COMPOSITION_CYCLE,
                operators=sequence,
                detail="composition revisits a procedure CID",
            )
        ids = tuple(item.procedure_id for item in sequence)
        if len(set(ids)) != len(ids):
            return CompositionDecision(
                accepted=False,
                reason_code=CompositionReason.COMPOSITION_CYCLE,
                operators=sequence,
                detail="composition revisits a procedure id",
            )
        specs: list[ProcedureSpec] = []
        for operator in sequence:
            if operator.spec is None:
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.MISSING_SPEC,
                    operators=sequence,
                    detail=operator.procedure_cid,
                )
            specs.append(operator.spec)
        for operator in sequence:
            if not _bindings_compatible(request.bindings, operator.bindings):
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_ENVIRONMENT,
                    operators=sequence,
                    detail="operator bindings are not exactly compatible",
                )
            if operator.bindings.environment_id != request.bindings.environment_id:
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_ENVIRONMENT,
                    operators=sequence,
                    detail="environment_id mismatch",
                )
        for index in range(len(specs) - 1):
            if specs[index].bindings.environment_id != specs[index + 1].bindings.environment_id:
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_ENVIRONMENT,
                    operators=sequence,
                    detail="composed procedures do not share an environment",
                )
            if not _authority_compatible(specs[index].authority, specs[index + 1].authority):
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_AUTHORITY,
                    operators=sequence,
                    detail="authority_policy_revision mismatch",
                )
            covered = _exact_entailment(specs[index], specs[index + 1])
            if not covered:
                covered = _evidence_entailment(
                    specs[index], specs[index + 1], request.entailment_evidence
                )
            if not covered:
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.MISSING_ENTAILMENT,
                    operators=sequence,
                    detail=(
                        f"{sequence[index].procedure_cid}->{sequence[index + 1].procedure_cid}"
                    ),
                )
        effects: list[EffectClass] = []
        capabilities: list[str] = []
        rollbacks: list[str] = []
        risk = RiskClass.OBSERVATION_ONLY
        resources: ProcedureResourceEnvelope | None = None
        prior_rank = -1
        for operator, spec in zip(sequence, specs):
            if not spec.postconditions or not _validation_complete(spec.validation):
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPLETE_VALIDATION,
                    operators=sequence,
                    detail=operator.procedure_cid,
                )
            declared = tuple(item.effect_class for item in spec.declared_effects)
            if not declared:
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_EFFECTS,
                    operators=sequence,
                    detail=operator.procedure_cid,
                )
            current_rank = _max_effect_rank(declared)
            extra = [item for item in declared if item not in effects]
            if (
                prior_rank >= 0
                and extra
                and current_rank > prior_rank
                and any(item in _HIGH_ESCALATION_EFFECTS for item in extra)
            ):
                allowed = set(request.allowed_effect_classes)
                if not allowed or not set(extra).issubset(allowed):
                    return CompositionDecision(
                        accepted=False,
                        reason_code=CompositionReason.HIDDEN_EFFECT_ESCALATION,
                        operators=sequence,
                        detail=operator.procedure_cid,
                    )
            if request.allowed_effect_classes and not set(declared).issubset(
                set(request.allowed_effect_classes)
            ):
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_EFFECTS,
                    operators=sequence,
                    detail=operator.procedure_cid,
                )
            if _risk_rank(spec.authority.risk_ceiling) > _risk_rank(request.max_risk):
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_AUTHORITY,
                    operators=sequence,
                    detail="risk exceeds request ceiling",
                )
            if request.capability_ids and not _subset(
                spec.authority.required_capability_ids, request.capability_ids
            ):
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.INCOMPATIBLE_AUTHORITY,
                    operators=sequence,
                    detail="required capabilities are not granted",
                )
            if not spec.rollback:
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.MISSING_COMPOSED_ROLLBACK,
                    operators=sequence,
                    detail=operator.procedure_cid,
                )
            effects.extend(item for item in declared if item not in effects)
            for capability in spec.authority.required_capability_ids:
                if capability not in capabilities:
                    capabilities.append(capability)
            if _risk_rank(spec.authority.risk_ceiling) > _risk_rank(risk):
                risk = spec.authority.risk_ceiling
            try:
                resources = (
                    spec.resources if resources is None else _add_resources(resources, spec.resources)
                )
            except ProcedureContractError as exc:
                return CompositionDecision(
                    accepted=False,
                    reason_code=CompositionReason.BUDGET_OVERFLOW,
                    operators=sequence,
                    detail=str(exc),
                )
            prior_rank = max(prior_rank, current_rank)
        if resources is None:
            return CompositionDecision(
                accepted=False,
                reason_code=CompositionReason.BUDGET_OVERFLOW,
                operators=sequence,
                detail="composed resources are missing",
            )
        if request.resource_budget is not None and _exceeds_budget(
            resources, request.resource_budget
        ):
            return CompositionDecision(
                accepted=False,
                reason_code=CompositionReason.BUDGET_OVERFLOW,
                operators=sequence,
                detail="composed resources exceed the request budget",
            )
        for spec in reversed(specs):
            rollbacks.extend(item.rollback_id for item in spec.rollback)
        if request.composed_rollback_ids and tuple(rollbacks) != request.composed_rollback_ids:
            return CompositionDecision(
                accepted=False,
                reason_code=CompositionReason.MISSING_COMPOSED_ROLLBACK,
                operators=sequence,
                detail="composed rollback does not match the declared target",
            )
        return CompositionDecision(
            accepted=True,
            reason_code=CompositionReason.ACCEPTED,
            operators=sequence,
            composed_rollback_ids=tuple(rollbacks),
            composed_effect_classes=tuple(effects),
            composed_capability_ids=tuple(capabilities),
            composed_risk=risk,
            composed_resources=resources,
        )


def _fallback_candidates() -> tuple[PlanCandidate, ...]:
    return tuple(
        PlanCandidate(stage=PlannerStage(stage), applicable=True, operators=())
        for stage in REQUIRED_PLANNER_ORDER[2:]
    )


class ProcedurePlannerAdapter:
    """Read registry and planner compatibility; emit ordered plan candidates."""

    revision: Final[str] = PLANNER_ADAPTER_REVISION

    def __init__(
        self,
        registry: ProcedureRegistry | None = None,
        catalog: ProcedureSpecCatalog | None = None,
        *,
        importer: Callable[[str], Any] | None = None,
        composition_validator: ProcedureCompositionValidator | None = None,
    ) -> None:
        if registry is not None and not isinstance(registry, ProcedureRegistry):
            raise PlannerAdapterError("registry must be a ProcedureRegistry")
        self._registry = registry
        self._catalog = catalog or InMemoryProcedureCatalog()
        self._importer = importer
        self._validator = composition_validator or ProcedureCompositionValidator()

    @property
    def registry(self) -> ProcedureRegistry | None:
        return self._registry

    def qualify(self) -> AdaptivePlannerCompatibility:
        return probe_adaptive_planner(importer=self._importer)

    def plan(self, request: PlanningRequest) -> PlannerDecision:
        if not isinstance(request, PlanningRequest):
            raise PlannerAdapterError("request must be a PlanningRequest")
        compatibility = self.qualify()
        if not compatibility.available:
            return PlannerDecision(
                status=PlannerAdapterStatus.TYPED_UNAVAILABLE,
                reason_code=compatibility.reason_code,
                compatibility=compatibility,
                selected=None,
                candidates=(),
                rejected=(),
                registry_consulted=False,
                procedures_dispatched=False,
            )
        operators, rejected = self._load_operators(request)
        need_kind, _need_id = request.requested_need
        matching: list[ProcedureOperator] = []
        for operator in operators:
            if operator.satisfaction_kind is need_kind:
                matching.append(operator)
                continue
            claims_more = (
                _SATISFACTION_RANK[operator.satisfaction_kind]
                > _SATISFACTION_RANK[need_kind]
            )
            rejected = rejected + (
                OperatorRejection(
                    procedure_cid=operator.procedure_cid,
                    reason_code=(
                        RejectionReason.CLAIMS_MORE_THAN_SATISFIED
                        if claims_more
                        else RejectionReason.SATISFACTION_KIND_MISMATCH
                    ),
                    detail=(
                        f"{operator.satisfaction_kind.value} cannot satisfy "
                        f"{need_kind.value}"
                    ),
                ),
            )
        exact, exact_rejected = self._select_exact(matching, request)
        rejected = rejected + exact_rejected
        composition, composition_rejected = self._select_composition(matching, request)
        rejected = rejected + composition_rejected
        candidates: list[PlanCandidate] = [
            PlanCandidate(
                stage=PlannerStage.EXACT_VERIFIED_PROCEDURE,
                applicable=exact is not None,
                operators=() if exact is None else (exact,),
                claims_task=False if exact is None else exact.claims_task,
            ),
            PlanCandidate(
                stage=PlannerStage.COMPOSABLE_VERIFIED_PROCEDURES,
                applicable=composition is not None and composition.accepted,
                operators=() if composition is None else composition.operators,
                composition=composition,
                claims_task=bool(
                    composition is not None
                    and composition.accepted
                    and any(item.claims_task for item in composition.operators)
                ),
            ),
        ]
        candidates.extend(_fallback_candidates())
        selected = next((item for item in candidates if item.applicable), None)
        dispatched = bool(selected is not None and selected.uses_procedure)
        return PlannerDecision(
            status=PlannerAdapterStatus.AVAILABLE,
            reason_code="planner_qualified",
            compatibility=compatibility,
            selected=selected,
            candidates=tuple(candidates),
            rejected=rejected,
            registry_consulted=self._registry is not None,
            procedures_dispatched=dispatched,
        )

    def _load_operators(
        self, request: PlanningRequest
    ) -> tuple[tuple[ProcedureOperator, ...], tuple[OperatorRejection, ...]]:
        if self._registry is None:
            return (), ()
        query = RegistryFilter(
            procedure_cid=request.procedure_cid,
            task_family_cid=request.task_family_id,
            capability_ids=request.capability_ids,
            max_risk=request.max_risk,
            environment_id=request.bindings.environment_id,
            repository_id=request.bindings.repository_id,
            tree_id=request.bindings.tree_id,
            policy_revision=request.bindings.policy_revision,
            language_classes=request.language_classes,
            framework_classes=request.framework_classes,
            repository_families=request.repository_families,
            usable_only=True,
        )
        revisions = self._registry.filter(query)
        operators: list[ProcedureOperator] = []
        rejected: list[OperatorRejection] = []
        for revision in revisions:
            if len(operators) >= MAX_OPERATORS:
                break
            if not _bindings_compatible(request.bindings, revision.bindings):
                rejected.append(
                    OperatorRejection(
                        procedure_cid=revision.procedure_cid,
                        reason_code=RejectionReason.INCOMPATIBLE_BOUNDARY,
                        detail="bindings are not exactly compatible",
                    )
                )
                continue
            if request.task_family_id and revision.task_family_cid != request.task_family_id:
                rejected.append(
                    OperatorRejection(
                        procedure_cid=revision.procedure_cid,
                        reason_code=RejectionReason.FAMILY_MISMATCH,
                    )
                )
                continue
            if request.language_classes and not _subset(
                request.language_classes, revision.supported_language_classes
            ):
                rejected.append(
                    OperatorRejection(
                        procedure_cid=revision.procedure_cid,
                        reason_code=RejectionReason.LANGUAGE_MISMATCH,
                    )
                )
                continue
            if request.framework_classes and not _subset(
                request.framework_classes, revision.supported_framework_classes
            ):
                rejected.append(
                    OperatorRejection(
                        procedure_cid=revision.procedure_cid,
                        reason_code=RejectionReason.FRAMEWORK_MISMATCH,
                    )
                )
                continue
            if request.repository_families and not _subset(
                request.repository_families, revision.repository_families
            ):
                rejected.append(
                    OperatorRejection(
                        procedure_cid=revision.procedure_cid,
                        reason_code=RejectionReason.REPOSITORY_FAMILY_MISMATCH,
                    )
                )
                continue
            entry = self._catalog.get(revision.procedure_cid)
            operators.append(ProcedureOperator.from_revision(revision, entry))
        return tuple(operators), tuple(rejected)

    def _select_exact(
        self,
        operators: Sequence[ProcedureOperator],
        request: PlanningRequest,
    ) -> tuple[ProcedureOperator | None, tuple[OperatorRejection, ...]]:
        _need_kind, need_id = request.requested_need
        rejected: list[OperatorRejection] = []
        for operator in operators:
            if request.procedure_cid:
                if operator.procedure_cid == request.procedure_cid:
                    return operator, tuple(rejected)
                continue
            if operator.satisfaction_id == need_id or operator.procedure_id == need_id:
                return operator, tuple(rejected)
            rejected.append(
                OperatorRejection(
                    procedure_cid=operator.procedure_cid,
                    reason_code=RejectionReason.SATISFACTION_KIND_MISMATCH,
                    detail="operator does not exactly satisfy the requested identity",
                )
            )
        return None, tuple(rejected)

    def _select_composition(
        self,
        operators: Sequence[ProcedureOperator],
        request: PlanningRequest,
    ) -> tuple[CompositionDecision | None, tuple[OperatorRejection, ...]]:
        usable = tuple(item for item in operators if item.spec is not None)
        rejected: list[OperatorRejection] = []
        if len(usable) < 2:
            return None, tuple(rejected)
        best: CompositionDecision | None = None
        for length in range(2, min(MAX_COMPOSITION_LENGTH, len(usable)) + 1):
            for sequence in _bounded_sequences(usable, length):
                decision = self._validator.validate(sequence, request)
                if decision.accepted:
                    if best is None or len(decision.operators) < len(best.operators):
                        best = decision
                        if len(best.operators) == 2:
                            return best, tuple(rejected)
                else:
                    rejected.append(
                        OperatorRejection(
                            procedure_cid="+".join(item.procedure_cid for item in sequence),
                            reason_code=decision.reason_code,
                            detail=decision.detail,
                        )
                    )
            if best is not None:
                return best, tuple(rejected)
        return best, tuple(rejected)


def _bounded_sequences(
    operators: Sequence[ProcedureOperator], length: int
) -> tuple[tuple[ProcedureOperator, ...], ...]:
    if length <= 1:
        return tuple((item,) for item in operators)
    sequences: list[tuple[ProcedureOperator, ...]] = []
    limit = MAX_PLAN_CANDIDATES * 4

    def walk(prefix: tuple[ProcedureOperator, ...], remaining: tuple[ProcedureOperator, ...]) -> None:
        if len(sequences) >= limit:
            return
        if len(prefix) == length:
            sequences.append(prefix)
            return
        for index, item in enumerate(remaining):
            walk(prefix + (item,), remaining[:index] + remaining[index + 1 :])

    walk((), tuple(operators))
    return tuple(sequences)


__all__ = [
    "ADAPTIVE_PLANNER_MODULE",
    "ADAPTIVE_PLANNER_SYMBOL",
    "COMPATIBLE_BINDING_FIELDS",
    "HAMMER_TRACE_IMPORT_SIGNATURE",
    "HAMMER_TRACE_REASON_CODE",
    "MAX_COMPOSITION_LENGTH",
    "PLANNER_ADAPTER_REVISION",
    "PLANNER_ADAPTER_SCHEMA",
    "REQUIRED_PLANNER_ORDER",
    "AdaptivePlannerCompatibility",
    "AdaptivePlannerCompatibilityStatus",
    "CatalogEntry",
    "CompositionDecision",
    "CompositionReason",
    "EntailmentEvidence",
    "InMemoryProcedureCatalog",
    "OperatorRejection",
    "PlanCandidate",
    "PlannerAdapterError",
    "PlannerAdapterStatus",
    "PlannerDecision",
    "PlannerStage",
    "PlanningRequest",
    "ProcedureCompositionValidator",
    "ProcedureOperator",
    "ProcedurePlannerAdapter",
    "RejectionReason",
    "SatisfactionKind",
    "probe_adaptive_planner",
]
