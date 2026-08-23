"""Verified procedure operators for adaptive planning (PCPC-019).

This adapter never replaces or reinterprets AdaptivePlanner.  It re-evaluates
the current AdaptivePlanner import before any procedure dispatch.  Unresolved
incompatibility is a typed-unavailable result: other procedure-compiler
runtime stays usable, and no procedure operator is offered as a plan
candidate.

When the planner import qualifies, candidate order is the closed sequence
from the procedure-compiler plan.  Procedures occupy only the first two
ranks, and only on exact compatible boundaries.  A procedure may satisfy a
task, criterion, subgoal, repair suffix, or validation stage without claiming
more.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Protocol

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    MAX_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ProcedureAuthorityEnvelope,
    ProcedureCertificate,
    ProcedureContractError,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    ProcedureSpec,
    ProcedureValidationPlan,
    RiskClass,
    StepOperation,
    _bounded,
    _decode_fields,
    _enum,
    _enums,
    _identifier,
    _nested,
    _nonnegative_int,
    _schema_name,
    _strings,
    _text,
    _verify_identity,
)
from .registry import (
    ProcedureRegistry,
    RegistryLifecycleState,
    USABLE_STATES,
)


ADAPTER_REVISION: Final[str] = "ProcedurePlannerAdapter@1"
OPERATOR_REVISION: Final[str] = "ProcedureOperator@1"
COMPOSITION_VALIDATOR_REVISION: Final[str] = "ProcedureCompositionValidator@1"
ADAPTIVE_PLANNER_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner"
)
ADAPTIVE_PLANNER_VERSION_EXPECTED: Final[int] = 2
ADAPTIVE_PLAN_SELECTION_SCHEMA_EXPECTED: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adaptive-plan-selection@2"
)
HAMMER_TRACE_SCHEMA_SIGNATURE: Final[str] = (
    "NameError: name 'HAMMER_TRACE_SCHEMA' is not defined"
)
ADAPTIVE_PLANNER_HAMMER_BLOCKER: Final[str] = (
    "adaptive_planner_import_undefined_hammer_trace_schema"
)
MULTI_PROVER_ROUTER_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router"
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
RESOURCE_MAXIMA: Final[Mapping[str, int]] = {
    "wall_time_ms": 86_400_000,
    "cpu_time_ms": 86_400_000,
    "memory_bytes": 1 << 50,
    "disk_bytes": 1 << 50,
    "model_token_limit": 10_000_000,
    "model_call_limit": 1_024,
    "subprocess_limit": 1_024,
    "network_request_limit": 1_024,
}
REQUIRED_MATCH_DIMENSIONS: Final[tuple[str, ...]] = (
    "claim",
    "task_family",
    "repository",
    "tree",
    "policy",
    "environment",
    "language",
    "framework",
    "effect",
    "authority",
    "validation",
    "risk",
    "scope",
    "certificate",
)
REQUIRED_COMPOSITION_DIMENSIONS: Final[tuple[str, ...]] = (
    "entailment",
    "effect",
    "authority",
    "budget",
    "rollback",
    "validation",
    "environment",
    "acyclicity",
)
_EFFECT_RANK: Final[dict[EffectClass, int]] = {
    EffectClass.OBSERVE: 0,
    EffectClass.VALIDATION: 1,
    EffectClass.PROOF: 1,
    EffectClass.RECEIPT_EMIT: 1,
    EffectClass.ARTIFACT_PERSIST: 2,
    EffectClass.MODEL_REQUEST: 2,
    EffectClass.WORKTREE_CREATE: 3,
    EffectClass.REPOSITORY_WRITE: 4,
    EffectClass.ROLLBACK: 4,
    EffectClass.MERGE_PREPARE: 5,
    EffectClass.MERGE: 6,
    EffectClass.ESCALATION: 7,
}
_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}
_VERIFIED_CERTIFICATE_STATES: Final[frozenset[ArtifactState]] = frozenset(
    {ArtifactState.VERIFIED, ArtifactState.PROMOTED}
)


class PlannerAdapterError(ProcedureContractError):
    """A planner-adapter request, operator, or decision is unsafe."""


class PlannerCompatibilityError(PlannerAdapterError):
    """AdaptivePlanner import qualification failed closed."""


class PlannerMatchError(PlannerAdapterError):
    """A procedure operator is not an exact compatible match."""


class PlannerCompositionError(PlannerAdapterError):
    """Procedure composition failed a required exact boundary."""


class PlannerOperatorKind(str, Enum):
    EXACT_VERIFIED_PROCEDURE = "exact-verified-procedure"
    COMPOSABLE_VERIFIED_PROCEDURES = "composable-verified-procedures"
    DETERMINISTIC_BASELINE = "deterministic-baseline"
    BOUNDED_LOCAL_SYNTHESIS = "bounded-local-synthesis"
    SMALL_LOCAL_MODEL = "small-local-model"
    STANDARD_REMOTE_MODEL = "standard-remote-model"
    STRONG_REMOTE_MODEL = "strong-remote-model"
    HUMAN_ESCALATION = "human-escalation"


PLANNER_OPERATOR_ORDER: Final[tuple[PlannerOperatorKind, ...]] = (
    PlannerOperatorKind.EXACT_VERIFIED_PROCEDURE,
    PlannerOperatorKind.COMPOSABLE_VERIFIED_PROCEDURES,
    PlannerOperatorKind.DETERMINISTIC_BASELINE,
    PlannerOperatorKind.BOUNDED_LOCAL_SYNTHESIS,
    PlannerOperatorKind.SMALL_LOCAL_MODEL,
    PlannerOperatorKind.STANDARD_REMOTE_MODEL,
    PlannerOperatorKind.STRONG_REMOTE_MODEL,
    PlannerOperatorKind.HUMAN_ESCALATION,
)
PROCEDURE_OPERATOR_KINDS: Final[frozenset[PlannerOperatorKind]] = frozenset(
    {
        PlannerOperatorKind.EXACT_VERIFIED_PROCEDURE,
        PlannerOperatorKind.COMPOSABLE_VERIFIED_PROCEDURES,
    }
)


class ProcedureClaimScope(str, Enum):
    TASK = "task"
    CRITERION = "criterion"
    SUBGOAL = "subgoal"
    REPAIR_SUFFIX = "repair-suffix"
    VALIDATION_STAGE = "validation-stage"


class PlannerCompatibilityStatus(str, Enum):
    QUALIFIED = "qualified"
    TYPED_UNAVAILABLE = "typed_unavailable"


class PlannerMatchAction(str, Enum):
    MATCH = "match"
    REJECT = "reject"


class PlannerMatchReason(str, Enum):
    EXACT_COMPATIBLE = "exact-compatible"
    PARTIAL_CRITERION = "partial-criterion"
    CLAIM_INCOMPATIBLE = "claim-incompatible"
    TASK_FAMILY_INCOMPATIBLE = "task-family-incompatible"
    REPOSITORY_INCOMPATIBLE = "repository-incompatible"
    TREE_INCOMPATIBLE = "tree-incompatible"
    POLICY_INCOMPATIBLE = "policy-incompatible"
    ENVIRONMENT_INCOMPATIBLE = "environment-incompatible"
    LANGUAGE_INCOMPATIBLE = "language-incompatible"
    FRAMEWORK_INCOMPATIBLE = "framework-incompatible"
    EFFECT_INCOMPATIBLE = "effect-incompatible"
    AUTHORITY_INCOMPATIBLE = "authority-incompatible"
    VALIDATION_INCOMPATIBLE = "validation-incompatible"
    RISK_INCOMPATIBLE = "risk-incompatible"
    SCOPE_INCOMPATIBLE = "scope-incompatible"
    CERTIFICATE_UNVERIFIED = "certificate-unverified"
    REGISTRY_UNUSABLE = "registry-unusable"
    PARTIAL_DIMENSION = "partial-dimension"
    NEAR_MATCH_REJECTED = "near-match-rejected"


class CompositionAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


class CompositionReason(str, Enum):
    COMPATIBLE = "compatible"
    ENTAILMENT_MISSING = "entailment-missing"
    ENTAILMENT_INEXACT = "entailment-inexact"
    EFFECT_INCOMPATIBLE = "effect-incompatible"
    HIDDEN_EFFECT_ESCALATION = "hidden-effect-escalation"
    AUTHORITY_INCOMPATIBLE = "authority-incompatible"
    HIDDEN_AUTHORITY_ESCALATION = "hidden-authority-escalation"
    BUDGET_INCOMPATIBLE = "budget-incompatible"
    ROLLBACK_INCOMPLETE = "rollback-incomplete"
    VALIDATION_INCOMPLETE = "validation-incomplete"
    ENVIRONMENT_INCOMPATIBLE = "environment-incompatible"
    CYCLE_REJECTED = "cycle-rejected"


class PlannerDispatchAction(str, Enum):
    CANDIDATES = "candidates"
    UNAVAILABLE = "unavailable"


class PlannerDispatchReason(str, Enum):
    QUALIFIED_ORDER = "qualified-order"
    ADAPTIVE_PLANNER_INCOMPATIBLE = "adaptive-planner-incompatible"


_MATCH_DIMENSION_REASON: Final[Mapping[str, PlannerMatchReason]] = {
    "claim": PlannerMatchReason.CLAIM_INCOMPATIBLE,
    "task_family": PlannerMatchReason.TASK_FAMILY_INCOMPATIBLE,
    "repository": PlannerMatchReason.REPOSITORY_INCOMPATIBLE,
    "tree": PlannerMatchReason.TREE_INCOMPATIBLE,
    "policy": PlannerMatchReason.POLICY_INCOMPATIBLE,
    "environment": PlannerMatchReason.ENVIRONMENT_INCOMPATIBLE,
    "language": PlannerMatchReason.LANGUAGE_INCOMPATIBLE,
    "framework": PlannerMatchReason.FRAMEWORK_INCOMPATIBLE,
    "effect": PlannerMatchReason.EFFECT_INCOMPATIBLE,
    "authority": PlannerMatchReason.AUTHORITY_INCOMPATIBLE,
    "validation": PlannerMatchReason.VALIDATION_INCOMPATIBLE,
    "risk": PlannerMatchReason.RISK_INCOMPATIBLE,
    "scope": PlannerMatchReason.SCOPE_INCOMPATIBLE,
    "certificate": PlannerMatchReason.CERTIFICATE_UNVERIFIED,
}


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise PlannerAdapterError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _ordered_unique(values: Sequence[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for item in values:
        if item not in result:
            result.append(item)
    return tuple(result)


def _path_is_within(path: str, prefixes: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    for prefix in prefixes:
        root = PurePosixPath(prefix)
        if prefix == ".":
            return True
        if candidate == root or root in candidate.parents:
            return True
    return False


def _set_eq(left: Sequence[Any], right: Sequence[Any]) -> bool:
    return set(left) == set(right)


def _effect_classes(procedure: ProcedureSpec) -> tuple[EffectClass, ...]:
    return _ordered_unique(tuple(item.effect_class for item in procedure.declared_effects))


def _effect_rank(values: Sequence[EffectClass]) -> int:
    if not values:
        return 0
    return max(_EFFECT_RANK[item] for item in values)


def _risk_rank(value: RiskClass) -> int:
    return _RISK_RANK[value]


def _condition_signature(
    condition: ProcedurePrecondition | ProcedurePostcondition,
) -> tuple[Any, ...]:
    return (
        condition.binding,
        condition.operator.value,
        condition.operand,
        condition.evidence_producer,
        condition.evidence_type,
        condition.required,
    )


def _validation_contracts(plan: ProcedureValidationPlan) -> tuple[str, ...]:
    return _ordered_unique(
        (
            *plan.required_test_contracts,
            *plan.required_proof_contracts,
            *plan.post_merge_validation_contracts,
        )
    )


def _exception_signature(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _reached_multi_prover_router(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        trace = getattr(current, "__traceback__", None)
        while trace is not None:
            filename = str(getattr(trace.tb_frame.f_code, "co_filename", ""))
            if filename.replace("\\", "/").endswith(
                "/agent_supervisor/proof/multi_prover_router.py"
            ):
                return True
            trace = trace.tb_next
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class PlannerCompatibility:
    """Closed AdaptivePlanner import qualification.  Never a capability grant."""

    status: PlannerCompatibilityStatus
    reason_code: str
    diagnostic: str
    module_name: str = ADAPTIVE_PLANNER_MODULE
    planner_class_present: bool = False
    other_runtime_usable: bool = True
    blocker: str = ""
    reached_module: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum(self.status, PlannerCompatibilityStatus, "status"),
        )
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "diagnostic", _text(self.diagnostic, "diagnostic", required=False))
        object.__setattr__(self, "module_name", _identifier(self.module_name, "module_name"))
        object.__setattr__(
            self,
            "planner_class_present",
            _bool(self.planner_class_present, "planner_class_present"),
        )
        object.__setattr__(
            self,
            "other_runtime_usable",
            _bool(self.other_runtime_usable, "other_runtime_usable"),
        )
        object.__setattr__(
            self, "blocker", _identifier(self.blocker, "blocker", required=False)
        )
        object.__setattr__(
            self,
            "reached_module",
            _identifier(self.reached_module, "reached_module", required=False),
        )
        if self.status is PlannerCompatibilityStatus.QUALIFIED:
            if not self.planner_class_present:
                raise PlannerCompatibilityError(
                    "qualified AdaptivePlanner compatibility requires the class"
                )
            if self.blocker:
                raise PlannerCompatibilityError(
                    "qualified AdaptivePlanner compatibility cannot carry a blocker"
                )
        else:
            if not self.reason_code:
                raise PlannerCompatibilityError(
                    "typed unavailable AdaptivePlanner compatibility requires a reason"
                )
        if not self.other_runtime_usable:
            raise PlannerCompatibilityError(
                "planner incompatibility cannot disable other procedure-compiler runtime"
            )

    @property
    def qualified(self) -> bool:
        return self.status is PlannerCompatibilityStatus.QUALIFIED

    @property
    def typed_unavailable(self) -> bool:
        return self.status is PlannerCompatibilityStatus.TYPED_UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocker": self.blocker,
            "diagnostic": self.diagnostic,
            "module_name": self.module_name,
            "other_runtime_usable": True,
            "planner_class_present": self.planner_class_present,
            "reached_module": self.reached_module,
            "reason_code": self.reason_code,
            "status": self.status.value,
        }


class PlannerCompatibilityProbe(Protocol):
    def __call__(self) -> PlannerCompatibility:
        ...


def probe_adaptive_planner_compatibility() -> PlannerCompatibility:
    """Re-evaluate the committed AdaptivePlanner import on the current tree."""

    try:
        module = importlib.import_module(ADAPTIVE_PLANNER_MODULE)
        planner = getattr(module, "AdaptivePlanner", None)
        version = getattr(module, "ADAPTIVE_PLANNER_VERSION", None)
        schema = getattr(module, "ADAPTIVE_PLAN_SELECTION_SCHEMA", None)
        if planner is None:
            return PlannerCompatibility(
                status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
                reason_code="adaptive_planner_class_missing",
                diagnostic="AdaptivePlanner is not exported from its committed module",
                planner_class_present=False,
                blocker="adaptive_planner_class_missing",
            )
        if version != ADAPTIVE_PLANNER_VERSION_EXPECTED:
            return PlannerCompatibility(
                status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
                reason_code="adaptive_planner_version_incompatible",
                diagnostic="ADAPTIVE_PLANNER_VERSION is not the committed interface",
                planner_class_present=True,
                blocker="adaptive_planner_version_incompatible",
            )
        if schema != ADAPTIVE_PLAN_SELECTION_SCHEMA_EXPECTED:
            return PlannerCompatibility(
                status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
                reason_code="adaptive_planner_schema_incompatible",
                diagnostic="ADAPTIVE_PLAN_SELECTION_SCHEMA is not the committed interface",
                planner_class_present=True,
                blocker="adaptive_planner_schema_incompatible",
            )
        return PlannerCompatibility(
            status=PlannerCompatibilityStatus.QUALIFIED,
            reason_code="adaptive_planner_qualified",
            diagnostic="",
            planner_class_present=True,
        )
    except Exception as exc:
        diagnostic = _exception_signature(exc)
        chain: list[str] = []
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(_exception_signature(current))
            current = current.__cause__ or current.__context__
        joined = " | ".join(chain)
        hammer = (
            HAMMER_TRACE_SCHEMA_SIGNATURE in joined
            or "HAMMER_TRACE_SCHEMA" in joined
            or _reached_multi_prover_router(exc)
        )
        if hammer:
            return PlannerCompatibility(
                status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
                reason_code=ADAPTIVE_PLANNER_HAMMER_BLOCKER,
                diagnostic=HAMMER_TRACE_SCHEMA_SIGNATURE,
                planner_class_present=False,
                blocker=ADAPTIVE_PLANNER_HAMMER_BLOCKER,
                reached_module=MULTI_PROVER_ROUTER_MODULE,
            )
        return PlannerCompatibility(
            status=PlannerCompatibilityStatus.TYPED_UNAVAILABLE,
            reason_code="adaptive_planner_import_failed",
            diagnostic=diagnostic,
            planner_class_present=False,
            blocker="adaptive_planner_import_failed",
        )


def qualified_planner_compatibility() -> PlannerCompatibility:
    """Test helper: a separately reviewed qualified AdaptivePlanner probe."""

    return PlannerCompatibility(
        status=PlannerCompatibilityStatus.QUALIFIED,
        reason_code="adaptive_planner_qualified",
        diagnostic="",
        planner_class_present=True,
    )


@dataclass(frozen=True)
class ProcedureOperator(CanonicalContract):
    """A verified procedure offered as a planning operator, never as authority."""

    SCHEMA: ClassVar[str] = _schema_name("ProcedureOperator")

    bindings: ArtifactBindings
    procedure: ProcedureSpec
    certificate: ProcedureCertificate
    claim_scope: ProcedureClaimScope
    claim_id: str
    registry_revision_id: str = ""
    registry_state: str = ""
    operator_revision: str = OPERATOR_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "procedure", _nested(self.procedure, ProcedureSpec, "procedure")
        )
        object.__setattr__(
            self,
            "certificate",
            _nested(self.certificate, ProcedureCertificate, "certificate"),
        )
        object.__setattr__(
            self,
            "claim_scope",
            _enum(self.claim_scope, ProcedureClaimScope, "claim_scope"),
        )
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(
            self,
            "registry_revision_id",
            _identifier(self.registry_revision_id, "registry_revision_id", required=False),
        )
        object.__setattr__(
            self,
            "registry_state",
            _identifier(self.registry_state, "registry_state", required=False),
        )
        object.__setattr__(
            self,
            "operator_revision",
            _identifier(self.operator_revision, "operator_revision"),
        )
        if self.operator_revision != OPERATOR_REVISION:
            raise PlannerAdapterError("procedure operator revision is not current")
        if self.bindings != self.procedure.bindings:
            raise PlannerAdapterError("operator bindings must equal the procedure bindings")
        if self.certificate.procedure_cid != self.procedure.content_id:
            raise PlannerAdapterError("operator certificate does not bind this procedure")
        if self.certificate.task_family_cid != self.procedure.task_family_id:
            raise PlannerAdapterError("operator certificate family is not exact")
        if self.certificate.bindings.repository_id != self.bindings.repository_id:
            raise PlannerAdapterError("operator certificate repository is not exact")
        if self.certificate.bindings.tree_id != self.bindings.tree_id:
            raise PlannerAdapterError("operator certificate tree is not exact")
        if self.certificate.bindings.policy_revision != self.bindings.policy_revision:
            raise PlannerAdapterError("operator certificate policy is not exact")
        if self.certificate.bindings.environment_id != self.bindings.environment_id:
            raise PlannerAdapterError("operator certificate environment is not exact")
        if self.registry_state and self.registry_state not in {
            item.value for item in RegistryLifecycleState
        }:
            raise PlannerAdapterError("operator registry state is outside the closed lifecycle")
        _bounded(self, "ProcedureOperator")

    @property
    def procedure_cid(self) -> str:
        return self.procedure.content_id

    @property
    def procedure_id(self) -> str:
        return self.procedure.name

    @property
    def verified(self) -> bool:
        return (
            self.certificate.state in _VERIFIED_CERTIFICATE_STATES
            and self.certificate.procedure_cid == self.procedure.content_id
        )

    @property
    def claims_task(self) -> bool:
        return self.claim_scope is ProcedureClaimScope.TASK

    @property
    def can_authorize(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_complete(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "procedure": self.procedure,
            "certificate": self.certificate,
            "claim_scope": self.claim_scope.value,
            "claim_id": self.claim_id,
            "registry_revision_id": self.registry_revision_id,
            "registry_state": self.registry_state,
            "operator_revision": OPERATOR_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProcedureOperator:
        fields = (
            "bindings",
            "procedure",
            "certificate",
            "claim_scope",
            "claim_id",
            "registry_revision_id",
            "registry_state",
            "operator_revision",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "procedure" in values:
            values["procedure"] = _nested(values["procedure"], ProcedureSpec, "procedure")
        if "certificate" in values:
            values["certificate"] = _nested(
                values["certificate"], ProcedureCertificate, "certificate"
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class PlannerMatchRequest:
    """Exact boundary a procedure operator must satisfy without claiming more."""

    bindings: ArtifactBindings
    task_family_id: str
    claim_scope: ProcedureClaimScope
    claim_id: str
    language_classes: tuple[str, ...]
    framework_classes: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...]
    authority_policy_revision: str
    authority_requirement_ids: tuple[str, ...]
    required_capability_ids: tuple[str, ...]
    validation_contracts: tuple[str, ...]
    risk_ceiling: RiskClass
    scope_paths: tuple[str, ...]
    repository_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "task_family_id", _identifier(self.task_family_id, "task_family_id")
        )
        object.__setattr__(
            self,
            "claim_scope",
            _enum(self.claim_scope, ProcedureClaimScope, "claim_scope"),
        )
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(
            self,
            "language_classes",
            _strings(self.language_classes, "language_classes", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "framework_classes",
            _strings(
                self.framework_classes, "framework_classes", identifiers=True, required=True
            ),
        )
        object.__setattr__(
            self,
            "effect_classes",
            _enums(
                self.effect_classes,
                EffectClass,
                "effect_classes",
                limit=len(EffectClass),
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "authority_policy_revision",
            _identifier(self.authority_policy_revision, "authority_policy_revision"),
        )
        object.__setattr__(
            self,
            "authority_requirement_ids",
            _strings(
                self.authority_requirement_ids,
                "authority_requirement_ids",
                identifiers=True,
                required=True,
            ),
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
        object.__setattr__(
            self,
            "validation_contracts",
            _strings(
                self.validation_contracts,
                "validation_contracts",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self, "risk_ceiling", _enum(self.risk_ceiling, RiskClass, "risk_ceiling")
        )
        object.__setattr__(
            self,
            "scope_paths",
            _strings(self.scope_paths, "scope_paths", paths=True, required=True),
        )
        object.__setattr__(
            self,
            "repository_families",
            _strings(self.repository_families, "repository_families", identifiers=True),
        )


@dataclass(frozen=True)
class PlannerMatchDecision(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("PlannerMatchDecision")

    bindings: ArtifactBindings
    action: PlannerMatchAction
    reason_code: PlannerMatchReason
    claim_scope: ProcedureClaimScope
    claim_id: str
    procedure_cid: str
    compatible_dimensions: tuple[str, ...]
    incompatible_dimensions: tuple[str, ...]
    matched: bool
    claims_task: bool
    adapter_revision: str = ADAPTER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "action", _enum(self.action, PlannerMatchAction, "action"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, PlannerMatchReason, "reason_code")
        )
        object.__setattr__(
            self,
            "claim_scope",
            _enum(self.claim_scope, ProcedureClaimScope, "claim_scope"),
        )
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(
            self, "procedure_cid", _identifier(self.procedure_cid, "procedure_cid")
        )
        object.__setattr__(
            self,
            "compatible_dimensions",
            _strings(self.compatible_dimensions, "compatible_dimensions", identifiers=True),
        )
        object.__setattr__(
            self,
            "incompatible_dimensions",
            _strings(
                self.incompatible_dimensions, "incompatible_dimensions", identifiers=True
            ),
        )
        object.__setattr__(self, "matched", _bool(self.matched, "matched"))
        object.__setattr__(self, "claims_task", _bool(self.claims_task, "claims_task"))
        object.__setattr__(
            self, "adapter_revision", _identifier(self.adapter_revision, "adapter_revision")
        )
        if self.adapter_revision != ADAPTER_REVISION:
            raise PlannerAdapterError("planner match adapter revision is not current")
        overlap = set(self.compatible_dimensions).intersection(self.incompatible_dimensions)
        if overlap:
            raise PlannerAdapterError("match dimensions cannot be both compatible and not")
        if self.action is PlannerMatchAction.MATCH:
            if not self.matched:
                raise PlannerAdapterError("a match decision must set matched")
            if self.incompatible_dimensions:
                raise PlannerAdapterError("an exact match cannot carry incompatible dimensions")
            if set(self.compatible_dimensions) != set(REQUIRED_MATCH_DIMENSIONS):
                raise PlannerAdapterError(
                    "an exact match must retain every required match dimension"
                )
            if self.reason_code not in {
                PlannerMatchReason.EXACT_COMPATIBLE,
                PlannerMatchReason.PARTIAL_CRITERION,
            }:
                raise PlannerAdapterError("an exact match must use a compatible reason")
            if self.reason_code is PlannerMatchReason.PARTIAL_CRITERION:
                if self.claim_scope is ProcedureClaimScope.TASK or self.claims_task:
                    raise PlannerAdapterError(
                        "a partial-criterion match cannot claim the whole task"
                    )
            elif self.claim_scope is ProcedureClaimScope.TASK and not self.claims_task:
                raise PlannerAdapterError("a task match must claim the task")
        else:
            if self.matched:
                raise PlannerAdapterError("a rejected match cannot be matched")
            if self.reason_code in {
                PlannerMatchReason.EXACT_COMPATIBLE,
                PlannerMatchReason.PARTIAL_CRITERION,
            }:
                raise PlannerAdapterError("a rejected match cannot be labeled compatible")
            if not self.incompatible_dimensions:
                raise PlannerAdapterError("a rejected match must name incompatible dimensions")
        _bounded(self, "PlannerMatchDecision")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "claim_scope": self.claim_scope.value,
            "claim_id": self.claim_id,
            "procedure_cid": self.procedure_cid,
            "compatible_dimensions": self.compatible_dimensions,
            "incompatible_dimensions": self.incompatible_dimensions,
            "matched": self.matched,
            "claims_task": self.claims_task,
            "adapter_revision": ADAPTER_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerMatchDecision:
        fields = (
            "bindings",
            "action",
            "reason_code",
            "claim_scope",
            "claim_id",
            "procedure_cid",
            "compatible_dimensions",
            "incompatible_dimensions",
            "matched",
            "claims_task",
            "adapter_revision",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class EntailmentEvidence:
    """Exact post(A)-to-pre(B) entailment.  Similarity is not evidence."""

    predecessor_procedure_cid: str
    successor_procedure_cid: str
    predecessor_postcondition_id: str
    successor_precondition_id: str
    evidence_cid: str

    def __post_init__(self) -> None:
        for name in (
            "predecessor_procedure_cid",
            "successor_procedure_cid",
            "predecessor_postcondition_id",
            "successor_precondition_id",
            "evidence_cid",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))

    def to_record(self) -> dict[str, str]:
        return {
            "predecessor_procedure_cid": self.predecessor_procedure_cid,
            "successor_procedure_cid": self.successor_procedure_cid,
            "predecessor_postcondition_id": self.predecessor_postcondition_id,
            "successor_precondition_id": self.successor_precondition_id,
            "evidence_cid": self.evidence_cid,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> EntailmentEvidence:
        if not isinstance(payload, Mapping):
            raise PlannerCompositionError("entailment evidence must be a mapping")
        return cls(
            predecessor_procedure_cid=payload.get("predecessor_procedure_cid", ""),
            successor_procedure_cid=payload.get("successor_procedure_cid", ""),
            predecessor_postcondition_id=payload.get("predecessor_postcondition_id", ""),
            successor_precondition_id=payload.get("successor_precondition_id", ""),
            evidence_cid=payload.get("evidence_cid", ""),
        )


@dataclass(frozen=True)
class CompositionRequest:
    operators: tuple[ProcedureOperator, ...]
    entailment: tuple[EntailmentEvidence, ...]
    composed_effects: tuple[EffectClass, ...]
    composed_authority: ProcedureAuthorityEnvelope
    composed_resources: ProcedureResourceEnvelope
    composed_validation: ProcedureValidationPlan
    composed_rollback: tuple[ProcedureRollback, ...]

    def __post_init__(self) -> None:
        operators = tuple(self.operators)
        if len(operators) < 2:
            raise PlannerCompositionError("composition requires at least two verified procedures")
        if len(operators) > MAX_ITEMS:
            raise PlannerCompositionError("composition exceeds its operator bound")
        if any(not isinstance(item, ProcedureOperator) for item in operators):
            raise PlannerCompositionError("composition operators must be ProcedureOperator")
        object.__setattr__(self, "operators", operators)
        evidence: list[EntailmentEvidence] = []
        for item in self.entailment:
            if isinstance(item, EntailmentEvidence):
                evidence.append(item)
            elif isinstance(item, Mapping):
                evidence.append(EntailmentEvidence.from_record(item))
            else:
                raise PlannerCompositionError("entailment evidence is malformed")
        object.__setattr__(self, "entailment", tuple(evidence))
        object.__setattr__(
            self,
            "composed_effects",
            _enums(
                self.composed_effects,
                EffectClass,
                "composed_effects",
                limit=len(EffectClass),
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "composed_authority",
            _nested(self.composed_authority, ProcedureAuthorityEnvelope, "composed_authority"),
        )
        object.__setattr__(
            self,
            "composed_resources",
            _nested(self.composed_resources, ProcedureResourceEnvelope, "composed_resources"),
        )
        object.__setattr__(
            self,
            "composed_validation",
            _nested(self.composed_validation, ProcedureValidationPlan, "composed_validation"),
        )
        rollbacks: list[ProcedureRollback] = []
        for item in self.composed_rollback:
            rollbacks.append(_nested(item, ProcedureRollback, "composed_rollback"))
        object.__setattr__(self, "composed_rollback", tuple(rollbacks))


@dataclass(frozen=True)
class CompositionDecision(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("CompositionDecision")

    bindings: ArtifactBindings
    action: CompositionAction
    reason_code: CompositionReason
    procedure_cids: tuple[str, ...]
    compatible_dimensions: tuple[str, ...]
    incompatible_dimensions: tuple[str, ...]
    accepted: bool
    validator_revision: str = COMPOSITION_VALIDATOR_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "action", _enum(self.action, CompositionAction, "action"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, CompositionReason, "reason_code")
        )
        object.__setattr__(
            self,
            "procedure_cids",
            _strings(self.procedure_cids, "procedure_cids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "compatible_dimensions",
            _strings(self.compatible_dimensions, "compatible_dimensions", identifiers=True),
        )
        object.__setattr__(
            self,
            "incompatible_dimensions",
            _strings(
                self.incompatible_dimensions, "incompatible_dimensions", identifiers=True
            ),
        )
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(
            self,
            "validator_revision",
            _identifier(self.validator_revision, "validator_revision"),
        )
        if self.validator_revision != COMPOSITION_VALIDATOR_REVISION:
            raise PlannerCompositionError("composition validator revision is not current")
        if self.action is CompositionAction.ACCEPT:
            if not self.accepted:
                raise PlannerCompositionError("an accepted composition must set accepted")
            if self.incompatible_dimensions:
                raise PlannerCompositionError(
                    "an accepted composition cannot carry incompatible dimensions"
                )
            if set(self.compatible_dimensions) != set(REQUIRED_COMPOSITION_DIMENSIONS):
                raise PlannerCompositionError(
                    "an accepted composition must retain every required dimension"
                )
            if self.reason_code is not CompositionReason.COMPATIBLE:
                raise PlannerCompositionError(
                    "an accepted composition must be labeled compatible"
                )
        else:
            if self.accepted:
                raise PlannerCompositionError("a rejected composition cannot be accepted")
            if self.reason_code is CompositionReason.COMPATIBLE:
                raise PlannerCompositionError(
                    "a rejected composition cannot be labeled compatible"
                )
            if not self.incompatible_dimensions:
                raise PlannerCompositionError(
                    "a rejected composition must name incompatible dimensions"
                )
        _bounded(self, "CompositionDecision")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "procedure_cids": self.procedure_cids,
            "compatible_dimensions": self.compatible_dimensions,
            "incompatible_dimensions": self.incompatible_dimensions,
            "accepted": self.accepted,
            "validator_revision": COMPOSITION_VALIDATOR_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompositionDecision:
        fields = (
            "bindings",
            "action",
            "reason_code",
            "procedure_cids",
            "compatible_dimensions",
            "incompatible_dimensions",
            "accepted",
            "validator_revision",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class PlannerDispatchRequest:
    match: PlannerMatchRequest
    operators: tuple[ProcedureOperator, ...] = ()
    composition: CompositionRequest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.match, PlannerMatchRequest):
            raise PlannerAdapterError("dispatch match must be PlannerMatchRequest")
        operators = tuple(self.operators)
        if any(not isinstance(item, ProcedureOperator) for item in operators):
            raise PlannerAdapterError("dispatch operators must be ProcedureOperator")
        if len(operators) > MAX_ITEMS:
            raise PlannerAdapterError("dispatch operator bound exhausted")
        object.__setattr__(self, "operators", operators)
        if self.composition is not None and not isinstance(
            self.composition, CompositionRequest
        ):
            raise PlannerAdapterError("dispatch composition must be CompositionRequest")


@dataclass(frozen=True)
class PlannerDispatchDecision(CanonicalContract):
    SCHEMA: ClassVar[str] = _schema_name("PlannerDispatchDecision")

    bindings: ArtifactBindings
    action: PlannerDispatchAction
    reason_code: PlannerDispatchReason
    compatibility_status: PlannerCompatibilityStatus
    compatibility_reason_code: str
    selected_kind: str
    considered_kinds: tuple[str, ...]
    procedure_cids: tuple[str, ...]
    dispatched: bool
    other_runtime_usable: bool = True
    adapter_revision: str = ADAPTER_REVISION
    diagnostic: str = ""
    blocker: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "action", _enum(self.action, PlannerDispatchAction, "action")
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, PlannerDispatchReason, "reason_code"),
        )
        object.__setattr__(
            self,
            "compatibility_status",
            _enum(
                self.compatibility_status,
                PlannerCompatibilityStatus,
                "compatibility_status",
            ),
        )
        object.__setattr__(
            self,
            "compatibility_reason_code",
            _identifier(self.compatibility_reason_code, "compatibility_reason_code"),
        )
        object.__setattr__(
            self,
            "selected_kind",
            _identifier(self.selected_kind, "selected_kind", required=False),
        )
        object.__setattr__(
            self,
            "considered_kinds",
            _strings(self.considered_kinds, "considered_kinds", identifiers=True),
        )
        object.__setattr__(
            self,
            "procedure_cids",
            _strings(self.procedure_cids, "procedure_cids", identifiers=True),
        )
        object.__setattr__(self, "dispatched", _bool(self.dispatched, "dispatched"))
        object.__setattr__(
            self,
            "other_runtime_usable",
            _bool(self.other_runtime_usable, "other_runtime_usable"),
        )
        object.__setattr__(
            self, "adapter_revision", _identifier(self.adapter_revision, "adapter_revision")
        )
        object.__setattr__(
            self, "diagnostic", _text(self.diagnostic, "diagnostic", required=False)
        )
        object.__setattr__(
            self, "blocker", _identifier(self.blocker, "blocker", required=False)
        )
        if self.adapter_revision != ADAPTER_REVISION:
            raise PlannerAdapterError("planner dispatch adapter revision is not current")
        if not self.other_runtime_usable:
            raise PlannerAdapterError(
                "planner incompatibility cannot disable other procedure-compiler runtime"
            )
        if self.action is PlannerDispatchAction.UNAVAILABLE:
            if self.dispatched:
                raise PlannerAdapterError("typed unavailable dispatch cannot emit procedures")
            if self.procedure_cids:
                raise PlannerAdapterError(
                    "typed unavailable dispatch cannot carry procedure candidates"
                )
            if self.compatibility_status is PlannerCompatibilityStatus.QUALIFIED:
                raise PlannerAdapterError(
                    "qualified AdaptivePlanner cannot produce typed unavailable dispatch"
                )
            if self.reason_code is not PlannerDispatchReason.ADAPTIVE_PLANNER_INCOMPATIBLE:
                raise PlannerAdapterError(
                    "incompatible AdaptivePlanner dispatch must use the declared reason"
                )
            if self.selected_kind:
                raise PlannerAdapterError(
                    "typed unavailable dispatch cannot select a planner operator"
                )
        else:
            if self.compatibility_status is not PlannerCompatibilityStatus.QUALIFIED:
                raise PlannerAdapterError(
                    "procedure candidates require a qualified AdaptivePlanner import"
                )
            if self.reason_code is not PlannerDispatchReason.QUALIFIED_ORDER:
                raise PlannerAdapterError("qualified dispatch must record planner order")
            if not self.selected_kind:
                raise PlannerAdapterError("qualified dispatch must select a planner-order kind")
            selected = PlannerOperatorKind(self.selected_kind)
            expected = tuple(item.value for item in PLANNER_OPERATOR_ORDER)
            if tuple(self.considered_kinds) != expected:
                raise PlannerAdapterError("qualified dispatch must consider the required order")
            if selected.value not in expected:
                raise PlannerAdapterError("qualified dispatch selected an unknown planner-order kind")
            if selected in PROCEDURE_OPERATOR_KINDS and not self.procedure_cids:
                raise PlannerAdapterError(
                    "procedure ranks require exact compatible procedure candidates"
                )
            if selected not in PROCEDURE_OPERATOR_KINDS and self.procedure_cids:
                raise PlannerAdapterError(
                    "non-procedure ranks cannot carry procedure candidates"
                )
            if not self.dispatched and selected in PROCEDURE_OPERATOR_KINDS:
                raise PlannerAdapterError("procedure candidates must be marked dispatched")
            if self.dispatched and selected not in PROCEDURE_OPERATOR_KINDS:
                raise PlannerAdapterError(
                    "non-procedure ranks do not dispatch procedure operators"
                )
        _bounded(self, "PlannerDispatchDecision")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "compatibility_status": self.compatibility_status.value,
            "compatibility_reason_code": self.compatibility_reason_code,
            "selected_kind": self.selected_kind,
            "considered_kinds": self.considered_kinds,
            "procedure_cids": self.procedure_cids,
            "dispatched": self.dispatched,
            "other_runtime_usable": True,
            "adapter_revision": ADAPTER_REVISION,
            "diagnostic": self.diagnostic,
            "blocker": self.blocker,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerDispatchDecision:
        fields = (
            "bindings",
            "action",
            "reason_code",
            "compatibility_status",
            "compatibility_reason_code",
            "selected_kind",
            "considered_kinds",
            "procedure_cids",
            "dispatched",
            "other_runtime_usable",
            "adapter_revision",
            "diagnostic",
            "blocker",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if values.get("other_runtime_usable") is False:
            raise PlannerAdapterError(
                "planner incompatibility cannot disable other procedure-compiler runtime"
            )
        record = cls(**values)
        _verify_identity(payload, record)
        return record


def _match_scope(procedure: ProcedureSpec, prefixes: Sequence[str]) -> bool:
    if not _set_eq(procedure.scope_paths, prefixes):
        return False
    for path in procedure.declared_reads:
        if not _path_is_within(path, prefixes):
            return False
    for effect in procedure.declared_effects:
        for target in effect.targets:
            if not _path_is_within(target, prefixes):
                return False
    return True


def _certificate_verified(operator: ProcedureOperator) -> bool:
    certificate = operator.certificate
    procedure = operator.procedure
    if not operator.verified:
        return False
    if certificate.authority_policy_revision != procedure.authority.authority_policy_revision:
        return False
    if certificate.risk_ceiling != procedure.authority.risk_ceiling:
        return False
    return True


class ProcedureCompositionValidator:
    """Exact post-to-pre, effect, authority, budget, rollback, and cycle checks."""

    revision: Final[str] = COMPOSITION_VALIDATOR_REVISION

    def validate(self, request: CompositionRequest) -> CompositionDecision:
        if not isinstance(request, CompositionRequest):
            raise PlannerCompositionError("composition request is required")
        operators = request.operators
        bindings = operators[0].bindings
        procedure_cids = tuple(item.procedure_cid for item in operators)
        compatible: list[str] = []
        incompatible: list[str] = []
        reasons: list[CompositionReason] = []

        def fail(dimension: str, reason: CompositionReason) -> None:
            if dimension not in incompatible:
                incompatible.append(dimension)
            if reason not in reasons:
                reasons.append(reason)

        def pass_dimension(dimension: str) -> None:
            if dimension not in compatible:
                compatible.append(dimension)

        if len(set(procedure_cids)) != len(procedure_cids) or len(
            {item.procedure_id for item in operators}
        ) != len(operators):
            fail("acyclicity", CompositionReason.CYCLE_REJECTED)
        else:
            pass_dimension("acyclicity")

        environment_ok = True
        for operator in operators[1:]:
            left = operators[0].bindings
            right = operator.bindings
            if (
                left.repository_id != right.repository_id
                or left.tree_id != right.tree_id
                or left.policy_revision != right.policy_revision
                or left.environment_id != right.environment_id
                or left.repository_commit != right.repository_commit
            ):
                environment_ok = False
                break
        if environment_ok:
            pass_dimension("environment")
        else:
            fail("environment", CompositionReason.ENVIRONMENT_INCOMPATIBLE)

        entailment_reason = self._entailment_reason(operators, request.entailment)
        if entailment_reason is None:
            pass_dimension("entailment")
        else:
            fail("entailment", entailment_reason)

        component_effects = _ordered_unique(
            tuple(cls for operator in operators for cls in _effect_classes(operator.procedure))
        )
        declared_effects = _ordered_unique(request.composed_effects)
        component_rank = _effect_rank(component_effects)
        declared_rank = _effect_rank(declared_effects)
        if set(declared_effects) != set(component_effects) or declared_rank > component_rank:
            fail("effect", CompositionReason.HIDDEN_EFFECT_ESCALATION)
        else:
            pass_dimension("effect")

        authority_reason = self._authority_reason(operators, request.composed_authority)
        if authority_reason is None:
            pass_dimension("authority")
        else:
            fail("authority", authority_reason)

        if self._budget_compatible(operators, request.composed_resources):
            pass_dimension("budget")
        else:
            fail("budget", CompositionReason.BUDGET_INCOMPATIBLE)

        if self._rollback_complete(operators, request.composed_rollback):
            pass_dimension("rollback")
        else:
            fail("rollback", CompositionReason.ROLLBACK_INCOMPLETE)

        if self._validation_complete(operators, request.composed_validation):
            pass_dimension("validation")
        else:
            fail("validation", CompositionReason.VALIDATION_INCOMPLETE)

        accepted = not incompatible and set(compatible) == set(REQUIRED_COMPOSITION_DIMENSIONS)
        if accepted:
            action = CompositionAction.ACCEPT
            reason = CompositionReason.COMPATIBLE
        else:
            action = CompositionAction.REJECT
            reason = reasons[0] if reasons else CompositionReason.EFFECT_INCOMPATIBLE
        return CompositionDecision(
            bindings=bindings,
            action=action,
            reason_code=reason,
            procedure_cids=procedure_cids,
            compatible_dimensions=tuple(compatible),
            incompatible_dimensions=tuple(incompatible),
            accepted=accepted,
        )

    def _entailment_reason(
        self,
        operators: Sequence[ProcedureOperator],
        evidence: Sequence[EntailmentEvidence],
    ) -> CompositionReason | None:
        by_pair: dict[tuple[str, str], list[EntailmentEvidence]] = {}
        for item in evidence:
            key = (item.predecessor_procedure_cid, item.successor_procedure_cid)
            by_pair.setdefault(key, []).append(item)
        for index in range(len(operators) - 1):
            predecessor = operators[index]
            successor = operators[index + 1]
            required = tuple(
                condition
                for condition in successor.procedure.preconditions
                if condition.required
            )
            if not required:
                return CompositionReason.ENTAILMENT_MISSING
            offered = by_pair.get(
                (predecessor.procedure_cid, successor.procedure_cid), ()
            )
            posts = {
                condition.condition_id: condition
                for condition in predecessor.procedure.postconditions
            }
            pres = {condition.condition_id: condition for condition in required}
            covered: set[str] = set()
            for item in offered:
                post = posts.get(item.predecessor_postcondition_id)
                pre = pres.get(item.successor_precondition_id)
                if post is None or pre is None:
                    return CompositionReason.ENTAILMENT_INEXACT
                if _condition_signature(post) != _condition_signature(pre):
                    return CompositionReason.ENTAILMENT_INEXACT
                if not item.evidence_cid:
                    return CompositionReason.ENTAILMENT_MISSING
                covered.add(pre.condition_id)
            if set(pres) != covered:
                return CompositionReason.ENTAILMENT_MISSING
        return None

    def _authority_reason(
        self,
        operators: Sequence[ProcedureOperator],
        declared: ProcedureAuthorityEnvelope,
    ) -> CompositionReason | None:
        policies = {item.procedure.authority.authority_policy_revision for item in operators}
        if len(policies) != 1:
            return CompositionReason.AUTHORITY_INCOMPATIBLE
        policy = next(iter(policies))
        requirements = _ordered_unique(
            tuple(
                req
                for item in operators
                for req in item.procedure.authority.requirement_ids
            )
        )
        capabilities = _ordered_unique(
            tuple(
                cap
                for item in operators
                for cap in item.procedure.authority.required_capability_ids
            )
        )
        operations = _ordered_unique(
            tuple(
                op
                for item in operators
                for op in item.procedure.authority.allowed_operations
            )
        )
        risk = max(
            (item.procedure.authority.risk_ceiling for item in operators),
            key=_risk_rank,
        )
        confirmation = any(item.procedure.authority.confirmation_required for item in operators)
        if declared.authority_policy_revision != policy:
            return CompositionReason.AUTHORITY_INCOMPATIBLE
        if set(declared.requirement_ids) != set(requirements):
            extra = set(declared.requirement_ids) - set(requirements)
            return (
                CompositionReason.HIDDEN_AUTHORITY_ESCALATION
                if extra
                else CompositionReason.AUTHORITY_INCOMPATIBLE
            )
        if set(declared.required_capability_ids) != set(capabilities):
            extra = set(declared.required_capability_ids) - set(capabilities)
            return (
                CompositionReason.HIDDEN_AUTHORITY_ESCALATION
                if extra
                else CompositionReason.AUTHORITY_INCOMPATIBLE
            )
        if set(declared.allowed_operations) != set(operations):
            extra = set(declared.allowed_operations) - set(operations)
            return (
                CompositionReason.HIDDEN_AUTHORITY_ESCALATION
                if extra
                else CompositionReason.AUTHORITY_INCOMPATIBLE
            )
        if _risk_rank(declared.risk_ceiling) > _risk_rank(risk):
            return CompositionReason.HIDDEN_AUTHORITY_ESCALATION
        if declared.risk_ceiling != risk:
            return CompositionReason.AUTHORITY_INCOMPATIBLE
        if declared.confirmation_required and not confirmation:
            return CompositionReason.HIDDEN_AUTHORITY_ESCALATION
        if declared.confirmation_required != confirmation:
            return CompositionReason.AUTHORITY_INCOMPATIBLE
        return None

    def _budget_compatible(
        self,
        operators: Sequence[ProcedureOperator],
        declared: ProcedureResourceEnvelope,
    ) -> bool:
        totals = {name: 0 for name in RESOURCE_FIELDS}
        for operator in operators:
            resources = operator.procedure.resources
            if resources is None:
                return False
            for name in RESOURCE_FIELDS:
                totals[name] += int(getattr(resources, name))
        for name in RESOURCE_FIELDS:
            observed = int(getattr(declared, name))
            if observed < totals[name]:
                return False
            if observed > RESOURCE_MAXIMA[name]:
                return False
        if declared.wall_time_ms <= 0:
            return False
        return True

    def _rollback_complete(
        self,
        operators: Sequence[ProcedureOperator],
        declared: Sequence[ProcedureRollback],
    ) -> bool:
        if not declared:
            return False
        required_effects = {
            effect_id
            for operator in operators
            for rollback in operator.procedure.rollback
            for effect_id in rollback.trigger_effect_ids
        }
        required_steps = {
            step_id
            for operator in operators
            for rollback in operator.procedure.rollback
            for step_id in rollback.step_ids
        }
        required_observations = {
            observation_id
            for operator in operators
            for rollback in operator.procedure.rollback
            for observation_id in rollback.verification_observation_ids
        }
        if not required_effects or not required_steps or not required_observations:
            return False
        declared_effects = {
            effect_id for rollback in declared for effect_id in rollback.trigger_effect_ids
        }
        declared_steps = {step_id for rollback in declared for step_id in rollback.step_ids}
        declared_observations = {
            observation_id
            for rollback in declared
            for observation_id in rollback.verification_observation_ids
        }
        if not required_effects.issubset(declared_effects):
            return False
        if not required_steps.issubset(declared_steps):
            return False
        if not required_observations.issubset(declared_observations):
            return False
        if any(not rollback.exact_target_cid for rollback in declared):
            return False
        return True

    def _validation_complete(
        self,
        operators: Sequence[ProcedureOperator],
        declared: ProcedureValidationPlan,
    ) -> bool:
        required_steps = {
            step_id
            for operator in operators
            for step_id in operator.procedure.validation.required_step_ids
        }
        required_observations = {
            observation_id
            for operator in operators
            for observation_id in operator.procedure.validation.required_observation_ids
        }
        required_tests = {
            contract
            for operator in operators
            for contract in operator.procedure.validation.required_test_contracts
        }
        required_proofs = {
            contract
            for operator in operators
            for contract in operator.procedure.validation.required_proof_contracts
        }
        required_post_merge = {
            contract
            for operator in operators
            for contract in operator.procedure.validation.post_merge_validation_contracts
        }
        if set(declared.required_step_ids) != required_steps:
            return False
        if set(declared.required_observation_ids) != required_observations:
            return False
        if set(declared.required_test_contracts) != required_tests:
            return False
        if set(declared.required_proof_contracts) != required_proofs:
            return False
        if set(declared.post_merge_validation_contracts) != required_post_merge:
            return False
        return True


class ProcedurePlannerAdapter:
    """Expose verified procedures as planner operators on exact boundaries."""

    revision: Final[str] = ADAPTER_REVISION

    def __init__(
        self,
        *,
        registry: ProcedureRegistry | None = None,
        compatibility_probe: PlannerCompatibilityProbe | None = None,
        composition_validator: ProcedureCompositionValidator | None = None,
    ) -> None:
        if registry is not None and not isinstance(registry, ProcedureRegistry):
            raise PlannerAdapterError("registry must be a ProcedureRegistry")
        self._registry = registry
        self._compatibility_probe = compatibility_probe or probe_adaptive_planner_compatibility
        self._composition_validator = composition_validator or ProcedureCompositionValidator()

    @property
    def planner_operator_order(self) -> tuple[PlannerOperatorKind, ...]:
        return PLANNER_OPERATOR_ORDER

    def probe_compatibility(self) -> PlannerCompatibility:
        result = self._compatibility_probe()
        if not isinstance(result, PlannerCompatibility):
            raise PlannerCompatibilityError("compatibility probe must return PlannerCompatibility")
        return result

    def match(
        self,
        request: PlannerMatchRequest,
        operator: ProcedureOperator,
    ) -> PlannerMatchDecision:
        if not isinstance(request, PlannerMatchRequest):
            raise PlannerMatchError("match request is required")
        if not isinstance(operator, ProcedureOperator):
            raise PlannerMatchError("match operator must be ProcedureOperator")
        compatible: list[str] = []
        incompatible: list[str] = []

        def check(dimension: str, ok: bool) -> None:
            if ok:
                compatible.append(dimension)
            else:
                incompatible.append(dimension)

        check(
            "claim",
            operator.claim_scope is request.claim_scope and operator.claim_id == request.claim_id,
        )
        check("task_family", operator.procedure.task_family_id == request.task_family_id)
        check(
            "repository",
            operator.bindings.repository_id == request.bindings.repository_id,
        )
        check("tree", operator.bindings.tree_id == request.bindings.tree_id)
        check(
            "policy",
            operator.bindings.policy_revision == request.bindings.policy_revision
            and operator.procedure.authority.authority_policy_revision
            == request.authority_policy_revision,
        )
        check(
            "environment",
            operator.bindings.environment_id == request.bindings.environment_id,
        )
        check(
            "language",
            _set_eq(operator.certificate.supported_language_classes, request.language_classes),
        )
        check(
            "framework",
            _set_eq(
                operator.certificate.supported_framework_classes, request.framework_classes
            ),
        )
        check("effect", _set_eq(_effect_classes(operator.procedure), request.effect_classes))
        check(
            "authority",
            _set_eq(
                operator.procedure.authority.requirement_ids,
                request.authority_requirement_ids,
            )
            and _set_eq(
                operator.procedure.authority.required_capability_ids,
                request.required_capability_ids,
            ),
        )
        check(
            "validation",
            _set_eq(
                _validation_contracts(operator.procedure.validation),
                request.validation_contracts,
            ),
        )
        check("risk", operator.procedure.authority.risk_ceiling is request.risk_ceiling)
        check("scope", _match_scope(operator.procedure, request.scope_paths))
        registry_ok = self._registry_usable(operator)
        check("certificate", _certificate_verified(operator) and registry_ok)

        matched = not incompatible and set(compatible) == set(REQUIRED_MATCH_DIMENSIONS)
        if matched:
            partial = request.claim_scope is not ProcedureClaimScope.TASK
            return PlannerMatchDecision(
                bindings=request.bindings,
                action=PlannerMatchAction.MATCH,
                reason_code=(
                    PlannerMatchReason.PARTIAL_CRITERION
                    if partial
                    else PlannerMatchReason.EXACT_COMPATIBLE
                ),
                claim_scope=request.claim_scope,
                claim_id=request.claim_id,
                procedure_cid=operator.procedure_cid,
                compatible_dimensions=tuple(compatible),
                incompatible_dimensions=(),
                matched=True,
                claims_task=request.claim_scope is ProcedureClaimScope.TASK,
            )

        if not registry_ok and "certificate" in incompatible:
            reason = PlannerMatchReason.REGISTRY_UNUSABLE
        elif not _certificate_verified(operator):
            reason = PlannerMatchReason.CERTIFICATE_UNVERIFIED
        elif len(incompatible) == 1:
            reason = _MATCH_DIMENSION_REASON[incompatible[0]]
        elif operator.procedure.task_family_id == request.task_family_id:
            reason = PlannerMatchReason.NEAR_MATCH_REJECTED
        elif len(compatible) and len(incompatible):
            reason = PlannerMatchReason.PARTIAL_DIMENSION
        else:
            reason = _MATCH_DIMENSION_REASON[incompatible[0]]
        return PlannerMatchDecision(
            bindings=request.bindings,
            action=PlannerMatchAction.REJECT,
            reason_code=reason,
            claim_scope=request.claim_scope,
            claim_id=request.claim_id,
            procedure_cid=operator.procedure_cid,
            compatible_dimensions=tuple(compatible),
            incompatible_dimensions=tuple(incompatible),
            matched=False,
            claims_task=False,
        )

    def compose(self, request: CompositionRequest) -> CompositionDecision:
        return self._composition_validator.validate(request)

    def plan(self, request: PlannerDispatchRequest) -> PlannerDispatchDecision:
        if not isinstance(request, PlannerDispatchRequest):
            raise PlannerAdapterError("dispatch request is required")
        compatibility = self.probe_compatibility()
        if not compatibility.qualified:
            return PlannerDispatchDecision(
                bindings=request.match.bindings,
                action=PlannerDispatchAction.UNAVAILABLE,
                reason_code=PlannerDispatchReason.ADAPTIVE_PLANNER_INCOMPATIBLE,
                compatibility_status=compatibility.status,
                compatibility_reason_code=compatibility.reason_code,
                selected_kind="",
                considered_kinds=(),
                procedure_cids=(),
                dispatched=False,
                diagnostic=compatibility.diagnostic,
                blocker=compatibility.blocker or compatibility.reason_code,
            )

        considered = tuple(item.value for item in PLANNER_OPERATOR_ORDER)
        exact: list[ProcedureOperator] = []
        for operator in sorted(request.operators, key=lambda item: item.procedure_cid):
            decision = self.match(request.match, operator)
            if decision.matched:
                exact.append(operator)

        composition_accepted = False
        composition_cids: tuple[str, ...] = ()
        if request.composition is not None:
            composition = self.compose(request.composition)
            composition_accepted = composition.accepted and self._composition_matches_request(
                request.match, request.composition.operators
            )
            if composition_accepted:
                composition_cids = composition.procedure_cids

        if exact:
            selected = PlannerOperatorKind.EXACT_VERIFIED_PROCEDURE
            procedure_cids = (exact[0].procedure_cid,)
            dispatched = True
        elif composition_accepted:
            selected = PlannerOperatorKind.COMPOSABLE_VERIFIED_PROCEDURES
            procedure_cids = composition_cids
            dispatched = True
        else:
            selected = PlannerOperatorKind.DETERMINISTIC_BASELINE
            procedure_cids = ()
            dispatched = False

        return PlannerDispatchDecision(
            bindings=request.match.bindings,
            action=PlannerDispatchAction.CANDIDATES,
            reason_code=PlannerDispatchReason.QUALIFIED_ORDER,
            compatibility_status=compatibility.status,
            compatibility_reason_code=compatibility.reason_code,
            selected_kind=selected.value,
            considered_kinds=considered,
            procedure_cids=procedure_cids,
            dispatched=dispatched,
        )

    def _composition_matches_request(
        self,
        request: PlannerMatchRequest,
        operators: Sequence[ProcedureOperator],
    ) -> bool:
        for operator in operators:
            if operator.procedure.task_family_id != request.task_family_id:
                return False
            if operator.bindings.repository_id != request.bindings.repository_id:
                return False
            if operator.bindings.tree_id != request.bindings.tree_id:
                return False
            if operator.bindings.policy_revision != request.bindings.policy_revision:
                return False
            if operator.bindings.environment_id != request.bindings.environment_id:
                return False
            if not _certificate_verified(operator):
                return False
        return True

    def _registry_usable(self, operator: ProcedureOperator) -> bool:
        if self._registry is None:
            if operator.registry_state:
                return operator.registry_state in {item.value for item in USABLE_STATES}
            return True
        revision = self._registry.lookup_exact(
            operator.procedure_cid, bindings=operator.bindings, usable_only=True
        )
        if revision is None:
            return False
        if operator.registry_revision_id and operator.registry_revision_id != revision.revision_id:
            return False
        return revision.state in USABLE_STATES


def match_procedure_operator(
    request: PlannerMatchRequest,
    operator: ProcedureOperator,
    *,
    registry: ProcedureRegistry | None = None,
) -> PlannerMatchDecision:
    return ProcedurePlannerAdapter(registry=registry).match(request, operator)


def compose_procedure_operators(request: CompositionRequest) -> CompositionDecision:
    return ProcedureCompositionValidator().validate(request)


ARTIFACT_TYPES_BY_SCHEMA[ProcedureOperator.SCHEMA] = ProcedureOperator
ARTIFACT_TYPES_BY_SCHEMA[PlannerMatchDecision.SCHEMA] = PlannerMatchDecision
ARTIFACT_TYPES_BY_SCHEMA[CompositionDecision.SCHEMA] = CompositionDecision
ARTIFACT_TYPES_BY_SCHEMA[PlannerDispatchDecision.SCHEMA] = PlannerDispatchDecision


__all__ = [
    "ADAPTER_REVISION",
    "ADAPTIVE_PLANNER_HAMMER_BLOCKER",
    "ADAPTIVE_PLANNER_MODULE",
    "COMPOSITION_VALIDATOR_REVISION",
    "HAMMER_TRACE_SCHEMA_SIGNATURE",
    "OPERATOR_REVISION",
    "PLANNER_OPERATOR_ORDER",
    "PROCEDURE_OPERATOR_KINDS",
    "REQUIRED_COMPOSITION_DIMENSIONS",
    "REQUIRED_MATCH_DIMENSIONS",
    "CompositionAction",
    "CompositionDecision",
    "CompositionReason",
    "CompositionRequest",
    "EntailmentEvidence",
    "PlannerAdapterError",
    "PlannerCompatibility",
    "PlannerCompatibilityError",
    "PlannerCompatibilityStatus",
    "PlannerCompositionError",
    "PlannerDispatchAction",
    "PlannerDispatchDecision",
    "PlannerDispatchReason",
    "PlannerDispatchRequest",
    "PlannerMatchAction",
    "PlannerMatchDecision",
    "PlannerMatchError",
    "PlannerMatchReason",
    "PlannerMatchRequest",
    "PlannerOperatorKind",
    "ProcedureClaimScope",
    "ProcedureCompositionValidator",
    "ProcedureOperator",
    "ProcedurePlannerAdapter",
    "compose_procedure_operators",
    "match_procedure_operator",
    "probe_adaptive_planner_compatibility",
    "qualified_planner_compatibility",
]
