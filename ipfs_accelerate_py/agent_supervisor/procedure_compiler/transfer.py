"""Cross-repository procedure transfer checks (PCPC-025).

A transfer never mutates a target repository.  Compatible transfers receive
bounded candidate eligibility only: they cannot grant authority, promotion,
completion, proof, or postconditions.  Every changed assumption is a typed
refusal.  Similar names, embeddings, descriptions, language, or maintainer
never establish portability.  The unsafe-transfer count remains zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    MAX_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    GeneralizationBoundary as GeneralizationBoundaryArtifact,
    ProcedureCertificate,
    ProcedureContractError,
    ProcedureSpec,
    ProcedureValidationPlan,
    RiskClass,
    StepOperation,
    TaskFamily,
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
from .task_family import (
    REQUIRED_BOUNDARY_DIMENSIONS,
    BoundaryViolationClass,
    TaskFamilyBoundaryError,
    TaskFamilyBoundaryValidator,
)


GATE_REVISION: Final[str] = "procedure-transfer-gate@1"
BOUNDARY_EVALUATOR_REVISION: Final[str] = "generalization-boundary-evaluator@1"
UNSAFE_TRANSFER_COUNT: Final[int] = 0

REQUIRED_COMPATIBILITY_DIMENSIONS: Final[tuple[str, ...]] = (
    "operation",
    "effect",
    "authority",
    "language",
    "framework",
    "validation",
    "family",
    "path",
    "held-out",
)
INSUFFICIENT_SIMILARITY_SIGNALS: Final[tuple[str, ...]] = (
    "name",
    "embedding",
    "description",
    "language",
    "maintainer",
)

_RISK_RANK: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}


class TransferError(ProcedureContractError):
    """A transfer request, assumption, or decision is unsafe."""


class TransferDeclarationError(TransferError):
    """A required transfer assumption is missing or malformed."""


class TransferRefusalError(TransferError):
    """A transfer request was refused and the caller required admission."""

    def __init__(self, message: str, decision: TransferDecision | None = None) -> None:
        super().__init__(message)
        self.decision = decision


class TransferAction(str, Enum):
    ELIGIBLE = "eligible"
    REFUSE = "refuse"


class TransferReason(str, Enum):
    COMPATIBLE = "compatible"
    OPERATION_INCOMPATIBLE = "operation-incompatible"
    EFFECT_INCOMPATIBLE = "effect-incompatible"
    AUTHORITY_INCOMPATIBLE = "authority-incompatible"
    LANGUAGE_INCOMPATIBLE = "language-incompatible"
    FRAMEWORK_INCOMPATIBLE = "framework-incompatible"
    VALIDATION_INCOMPATIBLE = "validation-incompatible"
    FAMILY_INCOMPATIBLE = "family-incompatible"
    PATH_INCOMPATIBLE = "path-incompatible"
    HELD_OUT_MISSING = "held-out-missing"
    HELD_OUT_FAILED = "held-out-failed"
    HELD_OUT_REPOSITORY_MISMATCH = "held-out-repository-mismatch"
    HELD_OUT_UNSAFE = "held-out-unsafe"
    SIMILARITY_INSUFFICIENT = "similarity-insufficient"
    CROSS_REPOSITORY_MUTATION = "cross-repository-mutation"
    PRODUCTION_MUTATION = "production-mutation"
    POLICY_MUTATION = "policy-mutation"
    UNSAFE_FIXTURE = "unsafe-fixture"
    BINDING_MISMATCH = "binding-mismatch"
    MISSING_CERTIFICATE = "missing-certificate"
    MISSING_FAMILY = "missing-family"
    INCOMPLETE_BOUNDARY = "incomplete-boundary"
    INCOMPLETE_ASSUMPTIONS = "incomplete-assumptions"
    RISK_CEILING = "risk-ceiling"
    TARGET_NOT_AUTHORIZED = "target-not-authorized"
    EXPERIMENT_CANNOT_AUTHORIZE = "experiment-cannot-authorize"
    ELIGIBILITY_NOT_CANDIDATE = "eligibility-not-candidate"


class TransferDimension(str, Enum):
    OPERATION = "operation"
    EFFECT = "effect"
    AUTHORITY = "authority"
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    VALIDATION = "validation"
    FAMILY = "family"
    PATH = "path"
    HELD_OUT = "held-out"


_DIMENSION_REASON: Final[dict[TransferDimension, TransferReason]] = {
    TransferDimension.OPERATION: TransferReason.OPERATION_INCOMPATIBLE,
    TransferDimension.EFFECT: TransferReason.EFFECT_INCOMPATIBLE,
    TransferDimension.AUTHORITY: TransferReason.AUTHORITY_INCOMPATIBLE,
    TransferDimension.LANGUAGE: TransferReason.LANGUAGE_INCOMPATIBLE,
    TransferDimension.FRAMEWORK: TransferReason.FRAMEWORK_INCOMPATIBLE,
    TransferDimension.VALIDATION: TransferReason.VALIDATION_INCOMPATIBLE,
    TransferDimension.FAMILY: TransferReason.FAMILY_INCOMPATIBLE,
    TransferDimension.PATH: TransferReason.PATH_INCOMPATIBLE,
    TransferDimension.HELD_OUT: TransferReason.HELD_OUT_MISSING,
}


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TransferError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _risk_rank(value: RiskClass) -> int:
    return _RISK_RANK[value]


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for item in values:
        if item not in result:
            result.append(item)
    return tuple(result)


def _subset(required: Sequence[Any], available: Sequence[Any]) -> bool:
    return set(required).issubset(set(available))


def _path_is_within(path: str, prefixes: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    for prefix in prefixes:
        root = PurePosixPath(prefix)
        if prefix == ".":
            return True
        if candidate == root or root in candidate.parents:
            return True
    return False


def _source_path_assumptions(procedure: ProcedureSpec) -> tuple[str, ...]:
    paths = list(procedure.scope_paths)
    for item in procedure.declared_reads:
        if item not in paths:
            paths.append(item)
    for effect in procedure.declared_effects:
        for target in effect.targets:
            if target not in paths:
                paths.append(target)
    return tuple(paths)


def _incomplete_family_dimensions(family: TaskFamily) -> tuple[str, ...]:
    boundary = family.boundary
    present = {
        "positive_member_cids": boundary.positive_member_cids,
        "negative_example_cids": boundary.negative_example_cids,
        "boundary_example_cids": boundary.boundary_example_cids,
        "unknown_case_cids": boundary.unknown_case_cids,
        "risk_ceiling": (boundary.risk_ceiling.value,),
        "permitted_repositories": boundary.permitted_repositories,
        "permitted_languages": boundary.permitted_languages,
        "permitted_frameworks": boundary.permitted_frameworks,
        "permitted_effect_classes": boundary.permitted_effect_classes,
        "required_operation_contracts": family.required_operation_contracts,
        "validation_structure": family.validation_structure,
        "rollback_structure": family.rollback_structure,
        "postcondition_shape": family.postcondition_shape,
    }
    return tuple(name for name in REQUIRED_BOUNDARY_DIMENSIONS if not present[name])


def _nested_record(value: Any, cls: type[Any], field_name: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_record(value)
    raise TransferDeclarationError(f"{field_name} must be {cls.__name__}")


@dataclass(frozen=True)
class SimilaritySignals:
    """Surface similarity that never establishes portability."""

    name_similar: bool = False
    embedding_similar: bool = False
    description_similar: bool = False
    language_similar: bool = False
    maintainer_similar: bool = False

    def __post_init__(self) -> None:
        for name in (
            "name_similar",
            "embedding_similar",
            "description_similar",
            "language_similar",
            "maintainer_similar",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))

    @property
    def any_similar(self) -> bool:
        return any(
            (
                self.name_similar,
                self.embedding_similar,
                self.description_similar,
                self.language_similar,
                self.maintainer_similar,
            )
        )

    @property
    def asserted_signals(self) -> tuple[str, ...]:
        mapping = (
            ("name", self.name_similar),
            ("embedding", self.embedding_similar),
            ("description", self.description_similar),
            ("language", self.language_similar),
            ("maintainer", self.maintainer_similar),
        )
        return tuple(name for name, flag in mapping if flag)

    def to_record(self) -> dict[str, bool]:
        return {
            "name_similar": self.name_similar,
            "embedding_similar": self.embedding_similar,
            "description_similar": self.description_similar,
            "language_similar": self.language_similar,
            "maintainer_similar": self.maintainer_similar,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> SimilaritySignals:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TransferDeclarationError("similarity must be a mapping")
        return cls(
            name_similar=payload.get("name_similar", False),
            embedding_similar=payload.get("embedding_similar", False),
            description_similar=payload.get("description_similar", False),
            language_similar=payload.get("language_similar", False),
            maintainer_similar=payload.get("maintainer_similar", False),
        )


@dataclass(frozen=True)
class TargetRepository:
    """Declared target-repository assumptions.  Similarity fields are non-evidence."""

    repository_id: str
    tree_id: str
    repository_family: str
    language_classes: tuple[str, ...]
    framework_classes: tuple[str, ...]
    permitted_operations: tuple[StepOperation, ...]
    permitted_operation_contracts: tuple[str, ...]
    permitted_effect_classes: tuple[EffectClass, ...]
    authority_policy_revision: str
    authority_requirement_ids: tuple[str, ...]
    required_capability_ids: tuple[str, ...]
    validation_contracts: tuple[str, ...]
    validation_structure: tuple[str, ...]
    path_prefixes: tuple[str, ...]
    risk_ceiling: RiskClass
    operation_catalog_revision: str
    effect_policy_revision: str
    authorized: bool = True
    production: bool = False
    policy_mutable: bool = False
    name: str = ""
    maintainer_id: str = ""
    description: str = ""
    embedding_id: str = ""

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "repository_family"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "language_classes",
            "framework_classes",
            "permitted_operation_contracts",
            "authority_requirement_ids",
            "required_capability_ids",
            "validation_contracts",
            "validation_structure",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        object.__setattr__(
            self,
            "permitted_operations",
            _enums(
                self.permitted_operations,
                StepOperation,
                "permitted_operations",
                limit=len(StepOperation),
            ),
        )
        object.__setattr__(
            self,
            "permitted_effect_classes",
            _enums(
                self.permitted_effect_classes,
                EffectClass,
                "permitted_effect_classes",
                limit=len(EffectClass),
            ),
        )
        for name in (
            "authority_policy_revision",
            "operation_catalog_revision",
            "effect_policy_revision",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "path_prefixes",
            _strings(self.path_prefixes, "path_prefixes", paths=True),
        )
        object.__setattr__(
            self, "risk_ceiling", _enum(self.risk_ceiling, RiskClass, "risk_ceiling")
        )
        for name in ("authorized", "production", "policy_mutable"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        for name in ("name", "description"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        for name in ("maintainer_id", "embedding_id"):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name, required=False),
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "repository_family": self.repository_family,
            "language_classes": self.language_classes,
            "framework_classes": self.framework_classes,
            "permitted_operations": tuple(item.value for item in self.permitted_operations),
            "permitted_operation_contracts": self.permitted_operation_contracts,
            "permitted_effect_classes": tuple(
                item.value for item in self.permitted_effect_classes
            ),
            "authority_policy_revision": self.authority_policy_revision,
            "authority_requirement_ids": self.authority_requirement_ids,
            "required_capability_ids": self.required_capability_ids,
            "validation_contracts": self.validation_contracts,
            "validation_structure": self.validation_structure,
            "path_prefixes": self.path_prefixes,
            "risk_ceiling": self.risk_ceiling.value,
            "operation_catalog_revision": self.operation_catalog_revision,
            "effect_policy_revision": self.effect_policy_revision,
            "authorized": self.authorized,
            "production": self.production,
            "policy_mutable": self.policy_mutable,
            "name": self.name,
            "maintainer_id": self.maintainer_id,
            "description": self.description,
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> TargetRepository:
        if payload is None or not isinstance(payload, Mapping):
            raise TransferDeclarationError("target must be a mapping")
        return cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            repository_family=payload.get("repository_family", ""),
            language_classes=payload.get("language_classes", ()),
            framework_classes=payload.get("framework_classes", ()),
            permitted_operations=payload.get("permitted_operations", ()),
            permitted_operation_contracts=payload.get("permitted_operation_contracts", ()),
            permitted_effect_classes=payload.get("permitted_effect_classes", ()),
            authority_policy_revision=payload.get("authority_policy_revision", ""),
            authority_requirement_ids=payload.get("authority_requirement_ids", ()),
            required_capability_ids=payload.get("required_capability_ids", ()),
            validation_contracts=payload.get("validation_contracts", ()),
            validation_structure=payload.get("validation_structure", ()),
            path_prefixes=payload.get("path_prefixes", ()),
            risk_ceiling=payload.get("risk_ceiling", ""),
            operation_catalog_revision=payload.get("operation_catalog_revision", ""),
            effect_policy_revision=payload.get("effect_policy_revision", ""),
            authorized=payload.get("authorized", True),
            production=payload.get("production", False),
            policy_mutable=payload.get("policy_mutable", False),
            name=payload.get("name", ""),
            maintainer_id=payload.get("maintainer_id", ""),
            description=payload.get("description", ""),
            embedding_id=payload.get("embedding_id", ""),
        )


@dataclass(frozen=True)
class HeldOutRepositoryResult:
    """Explicit held-out evaluation bound to one target repository fixture."""

    evaluation_cid: str
    repository_id: str
    tree_id: str
    passed: bool
    read_only: bool = True
    disposable: bool = True
    production: bool = False
    policy_mutable: bool = False
    authorized: bool = True
    mutate: bool = False
    scope_paths: tuple[str, ...] = ()
    observed_postcondition_count: int = 0
    observed_validation_count: int = 0

    def __post_init__(self) -> None:
        for name in ("evaluation_cid", "repository_id", "tree_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "passed",
            "read_only",
            "disposable",
            "production",
            "policy_mutable",
            "authorized",
            "mutate",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "scope_paths",
            _strings(self.scope_paths, "scope_paths", paths=True),
        )
        object.__setattr__(
            self,
            "observed_postcondition_count",
            _nonnegative_int(
                self.observed_postcondition_count,
                "observed_postcondition_count",
                maximum=MAX_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "observed_validation_count",
            _nonnegative_int(
                self.observed_validation_count,
                "observed_validation_count",
                maximum=MAX_ITEMS,
            ),
        )

    def isolation_reason(self) -> TransferReason | None:
        if self.mutate:
            return TransferReason.CROSS_REPOSITORY_MUTATION
        if self.production:
            return TransferReason.PRODUCTION_MUTATION
        if self.policy_mutable:
            return TransferReason.POLICY_MUTATION
        if not self.read_only or not self.disposable:
            return TransferReason.UNSAFE_FIXTURE
        if not self.authorized:
            return TransferReason.TARGET_NOT_AUTHORIZED
        return None

    @property
    def is_safe_fixture(self) -> bool:
        return self.isolation_reason() is None

    @property
    def is_nonvacuous_pass(self) -> bool:
        return (
            self.passed
            and self.observed_validation_count > 0
            and self.observed_postcondition_count > 0
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "evaluation_cid": self.evaluation_cid,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "passed": self.passed,
            "read_only": self.read_only,
            "disposable": self.disposable,
            "production": self.production,
            "policy_mutable": self.policy_mutable,
            "authorized": self.authorized,
            "mutate": self.mutate,
            "scope_paths": self.scope_paths,
            "observed_postcondition_count": self.observed_postcondition_count,
            "observed_validation_count": self.observed_validation_count,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> HeldOutRepositoryResult:
        if payload is None or not isinstance(payload, Mapping):
            raise TransferDeclarationError("held_out must be a mapping")
        return cls(
            evaluation_cid=payload.get("evaluation_cid", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            passed=payload.get("passed", False),
            read_only=payload.get("read_only", True),
            disposable=payload.get("disposable", True),
            production=payload.get("production", False),
            policy_mutable=payload.get("policy_mutable", False),
            authorized=payload.get("authorized", True),
            mutate=payload.get("mutate", False),
            scope_paths=payload.get("scope_paths", ()),
            observed_postcondition_count=payload.get("observed_postcondition_count", 0),
            observed_validation_count=payload.get("observed_validation_count", 0),
        )


@dataclass(frozen=True)
class TransferRequest:
    """One cross-repository transfer check.  Mutation of the target is forbidden."""

    bindings: ArtifactBindings
    procedure: ProcedureSpec
    certificate: ProcedureCertificate
    family: TaskFamily
    target: TargetRepository
    held_out: HeldOutRepositoryResult | None = None
    similarity: SimilaritySignals = field(default_factory=SimilaritySignals)
    mutate_target: bool = False
    experiment_authorizes: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        if not isinstance(self.procedure, ProcedureSpec):
            raise TransferDeclarationError("procedure must be ProcedureSpec")
        if not isinstance(self.certificate, ProcedureCertificate):
            raise TransferDeclarationError("certificate must be ProcedureCertificate")
        if not isinstance(self.family, TaskFamily):
            raise TransferDeclarationError("family must be TaskFamily")
        object.__setattr__(
            self, "target", _nested_record(self.target, TargetRepository, "target")
        )
        if self.held_out is not None:
            object.__setattr__(
                self,
                "held_out",
                _nested_record(self.held_out, HeldOutRepositoryResult, "held_out"),
            )
        object.__setattr__(
            self,
            "similarity",
            _nested_record(self.similarity, SimilaritySignals, "similarity"),
        )
        object.__setattr__(self, "mutate_target", _bool(self.mutate_target, "mutate_target"))
        object.__setattr__(
            self,
            "experiment_authorizes",
            _bool(self.experiment_authorizes, "experiment_authorizes"),
        )


@dataclass(frozen=True)
class GeneralizationBoundaryEvaluation:
    """Typed family/path/effect boundary result used by the transfer gate."""

    admitted: bool
    reason_code: TransferReason
    changed_dimensions: tuple[TransferDimension, ...] = ()
    reason_codes: tuple[TransferReason, ...] = ()
    violation_classes: tuple[str, ...] = ()
    missing_dimensions: tuple[str, ...] = ()
    artifact: GeneralizationBoundaryArtifact | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, TransferReason, "reason_code")
        )
        object.__setattr__(
            self,
            "changed_dimensions",
            _enums(
                self.changed_dimensions,
                TransferDimension,
                "changed_dimensions",
                limit=len(TransferDimension),
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _enums(
                self.reason_codes,
                TransferReason,
                "reason_codes",
                limit=len(TransferReason),
            ),
        )
        object.__setattr__(
            self,
            "violation_classes",
            _strings(self.violation_classes, "violation_classes", identifiers=True),
        )
        object.__setattr__(
            self,
            "missing_dimensions",
            _strings(self.missing_dimensions, "missing_dimensions", identifiers=True),
        )


@dataclass(frozen=True)
class TransferDecision(CanonicalContract):
    """Typed eligible/refuse result.  Eligibility is candidate-only."""

    SCHEMA: ClassVar[str] = _schema_name("TransferDecision")

    bindings: ArtifactBindings
    source_procedure_cid: str
    source_certificate_cid: str
    source_family_cid: str
    target_repository_id: str
    target_tree_id: str
    action: TransferAction
    reason_code: TransferReason
    reason_codes: tuple[TransferReason, ...] = ()
    changed_assumptions: tuple[str, ...] = ()
    compatible_dimensions: tuple[str, ...] = ()
    held_out_evaluation_cid: str = ""
    held_out_passed: bool = False
    eligibility_state: ArtifactState = ArtifactState.REJECTED
    eligible: bool = False
    unsafe_transfer_count: int = UNSAFE_TRANSFER_COUNT
    can_mutate_target: bool = False
    can_authorize: bool = False
    can_promote: bool = False
    similarity_used_as_evidence: bool = False
    similarity_signals: tuple[str, ...] = ()
    gate_revision: str = GATE_REVISION
    boundary_reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "source_procedure_cid",
            "source_certificate_cid",
            "source_family_cid",
            "target_repository_id",
            "target_tree_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "action", _enum(self.action, TransferAction, "action"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, TransferReason, "reason_code")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _enums(
                self.reason_codes or (self.reason_code,),
                TransferReason,
                "reason_codes",
                limit=len(TransferReason),
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "changed_assumptions",
            _strings(self.changed_assumptions, "changed_assumptions", identifiers=True),
        )
        object.__setattr__(
            self,
            "compatible_dimensions",
            _strings(self.compatible_dimensions, "compatible_dimensions", identifiers=True),
        )
        object.__setattr__(
            self,
            "held_out_evaluation_cid",
            _identifier(
                self.held_out_evaluation_cid, "held_out_evaluation_cid", required=False
            ),
        )
        object.__setattr__(
            self, "held_out_passed", _bool(self.held_out_passed, "held_out_passed")
        )
        object.__setattr__(
            self,
            "eligibility_state",
            _enum(self.eligibility_state, ArtifactState, "eligibility_state"),
        )
        for name in (
            "eligible",
            "can_mutate_target",
            "can_authorize",
            "can_promote",
            "similarity_used_as_evidence",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "unsafe_transfer_count",
            _nonnegative_int(self.unsafe_transfer_count, "unsafe_transfer_count"),
        )
        object.__setattr__(
            self,
            "similarity_signals",
            _strings(self.similarity_signals, "similarity_signals", identifiers=True),
        )
        object.__setattr__(
            self, "gate_revision", _identifier(self.gate_revision, "gate_revision")
        )
        object.__setattr__(
            self,
            "boundary_reason_code",
            _identifier(self.boundary_reason_code, "boundary_reason_code", required=False),
        )
        if self.gate_revision != GATE_REVISION:
            raise TransferError("transfer decision gate revision is not current")
        if self.can_mutate_target or self.can_authorize or self.can_promote:
            raise TransferError("transfer decisions cannot mutate, authorize, or promote")
        if self.similarity_used_as_evidence:
            raise TransferError("similarity cannot be used as transfer evidence")
        if self.unsafe_transfer_count != UNSAFE_TRANSFER_COUNT:
            raise TransferError("unsafe transfer count must remain zero")
        if self.action is TransferAction.ELIGIBLE:
            if not self.eligible:
                raise TransferError("an eligible transfer must set eligible")
            if self.eligibility_state is not ArtifactState.CANDIDATE:
                raise TransferError("compatible transfer eligibility is candidate-only")
            if self.reason_code is not TransferReason.COMPATIBLE:
                raise TransferError("an eligible transfer must be labeled compatible")
            if self.changed_assumptions:
                raise TransferError("an eligible transfer cannot carry changed assumptions")
            if set(self.compatible_dimensions) != set(REQUIRED_COMPATIBILITY_DIMENSIONS):
                raise TransferError(
                    "an eligible transfer must retain every compatibility dimension"
                )
            if not self.held_out_passed or not self.held_out_evaluation_cid:
                raise TransferError("an eligible transfer requires explicit held-out results")
        else:
            if self.eligible:
                raise TransferError("a refused transfer cannot be eligible")
            if self.eligibility_state not in {
                ArtifactState.REJECTED,
                ArtifactState.CANDIDATE,
            }:
                raise TransferError("a refused transfer cannot claim verified or promoted state")
            if self.reason_code is TransferReason.COMPATIBLE:
                raise TransferError("a refused transfer cannot be labeled compatible")
        if self.eligibility_state in {
            ArtifactState.VERIFIED,
            ArtifactState.PROMOTED,
            ArtifactState.SHADOW,
        }:
            raise TransferError("transfer eligibility cannot claim verified or promoted state")
        _bounded(self, "TransferDecision")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_establish_proof(self) -> bool:
        return False

    @property
    def can_establish_postcondition(self) -> bool:
        return False

    @property
    def can_establish_completion(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "source_procedure_cid": self.source_procedure_cid,
            "source_certificate_cid": self.source_certificate_cid,
            "source_family_cid": self.source_family_cid,
            "target_repository_id": self.target_repository_id,
            "target_tree_id": self.target_tree_id,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "reason_codes": tuple(item.value for item in self.reason_codes),
            "changed_assumptions": self.changed_assumptions,
            "compatible_dimensions": self.compatible_dimensions,
            "held_out_evaluation_cid": self.held_out_evaluation_cid,
            "held_out_passed": self.held_out_passed,
            "eligibility_state": self.eligibility_state.value,
            "eligible": self.eligible,
            "unsafe_transfer_count": UNSAFE_TRANSFER_COUNT,
            "can_mutate_target": False,
            "can_authorize": False,
            "can_promote": False,
            "similarity_used_as_evidence": False,
            "similarity_signals": self.similarity_signals,
            "gate_revision": self.gate_revision,
            "boundary_reason_code": self.boundary_reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TransferDecision:
        fields = (
            "bindings",
            "source_procedure_cid",
            "source_certificate_cid",
            "source_family_cid",
            "target_repository_id",
            "target_tree_id",
            "action",
            "reason_code",
            "reason_codes",
            "changed_assumptions",
            "compatible_dimensions",
            "held_out_evaluation_cid",
            "held_out_passed",
            "eligibility_state",
            "eligible",
            "unsafe_transfer_count",
            "can_mutate_target",
            "can_authorize",
            "can_promote",
            "similarity_used_as_evidence",
            "similarity_signals",
            "gate_revision",
            "boundary_reason_code",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        for forbidden in (
            "can_mutate_target",
            "can_authorize",
            "can_promote",
            "similarity_used_as_evidence",
        ):
            if values.get(forbidden):
                raise TransferError("transfer decisions cannot mutate, authorize, or promote")
        if values.get("unsafe_transfer_count", UNSAFE_TRANSFER_COUNT) != UNSAFE_TRANSFER_COUNT:
            raise TransferError("unsafe transfer count must remain zero")
        record = cls(**values)
        _verify_identity(payload, record)
        return record


ARTIFACT_TYPES_BY_SCHEMA[TransferDecision.SCHEMA] = TransferDecision


def _procedure_operations(procedure: ProcedureSpec) -> tuple[StepOperation, ...]:
    result: list[StepOperation] = []
    for step in procedure.steps:
        if step.operation not in result:
            result.append(step.operation)
    return tuple(result)


def _procedure_operation_contracts(procedure: ProcedureSpec) -> tuple[str, ...]:
    return _ordered_unique(tuple(step.operation_contract for step in procedure.steps))


def _procedure_effects(procedure: ProcedureSpec) -> tuple[EffectClass, ...]:
    result: list[EffectClass] = []
    for effect in procedure.declared_effects:
        if effect.effect_class not in result:
            result.append(effect.effect_class)
    return tuple(result)


def _procedure_authorities(procedure: ProcedureSpec) -> tuple[str, ...]:
    ids = list(procedure.authority.requirement_ids)
    for step in procedure.steps:
        for item in step.required_authority_ids:
            if item not in ids:
                ids.append(item)
    return tuple(ids)


def _procedure_validation_contracts(procedure: ProcedureSpec) -> tuple[str, ...]:
    plan = procedure.validation
    if not isinstance(plan, ProcedureValidationPlan):
        return ()
    return _ordered_unique(
        (
            *plan.required_test_contracts,
            *plan.required_proof_contracts,
            *plan.post_merge_validation_contracts,
            *((plan.full_test_fallback_contract,) if plan.full_test_fallback_contract else ()),
        )
    )


def _family_matches_certificate(family: TaskFamily, certificate: ProcedureCertificate) -> bool:
    return certificate.task_family_cid in {family.name, family.content_id}


class GeneralizationBoundaryEvaluator:
    """Evaluate whether a target stays inside a family's declared boundary."""

    revision: ClassVar[str] = BOUNDARY_EVALUATOR_REVISION

    def __init__(self) -> None:
        self._family_validator = TaskFamilyBoundaryValidator()

    def evaluate(
        self,
        family: TaskFamily,
        target: TargetRepository,
        *,
        procedure: ProcedureSpec | None = None,
        certificate: ProcedureCertificate | None = None,
        emitted_at_ms: int = 0,
    ) -> GeneralizationBoundaryEvaluation:
        if not isinstance(family, TaskFamily):
            raise TransferDeclarationError("family must be TaskFamily")
        if not isinstance(target, TargetRepository):
            target = _nested_record(target, TargetRepository, "target")

        changed: list[TransferDimension] = []
        reasons: list[TransferReason] = []
        violations: list[str] = []
        try:
            family = self._family_validator.validate_family(family)
        except TaskFamilyBoundaryError as exc:
            missing = ()
            if exc.decision is not None:
                missing = exc.decision.missing_dimensions
            if not missing:
                missing = _incomplete_family_dimensions(family)
            return self._result(
                family,
                target,
                admitted=False,
                reason=TransferReason.INCOMPLETE_BOUNDARY,
                changed=(TransferDimension.FAMILY,),
                reasons=(TransferReason.INCOMPLETE_BOUNDARY,),
                violations=(BoundaryViolationClass.INCOMPLETE_BOUNDARY.value,),
                missing=missing,
                emitted_at_ms=emitted_at_ms,
            )

        if target.repository_id not in family.boundary.permitted_repositories:
            changed.append(TransferDimension.FAMILY)
            reasons.append(TransferReason.FAMILY_INCOMPATIBLE)
            violations.append(BoundaryViolationClass.REPOSITORY.value)
            violations.append(BoundaryViolationClass.OWNERSHIP.value)
        if not _subset(target.language_classes, family.boundary.permitted_languages):
            changed.append(TransferDimension.LANGUAGE)
            reasons.append(TransferReason.LANGUAGE_INCOMPATIBLE)
            violations.append(BoundaryViolationClass.LANGUAGE.value)
        if not _subset(target.framework_classes, family.boundary.permitted_frameworks):
            changed.append(TransferDimension.FRAMEWORK)
            reasons.append(TransferReason.FRAMEWORK_INCOMPATIBLE)
            violations.append(BoundaryViolationClass.FRAMEWORK.value)
        if not _subset(target.permitted_effect_classes, family.boundary.permitted_effect_classes):
            changed.append(TransferDimension.EFFECT)
            reasons.append(TransferReason.EFFECT_INCOMPATIBLE)
            violations.append(BoundaryViolationClass.EFFECT.value)
        if family.required_operation_contracts and not _subset(
            family.required_operation_contracts, target.permitted_operation_contracts
        ):
            changed.append(TransferDimension.OPERATION)
            reasons.append(TransferReason.OPERATION_INCOMPATIBLE)
            violations.append(BoundaryViolationClass.AUTHORITY.value)
        if family.validation_structure and not _subset(
            family.validation_structure, target.validation_structure
        ):
            changed.append(TransferDimension.VALIDATION)
            reasons.append(TransferReason.VALIDATION_INCOMPATIBLE)
            violations.append(BoundaryViolationClass.VALIDATION.value)
        if _risk_rank(target.risk_ceiling) > _risk_rank(family.boundary.risk_ceiling):
            changed.append(TransferDimension.AUTHORITY)
            reasons.append(TransferReason.RISK_CEILING)
            violations.append(BoundaryViolationClass.RISK_CEILING.value)
        if procedure is not None:
            if not isinstance(procedure, ProcedureSpec):
                raise TransferDeclarationError("procedure must be ProcedureSpec")
            source_paths = _source_path_assumptions(procedure)
            if not source_paths or not target.path_prefixes:
                changed.append(TransferDimension.PATH)
                reasons.append(TransferReason.PATH_INCOMPATIBLE)
            elif any(not _path_is_within(path, target.path_prefixes) for path in source_paths):
                changed.append(TransferDimension.PATH)
                reasons.append(TransferReason.PATH_INCOMPATIBLE)
            if not _subset(_procedure_effects(procedure), family.boundary.permitted_effect_classes):
                changed.append(TransferDimension.EFFECT)
                reasons.append(TransferReason.EFFECT_INCOMPATIBLE)
                violations.append(BoundaryViolationClass.EFFECT.value)
            if not _subset(
                _procedure_operation_contracts(procedure), family.required_operation_contracts
            ):
                changed.append(TransferDimension.OPERATION)
                reasons.append(TransferReason.OPERATION_INCOMPATIBLE)
        if certificate is not None:
            if not isinstance(certificate, ProcedureCertificate):
                raise TransferDeclarationError("certificate must be ProcedureCertificate")
            if not _family_matches_certificate(family, certificate):
                changed.append(TransferDimension.FAMILY)
                reasons.append(TransferReason.FAMILY_INCOMPATIBLE)
                violations.append(BoundaryViolationClass.MEMBERSHIP_CONTRADICTION.value)
            if target.repository_family and target.repository_family not in {
                *certificate.repository_families,
                *family.boundary.permitted_repositories,
            }:
                changed.append(TransferDimension.FAMILY)
                reasons.append(TransferReason.FAMILY_INCOMPATIBLE)
                violations.append(BoundaryViolationClass.REPOSITORY.value)

        ordered_changed = tuple(dict.fromkeys(changed))
        ordered_reasons = tuple(dict.fromkeys(reasons))
        ordered_violations = tuple(dict.fromkeys(violations))
        admitted = not ordered_changed
        reason = TransferReason.COMPATIBLE if admitted else ordered_reasons[0]
        return self._result(
            family,
            target,
            admitted=admitted,
            reason=reason,
            changed=ordered_changed,
            reasons=ordered_reasons,
            violations=ordered_violations,
            missing=(),
            emitted_at_ms=emitted_at_ms,
        )

    def _result(
        self,
        family: TaskFamily,
        target: TargetRepository,
        *,
        admitted: bool,
        reason: TransferReason,
        changed: tuple[TransferDimension, ...],
        reasons: tuple[TransferReason, ...],
        violations: tuple[str, ...],
        missing: tuple[str, ...],
        emitted_at_ms: int,
    ) -> GeneralizationBoundaryEvaluation:
        state = ArtifactState.CANDIDATE if admitted else ArtifactState.REJECTED
        artifact = GeneralizationBoundaryArtifact(
            bindings=family.bindings,
            state=state,
            subject_cid=family.content_id,
            reference_cids=(target.repository_id, target.tree_id),
            labels=(
                "generalization-boundary",
                "transfer",
                reason.value,
            ),
            facts={
                "admitted": admitted,
                "reason_code": reason.value,
                "changed_dimensions": tuple(item.value for item in changed),
                "reason_codes": tuple(item.value for item in reasons),
                "violation_classes": violations,
                "missing_dimensions": missing,
                "target_repository_id": target.repository_id,
                "evaluator_revision": self.revision,
                "can_authorize": False,
                "can_promote": False,
            },
            created_at_ms=emitted_at_ms,
        )
        return GeneralizationBoundaryEvaluation(
            admitted=admitted,
            reason_code=reason,
            changed_dimensions=changed,
            reason_codes=reasons,
            violation_classes=violations,
            missing_dimensions=missing,
            artifact=artifact,
        )


def _check_operation(
    procedure: ProcedureSpec, certificate: ProcedureCertificate, target: TargetRepository
) -> TransferReason | None:
    operations = _procedure_operations(procedure)
    contracts = _procedure_operation_contracts(procedure)
    if not operations or not contracts:
        return TransferReason.INCOMPLETE_ASSUMPTIONS
    if not target.permitted_operations or not _subset(operations, target.permitted_operations):
        return TransferReason.OPERATION_INCOMPATIBLE
    if not _subset(contracts, target.permitted_operation_contracts):
        return TransferReason.OPERATION_INCOMPATIBLE
    if (
        target.operation_catalog_revision
        and target.operation_catalog_revision != certificate.operation_catalog_revision
    ):
        return TransferReason.OPERATION_INCOMPATIBLE
    if procedure.authority.allowed_operations and not _subset(
        operations, procedure.authority.allowed_operations
    ):
        return TransferReason.OPERATION_INCOMPATIBLE
    return None


def _check_effect(
    procedure: ProcedureSpec, certificate: ProcedureCertificate, target: TargetRepository
) -> TransferReason | None:
    effects = _procedure_effects(procedure)
    if not effects:
        return TransferReason.INCOMPLETE_ASSUMPTIONS
    if not target.permitted_effect_classes or not _subset(
        effects, target.permitted_effect_classes
    ):
        return TransferReason.EFFECT_INCOMPATIBLE
    if (
        target.effect_policy_revision
        and target.effect_policy_revision != certificate.effect_policy_revision
    ):
        return TransferReason.EFFECT_INCOMPATIBLE
    return None


def _check_authority(
    procedure: ProcedureSpec, certificate: ProcedureCertificate, target: TargetRepository
) -> TransferReason | None:
    if target.authority_policy_revision != certificate.authority_policy_revision:
        return TransferReason.AUTHORITY_INCOMPATIBLE
    if procedure.authority.authority_policy_revision != target.authority_policy_revision:
        return TransferReason.AUTHORITY_INCOMPATIBLE
    requirements = _procedure_authorities(procedure)
    if requirements and not _subset(requirements, target.authority_requirement_ids):
        return TransferReason.AUTHORITY_INCOMPATIBLE
    capabilities = procedure.authority.required_capability_ids
    if capabilities and not _subset(capabilities, target.required_capability_ids):
        return TransferReason.AUTHORITY_INCOMPATIBLE
    if _risk_rank(certificate.risk_ceiling) > _risk_rank(target.risk_ceiling):
        return TransferReason.RISK_CEILING
    if _risk_rank(procedure.authority.risk_ceiling) > _risk_rank(target.risk_ceiling):
        return TransferReason.RISK_CEILING
    return None


def _check_language(
    certificate: ProcedureCertificate, family: TaskFamily, target: TargetRepository
) -> TransferReason | None:
    source = certificate.supported_language_classes
    if not source:
        return TransferReason.INCOMPLETE_ASSUMPTIONS
    if not target.language_classes or not _subset(source, target.language_classes):
        return TransferReason.LANGUAGE_INCOMPATIBLE
    if not _subset(target.language_classes, family.boundary.permitted_languages):
        return TransferReason.LANGUAGE_INCOMPATIBLE
    return None


def _check_framework(
    certificate: ProcedureCertificate, family: TaskFamily, target: TargetRepository
) -> TransferReason | None:
    source = certificate.supported_framework_classes
    if not source:
        return TransferReason.INCOMPLETE_ASSUMPTIONS
    if not target.framework_classes or not _subset(source, target.framework_classes):
        return TransferReason.FRAMEWORK_INCOMPATIBLE
    if not _subset(target.framework_classes, family.boundary.permitted_frameworks):
        return TransferReason.FRAMEWORK_INCOMPATIBLE
    return None


def _check_validation(
    procedure: ProcedureSpec, family: TaskFamily, target: TargetRepository
) -> TransferReason | None:
    required = _procedure_validation_contracts(procedure)
    if not required:
        return TransferReason.INCOMPLETE_ASSUMPTIONS
    if not target.validation_contracts or not _subset(required, target.validation_contracts):
        return TransferReason.VALIDATION_INCOMPATIBLE
    if family.validation_structure and not _subset(
        family.validation_structure, target.validation_structure
    ):
        return TransferReason.VALIDATION_INCOMPATIBLE
    return None


def _check_family(
    family: TaskFamily, certificate: ProcedureCertificate, target: TargetRepository
) -> TransferReason | None:
    missing = _incomplete_family_dimensions(family)
    if missing:
        return TransferReason.INCOMPLETE_BOUNDARY
    if not _family_matches_certificate(family, certificate):
        return TransferReason.FAMILY_INCOMPATIBLE
    if target.repository_id not in family.boundary.permitted_repositories:
        return TransferReason.FAMILY_INCOMPATIBLE
    if (
        target.repository_family
        and target.repository_family not in certificate.repository_families
        and target.repository_family not in family.boundary.permitted_repositories
    ):
        return TransferReason.FAMILY_INCOMPATIBLE
    return None


def _check_path(procedure: ProcedureSpec, target: TargetRepository) -> TransferReason | None:
    source_paths = _source_path_assumptions(procedure)
    if not source_paths:
        return TransferReason.INCOMPLETE_ASSUMPTIONS
    if not target.path_prefixes:
        return TransferReason.PATH_INCOMPATIBLE
    if any(not _path_is_within(path, target.path_prefixes) for path in source_paths):
        return TransferReason.PATH_INCOMPATIBLE
    if procedure.scope_paths and any(
        not _path_is_within(path, target.path_prefixes) for path in procedure.scope_paths
    ):
        return TransferReason.PATH_INCOMPATIBLE
    return None


def _check_held_out(
    request: TransferRequest,
) -> TransferReason | None:
    held_out = request.held_out
    if held_out is None:
        return TransferReason.HELD_OUT_MISSING
    isolation = held_out.isolation_reason()
    if isolation is not None:
        return isolation
    if held_out.repository_id != request.target.repository_id:
        return TransferReason.HELD_OUT_REPOSITORY_MISMATCH
    if held_out.tree_id and held_out.tree_id != request.target.tree_id:
        return TransferReason.HELD_OUT_REPOSITORY_MISMATCH
    if not held_out.is_nonvacuous_pass:
        return TransferReason.HELD_OUT_FAILED
    return None


class ProcedureTransferGate:
    """Admit only explicitly compatible transfers as bounded candidate eligibility."""

    revision: ClassVar[str] = GATE_REVISION

    def __init__(self) -> None:
        self._boundary = GeneralizationBoundaryEvaluator()
        self._unsafe_transfer_count = UNSAFE_TRANSFER_COUNT

    @property
    def unsafe_transfer_count(self) -> int:
        return self._unsafe_transfer_count

    def evaluate(
        self,
        request: TransferRequest,
        *,
        emitted_at_ms: int = 0,
    ) -> TransferDecision:
        if not isinstance(request, TransferRequest):
            raise TransferDeclarationError("request must be TransferRequest")
        procedure = request.procedure
        certificate = request.certificate
        family = request.family
        target = request.target

        safety_reasons: list[TransferReason] = []
        if request.mutate_target:
            safety_reasons.append(TransferReason.CROSS_REPOSITORY_MUTATION)
        if request.experiment_authorizes:
            safety_reasons.append(TransferReason.EXPERIMENT_CANNOT_AUTHORIZE)
        if target.production:
            safety_reasons.append(TransferReason.PRODUCTION_MUTATION)
        if target.policy_mutable:
            safety_reasons.append(TransferReason.POLICY_MUTATION)
        if not target.authorized:
            safety_reasons.append(TransferReason.TARGET_NOT_AUTHORIZED)

        if procedure.bindings != request.bindings:
            safety_reasons.append(TransferReason.BINDING_MISMATCH)
        if certificate.bindings.policy_revision != request.bindings.policy_revision:
            safety_reasons.append(TransferReason.BINDING_MISMATCH)
        if certificate.procedure_cid != procedure.content_id:
            safety_reasons.append(TransferReason.BINDING_MISMATCH)
        if family.bindings.policy_revision != request.bindings.policy_revision:
            safety_reasons.append(TransferReason.BINDING_MISMATCH)

        boundary = self._boundary.evaluate(
            family,
            target,
            procedure=procedure,
            certificate=certificate,
            emitted_at_ms=emitted_at_ms,
        )

        changed: list[str] = []
        reasons: list[TransferReason] = []
        compatible: list[str] = []
        checkers = {
            TransferDimension.OPERATION: lambda: _check_operation(
                procedure, certificate, target
            ),
            TransferDimension.EFFECT: lambda: _check_effect(procedure, certificate, target),
            TransferDimension.AUTHORITY: lambda: _check_authority(
                procedure, certificate, target
            ),
            TransferDimension.LANGUAGE: lambda: _check_language(certificate, family, target),
            TransferDimension.FRAMEWORK: lambda: _check_framework(
                certificate, family, target
            ),
            TransferDimension.VALIDATION: lambda: _check_validation(
                procedure, family, target
            ),
            TransferDimension.FAMILY: lambda: _check_family(family, certificate, target),
            TransferDimension.PATH: lambda: _check_path(procedure, target),
            TransferDimension.HELD_OUT: lambda: _check_held_out(request),
        }
        for dimension in TransferDimension:
            reason = checkers[dimension]()
            if reason is None:
                compatible.append(dimension.value)
            else:
                changed.append(dimension.value)
                reasons.append(reason)

        for dimension in boundary.changed_dimensions:
            if dimension.value not in changed:
                changed.append(dimension.value)
                reasons.append(_DIMENSION_REASON[dimension])

        held_out_cid = request.held_out.evaluation_cid if request.held_out is not None else ""
        held_out_passed = bool(
            request.held_out is not None and request.held_out.is_nonvacuous_pass
        )

        primary: TransferReason
        action = TransferAction.REFUSE
        eligible = False
        state = ArtifactState.REJECTED
        if safety_reasons:
            primary = safety_reasons[0]
            reasons = list(dict.fromkeys([*safety_reasons, *reasons]))
        elif reasons:
            primary = reasons[0]
            reasons = list(dict.fromkeys(reasons))
            if (
                request.similarity.any_similar
                and primary
                not in {
                    TransferReason.HELD_OUT_MISSING,
                    TransferReason.HELD_OUT_FAILED,
                    TransferReason.HELD_OUT_REPOSITORY_MISMATCH,
                    TransferReason.HELD_OUT_UNSAFE,
                }
            ):
                if TransferReason.SIMILARITY_INSUFFICIENT not in reasons:
                    reasons.append(TransferReason.SIMILARITY_INSUFFICIENT)
        else:
            primary = TransferReason.COMPATIBLE
            action = TransferAction.ELIGIBLE
            eligible = True
            state = ArtifactState.CANDIDATE
            reasons = (TransferReason.COMPATIBLE,)

        # Similarity may be recorded but is never evidence, including on
        # the compatible path.
        similarity_signals = request.similarity.asserted_signals
        decision = TransferDecision(
            bindings=request.bindings,
            source_procedure_cid=procedure.content_id,
            source_certificate_cid=certificate.content_id,
            source_family_cid=family.content_id,
            target_repository_id=target.repository_id,
            target_tree_id=target.tree_id,
            action=action,
            reason_code=primary,
            reason_codes=tuple(reasons) if action is TransferAction.REFUSE else (TransferReason.COMPATIBLE,),
            changed_assumptions=tuple(changed) if action is TransferAction.REFUSE else (),
            compatible_dimensions=tuple(compatible)
            if action is TransferAction.REFUSE
            else REQUIRED_COMPATIBILITY_DIMENSIONS,
            held_out_evaluation_cid=held_out_cid,
            held_out_passed=held_out_passed,
            eligibility_state=state,
            eligible=eligible,
            similarity_signals=similarity_signals,
            boundary_reason_code=boundary.reason_code.value,
        )
        return decision

    def require(
        self,
        request: TransferRequest,
        *,
        emitted_at_ms: int = 0,
    ) -> TransferDecision:
        decision = self.evaluate(request, emitted_at_ms=emitted_at_ms)
        if decision.action is not TransferAction.ELIGIBLE:
            raise TransferRefusalError(
                f"transfer refused: {decision.reason_code.value}",
                decision=decision,
            )
        return decision

    def transfer(
        self,
        request: TransferRequest,
        *,
        emitted_at_ms: int = 0,
    ) -> TransferDecision:
        return self.evaluate(request, emitted_at_ms=emitted_at_ms)


def evaluate_transfer(
    request: TransferRequest,
    *,
    emitted_at_ms: int = 0,
) -> TransferDecision:
    """Module-level transfer check.  Unsafe transfers are never admitted."""

    return ProcedureTransferGate().evaluate(request, emitted_at_ms=emitted_at_ms)
