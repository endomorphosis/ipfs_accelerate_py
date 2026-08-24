"""Structural anti-unification of admitted procedure traces.

``ProcedureAntiUnifier`` infers a candidate pattern from two or more positive
traces in one task family.  It is a structural generalizer only: it does not
mine specifications, certify procedures, or promote anything.

Fail-closed floors:

- semantic order of shared operations is preserved
- every validation and postcondition is retained (union; never dropped)
- paths and credentials are never turned into parameters
- omitted tests, missing postconditions, effect splits, and uncertain
  authority produce immutable counterexamples
- every value that is not kept as a concrete constant is recorded as a lost
  detail
- remaining uncertainty becomes an allowed typed hole, never a forbidden one
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .contracts import (
    FORBIDDEN_HOLE_TYPES,
    FORBIDDEN_STEP_OPERATIONS,
    MAX_BRANCHES,
    MAX_HOLES,
    MAX_ITEMS,
    MAX_STEPS,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ExecutionTrajectory,
    FailureTransition,
    FamilyMembershipClass,
    HoleType,
    ProcedureContractError,
    ProcedureSpec,
    ProviderClass,
    StepOperation,
    TaskFamily,
    TaskFamilyMembership,
    TrajectoryTerminalStatus,
    ValueType,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _relative_path,
    _strings,
    _unsafe_key,
    canonical_json_bytes,
)
from .contracts import (
    AntiUnificationPattern as AntiUnificationPatternArtifact,
)
from .contracts import (
    GeneralizationBoundary as GeneralizationBoundaryArtifact,
)
from .contracts import (
    GeneralizationCounterexample as GeneralizationCounterexampleArtifact,
)


UNIFIER_REVISION: Final[str] = "ProcedureAntiUnifier@1"
MAX_OPTIONAL_BRANCHES: Final[int] = MAX_BRANCHES
MAX_PATTERN_HOLES: Final[int] = MAX_HOLES
MAX_LOST_DETAILS: Final[int] = MAX_ITEMS
MAX_COUNTEREXAMPLES: Final[int] = MAX_ITEMS

POSITIVE_TERMINAL_STATUSES: Final[frozenset[TrajectoryTerminalStatus]] = frozenset(
    {
        TrajectoryTerminalStatus.ACCEPTED,
        TrajectoryTerminalStatus.FAILED_RECOVERED,
        TrajectoryTerminalStatus.ROLLED_BACK,
    }
)

VALIDATION_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {
        StepOperation.RUN_STATIC_ANALYSIS,
        StepOperation.RUN_TYPE_CHECK,
        StepOperation.RUN_SELECTED_TESTS,
        StepOperation.RUN_FULL_TEST_FALLBACK,
        StepOperation.RUN_PROOF,
        StepOperation.RUN_ADVERSARIAL_ASSURANCE,
        StepOperation.CHECK_DIFF,
        StepOperation.CHECK_SCOPE,
        StepOperation.CHECK_POSTCONDITION,
        StepOperation.VERIFY_MERGED_TREE,
    }
)

TEST_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {
        StepOperation.RUN_SELECTED_TESTS,
        StepOperation.RUN_FULL_TEST_FALLBACK,
        StepOperation.RUN_ADVERSARIAL_ASSURANCE,
    }
)

POSTCONDITION_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {StepOperation.CHECK_POSTCONDITION}
)

AUTHORITY_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {
        StepOperation.CHECK_AUTHORITY,
        StepOperation.CHECK_CAPABILITY,
        StepOperation.CHECK_POLICY,
    }
)

OPTIONAL_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {
        StepOperation.QUERY_AST_INDEX,
        StepOperation.QUERY_DEPENDENCY_GRAPH,
        StepOperation.QUERY_SEMANTIC_INDEX,
        StepOperation.QUERY_RECEIPT_CACHE,
        StepOperation.SELECT_EVIDENCE,
        StepOperation.EXPAND_CONTEXT_REFERENCE,
        StepOperation.READ_STATE,
    }
)

MATERIAL_EFFECT_CLASSES: Final[frozenset[EffectClass]] = frozenset(
    {
        EffectClass.WORKTREE_CREATE,
        EffectClass.REPOSITORY_WRITE,
        EffectClass.MERGE_PREPARE,
        EffectClass.MERGE,
        EffectClass.ARTIFACT_PERSIST,
        EffectClass.ROLLBACK,
        EffectClass.ESCALATION,
        EffectClass.PROOF,
        EffectClass.VALIDATION,
    }
)

FORBIDDEN_GENERALIZATIONS: Final[tuple[str, ...]] = (
    "path",
    "credential",
    "omitted-test",
    "missing-postcondition",
    "uncertain-authority",
)

_OPERATION_EFFECT: Final[Mapping[StepOperation, EffectClass]] = MappingProxyType(
    {
        StepOperation.READ_STATE: EffectClass.OBSERVE,
        StepOperation.QUERY_AST_INDEX: EffectClass.OBSERVE,
        StepOperation.QUERY_DEPENDENCY_GRAPH: EffectClass.OBSERVE,
        StepOperation.QUERY_SEMANTIC_INDEX: EffectClass.OBSERVE,
        StepOperation.QUERY_RECEIPT_CACHE: EffectClass.OBSERVE,
        StepOperation.SELECT_EVIDENCE: EffectClass.OBSERVE,
        StepOperation.EXPAND_CONTEXT_REFERENCE: EffectClass.OBSERVE,
        StepOperation.CHECK_CAPABILITY: EffectClass.OBSERVE,
        StepOperation.CHECK_POLICY: EffectClass.OBSERVE,
        StepOperation.CHECK_AUTHORITY: EffectClass.OBSERVE,
        StepOperation.CREATE_ISOLATED_WORKTREE: EffectClass.WORKTREE_CREATE,
        StepOperation.APPLY_APPROVED_PATCH_TEMPLATE: EffectClass.REPOSITORY_WRITE,
        StepOperation.REQUEST_TYPED_MODEL_HOLE: EffectClass.MODEL_REQUEST,
        StepOperation.RUN_STATIC_ANALYSIS: EffectClass.VALIDATION,
        StepOperation.RUN_TYPE_CHECK: EffectClass.VALIDATION,
        StepOperation.RUN_SELECTED_TESTS: EffectClass.VALIDATION,
        StepOperation.RUN_FULL_TEST_FALLBACK: EffectClass.VALIDATION,
        StepOperation.RUN_PROOF: EffectClass.PROOF,
        StepOperation.RUN_ADVERSARIAL_ASSURANCE: EffectClass.PROOF,
        StepOperation.CHECK_DIFF: EffectClass.VALIDATION,
        StepOperation.CHECK_SCOPE: EffectClass.VALIDATION,
        StepOperation.CHECK_POSTCONDITION: EffectClass.VALIDATION,
        StepOperation.PREPARE_MERGE: EffectClass.MERGE_PREPARE,
        StepOperation.MERGE_IN_ISOLATED_TRAIN: EffectClass.MERGE,
        StepOperation.VERIFY_MERGED_TREE: EffectClass.MERGE,
        StepOperation.PERSIST_ARTIFACT: EffectClass.ARTIFACT_PERSIST,
        StepOperation.EMIT_RECEIPT: EffectClass.RECEIPT_EMIT,
        StepOperation.ROLLBACK: EffectClass.ROLLBACK,
        StepOperation.ESCALATE: EffectClass.ESCALATION,
    }
)

_EFFECT_ALIASES: Final[Mapping[str, EffectClass]] = MappingProxyType(
    {
        **{item.value: item for item in EffectClass},
        **{item.value.replace("_", "-"): item for item in EffectClass},
        "model-request": EffectClass.MODEL_REQUEST,
        "repository-write": EffectClass.REPOSITORY_WRITE,
        "worktree-create": EffectClass.WORKTREE_CREATE,
        "merge-prepare": EffectClass.MERGE_PREPARE,
        "receipt-emit": EffectClass.RECEIPT_EMIT,
        "artifact-persist": EffectClass.ARTIFACT_PERSIST,
    }
)

TraceSource = ExecutionTrajectory | ProcedureSpec


class AntiUnificationError(ProcedureContractError):
    """Traces could not be admitted for structural anti-unification."""


class UnsafeMergeClass(str, Enum):
    """Closed reasons an anti-unification cannot remain a candidate."""

    AUTHORITY_SPLIT = "authority-split"
    EFFECT_SPLIT = "effect-split"
    VALIDATION_SPLIT = "validation-split"
    OMITTED_TEST = "omitted-test"
    MISSING_POSTCONDITION = "missing-postcondition"
    PATH_GENERALIZATION = "path-generalization"
    CREDENTIAL_GENERALIZATION = "credential-generalization"
    UNCERTAIN_AUTHORITY = "uncertain-authority"
    FORBIDDEN_HOLE = "forbidden-hole"
    FAMILY_MISMATCH = "family-mismatch"


class LostDetailKind(str, Enum):
    """Closed classes of detail that left the concrete constant set."""

    INSTANCE_STATE = "instance-state"
    PARAMETER = "parameter"
    OPTIONAL_BRANCH = "optional-branch"
    TYPED_HOLE = "typed-hole"
    PATH = "path"
    CREDENTIAL = "credential"
    AUTHORITY = "authority"
    EFFECT = "effect"
    VALIDATION = "validation"
    POSTCONDITION = "postcondition"
    PRECONDITION = "precondition"
    ORDER = "order"
    FAILURE = "failure"
    HOLE = "hole"
    BINDING = "binding"


class LostDetailDisposition(str, Enum):
    """How a lost concrete value was treated in the pattern."""

    PARAMETER = "parameter"
    OPTIONAL_BRANCH = "optional-branch"
    TYPED_HOLE = "typed-hole"
    INSTANCE_STATE = "instance-state"
    RETAINED_UNION = "retained-union"
    COUNTEREXAMPLE = "counterexample"
    STRIPPED = "stripped"


class StepPresence(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    HOLE = "hole"


class PatternStatus(str, Enum):
    CANDIDATE = "candidate"
    REJECTED = "rejected"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise AntiUnificationError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _family(value: Any) -> TaskFamily:
    return _nested(value, TaskFamily, "family")


def _canonical(value: Any, field_name: str) -> Any:
    try:
        return _freeze(value, field_name)
    except ProcedureContractError as exc:
        raise AntiUnificationError(str(exc)) from exc


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for item in values:
        if item and item not in result:
            result.append(item)
    return tuple(result)


def _effect_class_for_id(effect_id: str) -> EffectClass | None:
    key = effect_id.replace("-", "_")
    return _EFFECT_ALIASES.get(effect_id) or _EFFECT_ALIASES.get(key)


def _effect_classes_for_operation(operation: StepOperation) -> tuple[EffectClass, ...]:
    mapped = _OPERATION_EFFECT.get(operation)
    if mapped is None:
        return ()
    return (mapped,)


def _looks_like_path(value: Any) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        normalized = _relative_path(value, "path")
    except ProcedureContractError:
        return False
    return "/" in normalized or normalized.endswith((".py", ".md", ".json", ".toml"))


def _lcs_pairs(left: Sequence[Any], right: Sequence[Any]) -> tuple[tuple[int, int], ...]:
    n = len(left)
    m = len(right)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if left[i] == right[j]:
                table[i][j] = table[i + 1][j + 1] + 1
            elif table[i + 1][j] >= table[i][j + 1]:
                table[i][j] = table[i + 1][j]
            else:
                table[i][j] = table[i][j + 1]
    pairs: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < n and j < m:
        if left[i] == right[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return tuple(pairs)


def _value_type_for(values: Sequence[Any]) -> ValueType | None:
    if not values:
        return None
    python_types = {type(item) for item in values}
    if python_types == {int}:
        return ValueType.INTEGER
    if python_types == {bool}:
        return ValueType.BOOLEAN
    if python_types != {str}:
        return None
    texts = tuple(str(item) for item in values)
    if all(_looks_like_path(item) for item in texts):
        return ValueType.RELATIVE_PATH
    try:
        for item in texts:
            _identifier(item, "parameter_value")
    except ProcedureContractError:
        return ValueType.STRING
    return ValueType.IDENTIFIER


def _hole_type(value: Any, *, required: bool = False) -> HoleType | None:
    if value in (None, ""):
        if required:
            raise AntiUnificationError("hole_type is required")
        return None
    if isinstance(value, HoleType):
        if value.value in FORBIDDEN_HOLE_TYPES:
            raise AntiUnificationError("forbidden hole types cannot enter anti-unification")
        return value
    if type(value) is not str:
        raise AntiUnificationError("hole_type must be a string")
    if value in FORBIDDEN_HOLE_TYPES:
        raise AntiUnificationError("forbidden hole types cannot enter anti-unification")
    try:
        return HoleType(value)
    except ValueError as exc:
        raise AntiUnificationError("hole_type must be an allowed typed hole") from exc


@dataclass(frozen=True)
class TraceStepView:
    """Normalized step extracted from a trajectory or procedure spec."""

    index: int
    operation: StepOperation
    operation_contract: str
    effect_ids: tuple[str, ...] = ()
    effect_classes: tuple[EffectClass, ...] = ()
    authority_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()
    hole_type: str = ""
    hole_id: str = ""
    failure: str = ""
    paths: tuple[str, ...] = ()
    credential_fields: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    input_bindings: tuple[tuple[str, str], ...] = ()
    output_bindings: tuple[tuple[str, str], ...] = ()
    instance_state: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TraceView:
    """Closed structural projection of one admitted trace."""

    cid: str
    bindings: ArtifactBindings
    kind: str
    family_hint: str
    steps: tuple[TraceStepView, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    validation_contracts: tuple[str, ...]
    validation_operations: tuple[str, ...]
    failure_transitions: tuple[str, ...]
    authority_ids: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...]
    paths: tuple[str, ...]
    credential_fields: tuple[str, ...]
    holes: tuple[tuple[str, str], ...]
    policy_revision: str
    terminal_status: str = ""


@dataclass(frozen=True)
class AlignedSlot:
    """One aligned operation across traces; missing members are absences."""

    operation: StepOperation
    members: Mapping[int, TraceStepView]


@dataclass(frozen=True)
class PatternParameter:
    """Closed parameter inferred from safe, non-path differences."""

    name: str
    value_type: ValueType
    source_field: str
    allowed_values: tuple[Any, ...]
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "name"))
        object.__setattr__(self, "value_type", _enum(self.value_type, ValueType, "value_type"))
        object.__setattr__(self, "source_field", _identifier(self.source_field, "source_field"))
        object.__setattr__(self, "allowed_values", _canonical(self.allowed_values, "allowed_values"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        if self.value_type is ValueType.RELATIVE_PATH:
            raise AntiUnificationError("paths are never generalized into parameters")

    def to_facts(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type.value,
            "source_field": self.source_field,
            "allowed_values": self.allowed_values,
            "required": self.required,
        }


@dataclass(frozen=True)
class PatternStep:
    """One required or optional step in the anti-unified skeleton."""

    step_index: int
    operation: StepOperation
    operation_contract: str
    presence: StepPresence
    effect_ids: tuple[str, ...] = ()
    effect_classes: tuple[str, ...] = ()
    authority_ids: tuple[str, ...] = ()
    parameter_names: tuple[str, ...] = ()
    hole_id: str = ""
    failure_transition: str = ""
    source_trace_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_index", _nonnegative_int(self.step_index, "step_index")
        )
        object.__setattr__(self, "operation", _enum(self.operation, StepOperation, "operation"))
        object.__setattr__(
            self,
            "operation_contract",
            _identifier(self.operation_contract, "operation_contract", required=False),
        )
        object.__setattr__(self, "presence", _enum(self.presence, StepPresence, "presence"))
        object.__setattr__(
            self, "effect_ids", _strings(self.effect_ids, "effect_ids", identifiers=True)
        )
        object.__setattr__(
            self,
            "effect_classes",
            _strings(self.effect_classes, "effect_classes", identifiers=True),
        )
        object.__setattr__(
            self,
            "authority_ids",
            _strings(self.authority_ids, "authority_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "parameter_names",
            _strings(self.parameter_names, "parameter_names", identifiers=True),
        )
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id", required=False))
        object.__setattr__(
            self,
            "failure_transition",
            _identifier(self.failure_transition, "failure_transition", required=False),
        )
        object.__setattr__(
            self,
            "source_trace_cids",
            _strings(self.source_trace_cids, "source_trace_cids", identifiers=True),
        )

    def to_facts(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "operation": self.operation.value,
            "operation_contract": self.operation_contract,
            "presence": self.presence.value,
            "effect_ids": self.effect_ids,
            "effect_classes": self.effect_classes,
            "authority_ids": self.authority_ids,
            "parameter_names": self.parameter_names,
            "hole_id": self.hole_id,
            "failure_transition": self.failure_transition,
            "source_trace_cids": self.source_trace_cids,
        }


@dataclass(frozen=True)
class OptionalBranch:
    """Bounded optional insertion inferred from a proper subset of traces."""

    branch_id: str
    operation: StepOperation
    operation_contract: str
    predecessor_operation: str
    successor_operation: str
    source_trace_cids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _identifier(self.branch_id, "branch_id"))
        object.__setattr__(self, "operation", _enum(self.operation, StepOperation, "operation"))
        object.__setattr__(
            self,
            "operation_contract",
            _identifier(self.operation_contract, "operation_contract", required=False),
        )
        object.__setattr__(
            self,
            "predecessor_operation",
            _identifier(self.predecessor_operation, "predecessor_operation", required=False),
        )
        object.__setattr__(
            self,
            "successor_operation",
            _identifier(self.successor_operation, "successor_operation", required=False),
        )
        object.__setattr__(
            self,
            "source_trace_cids",
            _strings(
                self.source_trace_cids,
                "source_trace_cids",
                identifiers=True,
                required=True,
            ),
        )

    def to_facts(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "operation": self.operation.value,
            "operation_contract": self.operation_contract,
            "predecessor_operation": self.predecessor_operation,
            "successor_operation": self.successor_operation,
            "source_trace_cids": self.source_trace_cids,
        }


@dataclass(frozen=True)
class PatternHole:
    """Allowed typed hole for residual uncertainty."""

    hole_id: str
    hole_type: HoleType
    reason: str
    source_field: str
    source_values: tuple[Any, ...]
    validation_observation_ids: tuple[str, ...]
    allowed_provider_classes: tuple[str, ...] = (ProviderClass.DECLARATIVE_RULE.value,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hole_id", _identifier(self.hole_id, "hole_id"))
        object.__setattr__(self, "hole_type", _enum(self.hole_type, HoleType, "hole_type"))
        if self.hole_type.value in FORBIDDEN_HOLE_TYPES:
            raise AntiUnificationError("forbidden hole types cannot be inferred")
        object.__setattr__(self, "reason", _identifier(self.reason, "reason"))
        object.__setattr__(
            self, "source_field", _identifier(self.source_field, "source_field")
        )
        object.__setattr__(self, "source_values", _canonical(self.source_values, "source_values"))
        object.__setattr__(
            self,
            "validation_observation_ids",
            _strings(
                self.validation_observation_ids,
                "validation_observation_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "allowed_provider_classes",
            _strings(
                self.allowed_provider_classes,
                "allowed_provider_classes",
                identifiers=True,
                required=True,
            ),
        )

    def to_facts(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "hole_type": self.hole_type.value,
            "reason": self.reason,
            "source_field": self.source_field,
            "source_values": self.source_values,
            "validation_observation_ids": self.validation_observation_ids,
            "allowed_provider_classes": self.allowed_provider_classes,
        }


@dataclass(frozen=True)
class LostDetail:
    """One concrete value that was not kept as a shared constant."""

    detail_id: str
    kind: LostDetailKind
    location: str
    disposition: LostDetailDisposition
    source_values: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail_id", _identifier(self.detail_id, "detail_id"))
        object.__setattr__(self, "kind", _enum(self.kind, LostDetailKind, "kind"))
        object.__setattr__(self, "location", _identifier(self.location, "location"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, LostDetailDisposition, "disposition")
        )
        pairs: list[tuple[str, Any]] = []
        if not isinstance(self.source_values, Sequence) or isinstance(
            self.source_values, (str, bytes, bytearray, memoryview)
        ):
            raise AntiUnificationError("source_values must be a sequence of pairs")
        for item in self.source_values:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)) or len(item) != 2:
                raise AntiUnificationError("source_values entries must be (trace_cid, value)")
            pairs.append(
                (
                    _identifier(item[0], "source_values.trace_cid"),
                    _canonical(item[1], "source_values.value"),
                )
            )
        object.__setattr__(self, "source_values", tuple(pairs))

    def to_facts(self) -> dict[str, Any]:
        return {
            "detail_id": self.detail_id,
            "kind": self.kind.value,
            "location": self.location,
            "disposition": self.disposition.value,
            "source_values": tuple(
                {"trace_cid": cid, "value": value} for cid, value in self.source_values
            ),
        }


@dataclass(frozen=True)
class GeneralizationCounterexample:
    """Immutable record of an unsafe merge attempt."""

    violation_class: UnsafeMergeClass
    left_trace_cid: str
    right_trace_cid: str
    location: str
    left_value: Any
    right_value: Any
    evidence_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "violation_class",
            _enum(self.violation_class, UnsafeMergeClass, "violation_class"),
        )
        object.__setattr__(
            self, "left_trace_cid", _identifier(self.left_trace_cid, "left_trace_cid")
        )
        object.__setattr__(
            self, "right_trace_cid", _identifier(self.right_trace_cid, "right_trace_cid")
        )
        object.__setattr__(self, "location", _identifier(self.location, "location"))
        object.__setattr__(self, "left_value", _canonical(self.left_value, "left_value"))
        object.__setattr__(self, "right_value", _canonical(self.right_value, "right_value"))
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )

    def to_facts(self) -> dict[str, Any]:
        return {
            "violation_class": self.violation_class.value,
            "left_trace_cid": self.left_trace_cid,
            "right_trace_cid": self.right_trace_cid,
            "location": self.location,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "evidence_cids": self.evidence_cids,
        }

    def to_artifact(
        self,
        bindings: ArtifactBindings,
        *,
        family_cid: str,
        emitted_at_ms: int = 0,
    ) -> GeneralizationCounterexampleArtifact:
        references = _unique(
            (self.left_trace_cid, self.right_trace_cid, family_cid, *self.evidence_cids)
        )
        return GeneralizationCounterexampleArtifact(
            bindings=bindings,
            state=ArtifactState.REJECTED,
            subject_cid=family_cid,
            reference_cids=references,
            labels=(self.violation_class.value,),
            facts=self.to_facts(),
            created_at_ms=emitted_at_ms,
        )


@dataclass(frozen=True)
class AntiUnificationPattern:
    """Candidate or rejected structural generalization of positive traces.

    The pattern records constants, parameters, optional branches, typed holes,
    retained validations/postconditions/failures, and every lost detail.  It
    never asserts verification or promotion.
    """

    bindings: ArtifactBindings
    task_family_id: str
    task_family_cid: str
    source_trace_cids: tuple[str, ...]
    constants: Mapping[str, Any]
    parameters: tuple[PatternParameter, ...]
    steps: tuple[PatternStep, ...]
    optional_branches: tuple[OptionalBranch, ...]
    holes: tuple[PatternHole, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    validation_contracts: tuple[str, ...]
    validation_operations: tuple[str, ...]
    failure_transitions: tuple[str, ...]
    lost_details: tuple[LostDetail, ...]
    status: PatternStatus
    unifier_revision: str = UNIFIER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self, "task_family_id", _identifier(self.task_family_id, "task_family_id")
        )
        object.__setattr__(
            self, "task_family_cid", _identifier(self.task_family_cid, "task_family_cid")
        )
        object.__setattr__(
            self,
            "source_trace_cids",
            _strings(
                self.source_trace_cids,
                "source_trace_cids",
                identifiers=True,
                required=True,
            ),
        )
        constants = _canonical(self.constants, "constants")
        if not isinstance(constants, Mapping):
            raise AntiUnificationError("constants must be a mapping")
        object.__setattr__(self, "constants", constants)
        if not isinstance(self.parameters, Sequence) or isinstance(
            self.parameters, (str, bytes, bytearray, memoryview)
        ):
            raise AntiUnificationError("parameters must be a sequence")
        if len(self.parameters) > MAX_ITEMS:
            raise AntiUnificationError("parameters exceeds its item bound")
        object.__setattr__(
            self,
            "parameters",
            tuple(
                item if isinstance(item, PatternParameter) else PatternParameter(**item)
                for item in self.parameters
            ),
        )
        for field_name, cls, limit in (
            ("steps", PatternStep, MAX_STEPS),
            ("optional_branches", OptionalBranch, MAX_OPTIONAL_BRANCHES),
            ("holes", PatternHole, MAX_PATTERN_HOLES),
            ("lost_details", LostDetail, MAX_LOST_DETAILS),
        ):
            raw = getattr(self, field_name)
            if not isinstance(raw, Sequence) or isinstance(
                raw, (str, bytes, bytearray, memoryview)
            ):
                raise AntiUnificationError(f"{field_name} must be a sequence")
            if len(raw) > limit:
                raise AntiUnificationError(f"{field_name} exceeds its item bound")
            object.__setattr__(
                self,
                field_name,
                tuple(item if isinstance(item, cls) else cls(**item) for item in raw),
            )
        for name in (
            "preconditions",
            "postconditions",
            "validation_contracts",
            "validation_operations",
            "failure_transitions",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        object.__setattr__(self, "status", _enum(self.status, PatternStatus, "status"))
        object.__setattr__(
            self,
            "unifier_revision",
            _identifier(self.unifier_revision, "unifier_revision"),
        )
        if any(item.value_type is ValueType.RELATIVE_PATH for item in self.parameters):
            raise AntiUnificationError("paths are never generalized into parameters")
        if any(_unsafe_key(item.name) or _unsafe_key(item.source_field) for item in self.parameters):
            raise AntiUnificationError("credentials are never generalized into parameters")

    @property
    def required_operations(self) -> tuple[str, ...]:
        return tuple(
            step.operation.value
            for step in self.steps
            if step.presence is StepPresence.REQUIRED
        )

    def to_facts(self) -> dict[str, Any]:
        return {
            "task_family_id": self.task_family_id,
            "task_family_cid": self.task_family_cid,
            "source_trace_cids": self.source_trace_cids,
            "constants": dict(self.constants),
            "parameters": tuple(item.to_facts() for item in self.parameters),
            "steps": tuple(item.to_facts() for item in self.steps),
            "optional_branches": tuple(item.to_facts() for item in self.optional_branches),
            "holes": tuple(item.to_facts() for item in self.holes),
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "validation_contracts": self.validation_contracts,
            "validation_operations": self.validation_operations,
            "failure_transitions": self.failure_transitions,
            "lost_details": tuple(item.to_facts() for item in self.lost_details),
            "pattern_status": self.status.value,
            "unifier_revision": self.unifier_revision,
            "required_operations": self.required_operations,
            "forbidden_generalizations": FORBIDDEN_GENERALIZATIONS,
        }

    def to_artifact(self, *, emitted_at_ms: int = 0) -> AntiUnificationPatternArtifact:
        state = (
            ArtifactState.CANDIDATE
            if self.status is PatternStatus.CANDIDATE
            else ArtifactState.REJECTED
        )
        return AntiUnificationPatternArtifact(
            bindings=self.bindings,
            state=state,
            subject_cid=self.task_family_cid,
            reference_cids=self.source_trace_cids,
            labels=(self.status.value, self.unifier_revision),
            facts=self.to_facts(),
            created_at_ms=emitted_at_ms,
        )

    @classmethod
    def from_artifact(cls, artifact: AntiUnificationPatternArtifact) -> AntiUnificationPattern:
        if not isinstance(artifact, AntiUnificationPatternArtifact):
            raise AntiUnificationError("artifact must be AntiUnificationPattern")
        facts = artifact.facts
        return cls(
            bindings=artifact.bindings,
            task_family_id=facts["task_family_id"],
            task_family_cid=facts["task_family_cid"],
            source_trace_cids=facts["source_trace_cids"],
            constants=facts["constants"],
            parameters=tuple(
                PatternParameter(**item) for item in facts.get("parameters", ())
            ),
            steps=tuple(PatternStep(**item) for item in facts.get("steps", ())),
            optional_branches=tuple(
                OptionalBranch(**item) for item in facts.get("optional_branches", ())
            ),
            holes=tuple(PatternHole(**item) for item in facts.get("holes", ())),
            preconditions=facts.get("preconditions", ()),
            postconditions=facts.get("postconditions", ()),
            validation_contracts=facts.get("validation_contracts", ()),
            validation_operations=facts.get("validation_operations", ()),
            failure_transitions=facts.get("failure_transitions", ()),
            lost_details=tuple(
                LostDetail(
                    detail_id=item["detail_id"],
                    kind=item["kind"],
                    location=item["location"],
                    disposition=item["disposition"],
                    source_values=tuple(
                        (entry["trace_cid"], entry["value"])
                        for entry in item.get("source_values", ())
                    ),
                )
                for item in facts.get("lost_details", ())
            ),
            status=facts["pattern_status"],
            unifier_revision=facts.get("unifier_revision", UNIFIER_REVISION),
        )


@dataclass(frozen=True)
class AntiUnificationResult:
    """Deterministic anti-unification output: pattern plus any counterexamples."""

    bindings: ArtifactBindings
    pattern: AntiUnificationPattern
    counterexamples: tuple[GeneralizationCounterexample, ...]
    pattern_artifact: AntiUnificationPatternArtifact
    counterexample_artifacts: tuple[GeneralizationCounterexampleArtifact, ...]
    boundary_artifact: GeneralizationBoundaryArtifact
    lost_details: tuple[LostDetail, ...]
    retained_validations: tuple[str, ...]
    retained_postconditions: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return self.pattern.status is PatternStatus.CANDIDATE and not self.counterexamples


def _credential_fields_from_mapping(values: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(key for key in values if _unsafe_key(str(key)))


def _paths_from_values(values: Sequence[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for item in values:
        if _looks_like_path(item) and item not in result:
            result.append(str(item))
    return tuple(result)


def _step_effect_classes(
    operation: StepOperation, effect_ids: Sequence[str]
) -> tuple[EffectClass, ...]:
    classes: list[EffectClass] = []
    for effect_id in effect_ids:
        mapped = _effect_class_for_id(effect_id)
        if mapped is not None and mapped not in classes:
            classes.append(mapped)
    if not classes:
        classes.extend(_effect_classes_for_operation(operation))
    return tuple(classes)


def _project_trajectory(trajectory: ExecutionTrajectory) -> TraceView:
    if trajectory.outcome.status not in POSITIVE_TERMINAL_STATUSES:
        raise AntiUnificationError(
            "only accepted, recovered, or rolled-back trajectories can be anti-unified"
        )
    steps: list[TraceStepView] = []
    failures: list[str] = []
    holes: list[tuple[str, str]] = []
    credentials: list[str] = []
    for step in trajectory.steps:
        if step.operation.value in FORBIDDEN_STEP_OPERATIONS:
            raise AntiUnificationError("forbidden operations cannot enter anti-unification")
        hole = _hole_type(step.hole_type)
        hole_type = hole.value if hole is not None else ""
        if hole_type:
            holes.append((f"hole.step-{step.sequence}", hole_type))
        if step.status.value in {"failed", "rolled_back", "retrying"}:
            failures.append(step.status.value)
        validation_ids = step.validation_receipt_cids if step.operation in VALIDATION_OPERATIONS else ()
        instance_state = (
            ("initial_state_cid", step.initial_state_cid),
            ("terminal_state_cid", step.terminal_state_cid),
        )
        view = TraceStepView(
            index=step.sequence,
            operation=step.operation,
            operation_contract=step.operation_contract,
            effect_ids=step.effect_ids,
            effect_classes=_step_effect_classes(step.operation, step.effect_ids),
            validation_ids=validation_ids,
            hole_type=hole_type,
            failure=step.status.value,
            observations=step.observation_cids,
            instance_state=instance_state,
        )
        steps.append(view)
    if trajectory.outcome.status is TrajectoryTerminalStatus.FAILED_RECOVERED:
        failures.append(FailureTransition.RETRY.value)
    elif trajectory.outcome.status is TrajectoryTerminalStatus.ROLLED_BACK:
        failures.append(FailureTransition.ROLLBACK.value)
    validation_ops = tuple(
        step.operation.value for step in steps if step.operation in VALIDATION_OPERATIONS
    )
    validation_contracts = tuple(
        step.operation_contract for step in steps if step.operation in VALIDATION_OPERATIONS
    )
    effect_classes = _unique_enums(
        tuple(item for step in steps for item in step.effect_classes)
    )
    return TraceView(
        cid=trajectory.content_id,
        bindings=trajectory.bindings,
        kind="trajectory",
        family_hint=trajectory.task_family_hint,
        steps=tuple(steps),
        preconditions=trajectory.objective_criterion_ids,
        postconditions=trajectory.outcome.accepted_criterion_ids,
        validation_contracts=_unique(validation_contracts),
        validation_operations=_unique(validation_ops),
        failure_transitions=_unique(failures),
        authority_ids=(),
        effect_classes=effect_classes,
        paths=(),
        credential_fields=tuple(credentials),
        holes=tuple(holes),
        policy_revision=trajectory.bindings.policy_revision,
        terminal_status=trajectory.outcome.status.value,
    )


def _unique_enums(values: Sequence[EffectClass]) -> tuple[EffectClass, ...]:
    result: list[EffectClass] = []
    for item in values:
        if item not in result:
            result.append(item)
    return tuple(result)


def _project_procedure(spec: ProcedureSpec) -> TraceView:
    if spec.state in {ArtifactState.PROMOTED, ArtifactState.VERIFIED}:
        raise AntiUnificationError("a ProcedureSpec cannot anti-unify a self-certified artifact")
    steps: list[TraceStepView] = []
    failures: list[str] = []
    holes: list[tuple[str, str]] = []
    credentials: list[str] = []
    paths: list[str] = []
    path_values = list(spec.declared_reads) + list(spec.scope_paths)
    for effect in spec.declared_effects:
        path_values.extend(effect.targets)
    paths.extend(_paths_from_values(path_values))
    hole_types = {item.hole_id: item.hole_type.value for item in spec.holes}
    effect_by_id = {item.effect_id: item for item in spec.declared_effects}
    for index, step in enumerate(spec.steps):
        if step.operation.value in FORBIDDEN_STEP_OPERATIONS:
            raise AntiUnificationError("forbidden operations cannot enter anti-unification")
        hole_type = hole_types.get(step.hole_id, "")
        if hole_type:
            _hole_type(hole_type, required=True)
            holes.append((step.hole_id, hole_type))
        failures.append(step.failure_transition.value)
        creds = _credential_fields_from_mapping(step.input_bindings)
        creds += _credential_fields_from_mapping(step.output_bindings)
        for name in creds:
            if name not in credentials:
                credentials.append(name)
        collected_paths = [value for _, value in step.input_bindings.items()]
        for effect_id in step.declared_effect_ids:
            effect = effect_by_id.get(effect_id)
            if effect is not None:
                collected_paths.extend(effect.targets)
        step_paths = _paths_from_values(collected_paths)
        for path in step_paths:
            if path not in paths:
                paths.append(path)
        effect_classes = []
        for effect_id in step.declared_effect_ids:
            effect = effect_by_id.get(effect_id)
            if effect is not None and effect.effect_class not in effect_classes:
                effect_classes.append(effect.effect_class)
        if not effect_classes:
            effect_classes.extend(_effect_classes_for_operation(step.operation))
        validation_ids = step.evidence_outputs if step.operation in VALIDATION_OPERATIONS else ()
        steps.append(
            TraceStepView(
                index=index,
                operation=step.operation,
                operation_contract=step.operation_contract,
                effect_ids=step.declared_effect_ids,
                effect_classes=tuple(effect_classes),
                authority_ids=step.required_authority_ids,
                validation_ids=validation_ids,
                hole_type=hole_type,
                hole_id=step.hole_id,
                failure=step.failure_transition.value,
                paths=step_paths,
                credential_fields=creds,
                observations=step.evidence_outputs,
                input_bindings=tuple(sorted(step.input_bindings.items())),
                output_bindings=tuple(sorted(step.output_bindings.items())),
            )
        )
    for parameter in spec.parameters:
        if _unsafe_key(parameter.name):
            if parameter.name not in credentials:
                credentials.append(parameter.name)
        if parameter.value_type is ValueType.RELATIVE_PATH or parameter.path_scoped:
            paths.extend(_paths_from_values(tuple(str(item) for item in parameter.allowed_values)))
            if parameter.default_value is not None:
                paths.extend(_paths_from_values((str(parameter.default_value),)))
    authority_ids = spec.authority.requirement_ids if spec.authority is not None else ()
    validation = spec.validation
    validation_contracts = ()
    if validation is not None:
        validation_contracts = _unique(
            tuple(validation.required_test_contracts)
            + tuple(validation.required_proof_contracts)
            + tuple(validation.post_merge_validation_contracts)
            + ((validation.full_test_fallback_contract,) if validation.full_test_fallback_contract else ())
        )
    validation_ops = _unique(
        tuple(step.operation.value for step in steps if step.operation in VALIDATION_OPERATIONS)
        + tuple(validation.required_step_ids if validation is not None else ())
    )
    effect_classes = _unique_enums(
        tuple(item.effect_class for item in spec.declared_effects)
        + tuple(item for step in steps for item in step.effect_classes)
    )
    return TraceView(
        cid=spec.content_id,
        bindings=spec.bindings,
        kind="procedure",
        family_hint=spec.task_family_id,
        steps=tuple(steps),
        preconditions=tuple(item.condition_id for item in spec.preconditions),
        postconditions=tuple(item.condition_id for item in spec.postconditions),
        validation_contracts=validation_contracts,
        validation_operations=validation_ops,
        failure_transitions=_unique(failures),
        authority_ids=_unique(tuple(authority_ids) + tuple(
            auth for step in steps for auth in step.authority_ids
        )),
        effect_classes=effect_classes,
        paths=_unique(tuple(paths)),
        credential_fields=tuple(credentials),
        holes=tuple(holes),
        policy_revision=spec.bindings.policy_revision
        if spec.authority is None
        else spec.authority.authority_policy_revision,
        terminal_status=spec.state.value,
    )


def project_trace(source: TraceSource) -> TraceView:
    """Project one admitted trajectory or procedure into a structural view."""

    if isinstance(source, ExecutionTrajectory):
        return _project_trajectory(source)
    if isinstance(source, ProcedureSpec):
        return _project_procedure(source)
    raise AntiUnificationError("traces must be ExecutionTrajectory or ProcedureSpec")


def _align_slots(traces: Sequence[TraceView]) -> tuple[AlignedSlot, ...]:
    slots = [
        AlignedSlot(operation=step.operation, members=MappingProxyType({0: step}))
        for step in traces[0].steps
    ]
    for trace_index, trace in enumerate(traces[1:], start=1):
        left_ops = tuple(slot.operation for slot in slots)
        right_ops = tuple(step.operation for step in trace.steps)
        pairs = _lcs_pairs(left_ops, right_ops)
        pair_index = 0
        merged: list[AlignedSlot] = []
        i = 0
        j = 0
        while i < len(slots) or j < len(trace.steps):
            matched = (
                pair_index < len(pairs)
                and i == pairs[pair_index][0]
                and j == pairs[pair_index][1]
            )
            if matched:
                members = dict(slots[i].members)
                members[trace_index] = trace.steps[j]
                merged.append(
                    AlignedSlot(operation=slots[i].operation, members=MappingProxyType(members))
                )
                i += 1
                j += 1
                pair_index += 1
            elif i < len(slots) and (
                pair_index >= len(pairs) or i < pairs[pair_index][0]
            ):
                merged.append(slots[i])
                i += 1
            else:
                merged.append(
                    AlignedSlot(
                        operation=trace.steps[j].operation,
                        members=MappingProxyType({trace_index: trace.steps[j]}),
                    )
                )
                j += 1
        slots = merged
    if len(slots) > MAX_STEPS:
        raise AntiUnificationError("aligned pattern exceeds the step bound")
    return tuple(slots)


def _predecessor_operation(slots: Sequence[AlignedSlot], index: int) -> str:
    for current in range(index - 1, -1, -1):
        if len(slots[current].members) > 0:
            return slots[current].operation.value
    return ""


def _successor_operation(slots: Sequence[AlignedSlot], index: int) -> str:
    for current in range(index + 1, len(slots)):
        if len(slots[current].members) > 0:
            return slots[current].operation.value
    return ""


class _Builder:
    def __init__(self, traces: Sequence[TraceView], family: TaskFamily) -> None:
        self.traces = traces
        self.family = family
        self.parameters: list[PatternParameter] = []
        self.holes: list[PatternHole] = []
        self.lost: list[LostDetail] = []
        self.counterexamples: list[GeneralizationCounterexample] = []
        self._param_index = 0
        self._hole_index = 0
        self._lost_index = 0
        self._branch_index = 0
        self._seen_counterexamples: set[tuple[Any, ...]] = set()

    def _next_param(self, field: str) -> str:
        self._param_index += 1
        return f"param.{field}.{self._param_index}"

    def _next_hole(self, reason: str) -> str:
        self._hole_index += 1
        return f"hole.{reason}.{self._hole_index}"

    def _next_lost(self, kind: str) -> str:
        self._lost_index += 1
        return f"lost.{kind}.{self._lost_index}"

    def _next_branch(self) -> str:
        self._branch_index += 1
        return f"branch.optional.{self._branch_index}"

    def add_lost(
        self,
        kind: LostDetailKind,
        location: str,
        disposition: LostDetailDisposition,
        source_values: Sequence[tuple[str, Any]],
    ) -> LostDetail:
        detail = LostDetail(
            detail_id=self._next_lost(kind.value),
            kind=kind,
            location=location,
            disposition=disposition,
            source_values=tuple(source_values),
        )
        if len(self.lost) < MAX_LOST_DETAILS:
            self.lost.append(detail)
        return detail

    def add_counterexample(
        self,
        violation: UnsafeMergeClass,
        left_index: int,
        right_index: int,
        location: str,
        left_value: Any,
        right_value: Any,
    ) -> None:
        key = (
            violation.value,
            self.traces[left_index].cid,
            self.traces[right_index].cid,
            location,
            canonical_json_bytes({"v": _canonical(left_value, "left_value")}).decode("utf-8"),
            canonical_json_bytes({"v": _canonical(right_value, "right_value")}).decode("utf-8"),
        )
        if key in self._seen_counterexamples:
            return
        self._seen_counterexamples.add(key)
        if len(self.counterexamples) >= MAX_COUNTEREXAMPLES:
            return
        self.counterexamples.append(
            GeneralizationCounterexample(
                violation_class=violation,
                left_trace_cid=self.traces[left_index].cid,
                right_trace_cid=self.traces[right_index].cid,
                location=location,
                left_value=left_value,
                right_value=right_value,
                evidence_cids=(self.traces[left_index].cid, self.traces[right_index].cid),
            )
        )

    def pairwise_counterexamples(
        self,
        violation: UnsafeMergeClass,
        location: str,
        values: Sequence[Any],
        *,
        present: Sequence[bool] | None = None,
    ) -> None:
        for left in range(len(self.traces)):
            for right in range(left + 1, len(self.traces)):
                if present is not None and not (present[left] or present[right]):
                    continue
                if present is not None and present[left] == present[right] and values[left] == values[right]:
                    continue
                if present is None and values[left] == values[right]:
                    continue
                self.add_counterexample(
                    violation, left, right, location, values[left], values[right]
                )

    def validation_ids(self) -> tuple[str, ...]:
        ids = [
            item
            for trace in self.traces
            for item in (trace.validation_operations + trace.validation_contracts)
        ]
        if not ids:
            ids = ["observation.retained-validation"]
        return _unique(ids)[:MAX_ITEMS] or ("observation.retained-validation",)

    def add_hole(
        self,
        hole_type: HoleType,
        reason: str,
        source_field: str,
        source_values: Sequence[Any],
    ) -> PatternHole:
        if len(self.holes) >= MAX_PATTERN_HOLES:
            raise AntiUnificationError("typed holes exceed the hole bound")
        hole = PatternHole(
            hole_id=self._next_hole(reason),
            hole_type=hole_type,
            reason=reason,
            source_field=source_field,
            source_values=tuple(source_values),
            validation_observation_ids=self.validation_ids(),
        )
        self.holes.append(hole)
        return hole

    def add_parameter(
        self,
        source_field: str,
        values: Sequence[Any],
        *,
        required: bool = True,
    ) -> PatternParameter | None:
        value_type = _value_type_for(values)
        if value_type is None:
            self.add_hole(
                HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
                "uncertain-value",
                source_field,
                tuple(
                    item if isinstance(item, (str, int, bool)) else str(item)
                    for item in values
                ),
            )
            return None
        if value_type is ValueType.RELATIVE_PATH:
            return None
        if value_type in {ValueType.IDENTIFIER, ValueType.STRING, ValueType.ENUM}:
            allowed_values: tuple[Any, ...] = _unique(
                tuple(str(item) for item in values if item not in (None, ""))
            )
        else:
            allowed_values = tuple(dict.fromkeys(values))
        if value_type is ValueType.STRING and all(
            _identifier_ok(str(item)) for item in allowed_values
        ):
            value_type = ValueType.IDENTIFIER
        parameter = PatternParameter(
            name=self._next_param(source_field),
            value_type=value_type,
            source_field=source_field,
            allowed_values=allowed_values,
            required=required,
        )
        self.parameters.append(parameter)
        return parameter


def _identifier_ok(value: str) -> bool:
    try:
        _identifier(value, "parameter_value")
    except ProcedureContractError:
        return False
    return True


def _infer_from_slots(
    builder: _Builder, slots: Sequence[AlignedSlot]
) -> tuple[tuple[PatternStep, ...], tuple[OptionalBranch, ...]]:
    traces = builder.traces
    n = len(traces)
    steps: list[PatternStep] = []
    branches: list[OptionalBranch] = []
    required_index = 0
    for slot_index, slot in enumerate(slots):
        present_indexes = tuple(sorted(slot.members))
        present_cids = tuple(traces[index].cid for index in present_indexes)
        members = tuple(slot.members[index] for index in present_indexes)
        missing = n - len(present_indexes)
        location = f"step.{slot.operation.value}.{slot_index}"
        if missing and slot.operation in TEST_OPERATIONS:
            flags = tuple(index in slot.members for index in range(n))
            values = tuple(
                traces[index].steps[slot.members[index].index].operation_contract
                if index in slot.members
                else ""
                for index in range(n)
            )
            builder.pairwise_counterexamples(
                UnsafeMergeClass.OMITTED_TEST, location, values, present=flags
            )
            builder.add_lost(
                LostDetailKind.VALIDATION,
                location,
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple(
                    (traces[index].cid, "present" if index in slot.members else "absent")
                    for index in range(n)
                ),
            )
        elif missing and slot.operation in POSTCONDITION_OPERATIONS:
            flags = tuple(index in slot.members for index in range(n))
            values = tuple("present" if flag else "absent" for flag in flags)
            builder.pairwise_counterexamples(
                UnsafeMergeClass.MISSING_POSTCONDITION, location, values, present=flags
            )
            builder.add_lost(
                LostDetailKind.POSTCONDITION,
                location,
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple((traces[index].cid, values[index]) for index in range(n)),
            )
        elif missing and slot.operation in VALIDATION_OPERATIONS:
            flags = tuple(index in slot.members for index in range(n))
            values = tuple("present" if flag else "absent" for flag in flags)
            builder.pairwise_counterexamples(
                UnsafeMergeClass.VALIDATION_SPLIT, location, values, present=flags
            )
            builder.add_lost(
                LostDetailKind.VALIDATION,
                location,
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple((traces[index].cid, values[index]) for index in range(n)),
            )
        elif missing and slot.operation in AUTHORITY_OPERATIONS:
            flags = tuple(index in slot.members for index in range(n))
            values = tuple("present" if flag else "absent" for flag in flags)
            builder.pairwise_counterexamples(
                UnsafeMergeClass.UNCERTAIN_AUTHORITY, location, values, present=flags
            )
            builder.add_lost(
                LostDetailKind.AUTHORITY,
                location,
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple((traces[index].cid, values[index]) for index in range(n)),
            )
        elif missing:
            material = any(
                effect in MATERIAL_EFFECT_CLASSES
                for member in members
                for effect in member.effect_classes
            )
            if material or slot.operation not in OPTIONAL_OPERATIONS:
                hole_type = (
                    HoleType.CHOOSE_APPROVED_REPAIR_TEMPLATE
                    if slot.operation is StepOperation.APPLY_APPROVED_PATCH_TEMPLATE
                    else HoleType.CLASSIFY_FAILURE
                    if slot.operation is StepOperation.REQUEST_TYPED_MODEL_HOLE
                    else HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS
                )
                hole = builder.add_hole(
                    hole_type,
                    "uncertain-step",
                    location,
                    (slot.operation.value,),
                )
                builder.add_lost(
                    LostDetailKind.TYPED_HOLE,
                    location,
                    LostDetailDisposition.TYPED_HOLE,
                    tuple(
                        (traces[index].cid, "present" if index in slot.members else "absent")
                        for index in range(n)
                    ),
                )
                steps.append(
                    PatternStep(
                        step_index=required_index,
                        operation=slot.operation,
                        operation_contract=members[0].operation_contract
                        if len({item.operation_contract for item in members}) == 1
                        else "",
                        presence=StepPresence.HOLE,
                        effect_ids=_unique(
                            tuple(item for member in members for item in member.effect_ids)
                        ),
                        effect_classes=_unique(
                            tuple(
                                item.value
                                for member in members
                                for item in member.effect_classes
                            )
                        ),
                        hole_id=hole.hole_id,
                        source_trace_cids=present_cids,
                    )
                )
                required_index += 1
                continue
            if len(branches) >= MAX_OPTIONAL_BRANCHES:
                builder.add_hole(
                    HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
                    "optional-branch-bound",
                    location,
                    (slot.operation.value,),
                )
            else:
                branches.append(
                    OptionalBranch(
                        branch_id=builder._next_branch(),
                        operation=slot.operation,
                        operation_contract=members[0].operation_contract
                        if len({item.operation_contract for item in members}) == 1
                        else "",
                        predecessor_operation=_predecessor_operation(slots, slot_index),
                        successor_operation=_successor_operation(slots, slot_index),
                        source_trace_cids=present_cids,
                    )
                )
            builder.add_lost(
                LostDetailKind.OPTIONAL_BRANCH,
                location,
                LostDetailDisposition.OPTIONAL_BRANCH,
                tuple(
                    (traces[index].cid, "present" if index in slot.members else "absent")
                    for index in range(n)
                ),
            )
            continue

        contracts = tuple(member.operation_contract for member in members)
        parameter_names: list[str] = []
        operation_contract = contracts[0]
        if len(set(contracts)) != 1:
            if slot.operation in AUTHORITY_OPERATIONS:
                builder.pairwise_counterexamples(
                    UnsafeMergeClass.AUTHORITY_SPLIT, f"{location}.operation_contract", contracts
                )
                builder.add_lost(
                    LostDetailKind.AUTHORITY,
                    f"{location}.operation_contract",
                    LostDetailDisposition.COUNTEREXAMPLE,
                    tuple((cid, value) for cid, value in zip(present_cids, contracts)),
                )
                operation_contract = ""
            elif slot.operation in TEST_OPERATIONS:
                parameter = builder.add_parameter("operation_contract", contracts)
                if parameter is not None:
                    parameter_names.append(parameter.name)
                    operation_contract = parameter.name
                builder.add_lost(
                    LostDetailKind.PARAMETER,
                    f"{location}.operation_contract",
                    LostDetailDisposition.PARAMETER,
                    tuple((cid, value) for cid, value in zip(present_cids, contracts)),
                )
            else:
                parameter = builder.add_parameter("operation_contract", contracts)
                if parameter is not None:
                    parameter_names.append(parameter.name)
                    operation_contract = parameter.name
                builder.add_lost(
                    LostDetailKind.PARAMETER,
                    f"{location}.operation_contract",
                    LostDetailDisposition.PARAMETER,
                    tuple((cid, value) for cid, value in zip(present_cids, contracts)),
                )

        effect_sets = tuple(frozenset(item.value for item in member.effect_classes) for member in members)
        if len(set(effect_sets)) != 1:
            material_names = {item.value for item in MATERIAL_EFFECT_CLASSES}
            if any((left ^ right) & material_names for left, right in zip(effect_sets, effect_sets[1:])):
                builder.pairwise_counterexamples(
                    UnsafeMergeClass.EFFECT_SPLIT,
                    f"{location}.effect_classes",
                    tuple(tuple(sorted(item)) for item in effect_sets),
                )
                builder.add_lost(
                    LostDetailKind.EFFECT,
                    f"{location}.effect_classes",
                    LostDetailDisposition.COUNTEREXAMPLE,
                    tuple(
                        (cid, tuple(sorted(value)))
                        for cid, value in zip(present_cids, effect_sets)
                    ),
                )

        authority_sets = tuple(frozenset(member.authority_ids) for member in members)
        if len(set(authority_sets)) != 1:
            builder.pairwise_counterexamples(
                UnsafeMergeClass.AUTHORITY_SPLIT,
                f"{location}.authority_ids",
                tuple(tuple(sorted(item)) for item in authority_sets),
            )
            builder.add_lost(
                LostDetailKind.AUTHORITY,
                f"{location}.authority_ids",
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple((cid, tuple(sorted(value))) for cid, value in zip(present_cids, authority_sets)),
            )

        path_sets = tuple(frozenset(member.paths) for member in members)
        if any(member.paths for member in members) and len(set(path_sets)) != 1:
            builder.pairwise_counterexamples(
                UnsafeMergeClass.PATH_GENERALIZATION,
                f"{location}.paths",
                tuple(tuple(sorted(item)) for item in path_sets),
            )
            builder.add_lost(
                LostDetailKind.PATH,
                f"{location}.paths",
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple((cid, tuple(sorted(value))) for cid, value in zip(present_cids, path_sets)),
            )

        credential_sets = tuple(frozenset(member.credential_fields) for member in members)
        if any(member.credential_fields for member in members):
            for left in range(len(present_indexes)):
                for right in range(left + 1, len(present_indexes)):
                    builder.add_counterexample(
                        UnsafeMergeClass.CREDENTIAL_GENERALIZATION,
                        present_indexes[left],
                        present_indexes[right],
                        f"{location}.credentials",
                        tuple(sorted(credential_sets[left])),
                        tuple(sorted(credential_sets[right])),
                    )
            builder.add_lost(
                LostDetailKind.CREDENTIAL,
                f"{location}.credentials",
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple(
                    (cid, tuple(sorted(value)))
                    for cid, value in zip(present_cids, credential_sets)
                ),
            )

        hole_types = tuple(member.hole_type for member in members)
        hole_id = ""
        if any(hole_types):
            if any(item in FORBIDDEN_HOLE_TYPES for item in hole_types):
                builder.pairwise_counterexamples(
                    UnsafeMergeClass.FORBIDDEN_HOLE, f"{location}.hole_type", hole_types
                )
            elif len(set(item for item in hole_types if item)) > 1:
                parameter = builder.add_parameter("hole_type", tuple(item for item in hole_types if item))
                if parameter is not None:
                    parameter_names.append(parameter.name)
                hole = builder.add_hole(
                    HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
                    "hole-type",
                    f"{location}.hole_type",
                    tuple(item for item in hole_types if item),
                )
                hole_id = hole.hole_id
                builder.add_lost(
                    LostDetailKind.HOLE,
                    f"{location}.hole_type",
                    LostDetailDisposition.TYPED_HOLE,
                    tuple((cid, value) for cid, value in zip(present_cids, hole_types)),
                )
            elif slot.operation is StepOperation.REQUEST_TYPED_MODEL_HOLE:
                hole_id = members[0].hole_id

        instance_states = tuple(
            tuple(f"{key}:{value}" for key, value in member.instance_state)
            for member in members
        )
        if any(instance_states) and len(set(instance_states)) != 1:
            builder.add_lost(
                LostDetailKind.INSTANCE_STATE,
                f"{location}.instance_state",
                LostDetailDisposition.INSTANCE_STATE,
                tuple((cid, value) for cid, value in zip(present_cids, instance_states)),
            )

        failures = tuple(member.failure for member in members)
        failure = failures[0] if len(set(failures)) == 1 else ""
        if len(set(failures)) != 1:
            builder.add_lost(
                LostDetailKind.FAILURE,
                f"{location}.failure",
                LostDetailDisposition.RETAINED_UNION,
                tuple((cid, value) for cid, value in zip(present_cids, failures)),
            )

        presence = StepPresence.REQUIRED
        if hole_id and slot.operation is StepOperation.REQUEST_TYPED_MODEL_HOLE and len(set(item for item in hole_types if item)) > 1:
            presence = StepPresence.REQUIRED
        steps.append(
            PatternStep(
                step_index=required_index,
                operation=slot.operation,
                operation_contract=operation_contract,
                presence=presence,
                effect_ids=_unique(tuple(item for member in members for item in member.effect_ids)),
                effect_classes=_unique(
                    tuple(item.value for member in members for item in member.effect_classes)
                ),
                authority_ids=_unique(
                    tuple(item for member in members for item in member.authority_ids)
                ),
                parameter_names=tuple(parameter_names),
                hole_id=hole_id,
                failure_transition=failure,
                source_trace_cids=present_cids,
            )
        )
        required_index += 1
    return tuple(steps), tuple(branches)


def _detect_order_inversions(builder: _Builder) -> None:
    traces = builder.traces
    for left in range(len(traces)):
        for right in range(left + 1, len(traces)):
            left_ops = tuple(step.operation for step in traces[left].steps)
            right_ops = tuple(step.operation for step in traces[right].steps)
            left_index = {operation: index for index, operation in enumerate(left_ops)}
            right_index = {operation: index for index, operation in enumerate(right_ops)}
            shared = [operation for operation in left_ops if operation in right_index]
            for first, second in zip(shared, shared[1:]):
                if first not in right_index or second not in right_index:
                    continue
                if left_index[first] < left_index[second] and right_index[first] > right_index[second]:
                    location = f"order.{first.value}.{second.value}"
                    if first in VALIDATION_OPERATIONS or second in VALIDATION_OPERATIONS:
                        builder.add_counterexample(
                            UnsafeMergeClass.VALIDATION_SPLIT,
                            left,
                            right,
                            location,
                            (first.value, second.value),
                            (second.value, first.value),
                        )
                        builder.add_lost(
                            LostDetailKind.ORDER,
                            location,
                            LostDetailDisposition.COUNTEREXAMPLE,
                            (
                                (traces[left].cid, (first.value, second.value)),
                                (traces[right].cid, (second.value, first.value)),
                            ),
                        )
                    else:
                        builder.add_hole(
                            HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
                            "order",
                            location,
                            ((first.value, second.value), (second.value, first.value)),
                        )
                        builder.add_lost(
                            LostDetailKind.ORDER,
                            location,
                            LostDetailDisposition.TYPED_HOLE,
                            (
                                (traces[left].cid, (first.value, second.value)),
                                (traces[right].cid, (second.value, first.value)),
                            ),
                        )


def _detect_global_splits(builder: _Builder) -> None:
    traces = builder.traces
    n = len(traces)
    path_sets = tuple(frozenset(trace.paths) for trace in traces)
    if any(trace.paths for trace in traces) and len(set(path_sets)) != 1:
        builder.pairwise_counterexamples(
            UnsafeMergeClass.PATH_GENERALIZATION,
            "trace.paths",
            tuple(tuple(sorted(item)) for item in path_sets),
        )
        builder.add_lost(
            LostDetailKind.PATH,
            "trace.paths",
            LostDetailDisposition.COUNTEREXAMPLE,
            tuple((trace.cid, tuple(sorted(trace.paths))) for trace in traces),
        )
    elif any(trace.paths for trace in traces):
        builder.add_lost(
            LostDetailKind.PATH,
            "trace.paths",
            LostDetailDisposition.INSTANCE_STATE,
            tuple((trace.cid, tuple(sorted(trace.paths))) for trace in traces),
        )

    credential_sets = tuple(frozenset(trace.credential_fields) for trace in traces)
    if any(trace.credential_fields for trace in traces):
        for left in range(n):
            for right in range(left + 1, n):
                builder.add_counterexample(
                    UnsafeMergeClass.CREDENTIAL_GENERALIZATION,
                    left,
                    right,
                    "trace.credentials",
                    tuple(sorted(credential_sets[left])),
                    tuple(sorted(credential_sets[right])),
                )
        builder.add_lost(
            LostDetailKind.CREDENTIAL,
            "trace.credentials",
            LostDetailDisposition.STRIPPED,
            tuple((trace.cid, tuple(sorted(trace.credential_fields))) for trace in traces),
        )

    policies = tuple(trace.policy_revision for trace in traces)
    if len(set(policies)) != 1:
        builder.pairwise_counterexamples(
            UnsafeMergeClass.AUTHORITY_SPLIT, "bindings.policy_revision", policies
        )
        builder.add_lost(
            LostDetailKind.AUTHORITY,
            "bindings.policy_revision",
            LostDetailDisposition.COUNTEREXAMPLE,
            tuple((trace.cid, trace.policy_revision) for trace in traces),
        )

    authority_sets = tuple(frozenset(trace.authority_ids) for trace in traces)
    if len(set(authority_sets)) != 1:
        empty = tuple(not item for item in authority_sets)
        if any(empty) and not all(empty):
            builder.pairwise_counterexamples(
                UnsafeMergeClass.UNCERTAIN_AUTHORITY,
                "trace.authority_ids",
                tuple(tuple(sorted(item)) for item in authority_sets),
            )
        else:
            builder.pairwise_counterexamples(
                UnsafeMergeClass.AUTHORITY_SPLIT,
                "trace.authority_ids",
                tuple(tuple(sorted(item)) for item in authority_sets),
            )
        builder.add_lost(
            LostDetailKind.AUTHORITY,
            "trace.authority_ids",
            LostDetailDisposition.COUNTEREXAMPLE,
            tuple((trace.cid, tuple(sorted(trace.authority_ids))) for trace in traces),
        )

    effect_sets = tuple(frozenset(trace.effect_classes) for trace in traces)
    if len(set(effect_sets)) != 1:
        material_diff = False
        for left in range(n):
            for right in range(left + 1, n):
                delta = effect_sets[left] ^ effect_sets[right]
                if delta & MATERIAL_EFFECT_CLASSES:
                    material_diff = True
                    builder.add_counterexample(
                        UnsafeMergeClass.EFFECT_SPLIT,
                        left,
                        right,
                        "trace.effect_classes",
                        tuple(sorted(item.value for item in effect_sets[left])),
                        tuple(sorted(item.value for item in effect_sets[right])),
                    )
        if material_diff:
            builder.add_lost(
                LostDetailKind.EFFECT,
                "trace.effect_classes",
                LostDetailDisposition.COUNTEREXAMPLE,
                tuple(
                    (trace.cid, tuple(sorted(item.value for item in trace.effect_classes)))
                    for trace in traces
                ),
            )

    family_effects = frozenset(builder.family.effect_classes) | frozenset(
        builder.family.boundary.permitted_effect_classes
    )
    for index, trace in enumerate(traces):
        extra = frozenset(trace.effect_classes) - family_effects
        if extra:
            other = 0 if index else min(1, n - 1)
            builder.add_counterexample(
                UnsafeMergeClass.EFFECT_SPLIT,
                index,
                other,
                "family.effect_classes",
                tuple(sorted(item.value for item in extra)),
                tuple(sorted(item.value for item in family_effects)),
            )

    post_sets = tuple(frozenset(trace.postconditions) for trace in traces)
    if any(trace.terminal_status == TrajectoryTerminalStatus.ACCEPTED.value for trace in traces):
        if len(set(post_sets)) != 1:
            # Union is retained; a trace that dropped a postcondition is unsafe.
            for left in range(n):
                for right in range(left + 1, n):
                    if post_sets[left] != post_sets[right] and (
                        traces[left].terminal_status == TrajectoryTerminalStatus.ACCEPTED.value
                        or traces[right].terminal_status == TrajectoryTerminalStatus.ACCEPTED.value
                    ):
                        builder.add_counterexample(
                            UnsafeMergeClass.MISSING_POSTCONDITION,
                            left,
                            right,
                            "trace.postconditions",
                            tuple(sorted(post_sets[left])),
                            tuple(sorted(post_sets[right])),
                        )
            builder.add_lost(
                LostDetailKind.POSTCONDITION,
                "trace.postconditions",
                LostDetailDisposition.RETAINED_UNION,
                tuple((trace.cid, tuple(sorted(trace.postconditions))) for trace in traces),
            )

    validation_op_sets = tuple(frozenset(trace.validation_operations) for trace in traces)
    if len(set(validation_op_sets)) != 1:
        for left in range(n):
            for right in range(left + 1, n):
                if validation_op_sets[left] == validation_op_sets[right]:
                    continue
                delta = validation_op_sets[left] ^ validation_op_sets[right]
                violation = (
                    UnsafeMergeClass.OMITTED_TEST
                    if delta & {item.value for item in TEST_OPERATIONS}
                    else UnsafeMergeClass.VALIDATION_SPLIT
                )
                builder.add_counterexample(
                    violation,
                    left,
                    right,
                    "trace.validation_operations",
                    tuple(sorted(validation_op_sets[left])),
                    tuple(sorted(validation_op_sets[right])),
                )
        builder.add_lost(
            LostDetailKind.VALIDATION,
            "trace.validation_operations",
            LostDetailDisposition.RETAINED_UNION,
            tuple((trace.cid, tuple(sorted(trace.validation_operations))) for trace in traces),
        )

    validation_contract_sets = tuple(frozenset(trace.validation_contracts) for trace in traces)
    if len(set(validation_contract_sets)) != 1:
        builder.add_lost(
            LostDetailKind.VALIDATION,
            "trace.validation_contracts",
            LostDetailDisposition.RETAINED_UNION,
            tuple((trace.cid, tuple(sorted(trace.validation_contracts))) for trace in traces),
        )

    pre_sets = tuple(frozenset(trace.preconditions) for trace in traces)
    if len(set(pre_sets)) != 1:
        builder.add_lost(
            LostDetailKind.PRECONDITION,
            "trace.preconditions",
            LostDetailDisposition.RETAINED_UNION,
            tuple((trace.cid, tuple(sorted(trace.preconditions))) for trace in traces),
        )

    tree_ids = tuple(trace.bindings.tree_id for trace in traces)
    if len(set(tree_ids)) != 1:
        builder.add_lost(
            LostDetailKind.BINDING,
            "bindings.tree_id",
            LostDetailDisposition.INSTANCE_STATE,
            tuple((trace.cid, trace.bindings.tree_id) for trace in traces),
        )
    commits = tuple(trace.bindings.repository_commit for trace in traces)
    if len(set(commits)) != 1:
        builder.add_lost(
            LostDetailKind.BINDING,
            "bindings.repository_commit",
            LostDetailDisposition.INSTANCE_STATE,
            tuple((trace.cid, trace.bindings.repository_commit) for trace in traces),
        )


def _union_ordered(groups: Sequence[Sequence[str]]) -> tuple[str, ...]:
    result: list[str] = []
    for group in groups:
        for item in group:
            if item and item not in result:
                result.append(item)
    return tuple(result)


def _intersection_ordered(groups: Sequence[Sequence[str]]) -> tuple[str, ...]:
    if not groups:
        return ()
    shared = [item for item in groups[0] if all(item in group for group in groups[1:])]
    return tuple(shared)


def _admit_family_and_memberships(
    family: TaskFamily,
    traces: Sequence[TraceView],
    memberships: Sequence[TaskFamilyMembership],
) -> None:
    membership_by_cid = {}
    for item in memberships:
        record = item if isinstance(item, TaskFamilyMembership) else TaskFamilyMembership(**item)
        if record.task_family_cid not in {"", family.content_id} and record.task_family_cid != family.content_id:
            raise AntiUnificationError("membership task_family_cid does not match the family")
        membership_by_cid[record.trajectory_cid] = record
    negative = set(family.boundary.negative_example_cids)
    boundary = set(family.boundary.boundary_example_cids)
    unknown = set(family.boundary.unknown_case_cids)
    permitted_repos = set(family.boundary.permitted_repositories)
    for trace in traces:
        if trace.family_hint and trace.family_hint != family.name:
            raise AntiUnificationError("trace task-family hint does not match the family")
        if trace.cid in negative or trace.cid in boundary or trace.cid in unknown:
            raise AntiUnificationError("negative, boundary, or unknown traces cannot be anti-unified")
        membership = membership_by_cid.get(trace.cid)
        if membership is not None and membership.membership is not FamilyMembershipClass.POSITIVE:
            raise AntiUnificationError("only positive family members can be anti-unified")
        if trace.bindings.repository_id not in permitted_repos:
            raise AntiUnificationError("trace repository is outside the family boundary")
        if trace.bindings.objective_id != family.bindings.objective_id:
            raise AntiUnificationError("trace objective_id does not match the family")
        if trace.bindings.contract_revision != family.bindings.contract_revision:
            raise AntiUnificationError("trace contract_revision does not match the family")


def _compatible_bindings(traces: Sequence[TraceView]) -> ArtifactBindings:
    first = traces[0].bindings
    for trace in traces[1:]:
        if trace.bindings.objective_id != first.objective_id:
            raise AntiUnificationError("traces must share objective_id")
        if trace.bindings.contract_revision != first.contract_revision:
            raise AntiUnificationError("traces must share contract_revision")
    return first


def _constants_from(
    family: TaskFamily,
    traces: Sequence[TraceView],
    steps: Sequence[PatternStep],
    preconditions: Sequence[str],
    postconditions: Sequence[str],
    validation_operations: Sequence[str],
    validation_contracts: Sequence[str],
    failure_transitions: Sequence[str],
) -> dict[str, Any]:
    shared_contracts = _intersection_ordered(
        tuple(tuple(step.operation_contract for step in trace.steps) for trace in traces)
    )
    constants: dict[str, Any] = {
        "task_family_id": family.name,
        "required_operations": tuple(
            step.operation.value for step in steps if step.presence is StepPresence.REQUIRED
        ),
        "shared_operation_contracts": shared_contracts,
        "preconditions": tuple(preconditions),
        "postconditions": tuple(postconditions),
        "validation_operations": tuple(validation_operations),
        "validation_contracts": tuple(validation_contracts),
        "failure_transitions": tuple(failure_transitions),
        "effect_classes": tuple(
            sorted({item.value for trace in traces for item in trace.effect_classes})
        ),
        "unifier_revision": UNIFIER_REVISION,
    }
    shared_authority = _intersection_ordered(tuple(trace.authority_ids for trace in traces))
    if shared_authority:
        constants["authority_ids"] = shared_authority
    shared_paths = _intersection_ordered(tuple(trace.paths for trace in traces))
    if shared_paths:
        constants["paths"] = shared_paths
    return constants


def _boundary_artifact(
    bindings: ArtifactBindings,
    family: TaskFamily,
    pattern: AntiUnificationPattern,
    *,
    emitted_at_ms: int,
) -> GeneralizationBoundaryArtifact:
    return GeneralizationBoundaryArtifact(
        bindings=bindings,
        state=pattern.to_artifact(emitted_at_ms=emitted_at_ms).state,
        subject_cid=family.content_id,
        reference_cids=pattern.source_trace_cids,
        labels=("generalization-boundary", pattern.status.value),
        facts={
            "task_family_id": family.name,
            "forbidden_generalizations": FORBIDDEN_GENERALIZATIONS,
            "permitted_parameter_names": tuple(item.name for item in pattern.parameters),
            "retained_validations": pattern.validation_operations,
            "retained_validation_contracts": pattern.validation_contracts,
            "retained_postconditions": pattern.postconditions,
            "required_operations": pattern.required_operations,
            "typed_hole_ids": tuple(item.hole_id for item in pattern.holes),
        },
        created_at_ms=emitted_at_ms,
    )


class ProcedureAntiUnifier:
    """Infer a structural pattern from positive traces in one task family."""

    def __init__(self, *, unifier_revision: str = UNIFIER_REVISION, emitted_at_ms: int = 0) -> None:
        self.unifier_revision = _identifier(unifier_revision, "unifier_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")

    def anti_unify(
        self,
        family: TaskFamily,
        traces: Sequence[TraceSource],
        *,
        memberships: Sequence[TaskFamilyMembership] = (),
        emitted_at_ms: int | None = None,
    ) -> AntiUnificationResult:
        family = _family(family)
        if not isinstance(traces, Sequence) or isinstance(
            traces, (str, bytes, bytearray, memoryview)
        ):
            raise AntiUnificationError("traces must be a sequence")
        if len(traces) < 2:
            raise AntiUnificationError("at least two positive traces are required")
        if len(traces) > MAX_ITEMS:
            raise AntiUnificationError("traces exceeds its item bound")
        if not isinstance(memberships, Sequence) or isinstance(
            memberships, (str, bytes, bytearray, memoryview)
        ):
            raise AntiUnificationError("memberships must be a sequence")
        stamp = self.emitted_at_ms if emitted_at_ms is None else _nonnegative_int(
            emitted_at_ms, "emitted_at_ms"
        )
        views = tuple(project_trace(item) for item in traces)
        seen: set[str] = set()
        for view in views:
            if view.cid in seen:
                raise AntiUnificationError("trace content identities must be unique")
            seen.add(view.cid)
        _admit_family_and_memberships(family, views, memberships)
        bindings = _compatible_bindings(views)
        builder = _Builder(views, family)
        _detect_global_splits(builder)
        _detect_order_inversions(builder)
        slots = _align_slots(views)
        steps, branches = _infer_from_slots(builder, slots)
        preconditions = _union_ordered(tuple(view.preconditions for view in views))
        postconditions = _union_ordered(tuple(view.postconditions for view in views))
        validation_operations = _union_ordered(
            tuple(view.validation_operations for view in views)
        )
        validation_contracts = _union_ordered(
            tuple(view.validation_contracts for view in views)
        )
        failure_transitions = _union_ordered(
            tuple(view.failure_transitions for view in views)
        )
        if not validation_operations and not validation_contracts:
            raise AntiUnificationError("validation cannot disappear from an anti-unified pattern")
        if not postconditions and any(
            view.terminal_status == TrajectoryTerminalStatus.ACCEPTED.value for view in views
        ):
            raise AntiUnificationError("postconditions cannot disappear from an anti-unified pattern")
        constants = _constants_from(
            family,
            views,
            steps,
            preconditions,
            postconditions,
            validation_operations,
            validation_contracts,
            failure_transitions,
        )
        status = (
            PatternStatus.REJECTED if builder.counterexamples else PatternStatus.CANDIDATE
        )
        pattern = AntiUnificationPattern(
            bindings=bindings,
            task_family_id=family.name,
            task_family_cid=family.content_id,
            source_trace_cids=tuple(view.cid for view in views),
            constants=constants,
            parameters=tuple(builder.parameters),
            steps=steps,
            optional_branches=branches,
            holes=tuple(builder.holes),
            preconditions=preconditions,
            postconditions=postconditions,
            validation_contracts=validation_contracts,
            validation_operations=validation_operations,
            failure_transitions=failure_transitions,
            lost_details=tuple(builder.lost),
            status=status,
            unifier_revision=self.unifier_revision,
        )
        pattern_artifact = pattern.to_artifact(emitted_at_ms=stamp)
        counterexample_artifacts = tuple(
            item.to_artifact(bindings, family_cid=family.content_id, emitted_at_ms=stamp)
            for item in builder.counterexamples
        )
        return AntiUnificationResult(
            bindings=bindings,
            pattern=pattern,
            counterexamples=tuple(builder.counterexamples),
            pattern_artifact=pattern_artifact,
            counterexample_artifacts=counterexample_artifacts,
            boundary_artifact=_boundary_artifact(
                bindings, family, pattern, emitted_at_ms=stamp
            ),
            lost_details=pattern.lost_details,
            retained_validations=pattern.validation_operations,
            retained_postconditions=pattern.postconditions,
        )

    def anti_unify_procedures(
        self,
        family: TaskFamily,
        procedures: Sequence[ProcedureSpec],
        *,
        emitted_at_ms: int | None = None,
    ) -> AntiUnificationResult:
        return self.anti_unify(family, procedures, emitted_at_ms=emitted_at_ms)


def anti_unify(
    family: TaskFamily,
    traces: Sequence[TraceSource],
    *,
    memberships: Sequence[TaskFamilyMembership] = (),
    emitted_at_ms: int = 0,
) -> AntiUnificationResult:
    """Anti-unify positive traces into a candidate pattern or counterexamples."""

    return ProcedureAntiUnifier(emitted_at_ms=emitted_at_ms).anti_unify(
        family, traces, memberships=memberships, emitted_at_ms=emitted_at_ms
    )


def anti_unify_trajectories(
    family: TaskFamily,
    trajectories: Sequence[ExecutionTrajectory],
    *,
    memberships: Sequence[TaskFamilyMembership] = (),
    emitted_at_ms: int = 0,
) -> AntiUnificationResult:
    return anti_unify(
        family, trajectories, memberships=memberships, emitted_at_ms=emitted_at_ms
    )


def anti_unify_procedures(
    family: TaskFamily,
    procedures: Sequence[ProcedureSpec],
    *,
    emitted_at_ms: int = 0,
) -> AntiUnificationResult:
    return ProcedureAntiUnifier(emitted_at_ms=emitted_at_ms).anti_unify_procedures(
        family, procedures, emitted_at_ms=emitted_at_ms
    )
