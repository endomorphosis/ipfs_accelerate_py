"""Canonical, provider-independent contracts for formal supervisor work plans.

The records in this module describe *intended supervisor work*.  They reuse
the strict canonical JSON and content identity boundary from
``formal_verification_contracts`` but deliberately do not claim that a valid
or consistent plan proves anything about generated source code.

Every logical expression is referenced by the content identity of a formula
constructed by :mod:`formal_logic_vocabulary`.  Free-form text is allowed only
as non-semantic metadata; it is never parsed into a formula by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from .formal_logic_vocabulary import (
    Formula,
    FormulaOperator,
    LOGIC_VOCABULARY_VERSION,
    ReviewedPredicate,
    TermKind,
    TermSort,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    _enum,
    _mapping,
    _text,
)


FORMAL_PLANNING_CONTRACT_VERSION = 1
PLANNING_CONTRACT_VERSION = FORMAL_PLANNING_CONTRACT_VERSION
SCHEMA_VERSION = FORMAL_PLANNING_CONTRACT_VERSION

FORMAL_WORK_PLAN_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-work-plan@1"
ACTOR_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-actor@1"
GOAL_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-goal@1"
SUBGOAL_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-subgoal@1"
PLAN_TASK_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-task@1"
PLAN_EVENT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-event@1"
FLUENT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-fluent@1"
PRECONDITION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-precondition@1"
EFFECT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-effect@1"
NORM_SCHEMA = "ipfs_accelerate_py/agent-supervisor/formal-plan-norm@1"
TEMPORAL_CONSTRAINT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-temporal-constraint@1"
)
EVIDENCE_REQUIREMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-evidence-requirement@1"
)
PLAN_ASSURANCE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/plan-assurance@1"


FormalPlanningValidationError = ContractValidationError


class ActorKind(str, Enum):
    SUPERVISOR = "supervisor"
    AGENT = "agent"
    HUMAN = "human"
    SERVICE = "service"
    PROVER = "prover"


class EventKind(str, Enum):
    ASSIGNED = "assigned"
    DELEGATED = "delegated"
    STARTED = "started"
    EXECUTED = "executed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EVIDENCE_PRODUCED = "evidence_produced"


class FluentValueType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    STRING = "string"
    SYMBOL = "symbol"


class EffectOperation(str, Enum):
    INITIATE = "initiate"
    TERMINATE = "terminate"
    ASSIGN = "assign"
    EMIT = "emit"


class NormKind(str, Enum):
    OBLIGATION = "obligation"
    PERMISSION = "permission"
    PROHIBITION = "prohibition"


class TemporalConstraintKind(str, Enum):
    DEPENDENCY_ORDER = "dependency_order"
    DEADLINE = "deadline"
    LIVENESS = "liveness"
    SAFETY = "safety"
    GOAL_SATISFACTION = "goal_satisfaction"


class EvidenceRequirementKind(str, Enum):
    TEST = "test"
    STATIC_ANALYSIS = "static_analysis"
    PLAN_CHECK = "plan_check"
    PLAN_CONFORMANCE = "plan_conformance"
    CODE_PROOF = "code_proof"
    REVIEW = "review"
    ARTIFACT = "artifact"


class RefinementMode(str, Enum):
    """Logical relationship asserted between a subgoal and its parent.

    ``SUFFICIENT`` records only the safe child-to-parent obligation.  The
    stronger bidirectional ``EQUIVALENT`` relationship must be selected
    explicitly and is never inferred from a legacy record.
    """

    SUFFICIENT = "sufficient"
    EQUIVALENT = "equivalent"

    # Readable aliases used by callers that name the relationship rather than
    # its short wire value.  Enum aliases retain one deterministic wire value.
    SUFFICIENT_REFINEMENT = "sufficient"
    EQUIVALENCE = "equivalent"


class PlanConsistencyLevel(str, Enum):
    UNCHECKED = "unchecked"
    INCONCLUSIVE = "inconclusive"
    COUNTEREXAMPLE = "counterexample"
    BOUNDED_CONSISTENT = "bounded_consistent"
    KERNEL_VERIFIED = "kernel_verified"


class PlanConformanceLevel(str, Enum):
    UNOBSERVED = "unobserved"
    INCONCLUSIVE = "inconclusive"
    VIOLATED = "violated"
    BOUNDED_CONFORMANT = "bounded_conformant"
    ATTESTED = "attested"


T = TypeVar("T")
E = TypeVar("E", bound=Enum)


def _strings(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
) -> Tuple[str, ...]:
    if value is None:
        items: Iterable[Any] = ()
    elif isinstance(value, str):
        items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ContractValidationError("%s must be a sequence of strings" % field_name)
    result: List[str] = []
    for item in items:
        normalized = _text(item, field_name=field_name, required=True)
        if normalized not in result:
            result.append(normalized)
    if required and not result:
        raise ContractValidationError("%s must not be empty" % field_name)
    return tuple(result if preserve_order else sorted(result))


def _optional_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(
            "%s must be a non-negative integer or null" % field_name
        )
    return value


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContractValidationError(
            "unsupported schema %r; expected %s" % (supplied, expected)
        )


def _claimed_identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id") or payload.get("identity")
    if claimed and claimed != actual:
        raise ContractValidationError(
            "%s content identity does not match payload" % noun
        )


class PlanningContract(CanonicalContract):
    """Canonical contract whose schema version belongs to the planning layer."""

    @property
    def schema_version(self) -> int:
        return FORMAL_PLANNING_CONTRACT_VERSION

    def _versioned(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "contract_version": FORMAL_PLANNING_CONTRACT_VERSION,
            **dict(payload),
        }


@dataclass(frozen=True)
class Actor(PlanningContract):
    SCHEMA: ClassVar[str] = ACTOR_SCHEMA

    actor_id: str
    kind: ActorKind = ActorKind.AGENT
    principal_id: str = ""
    capabilities: Tuple[str, ...] = ()
    authority_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "actor_id", _text(self.actor_id, field_name="actor_id", required=True)
        )
        object.__setattr__(
            self, "principal_id", _text(self.principal_id, field_name="principal_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, ActorKind, field_name="kind"))
        object.__setattr__(
            self, "capabilities", _strings(self.capabilities, field_name="capabilities")
        )
        object.__setattr__(
            self,
            "authority_ids",
            _strings(self.authority_ids, field_name="authority_ids"),
        )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "actor_id": self.actor_id,
                "kind": self.kind,
                "principal_id": self.principal_id,
                "capabilities": self.capabilities,
                "authority_ids": self.authority_ids,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Actor":
        _schema(payload, cls.SCHEMA)
        result = cls(
            actor_id=payload.get("actor_id", ""),
            kind=payload.get("kind", ActorKind.AGENT),
            principal_id=payload.get("principal_id", ""),
            capabilities=tuple(payload.get("capabilities") or ()),
            authority_ids=tuple(payload.get("authority_ids") or ()),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "actor")
        return result


@dataclass(frozen=True)
class Goal(PlanningContract):
    SCHEMA: ClassVar[str] = GOAL_SCHEMA

    goal_id: str
    owner_actor_id: str
    satisfaction_formula_id: str
    evidence_requirement_ids: Tuple[str, ...] = ()
    terminal_states: Tuple[str, ...] = ("satisfied",)
    source_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("goal_id", "owner_actor_id", "satisfaction_formula_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(
            self,
            "evidence_requirement_ids",
            _strings(
                self.evidence_requirement_ids, field_name="evidence_requirement_ids"
            ),
        )
        object.__setattr__(
            self,
            "terminal_states",
            _strings(self.terminal_states, field_name="terminal_states", required=True),
        )
        object.__setattr__(
            self, "source_ids", _strings(self.source_ids, field_name="source_ids")
        )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "goal_id": self.goal_id,
                "owner_actor_id": self.owner_actor_id,
                "satisfaction_formula_id": self.satisfaction_formula_id,
                "evidence_requirement_ids": self.evidence_requirement_ids,
                "terminal_states": self.terminal_states,
                "source_ids": self.source_ids,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Goal":
        _schema(payload, cls.SCHEMA)
        result = cls(
            goal_id=payload.get("goal_id", ""),
            owner_actor_id=payload.get("owner_actor_id", ""),
            satisfaction_formula_id=payload.get("satisfaction_formula_id", ""),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            terminal_states=tuple(payload.get("terminal_states") or ("satisfied",)),
            source_ids=tuple(payload.get("source_ids") or ()),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "goal")
        return result


@dataclass(frozen=True)
class Subgoal(PlanningContract):
    SCHEMA: ClassVar[str] = SUBGOAL_SCHEMA

    subgoal_id: str
    goal_id: str
    satisfaction_formula_id: str
    parent_id: str = ""
    refinement_mode: RefinementMode = RefinementMode.SUFFICIENT
    depends_on: Tuple[str, ...] = ()
    evidence_requirement_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("subgoal_id", "goal_id", "satisfaction_formula_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        parent_id = _text(self.parent_id, field_name="parent_id")
        # Version-1 records only carried goal_id and therefore denoted a
        # direct child of that goal.  Preserve that meaning without upgrading
        # it to equivalence.
        object.__setattr__(self, "parent_id", parent_id or self.goal_id)
        object.__setattr__(
            self,
            "refinement_mode",
            _enum(self.refinement_mode, RefinementMode, field_name="refinement_mode"),
        )
        object.__setattr__(
            self, "depends_on", _strings(self.depends_on, field_name="depends_on")
        )
        object.__setattr__(
            self,
            "evidence_requirement_ids",
            _strings(
                self.evidence_requirement_ids, field_name="evidence_requirement_ids"
            ),
        )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )
        if self.subgoal_id in self.depends_on:
            raise ContractValidationError("a subgoal cannot depend on itself")
        if self.subgoal_id == self.parent_id:
            raise ContractValidationError("a subgoal cannot be its own parent")

    @property
    def parent_goal_id(self) -> str:
        """Compatibility view of the root goal for this refinement."""

        return self.goal_id

    @property
    def is_equivalent_refinement(self) -> bool:
        return self.refinement_mode is RefinementMode.EQUIVALENT

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "subgoal_id": self.subgoal_id,
                "goal_id": self.goal_id,
                "parent_id": self.parent_id,
                "refinement_mode": self.refinement_mode,
                "satisfaction_formula_id": self.satisfaction_formula_id,
                "depends_on": self.depends_on,
                "evidence_requirement_ids": self.evidence_requirement_ids,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Subgoal":
        _schema(payload, cls.SCHEMA)
        result = cls(
            subgoal_id=payload.get("subgoal_id", ""),
            goal_id=payload.get("goal_id", ""),
            parent_id=payload.get("parent_id", ""),
            refinement_mode=payload.get(
                "refinement_mode", RefinementMode.SUFFICIENT
            ),
            satisfaction_formula_id=payload.get("satisfaction_formula_id", ""),
            depends_on=tuple(payload.get("depends_on") or ()),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "subgoal")
        return result


@dataclass(frozen=True)
class PlanTask(PlanningContract):
    SCHEMA: ClassVar[str] = PLAN_TASK_SCHEMA

    task_id: str
    goal_id: str
    subgoal_id: str = ""
    actor_ids: Tuple[str, ...] = ()
    depends_on: Tuple[str, ...] = ()
    precondition_ids: Tuple[str, ...] = ()
    effect_ids: Tuple[str, ...] = ()
    event_ids: Tuple[str, ...] = ()
    evidence_requirement_ids: Tuple[str, ...] = ()
    terminal_states: Tuple[str, ...] = ("completed", "failed", "cancelled")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "goal_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(
            self, "subgoal_id", _text(self.subgoal_id, field_name="subgoal_id")
        )
        for name in (
            "actor_ids",
            "depends_on",
            "precondition_ids",
            "effect_ids",
            "event_ids",
            "evidence_requirement_ids",
            "terminal_states",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    required=name in ("actor_ids", "terminal_states"),
                ),
            )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )
        if self.task_id in self.depends_on:
            raise ContractValidationError("a task cannot depend on itself")

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "task_id": self.task_id,
                "goal_id": self.goal_id,
                "subgoal_id": self.subgoal_id,
                "actor_ids": self.actor_ids,
                "depends_on": self.depends_on,
                "precondition_ids": self.precondition_ids,
                "effect_ids": self.effect_ids,
                "event_ids": self.event_ids,
                "evidence_requirement_ids": self.evidence_requirement_ids,
                "terminal_states": self.terminal_states,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanTask":
        _schema(payload, cls.SCHEMA)
        result = cls(
            task_id=payload.get("task_id", ""),
            goal_id=payload.get("goal_id", ""),
            subgoal_id=payload.get("subgoal_id", ""),
            actor_ids=tuple(payload.get("actor_ids") or ()),
            depends_on=tuple(payload.get("depends_on") or ()),
            precondition_ids=tuple(payload.get("precondition_ids") or ()),
            effect_ids=tuple(payload.get("effect_ids") or ()),
            event_ids=tuple(payload.get("event_ids") or ()),
            evidence_requirement_ids=tuple(
                payload.get("evidence_requirement_ids") or ()
            ),
            terminal_states=tuple(
                payload.get("terminal_states") or ("completed", "failed", "cancelled")
            ),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "task")
        return result


Task = PlanTask


@dataclass(frozen=True)
class PlanEvent(PlanningContract):
    SCHEMA: ClassVar[str] = PLAN_EVENT_SCHEMA

    event_id: str
    kind: EventKind
    actor_id: str
    task_id: str
    logical_time: int
    provenance_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_id", "actor_id", "task_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(self, "kind", _enum(self.kind, EventKind, field_name="kind"))
        logical_time = _optional_int(self.logical_time, field_name="logical_time")
        if logical_time is None:
            raise ContractValidationError("logical_time is required")
        object.__setattr__(self, "logical_time", logical_time)
        object.__setattr__(
            self,
            "provenance_ids",
            _strings(self.provenance_ids, field_name="provenance_ids"),
        )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "event_id": self.event_id,
                "kind": self.kind,
                "actor_id": self.actor_id,
                "task_id": self.task_id,
                "logical_time": self.logical_time,
                "provenance_ids": self.provenance_ids,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanEvent":
        _schema(payload, cls.SCHEMA)
        result = cls(
            event_id=payload.get("event_id", ""),
            kind=payload.get("kind", EventKind.EXECUTED),
            actor_id=payload.get("actor_id", ""),
            task_id=payload.get("task_id", ""),
            logical_time=payload.get("logical_time", -1),
            provenance_ids=tuple(payload.get("provenance_ids") or ()),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "event")
        return result


Event = PlanEvent


@dataclass(frozen=True)
class Fluent(PlanningContract):
    SCHEMA: ClassVar[str] = FLUENT_SCHEMA

    fluent_id: str
    value_type: FluentValueType
    initial_value: Any = None
    inertial: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fluent_id",
            _text(self.fluent_id, field_name="fluent_id", required=True),
        )
        object.__setattr__(
            self,
            "value_type",
            _enum(self.value_type, FluentValueType, field_name="value_type"),
        )
        value = _canonical_value(self.initial_value)
        if (
            self.value_type is FluentValueType.BOOLEAN
            and value is not None
            and not isinstance(value, bool)
        ):
            raise ContractValidationError(
                "boolean fluent initial_value must be boolean or null"
            )
        if (
            self.value_type is FluentValueType.INTEGER
            and value is not None
            and (isinstance(value, bool) or not isinstance(value, int))
        ):
            raise ContractValidationError(
                "integer fluent initial_value must be integer or null"
            )
        if (
            self.value_type in (FluentValueType.STRING, FluentValueType.SYMBOL)
            and value is not None
            and not isinstance(value, str)
        ):
            raise ContractValidationError(
                "string/symbol fluent initial_value must be string or null"
            )
        object.__setattr__(self, "initial_value", value)
        if not isinstance(self.inertial, bool):
            raise ContractValidationError("inertial must be a boolean")
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "fluent_id": self.fluent_id,
                "value_type": self.value_type,
                "initial_value": self.initial_value,
                "inertial": self.inertial,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Fluent":
        _schema(payload, cls.SCHEMA)
        result = cls(
            fluent_id=payload.get("fluent_id", ""),
            value_type=payload.get("value_type", FluentValueType.SYMBOL),
            initial_value=payload.get("initial_value"),
            inertial=payload.get("inertial", True),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "fluent")
        return result


@dataclass(frozen=True)
class Precondition(PlanningContract):
    SCHEMA: ClassVar[str] = PRECONDITION_SCHEMA

    precondition_id: str
    formula_id: str
    task_id: str
    event_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("precondition_id", "formula_id", "task_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(
            self, "event_id", _text(self.event_id, field_name="event_id")
        )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "precondition_id": self.precondition_id,
                "formula_id": self.formula_id,
                "task_id": self.task_id,
                "event_id": self.event_id,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Precondition":
        _schema(payload, cls.SCHEMA)
        result = cls(
            precondition_id=payload.get("precondition_id", ""),
            formula_id=payload.get("formula_id", ""),
            task_id=payload.get("task_id", ""),
            event_id=payload.get("event_id", ""),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "precondition")
        return result


@dataclass(frozen=True)
class Effect(PlanningContract):
    SCHEMA: ClassVar[str] = EFFECT_SCHEMA

    effect_id: str
    operation: EffectOperation
    task_id: str
    fluent_id: str = ""
    event_id: str = ""
    value: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("effect_id", "task_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(
            self,
            "operation",
            _enum(self.operation, EffectOperation, field_name="operation"),
        )
        object.__setattr__(
            self, "fluent_id", _text(self.fluent_id, field_name="fluent_id")
        )
        object.__setattr__(
            self, "event_id", _text(self.event_id, field_name="event_id")
        )
        if self.operation is EffectOperation.EMIT:
            if not self.event_id:
                raise ContractValidationError("emit effects require event_id")
        elif not self.fluent_id:
            raise ContractValidationError(
                "%s effects require fluent_id" % self.operation.value
            )
        object.__setattr__(self, "value", _canonical_value(self.value))
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "effect_id": self.effect_id,
                "operation": self.operation,
                "task_id": self.task_id,
                "fluent_id": self.fluent_id,
                "event_id": self.event_id,
                "value": self.value,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Effect":
        _schema(payload, cls.SCHEMA)
        result = cls(
            effect_id=payload.get("effect_id", ""),
            operation=payload.get("operation", EffectOperation.ASSIGN),
            task_id=payload.get("task_id", ""),
            fluent_id=payload.get("fluent_id", ""),
            event_id=payload.get("event_id", ""),
            value=payload.get("value"),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "effect")
        return result


@dataclass(frozen=True)
class Norm(PlanningContract):
    SCHEMA: ClassVar[str] = NORM_SCHEMA

    norm_id: str
    kind: NormKind
    bearer_actor_id: str
    action_id: str
    issuer_actor_id: str = ""
    activation_formula_id: str = ""
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("norm_id", "bearer_actor_id", "action_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name, required=True)
            )
        object.__setattr__(
            self,
            "issuer_actor_id",
            _text(self.issuer_actor_id, field_name="issuer_actor_id"),
        )
        object.__setattr__(
            self,
            "activation_formula_id",
            _text(self.activation_formula_id, field_name="activation_formula_id"),
        )
        object.__setattr__(self, "kind", _enum(self.kind, NormKind, field_name="kind"))
        object.__setattr__(
            self, "valid_from", _optional_int(self.valid_from, field_name="valid_from")
        )
        object.__setattr__(
            self,
            "valid_until",
            _optional_int(self.valid_until, field_name="valid_until"),
        )
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_from > self.valid_until
        ):
            raise ContractValidationError("norm valid_from must not exceed valid_until")
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "norm_id": self.norm_id,
                "kind": self.kind,
                "bearer_actor_id": self.bearer_actor_id,
                "action_id": self.action_id,
                "issuer_actor_id": self.issuer_actor_id,
                "activation_formula_id": self.activation_formula_id,
                "valid_from": self.valid_from,
                "valid_until": self.valid_until,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Norm":
        _schema(payload, cls.SCHEMA)
        result = cls(
            norm_id=payload.get("norm_id", ""),
            kind=payload.get("kind", NormKind.OBLIGATION),
            bearer_actor_id=payload.get("bearer_actor_id", ""),
            action_id=payload.get("action_id", ""),
            issuer_actor_id=payload.get("issuer_actor_id", ""),
            activation_formula_id=payload.get("activation_formula_id", ""),
            valid_from=payload.get("valid_from"),
            valid_until=payload.get("valid_until"),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "norm")
        return result


@dataclass(frozen=True)
class TemporalConstraint(PlanningContract):
    SCHEMA: ClassVar[str] = TEMPORAL_CONSTRAINT_SCHEMA

    constraint_id: str
    kind: TemporalConstraintKind
    subject_ids: Tuple[str, ...]
    formula_id: str
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constraint_id",
            _text(self.constraint_id, field_name="constraint_id", required=True),
        )
        object.__setattr__(
            self,
            "formula_id",
            _text(self.formula_id, field_name="formula_id", required=True),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, TemporalConstraintKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "subject_ids",
            _strings(self.subject_ids, field_name="subject_ids", required=True),
        )
        object.__setattr__(
            self,
            "lower_bound",
            _optional_int(self.lower_bound, field_name="lower_bound"),
        )
        object.__setattr__(
            self,
            "upper_bound",
            _optional_int(self.upper_bound, field_name="upper_bound"),
        )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ContractValidationError(
                "temporal lower_bound must not exceed upper_bound"
            )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "constraint_id": self.constraint_id,
                "kind": self.kind,
                "subject_ids": self.subject_ids,
                "formula_id": self.formula_id,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalConstraint":
        _schema(payload, cls.SCHEMA)
        result = cls(
            constraint_id=payload.get("constraint_id", ""),
            kind=payload.get("kind", TemporalConstraintKind.SAFETY),
            subject_ids=tuple(payload.get("subject_ids") or ()),
            formula_id=payload.get("formula_id", ""),
            lower_bound=payload.get("lower_bound"),
            upper_bound=payload.get("upper_bound"),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "temporal constraint")
        return result


@dataclass(frozen=True)
class EvidenceRequirement(PlanningContract):
    SCHEMA: ClassVar[str] = EVIDENCE_REQUIREMENT_SCHEMA

    requirement_id: str
    kind: EvidenceRequirementKind
    subject_ids: Tuple[str, ...]
    source_scope_ids: Tuple[str, ...] = ()
    minimum_code_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    freshness_seconds: Optional[int] = None
    fallback_check_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _text(self.requirement_id, field_name="requirement_id", required=True),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, EvidenceRequirementKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "subject_ids",
            _strings(self.subject_ids, field_name="subject_ids", required=True),
        )
        object.__setattr__(
            self,
            "source_scope_ids",
            _strings(self.source_scope_ids, field_name="source_scope_ids"),
        )
        object.__setattr__(
            self,
            "fallback_check_ids",
            _strings(self.fallback_check_ids, field_name="fallback_check_ids"),
        )
        object.__setattr__(
            self,
            "minimum_code_assurance",
            _enum(
                self.minimum_code_assurance,
                AssuranceLevel,
                field_name="minimum_code_assurance",
            ),
        )
        object.__setattr__(
            self,
            "freshness_seconds",
            _optional_int(self.freshness_seconds, field_name="freshness_seconds"),
        )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "requirement_id": self.requirement_id,
                "kind": self.kind,
                "subject_ids": self.subject_ids,
                "source_scope_ids": self.source_scope_ids,
                "minimum_code_assurance": self.minimum_code_assurance,
                "freshness_seconds": self.freshness_seconds,
                "fallback_check_ids": self.fallback_check_ids,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRequirement":
        _schema(payload, cls.SCHEMA)
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            kind=payload.get("kind", EvidenceRequirementKind.ARTIFACT),
            subject_ids=tuple(payload.get("subject_ids") or ()),
            source_scope_ids=tuple(payload.get("source_scope_ids") or ()),
            minimum_code_assurance=payload.get(
                "minimum_code_assurance", AssuranceLevel.UNVERIFIED
            ),
            freshness_seconds=payload.get("freshness_seconds"),
            fallback_check_ids=tuple(payload.get("fallback_check_ids") or ()),
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "evidence requirement")
        return result


def _records(
    values: Any,
    *,
    cls: Type[T],
    key: str,
    field_name: str,
    required: bool = False,
) -> Tuple[T, ...]:
    if values is None:
        values = ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ContractValidationError("%s must be a sequence" % field_name)
    result: List[T] = []
    for value in values:
        if isinstance(value, cls):
            record = value
        elif isinstance(value, Mapping):
            record = cls.from_dict(value)  # type: ignore[attr-defined]
        else:
            raise ContractValidationError(
                "%s must contain %s records" % (field_name, cls.__name__)
            )
        result.append(record)
    if required and not result:
        raise ContractValidationError("%s must not be empty" % field_name)
    result.sort(key=lambda item: getattr(item, key))
    keys = [getattr(item, key) for item in result]
    if len(keys) != len(set(keys)):
        raise ContractValidationError("%s identifiers must be unique" % field_name)
    return tuple(result)


def _formula_records(values: Any) -> Tuple[Formula, ...]:
    if values is None:
        values = ()
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray)
    ):
        raise ContractValidationError("formulas must be a sequence")
    result: List[Formula] = []
    for value in values:
        if isinstance(value, Formula):
            formula = value
        elif isinstance(value, Mapping):
            formula = Formula.from_dict(value)
        else:
            raise ContractValidationError(
                "formulas must contain reviewed Formula records"
            )
        result.append(formula)
    result.sort(key=lambda item: item.formula_id)
    identifiers = [item.formula_id for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise ContractValidationError("formula identifiers must be unique")
    return tuple(result)


def _subgoal_satisfaction_target(formula: Formula) -> Optional[str]:
    """Return the typed subgoal target, or ``None`` for another formula."""

    candidate = formula
    if formula.operator is FormulaOperator.GOAL_SATISFACTION:
        if len(formula.operands) != 1:
            return None
        candidate = formula.operands[0]
    if (
        candidate.operator is FormulaOperator.ATOM
        and candidate.predicate is ReviewedPredicate.SUBGOAL_SATISFIED
        and len(candidate.terms) == 1
        and candidate.terms[0].sort is TermSort.SUBGOAL
        and candidate.terms[0].kind is TermKind.CONSTANT
    ):
        return str(candidate.terms[0].value)
    return None


def _acyclic(
    nodes: Iterable[str], dependencies: Mapping[str, Tuple[str, ...]], noun: str
) -> None:
    node_set = set(nodes)
    visiting = set()
    visited = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ContractValidationError("%s dependencies must be acyclic" % noun)
        visiting.add(node)
        for dependency in dependencies.get(node, ()):
            if dependency not in node_set:
                raise ContractValidationError(
                    "%s %s has unknown dependency %s" % (noun, node, dependency)
                )
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(node_set):
        visit(node)


@dataclass(frozen=True)
class FormalWorkPlan(PlanningContract):
    """Complete canonical semantic model of intended supervisor work."""

    SCHEMA: ClassVar[str] = FORMAL_WORK_PLAN_SCHEMA

    vocabulary_profile_id: str
    vocabulary_version: int
    actors: Tuple[Actor, ...]
    goals: Tuple[Goal, ...]
    subgoals: Tuple[Subgoal, ...]
    tasks: Tuple[PlanTask, ...]
    events: Tuple[PlanEvent, ...]
    fluents: Tuple[Fluent, ...]
    preconditions: Tuple[Precondition, ...]
    effects: Tuple[Effect, ...]
    norms: Tuple[Norm, ...]
    temporal_constraints: Tuple[TemporalConstraint, ...]
    evidence_requirements: Tuple[EvidenceRequirement, ...]
    formulas: Tuple[Formula, ...] = ()
    source_ids: Tuple[str, ...] = ()
    repository_tree_id: str = ""
    trace_bound: int = 1
    abstraction_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vocabulary_profile_id",
            _text(
                self.vocabulary_profile_id,
                field_name="vocabulary_profile_id",
                required=True,
            ),
        )
        if (
            isinstance(self.vocabulary_version, bool)
            or not isinstance(self.vocabulary_version, int)
            or self.vocabulary_version <= 0
        ):
            raise ContractValidationError(
                "vocabulary_version must be a positive integer"
            )
        if (
            isinstance(self.trace_bound, bool)
            or not isinstance(self.trace_bound, int)
            or self.trace_bound <= 0
        ):
            raise ContractValidationError("trace_bound must be a positive integer")
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self,
            "source_ids",
            _strings(self.source_ids, field_name="source_ids", required=True),
        )
        object.__setattr__(
            self,
            "abstraction_ids",
            _strings(self.abstraction_ids, field_name="abstraction_ids"),
        )
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

        specifications = (
            ("actors", Actor, "actor_id", True),
            ("goals", Goal, "goal_id", True),
            ("subgoals", Subgoal, "subgoal_id", False),
            ("tasks", PlanTask, "task_id", True),
            ("events", PlanEvent, "event_id", False),
            ("fluents", Fluent, "fluent_id", False),
            ("preconditions", Precondition, "precondition_id", False),
            ("effects", Effect, "effect_id", False),
            ("norms", Norm, "norm_id", False),
            ("temporal_constraints", TemporalConstraint, "constraint_id", False),
            ("evidence_requirements", EvidenceRequirement, "requirement_id", False),
        )
        for name, cls, key, required in specifications:
            object.__setattr__(
                self,
                name,
                _records(
                    getattr(self, name),
                    cls=cls,
                    key=key,
                    field_name=name,
                    required=required,
                ),
            )
        object.__setattr__(self, "formulas", _formula_records(self.formulas))
        self._validate_references()

    def _validate_references(self) -> None:
        actor_ids = {item.actor_id for item in self.actors}
        goal_ids = {item.goal_id for item in self.goals}
        subgoal_ids = {item.subgoal_id for item in self.subgoals}
        task_ids = {item.task_id for item in self.tasks}
        event_ids = {item.event_id for item in self.events}
        fluent_ids = {item.fluent_id for item in self.fluents}
        precondition_ids = {item.precondition_id for item in self.preconditions}
        effect_ids = {item.effect_id for item in self.effects}
        requirement_ids = {item.requirement_id for item in self.evidence_requirements}
        formulas_by_id = {item.formula_id: item for item in self.formulas}
        all_subject_ids = (
            actor_ids | goal_ids | subgoal_ids | task_ids | event_ids | fluent_ids
        )
        if goal_ids & subgoal_ids:
            raise ContractValidationError(
                "goal and subgoal identifiers must occupy distinct typed namespaces"
            )
        if self.subgoals and self.vocabulary_version != LOGIC_VOCABULARY_VERSION:
            raise ContractValidationError(
                "typed subgoals require reviewed vocabulary version %s"
                % LOGIC_VOCABULARY_VERSION
            )

        def require(value: str, known: set, context: str) -> None:
            if value and value not in known:
                raise ContractValidationError(
                    "%s references unknown id %s" % (context, value)
                )

        for goal in self.goals:
            require(goal.owner_actor_id, actor_ids, "goal %s" % goal.goal_id)
            for value in goal.evidence_requirement_ids:
                require(value, requirement_ids, "goal %s" % goal.goal_id)
        for subgoal in self.subgoals:
            require(subgoal.goal_id, goal_ids, "subgoal %s" % subgoal.subgoal_id)
            require(
                subgoal.parent_id,
                goal_ids | subgoal_ids,
                "subgoal %s parent" % subgoal.subgoal_id,
            )
            if subgoal.parent_id in goal_ids and subgoal.parent_id != subgoal.goal_id:
                raise ContractValidationError(
                    "subgoal %s parent goal does not match goal_id"
                    % subgoal.subgoal_id
                )
            if subgoal.parent_id in subgoal_ids:
                parent = next(
                    item
                    for item in self.subgoals
                    if item.subgoal_id == subgoal.parent_id
                )
                if parent.goal_id != subgoal.goal_id:
                    raise ContractValidationError(
                        "subgoal %s parent belongs to a different root goal"
                        % subgoal.subgoal_id
                    )
            for value in subgoal.evidence_requirement_ids:
                require(value, requirement_ids, "subgoal %s" % subgoal.subgoal_id)
            formula = formulas_by_id.get(subgoal.satisfaction_formula_id)
            if formula is None:
                raise ContractValidationError(
                    "subgoal %s references unknown satisfaction formula %s"
                    % (subgoal.subgoal_id, subgoal.satisfaction_formula_id)
                )
            target = _subgoal_satisfaction_target(formula)
            if target != subgoal.subgoal_id:
                raise ContractValidationError(
                    "subgoal %s satisfaction formula must use the reviewed "
                    "subgoal_satisfied predicate for that subgoal"
                    % subgoal.subgoal_id
                )
        _acyclic(
            subgoal_ids,
            {item.subgoal_id: item.depends_on for item in self.subgoals},
            "subgoal",
        )
        _acyclic(
            subgoal_ids,
            {
                item.subgoal_id: tuple(
                    sorted(
                        {
                            *item.depends_on,
                            *(
                                (item.parent_id,)
                                if item.parent_id in subgoal_ids
                                else ()
                            ),
                        }
                    )
                )
                for item in self.subgoals
            },
            "subgoal parent/dependency graph",
        )
        for task in self.tasks:
            require(task.goal_id, goal_ids, "task %s" % task.task_id)
            require(task.subgoal_id, subgoal_ids, "task %s" % task.task_id)
            if task.subgoal_id:
                subgoal = next(
                    item
                    for item in self.subgoals
                    if item.subgoal_id == task.subgoal_id
                )
                if task.goal_id != subgoal.goal_id:
                    raise ContractValidationError(
                        "task %s and subgoal %s belong to different goals"
                        % (task.task_id, task.subgoal_id)
                    )
            for value in task.actor_ids:
                require(value, actor_ids, "task %s" % task.task_id)
            for value in task.precondition_ids:
                require(value, precondition_ids, "task %s" % task.task_id)
            for value in task.effect_ids:
                require(value, effect_ids, "task %s" % task.task_id)
            for value in task.event_ids:
                require(value, event_ids, "task %s" % task.task_id)
            for value in task.evidence_requirement_ids:
                require(value, requirement_ids, "task %s" % task.task_id)
        _acyclic(
            task_ids, {item.task_id: item.depends_on for item in self.tasks}, "task"
        )
        for event in self.events:
            require(event.actor_id, actor_ids, "event %s" % event.event_id)
            require(event.task_id, task_ids, "event %s" % event.event_id)
            if event.logical_time > self.trace_bound:
                raise ContractValidationError(
                    "event %s is outside trace_bound" % event.event_id
                )
        for precondition in self.preconditions:
            require(
                precondition.task_id,
                task_ids,
                "precondition %s" % precondition.precondition_id,
            )
            require(
                precondition.event_id,
                event_ids,
                "precondition %s" % precondition.precondition_id,
            )
        for effect in self.effects:
            require(effect.task_id, task_ids, "effect %s" % effect.effect_id)
            require(effect.event_id, event_ids, "effect %s" % effect.effect_id)
            require(effect.fluent_id, fluent_ids, "effect %s" % effect.effect_id)
            if effect.fluent_id:
                fluent = next(
                    item for item in self.fluents if item.fluent_id == effect.fluent_id
                )
                value = effect.value
                valid_value = (
                    value is None
                    or (
                        fluent.value_type is FluentValueType.BOOLEAN
                        and isinstance(value, bool)
                    )
                    or (
                        fluent.value_type is FluentValueType.INTEGER
                        and isinstance(value, int)
                        and not isinstance(value, bool)
                    )
                    or (
                        fluent.value_type
                        in (FluentValueType.STRING, FluentValueType.SYMBOL)
                        and isinstance(value, str)
                    )
                )
                if effect.operation is EffectOperation.ASSIGN and not valid_value:
                    raise ContractValidationError(
                        "effect %s value does not match fluent %s type"
                        % (effect.effect_id, effect.fluent_id)
                    )
        for norm in self.norms:
            require(norm.bearer_actor_id, actor_ids, "norm %s" % norm.norm_id)
            require(norm.issuer_actor_id, actor_ids, "norm %s" % norm.norm_id)
            require(norm.action_id, task_ids | event_ids, "norm %s" % norm.norm_id)
        for constraint in self.temporal_constraints:
            for value in constraint.subject_ids:
                require(
                    value, all_subject_ids, "constraint %s" % constraint.constraint_id
                )
            if (
                constraint.upper_bound is not None
                and constraint.upper_bound > self.trace_bound
            ):
                raise ContractValidationError(
                    "constraint %s exceeds trace_bound" % constraint.constraint_id
                )
        for requirement in self.evidence_requirements:
            for value in requirement.subject_ids:
                require(
                    value,
                    goal_ids | subgoal_ids | task_ids,
                    "evidence requirement %s" % requirement.requirement_id,
                )

        if self.formulas:
            referenced_formula_ids = {
                *(item.satisfaction_formula_id for item in self.goals),
                *(item.satisfaction_formula_id for item in self.subgoals),
                *(item.formula_id for item in self.preconditions),
                *(item.formula_id for item in self.temporal_constraints),
                *(
                    item.activation_formula_id
                    for item in self.norms
                    if item.activation_formula_id
                ),
            }
            unknown_formula_ids = referenced_formula_ids.difference(formulas_by_id)
            if unknown_formula_ids:
                raise ContractValidationError(
                    "plan references unknown formula %s"
                    % sorted(unknown_formula_ids)[0]
                )

        assignments: Dict[Tuple[str, str], Any] = {}
        for effect in self.effects:
            if effect.operation is EffectOperation.ASSIGN:
                transition = effect.event_id or effect.task_id
                key = (transition, effect.fluent_id)
                if key in assignments and assignments[key] != effect.value:
                    raise ContractValidationError(
                        "conflicting effects assign %s at %s"
                        % (effect.fluent_id, transition)
                    )
                assignments[key] = effect.value

    @property
    def plan_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "vocabulary_profile_id": self.vocabulary_profile_id,
            "vocabulary_version": self.vocabulary_version,
            "source_ids": self.source_ids,
            "repository_tree_id": self.repository_tree_id,
            "trace_bound": self.trace_bound,
            "actors": self.actors,
            "goals": self.goals,
            "subgoals": self.subgoals,
            "tasks": self.tasks,
            "events": self.events,
            "fluents": self.fluents,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "norms": self.norms,
            "temporal_constraints": self.temporal_constraints,
            "evidence_requirements": self.evidence_requirements,
            "abstraction_ids": self.abstraction_ids,
            "metadata": self.metadata,
        }
        # Omitting an empty registry preserves the canonical identity of
        # version-1 plans, which had no first-class formula collection.
        if self.formulas:
            payload["formulas"] = self.formulas
        return self._versioned(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalWorkPlan":
        _schema(payload, cls.SCHEMA)
        result = cls(
            vocabulary_profile_id=payload.get("vocabulary_profile_id", ""),
            vocabulary_version=payload.get("vocabulary_version", 0),
            source_ids=tuple(payload.get("source_ids") or ()),
            repository_tree_id=payload.get("repository_tree_id", ""),
            trace_bound=payload.get("trace_bound", 0),
            actors=tuple(payload.get("actors") or ()),
            goals=tuple(payload.get("goals") or ()),
            subgoals=tuple(payload.get("subgoals") or ()),
            tasks=tuple(payload.get("tasks") or ()),
            events=tuple(payload.get("events") or ()),
            fluents=tuple(payload.get("fluents") or ()),
            preconditions=tuple(payload.get("preconditions") or ()),
            effects=tuple(payload.get("effects") or ()),
            norms=tuple(payload.get("norms") or ()),
            temporal_constraints=tuple(payload.get("temporal_constraints") or ()),
            evidence_requirements=tuple(payload.get("evidence_requirements") or ()),
            formulas=tuple(
                payload.get("formulas")
                or payload.get("reviewed_formulas")
                or ()
            ),
            abstraction_ids=tuple(payload.get("abstraction_ids") or ()),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("plan_id") or payload.get("content_id")
        if claimed and claimed != result.plan_id:
            raise ContractValidationError(
                "formal work-plan identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class PlanAssurance(PlanningContract):
    """Independent assurance dimensions; none is inferred from another."""

    SCHEMA: ClassVar[str] = PLAN_ASSURANCE_SCHEMA

    plan_id: str
    consistency: PlanConsistencyLevel = PlanConsistencyLevel.UNCHECKED
    conformance: PlanConformanceLevel = PlanConformanceLevel.UNOBSERVED
    generated_code_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    consistency_receipt_ids: Tuple[str, ...] = ()
    conformance_receipt_ids: Tuple[str, ...] = ()
    code_proof_receipt_ids: Tuple[str, ...] = ()
    bounds: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "plan_id", _text(self.plan_id, field_name="plan_id", required=True)
        )
        object.__setattr__(
            self,
            "consistency",
            _enum(self.consistency, PlanConsistencyLevel, field_name="consistency"),
        )
        object.__setattr__(
            self,
            "conformance",
            _enum(self.conformance, PlanConformanceLevel, field_name="conformance"),
        )
        object.__setattr__(
            self,
            "generated_code_assurance",
            _enum(
                self.generated_code_assurance,
                AssuranceLevel,
                field_name="generated_code_assurance",
            ),
        )
        for name in (
            "consistency_receipt_ids",
            "conformance_receipt_ids",
            "code_proof_receipt_ids",
        ):
            object.__setattr__(
                self, name, _strings(getattr(self, name), field_name=name)
            )
        if (
            self.consistency is not PlanConsistencyLevel.UNCHECKED
            and not self.consistency_receipt_ids
        ):
            raise ContractValidationError(
                "checked plan consistency requires a consistency receipt"
            )
        if (
            self.conformance is not PlanConformanceLevel.UNOBSERVED
            and not self.conformance_receipt_ids
        ):
            raise ContractValidationError(
                "observed plan conformance requires a conformance receipt"
            )
        if (
            self.generated_code_assurance is not AssuranceLevel.UNVERIFIED
            and not self.code_proof_receipt_ids
        ):
            raise ContractValidationError(
                "generated-code assurance requires an independent code-proof receipt"
            )
        planning_receipts = set(self.consistency_receipt_ids) | set(
            self.conformance_receipt_ids
        )
        if planning_receipts & set(self.code_proof_receipt_ids):
            raise ContractValidationError(
                "plan receipts cannot be promoted or reused as code-proof receipts"
            )
        object.__setattr__(self, "bounds", _mapping(self.bounds, field_name="bounds"))
        object.__setattr__(
            self, "metadata", _mapping(self.metadata, field_name="metadata")
        )

    @property
    def plan_consistency(self) -> PlanConsistencyLevel:
        return self.consistency

    @property
    def plan_conformance(self) -> PlanConformanceLevel:
        return self.conformance

    @property
    def code_assurance(self) -> AssuranceLevel:
        return self.generated_code_assurance

    def _payload(self) -> Dict[str, Any]:
        return self._versioned(
            {
                "plan_id": self.plan_id,
                "consistency": self.consistency,
                "conformance": self.conformance,
                "generated_code_assurance": self.generated_code_assurance,
                "consistency_receipt_ids": self.consistency_receipt_ids,
                "conformance_receipt_ids": self.conformance_receipt_ids,
                "code_proof_receipt_ids": self.code_proof_receipt_ids,
                "bounds": self.bounds,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanAssurance":
        _schema(payload, cls.SCHEMA)
        result = cls(
            plan_id=payload.get("plan_id", ""),
            consistency=payload.get("consistency", PlanConsistencyLevel.UNCHECKED),
            conformance=payload.get("conformance", PlanConformanceLevel.UNOBSERVED),
            generated_code_assurance=payload.get(
                "generated_code_assurance", AssuranceLevel.UNVERIFIED
            ),
            consistency_receipt_ids=tuple(payload.get("consistency_receipt_ids") or ()),
            conformance_receipt_ids=tuple(payload.get("conformance_receipt_ids") or ()),
            code_proof_receipt_ids=tuple(payload.get("code_proof_receipt_ids") or ()),
            bounds=payload.get("bounds") or {},
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "plan assurance")
        return result


FormalPlanAssurance = PlanAssurance


__all__ = [
    "ACTOR_SCHEMA",
    "EFFECT_SCHEMA",
    "EVIDENCE_REQUIREMENT_SCHEMA",
    "FLUENT_SCHEMA",
    "FORMAL_PLANNING_CONTRACT_VERSION",
    "FORMAL_WORK_PLAN_SCHEMA",
    "GOAL_SCHEMA",
    "NORM_SCHEMA",
    "PLAN_ASSURANCE_SCHEMA",
    "PLAN_EVENT_SCHEMA",
    "PLAN_TASK_SCHEMA",
    "PLANNING_CONTRACT_VERSION",
    "PRECONDITION_SCHEMA",
    "SCHEMA_VERSION",
    "SUBGOAL_SCHEMA",
    "TEMPORAL_CONSTRAINT_SCHEMA",
    "Actor",
    "ActorKind",
    "Effect",
    "EffectOperation",
    "Event",
    "EventKind",
    "EvidenceRequirement",
    "EvidenceRequirementKind",
    "Fluent",
    "FluentValueType",
    "FormalPlanAssurance",
    "FormalPlanningValidationError",
    "FormalWorkPlan",
    "Goal",
    "Norm",
    "NormKind",
    "PlanAssurance",
    "PlanConformanceLevel",
    "PlanConsistencyLevel",
    "PlanEvent",
    "PlanTask",
    "PlanningContract",
    "Precondition",
    "RefinementMode",
    "Subgoal",
    "Task",
    "TemporalConstraint",
    "TemporalConstraintKind",
]
