"""Finite CASF specifications and bounded model-check receipts.

The existing supervisor TLA+ translator remains the only translator used by
this module.  CASF contributes six closed transition schemas plus a small
hermetic explorer for federation-specific invariants which the generic model
does not encode (for example budget conservation and causal-parent order).

Hermetic and external results are bounded evidence only.  An absent or
uncertified TLC/Apalache entry is recorded as ``unavailable`` or ``not_run``;
it can never be reported as a pass.  Neither model construction nor checking
opens the control database, invokes Quack, changes board state, or mints
policy, admission, merge, or completion authority.
"""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import content_identity
from ...proof.prover_matrix_registry import (
    CommandResult,
    ProverMatrixEntry,
    ProverMatrixSnapshot,
)
from ...self_improvement.supervisor_state_model import (
    LIVENESS_PROPERTIES,
    SAFETY_PROPERTIES,
    GeneratedSupervisorStateModel,
    ModelCheckBounds,
    ModelCheckerTool,
    ModelCheckReceipt,
    ModelCheckStatus,
    SupervisorStateModelChecker,
    SupervisorTransitionSchema,
    TransitionRule,
    generate_supervisor_state_model,
)
from ..contracts import FederationLifecycleState
from ..lifecycle import legal_transitions

CASF_FORMAL_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/formal-identity@1"
)
CASF_FORMAL_SUITE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/formal-suite@1"
)
CASF_FORMAL_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/formal-check-receipt@1"
)
CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/external-model-check-receipt@2"
)
CASF_FORMAL_TASK_ID: Final = "CASF-036"
MAX_HERMETIC_EXPLORED_STATES: Final = 200_000
_GIT_OID = re.compile(r"[0-9a-f]{40}")


class FederationFormalError(ValueError):
    """A formal-model input or identity is malformed."""


class FederationFormalProperty(str, Enum):
    EVENT_DELIVERY = "event_delivery"
    CLAIM_LEASE_FENCE = "claim_lease_fence"
    LIFECYCLE = "lifecycle"
    SHARD_TRANSFER = "shard_transfer"
    BUDGET_CONSERVATION = "budget_conservation"
    CAUSAL_PROPAGATION = "causal_propagation"


class AdversarialMutation(str, Enum):
    NONE = "none"
    DUPLICATE_EVENT_EFFECT = "duplicate_event_effect"
    STALE_FENCE_MUTATION = "stale_fence_mutation"
    ILLEGAL_LIFECYCLE_TRANSITION = "illegal_lifecycle_transition"
    DUAL_SHARD_OWNER = "dual_shard_owner"
    MINT_BUDGET = "mint_budget"
    ORPHAN_CAUSAL_PROPAGATION = "orphan_causal_propagation"


ADVERSARIAL_PROPERTY: Final[Mapping[AdversarialMutation, FederationFormalProperty]] = (
    MappingProxyType(
        {
            AdversarialMutation.DUPLICATE_EVENT_EFFECT: (FederationFormalProperty.EVENT_DELIVERY),
            AdversarialMutation.STALE_FENCE_MUTATION: (FederationFormalProperty.CLAIM_LEASE_FENCE),
            AdversarialMutation.ILLEGAL_LIFECYCLE_TRANSITION: (FederationFormalProperty.LIFECYCLE),
            AdversarialMutation.DUAL_SHARD_OWNER: (FederationFormalProperty.SHARD_TRANSFER),
            AdversarialMutation.MINT_BUDGET: (FederationFormalProperty.BUDGET_CONSERVATION),
            AdversarialMutation.ORPHAN_CAUSAL_PROPAGATION: (
                FederationFormalProperty.CAUSAL_PROPAGATION
            ),
        }
    )
)


class HermeticCheckStatus(str, Enum):
    PASSED = "passed"
    COUNTEREXAMPLE = "counterexample"
    INCONCLUSIVE = "inconclusive"


class ExternalCheckStatus(str, Enum):
    PASSED = "passed"
    COUNTEREXAMPLE = "counterexample"
    UNKNOWN = "unknown"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"
    NOT_RUN = "not_run"
    ERROR = "error"


class ExternalModelInvariant(str, Enum):
    """Closed generic invariants which an external checker actually checks."""

    SUPERVISOR_SAFETY = "Safety"


def _identity_token(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise FederationFormalError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise FederationFormalError(f"{name} must not be empty")
    if text != value or any(character.isspace() for character in text):
        raise FederationFormalError(f"{name} must be an exact whitespace-free identity")
    if len(text.encode("utf-8")) > 1_024:
        raise FederationFormalError(f"{name} exceeds the identity bound")
    return text


def _bounded_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FederationFormalError(f"{name} must be nonempty exact text")
    if len(value.encode("utf-8")) > 4_096:
        raise FederationFormalError(f"{name} exceeds the text bound")
    return value


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FederationFormalError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise FederationFormalError(f"{name} must be {qualifier}")
    return value


def _identity_tuple(
    values: Sequence[str],
    name: str,
    *,
    minimum: int = 0,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FederationFormalError(f"{name} must be an identity array")
    result = tuple(_identity_token(item, name) for item in values)
    if len(result) < minimum:
        raise FederationFormalError(f"{name} requires at least {minimum} identities")
    if len(result) != len(set(result)):
        raise FederationFormalError(f"{name} contains duplicate identities")
    return result if preserve_order else tuple(sorted(result))


def _typed_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}:{content_identity(dict(payload))}"


@dataclass(frozen=True)
class FederationFormalIdentity:
    """Exact source, authority, execution, lease, and fence binding."""

    SCHEMA: ClassVar[str] = CASF_FORMAL_IDENTITY_SCHEMA

    source_revision: str
    source_tree: str
    state_schema: str
    generation_id: str
    policy_id: str
    policy_revision: int
    capability_ids: tuple[str, ...]
    federation_id: str
    supervisor_ids: tuple[str, ...]
    task_id: str
    attempt_id: str
    lease_id: str
    fencing_epoch: int
    assignment_revision: int
    worktree_id: str
    schema: str = CASF_FORMAL_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != self.SCHEMA:
            raise FederationFormalError("unsupported federation formal identity schema")
        for name in ("source_revision", "source_tree"):
            value = _identity_token(getattr(self, name), name)
            if _GIT_OID.fullmatch(value) is None:
                raise FederationFormalError(f"{name} must be a lowercase 40-hex Git object id")
        for name in (
            "state_schema",
            "generation_id",
            "policy_id",
            "federation_id",
            "task_id",
            "attempt_id",
            "lease_id",
            "worktree_id",
        ):
            object.__setattr__(self, name, _identity_token(getattr(self, name), name))
        if self.task_id != CASF_FORMAL_TASK_ID:
            raise FederationFormalError("task_id must be the exact CASF-036 identity")
        object.__setattr__(
            self,
            "capability_ids",
            _identity_tuple(self.capability_ids, "capability_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "supervisor_ids",
            _identity_tuple(self.supervisor_ids, "supervisor_ids", minimum=2),
        )
        _positive_integer(self.policy_revision, "policy_revision")
        _positive_integer(self.fencing_epoch, "fencing_epoch")
        _positive_integer(self.assignment_revision, "assignment_revision")

    @property
    def identity(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "source_revision": self.source_revision,
            "source_tree": self.source_tree,
            "state_schema": self.state_schema,
            "generation_id": self.generation_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "capability_ids": list(self.capability_ids),
            "federation_id": self.federation_id,
            "supervisor_ids": list(self.supervisor_ids),
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "assignment_revision": self.assignment_revision,
            "worktree_id": self.worktree_id,
        }
        if include_identity:
            payload["identity"] = self.identity
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FederationFormalIdentity:
        if not isinstance(value, Mapping):
            raise FederationFormalError("formal identity must be an object")
        # This is a wire boundary, rather than a convenience constructor: a
        # silently accepted field could make a signed/formal identity appear
        # to bind more authority than the model actually consumes.  Keep the
        # optional content identity separate from the normative closed shape.
        allowed = {
            "schema",
            "source_revision",
            "source_tree",
            "state_schema",
            "generation_id",
            "policy_id",
            "policy_revision",
            "capability_ids",
            "federation_id",
            "supervisor_ids",
            "task_id",
            "attempt_id",
            "lease_id",
            "fencing_epoch",
            "assignment_revision",
            "worktree_id",
            "identity",
        }
        unknown = set(value) - allowed
        if unknown:
            raise FederationFormalError(
                "formal identity has unknown fields: "
                + repr(sorted(str(item) for item in unknown))
            )
        result = cls(
            source_revision=value.get("source_revision", ""),
            source_tree=value.get("source_tree", ""),
            state_schema=value.get("state_schema", ""),
            generation_id=value.get("generation_id", ""),
            policy_id=value.get("policy_id", ""),
            policy_revision=value.get("policy_revision", 0),
            capability_ids=tuple(value.get("capability_ids") or ()),
            federation_id=value.get("federation_id", ""),
            supervisor_ids=tuple(value.get("supervisor_ids") or ()),
            task_id=value.get("task_id", ""),
            attempt_id=value.get("attempt_id", ""),
            lease_id=value.get("lease_id", ""),
            fencing_epoch=value.get("fencing_epoch", 0),
            assignment_revision=value.get("assignment_revision", 0),
            worktree_id=value.get("worktree_id", ""),
            schema=value.get("schema", ""),
        )
        claimed = value.get("identity")
        if claimed is not None and claimed != result.identity:
            raise FederationFormalError("claimed formal identity does not match its content")
        return result


@dataclass(frozen=True)
class FederationModelState:
    """One immutable state shared by the six small finite specifications."""

    stage: str
    logical_time: int = 0
    delivery_attempts: int = 0
    applied_event_ids: tuple[str, ...] = ()
    claim_owners: tuple[str, ...] = ()
    lease_owner: str = ""
    lease_expires_at: int = 0
    claim_fence: int = 0
    current_fence: int = 1
    last_mutation_fence: int = 1
    shard_owners: tuple[str, ...] = ()
    claims_open: bool = True
    source_budget: int = 0
    target_budget: int = 0
    consumed_budget: int = 0
    initial_budget: int = 0
    observed_causal_ids: tuple[str, ...] = ()
    propagated_causal_ids: tuple[str, ...] = ()
    active_attempts: int = 0
    active_effects: int = 0
    preserved_identity_refs: tuple[str, ...] = ()
    last_lifecycle_source: str = ""
    last_lifecycle_target: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _identity_token(self.stage, "stage"))
        for name in (
            "logical_time",
            "delivery_attempts",
            "lease_expires_at",
            "claim_fence",
            "source_budget",
            "target_budget",
            "consumed_budget",
            "initial_budget",
            "active_attempts",
            "active_effects",
        ):
            _positive_integer(getattr(self, name), name, allow_zero=True)
        for name in ("current_fence", "last_mutation_fence"):
            _positive_integer(getattr(self, name), name)
        for name in (
            "applied_event_ids",
            "observed_causal_ids",
            "propagated_causal_ids",
            "preserved_identity_refs",
        ):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise FederationFormalError(f"{name} must be an identity array")
            object.__setattr__(
                self,
                name,
                tuple(_identity_token(item, name) for item in values),
            )
        for name in ("claim_owners", "shard_owners"):
            object.__setattr__(
                self,
                name,
                _identity_tuple(getattr(self, name), name, preserve_order=True),
            )
        for name in (
            "lease_owner",
            "last_lifecycle_source",
            "last_lifecycle_target",
        ):
            object.__setattr__(
                self,
                name,
                _identity_token(getattr(self, name), name, required=False),
            )
        if type(self.claims_open) is not bool:
            raise FederationFormalError("claims_open must be boolean")

    @property
    def state_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "logical_time": self.logical_time,
            "delivery_attempts": self.delivery_attempts,
            "applied_event_ids": list(self.applied_event_ids),
            "claim_owners": list(self.claim_owners),
            "lease_owner": self.lease_owner,
            "lease_expires_at": self.lease_expires_at,
            "claim_fence": self.claim_fence,
            "current_fence": self.current_fence,
            "last_mutation_fence": self.last_mutation_fence,
            "shard_owners": list(self.shard_owners),
            "claims_open": self.claims_open,
            "source_budget": self.source_budget,
            "target_budget": self.target_budget,
            "consumed_budget": self.consumed_budget,
            "initial_budget": self.initial_budget,
            "observed_causal_ids": list(self.observed_causal_ids),
            "propagated_causal_ids": list(self.propagated_causal_ids),
            "active_attempts": self.active_attempts,
            "active_effects": self.active_effects,
            "preserved_identity_refs": list(self.preserved_identity_refs),
            "last_lifecycle_source": self.last_lifecycle_source,
            "last_lifecycle_target": self.last_lifecycle_target,
        }


@dataclass(frozen=True)
class FederationFormalScenario:
    property: FederationFormalProperty
    transition_schema: SupervisorTransitionSchema
    generated_model: GeneratedSupervisorStateModel
    initial_state: FederationModelState
    goal_state: str
    subject_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.property, FederationFormalProperty):
            raise FederationFormalError("scenario property is not closed")
        if not isinstance(self.transition_schema, SupervisorTransitionSchema):
            raise FederationFormalError("scenario requires a supervisor transition schema")
        if not isinstance(self.generated_model, GeneratedSupervisorStateModel):
            raise FederationFormalError("scenario requires a generated TLA model")
        if not isinstance(self.initial_state, FederationModelState):
            raise FederationFormalError("scenario requires a typed finite initial state")
        if self.generated_model.transition_schema != self.transition_schema:
            raise FederationFormalError("generated model belongs to a different schema")
        if self.initial_state.stage not in self.transition_schema.states:
            raise FederationFormalError("initial finite state is outside the transition schema")
        if self.goal_state not in self.transition_schema.states:
            raise FederationFormalError("goal state is outside the transition schema")
        if not isinstance(self.subject_ids, Mapping) or not self.subject_ids:
            raise FederationFormalError("scenario requires exact subject identities")
        normalized = {
            _identity_token(key, "subject kind"): _identity_token(value, "subject identity")
            for key, value in self.subject_ids.items()
        }
        object.__setattr__(self, "subject_ids", MappingProxyType(dict(sorted(normalized.items()))))

    @property
    def scenario_id(self) -> str:
        return content_identity(
            {
                "property": self.property.value,
                "transition_schema_identity": self.transition_schema.schema_identity,
                "generated_model_identity": self.generated_model.artifact_identity,
                "initial_state_id": self.initial_state.state_id,
                "goal_state": self.goal_state,
                "subject_ids": dict(self.subject_ids),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "property": self.property.value,
            "external_model_invariant": ExternalModelInvariant.SUPERVISOR_SAFETY.value,
            "external_model_satisfies_casf_property_alone": False,
            "transition_schema_identity": self.transition_schema.schema_identity,
            "generated_model_identity": self.generated_model.artifact_identity,
            "model_identity": self.generated_model.model_identity,
            "bounds_identity": self.generated_model.bounds.identity,
            "initial_state_id": self.initial_state.state_id,
            "goal_state": self.goal_state,
            "subject_ids": dict(self.subject_ids),
        }


@dataclass(frozen=True)
class FederationFormalSuite:
    identity: FederationFormalIdentity
    bounds: ModelCheckBounds
    scenarios: tuple[FederationFormalScenario, ...]
    schema: str = CASF_FORMAL_SUITE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_FORMAL_SUITE_SCHEMA:
            raise FederationFormalError("unsupported formal suite schema")
        if not isinstance(self.identity, FederationFormalIdentity):
            raise FederationFormalError("suite identity is not typed")
        if not isinstance(self.bounds, ModelCheckBounds):
            raise FederationFormalError("suite bounds must be ModelCheckBounds")
        if not isinstance(self.scenarios, tuple) or not all(
            isinstance(item, FederationFormalScenario) for item in self.scenarios
        ):
            raise FederationFormalError("suite scenarios must be typed and immutable")
        expected = set(FederationFormalProperty)
        observed = [scenario.property for scenario in self.scenarios]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise FederationFormalError("formal suite must contain each property exactly once")
        ordered = tuple(sorted(self.scenarios, key=lambda item: item.property.value))
        if any(item.generated_model.bounds != self.bounds for item in ordered):
            raise FederationFormalError("scenario bounds differ from suite bounds")
        if any(
            item.transition_schema.source_identity != self.identity.identity for item in ordered
        ):
            raise FederationFormalError("scenario source identity differs from suite authority")
        object.__setattr__(self, "scenarios", ordered)

    @property
    def suite_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def scenario(self, property_: FederationFormalProperty | str) -> FederationFormalScenario:
        selected = FederationFormalProperty(property_)
        return next(item for item in self.scenarios if item.property is selected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "bounded": True,
            "unbounded_proof": False,
            "authority_created": False,
            "identity": self.identity.to_dict(),
            "bounds": self.bounds.to_dict(),
            "scenarios": [item.to_dict() for item in self.scenarios],
        }
        if include_id:
            payload["suite_id"] = self.suite_id
        return payload


@dataclass(frozen=True)
class FormalTraceStep:
    transition: str
    before_state_id: str
    after_state_id: str

    def __post_init__(self) -> None:
        _identity_token(self.transition, "transition")
        _identity_token(self.before_state_id, "before_state_id")
        _identity_token(self.after_state_id, "after_state_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "transition": self.transition,
            "before_state_id": self.before_state_id,
            "after_state_id": self.after_state_id,
        }


@dataclass(frozen=True)
class FormalCounterexample:
    invariant: str
    mutation: AdversarialMutation
    state: FederationModelState
    trace: tuple[FormalTraceStep, ...]

    def __post_init__(self) -> None:
        _identity_token(self.invariant, "invariant")
        if not isinstance(self.mutation, AdversarialMutation):
            raise FederationFormalError("counterexample mutation is not closed")
        if not isinstance(self.state, FederationModelState):
            raise FederationFormalError("counterexample state is not typed")
        if (
            not isinstance(self.trace, tuple)
            or not self.trace
            or not all(isinstance(item, FormalTraceStep) for item in self.trace)
        ):
            raise FederationFormalError("counterexample requires an immutable typed trace")
        if self.trace[-1].after_state_id != self.state.state_id:
            raise FederationFormalError("counterexample trace does not end at its bound state")

    @property
    def counterexample_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "invariant": self.invariant,
            "mutation": self.mutation.value,
            "state": self.state.to_dict(),
            "state_id": self.state.state_id,
            "trace": [item.to_dict() for item in self.trace],
        }
        if include_id:
            payload["counterexample_id"] = self.counterexample_id
        return payload


@dataclass(frozen=True)
class HermeticModelCheckReceipt:
    scenario_id: str
    property: FederationFormalProperty
    status: HermeticCheckStatus
    mutation: AdversarialMutation
    bounds: ModelCheckBounds
    explored_states: int
    explored_transitions: int
    goal_reached: bool
    reason: str
    counterexample: FormalCounterexample | None = None
    schema: str = CASF_FORMAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_FORMAL_RECEIPT_SCHEMA:
            raise FederationFormalError("unsupported formal receipt schema")
        _identity_token(self.scenario_id, "scenario_id")
        _bounded_text(self.reason, "reason")
        if not isinstance(self.property, FederationFormalProperty):
            raise FederationFormalError("receipt property is not closed")
        if not isinstance(self.status, HermeticCheckStatus):
            raise FederationFormalError("receipt status is not closed")
        if not isinstance(self.mutation, AdversarialMutation):
            raise FederationFormalError("receipt mutation is not closed")
        if not isinstance(self.bounds, ModelCheckBounds):
            raise FederationFormalError("receipt bounds are not typed")
        if type(self.goal_reached) is not bool:
            raise FederationFormalError("goal_reached must be boolean")
        _positive_integer(self.explored_states, "explored_states", allow_zero=True)
        _positive_integer(self.explored_transitions, "explored_transitions", allow_zero=True)
        if self.status is HermeticCheckStatus.COUNTEREXAMPLE:
            if self.counterexample is None:
                raise FederationFormalError("counterexample status requires a trace")
        elif self.counterexample is not None:
            raise FederationFormalError("non-counterexample receipt cannot contain a trace")
        if self.counterexample is not None and (self.counterexample.mutation is not self.mutation):
            raise FederationFormalError("counterexample mutation differs from its receipt")

    @property
    def passed(self) -> bool:
        return self.status is HermeticCheckStatus.PASSED

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "scenario_id": self.scenario_id,
            "property": self.property.value,
            "status": self.status.value,
            "mutation": self.mutation.value,
            "bounded": True,
            "unbounded_proof": False,
            "authority_created": False,
            "bounds": self.bounds.to_dict(),
            "explored_states": self.explored_states,
            "explored_transitions": self.explored_transitions,
            "goal_reached": self.goal_reached,
            "reason": self.reason,
            "counterexample": (self.counterexample.to_dict() if self.counterexample else None),
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class ExternalModelCheckReceipt:
    scenario_id: str
    scenario_property: FederationFormalProperty
    property: ExternalModelInvariant
    tool: ModelCheckerTool
    status: ExternalCheckStatus
    ran: bool
    matrix_snapshot_id: str
    matrix_entry_state: str
    generated_model_identity: str
    model_check_receipt_id: str
    paired_hermetic_receipt_id: str
    casf_property_satisfied_by_pair: bool
    reason: str
    schema: str = CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA:
            raise FederationFormalError("unsupported external check receipt schema")
        if not isinstance(self.scenario_property, FederationFormalProperty):
            raise FederationFormalError("external receipt scenario property is not closed")
        if not isinstance(self.property, ExternalModelInvariant):
            raise FederationFormalError("external receipt checked invariant is not closed")
        if self.property is not ExternalModelInvariant.SUPERVISOR_SAFETY:
            raise FederationFormalError("external receipt checked invariant is unsupported")
        if not isinstance(self.tool, ModelCheckerTool):
            raise FederationFormalError("external receipt tool is not closed")
        if not isinstance(self.status, ExternalCheckStatus):
            raise FederationFormalError("external receipt status is not closed")
        if type(self.ran) is not bool:
            raise FederationFormalError("external receipt ran flag must be boolean")
        if type(self.casf_property_satisfied_by_pair) is not bool:
            raise FederationFormalError("paired CASF satisfaction flag must be boolean")
        for name in (
            "scenario_id",
            "matrix_snapshot_id",
            "matrix_entry_state",
            "generated_model_identity",
        ):
            _identity_token(getattr(self, name), name)
        _identity_token(
            self.model_check_receipt_id,
            "model_check_receipt_id",
            required=False,
        )
        _identity_token(
            self.paired_hermetic_receipt_id,
            "paired_hermetic_receipt_id",
            required=False,
        )
        _bounded_text(self.reason, "reason")
        if self.status in {ExternalCheckStatus.UNAVAILABLE, ExternalCheckStatus.NOT_RUN}:
            if self.ran or self.model_check_receipt_id:
                raise FederationFormalError("unavailable/not-run checks cannot claim execution")
        elif not self.ran or not self.model_check_receipt_id:
            raise FederationFormalError("executed external outcomes require an exact receipt")
        expected_pair_satisfaction = bool(
            self.passed and self.paired_hermetic_receipt_id
        )
        if self.casf_property_satisfied_by_pair != expected_pair_satisfaction:
            raise FederationFormalError(
                "CASF property satisfaction requires both a generic external pass "
                "and paired hermetic evidence"
            )

    @property
    def passed(self) -> bool:
        """Whether the generic external invariant passed, not the CASF property."""

        return self.status is ExternalCheckStatus.PASSED

    @property
    def casf_property_passed(self) -> bool:
        """Whether the external pass has exact trusted hermetic CASF evidence."""

        return self.casf_property_satisfied_by_pair

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "scenario_id": self.scenario_id,
            "scenario_property": self.scenario_property.value,
            "property": self.property.value,
            "property_scope": "generic_supervisor_state_model",
            "external_model_satisfies_casf_property_alone": False,
            "tool": self.tool.value,
            "status": self.status.value,
            "ran": self.ran,
            "bounded": True,
            "unbounded_proof": False,
            "authority_created": False,
            "matrix_snapshot_id": self.matrix_snapshot_id,
            "matrix_entry_state": self.matrix_entry_state,
            "generated_model_identity": self.generated_model_identity,
            "model_check_receipt_id": self.model_check_receipt_id,
            "paired_hermetic_receipt_id": self.paired_hermetic_receipt_id,
            "casf_property_satisfied_by_pair": (
                self.casf_property_satisfied_by_pair
            ),
            "reason": self.reason,
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


def _rule(
    name: str,
    sources: Sequence[str],
    target: str,
    operation: str,
    **kwargs: Any,
) -> TransitionRule:
    return TransitionRule(
        name=name,
        source_states=tuple(sources),
        target_state=target,
        metadata={"operation": operation},
        **kwargs,
    )


def _lifecycle_rules() -> tuple[TransitionRule, ...]:
    rules: list[TransitionRule] = []
    for source in FederationLifecycleState:
        for target in sorted(legal_transitions(source), key=lambda item: item.value):
            name = "Lifecycle" + "".join(
                part.title() for part in f"{source.value}_to_{target.value}".split("_")
            )
            rules.append(_rule(name, (source.value,), target.value, "lifecycle"))
    rules.extend(
        (
            _rule(
                "DrainAttempt",
                (FederationLifecycleState.DRAINING.value,),
                FederationLifecycleState.DRAINING.value,
                "drain_attempt",
            ),
            _rule(
                "DrainEffect",
                (FederationLifecycleState.DRAINING.value,),
                FederationLifecycleState.DRAINING.value,
                "drain_effect",
            ),
        )
    )
    return tuple(rules)


def _scenario_definition(
    property_: FederationFormalProperty,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[TransitionRule, ...], str, str]:
    if property_ is FederationFormalProperty.EVENT_DELIVERY:
        return (
            ("pending", "delivered", "processed", "retry", "acknowledged", "dead_lettered"),
            ("acknowledged", "dead_lettered"),
            (
                _rule("Deliver", ("pending", "retry"), "delivered", "deliver"),
                _rule("ApplyEffect", ("delivered",), "processed", "apply_effect"),
                _rule("Acknowledge", ("processed",), "acknowledged", "acknowledge"),
                _rule(
                    "RetryDelivery",
                    ("delivered", "processed"),
                    "retry",
                    "retry_delivery",
                    increments_retry=True,
                ),
                _rule(
                    "DeadLetter",
                    ("delivered", "processed", "retry"),
                    "dead_lettered",
                    "dead_letter",
                ),
            ),
            "pending",
            "acknowledged",
        )
    if property_ is FederationFormalProperty.CLAIM_LEASE_FENCE:
        return (
            ("unclaimed", "claimed", "active", "expired", "released", "completed"),
            ("expired", "released", "completed"),
            (
                _rule(
                    "AcceptClaim",
                    ("unclaimed",),
                    "claimed",
                    "accept_claim",
                    accepts_claim=True,
                    increments_fence=True,
                ),
                _rule(
                    "AuthorizedMutation",
                    ("claimed", "active"),
                    "active",
                    "authorized_mutation",
                    requires_owner=True,
                ),
                _rule("TickLease", ("claimed", "active"), "active", "tick_lease"),
                _rule(
                    "ExpireLease",
                    ("claimed", "active"),
                    "expired",
                    "expire_lease",
                    requires_owner=True,
                    increments_fence=True,
                    clears_claim=True,
                ),
                _rule(
                    "ReleaseClaim",
                    ("claimed", "active"),
                    "released",
                    "release_claim",
                    requires_owner=True,
                    clears_claim=True,
                ),
                _rule(
                    "CompleteClaim",
                    ("active",),
                    "completed",
                    "complete_claim",
                    requires_owner=True,
                    clears_claim=True,
                ),
            ),
            "unclaimed",
            "expired",
        )
    if property_ is FederationFormalProperty.LIFECYCLE:
        states = tuple(item.value for item in FederationLifecycleState)
        return (
            states,
            (FederationLifecycleState.STOPPED.value,),
            _lifecycle_rules(),
            FederationLifecycleState.DECLARED.value,
            FederationLifecycleState.STOPPED.value,
        )
    if property_ is FederationFormalProperty.SHARD_TRANSFER:
        return (
            (
                "source_active",
                "frozen",
                "drained",
                "transferred",
                "target_active",
                "rolled_back",
            ),
            ("target_active", "rolled_back"),
            (
                _rule(
                    "BeginIrreversibleEffect",
                    ("source_active",),
                    "source_active",
                    "begin_irreversible",
                    capacity_delta=1,
                ),
                _rule(
                    "FinishIrreversibleEffect",
                    ("source_active",),
                    "source_active",
                    "finish_irreversible",
                    capacity_delta=-1,
                ),
                _rule("FreezeShard", ("source_active",), "frozen", "freeze_shard"),
                _rule("DrainShard", ("frozen",), "drained", "drain_shard"),
                _rule(
                    "TransferShard",
                    ("drained",),
                    "transferred",
                    "transfer_shard",
                    increments_fence=True,
                ),
                _rule(
                    "ActivateTarget",
                    ("transferred",),
                    "target_active",
                    "activate_target",
                ),
                _rule(
                    "RollbackTransfer",
                    ("transferred",),
                    "rolled_back",
                    "rollback_transfer",
                ),
            ),
            "source_active",
            "target_active",
        )
    if property_ is FederationFormalProperty.BUDGET_CONSERVATION:
        return (
            ("allocated", "transferred", "consumed", "released"),
            ("released",),
            (
                _rule("TransferBudget", ("allocated",), "transferred", "transfer_budget"),
                _rule("ConsumeBudget", ("transferred",), "consumed", "consume_budget"),
                _rule("ReleaseBudget", ("consumed",), "released", "release_budget"),
            ),
            "allocated",
            "released",
        )
    return (
        (
            "source_changed",
            "event_committed",
            "parent_observed",
            "dependent_invalidated",
            "replanned",
            "completed",
        ),
        ("completed",),
        (
            _rule("CommitCausalEvent", ("source_changed",), "event_committed", "commit_event"),
            _rule("ObserveCausalParent", ("event_committed",), "parent_observed", "observe_parent"),
            _rule(
                "PropagateInvalidation",
                ("parent_observed",),
                "dependent_invalidated",
                "propagate_invalidation",
            ),
            _rule("ReplanDependent", ("dependent_invalidated",), "replanned", "replan"),
            _rule("SettleCausalChange", ("replanned",), "completed", "settle"),
        ),
        "source_changed",
        "completed",
    )


def _subject_ids(
    identity: FederationFormalIdentity,
    property_: FederationFormalProperty,
) -> dict[str, str]:
    common = {
        "formal_identity": identity.identity,
        "task": identity.task_id,
        "attempt": identity.attempt_id,
        "lease": identity.lease_id,
    }
    common.update(
        {
            "event": _typed_id(
                "event",
                {"identity": identity.identity, "property": property_.value},
            ),
            "causal_parent": _typed_id(
                "causal-parent",
                {"identity": identity.identity, "property": property_.value},
            ),
            "causal_child": _typed_id(
                "causal-child",
                {"identity": identity.identity, "property": property_.value},
            ),
            "shard": _typed_id(
                "shard",
                {"identity": identity.identity, "assignment": identity.assignment_revision},
            ),
            "budget": _typed_id(
                "budget",
                {"identity": identity.identity, "policy": identity.policy_id},
            ),
            "checkpoint": _typed_id(
                "checkpoint", {"identity": identity.identity, "attempt": identity.attempt_id}
            ),
            "cursor": _typed_id(
                "cursor", {"identity": identity.identity, "attempt": identity.attempt_id}
            ),
        }
    )
    return common


def _initial_state(
    property_: FederationFormalProperty,
    identity: FederationFormalIdentity,
    bounds: ModelCheckBounds,
    subjects: Mapping[str, str],
    initial_stage: str,
) -> FederationModelState:
    common = {
        "stage": initial_stage,
        "current_fence": identity.fencing_epoch,
        "last_mutation_fence": identity.fencing_epoch,
    }
    if property_ is FederationFormalProperty.SHARD_TRANSFER:
        return FederationModelState(
            **common,
            shard_owners=(identity.supervisor_ids[0],),
            preserved_identity_refs=(
                identity.attempt_id,
                subjects["checkpoint"],
                subjects["cursor"],
            ),
        )
    if property_ is FederationFormalProperty.BUDGET_CONSERVATION:
        total = max(2, min(bounds.max_tasks, 8))
        return FederationModelState(
            **common,
            source_budget=total,
            initial_budget=total,
        )
    return FederationModelState(**common)


def _external_invariant_is_bound(
    model: GeneratedSupervisorStateModel,
    tool: ModelCheckerTool,
) -> bool:
    invariant = ExternalModelInvariant.SUPERVISOR_SAFETY.value
    configuration_lines = model.configuration_for(tool).splitlines()
    safety_conjuncts = ("TypeOK", *SAFETY_PROPERTIES)
    safety_definition = "\n".join(
        (f"{invariant} ==", *(f"    /\\ {item}" for item in safety_conjuncts))
    )
    return (
        model.model_text.count(f"\n{invariant} ==\n") == 1
        and model.model_text.count(safety_definition) == 1
        and configuration_lines.count(f"INVARIANT {invariant}") == 1
    )


def _bind_generic_external_invariant(
    model: GeneratedSupervisorStateModel,
) -> GeneratedSupervisorStateModel:
    """Configure the generic ``Safety`` invariant for both external tools."""

    invariant = ExternalModelInvariant.SUPERVISOR_SAFETY.value
    if f"\n{invariant} ==\n" not in model.model_text:
        raise FederationFormalError("generated model omits the generic external invariant")
    tlc_lines = model.tlc_config_text.splitlines()
    invariant_line = f"INVARIANT {invariant}"
    if invariant_line not in tlc_lines:
        insertion = next(
            (index for index, line in enumerate(tlc_lines) if line.startswith("PROPERTY ")),
            len(tlc_lines),
        )
        tlc_lines.insert(insertion, invariant_line)
    bound = replace(model, tlc_config_text="\n".join(tlc_lines) + "\n")
    if not all(
        _external_invariant_is_bound(bound, tool)
        for tool in (ModelCheckerTool.TLC, ModelCheckerTool.APALACHE)
    ):
        raise FederationFormalError("generic external invariant is not configured for every tool")
    return bound


def build_federation_formal_suite(
    identity: FederationFormalIdentity,
    *,
    bounds: ModelCheckBounds,
) -> FederationFormalSuite:
    """Build the six deterministic TLA+/hermetic CASF scenarios."""

    if not isinstance(identity, FederationFormalIdentity):
        raise FederationFormalError("identity must be FederationFormalIdentity")
    if not isinstance(bounds, ModelCheckBounds):
        raise FederationFormalError("bounds must be ModelCheckBounds")
    if identity.fencing_epoch + 2 > bounds.max_fence:
        raise FederationFormalError(
            "max_fence must represent claim acceptance followed by lease expiry"
        )
    scenarios: list[FederationFormalScenario] = []
    for property_ in FederationFormalProperty:
        states, terminal, transitions, initial_stage, goal = _scenario_definition(property_)
        subjects = _subject_ids(identity, property_)
        transition_schema = SupervisorTransitionSchema(
            tasks=(f"{identity.task_id}:{property_.value}",),
            agents=identity.supervisor_ids,
            states=states,
            initial_state=initial_stage,
            terminal_states=terminal,
            dependency_satisfied_states=terminal,
            transitions=transitions,
            dependencies={},
            required_evidence={},
            capacity=len(identity.supervisor_ids),
            source_identity=identity.identity,
            metadata={
                "formal_identity": identity.identity,
                "property": property_.value,
                "subject_ids": subjects,
                "authority_created": False,
            },
        )
        model = _bind_generic_external_invariant(
            generate_supervisor_state_model(
                transition_schema,
                bounds=bounds,
                module_name=(
                    "CASF036_"
                    + "".join(part.title() for part in property_.value.split("_"))
                ),
            )
        )
        scenarios.append(
            FederationFormalScenario(
                property=property_,
                transition_schema=transition_schema,
                generated_model=model,
                initial_state=_initial_state(property_, identity, bounds, subjects, initial_stage),
                goal_state=goal,
                subject_ids=subjects,
            )
        )
    return FederationFormalSuite(identity=identity, bounds=bounds, scenarios=tuple(scenarios))


def _advance(state: FederationModelState, stage: str, **updates: Any) -> FederationModelState:
    return replace(
        state,
        stage=stage,
        logical_time=state.logical_time + 1,
        **updates,
    )


def _event_successors(
    scenario: FederationFormalScenario,
    state: FederationModelState,
    mutation: AdversarialMutation,
) -> tuple[tuple[str, FederationModelState], ...]:
    event_id = scenario.subject_ids["event"]
    retries = scenario.generated_model.bounds.max_retries
    maximum_attempts = retries + 1
    result: list[tuple[str, FederationModelState]] = []
    if state.stage in {"pending", "retry"} and state.delivery_attempts < maximum_attempts:
        result.append(
            (
                "Deliver",
                _advance(
                    state,
                    "delivered",
                    delivery_attempts=state.delivery_attempts + 1,
                ),
            )
        )
    if state.stage == "delivered":
        applied = state.applied_event_ids
        if event_id not in applied or mutation is AdversarialMutation.DUPLICATE_EVENT_EFFECT:
            applied = (*applied, event_id)
        result.append(("ApplyEffect", _advance(state, "processed", applied_event_ids=applied)))
    if state.stage == "processed" and event_id in state.applied_event_ids:
        result.append(("Acknowledge", _advance(state, "acknowledged")))
    if state.stage in {"delivered", "processed"} and state.delivery_attempts <= retries:
        result.append(("RetryDelivery", _advance(state, "retry")))
    if (
        state.stage in {"delivered", "processed", "retry"}
        and state.delivery_attempts >= maximum_attempts
    ):
        result.append(("DeadLetter", _advance(state, "dead_lettered")))
    return tuple(result)


def _claim_successors(
    scenario: FederationFormalScenario,
    state: FederationModelState,
    mutation: AdversarialMutation,
) -> tuple[tuple[str, FederationModelState], ...]:
    owner = scenario.transition_schema.agents[0]
    result: list[tuple[str, FederationModelState]] = []
    if state.stage == "unclaimed":
        next_fence = state.current_fence + 1
        result.append(
            (
                "AcceptClaim",
                _advance(
                    state,
                    "claimed",
                    claim_owners=(owner,),
                    lease_owner=owner,
                    lease_expires_at=state.logical_time + 2,
                    claim_fence=next_fence,
                    current_fence=next_fence,
                    last_mutation_fence=next_fence,
                ),
            )
        )
    if state.stage in {"claimed", "active"} and state.claim_owners:
        if state.logical_time < state.lease_expires_at:
            result.append(
                (
                    "AuthorizedMutation",
                    _advance(
                        state,
                        "active",
                        last_mutation_fence=state.claim_fence,
                    ),
                )
            )
        result.append(("TickLease", _advance(state, "active")))
        result.append(
            (
                "ReleaseClaim",
                _advance(state, "released", claim_owners=(), lease_owner=""),
            )
        )
        if state.logical_time >= state.lease_expires_at:
            next_fence = state.current_fence + 1
            result.append(
                (
                    "ExpireLease",
                    _advance(
                        state,
                        "expired",
                        claim_owners=(),
                        lease_owner="",
                        current_fence=next_fence,
                        last_mutation_fence=next_fence,
                    ),
                )
            )
            if mutation is AdversarialMutation.STALE_FENCE_MUTATION:
                result.append(
                    (
                        "AdversaryStaleFenceMutation",
                        _advance(
                            state,
                            "completed",
                            current_fence=state.current_fence + 1,
                            last_mutation_fence=state.claim_fence,
                        ),
                    )
                )
        if state.stage == "active" and state.logical_time < state.lease_expires_at:
            result.append(
                (
                    "CompleteClaim",
                    _advance(state, "completed", claim_owners=(), lease_owner=""),
                )
            )
    return tuple(result)


def _lifecycle_successors(
    state: FederationModelState,
    mutation: AdversarialMutation,
) -> tuple[tuple[str, FederationModelState], ...]:
    result: list[tuple[str, FederationModelState]] = []
    source = FederationLifecycleState(state.stage)
    for target in sorted(legal_transitions(source), key=lambda item: item.value):
        if target is FederationLifecycleState.COMPLETED and (
            state.active_attempts or state.active_effects
        ):
            continue
        name = "Lifecycle" + "".join(
            part.title() for part in f"{source.value}_to_{target.value}".split("_")
        )
        updates: dict[str, Any] = {
            "last_lifecycle_source": source.value,
            "last_lifecycle_target": target.value,
        }
        if target is FederationLifecycleState.ACTIVE and not (
            state.active_attempts or state.active_effects
        ):
            updates.update(active_attempts=1, active_effects=1)
        result.append((name, _advance(state, target.value, **updates)))
    if state.stage == FederationLifecycleState.DRAINING.value:
        if state.active_attempts:
            result.append(
                (
                    "DrainAttempt",
                    _advance(
                        state,
                        state.stage,
                        active_attempts=state.active_attempts - 1,
                    ),
                )
            )
        if state.active_effects:
            result.append(
                (
                    "DrainEffect",
                    _advance(
                        state,
                        state.stage,
                        active_effects=state.active_effects - 1,
                    ),
                )
            )
    if (
        mutation is AdversarialMutation.ILLEGAL_LIFECYCLE_TRANSITION
        and state.stage == FederationLifecycleState.DECLARED.value
    ):
        result.append(
            (
                "AdversaryDeclaredToCompleted",
                _advance(
                    state,
                    FederationLifecycleState.COMPLETED.value,
                    last_lifecycle_source=FederationLifecycleState.DECLARED.value,
                    last_lifecycle_target=FederationLifecycleState.COMPLETED.value,
                ),
            )
        )
    return tuple(result)


def _shard_successors(
    scenario: FederationFormalScenario,
    state: FederationModelState,
    mutation: AdversarialMutation,
) -> tuple[tuple[str, FederationModelState], ...]:
    source, target = scenario.transition_schema.agents[:2]
    result: list[tuple[str, FederationModelState]] = []
    if state.stage == "source_active":
        if state.active_effects == 0:
            result.extend(
                (
                    (
                        "BeginIrreversibleEffect",
                        _advance(state, "source_active", active_effects=1),
                    ),
                    (
                        "FreezeShard",
                        _advance(state, "frozen", claims_open=False),
                    ),
                )
            )
        else:
            result.append(
                (
                    "FinishIrreversibleEffect",
                    _advance(state, "source_active", active_effects=0),
                )
            )
    elif state.stage == "frozen" and state.active_effects == 0:
        result.append(("DrainShard", _advance(state, "drained")))
    elif state.stage == "drained":
        next_fence = state.current_fence + 1
        owners = (source, target) if mutation is AdversarialMutation.DUAL_SHARD_OWNER else ()
        transition = (
            "AdversaryDualShardOwner"
            if mutation is AdversarialMutation.DUAL_SHARD_OWNER
            else "TransferShard"
        )
        result.append(
            (
                transition,
                _advance(
                    state,
                    "transferred",
                    shard_owners=owners,
                    current_fence=next_fence,
                    last_mutation_fence=next_fence,
                ),
            )
        )
    elif state.stage == "transferred":
        result.extend(
            (
                (
                    "ActivateTarget",
                    _advance(
                        state,
                        "target_active",
                        shard_owners=(target,),
                        claims_open=True,
                    ),
                ),
                (
                    "RollbackTransfer",
                    _advance(
                        state,
                        "rolled_back",
                        shard_owners=(source,),
                        claims_open=True,
                    ),
                ),
            )
        )
    return tuple(result)


def _budget_successors(
    state: FederationModelState,
    mutation: AdversarialMutation,
) -> tuple[tuple[str, FederationModelState], ...]:
    transfer = max(1, state.initial_budget // 2)
    if state.stage == "allocated":
        source = (
            state.source_budget
            if mutation is AdversarialMutation.MINT_BUDGET
            else state.source_budget - transfer
        )
        name = (
            "AdversaryMintBudget"
            if mutation is AdversarialMutation.MINT_BUDGET
            else "TransferBudget"
        )
        return (
            (
                name,
                _advance(
                    state,
                    "transferred",
                    source_budget=source,
                    target_budget=state.target_budget + transfer,
                ),
            ),
        )
    if state.stage == "transferred" and state.target_budget:
        return (
            (
                "ConsumeBudget",
                _advance(
                    state,
                    "consumed",
                    target_budget=state.target_budget - 1,
                    consumed_budget=state.consumed_budget + 1,
                ),
            ),
        )
    if state.stage == "consumed":
        return (
            (
                "ReleaseBudget",
                _advance(
                    state,
                    "released",
                    source_budget=state.source_budget + state.target_budget,
                    target_budget=0,
                ),
            ),
        )
    return ()


def _causal_successors(
    scenario: FederationFormalScenario,
    state: FederationModelState,
    mutation: AdversarialMutation,
) -> tuple[tuple[str, FederationModelState], ...]:
    parent = scenario.subject_ids["causal_parent"]
    child = scenario.subject_ids["causal_child"]
    if (
        state.stage == "source_changed"
        and mutation is AdversarialMutation.ORPHAN_CAUSAL_PROPAGATION
    ):
        return (
            (
                "AdversaryOrphanCausalPropagation",
                _advance(
                    state,
                    "dependent_invalidated",
                    propagated_causal_ids=(child,),
                ),
            ),
        )
    if state.stage == "source_changed":
        return (("CommitCausalEvent", _advance(state, "event_committed")),)
    if state.stage == "event_committed":
        return (
            (
                "ObserveCausalParent",
                _advance(
                    state,
                    "parent_observed",
                    observed_causal_ids=(parent,),
                ),
            ),
        )
    if state.stage == "parent_observed":
        return (
            (
                "PropagateInvalidation",
                _advance(
                    state,
                    "dependent_invalidated",
                    propagated_causal_ids=(child,),
                ),
            ),
        )
    if state.stage == "dependent_invalidated":
        return (("ReplanDependent", _advance(state, "replanned")),)
    if state.stage == "replanned":
        return (("SettleCausalChange", _advance(state, "completed")),)
    return ()


def _successors(
    scenario: FederationFormalScenario,
    state: FederationModelState,
    mutation: AdversarialMutation,
) -> tuple[tuple[str, FederationModelState], ...]:
    if scenario.property is FederationFormalProperty.EVENT_DELIVERY:
        return _event_successors(scenario, state, mutation)
    if scenario.property is FederationFormalProperty.CLAIM_LEASE_FENCE:
        return _claim_successors(scenario, state, mutation)
    if scenario.property is FederationFormalProperty.LIFECYCLE:
        return _lifecycle_successors(state, mutation)
    if scenario.property is FederationFormalProperty.SHARD_TRANSFER:
        return _shard_successors(scenario, state, mutation)
    if scenario.property is FederationFormalProperty.BUDGET_CONSERVATION:
        return _budget_successors(state, mutation)
    return _causal_successors(scenario, state, mutation)


def _violations(
    scenario: FederationFormalScenario,
    state: FederationModelState,
) -> tuple[str, ...]:
    violations: list[str] = []
    if state.stage not in scenario.transition_schema.states:
        violations.append("ClosedStateVocabulary")
    if state.logical_time > scenario.generated_model.bounds.max_steps:
        violations.append("FiniteStepBound")
    if scenario.property is FederationFormalProperty.EVENT_DELIVERY:
        event_id = scenario.subject_ids["event"]
        if state.delivery_attempts > scenario.generated_model.bounds.max_retries + 1:
            violations.append("BoundedDeliveryAttempts")
        if len(state.applied_event_ids) != len(set(state.applied_event_ids)):
            violations.append("IdempotentAuthoritativeEffect")
        if state.stage == "acknowledged" and event_id not in state.applied_event_ids:
            violations.append("AcknowledgementRequiresAppliedEvent")
        if state.applied_event_ids and state.delivery_attempts < 1:
            violations.append("AtLeastOnceDeliveryBeforeEffect")
    elif scenario.property is FederationFormalProperty.CLAIM_LEASE_FENCE:
        if len(state.claim_owners) > 1:
            violations.append("UniqueClaimOwner")
        if state.last_mutation_fence != state.current_fence:
            violations.append("CurrentFenceWins")
        if state.claim_owners and (
            state.lease_owner != state.claim_owners[0] or state.claim_fence != state.current_fence
        ):
            violations.append("LeaseClaimFenceBinding")
    elif scenario.property is FederationFormalProperty.LIFECYCLE:
        if state.last_lifecycle_source and state.last_lifecycle_target:
            target = FederationLifecycleState(state.last_lifecycle_target)
            if target not in legal_transitions(state.last_lifecycle_source):
                violations.append("LegalLifecycleTransition")
        if state.stage == FederationLifecycleState.COMPLETED.value and (
            state.active_attempts or state.active_effects
        ):
            violations.append("CompletionRequiresDrainedEffects")
    elif scenario.property is FederationFormalProperty.SHARD_TRANSFER:
        source, target = scenario.transition_schema.agents[:2]
        if len(state.shard_owners) > 1:
            violations.append("UniqueShardOwner")
        expected = {
            "source_active": (source,),
            "frozen": (source,),
            "drained": (source,),
            "transferred": (),
            "target_active": (target,),
            "rolled_back": (source,),
        }
        if state.stage in expected and state.shard_owners != expected[state.stage]:
            violations.append("FencedShardOwnership")
        if state.stage in {"frozen", "drained", "transferred"} and state.claims_open:
            violations.append("FrozenShardRejectsClaims")
        if state.stage not in {"source_active"} and state.active_effects:
            violations.append("IrreversibleEffectDoesNotMove")
        required = {
            scenario.subject_ids["attempt"],
            scenario.subject_ids["checkpoint"],
            scenario.subject_ids["cursor"],
        }
        if set(state.preserved_identity_refs) != required:
            violations.append("TransferPreservesExactIdentities")
        if state.last_mutation_fence != state.current_fence:
            violations.append("TransferIncrementsCurrentFence")
    elif scenario.property is FederationFormalProperty.BUDGET_CONSERVATION:
        if min(state.source_budget, state.target_budget, state.consumed_budget) < 0:
            violations.append("NonNegativeBudget")
        if (
            state.source_budget + state.target_budget + state.consumed_budget
            != state.initial_budget
        ):
            violations.append("BudgetConservation")
    else:
        parent = scenario.subject_ids["causal_parent"]
        if len(state.observed_causal_ids) != len(set(state.observed_causal_ids)):
            violations.append("UniqueCausalObservation")
        if len(state.propagated_causal_ids) != len(set(state.propagated_causal_ids)):
            violations.append("UniqueCausalPropagation")
        if state.propagated_causal_ids and parent not in state.observed_causal_ids:
            violations.append("CausalParentBeforeDependent")
    return tuple(violations)


def check_federation_scenario(
    scenario: FederationFormalScenario,
    *,
    mutation: AdversarialMutation = AdversarialMutation.NONE,
) -> HermeticModelCheckReceipt:
    """Exhaustively explore one finite scenario inside its recorded bounds."""

    if not isinstance(scenario, FederationFormalScenario):
        raise FederationFormalError("scenario must be FederationFormalScenario")
    try:
        selected_mutation = AdversarialMutation(mutation)
    except ValueError as exc:
        raise FederationFormalError("adversarial mutation is not closed") from exc
    if selected_mutation is not AdversarialMutation.NONE and (
        ADVERSARIAL_PROPERTY[selected_mutation] is not scenario.property
    ):
        raise FederationFormalError("adversarial mutation belongs to a different property")

    queue: deque[tuple[FederationModelState, tuple[FormalTraceStep, ...]]] = deque(
        ((scenario.initial_state, ()),)
    )
    visited: set[str] = set()
    explored_transitions = 0
    goal_reached = False
    while queue:
        state, trace = queue.popleft()
        state_id = state.state_id
        if state_id in visited:
            continue
        visited.add(state_id)
        violation = _violations(scenario, state)
        if violation:
            counterexample = FormalCounterexample(
                invariant=violation[0],
                mutation=selected_mutation,
                state=state,
                trace=trace,
            )
            return HermeticModelCheckReceipt(
                scenario_id=scenario.scenario_id,
                property=scenario.property,
                status=HermeticCheckStatus.COUNTEREXAMPLE,
                mutation=selected_mutation,
                bounds=scenario.generated_model.bounds,
                explored_states=len(visited),
                explored_transitions=explored_transitions,
                goal_reached=goal_reached,
                reason=f"bounded exploration found {violation[0]}",
                counterexample=counterexample,
            )
        goal_reached = goal_reached or state.stage == scenario.goal_state
        if state.logical_time >= scenario.generated_model.bounds.max_steps:
            continue
        for transition, successor in _successors(scenario, state, selected_mutation):
            explored_transitions += 1
            step = FormalTraceStep(
                transition=transition,
                before_state_id=state_id,
                after_state_id=successor.state_id,
            )
            queue.append((successor, (*trace, step)))
        if len(visited) + len(queue) > MAX_HERMETIC_EXPLORED_STATES:
            return HermeticModelCheckReceipt(
                scenario_id=scenario.scenario_id,
                property=scenario.property,
                status=HermeticCheckStatus.INCONCLUSIVE,
                mutation=selected_mutation,
                bounds=scenario.generated_model.bounds,
                explored_states=len(visited),
                explored_transitions=explored_transitions,
                goal_reached=goal_reached,
                reason="finite exploration state ceiling was reached",
            )
    status = HermeticCheckStatus.PASSED if goal_reached else HermeticCheckStatus.INCONCLUSIVE
    reason = (
        "all reachable states satisfied the invariants and the goal was reachable "
        "inside the recorded finite bounds"
        if goal_reached
        else "no invariant failed, but the goal was not reachable inside the bound"
    )
    return HermeticModelCheckReceipt(
        scenario_id=scenario.scenario_id,
        property=scenario.property,
        status=status,
        mutation=selected_mutation,
        bounds=scenario.generated_model.bounds,
        explored_states=len(visited),
        explored_transitions=explored_transitions,
        goal_reached=goal_reached,
        reason=reason,
    )


def check_federation_formal_suite(
    suite: FederationFormalSuite,
) -> tuple[HermeticModelCheckReceipt, ...]:
    """Run every hermetic scenario without adversarial mutation."""

    if not isinstance(suite, FederationFormalSuite):
        raise FederationFormalError("suite must be FederationFormalSuite")
    return tuple(check_federation_scenario(item) for item in suite.scenarios)


_EXTERNAL_STATUS: Final[Mapping[ModelCheckStatus, ExternalCheckStatus]] = MappingProxyType(
    {
        ModelCheckStatus.PASSED: ExternalCheckStatus.PASSED,
        ModelCheckStatus.COUNTEREXAMPLE: ExternalCheckStatus.COUNTEREXAMPLE,
        ModelCheckStatus.UNKNOWN: ExternalCheckStatus.UNKNOWN,
        ModelCheckStatus.TIMED_OUT: ExternalCheckStatus.TIMED_OUT,
        ModelCheckStatus.UNAVAILABLE: ExternalCheckStatus.UNAVAILABLE,
        ModelCheckStatus.ERROR: ExternalCheckStatus.ERROR,
    }
)

# Capture the reviewed implementation itself.  Calling this unbound method,
# together with the exact-type gate below, prevents a checker subclass or an
# instance-level ``check`` replacement from substituting a fabricated receipt.
_TRUSTED_MODEL_CHECK: Final = SupervisorStateModelChecker.check
_TRUSTED_MODEL_CLASSIFY: Final = SupervisorStateModelChecker._classify
_TRUSTED_HERMETIC_CHECK: Final = check_federation_scenario


def _validated_hermetic_pairings(
    suite: FederationFormalSuite,
    receipts: Sequence[HermeticModelCheckReceipt],
) -> Mapping[FederationFormalProperty, HermeticModelCheckReceipt]:
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise FederationFormalError("hermetic receipts must be a typed sequence")
    pairings: dict[FederationFormalProperty, HermeticModelCheckReceipt] = {}
    for receipt in receipts:
        if type(receipt) is not HermeticModelCheckReceipt:
            raise FederationFormalError("hermetic pairing receipt is not canonical")
        scenario = suite.scenario(receipt.property)
        expected = _TRUSTED_HERMETIC_CHECK(scenario)
        if receipt != expected or receipt.receipt_id != expected.receipt_id:
            raise FederationFormalError(
                "hermetic pairing receipt does not match trusted finite exploration"
            )
        if receipt.property in pairings:
            raise FederationFormalError("hermetic pairing contains a duplicate property")
        pairings[receipt.property] = receipt
    return MappingProxyType(pairings)


def _validate_external_model_check_receipt(
    receipt: ModelCheckReceipt,
    *,
    scenario: FederationFormalScenario,
    tool: ModelCheckerTool,
    entry: ProverMatrixEntry,
) -> None:
    """Reject receipts which do not prove an exact checker invocation.

    ``ModelCheckReceipt`` is also used as a general execution record, so its
    constructor deliberately permits incomplete non-passing outcomes.  A CASF
    external *pass* has a narrower truth boundary: the qualified matrix
    executable must have produced successful version and check executions,
    the commands and generated model must be exact, and all properties the
    trusted checker claims for that tool must be present.
    """

    if not _external_invariant_is_bound(scenario.generated_model, tool):
        raise FederationFormalError(
            "generic external invariant is not bound to the model and configuration"
        )

    if type(receipt) is not ModelCheckReceipt:
        raise FederationFormalError("checker returned a non-canonical receipt type")
    if receipt.tool is not tool:
        raise FederationFormalError("checker receipt tool does not match the requested tool")
    if (
        receipt.model is not scenario.generated_model
        or receipt.model.artifact_identity != scenario.generated_model.artifact_identity
    ):
        raise FederationFormalError("checker receipt does not bind the generated scenario model")
    if receipt.configuration_text != scenario.generated_model.configuration_for(tool):
        raise FederationFormalError("checker receipt configuration does not bind the scenario")

    executable = entry.executable_path or ""
    if not executable or receipt.executable != executable:
        raise FederationFormalError("checker receipt executable does not match the matrix")
    if not receipt.version_command or receipt.version_command[0] != executable:
        raise FederationFormalError("checker version command does not invoke the matrix executable")
    if not receipt.command or receipt.command[0] != executable:
        raise FederationFormalError("checker command does not invoke the matrix executable")

    expected_version_command = (
        (executable, "--version")
        if tool is ModelCheckerTool.TLC
        else (executable, "version")
    )
    if receipt.version_command != expected_version_command:
        raise FederationFormalError("checker version command is not the reviewed command")
    module_name = scenario.generated_model.module_name
    if tool is ModelCheckerTool.TLC:
        command_shape_valid = (
            len(receipt.command) == 4
            and receipt.command[1] == "-config"
            and receipt.command[2].endswith(f"/{module_name}.cfg")
            and receipt.command[3].endswith(f"/{module_name}.tla")
        )
    else:
        command_shape_valid = (
            len(receipt.command) == 7
            and receipt.command[1] == "check"
            and receipt.command[2].endswith("/apalache.cfg")
            and receipt.command[2].startswith("--config=")
            and receipt.command[3] == f"--length={scenario.generated_model.bounds.max_steps}"
            and receipt.command[4:6] == ("--inv=Safety", "--no-deadlock")
            and receipt.command[6].endswith(f"/{module_name}.tla")
        )
    if not command_shape_valid:
        raise FederationFormalError("checker command is not the reviewed bounded command")

    if receipt.status is not ModelCheckStatus.PASSED:
        return

    if type(receipt.version_returncode) is not int or receipt.version_returncode != 0:
        raise FederationFormalError("passed receipt lacks a successful version return code")
    if type(receipt.returncode) is not int or receipt.returncode != 0:
        raise FederationFormalError("passed receipt lacks a successful check return code")
    version_output = (receipt.version_stdout or receipt.version_stderr).strip()
    version_lines = tuple(
        line.strip()
        for line in f"{receipt.version_stdout}\n{receipt.version_stderr}".splitlines()
        if line.strip()
    )
    if not version_output or receipt.tool_version != version_output:
        raise FederationFormalError("passed receipt lacks exact version-command evidence")
    if not entry.executable_version or not version_lines:
        raise FederationFormalError("passed receipt is not bound to a matrix executable version")
    if version_lines[0] != entry.executable_version:
        raise FederationFormalError("passed receipt executable version differs from the matrix")

    required_safety = tuple(sorted(SAFETY_PROPERTIES))
    required_liveness = (
        tuple(sorted(LIVENESS_PROPERTIES)) if tool is ModelCheckerTool.TLC else ()
    )
    if receipt.checked_safety_properties != required_safety:
        raise FederationFormalError("passed receipt omits required safety properties")
    if receipt.checked_liveness_properties != required_liveness:
        raise FederationFormalError("passed receipt omits required liveness properties")
    if receipt.output_truncated:
        raise FederationFormalError("truncated checker output cannot establish a pass")
    if receipt.counterexample is not None:
        raise FederationFormalError("passed receipt cannot contain a counterexample")

    combined = "\n".join(part for part in (receipt.stdout, receipt.stderr) if part)
    classified_status, classified_reason = _TRUSTED_MODEL_CLASSIFY(
        tool,
        CommandResult(
            returncode=receipt.returncode,
            stdout=receipt.stdout,
            stderr=receipt.stderr,
            output_truncated=receipt.output_truncated,
        ),
        combined,
    )
    if classified_status is not ModelCheckStatus.PASSED:
        raise FederationFormalError("passed receipt output does not establish a reviewed pass")
    if receipt.reason != classified_reason:
        raise FederationFormalError("passed receipt reason is inconsistent with checker output")


def run_external_model_checks(
    suite: FederationFormalSuite,
    *,
    matrix: ProverMatrixSnapshot,
    checker: SupervisorStateModelChecker | None = None,
    hermetic_receipts: Sequence[HermeticModelCheckReceipt] = (),
) -> tuple[ExternalModelCheckReceipt, ...]:
    """Run certified TLC/Apalache lanes and record every unavailable lane.

    Discovery is insufficient.  A lane runs only after the prover matrix has
    an executable-backed, smoke-tested, translation-conformant entry that is
    authoritative for bounded state-machine checks.  External results are
    labeled only with the generic ``Safety`` invariant actually present in
    their model and configuration.  They can satisfy a scenario's CASF
    property only as an explicit pair with an exactly reconstructed hermetic
    receipt.  Every result remains bounded and non-promoting.
    """

    if not isinstance(suite, FederationFormalSuite):
        raise FederationFormalError("suite must be FederationFormalSuite")
    if not isinstance(matrix, ProverMatrixSnapshot):
        raise FederationFormalError("matrix must be ProverMatrixSnapshot")
    pairings = _validated_hermetic_pairings(suite, hermetic_receipts)
    if checker is not None and type(checker) is not SupervisorStateModelChecker:
        raise FederationFormalError(
            "checker must be an exact SupervisorStateModelChecker instance"
        )
    engine = checker or SupervisorStateModelChecker()
    results: list[ExternalModelCheckReceipt] = []
    for scenario in suite.scenarios:
        for tool, prover_id in (
            (ModelCheckerTool.TLC, "tla_tlc"),
            (ModelCheckerTool.APALACHE, "apalache"),
        ):
            entry = matrix.capabilities.get(prover_id)
            hermetic_receipt = pairings.get(scenario.property)
            common = {
                "scenario_id": scenario.scenario_id,
                "scenario_property": scenario.property,
                "property": ExternalModelInvariant.SUPERVISOR_SAFETY,
                "tool": tool,
                "matrix_snapshot_id": matrix.snapshot_id,
                "generated_model_identity": scenario.generated_model.artifact_identity,
                "paired_hermetic_receipt_id": (
                    hermetic_receipt.receipt_id if hermetic_receipt is not None else ""
                ),
            }
            if entry is None or entry.absent:
                results.append(
                    ExternalModelCheckReceipt(
                        **common,
                        status=ExternalCheckStatus.UNAVAILABLE,
                        ran=False,
                        matrix_entry_state=(
                            entry.highest_state.value if entry is not None else "absent"
                        ),
                        model_check_receipt_id="",
                        casf_property_satisfied_by_pair=False,
                        reason=f"{prover_id} is unavailable in the bound prover matrix",
                    )
                )
                continue
            qualified = (
                entry.executable_path
                and entry.smoke_tested
                and entry.translation_conformant
                and "bounded_state_machine" in entry.authoritative_for
            )
            if not qualified:
                results.append(
                    ExternalModelCheckReceipt(
                        **common,
                        status=ExternalCheckStatus.NOT_RUN,
                        ran=False,
                        matrix_entry_state=entry.highest_state.value,
                        model_check_receipt_id="",
                        casf_property_satisfied_by_pair=False,
                        reason=(
                            f"{prover_id} lacks an executable-backed, smoke-tested, "
                            "translation-conformant bounded-state-machine capability"
                        ),
                    )
                )
                continue
            try:
                receipt = _TRUSTED_MODEL_CHECK(
                    engine,
                    scenario.generated_model,
                    tool=tool,
                    executable=entry.executable_path,
                )
                _validate_external_model_check_receipt(
                    receipt,
                    scenario=scenario,
                    tool=tool,
                    entry=entry,
                )
            except Exception as exc:
                # The matrix establishes that the executable was qualified,
                # not that this particular invocation completed.  Without a
                # typed checker receipt we cannot bind a run to this exact
                # model, so preserve the failure as non-execution rather than
                # manufacturing an executable result or letting a provider
                # exception turn into an ambiguous caller-side outcome.
                results.append(
                    ExternalModelCheckReceipt(
                        **common,
                        status=ExternalCheckStatus.NOT_RUN,
                        ran=False,
                        matrix_entry_state=entry.highest_state.value,
                        model_check_receipt_id="",
                        casf_property_satisfied_by_pair=False,
                        reason=(
                            "qualified checker did not produce a valid execution receipt: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                )
                continue
            if receipt.status is ModelCheckStatus.UNAVAILABLE:
                results.append(
                    ExternalModelCheckReceipt(
                        **common,
                        status=ExternalCheckStatus.UNAVAILABLE,
                        ran=False,
                        matrix_entry_state=entry.highest_state.value,
                        model_check_receipt_id="",
                        casf_property_satisfied_by_pair=False,
                        reason=receipt.reason,
                    )
                )
                continue
            results.append(
                ExternalModelCheckReceipt(
                    **common,
                    status=_EXTERNAL_STATUS[receipt.status],
                    ran=True,
                    matrix_entry_state=entry.highest_state.value,
                    model_check_receipt_id=receipt.receipt_id,
                    casf_property_satisfied_by_pair=(
                        receipt.status is ModelCheckStatus.PASSED
                        and hermetic_receipt is not None
                    ),
                    reason=receipt.reason,
                )
            )
    return tuple(results)


__all__ = [
    "ADVERSARIAL_PROPERTY",
    "CASF_EXTERNAL_CHECK_RECEIPT_SCHEMA",
    "CASF_FORMAL_IDENTITY_SCHEMA",
    "CASF_FORMAL_RECEIPT_SCHEMA",
    "CASF_FORMAL_SUITE_SCHEMA",
    "AdversarialMutation",
    "ExternalCheckStatus",
    "ExternalModelCheckReceipt",
    "ExternalModelInvariant",
    "FederationFormalError",
    "FederationFormalIdentity",
    "FederationFormalProperty",
    "FederationFormalScenario",
    "FederationFormalSuite",
    "FederationModelState",
    "FormalCounterexample",
    "FormalTraceStep",
    "HermeticCheckStatus",
    "HermeticModelCheckReceipt",
    "build_federation_formal_suite",
    "check_federation_formal_suite",
    "check_federation_scenario",
    "run_external_model_checks",
]
