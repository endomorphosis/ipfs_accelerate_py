"""Scheduler authority and concurrency contracts (SchedulerContractCatalog@1).

Resolves the SwissKnife / accelerator scheduler surfaces into a fail-closed
catalog.  Each declared scheduler is exactly one of:

* ``canonical`` — reviewed primary implementation for a role;
* ``proved_adapter`` — explicit, version-bound adapter of a canonical root;
* ``legacy_only`` — retained only for compatibility, never primary; or
* ``contradictory`` — open conflict that cannot grant authority.

Shared class or method names never prove equivalence.  Concurrency claims
bind the modeled bound, implementation id, and version.  Lease ownership and
fencing-token checks dominate every effectful transition.  Bounded
interleavings conserve admitted work and terminal outcomes; retry, cancel,
and crash recovery paths cannot duplicate or lose tasks.

Interface: ``SchedulerContractCatalog@1`` (depends on
``RuntimeComponentCatalog@1`` component identities when supplied).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .content_identity_bridge import identify_strict_artifact


SCHEDULER_CONTRACT_CATALOG_INTERFACE: Final = "SchedulerContractCatalog@1"
SCHEDULER_CONTRACT_EXTRACTOR_INTERFACE: Final = "SchedulerContractExtractor@1"
CATALOG_VERSION: Final = "1"

SCHEDULER_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-implementation-authority@1"
)
SCHEDULER_SURFACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-surface@1"
)
SCHEDULER_RELATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-relation@1"
)
SCHEDULER_INVARIANT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-invariant@1"
)
SCHEDULER_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-contract-catalog@1"
)
LEASE_FENCE_GATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-lease-fence-gate@1"
)
INTERLEAVING_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-interleaving-trace@1"
)
RECOVERY_PATH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/scheduler-recovery-path@1"
)

# Hard bound for exhaustive interleaving exploration of small fixtures.
DEFAULT_MAX_INTERLEAVING_STEPS: Final = 32
DEFAULT_MAX_INTERLEAVING_BRANCHES: Final = 4096

# Queue accounting buckets that must be conserved under modeled bounds.
CONSERVED_BUCKETS: Final[tuple[str, ...]] = (
    "admitted",
    "reserved",
    "running",
    "retrying",
    "completed",
    "cancelled",
    "failed",
)
TERMINAL_BUCKETS: Final[frozenset[str]] = frozenset(
    {"completed", "cancelled", "failed"}
)
ACTIVE_BUCKETS: Final[frozenset[str]] = frozenset(
    {"admitted", "reserved", "running", "retrying"}
)


class SchedulerContractError(ValueError):
    """Base class for fail-closed scheduler contract errors."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "scheduler_contract_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class MissingSchedulerError(SchedulerContractError):
    """A required scheduler surface is absent."""


class DuplicateSchedulerError(SchedulerContractError):
    """A scheduler identity is duplicated."""


class SchedulerAuthorityError(SchedulerContractError):
    """Implementation authority is invalid, unresolved, or name-only."""


class SchedulerCIDError(SchedulerContractError):
    """A stored CID is absent or does not match its canonical preimage."""


class SchedulerInvariantError(SchedulerContractError):
    """A lease, fence, conservation, or recovery invariant failed."""


class SchedulerSourceError(SchedulerContractError):
    """A cataloged source file or symbol cannot be found."""


class SchedulerAuthorityKind(str, Enum):
    """Closed authority vocabulary from the SCA-173 acceptance criteria."""

    CANONICAL = "canonical"
    PROVED_ADAPTER = "proved_adapter"
    LEGACY_ONLY = "legacy_only"
    CONTRADICTORY = "contradictory"


class SchedulerRole(str, Enum):
    """Distinct scheduler roles; shared names never collapse these roles."""

    DETERMINISTIC_OWNERSHIP = "deterministic_ownership"
    LEGACY_WORKFLOW = "legacy_workflow"
    MCP_WORKFLOW = "mcp_workflow"
    MCP_RISK = "mcp_risk"
    SWISSKNIFE_MCP = "swissknife_mcp"
    SUPERVISOR_RESOURCE = "supervisor_resource"
    SUPERVISOR_PROVIDER = "supervisor_provider"
    VALIDATION = "validation"
    PROOF = "proof"


class SchedulerRelationKind(str, Enum):
    """Explicit, version-bound relations between scheduler surfaces."""

    PROVED_EQUIVALENCE = "proved_equivalence"
    EXPLICIT_ADAPTER = "explicit_adapter"
    LEGACY_DELEGATION = "legacy_delegation"
    CONTRADICTION = "contradiction"


class QueueBucket(str, Enum):
    """Accounting buckets for admitted work."""

    ADMITTED = "admitted"
    RESERVED = "reserved"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TransitionKind(str, Enum):
    """Effectful and non-effectful scheduler transitions."""

    ADMIT = "admit"
    RESERVE = "reserve"
    START = "start"
    COMPLETE = "complete"
    FAIL = "fail"
    CANCEL = "cancel"
    RETRY = "retry"
    HEARTBEAT = "heartbeat"
    CRASH_RECOVER = "crash_recover"
    # Non-effectful observation / query paths.
    OBSERVE = "observe"


class InvariantFamily(str, Enum):
    """Reviewed property families emitted for scheduler surfaces."""

    CANONICAL_IMPLEMENTATION_SELECTED = "CanonicalImplementationSelected"
    LEASE_FENCE_BEFORE_EFFECT = "LeaseFenceBeforeEffect"
    QUEUE_ACCOUNTING_CONSERVED = "QueueAccountingConserved"
    LIFECYCLE_STATE_MACHINE_CONFORMS = "LifecycleStateMachineConforms"
    NO_DUPLICATE_OR_LOST_WORK = "NoDuplicateOrLostWork"


# Effectful transitions require a current lease and matching fence token.
EFFECTFUL_TRANSITIONS: Final[frozenset[TransitionKind]] = frozenset(
    {
        TransitionKind.ADMIT,
        TransitionKind.RESERVE,
        TransitionKind.START,
        TransitionKind.COMPLETE,
        TransitionKind.FAIL,
        TransitionKind.CANCEL,
        TransitionKind.RETRY,
        TransitionKind.HEARTBEAT,
        TransitionKind.CRASH_RECOVER,
    }
)

# Legal single-task transitions (source_bucket -> transition -> dest_bucket).
_LEGAL_TRANSITIONS: Final[
    frozenset[tuple[QueueBucket, TransitionKind, QueueBucket]]
] = frozenset(
    {
        (QueueBucket.ADMITTED, TransitionKind.RESERVE, QueueBucket.RESERVED),
        (QueueBucket.ADMITTED, TransitionKind.CANCEL, QueueBucket.CANCELLED),
        (QueueBucket.ADMITTED, TransitionKind.FAIL, QueueBucket.FAILED),
        (QueueBucket.RESERVED, TransitionKind.START, QueueBucket.RUNNING),
        (QueueBucket.RESERVED, TransitionKind.CANCEL, QueueBucket.CANCELLED),
        (QueueBucket.RESERVED, TransitionKind.FAIL, QueueBucket.FAILED),
        (QueueBucket.RUNNING, TransitionKind.COMPLETE, QueueBucket.COMPLETED),
        (QueueBucket.RUNNING, TransitionKind.FAIL, QueueBucket.FAILED),
        (QueueBucket.RUNNING, TransitionKind.CANCEL, QueueBucket.CANCELLED),
        (QueueBucket.RUNNING, TransitionKind.RETRY, QueueBucket.RETRYING),
        (QueueBucket.RUNNING, TransitionKind.CRASH_RECOVER, QueueBucket.ADMITTED),
        (QueueBucket.RETRYING, TransitionKind.START, QueueBucket.RUNNING),
        (QueueBucket.RETRYING, TransitionKind.CANCEL, QueueBucket.CANCELLED),
        (QueueBucket.RETRYING, TransitionKind.FAIL, QueueBucket.FAILED),
        (QueueBucket.RETRYING, TransitionKind.CRASH_RECOVER, QueueBucket.ADMITTED),
    }
)


def _cid(payload: Mapping[str, Any]) -> str:
    return identify_strict_artifact(payload).cid


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchedulerContractError(
            f"{field_name} must be an object",
            reason_code="invalid_scheduler_field",
            details={"field": field_name},
        )
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchedulerContractError(
            f"{field_name} must be an array",
            reason_code="invalid_scheduler_field",
            details={"field": field_name},
        )
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchedulerContractError(
            f"{field_name} must be a nonempty string",
            reason_code="invalid_scheduler_field",
            details={"field": field_name},
        )
    return value


def _source_path(value: object, field_name: str) -> str:
    source = _text(value, field_name)
    parsed = PurePosixPath(source)
    if parsed.is_absolute() or ".." in parsed.parts or source != parsed.as_posix():
        raise SchedulerContractError(
            f"{field_name} must be a normalized relative POSIX path",
            reason_code="invalid_source_path",
            details={"field": field_name, "value": source},
        )
    return source


def _enum(enum_type: type[Enum], value: object, field_name: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise SchedulerContractError(
            f"{field_name} has an unsupported value",
            reason_code="invalid_scheduler_enum",
            details={"field": field_name, "value": value},
        ) from exc


def _nonneg_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchedulerContractError(
            f"{field_name} must be a non-negative integer",
            reason_code="invalid_scheduler_int",
            details={"field": field_name, "value": value},
        )
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchedulerContractError(
            f"{field_name} must be a positive integer",
            reason_code="invalid_scheduler_int",
            details={"field": field_name, "value": value},
        )
    return value


def _verified_cid(
    data: Mapping[str, Any],
    field_name: str,
    preimage: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> str:
    expected = _cid(preimage)
    stored = data.get(field_name)
    if stored is None and not require_stored_cids:
        return expected
    if not isinstance(stored, str) or not stored:
        raise SchedulerCIDError(
            f"{field_name} is required",
            reason_code="scheduler_cid_missing",
            details={"field": field_name},
        )
    if stored != expected:
        raise SchedulerCIDError(
            f"{field_name} does not match its canonical preimage",
            reason_code="scheduler_cid_mismatch",
            details={"field": field_name, "stored": stored, "expected": expected},
        )
    return stored


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SchedulerContractError(
            f"{field_name} must be a boolean",
            reason_code="invalid_scheduler_bool",
            details={"field": field_name, "value": value},
        )
    return value


# ---------------------------------------------------------------------------
# Catalog records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerAuthority:
    """Typed authority for one scheduler surface."""

    kind: SchedulerAuthorityKind
    canonical_scheduler_id: str
    decision: str
    adapter_contract_id: str
    version: str
    source_path: str
    authority_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULER_AUTHORITY_SCHEMA,
            "kind": self.kind.value,
            "canonicalSchedulerId": self.canonical_scheduler_id,
            "decision": self.decision,
            "adapterContractId": self.adapter_contract_id,
            "version": self.version,
            "sourcePath": self.source_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "authorityCid": self.authority_cid}


@dataclass(frozen=True)
class SchedulerSurface:
    """One concrete scheduler implementation surface."""

    scheduler_id: str
    display_name: str
    role: SchedulerRole
    implementation_symbol: str
    source_path: str
    package_id: str
    version: str
    concurrency_bound: int
    supports_lease: bool
    supports_fence: bool
    authority: SchedulerAuthority
    surface_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULER_SURFACE_SCHEMA,
            "schedulerId": self.scheduler_id,
            "displayName": self.display_name,
            "role": self.role.value,
            "implementationSymbol": self.implementation_symbol,
            "sourcePath": self.source_path,
            "packageId": self.package_id,
            "version": self.version,
            "concurrencyBound": self.concurrency_bound,
            "supportsLease": self.supports_lease,
            "supportsFence": self.supports_fence,
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "surfaceCid": self.surface_cid}


@dataclass(frozen=True)
class SchedulerRelation:
    """Explicit relation between two scheduler surfaces (never name-only)."""

    relation_id: str
    kind: SchedulerRelationKind
    source_scheduler_id: str
    target_scheduler_id: str
    source_version: str
    target_version: str
    adapter_contract_id: str
    proof_binding: str
    relation_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULER_RELATION_SCHEMA,
            "relationId": self.relation_id,
            "kind": self.kind.value,
            "sourceSchedulerId": self.source_scheduler_id,
            "targetSchedulerId": self.target_scheduler_id,
            "sourceVersion": self.source_version,
            "targetVersion": self.target_version,
            "adapterContractId": self.adapter_contract_id,
            "proofBinding": self.proof_binding,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "relationCid": self.relation_cid}


@dataclass(frozen=True)
class SchedulerInvariant:
    """A reviewed invariant bound to one or more scheduler surfaces."""

    invariant_id: str
    family: InvariantFamily
    scheduler_ids: tuple[str, ...]
    statement: str
    bound: int
    version: str
    invariant_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULER_INVARIANT_SCHEMA,
            "invariantId": self.invariant_id,
            "family": self.family.value,
            "schedulerIds": list(self.scheduler_ids),
            "statement": self.statement,
            "bound": self.bound,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "invariantCid": self.invariant_cid}


@dataclass(frozen=True)
class SchedulerContractCatalog:
    """CID-bound catalog of scheduler authority and concurrency contracts."""

    surfaces: tuple[SchedulerSurface, ...]
    relations: tuple[SchedulerRelation, ...]
    invariants: tuple[SchedulerInvariant, ...]
    runtime_component_id: str
    catalog_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULER_CATALOG_SCHEMA,
            "catalogVersion": CATALOG_VERSION,
            "runtimeComponentId": self.runtime_component_id,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "relations": [relation.to_dict() for relation in self.relations],
            "invariants": [invariant.to_dict() for invariant in self.invariants],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "catalogCid": self.catalog_cid}

    def surface(self, scheduler_id: str) -> SchedulerSurface:
        matches = [s for s in self.surfaces if s.scheduler_id == scheduler_id]
        if len(matches) != 1:
            raise MissingSchedulerError(
                f"scheduler id does not resolve uniquely: {scheduler_id}",
                reason_code="scheduler_lookup_failed",
                details={"schedulerId": scheduler_id, "matches": len(matches)},
            )
        return matches[0]

    def surfaces_by_role(self, role: SchedulerRole | str) -> tuple[SchedulerSurface, ...]:
        role_enum = _enum(SchedulerRole, role if isinstance(role, str) else role.value, "role")
        return tuple(s for s in self.surfaces if s.role is role_enum)

    def canonical_for_role(self, role: SchedulerRole | str) -> SchedulerSurface:
        candidates = [
            s
            for s in self.surfaces_by_role(role)
            if s.authority.kind is SchedulerAuthorityKind.CANONICAL
        ]
        if len(candidates) != 1:
            raise SchedulerAuthorityError(
                f"role {role} does not have exactly one canonical surface",
                reason_code="role_canonical_ambiguous",
                details={
                    "role": role if isinstance(role, str) else role.value,
                    "matches": len(candidates),
                },
            )
        return candidates[0]

    def open_contradictions(self) -> tuple[SchedulerSurface, ...]:
        return tuple(
            s
            for s in self.surfaces
            if s.authority.kind is SchedulerAuthorityKind.CONTRADICTORY
        )


# ---------------------------------------------------------------------------
# Lease / fence gate and concurrency models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaseFenceState:
    """Current lease ownership and fencing epoch for one work unit."""

    owner_id: str
    fencing_token: int
    lease_expires_at_ms: int
    now_ms: int
    held: bool = True

    def current(self) -> bool:
        return self.held and self.now_ms <= self.lease_expires_at_ms and self.fencing_token > 0


@dataclass(frozen=True)
class EffectAttempt:
    """One attempted transition against a lease/fence gate."""

    transition: TransitionKind
    actor_id: str
    presented_fencing_token: int
    task_id: str
    effectful: bool


@dataclass(frozen=True)
class LeaseFenceGateDecision:
    """Whether an effect may proceed under the current lease/fence state."""

    allowed: bool
    reason_code: str
    requires_lease: bool
    requires_fence: bool
    decision_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": LEASE_FENCE_GATE_SCHEMA,
            "allowed": self.allowed,
            "reasonCode": self.reason_code,
            "requiresLease": self.requires_lease,
            "requiresFence": self.requires_fence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "decisionCid": self.decision_cid}


def evaluate_lease_fence_gate(
    state: LeaseFenceState,
    attempt: EffectAttempt,
    *,
    surface: SchedulerSurface | None = None,
) -> LeaseFenceGateDecision:
    """Lease and fence checks dominate every effectful transition.

    Observation transitions may proceed without a lease.  Effectful
    transitions are denied when the lease is expired, the actor is not the
    owner, or the presented fencing token is stale.  When a surface declares
    that it does not support lease/fence, effectful transitions fail closed
    rather than silently bypassing the gate.
    """

    requires_lease = attempt.effectful or attempt.transition in EFFECTFUL_TRANSITIONS
    requires_fence = requires_lease

    if surface is not None:
        if requires_lease and not surface.supports_lease:
            decision = LeaseFenceGateDecision(
                allowed=False,
                reason_code="lease_unsupported_effect_denied",
                requires_lease=True,
                requires_fence=True,
                decision_cid="",
            )
            return LeaseFenceGateDecision(
                **{**decision.__dict__, "decision_cid": _cid(decision.preimage())}
            )
        if requires_fence and not surface.supports_fence:
            decision = LeaseFenceGateDecision(
                allowed=False,
                reason_code="fence_unsupported_effect_denied",
                requires_lease=True,
                requires_fence=True,
                decision_cid="",
            )
            return LeaseFenceGateDecision(
                **{**decision.__dict__, "decision_cid": _cid(decision.preimage())}
            )

    if not requires_lease:
        decision = LeaseFenceGateDecision(
            allowed=True,
            reason_code="observation_without_effect",
            requires_lease=False,
            requires_fence=False,
            decision_cid="",
        )
        return LeaseFenceGateDecision(
            **{**decision.__dict__, "decision_cid": _cid(decision.preimage())}
        )

    if not state.current():
        reason = "lease_expired" if state.held else "lease_not_held"
        if state.fencing_token <= 0:
            reason = "fence_token_invalid"
        decision = LeaseFenceGateDecision(
            allowed=False,
            reason_code=reason,
            requires_lease=True,
            requires_fence=True,
            decision_cid="",
        )
        return LeaseFenceGateDecision(
            **{**decision.__dict__, "decision_cid": _cid(decision.preimage())}
        )

    if attempt.actor_id != state.owner_id:
        decision = LeaseFenceGateDecision(
            allowed=False,
            reason_code="lease_owner_mismatch",
            requires_lease=True,
            requires_fence=True,
            decision_cid="",
        )
        return LeaseFenceGateDecision(
            **{**decision.__dict__, "decision_cid": _cid(decision.preimage())}
        )

    if attempt.presented_fencing_token != state.fencing_token:
        decision = LeaseFenceGateDecision(
            allowed=False,
            reason_code="stale_fencing_token",
            requires_lease=True,
            requires_fence=True,
            decision_cid="",
        )
        return LeaseFenceGateDecision(
            **{**decision.__dict__, "decision_cid": _cid(decision.preimage())}
        )

    decision = LeaseFenceGateDecision(
        allowed=True,
        reason_code="lease_fence_ok",
        requires_lease=True,
        requires_fence=True,
        decision_cid="",
    )
    return LeaseFenceGateDecision(
        **{**decision.__dict__, "decision_cid": _cid(decision.preimage())}
    )


@dataclass(frozen=True)
class QueueAccounting:
    """Counts of tasks in each conserved bucket."""

    admitted: int = 0
    reserved: int = 0
    running: int = 0
    retrying: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        for name in CONSERVED_BUCKETS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchedulerInvariantError(
                    f"queue bucket {name} must be a non-negative integer",
                    reason_code="invalid_queue_bucket",
                    details={"bucket": name, "value": value},
                )

    def total(self) -> int:
        return sum(getattr(self, name) for name in CONSERVED_BUCKETS)

    def active(self) -> int:
        return sum(getattr(self, name) for name in ACTIVE_BUCKETS)

    def terminal(self) -> int:
        return sum(getattr(self, name) for name in TERMINAL_BUCKETS)

    def bucket(self, name: QueueBucket | str) -> int:
        key = name.value if isinstance(name, QueueBucket) else name
        if key not in CONSERVED_BUCKETS:
            raise SchedulerInvariantError(
                f"unknown queue bucket: {key}",
                reason_code="unknown_queue_bucket",
                details={"bucket": key},
            )
        return int(getattr(self, key))

    def with_delta(self, *, dec: QueueBucket | None = None, inc: QueueBucket | None = None) -> "QueueAccounting":
        values = {name: getattr(self, name) for name in CONSERVED_BUCKETS}
        if dec is not None:
            values[dec.value] = values[dec.value] - 1
            if values[dec.value] < 0:
                raise SchedulerInvariantError(
                    f"cannot decrement empty bucket {dec.value}",
                    reason_code="queue_underflow",
                    details={"bucket": dec.value},
                )
        if inc is not None:
            values[inc.value] = values[inc.value] + 1
        return QueueAccounting(**values)

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in CONSERVED_BUCKETS}


@dataclass(frozen=True)
class ScheduledTask:
    """One unit of admitted work with optional lease ownership."""

    task_id: str
    bucket: QueueBucket
    owner_id: str = ""
    fencing_token: int = 0
    identity_key: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            raise SchedulerContractError(
                "task_id is required",
                reason_code="invalid_task_id",
            )
        if not self.identity_key:
            object.__setattr__(self, "identity_key", self.task_id)


@dataclass(frozen=True)
class InterleavingStep:
    """One step in a bounded interleaving fixture."""

    task_id: str
    transition: TransitionKind
    actor_id: str = ""
    presented_fencing_token: int = 0
    # For ADMIT: whether this is a new admission (increases total).
    new_admission: bool = False


@dataclass(frozen=True)
class InterleavingTrace:
    """Result of checking a bounded interleaving against conservation."""

    conserved: bool
    steps_applied: int
    initial_total: int
    final_total: int
    terminal_total: int
    active_total: int
    reason_code: str
    final_accounting: QueueAccounting
    duplicate_task_ids: tuple[str, ...]
    lost_task_ids: tuple[str, ...]
    trace_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": INTERLEAVING_TRACE_SCHEMA,
            "conserved": self.conserved,
            "stepsApplied": self.steps_applied,
            "initialTotal": self.initial_total,
            "finalTotal": self.final_total,
            "terminalTotal": self.terminal_total,
            "activeTotal": self.active_total,
            "reasonCode": self.reason_code,
            "finalAccounting": self.final_accounting.to_dict(),
            "duplicateTaskIds": list(self.duplicate_task_ids),
            "lostTaskIds": list(self.lost_task_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "traceCid": self.trace_cid}


def _tasks_to_accounting(tasks: Mapping[str, ScheduledTask]) -> QueueAccounting:
    counts = {name: 0 for name in CONSERVED_BUCKETS}
    for task in tasks.values():
        counts[task.bucket.value] += 1
    return QueueAccounting(**counts)


def apply_interleaving(
    initial_tasks: Sequence[ScheduledTask],
    steps: Sequence[InterleavingStep],
    *,
    lease_state: LeaseFenceState | None = None,
    surface: SchedulerSurface | None = None,
    max_steps: int = DEFAULT_MAX_INTERLEAVING_STEPS,
    concurrency_bound: int | None = None,
) -> InterleavingTrace:
    """Apply a bounded interleaving and prove queue conservation.

    Conservation rules:

    * Every non-admit step moves exactly one existing task between buckets.
    * Admit steps introduce exactly one new task id (no duplicates).
    * Terminal and active counts always sum to the admitted total.
    * Retry/cancel/crash_recover never invent a second identity for a task.
    * Effectful steps are gated by lease/fence when a lease state is supplied.
    """

    if max_steps <= 0:
        raise SchedulerContractError(
            "max_steps must be positive",
            reason_code="invalid_interleaving_bound",
            details={"max_steps": max_steps},
        )
    if len(steps) > max_steps:
        raise SchedulerInvariantError(
            "interleaving exceeds modeled step bound",
            reason_code="interleaving_bound_exceeded",
            details={"steps": len(steps), "max_steps": max_steps},
        )

    tasks: dict[str, ScheduledTask] = {}
    identity_index: dict[str, str] = {}
    for task in initial_tasks:
        if task.task_id in tasks:
            raise SchedulerInvariantError(
                f"duplicate initial task id: {task.task_id}",
                reason_code="duplicate_task",
                details={"taskId": task.task_id},
            )
        if task.identity_key in identity_index:
            raise SchedulerInvariantError(
                f"duplicate initial identity key: {task.identity_key}",
                reason_code="duplicate_identity",
                details={"identityKey": task.identity_key},
            )
        tasks[task.task_id] = task
        identity_index[task.identity_key] = task.task_id

    initial_total = len(tasks)
    bound = concurrency_bound
    if bound is None and surface is not None:
        bound = surface.concurrency_bound

    applied = 0
    try:
        for step in steps:
            if step.transition is TransitionKind.OBSERVE:
                applied += 1
                continue

            if lease_state is not None or (
                surface is not None and step.transition in EFFECTFUL_TRANSITIONS
            ):
                gate_state = lease_state or LeaseFenceState(
                    owner_id="",
                    fencing_token=0,
                    lease_expires_at_ms=0,
                    now_ms=1,
                    held=False,
                )
                decision = evaluate_lease_fence_gate(
                    gate_state,
                    EffectAttempt(
                        transition=step.transition,
                        actor_id=step.actor_id or gate_state.owner_id,
                        presented_fencing_token=(
                            step.presented_fencing_token or gate_state.fencing_token
                        ),
                        task_id=step.task_id,
                        effectful=step.transition in EFFECTFUL_TRANSITIONS,
                    ),
                    surface=surface,
                )
                if not decision.allowed:
                    raise SchedulerInvariantError(
                        f"lease/fence denied {step.transition.value}",
                        reason_code=decision.reason_code,
                        details={
                            "taskId": step.task_id,
                            "transition": step.transition.value,
                        },
                    )

            is_admit = (
                step.transition is TransitionKind.ADMIT or step.new_admission
            )
            if is_admit:
                if step.task_id in tasks:
                    raise SchedulerInvariantError(
                        f"admit would duplicate task {step.task_id}",
                        reason_code="duplicate_task",
                        details={"taskId": step.task_id},
                    )
                if step.task_id in identity_index:
                    raise SchedulerInvariantError(
                        f"admit reuses identity key {step.task_id}",
                        reason_code="duplicate_identity",
                        details={"identityKey": step.task_id},
                    )
                if bound is not None:
                    active = sum(
                        1 for t in tasks.values() if t.bucket.value in ACTIVE_BUCKETS
                    )
                    if active >= bound:
                        raise SchedulerInvariantError(
                            "admission exceeds concurrency bound",
                            reason_code="concurrency_bound_exceeded",
                            details={"bound": bound, "active": active},
                        )
                tasks[step.task_id] = ScheduledTask(
                    task_id=step.task_id,
                    bucket=QueueBucket.ADMITTED,
                    owner_id=step.actor_id,
                    fencing_token=step.presented_fencing_token,
                    identity_key=step.task_id,
                )
                identity_index[step.task_id] = step.task_id
                applied += 1
                continue

            current = tasks.get(step.task_id)
            if current is None:
                raise SchedulerInvariantError(
                    f"transition targets unknown task {step.task_id}",
                    reason_code="lost_task",
                    details={
                        "taskId": step.task_id,
                        "transition": step.transition.value,
                    },
                )

            dest: QueueBucket | None = None
            for src, kind, dst in _LEGAL_TRANSITIONS:
                if src is current.bucket and kind is step.transition:
                    dest = dst
                    break
            if dest is None:
                raise SchedulerInvariantError(
                    f"illegal transition {current.bucket.value} --"
                    f"{step.transition.value}",
                    reason_code="illegal_transition",
                    details={
                        "taskId": step.task_id,
                        "from": current.bucket.value,
                        "transition": step.transition.value,
                    },
                )

            # Crash recovery and retry keep the same identity; never fork.
            if step.transition in {
                TransitionKind.RETRY,
                TransitionKind.CRASH_RECOVER,
                TransitionKind.CANCEL,
            }:
                if current.identity_key not in identity_index:
                    raise SchedulerInvariantError(
                        "recovery path lost task identity",
                        reason_code="lost_task_identity",
                        details={"taskId": step.task_id},
                    )
                if identity_index[current.identity_key] != step.task_id:
                    raise SchedulerInvariantError(
                        "recovery path would duplicate task identity",
                        reason_code="duplicate_identity",
                        details={
                            "taskId": step.task_id,
                            "identityKey": current.identity_key,
                        },
                    )

            tasks[step.task_id] = ScheduledTask(
                task_id=current.task_id,
                bucket=dest,
                owner_id=step.actor_id or current.owner_id,
                fencing_token=step.presented_fencing_token or current.fencing_token,
                identity_key=current.identity_key,
            )
            applied += 1
    except SchedulerInvariantError as exc:
        accounting = _tasks_to_accounting(tasks)
        lost_ids: tuple[str, ...] = ()
        if exc.reason_code in {"lost_task", "lost_task_identity"}:
            lost_id = str(exc.details.get("taskId") or "")
            if lost_id:
                lost_ids = (lost_id,)
        provisional = InterleavingTrace(
            conserved=False,
            steps_applied=applied,
            initial_total=initial_total,
            final_total=accounting.total(),
            terminal_total=accounting.terminal(),
            active_total=accounting.active(),
            reason_code=exc.reason_code,
            final_accounting=accounting,
            duplicate_task_ids=tuple(
                tid
                for tid in tasks
                if list(identity_index.values()).count(tid) > 1
            ),
            lost_task_ids=lost_ids,
            trace_cid="",
        )
        return InterleavingTrace(
            **{
                **provisional.__dict__,
                "trace_cid": _cid(provisional.preimage()),
            }
        )

    accounting = _tasks_to_accounting(tasks)
    final_total = accounting.total()
    # Count each applied step that introduced a task at most once.
    admitted_by_steps = 0
    for step in steps[:applied]:
        if step.transition is TransitionKind.ADMIT or step.new_admission:
            admitted_by_steps += 1
    conserved = (
        final_total == initial_total + admitted_by_steps
        and accounting.active() + accounting.terminal() == final_total
        and len(tasks) == len(identity_index)
        and len(set(identity_index.values())) == len(identity_index)
    )
    reason = "queue_conserved" if conserved else "queue_not_conserved"
    provisional = InterleavingTrace(
        conserved=conserved,
        steps_applied=applied,
        initial_total=initial_total,
        final_total=final_total,
        terminal_total=accounting.terminal(),
        active_total=accounting.active(),
        reason_code=reason,
        final_accounting=accounting,
        duplicate_task_ids=(),
        lost_task_ids=(),
        trace_cid="",
    )
    return InterleavingTrace(
        **{**provisional.__dict__, "trace_cid": _cid(provisional.preimage())}
    )


@dataclass(frozen=True)
class RecoveryPath:
    """A retry, cancel, or crash recovery path under one scheduler version."""

    path_id: str
    kind: str  # retry | cancel | crash
    scheduler_id: str
    version: str
    steps: tuple[InterleavingStep, ...]
    path_cid: str = ""

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_PATH_SCHEMA,
            "pathId": self.path_id,
            "kind": self.kind,
            "schedulerId": self.scheduler_id,
            "version": self.version,
            "steps": [
                {
                    "taskId": step.task_id,
                    "transition": step.transition.value,
                    "actorId": step.actor_id,
                    "presentedFencingToken": step.presented_fencing_token,
                    "newAdmission": step.new_admission,
                }
                for step in self.steps
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        cid = self.path_cid or _cid(self.preimage())
        return {**self.preimage(), "pathCid": cid}


def check_recovery_path(
    path: RecoveryPath,
    *,
    initial_tasks: Sequence[ScheduledTask],
    lease_state: LeaseFenceState | None = None,
    surface: SchedulerSurface | None = None,
) -> InterleavingTrace:
    """Prove a recovery path neither duplicates nor loses tasks."""

    if path.kind not in {"retry", "cancel", "crash"}:
        raise SchedulerContractError(
            "recovery path kind must be retry, cancel, or crash",
            reason_code="invalid_recovery_kind",
            details={"kind": path.kind},
        )
    # Recovery paths must not introduce a second identity for the same work.
    for step in path.steps:
        if step.new_admission and path.kind in {"retry", "cancel", "crash"}:
            # Crash/retry recovery re-admits only via CRASH_RECOVER on the
            # same task id, never a fresh admit with a new id.
            if step.transition is TransitionKind.ADMIT:
                raise SchedulerInvariantError(
                    "recovery path cannot admit a new task identity",
                    reason_code="recovery_duplicate_admit",
                    details={"pathId": path.path_id, "taskId": step.task_id},
                )
    return apply_interleaving(
        initial_tasks,
        path.steps,
        lease_state=lease_state,
        surface=surface,
    )


def enumerate_bounded_interleavings(
    task_ids: Sequence[str],
    *,
    max_branching: int = DEFAULT_MAX_INTERLEAVING_BRANCHES,
    concurrency_bound: int = 2,
) -> tuple[InterleavingTrace, ...]:
    """Enumerate small interleavings and require every branch conserve work.

    The generator explores admit → reserve → start → {complete, fail, cancel,
    retry→start→complete} paths under ``concurrency_bound``.  It is a fixture
    helper for contract tests, not a general model checker.
    """

    if not task_ids:
        raise SchedulerContractError(
            "task_ids must be nonempty",
            reason_code="empty_interleaving",
        )
    if concurrency_bound <= 0:
        raise SchedulerContractError(
            "concurrency_bound must be positive",
            reason_code="invalid_concurrency_bound",
        )

    terminal_choices: tuple[TransitionKind, ...] = (
        TransitionKind.COMPLETE,
        TransitionKind.FAIL,
        TransitionKind.CANCEL,
    )
    traces: list[InterleavingTrace] = []
    branches = 0

    def explore(
        remaining: list[str],
        active: dict[str, QueueBucket],
        steps: list[InterleavingStep],
    ) -> None:
        nonlocal branches
        if branches >= max_branching:
            raise SchedulerInvariantError(
                "interleaving branch bound exceeded",
                reason_code="interleaving_branch_bound_exceeded",
                details={"max_branching": max_branching},
            )
        if not remaining and not active:
            initial: list[ScheduledTask] = []
            # Rebuild from steps only (empty initial); conservation includes admits.
            traces.append(
                apply_interleaving(
                    initial,
                    steps,
                    concurrency_bound=concurrency_bound,
                    max_steps=DEFAULT_MAX_INTERLEAVING_STEPS,
                )
            )
            branches += 1
            return

        # Prefer progressing active tasks so terminals drain.
        for task_id, bucket in list(active.items()):
            if bucket is QueueBucket.ADMITTED:
                nxt = list(steps) + [
                    InterleavingStep(task_id=task_id, transition=TransitionKind.RESERVE)
                ]
                new_active = dict(active)
                new_active[task_id] = QueueBucket.RESERVED
                explore(remaining, new_active, nxt)
                return
            if bucket is QueueBucket.RESERVED:
                nxt = list(steps) + [
                    InterleavingStep(task_id=task_id, transition=TransitionKind.START)
                ]
                new_active = dict(active)
                new_active[task_id] = QueueBucket.RUNNING
                explore(remaining, new_active, nxt)
                return
            if bucket is QueueBucket.RUNNING:
                for choice in terminal_choices:
                    nxt = list(steps) + [
                        InterleavingStep(task_id=task_id, transition=choice)
                    ]
                    new_active = dict(active)
                    del new_active[task_id]
                    explore(remaining, new_active, nxt)
                # Also explore one retry cycle that must reconverge.
                retry_steps = list(steps) + [
                    InterleavingStep(task_id=task_id, transition=TransitionKind.RETRY),
                    InterleavingStep(task_id=task_id, transition=TransitionKind.START),
                    InterleavingStep(
                        task_id=task_id, transition=TransitionKind.COMPLETE
                    ),
                ]
                new_active = dict(active)
                del new_active[task_id]
                explore(remaining, new_active, retry_steps)
                return
            if bucket is QueueBucket.RETRYING:
                nxt = list(steps) + [
                    InterleavingStep(task_id=task_id, transition=TransitionKind.START)
                ]
                new_active = dict(active)
                new_active[task_id] = QueueBucket.RUNNING
                explore(remaining, new_active, nxt)
                return

        if remaining and len(active) < concurrency_bound:
            task_id = remaining[0]
            nxt = list(steps) + [
                InterleavingStep(
                    task_id=task_id,
                    transition=TransitionKind.ADMIT,
                    new_admission=True,
                )
            ]
            new_remaining = remaining[1:]
            new_active = dict(active)
            new_active[task_id] = QueueBucket.ADMITTED
            explore(new_remaining, new_active, nxt)

    explore(list(task_ids), {}, [])
    if not traces:
        raise SchedulerInvariantError(
            "no interleavings were generated",
            reason_code="empty_interleaving_set",
        )
    for trace in traces:
        if not trace.conserved:
            raise SchedulerInvariantError(
                "bounded interleaving failed conservation",
                reason_code=trace.reason_code,
                details=trace.to_dict(),
            )
    return tuple(traces)


# ---------------------------------------------------------------------------
# Parsing / building
# ---------------------------------------------------------------------------


def _parse_authority(
    raw: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> SchedulerAuthority:
    kind = _enum(SchedulerAuthorityKind, raw.get("kind"), "authority.kind")
    adapter_contract_id = str(raw.get("adapterContractId") or "")
    if kind is SchedulerAuthorityKind.PROVED_ADAPTER and not adapter_contract_id:
        raise SchedulerAuthorityError(
            "proved adapters require an adapterContractId",
            reason_code="adapter_contract_missing",
        )
    # Name-only joins are forbidden: adapter contracts must be explicit ids.
    if adapter_contract_id and adapter_contract_id.startswith("name:"):
        raise SchedulerAuthorityError(
            "adapter contracts cannot be name-only joins",
            reason_code="name_only_adapter_forbidden",
            details={"adapterContractId": adapter_contract_id},
        )
    provisional = SchedulerAuthority(
        kind=kind,
        canonical_scheduler_id=_text(
            raw.get("canonicalSchedulerId"),
            "authority.canonicalSchedulerId",
        ),
        decision=_text(raw.get("decision"), "authority.decision"),
        adapter_contract_id=adapter_contract_id,
        version=_text(raw.get("version"), "authority.version"),
        source_path=_source_path(raw.get("sourcePath"), "authority.sourcePath"),
        authority_cid="",
    )
    return SchedulerAuthority(
        **{
            **provisional.__dict__,
            "authority_cid": _verified_cid(
                raw,
                "authorityCid",
                provisional.preimage(),
                require_stored_cids=require_stored_cids,
            ),
        }
    )


def _parse_surface(
    raw: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> SchedulerSurface:
    provisional = SchedulerSurface(
        scheduler_id=_text(raw.get("schedulerId"), "schedulerId"),
        display_name=_text(raw.get("displayName"), "displayName"),
        role=_enum(SchedulerRole, raw.get("role"), "role"),
        implementation_symbol=_text(
            raw.get("implementationSymbol"),
            "implementationSymbol",
        ),
        source_path=_source_path(raw.get("sourcePath"), "sourcePath"),
        package_id=_text(raw.get("packageId"), "packageId"),
        version=_text(raw.get("version"), "version"),
        concurrency_bound=_positive_int(
            raw.get("concurrencyBound"),
            "concurrencyBound",
        ),
        supports_lease=_bool(raw.get("supportsLease"), "supportsLease"),
        supports_fence=_bool(raw.get("supportsFence"), "supportsFence"),
        authority=_parse_authority(
            _mapping(raw.get("authority"), "authority"),
            require_stored_cids=require_stored_cids,
        ),
        surface_cid="",
    )
    return SchedulerSurface(
        **{
            **provisional.__dict__,
            "surface_cid": _verified_cid(
                raw,
                "surfaceCid",
                provisional.preimage(),
                require_stored_cids=require_stored_cids,
            ),
        }
    )


def _parse_relation(
    raw: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> SchedulerRelation:
    adapter_contract_id = str(raw.get("adapterContractId") or "")
    kind = _enum(SchedulerRelationKind, raw.get("kind"), "relation.kind")
    if kind is SchedulerRelationKind.EXPLICIT_ADAPTER and not adapter_contract_id:
        raise SchedulerAuthorityError(
            "explicit adapter relations require adapterContractId",
            reason_code="adapter_contract_missing",
        )
    if adapter_contract_id.startswith("name:"):
        raise SchedulerAuthorityError(
            "relations cannot use name-only adapter contracts",
            reason_code="name_only_adapter_forbidden",
            details={"adapterContractId": adapter_contract_id},
        )
    provisional = SchedulerRelation(
        relation_id=_text(raw.get("relationId"), "relationId"),
        kind=kind,
        source_scheduler_id=_text(raw.get("sourceSchedulerId"), "sourceSchedulerId"),
        target_scheduler_id=_text(raw.get("targetSchedulerId"), "targetSchedulerId"),
        source_version=_text(raw.get("sourceVersion"), "sourceVersion"),
        target_version=_text(raw.get("targetVersion"), "targetVersion"),
        adapter_contract_id=adapter_contract_id,
        proof_binding=_text(raw.get("proofBinding"), "proofBinding"),
        relation_cid="",
    )
    return SchedulerRelation(
        **{
            **provisional.__dict__,
            "relation_cid": _verified_cid(
                raw,
                "relationCid",
                provisional.preimage(),
                require_stored_cids=require_stored_cids,
            ),
        }
    )


def _parse_invariant(
    raw: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> SchedulerInvariant:
    scheduler_ids = tuple(
        _text(item, "invariant.schedulerIds[]")
        for item in _sequence(raw.get("schedulerIds"), "schedulerIds")
    )
    if not scheduler_ids:
        raise SchedulerContractError(
            "invariant must bind at least one scheduler id",
            reason_code="empty_invariant_binding",
        )
    provisional = SchedulerInvariant(
        invariant_id=_text(raw.get("invariantId"), "invariantId"),
        family=_enum(InvariantFamily, raw.get("family"), "family"),
        scheduler_ids=scheduler_ids,
        statement=_text(raw.get("statement"), "statement"),
        bound=_nonneg_int(raw.get("bound"), "bound"),
        version=_text(raw.get("version"), "version"),
        invariant_cid="",
    )
    return SchedulerInvariant(
        **{
            **provisional.__dict__,
            "invariant_cid": _verified_cid(
                raw,
                "invariantCid",
                provisional.preimage(),
                require_stored_cids=require_stored_cids,
            ),
        }
    )


def _validate_catalog_consistency(
    surfaces: Sequence[SchedulerSurface],
    relations: Sequence[SchedulerRelation],
    invariants: Sequence[SchedulerInvariant],
) -> None:
    by_id: dict[str, SchedulerSurface] = {}
    for surface in surfaces:
        if surface.scheduler_id in by_id:
            raise DuplicateSchedulerError(
                f"duplicate scheduler id: {surface.scheduler_id}",
                reason_code="duplicate_scheduler_id",
                details={"schedulerId": surface.scheduler_id},
            )
        by_id[surface.scheduler_id] = surface

    # Every surface must be one of the four authority kinds (enum already).
    # Canonical surfaces are self-rooted; adapters/legacy/contradictory point
    # at a canonical of the same role (contradictory may point at self).
    for surface in surfaces:
        authority = surface.authority
        if authority.source_path != surface.source_path:
            raise SchedulerAuthorityError(
                "authority source must bind the surface source",
                reason_code="authority_source_mismatch",
                details={"schedulerId": surface.scheduler_id},
            )
        if authority.version != surface.version:
            raise SchedulerAuthorityError(
                "authority version must bind the surface version",
                reason_code="authority_version_mismatch",
                details={"schedulerId": surface.scheduler_id},
            )
        canonical = by_id.get(authority.canonical_scheduler_id)
        if authority.kind is SchedulerAuthorityKind.CANONICAL:
            if canonical is not surface:
                raise SchedulerAuthorityError(
                    "canonical authority must be self-rooted",
                    reason_code="canonical_not_self_rooted",
                    details={"schedulerId": surface.scheduler_id},
                )
            continue
        if authority.kind is SchedulerAuthorityKind.CONTRADICTORY:
            # Contradictions remain open; they may self-root or name the peer.
            continue
        if authority.kind is SchedulerAuthorityKind.LEGACY_ONLY:
            # Legacy-only surfaces may self-root or name the canonical
            # consumer that still imports them.  Role matching is not
            # required because legacy retention is compatibility-only.
            if (
                authority.canonical_scheduler_id != surface.scheduler_id
                and canonical is None
            ):
                raise SchedulerAuthorityError(
                    f"legacy authority target missing for {surface.scheduler_id}",
                    reason_code="canonical_target_missing",
                    details={
                        "schedulerId": surface.scheduler_id,
                        "canonicalSchedulerId": authority.canonical_scheduler_id,
                    },
                )
            continue
        if canonical is None:
            raise SchedulerAuthorityError(
                f"authority target missing for {surface.scheduler_id}",
                reason_code="canonical_target_missing",
                details={
                    "schedulerId": surface.scheduler_id,
                    "canonicalSchedulerId": authority.canonical_scheduler_id,
                },
            )
        if canonical.authority.kind is not SchedulerAuthorityKind.CANONICAL:
            raise SchedulerAuthorityError(
                "authority target must itself be canonical",
                reason_code="authority_target_not_canonical",
                details={
                    "schedulerId": surface.scheduler_id,
                    "canonicalSchedulerId": authority.canonical_scheduler_id,
                },
            )
        if canonical.role is not surface.role:
            raise SchedulerAuthorityError(
                "authority cannot cross scheduler roles",
                reason_code="authority_role_mismatch",
                details={
                    "schedulerId": surface.scheduler_id,
                    "role": surface.role.value,
                    "canonicalRole": canonical.role.value,
                },
            )
        if authority.kind is SchedulerAuthorityKind.PROVED_ADAPTER:
            if not authority.adapter_contract_id:
                raise SchedulerAuthorityError(
                    "proved adapter requires adapterContractId",
                    reason_code="adapter_contract_missing",
                    details={"schedulerId": surface.scheduler_id},
                )

    # At most one non-contradictory canonical per role.
    role_canonicals: dict[SchedulerRole, list[str]] = {}
    for surface in surfaces:
        if surface.authority.kind is SchedulerAuthorityKind.CANONICAL:
            role_canonicals.setdefault(surface.role, []).append(surface.scheduler_id)
    for role, ids in role_canonicals.items():
        if len(ids) > 1:
            raise SchedulerAuthorityError(
                f"multiple canonical surfaces for role {role.value}",
                reason_code="duplicate_role_canonical",
                details={"role": role.value, "schedulerIds": ids},
            )

    relation_ids: set[str] = set()
    for relation in relations:
        if relation.relation_id in relation_ids:
            raise DuplicateSchedulerError(
                f"duplicate relation id: {relation.relation_id}",
                reason_code="duplicate_relation_id",
                details={"relationId": relation.relation_id},
            )
        relation_ids.add(relation.relation_id)
        if relation.source_scheduler_id not in by_id:
            raise MissingSchedulerError(
                f"relation source missing: {relation.source_scheduler_id}",
                reason_code="relation_source_missing",
                details={"relationId": relation.relation_id},
            )
        if relation.target_scheduler_id not in by_id:
            raise MissingSchedulerError(
                f"relation target missing: {relation.target_scheduler_id}",
                reason_code="relation_target_missing",
                details={"relationId": relation.relation_id},
            )
        source = by_id[relation.source_scheduler_id]
        target = by_id[relation.target_scheduler_id]
        if relation.source_version != source.version:
            raise SchedulerAuthorityError(
                "relation source version must bind the surface version",
                reason_code="relation_version_mismatch",
                details={"relationId": relation.relation_id},
            )
        if relation.target_version != target.version:
            raise SchedulerAuthorityError(
                "relation target version must bind the surface version",
                reason_code="relation_version_mismatch",
                details={"relationId": relation.relation_id},
            )
        # Shared display names never prove a relation; proof_binding required.
        if not relation.proof_binding or relation.proof_binding.startswith("name:"):
            raise SchedulerAuthorityError(
                "relations require a non-name proof binding",
                reason_code="name_only_relation_forbidden",
                details={"relationId": relation.relation_id},
            )

    invariant_ids: set[str] = set()
    for invariant in invariants:
        if invariant.invariant_id in invariant_ids:
            raise DuplicateSchedulerError(
                f"duplicate invariant id: {invariant.invariant_id}",
                reason_code="duplicate_invariant_id",
                details={"invariantId": invariant.invariant_id},
            )
        invariant_ids.add(invariant.invariant_id)
        for scheduler_id in invariant.scheduler_ids:
            if scheduler_id not in by_id:
                raise MissingSchedulerError(
                    f"invariant binds unknown scheduler: {scheduler_id}",
                    reason_code="invariant_scheduler_missing",
                    details={
                        "invariantId": invariant.invariant_id,
                        "schedulerId": scheduler_id,
                    },
                )


def build_scheduler_contract_catalog(
    payload: Mapping[str, Any],
    *,
    require_stored_cids: bool = False,
) -> SchedulerContractCatalog:
    """Validate and normalize a scheduler contract catalog mapping."""

    if payload.get("schema") not in (None, SCHEDULER_CATALOG_SCHEMA):
        raise SchedulerContractError(
            "unsupported scheduler catalog schema",
            reason_code="unsupported_catalog_schema",
            details={"schema": payload.get("schema")},
        )

    surfaces = tuple(
        _parse_surface(
            _mapping(item, "surfaces[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("surfaces"), "surfaces")
    )
    relations = tuple(
        _parse_relation(
            _mapping(item, "relations[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("relations") or (), "relations")
    )
    invariants = tuple(
        _parse_invariant(
            _mapping(item, "invariants[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("invariants") or (), "invariants")
    )

    if not surfaces:
        raise MissingSchedulerError(
            "scheduler catalog requires at least one surface",
            reason_code="empty_scheduler_catalog",
        )

    _validate_catalog_consistency(surfaces, relations, invariants)

    runtime_component_id = str(payload.get("runtimeComponentId") or "scheduler")
    if not runtime_component_id:
        raise SchedulerContractError(
            "runtimeComponentId must be a nonempty string",
            reason_code="invalid_runtime_component_id",
        )

    provisional = SchedulerContractCatalog(
        surfaces=surfaces,
        relations=relations,
        invariants=invariants,
        runtime_component_id=runtime_component_id,
        catalog_cid="",
    )
    catalog_cid = _verified_cid(
        payload,
        "catalogCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return SchedulerContractCatalog(
        surfaces=surfaces,
        relations=relations,
        invariants=invariants,
        runtime_component_id=runtime_component_id,
        catalog_cid=catalog_cid,
    )


def materialize_scheduler_contract_catalog(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a fully CID-bound serializable form of an unmaterialized catalog."""

    return build_scheduler_contract_catalog(payload).to_dict()


def load_scheduler_contract_catalog(path: str | Path) -> SchedulerContractCatalog:
    """Load a fully materialized catalog, rejecting missing or stale CIDs."""

    catalog_path = Path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchedulerContractError(
            f"unable to load scheduler contract catalog: {catalog_path}",
            reason_code="catalog_load_failed",
            details={"path": str(catalog_path), "cause": repr(exc)},
        ) from exc
    return build_scheduler_contract_catalog(
        _mapping(payload, "catalog"),
        require_stored_cids=True,
    )


def validate_scheduler_sources(
    catalog: SchedulerContractCatalog,
    repository_root: str | Path,
    *,
    required_scheduler_ids: Iterable[str] | None = None,
) -> None:
    """Prove that declared source files and symbols exist under ``repository_root``.

    Paths are resolved relative to the repository root.  Package-relative
    accelerator paths (``ipfs_accelerate_py/...``) are also tried under
    ``external/ipfs_accelerate/`` for monorepo layouts.
    """

    root = Path(repository_root)
    required = set(required_scheduler_ids or ())
    for surface in catalog.surfaces:
        if required and surface.scheduler_id not in required:
            continue
        candidates = [
            root / surface.source_path,
            root / "external" / "ipfs_accelerate" / surface.source_path,
            root / "swissknife" / surface.source_path,
        ]
        candidate = next((path for path in candidates if path.is_file()), None)
        if candidate is None:
            raise SchedulerSourceError(
                f"scheduler source does not exist: {surface.source_path}",
                reason_code="scheduler_source_missing",
                details={
                    "schedulerId": surface.scheduler_id,
                    "sourcePath": surface.source_path,
                },
            )
        text = candidate.read_text(encoding="utf-8")
        if surface.implementation_symbol not in text:
            raise SchedulerSourceError(
                f"scheduler symbol does not exist: {surface.implementation_symbol}",
                reason_code="scheduler_symbol_missing",
                details={
                    "schedulerId": surface.scheduler_id,
                    "sourcePath": surface.source_path,
                    "symbol": surface.implementation_symbol,
                },
            )


# ---------------------------------------------------------------------------
# Default inventory for the accelerator / SwissKnife scheduler surfaces
# ---------------------------------------------------------------------------


def default_scheduler_inventory() -> dict[str, Any]:
    """Return the unmaterialized default inventory of known scheduler surfaces.

    Authority decisions are explicit and version-bound.  Shared symbols such
    as ``MerkleClock`` or ``schedule`` never create a relation by themselves.
    """

    return {
        "schema": SCHEDULER_CATALOG_SCHEMA,
        "catalogVersion": CATALOG_VERSION,
        "runtimeComponentId": "scheduler",
        "surfaces": [
            {
                "schedulerId": "deterministic-ownership-v1",
                "displayName": "Deterministic P2P ownership scheduler",
                "role": SchedulerRole.DETERMINISTIC_OWNERSHIP.value,
                "implementationSymbol": "select_owner_peer",
                "sourcePath": (
                    "ipfs_accelerate_py/p2p_tasks/deterministic_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 1,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "deterministic-ownership-v1",
                    "decision": "canonical_deterministic_ownership_clock",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/p2p_tasks/deterministic_scheduler.py"
                    ),
                },
            },
            {
                "schedulerId": "legacy-p2p-workflow-v1",
                "displayName": "Legacy P2P workflow scheduler",
                "role": SchedulerRole.LEGACY_WORKFLOW.value,
                "implementationSymbol": "P2PWorkflowScheduler",
                "sourcePath": "ipfs_accelerate_py/p2p_workflow_scheduler.py",
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 8,
                "supportsLease": True,
                "supportsFence": False,
                "authority": {
                    "kind": SchedulerAuthorityKind.LEGACY_ONLY.value,
                    "canonicalSchedulerId": "mcp-workflow-adapter-v1",
                    "decision": "legacy_workflow_retained_for_compatibility",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": "ipfs_accelerate_py/p2p_workflow_scheduler.py",
                },
            },
            {
                "schedulerId": "mcp-workflow-adapter-v1",
                "displayName": "MCP++ workflow scheduler adapter",
                "role": SchedulerRole.MCP_WORKFLOW.value,
                "implementationSymbol": "create_workflow_scheduler",
                "sourcePath": (
                    "ipfs_accelerate_py/mcp_server/mcplusplus/workflow_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 8,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "mcp-workflow-adapter-v1",
                    "decision": "canonical_mcp_workflow_surface",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/mcp_server/mcplusplus/"
                        "workflow_scheduler.py"
                    ),
                },
            },
            {
                "schedulerId": "mcp-risk-v1",
                "displayName": "MCP++ risk frontier scheduler",
                "role": SchedulerRole.MCP_RISK.value,
                "implementationSymbol": "RiskScheduler",
                "sourcePath": (
                    "ipfs_accelerate_py/mcp_server/mcplusplus/risk_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 16,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "mcp-risk-v1",
                    "decision": "canonical_mcp_risk_frontier",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/mcp_server/mcplusplus/risk_scheduler.py"
                    ),
                },
            },
            {
                "schedulerId": "swissknife-mcp-scheduler-v1",
                "displayName": "SwissKnife MCPScheduler",
                "role": SchedulerRole.SWISSKNIFE_MCP.value,
                "implementationSymbol": "MCPScheduler",
                "sourcePath": "src/services/mcp/mcp-scheduler.ts",
                "packageId": "swissknife",
                "version": "1",
                "concurrencyBound": 8,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "swissknife-mcp-scheduler-v1",
                    "decision": "canonical_swissknife_mcp_scheduler",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": "src/services/mcp/mcp-scheduler.ts",
                },
            },
            {
                "schedulerId": "supervisor-resource-v1",
                "displayName": "Supervisor runtime resource scheduler",
                "role": SchedulerRole.SUPERVISOR_RESOURCE.value,
                "implementationSymbol": "ResourceScheduler",
                "sourcePath": (
                    "ipfs_accelerate_py/agent_supervisor/runtime/"
                    "resource_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 32,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "supervisor-resource-v1",
                    "decision": "canonical_supervisor_resource_admission",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/agent_supervisor/runtime/"
                        "resource_scheduler.py"
                    ),
                },
            },
            {
                "schedulerId": "supervisor-resource-usage-adapter-v1",
                "displayName": "Endpoint-usage resource admission adapter",
                "role": SchedulerRole.SUPERVISOR_RESOURCE.value,
                "implementationSymbol": "UsageAwareResourceScheduler",
                "sourcePath": (
                    "ipfs_accelerate_py/agent_supervisor/resource_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 32,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.PROVED_ADAPTER.value,
                    "canonicalSchedulerId": "supervisor-resource-v1",
                    "decision": "explicit_usage_projection_adapter",
                    "adapterContractId": (
                        "adapter:endpoint-usage-fair-resource-admission.v1"
                    ),
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/agent_supervisor/resource_scheduler.py"
                    ),
                },
            },
            {
                "schedulerId": "supervisor-provider-batch-v1",
                "displayName": "Supervisor provider batch scheduler",
                "role": SchedulerRole.SUPERVISOR_PROVIDER.value,
                "implementationSymbol": "ProviderBatchScheduler",
                "sourcePath": (
                    "ipfs_accelerate_py/agent_supervisor/runtime/"
                    "provider_batch_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 16,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "supervisor-provider-batch-v1",
                    "decision": "canonical_provider_batch_stream",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/agent_supervisor/runtime/"
                        "provider_batch_scheduler.py"
                    ),
                },
            },
            {
                "schedulerId": "validation-scheduler-v1",
                "displayName": "Validation DAG scheduler",
                "role": SchedulerRole.VALIDATION.value,
                "implementationSymbol": "ValidationResultCache",
                "sourcePath": (
                    "ipfs_accelerate_py/agent_supervisor/validation/"
                    "validation_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 8,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "validation-scheduler-v1",
                    "decision": "canonical_validation_stage_scheduler",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/agent_supervisor/validation/"
                        "validation_scheduler.py"
                    ),
                },
            },
            {
                "schedulerId": "proof-scheduler-v1",
                "displayName": "Proof step scheduler",
                "role": SchedulerRole.PROOF.value,
                "implementationSymbol": "ProofSchedulerConfig",
                "sourcePath": (
                    "ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py"
                ),
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "concurrencyBound": 8,
                "supportsLease": True,
                "supportsFence": True,
                "authority": {
                    "kind": SchedulerAuthorityKind.CANONICAL.value,
                    "canonicalSchedulerId": "proof-scheduler-v1",
                    "decision": "canonical_proof_step_scheduler",
                    "adapterContractId": "",
                    "version": "1",
                    "sourcePath": (
                        "ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py"
                    ),
                },
            },
        ],
        "relations": [
            {
                "relationId": "rel-mcp-workflow-delegates-legacy-v1",
                "kind": SchedulerRelationKind.LEGACY_DELEGATION.value,
                "sourceSchedulerId": "mcp-workflow-adapter-v1",
                "targetSchedulerId": "legacy-p2p-workflow-v1",
                "sourceVersion": "1",
                "targetVersion": "1",
                "adapterContractId": "adapter:mcp-workflow-imports-p2p-workflow.v1",
                "proofBinding": (
                    "import:ipfs_accelerate_py.p2p_workflow_scheduler."
                    "P2PWorkflowScheduler"
                ),
            },
            {
                "relationId": "rel-resource-usage-adapter-v1",
                "kind": SchedulerRelationKind.EXPLICIT_ADAPTER.value,
                "sourceSchedulerId": "supervisor-resource-usage-adapter-v1",
                "targetSchedulerId": "supervisor-resource-v1",
                "sourceVersion": "1",
                "targetVersion": "1",
                "adapterContractId": (
                    "adapter:endpoint-usage-fair-resource-admission.v1"
                ),
                "proofBinding": (
                    "reexport:runtime.resource_scheduler+"
                    "UsageAwareResourceScheduler"
                ),
            },
            {
                "relationId": "rel-swissknife-mcp-risk-adapter-v1",
                "kind": SchedulerRelationKind.EXPLICIT_ADAPTER.value,
                "sourceSchedulerId": "mcp-risk-v1",
                "targetSchedulerId": "swissknife-mcp-scheduler-v1",
                "sourceVersion": "1",
                "targetVersion": "1",
                "adapterContractId": "adapter:mcp-risk-frontier-to-swissknife.v1",
                "proofBinding": (
                    "formula:denial_ratio*0.6+obligations+disputed;"
                    "swissknife:RiskScorer.normaliseScore"
                ),
            },
            {
                "relationId": "rel-deterministic-clock-from-legacy-v1",
                "kind": SchedulerRelationKind.PROVED_EQUIVALENCE.value,
                "sourceSchedulerId": "deterministic-ownership-v1",
                "targetSchedulerId": "legacy-p2p-workflow-v1",
                "sourceVersion": "1",
                "targetVersion": "1",
                "adapterContractId": "adapter:merkle-clock-hamming-ownership.v1",
                "proofBinding": (
                    "semantics:MerkleClock.vector+select_owner_peer.hamming"
                ),
            },
        ],
        "invariants": [
            {
                "invariantId": "inv-canonical-selected-v1",
                "family": InvariantFamily.CANONICAL_IMPLEMENTATION_SELECTED.value,
                "schedulerIds": [
                    "deterministic-ownership-v1",
                    "mcp-workflow-adapter-v1",
                    "mcp-risk-v1",
                    "swissknife-mcp-scheduler-v1",
                    "supervisor-resource-v1",
                    "supervisor-provider-batch-v1",
                    "validation-scheduler-v1",
                    "proof-scheduler-v1",
                ],
                "statement": (
                    "Every public scheduler role resolves to one reviewed "
                    "canonical implementation or an explicit versioned adapter; "
                    "legacy-only surfaces never become primary."
                ),
                "bound": 1,
                "version": "1",
            },
            {
                "invariantId": "inv-lease-fence-before-effect-v1",
                "family": InvariantFamily.LEASE_FENCE_BEFORE_EFFECT.value,
                "schedulerIds": [
                    "supervisor-resource-v1",
                    "supervisor-resource-usage-adapter-v1",
                    "supervisor-provider-batch-v1",
                    "proof-scheduler-v1",
                    "validation-scheduler-v1",
                ],
                "statement": (
                    "Current lease ownership and fencing-token validation "
                    "dominate every distributed or repository-mutating effect."
                ),
                "bound": 1,
                "version": "1",
            },
            {
                "invariantId": "inv-queue-accounting-conserved-v1",
                "family": InvariantFamily.QUEUE_ACCOUNTING_CONSERVED.value,
                "schedulerIds": [
                    "mcp-workflow-adapter-v1",
                    "swissknife-mcp-scheduler-v1",
                    "supervisor-resource-v1",
                    "supervisor-provider-batch-v1",
                    "proof-scheduler-v1",
                ],
                "statement": (
                    "Admission, reservation, running, retry, completion, "
                    "cancellation, and failure transitions neither lose nor "
                    "duplicate work under the modeled concurrency bounds."
                ),
                "bound": 32,
                "version": "1",
            },
            {
                "invariantId": "inv-no-duplicate-or-lost-work-v1",
                "family": InvariantFamily.NO_DUPLICATE_OR_LOST_WORK.value,
                "schedulerIds": [
                    "supervisor-resource-v1",
                    "proof-scheduler-v1",
                    "validation-scheduler-v1",
                    "supervisor-provider-batch-v1",
                ],
                "statement": (
                    "Retry, cancel, and crash recovery paths preserve a single "
                    "task identity and reach a terminal outcome without "
                    "duplicate execution."
                ),
                "bound": 1,
                "version": "1",
            },
            {
                "invariantId": "inv-lifecycle-state-machine-v1",
                "family": InvariantFamily.LIFECYCLE_STATE_MACHINE_CONFORMS.value,
                "schedulerIds": [
                    "deterministic-ownership-v1",
                    "mcp-workflow-adapter-v1",
                    "supervisor-resource-v1",
                    "proof-scheduler-v1",
                ],
                "statement": (
                    "Scheduler transitions respect versioned preconditions, "
                    "terminal states, cancellation, timeout, retry, and "
                    "recovery semantics."
                ),
                "bound": 1,
                "version": "1",
            },
        ],
    }


def extract_scheduler_contracts(
    payload: Mapping[str, Any] | None = None,
    *,
    require_stored_cids: bool = False,
) -> SchedulerContractCatalog:
    """Extract and validate the scheduler contract catalog.

    When ``payload`` is omitted, the default accelerator/SwissKnife inventory
    is used.  Callers may supply a partial or full catalog mapping.
    """

    source = dict(payload) if payload is not None else default_scheduler_inventory()
    return build_scheduler_contract_catalog(
        source,
        require_stored_cids=require_stored_cids,
    )


def assert_authority_partition(catalog: SchedulerContractCatalog) -> None:
    """Fail closed unless every surface is in the closed authority partition."""

    allowed = set(SchedulerAuthorityKind)
    for surface in catalog.surfaces:
        if surface.authority.kind not in allowed:
            raise SchedulerAuthorityError(
                f"unknown authority kind for {surface.scheduler_id}",
                reason_code="unknown_authority_kind",
                details={
                    "schedulerId": surface.scheduler_id,
                    "kind": surface.authority.kind.value,
                },
            )
        if surface.authority.kind is SchedulerAuthorityKind.CONTRADICTORY:
            # Contradictions are typed and retained; they grant no authority.
            continue


def assert_lease_fence_invariants(catalog: SchedulerContractCatalog) -> None:
    """Require lease/fence-capable surfaces for effectful supervisor roles."""

    effectful_roles = {
        SchedulerRole.SUPERVISOR_RESOURCE,
        SchedulerRole.SUPERVISOR_PROVIDER,
        SchedulerRole.VALIDATION,
        SchedulerRole.PROOF,
    }
    for surface in catalog.surfaces:
        if surface.role not in effectful_roles:
            continue
        if surface.authority.kind is SchedulerAuthorityKind.LEGACY_ONLY:
            continue
        if surface.authority.kind is SchedulerAuthorityKind.CONTRADICTORY:
            continue
        if not surface.supports_lease or not surface.supports_fence:
            raise SchedulerInvariantError(
                f"effectful scheduler {surface.scheduler_id} lacks lease/fence",
                reason_code="lease_fence_required",
                details={
                    "schedulerId": surface.scheduler_id,
                    "supportsLease": surface.supports_lease,
                    "supportsFence": surface.supports_fence,
                },
            )


def classify_scheduler_authority(
    surface: SchedulerSurface,
    catalog: SchedulerContractCatalog,
) -> SchedulerAuthorityKind:
    """Return the closed authority classification for ``surface``.

    Shared display names are ignored.  Classification is derived only from the
    typed authority record already validated into the catalog.
    """

    resolved = catalog.surface(surface.scheduler_id)
    return resolved.authority.kind


@dataclass
class SchedulerContractExtractor:
    """Object-oriented facade matching other SCA extractors."""

    require_stored_cids: bool = False

    def extract(
        self,
        payload: Mapping[str, Any] | None = None,
    ) -> SchedulerContractCatalog:
        catalog = extract_scheduler_contracts(
            payload,
            require_stored_cids=self.require_stored_cids,
        )
        assert_authority_partition(catalog)
        assert_lease_fence_invariants(catalog)
        return catalog


__all__ = [
    "SCHEDULER_CONTRACT_CATALOG_INTERFACE",
    "SCHEDULER_CONTRACT_EXTRACTOR_INTERFACE",
    "CATALOG_VERSION",
    "SCHEDULER_AUTHORITY_SCHEMA",
    "SCHEDULER_SURFACE_SCHEMA",
    "SCHEDULER_RELATION_SCHEMA",
    "SCHEDULER_INVARIANT_SCHEMA",
    "SCHEDULER_CATALOG_SCHEMA",
    "LEASE_FENCE_GATE_SCHEMA",
    "INTERLEAVING_TRACE_SCHEMA",
    "RECOVERY_PATH_SCHEMA",
    "DEFAULT_MAX_INTERLEAVING_STEPS",
    "DEFAULT_MAX_INTERLEAVING_BRANCHES",
    "CONSERVED_BUCKETS",
    "TERMINAL_BUCKETS",
    "ACTIVE_BUCKETS",
    "EFFECTFUL_TRANSITIONS",
    "SchedulerContractError",
    "MissingSchedulerError",
    "DuplicateSchedulerError",
    "SchedulerAuthorityError",
    "SchedulerCIDError",
    "SchedulerInvariantError",
    "SchedulerSourceError",
    "SchedulerAuthorityKind",
    "SchedulerRole",
    "SchedulerRelationKind",
    "QueueBucket",
    "TransitionKind",
    "InvariantFamily",
    "SchedulerAuthority",
    "SchedulerSurface",
    "SchedulerRelation",
    "SchedulerInvariant",
    "SchedulerContractCatalog",
    "LeaseFenceState",
    "EffectAttempt",
    "LeaseFenceGateDecision",
    "QueueAccounting",
    "ScheduledTask",
    "InterleavingStep",
    "InterleavingTrace",
    "RecoveryPath",
    "SchedulerContractExtractor",
    "evaluate_lease_fence_gate",
    "apply_interleaving",
    "check_recovery_path",
    "enumerate_bounded_interleavings",
    "build_scheduler_contract_catalog",
    "materialize_scheduler_contract_catalog",
    "load_scheduler_contract_catalog",
    "validate_scheduler_sources",
    "default_scheduler_inventory",
    "extract_scheduler_contracts",
    "assert_authority_partition",
    "assert_lease_fence_invariants",
    "classify_scheduler_authority",
]
