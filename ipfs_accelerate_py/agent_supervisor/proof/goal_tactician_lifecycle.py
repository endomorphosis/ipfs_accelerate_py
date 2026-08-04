"""GoalTacticianSupervisorLifecycle@1 — restartable, fenced tactician execution.

Conflict policy (FVT-G051 / FVT-031): this module owns the tactician supervisor
lifecycle and restart integration surface.  It reuses scheduler, proof-carrying
planner, event-store, lease, resource, cache, and completion-authority concepts
as durable projections rather than re-implementing parallel persistence engines.

Acceptance invariants:

* restart replays identical authoritative state;
* stale workers / receipts cannot close or mutate a plan;
* cancellation / timeout / backpressure are durable control signals;
* changed trees invalidate scoped work under exact cache keys; and
* completion requires all selected graph leaves and counterexamples to carry
  adequate fresh receipts bound to the current tree and fencing epoch.

The lifecycle is the fenced mutation boundary for end-goal, proof-graph,
candidate, verification, counterexample, closure, and completion transitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Final

from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE: Final = (
    "GoalTacticianSupervisorLifecycle@1"
)
GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_VERSION: Final = "1.0.0"
GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-supervisor-lifecycle@1"
)
LIFECYCLE_PLAN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-lifecycle-plan@1"
)
LIFECYCLE_STATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-lifecycle-state@1"
)
LIFECYCLE_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-lifecycle-transition@1"
)
LIFECYCLE_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-lifecycle-cache-key@1"
)
LIFECYCLE_LEASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-lifecycle-lease@1"
)
LIFECYCLE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-lifecycle-receipt@1"
)
LIFECYCLE_COMPLETION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-tactician-lifecycle-completion@1"
)

DEFAULT_STATE_FILENAME: Final = "goal_tactician_lifecycle.state.json"
DEFAULT_JOURNAL_FILENAME: Final = "goal_tactician_lifecycle.journal.jsonl"
DEFAULT_LEASE_SECONDS: Final = 300
DEFAULT_MAX_RETRIES: Final = 3
DEFAULT_REQUIRED_ASSURANCE: Final = AssuranceLevel.KERNEL_VERIFIED

_AUTHORITY_BOOLEAN_CLAIMS: Final = frozenset(
    {
        "admission_claimed",
        "admitted",
        "authoritative",
        "can_mark_complete",
        "can_satisfy_completion",
        "complete",
        "completion_evidence",
        "implementation_conformant",
        "kernel_checked",
        "proof_success",
        "trusted",
        "verified",
    }
)


# ---------------------------------------------------------------------------
# Errors and enums
# ---------------------------------------------------------------------------


class GoalTacticianLifecycleError(ContractValidationError):
    """Raised when a lifecycle request, lease, or artifact is invalid."""


class StaleWorkerError(GoalTacticianLifecycleError):
    """A worker whose fencing token / lease is no longer authoritative."""


class StaleReceiptError(GoalTacticianLifecycleError):
    """A receipt that is not fresh for the current tree / fencing epoch."""


class LifecycleControlActiveError(GoalTacticianLifecycleError):
    """Mutation blocked by a durable cancel / timeout / backpressure signal."""


class LifecycleCompletionError(GoalTacticianLifecycleError):
    """Completion rejected because evidence is incomplete or stale."""


class LifecycleTransitionKind(str, Enum):
    """Ordered durable transitions under content identity."""

    END_GOAL = "end_goal"
    PROOF_GRAPH = "proof_graph"
    CANDIDATE = "candidate"
    VERIFICATION = "verification"
    COUNTEREXAMPLE = "counterexample"
    CLOSURE = "closure"
    COMPLETION = "completion"
    CONTROL = "control"
    TREE_INVALIDATION = "tree_invalidation"
    LEASE_ACQUIRE = "lease_acquire"
    LEASE_RELEASE = "lease_release"
    RECONCILE = "reconcile"


class LifecycleControlSignal(str, Enum):
    """Durable control-plane signals that fence further mutation."""

    NONE = "none"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BACKPRESSURE = "backpressure"


class LifecyclePlanStatus(str, Enum):
    """Authoritative plan status projection."""

    OPEN = "open"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class ReceiptKind(str, Enum):
    """Kinds of evidence bound into the lifecycle for completion."""

    GRAPH_LEAF = "graph_leaf"
    COUNTEREXAMPLE = "counterexample"
    VERIFICATION = "verification"
    CLOSURE = "closure"


class Freshness(str, Enum):
    """Receipt freshness relative to the current tree / fencing epoch."""

    FRESH = "fresh"
    STALE_TREE = "stale_tree"
    STALE_EPOCH = "stale_epoch"
    STALE_WORKER = "stale_worker"
    INADEQUATE = "inadequate"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise GoalTacticianLifecycleError(f"{field_name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise GoalTacticianLifecycleError(f"{field_name} is required")
    if "\x00" in result:
        raise GoalTacticianLifecycleError(
            f"{field_name} must not contain NUL bytes"
        )
    return result


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        values = value
    else:
        raise GoalTacticianLifecycleError("expected a sequence of strings")
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _positive(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GoalTacticianLifecycleError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _non_negative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GoalTacticianLifecycleError(f"{name} must be a non-negative integer")
    return value


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoalTacticianLifecycleError(f"{field_name} must be an object")
    return value


def _public_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if not str(key).startswith("_")
    }


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({item.value for item in kind}))
        raise GoalTacticianLifecycleError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _assurance(value: Any) -> AssuranceLevel:
    if isinstance(value, AssuranceLevel):
        return value
    raw = getattr(value, "value", value)
    try:
        return AssuranceLevel(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        raise GoalTacticianLifecycleError(
            "assurance must be one of: "
            + ", ".join(sorted({item.value for item in AssuranceLevel}))
        ) from exc


def _sha256_hex(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
    else:
        payload = canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _now_ms(clock: Callable[[], float] | None = None) -> int:
    source = clock or time.time
    return int(source() * 1000)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def claims_authority(value: Any) -> bool:
    """Return True when nested payload claims completion authority unsafely."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = str(raw_name).strip().casefold().replace("-", "_")
            if name in _AUTHORITY_BOOLEAN_CLAIMS and item not in (
                False,
                None,
                0,
                "",
            ):
                return True
            if claims_authority(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(claims_authority(item) for item in value)
    return False


# ---------------------------------------------------------------------------
# Exact cache key / resource policy / lease
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactLifecycleCacheKey:
    """Exact cache identity for one fenced tactician lifecycle plan.

    Required components: tree, end-goal, proof-graph, provider, version,
    policy, bounds, resource class, and retry bound.
    """

    SCHEMA: ClassVar[str] = LIFECYCLE_CACHE_KEY_SCHEMA

    tree_id: str
    end_goal_id: str
    proof_graph_id: str
    provider_id: str
    provider_version: str
    policy_id: str
    bounds: Mapping[str, Any]
    resource_class: str
    max_retries: int = DEFAULT_MAX_RETRIES
    selected_leaf_ids: tuple[str, ...] = ()
    selected_counterexample_ids: tuple[str, ...] = ()
    toolchain_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, field_name="tree_id")
        )
        object.__setattr__(
            self,
            "end_goal_id",
            _text(self.end_goal_id, field_name="end_goal_id"),
        )
        object.__setattr__(
            self,
            "proof_graph_id",
            _text(self.proof_graph_id, field_name="proof_graph_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _text(self.provider_id, field_name="provider_id"),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(self.provider_version, field_name="provider_version"),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        if not isinstance(self.bounds, Mapping) or not dict(self.bounds):
            raise GoalTacticianLifecycleError("bounds must be a non-empty object")
        object.__setattr__(self, "bounds", dict(self.bounds))
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class, field_name="resource_class"),
        )
        object.__setattr__(
            self, "max_retries", _non_negative(self.max_retries, "max_retries")
        )
        object.__setattr__(
            self, "selected_leaf_ids", _strings(self.selected_leaf_ids)
        )
        object.__setattr__(
            self,
            "selected_counterexample_ids",
            _strings(self.selected_counterexample_ids),
        )
        object.__setattr__(
            self, "toolchain_id", str(self.toolchain_id or "").strip()
        )

    @property
    def bound_digest(self) -> str:
        return f"sha256:{_sha256_hex(self.bounds)}"

    @property
    def key_id(self) -> str:
        return content_identity(self.to_dict(include_schema=False))

    def to_dict(self, *, include_schema: bool = True) -> dict[str, Any]:
        payload = {
            "tree_id": self.tree_id,
            "end_goal_id": self.end_goal_id,
            "proof_graph_id": self.proof_graph_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "policy_id": self.policy_id,
            "bounds": dict(self.bounds),
            "bound_digest": self.bound_digest,
            "resource_class": self.resource_class,
            "max_retries": self.max_retries,
            "selected_leaf_ids": list(self.selected_leaf_ids),
            "selected_counterexample_ids": list(
                self.selected_counterexample_ids
            ),
            "toolchain_id": self.toolchain_id,
        }
        if include_schema:
            payload = {"schema": LIFECYCLE_CACHE_KEY_SCHEMA, **payload}
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactLifecycleCacheKey":
        value = _mapping(payload, field_name="cache_key")
        if value.get("schema") not in {None, LIFECYCLE_CACHE_KEY_SCHEMA}:
            raise GoalTacticianLifecycleError(
                "unsupported lifecycle cache key schema"
            )
        return cls(
            tree_id=value.get("tree_id", ""),
            end_goal_id=value.get("end_goal_id", ""),
            proof_graph_id=value.get("proof_graph_id", ""),
            provider_id=value.get("provider_id", ""),
            provider_version=value.get("provider_version", ""),
            policy_id=value.get("policy_id", ""),
            bounds=dict(value.get("bounds") or {}),
            resource_class=value.get("resource_class", ""),
            max_retries=int(value.get("max_retries", DEFAULT_MAX_RETRIES)),
            selected_leaf_ids=tuple(value.get("selected_leaf_ids") or ()),
            selected_counterexample_ids=tuple(
                value.get("selected_counterexample_ids") or ()
            ),
            toolchain_id=value.get("toolchain_id", ""),
        )


@dataclass(frozen=True)
class ResourcePolicy:
    """Hard resource policy bound into the lifecycle lease fence."""

    resource_class: str
    max_concurrent_workers: int = 1
    wall_time_ms: int = 60_000
    memory_bytes: int = 256 * 1024 * 1024
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class, field_name="resource_class"),
        )
        object.__setattr__(
            self,
            "max_concurrent_workers",
            _positive(self.max_concurrent_workers, "max_concurrent_workers"),
        )
        object.__setattr__(
            self, "wall_time_ms", _positive(self.wall_time_ms, "wall_time_ms")
        )
        object.__setattr__(
            self,
            "memory_bytes",
            _positive(self.memory_bytes, "memory_bytes"),
        )
        object.__setattr__(
            self, "max_retries", _non_negative(self.max_retries, "max_retries")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_class": self.resource_class,
            "max_concurrent_workers": self.max_concurrent_workers,
            "wall_time_ms": self.wall_time_ms,
            "memory_bytes": self.memory_bytes,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourcePolicy":
        value = dict(payload or {})
        return cls(
            resource_class=value.get("resource_class", "cpu-supervisor"),
            max_concurrent_workers=int(value.get("max_concurrent_workers", 1)),
            wall_time_ms=int(value.get("wall_time_ms", 60_000)),
            memory_bytes=int(value.get("memory_bytes", 256 * 1024 * 1024)),
            max_retries=int(value.get("max_retries", DEFAULT_MAX_RETRIES)),
        )


@dataclass(frozen=True)
class WorkerLease:
    """Fenced lease authorizing a single worker to mutate the plan.

    The monotonically increasing fencing token is the sole mutation credential.
    Lease bearer secrets are never persisted in authoritative state.
    """

    SCHEMA: ClassVar[str] = LIFECYCLE_LEASE_SCHEMA

    worker_id: str
    plan_id: str
    fencing_token: int
    fencing_epoch: int
    acquired_at_ms: int
    expires_at_ms: int
    resource_class: str
    active: bool = True
    release_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worker_id", _text(self.worker_id, field_name="worker_id")
        )
        object.__setattr__(
            self, "plan_id", _text(self.plan_id, field_name="plan_id")
        )
        object.__setattr__(
            self,
            "fencing_token",
            _positive(self.fencing_token, "fencing_token"),
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _positive(self.fencing_epoch, "fencing_epoch"),
        )
        object.__setattr__(
            self,
            "acquired_at_ms",
            _non_negative(self.acquired_at_ms, "acquired_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _non_negative(self.expires_at_ms, "expires_at_ms"),
        )
        object.__setattr__(
            self,
            "resource_class",
            _text(self.resource_class, field_name="resource_class"),
        )
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(
            self, "release_reason", str(self.release_reason or "").strip()
        )

    def is_expired(self, now_ms: int) -> bool:
        return bool(self.expires_at_ms and now_ms > self.expires_at_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_LEASE_SCHEMA,
            "worker_id": self.worker_id,
            "plan_id": self.plan_id,
            "fencing_token": self.fencing_token,
            "fencing_epoch": self.fencing_epoch,
            "acquired_at_ms": self.acquired_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "resource_class": self.resource_class,
            "active": self.active,
            "release_reason": self.release_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerLease":
        value = _mapping(payload, field_name="lease")
        if value.get("schema") not in {None, LIFECYCLE_LEASE_SCHEMA}:
            raise GoalTacticianLifecycleError("unsupported lifecycle lease schema")
        return cls(
            worker_id=value.get("worker_id", ""),
            plan_id=value.get("plan_id", ""),
            fencing_token=int(value.get("fencing_token", 0)),
            fencing_epoch=int(value.get("fencing_epoch", 1)),
            acquired_at_ms=int(value.get("acquired_at_ms", 0)),
            expires_at_ms=int(value.get("expires_at_ms", 0)),
            resource_class=value.get("resource_class", ""),
            active=bool(value.get("active", True)),
            release_reason=value.get("release_reason", ""),
        )


# ---------------------------------------------------------------------------
# Receipts, transitions, authoritative state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleReceipt:
    """Freshness-aware evidence bound to a leaf or counterexample."""

    SCHEMA: ClassVar[str] = LIFECYCLE_RECEIPT_SCHEMA

    receipt_id: str
    kind: ReceiptKind
    subject_id: str
    tree_id: str
    fencing_epoch: int
    fencing_token: int
    assurance: AssuranceLevel
    independently_validated: bool
    content_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, field_name="receipt_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, ReceiptKind, "kind"))
        object.__setattr__(
            self, "subject_id", _text(self.subject_id, field_name="subject_id")
        )
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, field_name="tree_id")
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _positive(self.fencing_epoch, "fencing_epoch"),
        )
        object.__setattr__(
            self,
            "fencing_token",
            _positive(self.fencing_token, "fencing_token"),
        )
        object.__setattr__(self, "assurance", _assurance(self.assurance))
        object.__setattr__(
            self, "independently_validated", bool(self.independently_validated)
        )
        digest = str(self.content_digest or "").strip()
        if not digest:
            digest = f"sha256:{_sha256_hex(self.to_dict(include_identity=False))}"
        object.__setattr__(self, "content_digest", digest)
        object.__setattr__(
            self, "metadata", _public_mapping(dict(self.metadata or {}))
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def freshness_against(
        self,
        *,
        tree_id: str,
        fencing_epoch: int,
        required_assurance: AssuranceLevel,
        current_fencing_token: int | None = None,
    ) -> Freshness:
        if self.tree_id != tree_id:
            return Freshness.STALE_TREE
        if self.fencing_epoch != fencing_epoch:
            return Freshness.STALE_EPOCH
        if (
            current_fencing_token is not None
            and self.fencing_token > current_fencing_token
        ):
            return Freshness.STALE_WORKER
        if not self.independently_validated:
            return Freshness.INADEQUATE
        if self.assurance.rank < required_assurance.rank:
            return Freshness.INADEQUATE
        return Freshness.FRESH

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LIFECYCLE_RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "tree_id": self.tree_id,
            "fencing_epoch": self.fencing_epoch,
            "fencing_token": self.fencing_token,
            "assurance": self.assurance.value,
            "independently_validated": self.independently_validated,
            "content_digest": self.content_digest,
            "metadata": dict(self.metadata),
        }
        if include_identity:
            payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleReceipt":
        value = _mapping(payload, field_name="receipt")
        if value.get("schema") not in {None, LIFECYCLE_RECEIPT_SCHEMA}:
            raise GoalTacticianLifecycleError(
                "unsupported lifecycle receipt schema"
            )
        return cls(
            receipt_id=value.get("receipt_id", ""),
            kind=value.get("kind", ReceiptKind.GRAPH_LEAF),
            subject_id=value.get("subject_id", ""),
            tree_id=value.get("tree_id", ""),
            fencing_epoch=int(value.get("fencing_epoch", 1)),
            fencing_token=int(value.get("fencing_token", 1)),
            assurance=value.get("assurance", AssuranceLevel.UNVERIFIED),
            independently_validated=bool(
                value.get("independently_validated", False)
            ),
            content_digest=value.get("content_digest", ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LifecycleTransition:
    """One durable content-addressed lifecycle transition."""

    SCHEMA: ClassVar[str] = LIFECYCLE_TRANSITION_SCHEMA

    kind: LifecycleTransitionKind
    plan_id: str
    sequence: int
    tree_id: str
    fencing_epoch: int
    fencing_token: int
    worker_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    recorded_at_ms: int = 0
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, LifecycleTransitionKind, "kind")
        )
        object.__setattr__(
            self, "plan_id", _text(self.plan_id, field_name="plan_id")
        )
        object.__setattr__(
            self, "sequence", _non_negative(self.sequence, "sequence")
        )
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, field_name="tree_id")
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _positive(self.fencing_epoch, "fencing_epoch"),
        )
        object.__setattr__(
            self,
            "fencing_token",
            _non_negative(self.fencing_token, "fencing_token"),
        )
        object.__setattr__(
            self, "worker_id", str(self.worker_id or "").strip()
        )
        object.__setattr__(
            self, "payload", _public_mapping(dict(self.payload or {}))
        )
        object.__setattr__(
            self,
            "recorded_at_ms",
            _non_negative(self.recorded_at_ms, "recorded_at_ms"),
        )
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "").strip()
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LIFECYCLE_TRANSITION_SCHEMA,
            "kind": self.kind.value,
            "plan_id": self.plan_id,
            "sequence": self.sequence,
            "tree_id": self.tree_id,
            "fencing_epoch": self.fencing_epoch,
            "fencing_token": self.fencing_token,
            "worker_id": self.worker_id,
            "payload": dict(self.payload),
            "recorded_at_ms": self.recorded_at_ms,
            "reason_code": self.reason_code,
        }
        if include_identity:
            payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleTransition":
        value = _mapping(payload, field_name="transition")
        if value.get("schema") not in {None, LIFECYCLE_TRANSITION_SCHEMA}:
            raise GoalTacticianLifecycleError(
                "unsupported lifecycle transition schema"
            )
        return cls(
            kind=value.get("kind", LifecycleTransitionKind.END_GOAL),
            plan_id=value.get("plan_id", ""),
            sequence=int(value.get("sequence", 0)),
            tree_id=value.get("tree_id", ""),
            fencing_epoch=int(value.get("fencing_epoch", 1)),
            fencing_token=int(value.get("fencing_token", 0)),
            worker_id=value.get("worker_id", ""),
            payload=dict(value.get("payload") or {}),
            recorded_at_ms=int(value.get("recorded_at_ms", 0)),
            reason_code=value.get("reason_code", ""),
        )


@dataclass(frozen=True)
class LifecycleCompletionDecision:
    """Authoritative completion gate projection (fail-closed)."""

    SCHEMA: ClassVar[str] = LIFECYCLE_COMPLETION_SCHEMA

    admitted: bool
    reason_codes: tuple[str, ...]
    missing_leaf_ids: tuple[str, ...] = ()
    missing_counterexample_ids: tuple[str, ...] = ()
    stale_receipt_ids: tuple[str, ...] = ()
    control_signal: LifecycleControlSignal = LifecycleControlSignal.NONE
    plan_status: LifecyclePlanStatus = LifecyclePlanStatus.OPEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_COMPLETION_SCHEMA,
            "admitted": self.admitted,
            "reason_codes": list(self.reason_codes),
            "missing_leaf_ids": list(self.missing_leaf_ids),
            "missing_counterexample_ids": list(self.missing_counterexample_ids),
            "stale_receipt_ids": list(self.stale_receipt_ids),
            "control_signal": self.control_signal.value,
            "plan_status": self.plan_status.value,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LifecycleCompletionDecision":
        value = _mapping(payload, field_name="completion")
        return cls(
            admitted=bool(value.get("admitted", False)),
            reason_codes=_strings(value.get("reason_codes")),
            missing_leaf_ids=_strings(value.get("missing_leaf_ids")),
            missing_counterexample_ids=_strings(
                value.get("missing_counterexample_ids")
            ),
            stale_receipt_ids=_strings(value.get("stale_receipt_ids")),
            control_signal=_enum(
                value.get("control_signal", LifecycleControlSignal.NONE),
                LifecycleControlSignal,
                "control_signal",
            ),
            plan_status=_enum(
                value.get("plan_status", LifecyclePlanStatus.OPEN),
                LifecyclePlanStatus,
                "plan_status",
            ),
        )


@dataclass(frozen=True)
class LifecycleAuthoritativeState:
    """Restart-safe authoritative projection of one tactician plan."""

    SCHEMA: ClassVar[str] = LIFECYCLE_STATE_SCHEMA

    plan_id: str
    cache_key: ExactLifecycleCacheKey
    status: LifecyclePlanStatus
    fencing_epoch: int
    fencing_token: int
    control_signal: LifecycleControlSignal
    resource_policy: ResourcePolicy
    required_assurance: AssuranceLevel
    sequence: int
    active_lease: WorkerLease | None = None
    transitions: tuple[LifecycleTransition, ...] = ()
    receipts: tuple[LifecycleReceipt, ...] = ()
    end_goal: Mapping[str, Any] = field(default_factory=dict)
    proof_graph: Mapping[str, Any] = field(default_factory=dict)
    candidates: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    completion: LifecycleCompletionDecision | None = None
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "plan_id", _text(self.plan_id, field_name="plan_id")
        )
        if not isinstance(self.cache_key, ExactLifecycleCacheKey):
            object.__setattr__(
                self,
                "cache_key",
                ExactLifecycleCacheKey.from_dict(
                    _mapping(self.cache_key, field_name="cache_key")
                ),
            )
        object.__setattr__(
            self, "status", _enum(self.status, LifecyclePlanStatus, "status")
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _positive(self.fencing_epoch, "fencing_epoch"),
        )
        object.__setattr__(
            self,
            "fencing_token",
            _non_negative(self.fencing_token, "fencing_token"),
        )
        object.__setattr__(
            self,
            "control_signal",
            _enum(
                self.control_signal, LifecycleControlSignal, "control_signal"
            ),
        )
        if not isinstance(self.resource_policy, ResourcePolicy):
            object.__setattr__(
                self,
                "resource_policy",
                ResourcePolicy.from_dict(
                    _mapping(self.resource_policy, field_name="resource_policy")
                ),
            )
        object.__setattr__(
            self, "required_assurance", _assurance(self.required_assurance)
        )
        object.__setattr__(
            self, "sequence", _non_negative(self.sequence, "sequence")
        )
        if self.active_lease is not None and not isinstance(
            self.active_lease, WorkerLease
        ):
            object.__setattr__(
                self,
                "active_lease",
                WorkerLease.from_dict(
                    _mapping(self.active_lease, field_name="active_lease")
                ),
            )
        transitions: list[LifecycleTransition] = []
        for item in self.transitions or ():
            if isinstance(item, LifecycleTransition):
                transitions.append(item)
            else:
                transitions.append(
                    LifecycleTransition.from_dict(
                        _mapping(item, field_name="transition")
                    )
                )
        object.__setattr__(self, "transitions", tuple(transitions))
        receipts: list[LifecycleReceipt] = []
        for item in self.receipts or ():
            if isinstance(item, LifecycleReceipt):
                receipts.append(item)
            else:
                receipts.append(
                    LifecycleReceipt.from_dict(
                        _mapping(item, field_name="receipt")
                    )
                )
        object.__setattr__(self, "receipts", tuple(receipts))
        object.__setattr__(
            self, "end_goal", _public_mapping(dict(self.end_goal or {}))
        )
        object.__setattr__(
            self, "proof_graph", _public_mapping(dict(self.proof_graph or {}))
        )
        candidates: list[dict[str, Any]] = []
        for item in self.candidates or ():
            candidates.append(
                _public_mapping(_mapping(item, field_name="candidate"))
            )
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(
            self, "metadata", _public_mapping(dict(self.metadata or {}))
        )
        if self.completion is not None and not isinstance(
            self.completion, LifecycleCompletionDecision
        ):
            object.__setattr__(
                self,
                "completion",
                LifecycleCompletionDecision.from_dict(
                    _mapping(self.completion, field_name="completion")
                ),
            )
        object.__setattr__(
            self, "updated_at_ms", _non_negative(self.updated_at_ms, "updated_at_ms")
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def tree_id(self) -> str:
        return self.cache_key.tree_id

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": LIFECYCLE_STATE_SCHEMA,
            "interface": GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE,
            "version": GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_VERSION,
            "plan_id": self.plan_id,
            "cache_key": self.cache_key.to_dict(),
            "status": self.status.value,
            "fencing_epoch": self.fencing_epoch,
            "fencing_token": self.fencing_token,
            "control_signal": self.control_signal.value,
            "resource_policy": self.resource_policy.to_dict(),
            "required_assurance": self.required_assurance.value,
            "sequence": self.sequence,
            "active_lease": (
                self.active_lease.to_dict()
                if self.active_lease is not None
                else None
            ),
            "transitions": [
                item.to_dict() for item in self.transitions
            ],
            "receipts": [item.to_dict() for item in self.receipts],
            "end_goal": dict(self.end_goal),
            "proof_graph": dict(self.proof_graph),
            "candidates": [dict(item) for item in self.candidates],
            "metadata": dict(self.metadata),
            "completion": (
                self.completion.to_dict()
                if self.completion is not None
                else None
            ),
            "updated_at_ms": self.updated_at_ms,
        }
        if include_identity:
            payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "LifecycleAuthoritativeState":
        value = _mapping(payload, field_name="state")
        if value.get("schema") not in {None, LIFECYCLE_STATE_SCHEMA}:
            raise GoalTacticianLifecycleError(
                "unsupported lifecycle state schema"
            )
        return cls(
            plan_id=value.get("plan_id", ""),
            cache_key=ExactLifecycleCacheKey.from_dict(
                value.get("cache_key") or {}
            ),
            status=value.get("status", LifecyclePlanStatus.OPEN),
            fencing_epoch=int(value.get("fencing_epoch", 1)),
            fencing_token=int(value.get("fencing_token", 0)),
            control_signal=value.get(
                "control_signal", LifecycleControlSignal.NONE
            ),
            resource_policy=ResourcePolicy.from_dict(
                value.get("resource_policy") or {}
            ),
            required_assurance=value.get(
                "required_assurance", DEFAULT_REQUIRED_ASSURANCE
            ),
            sequence=int(value.get("sequence", 0)),
            active_lease=value.get("active_lease"),
            transitions=tuple(value.get("transitions") or ()),
            receipts=tuple(value.get("receipts") or ()),
            end_goal=dict(value.get("end_goal") or {}),
            proof_graph=dict(value.get("proof_graph") or {}),
            candidates=tuple(value.get("candidates") or ()),
            metadata=dict(value.get("metadata") or {}),
            completion=value.get("completion"),
            updated_at_ms=int(value.get("updated_at_ms", 0)),
        )

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        _atomic_json(target, self.to_dict())
        return target

    @classmethod
    def load(cls, path: Path | str) -> "LifecycleAuthoritativeState":
        target = Path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GoalTacticianLifecycleError(
                f"lifecycle state is unreadable: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise GoalTacticianLifecycleError(
                "lifecycle state must be a JSON object"
            )
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# Lifecycle supervisor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalTacticianLifecycleConfig:
    """Operational controls for durable restart-safe lifecycle state."""

    state_dir: Path | str
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    state_filename: str = DEFAULT_STATE_FILENAME
    journal_filename: str = DEFAULT_JOURNAL_FILENAME
    required_assurance: AssuranceLevel | str = DEFAULT_REQUIRED_ASSURANCE

    def __post_init__(self) -> None:
        try:
            raw_state_dir = os.fspath(self.state_dir)
        except TypeError as exc:
            raise GoalTacticianLifecycleError(
                "state_dir must be a filesystem path"
            ) from exc
        if not isinstance(raw_state_dir, str) or not raw_state_dir.strip():
            raise GoalTacticianLifecycleError("state_dir must not be empty")
        object.__setattr__(self, "state_dir", Path(raw_state_dir))
        object.__setattr__(
            self,
            "lease_seconds",
            _positive(self.lease_seconds, "lease_seconds"),
        )
        for name in ("state_filename", "journal_filename"):
            value = str(getattr(self, name) or "").strip()
            if not value or value in {".", ".."} or Path(value).name != value:
                raise GoalTacticianLifecycleError(
                    f"{name} must be a plain file name"
                )
            object.__setattr__(self, name, value)
        if self.state_filename == self.journal_filename:
            raise GoalTacticianLifecycleError(
                "state and journal filenames must be distinct"
            )
        object.__setattr__(
            self, "required_assurance", _assurance(self.required_assurance)
        )

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / self.state_filename

    @property
    def journal_path(self) -> Path:
        return Path(self.state_dir) / self.journal_filename


class GoalTacticianSupervisorLifecycle:
    """Fenced, restartable supervisor boundary for goal-tactician plans.

    Interface: ``GoalTacticianSupervisorLifecycle@1``.
    """

    interface: ClassVar[str] = GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE
    schema: ClassVar[str] = GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_SCHEMA
    version: ClassVar[str] = GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_VERSION

    def __init__(
        self,
        config: GoalTacticianLifecycleConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(config, GoalTacticianLifecycleConfig):
            raise GoalTacticianLifecycleError(
                "config must be GoalTacticianLifecycleConfig"
            )
        self.config = config
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._state: LifecycleAuthoritativeState | None = None
        if self.config.state_path.exists():
            self._state = LifecycleAuthoritativeState.load(self.config.state_path)

    # -- public projections -------------------------------------------------

    @property
    def state(self) -> LifecycleAuthoritativeState | None:
        with self._lock:
            return self._state

    def authoritative_state(self) -> LifecycleAuthoritativeState:
        with self._lock:
            if self._state is None:
                raise GoalTacticianLifecycleError("no lifecycle plan is open")
            return self._state

    def authoritative_snapshot(self) -> dict[str, Any]:
        """Canonical durable projection used for restart identity checks."""

        return self.authoritative_state().to_dict()

    # -- plan open / restart ------------------------------------------------

    def open_plan(
        self,
        *,
        tree_id: str,
        end_goal_id: str,
        proof_graph_id: str,
        provider_id: str,
        provider_version: str,
        policy_id: str,
        bounds: Mapping[str, Any],
        resource_class: str = "cpu-supervisor",
        max_retries: int = DEFAULT_MAX_RETRIES,
        selected_leaf_ids: Sequence[str] = (),
        selected_counterexample_ids: Sequence[str] = (),
        toolchain_id: str = "",
        end_goal: Mapping[str, Any] | None = None,
        proof_graph: Mapping[str, Any] | None = None,
        resource_policy: ResourcePolicy | Mapping[str, Any] | None = None,
        plan_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> LifecycleAuthoritativeState:
        """Open a new fenced plan (or reject if one is already completed)."""

        with self._lock:
            cache_key = ExactLifecycleCacheKey(
                tree_id=tree_id,
                end_goal_id=end_goal_id,
                proof_graph_id=proof_graph_id,
                provider_id=provider_id,
                provider_version=provider_version,
                policy_id=policy_id,
                bounds=bounds,
                resource_class=resource_class,
                max_retries=max_retries,
                selected_leaf_ids=tuple(selected_leaf_ids),
                selected_counterexample_ids=tuple(selected_counterexample_ids),
                toolchain_id=toolchain_id,
            )
            if resource_policy is None:
                policy = ResourcePolicy(
                    resource_class=resource_class,
                    max_retries=max_retries,
                    wall_time_ms=int(
                        dict(bounds).get("wall_time_ms", 60_000)
                    ),
                    memory_bytes=int(
                        dict(bounds).get(
                            "memory_bytes", 256 * 1024 * 1024
                        )
                    ),
                )
            elif isinstance(resource_policy, ResourcePolicy):
                policy = resource_policy
            else:
                policy = ResourcePolicy.from_dict(resource_policy)

            resolved_plan_id = (
                _text(plan_id, field_name="plan_id", required=False)
                or f"plan:{cache_key.key_id}"
            )
            now = _now_ms(self._clock)
            state = LifecycleAuthoritativeState(
                plan_id=resolved_plan_id,
                cache_key=cache_key,
                status=LifecyclePlanStatus.OPEN,
                fencing_epoch=1,
                fencing_token=0,
                control_signal=LifecycleControlSignal.NONE,
                resource_policy=policy,
                required_assurance=self.config.required_assurance,
                sequence=0,
                end_goal=_public_mapping(
                    dict(end_goal or {"end_goal_id": end_goal_id})
                ),
                proof_graph=_public_mapping(
                    dict(
                        proof_graph
                        or {
                            "proof_graph_id": proof_graph_id,
                            "leaf_ids": list(selected_leaf_ids),
                        }
                    )
                ),
                metadata=_public_mapping(dict(metadata or {})),
                updated_at_ms=now,
            )
            # Record opening transitions under content identity.
            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.END_GOAL,
                worker_id="",
                fencing_token=0,
                payload=dict(state.end_goal),
                reason_code="plan_opened",
            )
            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.PROOF_GRAPH,
                worker_id="",
                fencing_token=0,
                payload=dict(state.proof_graph),
                reason_code="plan_opened",
            )
            self._commit(state)
            return state

    def reconcile(self) -> LifecycleAuthoritativeState:
        """Restart-safe reconciliation: reload durable state and re-fence.

        Expired leases are released.  Control signals remain durable.  The
        authoritative content identity is preserved when nothing material
        changes.
        """

        with self._lock:
            if self._state is None and self.config.state_path.exists():
                self._state = LifecycleAuthoritativeState.load(
                    self.config.state_path
                )
            if self._state is None:
                raise GoalTacticianLifecycleError(
                    "no durable lifecycle state to reconcile"
                )
            state = self._state
            now = _now_ms(self._clock)
            lease = state.active_lease
            changed = False
            if lease is not None and (
                not lease.active or lease.is_expired(now)
            ):
                released = WorkerLease(
                    worker_id=lease.worker_id,
                    plan_id=lease.plan_id,
                    fencing_token=lease.fencing_token,
                    fencing_epoch=lease.fencing_epoch,
                    acquired_at_ms=lease.acquired_at_ms,
                    expires_at_ms=lease.expires_at_ms,
                    resource_class=lease.resource_class,
                    active=False,
                    release_reason=(
                        lease.release_reason or "lease_expired_on_reconcile"
                    ),
                )
                state = LifecycleAuthoritativeState.from_dict(
                    {
                        **state.to_dict(include_identity=False),
                        "active_lease": released.to_dict(),
                        "status": (
                            LifecyclePlanStatus.BLOCKED.value
                            if state.control_signal
                            is not LifecycleControlSignal.NONE
                            else (
                                state.status.value
                                if state.status
                                is not LifecyclePlanStatus.RUNNING
                                else LifecyclePlanStatus.OPEN.value
                            )
                        ),
                        "updated_at_ms": now,
                    }
                )
                state = self._append_transition(
                    state,
                    kind=LifecycleTransitionKind.LEASE_RELEASE,
                    worker_id=released.worker_id,
                    fencing_token=released.fencing_token,
                    payload={"release_reason": released.release_reason},
                    reason_code="reconcile_expired_lease",
                )
                changed = True

            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.RECONCILE,
                worker_id="",
                fencing_token=state.fencing_token,
                payload={
                    "prior_content_id": state.content_id,
                    "control_signal": state.control_signal.value,
                    "status": state.status.value,
                },
                reason_code="restart_reconcile",
            )
            # Always persist the reconcile transition for auditability.
            self._commit(state)
            del changed  # reconcile always journals
            return state

    def restart(self) -> LifecycleAuthoritativeState:
        """Alias for :meth:`reconcile` — process restart entrypoint."""

        # Drop in-memory view so reconcile reloads from durable state.
        with self._lock:
            self._state = None
        return self.reconcile()

    # -- leases / fencing ---------------------------------------------------

    def acquire_lease(
        self,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
    ) -> WorkerLease:
        """Acquire a fenced lease, invalidating any prior active owner."""

        with self._lock:
            state = self._require_open_state()
            self._reject_if_control_blocks_mutation(state)
            if state.status in {
                LifecyclePlanStatus.COMPLETED,
                LifecyclePlanStatus.INVALIDATED,
            }:
                raise GoalTacticianLifecycleError(
                    f"cannot acquire lease on {state.status.value} plan"
                )
            now = _now_ms(self._clock)
            seconds = (
                self.config.lease_seconds
                if lease_seconds is None
                else _positive(lease_seconds, "lease_seconds")
            )
            next_token = state.fencing_token + 1
            lease = WorkerLease(
                worker_id=_text(worker_id, field_name="worker_id"),
                plan_id=state.plan_id,
                fencing_token=next_token,
                fencing_epoch=state.fencing_epoch,
                acquired_at_ms=now,
                expires_at_ms=now + seconds * 1000,
                resource_class=state.resource_policy.resource_class,
                active=True,
            )
            # Fence out the previous owner if present.
            if state.active_lease is not None and state.active_lease.active:
                prior = state.active_lease
                fenced = WorkerLease(
                    worker_id=prior.worker_id,
                    plan_id=prior.plan_id,
                    fencing_token=prior.fencing_token,
                    fencing_epoch=prior.fencing_epoch,
                    acquired_at_ms=prior.acquired_at_ms,
                    expires_at_ms=prior.expires_at_ms,
                    resource_class=prior.resource_class,
                    active=False,
                    release_reason="fenced_by_successor",
                )
                state = LifecycleAuthoritativeState.from_dict(
                    {
                        **state.to_dict(include_identity=False),
                        "active_lease": fenced.to_dict(),
                    }
                )
                state = self._append_transition(
                    state,
                    kind=LifecycleTransitionKind.LEASE_RELEASE,
                    worker_id=prior.worker_id,
                    fencing_token=prior.fencing_token,
                    payload={"release_reason": "fenced_by_successor"},
                    reason_code="fenced_by_successor",
                )

            state = LifecycleAuthoritativeState.from_dict(
                {
                    **state.to_dict(include_identity=False),
                    "active_lease": lease.to_dict(),
                    "fencing_token": next_token,
                    "status": LifecyclePlanStatus.RUNNING.value,
                    "updated_at_ms": now,
                }
            )
            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.LEASE_ACQUIRE,
                worker_id=lease.worker_id,
                fencing_token=lease.fencing_token,
                payload=lease.to_dict(),
                reason_code="lease_acquired",
            )
            self._commit(state)
            return lease

    def release_lease(
        self,
        lease: WorkerLease | Mapping[str, Any],
        *,
        reason: str = "released",
    ) -> LifecycleAuthoritativeState:
        with self._lock:
            state = self._require_open_state()
            held = self._coerce_lease(lease)
            self._assert_lease_authoritative(state, held)
            released = WorkerLease(
                worker_id=held.worker_id,
                plan_id=held.plan_id,
                fencing_token=held.fencing_token,
                fencing_epoch=held.fencing_epoch,
                acquired_at_ms=held.acquired_at_ms,
                expires_at_ms=held.expires_at_ms,
                resource_class=held.resource_class,
                active=False,
                release_reason=_text(reason, field_name="reason"),
            )
            now = _now_ms(self._clock)
            status = state.status
            if state.control_signal is not LifecycleControlSignal.NONE:
                status = LifecyclePlanStatus.BLOCKED
            elif status is LifecyclePlanStatus.RUNNING:
                status = LifecyclePlanStatus.OPEN
            state = LifecycleAuthoritativeState.from_dict(
                {
                    **state.to_dict(include_identity=False),
                    "active_lease": released.to_dict(),
                    "status": status.value,
                    "updated_at_ms": now,
                }
            )
            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.LEASE_RELEASE,
                worker_id=released.worker_id,
                fencing_token=released.fencing_token,
                payload={"release_reason": released.release_reason},
                reason_code=released.release_reason,
            )
            self._commit(state)
            return state

    # -- transitions --------------------------------------------------------

    def record_transition(
        self,
        kind: LifecycleTransitionKind | str,
        payload: Mapping[str, Any],
        lease: WorkerLease | Mapping[str, Any],
        *,
        reason_code: str = "",
        receipt: LifecycleReceipt | Mapping[str, Any] | None = None,
    ) -> LifecycleAuthoritativeState:
        """Record a fenced content-addressed lifecycle transition."""

        with self._lock:
            state = self._require_open_state()
            held = self._coerce_lease(lease)
            self._assert_lease_authoritative(state, held)
            self._reject_if_control_blocks_mutation(state)
            if state.status is LifecyclePlanStatus.INVALIDATED:
                raise GoalTacticianLifecycleError(
                    "cannot mutate an invalidated plan"
                )
            if state.status is LifecyclePlanStatus.COMPLETED:
                raise GoalTacticianLifecycleError(
                    "cannot mutate a completed plan"
                )

            kind_enum = _enum(kind, LifecycleTransitionKind, "kind")
            if kind_enum in {
                LifecycleTransitionKind.LEASE_ACQUIRE,
                LifecycleTransitionKind.LEASE_RELEASE,
                LifecycleTransitionKind.RECONCILE,
                LifecycleTransitionKind.CONTROL,
                LifecycleTransitionKind.TREE_INVALIDATION,
                LifecycleTransitionKind.COMPLETION,
            }:
                raise GoalTacticianLifecycleError(
                    f"{kind_enum.value} must use its dedicated API"
                )

            body = _public_mapping(dict(payload or {}))
            if claims_authority(body) and kind_enum is not LifecycleTransitionKind.VERIFICATION:
                # Authority claims only admissible via validated receipts.
                if receipt is None:
                    raise GoalTacticianLifecycleError(
                        "authority claims require an independently validated receipt"
                    )

            now = _now_ms(self._clock)
            next_state_fields: dict[str, Any] = {
                **state.to_dict(include_identity=False),
                "updated_at_ms": now,
            }

            if kind_enum is LifecycleTransitionKind.END_GOAL:
                next_state_fields["end_goal"] = body
            elif kind_enum is LifecycleTransitionKind.PROOF_GRAPH:
                next_state_fields["proof_graph"] = body
            elif kind_enum is LifecycleTransitionKind.CANDIDATE:
                existing = [dict(item) for item in state.candidates]
                existing.append(body)
                next_state_fields["candidates"] = existing

            receipts = list(state.receipts)
            if receipt is not None:
                bound = self._coerce_receipt(receipt)
                freshness = bound.freshness_against(
                    tree_id=state.tree_id,
                    fencing_epoch=state.fencing_epoch,
                    required_assurance=state.required_assurance,
                    current_fencing_token=state.fencing_token,
                )
                if freshness is not Freshness.FRESH:
                    raise StaleReceiptError(
                        f"receipt {bound.receipt_id} is {freshness.value}"
                    )
                if bound.fencing_token != held.fencing_token:
                    raise StaleReceiptError(
                        "receipt fencing token does not match the active lease"
                    )
                receipts.append(bound)
                next_state_fields["receipts"] = [
                    item.to_dict() for item in receipts
                ]

            state = LifecycleAuthoritativeState.from_dict(next_state_fields)
            state = self._append_transition(
                state,
                kind=kind_enum,
                worker_id=held.worker_id,
                fencing_token=held.fencing_token,
                payload=body,
                reason_code=reason_code or kind_enum.value,
            )
            self._commit(state)
            return state

    def signal_control(
        self,
        signal: LifecycleControlSignal | str,
        lease: WorkerLease | Mapping[str, Any] | None = None,
        *,
        reason_code: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> LifecycleAuthoritativeState:
        """Record a durable cancel / timeout / backpressure control signal."""

        with self._lock:
            state = self._require_open_state()
            signal_enum = _enum(signal, LifecycleControlSignal, "signal")
            if signal_enum is LifecycleControlSignal.NONE:
                raise GoalTacticianLifecycleError(
                    "cannot signal NONE; use clear_control if supported"
                )
            worker_id = ""
            fencing_token = state.fencing_token
            if lease is not None:
                held = self._coerce_lease(lease)
                self._assert_lease_authoritative(state, held)
                worker_id = held.worker_id
                fencing_token = held.fencing_token

            now = _now_ms(self._clock)
            state = LifecycleAuthoritativeState.from_dict(
                {
                    **state.to_dict(include_identity=False),
                    "control_signal": signal_enum.value,
                    "status": LifecyclePlanStatus.BLOCKED.value,
                    "updated_at_ms": now,
                }
            )
            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.CONTROL,
                worker_id=worker_id,
                fencing_token=fencing_token,
                payload=_public_mapping(
                    dict(
                        payload
                        or {
                            "signal": signal_enum.value,
                            "reason_code": reason_code,
                        }
                    )
                ),
                reason_code=reason_code or signal_enum.value,
            )
            self._commit(state)
            return state

    def invalidate_tree(
        self,
        new_tree_id: str,
        lease: WorkerLease | Mapping[str, Any],
        *,
        reason_code: str = "tree_changed",
    ) -> LifecycleAuthoritativeState:
        """Invalidate scoped work when the repository tree identity changes.

        Bumps the fencing epoch, rewrites the exact cache key tree component,
        drops prior receipts as stale, and fences any prior worker token.
        """

        with self._lock:
            state = self._require_open_state()
            held = self._coerce_lease(lease)
            self._assert_lease_authoritative(state, held)
            new_tree = _text(new_tree_id, field_name="new_tree_id")
            if new_tree == state.tree_id:
                raise GoalTacticianLifecycleError(
                    "new_tree_id must differ from the current tree_id"
                )

            prior_key = state.cache_key
            new_key = ExactLifecycleCacheKey(
                tree_id=new_tree,
                end_goal_id=prior_key.end_goal_id,
                proof_graph_id=prior_key.proof_graph_id,
                provider_id=prior_key.provider_id,
                provider_version=prior_key.provider_version,
                policy_id=prior_key.policy_id,
                bounds=prior_key.bounds,
                resource_class=prior_key.resource_class,
                max_retries=prior_key.max_retries,
                selected_leaf_ids=prior_key.selected_leaf_ids,
                selected_counterexample_ids=prior_key.selected_counterexample_ids,
                toolchain_id=prior_key.toolchain_id,
            )
            now = _now_ms(self._clock)
            next_epoch = state.fencing_epoch + 1
            # Fence the current lease under the new epoch — token continues but
            # receipts from the prior epoch become stale.
            state = LifecycleAuthoritativeState.from_dict(
                {
                    **state.to_dict(include_identity=False),
                    "cache_key": new_key.to_dict(),
                    "fencing_epoch": next_epoch,
                    "receipts": [],  # prior receipts are tree-stale
                    "status": LifecyclePlanStatus.INVALIDATED.value,
                    "completion": None,
                    "updated_at_ms": now,
                    "metadata": {
                        **dict(state.metadata),
                        "prior_tree_id": state.tree_id,
                        "invalidation_reason": reason_code,
                    },
                }
            )
            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.TREE_INVALIDATION,
                worker_id=held.worker_id,
                fencing_token=held.fencing_token,
                payload={
                    "prior_tree_id": prior_key.tree_id,
                    "new_tree_id": new_tree,
                    "prior_cache_key_id": prior_key.key_id,
                    "new_cache_key_id": new_key.key_id,
                    "fencing_epoch": next_epoch,
                },
                reason_code=reason_code,
            )
            # After invalidation the plan is reopened under the new epoch so
            # work can continue with a fresh lease.
            state = LifecycleAuthoritativeState.from_dict(
                {
                    **state.to_dict(include_identity=False),
                    "status": LifecyclePlanStatus.OPEN.value,
                    "active_lease": None,
                    "fencing_token": state.fencing_token,  # keep high-water mark
                    "updated_at_ms": _now_ms(self._clock),
                }
            )
            self._commit(state)
            return state

    def try_complete(
        self,
        lease: WorkerLease | Mapping[str, Any],
        *,
        force: bool = False,
    ) -> LifecycleCompletionDecision:
        """Fail-closed completion: all selected leaves + counterexamples fresh."""

        with self._lock:
            state = self._require_open_state()
            held = self._coerce_lease(lease)
            self._assert_lease_authoritative(state, held)
            decision = self.evaluate_completion(state)
            if not decision.admitted:
                if force:
                    raise LifecycleCompletionError(
                        "force completion is not permitted; evidence is incomplete: "
                        + ",".join(decision.reason_codes)
                    )
                # Persist the rejected completion attempt for audit.
                state = self._append_transition(
                    state,
                    kind=LifecycleTransitionKind.COMPLETION,
                    worker_id=held.worker_id,
                    fencing_token=held.fencing_token,
                    payload=decision.to_dict(),
                    reason_code="completion_rejected",
                )
                state = LifecycleAuthoritativeState.from_dict(
                    {
                        **state.to_dict(include_identity=False),
                        "completion": decision.to_dict(),
                        "updated_at_ms": _now_ms(self._clock),
                    }
                )
                self._commit(state)
                return decision

            now = _now_ms(self._clock)
            admitted = LifecycleCompletionDecision(
                admitted=True,
                reason_codes=("all_selected_evidence_fresh",),
                control_signal=state.control_signal,
                plan_status=LifecyclePlanStatus.COMPLETED,
            )
            state = LifecycleAuthoritativeState.from_dict(
                {
                    **state.to_dict(include_identity=False),
                    "status": LifecyclePlanStatus.COMPLETED.value,
                    "completion": admitted.to_dict(),
                    "updated_at_ms": now,
                }
            )
            state = self._append_transition(
                state,
                kind=LifecycleTransitionKind.COMPLETION,
                worker_id=held.worker_id,
                fencing_token=held.fencing_token,
                payload=admitted.to_dict(),
                reason_code="completion_admitted",
            )
            # Release lease on successful completion.
            if state.active_lease is not None:
                released = WorkerLease(
                    worker_id=state.active_lease.worker_id,
                    plan_id=state.active_lease.plan_id,
                    fencing_token=state.active_lease.fencing_token,
                    fencing_epoch=state.active_lease.fencing_epoch,
                    acquired_at_ms=state.active_lease.acquired_at_ms,
                    expires_at_ms=state.active_lease.expires_at_ms,
                    resource_class=state.active_lease.resource_class,
                    active=False,
                    release_reason="completed",
                )
                state = LifecycleAuthoritativeState.from_dict(
                    {
                        **state.to_dict(include_identity=False),
                        "active_lease": released.to_dict(),
                    }
                )
            self._commit(state)
            return admitted

    def evaluate_completion(
        self,
        state: LifecycleAuthoritativeState | None = None,
    ) -> LifecycleCompletionDecision:
        """Project whether completion evidence is currently adequate."""

        current = state if state is not None else self.authoritative_state()
        reasons: list[str] = []
        if current.control_signal is not LifecycleControlSignal.NONE:
            reasons.append(f"control_{current.control_signal.value}")
        if current.status is LifecyclePlanStatus.INVALIDATED:
            reasons.append("plan_invalidated")

        leaf_ids = set(current.cache_key.selected_leaf_ids)
        cex_ids = set(current.cache_key.selected_counterexample_ids)
        fresh_leaves: set[str] = set()
        fresh_cex: set[str] = set()
        stale_ids: list[str] = []

        for receipt in current.receipts:
            freshness = receipt.freshness_against(
                tree_id=current.tree_id,
                fencing_epoch=current.fencing_epoch,
                required_assurance=current.required_assurance,
            )
            if freshness is not Freshness.FRESH:
                stale_ids.append(receipt.receipt_id)
                continue
            if receipt.kind is ReceiptKind.GRAPH_LEAF:
                fresh_leaves.add(receipt.subject_id)
            elif receipt.kind is ReceiptKind.COUNTEREXAMPLE:
                fresh_cex.add(receipt.subject_id)

        missing_leaves = tuple(sorted(leaf_ids - fresh_leaves))
        missing_cex = tuple(sorted(cex_ids - fresh_cex))
        if missing_leaves:
            reasons.append("missing_leaf_receipts")
        if missing_cex:
            reasons.append("missing_counterexample_receipts")
        if stale_ids:
            reasons.append("stale_receipts_present")

        admitted = not reasons
        return LifecycleCompletionDecision(
            admitted=admitted,
            reason_codes=tuple(reasons) if reasons else ("all_selected_evidence_fresh",),
            missing_leaf_ids=missing_leaves,
            missing_counterexample_ids=missing_cex,
            stale_receipt_ids=tuple(stale_ids),
            control_signal=current.control_signal,
            plan_status=(
                LifecyclePlanStatus.COMPLETED
                if admitted
                else current.status
            ),
        )

    def build_receipt(
        self,
        *,
        receipt_id: str,
        kind: ReceiptKind | str,
        subject_id: str,
        lease: WorkerLease | Mapping[str, Any],
        assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
        independently_validated: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> LifecycleReceipt:
        """Construct a receipt bound to the current tree and active lease."""

        with self._lock:
            state = self._require_open_state()
            held = self._coerce_lease(lease)
            self._assert_lease_authoritative(state, held)
            return LifecycleReceipt(
                receipt_id=receipt_id,
                kind=kind,
                subject_id=subject_id,
                tree_id=state.tree_id,
                fencing_epoch=state.fencing_epoch,
                fencing_token=held.fencing_token,
                assurance=assurance,
                independently_validated=independently_validated,
                metadata=dict(metadata or {}),
            )

    # -- internals ----------------------------------------------------------

    def _require_open_state(self) -> LifecycleAuthoritativeState:
        if self._state is None:
            if self.config.state_path.exists():
                self._state = LifecycleAuthoritativeState.load(
                    self.config.state_path
                )
            else:
                raise GoalTacticianLifecycleError("no lifecycle plan is open")
        return self._state

    def _coerce_lease(
        self, lease: WorkerLease | Mapping[str, Any]
    ) -> WorkerLease:
        if isinstance(lease, WorkerLease):
            return lease
        return WorkerLease.from_dict(_mapping(lease, field_name="lease"))

    def _coerce_receipt(
        self, receipt: LifecycleReceipt | Mapping[str, Any]
    ) -> LifecycleReceipt:
        if isinstance(receipt, LifecycleReceipt):
            return receipt
        return LifecycleReceipt.from_dict(
            _mapping(receipt, field_name="receipt")
        )

    def _assert_lease_authoritative(
        self,
        state: LifecycleAuthoritativeState,
        lease: WorkerLease,
    ) -> None:
        if lease.plan_id != state.plan_id:
            raise StaleWorkerError("lease plan_id does not match open plan")
        if lease.fencing_epoch != state.fencing_epoch:
            raise StaleWorkerError(
                "lease fencing_epoch is stale for the current plan epoch"
            )
        if state.active_lease is None or not state.active_lease.active:
            raise StaleWorkerError("no active lease is held for this plan")
        if lease.fencing_token != state.active_lease.fencing_token:
            raise StaleWorkerError(
                "lease fencing_token is stale; a successor worker holds the fence"
            )
        if lease.worker_id != state.active_lease.worker_id:
            raise StaleWorkerError(
                "lease worker_id does not match the active fenced owner"
            )
        now = _now_ms(self._clock)
        if lease.is_expired(now) or state.active_lease.is_expired(now):
            raise StaleWorkerError("lease has expired")

    def _reject_if_control_blocks_mutation(
        self, state: LifecycleAuthoritativeState
    ) -> None:
        if state.control_signal is not LifecycleControlSignal.NONE:
            raise LifecycleControlActiveError(
                f"mutation blocked by durable control signal: "
                f"{state.control_signal.value}"
            )

    def _append_transition(
        self,
        state: LifecycleAuthoritativeState,
        *,
        kind: LifecycleTransitionKind,
        worker_id: str,
        fencing_token: int,
        payload: Mapping[str, Any],
        reason_code: str,
    ) -> LifecycleAuthoritativeState:
        sequence = state.sequence + 1
        transition = LifecycleTransition(
            kind=kind,
            plan_id=state.plan_id,
            sequence=sequence,
            tree_id=state.tree_id,
            fencing_epoch=state.fencing_epoch,
            fencing_token=fencing_token,
            worker_id=worker_id,
            payload=dict(payload),
            recorded_at_ms=_now_ms(self._clock),
            reason_code=reason_code,
        )
        transitions = list(state.transitions) + [transition]
        return LifecycleAuthoritativeState.from_dict(
            {
                **state.to_dict(include_identity=False),
                "sequence": sequence,
                "transitions": [item.to_dict() for item in transitions],
                "updated_at_ms": transition.recorded_at_ms,
            }
        )

    def _commit(self, state: LifecycleAuthoritativeState) -> None:
        state.write(self.config.state_path)
        if state.transitions:
            _append_jsonl(
                self.config.journal_path, state.transitions[-1].to_dict()
            )
        self._state = state


def create_goal_tactician_supervisor_lifecycle(
    state_dir: Path | str,
    **kwargs: Any,
) -> GoalTacticianSupervisorLifecycle:
    """Factory for :class:`GoalTacticianSupervisorLifecycle`."""

    config = GoalTacticianLifecycleConfig(state_dir=state_dir, **kwargs)
    return GoalTacticianSupervisorLifecycle(config)


__all__ = [
    "GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_INTERFACE",
    "GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_SCHEMA",
    "GOAL_TACTICIAN_SUPERVISOR_LIFECYCLE_VERSION",
    "LIFECYCLE_CACHE_KEY_SCHEMA",
    "LIFECYCLE_COMPLETION_SCHEMA",
    "LIFECYCLE_LEASE_SCHEMA",
    "LIFECYCLE_RECEIPT_SCHEMA",
    "LIFECYCLE_STATE_SCHEMA",
    "LIFECYCLE_TRANSITION_SCHEMA",
    "ExactLifecycleCacheKey",
    "Freshness",
    "GoalTacticianLifecycleConfig",
    "GoalTacticianLifecycleError",
    "GoalTacticianSupervisorLifecycle",
    "LifecycleAuthoritativeState",
    "LifecycleCompletionDecision",
    "LifecycleCompletionError",
    "LifecycleControlActiveError",
    "LifecycleControlSignal",
    "LifecyclePlanStatus",
    "LifecycleReceipt",
    "LifecycleTransition",
    "LifecycleTransitionKind",
    "ReceiptKind",
    "ResourcePolicy",
    "StaleReceiptError",
    "StaleWorkerError",
    "WorkerLease",
    "claims_authority",
    "create_goal_tactician_supervisor_lifecycle",
]
