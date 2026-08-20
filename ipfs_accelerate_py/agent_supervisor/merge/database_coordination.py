"""Consolidate task, resource, merge, and maintenance leases with fencing.

DQP-015 / FencedLease@1, TaskClaim@1, ResourceClaim@1, MaintenanceLease@1
=========================================================================

:class:`DatabaseCoordinator` is the unified lease authority for task claims,
path/resource claims, provider/prover capacity, merge ownership, schema
maintenance, backup, and offline recovery. All lease kinds share one vocabulary:
owner session, scope, expiry, revision, fencing token, and fence epoch.

Algorithms reuse the proven :class:`~.lease_coordination.LeaseCoordinator`
patterns (monotonic fencing tokens/epochs, expire-before-claim, fair ready
selection, stale-token rejection) without deleting legacy stores. Legacy
:mod:`.lease_coordination` remains the Profile-G task-lane store until canary
cutover.

Authority rules (fail-closed)
-----------------------------
* Four processes never own the same exclusive scope simultaneously.
* An expired session cannot renew or mutate under its former fence.
* Append/fair scheduling remains concurrent for non-conflicting scopes.
* A stale fencing epoch is rejected on every protected write.
* Task claim and task-attempt creation commit in one transaction.
* Response-loss retries with the same idempotency key are replay-safe.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.duckdb_state import (
    connect_duckdb_with_policy,
    is_quack_transport_target,
    open_duckdb_connection,
)
from ..task_sources.task_identity import canonical_json_bytes
from .lease_coordination import (
    MAX_LEASE_MS,
    MIN_LEASE_MS,
    DependencyNotReadyError,
    LeaseConflictError,
    LeaseError,
    LeaseExpiredError,
    StaleFencingTokenError,
)

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_COORDINATOR_INTERFACE: Final[str] = "DatabaseCoordinator@1"
FENCED_LEASE_INTERFACE: Final[str] = "FencedLease@1"
TASK_CLAIM_INTERFACE: Final[str] = "TaskClaim@1"
RESOURCE_CLAIM_INTERFACE: Final[str] = "ResourceClaim@1"
MAINTENANCE_LEASE_INTERFACE: Final[str] = "MaintenanceLease@1"

DATABASE_COORDINATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-coordination@1"
)
COORDINATION_REGISTRY_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/coordination-registry-projection@1"
)
FENCED_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/fenced-lease@1"
)
TASK_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-claim@1"
)
RESOURCE_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/resource-claim@1"
)
MAINTENANCE_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/maintenance-lease@1"
)
TASK_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-attempt@1"
)
LEASE_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/lease-event@1"
)

DEFAULT_LEASE_MS: Final[int] = 60_000
DEFAULT_MAINTENANCE_SCOPE: Final[str] = "control-plane"
MAX_PAYLOAD_BYTES: Final[int] = 262_144
MAX_DEPENDENCY_EVIDENCE: Final[int] = 32
MAX_PREPARED_COMPLETION_QUERY: Final[int] = 1_000
PREPARED_COMPLETION_STATUS: Final[str] = "prepared"
TASK_COMPLETION_PREPARATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-completion-preparation@1"
)
CROSS_STORE_FENCE_GUARD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cross-store-fence-guard@1"
)
CROSS_STORE_FENCE_GUARD_EVENT: Final[str] = "cross_store_fence_guard_succeeded"
CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD: Final[str] = (
    "requires_cross_store_fence_guard"
)
_AUTHORITATIVE_READY_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"}
)
_AUTHORITATIVE_COMPLETED_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "skipped", "complete", "done"}
)
_AUTHORITATIVE_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        *_AUTHORITATIVE_READY_TASK_STATUSES,
        *_AUTHORITATIVE_COMPLETED_TASK_STATUSES,
        "cancelled",
        "failed",
        "quarantined",
        "rejected",
        "claimed",
        "in_progress",
        "running",
        "blocked",
    }
)

# ---------------------------------------------------------------------------
# Errors (reuse LeaseCoordinator vocabulary; extend only when needed)
# ---------------------------------------------------------------------------


class DatabaseCoordinationError(LeaseError):
    """Base fail-closed error for unified database coordination."""

    code = "DQP_COORDINATION_ERROR"


class DatabaseCoordinationConflictError(LeaseConflictError, DatabaseCoordinationError):
    """Exclusive scope already owned, or identity conflict."""

    code = "DQP_SCOPE_CONFLICT"


class DatabaseCoordinationExpiredError(LeaseExpiredError, DatabaseCoordinationError):
    """Lease or session expired and cannot renew or mutate."""

    code = "DQP_LEASE_EXPIRED"


class DatabaseCoordinationStaleFenceError(
    StaleFencingTokenError, DatabaseCoordinationError
):
    """Stale fencing token or fence epoch on a protected write."""

    code = "DQP_STALE_FENCE"


class DatabaseCoordinationNotReadyError(DependencyNotReadyError):
    """Task claim blocked by unsatisfied dependencies."""

    code = "DQP_DEPENDENCY_NOT_READY"

    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        # DependencyNotReadyError requires evidence; provide a stable default.
        DependencyNotReadyError.__init__(self, message, evidence=dict(evidence or {}))


class DatabaseCoordinationNotOpenError(DatabaseCoordinationError):
    """Operation requires an open coordinator."""

    code = "DQP_NOT_OPEN"


class DatabaseCoordinationBoundsError(DatabaseCoordinationError, ValueError):
    """Payload or lease duration bound exceeded."""

    code = "DQP_BOUNDS"


class DuckDBUnavailableError(DatabaseCoordinationError):
    """Optional DuckDB dependency is not installed."""

    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class LeaseKind(str, Enum):
    """Unified lease kinds under one fencing vocabulary."""

    TASK = "task"
    RESOURCE = "resource"
    PATH = "path"
    MERGE = "merge"
    MAINTENANCE = "maintenance"
    PROVIDER_CAPACITY = "provider_capacity"
    PROVER_CAPACITY = "prover_capacity"


class LeaseMode(str, Enum):
    """Ownership mode for a scope."""

    EXCLUSIVE = "exclusive"
    SHARED = "shared"


class LeaseState(str, Enum):
    ACCEPTED = "accepted"
    RELEASED = "released"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    COMPLETED = "completed"


class AttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RELEASED = "released"
    EXPIRED = "expired"


ClockMs = Callable[[], int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _default_clock_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _utc_iso_from_ms(epoch_ms: int) -> str:
    return (
        datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseCoordinationError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseCoordinationError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseCoordinationBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseCoordinationBoundsError(f"{name} must be a positive integer")
    return value


def _lease_duration_ms(value: Any) -> int:
    duration = _positive_int(int(value), "lease_ms")
    if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
        raise DatabaseCoordinationBoundsError(
            f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]"
        )
    return duration


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise DatabaseCoordinationBoundsError(
            f"{name} exceeds the {max_bytes}-byte bound"
        )
    return raw


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        keys = []
    if keys:
        return {str(key): row[key] for key in keys}
    try:
        return {str(index): row[index] for index in range(len(row))}  # type: ignore[arg-type]
    except Exception:
        return {}


def _row_get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
        upper = name.upper()
        if upper in mapping and mapping[upper] is not None:
            return mapping[upper]
        lower = name.lower()
        if lower in mapping and mapping[lower] is not None:
            return mapping[lower]
    wanted = {name.lower() for name in names}
    for key, value in mapping.items():
        if str(key).lower() in wanted and value is not None:
            return value
    return default


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement or statement.startswith("--"):
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def exclusive_scope_key(
    *,
    lease_kind: LeaseKind | str,
    scope: str,
    resource_kind: str = "",
    resource_id: str = "",
    repository_id: str = "",
    path: str = "",
    task_cid: str = "",
) -> str:
    """Return the canonical exclusive-scope identity used for ownership checks.

    Reuses the LeaseCoordinator idea that one exclusive execution scope admits
    at most one accepted owner, generalized across lease kinds.
    """

    kind = (
        lease_kind
        if isinstance(lease_kind, LeaseKind)
        else LeaseKind(str(lease_kind).strip().lower())
    )
    if kind is LeaseKind.TASK:
        return f"task:{_text(scope or task_cid, 'scope')}"
    if kind is LeaseKind.PATH:
        repo = _text(repository_id or "repository:default", "repository_id")
        path_text = _text(path or scope, "path")
        return f"path:{repo}:{path_text}"
    if kind is LeaseKind.RESOURCE:
        rkind = _text(resource_kind or "resource", "resource_kind")
        rid = _text(resource_id or scope, "resource_id")
        return f"resource:{rkind}:{rid}"
    if kind is LeaseKind.MERGE:
        return f"merge:{_text(scope, 'scope')}"
    if kind is LeaseKind.MAINTENANCE:
        return f"maintenance:{_text(scope or DEFAULT_MAINTENANCE_SCOPE, 'scope')}"
    if kind is LeaseKind.PROVIDER_CAPACITY:
        return f"provider:{_text(scope or resource_id, 'scope')}"
    if kind is LeaseKind.PROVER_CAPACITY:
        return f"prover:{_text(scope or resource_id, 'scope')}"
    return f"{kind.value}:{_text(scope, 'scope')}"


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FencedLease:
    """Unified fenced ownership record for any coordinated scope."""

    INTERFACE: ClassVar[str] = FENCED_LEASE_INTERFACE
    SCHEMA: ClassVar[str] = FENCED_LEASE_SCHEMA

    lease_id: str
    lease_kind: LeaseKind
    scope_key: str
    scope: str
    mode: LeaseMode
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at_ms: int
    expires_at_ms: int
    state: LeaseState
    revision: int
    task_cid: str = ""
    worktree_id: str = ""
    resource_kind: str = ""
    resource_id: str = ""
    repository_id: str = ""
    path: str = ""
    claim_id: str = ""
    attempt_id: str = ""
    attempt_number: int = 0
    idempotency_key: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        kind = self.lease_kind
        if not isinstance(kind, LeaseKind):
            kind = LeaseKind(str(kind).strip().lower())
            object.__setattr__(self, "lease_kind", kind)
        mode = self.mode
        if not isinstance(mode, LeaseMode):
            mode = LeaseMode(str(mode).strip().lower())
            object.__setattr__(self, "mode", mode)
        state = self.state
        if not isinstance(state, LeaseState):
            state = LeaseState(str(state).strip().lower())
            object.__setattr__(self, "state", state)
        object.__setattr__(self, "scope_key", _text(self.scope_key, "scope_key"))
        object.__setattr__(self, "scope", _text(self.scope, "scope", required=False) or self.scope_key)
        object.__setattr__(
            self, "owner_session_id", _text(self.owner_session_id, "owner_session_id")
        )
        object.__setattr__(
            self, "fencing_token", _positive_int(int(self.fencing_token), "fencing_token")
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self, "acquired_at_ms", _nonneg_int(int(self.acquired_at_ms), "acquired_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _nonneg_int(int(self.expires_at_ms), "expires_at_ms")
        )
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self,
            "resource_kind",
            _text(self.resource_kind, "resource_kind", required=False),
        )
        object.__setattr__(
            self, "resource_id", _text(self.resource_id, "resource_id", required=False)
        )
        object.__setattr__(
            self,
            "repository_id",
            _text(self.repository_id, "repository_id", required=False),
        )
        object.__setattr__(self, "path", _text(self.path, "path", required=False))
        object.__setattr__(
            self, "claim_id", _text(self.claim_id, "claim_id", required=False)
        )
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", required=False)
        )
        object.__setattr__(
            self,
            "attempt_number",
            _nonneg_int(int(self.attempt_number), "attempt_number"),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key", required=False),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(dict(self.body or {}), name="body")
            ),
        )

    @property
    def active(self) -> bool:
        return self.state is LeaseState.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "lease_id": self.lease_id,
            "lease_kind": self.lease_kind.value,
            "scope_key": self.scope_key,
            "scope": self.scope,
            "mode": self.mode.value,
            "owner_session_id": self.owner_session_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "acquired_at_ms": int(self.acquired_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "acquired_at": _utc_iso_from_ms(self.acquired_at_ms)
            if self.acquired_at_ms
            else "",
            "expires_at": _utc_iso_from_ms(self.expires_at_ms)
            if self.expires_at_ms
            else "",
            "state": self.state.value,
            "revision": int(self.revision),
            "task_cid": self.task_cid,
            "worktree_id": self.worktree_id,
            "resource_kind": self.resource_kind,
            "resource_id": self.resource_id,
            "repository_id": self.repository_id,
            "path": self.path,
            "claim_id": self.claim_id,
            "attempt_id": self.attempt_id,
            "attempt_number": int(self.attempt_number),
            "idempotency_key": self.idempotency_key,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class TaskClaim:
    """Accepted task claim bound to a fencing epoch and task attempt."""

    INTERFACE: ClassVar[str] = TASK_CLAIM_INTERFACE
    SCHEMA: ClassVar[str] = TASK_CLAIM_SCHEMA

    claim_id: str
    task_cid: str
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    claimed_at_ms: int
    expires_at_ms: int
    state: LeaseState
    revision: int
    attempt_id: str
    attempt_number: int
    lease_id: str
    worktree_id: str = ""
    idempotency_key: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "owner_session_id", _text(self.owner_session_id, "owner_session_id")
        )
        object.__setattr__(
            self, "fencing_token", _positive_int(int(self.fencing_token), "fencing_token")
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self, "claimed_at_ms", _nonneg_int(int(self.claimed_at_ms), "claimed_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _nonneg_int(int(self.expires_at_ms), "expires_at_ms")
        )
        state = self.state
        if not isinstance(state, LeaseState):
            state = LeaseState(str(state).strip().lower())
            object.__setattr__(self, "state", state)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(
            self,
            "attempt_number",
            _positive_int(int(self.attempt_number), "attempt_number"),
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key", required=False),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "claim_id": self.claim_id,
            "task_cid": self.task_cid,
            "owner_session_id": self.owner_session_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "claimed_at_ms": int(self.claimed_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "claimed_at": _utc_iso_from_ms(self.claimed_at_ms)
            if self.claimed_at_ms
            else "",
            "expires_at": _utc_iso_from_ms(self.expires_at_ms)
            if self.expires_at_ms
            else "",
            "state": self.state.value,
            "revision": int(self.revision),
            "attempt_id": self.attempt_id,
            "attempt_number": int(self.attempt_number),
            "lease_id": self.lease_id,
            "worktree_id": self.worktree_id,
            "idempotency_key": self.idempotency_key,
            "body": dict(self.body),
        }

    def as_fenced_lease(self) -> FencedLease:
        return FencedLease(
            lease_id=self.lease_id,
            lease_kind=LeaseKind.TASK,
            scope_key=exclusive_scope_key(
                lease_kind=LeaseKind.TASK, scope=self.task_cid, task_cid=self.task_cid
            ),
            scope=self.task_cid,
            mode=LeaseMode.EXCLUSIVE,
            owner_session_id=self.owner_session_id,
            fencing_token=self.fencing_token,
            fence_epoch=self.fence_epoch,
            acquired_at_ms=self.claimed_at_ms,
            expires_at_ms=self.expires_at_ms,
            state=self.state,
            revision=self.revision,
            task_cid=self.task_cid,
            worktree_id=self.worktree_id,
            claim_id=self.claim_id,
            attempt_id=self.attempt_id,
            attempt_number=self.attempt_number,
            idempotency_key=self.idempotency_key,
            body=dict(self.body),
        )


@dataclass(frozen=True)
class ResourceClaim:
    """Path or capacity resource claim under unified fencing."""

    INTERFACE: ClassVar[str] = RESOURCE_CLAIM_INTERFACE
    SCHEMA: ClassVar[str] = RESOURCE_CLAIM_SCHEMA

    claim_id: str
    resource_kind: str
    resource_id: str
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at_ms: int
    expires_at_ms: int
    state: LeaseState
    revision: int
    lease_id: str
    task_cid: str = ""
    repository_id: str = ""
    path: str = ""
    worktree_id: str = ""
    mode: LeaseMode = LeaseMode.EXCLUSIVE
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(
            self, "resource_kind", _text(self.resource_kind, "resource_kind")
        )
        object.__setattr__(self, "resource_id", _text(self.resource_id, "resource_id"))
        object.__setattr__(
            self, "owner_session_id", _text(self.owner_session_id, "owner_session_id")
        )
        object.__setattr__(
            self, "fencing_token", _positive_int(int(self.fencing_token), "fencing_token")
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self, "acquired_at_ms", _nonneg_int(int(self.acquired_at_ms), "acquired_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _nonneg_int(int(self.expires_at_ms), "expires_at_ms")
        )
        state = self.state
        if not isinstance(state, LeaseState):
            state = LeaseState(str(state).strip().lower())
            object.__setattr__(self, "state", state)
        mode = self.mode
        if not isinstance(mode, LeaseMode):
            mode = LeaseMode(str(mode).strip().lower())
            object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self,
            "repository_id",
            _text(self.repository_id, "repository_id", required=False),
        )
        object.__setattr__(self, "path", _text(self.path, "path", required=False))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "claim_id": self.claim_id,
            "resource_kind": self.resource_kind,
            "resource_id": self.resource_id,
            "owner_session_id": self.owner_session_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "acquired_at_ms": int(self.acquired_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "state": self.state.value,
            "revision": int(self.revision),
            "lease_id": self.lease_id,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "path": self.path,
            "worktree_id": self.worktree_id,
            "mode": self.mode.value,
            "body": dict(self.body),
        }

    def as_fenced_lease(self) -> FencedLease:
        kind = (
            LeaseKind.PATH
            if self.resource_kind == "path"
            else LeaseKind.PROVIDER_CAPACITY
            if self.resource_kind == "provider"
            else LeaseKind.PROVER_CAPACITY
            if self.resource_kind == "prover"
            else LeaseKind.RESOURCE
        )
        return FencedLease(
            lease_id=self.lease_id,
            lease_kind=kind,
            scope_key=exclusive_scope_key(
                lease_kind=kind,
                scope=self.resource_id,
                resource_kind=self.resource_kind,
                resource_id=self.resource_id,
                repository_id=self.repository_id,
                path=self.path,
            ),
            scope=self.resource_id,
            mode=self.mode,
            owner_session_id=self.owner_session_id,
            fencing_token=self.fencing_token,
            fence_epoch=self.fence_epoch,
            acquired_at_ms=self.acquired_at_ms,
            expires_at_ms=self.expires_at_ms,
            state=self.state,
            revision=self.revision,
            task_cid=self.task_cid,
            worktree_id=self.worktree_id,
            resource_kind=self.resource_kind,
            resource_id=self.resource_id,
            repository_id=self.repository_id,
            path=self.path,
            claim_id=self.claim_id,
            body=dict(self.body),
        )


@dataclass(frozen=True)
class MaintenanceLease:
    """Exclusive maintenance lease for schema, backup, or offline recovery."""

    INTERFACE: ClassVar[str] = MAINTENANCE_LEASE_INTERFACE
    SCHEMA: ClassVar[str] = MAINTENANCE_LEASE_SCHEMA

    lease_id: str
    scope: str
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at_ms: int
    expires_at_ms: int
    state: LeaseState
    revision: int
    process_birth_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "scope", _text(self.scope or DEFAULT_MAINTENANCE_SCOPE, "scope")
        )
        object.__setattr__(
            self, "owner_session_id", _text(self.owner_session_id, "owner_session_id")
        )
        object.__setattr__(
            self, "fencing_token", _positive_int(int(self.fencing_token), "fencing_token")
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self, "acquired_at_ms", _nonneg_int(int(self.acquired_at_ms), "acquired_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _nonneg_int(int(self.expires_at_ms), "expires_at_ms")
        )
        state = self.state
        if not isinstance(state, LeaseState):
            state = LeaseState(str(state).strip().lower())
            object.__setattr__(self, "state", state)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self,
            "process_birth_id",
            _text(self.process_birth_id, "process_birth_id", required=False),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def active(self) -> bool:
        return self.state is LeaseState.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "lease_id": self.lease_id,
            "scope": self.scope,
            "owner_session_id": self.owner_session_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "acquired_at_ms": int(self.acquired_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "acquired_at": _utc_iso_from_ms(self.acquired_at_ms)
            if self.acquired_at_ms
            else "",
            "expires_at": _utc_iso_from_ms(self.expires_at_ms)
            if self.expires_at_ms
            else "",
            "state": self.state.value,
            "revision": int(self.revision),
            "process_birth_id": self.process_birth_id,
            "body": dict(self.body),
        }

    def as_fenced_lease(self) -> FencedLease:
        return FencedLease(
            lease_id=self.lease_id,
            lease_kind=LeaseKind.MAINTENANCE,
            scope_key=exclusive_scope_key(
                lease_kind=LeaseKind.MAINTENANCE, scope=self.scope
            ),
            scope=self.scope,
            mode=LeaseMode.EXCLUSIVE,
            owner_session_id=self.owner_session_id,
            fencing_token=self.fencing_token,
            fence_epoch=self.fence_epoch,
            acquired_at_ms=self.acquired_at_ms,
            expires_at_ms=self.expires_at_ms,
            state=self.state,
            revision=self.revision,
            body=dict(self.body),
        )


@dataclass(frozen=True)
class TaskAttempt:
    """Task attempt created atomically with a task claim."""

    SCHEMA: ClassVar[str] = TASK_ATTEMPT_SCHEMA

    attempt_id: str
    task_cid: str
    attempt_number: int
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    started_at_ms: int
    status: AttemptStatus
    revision: int
    finished_at_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt_id": self.attempt_id,
            "task_cid": self.task_cid,
            "attempt_number": int(self.attempt_number),
            "owner_session_id": self.owner_session_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "started_at_ms": int(self.started_at_ms),
            "finished_at_ms": self.finished_at_ms,
            "status": self.status.value
            if isinstance(self.status, AttemptStatus)
            else str(self.status),
            "revision": int(self.revision),
        }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS coordination_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS coordination_tasks (
    task_cid VARCHAR PRIMARY KEY,
    task_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL DEFAULT '',
    registered_at_ms BIGINT NOT NULL,
    ready BOOLEAN NOT NULL DEFAULT TRUE,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS coordination_tasks_ready_idx
    ON coordination_tasks(ready, registered_at_ms, task_cid);

CREATE TABLE IF NOT EXISTS task_dependencies (
    task_cid VARCHAR NOT NULL,
    dependency_task_cid VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, dependency_task_cid)
);
CREATE INDEX IF NOT EXISTS task_dependencies_dep_idx
    ON task_dependencies(dependency_task_cid);

CREATE TABLE IF NOT EXISTS task_completions (
    task_cid VARCHAR PRIMARY KEY,
    completed_at_ms BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS fenced_leases (
    lease_id VARCHAR PRIMARY KEY,
    lease_kind VARCHAR NOT NULL,
    scope_key VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    mode VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    resource_kind VARCHAR NOT NULL DEFAULT '',
    resource_id VARCHAR NOT NULL DEFAULT '',
    repository_id VARCHAR NOT NULL DEFAULT '',
    path VARCHAR NOT NULL DEFAULT '',
    claim_id VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    attempt_number BIGINT NOT NULL DEFAULT 0,
    idempotency_key VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS fenced_leases_scope_state_idx
    ON fenced_leases(scope_key, state, expires_at_ms);
CREATE INDEX IF NOT EXISTS fenced_leases_owner_idx
    ON fenced_leases(owner_session_id, state);
CREATE INDEX IF NOT EXISTS fenced_leases_idempotency_idx
    ON fenced_leases(idempotency_key);

CREATE TABLE IF NOT EXISTS token_history (
    scope_key VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    PRIMARY KEY (scope_key, fencing_token, fence_epoch)
);

CREATE TABLE IF NOT EXISTS lease_events (
    event_id VARCHAR PRIMARY KEY,
    lease_id VARCHAR NOT NULL,
    scope_key VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS lease_events_scope_idx
    ON lease_events(scope_key, observed_at_ms);

CREATE TABLE IF NOT EXISTS task_claims (
    claim_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    claimed_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    released_at_ms BIGINT,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    attempt_id VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    lease_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL DEFAULT '',
    idempotency_key VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS task_claims_task_idx ON task_claims(task_cid, state);
CREATE INDEX IF NOT EXISTS task_claims_idempotency_idx
    ON task_claims(idempotency_key);

CREATE TABLE IF NOT EXISTS task_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS task_attempts_task_number_uidx
    ON task_attempts(task_cid, attempt_number);

CREATE TABLE IF NOT EXISTS resource_claims (
    claim_id VARCHAR PRIMARY KEY,
    resource_kind VARCHAR NOT NULL,
    resource_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    lease_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    repository_id VARCHAR NOT NULL DEFAULT '',
    path VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    mode VARCHAR NOT NULL DEFAULT 'exclusive',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS resource_claims_resource_idx
    ON resource_claims(resource_kind, resource_id, state);

CREATE TABLE IF NOT EXISTS maintenance_leases (
    lease_id VARCHAR PRIMARY KEY,
    scope VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    released_at_ms BIGINT,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS maintenance_leases_scope_idx
    ON maintenance_leases(scope, state);
"""


_COORDINATION_REQUIRED_COLUMNS: Final[Mapping[str, tuple[tuple[str, str], ...]]] = {
    "coordination_metadata": (("key", "VARCHAR"), ("value", "VARCHAR")),
    "coordination_tasks": (
        ("task_cid", "VARCHAR"),
        ("task_id", "VARCHAR"),
        ("worktree_id", "VARCHAR"),
        ("registered_at_ms", "BIGINT"),
        ("ready", "BOOLEAN"),
        ("body_json", "VARCHAR"),
    ),
    "task_dependencies": (
        ("task_cid", "VARCHAR"),
        ("dependency_task_cid", "VARCHAR"),
    ),
    "task_completions": (
        ("task_cid", "VARCHAR"),
        ("completed_at_ms", "BIGINT"),
        ("status", "VARCHAR"),
        ("body_json", "VARCHAR"),
    ),
    "fenced_leases": (
        ("lease_id", "VARCHAR"),
        ("lease_kind", "VARCHAR"),
        ("scope_key", "VARCHAR"),
        ("scope", "VARCHAR"),
        ("mode", "VARCHAR"),
        ("owner_session_id", "VARCHAR"),
        ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"),
        ("acquired_at_ms", "BIGINT"),
        ("expires_at_ms", "BIGINT"),
        ("state", "VARCHAR"),
        ("revision", "BIGINT"),
        ("task_cid", "VARCHAR"),
        ("worktree_id", "VARCHAR"),
        ("resource_kind", "VARCHAR"),
        ("resource_id", "VARCHAR"),
        ("repository_id", "VARCHAR"),
        ("path", "VARCHAR"),
        ("claim_id", "VARCHAR"),
        ("attempt_id", "VARCHAR"),
        ("attempt_number", "BIGINT"),
        ("idempotency_key", "VARCHAR"),
        ("body_json", "VARCHAR"),
    ),
    "token_history": (
        ("scope_key", "VARCHAR"),
        ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"),
        ("recorded_at_ms", "BIGINT"),
    ),
    "lease_events": (
        ("event_id", "VARCHAR"),
        ("lease_id", "VARCHAR"),
        ("scope_key", "VARCHAR"),
        ("event_type", "VARCHAR"),
        ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"),
        ("observed_at_ms", "BIGINT"),
        ("body_json", "VARCHAR"),
    ),
    "task_claims": (
        ("claim_id", "VARCHAR"),
        ("task_cid", "VARCHAR"),
        ("owner_session_id", "VARCHAR"),
        ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"),
        ("claimed_at_ms", "BIGINT"),
        ("expires_at_ms", "BIGINT"),
        ("released_at_ms", "BIGINT"),
        ("state", "VARCHAR"),
        ("revision", "BIGINT"),
        ("attempt_id", "VARCHAR"),
        ("attempt_number", "BIGINT"),
        ("lease_id", "VARCHAR"),
        ("worktree_id", "VARCHAR"),
        ("idempotency_key", "VARCHAR"),
        ("body_json", "VARCHAR"),
    ),
    "task_attempts": (
        ("attempt_id", "VARCHAR"),
        ("task_cid", "VARCHAR"),
        ("attempt_number", "BIGINT"),
        ("owner_session_id", "VARCHAR"),
        ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"),
        ("started_at_ms", "BIGINT"),
        ("finished_at_ms", "BIGINT"),
        ("status", "VARCHAR"),
        ("revision", "BIGINT"),
    ),
    "resource_claims": (
        ("claim_id", "VARCHAR"),
        ("resource_kind", "VARCHAR"),
        ("resource_id", "VARCHAR"),
        ("owner_session_id", "VARCHAR"),
        ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"),
        ("acquired_at_ms", "BIGINT"),
        ("expires_at_ms", "BIGINT"),
        ("state", "VARCHAR"),
        ("revision", "BIGINT"),
        ("lease_id", "VARCHAR"),
        ("task_cid", "VARCHAR"),
        ("repository_id", "VARCHAR"),
        ("path", "VARCHAR"),
        ("worktree_id", "VARCHAR"),
        ("mode", "VARCHAR"),
        ("body_json", "VARCHAR"),
    ),
    "maintenance_leases": (
        ("lease_id", "VARCHAR"),
        ("scope", "VARCHAR"),
        ("owner_session_id", "VARCHAR"),
        ("process_birth_id", "VARCHAR"),
        ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"),
        ("acquired_at_ms", "BIGINT"),
        ("expires_at_ms", "BIGINT"),
        ("released_at_ms", "BIGINT"),
        ("state", "VARCHAR"),
        ("revision", "BIGINT"),
        ("body_json", "VARCHAR"),
    ),
}

_COORDINATION_REQUIRED_INDEXES: Final[frozenset[str]] = frozenset(
    {
        "coordination_tasks_ready_idx",
        "task_dependencies_dep_idx",
        "fenced_leases_scope_state_idx",
        "fenced_leases_owner_idx",
        "fenced_leases_idempotency_idx",
        "lease_events_scope_idx",
        "task_claims_task_idx",
        "task_claims_idempotency_idx",
        "task_attempts_task_number_uidx",
        "resource_claims_resource_idx",
        "maintenance_leases_scope_idx",
    }
)


def _coordination_row_value(row: Any, index: int, name: str) -> Any:
    mapping = _row_mapping(row)
    return _row_get(mapping, name, str(index))


def _decode_coordination_body(
    value: Any,
    *,
    table: str,
    identity: str,
) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value or "{}"))
    except json.JSONDecodeError as exc:
        raise DatabaseCoordinationStaleFenceError(
            f"{table} body for {identity} is not valid JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise DatabaseCoordinationStaleFenceError(
            f"{table} body for {identity} is not a mapping"
        )
    return dict(decoded)


def _validate_coordination_authority(connection: Any) -> None:
    """Validate the landed authority without installing or repairing it."""

    rows = connection.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()
    actual: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        table = str(_coordination_row_value(row, 0, "table_name") or "")
        actual.setdefault(table, []).append(
            (
                str(_coordination_row_value(row, 1, "column_name") or ""),
                str(_coordination_row_value(row, 2, "data_type") or "").upper(),
            )
        )
    for table, expected_columns in _COORDINATION_REQUIRED_COLUMNS.items():
        if tuple(actual.get(table, ())) != expected_columns:
            raise DatabaseCoordinationStaleFenceError(
                f"coordination authority schema mismatch for table {table}"
            )

    index_rows = connection.execute(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE schema_name = 'main'
        ORDER BY index_name
        """
    ).fetchall()
    index_names = {
        str(_coordination_row_value(row, 0, "index_name") or "")
        for row in index_rows
    }
    missing_indexes = _COORDINATION_REQUIRED_INDEXES - index_names
    if missing_indexes:
        raise DatabaseCoordinationStaleFenceError(
            "coordination authority is missing required indexes: "
            + ", ".join(sorted(missing_indexes))
        )

    metadata_rows = connection.execute(
        "SELECT key, value FROM coordination_metadata ORDER BY key"
    ).fetchall()
    metadata = {
        str(_coordination_row_value(row, 0, "key") or ""): str(
            _coordination_row_value(row, 1, "value") or ""
        )
        for row in metadata_rows
    }
    expected_metadata = {
        "interface": DATABASE_COORDINATOR_INTERFACE,
        "schema": DATABASE_COORDINATION_SCHEMA,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise DatabaseCoordinationStaleFenceError(
                f"coordination authority metadata mismatch for {key}"
            )


def _coordination_registry_projection_from_connection(
    connection: Any,
    *,
    validate_authority: bool,
) -> dict[str, Any]:
    """Project exact logical and fencing history from an open connection."""

    if validate_authority:
        _validate_coordination_authority(connection)

    def records(
        *,
        table: str,
        columns: tuple[str, ...],
        integer_columns: frozenset[str] = frozenset(),
        boolean_columns: frozenset[str] = frozenset(),
        body_column: str | None = None,
        identity_column: str,
        order_by: str,
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order_by}"
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {}
            for index, name in enumerate(columns):
                value = _coordination_row_value(row, index, name)
                if name == body_column:
                    continue
                if name in integer_columns:
                    item[name] = int(value or 0)
                elif name in boolean_columns:
                    item[name] = bool(value)
                else:
                    item[name] = str(value or "")
            if body_column is not None:
                body_index = columns.index(body_column)
                item["body"] = _decode_coordination_body(
                    _coordination_row_value(row, body_index, body_column),
                    table=table,
                    identity=str(item.get(identity_column, "")),
                )
            result.append(item)
        return result

    tasks = records(
        table="coordination_tasks",
        columns=("task_cid", "task_id", "worktree_id", "ready", "body_json"),
        boolean_columns=frozenset({"ready"}),
        body_column="body_json",
        identity_column="task_cid",
        order_by="task_cid",
    )
    dependencies = records(
        table="task_dependencies",
        columns=("task_cid", "dependency_task_cid"),
        identity_column="task_cid",
        order_by="task_cid, dependency_task_cid",
    )
    completions = records(
        table="task_completions",
        columns=("task_cid", "status", "body_json"),
        body_column="body_json",
        identity_column="task_cid",
        order_by="task_cid",
    )
    task_claims = records(
        table="task_claims",
        columns=(
            "claim_id",
            "task_cid",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "state",
            "revision",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "worktree_id",
            "idempotency_key",
            "body_json",
        ),
        integer_columns=frozenset(
            {"fencing_token", "fence_epoch", "revision", "attempt_number"}
        ),
        body_column="body_json",
        identity_column="claim_id",
        order_by="claim_id",
    )
    task_attempts = records(
        table="task_attempts",
        columns=(
            "attempt_id",
            "task_cid",
            "attempt_number",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "status",
            "revision",
        ),
        integer_columns=frozenset(
            {"attempt_number", "fencing_token", "fence_epoch", "revision"}
        ),
        identity_column="attempt_id",
        order_by="attempt_id",
    )
    fenced_leases = records(
        table="fenced_leases",
        columns=(
            "lease_id",
            "lease_kind",
            "scope_key",
            "scope",
            "mode",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "state",
            "revision",
            "task_cid",
            "worktree_id",
            "resource_kind",
            "resource_id",
            "repository_id",
            "path",
            "claim_id",
            "attempt_id",
            "attempt_number",
            "idempotency_key",
            "body_json",
        ),
        integer_columns=frozenset(
            {"fencing_token", "fence_epoch", "revision", "attempt_number"}
        ),
        body_column="body_json",
        identity_column="lease_id",
        order_by="lease_id",
    )
    resource_claims = records(
        table="resource_claims",
        columns=(
            "claim_id",
            "resource_kind",
            "resource_id",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "state",
            "revision",
            "lease_id",
            "task_cid",
            "repository_id",
            "path",
            "worktree_id",
            "mode",
            "body_json",
        ),
        integer_columns=frozenset(
            {"fencing_token", "fence_epoch", "revision"}
        ),
        body_column="body_json",
        identity_column="claim_id",
        order_by="claim_id",
    )
    maintenance_leases = records(
        table="maintenance_leases",
        columns=(
            "lease_id",
            "scope",
            "owner_session_id",
            "process_birth_id",
            "fencing_token",
            "fence_epoch",
            "state",
            "revision",
            "body_json",
        ),
        integer_columns=frozenset(
            {"fencing_token", "fence_epoch", "revision"}
        ),
        body_column="body_json",
        identity_column="lease_id",
        order_by="lease_id",
    )

    def grouped_counts(
        items: Sequence[Mapping[str, Any]],
        *columns: str,
    ) -> list[dict[str, Any]]:
        counts: dict[tuple[str, ...], int] = {}
        for item in items:
            key = tuple(str(item.get(column, "")) for column in columns)
            counts[key] = counts.get(key, 0) + 1
        return [
            {**dict(zip(columns, key, strict=True)), "count": count}
            for key, count in sorted(counts.items())
        ]

    def active_count(items: Sequence[Mapping[str, Any]], column: str, state: str) -> int:
        return sum(1 for item in items if item.get(column) == state)

    task_claim_states = grouped_counts(task_claims, "state")
    resource_claim_states = grouped_counts(resource_claims, "state")
    attempt_statuses = grouped_counts(task_attempts, "status")
    fenced_lease_kind_states = grouped_counts(fenced_leases, "lease_kind", "state")
    maintenance_lease_states = grouped_counts(maintenance_leases, "state")
    projection: dict[str, Any] = {
        "schema": COORDINATION_REGISTRY_PROJECTION_SCHEMA,
        "authority_schema": DATABASE_COORDINATION_SCHEMA,
        "tasks": tasks,
        "dependency_edges": dependencies,
        "logical_completions": completions,
        "task_claims": task_claims,
        "task_attempts": task_attempts,
        "fenced_leases": fenced_leases,
        "resource_claims": resource_claims,
        "maintenance_leases": maintenance_leases,
        "counts": {
            "registered_tasks": len(tasks),
            "dependency_edges": len(dependencies),
            "logical_completions": len(completions),
            "task_claims": len(task_claims),
            "active_task_claims": active_count(
                task_claims, "state", LeaseState.ACCEPTED.value
            ),
            "resource_claims": len(resource_claims),
            "active_resource_claims": active_count(
                resource_claims, "state", LeaseState.ACCEPTED.value
            ),
            "task_attempts": len(task_attempts),
            "active_task_attempts": active_count(
                task_attempts, "status", AttemptStatus.RUNNING.value
            ),
            "fenced_leases": len(fenced_leases),
            "active_fenced_leases": active_count(
                fenced_leases, "state", LeaseState.ACCEPTED.value
            ),
            "maintenance_leases": len(maintenance_leases),
            "active_maintenance_leases": active_count(
                maintenance_leases, "state", LeaseState.ACCEPTED.value
            ),
        },
        "task_claim_state_counts": task_claim_states,
        "resource_claim_state_counts": resource_claim_states,
        "task_attempt_status_counts": attempt_statuses,
        "fenced_lease_kind_state_counts": fenced_lease_kind_states,
        "maintenance_lease_state_counts": maintenance_lease_states,
    }
    projection["projection_root"] = _sha256_hex(
        _canonical_json(projection).encode("utf-8")
    )
    return projection


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class DatabaseCoordinator:
    """DuckDB-backed unified fenced lease authority.

    Interface: ``DatabaseCoordinator@1`` with projected records
    ``FencedLease@1``, ``TaskClaim@1``, ``ResourceClaim@1``,
    ``MaintenanceLease@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_COORDINATOR_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_COORDINATION_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        default_lease_ms: int = DEFAULT_LEASE_MS,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseCoordinator; install the optional "
                "duckdb dependency"
            )
        if is_quack_transport_target(database_path):
            self._open_target = str(database_path).strip()
            self._path = Path(self._open_target)
            self._quack_transport = True
        else:
            self._open_target = Path(database_path)
            self._path = self._open_target
            self._quack_transport = False
        self._clock_ms = clock_ms or _default_clock_ms
        self._default_lease_ms = _lease_duration_ms(int(default_lease_ms))
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True
        self._fenced_callback_active = False
        self._fenced_callback_reentry_detected = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseCoordinator":
        with self._lock:
            if self._fenced_callback_active:
                self._fenced_callback_reentry_detected = True
                raise DatabaseCoordinationConflictError(
                    "fenced callback must not re-enter DatabaseCoordinator"
                )
            if self.is_open:
                return self
            if not self._quack_transport:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._open_target)
            try:
                if not self._quack_transport:
                    for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                        connection.execute(statement)
                    for key, value in (
                        ("interface", DATABASE_COORDINATOR_INTERFACE),
                        ("schema", DATABASE_COORDINATION_SCHEMA),
                    ):
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO coordination_metadata(key, value)
                            VALUES (?, ?)
                            """,
                            [key, value],
                        )
                self._connection = connection
                self._closed = False
                if not self._quack_transport:
                    self._commit_if_idle(connection)
                return self
            except Exception:
                try:
                    connection.close()
                except Exception:
                    pass
                raise

    def close(self) -> None:
        with self._lock:
            if self._fenced_callback_active:
                self._fenced_callback_reentry_detected = True
                raise DatabaseCoordinationConflictError(
                    "fenced callback must not re-enter DatabaseCoordinator"
                )
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseCoordinator":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if self._fenced_callback_active:
            self._fenced_callback_reentry_detected = True
            raise DatabaseCoordinationConflictError(
                "fenced callback must not re-enter DatabaseCoordinator"
            )
        if not self.is_open or self._connection is None:
            raise DatabaseCoordinationNotOpenError("DatabaseCoordinator is not open")
        return self._connection

    def _begin(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        try:
            connection.execute("BEGIN TRANSACTION")
        except Exception:
            pass

    def _rollback_if_open(self, connection: Any) -> None:
        try:
            rollback = getattr(connection, "rollback", None)
            if callable(rollback) and getattr(connection, "in_transaction", False):
                rollback()
                return
            raw = getattr(connection, "_connection", None)
            raw_rollback = getattr(raw, "rollback", None) if raw is not None else None
            if callable(raw_rollback):
                raw_rollback()
        except Exception:
            pass

    def _commit_if_idle(self, connection: Any) -> None:
        try:
            if getattr(connection, "in_transaction", False):
                commit = getattr(connection, "commit", None)
                if callable(commit):
                    commit()
                    return
            raw = getattr(connection, "_connection", None)
            raw_commit = getattr(raw, "commit", None) if raw is not None else None
            if callable(raw_commit):
                raw_commit()
                return
            commit = getattr(connection, "commit", None)
            if callable(commit):
                commit()
        except Exception:
            pass

    def _now_ms(self) -> int:
        return int(self._clock_ms())

    # -- task registration / readiness --------------------------------------

    def register_task(
        self,
        *,
        task_cid: str,
        task_id: str | None = None,
        worktree_id: str = "",
        dependency_task_cids: Sequence[str] = (),
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Register a task identity and optional dependency edges."""

        cid = _text(task_cid, "task_cid")
        tid = _text(task_id or cid, "task_id")
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        deps = tuple(
            sorted({_text(item, "dependency_task_cid") for item in dependency_task_cids})
        )
        payload = _bounded_mapping(body, name="body")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT task_cid FROM coordination_tasks WHERE task_cid = ?",
                    [cid],
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO coordination_tasks(
                            task_cid, task_id, worktree_id, registered_at_ms,
                            ready, body_json
                        ) VALUES (?, ?, ?, ?, TRUE, ?)
                        """,
                        [
                            cid,
                            tid,
                            _text(worktree_id, "worktree_id", required=False),
                            now,
                            _canonical_json(payload),
                        ],
                    )
                    for dep in deps:
                        connection.execute(
                            """
                            INSERT INTO task_dependencies(task_cid, dependency_task_cid)
                            VALUES (?, ?)
                            """,
                            [cid, dep],
                        )
                # Already registered: do not UPDATE/DELETE/re-INSERT. DuckDB 1.5
                # can FATAL on unique-index maintenance for those writes.
                self._commit_if_idle(connection)
                return {
                    "task_cid": cid,
                    "task_id": tid,
                    "worktree_id": _text(worktree_id, "worktree_id", required=False),
                    "dependency_task_cids": list(deps),
                    "registered_at_ms": now,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def synchronize_authoritative_task(
        self,
        *,
        task_cid: str,
        task_id: str | None = None,
        dependency_task_cids: Sequence[str] = (),
        authoritative_status: str,
        authoritative_revision: int,
        authoritative_ready: bool,
        authoritative_completed: bool,
        restart_recovery_ready: bool = False,
        restart_recovery_owner_session_id: str = "",
        restart_recovery_binding: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Project one authoritative task into lane-local coordination.

        The task source remains the status and dependency authority.  This
        method only maintains the lane-local scheduling projection needed by
        :meth:`claim_ready_task`; it refuses identity or dependency drift and
        never overwrites an in-flight ``prepared`` completion or active claim.
        A successful local completion may be removed only when the
        authoritative source has explicitly reopened the same task identity.
        """

        cid = _text(task_cid, "task_cid")
        tid = _text(task_id or cid, "task_id")
        status = _text(authoritative_status, "authoritative_status").lower()
        revision = _positive_int(authoritative_revision, "authoritative_revision")
        if status not in _AUTHORITATIVE_TASK_STATUSES:
            raise DatabaseCoordinationConflictError(
                f"unknown authoritative task status {status!r}"
            )
        if type(authoritative_ready) is not bool:  # noqa: E721 - reject truthy values
            raise DatabaseCoordinationBoundsError(
                "authoritative_ready must be a boolean"
            )
        if type(authoritative_completed) is not bool:  # noqa: E721
            raise DatabaseCoordinationBoundsError(
                "authoritative_completed must be a boolean"
            )
        if authoritative_ready and authoritative_completed:
            raise DatabaseCoordinationConflictError(
                "an authoritative task cannot be ready and completed"
            )
        if authoritative_ready and status not in _AUTHORITATIVE_READY_TASK_STATUSES:
            raise DatabaseCoordinationConflictError(
                "authoritative ready=true requires a closed ready status"
            )
        if authoritative_completed != (
            status in _AUTHORITATIVE_COMPLETED_TASK_STATUSES
        ):
            raise DatabaseCoordinationConflictError(
                "authoritative completed flag contradicts task status"
            )
        if type(restart_recovery_ready) is not bool:  # noqa: E721
            raise DatabaseCoordinationBoundsError(
                "restart_recovery_ready must be a boolean"
            )
        if restart_recovery_ready and status != "in_progress":
            raise DatabaseCoordinationConflictError(
                "restart recovery readiness requires in_progress status"
            )
        if restart_recovery_ready and authoritative_ready:
            raise DatabaseCoordinationConflictError(
                "restart recovery and authoritative readiness are disjoint"
            )
        recovery_owner = _text(
            restart_recovery_owner_session_id,
            "restart_recovery_owner_session_id",
            required=False,
        )
        if restart_recovery_ready and not recovery_owner:
            raise DatabaseCoordinationConflictError(
                "restart recovery readiness requires an owner session"
            )
        if not restart_recovery_ready and recovery_owner:
            raise DatabaseCoordinationConflictError(
                "restart recovery owner requires restart recovery readiness"
            )
        recovery_binding = _bounded_mapping(
            restart_recovery_binding,
            name="restart_recovery_binding",
        )
        recovery_binding_fields = frozenset(
            {
                "claim_id",
                "attempt_id",
                "lease_id",
                "owner_session_id",
                "fencing_token",
                "fence_epoch",
            }
        )
        if restart_recovery_ready:
            if frozenset(recovery_binding) != recovery_binding_fields:
                raise DatabaseCoordinationConflictError(
                    "restart recovery binding must contain the exact claim tuple"
                )
            for field_name in ("claim_id", "attempt_id", "lease_id"):
                recovery_binding[field_name] = _text(
                    recovery_binding[field_name],
                    field_name,
                )
            recovery_binding["owner_session_id"] = _text(
                recovery_binding["owner_session_id"],
                "owner_session_id",
            )
            recovery_binding["fencing_token"] = _positive_int(
                recovery_binding["fencing_token"],
                "fencing_token",
            )
            recovery_binding["fence_epoch"] = _positive_int(
                recovery_binding["fence_epoch"],
                "fence_epoch",
            )
            if recovery_binding["owner_session_id"] != recovery_owner:
                raise DatabaseCoordinationConflictError(
                    "restart recovery binding owner does not match projection owner"
                )
        elif recovery_binding:
            raise DatabaseCoordinationConflictError(
                "restart recovery binding requires restart recovery readiness"
            )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        deps = tuple(
            sorted({_text(item, "dependency_task_cid") for item in dependency_task_cids})
        )
        projection_body = _bounded_mapping(
            {
                "authority": "task_source",
                "authoritative_status": status,
                "authoritative_revision": revision,
                "restart_recovery_ready": bool(restart_recovery_ready),
                "restart_recovery_owner_session_id": recovery_owner,
                "restart_recovery_binding": recovery_binding,
            },
            name="authoritative_task_projection",
        )
        scope_key = exclusive_scope_key(
            lease_kind=LeaseKind.TASK,
            scope=cid,
            task_cid=cid,
        )

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    """
                    SELECT task_id, ready, body_json
                    FROM coordination_tasks WHERE task_cid = ?
                    """,
                    [cid],
                ).fetchone()
                existing_ready = False
                existing_body: dict[str, Any] = {}
                changed = False
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO coordination_tasks(
                            task_cid, task_id, worktree_id, registered_at_ms,
                            ready, body_json
                        ) VALUES (?, ?, '', ?, FALSE, ?)
                        """,
                        [cid, tid, now, _canonical_json(projection_body)],
                    )
                    changed = True
                    for dep in deps:
                        connection.execute(
                            """
                            INSERT INTO task_dependencies(
                                task_cid, dependency_task_cid
                            ) VALUES (?, ?)
                            """,
                            [cid, dep],
                        )
                else:
                    existing_mapping = _row_mapping(existing)
                    existing_id = str(
                        _row_get(existing_mapping, "task_id", "0", default="")
                    )
                    existing_ready = bool(
                        _row_get(existing_mapping, "ready", "1", default=False)
                    )
                    existing_body = _decode_coordination_body(
                        _row_get(existing_mapping, "body_json", "2", default="{}"),
                        table="coordination_tasks",
                        identity=cid,
                    )
                    if existing_body.get("authority") == "task_source":
                        prior_revision = _positive_int(
                            existing_body.get("authoritative_revision"),
                            "stored_authoritative_revision",
                        )
                        prior_status = str(
                            existing_body.get("authoritative_status") or ""
                        ).strip().lower()
                        if prior_revision > revision:
                            raise DatabaseCoordinationConflictError(
                                f"authoritative revision regression for {cid}"
                            )
                        if prior_revision == revision and prior_status != status:
                            raise DatabaseCoordinationConflictError(
                                f"same-revision authoritative status drift for {cid}"
                            )
                    if existing_id != tid:
                        raise DatabaseCoordinationConflictError(
                            f"authoritative task identity drift for {cid}"
                        )
                    existing_dep_rows = connection.execute(
                        """
                        SELECT dependency_task_cid FROM task_dependencies
                        WHERE task_cid = ? ORDER BY dependency_task_cid
                        """,
                        [cid],
                    ).fetchall()
                    existing_deps = tuple(
                        str(
                            _row_get(
                                _row_mapping(row),
                                "dependency_task_cid",
                                "0",
                                default="",
                            )
                        )
                        for row in existing_dep_rows
                    )
                    if existing_deps != deps:
                        raise DatabaseCoordinationConflictError(
                            f"authoritative dependency drift for {cid}"
                        )

                expired_lease_ids = self._expire_scope(connection, scope_key, now)
                if expired_lease_ids:
                    changed = True
                active = bool(
                    connection.execute(
                        """
                        SELECT 1 FROM fenced_leases
                        WHERE scope_key = ? AND state = ? AND expires_at_ms > ?
                        LIMIT 1
                        """,
                        [scope_key, LeaseState.ACCEPTED.value, now],
                    ).fetchone()
                )
                if restart_recovery_ready:
                    prior_claim = connection.execute(
                        """
                        SELECT
                            claim.claim_id, claim.task_cid,
                            claim.owner_session_id, claim.fencing_token,
                            claim.fence_epoch, claim.attempt_id, claim.lease_id,
                            claim.attempt_number, claim.state,
                            attempt.task_cid, attempt.owner_session_id,
                            attempt.fencing_token, attempt.fence_epoch,
                            attempt.status,
                            lease.task_cid, lease.owner_session_id,
                            lease.fencing_token, lease.fence_epoch,
                            lease.claim_id, lease.attempt_id, lease.lease_id,
                            lease.state
                        FROM task_claims AS claim
                        JOIN task_attempts AS attempt
                          ON attempt.attempt_id = claim.attempt_id
                        JOIN fenced_leases AS lease
                          ON lease.lease_id = claim.lease_id
                        WHERE claim.task_cid = ? AND claim.claim_id = ?
                          AND claim.attempt_id = ? AND claim.lease_id = ?
                        """,
                        [
                            cid,
                            recovery_binding["claim_id"],
                            recovery_binding["attempt_id"],
                            recovery_binding["lease_id"],
                        ],
                    ).fetchone()
                    if prior_claim is None:
                        raise DatabaseCoordinationConflictError(
                            f"restart recovery for {cid} has no prior local claim"
                        )
                    prior = tuple(prior_claim[index] for index in range(22))
                    expected_identity = (
                        str(recovery_binding["claim_id"]),
                        cid,
                        recovery_owner,
                        int(recovery_binding["fencing_token"]),
                        int(recovery_binding["fence_epoch"]),
                        str(recovery_binding["attempt_id"]),
                        str(recovery_binding["lease_id"]),
                    )
                    observed_identity = tuple(
                        str(value) if index in {0, 1, 2, 5, 6} else int(value)
                        for index, value in enumerate(prior[:7])
                    )
                    if observed_identity != expected_identity:
                        raise DatabaseCoordinationConflictError(
                            f"restart recovery binding does not match latest claim for {cid}"
                        )
                    _positive_int(prior[7], "prior_attempt_number")
                    attempt_identity = (
                        str(prior[9]),
                        str(prior[10]),
                        int(prior[11]),
                        int(prior[12]),
                    )
                    lease_identity = (
                        str(prior[14]),
                        str(prior[15]),
                        int(prior[16]),
                        int(prior[17]),
                        str(prior[18]),
                        str(prior[19]),
                        str(prior[20]),
                    )
                    if attempt_identity != (
                        cid,
                        recovery_owner,
                        int(recovery_binding["fencing_token"]),
                        int(recovery_binding["fence_epoch"]),
                    ) or lease_identity != (
                        cid,
                        recovery_owner,
                        int(recovery_binding["fencing_token"]),
                        int(recovery_binding["fence_epoch"]),
                        str(recovery_binding["claim_id"]),
                        str(recovery_binding["attempt_id"]),
                        str(recovery_binding["lease_id"]),
                    ):
                        raise DatabaseCoordinationConflictError(
                            f"restart recovery local authority rows disagree for {cid}"
                        )
                    states = (str(prior[8]), str(prior[13]), str(prior[21]))
                    if states not in {
                        (
                            LeaseState.ACCEPTED.value,
                            AttemptStatus.RUNNING.value,
                            LeaseState.ACCEPTED.value,
                        ),
                        (
                            LeaseState.EXPIRED.value,
                            AttemptStatus.EXPIRED.value,
                            LeaseState.EXPIRED.value,
                        ),
                    }:
                        raise DatabaseCoordinationConflictError(
                            f"restart recovery local authority state is inadmissible for {cid}"
                        )
                    later_rows = connection.execute(
                        """
                        SELECT claim.state, attempt.status, lease.state
                        FROM task_claims AS claim
                        JOIN task_attempts AS attempt
                          ON attempt.attempt_id = claim.attempt_id
                        JOIN fenced_leases AS lease
                          ON lease.lease_id = claim.lease_id
                        WHERE claim.task_cid = ? AND claim.attempt_number > ?
                        ORDER BY claim.attempt_number, claim.claim_id
                        """,
                        [cid, int(prior[7])],
                    ).fetchall()
                    admitted_later_states = {
                        (
                            LeaseState.RELEASED.value,
                            AttemptStatus.RELEASED.value,
                            LeaseState.RELEASED.value,
                        ),
                        (
                            LeaseState.EXPIRED.value,
                            AttemptStatus.EXPIRED.value,
                            LeaseState.EXPIRED.value,
                        ),
                    }
                    for later_row in later_rows:
                        later_states = tuple(str(later_row[index]) for index in range(3))
                        if later_states not in admitted_later_states:
                            raise DatabaseCoordinationConflictError(
                                f"restart recovery has a later active or inadmissible claim for {cid}"
                            )
                completion = connection.execute(
                    """
                    SELECT status, body_json
                    FROM task_completions WHERE task_cid = ?
                    """,
                    [cid],
                ).fetchone()
                completion_body: dict[str, Any] = {}
                completion_status = (
                    ""
                    if completion is None
                    else str(
                        _row_get(
                            _row_mapping(completion),
                            "status",
                            "0",
                            default="",
                        )
                    )
                )
                if completion is not None:
                    completion_body = _decode_coordination_body(
                        _row_get(
                            _row_mapping(completion),
                            "body_json",
                            "1",
                            default="{}",
                        ),
                        table="task_completions",
                        identity=cid,
                    )

                # An active claim or prepared completion is an exact local
                # two-phase authority.  Projection sync can make it not-ready,
                # but cannot overwrite, delete, or supersede it.
                protected = active or completion_status == PREPARED_COMPLETION_STATUS
                if not protected and authoritative_completed:
                    if completion is None:
                        connection.execute(
                            """
                            INSERT INTO task_completions(
                                task_cid, completed_at_ms, status, body_json
                            ) VALUES (?, ?, ?, ?)
                            """,
                            [
                                cid,
                                now,
                                AttemptStatus.SUCCEEDED.value,
                                _canonical_json(projection_body),
                            ],
                        )
                        completion_status = AttemptStatus.SUCCEEDED.value
                        changed = True
                    elif completion_status != AttemptStatus.SUCCEEDED.value:
                        raise DatabaseCoordinationConflictError(
                            f"unsupported local completion state for {cid}: "
                            f"{completion_status}"
                        )
                    elif (
                        completion_body.get("authority") == "task_source"
                        and completion_body != projection_body
                    ):
                        connection.execute(
                            """
                            UPDATE task_completions SET body_json = ?
                            WHERE task_cid = ? AND status = ?
                            """,
                            [
                                _canonical_json(projection_body),
                                cid,
                                AttemptStatus.SUCCEEDED.value,
                            ],
                        )
                        completion_body = dict(projection_body)
                        changed = True
                elif (
                    not protected
                    and not authoritative_completed
                    and completion_status == AttemptStatus.SUCCEEDED.value
                ):
                    connection.execute(
                        """
                        DELETE FROM task_completions
                        WHERE task_cid = ? AND status = ?
                        """,
                        [cid, AttemptStatus.SUCCEEDED.value],
                    )
                    completion_status = ""
                    changed = True
                elif (
                    not protected
                    and completion_status
                    and completion_status != AttemptStatus.SUCCEEDED.value
                ):
                    raise DatabaseCoordinationConflictError(
                        f"unsupported local completion state for {cid}: "
                        f"{completion_status}"
                    )

                ready = bool(
                    (authoritative_ready or restart_recovery_ready)
                    and not active
                    and not completion_status
                )
                if (existing is None and ready) or (
                    existing is not None
                    and (existing_ready != ready or existing_body != projection_body)
                ):
                    connection.execute(
                        """
                        UPDATE coordination_tasks
                        SET ready = ?, body_json = ? WHERE task_cid = ?
                        """,
                        [ready, _canonical_json(projection_body), cid],
                    )
                    changed = True
                self._commit_if_idle(connection)
                return {
                    "task_cid": cid,
                    "task_id": tid,
                    "dependency_task_cids": list(deps),
                    "authoritative_status": status,
                    "authoritative_revision": revision,
                    "authoritative_ready": bool(authoritative_ready),
                    "authoritative_completed": bool(authoritative_completed),
                    "restart_recovery_ready": bool(restart_recovery_ready),
                    "restart_recovery_owner_session_id": recovery_owner,
                    "restart_recovery_binding": recovery_binding,
                    "ready": ready,
                    "changed": changed,
                    "active_claim_preserved": active,
                    "completion_status": completion_status,
                    "prepared_completion_preserved": (
                        completion_status == PREPARED_COMPLETION_STATUS
                    ),
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def mark_task_complete(
        self,
        task_cid: str,
        *,
        status: str = "succeeded",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Record successful prerequisite completion for dependency readiness."""

        cid = _text(task_cid, "task_cid")
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        status_text = _text(status, "status")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO task_completions(
                        task_cid, completed_at_ms, status, body_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [cid, now, status_text, _canonical_json(dict(body or {}))],
                )
                connection.execute(
                    "UPDATE coordination_tasks SET ready = FALSE WHERE task_cid = ?",
                    [cid],
                )
                self._commit_if_idle(connection)
                return {
                    "task_cid": cid,
                    "completed_at_ms": now,
                    "status": status_text,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def coordination_registry_projection(self) -> dict[str, Any]:
        """Return a deterministic, content-addressed coordination read model.

        The projection deliberately excludes registration, completion, claim,
        lease, and attempt timestamps.  Those values are execution evidence,
        but they are not part of the logical task registry identity.  Stored
        task and completion bodies remain exact and are decoded fail-closed so
        callers can compare the projection with the authority that populated
        this database.

        Claim, attempt, and lease records are represented by exact totals and
        closed state/kind breakdowns.  In particular, the ``active_*`` counts
        reflect rows still stored in the accepted/running state; this method
        does not sweep expiry or otherwise mutate coordination authority.
        """

        with self._lock:
            return _coordination_registry_projection_from_connection(
                self._require(),
                validate_authority=True,
            )

    def claimability(
        self,
        task_cid: str,
        *,
        max_evidence: int = MAX_DEPENDENCY_EVIDENCE,
    ) -> dict[str, Any]:
        """Return dependency readiness for a registered task.

        Reuses the LeaseCoordinator claimability idea: blocked tasks never
        become claimable while prerequisites lack a successful completion.
        """

        cid = _text(task_cid, "task_cid")
        limit = max(1, min(int(max_evidence), MAX_DEPENDENCY_EVIDENCE))
        with self._lock:
            connection = self._require()
            task = connection.execute(
                "SELECT * FROM coordination_tasks WHERE task_cid = ?",
                [cid],
            ).fetchone()
            if task is None:
                raise KeyError(f"unknown task CID: {cid}")
            return self._claimability_unlocked(
                connection,
                cid,
                max_evidence=limit,
            )

    # -- core fencing helpers -----------------------------------------------

    def _expire_scope(
        self,
        connection: Any,
        scope_key: str,
        now: int,
    ) -> list[str]:
        """Expire accepted leases past their deadline (LeaseCoordinator pattern)."""

        rows = connection.execute(
            """
            SELECT lease_id, fencing_token, fence_epoch FROM fenced_leases
            WHERE scope_key = ? AND state = ? AND expires_at_ms <= ?
            """,
            [scope_key, LeaseState.ACCEPTED.value, now],
        ).fetchall()
        expired_ids: list[str] = []
        for row in rows:
            mapping = _row_mapping(row)
            lease_id = str(_row_get(mapping, "lease_id", "0"))
            token = int(_row_get(mapping, "fencing_token", "1", default=0))
            epoch = int(_row_get(mapping, "fence_epoch", "2", default=0))
            connection.execute(
                """
                UPDATE fenced_leases SET state = ?, revision = revision + 1
                WHERE lease_id = ? AND state = ?
                """,
                [LeaseState.EXPIRED.value, lease_id, LeaseState.ACCEPTED.value],
            )
            connection.execute(
                """
                UPDATE task_claims SET state = ?, revision = revision + 1
                WHERE lease_id = ? AND state = ?
                """,
                [LeaseState.EXPIRED.value, lease_id, LeaseState.ACCEPTED.value],
            )
            connection.execute(
                """
                UPDATE resource_claims SET state = ?, revision = revision + 1
                WHERE lease_id = ? AND state = ?
                """,
                [LeaseState.EXPIRED.value, lease_id, LeaseState.ACCEPTED.value],
            )
            connection.execute(
                """
                UPDATE maintenance_leases SET state = ?, revision = revision + 1
                WHERE lease_id = ? AND state = ?
                """,
                [LeaseState.EXPIRED.value, lease_id, LeaseState.ACCEPTED.value],
            )
            connection.execute(
                """
                UPDATE task_attempts
                SET status = ?, finished_at_ms = ?, revision = revision + 1
                WHERE attempt_id IN (
                    SELECT attempt_id FROM fenced_leases WHERE lease_id = ?
                ) AND status = ?
                """,
                [
                    AttemptStatus.EXPIRED.value,
                    now,
                    lease_id,
                    AttemptStatus.RUNNING.value,
                ],
            )
            self._record_event(
                connection,
                lease_id=lease_id,
                scope_key=scope_key,
                event_type="expired",
                fencing_token=token,
                fence_epoch=epoch,
                observed_at_ms=now,
            )
            expired_ids.append(lease_id)
        return expired_ids

    def _active_owners(
        self,
        connection: Any,
        scope_key: str,
        now: int,
    ) -> list[dict[str, Any]]:
        self._expire_scope(connection, scope_key, now)
        rows = connection.execute(
            """
            SELECT * FROM fenced_leases
            WHERE scope_key = ? AND state = ? AND expires_at_ms > ?
            ORDER BY fencing_token DESC, lease_id
            """,
            [scope_key, LeaseState.ACCEPTED.value, now],
        ).fetchall()
        return [_row_mapping(row) for row in rows]

    def _next_fence(
        self,
        connection: Any,
        scope_key: str,
    ) -> tuple[int, int]:
        """Return the next monotonic (fencing_token, fence_epoch)."""

        row = connection.execute(
            """
            SELECT
                COALESCE(MAX(fencing_token), 0) AS max_token,
                COALESCE(MAX(fence_epoch), 0) AS max_epoch
            FROM token_history
            WHERE scope_key = ?
            """,
            [scope_key],
        ).fetchone()
        mapping = _row_mapping(row)
        token = int(_row_get(mapping, "max_token", "0", default=0)) + 1
        epoch = int(_row_get(mapping, "max_epoch", "1", default=0)) + 1
        # Also consider live lease history if token_history is empty after migration.
        live = connection.execute(
            """
            SELECT
                COALESCE(MAX(fencing_token), 0) AS max_token,
                COALESCE(MAX(fence_epoch), 0) AS max_epoch
            FROM fenced_leases
            WHERE scope_key = ?
            """,
            [scope_key],
        ).fetchone()
        live_map = _row_mapping(live)
        token = max(
            token, int(_row_get(live_map, "max_token", "0", default=0)) + 1
        )
        epoch = max(
            epoch, int(_row_get(live_map, "max_epoch", "1", default=0)) + 1
        )
        return token, epoch

    def _record_token(
        self,
        connection: Any,
        *,
        scope_key: str,
        fencing_token: int,
        fence_epoch: int,
        now: int,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO token_history(
                scope_key, fencing_token, fence_epoch, recorded_at_ms
            ) VALUES (?, ?, ?, ?)
            """,
            [scope_key, fencing_token, fence_epoch, now],
        )

    def _record_event(
        self,
        connection: Any,
        *,
        lease_id: str,
        scope_key: str,
        event_type: str,
        fencing_token: int,
        fence_epoch: int,
        observed_at_ms: int,
        body: Mapping[str, Any] | None = None,
    ) -> str:
        event_id = _new_id("lease-event")
        connection.execute(
            """
            INSERT INTO lease_events(
                event_id, lease_id, scope_key, event_type, fencing_token,
                fence_epoch, observed_at_ms, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event_id,
                lease_id,
                scope_key,
                event_type,
                fencing_token,
                fence_epoch,
                observed_at_ms,
                _canonical_json(dict(body or {})),
            ],
        )
        return event_id

    def _insert_fenced_lease(
        self,
        connection: Any,
        lease: FencedLease,
    ) -> None:
        connection.execute(
            """
            INSERT INTO fenced_leases(
                lease_id, lease_kind, scope_key, scope, mode, owner_session_id,
                fencing_token, fence_epoch, acquired_at_ms, expires_at_ms, state,
                revision, task_cid, worktree_id, resource_kind, resource_id,
                repository_id, path, claim_id, attempt_id, attempt_number,
                idempotency_key, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                lease.lease_id,
                lease.lease_kind.value,
                lease.scope_key,
                lease.scope,
                lease.mode.value,
                lease.owner_session_id,
                int(lease.fencing_token),
                int(lease.fence_epoch),
                int(lease.acquired_at_ms),
                int(lease.expires_at_ms),
                lease.state.value,
                int(lease.revision),
                lease.task_cid,
                lease.worktree_id,
                lease.resource_kind,
                lease.resource_id,
                lease.repository_id,
                lease.path,
                lease.claim_id,
                lease.attempt_id,
                int(lease.attempt_number),
                lease.idempotency_key,
                _canonical_json(dict(lease.body)),
            ],
        )

    def _lease_from_row(self, row: Mapping[str, Any] | Any) -> FencedLease:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        return FencedLease(
            lease_id=str(_row_get(mapping, "lease_id", default="")),
            lease_kind=LeaseKind(str(_row_get(mapping, "lease_kind", default="task"))),
            scope_key=str(_row_get(mapping, "scope_key", default="")),
            scope=str(_row_get(mapping, "scope", default="")),
            mode=LeaseMode(str(_row_get(mapping, "mode", default="exclusive"))),
            owner_session_id=str(_row_get(mapping, "owner_session_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=1)),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=1)),
            acquired_at_ms=int(_row_get(mapping, "acquired_at_ms", default=0)),
            expires_at_ms=int(_row_get(mapping, "expires_at_ms", default=0)),
            state=LeaseState(str(_row_get(mapping, "state", default="accepted"))),
            revision=int(_row_get(mapping, "revision", default=1)),
            task_cid=str(_row_get(mapping, "task_cid", default="") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id", default="") or ""),
            resource_kind=str(_row_get(mapping, "resource_kind", default="") or ""),
            resource_id=str(_row_get(mapping, "resource_id", default="") or ""),
            repository_id=str(_row_get(mapping, "repository_id", default="") or ""),
            path=str(_row_get(mapping, "path", default="") or ""),
            claim_id=str(_row_get(mapping, "claim_id", default="") or ""),
            attempt_id=str(_row_get(mapping, "attempt_id", default="") or ""),
            attempt_number=int(_row_get(mapping, "attempt_number", default=0) or 0),
            idempotency_key=str(_row_get(mapping, "idempotency_key", default="") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    def get_lease(self, lease_id: str) -> FencedLease | None:
        lid = _text(lease_id, "lease_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM fenced_leases WHERE lease_id = ?",
                [lid],
            ).fetchone()
            if row is None:
                return None
            return self._lease_from_row(row)

    def list_active_leases(
        self,
        *,
        lease_kind: LeaseKind | str | None = None,
        owner_session_id: str | None = None,
        now_ms: int | None = None,
    ) -> list[FencedLease]:
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                # Sweep all expired accepted leases.
                expired = connection.execute(
                    """
                    SELECT DISTINCT scope_key FROM fenced_leases
                    WHERE state = ? AND expires_at_ms <= ?
                    """,
                    [LeaseState.ACCEPTED.value, now],
                ).fetchall()
                for row in expired:
                    scope = str(_row_get(_row_mapping(row), "scope_key", "0"))
                    self._expire_scope(connection, scope, now)
                clauses = ["state = ?", "expires_at_ms > ?"]
                params: list[Any] = [LeaseState.ACCEPTED.value, now]
                if lease_kind is not None:
                    kind = (
                        lease_kind
                        if isinstance(lease_kind, LeaseKind)
                        else LeaseKind(str(lease_kind).strip().lower())
                    )
                    clauses.append("lease_kind = ?")
                    params.append(kind.value)
                if owner_session_id is not None:
                    clauses.append("owner_session_id = ?")
                    params.append(_text(owner_session_id, "owner_session_id"))
                sql = (
                    "SELECT * FROM fenced_leases WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY acquired_at_ms, lease_id"
                )
                rows = connection.execute(sql, params).fetchall()
                self._commit_if_idle(connection)
                return [self._lease_from_row(row) for row in rows]
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- acquire / renew / release / takeover --------------------------------

    def acquire(
        self,
        *,
        lease_kind: LeaseKind | str,
        scope: str,
        owner_session_id: str,
        mode: LeaseMode | str = LeaseMode.EXCLUSIVE,
        lease_ms: int | None = None,
        task_cid: str = "",
        worktree_id: str = "",
        resource_kind: str = "",
        resource_id: str = "",
        repository_id: str = "",
        path: str = "",
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Acquire a fenced lease for one coordinated scope.

        Exclusive scopes admit only one accepted owner. Shared scopes allow
        concurrent owners (append/fair capacity style).
        """

        kind = (
            lease_kind
            if isinstance(lease_kind, LeaseKind)
            else LeaseKind(str(lease_kind).strip().lower())
        )
        lease_mode = (
            mode if isinstance(mode, LeaseMode) else LeaseMode(str(mode).strip().lower())
        )
        owner = _text(owner_session_id, "owner_session_id")
        duration = (
            self._default_lease_ms
            if lease_ms is None
            else _lease_duration_ms(int(lease_ms))
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        scope_key = exclusive_scope_key(
            lease_kind=kind,
            scope=scope,
            resource_kind=resource_kind,
            resource_id=resource_id,
            repository_id=repository_id,
            path=path,
            task_cid=task_cid or scope,
        )
        idem = _text(idempotency_key, "idempotency_key", required=False)
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                if idem:
                    prior = connection.execute(
                        """
                        SELECT * FROM fenced_leases
                        WHERE idempotency_key = ? AND owner_session_id = ?
                        ORDER BY acquired_at_ms DESC
                        LIMIT 1
                        """,
                        [idem, owner],
                    ).fetchone()
                    if prior is not None:
                        lease = self._lease_from_row(prior)
                        # Response-loss replay: return the original grant when
                        # still accepted or when the same idempotency key is
                        # re-presented after network loss of the first reply.
                        self._commit_if_idle(connection)
                        return lease

                owners = self._active_owners(connection, scope_key, now)
                if owners:
                    exclusive_owners = [
                        item
                        for item in owners
                        if str(item.get("mode") or LeaseMode.EXCLUSIVE.value)
                        == LeaseMode.EXCLUSIVE.value
                    ]
                    if lease_mode is LeaseMode.EXCLUSIVE:
                        # Same-owner re-entry is idempotent for exclusive scopes.
                        same = [
                            item
                            for item in owners
                            if str(item.get("owner_session_id")) == owner
                        ]
                        if same and not any(
                            str(item.get("owner_session_id")) != owner
                            for item in owners
                        ):
                            self._commit_if_idle(connection)
                            return self._lease_from_row(same[0])
                        current = exclusive_owners[0] if exclusive_owners else owners[0]
                        raise DatabaseCoordinationConflictError(
                            f"exclusive scope {scope_key!r} is owned by "
                            f"{current.get('owner_session_id')}"
                        )
                    # Shared acquire is rejected while any exclusive owner is live.
                    if exclusive_owners:
                        raise DatabaseCoordinationConflictError(
                            f"exclusive scope {scope_key!r} is owned by "
                            f"{exclusive_owners[0].get('owner_session_id')}"
                        )

                token, epoch = self._next_fence(connection, scope_key)
                lease_id = _new_id("lease")
                lease = FencedLease(
                    lease_id=lease_id,
                    lease_kind=kind,
                    scope_key=scope_key,
                    scope=_text(scope, "scope"),
                    mode=lease_mode,
                    owner_session_id=owner,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=now + duration,
                    state=LeaseState.ACCEPTED,
                    revision=1,
                    task_cid=_text(task_cid, "task_cid", required=False),
                    worktree_id=_text(worktree_id, "worktree_id", required=False),
                    resource_kind=_text(resource_kind, "resource_kind", required=False),
                    resource_id=_text(resource_id, "resource_id", required=False),
                    repository_id=_text(repository_id, "repository_id", required=False),
                    path=_text(path, "path", required=False),
                    idempotency_key=idem,
                    body=dict(body or {}),
                )
                self._insert_fenced_lease(connection, lease)
                self._record_token(
                    connection,
                    scope_key=scope_key,
                    fencing_token=token,
                    fence_epoch=epoch,
                    now=now,
                )
                self._record_event(
                    connection,
                    lease_id=lease_id,
                    scope_key=scope_key,
                    event_type="acquired",
                    fencing_token=token,
                    fence_epoch=epoch,
                    observed_at_ms=now,
                    body={"owner_session_id": owner, "mode": lease_mode.value},
                )
                self._commit_if_idle(connection)
                return lease
            except Exception:
                self._rollback_if_open(connection)
                raise

    def renew(
        self,
        lease: FencedLease | Mapping[str, Any],
        *,
        lease_ms: int | None = None,
        expected_fencing_token: int | None = None,
        expected_fence_epoch: int | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Renew an accepted lease. Expired or stale fences fail closed."""

        current = (
            lease
            if isinstance(lease, FencedLease)
            else self._lease_from_row(dict(lease))
        )
        duration = (
            self._default_lease_ms
            if lease_ms is None
            else _lease_duration_ms(int(lease_ms))
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        token = (
            int(current.fencing_token)
            if expected_fencing_token is None
            else _positive_int(int(expected_fencing_token), "expected_fencing_token")
        )
        epoch = (
            int(current.fence_epoch)
            if expected_fence_epoch is None
            else _positive_int(int(expected_fence_epoch), "expected_fence_epoch")
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._expire_scope(connection, current.scope_key, now)
                row = connection.execute(
                    "SELECT * FROM fenced_leases WHERE lease_id = ?",
                    [current.lease_id],
                ).fetchone()
                if row is None:
                    raise DatabaseCoordinationError(
                        f"unknown lease: {current.lease_id}"
                    )
                stored = self._lease_from_row(row)
                if stored.state is not LeaseState.ACCEPTED:
                    raise DatabaseCoordinationExpiredError(
                        f"lease {stored.lease_id} is {stored.state.value}; "
                        "expired session cannot renew"
                    )
                if stored.expires_at_ms <= now:
                    self._expire_scope(connection, stored.scope_key, now)
                    raise DatabaseCoordinationExpiredError(
                        f"lease {stored.lease_id} expired; cannot renew"
                    )
                if (
                    int(stored.fencing_token) != token
                    or int(stored.fence_epoch) != epoch
                ):
                    raise DatabaseCoordinationStaleFenceError(
                        "stale fencing epoch or token rejected on renew"
                    )
                expires = now + duration
                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET expires_at_ms = ?, revision = revision + 1
                    WHERE lease_id = ? AND fencing_token = ? AND fence_epoch = ?
                      AND state = ?
                    """,
                    [
                        expires,
                        stored.lease_id,
                        token,
                        epoch,
                        LeaseState.ACCEPTED.value,
                    ],
                )
                if stored.claim_id:
                    connection.execute(
                        """
                        UPDATE task_claims
                        SET expires_at_ms = ?, revision = revision + 1
                        WHERE claim_id = ? AND fencing_token = ? AND fence_epoch = ?
                          AND state = ?
                        """,
                        [
                            expires,
                            stored.claim_id,
                            token,
                            epoch,
                            LeaseState.ACCEPTED.value,
                        ],
                    )
                if stored.resource_kind:
                    connection.execute(
                        """
                        UPDATE resource_claims
                        SET expires_at_ms = ?, revision = revision + 1
                        WHERE claim_id = ? AND fencing_token = ? AND fence_epoch = ?
                          AND state = ?
                        """,
                        [
                            expires,
                            stored.claim_id,
                            token,
                            epoch,
                            LeaseState.ACCEPTED.value,
                        ],
                    )
                if stored.lease_kind is LeaseKind.MAINTENANCE:
                    connection.execute(
                        """
                        UPDATE maintenance_leases
                        SET expires_at_ms = ?, revision = revision + 1
                        WHERE lease_id = ? AND fencing_token = ? AND fence_epoch = ?
                          AND state = ?
                        """,
                        [
                            expires,
                            stored.lease_id,
                            token,
                            epoch,
                            LeaseState.ACCEPTED.value,
                        ],
                    )
                self._record_event(
                    connection,
                    lease_id=stored.lease_id,
                    scope_key=stored.scope_key,
                    event_type="renewed",
                    fencing_token=token,
                    fence_epoch=epoch,
                    observed_at_ms=now,
                    body={"expires_at_ms": expires},
                )
                self._commit_if_idle(connection)
                refreshed = self.get_lease(stored.lease_id)
                assert refreshed is not None
                return refreshed
            except Exception:
                self._rollback_if_open(connection)
                raise

    def release(
        self,
        lease: FencedLease | Mapping[str, Any],
        *,
        reason: str = "released",
        expected_fencing_token: int | None = None,
        expected_fence_epoch: int | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Voluntarily release an accepted lease under the current fence."""

        current = (
            lease
            if isinstance(lease, FencedLease)
            else self._lease_from_row(dict(lease))
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        token = (
            int(current.fencing_token)
            if expected_fencing_token is None
            else _positive_int(int(expected_fencing_token), "expected_fencing_token")
        )
        epoch = (
            int(current.fence_epoch)
            if expected_fence_epoch is None
            else _positive_int(int(expected_fence_epoch), "expected_fence_epoch")
        )
        reason_text = str(reason or "released")[:256]
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._expire_scope(connection, current.scope_key, now)
                row = connection.execute(
                    "SELECT * FROM fenced_leases WHERE lease_id = ?",
                    [current.lease_id],
                ).fetchone()
                if row is None:
                    raise DatabaseCoordinationError(
                        f"unknown lease: {current.lease_id}"
                    )
                stored = self._lease_from_row(row)
                if stored.state is not LeaseState.ACCEPTED:
                    raise DatabaseCoordinationExpiredError(
                        f"lease {stored.lease_id} is {stored.state.value}; "
                        "cannot release"
                    )
                if (
                    int(stored.fencing_token) != token
                    or int(stored.fence_epoch) != epoch
                ):
                    raise DatabaseCoordinationStaleFenceError(
                        "stale fencing epoch or token rejected on release"
                    )
                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET state = ?, revision = revision + 1
                    WHERE lease_id = ? AND fencing_token = ? AND fence_epoch = ?
                    """,
                    [LeaseState.RELEASED.value, stored.lease_id, token, epoch],
                )
                if stored.claim_id:
                    connection.execute(
                        """
                        UPDATE task_claims
                        SET state = ?, released_at_ms = ?, revision = revision + 1
                        WHERE claim_id = ?
                        """,
                        [LeaseState.RELEASED.value, now, stored.claim_id],
                    )
                    if stored.attempt_id:
                        connection.execute(
                            """
                            UPDATE task_attempts
                            SET status = ?, finished_at_ms = ?, revision = revision + 1
                            WHERE attempt_id = ? AND status = ?
                            """,
                            [
                                AttemptStatus.RELEASED.value,
                                now,
                                stored.attempt_id,
                                AttemptStatus.RUNNING.value,
                            ],
                        )
                if stored.resource_kind and stored.claim_id:
                    connection.execute(
                        """
                        UPDATE resource_claims
                        SET state = ?, revision = revision + 1
                        WHERE claim_id = ?
                        """,
                        [LeaseState.RELEASED.value, stored.claim_id],
                    )
                if stored.lease_kind is LeaseKind.MAINTENANCE:
                    connection.execute(
                        """
                        UPDATE maintenance_leases
                        SET state = ?, released_at_ms = ?, revision = revision + 1
                        WHERE lease_id = ?
                        """,
                        [LeaseState.RELEASED.value, now, stored.lease_id],
                    )
                self._record_event(
                    connection,
                    lease_id=stored.lease_id,
                    scope_key=stored.scope_key,
                    event_type="released",
                    fencing_token=token,
                    fence_epoch=epoch,
                    observed_at_ms=now,
                    body={"reason": reason_text},
                )
                self._commit_if_idle(connection)
                refreshed = self.get_lease(stored.lease_id)
                assert refreshed is not None
                return refreshed
            except Exception:
                self._rollback_if_open(connection)
                raise

    def takeover(
        self,
        *,
        lease_kind: LeaseKind | str,
        scope: str,
        owner_session_id: str,
        mode: LeaseMode | str = LeaseMode.EXCLUSIVE,
        lease_ms: int | None = None,
        task_cid: str = "",
        worktree_id: str = "",
        resource_kind: str = "",
        resource_id: str = "",
        repository_id: str = "",
        path: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Take over an expired or released exclusive scope with a new fence.

        Active non-expired ownership is never pre-empted (LeaseCoordinator
        steal semantics).
        """

        kind = (
            lease_kind
            if isinstance(lease_kind, LeaseKind)
            else LeaseKind(str(lease_kind).strip().lower())
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        scope_key = exclusive_scope_key(
            lease_kind=kind,
            scope=scope,
            resource_kind=resource_kind,
            resource_id=resource_id,
            repository_id=repository_id,
            path=path,
            task_cid=task_cid or scope,
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                owners = self._active_owners(connection, scope_key, now)
                if owners:
                    raise DatabaseCoordinationConflictError(
                        f"cannot takeover active scope {scope_key!r} owned by "
                        f"{owners[0].get('owner_session_id')}"
                    )
                # Mark any prior accepted rows (already expired) as superseded
                # so history retains monotonic epochs.
                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET state = CASE
                        WHEN state = ? THEN ?
                        ELSE state
                    END
                    WHERE scope_key = ? AND state IN (?, ?)
                    """,
                    [
                        LeaseState.EXPIRED.value,
                        LeaseState.SUPERSEDED.value,
                        scope_key,
                        LeaseState.EXPIRED.value,
                        LeaseState.RELEASED.value,
                    ],
                )
                self._commit_if_idle(connection)
            except Exception:
                self._rollback_if_open(connection)
                raise
        # Re-enter acquire path after superseding prior ownership.
        return self.acquire(
            lease_kind=kind,
            scope=scope,
            owner_session_id=owner_session_id,
            mode=mode,
            lease_ms=lease_ms,
            task_cid=task_cid,
            worktree_id=worktree_id,
            resource_kind=resource_kind,
            resource_id=resource_id,
            repository_id=repository_id,
            path=path,
            body=body,
            now_ms=now,
        )

    def protect_write(
        self,
        lease: FencedLease | Mapping[str, Any],
        *,
        expected_fencing_token: int | None = None,
        expected_fence_epoch: int | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Validate fence authority for a protected mutation.

        Rejects expired leases and stale fencing epochs/tokens.
        """

        current = (
            lease
            if isinstance(lease, FencedLease)
            else self._lease_from_row(dict(lease))
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        token = (
            int(current.fencing_token)
            if expected_fencing_token is None
            else _positive_int(int(expected_fencing_token), "expected_fencing_token")
        )
        epoch = (
            int(current.fence_epoch)
            if expected_fence_epoch is None
            else _positive_int(int(expected_fence_epoch), "expected_fence_epoch")
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._expire_scope(connection, current.scope_key, now)
                row = connection.execute(
                    "SELECT * FROM fenced_leases WHERE lease_id = ?",
                    [current.lease_id],
                ).fetchone()
                if row is None:
                    raise DatabaseCoordinationError(
                        f"unknown lease: {current.lease_id}"
                    )
                stored = self._lease_from_row(row)
                if stored.state is not LeaseState.ACCEPTED or stored.expires_at_ms <= now:
                    raise DatabaseCoordinationExpiredError(
                        f"lease {stored.lease_id} cannot mutate; expired or inactive"
                    )
                if (
                    int(stored.fencing_token) != token
                    or int(stored.fence_epoch) != epoch
                ):
                    raise DatabaseCoordinationStaleFenceError(
                        "stale fencing epoch rejected on protected write"
                    )
                self._record_event(
                    connection,
                    lease_id=stored.lease_id,
                    scope_key=stored.scope_key,
                    event_type="protected_write",
                    fencing_token=token,
                    fence_epoch=epoch,
                    observed_at_ms=now,
                )
                self._commit_if_idle(connection)
                return stored
            except Exception:
                self._rollback_if_open(connection)
                raise

    @staticmethod
    def _task_claim_identity(
        claim: TaskClaim | Mapping[str, Any],
    ) -> dict[str, Any]:
        mapping = claim.to_dict() if isinstance(claim, TaskClaim) else dict(claim)
        return {
            "task_cid": _text(mapping.get("task_cid"), "task_cid"),
            "claim_id": _text(mapping.get("claim_id"), "claim_id"),
            "attempt_id": _text(mapping.get("attempt_id"), "attempt_id"),
            "attempt_number": _positive_int(
                int(mapping.get("attempt_number") or 0), "attempt_number"
            ),
            "owner_session_id": _text(
                mapping.get("owner_session_id"), "owner_session_id"
            ),
            "lease_id": _text(mapping.get("lease_id"), "lease_id"),
            "fencing_token": _positive_int(
                int(mapping.get("fencing_token") or 0), "fencing_token"
            ),
            "fence_epoch": _positive_int(
                int(mapping.get("fence_epoch") or 0), "fence_epoch"
            ),
        }

    @staticmethod
    def _resource_claim_identity(
        claim: ResourceClaim | Mapping[str, Any],
    ) -> dict[str, Any]:
        mapping = claim.to_dict() if isinstance(claim, ResourceClaim) else dict(claim)
        raw_body = mapping.get("body")
        if raw_body is not None and not isinstance(raw_body, Mapping):
            raise DatabaseCoordinationStaleFenceError(
                "resource claim body is not a mapping"
            )
        return {
            "claim_id": _text(mapping.get("claim_id"), "claim_id"),
            "resource_kind": _text(
                mapping.get("resource_kind"), "resource_kind"
            ),
            "resource_id": _text(mapping.get("resource_id"), "resource_id"),
            "owner_session_id": _text(
                mapping.get("owner_session_id"), "owner_session_id"
            ),
            "lease_id": _text(mapping.get("lease_id"), "lease_id"),
            "task_cid": _text(
                mapping.get("task_cid"), "task_cid", required=False
            ),
            "repository_id": _text(
                mapping.get("repository_id"), "repository_id", required=False
            ),
            "path": _text(mapping.get("path"), "path", required=False),
            "worktree_id": _text(
                mapping.get("worktree_id"), "worktree_id", required=False
            ),
            "mode": (
                mapping.get("mode")
                if isinstance(mapping.get("mode"), LeaseMode)
                else LeaseMode(str(mapping.get("mode") or "").strip().lower())
            ),
            "fencing_token": _positive_int(
                int(mapping.get("fencing_token") or 0), "fencing_token"
            ),
            "fence_epoch": _positive_int(
                int(mapping.get("fence_epoch") or 0), "fence_epoch"
            ),
            "body": _bounded_mapping(
                dict(raw_body or {}),
                name="resource_claim_body",
            ),
        }

    def _protect_task_claim_unlocked(
        self,
        connection: Any,
        *,
        identity: Mapping[str, Any],
        now: int,
        expected_attempt_status: AttemptStatus,
        allow_logically_completed: bool,
        record_event: bool,
        expected_lease_state: LeaseState = LeaseState.ACCEPTED,
    ) -> FencedLease:
        """Validate one exact task/claim/attempt/fence tuple inside a transaction."""

        task_cid = str(identity["task_cid"])
        claim_id = str(identity["claim_id"])
        attempt_id = str(identity["attempt_id"])
        attempt_number = int(identity["attempt_number"])
        owner_session_id = str(identity["owner_session_id"])
        lease_id = str(identity["lease_id"])
        token = int(identity["fencing_token"])
        epoch = int(identity["fence_epoch"])
        scope_key = exclusive_scope_key(
            lease_kind=LeaseKind.TASK,
            scope=task_cid,
            task_cid=task_cid,
        )

        # Expire before inspecting any projection so a deadline boundary never
        # validates stale authority.
        self._expire_scope(connection, scope_key, now)
        claim_row = connection.execute(
            "SELECT * FROM task_claims WHERE claim_id = ?",
            [claim_id],
        ).fetchone()
        lease_row = connection.execute(
            "SELECT * FROM fenced_leases WHERE lease_id = ?",
            [lease_id],
        ).fetchone()
        attempt_row = connection.execute(
            "SELECT * FROM task_attempts WHERE attempt_id = ?",
            [attempt_id],
        ).fetchone()
        if claim_row is None or lease_row is None or attempt_row is None:
            missing = [
                name
                for name, row in (
                    ("task_claim", claim_row),
                    ("fenced_lease", lease_row),
                    ("task_attempt", attempt_row),
                )
                if row is None
            ]
            raise DatabaseCoordinationStaleFenceError(
                "task claim authority is incomplete: " + ", ".join(missing)
            )

        claim_mapping = _row_mapping(claim_row)
        lease = self._lease_from_row(lease_row)
        attempt_mapping = _row_mapping(attempt_row)
        exact_checks = {
            "claim.task_cid": str(
                _row_get(claim_mapping, "task_cid", default="") or ""
            )
            == task_cid,
            "claim.owner_session_id": str(
                _row_get(claim_mapping, "owner_session_id", default="") or ""
            )
            == owner_session_id,
            "claim.attempt_id": str(
                _row_get(claim_mapping, "attempt_id", default="") or ""
            )
            == attempt_id,
            "claim.attempt_number": int(
                _row_get(claim_mapping, "attempt_number", default=0)
            )
            == attempt_number,
            "claim.lease_id": str(
                _row_get(claim_mapping, "lease_id", default="") or ""
            )
            == lease_id,
            "claim.fencing_token": int(
                _row_get(claim_mapping, "fencing_token", default=0)
            )
            == token,
            "claim.fence_epoch": int(
                _row_get(claim_mapping, "fence_epoch", default=0)
            )
            == epoch,
            "lease.lease_kind": lease.lease_kind is LeaseKind.TASK,
            "lease.scope_key": lease.scope_key == scope_key,
            "lease.task_cid": lease.task_cid == task_cid,
            "lease.owner_session_id": lease.owner_session_id == owner_session_id,
            "lease.claim_id": lease.claim_id == claim_id,
            "lease.attempt_id": lease.attempt_id == attempt_id,
            "lease.attempt_number": int(lease.attempt_number) == attempt_number,
            "lease.fencing_token": int(lease.fencing_token) == token,
            "lease.fence_epoch": int(lease.fence_epoch) == epoch,
            "attempt.task_cid": str(
                _row_get(attempt_mapping, "task_cid", default="") or ""
            )
            == task_cid,
            "attempt.attempt_number": int(
                _row_get(attempt_mapping, "attempt_number", default=0)
            )
            == attempt_number,
            "attempt.owner_session_id": str(
                _row_get(attempt_mapping, "owner_session_id", default="") or ""
            )
            == owner_session_id,
            "attempt.fencing_token": int(
                _row_get(attempt_mapping, "fencing_token", default=0)
            )
            == token,
            "attempt.fence_epoch": int(
                _row_get(attempt_mapping, "fence_epoch", default=0)
            )
            == epoch,
        }
        mismatches = [name for name, matches in exact_checks.items() if not matches]
        if mismatches:
            raise DatabaseCoordinationStaleFenceError(
                "stale or mismatched task claim authority: "
                + ", ".join(mismatches)
            )

        claim_state = LeaseState(
            str(_row_get(claim_mapping, "state", default="accepted"))
        )
        claim_expires_at_ms = int(
            _row_get(claim_mapping, "expires_at_ms", default=0)
        )
        attempt_status = AttemptStatus(
            str(_row_get(attempt_mapping, "status", default="running"))
        )
        if (
            claim_state is not expected_lease_state
            or lease.state is not expected_lease_state
        ):
            raise DatabaseCoordinationExpiredError(
                f"task claim {claim_id} is not {expected_lease_state.value}"
            )
        if expected_lease_state is LeaseState.ACCEPTED and (
            claim_expires_at_ms <= now or lease.expires_at_ms <= now
        ):
            raise DatabaseCoordinationExpiredError(
                f"task claim {claim_id} cannot mutate; expired or inactive"
            )
        if claim_expires_at_ms != lease.expires_at_ms:
            raise DatabaseCoordinationStaleFenceError(
                "task claim and fenced lease expiry projections disagree"
            )
        if attempt_status is not expected_attempt_status:
            raise DatabaseCoordinationExpiredError(
                f"task attempt {attempt_id} is {attempt_status.value}; expected "
                f"{expected_attempt_status.value}"
            )

        latest_fence_row = connection.execute(
            """
            SELECT COALESCE(MAX(fencing_token), 0) AS max_token,
                   COALESCE(MAX(fence_epoch), 0) AS max_epoch
            FROM token_history WHERE scope_key = ?
            """,
            [scope_key],
        ).fetchone()
        latest_fence = _row_mapping(latest_fence_row)
        if (
            int(_row_get(latest_fence, "max_token", "0", default=0)) != token
            or int(_row_get(latest_fence, "max_epoch", "1", default=0)) != epoch
        ):
            raise DatabaseCoordinationStaleFenceError(
                "task claim is not the latest fencing epoch and token"
            )

        completion_row = connection.execute(
            "SELECT status FROM task_completions WHERE task_cid = ?",
            [task_cid],
        ).fetchone()
        if completion_row is not None and not allow_logically_completed:
            raise DatabaseCoordinationNotReadyError(
                f"task {task_cid} already has a logical completion",
                evidence={
                    "task_cid": task_cid,
                    "completion_status": str(
                        _row_get(
                            _row_mapping(completion_row),
                            "status",
                            "0",
                            default="",
                        )
                    ),
                    "reason": "already_completed",
                },
            )
        if completion_row is not None and allow_logically_completed:
            self._task_completion_for_identity_unlocked(
                connection,
                identity=identity,
                required=True,
                expected_statuses=(
                    PREPARED_COMPLETION_STATUS,
                    AttemptStatus.SUCCEEDED.value,
                ),
            )

        if record_event:
            self._record_event(
                connection,
                lease_id=lease_id,
                scope_key=scope_key,
                event_type="protected_task_write",
                fencing_token=token,
                fence_epoch=epoch,
                observed_at_ms=now,
                body={
                    "task_cid": task_cid,
                    "claim_id": claim_id,
                    "attempt_id": attempt_id,
                    "owner_session_id": owner_session_id,
                    "attempt_status": attempt_status.value,
                },
            )
        return lease

    def _protect_resource_claim_unlocked(
        self,
        connection: Any,
        *,
        identity: Mapping[str, Any],
        now: int,
        record_event: bool,
    ) -> FencedLease:
        """Validate one exact, exclusive resource claim inside a transaction."""

        claim_id = str(identity["claim_id"])
        resource_kind = str(identity["resource_kind"])
        resource_id = str(identity["resource_id"])
        owner_session_id = str(identity["owner_session_id"])
        lease_id = str(identity["lease_id"])
        task_cid = str(identity["task_cid"])
        repository_id = str(identity["repository_id"])
        path = str(identity["path"])
        worktree_id = str(identity["worktree_id"])
        mode = identity["mode"]
        token = int(identity["fencing_token"])
        epoch = int(identity["fence_epoch"])
        expected_body = dict(identity["body"])
        expected_kind = (
            LeaseKind.PATH
            if resource_kind == "path"
            else LeaseKind.PROVIDER_CAPACITY
            if resource_kind == "provider"
            else LeaseKind.PROVER_CAPACITY
            if resource_kind == "prover"
            else LeaseKind.MERGE
            if resource_kind == "merge"
            else LeaseKind.RESOURCE
        )
        scope = (
            path or resource_id
            if expected_kind is LeaseKind.PATH
            else resource_id
        )
        scope_key = exclusive_scope_key(
            lease_kind=expected_kind,
            scope=scope,
            resource_kind=resource_kind,
            resource_id=resource_id,
            repository_id=repository_id,
            path=path,
            task_cid=task_cid,
        )

        if mode is not LeaseMode.EXCLUSIVE:
            raise DatabaseCoordinationConflictError(
                "cross-store writer resource claim must be exclusive"
            )

        self._expire_scope(connection, scope_key, now)
        claim_row = connection.execute(
            "SELECT * FROM resource_claims WHERE claim_id = ?",
            [claim_id],
        ).fetchone()
        lease_row = connection.execute(
            "SELECT * FROM fenced_leases WHERE lease_id = ?",
            [lease_id],
        ).fetchone()
        if claim_row is None or lease_row is None:
            missing = [
                name
                for name, row in (
                    ("resource_claim", claim_row),
                    ("fenced_lease", lease_row),
                )
                if row is None
            ]
            raise DatabaseCoordinationStaleFenceError(
                "writer resource authority is incomplete: " + ", ".join(missing)
            )

        claim_mapping = _row_mapping(claim_row)
        lease = self._lease_from_row(lease_row)
        stored_body = _decode_coordination_body(
            _row_get(claim_mapping, "body_json", default="{}"),
            table="resource_claims",
            identity=claim_id,
        )
        exact_checks = {
            "claim.resource_kind": str(
                _row_get(claim_mapping, "resource_kind", default="") or ""
            )
            == resource_kind,
            "claim.resource_id": str(
                _row_get(claim_mapping, "resource_id", default="") or ""
            )
            == resource_id,
            "claim.owner_session_id": str(
                _row_get(claim_mapping, "owner_session_id", default="") or ""
            )
            == owner_session_id,
            "claim.lease_id": str(
                _row_get(claim_mapping, "lease_id", default="") or ""
            )
            == lease_id,
            "claim.task_cid": str(
                _row_get(claim_mapping, "task_cid", default="") or ""
            )
            == task_cid,
            "claim.repository_id": str(
                _row_get(claim_mapping, "repository_id", default="") or ""
            )
            == repository_id,
            "claim.path": str(
                _row_get(claim_mapping, "path", default="") or ""
            )
            == path,
            "claim.worktree_id": str(
                _row_get(claim_mapping, "worktree_id", default="") or ""
            )
            == worktree_id,
            "claim.mode": str(
                _row_get(claim_mapping, "mode", default="") or ""
            )
            == LeaseMode.EXCLUSIVE.value,
            "claim.fencing_token": int(
                _row_get(claim_mapping, "fencing_token", default=0)
            )
            == token,
            "claim.fence_epoch": int(
                _row_get(claim_mapping, "fence_epoch", default=0)
            )
            == epoch,
            "claim.body": stored_body == expected_body,
            "lease.lease_kind": lease.lease_kind is expected_kind,
            "lease.scope_key": lease.scope_key == scope_key,
            "lease.mode": lease.mode is LeaseMode.EXCLUSIVE,
            "lease.owner_session_id": lease.owner_session_id == owner_session_id,
            "lease.claim_id": lease.claim_id == claim_id,
            "lease.task_cid": lease.task_cid == task_cid,
            "lease.resource_kind": lease.resource_kind == resource_kind,
            "lease.resource_id": lease.resource_id == resource_id,
            "lease.repository_id": lease.repository_id == repository_id,
            "lease.path": lease.path == path,
            "lease.worktree_id": lease.worktree_id == worktree_id,
            "lease.fencing_token": int(lease.fencing_token) == token,
            "lease.fence_epoch": int(lease.fence_epoch) == epoch,
            "lease.body": dict(lease.body) == expected_body,
        }
        mismatches = [name for name, matches in exact_checks.items() if not matches]
        if mismatches:
            raise DatabaseCoordinationStaleFenceError(
                "stale or mismatched writer resource authority: "
                + ", ".join(mismatches)
            )

        latest_fence_row = connection.execute(
            """
            SELECT COALESCE(MAX(fencing_token), 0) AS max_token,
                   COALESCE(MAX(fence_epoch), 0) AS max_epoch
            FROM token_history WHERE scope_key = ?
            """,
            [scope_key],
        ).fetchone()
        latest_fence = _row_mapping(latest_fence_row)
        if (
            int(_row_get(latest_fence, "max_token", "0", default=0)) != token
            or int(_row_get(latest_fence, "max_epoch", "1", default=0)) != epoch
        ):
            raise DatabaseCoordinationStaleFenceError(
                "writer resource claim is not the latest fencing epoch and token"
            )

        claim_state = LeaseState(
            str(_row_get(claim_mapping, "state", default="accepted"))
        )
        claim_expires_at_ms = int(
            _row_get(claim_mapping, "expires_at_ms", default=0)
        )
        if (
            claim_state is not LeaseState.ACCEPTED
            or lease.state is not LeaseState.ACCEPTED
        ):
            raise DatabaseCoordinationExpiredError(
                f"writer resource claim {claim_id} is not accepted"
            )
        if claim_expires_at_ms <= now or lease.expires_at_ms <= now:
            raise DatabaseCoordinationExpiredError(
                f"writer resource claim {claim_id} cannot mutate; expired or inactive"
            )
        if claim_expires_at_ms != lease.expires_at_ms:
            raise DatabaseCoordinationStaleFenceError(
                "writer resource claim and fenced lease expiry projections disagree"
            )

        if record_event:
            self._record_event(
                connection,
                lease_id=lease_id,
                scope_key=scope_key,
                event_type="protected_resource_write",
                fencing_token=token,
                fence_epoch=epoch,
                observed_at_ms=now,
                body={
                    "claim_id": claim_id,
                    "task_cid": task_cid,
                    "owner_session_id": owner_session_id,
                    "resource_kind": resource_kind,
                    "resource_id": resource_id,
                },
            )
        return lease

    def _task_completion_for_identity_unlocked(
        self,
        connection: Any,
        *,
        identity: Mapping[str, Any],
        required: bool,
        expected_statuses: Sequence[str] = (AttemptStatus.SUCCEEDED.value,),
    ) -> dict[str, Any] | None:
        task_cid = str(identity["task_cid"])
        completion_row = connection.execute(
            """
            SELECT completed_at_ms, status, body_json
            FROM task_completions WHERE task_cid = ?
            """,
            [task_cid],
        ).fetchone()
        if completion_row is None:
            if required:
                raise DatabaseCoordinationNotReadyError(
                    f"task {task_cid} has no logical completion to settle",
                    evidence={
                        "task_cid": task_cid,
                        "claim_id": str(identity["claim_id"]),
                        "attempt_id": str(identity["attempt_id"]),
                        "reason": "completion_missing",
                    },
                )
            return None
        completion_mapping = _row_mapping(completion_row)
        completion_status = str(
            _row_get(completion_mapping, "status", "1", default="") or ""
        )
        completion_body_raw = _row_get(
            completion_mapping,
            "body_json",
            "2",
            default="{}",
        )
        try:
            completion_body = json.loads(str(completion_body_raw or "{}"))
        except json.JSONDecodeError as exc:
            raise DatabaseCoordinationStaleFenceError(
                "logical completion body is not valid JSON"
            ) from exc
        if not isinstance(completion_body, Mapping):
            raise DatabaseCoordinationStaleFenceError(
                "logical completion body is not a mapping"
            )
        expected_completion = {
            "attempt_id": str(identity["attempt_id"]),
            "attempt_number": int(identity["attempt_number"]),
            "claim_id": str(identity["claim_id"]),
            "lease_id": str(identity["lease_id"]),
            "owner_session_id": str(identity["owner_session_id"]),
            "fencing_token": int(identity["fencing_token"]),
            "fence_epoch": int(identity["fence_epoch"]),
        }
        allowed_statuses = {
            _text(item, "expected_completion_status")
            for item in expected_statuses
        }
        if completion_status not in allowed_statuses or any(
            completion_body.get(name) != expected
            for name, expected in expected_completion.items()
        ):
            raise DatabaseCoordinationStaleFenceError(
                "logical completion belongs to a different task authority"
            )
        return {
            "completed_at_ms": int(
                _row_get(completion_mapping, "completed_at_ms", "0", default=0)
            ),
            "status": completion_status,
            "body": dict(completion_body),
        }

    @staticmethod
    def _preparation_digest(body: Mapping[str, Any]) -> str:
        payload = dict(body)
        payload.pop("preparation_digest", None)
        payload.pop("control_completion", None)
        payload.pop("cross_store_guard", None)
        return _sha256_hex(_canonical_json(payload).encode("utf-8"))

    def _validate_preparation_mapping(
        self,
        prepared: Mapping[str, Any],
        *,
        task_cid: str,
    ) -> dict[str, Any]:
        normalized = dict(prepared)
        identity = self._task_claim_identity(normalized)
        if str(identity["task_cid"]) != task_cid:
            raise DatabaseCoordinationStaleFenceError(
                "prepared completion task identity does not match its row"
            )
        if normalized.get("schema") != TASK_COMPLETION_PREPARATION_SCHEMA:
            raise DatabaseCoordinationStaleFenceError(
                "prepared completion schema is not authoritative"
            )
        preparation_digest = _text(
            normalized.get("preparation_digest"),
            "preparation_digest",
        )
        if preparation_digest != self._preparation_digest(normalized):
            raise DatabaseCoordinationStaleFenceError(
                "prepared completion digest does not match its bound body"
            )
        normalized["control_expected_revision"] = _positive_int(
            int(normalized.get("control_expected_revision") or 0),
            "control_expected_revision",
        )
        normalized["control_expected_status"] = _text(
            normalized.get("control_expected_status"),
            "control_expected_status",
        )
        normalized["evidence_digest"] = _text(
            normalized.get("evidence_digest"),
            "evidence_digest",
        )
        normalized["preparation_digest"] = preparation_digest
        normalized["prepared_at_ms"] = _nonneg_int(
            int(normalized.get("prepared_at_ms") or 0),
            "prepared_at_ms",
        )
        return normalized

    def _prepared_completion_unlocked(
        self,
        connection: Any,
        task_cid: str,
        *,
        required: bool,
        include_promoted: bool = False,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT completed_at_ms, status, body_json
            FROM task_completions WHERE task_cid = ?
            """,
            [task_cid],
        ).fetchone()
        if row is None:
            if required:
                raise DatabaseCoordinationNotReadyError(
                    f"task {task_cid} has no prepared completion",
                    evidence={"task_cid": task_cid, "reason": "preparation_missing"},
                )
            return None
        mapping = _row_mapping(row)
        status = str(_row_get(mapping, "status", "1", default="") or "")
        allowed_statuses = {PREPARED_COMPLETION_STATUS}
        if include_promoted:
            allowed_statuses.add(AttemptStatus.SUCCEEDED.value)
        if status not in allowed_statuses:
            if required:
                raise DatabaseCoordinationNotReadyError(
                    f"task {task_cid} completion is {status!r}, not prepared",
                    evidence={
                        "task_cid": task_cid,
                        "completion_status": status,
                        "reason": "preparation_not_pending",
                    },
                )
            return None
        body_raw = _row_get(mapping, "body_json", "2", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError as exc:
            raise DatabaseCoordinationStaleFenceError(
                "prepared completion body is not valid JSON"
            ) from exc
        if not isinstance(body, Mapping):
            raise DatabaseCoordinationStaleFenceError(
                "prepared completion body is not a mapping"
            )
        prepared = self._validate_preparation_mapping(body, task_cid=task_cid)
        identity = self._task_claim_identity(prepared)
        self._task_completion_for_identity_unlocked(
            connection,
            identity=identity,
            required=True,
            expected_statuses=tuple(sorted(allowed_statuses)),
        )
        return {
            **prepared,
            "status": status,
        }

    def _completion_authority_state_unlocked(
        self,
        connection: Any,
        *,
        prepared: Mapping[str, Any],
        now: int,
    ) -> tuple[dict[str, Any], FencedLease, LeaseState, AttemptStatus]:
        """Expire, then validate, the exact authority bound to a barrier.

        Prepared-completion recovery is itself an expiry sweep.  Callers must
        not rely on another coordinator mutation having happened after the
        wall-clock deadline: this helper transitions every overdue projection
        and validates the resulting claim/lease/attempt tuple in the caller's
        transaction.
        """

        identity = self._task_claim_identity(prepared)
        task_cid = str(identity["task_cid"])
        scope_key = exclusive_scope_key(
            lease_kind=LeaseKind.TASK,
            scope=task_cid,
            task_cid=task_cid,
        )
        self._expire_scope(connection, scope_key, now)
        claim_row = connection.execute(
            "SELECT state FROM task_claims WHERE claim_id = ?",
            [str(identity["claim_id"])],
        ).fetchone()
        lease_row = connection.execute(
            "SELECT state FROM fenced_leases WHERE lease_id = ?",
            [str(identity["lease_id"])],
        ).fetchone()
        attempt_row = connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            [str(identity["attempt_id"])],
        ).fetchone()
        if claim_row is None or lease_row is None or attempt_row is None:
            raise DatabaseCoordinationStaleFenceError(
                "completion barrier has incomplete task authority"
            )
        claim_state = LeaseState(
            str(_row_get(_row_mapping(claim_row), "state", "0", default=""))
        )
        lease_state = LeaseState(
            str(_row_get(_row_mapping(lease_row), "state", "0", default=""))
        )
        attempt_status = AttemptStatus(
            str(_row_get(_row_mapping(attempt_row), "status", "0", default=""))
        )
        if claim_state is not lease_state:
            raise DatabaseCoordinationStaleFenceError(
                "completion barrier claim and lease states disagree"
            )
        expected_attempt_by_state = {
            LeaseState.ACCEPTED: AttemptStatus.RUNNING,
            LeaseState.EXPIRED: AttemptStatus.EXPIRED,
            LeaseState.RELEASED: AttemptStatus.SUCCEEDED,
            LeaseState.COMPLETED: AttemptStatus.SUCCEEDED,
        }
        expected_attempt = expected_attempt_by_state.get(lease_state)
        if expected_attempt is None or attempt_status is not expected_attempt:
            raise DatabaseCoordinationStaleFenceError(
                "completion barrier authority has an invalid terminal-state pairing"
            )
        lease = self._protect_task_claim_unlocked(
            connection,
            identity=identity,
            now=now,
            expected_attempt_status=attempt_status,
            allow_logically_completed=True,
            record_event=False,
            expected_lease_state=lease_state,
        )
        return identity, lease, lease_state, attempt_status

    @staticmethod
    def _control_task_projection(
        receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        raw = _bounded_mapping(receipt, name="control_completion_receipt")
        nested = isinstance(raw.get("task"), Mapping)
        task = dict(raw["task"]) if nested else dict(raw)
        if not task:
            raise DatabaseCoordinationStaleFenceError(
                "control completion receipt has no task projection"
            )
        return raw, task, nested

    def _validate_control_completion_receipt(
        self,
        *,
        prepared: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw, task, nested = self._control_task_projection(receipt)
        task_cid = str(prepared["task_cid"])
        expected_revision = int(prepared["control_expected_revision"])
        resulting_revision = expected_revision + 1
        task_revision = _positive_int(
            int(task.get("revision") or 0),
            "control_task_revision",
        )
        task_status = _text(task.get("status"), "control_task_status").lower()
        if str(task.get("task_cid") or "") != task_cid:
            raise DatabaseCoordinationStaleFenceError(
                "control completion receipt task does not match preparation"
            )
        if task_status not in {"completed", "complete", "done"}:
            raise DatabaseCoordinationNotReadyError(
                f"control task {task_cid} is not successfully completed",
                evidence={
                    "task_cid": task_cid,
                    "control_status": task_status,
                    "reason": "control_completion_missing",
                },
            )
        if task_revision != resulting_revision:
            raise DatabaseCoordinationStaleFenceError(
                "control completion revision does not match prepared CAS revision"
            )
        if nested:
            if raw.get("changed") is not True:
                raise DatabaseCoordinationStaleFenceError(
                    "fresh control CAS receipt did not record a change"
                )
            if int(raw.get("revision") or 0) != task_revision:
                raise DatabaseCoordinationStaleFenceError(
                    "control CAS result revision disagrees with its task"
                )
            if str(raw.get("previous_status") or "") != str(
                prepared["control_expected_status"]
            ):
                raise DatabaseCoordinationStaleFenceError(
                    "control CAS prior status does not match preparation"
                )
            if not str(raw.get("receipt_cid") or "").strip():
                raise DatabaseCoordinationStaleFenceError(
                    "fresh control CAS result has no completion receipt CID"
                )
        task_body = task.get("body")
        if not isinstance(task_body, Mapping):
            raise DatabaseCoordinationStaleFenceError(
                "control task projection has no durable body"
            )
        completion_receipt = task_body.get("completion_receipt")
        if not isinstance(completion_receipt, Mapping):
            raise DatabaseCoordinationStaleFenceError(
                "control task has no persisted completion receipt"
            )
        binding = completion_receipt.get("coordination_preparation")
        if not isinstance(binding, Mapping):
            raise DatabaseCoordinationStaleFenceError(
                "control completion receipt has no coordination preparation"
            )
        exact_binding_fields = (
            "task_cid",
            "claim_id",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "control_expected_revision",
            "control_expected_status",
            "evidence_digest",
            "preparation_digest",
        )
        mismatches = [
            name
            for name in exact_binding_fields
            if binding.get(name) != prepared.get(name)
        ]
        if mismatches:
            raise DatabaseCoordinationStaleFenceError(
                "control completion receipt is not bound to its preparation: "
                + ", ".join(mismatches)
            )
        return {
            "task_cid": task_cid,
            "status": task_status,
            "revision": task_revision,
            "receipt_cid": str(raw.get("receipt_cid") or ""),
            "receipt_digest": _sha256_hex(
                _canonical_json(raw).encode("utf-8")
            ),
        }

    @staticmethod
    def _requires_cross_store_fence_guard(prepared: Mapping[str, Any]) -> bool:
        body = prepared.get("body")
        if not isinstance(body, Mapping):
            return False
        value = body.get(CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD)
        if value is None:
            return False
        if type(value) is not bool:
            raise DatabaseCoordinationStaleFenceError(
                "cross-store fence guard requirement must be a boolean"
            )
        return value

    def _required_cross_store_fence_guard_unlocked(
        self,
        connection: Any,
        *,
        prepared: Mapping[str, Any],
        control_summary: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return the matching durable guard or fail a guarded preparation."""

        if not self._requires_cross_store_fence_guard(prepared):
            return None
        rows = connection.execute(
            """
            SELECT event_id, body_json
            FROM lease_events
            WHERE lease_id = ? AND scope_key = ? AND event_type = ?
              AND fencing_token = ? AND fence_epoch = ?
            ORDER BY event_id
            """,
            [
                str(prepared["lease_id"]),
                exclusive_scope_key(
                    lease_kind=LeaseKind.TASK,
                    scope=str(prepared["task_cid"]),
                    task_cid=str(prepared["task_cid"]),
                ),
                CROSS_STORE_FENCE_GUARD_EVENT,
                int(prepared["fencing_token"]),
                int(prepared["fence_epoch"]),
            ],
        ).fetchall()
        expected = {
            "schema": CROSS_STORE_FENCE_GUARD_SCHEMA,
            "preparation_digest": str(prepared["preparation_digest"]),
            "task_cid": str(prepared["task_cid"]),
            "claim_id": str(prepared["claim_id"]),
            "attempt_id": str(prepared["attempt_id"]),
            "attempt_number": int(prepared["attempt_number"]),
            "lease_id": str(prepared["lease_id"]),
            "owner_session_id": str(prepared["owner_session_id"]),
            "fencing_token": int(prepared["fencing_token"]),
            "fence_epoch": int(prepared["fence_epoch"]),
            "control_result_digest": str(control_summary["receipt_digest"]),
        }
        for row in rows:
            mapping = _row_mapping(row)
            event_id = str(_row_get(mapping, "event_id", "0", default=""))
            body = _decode_coordination_body(
                _row_get(mapping, "body_json", "1", default="{}"),
                table="lease_events",
                identity=event_id,
            )
            if any(body.get(name) != value for name, value in expected.items()):
                continue
            writer_fields = (
                "writer_claim_id",
                "writer_lease_id",
                "writer_resource_kind",
                "writer_resource_id",
            )
            if (
                any(not str(body.get(name) or "") for name in writer_fields)
                or body.get("writer_owner_session_id")
                != prepared["owner_session_id"]
                or body.get("writer_task_cid") != prepared["task_cid"]
                or body.get("writer_mode") != LeaseMode.EXCLUSIVE.value
                or int(body.get("writer_fencing_token") or 0) < 1
                or int(body.get("writer_fence_epoch") or 0) < 1
            ):
                continue
            return {
                "schema": CROSS_STORE_FENCE_GUARD_SCHEMA,
                "event_id": event_id,
                "guard_digest": _sha256_hex(
                    _canonical_json(body).encode("utf-8")
                ),
                "preparation_digest": expected["preparation_digest"],
                "control_result_digest": expected["control_result_digest"],
                "writer_claim_id": str(body["writer_claim_id"]),
                "writer_lease_id": str(body["writer_lease_id"]),
                "writer_fencing_token": int(body["writer_fencing_token"]),
                "writer_fence_epoch": int(body["writer_fence_epoch"]),
            }
        raise DatabaseCoordinationNotReadyError(
            "guarded control completion has no matching cross-store fence receipt",
            evidence={
                "task_cid": str(prepared["task_cid"]),
                "claim_id": str(prepared["claim_id"]),
                "preparation_digest": str(prepared["preparation_digest"]),
                "control_result_digest": str(control_summary["receipt_digest"]),
                "reason": "cross_store_fence_guard_missing",
            },
        )

    def _validate_control_incomplete_observation(
        self,
        *,
        prepared: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        _raw, task, _nested = self._control_task_projection(observation)
        task_cid = str(prepared["task_cid"])
        task_status = _text(task.get("status"), "control_task_status").lower()
        task_revision = _positive_int(
            int(task.get("revision") or 0),
            "control_task_revision",
        )
        if str(task.get("task_cid") or "") != task_cid:
            raise DatabaseCoordinationStaleFenceError(
                "control task observation does not match preparation"
            )
        if task_status != str(prepared["control_expected_status"]):
            raise DatabaseCoordinationStaleFenceError(
                "control task status changed after preparation"
            )
        if task_revision != int(prepared["control_expected_revision"]):
            raise DatabaseCoordinationStaleFenceError(
                "control task revision changed after preparation"
            )
        task_body = task.get("body")
        if isinstance(task_body, Mapping):
            completion_receipt = task_body.get("completion_receipt")
            if isinstance(completion_receipt, Mapping):
                binding = completion_receipt.get("coordination_preparation")
                if isinstance(binding, Mapping) and binding.get(
                    "preparation_digest"
                ) == prepared.get("preparation_digest"):
                    raise DatabaseCoordinationStaleFenceError(
                        "control task already contains this completion preparation"
                    )
        return {
            "task_cid": task_cid,
            "status": task_status,
            "revision": task_revision,
        }

    def protect_task_claim(
        self,
        claim: TaskClaim | Mapping[str, Any],
        *,
        expected_task_cid: str | None = None,
        expected_attempt_id: str | None = None,
        expected_owner_session_id: str | None = None,
        expected_fencing_token: int | None = None,
        expected_fence_epoch: int | None = None,
        expected_attempt_status: AttemptStatus | str = AttemptStatus.RUNNING,
        allow_logically_completed: bool = False,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Protect a durable task write with its exact live authority tuple.

        Unlike :meth:`protect_write`, this validates the task claim, fenced
        lease, and task-attempt projections together.  The caller-provided
        identity must name the same task, claim, attempt, owner, lease, token,
        and epoch in all three rows.  Expired, released, completed, or taken-
        over attempts fail closed.
        """

        identity = self._task_claim_identity(claim)
        expected_values = {
            "task_cid": expected_task_cid,
            "attempt_id": expected_attempt_id,
            "owner_session_id": expected_owner_session_id,
            "fencing_token": expected_fencing_token,
            "fence_epoch": expected_fence_epoch,
        }
        for name, expected in expected_values.items():
            if expected is None:
                continue
            actual = identity[name]
            normalized = (
                _positive_int(int(expected), name)
                if name in {"fencing_token", "fence_epoch"}
                else _text(expected, name)
            )
            if actual != normalized:
                raise DatabaseCoordinationStaleFenceError(
                    f"caller {name} does not match the task claim"
                )
        status = (
            expected_attempt_status
            if isinstance(expected_attempt_status, AttemptStatus)
            else AttemptStatus(str(expected_attempt_status).strip().lower())
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                lease = self._protect_task_claim_unlocked(
                    connection,
                    identity=identity,
                    now=now,
                    expected_attempt_status=status,
                    allow_logically_completed=bool(allow_logically_completed),
                    record_event=True,
                )
                self._commit_if_idle(connection)
                return lease
            except Exception:
                self._rollback_if_open(connection)
                raise

    def execute_with_task_and_resource_fences(
        self,
        claim: TaskClaim | Mapping[str, Any],
        writer_claim: ResourceClaim | Mapping[str, Any],
        callback: Callable[[], Any],
        *,
        allow_logically_completed: bool = False,
    ) -> Any:
        """Execute an external CAS while exact coordinator fences stay stable.

        The task claim/attempt/lease tuple and the exclusive resource
        claim/lease tuple are validated before the callback and again
        immediately after it.  This coordinator's re-entrant lock, its
        process-shared DuckDB file lock, and one write transaction prevent a
        coordinator mutation from interleaving with the callback.  Callback
        re-entry into this coordinator is rejected.  A callback exception or
        failed post-check rolls the coordinator guard transaction back.

        This is deliberately *not* a distributed transaction with the store
        mutated by ``callback``.  If that external CAS commits and this
        process crashes, a fence expires, or the post-check fails, its effect
        cannot be rolled back here.  The external CAS therefore must be
        idempotent, bind both fence identities, and support receipt-based
        reconciliation before coordinator completion is accepted.

        A preparation whose ``body`` sets
        ``requires_cross_store_fence_guard`` to true makes this durable: the
        callback must return the same mapping receipt later supplied to
        :meth:`complete_task_claim` or recovery.  Successful post-validation
        records its digest with both fences; promotion and recovery then fail
        closed when that guard receipt is absent.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        task_identity = self._task_claim_identity(claim)
        resource_identity = self._resource_claim_identity(writer_claim)
        if resource_identity["task_cid"] != task_identity["task_cid"]:
            raise DatabaseCoordinationStaleFenceError(
                "writer resource claim is bound to a different task"
            )
        if resource_identity["owner_session_id"] != task_identity["owner_session_id"]:
            raise DatabaseCoordinationStaleFenceError(
                "writer resource claim is bound to a different owner"
            )

        with self._lock:
            connection = self._require()
            self._begin(connection)
            if not getattr(connection, "in_transaction", False):
                raise DatabaseCoordinationError(
                    "could not start fenced cross-store callback transaction"
                )
            self._fenced_callback_reentry_detected = False
            try:
                before_ms = self._now_ms()
                task_lease = self._protect_task_claim_unlocked(
                    connection,
                    identity=task_identity,
                    now=before_ms,
                    expected_attempt_status=AttemptStatus.RUNNING,
                    allow_logically_completed=bool(allow_logically_completed),
                    record_event=True,
                )
                writer_lease = self._protect_resource_claim_unlocked(
                    connection,
                    identity=resource_identity,
                    now=before_ms,
                    record_event=True,
                )
                prepared = self._prepared_completion_unlocked(
                    connection,
                    str(task_identity["task_cid"]),
                    required=False,
                )
                guard_required = bool(
                    prepared is not None
                    and self._requires_cross_store_fence_guard(prepared)
                )

                self._fenced_callback_active = True
                try:
                    result = callback()
                finally:
                    self._fenced_callback_active = False
                if self._fenced_callback_reentry_detected:
                    raise DatabaseCoordinationConflictError(
                        "fenced callback attempted to re-enter DatabaseCoordinator"
                    )
                callback_projection: dict[str, Any] | None = None
                if isinstance(result, Mapping):
                    callback_projection = _bounded_mapping(
                        result,
                        name="cross_store_callback_result",
                    )
                else:
                    to_dict = getattr(result, "to_dict", None)
                    if callable(to_dict):
                        projected = to_dict()
                        if isinstance(projected, Mapping):
                            callback_projection = _bounded_mapping(
                                projected,
                                name="cross_store_callback_result",
                            )
                if guard_required and callback_projection is None:
                    raise DatabaseCoordinationStaleFenceError(
                        "guarded cross-store callback returned no mapping receipt"
                    )

                after_ms = self._now_ms()
                if after_ms < before_ms:
                    raise DatabaseCoordinationStaleFenceError(
                        "coordination clock moved backwards during fenced callback"
                    )
                self._protect_task_claim_unlocked(
                    connection,
                    identity=task_identity,
                    now=after_ms,
                    expected_attempt_status=AttemptStatus.RUNNING,
                    allow_logically_completed=bool(allow_logically_completed),
                    record_event=False,
                )
                self._protect_resource_claim_unlocked(
                    connection,
                    identity=resource_identity,
                    now=after_ms,
                    record_event=False,
                )
                if guard_required:
                    assert prepared is not None
                    assert callback_projection is not None
                    self._record_event(
                        connection,
                        lease_id=task_lease.lease_id,
                        scope_key=task_lease.scope_key,
                        event_type=CROSS_STORE_FENCE_GUARD_EVENT,
                        fencing_token=int(task_identity["fencing_token"]),
                        fence_epoch=int(task_identity["fence_epoch"]),
                        observed_at_ms=after_ms,
                        body={
                            "schema": CROSS_STORE_FENCE_GUARD_SCHEMA,
                            "preparation_digest": str(
                                prepared["preparation_digest"]
                            ),
                            "task_cid": str(task_identity["task_cid"]),
                            "claim_id": str(task_identity["claim_id"]),
                            "attempt_id": str(task_identity["attempt_id"]),
                            "attempt_number": int(
                                task_identity["attempt_number"]
                            ),
                            "lease_id": str(task_identity["lease_id"]),
                            "owner_session_id": str(
                                task_identity["owner_session_id"]
                            ),
                            "fencing_token": int(
                                task_identity["fencing_token"]
                            ),
                            "fence_epoch": int(task_identity["fence_epoch"]),
                            "writer_claim_id": str(
                                resource_identity["claim_id"]
                            ),
                            "writer_lease_id": writer_lease.lease_id,
                            "writer_resource_kind": str(
                                resource_identity["resource_kind"]
                            ),
                            "writer_resource_id": str(
                                resource_identity["resource_id"]
                            ),
                            "writer_owner_session_id": str(
                                resource_identity["owner_session_id"]
                            ),
                            "writer_task_cid": str(
                                resource_identity["task_cid"]
                            ),
                            "writer_mode": LeaseMode.EXCLUSIVE.value,
                            "writer_fencing_token": int(
                                resource_identity["fencing_token"]
                            ),
                            "writer_fence_epoch": int(
                                resource_identity["fence_epoch"]
                            ),
                            "control_result_digest": _sha256_hex(
                                _canonical_json(callback_projection).encode("utf-8")
                            ),
                        },
                    )
                connection.commit()
                return result
            except BaseException:
                self._rollback_if_open(connection)
                raise
            finally:
                self._fenced_callback_active = False
                self._fenced_callback_reentry_detected = False

    def expire_task_claim(
        self,
        claim: TaskClaim | Mapping[str, Any],
        *,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Persist wall-clock expiry for one exact task authority tuple.

        The transition is idempotent only while this claim remains the latest
        fence for its task.  A still-live claim fails without mutation, and a
        claim superseded by a later fence fails closed.
        """

        identity = self._task_claim_identity(claim)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        task_cid = str(identity["task_cid"])
        scope_key = exclusive_scope_key(
            lease_kind=LeaseKind.TASK,
            scope=task_cid,
            task_cid=task_cid,
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._expire_scope(connection, scope_key, now)
                completion_exists = (
                    connection.execute(
                        "SELECT 1 FROM task_completions WHERE task_cid = ?",
                        [task_cid],
                    ).fetchone()
                    is not None
                )
                lease = self._protect_task_claim_unlocked(
                    connection,
                    identity=identity,
                    now=now,
                    expected_attempt_status=AttemptStatus.EXPIRED,
                    allow_logically_completed=completion_exists,
                    record_event=False,
                    expected_lease_state=LeaseState.EXPIRED,
                )
                self._commit_if_idle(connection)
                return lease
            except Exception:
                self._rollback_if_open(connection)
                raise

    def prepare_task_completion(
        self,
        claim: TaskClaim | Mapping[str, Any],
        *,
        control_expected_revision: int,
        evidence_digest: str,
        control_expected_status: str = "in_progress",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Install a PREPARED barrier before the control-store completion CAS.

        PREPARED makes the task itself unclaimable and clears its ready bit but
        does not satisfy dependents.  The exact claim remains live so the
        daemon can perform the separately durable control-store CAS.
        """

        identity = self._task_claim_identity(claim)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        supplied_body = _bounded_mapping(body, name="body")
        expected_revision = _positive_int(
            int(control_expected_revision),
            "control_expected_revision",
        )
        expected_status = _text(
            control_expected_status,
            "control_expected_status",
        ).lower()
        evidence = _text(evidence_digest, "evidence_digest")
        task_cid = str(identity["task_cid"])
        claim_id = str(identity["claim_id"])
        attempt_id = str(identity["attempt_id"])
        lease_id = str(identity["lease_id"])
        token = int(identity["fencing_token"])
        epoch = int(identity["fence_epoch"])
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = self._prepared_completion_unlocked(
                    connection,
                    task_cid,
                    required=False,
                )
                if existing is not None:
                    self._protect_task_claim_unlocked(
                        connection,
                        identity=identity,
                        now=now,
                        expected_attempt_status=AttemptStatus.RUNNING,
                        allow_logically_completed=True,
                        record_event=False,
                    )
                    expected_fields = {
                        "control_expected_revision": expected_revision,
                        "control_expected_status": expected_status,
                        "evidence_digest": evidence,
                    }
                    mismatches = [
                        name
                        for name, expected in expected_fields.items()
                        if existing.get(name) != expected
                    ]
                    if mismatches:
                        raise DatabaseCoordinationStaleFenceError(
                            "prepared completion replay changed its control binding: "
                            + ", ".join(mismatches)
                        )
                    self._commit_if_idle(connection)
                    return {
                        **existing,
                        "replayed": True,
                    }

                lease = self._protect_task_claim_unlocked(
                    connection,
                    identity=identity,
                    now=now,
                    expected_attempt_status=AttemptStatus.RUNNING,
                    allow_logically_completed=False,
                    record_event=False,
                )
                preparation_body: dict[str, Any] = {
                    "schema": TASK_COMPLETION_PREPARATION_SCHEMA,
                    "task_cid": task_cid,
                    "attempt_id": attempt_id,
                    "attempt_number": int(identity["attempt_number"]),
                    "claim_id": claim_id,
                    "lease_id": lease_id,
                    "owner_session_id": str(identity["owner_session_id"]),
                    "fencing_token": token,
                    "fence_epoch": epoch,
                    "control_expected_revision": expected_revision,
                    "control_expected_status": expected_status,
                    "evidence_digest": evidence,
                    "prepared_at_ms": now,
                    "body": supplied_body,
                }
                preparation_body["preparation_digest"] = self._preparation_digest(
                    preparation_body
                )
                preparation_body = _bounded_mapping(
                    preparation_body,
                    name="task_completion_preparation",
                )
                connection.execute(
                    """
                    INSERT INTO task_completions(
                        task_cid, completed_at_ms, status, body_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        task_cid,
                        now,
                        PREPARED_COMPLETION_STATUS,
                        _canonical_json(preparation_body),
                    ],
                )
                connection.execute(
                    "UPDATE coordination_tasks SET ready = FALSE WHERE task_cid = ?",
                    [task_cid],
                )
                self._record_event(
                    connection,
                    lease_id=lease.lease_id,
                    scope_key=lease.scope_key,
                    event_type="task_completion_prepared",
                    fencing_token=token,
                    fence_epoch=epoch,
                    observed_at_ms=now,
                    body={
                        "task_cid": task_cid,
                        "claim_id": claim_id,
                        "attempt_id": attempt_id,
                    },
                )
                self._commit_if_idle(connection)
                return {
                    **preparation_body,
                    "status": PREPARED_COMPLETION_STATUS,
                    "replayed": False,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def complete_task_claim(
        self,
        claim: TaskClaim | Mapping[str, Any],
        *,
        control_completion_receipt: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Promote the exact PREPARED barrier after the control CAS commits."""

        identity = self._task_claim_identity(claim)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        task_cid = str(identity["task_cid"])
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                completion = self._task_completion_for_identity_unlocked(
                    connection,
                    identity=identity,
                    required=True,
                    expected_statuses=(
                        PREPARED_COMPLETION_STATUS,
                        AttemptStatus.SUCCEEDED.value,
                    ),
                )
                prepared = self._validate_preparation_mapping(
                    completion["body"],
                    task_cid=task_cid,
                )
                control_summary = self._validate_control_completion_receipt(
                    prepared=prepared,
                    receipt=control_completion_receipt,
                )
                guard_summary = self._required_cross_store_fence_guard_unlocked(
                    connection,
                    prepared=prepared,
                    control_summary=control_summary,
                )
                self._protect_task_claim_unlocked(
                    connection,
                    identity=identity,
                    now=now,
                    expected_attempt_status=AttemptStatus.RUNNING,
                    allow_logically_completed=True,
                    record_event=False,
                )
                if completion["status"] == AttemptStatus.SUCCEEDED.value:
                    self._commit_if_idle(connection)
                    return {
                        "task_cid": task_cid,
                        "claim_id": str(identity["claim_id"]),
                        "attempt_id": str(identity["attempt_id"]),
                        "lease_id": str(identity["lease_id"]),
                        "fencing_token": int(identity["fencing_token"]),
                        "fence_epoch": int(identity["fence_epoch"]),
                        "completed_at_ms": int(completion["completed_at_ms"]),
                        "status": AttemptStatus.SUCCEEDED.value,
                        "replayed": True,
                    }
                promoted_payload = {
                    **prepared,
                    "control_completion": control_summary,
                }
                if guard_summary is not None:
                    promoted_payload["cross_store_guard"] = guard_summary
                promoted_body = _bounded_mapping(
                    promoted_payload,
                    name="task_completion_body",
                )
                connection.execute(
                    """
                    UPDATE task_completions
                    SET completed_at_ms = ?, status = ?, body_json = ?
                    WHERE task_cid = ? AND status = ?
                    """,
                    [
                        now,
                        AttemptStatus.SUCCEEDED.value,
                        _canonical_json(promoted_body),
                        task_cid,
                        PREPARED_COMPLETION_STATUS,
                    ],
                )
                promoted = connection.execute(
                    "SELECT status FROM task_completions WHERE task_cid = ?",
                    [task_cid],
                ).fetchone()
                if promoted is None or str(
                    _row_get(_row_mapping(promoted), "status", "0", default="")
                ) != AttemptStatus.SUCCEEDED.value:
                    raise DatabaseCoordinationStaleFenceError(
                        "prepared completion promotion lost its exact barrier"
                    )
                lease = self._lease_from_row(
                    connection.execute(
                        "SELECT * FROM fenced_leases WHERE lease_id = ?",
                        [str(identity["lease_id"])],
                    ).fetchone()
                )
                self._record_event(
                    connection,
                    lease_id=lease.lease_id,
                    scope_key=lease.scope_key,
                    event_type="task_completion_promoted",
                    fencing_token=int(identity["fencing_token"]),
                    fence_epoch=int(identity["fence_epoch"]),
                    observed_at_ms=now,
                    body={
                        "task_cid": task_cid,
                        "claim_id": str(identity["claim_id"]),
                        "attempt_id": str(identity["attempt_id"]),
                        "control_revision": int(control_summary["revision"]),
                    },
                )
                self._commit_if_idle(connection)
                return {
                    "task_cid": task_cid,
                    "claim_id": str(identity["claim_id"]),
                    "attempt_id": str(identity["attempt_id"]),
                    "lease_id": str(identity["lease_id"]),
                    "fencing_token": int(identity["fencing_token"]),
                    "fence_epoch": int(identity["fence_epoch"]),
                    "completed_at_ms": now,
                    "status": AttemptStatus.SUCCEEDED.value,
                    "replayed": False,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_prepared_task_completion(
        self,
        task_cid: str,
    ) -> dict[str, Any] | None:
        """Return one validated preparation, including a promoted barrier."""

        cid = _text(task_cid, "task_cid")
        with self._lock:
            connection = self._require()
            return self._prepared_completion_unlocked(
                connection,
                cid,
                required=False,
                include_promoted=True,
            )

    def list_prepared_task_completions(
        self,
        *,
        limit: int = 100,
        now_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return validated PREPARED barriers after an atomic expiry sweep."""

        bound = max(1, min(int(limit), MAX_PREPARED_COMPLETION_QUERY))
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                rows = connection.execute(
                    f"""
                    SELECT completion.task_cid
                    FROM task_completions AS completion
                    JOIN task_claims AS claim
                      ON claim.task_cid = completion.task_cid
                     AND claim.claim_id = json_extract_string(
                         completion.body_json, '$.claim_id'
                     )
                    WHERE completion.status = ?
                      AND claim.state IN (?, ?)
                    ORDER BY
                        CASE
                            WHEN claim.state = ? OR claim.expires_at_ms <= ?
                            THEN 0 ELSE 1
                        END,
                        completion.completed_at_ms,
                        completion.task_cid
                    LIMIT {bound}
                    """,
                    [
                        PREPARED_COMPLETION_STATUS,
                        LeaseState.ACCEPTED.value,
                        LeaseState.EXPIRED.value,
                        LeaseState.EXPIRED.value,
                        now,
                    ],
                ).fetchall()
                prepared: list[dict[str, Any]] = []
                for row in rows:
                    task_cid = str(
                        _row_get(_row_mapping(row), "task_cid", "0", default="")
                    )
                    item = self._prepared_completion_unlocked(
                        connection,
                        task_cid,
                        required=True,
                    )
                    assert item is not None
                    _identity, _lease, lease_state, attempt_status = (
                        self._completion_authority_state_unlocked(
                            connection,
                            prepared=item,
                            now=now,
                        )
                    )
                    if lease_state not in {
                        LeaseState.ACCEPTED,
                        LeaseState.EXPIRED,
                    }:
                        raise DatabaseCoordinationStaleFenceError(
                            "pending preparation has terminal task authority"
                        )
                    prepared.append(
                        {
                            **item,
                            "claim_state": lease_state.value,
                            "lease_state": lease_state.value,
                            "attempt_status": attempt_status.value,
                        }
                    )
                self._commit_if_idle(connection)
                return prepared
            except Exception:
                self._rollback_if_open(connection)
                raise

    def list_unsettled_task_completions(
        self,
        *,
        limit: int = 100,
        now_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate exact PREPARED and promoted barriers needing settlement.

        Enumeration is a bounded coordination mutation: any exact bound lease
        whose wall-clock deadline has passed is atomically projected to
        EXPIRED before it is returned.  Already released/completed barriers
        are validated but omitted.
        """

        bound = max(1, min(int(limit), MAX_PREPARED_COMPLETION_QUERY))
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                rows = connection.execute(
                    f"""
                    SELECT completion.task_cid
                    FROM task_completions AS completion
                    JOIN task_claims AS claim
                      ON claim.task_cid = completion.task_cid
                     AND claim.claim_id = json_extract_string(
                         completion.body_json, '$.claim_id'
                     )
                    WHERE completion.status IN (?, ?)
                      AND claim.state IN (?, ?)
                    ORDER BY
                        CASE
                            WHEN completion.status = ? THEN 0
                            WHEN claim.state = ? OR claim.expires_at_ms <= ?
                            THEN 1 ELSE 2
                        END,
                        completion.completed_at_ms,
                        completion.task_cid
                    LIMIT {bound}
                    """,
                    [
                        PREPARED_COMPLETION_STATUS,
                        AttemptStatus.SUCCEEDED.value,
                        LeaseState.ACCEPTED.value,
                        LeaseState.EXPIRED.value,
                        AttemptStatus.SUCCEEDED.value,
                        LeaseState.EXPIRED.value,
                        now,
                    ],
                ).fetchall()
                unsettled: list[dict[str, Any]] = []
                for row in rows:
                    task_cid = str(
                        _row_get(_row_mapping(row), "task_cid", "0", default="")
                    )
                    item = self._prepared_completion_unlocked(
                        connection,
                        task_cid,
                        required=True,
                        include_promoted=True,
                    )
                    assert item is not None
                    _identity, _lease, lease_state, attempt_status = (
                        self._completion_authority_state_unlocked(
                            connection,
                            prepared=item,
                            now=now,
                        )
                    )
                    if item["status"] == PREPARED_COMPLETION_STATUS and (
                        lease_state
                        not in {LeaseState.ACCEPTED, LeaseState.EXPIRED}
                    ):
                        raise DatabaseCoordinationStaleFenceError(
                            "pending preparation has terminal task authority"
                        )
                    if lease_state in {
                        LeaseState.RELEASED,
                        LeaseState.COMPLETED,
                    }:
                        continue
                    unsettled.append(
                        {
                            **item,
                            "claim_state": lease_state.value,
                            "lease_state": lease_state.value,
                            "attempt_status": attempt_status.value,
                        }
                    )
                self._commit_if_idle(connection)
                return unsettled
            except Exception:
                self._rollback_if_open(connection)
                raise

    def recover_prepared_task_completion(
        self,
        task_cid: str,
        *,
        control_completion_receipt: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Promote and settle an expired PREPARED claim from control truth."""

        cid = _text(task_cid, "task_cid")
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                prepared = self._prepared_completion_unlocked(
                    connection,
                    cid,
                    required=True,
                )
                assert prepared is not None
                identity, lease, lease_state, _attempt_status = (
                    self._completion_authority_state_unlocked(
                        connection,
                        prepared=prepared,
                        now=now,
                    )
                )
                if lease_state is not LeaseState.EXPIRED:
                    raise DatabaseCoordinationExpiredError(
                        f"prepared task claim {identity['claim_id']} is still live"
                    )
                self._task_completion_for_identity_unlocked(
                    connection,
                    identity=identity,
                    required=True,
                    expected_statuses=(PREPARED_COMPLETION_STATUS,),
                )
                control_summary = self._validate_control_completion_receipt(
                    prepared=prepared,
                    receipt=control_completion_receipt,
                )
                guard_summary = self._required_cross_store_fence_guard_unlocked(
                    connection,
                    prepared=prepared,
                    control_summary=control_summary,
                )
                promoted_preparation = dict(prepared)
                promoted_preparation.pop("status", None)
                promoted_payload = {
                    **promoted_preparation,
                    "control_completion": control_summary,
                }
                if guard_summary is not None:
                    promoted_payload["cross_store_guard"] = guard_summary
                promoted_body = _bounded_mapping(
                    promoted_payload,
                    name="task_completion_body",
                )
                connection.execute(
                    """
                    UPDATE task_completions
                    SET completed_at_ms = ?, status = ?, body_json = ?
                    WHERE task_cid = ? AND status = ?
                    """,
                    [
                        now,
                        AttemptStatus.SUCCEEDED.value,
                        _canonical_json(promoted_body),
                        cid,
                        PREPARED_COMPLETION_STATUS,
                    ],
                )
                connection.execute(
                    "UPDATE coordination_tasks SET ready = FALSE WHERE task_cid = ?",
                    [cid],
                )
                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET state = ?, revision = revision + 1
                    WHERE lease_id = ? AND task_cid = ? AND claim_id = ?
                      AND attempt_id = ? AND owner_session_id = ?
                      AND fencing_token = ? AND fence_epoch = ? AND state = ?
                    """,
                    [
                        LeaseState.COMPLETED.value,
                        str(identity["lease_id"]),
                        cid,
                        str(identity["claim_id"]),
                        str(identity["attempt_id"]),
                        str(identity["owner_session_id"]),
                        int(identity["fencing_token"]),
                        int(identity["fence_epoch"]),
                        LeaseState.EXPIRED.value,
                    ],
                )
                connection.execute(
                    """
                    UPDATE task_claims
                    SET state = ?, released_at_ms = ?, revision = revision + 1
                    WHERE claim_id = ? AND task_cid = ? AND attempt_id = ?
                      AND lease_id = ? AND owner_session_id = ?
                      AND fencing_token = ? AND fence_epoch = ? AND state = ?
                    """,
                    [
                        LeaseState.COMPLETED.value,
                        now,
                        str(identity["claim_id"]),
                        cid,
                        str(identity["attempt_id"]),
                        str(identity["lease_id"]),
                        str(identity["owner_session_id"]),
                        int(identity["fencing_token"]),
                        int(identity["fence_epoch"]),
                        LeaseState.EXPIRED.value,
                    ],
                )
                connection.execute(
                    """
                    UPDATE task_attempts
                    SET status = ?, finished_at_ms = ?, revision = revision + 1
                    WHERE attempt_id = ? AND task_cid = ?
                      AND owner_session_id = ? AND fencing_token = ?
                      AND fence_epoch = ? AND status = ?
                    """,
                    [
                        AttemptStatus.SUCCEEDED.value,
                        now,
                        str(identity["attempt_id"]),
                        cid,
                        str(identity["owner_session_id"]),
                        int(identity["fencing_token"]),
                        int(identity["fence_epoch"]),
                        AttemptStatus.EXPIRED.value,
                    ],
                )
                final_lease = self._lease_from_row(
                    connection.execute(
                        "SELECT * FROM fenced_leases WHERE lease_id = ?",
                        [str(identity["lease_id"])],
                    ).fetchone()
                )
                final_claim = connection.execute(
                    "SELECT state FROM task_claims WHERE claim_id = ?",
                    [str(identity["claim_id"])],
                ).fetchone()
                final_attempt = connection.execute(
                    "SELECT status FROM task_attempts WHERE attempt_id = ?",
                    [str(identity["attempt_id"])],
                ).fetchone()
                if (
                    final_lease.state is not LeaseState.COMPLETED
                    or final_claim is None
                    or str(
                        _row_get(
                            _row_mapping(final_claim), "state", "0", default=""
                        )
                    )
                    != LeaseState.COMPLETED.value
                    or final_attempt is None
                    or str(
                        _row_get(
                            _row_mapping(final_attempt), "status", "0", default=""
                        )
                    )
                    != AttemptStatus.SUCCEEDED.value
                ):
                    raise DatabaseCoordinationStaleFenceError(
                        "prepared completion recovery lost its exact fence"
                    )
                self._record_event(
                    connection,
                    lease_id=lease.lease_id,
                    scope_key=lease.scope_key,
                    event_type="prepared_completion_recovered",
                    fencing_token=int(identity["fencing_token"]),
                    fence_epoch=int(identity["fence_epoch"]),
                    observed_at_ms=now,
                    body={
                        "task_cid": cid,
                        "claim_id": str(identity["claim_id"]),
                        "attempt_id": str(identity["attempt_id"]),
                        "control_revision": int(control_summary["revision"]),
                    },
                )
                self._commit_if_idle(connection)
                return {
                    "task_cid": cid,
                    "claim_id": str(identity["claim_id"]),
                    "attempt_id": str(identity["attempt_id"]),
                    "lease_id": str(identity["lease_id"]),
                    "status": AttemptStatus.SUCCEEDED.value,
                    "lease_state": LeaseState.COMPLETED.value,
                    "recovered": True,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def reconcile_promoted_task_completion(
        self,
        task_cid: str,
        *,
        control_completion_receipt: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Atomically settle a promoted barrier whether live or just expired.

        This closes the crash window after coordinator promotion and local
        execution-store COMPLETE but before ordinary settlement.  Current
        control truth is rebound to the persisted preparation, expiry is
        projected in this same transaction, and the exact authority becomes
        RELEASED (while live) or COMPLETED (after expiry).
        """

        cid = _text(task_cid, "task_cid")
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                promoted = self._prepared_completion_unlocked(
                    connection,
                    cid,
                    required=True,
                    include_promoted=True,
                )
                assert promoted is not None
                if promoted["status"] != AttemptStatus.SUCCEEDED.value:
                    raise DatabaseCoordinationNotReadyError(
                        f"task {cid} completion has not been promoted",
                        evidence={
                            "task_cid": cid,
                            "completion_status": str(promoted["status"]),
                            "reason": "promotion_missing",
                        },
                    )
                control_summary = self._validate_control_completion_receipt(
                    prepared=promoted,
                    receipt=control_completion_receipt,
                )
                self._required_cross_store_fence_guard_unlocked(
                    connection,
                    prepared=promoted,
                    control_summary=control_summary,
                )
                identity, lease, lease_state, _attempt_status = (
                    self._completion_authority_state_unlocked(
                        connection,
                        prepared=promoted,
                        now=now,
                    )
                )
                if lease_state in {
                    LeaseState.RELEASED,
                    LeaseState.COMPLETED,
                }:
                    self._commit_if_idle(connection)
                    return {
                        "task_cid": cid,
                        "claim_id": str(identity["claim_id"]),
                        "attempt_id": str(identity["attempt_id"]),
                        "lease_id": str(identity["lease_id"]),
                        "status": AttemptStatus.SUCCEEDED.value,
                        "lease_state": lease_state.value,
                        "replayed": True,
                    }

                target_state = (
                    LeaseState.RELEASED
                    if lease_state is LeaseState.ACCEPTED
                    else LeaseState.COMPLETED
                )
                if lease_state not in {
                    LeaseState.ACCEPTED,
                    LeaseState.EXPIRED,
                }:
                    raise DatabaseCoordinationStaleFenceError(
                        "promoted completion has unreconcilable task authority"
                    )
                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET state = ?, revision = revision + 1
                    WHERE lease_id = ? AND task_cid = ? AND claim_id = ?
                      AND attempt_id = ? AND owner_session_id = ?
                      AND fencing_token = ? AND fence_epoch = ? AND state = ?
                    """,
                    [
                        target_state.value,
                        str(identity["lease_id"]),
                        cid,
                        str(identity["claim_id"]),
                        str(identity["attempt_id"]),
                        str(identity["owner_session_id"]),
                        int(identity["fencing_token"]),
                        int(identity["fence_epoch"]),
                        lease_state.value,
                    ],
                )
                connection.execute(
                    """
                    UPDATE task_claims
                    SET state = ?, released_at_ms = ?, revision = revision + 1
                    WHERE claim_id = ? AND task_cid = ? AND attempt_id = ?
                      AND lease_id = ? AND owner_session_id = ?
                      AND fencing_token = ? AND fence_epoch = ? AND state = ?
                    """,
                    [
                        target_state.value,
                        now,
                        str(identity["claim_id"]),
                        cid,
                        str(identity["attempt_id"]),
                        str(identity["lease_id"]),
                        str(identity["owner_session_id"]),
                        int(identity["fencing_token"]),
                        int(identity["fence_epoch"]),
                        lease_state.value,
                    ],
                )
                connection.execute(
                    """
                    UPDATE task_attempts
                    SET status = ?, finished_at_ms = ?, revision = revision + 1
                    WHERE attempt_id = ? AND task_cid = ?
                      AND owner_session_id = ? AND fencing_token = ?
                      AND fence_epoch = ? AND status = ?
                    """,
                    [
                        AttemptStatus.SUCCEEDED.value,
                        now,
                        str(identity["attempt_id"]),
                        cid,
                        str(identity["owner_session_id"]),
                        int(identity["fencing_token"]),
                        int(identity["fence_epoch"]),
                        (
                            AttemptStatus.RUNNING.value
                            if lease_state is LeaseState.ACCEPTED
                            else AttemptStatus.EXPIRED.value
                        ),
                    ],
                )
                connection.execute(
                    "UPDATE coordination_tasks SET ready = FALSE WHERE task_cid = ?",
                    [cid],
                )
                final_lease = self._lease_from_row(
                    connection.execute(
                        "SELECT * FROM fenced_leases WHERE lease_id = ?",
                        [str(identity["lease_id"])],
                    ).fetchone()
                )
                final_claim = connection.execute(
                    "SELECT state FROM task_claims WHERE claim_id = ?",
                    [str(identity["claim_id"])],
                ).fetchone()
                final_attempt = connection.execute(
                    "SELECT status FROM task_attempts WHERE attempt_id = ?",
                    [str(identity["attempt_id"])],
                ).fetchone()
                if (
                    final_lease.state is not target_state
                    or final_claim is None
                    or str(
                        _row_get(
                            _row_mapping(final_claim), "state", "0", default=""
                        )
                    )
                    != target_state.value
                    or final_attempt is None
                    or str(
                        _row_get(
                            _row_mapping(final_attempt),
                            "status",
                            "0",
                            default="",
                        )
                    )
                    != AttemptStatus.SUCCEEDED.value
                ):
                    raise DatabaseCoordinationStaleFenceError(
                        "promoted completion reconciliation lost its exact fence"
                    )
                self._record_event(
                    connection,
                    lease_id=lease.lease_id,
                    scope_key=lease.scope_key,
                    event_type="promoted_completion_reconciled",
                    fencing_token=int(identity["fencing_token"]),
                    fence_epoch=int(identity["fence_epoch"]),
                    observed_at_ms=now,
                    body={
                        "task_cid": cid,
                        "claim_id": str(identity["claim_id"]),
                        "attempt_id": str(identity["attempt_id"]),
                        "control_revision": int(control_summary["revision"]),
                        "lease_state": target_state.value,
                    },
                )
                self._commit_if_idle(connection)
                return {
                    "task_cid": cid,
                    "claim_id": str(identity["claim_id"]),
                    "attempt_id": str(identity["attempt_id"]),
                    "lease_id": str(identity["lease_id"]),
                    "status": AttemptStatus.SUCCEEDED.value,
                    "lease_state": target_state.value,
                    "replayed": False,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def abort_prepared_task_completion(
        self,
        task_cid: str,
        *,
        control_task_observation: Mapping[str, Any],
        reason: str = "control_completion_absent",
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Abort an expired PREPARED barrier proven absent from control truth."""

        cid = _text(task_cid, "task_cid")
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        reason_text = _text(reason, "reason")[:256]
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                prepared = self._prepared_completion_unlocked(
                    connection,
                    cid,
                    required=True,
                )
                assert prepared is not None
                identity, lease, lease_state, _attempt_status = (
                    self._completion_authority_state_unlocked(
                        connection,
                        prepared=prepared,
                        now=now,
                    )
                )
                if lease_state is not LeaseState.EXPIRED:
                    raise DatabaseCoordinationExpiredError(
                        f"prepared task claim {identity['claim_id']} is still live"
                    )
                self._task_completion_for_identity_unlocked(
                    connection,
                    identity=identity,
                    required=True,
                    expected_statuses=(PREPARED_COMPLETION_STATUS,),
                )
                observation = self._validate_control_incomplete_observation(
                    prepared=prepared,
                    observation=control_task_observation,
                )
                connection.execute(
                    """
                    DELETE FROM task_completions
                    WHERE task_cid = ? AND status = ?
                    """,
                    [cid, PREPARED_COMPLETION_STATUS],
                )
                readiness = self._claimability_unlocked(connection, cid)
                ready = bool(readiness["claimable"])
                connection.execute(
                    "UPDATE coordination_tasks SET ready = ? WHERE task_cid = ?",
                    [ready, cid],
                )
                self._record_event(
                    connection,
                    lease_id=lease.lease_id,
                    scope_key=lease.scope_key,
                    event_type="prepared_completion_aborted",
                    fencing_token=int(identity["fencing_token"]),
                    fence_epoch=int(identity["fence_epoch"]),
                    observed_at_ms=now,
                    body={
                        "task_cid": cid,
                        "claim_id": str(identity["claim_id"]),
                        "attempt_id": str(identity["attempt_id"]),
                        "control_revision": int(observation["revision"]),
                        "reason": reason_text,
                        "ready": ready,
                    },
                )
                self._commit_if_idle(connection)
                return {
                    "task_cid": cid,
                    "claim_id": str(identity["claim_id"]),
                    "attempt_id": str(identity["attempt_id"]),
                    "status": "aborted",
                    "ready": ready,
                    "reason": reason_text,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def settle_task_claim(
        self,
        claim: TaskClaim | Mapping[str, Any],
        *,
        reason: str = "attempt_complete",
        now_ms: int | None = None,
    ) -> FencedLease:
        """Successfully settle and release an exactly completed task claim.

        This is the final coordination transition after the daemon has durably
        committed its own COMPLETE phase.  The matching logical completion is
        required; the claim and lease become ``released`` while the
        coordination task attempt becomes ``succeeded`` in one transaction.
        """

        identity = self._task_claim_identity(claim)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        task_cid = str(identity["task_cid"])
        claim_id = str(identity["claim_id"])
        attempt_id = str(identity["attempt_id"])
        token = int(identity["fencing_token"])
        epoch = int(identity["fence_epoch"])
        reason_text = str(reason or "attempt_complete")[:256]
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                state_row = connection.execute(
                    """
                    SELECT l.state AS lease_state, c.state AS claim_state,
                           a.status AS attempt_status
                    FROM fenced_leases AS l
                    JOIN task_claims AS c ON c.claim_id = l.claim_id
                    JOIN task_attempts AS a ON a.attempt_id = l.attempt_id
                    WHERE l.lease_id = ? AND c.claim_id = ? AND a.attempt_id = ?
                    """,
                    [str(identity["lease_id"]), claim_id, attempt_id],
                ).fetchone()
                state_mapping = _row_mapping(state_row)
                already_settled = (
                    str(_row_get(state_mapping, "lease_state", "0", default=""))
                    == LeaseState.RELEASED.value
                    and str(
                        _row_get(state_mapping, "claim_state", "1", default="")
                    )
                    == LeaseState.RELEASED.value
                    and str(
                        _row_get(state_mapping, "attempt_status", "2", default="")
                    )
                    == AttemptStatus.SUCCEEDED.value
                )
                lease = self._protect_task_claim_unlocked(
                    connection,
                    identity=identity,
                    now=now,
                    expected_attempt_status=(
                        AttemptStatus.SUCCEEDED
                        if already_settled
                        else AttemptStatus.RUNNING
                    ),
                    allow_logically_completed=True,
                    record_event=False,
                    expected_lease_state=(
                        LeaseState.RELEASED
                        if already_settled
                        else LeaseState.ACCEPTED
                    ),
                )
                self._task_completion_for_identity_unlocked(
                    connection,
                    identity=identity,
                    required=True,
                )
                if already_settled:
                    self._commit_if_idle(connection)
                    return lease

                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET state = ?, revision = revision + 1
                    WHERE lease_id = ? AND task_cid = ? AND claim_id = ?
                      AND attempt_id = ? AND owner_session_id = ?
                      AND fencing_token = ? AND fence_epoch = ? AND state = ?
                    """,
                    [
                        LeaseState.RELEASED.value,
                        str(identity["lease_id"]),
                        task_cid,
                        claim_id,
                        attempt_id,
                        str(identity["owner_session_id"]),
                        token,
                        epoch,
                        LeaseState.ACCEPTED.value,
                    ],
                )
                connection.execute(
                    """
                    UPDATE task_claims
                    SET state = ?, released_at_ms = ?, revision = revision + 1
                    WHERE claim_id = ? AND task_cid = ? AND attempt_id = ?
                      AND lease_id = ? AND owner_session_id = ?
                      AND fencing_token = ? AND fence_epoch = ? AND state = ?
                    """,
                    [
                        LeaseState.RELEASED.value,
                        now,
                        claim_id,
                        task_cid,
                        attempt_id,
                        str(identity["lease_id"]),
                        str(identity["owner_session_id"]),
                        token,
                        epoch,
                        LeaseState.ACCEPTED.value,
                    ],
                )
                connection.execute(
                    """
                    UPDATE task_attempts
                    SET status = ?, finished_at_ms = ?, revision = revision + 1
                    WHERE attempt_id = ? AND task_cid = ?
                      AND owner_session_id = ? AND fencing_token = ?
                      AND fence_epoch = ? AND status = ?
                    """,
                    [
                        AttemptStatus.SUCCEEDED.value,
                        now,
                        attempt_id,
                        task_cid,
                        str(identity["owner_session_id"]),
                        token,
                        epoch,
                        AttemptStatus.RUNNING.value,
                    ],
                )
                released_lease_row = connection.execute(
                    "SELECT * FROM fenced_leases WHERE lease_id = ?",
                    [str(identity["lease_id"])],
                ).fetchone()
                released_claim_row = connection.execute(
                    "SELECT state FROM task_claims WHERE claim_id = ?",
                    [claim_id],
                ).fetchone()
                settled_attempt_row = connection.execute(
                    "SELECT status FROM task_attempts WHERE attempt_id = ?",
                    [attempt_id],
                ).fetchone()
                if (
                    released_lease_row is None
                    or self._lease_from_row(released_lease_row).state
                    is not LeaseState.RELEASED
                    or released_claim_row is None
                    or str(
                        _row_get(
                            _row_mapping(released_claim_row),
                            "state",
                            "0",
                            default="",
                        )
                    )
                    != LeaseState.RELEASED.value
                    or settled_attempt_row is None
                    or str(
                        _row_get(
                            _row_mapping(settled_attempt_row),
                            "status",
                            "0",
                            default="",
                        )
                    )
                    != AttemptStatus.SUCCEEDED.value
                ):
                    raise DatabaseCoordinationStaleFenceError(
                        "task claim settlement lost its exact fence"
                    )
                self._record_event(
                    connection,
                    lease_id=lease.lease_id,
                    scope_key=lease.scope_key,
                    event_type="task_claim_settled",
                    fencing_token=token,
                    fence_epoch=epoch,
                    observed_at_ms=now,
                    body={
                        "task_cid": task_cid,
                        "claim_id": claim_id,
                        "attempt_id": attempt_id,
                        "reason": reason_text,
                    },
                )
                self._commit_if_idle(connection)
                refreshed = self.get_lease(lease.lease_id)
                assert refreshed is not None
                return refreshed
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- specialized claim APIs ---------------------------------------------

    def claim_task(
        self,
        *,
        task_cid: str,
        owner_session_id: str,
        lease_ms: int | None = None,
        worktree_id: str = "",
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> TaskClaim:
        """Claim a task and create its task attempt in one transaction."""

        cid = _text(task_cid, "task_cid")
        owner = _text(owner_session_id, "owner_session_id")
        duration = (
            self._default_lease_ms
            if lease_ms is None
            else _lease_duration_ms(int(lease_ms))
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        idem = _text(idempotency_key, "idempotency_key", required=False)
        scope_key = exclusive_scope_key(
            lease_kind=LeaseKind.TASK, scope=cid, task_cid=cid
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                task_row = connection.execute(
                    "SELECT * FROM coordination_tasks WHERE task_cid = ?",
                    [cid],
                ).fetchone()
                if task_row is None:
                    raise KeyError(f"unknown task CID: {cid}")
                completion_row = connection.execute(
                    "SELECT status FROM task_completions WHERE task_cid = ?",
                    [cid],
                ).fetchone()
                if completion_row is not None:
                    raise DatabaseCoordinationNotReadyError(
                        f"task {cid} already has a logical completion",
                        evidence={
                            "task_cid": cid,
                            "completion_status": str(
                                _row_get(
                                    _row_mapping(completion_row),
                                    "status",
                                    "0",
                                    default="",
                                )
                            ),
                            "reason": "already_completed",
                        },
                    )

                # Expire this task scope before considering response-loss
                # replay.  An idempotency key never revives an expired,
                # released, completed, or superseded attempt.
                self._expire_scope(connection, scope_key, now)
                if idem:
                    prior = connection.execute(
                        """
                        SELECT * FROM task_claims
                        WHERE task_cid = ? AND idempotency_key = ?
                          AND owner_session_id = ?
                          AND state = ? AND expires_at_ms > ?
                        ORDER BY claimed_at_ms DESC
                        LIMIT 1
                        """,
                        [
                            cid,
                            idem,
                            owner,
                            LeaseState.ACCEPTED.value,
                            now,
                        ],
                    ).fetchone()
                    if prior is not None:
                        claim = self._task_claim_from_row(prior)
                        self._protect_task_claim_unlocked(
                            connection,
                            identity=self._task_claim_identity(claim),
                            now=now,
                            expected_attempt_status=AttemptStatus.RUNNING,
                            allow_logically_completed=False,
                            record_event=False,
                        )
                        self._commit_if_idle(connection)
                        return claim

                readiness = self._claimability_unlocked(connection, cid)
                if not readiness["claimable"]:
                    raise DatabaseCoordinationNotReadyError(
                        f"task {cid} has unsatisfied dependencies",
                        evidence=readiness,
                    )
                task_map = _row_mapping(task_row)
                wt = _text(
                    worktree_id
                    or str(_row_get(task_map, "worktree_id", default="") or ""),
                    "worktree_id",
                    required=False,
                )

                owners = self._active_owners(connection, scope_key, now)
                if owners:
                    current = owners[0]
                    if str(current.get("owner_session_id")) == owner:
                        claim_id = str(current.get("claim_id") or "")
                        if claim_id:
                            row = connection.execute(
                                "SELECT * FROM task_claims WHERE claim_id = ?",
                                [claim_id],
                            ).fetchone()
                            if row is not None:
                                claim = self._task_claim_from_row(row)
                                self._protect_task_claim_unlocked(
                                    connection,
                                    identity=self._task_claim_identity(claim),
                                    now=now,
                                    expected_attempt_status=AttemptStatus.RUNNING,
                                    allow_logically_completed=False,
                                    record_event=False,
                                )
                                self._commit_if_idle(connection)
                                return claim
                    raise DatabaseCoordinationConflictError(
                        f"task {cid} is leased by {current.get('owner_session_id')}"
                    )

                # Attempt number is one-based and monotonic per task_cid.
                attempt_row = connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt_number), 0) AS max_attempt
                    FROM task_attempts WHERE task_cid = ?
                    """,
                    [cid],
                ).fetchone()
                attempt_number = (
                    int(
                        _row_get(
                            _row_mapping(attempt_row), "max_attempt", "0", default=0
                        )
                    )
                    + 1
                )
                token, epoch = self._next_fence(connection, scope_key)
                claim_id = _new_id("claim")
                attempt_id = _new_id("attempt")
                lease_id = _new_id("lease")
                expires = now + duration
                payload = _bounded_mapping(body, name="body")

                # Atomic: task_claim + task_attempt + fenced_lease.
                connection.execute(
                    """
                    INSERT INTO task_attempts(
                        attempt_id, task_cid, attempt_number, owner_session_id,
                        fencing_token, fence_epoch, started_at_ms, finished_at_ms,
                        status, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    [
                        attempt_id,
                        cid,
                        attempt_number,
                        owner,
                        token,
                        epoch,
                        now,
                        AttemptStatus.RUNNING.value,
                        1,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO task_claims(
                        claim_id, task_cid, owner_session_id, fencing_token,
                        fence_epoch, claimed_at_ms, expires_at_ms, released_at_ms,
                        state, revision, attempt_id, attempt_number, lease_id,
                        worktree_id, idempotency_key, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        claim_id,
                        cid,
                        owner,
                        token,
                        epoch,
                        now,
                        expires,
                        LeaseState.ACCEPTED.value,
                        1,
                        attempt_id,
                        attempt_number,
                        lease_id,
                        wt,
                        idem,
                        _canonical_json(payload),
                    ],
                )
                lease = FencedLease(
                    lease_id=lease_id,
                    lease_kind=LeaseKind.TASK,
                    scope_key=scope_key,
                    scope=cid,
                    mode=LeaseMode.EXCLUSIVE,
                    owner_session_id=owner,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    revision=1,
                    task_cid=cid,
                    worktree_id=wt,
                    claim_id=claim_id,
                    attempt_id=attempt_id,
                    attempt_number=attempt_number,
                    idempotency_key=idem,
                    body=payload,
                )
                self._insert_fenced_lease(connection, lease)
                self._record_token(
                    connection,
                    scope_key=scope_key,
                    fencing_token=token,
                    fence_epoch=epoch,
                    now=now,
                )
                self._record_event(
                    connection,
                    lease_id=lease_id,
                    scope_key=scope_key,
                    event_type="task_claimed",
                    fencing_token=token,
                    fence_epoch=epoch,
                    observed_at_ms=now,
                    body={
                        "claim_id": claim_id,
                        "attempt_id": attempt_id,
                        "attempt_number": attempt_number,
                    },
                )
                self._commit_if_idle(connection)
                return TaskClaim(
                    claim_id=claim_id,
                    task_cid=cid,
                    owner_session_id=owner,
                    fencing_token=token,
                    fence_epoch=epoch,
                    claimed_at_ms=now,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    revision=1,
                    attempt_id=attempt_id,
                    attempt_number=attempt_number,
                    lease_id=lease_id,
                    worktree_id=wt,
                    idempotency_key=idem,
                    body=payload,
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def claim_ready_task(
        self,
        *,
        owner_session_id: str,
        lease_ms: int | None = None,
        exclude_task_cids: Iterable[str] = (),
        now_ms: int | None = None,
    ) -> TaskClaim | None:
        """Fair-schedule: claim the oldest ready unclaimed task.

        Selection and acceptance share one transaction (LeaseCoordinator
        ``claim_ready`` algorithm).
        """

        owner = _text(owner_session_id, "owner_session_id")
        duration = (
            self._default_lease_ms
            if lease_ms is None
            else _lease_duration_ms(int(lease_ms))
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        excluded = {str(item) for item in exclude_task_cids}
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                # Expire all due task scopes first.
                due = connection.execute(
                    """
                    SELECT DISTINCT scope_key FROM fenced_leases
                    WHERE lease_kind = ? AND state = ? AND expires_at_ms <= ?
                    """,
                    [LeaseKind.TASK.value, LeaseState.ACCEPTED.value, now],
                ).fetchall()
                for row in due:
                    self._expire_scope(
                        connection,
                        str(_row_get(_row_mapping(row), "scope_key", "0")),
                        now,
                    )
                candidates = connection.execute(
                    """
                    SELECT * FROM coordination_tasks WHERE ready = TRUE
                    ORDER BY registered_at_ms, task_cid
                    """
                ).fetchall()
                for task in candidates:
                    mapping = _row_mapping(task)
                    cid = str(_row_get(mapping, "task_cid", default=""))
                    if not cid or cid in excluded:
                        continue
                    scope_key = exclusive_scope_key(
                        lease_kind=LeaseKind.TASK, scope=cid, task_cid=cid
                    )
                    if self._active_owners(connection, scope_key, now):
                        continue
                    recovery_owner = self._restart_recovery_owner_unlocked(
                        connection,
                        cid,
                    )
                    if recovery_owner and recovery_owner != owner:
                        continue
                    readiness = self._claimability_unlocked(connection, cid)
                    if not readiness["claimable"]:
                        continue
                    # Claim inside this same transaction.
                    claim = self._claim_task_in_transaction(
                        connection,
                        task_cid=cid,
                        owner_session_id=owner,
                        worktree_id=str(
                            _row_get(mapping, "worktree_id", default="") or ""
                        ),
                        duration=duration,
                        now=now,
                        idempotency_key="",
                        body={},
                    )
                    self._commit_if_idle(connection)
                    return claim
                self._commit_if_idle(connection)
                return None
            except Exception:
                self._rollback_if_open(connection)
                raise

    def _restart_recovery_owner_unlocked(
        self,
        connection: Any,
        task_cid: str,
    ) -> str:
        row = connection.execute(
            "SELECT body_json FROM coordination_tasks WHERE task_cid = ?",
            [task_cid],
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown task CID: {task_cid}")
        body = _decode_coordination_body(
            row[0],
            table="coordination_tasks",
            identity=task_cid,
        )
        if body.get("authority") != "task_source" or not body.get(
            "restart_recovery_ready"
        ):
            return ""
        owner = _text(
            body.get("restart_recovery_owner_session_id"),
            "restart_recovery_owner_session_id",
        )
        binding = body.get("restart_recovery_binding")
        if not isinstance(binding, Mapping):
            raise DatabaseCoordinationStaleFenceError(
                f"restart recovery projection for {task_cid} has no claim binding"
            )
        if str(binding.get("owner_session_id") or "") != owner:
            raise DatabaseCoordinationStaleFenceError(
                f"restart recovery projection owner disagrees for {task_cid}"
            )
        return owner

    def _claimability_unlocked(
        self,
        connection: Any,
        task_cid: str,
        *,
        max_evidence: int = MAX_DEPENDENCY_EVIDENCE,
    ) -> dict[str, Any]:
        limit = max(1, min(int(max_evidence), MAX_DEPENDENCY_EVIDENCE))
        completion_row = connection.execute(
            """
            SELECT completed_at_ms, status FROM task_completions
            WHERE task_cid = ?
            """,
            [task_cid],
        ).fetchone()
        completion_status = ""
        completed_at_ms: int | None = None
        if completion_row is not None:
            completion_mapping = _row_mapping(completion_row)
            completion_status = str(
                _row_get(completion_mapping, "status", "1", default="") or ""
            )
            completed_at_ms = int(
                _row_get(
                    completion_mapping,
                    "completed_at_ms",
                    "0",
                    default=0,
                )
            )
        dep_rows = connection.execute(
            """
            SELECT dependency_task_cid FROM task_dependencies
            WHERE task_cid = ? ORDER BY dependency_task_cid
            """,
            [task_cid],
        ).fetchall()
        deps = [
            str(_row_get(_row_mapping(row), "dependency_task_cid", "0"))
            for row in dep_rows
        ]
        missing: list[str] = []
        blocked: list[str] = []
        satisfied: list[str] = []
        repairs: list[dict[str, Any]] = []
        if completion_row is not None:
            repairs.append(
                {
                    "kind": "already_completed",
                    "task_cid": task_cid,
                    "completion_status": completion_status,
                    "completed_at_ms": completed_at_ms,
                }
            )
        for dep in deps:
            registered = connection.execute(
                "SELECT task_cid FROM coordination_tasks WHERE task_cid = ?",
                [dep],
            ).fetchone()
            completion = connection.execute(
                "SELECT status FROM task_completions WHERE task_cid = ?",
                [dep],
            ).fetchone()
            if registered is None and completion is None:
                missing.append(dep)
                repairs.append(
                    {
                        "kind": "missing_dependency",
                        "dependency_task_cid": dep,
                    }
                )
                continue
            if completion is None:
                blocked.append(dep)
                repairs.append(
                    {
                        "kind": "unsatisfied_dependency",
                        "dependency_task_cid": dep,
                        "latest_status": "missing",
                    }
                )
                continue
            status = str(_row_get(_row_mapping(completion), "status", "0", default=""))
            if status != "succeeded":
                blocked.append(dep)
                repairs.append(
                    {
                        "kind": "unsatisfied_dependency",
                        "dependency_task_cid": dep,
                        "latest_status": status,
                    }
                )
            else:
                satisfied.append(dep)
        return {
            "task_cid": task_cid,
            "claimable": completion_row is None and not missing and not blocked,
            "completion_status": completion_status,
            "completed_at_ms": completed_at_ms,
            "dependency_task_cids": deps,
            "missing_dependency_task_cids": missing,
            "blocked_dependency_task_cids": blocked,
            "satisfied_dependency_task_cids": satisfied,
            "repair_evidence": repairs[:limit],
            "evidence_truncated": len(repairs) > limit,
        }

    def _claim_task_in_transaction(
        self,
        connection: Any,
        *,
        task_cid: str,
        owner_session_id: str,
        worktree_id: str,
        duration: int,
        now: int,
        idempotency_key: str,
        body: Mapping[str, Any],
    ) -> TaskClaim:
        recovery_owner = self._restart_recovery_owner_unlocked(
            connection,
            task_cid,
        )
        if recovery_owner and recovery_owner != owner_session_id:
            raise DatabaseCoordinationNotReadyError(
                f"task {task_cid} restart recovery belongs to another owner",
                evidence={
                    "task_cid": task_cid,
                    "reason": "restart_recovery_owner_mismatch",
                },
            )
        scope_key = exclusive_scope_key(
            lease_kind=LeaseKind.TASK, scope=task_cid, task_cid=task_cid
        )
        attempt_row = connection.execute(
            """
            SELECT COALESCE(MAX(attempt_number), 0) AS max_attempt
            FROM task_attempts WHERE task_cid = ?
            """,
            [task_cid],
        ).fetchone()
        attempt_number = (
            int(_row_get(_row_mapping(attempt_row), "max_attempt", "0", default=0)) + 1
        )
        token, epoch = self._next_fence(connection, scope_key)
        claim_id = _new_id("claim")
        attempt_id = _new_id("attempt")
        lease_id = _new_id("lease")
        expires = now + duration
        payload = _bounded_mapping(body, name="body")
        connection.execute(
            """
            INSERT INTO task_attempts(
                attempt_id, task_cid, attempt_number, owner_session_id,
                fencing_token, fence_epoch, started_at_ms, finished_at_ms,
                status, revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            [
                attempt_id,
                task_cid,
                attempt_number,
                owner_session_id,
                token,
                epoch,
                now,
                AttemptStatus.RUNNING.value,
                1,
            ],
        )
        connection.execute(
            """
            INSERT INTO task_claims(
                claim_id, task_cid, owner_session_id, fencing_token,
                fence_epoch, claimed_at_ms, expires_at_ms, released_at_ms,
                state, revision, attempt_id, attempt_number, lease_id,
                worktree_id, idempotency_key, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                claim_id,
                task_cid,
                owner_session_id,
                token,
                epoch,
                now,
                expires,
                LeaseState.ACCEPTED.value,
                1,
                attempt_id,
                attempt_number,
                lease_id,
                worktree_id,
                idempotency_key,
                _canonical_json(payload),
            ],
        )
        lease = FencedLease(
            lease_id=lease_id,
            lease_kind=LeaseKind.TASK,
            scope_key=scope_key,
            scope=task_cid,
            mode=LeaseMode.EXCLUSIVE,
            owner_session_id=owner_session_id,
            fencing_token=token,
            fence_epoch=epoch,
            acquired_at_ms=now,
            expires_at_ms=expires,
            state=LeaseState.ACCEPTED,
            revision=1,
            task_cid=task_cid,
            worktree_id=worktree_id,
            claim_id=claim_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            idempotency_key=idempotency_key,
            body=payload,
        )
        self._insert_fenced_lease(connection, lease)
        self._record_token(
            connection,
            scope_key=scope_key,
            fencing_token=token,
            fence_epoch=epoch,
            now=now,
        )
        self._record_event(
            connection,
            lease_id=lease_id,
            scope_key=scope_key,
            event_type="task_claimed",
            fencing_token=token,
            fence_epoch=epoch,
            observed_at_ms=now,
            body={
                "claim_id": claim_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
            },
        )
        return TaskClaim(
            claim_id=claim_id,
            task_cid=task_cid,
            owner_session_id=owner_session_id,
            fencing_token=token,
            fence_epoch=epoch,
            claimed_at_ms=now,
            expires_at_ms=expires,
            state=LeaseState.ACCEPTED,
            revision=1,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            lease_id=lease_id,
            worktree_id=worktree_id,
            idempotency_key=idempotency_key,
            body=payload,
        )

    def _task_claim_from_row(self, row: Any) -> TaskClaim:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        return TaskClaim(
            claim_id=str(_row_get(mapping, "claim_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            owner_session_id=str(_row_get(mapping, "owner_session_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=1)),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=1)),
            claimed_at_ms=int(_row_get(mapping, "claimed_at_ms", default=0)),
            expires_at_ms=int(_row_get(mapping, "expires_at_ms", default=0)),
            state=LeaseState(str(_row_get(mapping, "state", default="accepted"))),
            revision=int(_row_get(mapping, "revision", default=1)),
            attempt_id=str(_row_get(mapping, "attempt_id", default="")),
            attempt_number=int(_row_get(mapping, "attempt_number", default=1)),
            lease_id=str(_row_get(mapping, "lease_id", default="")),
            worktree_id=str(_row_get(mapping, "worktree_id", default="") or ""),
            idempotency_key=str(_row_get(mapping, "idempotency_key", default="") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    def get_task_claim(self, claim_id: str) -> TaskClaim | None:
        cid = _text(claim_id, "claim_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM task_claims WHERE claim_id = ?",
                [cid],
            ).fetchone()
            if row is None:
                return None
            return self._task_claim_from_row(row)

    def get_task_attempt(self, attempt_id: str) -> TaskAttempt | None:
        aid = _text(attempt_id, "attempt_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM task_attempts WHERE attempt_id = ?",
                [aid],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            finished = _row_get(mapping, "finished_at_ms")
            return TaskAttempt(
                attempt_id=str(_row_get(mapping, "attempt_id", default="")),
                task_cid=str(_row_get(mapping, "task_cid", default="")),
                attempt_number=int(_row_get(mapping, "attempt_number", default=1)),
                owner_session_id=str(
                    _row_get(mapping, "owner_session_id", default="")
                ),
                fencing_token=int(_row_get(mapping, "fencing_token", default=1)),
                fence_epoch=int(_row_get(mapping, "fence_epoch", default=1)),
                started_at_ms=int(_row_get(mapping, "started_at_ms", default=0)),
                finished_at_ms=None if finished is None else int(finished),
                status=AttemptStatus(
                    str(_row_get(mapping, "status", default="running"))
                ),
                revision=int(_row_get(mapping, "revision", default=1)),
            )

    def claim_resource(
        self,
        *,
        resource_kind: str,
        resource_id: str,
        owner_session_id: str,
        lease_ms: int | None = None,
        mode: LeaseMode | str = LeaseMode.EXCLUSIVE,
        task_cid: str = "",
        repository_id: str = "",
        path: str = "",
        worktree_id: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> ResourceClaim:
        """Acquire a path/resource/capacity claim under unified fencing."""

        rkind = _text(resource_kind, "resource_kind")
        rid = _text(resource_id, "resource_id")
        kind = (
            LeaseKind.PATH
            if rkind == "path"
            else LeaseKind.PROVIDER_CAPACITY
            if rkind == "provider"
            else LeaseKind.PROVER_CAPACITY
            if rkind == "prover"
            else LeaseKind.MERGE
            if rkind == "merge"
            else LeaseKind.RESOURCE
        )
        lease = self.acquire(
            lease_kind=kind,
            scope=rid if rkind != "path" else path or rid,
            owner_session_id=owner_session_id,
            mode=mode,
            lease_ms=lease_ms,
            task_cid=task_cid,
            worktree_id=worktree_id,
            resource_kind=rkind,
            resource_id=rid,
            repository_id=repository_id,
            path=path if rkind == "path" else "",
            body=body,
            now_ms=now_ms,
        )
        claim_id = lease.claim_id or lease.lease_id
        claim = ResourceClaim(
            claim_id=claim_id,
            resource_kind=rkind,
            resource_id=rid,
            owner_session_id=lease.owner_session_id,
            fencing_token=lease.fencing_token,
            fence_epoch=lease.fence_epoch,
            acquired_at_ms=lease.acquired_at_ms,
            expires_at_ms=lease.expires_at_ms,
            state=lease.state,
            revision=lease.revision,
            lease_id=lease.lease_id,
            task_cid=lease.task_cid,
            repository_id=lease.repository_id,
            path=lease.path,
            worktree_id=lease.worktree_id,
            mode=lease.mode,
            body=dict(lease.body),
        )
        # Persist specialized projection and back-fill claim_id on the lease.
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT claim_id FROM resource_claims WHERE claim_id = ?",
                    [claim.claim_id],
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO resource_claims(
                            claim_id, resource_kind, resource_id, owner_session_id,
                            fencing_token, fence_epoch, acquired_at_ms, expires_at_ms,
                            state, revision, lease_id, task_cid, repository_id, path,
                            worktree_id, mode, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            claim.claim_id,
                            claim.resource_kind,
                            claim.resource_id,
                            claim.owner_session_id,
                            int(claim.fencing_token),
                            int(claim.fence_epoch),
                            int(claim.acquired_at_ms),
                            int(claim.expires_at_ms),
                            claim.state.value,
                            int(claim.revision),
                            claim.lease_id,
                            claim.task_cid,
                            claim.repository_id,
                            claim.path,
                            claim.worktree_id,
                            claim.mode.value,
                            _canonical_json(dict(claim.body)),
                        ],
                    )
                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET claim_id = ?, resource_kind = ?, resource_id = ?
                    WHERE lease_id = ?
                    """,
                    [
                        claim.claim_id,
                        claim.resource_kind,
                        claim.resource_id,
                        claim.lease_id,
                    ],
                )
                self._commit_if_idle(connection)
                return claim
            except Exception:
                self._rollback_if_open(connection)
                raise

    def acquire_maintenance_lease(
        self,
        *,
        owner_session_id: str,
        scope: str = DEFAULT_MAINTENANCE_SCOPE,
        process_birth_id: str = "",
        lease_ms: int | None = None,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> MaintenanceLease:
        """Acquire exclusive schema/backup/offline-recovery maintenance lease."""

        lease = self.acquire(
            lease_kind=LeaseKind.MAINTENANCE,
            scope=scope,
            owner_session_id=owner_session_id,
            mode=LeaseMode.EXCLUSIVE,
            lease_ms=lease_ms,
            body=body,
            now_ms=now_ms,
        )
        record = MaintenanceLease(
            lease_id=lease.lease_id,
            scope=lease.scope,
            owner_session_id=lease.owner_session_id,
            fencing_token=lease.fencing_token,
            fence_epoch=lease.fence_epoch,
            acquired_at_ms=lease.acquired_at_ms,
            expires_at_ms=lease.expires_at_ms,
            state=lease.state,
            revision=lease.revision,
            process_birth_id=_text(
                process_birth_id, "process_birth_id", required=False
            ),
            body=dict(lease.body),
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT lease_id FROM maintenance_leases WHERE lease_id = ?",
                    [record.lease_id],
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO maintenance_leases(
                            lease_id, scope, owner_session_id, process_birth_id,
                            fencing_token, fence_epoch, acquired_at_ms, expires_at_ms,
                            released_at_ms, state, revision, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                        """,
                        [
                            record.lease_id,
                            record.scope,
                            record.owner_session_id,
                            record.process_birth_id,
                            int(record.fencing_token),
                            int(record.fence_epoch),
                            int(record.acquired_at_ms),
                            int(record.expires_at_ms),
                            record.state.value,
                            int(record.revision),
                            _canonical_json(dict(record.body)),
                        ],
                    )
                self._commit_if_idle(connection)
                return record
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_maintenance_lease(self, lease_id: str) -> MaintenanceLease | None:
        lid = _text(lease_id, "lease_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM maintenance_leases WHERE lease_id = ?",
                [lid],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            body_raw = _row_get(mapping, "body_json", default="{}")
            try:
                body = json.loads(str(body_raw or "{}"))
            except json.JSONDecodeError:
                body = {}
            return MaintenanceLease(
                lease_id=str(_row_get(mapping, "lease_id", default="")),
                scope=str(_row_get(mapping, "scope", default="")),
                owner_session_id=str(
                    _row_get(mapping, "owner_session_id", default="")
                ),
                fencing_token=int(_row_get(mapping, "fencing_token", default=1)),
                fence_epoch=int(_row_get(mapping, "fence_epoch", default=1)),
                acquired_at_ms=int(_row_get(mapping, "acquired_at_ms", default=0)),
                expires_at_ms=int(_row_get(mapping, "expires_at_ms", default=0)),
                state=LeaseState(str(_row_get(mapping, "state", default="accepted"))),
                revision=int(_row_get(mapping, "revision", default=1)),
                process_birth_id=str(
                    _row_get(mapping, "process_birth_id", default="") or ""
                ),
                body=body if isinstance(body, Mapping) else {},
            )

    def lease_events(
        self,
        *,
        scope_key: str | None = None,
        lease_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent lease events for diagnostics."""

        bound = max(1, min(int(limit), 10_000))
        with self._lock:
            connection = self._require()
            clauses: list[str] = []
            params: list[Any] = []
            if scope_key is not None:
                clauses.append("scope_key = ?")
                params.append(_text(scope_key, "scope_key"))
            if lease_id is not None:
                clauses.append("lease_id = ?")
                params.append(_text(lease_id, "lease_id"))
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = connection.execute(
                f"""
                SELECT * FROM lease_events{where}
                ORDER BY observed_at_ms DESC, event_id DESC
                LIMIT {bound}
                """,
                params,
            ).fetchall()
            events: list[dict[str, Any]] = []
            for row in rows:
                mapping = _row_mapping(row)
                body_raw = _row_get(mapping, "body_json", default="{}")
                try:
                    body = json.loads(str(body_raw or "{}"))
                except json.JSONDecodeError:
                    body = {}
                events.append(
                    {
                        "schema": LEASE_EVENT_SCHEMA,
                        "event_id": str(_row_get(mapping, "event_id", default="")),
                        "lease_id": str(_row_get(mapping, "lease_id", default="")),
                        "scope_key": str(_row_get(mapping, "scope_key", default="")),
                        "event_type": str(_row_get(mapping, "event_type", default="")),
                        "fencing_token": int(
                            _row_get(mapping, "fencing_token", default=0)
                        ),
                        "fence_epoch": int(
                            _row_get(mapping, "fence_epoch", default=0)
                        ),
                        "observed_at_ms": int(
                            _row_get(mapping, "observed_at_ms", default=0)
                        ),
                        "body": body if isinstance(body, Mapping) else {},
                    }
                )
            return events


def open_database_coordinator(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    default_lease_ms: int = DEFAULT_LEASE_MS,
) -> DatabaseCoordinator:
    """Open a :class:`DatabaseCoordinator` on ``database_path``."""

    return DatabaseCoordinator(
        database_path,
        clock_ms=clock_ms,
        default_lease_ms=default_lease_ms,
    ).open()


def read_coordination_registry_projection(
    database_path: Path | str,
) -> dict[str, Any]:
    """Read the coordination authority without mutating or repairing it.

    The database must already exist and contain the exact landed coordination
    tables, indexes, and required authority metadata.  This function opens a
    DuckDB ``read_only`` connection directly: it does not create parent
    directories or lock files, install DDL, repair metadata, begin a write
    transaction, sweep expiry, or commit.  Missing and tampered authorities
    fail closed.
    """

    path = Path(database_path)
    if not path.is_file():
        raise DatabaseCoordinationStaleFenceError(
            f"coordination authority does not exist: {path}"
        )
    if not duckdb_available():
        raise DuckDBUnavailableError(
            "DuckDB is required for coordination projection; install the optional "
            "duckdb dependency"
        )
    import duckdb  # type: ignore

    try:
        connection = connect_duckdb_with_policy(
            duckdb,
            path,
            read_only=True,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
    except DatabaseCoordinationError:
        raise
    except Exception as exc:
        raise DatabaseCoordinationStaleFenceError(
            f"could not open coordination authority read-only: {path}"
        ) from exc
    try:
        return _coordination_registry_projection_from_connection(
            connection,
            validate_authority=True,
        )
    except DatabaseCoordinationError:
        raise
    except Exception as exc:
        raise DatabaseCoordinationStaleFenceError(
            "coordination authority could not be projected read-only"
        ) from exc
    finally:
        connection.close()


__all__ = [
    "DATABASE_COORDINATOR_INTERFACE",
    "FENCED_LEASE_INTERFACE",
    "TASK_CLAIM_INTERFACE",
    "RESOURCE_CLAIM_INTERFACE",
    "MAINTENANCE_LEASE_INTERFACE",
    "DATABASE_COORDINATION_SCHEMA",
    "COORDINATION_REGISTRY_PROJECTION_SCHEMA",
    "FENCED_LEASE_SCHEMA",
    "TASK_CLAIM_SCHEMA",
    "RESOURCE_CLAIM_SCHEMA",
    "MAINTENANCE_LEASE_SCHEMA",
    "CROSS_STORE_FENCE_GUARD_SCHEMA",
    "CROSS_STORE_FENCE_GUARD_EVENT",
    "CROSS_STORE_FENCE_GUARD_REQUIRED_FIELD",
    "DEFAULT_LEASE_MS",
    "DEFAULT_MAINTENANCE_SCOPE",
    "MIN_LEASE_MS",
    "MAX_LEASE_MS",
    "LeaseKind",
    "LeaseMode",
    "LeaseState",
    "AttemptStatus",
    "FencedLease",
    "TaskClaim",
    "ResourceClaim",
    "MaintenanceLease",
    "TaskAttempt",
    "DatabaseCoordinator",
    "DatabaseCoordinationError",
    "DatabaseCoordinationConflictError",
    "DatabaseCoordinationExpiredError",
    "DatabaseCoordinationStaleFenceError",
    "DatabaseCoordinationNotReadyError",
    "DatabaseCoordinationNotOpenError",
    "DatabaseCoordinationBoundsError",
    "DuckDBUnavailableError",
    "duckdb_available",
    "exclusive_scope_key",
    "open_database_coordinator",
    "read_coordination_registry_projection",
]
