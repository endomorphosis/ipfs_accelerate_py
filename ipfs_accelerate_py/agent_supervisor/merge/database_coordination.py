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

from ..task_sources.duckdb_state import open_duckdb_connection
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
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._default_lease_ms = _lease_duration_ms(int(default_lease_ms))
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseCoordinator":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
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
                else:
                    connection.execute(
                        """
                        UPDATE coordination_tasks
                        SET task_id = ?, worktree_id = ?, body_json = ?
                        WHERE task_cid = ?
                        """,
                        [
                            tid,
                            _text(worktree_id, "worktree_id", required=False),
                            _canonical_json(payload),
                            cid,
                        ],
                    )
                    connection.execute(
                        "DELETE FROM task_dependencies WHERE task_cid = ?",
                        [cid],
                    )
                for dep in deps:
                    connection.execute(
                        """
                        INSERT INTO task_dependencies(task_cid, dependency_task_cid)
                        VALUES (?, ?)
                        """,
                        [cid, dep],
                    )
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
                self._commit_if_idle(connection)
                return {
                    "task_cid": cid,
                    "completed_at_ms": now,
                    "status": status_text,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

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
            dep_rows = connection.execute(
                """
                SELECT dependency_task_cid FROM task_dependencies
                WHERE task_cid = ? ORDER BY dependency_task_cid
                """,
                [cid],
            ).fetchall()
            deps = [
                str(_row_get(_row_mapping(row), "dependency_task_cid", "0"))
                for row in dep_rows
            ]
            missing: list[str] = []
            blocked: list[str] = []
            satisfied: list[str] = []
            repairs: list[dict[str, Any]] = []
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
                            "message": "dependency is not registered or completed",
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
                status = str(
                    _row_get(_row_mapping(completion), "status", "0", default="")
                )
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
            truncated = len(repairs) > limit
            evidence = repairs[:limit]
            claimable = not missing and not blocked
            return {
                "task_cid": cid,
                "claimable": claimable,
                "dependency_task_cids": deps,
                "missing_dependency_task_cids": missing,
                "blocked_dependency_task_cids": blocked,
                "satisfied_dependency_task_cids": satisfied,
                "repair_evidence": evidence,
                "evidence_truncated": truncated,
            }

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
                if idem:
                    prior = connection.execute(
                        """
                        SELECT * FROM task_claims
                        WHERE idempotency_key = ? AND owner_session_id = ?
                        ORDER BY claimed_at_ms DESC
                        LIMIT 1
                        """,
                        [idem, owner],
                    ).fetchone()
                    if prior is not None:
                        claim = self._task_claim_from_row(prior)
                        self._commit_if_idle(connection)
                        return claim

                task_row = connection.execute(
                    "SELECT * FROM coordination_tasks WHERE task_cid = ?",
                    [cid],
                ).fetchone()
                if task_row is None:
                    raise KeyError(f"unknown task CID: {cid}")
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
                                self._commit_if_idle(connection)
                                return self._task_claim_from_row(row)
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
                    SELECT * FROM coordination_tasks
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

    def _claimability_unlocked(
        self,
        connection: Any,
        task_cid: str,
        *,
        max_evidence: int = MAX_DEPENDENCY_EVIDENCE,
    ) -> dict[str, Any]:
        limit = max(1, min(int(max_evidence), MAX_DEPENDENCY_EVIDENCE))
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
            "claimable": not missing and not blocked,
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


__all__ = [
    "DATABASE_COORDINATOR_INTERFACE",
    "FENCED_LEASE_INTERFACE",
    "TASK_CLAIM_INTERFACE",
    "RESOURCE_CLAIM_INTERFACE",
    "MAINTENANCE_LEASE_INTERFACE",
    "DATABASE_COORDINATION_SCHEMA",
    "FENCED_LEASE_SCHEMA",
    "TASK_CLAIM_SCHEMA",
    "RESOURCE_CLAIM_SCHEMA",
    "MAINTENANCE_LEASE_SCHEMA",
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
]
