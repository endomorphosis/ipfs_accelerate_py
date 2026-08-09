"""Consolidate task, resource, merge, and maintenance leases with fencing.

DQP-015 / FencedLease@1, TaskClaim@1, ResourceClaim@1, MaintenanceLease@1
=======================================================================

:class:`DatabaseCoordinator` is the unified durable authority for exclusive and
shared coordination scopes. Task claims, path/resource claims, provider/prover
capacity, merge ownership, schema maintenance, backup, and offline recovery
share one lease vocabulary keyed by canonical task, worktree, and session IDs.

Authority rules (fail-closed)
-----------------------------
* At most one accepted live owner exists per exclusive scope.
* Every accepted claim carries a monotonic fence epoch and fencing token.
* Expired or superseded sessions cannot renew or mutate under a stale fence.
* Shared append / fair-schedule scopes remain concurrent (non-exclusive).
* Every protected write CAS-checks fence epoch + token before committing.
* Task claim and task-attempt rows are created in one immediate transaction.

Conflict policy
---------------
Algorithms mirror :class:`~.lease_coordination.LeaseCoordinator` (epoch/token
advancement, expiry before claim, CAS renew/release). Legacy
``LeaseCoordinator`` stores remain until canary cutover; this module does not
delete or rewrite them.

Cold import of this module performs no filesystem, database, network, provider,
or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterator

from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes
from .lease_coordination import (
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

DATABASE_COORDINATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-coordinator@1"
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
OWNER_SESSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/coordination-session@1"
)
PROTECTED_WRITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protected-write@1"
)
FAIR_SCHEDULE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/fair-schedule-entry@1"
)

MIN_LEASE_MS: Final[int] = 5_000
MAX_LEASE_MS: Final[int] = 300_000
DEFAULT_LEASE_MS: Final[int] = 60_000
DEFAULT_SESSION_TTL_MS: Final[int] = 120_000
MAX_PAYLOAD_BYTES: Final[int] = 262_144
MAX_FAIR_QUEUE_NAME_BYTES: Final[int] = 512

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS coordination_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS owner_sessions (
    session_id VARCHAR PRIMARY KEY,
    owner_did VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL DEFAULT '',
    fence_epoch BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL,
    attached_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS owner_sessions_status_idx
    ON owner_sessions(status, expires_at_ms);

CREATE TABLE IF NOT EXISTS scope_epochs (
    scope_key VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL,
    last_owner_session_id VARCHAR NOT NULL DEFAULT '',
    updated_at_ms BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS fenced_leases (
    lease_id VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    scope_key VARCHAR NOT NULL,
    scope_mode VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    claim_id VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    released_at_ms BIGINT,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    capacity_units BIGINT NOT NULL DEFAULT 1,
    resource_kind VARCHAR NOT NULL DEFAULT '',
    resource_id VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS fenced_leases_scope_state_idx
    ON fenced_leases(scope_key, state, expires_at_ms);
CREATE INDEX IF NOT EXISTS fenced_leases_session_idx
    ON fenced_leases(owner_session_id, state);
CREATE INDEX IF NOT EXISTS fenced_leases_kind_idx
    ON fenced_leases(kind, state);

CREATE TABLE IF NOT EXISTS task_claims (
    claim_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    claimed_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    released_at_ms BIGINT,
    state VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS task_claims_task_idx
    ON task_claims(task_cid, state);
CREATE INDEX IF NOT EXISTS task_claims_idempotency_idx
    ON task_claims(idempotency_key);

CREATE TABLE IF NOT EXISTS task_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claim_id VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS task_attempts_task_number_uidx
    ON task_attempts(task_cid, attempt_number);
CREATE INDEX IF NOT EXISTS task_attempts_claim_idx
    ON task_attempts(claim_id);

CREATE TABLE IF NOT EXISTS resource_claims (
    claim_id VARCHAR PRIMARY KEY,
    resource_kind VARCHAR NOT NULL,
    resource_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    released_at_ms BIGINT,
    state VARCHAR NOT NULL,
    capacity_units BIGINT NOT NULL DEFAULT 1,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS resource_claims_resource_idx
    ON resource_claims(resource_kind, resource_id, state);

CREATE TABLE IF NOT EXISTS maintenance_leases (
    lease_id VARCHAR PRIMARY KEY,
    scope VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    released_at_ms BIGINT,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    purpose VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS maintenance_leases_scope_idx
    ON maintenance_leases(scope, state);

CREATE TABLE IF NOT EXISTS protected_writes (
    write_id VARCHAR PRIMARY KEY,
    lease_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    write_kind VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS protected_writes_lease_idx
    ON protected_writes(lease_id, recorded_at_ms);

CREATE TABLE IF NOT EXISTS fair_schedule_entries (
    entry_id VARCHAR PRIMARY KEY,
    queue_name VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL DEFAULT '',
    enqueued_at_ms BIGINT NOT NULL,
    ordinal BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS fair_schedule_queue_idx
    ON fair_schedule_entries(queue_name, state, ordinal);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseCoordinationError(LeaseError):
    """Base fail-closed error for consolidated lease coordination."""

    code = "G_COORDINATION_ERROR"


class DatabaseCoordinationConflictError(DatabaseCoordinationError, LeaseConflictError):
    """Exclusive-scope conflict, duplicate owner, or CAS mismatch."""

    code = "G_CLAIM_CONFLICT"


class DatabaseCoordinationExpiredError(DatabaseCoordinationError, LeaseExpiredError):
    """Session or lease is expired and cannot renew or mutate."""

    code = "G_LEASE_EXPIRED"


class DatabaseCoordinationFenceError(DatabaseCoordinationError, StaleFencingTokenError):
    """Stale fencing epoch or token rejected on a protected write."""

    code = "G_CLAIM_CONFLICT"


class DatabaseCoordinationBoundsError(DatabaseCoordinationError, ValueError):
    """Payload, duration, or identity bound exceeded."""


class DatabaseCoordinationNotOpenError(DatabaseCoordinationError):
    """Operation requires an open coordinator."""


class DuckDBUnavailableError(DatabaseCoordinationError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class LeaseKind(str, Enum):
    """Closed vocabulary of consolidated lease kinds."""

    TASK = "task"
    RESOURCE = "resource"
    PATH = "path"
    MERGE = "merge"
    MAINTENANCE = "maintenance"
    PROVIDER = "provider"
    PROVER = "prover"


class ScopeMode(str, Enum):
    """How a scope arbitrates concurrent owners."""

    EXCLUSIVE = "exclusive"
    SHARED_APPEND = "shared_append"
    SHARED_FAIR = "shared_fair"


class LeaseState(str, Enum):
    ACCEPTED = "accepted"
    RELEASED = "released"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    STOPPED = "stopped"


class AttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FairEntryState(str, Enum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    DONE = "done"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ClockMs = Callable[[], int]


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


def _lease_duration_ms(value: Any, *, default: int = DEFAULT_LEASE_MS) -> int:
    duration = default if value is None else int(value)
    if duration < MIN_LEASE_MS or duration > MAX_LEASE_MS:
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


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


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


def _parse_enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise DatabaseCoordinationError(f"unknown {name}: {text}") from exc


def exclusive_scope_key(
    kind: LeaseKind | str,
    *,
    task_cid: str = "",
    resource_kind: str = "",
    resource_id: str = "",
    path: str = "",
    repository_id: str = "",
    worktree_id: str = "",
    merge_target: str = "",
    maintenance_scope: str = "",
) -> str:
    """Return the canonical exclusive-scope identity for a lease kind."""

    kind_value = (
        kind if isinstance(kind, LeaseKind) else _parse_enum(kind, LeaseKind, "kind")
    )
    assert isinstance(kind_value, LeaseKind)
    if kind_value is LeaseKind.TASK:
        return f"task:{_text(task_cid, 'task_cid')}"
    if kind_value is LeaseKind.RESOURCE:
        return (
            f"resource:{_text(resource_kind, 'resource_kind')}:"
            f"{_text(resource_id, 'resource_id')}"
        )
    if kind_value is LeaseKind.PATH:
        return (
            f"path:{_text(repository_id, 'repository_id')}:"
            f"{_text(path, 'path')}"
        )
    if kind_value is LeaseKind.MERGE:
        return (
            f"merge:{_text(repository_id, 'repository_id')}:"
            f"{_text(merge_target, 'merge_target')}"
        )
    if kind_value is LeaseKind.MAINTENANCE:
        return f"maintenance:{_text(maintenance_scope, 'maintenance_scope')}"
    if kind_value is LeaseKind.PROVIDER:
        return f"provider:{_text(resource_id, 'resource_id')}"
    if kind_value is LeaseKind.PROVER:
        return f"prover:{_text(resource_id, 'resource_id')}"
    raise DatabaseCoordinationError(f"unsupported lease kind: {kind_value}")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerSession:
    """Coordination owner session bound to fence identity."""

    INTERFACE: ClassVar[str] = "OwnerSession@1"
    SCHEMA: ClassVar[str] = OWNER_SESSION_SCHEMA

    session_id: str
    owner_did: str
    process_birth_id: str = ""
    fence_epoch: int = 1
    fencing_token: int = 1
    attached_at_ms: int = 0
    expires_at_ms: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(self, "owner_did", _text(self.owner_did, "owner_did"))
        object.__setattr__(
            self,
            "process_birth_id",
            _text(self.process_birth_id, "process_birth_id", required=False),
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self,
            "fencing_token",
            _positive_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self,
            "attached_at_ms",
            _nonneg_int(int(self.attached_at_ms), "attached_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _nonneg_int(int(self.expires_at_ms), "expires_at_ms"),
        )
        status = self.status
        if not isinstance(status, SessionStatus):
            status = _parse_enum(status, SessionStatus, "status")
            object.__setattr__(self, "status", status)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def active(self) -> bool:
        return self.status is SessionStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "session_id": self.session_id,
            "owner_did": self.owner_did,
            "process_birth_id": self.process_birth_id,
            "fence_epoch": int(self.fence_epoch),
            "fencing_token": int(self.fencing_token),
            "attached_at_ms": int(self.attached_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "status": self.status.value
            if isinstance(self.status, SessionStatus)
            else str(self.status),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class FencedLease:
    """Unified fenced ownership grant for any consolidated lease kind."""

    INTERFACE: ClassVar[str] = FENCED_LEASE_INTERFACE
    SCHEMA: ClassVar[str] = FENCED_LEASE_SCHEMA

    lease_id: str
    kind: LeaseKind
    scope_key: str
    scope_mode: ScopeMode
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at_ms: int
    expires_at_ms: int
    state: LeaseState = LeaseState.ACCEPTED
    task_cid: str = ""
    worktree_id: str = ""
    claim_id: str = ""
    attempt_id: str = ""
    capacity_units: int = 1
    resource_kind: str = ""
    resource_id: str = ""
    revision: int = 1
    released_at_ms: int | None = None
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        kind = self.kind
        if not isinstance(kind, LeaseKind):
            kind = _parse_enum(kind, LeaseKind, "kind")
            object.__setattr__(self, "kind", kind)
        mode = self.scope_mode
        if not isinstance(mode, ScopeMode):
            mode = _parse_enum(mode, ScopeMode, "scope_mode")
            object.__setattr__(self, "scope_mode", mode)
        object.__setattr__(self, "scope_key", _text(self.scope_key, "scope_key"))
        object.__setattr__(
            self,
            "owner_session_id",
            _text(self.owner_session_id, "owner_session_id"),
        )
        object.__setattr__(
            self,
            "fencing_token",
            _positive_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self,
            "acquired_at_ms",
            _nonneg_int(int(self.acquired_at_ms), "acquired_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _nonneg_int(int(self.expires_at_ms), "expires_at_ms"),
        )
        state = self.state
        if not isinstance(state, LeaseState):
            state = _parse_enum(state, LeaseState, "state")
            object.__setattr__(self, "state", state)
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self,
            "worktree_id",
            _text(self.worktree_id, "worktree_id", required=False),
        )
        object.__setattr__(
            self, "claim_id", _text(self.claim_id, "claim_id", required=False)
        )
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", required=False)
        )
        object.__setattr__(
            self,
            "capacity_units",
            _positive_int(int(self.capacity_units), "capacity_units"),
        )
        object.__setattr__(
            self,
            "resource_kind",
            _text(self.resource_kind, "resource_kind", required=False),
        )
        object.__setattr__(
            self,
            "resource_id",
            _text(self.resource_id, "resource_id", required=False),
        )
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        if self.released_at_ms is not None:
            object.__setattr__(
                self,
                "released_at_ms",
                _nonneg_int(int(self.released_at_ms), "released_at_ms"),
            )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def accepted(self) -> bool:
        return self.state is LeaseState.ACCEPTED

    @property
    def logical_epoch(self) -> int:
        """Alias for LeaseCoordinator-compatible epoch naming."""

        return int(self.fence_epoch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "lease_id": self.lease_id,
            "kind": self.kind.value if isinstance(self.kind, LeaseKind) else str(self.kind),
            "scope_key": self.scope_key,
            "scope_mode": self.scope_mode.value
            if isinstance(self.scope_mode, ScopeMode)
            else str(self.scope_mode),
            "owner_session_id": self.owner_session_id,
            "task_cid": self.task_cid,
            "worktree_id": self.worktree_id,
            "claim_id": self.claim_id,
            "attempt_id": self.attempt_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "logical_epoch": int(self.fence_epoch),
            "acquired_at_ms": int(self.acquired_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "released_at_ms": self.released_at_ms,
            "state": self.state.value if isinstance(self.state, LeaseState) else str(self.state),
            "capacity_units": int(self.capacity_units),
            "resource_kind": self.resource_kind,
            "resource_id": self.resource_id,
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class TaskClaim:
    """Accepted task ownership bound to a task attempt in one transaction."""

    INTERFACE: ClassVar[str] = TASK_CLAIM_INTERFACE
    SCHEMA: ClassVar[str] = TASK_CLAIM_SCHEMA

    claim_id: str
    task_cid: str
    owner_session_id: str
    lease_id: str
    fencing_token: int
    fence_epoch: int
    claimed_at_ms: int
    expires_at_ms: int
    attempt_id: str
    attempt_number: int
    state: LeaseState = LeaseState.ACCEPTED
    revision: int = 1
    idempotency_key: str = ""
    worktree_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self,
            "owner_session_id",
            _text(self.owner_session_id, "owner_session_id"),
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self,
            "fencing_token",
            _positive_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self,
            "claimed_at_ms",
            _nonneg_int(int(self.claimed_at_ms), "claimed_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _nonneg_int(int(self.expires_at_ms), "expires_at_ms"),
        )
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(
            self,
            "attempt_number",
            _positive_int(int(self.attempt_number), "attempt_number"),
        )
        state = self.state
        if not isinstance(state, LeaseState):
            state = _parse_enum(state, LeaseState, "state")
            object.__setattr__(self, "state", state)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key", required=False),
        )
        object.__setattr__(
            self,
            "worktree_id",
            _text(self.worktree_id, "worktree_id", required=False),
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
            "lease_id": self.lease_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "logical_epoch": int(self.fence_epoch),
            "claimed_at_ms": int(self.claimed_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "attempt_id": self.attempt_id,
            "attempt_number": int(self.attempt_number),
            "state": self.state.value if isinstance(self.state, LeaseState) else str(self.state),
            "revision": int(self.revision),
            "idempotency_key": self.idempotency_key,
            "worktree_id": self.worktree_id,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class TaskAttempt:
    """Task attempt created atomically with its task claim."""

    SCHEMA: ClassVar[str] = TASK_ATTEMPT_SCHEMA

    attempt_id: str
    task_cid: str
    claim_id: str
    attempt_number: int
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    started_at_ms: int
    status: AttemptStatus = AttemptStatus.RUNNING
    finished_at_ms: int | None = None
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt_id": self.attempt_id,
            "task_cid": self.task_cid,
            "claim_id": self.claim_id,
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
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class ResourceClaim:
    """Provider, prover, path, or generic resource capacity claim."""

    INTERFACE: ClassVar[str] = RESOURCE_CLAIM_INTERFACE
    SCHEMA: ClassVar[str] = RESOURCE_CLAIM_SCHEMA

    claim_id: str
    resource_kind: str
    resource_id: str
    owner_session_id: str
    lease_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at_ms: int
    expires_at_ms: int
    state: LeaseState = LeaseState.ACCEPTED
    task_cid: str = ""
    capacity_units: int = 1
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(
            self, "resource_kind", _text(self.resource_kind, "resource_kind")
        )
        object.__setattr__(self, "resource_id", _text(self.resource_id, "resource_id"))
        object.__setattr__(
            self,
            "owner_session_id",
            _text(self.owner_session_id, "owner_session_id"),
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self,
            "fencing_token",
            _positive_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self,
            "acquired_at_ms",
            _nonneg_int(int(self.acquired_at_ms), "acquired_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _nonneg_int(int(self.expires_at_ms), "expires_at_ms"),
        )
        state = self.state
        if not isinstance(state, LeaseState):
            state = _parse_enum(state, LeaseState, "state")
            object.__setattr__(self, "state", state)
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self,
            "capacity_units",
            _positive_int(int(self.capacity_units), "capacity_units"),
        )
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
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
            "lease_id": self.lease_id,
            "task_cid": self.task_cid,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "logical_epoch": int(self.fence_epoch),
            "acquired_at_ms": int(self.acquired_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "state": self.state.value if isinstance(self.state, LeaseState) else str(self.state),
            "capacity_units": int(self.capacity_units),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class MaintenanceLease:
    """Schema maintenance, backup, merge serialization, or recovery lease."""

    INTERFACE: ClassVar[str] = MAINTENANCE_LEASE_INTERFACE
    SCHEMA: ClassVar[str] = MAINTENANCE_LEASE_SCHEMA

    lease_id: str
    scope: str
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at_ms: int
    expires_at_ms: int
    state: LeaseState = LeaseState.ACCEPTED
    purpose: str = ""
    revision: int = 1
    released_at_ms: int | None = None
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        object.__setattr__(
            self,
            "owner_session_id",
            _text(self.owner_session_id, "owner_session_id"),
        )
        object.__setattr__(
            self,
            "fencing_token",
            _positive_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _positive_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self,
            "acquired_at_ms",
            _nonneg_int(int(self.acquired_at_ms), "acquired_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _nonneg_int(int(self.expires_at_ms), "expires_at_ms"),
        )
        state = self.state
        if not isinstance(state, LeaseState):
            state = _parse_enum(state, LeaseState, "state")
            object.__setattr__(self, "state", state)
        object.__setattr__(
            self, "purpose", _text(self.purpose, "purpose", required=False)
        )
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        if self.released_at_ms is not None:
            object.__setattr__(
                self,
                "released_at_ms",
                _nonneg_int(int(self.released_at_ms), "released_at_ms"),
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
            "fence_epoch_alias": int(self.fence_epoch),
            "acquired_at_ms": int(self.acquired_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "released_at_ms": self.released_at_ms,
            "state": self.state.value if isinstance(self.state, LeaseState) else str(self.state),
            "purpose": self.purpose,
            "revision": int(self.revision),
            "body": dict(self.body),
            "acquired_at": _utc_iso_from_ms(self.acquired_at_ms),
            "expires_at": _utc_iso_from_ms(self.expires_at_ms),
        }


@dataclass(frozen=True)
class ProtectedWrite:
    """Receipt for a fence-checked mutation under an accepted lease."""

    SCHEMA: ClassVar[str] = PROTECTED_WRITE_SCHEMA

    write_id: str
    lease_id: str
    owner_session_id: str
    fencing_token: int
    fence_epoch: int
    write_kind: str
    recorded_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "write_id": self.write_id,
            "lease_id": self.lease_id,
            "owner_session_id": self.owner_session_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "write_kind": self.write_kind,
            "recorded_at_ms": int(self.recorded_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class FairScheduleEntry:
    """Concurrent fair-schedule / append entry (non-exclusive scope)."""

    SCHEMA: ClassVar[str] = FAIR_SCHEDULE_ENTRY_SCHEMA

    entry_id: str
    queue_name: str
    task_cid: str
    enqueued_at_ms: int
    ordinal: int
    state: FairEntryState = FairEntryState.QUEUED
    owner_session_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entry_id": self.entry_id,
            "queue_name": self.queue_name,
            "task_cid": self.task_cid,
            "owner_session_id": self.owner_session_id,
            "enqueued_at_ms": int(self.enqueued_at_ms),
            "ordinal": int(self.ordinal),
            "state": self.state.value
            if isinstance(self.state, FairEntryState)
            else str(self.state),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class TaskClaimBundle:
    """Atomic result of claim + attempt creation."""

    claim: TaskClaim
    attempt: TaskAttempt
    lease: FencedLease

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "attempt": self.attempt.to_dict(),
            "lease": self.lease.to_dict(),
        }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class DatabaseCoordinator:
    """Unified fenced lease authority for task/resource/merge/maintenance.

    Interface: ``DatabaseCoordinator@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_COORDINATOR_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_COORDINATOR_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        default_lease_ms: int = DEFAULT_LEASE_MS,
        default_session_ttl_ms: int = DEFAULT_SESSION_TTL_MS,
    ) -> None:
        self.database_path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._default_lease_ms = _lease_duration_ms(default_lease_ms)
        self._default_session_ttl_ms = _positive_int(
            int(default_session_ttl_ms), "default_session_ttl_ms"
        )
        self._lock = threading.RLock()
        self._connection: Any | None = None
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseCoordinator":
        with self._lock:
            if self.is_open:
                return self
            if not duckdb_available():
                raise DuckDBUnavailableError(
                    "DuckDB is required for DatabaseCoordinator"
                )
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self.database_path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO coordination_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    ["schema", self.SCHEMA],
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

    # -- sessions ------------------------------------------------------------

    def open_session(
        self,
        *,
        owner_did: str,
        session_id: str | None = None,
        process_birth_id: str = "",
        ttl_ms: int | None = None,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> OwnerSession:
        """Register an active owner session used by subsequent claims."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        ttl = (
            self._default_session_ttl_ms
            if ttl_ms is None
            else _positive_int(int(ttl_ms), "ttl_ms")
        )
        session = OwnerSession(
            session_id=_text(session_id or _new_id("session"), "session_id"),
            owner_did=_text(owner_did, "owner_did"),
            process_birth_id=_text(process_birth_id, "process_birth_id", required=False),
            fence_epoch=1,
            fencing_token=1,
            attached_at_ms=now,
            expires_at_ms=now + ttl,
            status=SessionStatus.ACTIVE,
            revision=1,
            body=dict(body or {}),
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._expire_sessions_locked(connection, now)
                existing = connection.execute(
                    "SELECT session_id FROM owner_sessions WHERE session_id = ?",
                    [session.session_id],
                ).fetchone()
                if existing is not None:
                    raise DatabaseCoordinationConflictError(
                        f"session already exists: {session.session_id}"
                    )
                connection.execute(
                    """
                    INSERT INTO owner_sessions(
                        session_id, owner_did, process_birth_id, fence_epoch,
                        fencing_token, attached_at_ms, expires_at_ms, status,
                        revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        session.session_id,
                        session.owner_did,
                        session.process_birth_id,
                        int(session.fence_epoch),
                        int(session.fencing_token),
                        int(session.attached_at_ms),
                        int(session.expires_at_ms),
                        session.status.value,
                        int(session.revision),
                        _canonical_json(dict(session.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return session
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_session(self, session_id: str) -> OwnerSession | None:
        with self._lock:
            connection = self._require()
            now = self._now_ms()
            self._begin(connection)
            try:
                self._expire_sessions_locked(connection, now)
                row = connection.execute(
                    "SELECT * FROM owner_sessions WHERE session_id = ?",
                    [_text(session_id, "session_id")],
                ).fetchone()
                self._commit_if_idle(connection)
                if row is None:
                    return None
                return self._session_from_row(_row_mapping(row))
            except Exception:
                self._rollback_if_open(connection)
                raise

    def renew_session(
        self,
        session_id: str,
        *,
        fencing_token: int | None = None,
        ttl_ms: int | None = None,
        now_ms: int | None = None,
    ) -> OwnerSession:
        """Extend an active non-expired session. Expired sessions fail closed."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        ttl = (
            self._default_session_ttl_ms
            if ttl_ms is None
            else _positive_int(int(ttl_ms), "ttl_ms")
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._expire_sessions_locked(connection, now)
                session = self._require_active_session_locked(
                    connection,
                    session_id,
                    now=now,
                    expected_fencing_token=fencing_token,
                )
                expires = now + ttl
                revision = int(session.revision) + 1
                connection.execute(
                    """
                    UPDATE owner_sessions
                    SET expires_at_ms = ?, revision = ?
                    WHERE session_id = ? AND status = ? AND fencing_token = ?
                    """,
                    [
                        expires,
                        revision,
                        session.session_id,
                        SessionStatus.ACTIVE.value,
                        int(session.fencing_token),
                    ],
                )
                self._commit_if_idle(connection)
                return OwnerSession(
                    session_id=session.session_id,
                    owner_did=session.owner_did,
                    process_birth_id=session.process_birth_id,
                    fence_epoch=session.fence_epoch,
                    fencing_token=session.fencing_token,
                    attached_at_ms=session.attached_at_ms,
                    expires_at_ms=expires,
                    status=SessionStatus.ACTIVE,
                    revision=revision,
                    body=dict(session.body),
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def stop_session(
        self,
        session_id: str,
        *,
        fencing_token: int | None = None,
        now_ms: int | None = None,
    ) -> OwnerSession:
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                session = self._load_session_locked(connection, session_id)
                if session is None:
                    raise DatabaseCoordinationError(f"unknown session: {session_id}")
                if fencing_token is not None and int(fencing_token) != int(
                    session.fencing_token
                ):
                    raise DatabaseCoordinationFenceError(
                        "session fencing token mismatch during stop"
                    )
                connection.execute(
                    """
                    UPDATE owner_sessions
                    SET status = ?, expires_at_ms = ?, revision = revision + 1
                    WHERE session_id = ?
                    """,
                    [SessionStatus.STOPPED.value, now, session.session_id],
                )
                # Release exclusive leases owned by the stopped session.
                self._release_session_leases_locked(connection, session.session_id, now)
                self._commit_if_idle(connection)
                return OwnerSession(
                    session_id=session.session_id,
                    owner_did=session.owner_did,
                    process_birth_id=session.process_birth_id,
                    fence_epoch=session.fence_epoch,
                    fencing_token=session.fencing_token,
                    attached_at_ms=session.attached_at_ms,
                    expires_at_ms=now,
                    status=SessionStatus.STOPPED,
                    revision=int(session.revision) + 1,
                    body=dict(session.body),
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- task claims (claim + attempt one TX) --------------------------------

    def claim_task(
        self,
        *,
        task_cid: str,
        owner_session_id: str,
        worktree_id: str = "",
        requested_lease_ms: int | None = None,
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> TaskClaimBundle:
        """Accept exclusive task ownership and create the attempt in one TX.

        Reuses LeaseCoordinator algorithms: expire first, reject active foreign
        owners, advance fence epoch/token, and bind claim + attempt atomically.
        """

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        duration = _lease_duration_ms(
            requested_lease_ms, default=self._default_lease_ms
        )
        task = _text(task_cid, "task_cid")
        scope_key = exclusive_scope_key(LeaseKind.TASK, task_cid=task)
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                session = self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                # Idempotent replay under the same key returns the prior bundle.
                if idempotency_key:
                    prior = connection.execute(
                        """
                        SELECT * FROM task_claims
                        WHERE idempotency_key = ? AND state = ?
                        """,
                        [_text(idempotency_key, "idempotency_key"), LeaseState.ACCEPTED.value],
                    ).fetchone()
                    if prior is not None:
                        prior_map = _row_mapping(prior)
                        if str(prior_map.get("task_cid")) != task:
                            raise DatabaseCoordinationConflictError(
                                "idempotency key bound to a different task"
                            )
                        if str(prior_map.get("owner_session_id")) != session.session_id:
                            raise DatabaseCoordinationConflictError(
                                "idempotency key bound to a different session"
                            )
                        claim = self._task_claim_from_row(prior_map)
                        attempt = self._load_attempt_locked(connection, claim.attempt_id)
                        lease = self._load_lease_locked(connection, claim.lease_id)
                        assert attempt is not None and lease is not None
                        self._commit_if_idle(connection)
                        return TaskClaimBundle(claim=claim, attempt=attempt, lease=lease)

                self._expire_scope_locked(connection, scope_key, now)
                active = self._active_exclusive_owner_locked(connection, scope_key, now)
                if active is not None:
                    if str(active.get("owner_session_id")) == session.session_id:
                        # Same session re-claim of still-live lease is a conflict
                        # (use renew). Distinct from takeover after expiry.
                        raise DatabaseCoordinationConflictError(
                            f"task is already leased by this session: {task}"
                        )
                    raise DatabaseCoordinationConflictError(
                        f"task is leased by session {active.get('owner_session_id')}"
                    )

                epoch, token = self._next_fence_locked(
                    connection, scope_key, kind=LeaseKind.TASK, now=now
                )
                attempt_number = self._next_attempt_number_locked(connection, task)
                claim_id = _new_id("claim")
                attempt_id = _new_id("attempt")
                lease_id = _new_id("lease")
                expires = now + duration
                lease = FencedLease(
                    lease_id=lease_id,
                    kind=LeaseKind.TASK,
                    scope_key=scope_key,
                    scope_mode=ScopeMode.EXCLUSIVE,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    task_cid=task,
                    worktree_id=_text(worktree_id, "worktree_id", required=False),
                    claim_id=claim_id,
                    attempt_id=attempt_id,
                    revision=1,
                    body=dict(body or {}),
                )
                claim = TaskClaim(
                    claim_id=claim_id,
                    task_cid=task,
                    owner_session_id=session.session_id,
                    lease_id=lease_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    claimed_at_ms=now,
                    expires_at_ms=expires,
                    attempt_id=attempt_id,
                    attempt_number=attempt_number,
                    state=LeaseState.ACCEPTED,
                    revision=1,
                    idempotency_key=_text(
                        idempotency_key, "idempotency_key", required=False
                    ),
                    worktree_id=lease.worktree_id,
                    body=dict(body or {}),
                )
                attempt = TaskAttempt(
                    attempt_id=attempt_id,
                    task_cid=task,
                    claim_id=claim_id,
                    attempt_number=attempt_number,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    started_at_ms=now,
                    status=AttemptStatus.RUNNING,
                    revision=1,
                    body=dict(body or {}),
                )
                # Single transaction: lease + claim + attempt or nothing.
                self._insert_lease_locked(connection, lease)
                connection.execute(
                    """
                    INSERT INTO task_claims(
                        claim_id, task_cid, owner_session_id, lease_id,
                        fencing_token, fence_epoch, claimed_at_ms, expires_at_ms,
                        released_at_ms, state, attempt_id, attempt_number,
                        revision, idempotency_key, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        claim.claim_id,
                        claim.task_cid,
                        claim.owner_session_id,
                        claim.lease_id,
                        int(claim.fencing_token),
                        int(claim.fence_epoch),
                        int(claim.claimed_at_ms),
                        int(claim.expires_at_ms),
                        None,
                        claim.state.value,
                        claim.attempt_id,
                        int(claim.attempt_number),
                        int(claim.revision),
                        claim.idempotency_key,
                        _canonical_json(dict(claim.body)),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO task_attempts(
                        attempt_id, task_cid, claim_id, attempt_number,
                        owner_session_id, fencing_token, fence_epoch,
                        started_at_ms, finished_at_ms, status, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        attempt.attempt_id,
                        attempt.task_cid,
                        attempt.claim_id,
                        int(attempt.attempt_number),
                        attempt.owner_session_id,
                        int(attempt.fencing_token),
                        int(attempt.fence_epoch),
                        int(attempt.started_at_ms),
                        None,
                        attempt.status.value,
                        int(attempt.revision),
                        _canonical_json(dict(attempt.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return TaskClaimBundle(claim=claim, attempt=attempt, lease=lease)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_task_claim(self, claim_id: str) -> TaskClaim | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM task_claims WHERE claim_id = ?",
                [_text(claim_id, "claim_id")],
            ).fetchone()
            if row is None:
                return None
            return self._task_claim_from_row(_row_mapping(row))

    def get_task_attempt(self, attempt_id: str) -> TaskAttempt | None:
        with self._lock:
            connection = self._require()
            return self._load_attempt_locked(connection, attempt_id)

    def list_active_task_claims(self, task_cid: str | None = None) -> list[TaskClaim]:
        with self._lock:
            connection = self._require()
            now = self._now_ms()
            self._begin(connection)
            try:
                self._expire_all_locked(connection, now)
                if task_cid is None:
                    rows = connection.execute(
                        "SELECT * FROM task_claims WHERE state = ? ORDER BY claimed_at_ms",
                        [LeaseState.ACCEPTED.value],
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT * FROM task_claims
                        WHERE state = ? AND task_cid = ?
                        ORDER BY claimed_at_ms
                        """,
                        [LeaseState.ACCEPTED.value, _text(task_cid, "task_cid")],
                    ).fetchall()
                self._commit_if_idle(connection)
                return [self._task_claim_from_row(_row_mapping(row)) for row in rows]
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- resource claims -----------------------------------------------------

    def claim_resource(
        self,
        *,
        resource_kind: str,
        resource_id: str,
        owner_session_id: str,
        task_cid: str = "",
        capacity_units: int = 1,
        requested_lease_ms: int | None = None,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> ResourceClaim:
        """Acquire exclusive ownership of a resource / provider / prover scope."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        duration = _lease_duration_ms(
            requested_lease_ms, default=self._default_lease_ms
        )
        kind_text = _text(resource_kind, "resource_kind")
        resource = _text(resource_id, "resource_id")
        # Map provider/prover kinds onto dedicated exclusive scopes.
        if kind_text in {LeaseKind.PROVIDER.value, "llm_provider"}:
            lease_kind = LeaseKind.PROVIDER
            scope_key = exclusive_scope_key(LeaseKind.PROVIDER, resource_id=resource)
        elif kind_text in {LeaseKind.PROVER.value, "proof_backend"}:
            lease_kind = LeaseKind.PROVER
            scope_key = exclusive_scope_key(LeaseKind.PROVER, resource_id=resource)
        elif kind_text == LeaseKind.PATH.value:
            lease_kind = LeaseKind.PATH
            # path claims encode repository_id:path in resource_id
            repository_id, _, path = resource.partition(":")
            scope_key = exclusive_scope_key(
                LeaseKind.PATH,
                repository_id=repository_id or "repo",
                path=path or resource,
            )
        else:
            lease_kind = LeaseKind.RESOURCE
            scope_key = exclusive_scope_key(
                LeaseKind.RESOURCE,
                resource_kind=kind_text,
                resource_id=resource,
            )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                session = self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                self._expire_scope_locked(connection, scope_key, now)
                active = self._active_exclusive_owner_locked(connection, scope_key, now)
                if active is not None:
                    raise DatabaseCoordinationConflictError(
                        f"resource scope {scope_key} is leased by "
                        f"{active.get('owner_session_id')}"
                    )
                epoch, token = self._next_fence_locked(
                    connection, scope_key, kind=lease_kind, now=now
                )
                claim_id = _new_id("rclaim")
                lease_id = _new_id("lease")
                expires = now + duration
                units = _positive_int(int(capacity_units), "capacity_units")
                lease = FencedLease(
                    lease_id=lease_id,
                    kind=lease_kind,
                    scope_key=scope_key,
                    scope_mode=ScopeMode.EXCLUSIVE,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    task_cid=_text(task_cid, "task_cid", required=False),
                    claim_id=claim_id,
                    capacity_units=units,
                    resource_kind=kind_text,
                    resource_id=resource,
                    revision=1,
                    body=dict(body or {}),
                )
                claim = ResourceClaim(
                    claim_id=claim_id,
                    resource_kind=kind_text,
                    resource_id=resource,
                    owner_session_id=session.session_id,
                    lease_id=lease_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    task_cid=lease.task_cid,
                    capacity_units=units,
                    revision=1,
                    body=dict(body or {}),
                )
                self._insert_lease_locked(connection, lease)
                connection.execute(
                    """
                    INSERT INTO resource_claims(
                        claim_id, resource_kind, resource_id, owner_session_id,
                        lease_id, task_cid, fencing_token, fence_epoch,
                        acquired_at_ms, expires_at_ms, released_at_ms, state,
                        capacity_units, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        claim.claim_id,
                        claim.resource_kind,
                        claim.resource_id,
                        claim.owner_session_id,
                        claim.lease_id,
                        claim.task_cid,
                        int(claim.fencing_token),
                        int(claim.fence_epoch),
                        int(claim.acquired_at_ms),
                        int(claim.expires_at_ms),
                        None,
                        claim.state.value,
                        int(claim.capacity_units),
                        int(claim.revision),
                        _canonical_json(dict(claim.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return claim
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_resource_claim(self, claim_id: str) -> ResourceClaim | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM resource_claims WHERE claim_id = ?",
                [_text(claim_id, "claim_id")],
            ).fetchone()
            if row is None:
                return None
            return self._resource_claim_from_row(_row_mapping(row))

    # -- maintenance / merge leases ------------------------------------------

    def acquire_maintenance_lease(
        self,
        *,
        scope: str,
        owner_session_id: str,
        purpose: str = "schema_maintenance",
        requested_lease_ms: int | None = None,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> MaintenanceLease:
        """Acquire exclusive maintenance (schema/backup/recovery) ownership."""

        return self._acquire_named_exclusive(
            kind=LeaseKind.MAINTENANCE,
            scope_label=_text(scope, "scope"),
            owner_session_id=owner_session_id,
            purpose=purpose,
            requested_lease_ms=requested_lease_ms,
            body=body,
            now_ms=now_ms,
        )

    def acquire_merge_lease(
        self,
        *,
        repository_id: str,
        merge_target: str,
        owner_session_id: str,
        requested_lease_ms: int | None = None,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Acquire exclusive merge ownership for a repository target branch."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        duration = _lease_duration_ms(
            requested_lease_ms, default=self._default_lease_ms
        )
        scope_key = exclusive_scope_key(
            LeaseKind.MERGE,
            repository_id=repository_id,
            merge_target=merge_target,
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                session = self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                self._expire_scope_locked(connection, scope_key, now)
                active = self._active_exclusive_owner_locked(connection, scope_key, now)
                if active is not None:
                    raise DatabaseCoordinationConflictError(
                        f"merge scope {scope_key} is leased by "
                        f"{active.get('owner_session_id')}"
                    )
                epoch, token = self._next_fence_locked(
                    connection, scope_key, kind=LeaseKind.MERGE, now=now
                )
                lease = FencedLease(
                    lease_id=_new_id("lease"),
                    kind=LeaseKind.MERGE,
                    scope_key=scope_key,
                    scope_mode=ScopeMode.EXCLUSIVE,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=now + duration,
                    state=LeaseState.ACCEPTED,
                    revision=1,
                    body={
                        **dict(body or {}),
                        "repository_id": _text(repository_id, "repository_id"),
                        "merge_target": _text(merge_target, "merge_target"),
                    },
                )
                self._insert_lease_locked(connection, lease)
                self._commit_if_idle(connection)
                return lease
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_maintenance_lease(self, lease_id: str) -> MaintenanceLease | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM maintenance_leases WHERE lease_id = ?",
                [_text(lease_id, "lease_id")],
            ).fetchone()
            if row is None:
                return None
            return self._maintenance_from_row(_row_mapping(row))

    def _acquire_named_exclusive(
        self,
        *,
        kind: LeaseKind,
        scope_label: str,
        owner_session_id: str,
        purpose: str,
        requested_lease_ms: int | None,
        body: Mapping[str, Any] | None,
        now_ms: int | None,
    ) -> MaintenanceLease:
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        duration = _lease_duration_ms(
            requested_lease_ms, default=self._default_lease_ms
        )
        scope_key = exclusive_scope_key(
            LeaseKind.MAINTENANCE, maintenance_scope=scope_label
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                session = self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                self._expire_scope_locked(connection, scope_key, now)
                active = self._active_exclusive_owner_locked(connection, scope_key, now)
                if active is not None:
                    raise DatabaseCoordinationConflictError(
                        f"maintenance scope {scope_label} is leased by "
                        f"{active.get('owner_session_id')}"
                    )
                epoch, token = self._next_fence_locked(
                    connection, scope_key, kind=kind, now=now
                )
                lease_id = _new_id("mlease")
                expires = now + duration
                fenced = FencedLease(
                    lease_id=lease_id,
                    kind=kind,
                    scope_key=scope_key,
                    scope_mode=ScopeMode.EXCLUSIVE,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    revision=1,
                    body={
                        **dict(body or {}),
                        "purpose": _text(purpose, "purpose", required=False),
                        "scope": scope_label,
                    },
                )
                maintenance = MaintenanceLease(
                    lease_id=lease_id,
                    scope=scope_label,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    purpose=_text(purpose, "purpose", required=False),
                    revision=1,
                    body=dict(body or {}),
                )
                self._insert_lease_locked(connection, fenced)
                connection.execute(
                    """
                    INSERT INTO maintenance_leases(
                        lease_id, scope, owner_session_id, fencing_token,
                        fence_epoch, acquired_at_ms, expires_at_ms, released_at_ms,
                        state, revision, purpose, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        maintenance.lease_id,
                        maintenance.scope,
                        maintenance.owner_session_id,
                        int(maintenance.fencing_token),
                        int(maintenance.fence_epoch),
                        int(maintenance.acquired_at_ms),
                        int(maintenance.expires_at_ms),
                        None,
                        maintenance.state.value,
                        int(maintenance.revision),
                        maintenance.purpose,
                        _canonical_json(dict(maintenance.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return maintenance
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- renew / release / protected write -----------------------------------

    def renew_lease(
        self,
        lease_id: str,
        *,
        owner_session_id: str,
        fencing_token: int,
        fence_epoch: int | None = None,
        requested_lease_ms: int | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Renew an accepted lease. Expired session/lease cannot renew."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        duration = _lease_duration_ms(
            requested_lease_ms, default=self._default_lease_ms
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                session = self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                lease = self._require_current_lease_locked(
                    connection,
                    lease_id,
                    owner_session_id=session.session_id,
                    fencing_token=fencing_token,
                    fence_epoch=fence_epoch,
                    now=now,
                )
                expires = now + duration
                revision = int(lease.revision) + 1
                connection.execute(
                    """
                    UPDATE fenced_leases
                    SET expires_at_ms = ?, revision = ?
                    WHERE lease_id = ? AND fencing_token = ? AND state = ?
                    """,
                    [
                        expires,
                        revision,
                        lease.lease_id,
                        int(lease.fencing_token),
                        LeaseState.ACCEPTED.value,
                    ],
                )
                if lease.kind is LeaseKind.TASK and lease.claim_id:
                    connection.execute(
                        """
                        UPDATE task_claims
                        SET expires_at_ms = ?, revision = revision + 1
                        WHERE claim_id = ? AND fencing_token = ? AND state = ?
                        """,
                        [
                            expires,
                            lease.claim_id,
                            int(lease.fencing_token),
                            LeaseState.ACCEPTED.value,
                        ],
                    )
                if lease.kind is LeaseKind.MAINTENANCE:
                    connection.execute(
                        """
                        UPDATE maintenance_leases
                        SET expires_at_ms = ?, revision = revision + 1
                        WHERE lease_id = ? AND fencing_token = ? AND state = ?
                        """,
                        [
                            expires,
                            lease.lease_id,
                            int(lease.fencing_token),
                            LeaseState.ACCEPTED.value,
                        ],
                    )
                if lease.kind in {
                    LeaseKind.RESOURCE,
                    LeaseKind.PATH,
                    LeaseKind.PROVIDER,
                    LeaseKind.PROVER,
                } and lease.claim_id:
                    connection.execute(
                        """
                        UPDATE resource_claims
                        SET expires_at_ms = ?, revision = revision + 1
                        WHERE claim_id = ? AND fencing_token = ? AND state = ?
                        """,
                        [
                            expires,
                            lease.claim_id,
                            int(lease.fencing_token),
                            LeaseState.ACCEPTED.value,
                        ],
                    )
                self._commit_if_idle(connection)
                return FencedLease(
                    lease_id=lease.lease_id,
                    kind=lease.kind,
                    scope_key=lease.scope_key,
                    scope_mode=lease.scope_mode,
                    owner_session_id=lease.owner_session_id,
                    fencing_token=lease.fencing_token,
                    fence_epoch=lease.fence_epoch,
                    acquired_at_ms=lease.acquired_at_ms,
                    expires_at_ms=expires,
                    state=LeaseState.ACCEPTED,
                    task_cid=lease.task_cid,
                    worktree_id=lease.worktree_id,
                    claim_id=lease.claim_id,
                    attempt_id=lease.attempt_id,
                    capacity_units=lease.capacity_units,
                    resource_kind=lease.resource_kind,
                    resource_id=lease.resource_id,
                    revision=revision,
                    body=dict(lease.body),
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def release_lease(
        self,
        lease_id: str,
        *,
        owner_session_id: str,
        fencing_token: int,
        fence_epoch: int | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Release an accepted lease under the current fencing token."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                lease = self._require_current_lease_locked(
                    connection,
                    lease_id,
                    owner_session_id=owner_session_id,
                    fencing_token=fencing_token,
                    fence_epoch=fence_epoch,
                    now=now,
                    require_active_session=False,
                )
                released = self._mark_lease_released_locked(connection, lease, now)
                self._commit_if_idle(connection)
                return released
            except Exception:
                self._rollback_if_open(connection)
                raise

    def protected_write(
        self,
        *,
        lease_id: str,
        owner_session_id: str,
        fencing_token: int,
        fence_epoch: int,
        write_kind: str,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> ProtectedWrite:
        """Commit a protected mutation only when fence epoch + token match.

        Stale fencing epochs are rejected for every protected write.
        """

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                lease = self._require_current_lease_locked(
                    connection,
                    lease_id,
                    owner_session_id=owner_session_id,
                    fencing_token=fencing_token,
                    fence_epoch=fence_epoch,
                    now=now,
                )
                # Explicit epoch CAS: reject any stale epoch even if token
                # somehow matched (defense in depth / takeover safety).
                if int(lease.fence_epoch) != int(fence_epoch):
                    raise DatabaseCoordinationFenceError(
                        f"stale fencing epoch: expected {lease.fence_epoch}, "
                        f"got {fence_epoch}"
                    )
                if int(lease.fencing_token) != int(fencing_token):
                    raise DatabaseCoordinationFenceError(
                        f"stale fencing token: expected {lease.fencing_token}, "
                        f"got {fencing_token}"
                    )
                write = ProtectedWrite(
                    write_id=_new_id("write"),
                    lease_id=lease.lease_id,
                    owner_session_id=owner_session_id,
                    fencing_token=int(fencing_token),
                    fence_epoch=int(fence_epoch),
                    write_kind=_text(write_kind, "write_kind"),
                    recorded_at_ms=now,
                    body=_bounded_mapping(body, name="body"),
                )
                connection.execute(
                    """
                    INSERT INTO protected_writes(
                        write_id, lease_id, owner_session_id, fencing_token,
                        fence_epoch, write_kind, recorded_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        write.write_id,
                        write.lease_id,
                        write.owner_session_id,
                        int(write.fencing_token),
                        int(write.fence_epoch),
                        write.write_kind,
                        int(write.recorded_at_ms),
                        _canonical_json(dict(write.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return write
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_lease(self, lease_id: str) -> FencedLease | None:
        with self._lock:
            connection = self._require()
            return self._load_lease_locked(connection, lease_id)

    def list_active_leases(
        self,
        *,
        kind: LeaseKind | str | None = None,
        scope_key: str | None = None,
    ) -> list[FencedLease]:
        with self._lock:
            connection = self._require()
            now = self._now_ms()
            self._begin(connection)
            try:
                self._expire_all_locked(connection, now)
                clauses = ["state = ?"]
                params: list[Any] = [LeaseState.ACCEPTED.value]
                if kind is not None:
                    kind_value = (
                        kind
                        if isinstance(kind, LeaseKind)
                        else _parse_enum(kind, LeaseKind, "kind")
                    )
                    assert isinstance(kind_value, LeaseKind)
                    clauses.append("kind = ?")
                    params.append(kind_value.value)
                if scope_key is not None:
                    clauses.append("scope_key = ?")
                    params.append(_text(scope_key, "scope_key"))
                sql = (
                    "SELECT * FROM fenced_leases WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY acquired_at_ms, lease_id"
                )
                rows = connection.execute(sql, params).fetchall()
                self._commit_if_idle(connection)
                return [self._lease_from_row(_row_mapping(row)) for row in rows]
            except Exception:
                self._rollback_if_open(connection)
                raise

    def active_owner_for_scope(self, scope_key: str) -> FencedLease | None:
        """Return the single accepted exclusive owner for a scope, if any."""

        with self._lock:
            connection = self._require()
            now = self._now_ms()
            self._begin(connection)
            try:
                self._expire_scope_locked(
                    connection, _text(scope_key, "scope_key"), now
                )
                active = self._active_exclusive_owner_locked(
                    connection, scope_key, now
                )
                self._commit_if_idle(connection)
                if active is None:
                    return None
                return self._lease_from_row(active)
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- fair / concurrent append scheduling ---------------------------------

    def enqueue_fair(
        self,
        *,
        queue_name: str,
        task_cid: str,
        owner_session_id: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> FairScheduleEntry:
        """Append one fair-schedule entry. Concurrent appends never exclusive-lock."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        queue = _text(queue_name, "queue_name")
        if len(queue.encode("utf-8")) > MAX_FAIR_QUEUE_NAME_BYTES:
            raise DatabaseCoordinationBoundsError("queue_name exceeds bound")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                if owner_session_id:
                    self._require_active_session_locked(
                        connection, owner_session_id, now=now
                    )
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(ordinal), 0) AS max_ordinal
                    FROM fair_schedule_entries WHERE queue_name = ?
                    """,
                    [queue],
                ).fetchone()
                mapping = _row_mapping(row)
                next_ordinal = int(mapping.get("max_ordinal") or 0) + 1
                entry = FairScheduleEntry(
                    entry_id=_new_id("fair"),
                    queue_name=queue,
                    task_cid=_text(task_cid, "task_cid"),
                    enqueued_at_ms=now,
                    ordinal=next_ordinal,
                    state=FairEntryState.QUEUED,
                    owner_session_id=_text(
                        owner_session_id, "owner_session_id", required=False
                    ),
                    body=_bounded_mapping(body, name="body"),
                )
                connection.execute(
                    """
                    INSERT INTO fair_schedule_entries(
                        entry_id, queue_name, task_cid, owner_session_id,
                        enqueued_at_ms, ordinal, state, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        entry.entry_id,
                        entry.queue_name,
                        entry.task_cid,
                        entry.owner_session_id,
                        int(entry.enqueued_at_ms),
                        int(entry.ordinal),
                        entry.state.value,
                        _canonical_json(dict(entry.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return entry
            except Exception:
                self._rollback_if_open(connection)
                raise

    def list_fair_queue(
        self,
        queue_name: str,
        *,
        states: Sequence[FairEntryState | str] | None = None,
        limit: int = 256,
    ) -> list[FairScheduleEntry]:
        """List fair-schedule entries ordered by ordinal (fair FIFO)."""

        queue = _text(queue_name, "queue_name")
        bound = _positive_int(int(limit), "limit")
        wanted: list[str]
        if states is None:
            wanted = [FairEntryState.QUEUED.value]
        else:
            wanted = [
                (
                    state.value
                    if isinstance(state, FairEntryState)
                    else _parse_enum(state, FairEntryState, "state").value  # type: ignore[union-attr]
                )
                for state in states
            ]
        with self._lock:
            connection = self._require()
            placeholders = ", ".join("?" for _ in wanted) or "?"
            params: list[Any] = [queue, *wanted, bound]
            rows = connection.execute(
                f"""
                SELECT * FROM fair_schedule_entries
                WHERE queue_name = ? AND state IN ({placeholders})
                ORDER BY ordinal, enqueued_at_ms
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [self._fair_from_row(_row_mapping(row)) for row in rows]

    def claim_next_fair(
        self,
        queue_name: str,
        *,
        owner_session_id: str,
        now_ms: int | None = None,
    ) -> FairScheduleEntry | None:
        """Atomically claim the oldest queued fair entry (concurrent-safe)."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        queue = _text(queue_name, "queue_name")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                row = connection.execute(
                    """
                    SELECT * FROM fair_schedule_entries
                    WHERE queue_name = ? AND state = ?
                    ORDER BY ordinal, enqueued_at_ms
                    LIMIT 1
                    """,
                    [queue, FairEntryState.QUEUED.value],
                ).fetchone()
                if row is None:
                    self._commit_if_idle(connection)
                    return None
                mapping = _row_mapping(row)
                entry_id = str(mapping["entry_id"])
                connection.execute(
                    """
                    UPDATE fair_schedule_entries
                    SET state = ?, owner_session_id = ?
                    WHERE entry_id = ? AND state = ?
                    """,
                    [
                        FairEntryState.CLAIMED.value,
                        _text(owner_session_id, "owner_session_id"),
                        entry_id,
                        FairEntryState.QUEUED.value,
                    ],
                )
                claimed = connection.execute(
                    "SELECT * FROM fair_schedule_entries WHERE entry_id = ?",
                    [entry_id],
                ).fetchone()
                self._commit_if_idle(connection)
                assert claimed is not None
                return self._fair_from_row(_row_mapping(claimed))
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- takeover after expiry -----------------------------------------------

    def takeover_expired(
        self,
        *,
        scope_key: str,
        owner_session_id: str,
        kind: LeaseKind | str | None = None,
        requested_lease_ms: int | None = None,
        now_ms: int | None = None,
    ) -> FencedLease:
        """Take over an expired exclusive scope with a higher fence epoch/token."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        duration = _lease_duration_ms(
            requested_lease_ms, default=self._default_lease_ms
        )
        scope = _text(scope_key, "scope_key")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                session = self._require_active_session_locked(
                    connection, owner_session_id, now=now
                )
                self._expire_scope_locked(connection, scope, now)
                active = self._active_exclusive_owner_locked(connection, scope, now)
                if active is not None:
                    raise DatabaseCoordinationConflictError(
                        f"scope {scope} still has an active owner "
                        f"{active.get('owner_session_id')}"
                    )
                prior = connection.execute(
                    """
                    SELECT kind FROM fenced_leases
                    WHERE scope_key = ?
                    ORDER BY fence_epoch DESC, fencing_token DESC
                    LIMIT 1
                    """,
                    [scope],
                ).fetchone()
                if prior is None and kind is None:
                    raise DatabaseCoordinationError(
                        f"no prior lease for scope {scope}; use claim/acquire first"
                    )
                prior_kind = (
                    kind
                    if kind is not None
                    else str(_row_mapping(prior).get("kind") or LeaseKind.TASK.value)
                )
                lease_kind = (
                    prior_kind
                    if isinstance(prior_kind, LeaseKind)
                    else _parse_enum(prior_kind, LeaseKind, "kind")
                )
                assert isinstance(lease_kind, LeaseKind)
                epoch, token = self._next_fence_locked(
                    connection, scope, kind=lease_kind, now=now
                )
                lease = FencedLease(
                    lease_id=_new_id("lease"),
                    kind=lease_kind,
                    scope_key=scope,
                    scope_mode=ScopeMode.EXCLUSIVE,
                    owner_session_id=session.session_id,
                    fencing_token=token,
                    fence_epoch=epoch,
                    acquired_at_ms=now,
                    expires_at_ms=now + duration,
                    state=LeaseState.ACCEPTED,
                    revision=1,
                    body={"takeover": True},
                )
                self._insert_lease_locked(connection, lease)
                self._commit_if_idle(connection)
                return lease
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- internal: fence / expiry --------------------------------------------

    def _next_fence_locked(
        self,
        connection: Any,
        scope_key: str,
        *,
        kind: LeaseKind,
        now: int,
    ) -> tuple[int, int]:
        """Advance fence epoch and token for a scope (LeaseCoordinator algorithm)."""

        row = connection.execute(
            """
            SELECT fence_epoch, fencing_token FROM scope_epochs
            WHERE scope_key = ?
            """,
            [scope_key],
        ).fetchone()
        if row is None:
            epoch, token = 1, 1
            connection.execute(
                """
                INSERT INTO scope_epochs(
                    scope_key, kind, fence_epoch, fencing_token,
                    last_owner_session_id, updated_at_ms
                ) VALUES (?, ?, ?, ?, '', ?)
                """,
                [scope_key, kind.value, epoch, token, now],
            )
            return epoch, token
        mapping = _row_mapping(row)
        epoch = int(mapping.get("fence_epoch") or 0) + 1
        token = int(mapping.get("fencing_token") or 0) + 1
        connection.execute(
            """
            UPDATE scope_epochs
            SET fence_epoch = ?, fencing_token = ?, kind = ?, updated_at_ms = ?
            WHERE scope_key = ?
            """,
            [epoch, token, kind.value, now, scope_key],
        )
        return epoch, token

    def _active_exclusive_owner_locked(
        self,
        connection: Any,
        scope_key: str,
        now: int,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT * FROM fenced_leases
            WHERE scope_key = ? AND state = ? AND scope_mode = ?
              AND expires_at_ms > ?
            ORDER BY fence_epoch DESC, fencing_token DESC
            LIMIT 1
            """,
            [
                scope_key,
                LeaseState.ACCEPTED.value,
                ScopeMode.EXCLUSIVE.value,
                now,
            ],
        ).fetchone()
        if row is None:
            return None
        return _row_mapping(row)

    def _expire_scope_locked(
        self, connection: Any, scope_key: str, now: int
    ) -> None:
        rows = connection.execute(
            """
            SELECT lease_id, claim_id, kind, fencing_token FROM fenced_leases
            WHERE scope_key = ? AND state = ? AND expires_at_ms <= ?
            """,
            [scope_key, LeaseState.ACCEPTED.value, now],
        ).fetchall()
        for row in rows:
            mapping = _row_mapping(row)
            self._expire_lease_row_locked(connection, mapping, now)

    def _expire_all_locked(self, connection: Any, now: int) -> None:
        self._expire_sessions_locked(connection, now)
        rows = connection.execute(
            """
            SELECT lease_id, claim_id, kind, fencing_token FROM fenced_leases
            WHERE state = ? AND expires_at_ms <= ?
            """,
            [LeaseState.ACCEPTED.value, now],
        ).fetchall()
        for row in rows:
            self._expire_lease_row_locked(connection, _row_mapping(row), now)

    def _expire_lease_row_locked(
        self, connection: Any, mapping: Mapping[str, Any], now: int
    ) -> None:
        lease_id = str(mapping["lease_id"])
        connection.execute(
            """
            UPDATE fenced_leases
            SET state = ?, released_at_ms = ?, revision = revision + 1
            WHERE lease_id = ? AND state = ?
            """,
            [LeaseState.EXPIRED.value, now, lease_id, LeaseState.ACCEPTED.value],
        )
        claim_id = str(mapping.get("claim_id") or "")
        kind = str(mapping.get("kind") or "")
        if claim_id and kind == LeaseKind.TASK.value:
            connection.execute(
                """
                UPDATE task_claims
                SET state = ?, released_at_ms = ?, revision = revision + 1
                WHERE claim_id = ? AND state = ?
                """,
                [
                    LeaseState.EXPIRED.value,
                    now,
                    claim_id,
                    LeaseState.ACCEPTED.value,
                ],
            )
        if claim_id and kind in {
            LeaseKind.RESOURCE.value,
            LeaseKind.PATH.value,
            LeaseKind.PROVIDER.value,
            LeaseKind.PROVER.value,
        }:
            connection.execute(
                """
                UPDATE resource_claims
                SET state = ?, released_at_ms = ?, revision = revision + 1
                WHERE claim_id = ? AND state = ?
                """,
                [
                    LeaseState.EXPIRED.value,
                    now,
                    claim_id,
                    LeaseState.ACCEPTED.value,
                ],
            )
        if kind == LeaseKind.MAINTENANCE.value:
            connection.execute(
                """
                UPDATE maintenance_leases
                SET state = ?, released_at_ms = ?, revision = revision + 1
                WHERE lease_id = ? AND state = ?
                """,
                [
                    LeaseState.EXPIRED.value,
                    now,
                    lease_id,
                    LeaseState.ACCEPTED.value,
                ],
            )

    def _expire_sessions_locked(self, connection: Any, now: int) -> None:
        expired = connection.execute(
            """
            SELECT session_id FROM owner_sessions
            WHERE status = ? AND expires_at_ms <= ?
            """,
            [SessionStatus.ACTIVE.value, now],
        ).fetchall()
        for row in expired:
            session_id = str(_row_mapping(row)["session_id"])
            connection.execute(
                """
                UPDATE owner_sessions
                SET status = ?, revision = revision + 1
                WHERE session_id = ?
                """,
                [SessionStatus.EXPIRED.value, session_id],
            )
            self._release_session_leases_locked(connection, session_id, now)

    def _release_session_leases_locked(
        self, connection: Any, session_id: str, now: int
    ) -> None:
        rows = connection.execute(
            """
            SELECT lease_id, claim_id, kind, fencing_token FROM fenced_leases
            WHERE owner_session_id = ? AND state = ?
            """,
            [session_id, LeaseState.ACCEPTED.value],
        ).fetchall()
        for row in rows:
            self._expire_lease_row_locked(connection, _row_mapping(row), now)

    def _require_active_session_locked(
        self,
        connection: Any,
        session_id: str,
        *,
        now: int,
        expected_fencing_token: int | None = None,
    ) -> OwnerSession:
        self._expire_sessions_locked(connection, now)
        session = self._load_session_locked(connection, session_id)
        if session is None:
            raise DatabaseCoordinationError(f"unknown session: {session_id}")
        if session.status is not SessionStatus.ACTIVE or session.expires_at_ms <= now:
            raise DatabaseCoordinationExpiredError(
                f"session is expired or stopped: {session_id}"
            )
        if expected_fencing_token is not None and int(expected_fencing_token) != int(
            session.fencing_token
        ):
            raise DatabaseCoordinationFenceError(
                "session fencing token mismatch"
            )
        return session

    def _require_current_lease_locked(
        self,
        connection: Any,
        lease_id: str,
        *,
        owner_session_id: str,
        fencing_token: int,
        fence_epoch: int | None,
        now: int,
        require_active_session: bool = True,
    ) -> FencedLease:
        if require_active_session:
            self._require_active_session_locked(
                connection, owner_session_id, now=now
            )
        else:
            # Still expire scopes so expiry is observed.
            pass
        self._expire_all_locked(connection, now)
        lease = self._load_lease_locked(connection, lease_id)
        if lease is None:
            raise DatabaseCoordinationError(f"unknown lease: {lease_id}")
        if lease.state is not LeaseState.ACCEPTED or lease.expires_at_ms <= now:
            raise DatabaseCoordinationExpiredError(
                "lease has expired or was released"
            )
        if lease.owner_session_id != owner_session_id:
            raise DatabaseCoordinationFenceError(
                "lease is no longer owned by this session"
            )
        if int(lease.fencing_token) != int(fencing_token):
            raise DatabaseCoordinationFenceError("fencing token is stale")
        if fence_epoch is not None and int(lease.fence_epoch) != int(fence_epoch):
            raise DatabaseCoordinationFenceError("fencing epoch is stale")
        return lease

    def _mark_lease_released_locked(
        self, connection: Any, lease: FencedLease, now: int
    ) -> FencedLease:
        connection.execute(
            """
            UPDATE fenced_leases
            SET state = ?, released_at_ms = ?, revision = revision + 1
            WHERE lease_id = ? AND fencing_token = ? AND state = ?
            """,
            [
                LeaseState.RELEASED.value,
                now,
                lease.lease_id,
                int(lease.fencing_token),
                LeaseState.ACCEPTED.value,
            ],
        )
        if lease.claim_id and lease.kind is LeaseKind.TASK:
            connection.execute(
                """
                UPDATE task_claims
                SET state = ?, released_at_ms = ?, revision = revision + 1
                WHERE claim_id = ? AND fencing_token = ? AND state = ?
                """,
                [
                    LeaseState.RELEASED.value,
                    now,
                    lease.claim_id,
                    int(lease.fencing_token),
                    LeaseState.ACCEPTED.value,
                ],
            )
        if lease.claim_id and lease.kind in {
            LeaseKind.RESOURCE,
            LeaseKind.PATH,
            LeaseKind.PROVIDER,
            LeaseKind.PROVER,
        }:
            connection.execute(
                """
                UPDATE resource_claims
                SET state = ?, released_at_ms = ?, revision = revision + 1
                WHERE claim_id = ? AND fencing_token = ? AND state = ?
                """,
                [
                    LeaseState.RELEASED.value,
                    now,
                    lease.claim_id,
                    int(lease.fencing_token),
                    LeaseState.ACCEPTED.value,
                ],
            )
        if lease.kind is LeaseKind.MAINTENANCE:
            connection.execute(
                """
                UPDATE maintenance_leases
                SET state = ?, released_at_ms = ?, revision = revision + 1
                WHERE lease_id = ? AND fencing_token = ? AND state = ?
                """,
                [
                    LeaseState.RELEASED.value,
                    now,
                    lease.lease_id,
                    int(lease.fencing_token),
                    LeaseState.ACCEPTED.value,
                ],
            )
        return FencedLease(
            lease_id=lease.lease_id,
            kind=lease.kind,
            scope_key=lease.scope_key,
            scope_mode=lease.scope_mode,
            owner_session_id=lease.owner_session_id,
            fencing_token=lease.fencing_token,
            fence_epoch=lease.fence_epoch,
            acquired_at_ms=lease.acquired_at_ms,
            expires_at_ms=lease.expires_at_ms,
            state=LeaseState.RELEASED,
            task_cid=lease.task_cid,
            worktree_id=lease.worktree_id,
            claim_id=lease.claim_id,
            attempt_id=lease.attempt_id,
            capacity_units=lease.capacity_units,
            resource_kind=lease.resource_kind,
            resource_id=lease.resource_id,
            revision=int(lease.revision) + 1,
            released_at_ms=now,
            body=dict(lease.body),
        )

    def _insert_lease_locked(self, connection: Any, lease: FencedLease) -> None:
        connection.execute(
            """
            INSERT INTO fenced_leases(
                lease_id, kind, scope_key, scope_mode, owner_session_id,
                task_cid, worktree_id, claim_id, attempt_id, fencing_token,
                fence_epoch, acquired_at_ms, expires_at_ms, released_at_ms,
                state, revision, capacity_units, resource_kind, resource_id,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                lease.lease_id,
                lease.kind.value,
                lease.scope_key,
                lease.scope_mode.value,
                lease.owner_session_id,
                lease.task_cid,
                lease.worktree_id,
                lease.claim_id,
                lease.attempt_id,
                int(lease.fencing_token),
                int(lease.fence_epoch),
                int(lease.acquired_at_ms),
                int(lease.expires_at_ms),
                lease.released_at_ms,
                lease.state.value,
                int(lease.revision),
                int(lease.capacity_units),
                lease.resource_kind,
                lease.resource_id,
                _canonical_json(dict(lease.body)),
            ],
        )
        connection.execute(
            """
            UPDATE scope_epochs
            SET last_owner_session_id = ?, updated_at_ms = ?
            WHERE scope_key = ?
            """,
            [lease.owner_session_id, int(lease.acquired_at_ms), lease.scope_key],
        )

    def _next_attempt_number_locked(self, connection: Any, task_cid: str) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(attempt_number), 0) AS max_attempt
            FROM task_attempts WHERE task_cid = ?
            """,
            [task_cid],
        ).fetchone()
        return int(_row_mapping(row).get("max_attempt") or 0) + 1

    # -- row loaders ---------------------------------------------------------

    def _load_session_locked(
        self, connection: Any, session_id: str
    ) -> OwnerSession | None:
        row = connection.execute(
            "SELECT * FROM owner_sessions WHERE session_id = ?",
            [_text(session_id, "session_id")],
        ).fetchone()
        if row is None:
            return None
        return self._session_from_row(_row_mapping(row))

    def _load_lease_locked(
        self, connection: Any, lease_id: str
    ) -> FencedLease | None:
        row = connection.execute(
            "SELECT * FROM fenced_leases WHERE lease_id = ?",
            [_text(lease_id, "lease_id")],
        ).fetchone()
        if row is None:
            return None
        return self._lease_from_row(_row_mapping(row))

    def _load_attempt_locked(
        self, connection: Any, attempt_id: str
    ) -> TaskAttempt | None:
        row = connection.execute(
            "SELECT * FROM task_attempts WHERE attempt_id = ?",
            [_text(attempt_id, "attempt_id")],
        ).fetchone()
        if row is None:
            return None
        return self._attempt_from_row(_row_mapping(row))

    @staticmethod
    def _session_from_row(mapping: Mapping[str, Any]) -> OwnerSession:
        body_raw = mapping.get("body_json") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else dict(body_raw or {})
        return OwnerSession(
            session_id=str(mapping["session_id"]),
            owner_did=str(mapping["owner_did"]),
            process_birth_id=str(mapping.get("process_birth_id") or ""),
            fence_epoch=int(mapping.get("fence_epoch") or 1),
            fencing_token=int(mapping.get("fencing_token") or 1),
            attached_at_ms=int(mapping.get("attached_at_ms") or 0),
            expires_at_ms=int(mapping.get("expires_at_ms") or 0),
            status=SessionStatus(str(mapping.get("status") or SessionStatus.ACTIVE.value)),
            revision=int(mapping.get("revision") or 1),
            body=body if isinstance(body, Mapping) else {},
        )

    @staticmethod
    def _lease_from_row(mapping: Mapping[str, Any]) -> FencedLease:
        body_raw = mapping.get("body_json") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else dict(body_raw or {})
        released = mapping.get("released_at_ms")
        return FencedLease(
            lease_id=str(mapping["lease_id"]),
            kind=LeaseKind(str(mapping["kind"])),
            scope_key=str(mapping["scope_key"]),
            scope_mode=ScopeMode(str(mapping["scope_mode"])),
            owner_session_id=str(mapping["owner_session_id"]),
            fencing_token=int(mapping["fencing_token"]),
            fence_epoch=int(mapping["fence_epoch"]),
            acquired_at_ms=int(mapping["acquired_at_ms"]),
            expires_at_ms=int(mapping["expires_at_ms"]),
            state=LeaseState(str(mapping["state"])),
            task_cid=str(mapping.get("task_cid") or ""),
            worktree_id=str(mapping.get("worktree_id") or ""),
            claim_id=str(mapping.get("claim_id") or ""),
            attempt_id=str(mapping.get("attempt_id") or ""),
            capacity_units=int(mapping.get("capacity_units") or 1),
            resource_kind=str(mapping.get("resource_kind") or ""),
            resource_id=str(mapping.get("resource_id") or ""),
            revision=int(mapping.get("revision") or 1),
            released_at_ms=None if released is None else int(released),
            body=body if isinstance(body, Mapping) else {},
        )

    @staticmethod
    def _task_claim_from_row(mapping: Mapping[str, Any]) -> TaskClaim:
        body_raw = mapping.get("body_json") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else dict(body_raw or {})
        return TaskClaim(
            claim_id=str(mapping["claim_id"]),
            task_cid=str(mapping["task_cid"]),
            owner_session_id=str(mapping["owner_session_id"]),
            lease_id=str(mapping["lease_id"]),
            fencing_token=int(mapping["fencing_token"]),
            fence_epoch=int(mapping["fence_epoch"]),
            claimed_at_ms=int(mapping["claimed_at_ms"]),
            expires_at_ms=int(mapping["expires_at_ms"]),
            attempt_id=str(mapping["attempt_id"]),
            attempt_number=int(mapping["attempt_number"]),
            state=LeaseState(str(mapping["state"])),
            revision=int(mapping.get("revision") or 1),
            idempotency_key=str(mapping.get("idempotency_key") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    @staticmethod
    def _attempt_from_row(mapping: Mapping[str, Any]) -> TaskAttempt:
        body_raw = mapping.get("body_json") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else dict(body_raw or {})
        finished = mapping.get("finished_at_ms")
        return TaskAttempt(
            attempt_id=str(mapping["attempt_id"]),
            task_cid=str(mapping["task_cid"]),
            claim_id=str(mapping["claim_id"]),
            attempt_number=int(mapping["attempt_number"]),
            owner_session_id=str(mapping["owner_session_id"]),
            fencing_token=int(mapping["fencing_token"]),
            fence_epoch=int(mapping["fence_epoch"]),
            started_at_ms=int(mapping["started_at_ms"]),
            finished_at_ms=None if finished is None else int(finished),
            status=AttemptStatus(str(mapping.get("status") or AttemptStatus.RUNNING.value)),
            revision=int(mapping.get("revision") or 1),
            body=body if isinstance(body, Mapping) else {},
        )

    @staticmethod
    def _resource_claim_from_row(mapping: Mapping[str, Any]) -> ResourceClaim:
        body_raw = mapping.get("body_json") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else dict(body_raw or {})
        return ResourceClaim(
            claim_id=str(mapping["claim_id"]),
            resource_kind=str(mapping["resource_kind"]),
            resource_id=str(mapping["resource_id"]),
            owner_session_id=str(mapping["owner_session_id"]),
            lease_id=str(mapping["lease_id"]),
            fencing_token=int(mapping["fencing_token"]),
            fence_epoch=int(mapping["fence_epoch"]),
            acquired_at_ms=int(mapping["acquired_at_ms"]),
            expires_at_ms=int(mapping["expires_at_ms"]),
            state=LeaseState(str(mapping["state"])),
            task_cid=str(mapping.get("task_cid") or ""),
            capacity_units=int(mapping.get("capacity_units") or 1),
            revision=int(mapping.get("revision") or 1),
            body=body if isinstance(body, Mapping) else {},
        )

    @staticmethod
    def _maintenance_from_row(mapping: Mapping[str, Any]) -> MaintenanceLease:
        body_raw = mapping.get("body_json") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else dict(body_raw or {})
        released = mapping.get("released_at_ms")
        return MaintenanceLease(
            lease_id=str(mapping["lease_id"]),
            scope=str(mapping["scope"]),
            owner_session_id=str(mapping["owner_session_id"]),
            fencing_token=int(mapping["fencing_token"]),
            fence_epoch=int(mapping["fence_epoch"]),
            acquired_at_ms=int(mapping["acquired_at_ms"]),
            expires_at_ms=int(mapping["expires_at_ms"]),
            state=LeaseState(str(mapping["state"])),
            purpose=str(mapping.get("purpose") or ""),
            revision=int(mapping.get("revision") or 1),
            released_at_ms=None if released is None else int(released),
            body=body if isinstance(body, Mapping) else {},
        )

    @staticmethod
    def _fair_from_row(mapping: Mapping[str, Any]) -> FairScheduleEntry:
        body_raw = mapping.get("body_json") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else dict(body_raw or {})
        return FairScheduleEntry(
            entry_id=str(mapping["entry_id"]),
            queue_name=str(mapping["queue_name"]),
            task_cid=str(mapping["task_cid"]),
            enqueued_at_ms=int(mapping["enqueued_at_ms"]),
            ordinal=int(mapping["ordinal"]),
            state=FairEntryState(str(mapping.get("state") or FairEntryState.QUEUED.value)),
            owner_session_id=str(mapping.get("owner_session_id") or ""),
            body=body if isinstance(body, Mapping) else {},
        )


def open_database_coordinator(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    default_lease_ms: int = DEFAULT_LEASE_MS,
    default_session_ttl_ms: int = DEFAULT_SESSION_TTL_MS,
) -> DatabaseCoordinator:
    """Open and return an initialized :class:`DatabaseCoordinator`."""

    return DatabaseCoordinator(
        database_path,
        clock_ms=clock_ms,
        default_lease_ms=default_lease_ms,
        default_session_ttl_ms=default_session_ttl_ms,
    ).open()


__all__ = (
    "DATABASE_COORDINATOR_INTERFACE",
    "DATABASE_COORDINATOR_SCHEMA",
    "DEFAULT_LEASE_MS",
    "DEFAULT_SESSION_TTL_MS",
    "FAIR_SCHEDULE_ENTRY_SCHEMA",
    "FENCED_LEASE_INTERFACE",
    "FENCED_LEASE_SCHEMA",
    "MAINTENANCE_LEASE_INTERFACE",
    "MAINTENANCE_LEASE_SCHEMA",
    "MAX_LEASE_MS",
    "MIN_LEASE_MS",
    "PROTECTED_WRITE_SCHEMA",
    "RESOURCE_CLAIM_INTERFACE",
    "RESOURCE_CLAIM_SCHEMA",
    "TASK_ATTEMPT_SCHEMA",
    "TASK_CLAIM_INTERFACE",
    "TASK_CLAIM_SCHEMA",
    "AttemptStatus",
    "DatabaseCoordinationBoundsError",
    "DatabaseCoordinationConflictError",
    "DatabaseCoordinationError",
    "DatabaseCoordinationExpiredError",
    "DatabaseCoordinationFenceError",
    "DatabaseCoordinationNotOpenError",
    "DatabaseCoordinator",
    "DuckDBUnavailableError",
    "FairEntryState",
    "FairScheduleEntry",
    "FencedLease",
    "LeaseKind",
    "LeaseState",
    "MaintenanceLease",
    "OwnerSession",
    "ProtectedWrite",
    "ResourceClaim",
    "ScopeMode",
    "SessionStatus",
    "TaskAttempt",
    "TaskClaim",
    "TaskClaimBundle",
    "duckdb_available",
    "exclusive_scope_key",
    "open_database_coordinator",
)
