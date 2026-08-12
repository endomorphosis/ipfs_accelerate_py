"""Register supervisors, daemons, sessions, process births, and heartbeats.

DQP-014 / SupervisorInstance@1, DaemonInstance@1, DaemonSession@1, Heartbeat@1
============================================================================

:class:`DaemonRegistry` is the durable authority for master/lane/supervisor/
daemon/worker session identity. Every active session binds run, role, shard,
process-birth identity, server generation, Quack connection, capability,
heartbeat, progress cursor, deadline, and exit/restart disposition.

Authority rules (fail-closed)
-----------------------------
* A raw PID never proves identity; multi-factor process birth is required.
* Dead, PID-reused, or unknown process births cannot renew or extend a session.
* Duplicate active role/lane ownership is fenced: only one live owner.
* Heartbeats and progress cursors are distinct records with distinct semantics.
* Status/PID files may mirror registry rows for legacy tooling but cannot create
  or extend a session.

Cold import of this module performs no filesystem, database, network, provider,
or process action.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    current_process_birth,
    owner_liveness,
    read_process_birth,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DAEMON_REGISTRY_INTERFACE: Final[str] = "DaemonRegistry@1"
SUPERVISOR_INSTANCE_INTERFACE: Final[str] = "SupervisorInstance@1"
DAEMON_INSTANCE_INTERFACE: Final[str] = "DaemonInstance@1"
DAEMON_SESSION_INTERFACE: Final[str] = "DaemonSession@1"
HEARTBEAT_INTERFACE: Final[str] = "Heartbeat@1"

DAEMON_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/daemon-registry@1"
)
SUPERVISOR_INSTANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-instance@1"
)
DAEMON_INSTANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/daemon-instance@1"
)
DAEMON_SESSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/daemon-session@1"
)
HEARTBEAT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/daemon-heartbeat@1"
)
PROGRESS_CURSOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/progress-cursor@1"
)
STATUS_FILE_MIRROR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/status-file-mirror@1"
)

DEFAULT_HEARTBEAT_TTL_MS: Final[int] = 15_000
DEFAULT_SESSION_TTL_MS: Final[int] = 60_000
DEFAULT_HEARTBEAT_RETAIN: Final[int] = 32
MAX_HEARTBEAT_RETAIN: Final[int] = 4_096
MAX_CAPABILITY_BYTES: Final[int] = 65_536
MAX_PAYLOAD_BYTES: Final[int] = 262_144

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS registry_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS supervisor_instances (
    supervisor_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    process_birth_json VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL DEFAULT '',
    server_generation BIGINT NOT NULL DEFAULT 1,
    started_at_ms BIGINT NOT NULL,
    stopped_at_ms BIGINT,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    capability_json VARCHAR NOT NULL DEFAULT '{}',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS supervisor_instances_status_idx
    ON supervisor_instances(status, started_at_ms);

CREATE TABLE IF NOT EXISTS daemon_instances (
    daemon_id VARCHAR PRIMARY KEY,
    supervisor_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    process_birth_json VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    lane_id VARCHAR NOT NULL DEFAULT '',
    shard_id VARCHAR NOT NULL DEFAULT '',
    run_id VARCHAR NOT NULL DEFAULT '',
    started_at_ms BIGINT NOT NULL,
    stopped_at_ms BIGINT,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    capability_json VARCHAR NOT NULL DEFAULT '{}',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS daemon_instances_supervisor_idx
    ON daemon_instances(supervisor_id, status);
CREATE INDEX IF NOT EXISTS daemon_instances_role_lane_idx
    ON daemon_instances(role, lane_id, status);

CREATE TABLE IF NOT EXISTS daemon_sessions (
    session_id VARCHAR PRIMARY KEY,
    daemon_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    process_birth_json VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    lane_id VARCHAR NOT NULL DEFAULT '',
    shard_id VARCHAR NOT NULL DEFAULT '',
    run_id VARCHAR NOT NULL DEFAULT '',
    server_id VARCHAR NOT NULL DEFAULT '',
    server_generation BIGINT NOT NULL DEFAULT 1,
    fence_epoch BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL,
    quack_connection VARCHAR NOT NULL DEFAULT '',
    capability_json VARCHAR NOT NULL DEFAULT '{}',
    attached_at_ms BIGINT NOT NULL,
    last_heartbeat_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    progress_cursor VARCHAR NOT NULL DEFAULT '',
    progress_updated_at_ms BIGINT NOT NULL DEFAULT 0,
    deadline_ms BIGINT NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL,
    exit_disposition VARCHAR NOT NULL DEFAULT 'running',
    restart_disposition VARCHAR NOT NULL DEFAULT 'none',
    revision BIGINT NOT NULL,
    ancestry_json VARCHAR NOT NULL DEFAULT '[]',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS daemon_sessions_daemon_idx
    ON daemon_sessions(daemon_id, status);
CREATE INDEX IF NOT EXISTS daemon_sessions_role_lane_idx
    ON daemon_sessions(role, lane_id, status);
CREATE INDEX IF NOT EXISTS daemon_sessions_expiry_idx
    ON daemon_sessions(status, expires_at_ms);

CREATE TABLE IF NOT EXISTS session_heartbeats (
    heartbeat_cid VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    sequence BIGINT NOT NULL,
    payload_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS session_heartbeats_session_idx
    ON session_heartbeats(session_id, sequence);
CREATE INDEX IF NOT EXISTS session_heartbeats_observed_idx
    ON session_heartbeats(session_id, observed_at_ms);

CREATE TABLE IF NOT EXISTS progress_records (
    progress_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    cursor_value VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    sequence BIGINT NOT NULL,
    payload_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS progress_records_session_idx
    ON progress_records(session_id, sequence);

CREATE TABLE IF NOT EXISTS status_file_mirrors (
    mirror_path VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    written_at_ms BIGINT NOT NULL,
    content_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DaemonRegistryError(RuntimeError):
    """Base error for daemon-registry failures."""


class DaemonRegistryConflictError(DaemonRegistryError):
    """Duplicate ownership, fence mismatch, or identity conflict."""


class DaemonRegistryIdentityError(DaemonRegistryError):
    """Process-birth identity mismatch, dead, reused, or unknown."""


class DaemonRegistrySessionError(DaemonRegistryError):
    """Session missing, expired, stopped, or not renewable."""


class DaemonRegistryBoundsError(DaemonRegistryError, ValueError):
    """Payload or retention bound exceeded."""


class DaemonRegistryNotOpenError(DaemonRegistryError):
    """Operation requires an open registry."""


class DuckDBUnavailableError(DaemonRegistryError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class InstanceStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    DEAD = "dead"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    STOPPED = "stopped"
    SUPERSEDED = "superseded"


class ExitDisposition(str, Enum):
    RUNNING = "running"
    CLEAN = "clean"
    ERROR = "error"
    KILLED = "killed"


class RestartDisposition(str, Enum):
    NONE = "none"
    RESTART = "restart"
    FAILOVER = "failover"
    ABANDON = "abandon"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LivenessProbe = Callable[[ProcessBirthIdentity], OwnerLiveness]
BirthReader = Callable[[int], ProcessBirthIdentity | None]
ClockMs = Callable[[], int]


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso_from_ms(epoch_ms: int) -> str:
    return (
        datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _default_clock_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DaemonRegistryError(f"{name} contains NUL")
    if required and not text:
        raise DaemonRegistryError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DaemonRegistryBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DaemonRegistryBoundsError(f"{name} must be a positive integer")
    return value


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
    max_bytes: int,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise DaemonRegistryBoundsError(
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
    # Positional fallback for drivers that omit aliases on aggregates.
    try:
        return {str(index): row[index] for index in range(len(row))}  # type: ignore[arg-type]
    except Exception:
        return {}


def _row_int(mapping: Mapping[str, Any], *names: str, default: int = 0) -> int:
    """Read an integer column by preferred name, then case-fold, then position."""

    for name in names:
        if name in mapping and mapping[name] is not None:
            return int(mapping[name])
        upper = name.upper()
        if upper in mapping and mapping[upper] is not None:
            return int(mapping[upper])
        lower = name.lower()
        if lower in mapping and mapping[lower] is not None:
            return int(mapping[lower])
    # Case-insensitive scan.
    wanted = {name.lower() for name in names}
    for key, value in mapping.items():
        if str(key).lower() in wanted and value is not None:
            return int(value)
    # Positional fallback: first numeric-looking entry.
    for key in ("0", 0):
        if key in mapping and mapping[key] is not None:
            return int(mapping[key])
    return int(default)


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


def process_birth_id(birth: ProcessBirthIdentity) -> str:
    """Return a stable content id for multi-factor process birth.

    PID alone is never sufficient; start-time ticks, boot id, and parent pid
    participate so PID reuse yields a different identity.
    """

    if not isinstance(birth, ProcessBirthIdentity):
        raise TypeError("birth must be ProcessBirthIdentity")
    material = (
        f"{int(birth.pid)}:{int(birth.start_time_ticks)}:"
        f"{str(birth.boot_id or '')}:{int(birth.parent_pid or 0)}"
    )
    return f"birth:{_sha256_hex(material.encode('utf-8'))[7:39]}"


def process_birth_from_mapping(payload: Mapping[str, Any] | None) -> ProcessBirthIdentity:
    return ProcessBirthIdentity.from_dict(payload or {})


def process_births_match(
    expected: ProcessBirthIdentity,
    observed: ProcessBirthIdentity | None,
) -> bool:
    """Return whether observed birth is an exact multi-factor match."""

    if observed is None:
        return False
    if int(expected.pid) != int(observed.pid):
        return False
    if int(expected.start_time_ticks) and int(observed.start_time_ticks):
        if int(expected.start_time_ticks) != int(observed.start_time_ticks):
            return False
    elif int(expected.start_time_ticks) or int(observed.start_time_ticks):
        # One side lacks start ticks: refuse equality (PID alone is not enough).
        return False
    if expected.boot_id and observed.boot_id and expected.boot_id != observed.boot_id:
        return False
    return True


def _parse_status(value: str, enum_cls: type[Enum], name: str) -> Enum:
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise DaemonRegistryError(f"unknown {name}: {text}") from exc


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorInstance:
    """Registered supervisor process bound to multi-factor process birth."""

    INTERFACE: ClassVar[str] = SUPERVISOR_INSTANCE_INTERFACE
    SCHEMA: ClassVar[str] = SUPERVISOR_INSTANCE_SCHEMA

    supervisor_id: str
    repository_id: str
    process_birth: ProcessBirthIdentity
    process_birth_id: str
    run_id: str = ""
    server_generation: int = 1
    started_at_ms: int = 0
    stopped_at_ms: int | None = None
    status: InstanceStatus = InstanceStatus.RUNNING
    revision: int = 1
    capability: Mapping[str, Any] = field(default_factory=dict)
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "supervisor_id", _text(self.supervisor_id, "supervisor_id")
        )
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        if not isinstance(self.process_birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        object.__setattr__(
            self,
            "process_birth_id",
            _text(self.process_birth_id or process_birth_id(self.process_birth), "process_birth_id"),
        )
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", required=False))
        object.__setattr__(
            self,
            "server_generation",
            _positive_int(int(self.server_generation), "server_generation"),
        )
        object.__setattr__(
            self, "started_at_ms", _nonneg_int(int(self.started_at_ms), "started_at_ms")
        )
        if self.stopped_at_ms is not None:
            object.__setattr__(
                self,
                "stopped_at_ms",
                _nonneg_int(int(self.stopped_at_ms), "stopped_at_ms"),
            )
        status = self.status
        if not isinstance(status, InstanceStatus):
            status = _parse_status(str(status), InstanceStatus, "status")
            object.__setattr__(self, "status", status)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self,
            "capability",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.capability or {}),
                    name="capability",
                    max_bytes=MAX_CAPABILITY_BYTES,
                )
            ),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "supervisor_id": self.supervisor_id,
            "repository_id": self.repository_id,
            "process_birth": self.process_birth.to_dict(),
            "process_birth_id": self.process_birth_id,
            "run_id": self.run_id,
            "server_generation": int(self.server_generation),
            "started_at_ms": int(self.started_at_ms),
            "started_at": _utc_iso_from_ms(self.started_at_ms) if self.started_at_ms else "",
            "stopped_at_ms": self.stopped_at_ms,
            "status": self.status.value,
            "revision": int(self.revision),
            "capability": dict(self.capability),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class DaemonInstance:
    """Registered daemon under one supervisor, with role and optional lane."""

    INTERFACE: ClassVar[str] = DAEMON_INSTANCE_INTERFACE
    SCHEMA: ClassVar[str] = DAEMON_INSTANCE_SCHEMA

    daemon_id: str
    supervisor_id: str
    process_birth: ProcessBirthIdentity
    process_birth_id: str
    role: str
    lane_id: str = ""
    shard_id: str = ""
    run_id: str = ""
    started_at_ms: int = 0
    stopped_at_ms: int | None = None
    status: InstanceStatus = InstanceStatus.RUNNING
    revision: int = 1
    capability: Mapping[str, Any] = field(default_factory=dict)
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "daemon_id", _text(self.daemon_id, "daemon_id"))
        object.__setattr__(
            self, "supervisor_id", _text(self.supervisor_id, "supervisor_id")
        )
        if not isinstance(self.process_birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        object.__setattr__(
            self,
            "process_birth_id",
            _text(self.process_birth_id or process_birth_id(self.process_birth), "process_birth_id"),
        )
        object.__setattr__(self, "role", _text(self.role, "role"))
        object.__setattr__(
            self, "lane_id", _text(self.lane_id, "lane_id", required=False)
        )
        object.__setattr__(
            self, "shard_id", _text(self.shard_id, "shard_id", required=False)
        )
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", required=False))
        object.__setattr__(
            self, "started_at_ms", _nonneg_int(int(self.started_at_ms), "started_at_ms")
        )
        if self.stopped_at_ms is not None:
            object.__setattr__(
                self,
                "stopped_at_ms",
                _nonneg_int(int(self.stopped_at_ms), "stopped_at_ms"),
            )
        status = self.status
        if not isinstance(status, InstanceStatus):
            status = _parse_status(str(status), InstanceStatus, "status")
            object.__setattr__(self, "status", status)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self,
            "capability",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.capability or {}),
                    name="capability",
                    max_bytes=MAX_CAPABILITY_BYTES,
                )
            ),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "daemon_id": self.daemon_id,
            "supervisor_id": self.supervisor_id,
            "process_birth": self.process_birth.to_dict(),
            "process_birth_id": self.process_birth_id,
            "role": self.role,
            "lane_id": self.lane_id,
            "shard_id": self.shard_id,
            "run_id": self.run_id,
            "started_at_ms": int(self.started_at_ms),
            "started_at": _utc_iso_from_ms(self.started_at_ms) if self.started_at_ms else "",
            "stopped_at_ms": self.stopped_at_ms,
            "status": self.status.value,
            "revision": int(self.revision),
            "capability": dict(self.capability),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class ProgressCursor:
    """Task/work progress distinct from liveness heartbeats."""

    SCHEMA: ClassVar[str] = PROGRESS_CURSOR_SCHEMA

    session_id: str
    cursor_value: str
    recorded_at_ms: int
    sequence: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(
            self, "cursor_value", _text(self.cursor_value, "cursor_value")
        )
        object.__setattr__(
            self,
            "recorded_at_ms",
            _nonneg_int(int(self.recorded_at_ms), "recorded_at_ms"),
        )
        object.__setattr__(
            self, "sequence", _nonneg_int(int(self.sequence), "sequence")
        )
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.payload or {}),
                    name="payload",
                    max_bytes=MAX_PAYLOAD_BYTES,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "session_id": self.session_id,
            "cursor_value": self.cursor_value,
            "recorded_at_ms": int(self.recorded_at_ms),
            "sequence": int(self.sequence),
            "payload": dict(self.payload),
            # Explicit non-liveness marker: progress is not a heartbeat.
            "kind": "progress",
            "extends_session": False,
        }


@dataclass(frozen=True)
class Heartbeat:
    """Liveness heartbeat for one fenced session."""

    INTERFACE: ClassVar[str] = HEARTBEAT_INTERFACE
    SCHEMA: ClassVar[str] = HEARTBEAT_SCHEMA

    heartbeat_cid: str
    session_id: str
    fencing_token: int
    observed_at_ms: int
    expires_at_ms: int
    sequence: int
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "heartbeat_cid", _text(self.heartbeat_cid, "heartbeat_cid")
        )
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(
            self,
            "fencing_token",
            _positive_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonneg_int(int(self.observed_at_ms), "observed_at_ms"),
        )
        object.__setattr__(
            self, "expires_at_ms", _nonneg_int(int(self.expires_at_ms), "expires_at_ms")
        )
        object.__setattr__(
            self, "sequence", _positive_int(int(self.sequence), "sequence")
        )
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.payload or {}),
                    name="payload",
                    max_bytes=MAX_PAYLOAD_BYTES,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "heartbeat_cid": self.heartbeat_cid,
            "session_id": self.session_id,
            "fencing_token": int(self.fencing_token),
            "observed_at_ms": int(self.observed_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "sequence": int(self.sequence),
            "payload": dict(self.payload),
            "kind": "heartbeat",
            "is_progress": False,
        }


@dataclass(frozen=True)
class DaemonSession:
    """Active or historical fenced daemon/worker session."""

    INTERFACE: ClassVar[str] = DAEMON_SESSION_INTERFACE
    SCHEMA: ClassVar[str] = DAEMON_SESSION_SCHEMA

    session_id: str
    daemon_id: str
    supervisor_id: str
    process_birth: ProcessBirthIdentity
    process_birth_id: str
    role: str
    lane_id: str = ""
    shard_id: str = ""
    run_id: str = ""
    server_id: str = ""
    server_generation: int = 1
    fence_epoch: int = 1
    fencing_token: int = 1
    quack_connection: str = ""
    capability: Mapping[str, Any] = field(default_factory=dict)
    attached_at_ms: int = 0
    last_heartbeat_at_ms: int = 0
    expires_at_ms: int = 0
    progress_cursor: str = ""
    progress_updated_at_ms: int = 0
    deadline_ms: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    exit_disposition: ExitDisposition = ExitDisposition.RUNNING
    restart_disposition: RestartDisposition = RestartDisposition.NONE
    revision: int = 1
    ancestry: Sequence[str] = field(default_factory=tuple)
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(self, "daemon_id", _text(self.daemon_id, "daemon_id"))
        object.__setattr__(
            self, "supervisor_id", _text(self.supervisor_id, "supervisor_id")
        )
        if not isinstance(self.process_birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        object.__setattr__(
            self,
            "process_birth_id",
            _text(
                self.process_birth_id or process_birth_id(self.process_birth),
                "process_birth_id",
            ),
        )
        object.__setattr__(self, "role", _text(self.role, "role"))
        object.__setattr__(
            self, "lane_id", _text(self.lane_id, "lane_id", required=False)
        )
        object.__setattr__(
            self, "shard_id", _text(self.shard_id, "shard_id", required=False)
        )
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", required=False))
        object.__setattr__(
            self, "server_id", _text(self.server_id, "server_id", required=False)
        )
        object.__setattr__(
            self,
            "server_generation",
            _positive_int(int(self.server_generation), "server_generation"),
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
            "quack_connection",
            _text(self.quack_connection, "quack_connection", required=False),
        )
        object.__setattr__(
            self, "attached_at_ms", _nonneg_int(int(self.attached_at_ms), "attached_at_ms")
        )
        object.__setattr__(
            self,
            "last_heartbeat_at_ms",
            _nonneg_int(int(self.last_heartbeat_at_ms), "last_heartbeat_at_ms"),
        )
        object.__setattr__(
            self, "expires_at_ms", _nonneg_int(int(self.expires_at_ms), "expires_at_ms")
        )
        object.__setattr__(
            self,
            "progress_cursor",
            _text(self.progress_cursor, "progress_cursor", required=False),
        )
        object.__setattr__(
            self,
            "progress_updated_at_ms",
            _nonneg_int(int(self.progress_updated_at_ms), "progress_updated_at_ms"),
        )
        object.__setattr__(
            self, "deadline_ms", _nonneg_int(int(self.deadline_ms), "deadline_ms")
        )
        status = self.status
        if not isinstance(status, SessionStatus):
            status = _parse_status(str(status), SessionStatus, "status")
            object.__setattr__(self, "status", status)
        exit_disp = self.exit_disposition
        if not isinstance(exit_disp, ExitDisposition):
            exit_disp = _parse_status(str(exit_disp), ExitDisposition, "exit_disposition")
            object.__setattr__(self, "exit_disposition", exit_disp)
        restart_disp = self.restart_disposition
        if not isinstance(restart_disp, RestartDisposition):
            restart_disp = _parse_status(
                str(restart_disp), RestartDisposition, "restart_disposition"
            )
            object.__setattr__(self, "restart_disposition", restart_disp)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        ancestry = tuple(str(item) for item in (self.ancestry or ()))
        object.__setattr__(self, "ancestry", ancestry)
        object.__setattr__(
            self,
            "capability",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.capability or {}),
                    name="capability",
                    max_bytes=MAX_CAPABILITY_BYTES,
                )
            ),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    @property
    def is_active(self) -> bool:
        return self.status is SessionStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "session_id": self.session_id,
            "daemon_id": self.daemon_id,
            "supervisor_id": self.supervisor_id,
            "process_birth": self.process_birth.to_dict(),
            "process_birth_id": self.process_birth_id,
            "role": self.role,
            "lane_id": self.lane_id,
            "shard_id": self.shard_id,
            "run_id": self.run_id,
            "server_id": self.server_id,
            "server_generation": int(self.server_generation),
            "fence_epoch": int(self.fence_epoch),
            "fencing_token": int(self.fencing_token),
            "quack_connection": self.quack_connection,
            "capability": dict(self.capability),
            "attached_at_ms": int(self.attached_at_ms),
            "last_heartbeat_at_ms": int(self.last_heartbeat_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "progress_cursor": self.progress_cursor,
            "progress_updated_at_ms": int(self.progress_updated_at_ms),
            "deadline_ms": int(self.deadline_ms),
            "status": self.status.value,
            "exit_disposition": self.exit_disposition.value,
            "restart_disposition": self.restart_disposition.value,
            "revision": int(self.revision),
            "ancestry": list(self.ancestry),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class StatusFileMirror:
    """Non-authoritative status-file projection of a session.

    Writing or touching a status file never creates or extends a session.
    """

    SCHEMA: ClassVar[str] = STATUS_FILE_MIRROR_SCHEMA

    mirror_path: str
    session_id: str
    written_at_ms: int
    content_digest: str
    body: Mapping[str, Any] = field(default_factory=dict)
    authoritative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "mirror_path": self.mirror_path,
            "session_id": self.session_id,
            "written_at_ms": int(self.written_at_ms),
            "content_digest": self.content_digest,
            "body": dict(self.body),
            "authoritative": False,
            "can_create_session": False,
            "can_extend_session": False,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DaemonRegistry:
    """DuckDB-backed supervisor/daemon/session/heartbeat authority."""

    INTERFACE: ClassVar[str] = DAEMON_REGISTRY_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        liveness: LivenessProbe | None = None,
        birth_reader: BirthReader | None = None,
        default_session_ttl_ms: int = DEFAULT_SESSION_TTL_MS,
        default_heartbeat_ttl_ms: int = DEFAULT_HEARTBEAT_TTL_MS,
        heartbeat_retain: int = DEFAULT_HEARTBEAT_RETAIN,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DaemonRegistry; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._liveness = liveness or (lambda birth: owner_liveness(birth))
        # When callers inject a custom liveness probe without an explicit
        # birth_reader (hermetic tests), do not cross-check against host /proc.
        # Synthetic PIDs often collide with live processes and would false-fail
        # renewals. Production callers that leave both defaults get OS checks.
        if birth_reader is not None:
            self._birth_reader = birth_reader
        elif liveness is not None:
            self._birth_reader = lambda _pid: None
        else:
            self._birth_reader = lambda pid: read_process_birth(int(pid))
        self._default_session_ttl_ms = _positive_int(
            int(default_session_ttl_ms), "default_session_ttl_ms"
        )
        self._default_heartbeat_ttl_ms = _positive_int(
            int(default_heartbeat_ttl_ms), "default_heartbeat_ttl_ms"
        )
        retain = _positive_int(int(heartbeat_retain), "heartbeat_retain")
        if retain > MAX_HEARTBEAT_RETAIN:
            raise DaemonRegistryBoundsError(
                f"heartbeat_retain exceeds the {MAX_HEARTBEAT_RETAIN} bound"
            )
        self._heartbeat_retain = retain
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True
        self._server_generation = 1

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    @property
    def server_generation(self) -> int:
        return int(self._server_generation)

    def open(self) -> "DaemonRegistry":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                for key, value in (
                    ("interface", DAEMON_REGISTRY_INTERFACE),
                    ("schema", DAEMON_REGISTRY_SCHEMA),
                ):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO registry_metadata(key, value)
                        VALUES (?, ?)
                        """,
                        [key, value],
                    )
                row = connection.execute(
                    "SELECT value FROM registry_metadata WHERE key = ?",
                    ["server_generation"],
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO registry_metadata(key, value) VALUES (?, ?)
                        """,
                        ["server_generation", "1"],
                    )
                    self._server_generation = 1
                else:
                    mapping = _row_mapping(row)
                    raw_generation = mapping.get("value")
                    if raw_generation is None:
                        raw_generation = mapping.get("VALUE")
                    if raw_generation is None:
                        raw_generation = mapping.get("0")
                    self._server_generation = max(1, int(raw_generation or 1))
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

    def __enter__(self) -> "DaemonRegistry":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DaemonRegistryNotOpenError("DaemonRegistry is not open")
        return self._connection

    def _begin(self, connection: Any) -> None:
        """Open an explicit write transaction when the connection is idle."""

        if getattr(connection, "in_transaction", False):
            return
        try:
            connection.execute("BEGIN TRANSACTION")
        except Exception:
            # Some adapters are always-autocommit; ignore redundant BEGIN.
            pass

    def _rollback_if_open(self, connection: Any) -> None:
        """Best-effort rollback so a failed write does not pin a transaction."""

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
        """Flush pending writes so subsequent reads observe mutations.

        Prefer the adapter's transaction-aware ``commit``. When the adapter is
        not tracking a transaction, still invoke the underlying DuckDB
        ``commit`` so multi-statement heartbeat/session updates cannot be lost
        behind an open implicit transaction.
        """

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

    # -- registration --------------------------------------------------------

    def register_supervisor(
        self,
        *,
        repository_id: str,
        process_birth: ProcessBirthIdentity | None = None,
        supervisor_id: str | None = None,
        run_id: str = "",
        server_generation: int | None = None,
        capability: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        status: InstanceStatus | str = InstanceStatus.RUNNING,
        now_ms: int | None = None,
    ) -> SupervisorInstance:
        """Register one supervisor instance bound to process birth."""

        birth = process_birth or current_process_birth()
        if not isinstance(birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        if int(birth.pid) <= 0:
            raise DaemonRegistryIdentityError("process birth pid must be positive")
        # PID alone is insufficient: require start_time_ticks for identity.
        if int(birth.start_time_ticks) <= 0:
            raise DaemonRegistryIdentityError(
                "process birth requires start_time_ticks; raw PID never proves identity"
            )
        birth_id = process_birth_id(birth)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        generation = (
            self._server_generation
            if server_generation is None
            else _positive_int(int(server_generation), "server_generation")
        )
        status_value = (
            status
            if isinstance(status, InstanceStatus)
            else _parse_status(str(status), InstanceStatus, "status")
        )
        instance = SupervisorInstance(
            supervisor_id=_text(supervisor_id or _new_id("supervisor"), "supervisor_id"),
            repository_id=_text(repository_id, "repository_id"),
            process_birth=birth,
            process_birth_id=birth_id,
            run_id=_text(run_id or _new_id("run"), "run_id", required=False),
            server_generation=generation,
            started_at_ms=now,
            status=status_value,  # type: ignore[arg-type]
            revision=1,
            capability=dict(capability or {}),
            body=dict(body or {}),
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            existing = connection.execute(
                """
                SELECT supervisor_id, process_birth_id, status
                FROM supervisor_instances WHERE supervisor_id = ?
                """,
                [instance.supervisor_id],
            ).fetchone()
            if existing is not None:
                mapping = _row_mapping(existing)
                if str(mapping.get("process_birth_id")) != birth_id:
                    self._commit_if_idle(connection)
                    raise DaemonRegistryConflictError(
                        "supervisor_id already bound to a different process birth"
                    )
                self._commit_if_idle(connection)
                loaded = self.get_supervisor(instance.supervisor_id)
                assert loaded is not None
                return loaded
            connection.execute(
                """
                INSERT INTO supervisor_instances(
                    supervisor_id, repository_id, process_birth_id,
                    process_birth_json, run_id, server_generation,
                    started_at_ms, stopped_at_ms, status, revision,
                    capability_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    instance.supervisor_id,
                    instance.repository_id,
                    instance.process_birth_id,
                    _canonical_json(instance.process_birth.to_dict()),
                    instance.run_id,
                    int(instance.server_generation),
                    int(instance.started_at_ms),
                    None,
                    instance.status.value,
                    int(instance.revision),
                    _canonical_json(dict(instance.capability)),
                    _canonical_json(dict(instance.body)),
                ],
            )
            self._commit_if_idle(connection)
            return instance

    def register_daemon(
        self,
        *,
        supervisor_id: str,
        role: str,
        process_birth: ProcessBirthIdentity | None = None,
        daemon_id: str | None = None,
        lane_id: str = "",
        shard_id: str = "",
        run_id: str = "",
        capability: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        status: InstanceStatus | str = InstanceStatus.RUNNING,
        now_ms: int | None = None,
    ) -> DaemonInstance:
        """Register one daemon under a supervisor."""

        supervisor = self.get_supervisor(_text(supervisor_id, "supervisor_id"))
        if supervisor is None:
            raise DaemonRegistryError(f"unknown supervisor_id: {supervisor_id}")
        birth = process_birth or current_process_birth()
        if not isinstance(birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        if int(birth.pid) <= 0 or int(birth.start_time_ticks) <= 0:
            raise DaemonRegistryIdentityError(
                "process birth requires pid and start_time_ticks; "
                "raw PID never proves identity"
            )
        birth_id = process_birth_id(birth)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        status_value = (
            status
            if isinstance(status, InstanceStatus)
            else _parse_status(str(status), InstanceStatus, "status")
        )
        instance = DaemonInstance(
            daemon_id=_text(daemon_id or _new_id("daemon"), "daemon_id"),
            supervisor_id=supervisor.supervisor_id,
            process_birth=birth,
            process_birth_id=birth_id,
            role=_text(role, "role"),
            lane_id=_text(lane_id, "lane_id", required=False),
            shard_id=_text(shard_id, "shard_id", required=False),
            run_id=_text(run_id or supervisor.run_id, "run_id", required=False),
            started_at_ms=now,
            status=status_value,  # type: ignore[arg-type]
            revision=1,
            capability=dict(capability or {}),
            body=dict(body or {}),
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            existing = connection.execute(
                "SELECT daemon_id FROM daemon_instances WHERE daemon_id = ?",
                [instance.daemon_id],
            ).fetchone()
            if existing is not None:
                self._commit_if_idle(connection)
                loaded = self.get_daemon(instance.daemon_id)
                assert loaded is not None
                if loaded.process_birth_id != birth_id:
                    raise DaemonRegistryConflictError(
                        "daemon_id already bound to a different process birth"
                    )
                return loaded
            connection.execute(
                """
                INSERT INTO daemon_instances(
                    daemon_id, supervisor_id, process_birth_id, process_birth_json,
                    role, lane_id, shard_id, run_id, started_at_ms, stopped_at_ms,
                    status, revision, capability_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    instance.daemon_id,
                    instance.supervisor_id,
                    instance.process_birth_id,
                    _canonical_json(instance.process_birth.to_dict()),
                    instance.role,
                    instance.lane_id,
                    instance.shard_id,
                    instance.run_id,
                    int(instance.started_at_ms),
                    None,
                    instance.status.value,
                    int(instance.revision),
                    _canonical_json(dict(instance.capability)),
                    _canonical_json(dict(instance.body)),
                ],
            )
            self._commit_if_idle(connection)
            return instance

    def open_session(
        self,
        *,
        daemon_id: str,
        process_birth: ProcessBirthIdentity | None = None,
        session_id: str | None = None,
        role: str | None = None,
        lane_id: str | None = None,
        shard_id: str | None = None,
        run_id: str | None = None,
        server_id: str = "",
        server_generation: int | None = None,
        quack_connection: str = "",
        capability: Mapping[str, Any] | None = None,
        deadline_ms: int = 0,
        ttl_ms: int | None = None,
        ancestry: Sequence[str] | None = None,
        body: Mapping[str, Any] | None = None,
        allow_replace_dead: bool = True,
        now_ms: int | None = None,
    ) -> DaemonSession:
        """Open a fenced session. Duplicate live role/lane owners are refused."""

        daemon = self.get_daemon(_text(daemon_id, "daemon_id"))
        if daemon is None:
            raise DaemonRegistryError(f"unknown daemon_id: {daemon_id}")
        birth = process_birth or current_process_birth()
        if not isinstance(birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        if int(birth.pid) <= 0 or int(birth.start_time_ticks) <= 0:
            raise DaemonRegistryIdentityError(
                "process birth requires pid and start_time_ticks; "
                "raw PID never proves identity"
            )
        # Opening also requires that the claimed birth is currently live.
        liveness = self._liveness(birth)
        if liveness is OwnerLiveness.DEAD:
            raise DaemonRegistryIdentityError(
                "dead process birth cannot open a session"
            )
        if liveness is OwnerLiveness.UNKNOWN:
            raise DaemonRegistryIdentityError(
                "unknown process birth cannot open a session"
            )
        birth_id = process_birth_id(birth)
        selected_role = _text(role if role is not None else daemon.role, "role")
        selected_lane = _text(
            lane_id if lane_id is not None else daemon.lane_id,
            "lane_id",
            required=False,
        )
        selected_shard = _text(
            shard_id if shard_id is not None else daemon.shard_id,
            "shard_id",
            required=False,
        )
        selected_run = _text(
            run_id if run_id is not None else daemon.run_id,
            "run_id",
            required=False,
        )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        ttl = (
            self._default_session_ttl_ms
            if ttl_ms is None
            else _positive_int(int(ttl_ms), "ttl_ms")
        )
        generation = (
            self._server_generation
            if server_generation is None
            else _positive_int(int(server_generation), "server_generation")
        )
        expires = now + ttl
        selected_deadline = _nonneg_int(int(deadline_ms), "deadline_ms")
        if selected_deadline and selected_deadline < expires:
            expires = selected_deadline

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                # Expire stale sessions first so fencing sees current truth.
                self._expire_sessions_locked(connection, now_ms=now)
                fence_epoch, fencing_token, ancestry_list = self._fence_role_lane_locked(
                    connection,
                    role=selected_role,
                    lane_id=selected_lane,
                    claimant_birth_id=birth_id,
                    allow_replace_dead=allow_replace_dead,
                    now_ms=now,
                )
                if ancestry:
                    ancestry_list = list(ancestry) + ancestry_list
                session = DaemonSession(
                    session_id=_text(session_id or _new_id("session"), "session_id"),
                    daemon_id=daemon.daemon_id,
                    supervisor_id=daemon.supervisor_id,
                    process_birth=birth,
                    process_birth_id=birth_id,
                    role=selected_role,
                    lane_id=selected_lane,
                    shard_id=selected_shard,
                    run_id=selected_run,
                    server_id=_text(server_id, "server_id", required=False),
                    server_generation=generation,
                    fence_epoch=fence_epoch,
                    fencing_token=fencing_token,
                    quack_connection=_text(
                        quack_connection, "quack_connection", required=False
                    ),
                    capability=dict(capability or dict(daemon.capability)),
                    attached_at_ms=now,
                    last_heartbeat_at_ms=now,
                    expires_at_ms=expires,
                    progress_cursor="",
                    progress_updated_at_ms=0,
                    deadline_ms=selected_deadline,
                    status=SessionStatus.ACTIVE,
                    exit_disposition=ExitDisposition.RUNNING,
                    restart_disposition=RestartDisposition.NONE,
                    revision=1,
                    ancestry=tuple(ancestry_list),
                    body=dict(body or {}),
                )
                existing = connection.execute(
                    "SELECT session_id FROM daemon_sessions WHERE session_id = ?",
                    [session.session_id],
                ).fetchone()
                if existing is not None:
                    raise DaemonRegistryConflictError(
                        f"session_id already exists: {session.session_id}"
                    )
                self._insert_session_locked(connection, session)
                # Initial heartbeat establishes liveness without progress.
                self._insert_heartbeat_locked(
                    connection,
                    session=session,
                    observed_at_ms=now,
                    expires_at_ms=expires,
                    payload={"kind": "session.open"},
                )
                self._commit_if_idle(connection)
                return session
            except Exception:
                self._rollback_if_open(connection)
                raise

    def adopt_session(
        self,
        session_id: str,
        *,
        process_birth: ProcessBirthIdentity,
        expected_fencing_token: int | None = None,
        now_ms: int | None = None,
    ) -> DaemonSession:
        """Adopt an existing session only with exact matching process birth.

        Adoption re-binds a still-active session to a proven-live birth that
        already owns it. Dead, reused, or unknown births cannot adopt.
        """

        birth = process_birth
        if not isinstance(birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        if int(birth.pid) <= 0 or int(birth.start_time_ticks) <= 0:
            raise DaemonRegistryIdentityError(
                "process birth requires pid and start_time_ticks; "
                "raw PID never proves identity"
            )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._expire_sessions_locked(connection, now_ms=now)
            session = self._load_session_locked(
                connection, _text(session_id, "session_id")
            )
            if session is None:
                raise DaemonRegistrySessionError(f"unknown session_id: {session_id}")
            if session.status is not SessionStatus.ACTIVE:
                raise DaemonRegistrySessionError(
                    f"session {session_id} is not active ({session.status.value})"
                )
            if expected_fencing_token is not None and int(
                expected_fencing_token
            ) != int(session.fencing_token):
                raise DaemonRegistryConflictError(
                    "session fencing token mismatch during adoption"
                )
            if session.process_birth_id != process_birth_id(birth):
                raise DaemonRegistryIdentityError(
                    "process birth does not match session owner; "
                    "PID reuse or different process cannot adopt"
                )
            if not process_births_match(session.process_birth, birth):
                raise DaemonRegistryIdentityError(
                    "process birth factors do not match session owner"
                )
            liveness = self._liveness(birth)
            if liveness is OwnerLiveness.DEAD:
                raise DaemonRegistryIdentityError(
                    "dead process birth cannot adopt a session"
                )
            if liveness is OwnerLiveness.UNKNOWN:
                raise DaemonRegistryIdentityError(
                    "unknown process birth cannot adopt a session"
                )
            # Adoption refreshes identity confirmation but does not extend
            # expiry by itself — only heartbeat does.
            return session

    def heartbeat(
        self,
        session_id: str,
        *,
        process_birth: ProcessBirthIdentity | None = None,
        process_birth_id_claim: str | None = None,
        pid_only: int | None = None,
        fencing_token: int | None = None,
        ttl_ms: int | None = None,
        payload: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> Heartbeat:
        """Record a liveness heartbeat. Progress is not updated here.

        Raw PID claims (``pid_only`` without full birth) are rejected.
        Dead, reused, and unknown births cannot renew.
        Late heartbeats after expiry fail closed.
        """

        if pid_only is not None and process_birth is None and not process_birth_id_claim:
            raise DaemonRegistryIdentityError(
                "raw PID never proves identity; provide process birth"
            )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        ttl = (
            self._default_heartbeat_ttl_ms
            if ttl_ms is None
            else _positive_int(int(ttl_ms), "ttl_ms")
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            self._expire_sessions_locked(connection, now_ms=now)
            session = self._load_session_locked(
                connection, _text(session_id, "session_id")
            )
            if session is None:
                self._commit_if_idle(connection)
                raise DaemonRegistrySessionError(f"unknown session_id: {session_id}")
            if session.status is not SessionStatus.ACTIVE:
                self._commit_if_idle(connection)
                raise DaemonRegistrySessionError(
                    f"session {session_id} is not active ({session.status.value}); "
                    "late or expired sessions cannot renew"
                )
            if fencing_token is not None and int(fencing_token) != int(
                session.fencing_token
            ):
                self._commit_if_idle(connection)
                raise DaemonRegistryConflictError("heartbeat fencing token mismatch")
            # Identity: full process birth or content id; never bare PID.
            if process_birth is not None:
                if not process_births_match(session.process_birth, process_birth):
                    self._commit_if_idle(connection)
                    raise DaemonRegistryIdentityError(
                        "process birth does not match session; cannot renew"
                    )
                claimed_id = process_birth_id(process_birth)
                probe_birth = process_birth
            elif process_birth_id_claim is not None:
                claimed_id = _text(process_birth_id_claim, "process_birth_id_claim")
                if claimed_id != session.process_birth_id:
                    self._commit_if_idle(connection)
                    raise DaemonRegistryIdentityError(
                        "process birth id does not match session; cannot renew"
                    )
                probe_birth = session.process_birth
            else:
                self._commit_if_idle(connection)
                raise DaemonRegistryIdentityError(
                    "process birth proof is required; raw PID never proves identity"
                )
            if claimed_id != session.process_birth_id:
                self._commit_if_idle(connection)
                raise DaemonRegistryIdentityError(
                    "process birth id does not match session; cannot renew"
                )
            liveness = self._liveness(probe_birth)
            if liveness is OwnerLiveness.DEAD:
                # Mark dead so the role/lane can be reclaimed.
                self._mark_session_status_locked(
                    connection,
                    session.session_id,
                    status=SessionStatus.EXPIRED,
                    exit_disposition=ExitDisposition.KILLED,
                    restart_disposition=RestartDisposition.RESTART,
                    now_ms=now,
                    reason="owner_dead_on_heartbeat",
                )
                self._commit_if_idle(connection)
                raise DaemonRegistryIdentityError(
                    "dead or reused process birth cannot renew a session"
                )
            if liveness is OwnerLiveness.UNKNOWN:
                self._commit_if_idle(connection)
                raise DaemonRegistryIdentityError(
                    "unknown process birth cannot renew a session"
                )
            # Optional cross-check against live reader when full birth supplied.
            if process_birth is not None:
                try:
                    observed = self._birth_reader(int(process_birth.pid))
                except Exception:
                    observed = None
                if observed is not None and not process_births_match(
                    process_birth, observed
                ):
                    self._mark_session_status_locked(
                        connection,
                        session.session_id,
                        status=SessionStatus.EXPIRED,
                        exit_disposition=ExitDisposition.KILLED,
                        restart_disposition=RestartDisposition.RESTART,
                        now_ms=now,
                        reason="pid_reuse_on_heartbeat",
                    )
                    self._commit_if_idle(connection)
                    raise DaemonRegistryIdentityError(
                        "process birth reused under same PID; cannot renew"
                    )
            if now > int(session.expires_at_ms):
                self._mark_session_status_locked(
                    connection,
                    session.session_id,
                    status=SessionStatus.EXPIRED,
                    exit_disposition=ExitDisposition.ERROR,
                    restart_disposition=RestartDisposition.RESTART,
                    now_ms=now,
                    reason="late_heartbeat_after_expiry",
                )
                self._commit_if_idle(connection)
                raise DaemonRegistrySessionError(
                    "late heartbeat after session expiry cannot renew"
                )
            expires = now + ttl
            if session.deadline_ms and session.deadline_ms < expires:
                expires = int(session.deadline_ms)
            # Heartbeat updates liveness only — never progress_cursor.
            connection.execute(
                """
                UPDATE daemon_sessions
                SET last_heartbeat_at_ms = ?,
                    expires_at_ms = ?,
                    revision = revision + 1
                WHERE session_id = ?
                """,
                [int(now), int(expires), session.session_id],
            )
            refreshed = DaemonSession(
                session_id=session.session_id,
                daemon_id=session.daemon_id,
                supervisor_id=session.supervisor_id,
                process_birth=session.process_birth,
                process_birth_id=session.process_birth_id,
                role=session.role,
                lane_id=session.lane_id,
                shard_id=session.shard_id,
                run_id=session.run_id,
                server_id=session.server_id,
                server_generation=session.server_generation,
                fence_epoch=session.fence_epoch,
                fencing_token=session.fencing_token,
                quack_connection=session.quack_connection,
                capability=dict(session.capability),
                attached_at_ms=session.attached_at_ms,
                last_heartbeat_at_ms=now,
                expires_at_ms=expires,
                progress_cursor=session.progress_cursor,
                progress_updated_at_ms=session.progress_updated_at_ms,
                deadline_ms=session.deadline_ms,
                status=session.status,
                exit_disposition=session.exit_disposition,
                restart_disposition=session.restart_disposition,
                revision=int(session.revision) + 1,
                ancestry=session.ancestry,
                body=dict(session.body),
            )
            heartbeat = self._insert_heartbeat_locked(
                connection,
                session=refreshed,
                observed_at_ms=now,
                expires_at_ms=expires,
                payload=dict(payload or {}),
            )
            self._commit_if_idle(connection)
            return heartbeat

    def record_progress(
        self,
        session_id: str,
        cursor_value: str,
        *,
        process_birth: ProcessBirthIdentity | None = None,
        process_birth_id_claim: str | None = None,
        payload: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> ProgressCursor:
        """Record work progress without extending session liveness/expiry.

        Progress and heartbeats are intentionally distinct. A progress update
        never renews ``expires_at_ms`` or ``last_heartbeat_at_ms``.
        """

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            self._expire_sessions_locked(connection, now_ms=now)
            session = self._load_session_locked(
                connection, _text(session_id, "session_id")
            )
            if session is None:
                raise DaemonRegistrySessionError(f"unknown session_id: {session_id}")
            if session.status is not SessionStatus.ACTIVE:
                raise DaemonRegistrySessionError(
                    f"session {session_id} is not active ({session.status.value})"
                )
            self._assert_birth_claim(
                session,
                process_birth=process_birth,
                process_birth_id_claim=process_birth_id_claim,
            )
            cursor = _text(cursor_value, "cursor_value")
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS max_seq
                FROM progress_records WHERE session_id = ?
                """,
                [session.session_id],
            ).fetchone()
            sequence = (
                _row_int(
                    _row_mapping(row), "max_seq", "MAX(sequence)", "coalesce", default=0
                )
                + 1
            )
            progress_id = _sha256_hex(
                _canonical_json(
                    {
                        "session_id": session.session_id,
                        "cursor_value": cursor,
                        "recorded_at_ms": now,
                        "sequence": sequence,
                    }
                ).encode("utf-8")
            )
            body = _bounded_mapping(
                dict(payload or {}), name="payload", max_bytes=MAX_PAYLOAD_BYTES
            )
            connection.execute(
                """
                INSERT INTO progress_records(
                    progress_id, session_id, cursor_value, recorded_at_ms,
                    sequence, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    progress_id,
                    session.session_id,
                    cursor,
                    now,
                    sequence,
                    _canonical_json(body),
                ],
            )
            # Update progress fields only — leave heartbeat/expiry untouched.
            connection.execute(
                """
                UPDATE daemon_sessions
                SET progress_cursor = ?,
                    progress_updated_at_ms = ?,
                    revision = revision + 1
                WHERE session_id = ?
                """,
                [cursor, now, session.session_id],
            )
            self._commit_if_idle(connection)
            return ProgressCursor(
                session_id=session.session_id,
                cursor_value=cursor,
                recorded_at_ms=now,
                sequence=sequence,
                payload=body,
            )

    def stop_session(
        self,
        session_id: str,
        *,
        process_birth: ProcessBirthIdentity | None = None,
        process_birth_id_claim: str | None = None,
        fencing_token: int | None = None,
        exit_disposition: ExitDisposition | str = ExitDisposition.CLEAN,
        restart_disposition: RestartDisposition | str = RestartDisposition.NONE,
        now_ms: int | None = None,
    ) -> DaemonSession:
        """Stop an active session with optional identity check."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        exit_value = (
            exit_disposition
            if isinstance(exit_disposition, ExitDisposition)
            else _parse_status(str(exit_disposition), ExitDisposition, "exit_disposition")
        )
        restart_value = (
            restart_disposition
            if isinstance(restart_disposition, RestartDisposition)
            else _parse_status(
                str(restart_disposition), RestartDisposition, "restart_disposition"
            )
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            session = self._load_session_locked(
                connection, _text(session_id, "session_id")
            )
            if session is None:
                self._commit_if_idle(connection)
                raise DaemonRegistrySessionError(f"unknown session_id: {session_id}")
            if fencing_token is not None and int(fencing_token) != int(
                session.fencing_token
            ):
                self._commit_if_idle(connection)
                raise DaemonRegistryConflictError("stop fencing token mismatch")
            if process_birth is not None or process_birth_id_claim is not None:
                self._assert_birth_claim(
                    session,
                    process_birth=process_birth,
                    process_birth_id_claim=process_birth_id_claim,
                    require_live=False,
                )
            self._mark_session_status_locked(
                connection,
                session.session_id,
                status=SessionStatus.STOPPED,
                exit_disposition=exit_value,  # type: ignore[arg-type]
                restart_disposition=restart_value,  # type: ignore[arg-type]
                now_ms=now,
                reason="stop",
            )
            self._commit_if_idle(connection)
            loaded = self._load_session_locked(connection, session.session_id)
            assert loaded is not None
            return loaded

    def expire_sessions(self, *, now_ms: int | None = None) -> list[str]:
        """Mark active sessions past their expiry as expired."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            expired = self._expire_sessions_locked(connection, now_ms=now)
            self._commit_if_idle(connection)
            return expired

    def record_server_restart(
        self,
        *,
        now_ms: int | None = None,
        reason: str = "server_restart",
    ) -> int:
        """Advance server generation and supersede active sessions.

        After a server restart, prior sessions cannot renew against the new
        generation; callers must open fresh sessions.
        """

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            self._server_generation = int(self._server_generation) + 1
            connection.execute(
                """
                INSERT OR REPLACE INTO registry_metadata(key, value)
                VALUES ('server_generation', ?)
                """,
                [str(self._server_generation)],
            )
            rows = connection.execute(
                """
                SELECT session_id FROM daemon_sessions WHERE status = ?
                """,
                [SessionStatus.ACTIVE.value],
            ).fetchall()
            for row in rows:
                mapping = _row_mapping(row)
                session_id = str(mapping.get("session_id") or "")
                if not session_id:
                    continue
                self._mark_session_status_locked(
                    connection,
                    session_id,
                    status=SessionStatus.SUPERSEDED,
                    exit_disposition=ExitDisposition.KILLED,
                    restart_disposition=RestartDisposition.RESTART,
                    now_ms=now,
                    reason=reason,
                )
            self._commit_if_idle(connection)
            return int(self._server_generation)

    def compact_heartbeats(
        self,
        session_id: str | None = None,
        *,
        retain: int | None = None,
    ) -> dict[str, int]:
        """Retain only the newest heartbeats per session."""

        keep = (
            self._heartbeat_retain
            if retain is None
            else _positive_int(int(retain), "retain")
        )
        if keep > MAX_HEARTBEAT_RETAIN:
            raise DaemonRegistryBoundsError(
                f"retain exceeds the {MAX_HEARTBEAT_RETAIN} bound"
            )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            if session_id is None:
                rows = connection.execute(
                    "SELECT DISTINCT session_id FROM session_heartbeats"
                ).fetchall()
                session_ids = [
                    str(_row_mapping(row).get("session_id") or "")
                    for row in rows
                ]
            else:
                session_ids = [_text(session_id, "session_id")]
            deleted = 0
            for sid in session_ids:
                if not sid:
                    continue
                deleted += self._compact_session_heartbeats_locked(
                    connection, sid, keep=keep
                )
            self._commit_if_idle(connection)
            return {"deleted": deleted, "retain": keep}

    # -- status file mirror (non-authoritative) ------------------------------

    def mirror_status_file(
        self,
        path: Path | str,
        session_id: str,
        *,
        now_ms: int | None = None,
    ) -> StatusFileMirror:
        """Write a legacy status-file projection of an existing session.

        The file is a mirror only: it cannot create a session and writing it
        never extends ``expires_at_ms`` / heartbeat state.
        """

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        session = self.get_session(_text(session_id, "session_id"))
        if session is None:
            raise DaemonRegistrySessionError(
                f"cannot mirror unknown session_id: {session_id}"
            )
        target = Path(path)
        body = {
            "schema": STATUS_FILE_MIRROR_SCHEMA,
            "authoritative": False,
            "can_create_session": False,
            "can_extend_session": False,
            "session": session.to_dict(),
            "mirrored_at_ms": now,
        }
        digest = _sha256_hex(_canonical_json(body).encode("utf-8"))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, target)
        mirror = StatusFileMirror(
            mirror_path=str(target),
            session_id=session.session_id,
            written_at_ms=now,
            content_digest=digest,
            body=body,
            authoritative=False,
        )
        with self._lock:
            connection = self._require()
            # Capture pre-mirror expiry to prove we never extend.
            before = self._load_session_locked(connection, session.session_id)
            connection.execute(
                """
                INSERT OR REPLACE INTO status_file_mirrors(
                    mirror_path, session_id, written_at_ms, content_digest, body_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    str(target),
                    session.session_id,
                    now,
                    digest,
                    _canonical_json(body),
                ],
            )
            after = self._load_session_locked(connection, session.session_id)
            self._commit_if_idle(connection)
            if before is not None and after is not None:
                if int(after.expires_at_ms) != int(before.expires_at_ms):
                    raise DaemonRegistryError(
                        "status file mirror must not extend session expiry"
                    )
                if int(after.last_heartbeat_at_ms) != int(before.last_heartbeat_at_ms):
                    raise DaemonRegistryError(
                        "status file mirror must not update heartbeat"
                    )
            return mirror

    def load_status_file(self, path: Path | str) -> StatusFileMirror | None:
        """Load a status-file mirror without creating or extending a session."""

        target = Path(path)
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        session_id = str(payload.get("session_id") or "")
        if not session_id and isinstance(payload.get("session"), Mapping):
            session_id = str(payload["session"].get("session_id") or "")
        digest = _sha256_hex(_canonical_json(dict(payload)).encode("utf-8"))
        return StatusFileMirror(
            mirror_path=str(target),
            session_id=session_id,
            written_at_ms=int(payload.get("mirrored_at_ms") or 0),
            content_digest=digest,
            body=dict(payload),
            authoritative=False,
        )

    def ingest_status_file(
        self,
        path: Path | str,
        *,
        now_ms: int | None = None,
    ) -> StatusFileMirror | None:
        """Attempt to ingest a status file as authority — always non-creating.

        Explicit API so callers cannot accidentally treat file mtime as
        session creation or renewal. Returns the mirror projection only.
        """

        del now_ms  # Intentionally unused: file age has no authority effect.
        return self.load_status_file(path)

    # -- queries -------------------------------------------------------------

    def get_supervisor(self, supervisor_id: str) -> SupervisorInstance | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT supervisor_id, repository_id, process_birth_id,
                       process_birth_json, run_id, server_generation,
                       started_at_ms, stopped_at_ms, status, revision,
                       capability_json, body_json
                FROM supervisor_instances WHERE supervisor_id = ?
                """,
                [_text(supervisor_id, "supervisor_id")],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_supervisor(_row_mapping(row))

    def get_daemon(self, daemon_id: str) -> DaemonInstance | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT daemon_id, supervisor_id, process_birth_id,
                       process_birth_json, role, lane_id, shard_id, run_id,
                       started_at_ms, stopped_at_ms, status, revision,
                       capability_json, body_json
                FROM daemon_instances WHERE daemon_id = ?
                """,
                [_text(daemon_id, "daemon_id")],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_daemon(_row_mapping(row))

    def get_session(self, session_id: str) -> DaemonSession | None:
        with self._lock:
            return self._load_session_locked(
                self._require(), _text(session_id, "session_id")
            )

    def active_session_for_role_lane(
        self,
        role: str,
        lane_id: str = "",
        *,
        now_ms: int | None = None,
    ) -> DaemonSession | None:
        """Return the active owner of a role/lane, if any."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._expire_sessions_locked(connection, now_ms=now)
            row = connection.execute(
                """
                SELECT session_id FROM daemon_sessions
                WHERE role = ? AND lane_id = ? AND status = ?
                ORDER BY fencing_token DESC
                LIMIT 1
                """,
                [
                    _text(role, "role"),
                    _text(lane_id, "lane_id", required=False),
                    SessionStatus.ACTIVE.value,
                ],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            return self._load_session_locked(
                connection, str(mapping.get("session_id") or "")
            )

    def list_heartbeats(
        self,
        session_id: str,
        *,
        limit: int = DEFAULT_HEARTBEAT_RETAIN,
    ) -> list[Heartbeat]:
        page = _positive_int(int(limit), "limit")
        with self._lock:
            connection = self._require()
            # Interpolate validated LIMIT; some DuckDB builds reject bound LIMIT.
            rows = connection.execute(
                f"""
                SELECT heartbeat_cid, session_id, fencing_token, observed_at_ms,
                       expires_at_ms, sequence, payload_json
                FROM session_heartbeats
                WHERE session_id = ?
                ORDER BY sequence DESC, observed_at_ms DESC, heartbeat_cid DESC
                LIMIT {page}
                """,
                [_text(session_id, "session_id")],
            ).fetchall()
            return [self._row_to_heartbeat(_row_mapping(row)) for row in rows]

    def list_progress(
        self,
        session_id: str,
        *,
        limit: int = 32,
    ) -> list[ProgressCursor]:
        page = _positive_int(int(limit), "limit")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                f"""
                SELECT progress_id, session_id, cursor_value, recorded_at_ms,
                       sequence, payload_json
                FROM progress_records
                WHERE session_id = ?
                ORDER BY sequence DESC, recorded_at_ms DESC
                LIMIT {page}
                """,
                [_text(session_id, "session_id")],
            ).fetchall()
            result: list[ProgressCursor] = []
            for row in rows:
                mapping = _row_mapping(row)
                payload = json.loads(str(mapping.get("payload_json") or "{}"))
                result.append(
                    ProgressCursor(
                        session_id=str(mapping["session_id"]),
                        cursor_value=str(mapping["cursor_value"]),
                        recorded_at_ms=int(mapping["recorded_at_ms"]),
                        sequence=int(mapping["sequence"]),
                        payload=payload if isinstance(payload, dict) else {},
                    )
                )
            return result

    def exact_ancestry(self, session_id: str) -> tuple[str, ...]:
        """Return exact session ancestry (superseded owners) for one session."""

        session = self.get_session(session_id)
        if session is None:
            return ()
        return tuple(session.ancestry)

    # -- internal helpers ----------------------------------------------------

    def _assert_birth_claim(
        self,
        session: DaemonSession,
        *,
        process_birth: ProcessBirthIdentity | None,
        process_birth_id_claim: str | None,
        require_live: bool = True,
    ) -> None:
        if process_birth is None and process_birth_id_claim is None:
            raise DaemonRegistryIdentityError(
                "process birth proof is required; raw PID never proves identity"
            )
        if process_birth is not None:
            if not process_births_match(session.process_birth, process_birth):
                raise DaemonRegistryIdentityError(
                    "process birth does not match session owner"
                )
            if process_birth_id(process_birth) != session.process_birth_id:
                raise DaemonRegistryIdentityError(
                    "process birth id does not match session owner"
                )
            probe = process_birth
        else:
            claim = _text(process_birth_id_claim, "process_birth_id_claim")
            if claim != session.process_birth_id:
                raise DaemonRegistryIdentityError(
                    "process birth id does not match session owner"
                )
            probe = session.process_birth
        if not require_live:
            return
        liveness = self._liveness(probe)
        if liveness is OwnerLiveness.DEAD:
            raise DaemonRegistryIdentityError(
                "dead or reused process birth cannot act on session"
            )
        if liveness is OwnerLiveness.UNKNOWN:
            raise DaemonRegistryIdentityError(
                "unknown process birth cannot act on session"
            )

    def _fence_role_lane_locked(
        self,
        connection: Any,
        *,
        role: str,
        lane_id: str,
        claimant_birth_id: str,
        allow_replace_dead: bool,
        now_ms: int,
    ) -> tuple[int, int, list[str]]:
        """Fence exclusive active ownership of (role, lane)."""

        rows = connection.execute(
            """
            SELECT session_id, process_birth_id, process_birth_json,
                   fence_epoch, fencing_token, status, expires_at_ms
            FROM daemon_sessions
            WHERE role = ? AND lane_id = ? AND status = ?
            ORDER BY fencing_token DESC
            """,
            [role, lane_id, SessionStatus.ACTIVE.value],
        ).fetchall()
        ancestry: list[str] = []
        fence_epoch = 1
        fencing_token = 1
        for row in rows:
            mapping = _row_mapping(row)
            owner_session_id = str(mapping.get("session_id") or "")
            owner_birth = process_birth_from_mapping(
                json.loads(str(mapping.get("process_birth_json") or "{}"))
            )
            owner_birth_id = str(mapping.get("process_birth_id") or "")
            fence_epoch = max(fence_epoch, int(mapping.get("fence_epoch") or 1))
            fencing_token = max(
                fencing_token, int(mapping.get("fencing_token") or 1) + 1
            )
            if owner_birth_id == claimant_birth_id:
                # Same process re-registering: supersede prior session.
                self._mark_session_status_locked(
                    connection,
                    owner_session_id,
                    status=SessionStatus.SUPERSEDED,
                    exit_disposition=ExitDisposition.CLEAN,
                    restart_disposition=RestartDisposition.RESTART,
                    now_ms=now_ms,
                    reason="same_birth_reopen",
                )
                ancestry.append(owner_session_id)
                continue
            liveness = self._liveness(owner_birth)
            if liveness is OwnerLiveness.ALIVE:
                raise DaemonRegistryConflictError(
                    f"role/lane ownership fenced by live session {owner_session_id} "
                    f"(role={role!r}, lane={lane_id!r})"
                )
            if liveness is OwnerLiveness.UNKNOWN:
                raise DaemonRegistryConflictError(
                    f"role/lane ownership fenced; owner liveness unknown for "
                    f"session {owner_session_id}"
                )
            # DEAD owner
            if not allow_replace_dead:
                raise DaemonRegistryConflictError(
                    f"role/lane still held by dead session {owner_session_id}; "
                    "replacement disabled"
                )
            self._mark_session_status_locked(
                connection,
                owner_session_id,
                status=SessionStatus.SUPERSEDED,
                exit_disposition=ExitDisposition.KILLED,
                restart_disposition=RestartDisposition.RESTART,
                now_ms=now_ms,
                reason="dead_owner_reclaim",
            )
            ancestry.append(owner_session_id)
            fence_epoch = max(fence_epoch, int(mapping.get("fence_epoch") or 1) + 1)
        return fence_epoch, fencing_token, ancestry

    def _expire_sessions_locked(
        self, connection: Any, *, now_ms: int
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT session_id FROM daemon_sessions
            WHERE status = ? AND expires_at_ms < ?
            """,
            [SessionStatus.ACTIVE.value, now_ms],
        ).fetchall()
        expired: list[str] = []
        for row in rows:
            mapping = _row_mapping(row)
            session_id = str(mapping.get("session_id") or "")
            if not session_id:
                continue
            self._mark_session_status_locked(
                connection,
                session_id,
                status=SessionStatus.EXPIRED,
                exit_disposition=ExitDisposition.ERROR,
                restart_disposition=RestartDisposition.RESTART,
                now_ms=now_ms,
                reason="ttl_expired",
            )
            expired.append(session_id)
        return expired

    def _mark_session_status_locked(
        self,
        connection: Any,
        session_id: str,
        *,
        status: SessionStatus,
        exit_disposition: ExitDisposition,
        restart_disposition: RestartDisposition,
        now_ms: int,
        reason: str,
    ) -> None:
        connection.execute(
            """
            UPDATE daemon_sessions
            SET status = ?,
                exit_disposition = ?,
                restart_disposition = ?,
                revision = revision + 1,
                body_json = ?
            WHERE session_id = ?
            """,
            [
                status.value,
                exit_disposition.value,
                restart_disposition.value,
                _canonical_json(
                    {
                        "terminal_reason": reason,
                        "terminal_at_ms": now_ms,
                    }
                ),
                session_id,
            ],
        )

    def _insert_session_locked(
        self, connection: Any, session: DaemonSession
    ) -> None:
        connection.execute(
            """
            INSERT INTO daemon_sessions(
                session_id, daemon_id, supervisor_id, process_birth_id,
                process_birth_json, role, lane_id, shard_id, run_id,
                server_id, server_generation, fence_epoch, fencing_token,
                quack_connection, capability_json, attached_at_ms,
                last_heartbeat_at_ms, expires_at_ms, progress_cursor,
                progress_updated_at_ms, deadline_ms, status,
                exit_disposition, restart_disposition, revision,
                ancestry_json, body_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                session.session_id,
                session.daemon_id,
                session.supervisor_id,
                session.process_birth_id,
                _canonical_json(session.process_birth.to_dict()),
                session.role,
                session.lane_id,
                session.shard_id,
                session.run_id,
                session.server_id,
                int(session.server_generation),
                int(session.fence_epoch),
                int(session.fencing_token),
                session.quack_connection,
                _canonical_json(dict(session.capability)),
                int(session.attached_at_ms),
                int(session.last_heartbeat_at_ms),
                int(session.expires_at_ms),
                session.progress_cursor,
                int(session.progress_updated_at_ms),
                int(session.deadline_ms),
                session.status.value,
                session.exit_disposition.value,
                session.restart_disposition.value,
                int(session.revision),
                _canonical_json(list(session.ancestry)),
                _canonical_json(dict(session.body)),
            ],
        )

    def _compact_session_heartbeats_locked(
        self,
        connection: Any,
        session_id: str,
        *,
        keep: int,
    ) -> int:
        """Drop oldest heartbeats so at most ``keep`` remain. Returns deleted count."""

        keep = max(1, int(keep))
        # Count first so we can report deleted rows without relying on driver
        # rowcount for DELETE ... WHERE cid NOT IN (SELECT ... LIMIT n).
        count_row = connection.execute(
            """
            SELECT COUNT(*) AS heartbeat_count
            FROM session_heartbeats
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()
        total = _row_int(_row_mapping(count_row), "heartbeat_count", "count", default=0)
        if total <= keep:
            return 0
        deleted = 0
        # Prefer set-based retention; fall back when the engine rejects
        # DELETE ... NOT IN (SELECT ... LIMIT n).
        try:
            connection.execute(
                f"""
                DELETE FROM session_heartbeats
                WHERE session_id = ?
                  AND heartbeat_cid NOT IN (
                    SELECT heartbeat_cid
                    FROM session_heartbeats
                    WHERE session_id = ?
                    ORDER BY sequence DESC, observed_at_ms DESC, heartbeat_cid DESC
                    LIMIT {keep}
                  )
                """,
                [session_id, session_id],
            )
            after_row = connection.execute(
                """
                SELECT COUNT(*) AS heartbeat_count
                FROM session_heartbeats
                WHERE session_id = ?
                """,
                [session_id],
            ).fetchone()
            remaining = _row_int(
                _row_mapping(after_row), "heartbeat_count", "count", default=0
            )
            deleted = max(0, total - remaining)
        except Exception:
            deleted = 0
        if deleted == 0 and total > keep:
            rows = connection.execute(
                """
                SELECT heartbeat_cid, sequence, observed_at_ms
                FROM session_heartbeats
                WHERE session_id = ?
                ORDER BY sequence DESC, observed_at_ms DESC, heartbeat_cid DESC
                """,
                [session_id],
            ).fetchall()
            for row in rows[keep:]:
                mapping = _row_mapping(row)
                heartbeat_cid = str(
                    mapping.get("heartbeat_cid")
                    or mapping.get("HEARTBEAT_CID")
                    or mapping.get("0")
                    or ""
                )
                if not heartbeat_cid:
                    try:
                        heartbeat_cid = str(row[0])  # type: ignore[index]
                    except Exception:
                        continue
                if not heartbeat_cid:
                    continue
                connection.execute(
                    "DELETE FROM session_heartbeats WHERE heartbeat_cid = ?",
                    [heartbeat_cid],
                )
                deleted += 1
        return deleted

    def _insert_heartbeat_locked(
        self,
        connection: Any,
        *,
        session: DaemonSession,
        observed_at_ms: int,
        expires_at_ms: int,
        payload: Mapping[str, Any],
    ) -> Heartbeat:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) AS max_seq
            FROM session_heartbeats WHERE session_id = ?
            """,
            [session.session_id],
        ).fetchone()
        sequence = (
            _row_int(_row_mapping(row), "max_seq", "MAX(sequence)", "coalesce", default=0)
            + 1
        )
        body = _bounded_mapping(
            {
                **dict(payload or {}),
                # Heartbeat markers win over caller payload keys.
                "kind": "heartbeat",
                "is_progress": False,
            },
            name="payload",
            max_bytes=MAX_PAYLOAD_BYTES,
        )
        material = {
            "session_id": session.session_id,
            "fencing_token": int(session.fencing_token),
            "observed_at_ms": int(observed_at_ms),
            "expires_at_ms": int(expires_at_ms),
            "sequence": sequence,
            "payload": body,
        }
        heartbeat_cid = _sha256_hex(_canonical_json(material).encode("utf-8"))
        connection.execute(
            """
            INSERT INTO session_heartbeats(
                heartbeat_cid, session_id, fencing_token, observed_at_ms,
                expires_at_ms, sequence, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                heartbeat_cid,
                session.session_id,
                int(session.fencing_token),
                int(observed_at_ms),
                int(expires_at_ms),
                sequence,
                _canonical_json(body),
            ],
        )
        # Opportunistic compaction for this session.
        self._compact_session_heartbeats_locked(
            connection, session.session_id, keep=self._heartbeat_retain
        )
        return Heartbeat(
            heartbeat_cid=heartbeat_cid,
            session_id=session.session_id,
            fencing_token=int(session.fencing_token),
            observed_at_ms=int(observed_at_ms),
            expires_at_ms=int(expires_at_ms),
            sequence=sequence,
            payload=body,
        )

    def _load_session_locked(
        self, connection: Any, session_id: str
    ) -> DaemonSession | None:
        if not session_id:
            return None
        row = connection.execute(
            """
            SELECT session_id, daemon_id, supervisor_id, process_birth_id,
                   process_birth_json, role, lane_id, shard_id, run_id,
                   server_id, server_generation, fence_epoch, fencing_token,
                   quack_connection, capability_json, attached_at_ms,
                   last_heartbeat_at_ms, expires_at_ms, progress_cursor,
                   progress_updated_at_ms, deadline_ms, status,
                   exit_disposition, restart_disposition, revision,
                   ancestry_json, body_json
            FROM daemon_sessions WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(_row_mapping(row))

    @staticmethod
    def _row_to_supervisor(mapping: Mapping[str, Any]) -> SupervisorInstance:
        birth = process_birth_from_mapping(
            json.loads(str(mapping.get("process_birth_json") or "{}"))
        )
        capability = json.loads(str(mapping.get("capability_json") or "{}"))
        body = json.loads(str(mapping.get("body_json") or "{}"))
        stopped = mapping.get("stopped_at_ms")
        return SupervisorInstance(
            supervisor_id=str(mapping["supervisor_id"]),
            repository_id=str(mapping["repository_id"]),
            process_birth=birth,
            process_birth_id=str(mapping["process_birth_id"]),
            run_id=str(mapping.get("run_id") or ""),
            server_generation=int(mapping.get("server_generation") or 1),
            started_at_ms=int(mapping.get("started_at_ms") or 0),
            stopped_at_ms=None if stopped is None else int(stopped),
            status=InstanceStatus(str(mapping.get("status") or "running")),
            revision=int(mapping.get("revision") or 1),
            capability=capability if isinstance(capability, dict) else {},
            body=body if isinstance(body, dict) else {},
        )

    @staticmethod
    def _row_to_daemon(mapping: Mapping[str, Any]) -> DaemonInstance:
        birth = process_birth_from_mapping(
            json.loads(str(mapping.get("process_birth_json") or "{}"))
        )
        capability = json.loads(str(mapping.get("capability_json") or "{}"))
        body = json.loads(str(mapping.get("body_json") or "{}"))
        stopped = mapping.get("stopped_at_ms")
        return DaemonInstance(
            daemon_id=str(mapping["daemon_id"]),
            supervisor_id=str(mapping["supervisor_id"]),
            process_birth=birth,
            process_birth_id=str(mapping["process_birth_id"]),
            role=str(mapping["role"]),
            lane_id=str(mapping.get("lane_id") or ""),
            shard_id=str(mapping.get("shard_id") or ""),
            run_id=str(mapping.get("run_id") or ""),
            started_at_ms=int(mapping.get("started_at_ms") or 0),
            stopped_at_ms=None if stopped is None else int(stopped),
            status=InstanceStatus(str(mapping.get("status") or "running")),
            revision=int(mapping.get("revision") or 1),
            capability=capability if isinstance(capability, dict) else {},
            body=body if isinstance(body, dict) else {},
        )

    @staticmethod
    def _row_to_session(mapping: Mapping[str, Any]) -> DaemonSession:
        birth = process_birth_from_mapping(
            json.loads(str(mapping.get("process_birth_json") or "{}"))
        )
        capability = json.loads(str(mapping.get("capability_json") or "{}"))
        body = json.loads(str(mapping.get("body_json") or "{}"))
        ancestry = json.loads(str(mapping.get("ancestry_json") or "[]"))
        return DaemonSession(
            session_id=str(mapping["session_id"]),
            daemon_id=str(mapping["daemon_id"]),
            supervisor_id=str(mapping["supervisor_id"]),
            process_birth=birth,
            process_birth_id=str(mapping["process_birth_id"]),
            role=str(mapping["role"]),
            lane_id=str(mapping.get("lane_id") or ""),
            shard_id=str(mapping.get("shard_id") or ""),
            run_id=str(mapping.get("run_id") or ""),
            server_id=str(mapping.get("server_id") or ""),
            server_generation=int(mapping.get("server_generation") or 1),
            fence_epoch=int(mapping.get("fence_epoch") or 1),
            fencing_token=int(mapping.get("fencing_token") or 1),
            quack_connection=str(mapping.get("quack_connection") or ""),
            capability=capability if isinstance(capability, dict) else {},
            attached_at_ms=int(mapping.get("attached_at_ms") or 0),
            last_heartbeat_at_ms=int(mapping.get("last_heartbeat_at_ms") or 0),
            expires_at_ms=int(mapping.get("expires_at_ms") or 0),
            progress_cursor=str(mapping.get("progress_cursor") or ""),
            progress_updated_at_ms=int(mapping.get("progress_updated_at_ms") or 0),
            deadline_ms=int(mapping.get("deadline_ms") or 0),
            status=SessionStatus(str(mapping.get("status") or "active")),
            exit_disposition=ExitDisposition(
                str(mapping.get("exit_disposition") or "running")
            ),
            restart_disposition=RestartDisposition(
                str(mapping.get("restart_disposition") or "none")
            ),
            revision=int(mapping.get("revision") or 1),
            ancestry=tuple(ancestry) if isinstance(ancestry, list) else (),
            body=body if isinstance(body, dict) else {},
        )

    @staticmethod
    def _row_to_heartbeat(mapping: Mapping[str, Any]) -> Heartbeat:
        payload = json.loads(str(mapping.get("payload_json") or "{}"))
        return Heartbeat(
            heartbeat_cid=str(mapping["heartbeat_cid"]),
            session_id=str(mapping["session_id"]),
            fencing_token=int(mapping["fencing_token"]),
            observed_at_ms=int(mapping["observed_at_ms"]),
            expires_at_ms=int(mapping["expires_at_ms"]),
            sequence=int(mapping["sequence"]),
            payload=payload if isinstance(payload, dict) else {},
        )


def open_daemon_registry(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    liveness: LivenessProbe | None = None,
    birth_reader: BirthReader | None = None,
    default_session_ttl_ms: int = DEFAULT_SESSION_TTL_MS,
    default_heartbeat_ttl_ms: int = DEFAULT_HEARTBEAT_TTL_MS,
    heartbeat_retain: int = DEFAULT_HEARTBEAT_RETAIN,
) -> DaemonRegistry:
    """Open and return an initialized :class:`DaemonRegistry`."""

    return DaemonRegistry(
        database_path,
        clock_ms=clock_ms,
        liveness=liveness,
        birth_reader=birth_reader,
        default_session_ttl_ms=default_session_ttl_ms,
        default_heartbeat_ttl_ms=default_heartbeat_ttl_ms,
        heartbeat_retain=heartbeat_retain,
    ).open()


__all__ = (
    "DAEMON_INSTANCE_INTERFACE",
    "DAEMON_INSTANCE_SCHEMA",
    "DAEMON_REGISTRY_INTERFACE",
    "DAEMON_REGISTRY_SCHEMA",
    "DAEMON_SESSION_INTERFACE",
    "DAEMON_SESSION_SCHEMA",
    "DEFAULT_HEARTBEAT_RETAIN",
    "DEFAULT_HEARTBEAT_TTL_MS",
    "DEFAULT_SESSION_TTL_MS",
    "DaemonInstance",
    "DaemonRegistry",
    "DaemonRegistryBoundsError",
    "DaemonRegistryConflictError",
    "DaemonRegistryError",
    "DaemonRegistryIdentityError",
    "DaemonRegistryNotOpenError",
    "DaemonRegistrySessionError",
    "DaemonSession",
    "DuckDBUnavailableError",
    "ExitDisposition",
    "HEARTBEAT_INTERFACE",
    "HEARTBEAT_SCHEMA",
    "Heartbeat",
    "InstanceStatus",
    "PROGRESS_CURSOR_SCHEMA",
    "ProcessBirthIdentity",
    "ProgressCursor",
    "RestartDisposition",
    "STATUS_FILE_MIRROR_SCHEMA",
    "SUPERVISOR_INSTANCE_INTERFACE",
    "SUPERVISOR_INSTANCE_SCHEMA",
    "SessionStatus",
    "StatusFileMirror",
    "SupervisorInstance",
    "duckdb_available",
    "open_daemon_registry",
    "process_birth_id",
    "process_birth_from_mapping",
    "process_births_match",
)
