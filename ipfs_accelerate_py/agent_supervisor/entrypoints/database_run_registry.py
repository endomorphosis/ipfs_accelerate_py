"""DuckDB-backed durable run registry, idempotency, and audit store.

DQP-031 / DatabaseRunRegistry@1
===============================

:class:`DatabaseRunRegistry` is the transactional authority for run roots,
CAS heads, handle snapshots, namespace current pointers, control-plane
idempotency records, and explicit audit rows. Filesystem run trees are export
or compatibility adapters only: a directory scan cannot create a run.

Idempotency contract:

* Exact request replay under the same key returns the prior result.
* Reusing a key with a different request body conflicts and fails closed.
* Lost-response retries with an identical request hash are safe.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    redact_mapping,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_RUN_REGISTRY_INTERFACE: Final[str] = "DatabaseRunRegistry@1"
DATABASE_RUN_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-registry@1"
)
RUN_ROOT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-root@1"
)
RUN_HEAD_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-head@1"
)
RUN_HANDLE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-handle@1"
)
NAMESPACE_CURRENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-namespace-current@1"
)
IDEMPOTENCY_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-idempotency-record@1"
)
RUN_AUDIT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-audit@1"
)
RUN_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-export-receipt@1"
)
DIRECTORY_SCAN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-directory-scan@1"
)

DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:database-run-registry"
AUTHORITY_CLASS: Final[str] = "database_authority"
EXPORT_AUTHORITY: Final[str] = "export_adapter_only"
DIRECTORY_SCAN_AUTHORITY: Final[str] = "scan_non_authoritative"

DEFAULT_PAGE_LIMIT: Final[int] = 256
MAX_PAGE_LIMIT: Final[int] = 4_096
MAX_BODY_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_ID_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 64
MAX_RECURSION_DEPTH: Final[int] = 8

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS run_registry_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS run_roots (
    run_id VARCHAR PRIMARY KEY,
    run_namespace VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    checkout_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    lease_id VARCHAR NOT NULL DEFAULT '',
    target_resolution_receipt_cid VARCHAR NOT NULL DEFAULT '',
    invocation_cid VARCHAR NOT NULL DEFAULT '',
    prompt_cid VARCHAR NOT NULL DEFAULT '',
    objective_cid VARCHAR NOT NULL DEFAULT '',
    lifecycle_profile_cid VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    initial_handle_cid VARCHAR NOT NULL DEFAULT '',
    initial_revision BIGINT NOT NULL DEFAULT 1,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS run_roots_namespace_idx
    ON run_roots(run_namespace, created_at);
CREATE INDEX IF NOT EXISTS run_roots_repository_idx
    ON run_roots(repository_id, created_at);

CREATE TABLE IF NOT EXISTS run_heads (
    run_id VARCHAR PRIMARY KEY,
    run_revision BIGINT NOT NULL,
    handle_cid VARCHAR NOT NULL,
    semantic_id VARCHAR NOT NULL DEFAULT '',
    state VARCHAR NOT NULL,
    health VARCHAR NOT NULL DEFAULT 'unknown',
    event_cursor VARCHAR NOT NULL DEFAULT '',
    updated_at VARCHAR NOT NULL,
    previous_handle_cid VARCHAR NOT NULL DEFAULT '',
    previous_revision BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS run_heads_state_idx
    ON run_heads(state, updated_at);

CREATE TABLE IF NOT EXISTS run_handles (
    handle_cid VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    run_revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS run_handles_run_idx
    ON run_handles(run_id, run_revision);

CREATE TABLE IF NOT EXISTS namespace_current (
    run_namespace VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL DEFAULT '',
    checkout_id VARCHAR NOT NULL DEFAULT '',
    selected_run_id VARCHAR NOT NULL DEFAULT '',
    integrity_cid VARCHAR NOT NULL DEFAULT '',
    pointer_revision BIGINT NOT NULL DEFAULT 1,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS idempotency_records (
    record_id VARCHAR PRIMARY KEY,
    idempotency_key VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    caller VARCHAR NOT NULL DEFAULT '',
    repository_id VARCHAR NOT NULL DEFAULT '',
    objective_id VARCHAR NOT NULL DEFAULT '',
    request_digest VARCHAR NOT NULL,
    request_json VARCHAR NOT NULL,
    result_json VARCHAR NOT NULL,
    result_digest VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'committed',
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idempotency_records_key_uidx
    ON idempotency_records(idempotency_key, operation, caller, repository_id);
CREATE INDEX IF NOT EXISTS idempotency_records_digest_idx
    ON idempotency_records(request_digest);

CREATE TABLE IF NOT EXISTS run_audit_records (
    audit_id VARCHAR PRIMARY KEY,
    actor_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    redacted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS run_audit_subject_idx
    ON run_audit_records(subject_kind, subject_id, recorded_at);
CREATE INDEX IF NOT EXISTS run_audit_run_idx
    ON run_audit_records(run_id, recorded_at);

CREATE TABLE IF NOT EXISTS run_exports (
    export_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    target_path VARCHAR NOT NULL,
    run_count BIGINT NOT NULL DEFAULT 0,
    export_digest VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'export_adapter_only',
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseRunRegistryError(RuntimeError):
    """Base error for database run-registry failures."""


class DatabaseRunRegistryNotOpenError(DatabaseRunRegistryError):
    """Operation requires an open registry."""


class DatabaseRunRegistryBoundsError(DatabaseRunRegistryError, ValueError):
    """A payload, page, or identity bound was exceeded."""


class DatabaseRunRegistryConflictError(DatabaseRunRegistryError):
    """CAS, identity, or idempotency conflict."""


class DatabaseRunRegistryNotFoundError(DatabaseRunRegistryError, LookupError):
    """Requested run or record is absent."""


class DatabaseRunRegistryIntegrityError(DatabaseRunRegistryError, ValueError):
    """Digest, schema, or payload integrity failure."""


class DuckDBUnavailableError(DatabaseRunRegistryError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RunLifecycleState(str, Enum):
    RECEIVED = "received"
    STARTING = "starting"
    RUNNING = "running"
    DRAINED = "drained"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    QUARANTINED = "quarantined"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @classmethod
    def coerce(cls, value: Any) -> "RunLifecycleState":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        for item in cls:
            if item.value == raw:
                return item
        raise DatabaseRunRegistryBoundsError(
            f"unsupported run lifecycle state: {value!r}"
        )


class RunHealthState(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    TERMINAL = "terminal"

    @classmethod
    def coerce(cls, value: Any) -> "RunHealthState":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        for item in cls:
            if item.value == raw:
                return item
        raise DatabaseRunRegistryBoundsError(
            f"unsupported run health state: {value!r}"
        )


class AuditAction(str, Enum):
    CREATE = "create"
    CAS_UPDATE = "cas_update"
    SET_CURRENT = "set_current"
    IDEMPOTENCY_COMMIT = "idempotency_commit"
    IDEMPOTENCY_REPLAY = "idempotency_replay"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    EXPORT = "export"
    DIRECTORY_SCAN = "directory_scan"
    LOOKUP = "lookup"
    REDACT = "redact"

    @classmethod
    def coerce(cls, value: Any) -> "AuditAction":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        for item in cls:
            if item.value == raw:
                return item
        raise DatabaseRunRegistryBoundsError(f"unsupported audit action: {value!r}")


class IdempotencyStatus(str, Enum):
    COMMITTED = "committed"
    REPLAYED = "replayed"
    CONFLICT = "conflict"


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


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseRunRegistryError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseRunRegistryError(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise DatabaseRunRegistryBoundsError(f"{name} exceeds {maximum} bytes")
    return text


def _safe_id(value: Any, name: str) -> str:
    text = _text(value, name, maximum=MAX_ID_BYTES)
    if not _SAFE_ID_RE.match(text):
        raise DatabaseRunRegistryBoundsError(f"{name} is not a closed identity token")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseRunRegistryBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseRunRegistryBoundsError(f"{name} must be a positive integer")
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


def _bounded_body(
    body: Mapping[str, Any] | None,
    *,
    redact: bool,
    depth: int = 0,
    name: str = "body",
) -> dict[str, Any]:
    if depth > MAX_RECURSION_DEPTH:
        raise DatabaseRunRegistryBoundsError(
            f"{name} exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    raw = dict(body or {})
    cleaned = redact_mapping(raw) if redact else raw
    if not isinstance(cleaned, dict):
        raise DatabaseRunRegistryError(f"{name} must project to an object")
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise DatabaseRunRegistryBoundsError(
            f"{name} exceeds the {MAX_BODY_BYTES}-byte bound"
        )
    return cleaned


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


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


def _identity(prefix: str, material: Mapping[str, Any]) -> str:
    digest = _sha256_hex(_canonical_json(dict(material)).encode("utf-8"))
    return f"{prefix}:{digest[7:39]}"


def _reason_codes(values: Sequence[str] | None) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in values or ():
        text = str(item).strip()
        if not text or text in cleaned:
            continue
        cleaned.append(text)
        if len(cleaned) >= MAX_REASON_CODES:
            break
    return tuple(cleaned)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRootRecord:
    """Immutable identity binding for one durable run."""

    run_id: str
    run_namespace: str
    repository_id: str
    created_at: str
    content_digest: str
    checkout_id: str = ""
    worktree_id: str = ""
    session_id: str = ""
    lease_id: str = ""
    target_resolution_receipt_cid: str = ""
    invocation_cid: str = ""
    prompt_cid: str = ""
    objective_cid: str = ""
    lifecycle_profile_cid: str = ""
    initial_handle_cid: str = ""
    initial_revision: int = 1
    body: Mapping[str, Any] = MappingProxyType({})
    schema: str = RUN_ROOT_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "run_namespace": self.run_namespace,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "worktree_id": self.worktree_id,
            "session_id": self.session_id,
            "lease_id": self.lease_id,
            "target_resolution_receipt_cid": self.target_resolution_receipt_cid,
            "invocation_cid": self.invocation_cid,
            "prompt_cid": self.prompt_cid,
            "objective_cid": self.objective_cid,
            "lifecycle_profile_cid": self.lifecycle_profile_cid,
            "created_at": self.created_at,
            "initial_handle_cid": self.initial_handle_cid,
            "initial_revision": self.initial_revision,
            "body": dict(self.body),
            "content_digest": self.content_digest,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class RunHeadRecord:
    """Mutable CAS head for one run."""

    run_id: str
    run_revision: int
    handle_cid: str
    state: str
    updated_at: str
    content_digest: str
    semantic_id: str = ""
    health: str = RunHealthState.UNKNOWN.value
    event_cursor: str = ""
    previous_handle_cid: str = ""
    previous_revision: int = 0
    body: Mapping[str, Any] = MappingProxyType({})
    schema: str = RUN_HEAD_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "run_revision": self.run_revision,
            "handle_cid": self.handle_cid,
            "semantic_id": self.semantic_id,
            "state": self.state,
            "health": self.health,
            "event_cursor": self.event_cursor,
            "updated_at": self.updated_at,
            "previous_handle_cid": self.previous_handle_cid,
            "previous_revision": self.previous_revision,
            "body": dict(self.body),
            "content_digest": self.content_digest,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class IdempotencyRecord:
    """Durable control result bound to one caller-scoped idempotency key."""

    record_id: str
    idempotency_key: str
    operation: str
    request_digest: str
    result: Mapping[str, Any]
    result_digest: str
    created_at: str
    updated_at: str
    caller: str = ""
    repository_id: str = ""
    objective_id: str = ""
    status: str = IdempotencyStatus.COMMITTED.value
    request: Mapping[str, Any] = MappingProxyType({})
    schema: str = IDEMPOTENCY_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_id": self.record_id,
            "idempotency_key": self.idempotency_key,
            "operation": self.operation,
            "caller": self.caller,
            "repository_id": self.repository_id,
            "objective_id": self.objective_id,
            "request_digest": self.request_digest,
            "request": dict(self.request),
            "result": dict(self.result),
            "result_digest": self.result_digest,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class DirectoryScanReceipt:
    """Non-authoritative observation of a filesystem run tree."""

    scan_id: str
    directory: str
    observed_run_ids: tuple[str, ...]
    observed_count: int
    created_at: str
    authority: str = DIRECTORY_SCAN_AUTHORITY
    creates_runs: bool = False
    schema: str = DIRECTORY_SCAN_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "scan_id": self.scan_id,
            "directory": self.directory,
            "observed_run_ids": list(self.observed_run_ids),
            "observed_count": self.observed_count,
            "created_at": self.created_at,
            "authority": self.authority,
            "creates_runs": self.creates_runs,
            "authoritative": False,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseRunRegistry:
    """DuckDB authority for runs, CAS heads, idempotency, and audit."""

    INTERFACE: Final[str] = DATABASE_RUN_REGISTRY_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        auto_redact: bool = True,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseRunRegistry; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._snapshot_id = _text(snapshot_id, "snapshot_id")
        self._auto_redact = bool(auto_redact)
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseRunRegistry":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_RUN_REGISTRY_INTERFACE),
                ("schema", DATABASE_RUN_REGISTRY_SCHEMA),
                ("snapshot_id", self._snapshot_id),
                ("authority", AUTHORITY_CLASS),
                ("directory_scan_creates_runs", "false"),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO run_registry_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

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

    def __enter__(self) -> "DatabaseRunRegistry":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseRunRegistryNotOpenError("DatabaseRunRegistry is not open")
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    def metadata(self) -> dict[str, str]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                "SELECT key, value FROM run_registry_metadata ORDER BY key"
            ).fetchall()
            return {
                str(_row_mapping(row)["key"]): str(_row_mapping(row)["value"])
                for row in rows
            }

    # -- create / cas --------------------------------------------------------

    def create_run(
        self,
        *,
        run_id: str | None = None,
        run_namespace: str,
        repository_id: str,
        checkout_id: str = "",
        worktree_id: str = "",
        session_id: str = "",
        lease_id: str = "",
        target_resolution_receipt_cid: str = "",
        invocation_cid: str = "",
        prompt_cid: str = "",
        objective_cid: str = "",
        lifecycle_profile_cid: str = "",
        state: RunLifecycleState | str = RunLifecycleState.RECEIVED,
        health: RunHealthState | str = RunHealthState.UNKNOWN,
        handle: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        actor_id: str = "system",
        created_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Explicitly create one durable run root + initial head.

        Directory scans and filesystem observation never call this path.
        """

        namespace = _safe_id(run_namespace, "run_namespace")
        repo = _safe_id(repository_id, "repository_id")
        selected_state = RunLifecycleState.coerce(state)
        selected_health = RunHealthState.coerce(health)
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        handle_body = _bounded_body(handle, redact=do_redact, name="handle")
        stamp = _text(created_at or _utc_iso(), "created_at")
        material = {
            "run_namespace": namespace,
            "repository_id": repo,
            "checkout_id": _text(checkout_id, "checkout_id", required=False),
            "worktree_id": _text(worktree_id, "worktree_id", required=False),
            "session_id": _text(session_id, "session_id", required=False),
            "lease_id": _text(lease_id, "lease_id", required=False),
            "target_resolution_receipt_cid": _text(
                target_resolution_receipt_cid,
                "target_resolution_receipt_cid",
                required=False,
            ),
            "invocation_cid": _text(invocation_cid, "invocation_cid", required=False),
            "prompt_cid": _text(prompt_cid, "prompt_cid", required=False),
            "objective_cid": _text(objective_cid, "objective_cid", required=False),
            "lifecycle_profile_cid": _text(
                lifecycle_profile_cid, "lifecycle_profile_cid", required=False
            ),
            "created_at": stamp,
            "body": payload,
            "handle": handle_body,
        }
        selected_run_id = _safe_id(
            run_id or _identity("run", material),
            "run_id",
        )
        handle_material = {
            "run_id": selected_run_id,
            "run_revision": 1,
            "state": selected_state.value,
            "health": selected_health.value,
            "handle": handle_body,
            "body": payload,
        }
        handle_cid = _identity("handle", handle_material)
        root_digest = _sha256_hex(
            _canonical_json({**material, "run_id": selected_run_id}).encode("utf-8")
        )
        head_digest = _sha256_hex(
            _canonical_json(
                {
                    "run_id": selected_run_id,
                    "run_revision": 1,
                    "handle_cid": handle_cid,
                    "state": selected_state.value,
                    "health": selected_health.value,
                }
            ).encode("utf-8")
        )

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT run_id FROM run_roots WHERE run_id = ?",
                [selected_run_id],
            ).fetchone()
            if existing is not None:
                raise DatabaseRunRegistryConflictError(
                    f"run already registered: {selected_run_id}"
                )
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    """
                    INSERT INTO run_roots(
                        run_id, run_namespace, repository_id, checkout_id,
                        worktree_id, session_id, lease_id,
                        target_resolution_receipt_cid, invocation_cid,
                        prompt_cid, objective_cid, lifecycle_profile_cid,
                        created_at, initial_handle_cid, initial_revision,
                        body_json, content_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        selected_run_id,
                        namespace,
                        repo,
                        material["checkout_id"],
                        material["worktree_id"],
                        material["session_id"],
                        material["lease_id"],
                        material["target_resolution_receipt_cid"],
                        material["invocation_cid"],
                        material["prompt_cid"],
                        material["objective_cid"],
                        material["lifecycle_profile_cid"],
                        stamp,
                        handle_cid,
                        1,
                        _canonical_json(payload),
                        root_digest,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO run_handles(
                        handle_cid, run_id, run_revision, body_json,
                        content_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        handle_cid,
                        selected_run_id,
                        1,
                        _canonical_json(handle_body or handle_material),
                        _sha256_hex(
                            _canonical_json(handle_body or handle_material).encode(
                                "utf-8"
                            )
                        ),
                        stamp,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO run_heads(
                        run_id, run_revision, handle_cid, semantic_id, state,
                        health, event_cursor, updated_at, previous_handle_cid,
                        previous_revision, body_json, content_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        selected_run_id,
                        1,
                        handle_cid,
                        handle_cid,
                        selected_state.value,
                        selected_health.value,
                        "",
                        stamp,
                        "",
                        0,
                        _canonical_json({"created": True}),
                        head_digest,
                    ],
                )
                self._insert_audit(
                    connection,
                    actor_id=actor_id,
                    action=AuditAction.CREATE,
                    subject_kind="run",
                    subject_id=selected_run_id,
                    run_id=selected_run_id,
                    body={
                        "run_namespace": namespace,
                        "repository_id": repo,
                        "handle_cid": handle_cid,
                        "revision": 1,
                    },
                    recorded_at=stamp,
                    redact=do_redact,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return self.get_run(selected_run_id)

    def cas_update(
        self,
        run_id: str,
        *,
        expected_revision: int,
        state: RunLifecycleState | str | None = None,
        health: RunHealthState | str | None = None,
        handle: Mapping[str, Any] | None = None,
        event_cursor: str = "",
        body: Mapping[str, Any] | None = None,
        actor_id: str = "system",
        updated_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Compare-and-swap the run head to the next revision."""

        selected_run_id = _safe_id(run_id, "run_id")
        expected = _nonneg_int(expected_revision, "expected_revision")
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        handle_body = _bounded_body(handle, redact=do_redact, name="handle")
        stamp = _text(updated_at or _utc_iso(), "updated_at")

        with self._lock:
            connection = self._require()
            head_row = connection.execute(
                "SELECT * FROM run_heads WHERE run_id = ?",
                [selected_run_id],
            ).fetchone()
            if head_row is None:
                raise DatabaseRunRegistryNotFoundError(
                    f"run head not found: {selected_run_id}"
                )
            head = self._head_from_row(_row_mapping(head_row))
            if head.run_revision != expected:
                raise DatabaseRunRegistryConflictError(
                    "CAS conflict: expected revision does not match head"
                )
            next_revision = head.run_revision + 1
            selected_state = (
                RunLifecycleState.coerce(state)
                if state is not None
                else RunLifecycleState.coerce(head.state)
            )
            selected_health = (
                RunHealthState.coerce(health)
                if health is not None
                else RunHealthState.coerce(head.health)
            )
            handle_material = {
                "run_id": selected_run_id,
                "run_revision": next_revision,
                "state": selected_state.value,
                "health": selected_health.value,
                "handle": handle_body,
                "body": payload,
                "previous_handle_cid": head.handle_cid,
            }
            handle_cid = _identity("handle", handle_material)
            head_digest = _sha256_hex(
                _canonical_json(
                    {
                        "run_id": selected_run_id,
                        "run_revision": next_revision,
                        "handle_cid": handle_cid,
                        "state": selected_state.value,
                        "health": selected_health.value,
                        "previous_revision": head.run_revision,
                    }
                ).encode("utf-8")
            )
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    """
                    INSERT INTO run_handles(
                        handle_cid, run_id, run_revision, body_json,
                        content_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        handle_cid,
                        selected_run_id,
                        next_revision,
                        _canonical_json(handle_body or handle_material),
                        _sha256_hex(
                            _canonical_json(handle_body or handle_material).encode(
                                "utf-8"
                            )
                        ),
                        stamp,
                    ],
                )
                connection.execute(
                    """
                    UPDATE run_heads
                    SET run_revision = ?,
                        handle_cid = ?,
                        semantic_id = ?,
                        state = ?,
                        health = ?,
                        event_cursor = ?,
                        updated_at = ?,
                        previous_handle_cid = ?,
                        previous_revision = ?,
                        body_json = ?,
                        content_digest = ?
                    WHERE run_id = ? AND run_revision = ?
                    """,
                    [
                        next_revision,
                        handle_cid,
                        handle_cid,
                        selected_state.value,
                        selected_health.value,
                        _text(event_cursor, "event_cursor", required=False),
                        stamp,
                        head.handle_cid,
                        head.run_revision,
                        _canonical_json(payload),
                        head_digest,
                        selected_run_id,
                        expected,
                    ],
                )
                self._insert_audit(
                    connection,
                    actor_id=actor_id,
                    action=AuditAction.CAS_UPDATE,
                    subject_kind="run",
                    subject_id=selected_run_id,
                    run_id=selected_run_id,
                    body={
                        "previous_revision": head.run_revision,
                        "run_revision": next_revision,
                        "handle_cid": handle_cid,
                        "state": selected_state.value,
                    },
                    recorded_at=stamp,
                    redact=do_redact,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return self.get_run(selected_run_id)

    def set_current(
        self,
        *,
        run_namespace: str,
        run_id: str,
        repository_id: str = "",
        checkout_id: str = "",
        expected_pointer_revision: int | None = None,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """CAS the namespace current-run pointer onto an existing run."""

        namespace = _safe_id(run_namespace, "run_namespace")
        selected_run_id = _safe_id(run_id, "run_id")
        stamp = _utc_iso()
        with self._lock:
            connection = self._require()
            root = connection.execute(
                "SELECT * FROM run_roots WHERE run_id = ?",
                [selected_run_id],
            ).fetchone()
            if root is None:
                raise DatabaseRunRegistryNotFoundError(
                    f"run not found: {selected_run_id}"
                )
            root_map = _row_mapping(root)
            if str(root_map["run_namespace"]) != namespace:
                raise DatabaseRunRegistryConflictError(
                    "run_namespace does not match the selected run"
                )
            head = connection.execute(
                "SELECT handle_cid, content_digest FROM run_heads WHERE run_id = ?",
                [selected_run_id],
            ).fetchone()
            if head is None:
                raise DatabaseRunRegistryIntegrityError(
                    f"run head missing for {selected_run_id}"
                )
            head_map = _row_mapping(head)
            current = connection.execute(
                "SELECT * FROM namespace_current WHERE run_namespace = ?",
                [namespace],
            ).fetchone()
            if current is None:
                pointer_revision = 1
                if expected_pointer_revision is not None and expected_pointer_revision != 0:
                    raise DatabaseRunRegistryConflictError(
                        "namespace current pointer CAS conflict"
                    )
            else:
                current_map = _row_mapping(current)
                current_rev = int(current_map["pointer_revision"])
                if (
                    expected_pointer_revision is not None
                    and expected_pointer_revision != current_rev
                ):
                    raise DatabaseRunRegistryConflictError(
                        "namespace current pointer CAS conflict"
                    )
                pointer_revision = current_rev + 1
            integrity = str(head_map["content_digest"])
            connection.execute(
                """
                INSERT OR REPLACE INTO namespace_current(
                    run_namespace, repository_id, checkout_id, selected_run_id,
                    integrity_cid, pointer_revision, updated_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    namespace,
                    _text(
                        repository_id or root_map["repository_id"],
                        "repository_id",
                        required=False,
                    ),
                    _text(
                        checkout_id or root_map.get("checkout_id") or "",
                        "checkout_id",
                        required=False,
                    ),
                    selected_run_id,
                    integrity,
                    pointer_revision,
                    stamp,
                    _canonical_json({"selected": True}),
                ],
            )
            self._insert_audit(
                connection,
                actor_id=actor_id,
                action=AuditAction.SET_CURRENT,
                subject_kind="namespace",
                subject_id=namespace,
                run_id=selected_run_id,
                body={
                    "pointer_revision": pointer_revision,
                    "integrity_cid": integrity,
                },
                recorded_at=stamp,
                redact=False,
            )
            self._commit_if_idle(connection)
            return {
                "schema": NAMESPACE_CURRENT_SCHEMA,
                "run_namespace": namespace,
                "selected_run_id": selected_run_id,
                "integrity_cid": integrity,
                "pointer_revision": pointer_revision,
                "updated_at": stamp,
                "authority": AUTHORITY_CLASS,
            }

    # -- idempotency ---------------------------------------------------------

    def commit_idempotent_result(
        self,
        *,
        idempotency_key: str,
        operation: str,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
        caller: str = "",
        repository_id: str = "",
        objective_id: str = "",
        actor_id: str = "system",
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Commit or replay a control result under a caller-scoped key.

        Exact request digest replay returns the prior result. A different
        request under the same key raises :class:`DatabaseRunRegistryConflictError`.
        """

        key = _safe_id(idempotency_key, "idempotency_key")
        op = _safe_id(operation, "operation")
        selected_caller = _text(caller, "caller", required=False)
        selected_repo = _text(repository_id, "repository_id", required=False)
        selected_objective = _text(objective_id, "objective_id", required=False)
        do_redact = self._auto_redact if redact is None else bool(redact)
        request_body = _bounded_body(request, redact=do_redact, name="request")
        result_body = _bounded_body(result, redact=do_redact, name="result")
        request_digest = _sha256_hex(_canonical_json(request_body).encode("utf-8"))
        result_digest = _sha256_hex(_canonical_json(result_body).encode("utf-8"))
        stamp = _utc_iso()
        scope_material = {
            "idempotency_key": key,
            "operation": op,
            "caller": selected_caller,
            "repository_id": selected_repo,
        }

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                """
                SELECT * FROM idempotency_records
                WHERE idempotency_key = ?
                  AND operation = ?
                  AND caller = ?
                  AND repository_id = ?
                LIMIT 1
                """,
                [key, op, selected_caller, selected_repo],
            ).fetchone()
            if existing is not None:
                record = self._idempotency_from_row(_row_mapping(existing))
                if record.request_digest != request_digest:
                    self._insert_audit(
                        connection,
                        actor_id=actor_id,
                        action=AuditAction.IDEMPOTENCY_CONFLICT,
                        subject_kind="idempotency",
                        subject_id=key,
                        run_id="",
                        body={
                            "operation": op,
                            "existing_request_digest": record.request_digest,
                            "incoming_request_digest": request_digest,
                        },
                        recorded_at=stamp,
                        redact=do_redact,
                    )
                    self._commit_if_idle(connection)
                    raise DatabaseRunRegistryConflictError(
                        "idempotency key is already bound to a different request"
                    )
                # Exact replay: return prior result without mutation.
                connection.execute(
                    """
                    UPDATE idempotency_records
                    SET status = ?, updated_at = ?
                    WHERE record_id = ?
                    """,
                    [IdempotencyStatus.REPLAYED.value, stamp, record.record_id],
                )
                self._insert_audit(
                    connection,
                    actor_id=actor_id,
                    action=AuditAction.IDEMPOTENCY_REPLAY,
                    subject_kind="idempotency",
                    subject_id=key,
                    run_id="",
                    body={
                        "operation": op,
                        "record_id": record.record_id,
                        "request_digest": request_digest,
                    },
                    recorded_at=stamp,
                    redact=do_redact,
                )
                self._commit_if_idle(connection)
                replayed = self._idempotency_from_row(
                    _row_mapping(
                        connection.execute(
                            "SELECT * FROM idempotency_records WHERE record_id = ?",
                            [record.record_id],
                        ).fetchone()
                    )
                )
                payload = replayed.to_dict()
                payload["replayed"] = True
                return payload

            record_id = _identity(
                "idem",
                {**scope_material, "request_digest": request_digest},
            )
            connection.execute(
                """
                INSERT INTO idempotency_records(
                    record_id, idempotency_key, operation, caller,
                    repository_id, objective_id, request_digest, request_json,
                    result_json, result_digest, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record_id,
                    key,
                    op,
                    selected_caller,
                    selected_repo,
                    selected_objective,
                    request_digest,
                    _canonical_json(request_body),
                    _canonical_json(result_body),
                    result_digest,
                    IdempotencyStatus.COMMITTED.value,
                    stamp,
                    stamp,
                ],
            )
            self._insert_audit(
                connection,
                actor_id=actor_id,
                action=AuditAction.IDEMPOTENCY_COMMIT,
                subject_kind="idempotency",
                subject_id=key,
                run_id="",
                body={
                    "operation": op,
                    "record_id": record_id,
                    "request_digest": request_digest,
                    "result_digest": result_digest,
                },
                recorded_at=stamp,
                redact=do_redact,
            )
            self._commit_if_idle(connection)
            payload = IdempotencyRecord(
                record_id=record_id,
                idempotency_key=key,
                operation=op,
                caller=selected_caller,
                repository_id=selected_repo,
                objective_id=selected_objective,
                request_digest=request_digest,
                request=MappingProxyType(request_body),
                result=MappingProxyType(result_body),
                result_digest=result_digest,
                status=IdempotencyStatus.COMMITTED.value,
                created_at=stamp,
                updated_at=stamp,
            ).to_dict()
            payload["replayed"] = False
            return payload

    def get_idempotency(
        self,
        *,
        idempotency_key: str,
        operation: str,
        caller: str = "",
        repository_id: str = "",
    ) -> dict[str, Any] | None:
        key = _safe_id(idempotency_key, "idempotency_key")
        op = _safe_id(operation, "operation")
        selected_caller = _text(caller, "caller", required=False)
        selected_repo = _text(repository_id, "repository_id", required=False)
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT * FROM idempotency_records
                WHERE idempotency_key = ?
                  AND operation = ?
                  AND caller = ?
                  AND repository_id = ?
                LIMIT 1
                """,
                [key, op, selected_caller, selected_repo],
            ).fetchone()
            if row is None:
                return None
            return self._idempotency_from_row(_row_mapping(row)).to_dict()

    # -- reads ---------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any]:
        selected = _safe_id(run_id, "run_id")
        with self._lock:
            connection = self._require()
            root_row = connection.execute(
                "SELECT * FROM run_roots WHERE run_id = ?",
                [selected],
            ).fetchone()
            if root_row is None:
                raise DatabaseRunRegistryNotFoundError(f"run not found: {selected}")
            head_row = connection.execute(
                "SELECT * FROM run_heads WHERE run_id = ?",
                [selected],
            ).fetchone()
            if head_row is None:
                raise DatabaseRunRegistryIntegrityError(
                    f"run head missing for {selected}"
                )
            root = self._root_from_row(_row_mapping(root_row))
            head = self._head_from_row(_row_mapping(head_row))
            return {
                "schema": DATABASE_RUN_REGISTRY_SCHEMA,
                "authority": AUTHORITY_CLASS,
                "root": root.to_dict(),
                "head": head.to_dict(),
            }

    def list_runs(
        self,
        *,
        run_namespace: str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[dict[str, Any], ...]:
        page_limit = min(_positive_int(limit, "limit"), MAX_PAGE_LIMIT)
        page_offset = _nonneg_int(offset, "offset")
        with self._lock:
            connection = self._require()
            if run_namespace is not None:
                namespace = _safe_id(run_namespace, "run_namespace")
                rows = connection.execute(
                    """
                    SELECT r.run_id
                    FROM run_roots r
                    WHERE r.run_namespace = ?
                    ORDER BY r.created_at ASC, r.run_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    [namespace, page_limit, page_offset],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT r.run_id
                    FROM run_roots r
                    ORDER BY r.created_at ASC, r.run_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    [page_limit, page_offset],
                ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                run_id = str(_row_mapping(row)["run_id"])
                root_row = connection.execute(
                    "SELECT * FROM run_roots WHERE run_id = ?",
                    [run_id],
                ).fetchone()
                head_row = connection.execute(
                    "SELECT * FROM run_heads WHERE run_id = ?",
                    [run_id],
                ).fetchone()
                if root_row is None or head_row is None:
                    continue
                results.append(
                    {
                        "schema": DATABASE_RUN_REGISTRY_SCHEMA,
                        "authority": AUTHORITY_CLASS,
                        "root": self._root_from_row(_row_mapping(root_row)).to_dict(),
                        "head": self._head_from_row(_row_mapping(head_row)).to_dict(),
                    }
                )
            return tuple(results)

    def get_current(self, run_namespace: str) -> dict[str, Any] | None:
        namespace = _safe_id(run_namespace, "run_namespace")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM namespace_current WHERE run_namespace = ?",
                [namespace],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            return {
                "schema": NAMESPACE_CURRENT_SCHEMA,
                "run_namespace": str(mapping["run_namespace"]),
                "repository_id": str(mapping.get("repository_id") or ""),
                "checkout_id": str(mapping.get("checkout_id") or ""),
                "selected_run_id": str(mapping.get("selected_run_id") or ""),
                "integrity_cid": str(mapping.get("integrity_cid") or ""),
                "pointer_revision": int(mapping["pointer_revision"]),
                "updated_at": str(mapping["updated_at"]),
                "authority": AUTHORITY_CLASS,
            }

    def list_audits(
        self,
        *,
        subject_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        page_limit = min(_positive_int(limit, "limit"), MAX_PAGE_LIMIT)
        with self._lock:
            connection = self._require()
            if subject_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM run_audit_records
                    WHERE subject_id = ?
                    ORDER BY recorded_at ASC
                    LIMIT ?
                    """,
                    [_text(subject_id, "subject_id"), page_limit],
                ).fetchall()
            elif run_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM run_audit_records
                    WHERE run_id = ?
                    ORDER BY recorded_at ASC
                    LIMIT ?
                    """,
                    [_safe_id(run_id, "run_id"), page_limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM run_audit_records
                    ORDER BY recorded_at ASC
                    LIMIT ?
                    """,
                    [page_limit],
                ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                mapping = _row_mapping(row)
                body = json.loads(str(mapping.get("body_json") or "{}"))
                results.append(
                    {
                        "schema": RUN_AUDIT_RECORD_SCHEMA,
                        "audit_id": mapping["audit_id"],
                        "actor_id": mapping["actor_id"],
                        "action": mapping["action"],
                        "subject_kind": mapping["subject_kind"],
                        "subject_id": mapping["subject_id"],
                        "run_id": mapping.get("run_id") or "",
                        "recorded_at": mapping["recorded_at"],
                        "body": body,
                        "redacted": bool(mapping.get("redacted")),
                    }
                )
            return tuple(results)

    # -- directory scan (non-authoritative) ----------------------------------

    def scan_directory(
        self,
        directory: Path | str,
        *,
        actor_id: str = "system",
    ) -> DirectoryScanReceipt:
        """Observe a filesystem run tree without creating any database run.

        Acceptance: directory scan cannot create a run. This method never
        inserts into ``run_roots`` / ``run_heads``.
        """

        root = Path(directory)
        observed: list[str] = []
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                # Compatibility layout: runs/<run_id>/ or bare <run_id>/.
                if child.name == "runs" and child.is_dir():
                    for run_dir in sorted(child.iterdir()):
                        if run_dir.is_dir() and not run_dir.name.startswith("."):
                            observed.append(run_dir.name)
                else:
                    marker = child / "root.json"
                    head = child / "head.json"
                    if marker.is_file() or head.is_file() or child.name.startswith("run"):
                        observed.append(child.name)
        stamp = _utc_iso()
        scan_id = _identity(
            "scan",
            {
                "directory": str(root),
                "observed_run_ids": observed,
                "created_at": stamp,
            },
        )
        receipt = DirectoryScanReceipt(
            scan_id=scan_id,
            directory=str(root),
            observed_run_ids=tuple(observed),
            observed_count=len(observed),
            created_at=stamp,
        )
        with self._lock:
            connection = self._require()
            self._insert_audit(
                connection,
                actor_id=actor_id,
                action=AuditAction.DIRECTORY_SCAN,
                subject_kind="directory",
                subject_id=str(root),
                run_id="",
                body={
                    "scan_id": scan_id,
                    "observed_count": len(observed),
                    "creates_runs": False,
                    "authority": DIRECTORY_SCAN_AUTHORITY,
                },
                recorded_at=stamp,
                redact=False,
            )
            self._commit_if_idle(connection)
        return receipt

    def import_from_directory_scan(
        self,
        directory: Path | str,
        *,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """Explicitly refuse automatic creation from directory observation.

        Operators must call :meth:`create_run` with typed fields. This method
        only returns the non-authoritative scan receipt and a closed refusal.
        """

        receipt = self.scan_directory(directory, actor_id=actor_id)
        return {
            "schema": DIRECTORY_SCAN_RECEIPT_SCHEMA,
            "accepted": False,
            "reason_codes": ["directory_scan_cannot_create_run"],
            "creates_runs": False,
            "authority": DIRECTORY_SCAN_AUTHORITY,
            "scan": receipt.to_dict(),
        }

    # -- export (non-authoritative) ------------------------------------------

    def export_runs(
        self,
        target_path: Path | str,
        *,
        run_namespace: str | None = None,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """Render admitted runs to JSON. Exports never grant authority."""

        destination = Path(target_path)
        runs = self.list_runs(run_namespace=run_namespace, limit=MAX_PAGE_LIMIT)
        payload = {
            "schema": RUN_EXPORT_RECEIPT_SCHEMA,
            "snapshot_id": self._snapshot_id,
            "authority": EXPORT_AUTHORITY,
            "authoritative": False,
            "runs": [dict(item) for item in runs],
        }
        encoded = _canonical_json(payload).encode("utf-8")
        digest = _sha256_hex(encoded)
        stamp = _utc_iso()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        export_id = _identity(
            "export",
            {
                "target_path": str(destination),
                "export_digest": digest,
                "run_count": len(runs),
            },
        )
        with self._lock:
            connection = self._require()
            connection.execute(
                """
                INSERT INTO run_exports(
                    export_id, snapshot_id, target_path, run_count,
                    export_digest, authority, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    export_id,
                    self._snapshot_id,
                    str(destination),
                    len(runs),
                    digest,
                    EXPORT_AUTHORITY,
                    stamp,
                    _canonical_json({"run_namespace": run_namespace or ""}),
                ],
            )
            self._insert_audit(
                connection,
                actor_id=actor_id,
                action=AuditAction.EXPORT,
                subject_kind="export",
                subject_id=export_id,
                run_id="",
                body={
                    "target_path": str(destination),
                    "run_count": len(runs),
                    "export_digest": digest,
                    "authority": EXPORT_AUTHORITY,
                },
                recorded_at=stamp,
                redact=False,
            )
            self._commit_if_idle(connection)
        return {
            "schema": RUN_EXPORT_RECEIPT_SCHEMA,
            "export_id": export_id,
            "snapshot_id": self._snapshot_id,
            "target_path": str(destination),
            "run_count": len(runs),
            "export_digest": digest,
            "authority": EXPORT_AUTHORITY,
            "authoritative": False,
            "created_at": stamp,
        }

    # -- internal helpers ----------------------------------------------------

    def _insert_audit(
        self,
        connection: Any,
        *,
        actor_id: str,
        action: AuditAction | str,
        subject_kind: str,
        subject_id: str,
        run_id: str,
        body: Mapping[str, Any],
        recorded_at: str,
        redact: bool,
    ) -> None:
        selected_action = AuditAction.coerce(action)
        payload = _bounded_body(body, redact=redact, name="audit_body")
        material = {
            "actor_id": _text(actor_id, "actor_id"),
            "action": selected_action.value,
            "subject_kind": _text(subject_kind, "subject_kind"),
            "subject_id": _text(subject_id, "subject_id"),
            "run_id": _text(run_id, "run_id", required=False),
            "recorded_at": recorded_at,
            "body": payload,
        }
        audit_id = _identity("audit", material)
        connection.execute(
            """
            INSERT INTO run_audit_records(
                audit_id, actor_id, action, subject_kind, subject_id,
                run_id, recorded_at, body_json, redacted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                audit_id,
                material["actor_id"],
                selected_action.value,
                material["subject_kind"],
                material["subject_id"],
                material["run_id"],
                recorded_at,
                _canonical_json(payload),
                bool(redact and payload != dict(body or {})),
            ],
        )

    def _root_from_row(self, row: Mapping[str, Any]) -> RunRootRecord:
        body_raw = row.get("body_json") or "{}"
        if isinstance(body_raw, Mapping):
            body = dict(body_raw)
        else:
            try:
                body = json.loads(str(body_raw))
            except json.JSONDecodeError:
                body = {}
        if not isinstance(body, dict):
            body = {}
        return RunRootRecord(
            run_id=str(row["run_id"]),
            run_namespace=str(row["run_namespace"]),
            repository_id=str(row["repository_id"]),
            checkout_id=str(row.get("checkout_id") or ""),
            worktree_id=str(row.get("worktree_id") or ""),
            session_id=str(row.get("session_id") or ""),
            lease_id=str(row.get("lease_id") or ""),
            target_resolution_receipt_cid=str(
                row.get("target_resolution_receipt_cid") or ""
            ),
            invocation_cid=str(row.get("invocation_cid") or ""),
            prompt_cid=str(row.get("prompt_cid") or ""),
            objective_cid=str(row.get("objective_cid") or ""),
            lifecycle_profile_cid=str(row.get("lifecycle_profile_cid") or ""),
            created_at=str(row["created_at"]),
            initial_handle_cid=str(row.get("initial_handle_cid") or ""),
            initial_revision=int(row.get("initial_revision") or 1),
            body=MappingProxyType(body),
            content_digest=str(row["content_digest"]),
        )

    def _head_from_row(self, row: Mapping[str, Any]) -> RunHeadRecord:
        body_raw = row.get("body_json") or "{}"
        if isinstance(body_raw, Mapping):
            body = dict(body_raw)
        else:
            try:
                body = json.loads(str(body_raw))
            except json.JSONDecodeError:
                body = {}
        if not isinstance(body, dict):
            body = {}
        return RunHeadRecord(
            run_id=str(row["run_id"]),
            run_revision=int(row["run_revision"]),
            handle_cid=str(row["handle_cid"]),
            semantic_id=str(row.get("semantic_id") or ""),
            state=str(row["state"]),
            health=str(row.get("health") or RunHealthState.UNKNOWN.value),
            event_cursor=str(row.get("event_cursor") or ""),
            updated_at=str(row["updated_at"]),
            previous_handle_cid=str(row.get("previous_handle_cid") or ""),
            previous_revision=int(row.get("previous_revision") or 0),
            body=MappingProxyType(body),
            content_digest=str(row["content_digest"]),
        )

    def _idempotency_from_row(self, row: Mapping[str, Any]) -> IdempotencyRecord:
        def _load(field: str) -> dict[str, Any]:
            raw = row.get(field) or "{}"
            if isinstance(raw, Mapping):
                value = dict(raw)
            else:
                try:
                    value = json.loads(str(raw))
                except json.JSONDecodeError:
                    value = {}
            return value if isinstance(value, dict) else {}

        return IdempotencyRecord(
            record_id=str(row["record_id"]),
            idempotency_key=str(row["idempotency_key"]),
            operation=str(row["operation"]),
            caller=str(row.get("caller") or ""),
            repository_id=str(row.get("repository_id") or ""),
            objective_id=str(row.get("objective_id") or ""),
            request_digest=str(row["request_digest"]),
            request=MappingProxyType(_load("request_json")),
            result=MappingProxyType(_load("result_json")),
            result_digest=str(row["result_digest"]),
            status=str(row.get("status") or IdempotencyStatus.COMMITTED.value),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


def open_database_run_registry(
    database_path: Path | str,
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    auto_redact: bool = True,
) -> DatabaseRunRegistry:
    """Open and initialize a :class:`DatabaseRunRegistry`."""

    return DatabaseRunRegistry(
        database_path,
        snapshot_id=snapshot_id,
        auto_redact=auto_redact,
    ).open()


__all__ = (
    "AUTHORITY_CLASS",
    "AuditAction",
    "DATABASE_RUN_REGISTRY_INTERFACE",
    "DATABASE_RUN_REGISTRY_SCHEMA",
    "DEFAULT_PAGE_LIMIT",
    "DEFAULT_SNAPSHOT_ID",
    "DIRECTORY_SCAN_AUTHORITY",
    "DIRECTORY_SCAN_RECEIPT_SCHEMA",
    "DatabaseRunRegistry",
    "DatabaseRunRegistryBoundsError",
    "DatabaseRunRegistryConflictError",
    "DatabaseRunRegistryError",
    "DatabaseRunRegistryIntegrityError",
    "DatabaseRunRegistryNotFoundError",
    "DatabaseRunRegistryNotOpenError",
    "DirectoryScanReceipt",
    "DuckDBUnavailableError",
    "EXPORT_AUTHORITY",
    "IDEMPOTENCY_RECORD_SCHEMA",
    "IdempotencyRecord",
    "IdempotencyStatus",
    "NAMESPACE_CURRENT_SCHEMA",
    "REDACTION_MARKER",
    "RUN_AUDIT_RECORD_SCHEMA",
    "RUN_EXPORT_RECEIPT_SCHEMA",
    "RUN_HANDLE_SNAPSHOT_SCHEMA",
    "RUN_HEAD_RECORD_SCHEMA",
    "RUN_ROOT_RECORD_SCHEMA",
    "RunHeadRecord",
    "RunHealthState",
    "RunLifecycleState",
    "RunRootRecord",
    "duckdb_available",
    "open_database_run_registry",
)
