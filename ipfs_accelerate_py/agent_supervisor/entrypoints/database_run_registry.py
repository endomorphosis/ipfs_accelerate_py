"""DuckDB-backed durable run registry, idempotency, and audit (DQP-031).

Interface: ``DatabaseRunRegistry@1``

:class:`DatabaseRunRegistry` is the transactional authority for run roots,
CAS heads, current-run pointers, lifecycle handle snapshots, idempotent
control results, and application audit receipts. Filesystem run trees are
export/compatibility adapters only: deleting or tampering with an export
cannot create, advance, or revive a run.

Authority rules (fail-closed)
-----------------------------
* Directory scans cannot create a run (``source_kind=directory_scan`` is
  rejected before any row is written).
* An idempotency key reused with a different request digest conflicts.
* Exact idempotent replay returns the prior result without re-dispatch.
* Concurrent create and head CAS use optimistic revision checks under a
  process-local exclusive lock.
* Audit bodies are redacted; secret material never persists as authority.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
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
from typing import Any, ClassVar, Final

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
RUN_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-root@1"
)
RUN_HEAD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-head@1"
)
RUN_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-handle@1"
)
NAMESPACE_CURRENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-namespace-current@1"
)
IDEMPOTENCY_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-idempotency@1"
)
AUDIT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-audit@1"
)
REGISTRY_TX_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-registry-tx@1"
)
EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-run-export-receipt@1"
)

DEFAULT_PAGE_LIMIT: Final[int] = 50
MAX_PAGE_LIMIT: Final[int] = 500
HARD_MAX_LIST: Final[int] = 4_096
MAX_PAYLOAD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_ID_BYTES: Final[int] = 512
EXPORT_AUTHORITY: Final[str] = "export_adapter_only"

# Closed source kinds that may create a durable run.
ALLOWED_CREATE_SOURCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "control_api",
        "lifecycle_orchestrator",
        "prompt_broker",
        "plan_materializer",
        "daemon",
        "operator",
        "test",
        "import",
        "recovery",
    }
)
FORBIDDEN_CREATE_SOURCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "directory_scan",
        "prompt_directory_scan",
        "scan",
        "filesystem_scan",
        "export",
        "status_file",
        "pid_file",
    }
)

ClockMs = Callable[[], int]


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
    source_kind VARCHAR NOT NULL,
    invocation_cid VARCHAR NOT NULL DEFAULT '',
    prompt_cid VARCHAR NOT NULL DEFAULT '',
    objective_cid VARCHAR NOT NULL DEFAULT '',
    target_resolution_receipt_cid VARCHAR NOT NULL DEFAULT '',
    lifecycle_profile_cid VARCHAR NOT NULL DEFAULT '',
    created_at_ms BIGINT NOT NULL,
    initial_handle_cid VARCHAR NOT NULL,
    initial_revision BIGINT NOT NULL DEFAULT 1,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS run_roots_namespace_idx
    ON run_roots(run_namespace, created_at_ms);
CREATE INDEX IF NOT EXISTS run_roots_worktree_idx
    ON run_roots(worktree_id, created_at_ms);

CREATE TABLE IF NOT EXISTS run_heads (
    run_id VARCHAR PRIMARY KEY,
    run_revision BIGINT NOT NULL,
    handle_cid VARCHAR NOT NULL,
    semantic_id VARCHAR NOT NULL DEFAULT '',
    state VARCHAR NOT NULL DEFAULT 'created',
    health VARCHAR NOT NULL DEFAULT 'unknown',
    event_cursor VARCHAR NOT NULL DEFAULT '',
    updated_at_ms BIGINT NOT NULL,
    previous_handle_cid VARCHAR NOT NULL DEFAULT '',
    previous_revision BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS run_heads_revision_idx
    ON run_heads(run_revision, updated_at_ms);

CREATE TABLE IF NOT EXISTS run_handles (
    handle_cid VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    run_revision BIGINT NOT NULL,
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS run_handles_run_idx
    ON run_handles(run_id, run_revision);

CREATE TABLE IF NOT EXISTS namespace_current (
    run_namespace VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    selected_run_id VARCHAR NOT NULL,
    handle_cid VARCHAR NOT NULL,
    pointer_revision BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS run_idempotency_records (
    idempotency_key VARCHAR PRIMARY KEY,
    request_digest VARCHAR NOT NULL,
    command_kind VARCHAR NOT NULL,
    result_digest VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL DEFAULT '',
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS run_idempotency_run_idx
    ON run_idempotency_records(run_id, created_at_ms);

CREATE TABLE IF NOT EXISTS run_audit_records (
    audit_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL DEFAULT '',
    actor_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    redacted BOOLEAN NOT NULL DEFAULT TRUE,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS run_audit_run_idx
    ON run_audit_records(run_id, recorded_at_ms);
CREATE INDEX IF NOT EXISTS run_audit_action_idx
    ON run_audit_records(action, recorded_at_ms);

CREATE TABLE IF NOT EXISTS run_export_receipts (
    export_id VARCHAR PRIMARY KEY,
    target_path VARCHAR NOT NULL,
    run_count BIGINT NOT NULL DEFAULT 0,
    export_digest VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'export_adapter_only',
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseRunRegistryError(RuntimeError):
    """Base fail-closed error for database run-registry operations."""

    code = "DQP_RUN_REGISTRY_ERROR"


class DatabaseRunRegistryNotOpenError(DatabaseRunRegistryError):
    """Operation requires an open registry."""

    code = "DQP_RUN_REGISTRY_NOT_OPEN"


class DatabaseRunRegistryBoundsError(DatabaseRunRegistryError, ValueError):
    """Payload or list bound exceeded."""

    code = "DQP_RUN_REGISTRY_BOUNDS"


class DatabaseRunNotFoundError(DatabaseRunRegistryError):
    """Run identity is absent."""

    code = "DQP_RUN_NOT_FOUND"


class DatabaseRunExistsError(DatabaseRunRegistryError):
    """Create would overwrite an existing run root."""

    code = "DQP_RUN_EXISTS"


class DatabaseRunCasConflictError(DatabaseRunRegistryError):
    """Compare-and-swap revision does not match the head."""

    code = "DQP_RUN_CAS_CONFLICT"

    def __init__(
        self,
        message: str,
        *,
        receipt: "RegistryTransactionReceipt | None" = None,
    ) -> None:
        super().__init__(message)
        self.receipt = receipt


class DatabaseRunSourceError(DatabaseRunRegistryError):
    """Create source is forbidden (e.g. directory scan)."""

    code = "DQP_RUN_SOURCE_FORBIDDEN"


class DatabaseIdempotencyConflictError(DatabaseRunRegistryError):
    """Idempotency key reused with a different request digest."""

    code = "DQP_IDEMPOTENCY_CONFLICT"


class DuckDBUnavailableError(DatabaseRunRegistryError):
    """Optional DuckDB dependency is not installed."""

    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RegistryTxOutcome(str, Enum):
    COMMITTED = "committed"
    CONFLICT = "conflict"
    NOOP = "noop"
    REPLAYED = "replayed"
    REJECTED = "rejected"


class RegistryOperation(str, Enum):
    CREATE = "create"
    CAS_UPDATE = "cas_update"
    SET_CURRENT = "set_current"
    CONTROL_MUTATION = "control_mutation"
    AUDIT = "audit"
    EXPORT = "export"
    LOOKUP = "lookup"


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


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_ID_BYTES) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseRunRegistryError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseRunRegistryError(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise DatabaseRunRegistryBoundsError(f"{name} exceeds {maximum} bytes")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DatabaseRunRegistryBoundsError(f"{name} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DatabaseRunRegistryBoundsError(f"{name} must be a non-negative integer") from exc
    if number < 0:
        raise DatabaseRunRegistryBoundsError(f"{name} must be a non-negative integer")
    return number


def _positive_int(value: Any, name: str) -> int:
    number = _nonneg_int(value, name)
    if number < 1:
        raise DatabaseRunRegistryBoundsError(f"{name} must be a positive integer")
    return number


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


def _digest_of(value: Any) -> str:
    return _sha256_hex(_canonical_json(value).encode("utf-8"))


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise DatabaseRunRegistryBoundsError(
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


def _normalize_source_kind(value: Any) -> str:
    text = _text(value, "source_kind").casefold().replace("-", "_")
    return text


def _assert_create_source_allowed(source_kind: str) -> str:
    kind = _normalize_source_kind(source_kind)
    if kind in FORBIDDEN_CREATE_SOURCE_KINDS or kind.endswith("_scan"):
        raise DatabaseRunSourceError(
            f"source_kind {kind!r} cannot create a durable run "
            "(directory/filesystem scans are discovery-only)"
        )
    if kind not in ALLOWED_CREATE_SOURCE_KINDS:
        raise DatabaseRunSourceError(
            f"source_kind {kind!r} is not an admitted run-create authority"
        )
    return kind


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryTransactionReceipt:
    """Durable outcome of one registry mutation."""

    SCHEMA: ClassVar[str] = REGISTRY_TX_SCHEMA

    operation: RegistryOperation | str
    outcome: RegistryTxOutcome | str
    run_id: str = ""
    run_revision: int = 0
    handle_cid: str = ""
    integrity_cid: str = ""
    previous_revision: int = 0
    previous_handle_cid: str = ""
    reason_codes: tuple[str, ...] = ()
    committed_at_ms: int = 0
    replayed: bool = False
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = self.operation
        if not isinstance(operation, RegistryOperation):
            operation = RegistryOperation(str(operation))
        outcome = self.outcome
        if not isinstance(outcome, RegistryTxOutcome):
            outcome = RegistryTxOutcome(str(outcome))
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", required=False))
        object.__setattr__(
            self, "run_revision", _nonneg_int(int(self.run_revision), "run_revision")
        )
        object.__setattr__(
            self, "handle_cid", _text(self.handle_cid, "handle_cid", required=False)
        )
        object.__setattr__(
            self,
            "integrity_cid",
            _text(self.integrity_cid, "integrity_cid", required=False),
        )
        object.__setattr__(
            self,
            "previous_revision",
            _nonneg_int(int(self.previous_revision), "previous_revision"),
        )
        object.__setattr__(
            self,
            "previous_handle_cid",
            _text(self.previous_handle_cid, "previous_handle_cid", required=False),
        )
        reasons = tuple(
            str(item).strip()
            for item in (self.reason_codes or ())
            if str(item).strip()
        )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "committed_at_ms",
            _nonneg_int(int(self.committed_at_ms), "committed_at_ms"),
        )
        object.__setattr__(self, "replayed", bool(self.replayed))
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(dict(self.body or {}), name="body")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "operation": self.operation.value,
            "outcome": self.outcome.value,
            "run_id": self.run_id,
            "run_revision": int(self.run_revision),
            "handle_cid": self.handle_cid,
            "integrity_cid": self.integrity_cid,
            "previous_revision": int(self.previous_revision),
            "previous_handle_cid": self.previous_handle_cid,
            "reason_codes": list(self.reason_codes),
            "committed_at_ms": int(self.committed_at_ms),
            "replayed": bool(self.replayed),
            "body": dict(self.body),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegistryTransactionReceipt":
        return cls(
            operation=str(payload.get("operation") or ""),
            outcome=str(payload.get("outcome") or ""),
            run_id=str(payload.get("run_id") or ""),
            run_revision=int(payload.get("run_revision") or 0),
            handle_cid=str(payload.get("handle_cid") or ""),
            integrity_cid=str(payload.get("integrity_cid") or ""),
            previous_revision=int(payload.get("previous_revision") or 0),
            previous_handle_cid=str(payload.get("previous_handle_cid") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            committed_at_ms=int(payload.get("committed_at_ms") or 0),
            replayed=bool(payload.get("replayed") or False),
            body=dict(payload.get("body") or {}),
        )


@dataclass(frozen=True)
class RunRecord:
    """Immutable root plus current head projection."""

    SCHEMA: ClassVar[str] = RUN_ROOT_SCHEMA

    run_id: str
    run_namespace: str
    repository_id: str
    checkout_id: str = ""
    worktree_id: str = ""
    source_kind: str = "control_api"
    invocation_cid: str = ""
    prompt_cid: str = ""
    objective_cid: str = ""
    target_resolution_receipt_cid: str = ""
    lifecycle_profile_cid: str = ""
    created_at_ms: int = 0
    initial_handle_cid: str = ""
    initial_revision: int = 1
    run_revision: int = 1
    handle_cid: str = ""
    semantic_id: str = ""
    state: str = "created"
    health: str = "unknown"
    event_cursor: str = ""
    updated_at_ms: int = 0
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_id": self.run_id,
            "run_namespace": self.run_namespace,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "worktree_id": self.worktree_id,
            "source_kind": self.source_kind,
            "invocation_cid": self.invocation_cid,
            "prompt_cid": self.prompt_cid,
            "objective_cid": self.objective_cid,
            "target_resolution_receipt_cid": self.target_resolution_receipt_cid,
            "lifecycle_profile_cid": self.lifecycle_profile_cid,
            "created_at_ms": int(self.created_at_ms),
            "initial_handle_cid": self.initial_handle_cid,
            "initial_revision": int(self.initial_revision),
            "run_revision": int(self.run_revision),
            "handle_cid": self.handle_cid,
            "semantic_id": self.semantic_id,
            "state": self.state,
            "health": self.health,
            "event_cursor": self.event_cursor,
            "updated_at_ms": int(self.updated_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class RunListPage:
    """Bounded page of runs with pagination cursor."""

    items: tuple[RunRecord, ...]
    next_cursor: str = ""
    has_more: bool = False
    total_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "next_cursor": self.next_cursor,
            "has_more": bool(self.has_more),
            "total_estimate": int(self.total_estimate),
        }


@dataclass(frozen=True)
class AuditRecord:
    """Explicit application audit receipt (redacted by default)."""

    SCHEMA: ClassVar[str] = AUDIT_RECORD_SCHEMA

    audit_id: str
    run_id: str
    actor_id: str
    action: str
    outcome: str
    recorded_at_ms: int
    redacted: bool = True
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "audit_id": self.audit_id,
            "run_id": self.run_id,
            "actor_id": self.actor_id,
            "action": self.action,
            "outcome": self.outcome,
            "recorded_at_ms": int(self.recorded_at_ms),
            "redacted": bool(self.redacted),
            "body": dict(self.body),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DatabaseRunRegistry:
    """DuckDB-backed run roots, CAS heads, idempotency, and audit authority."""

    INTERFACE: ClassVar[str] = DATABASE_RUN_REGISTRY_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_RUN_REGISTRY_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        max_list: int = HARD_MAX_LIST,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseRunRegistry; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        if not 1 <= int(max_list) <= HARD_MAX_LIST:
            raise DatabaseRunRegistryBoundsError(
                f"max_list must be in 1..{HARD_MAX_LIST}"
            )
        self._max_list = int(max_list)
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

    def open(self) -> "DatabaseRunRegistry":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                for key, value in (
                    ("interface", DATABASE_RUN_REGISTRY_INTERFACE),
                    ("schema", DATABASE_RUN_REGISTRY_SCHEMA),
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

    def __enter__(self) -> "DatabaseRunRegistry":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def authority_policy(self) -> dict[str, Any]:
        """Explicit authority surface for operators and tests."""

        return {
            "semantic_authority": "database",
            "filesystem_run_trees": EXPORT_AUTHORITY,
            "directory_scan_create": "prohibited",
            "idempotency": "exact_request_digest_replay",
            "cas": "optimistic_revision",
            "audit": "explicit_redacted",
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
        }

    # -- connection helpers --------------------------------------------------

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseRunRegistryNotOpenError("DatabaseRunRegistry is not open")
        return self._connection

    def _begin(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        try:
            connection.execute("BEGIN TRANSACTION")
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

    # -- public API ----------------------------------------------------------

    def create_run(
        self,
        *,
        run_namespace: str,
        repository_id: str,
        source_kind: str,
        run_id: str | None = None,
        checkout_id: str = "",
        worktree_id: str = "",
        invocation_cid: str = "",
        prompt_cid: str = "",
        objective_cid: str = "",
        target_resolution_receipt_cid: str = "",
        lifecycle_profile_cid: str = "",
        state: str = "created",
        health: str = "unknown",
        event_cursor: str = "",
        handle_body: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        idempotency_key: str = "",
        request: Mapping[str, Any] | None = None,
        actor_id: str = "system",
    ) -> RegistryTransactionReceipt:
        """Create an immutable run root and initial CAS head.

        ``source_kind=directory_scan`` (and other scan sources) are rejected
        before any durable row is written.
        """

        kind = _assert_create_source_allowed(source_kind)
        namespace = _text(run_namespace, "run_namespace")
        repo = _text(repository_id, "repository_id", maximum=MAX_TEXT_BYTES)
        checkout = _text(checkout_id, "checkout_id", required=False)
        worktree = _text(worktree_id, "worktree_id", required=False)
        handle_payload = _bounded_mapping(handle_body, name="handle_body")
        root_body = _bounded_mapping(body, name="body")
        now_ms = int(self._clock_ms())
        resolved_run_id = _text(run_id or _new_id("run"), "run_id")

        request_payload = dict(request or {})
        if not request_payload:
            request_payload = {
                "operation": RegistryOperation.CREATE.value,
                "run_id": resolved_run_id,
                "run_namespace": namespace,
                "repository_id": repo,
                "checkout_id": checkout,
                "worktree_id": worktree,
                "source_kind": kind,
                "handle_body": handle_payload,
                "body": root_body,
            }
        request_digest = _digest_of(request_payload)

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                if idempotency_key:
                    replay = self._lookup_idempotency(
                        connection,
                        idempotency_key=idempotency_key,
                        request_digest=request_digest,
                    )
                    if replay is not None:
                        self._commit_if_idle(connection)
                        return replay

                existing = connection.execute(
                    "SELECT run_id FROM run_roots WHERE run_id = ?",
                    [resolved_run_id],
                ).fetchone()
                if existing is not None:
                    raise DatabaseRunExistsError(
                        f"run already registered: {resolved_run_id}"
                    )

                handle_cid = _digest_of(
                    {
                        "schema": RUN_HANDLE_SCHEMA,
                        "run_id": resolved_run_id,
                        "run_revision": 1,
                        "payload": handle_payload,
                    }
                )
                semantic_id = _digest_of(
                    {
                        "run_id": resolved_run_id,
                        "namespace": namespace,
                        "repository_id": repo,
                        "revision": 1,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO run_roots (
                        run_id, run_namespace, repository_id, checkout_id,
                        worktree_id, source_kind, invocation_cid, prompt_cid,
                        objective_cid, target_resolution_receipt_cid,
                        lifecycle_profile_cid, created_at_ms, initial_handle_cid,
                        initial_revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        resolved_run_id,
                        namespace,
                        repo,
                        checkout,
                        worktree,
                        kind,
                        _text(invocation_cid, "invocation_cid", required=False),
                        _text(prompt_cid, "prompt_cid", required=False),
                        _text(objective_cid, "objective_cid", required=False),
                        _text(
                            target_resolution_receipt_cid,
                            "target_resolution_receipt_cid",
                            required=False,
                        ),
                        _text(
                            lifecycle_profile_cid,
                            "lifecycle_profile_cid",
                            required=False,
                        ),
                        now_ms,
                        handle_cid,
                        1,
                        _canonical_json(root_body),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO run_heads (
                        run_id, run_revision, handle_cid, semantic_id, state,
                        health, event_cursor, updated_at_ms, previous_handle_cid,
                        previous_revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        resolved_run_id,
                        1,
                        handle_cid,
                        semantic_id,
                        _text(state, "state"),
                        _text(health, "health"),
                        _text(event_cursor, "event_cursor", required=False),
                        now_ms,
                        "",
                        0,
                        _canonical_json({"schema": RUN_HEAD_SCHEMA}),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO run_handles (
                        handle_cid, run_id, run_revision, created_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        handle_cid,
                        resolved_run_id,
                        1,
                        now_ms,
                        _canonical_json(
                            {
                                "schema": RUN_HANDLE_SCHEMA,
                                "run_id": resolved_run_id,
                                "run_revision": 1,
                                "payload": handle_payload,
                            }
                        ),
                    ],
                )
                head_integrity = _digest_of(
                    {
                        "schema": RUN_HEAD_SCHEMA,
                        "run_id": resolved_run_id,
                        "run_revision": 1,
                        "handle_cid": handle_cid,
                        "semantic_id": semantic_id,
                    }
                )
                receipt = RegistryTransactionReceipt(
                    operation=RegistryOperation.CREATE,
                    outcome=RegistryTxOutcome.COMMITTED,
                    run_id=resolved_run_id,
                    run_revision=1,
                    handle_cid=handle_cid,
                    integrity_cid=head_integrity,
                    previous_revision=0,
                    previous_handle_cid="",
                    reason_codes=("created",),
                    committed_at_ms=now_ms,
                    body={
                        "run_namespace": namespace,
                        "repository_id": repo,
                        "worktree_id": worktree,
                        "source_kind": kind,
                    },
                )
                if idempotency_key:
                    self._store_idempotency(
                        connection,
                        idempotency_key=idempotency_key,
                        request_digest=request_digest,
                        command_kind=RegistryOperation.CREATE.value,
                        receipt=receipt,
                    )
                self._insert_audit(
                    connection,
                    run_id=resolved_run_id,
                    actor_id=actor_id,
                    action="create_run",
                    outcome=RegistryTxOutcome.COMMITTED.value,
                    body={
                        "source_kind": kind,
                        "handle_cid": handle_cid,
                        "worktree_id": worktree,
                    },
                    recorded_at_ms=now_ms,
                )
                self._commit_if_idle(connection)
                return receipt
            except Exception:
                self._rollback_if_open(connection)
                raise

    def cas_update(
        self,
        run_id: str,
        *,
        expected_revision: int,
        handle_body: Mapping[str, Any] | None = None,
        state: str | None = None,
        health: str | None = None,
        event_cursor: str | None = None,
        idempotency_key: str = "",
        request: Mapping[str, Any] | None = None,
        actor_id: str = "system",
    ) -> RegistryTransactionReceipt:
        """Compare-and-swap the run head to the next revision."""

        rid = _text(run_id, "run_id")
        expected = _positive_int(int(expected_revision), "expected_revision")
        handle_payload = _bounded_mapping(handle_body, name="handle_body")
        now_ms = int(self._clock_ms())
        request_payload = dict(request or {})
        if not request_payload:
            request_payload = {
                "operation": RegistryOperation.CAS_UPDATE.value,
                "run_id": rid,
                "expected_revision": expected,
                "handle_body": handle_payload,
                "state": state,
                "health": health,
                "event_cursor": event_cursor,
            }
        request_digest = _digest_of(request_payload)

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                if idempotency_key:
                    replay = self._lookup_idempotency(
                        connection,
                        idempotency_key=idempotency_key,
                        request_digest=request_digest,
                    )
                    if replay is not None:
                        self._commit_if_idle(connection)
                        return replay

                head_row = connection.execute(
                    """
                    SELECT run_id, run_revision, handle_cid, semantic_id, state,
                           health, event_cursor, updated_at_ms
                    FROM run_heads WHERE run_id = ?
                    """,
                    [rid],
                ).fetchone()
                if head_row is None:
                    raise DatabaseRunNotFoundError(f"run not found: {rid}")
                head = _row_mapping(head_row)
                current_revision = int(_row_get(head, "run_revision", default=0))
                current_handle = str(_row_get(head, "handle_cid", default="") or "")
                if current_revision != expected:
                    receipt = RegistryTransactionReceipt(
                        operation=RegistryOperation.CAS_UPDATE,
                        outcome=RegistryTxOutcome.CONFLICT,
                        run_id=rid,
                        run_revision=current_revision,
                        handle_cid=current_handle,
                        integrity_cid=_digest_of(dict(head)),
                        previous_revision=current_revision,
                        previous_handle_cid=current_handle,
                        reason_codes=("revision_mismatch",),
                        committed_at_ms=now_ms,
                    )
                    self._rollback_if_open(connection)
                    raise DatabaseRunCasConflictError(
                        "CAS conflict: expected revision does not match head",
                        receipt=receipt,
                    )

                new_revision = current_revision + 1
                new_handle_cid = _digest_of(
                    {
                        "schema": RUN_HANDLE_SCHEMA,
                        "run_id": rid,
                        "run_revision": new_revision,
                        "payload": handle_payload,
                    }
                )
                new_semantic = _digest_of(
                    {
                        "run_id": rid,
                        "revision": new_revision,
                        "handle_cid": new_handle_cid,
                    }
                )
                new_state = (
                    _text(state, "state")
                    if state is not None
                    else str(_row_get(head, "state", default="created") or "created")
                )
                new_health = (
                    _text(health, "health")
                    if health is not None
                    else str(_row_get(head, "health", default="unknown") or "unknown")
                )
                new_cursor = (
                    _text(event_cursor, "event_cursor", required=False)
                    if event_cursor is not None
                    else str(_row_get(head, "event_cursor", default="") or "")
                )
                connection.execute(
                    """
                    UPDATE run_heads
                    SET run_revision = ?, handle_cid = ?, semantic_id = ?,
                        state = ?, health = ?, event_cursor = ?,
                        updated_at_ms = ?, previous_handle_cid = ?,
                        previous_revision = ?
                    WHERE run_id = ? AND run_revision = ?
                    """,
                    [
                        new_revision,
                        new_handle_cid,
                        new_semantic,
                        new_state,
                        new_health,
                        new_cursor,
                        now_ms,
                        current_handle,
                        current_revision,
                        rid,
                        expected,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO run_handles (
                        handle_cid, run_id, run_revision, created_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        new_handle_cid,
                        rid,
                        new_revision,
                        now_ms,
                        _canonical_json(
                            {
                                "schema": RUN_HANDLE_SCHEMA,
                                "run_id": rid,
                                "run_revision": new_revision,
                                "payload": handle_payload,
                            }
                        ),
                    ],
                )
                integrity = _digest_of(
                    {
                        "schema": RUN_HEAD_SCHEMA,
                        "run_id": rid,
                        "run_revision": new_revision,
                        "handle_cid": new_handle_cid,
                        "semantic_id": new_semantic,
                    }
                )
                receipt = RegistryTransactionReceipt(
                    operation=RegistryOperation.CAS_UPDATE,
                    outcome=RegistryTxOutcome.COMMITTED,
                    run_id=rid,
                    run_revision=new_revision,
                    handle_cid=new_handle_cid,
                    integrity_cid=integrity,
                    previous_revision=current_revision,
                    previous_handle_cid=current_handle,
                    reason_codes=("cas_committed",),
                    committed_at_ms=now_ms,
                )
                if idempotency_key:
                    self._store_idempotency(
                        connection,
                        idempotency_key=idempotency_key,
                        request_digest=request_digest,
                        command_kind=RegistryOperation.CAS_UPDATE.value,
                        receipt=receipt,
                    )
                self._insert_audit(
                    connection,
                    run_id=rid,
                    actor_id=actor_id,
                    action="cas_update",
                    outcome=RegistryTxOutcome.COMMITTED.value,
                    body={
                        "expected_revision": expected,
                        "new_revision": new_revision,
                        "handle_cid": new_handle_cid,
                    },
                    recorded_at_ms=now_ms,
                )
                self._commit_if_idle(connection)
                return receipt
            except DatabaseRunCasConflictError:
                raise
            except Exception:
                self._rollback_if_open(connection)
                raise

    def set_current(
        self,
        *,
        run_namespace: str,
        repository_id: str,
        run_id: str,
        expected_pointer_revision: int | None = None,
        actor_id: str = "system",
    ) -> RegistryTransactionReceipt:
        """CAS-update the namespace current-run pointer."""

        namespace = _text(run_namespace, "run_namespace")
        repo = _text(repository_id, "repository_id", maximum=MAX_TEXT_BYTES)
        rid = _text(run_id, "run_id")
        now_ms = int(self._clock_ms())

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                root_row = connection.execute(
                    """
                    SELECT run_id, run_namespace, repository_id
                    FROM run_roots WHERE run_id = ?
                    """,
                    [rid],
                ).fetchone()
                if root_row is None:
                    raise DatabaseRunNotFoundError(f"run not found: {rid}")
                root = _row_mapping(root_row)
                if str(_row_get(root, "run_namespace", default="")) != namespace:
                    raise DatabaseRunRegistryError(
                        "run namespace does not match pointer"
                    )
                if str(_row_get(root, "repository_id", default="")) != repo:
                    raise DatabaseRunRegistryError(
                        "run repository_id does not match pointer"
                    )
                head_row = connection.execute(
                    "SELECT handle_cid, run_revision FROM run_heads WHERE run_id = ?",
                    [rid],
                ).fetchone()
                if head_row is None:
                    raise DatabaseRunNotFoundError(f"run head missing: {rid}")
                head = _row_mapping(head_row)
                handle_cid = str(_row_get(head, "handle_cid", default="") or "")

                current_row = connection.execute(
                    """
                    SELECT selected_run_id, pointer_revision, handle_cid
                    FROM namespace_current WHERE run_namespace = ?
                    """,
                    [namespace],
                ).fetchone()
                previous_revision = 0
                previous_handle = ""
                if current_row is not None:
                    current = _row_mapping(current_row)
                    previous_revision = int(
                        _row_get(current, "pointer_revision", default=0)
                    )
                    previous_handle = str(
                        _row_get(current, "selected_run_id", default="") or ""
                    )
                    if (
                        expected_pointer_revision is not None
                        and previous_revision != int(expected_pointer_revision)
                    ):
                        receipt = RegistryTransactionReceipt(
                            operation=RegistryOperation.SET_CURRENT,
                            outcome=RegistryTxOutcome.CONFLICT,
                            run_id=rid,
                            run_revision=int(
                                _row_get(head, "run_revision", default=0)
                            ),
                            handle_cid=handle_cid,
                            previous_revision=previous_revision,
                            previous_handle_cid=previous_handle,
                            reason_codes=("pointer_revision_mismatch",),
                            committed_at_ms=now_ms,
                        )
                        self._rollback_if_open(connection)
                        raise DatabaseRunCasConflictError(
                            "CAS conflict: expected pointer revision mismatch",
                            receipt=receipt,
                        )

                new_revision = previous_revision + 1
                if current_row is None:
                    connection.execute(
                        """
                        INSERT INTO namespace_current (
                            run_namespace, repository_id, selected_run_id,
                            handle_cid, pointer_revision, updated_at_ms, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            namespace,
                            repo,
                            rid,
                            handle_cid,
                            new_revision,
                            now_ms,
                            _canonical_json({"schema": NAMESPACE_CURRENT_SCHEMA}),
                        ],
                    )
                else:
                    connection.execute(
                        """
                        UPDATE namespace_current
                        SET repository_id = ?, selected_run_id = ?, handle_cid = ?,
                            pointer_revision = ?, updated_at_ms = ?
                        WHERE run_namespace = ?
                        """,
                        [repo, rid, handle_cid, new_revision, now_ms, namespace],
                    )
                receipt = RegistryTransactionReceipt(
                    operation=RegistryOperation.SET_CURRENT,
                    outcome=RegistryTxOutcome.COMMITTED,
                    run_id=rid,
                    run_revision=int(_row_get(head, "run_revision", default=0)),
                    handle_cid=handle_cid,
                    integrity_cid=_digest_of(
                        {
                            "namespace": namespace,
                            "run_id": rid,
                            "pointer_revision": new_revision,
                        }
                    ),
                    previous_revision=previous_revision,
                    previous_handle_cid=previous_handle,
                    reason_codes=("current_set",),
                    committed_at_ms=now_ms,
                    body={"pointer_revision": new_revision, "run_namespace": namespace},
                )
                self._insert_audit(
                    connection,
                    run_id=rid,
                    actor_id=actor_id,
                    action="set_current",
                    outcome=RegistryTxOutcome.COMMITTED.value,
                    body={"run_namespace": namespace, "pointer_revision": new_revision},
                    recorded_at_ms=now_ms,
                )
                self._commit_if_idle(connection)
                return receipt
            except DatabaseRunCasConflictError:
                raise
            except Exception:
                self._rollback_if_open(connection)
                raise

    def execute_control_mutation(
        self,
        *,
        command_kind: str,
        request: Mapping[str, Any],
        idempotency_key: str,
        result_body: Mapping[str, Any] | None = None,
        run_id: str = "",
        actor_id: str = "system",
        effect_fn: Callable[[], Mapping[str, Any]] | None = None,
    ) -> RegistryTransactionReceipt:
        """Execute a control mutation under exact-request idempotency.

        Exact replay of the same key + request digest returns the prior
        result without invoking ``effect_fn``. A different request under the
        same key raises :class:`DatabaseIdempotencyConflictError`.
        """

        key = _text(idempotency_key, "idempotency_key")
        kind = _text(command_kind, "command_kind")
        request_payload = _bounded_mapping(request, name="request")
        request_digest = _digest_of(request_payload)
        now_ms = int(self._clock_ms())

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                replay = self._lookup_idempotency(
                    connection,
                    idempotency_key=key,
                    request_digest=request_digest,
                )
                if replay is not None:
                    self._commit_if_idle(connection)
                    return replay

                if effect_fn is not None:
                    produced = dict(effect_fn() or {})
                else:
                    produced = dict(result_body or {})
                produced = _bounded_mapping(produced, name="result_body")
                receipt = RegistryTransactionReceipt(
                    operation=RegistryOperation.CONTROL_MUTATION,
                    outcome=RegistryTxOutcome.COMMITTED,
                    run_id=_text(run_id, "run_id", required=False),
                    run_revision=0,
                    handle_cid="",
                    integrity_cid=_digest_of(produced),
                    reason_codes=("control_committed", kind),
                    committed_at_ms=now_ms,
                    body={"command_kind": kind, "result": produced},
                )
                self._store_idempotency(
                    connection,
                    idempotency_key=key,
                    request_digest=request_digest,
                    command_kind=kind,
                    receipt=receipt,
                )
                self._insert_audit(
                    connection,
                    run_id=str(run_id or ""),
                    actor_id=actor_id,
                    action=f"control:{kind}",
                    outcome=RegistryTxOutcome.COMMITTED.value,
                    body={"command_kind": kind, "request_digest": request_digest},
                    recorded_at_ms=now_ms,
                )
                self._commit_if_idle(connection)
                return receipt
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_run(self, run_id: str) -> RunRecord:
        """Return root + head projection for one run."""

        rid = _text(run_id, "run_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT r.run_id, r.run_namespace, r.repository_id, r.checkout_id,
                       r.worktree_id, r.source_kind, r.invocation_cid, r.prompt_cid,
                       r.objective_cid, r.target_resolution_receipt_cid,
                       r.lifecycle_profile_cid, r.created_at_ms, r.initial_handle_cid,
                       r.initial_revision, r.body_json,
                       h.run_revision, h.handle_cid, h.semantic_id, h.state,
                       h.health, h.event_cursor, h.updated_at_ms
                FROM run_roots r
                JOIN run_heads h ON h.run_id = r.run_id
                WHERE r.run_id = ?
                """,
                [rid],
            ).fetchone()
            if row is None:
                raise DatabaseRunNotFoundError(f"run not found: {rid}")
            return self._row_to_run(_row_mapping(row))

    def get_head(self, run_id: str) -> dict[str, Any]:
        rid = _text(run_id, "run_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM run_heads WHERE run_id = ?",
                [rid],
            ).fetchone()
            if row is None:
                raise DatabaseRunNotFoundError(f"run head not found: {rid}")
            mapping = _row_mapping(row)
            return {
                "schema": RUN_HEAD_SCHEMA,
                "run_id": str(_row_get(mapping, "run_id", default="") or ""),
                "run_revision": int(_row_get(mapping, "run_revision", default=0)),
                "handle_cid": str(_row_get(mapping, "handle_cid", default="") or ""),
                "semantic_id": str(_row_get(mapping, "semantic_id", default="") or ""),
                "state": str(_row_get(mapping, "state", default="") or ""),
                "health": str(_row_get(mapping, "health", default="") or ""),
                "event_cursor": str(
                    _row_get(mapping, "event_cursor", default="") or ""
                ),
                "updated_at_ms": int(
                    _row_get(mapping, "updated_at_ms", default=0)
                ),
                "previous_handle_cid": str(
                    _row_get(mapping, "previous_handle_cid", default="") or ""
                ),
                "previous_revision": int(
                    _row_get(mapping, "previous_revision", default=0)
                ),
            }

    def get_current(self, run_namespace: str) -> dict[str, Any] | None:
        namespace = _text(run_namespace, "run_namespace")
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
                "run_namespace": str(
                    _row_get(mapping, "run_namespace", default="") or ""
                ),
                "repository_id": str(
                    _row_get(mapping, "repository_id", default="") or ""
                ),
                "selected_run_id": str(
                    _row_get(mapping, "selected_run_id", default="") or ""
                ),
                "handle_cid": str(_row_get(mapping, "handle_cid", default="") or ""),
                "pointer_revision": int(
                    _row_get(mapping, "pointer_revision", default=0)
                ),
                "updated_at_ms": int(
                    _row_get(mapping, "updated_at_ms", default=0)
                ),
            }

    def list_runs(
        self,
        *,
        run_namespace: str | None = None,
        worktree_id: str | None = None,
        cursor: str = "",
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> RunListPage:
        """List runs with stable ordering and bounded pagination."""

        page_limit = int(limit)
        if page_limit < 1 or page_limit > MAX_PAGE_LIMIT:
            raise DatabaseRunRegistryBoundsError(
                f"limit must be in 1..{MAX_PAGE_LIMIT}"
            )
        page_limit = min(page_limit, self._max_list)
        offset = 0
        if cursor:
            try:
                offset = max(0, int(cursor))
            except (TypeError, ValueError) as exc:
                raise DatabaseRunRegistryBoundsError(
                    "cursor must be a non-negative integer offset"
                ) from exc

        clauses: list[str] = []
        params: list[Any] = []
        if run_namespace is not None:
            clauses.append("r.run_namespace = ?")
            params.append(_text(run_namespace, "run_namespace"))
        if worktree_id is not None:
            clauses.append("r.worktree_id = ?")
            params.append(_text(worktree_id, "worktree_id"))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            connection = self._require()
            count_row = connection.execute(
                f"SELECT COUNT(*) AS n FROM run_roots r {where}",
                params,
            ).fetchone()
            count_mapping = _row_mapping(count_row)
            total_raw = _row_get(count_mapping, "n", "count_star()", "COUNT_STAR()", default=None)
            if total_raw is None and count_mapping:
                # Positional fallback for drivers that drop aliases.
                total_raw = next(iter(count_mapping.values()))
            total = int(total_raw or 0)
            rows = connection.execute(
                f"""
                SELECT r.run_id, r.run_namespace, r.repository_id, r.checkout_id,
                       r.worktree_id, r.source_kind, r.invocation_cid, r.prompt_cid,
                       r.objective_cid, r.target_resolution_receipt_cid,
                       r.lifecycle_profile_cid, r.created_at_ms, r.initial_handle_cid,
                       r.initial_revision, r.body_json,
                       h.run_revision, h.handle_cid, h.semantic_id, h.state,
                       h.health, h.event_cursor, h.updated_at_ms
                FROM run_roots r
                JOIN run_heads h ON h.run_id = r.run_id
                {where}
                ORDER BY r.created_at_ms ASC, r.run_id ASC
                LIMIT ? OFFSET ?
                """,
                [*params, page_limit + 1, offset],
            ).fetchall()
            items = [self._row_to_run(_row_mapping(row)) for row in rows[:page_limit]]
            has_more = len(rows) > page_limit
            next_cursor = str(offset + page_limit) if has_more else ""
            return RunListPage(
                items=tuple(items),
                next_cursor=next_cursor,
                has_more=has_more,
                total_estimate=total,
            )

    def exists(self, run_id: str) -> bool:
        rid = _text(run_id, "run_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT 1 FROM run_roots WHERE run_id = ?",
                [rid],
            ).fetchone()
            return row is not None

    def append_audit(
        self,
        *,
        actor_id: str,
        action: str,
        outcome: str,
        run_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        """Append an explicit redacted audit receipt."""

        now_ms = int(self._clock_ms())
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                record = self._insert_audit(
                    connection,
                    run_id=_text(run_id, "run_id", required=False),
                    actor_id=actor_id,
                    action=action,
                    outcome=outcome,
                    body=dict(body or {}),
                    recorded_at_ms=now_ms,
                )
                self._commit_if_idle(connection)
                return record
            except Exception:
                self._rollback_if_open(connection)
                raise

    def list_audits(
        self,
        *,
        run_id: str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> tuple[AuditRecord, ...]:
        page_limit = min(max(1, int(limit)), MAX_PAGE_LIMIT)
        with self._lock:
            connection = self._require()
            if run_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM run_audit_records
                    WHERE run_id = ?
                    ORDER BY recorded_at_ms ASC, audit_id ASC
                    LIMIT ?
                    """,
                    [_text(run_id, "run_id"), page_limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM run_audit_records
                    ORDER BY recorded_at_ms ASC, audit_id ASC
                    LIMIT ?
                    """,
                    [page_limit],
                ).fetchall()
            return tuple(self._row_to_audit(_row_mapping(row)) for row in rows)

    def lookup_idempotency(self, idempotency_key: str) -> RegistryTransactionReceipt | None:
        """Return a stored idempotent result without digest checks."""

        key = _text(idempotency_key, "idempotency_key")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT body_json FROM run_idempotency_records
                WHERE idempotency_key = ?
                """,
                [key],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            body = json.loads(str(_row_get(mapping, "body_json", default="{}") or "{}"))
            result = body.get("result") if isinstance(body, dict) else None
            if not isinstance(result, Mapping):
                return None
            receipt = RegistryTransactionReceipt.from_dict(result)
            # Present as a replay observation.
            return RegistryTransactionReceipt(
                operation=receipt.operation,
                outcome=RegistryTxOutcome.REPLAYED,
                run_id=receipt.run_id,
                run_revision=receipt.run_revision,
                handle_cid=receipt.handle_cid,
                integrity_cid=receipt.integrity_cid,
                previous_revision=receipt.previous_revision,
                previous_handle_cid=receipt.previous_handle_cid,
                reason_codes=tuple(receipt.reason_codes) + ("idempotent_replay",),
                committed_at_ms=receipt.committed_at_ms,
                replayed=True,
                body=dict(receipt.body),
            )

    def export_filesystem_tree(self, target_dir: Path | str) -> dict[str, Any]:
        """Export runs as a non-authoritative filesystem tree.

        Deleting or editing the export cannot affect registry authority.
        """

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        page = self.list_runs(limit=MAX_PAGE_LIMIT)
        exported = 0
        for record in page.items:
            ns_dir = target / "namespaces" / record.run_namespace.replace(":", "~")
            run_dir = ns_dir / "runs" / record.run_id.replace(":", "_")
            run_dir.mkdir(parents=True, exist_ok=True)
            root_path = run_dir / "root.json"
            head_path = run_dir / "head.json"
            root_path.write_text(
                _canonical_json(
                    {
                        "schema": RUN_ROOT_SCHEMA,
                        "run_id": record.run_id,
                        "run_namespace": record.run_namespace,
                        "repository_id": record.repository_id,
                        "worktree_id": record.worktree_id,
                        "source_kind": record.source_kind,
                        "created_at_ms": record.created_at_ms,
                        "authority": EXPORT_AUTHORITY,
                    }
                ),
                encoding="utf-8",
            )
            head_path.write_text(
                _canonical_json(
                    {
                        "schema": RUN_HEAD_SCHEMA,
                        "run_id": record.run_id,
                        "run_revision": record.run_revision,
                        "handle_cid": record.handle_cid,
                        "state": record.state,
                        "health": record.health,
                        "authority": EXPORT_AUTHORITY,
                    }
                ),
                encoding="utf-8",
            )
            exported += 1

        now_ms = int(self._clock_ms())
        export_digest = _digest_of(
            {
                "target": str(target),
                "run_count": exported,
                "created_at_ms": now_ms,
            }
        )
        receipt = {
            "schema": EXPORT_RECEIPT_SCHEMA,
            "export_id": _new_id("export"),
            "target_path": str(target),
            "run_count": exported,
            "export_digest": export_digest,
            "authority": EXPORT_AUTHORITY,
            "created_at_ms": now_ms,
            "created_at": _utc_iso_from_ms(now_ms),
        }
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                connection.execute(
                    """
                    INSERT INTO run_export_receipts (
                        export_id, target_path, run_count, export_digest,
                        authority, created_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt["export_id"],
                        receipt["target_path"],
                        exported,
                        export_digest,
                        EXPORT_AUTHORITY,
                        now_ms,
                        _canonical_json(receipt),
                    ],
                )
                self._commit_if_idle(connection)
            except Exception:
                self._rollback_if_open(connection)
                raise
        marker = target / "EXPORT_NON_AUTHORITATIVE.json"
        marker.write_text(_canonical_json(receipt), encoding="utf-8")
        return receipt

    # -- internal helpers ----------------------------------------------------

    def _lookup_idempotency(
        self,
        connection: Any,
        *,
        idempotency_key: str,
        request_digest: str,
    ) -> RegistryTransactionReceipt | None:
        row = connection.execute(
            """
            SELECT request_digest, body_json
            FROM run_idempotency_records
            WHERE idempotency_key = ?
            """,
            [idempotency_key],
        ).fetchone()
        if row is None:
            return None
        mapping = _row_mapping(row)
        stored_digest = str(_row_get(mapping, "request_digest", default="") or "")
        if stored_digest != request_digest:
            raise DatabaseIdempotencyConflictError(
                "idempotency key is already bound to a different request"
            )
        body = json.loads(str(_row_get(mapping, "body_json", default="{}") or "{}"))
        result = body.get("result") if isinstance(body, dict) else None
        if not isinstance(result, Mapping):
            raise DatabaseIdempotencyConflictError(
                "idempotency state contains an invalid matching result"
            )
        receipt = RegistryTransactionReceipt.from_dict(result)
        return RegistryTransactionReceipt(
            operation=receipt.operation,
            outcome=RegistryTxOutcome.REPLAYED,
            run_id=receipt.run_id,
            run_revision=receipt.run_revision,
            handle_cid=receipt.handle_cid,
            integrity_cid=receipt.integrity_cid,
            previous_revision=receipt.previous_revision,
            previous_handle_cid=receipt.previous_handle_cid,
            reason_codes=tuple(receipt.reason_codes) + ("idempotent_replay",),
            committed_at_ms=receipt.committed_at_ms,
            replayed=True,
            body=dict(receipt.body),
        )

    def _store_idempotency(
        self,
        connection: Any,
        *,
        idempotency_key: str,
        request_digest: str,
        command_kind: str,
        receipt: RegistryTransactionReceipt,
    ) -> None:
        result = receipt.to_dict()
        result_digest = _digest_of(result)
        body = {
            "schema": IDEMPOTENCY_RECORD_SCHEMA,
            "idempotency_key": idempotency_key,
            "request_digest": request_digest,
            "command_kind": command_kind,
            "result_digest": result_digest,
            "result": result,
        }
        connection.execute(
            """
            INSERT INTO run_idempotency_records (
                idempotency_key, request_digest, command_kind, result_digest,
                run_id, created_at_ms, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                idempotency_key,
                request_digest,
                command_kind,
                result_digest,
                receipt.run_id,
                int(receipt.committed_at_ms),
                _canonical_json(body),
            ],
        )

    def _insert_audit(
        self,
        connection: Any,
        *,
        run_id: str,
        actor_id: str,
        action: str,
        outcome: str,
        body: Mapping[str, Any],
        recorded_at_ms: int,
    ) -> AuditRecord:
        audit_id = _new_id("audit")
        redacted_body = redact_mapping(dict(body or {}))
        if not isinstance(redacted_body, Mapping):
            redacted_body = {"value": REDACTION_MARKER}
        # Ensure nested secret keys are markers.
        safe_body = _bounded_mapping(redacted_body, name="audit_body")
        connection.execute(
            """
            INSERT INTO run_audit_records (
                audit_id, run_id, actor_id, action, outcome,
                recorded_at_ms, redacted, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                audit_id,
                _text(run_id, "run_id", required=False),
                _text(actor_id, "actor_id"),
                _text(action, "action"),
                _text(outcome, "outcome"),
                int(recorded_at_ms),
                True,
                _canonical_json(safe_body),
            ],
        )
        return AuditRecord(
            audit_id=audit_id,
            run_id=str(run_id or ""),
            actor_id=str(actor_id),
            action=str(action),
            outcome=str(outcome),
            recorded_at_ms=int(recorded_at_ms),
            redacted=True,
            body=safe_body,
        )

    @staticmethod
    def _row_to_run(mapping: Mapping[str, Any]) -> RunRecord:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return RunRecord(
            run_id=str(_row_get(mapping, "run_id", default="") or ""),
            run_namespace=str(_row_get(mapping, "run_namespace", default="") or ""),
            repository_id=str(_row_get(mapping, "repository_id", default="") or ""),
            checkout_id=str(_row_get(mapping, "checkout_id", default="") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id", default="") or ""),
            source_kind=str(_row_get(mapping, "source_kind", default="") or ""),
            invocation_cid=str(_row_get(mapping, "invocation_cid", default="") or ""),
            prompt_cid=str(_row_get(mapping, "prompt_cid", default="") or ""),
            objective_cid=str(_row_get(mapping, "objective_cid", default="") or ""),
            target_resolution_receipt_cid=str(
                _row_get(mapping, "target_resolution_receipt_cid", default="") or ""
            ),
            lifecycle_profile_cid=str(
                _row_get(mapping, "lifecycle_profile_cid", default="") or ""
            ),
            created_at_ms=int(_row_get(mapping, "created_at_ms", default=0)),
            initial_handle_cid=str(
                _row_get(mapping, "initial_handle_cid", default="") or ""
            ),
            initial_revision=int(_row_get(mapping, "initial_revision", default=1)),
            run_revision=int(_row_get(mapping, "run_revision", default=1)),
            handle_cid=str(_row_get(mapping, "handle_cid", default="") or ""),
            semantic_id=str(_row_get(mapping, "semantic_id", default="") or ""),
            state=str(_row_get(mapping, "state", default="") or ""),
            health=str(_row_get(mapping, "health", default="") or ""),
            event_cursor=str(_row_get(mapping, "event_cursor", default="") or ""),
            updated_at_ms=int(_row_get(mapping, "updated_at_ms", default=0)),
            body=body,
        )

    @staticmethod
    def _row_to_audit(mapping: Mapping[str, Any]) -> AuditRecord:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return AuditRecord(
            audit_id=str(_row_get(mapping, "audit_id", default="") or ""),
            run_id=str(_row_get(mapping, "run_id", default="") or ""),
            actor_id=str(_row_get(mapping, "actor_id", default="") or ""),
            action=str(_row_get(mapping, "action", default="") or ""),
            outcome=str(_row_get(mapping, "outcome", default="") or ""),
            recorded_at_ms=int(_row_get(mapping, "recorded_at_ms", default=0)),
            redacted=bool(_row_get(mapping, "redacted", default=True)),
            body=body,
        )


def open_database_run_registry(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    max_list: int = HARD_MAX_LIST,
) -> DatabaseRunRegistry:
    """Open and return an initialized :class:`DatabaseRunRegistry`."""

    return DatabaseRunRegistry(
        database_path,
        clock_ms=clock_ms,
        max_list=max_list,
    ).open()


__all__ = (
    "ALLOWED_CREATE_SOURCE_KINDS",
    "AUDIT_RECORD_SCHEMA",
    "DATABASE_RUN_REGISTRY_INTERFACE",
    "DATABASE_RUN_REGISTRY_SCHEMA",
    "DEFAULT_PAGE_LIMIT",
    "EXPORT_AUTHORITY",
    "FORBIDDEN_CREATE_SOURCE_KINDS",
    "HARD_MAX_LIST",
    "IDEMPOTENCY_RECORD_SCHEMA",
    "MAX_PAGE_LIMIT",
    "AuditRecord",
    "DatabaseIdempotencyConflictError",
    "DatabaseRunCasConflictError",
    "DatabaseRunExistsError",
    "DatabaseRunNotFoundError",
    "DatabaseRunRegistry",
    "DatabaseRunRegistryBoundsError",
    "DatabaseRunRegistryError",
    "DatabaseRunRegistryNotOpenError",
    "DatabaseRunSourceError",
    "DuckDBUnavailableError",
    "REDACTION_MARKER",
    "RegistryOperation",
    "RegistryTransactionReceipt",
    "RegistryTxOutcome",
    "RunListPage",
    "RunRecord",
    "duckdb_available",
    "open_database_run_registry",
)
