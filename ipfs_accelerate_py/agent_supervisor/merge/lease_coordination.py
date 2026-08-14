"""Lease-safe Profile G adapters and DuckDB coordination for daemon lanes.

Bundle supervisors are separate processes, while DuckDB permits one external
writer process at a time. Each operation therefore takes a process-shared file
lock, opens a short-lived DuckDB connection, and checks the accepted claim and
fencing token inside one transaction. An expired worker cannot publish progress
or a terminal receipt after a takeover.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

from ..task_sources.duckdb_state import (
    DuckDBConnection as _DuckConnection,
)
from ..task_sources.duckdb_state import (
    DuckDBRow as _DuckRow,
)
from ..task_sources.duckdb_state import (
    connect_duckdb_with_policy as _connect_duckdb_with_policy,
)
from ..task_sources.duckdb_state import (
    exclusive_file_lock as _exclusive_file_lock,
)
from ..task_sources.task_identity import canonical_bundle_identity

MIN_LEASE_MS = 5_000
MAX_LEASE_MS = 300_000
PROVIDER_VERSION = "3.2.0"
MAX_PERSISTED_DEPENDENCY_REPAIRS = 256
STRUCTURAL_DEPENDENCY_REPAIR_KINDS = frozenset(
    {"missing_dependency", "dependency_cycle", "duplicate_alias", "duplicate_task"}
)
READY_BUNDLE_TASK_STATUSES = frozenset({"todo", "ready", "needed", "queued", "in_progress"})
COORDINATION_STORE_SCHEMA = "ipfs_accelerate_py.agent_supervisor.lease-coordination-duckdb@1"
COORDINATION_LOCK_TIMEOUT_SECONDS = 30.0
COORDINATION_DUCKDB_MEMORY_LIMIT = "256MB"
COORDINATION_DUCKDB_CONNECT_MAX_ATTEMPTS = 8
COORDINATION_DUCKDB_CONNECT_INITIAL_BACKOFF_SECONDS = 0.1
COORDINATION_DUCKDB_CONNECT_MAX_BACKOFF_SECONDS = 2.0
MAX_PERSISTED_HEARTBEATS_PER_LEASE = 8
SMALL_STORE_FULL_ARTIFACT_LIMIT = 10_000
DISTRIBUTED_INPUT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor/immutable-lane-input@1"
)
WORKER_CAPABILITY_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor/worker-capability-receipt@1"
)
WORKER_ENVIRONMENT_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor/worker-environment-receipt@1"
)
DISTRIBUTED_LANE_DISPATCH_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor/distributed-lane-dispatch@1"
)
REMOTE_LANE_RESULT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor/remote-lane-result@1"
)
DISTRIBUTED_PUBLICATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor/distributed-publication@1"
)
DISTRIBUTED_QUARANTINE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor/distributed-result-quarantine@1"
)
SINGLE_FLIGHT_STORE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.distributed-single-flight@1"
)
SINGLE_FLIGHT_OUTCOME_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/distributed-single-flight-outcome@1"
)
SINGLE_FLIGHT_ATTESTATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/distributed-single-flight-attestation@1"
)
DEFAULT_SINGLE_FLIGHT_LEASE_SECONDS = 30.0
DEFAULT_SINGLE_FLIGHT_OUTCOME_TTL_SECONDS = 60.0
DEFAULT_SINGLE_FLIGHT_POLL_SECONDS = 0.02
DEFAULT_SINGLE_FLIGHT_MAX_OUTCOME_BYTES = 256 * 1024
# Profile-G v1 TaskSpec attempt ceiling (schema range [1, 100]); supervisor
# policy may still use 0 as the unlimited sentinel, mapped to this ceiling
# only when projecting immutable TaskSpecs.
PROFILE_G_MAX_TASK_ATTEMPTS = 100

_COORDINATION_COMPACTION_TABLES = (
    "artifacts",
    "tasks",
    "task_aliases",
    "task_dependencies",
    "task_dependency_repairs",
    "task_dependency_repair_state",
    "leases",
    "token_history",
    "heartbeats",
    "receipts",
    "distributed_inputs",
    "worker_capability_receipts",
    "worker_environment_receipts",
    "distributed_dispatches",
    "distributed_publications",
    "coordination_metadata",
)

# DuckDB's ``SHOW TABLES`` and ``PRAGMA table_info`` are intentionally narrow:
# the former only sees the current schema and the latter omits constraints,
# collations, indexes, and other persistent catalog objects.  Compaction must
# compare the complete path-independent catalog of its source and freshly
# initialized target or it could silently erase state it does not understand.
# Object identifiers, database names/paths, estimated sizes, and read-only
# connection state are excluded because they necessarily differ between the
# authoritative source and temporary target.
_COORDINATION_PERSISTENT_CATALOG_QUERIES = (
    (
        "database",
        """
        SELECT database_name, comment, internal, type, encrypted, cipher, options
        FROM duckdb_databases()
        ORDER BY database_name
        """,
    ),
    (
        "schemas",
        """
        SELECT schema_name, comment, tags, internal, sql
        FROM duckdb_schemas()
        WHERE database_name=current_database()
        ORDER BY schema_name
        """,
    ),
    (
        "tables",
        """
        SELECT schema_name, table_name, comment, tags, internal, temporary,
               has_primary_key, column_count, index_count,
               check_constraint_count, sql
        FROM duckdb_tables()
        WHERE database_name=current_database()
        ORDER BY schema_name, table_name
        """,
    ),
    (
        "views",
        """
        SELECT schema_name, view_name, comment, tags, internal, temporary,
               column_count, sql, is_bound
        FROM duckdb_views()
        WHERE database_name=current_database()
        ORDER BY schema_name, view_name
        """,
    ),
    (
        "sequences",
        """
        SELECT schema_name, sequence_name, comment, tags, temporary,
               start_value, min_value, max_value, increment_by, cycle,
               last_value, sql
        FROM duckdb_sequences()
        WHERE database_name=current_database()
        ORDER BY schema_name, sequence_name
        """,
    ),
    (
        "functions",
        """
        SELECT schema_name, function_name, alias_of, function_type, comment,
               tags, return_type, parameters, parameter_types, varargs,
               macro_definition, has_side_effects, internal, stability,
               categories
        FROM duckdb_functions()
        WHERE database_name=current_database()
        ORDER BY schema_name, function_name, function_type
        """,
    ),
    (
        "types",
        """
        SELECT schema_name, type_name, type_size, logical_type, type_category,
               comment, tags, internal, labels
        FROM duckdb_types()
        WHERE database_name=current_database()
        ORDER BY schema_name, type_name
        """,
    ),
    (
        "indexes",
        """
        SELECT schema_name, index_name, table_name, comment, tags, is_unique,
               is_primary, expressions, sql
        FROM duckdb_indexes()
        WHERE database_name=current_database()
        ORDER BY schema_name, index_name
        """,
    ),
    (
        "constraints",
        """
        SELECT schema_name, table_name, constraint_type, constraint_text,
               expression, constraint_column_indexes, constraint_column_names,
               constraint_name, referenced_table, referenced_column_names
        FROM duckdb_constraints()
        WHERE database_name=current_database()
        ORDER BY schema_name, table_name, constraint_type, constraint_text,
                 constraint_name
        """,
    ),
    (
        "columns",
        """
        SELECT schema_name, table_name, column_name, column_index, comment,
               internal, column_default, is_nullable, data_type,
               character_maximum_length, numeric_precision,
               numeric_precision_radix, numeric_scale
        FROM duckdb_columns()
        WHERE database_name=current_database()
        ORDER BY schema_name, table_name, column_index
        """,
    ),
)

_COORDINATION_ADDITIVE_NOT_NULL_DEFAULT_COLUMNS = {
    ("tasks", "registered_at_ms"): ("BIGINT", "0"),
    ("tasks", "updated_at_ms"): ("BIGINT", "0"),
    ("leases", "retry_not_before_ms"): ("BIGINT", "0"),
}


def _is_transient_duckdb_lock_error(exc: Exception) -> bool:
    """Return whether ``exc`` is DuckDB's narrow external-lock conflict."""

    exception_type = type(exc)
    if (
        exception_type.__module__ not in {"duckdb", "_duckdb"}
        or exception_type.__name__ not in {"IOException", "OperationalError"}
    ):
        return False
    message = str(exc).casefold()
    return "could not set lock" in message and "conflicting lock" in message


def profile_g_task_attempt_limit(value: Any, *, default: int = 3) -> int:
    """Return the supervisor attempt policy accepted beside Profile-G tasks.

    Zero is the unlimited sentinel. Values above the Profile-G v1 boundary
    are rejected instead of being silently rewritten across queue layers.
    """

    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("max_attempts must be an integer")
    if raw < 0 or raw > PROFILE_G_MAX_TASK_ATTEMPTS:
        raise ValueError(
            "max_attempts must be between 0 and "
            f"{PROFILE_G_MAX_TASK_ATTEMPTS} for Profile-G tasks"
        )
    return raw


def _profile_g_task_spec_attempt_limit(value: Any, *, default: int = 3) -> int:
    """Translate supervisor attempt policy into strict Profile-G v1 syntax.

    Profile-G v1 has no unlimited sentinel and accepts only values in
    ``[1, 100]``. The executable bundle retains zero as its authoritative
    unlimited policy; only the immutable TaskSpec uses the schema ceiling.
    """

    selected = profile_g_task_attempt_limit(value, default=default)
    return PROFILE_G_MAX_TASK_ATTEMPTS if selected == 0 else selected


def _coordinator_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Open one flock-serialized DuckDB connection for a public operation."""

    @wraps(method)
    def wrapped(self: LeaseCoordinator, *args: Any, **kwargs: Any) -> Any:
        with self._database_operation():
            return method(self, *args, **kwargs)

    return wrapped


def _is_transient_duckdb_file_lock_error(
    error: BaseException,
    duckdb_module: Any,
) -> bool:
    """Recognize only DuckDB's cross-process file-lock connect failure."""

    io_exception = getattr(duckdb_module, "IOException", None)
    if not isinstance(io_exception, type) or not isinstance(error, io_exception):
        return False
    message = str(error).casefold()
    return (
        "could not set lock on file" in message
        and "conflicting lock is held" in message
    )


def _connect_coordination_duckdb(
    duckdb_module: Any,
    path: Path | str,
    *,
    read_only: bool = False,
) -> Any:
    """Open a coordination connection with bounded lock retries.

    The surrounding advisory lock remains held while this retries.  The retry
    is deliberately limited to DuckDB's explicit cross-process file-lock
    ``IOException``; corrupt stores, permission failures, and other I/O errors
    fail immediately with their original exception.
    """

    delay = COORDINATION_DUCKDB_CONNECT_INITIAL_BACKOFF_SECONDS
    for attempt in range(1, COORDINATION_DUCKDB_CONNECT_MAX_ATTEMPTS + 1):
        try:
            return _connect_duckdb_with_policy(
                duckdb_module,
                path,
                read_only=read_only,
                configuration={
                    "threads": 1,
                    "memory_limit": COORDINATION_DUCKDB_MEMORY_LIMIT,
                },
            )
        except Exception as exc:
            if (
                not _is_transient_duckdb_file_lock_error(exc, duckdb_module)
                or attempt >= COORDINATION_DUCKDB_CONNECT_MAX_ATTEMPTS
            ):
                raise
            time.sleep(delay)
            delay = min(
                delay * 2,
                COORDINATION_DUCKDB_CONNECT_MAX_BACKOFF_SECONDS,
            )
    raise AssertionError("unreachable DuckDB connection retry state")


def _coordination_table_schema(connection: Any, table: str) -> tuple[tuple[Any, ...], ...]:
    if table not in _COORDINATION_COMPACTION_TABLES:
        raise ValueError("coordination compaction table is outside the closed schema")
    return tuple(
        tuple(row)
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    )


def _add_coordination_not_null_default_column(
    connection: Any,
    *,
    table: str,
    column: str,
) -> None:
    """Transactionally add one closed legacy column on DuckDB 1.5.x."""

    try:
        column_type, default = _COORDINATION_ADDITIVE_NOT_NULL_DEFAULT_COLUMNS[
            (table, column)
        ]
    except KeyError as exc:
        raise ValueError(
            "coordination schema evolution requested an unknown column"
        ) from exc
    # DuckDB 1.5 rejects ADD COLUMN when NOT NULL is written in the same DDL.
    # Adding a DEFAULT backfills existing rows; the second statement can then
    # install the constraint.  The caller's schema transaction makes a crash
    # or exception between these statements roll back the entire evolution.
    connection.execute(
        f'ALTER TABLE "{table}" ADD COLUMN "{column}" '
        f"{column_type} DEFAULT {default}"
    )
    connection.execute(
        f'ALTER TABLE "{table}" ALTER COLUMN "{column}" SET NOT NULL'
    )


def _coordination_persistent_catalog(
    connection: Any,
) -> tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]:
    """Return the complete path-independent persistent DuckDB catalog."""

    current_row = connection.execute("SELECT current_database()").fetchone()
    if (
        not isinstance(current_row, tuple)
        or len(current_row) != 1
        or type(current_row[0]) is not str
        or not current_row[0]
    ):
        raise RuntimeError(
            "coordination compaction could not identify the current catalog"
        )
    current_database = current_row[0]
    result: list[tuple[str, tuple[tuple[Any, ...], ...]]] = []
    for category, query in _COORDINATION_PERSISTENT_CATALOG_QUERIES:
        rows = [tuple(row) for row in connection.execute(query).fetchall()]
        if category == "database":
            # File-backed database names differ because the target has a
            # temporary filename.  Normalize only the exact current database;
            # any attached catalog remains visible under its real name.
            rows = [
                (
                    "current" if row[0] == current_database else row[0],
                    *row[1:],
                )
                for row in rows
            ]
            rows.sort(key=lambda row: tuple(repr(value) for value in row))
        result.append((category, tuple(rows)))
    return tuple(result)


def _copy_coordination_store_rows(source: Any, target: Any) -> None:
    """Copy only the sealed coordination catalog between local connections."""

    expected_tables = set(_COORDINATION_COMPACTION_TABLES)
    source_tables = {
        str(row[0]) for row in source.execute("SHOW TABLES").fetchall()
    }
    target_tables = {
        str(row[0]) for row in target.execute("SHOW TABLES").fetchall()
    }
    if source_tables != expected_tables or target_tables != expected_tables:
        raise RuntimeError("coordination compaction observed a foreign table set")

    source_catalog = _coordination_persistent_catalog(source)
    target_catalog = _coordination_persistent_catalog(target)
    if source_catalog != target_catalog:
        source_by_category = dict(source_catalog)
        target_by_category = dict(target_catalog)
        changed = sorted(
            category
            for category in source_by_category.keys() | target_by_category.keys()
            if source_by_category.get(category) != target_by_category.get(category)
        )
        raise RuntimeError(
            "coordination compaction persistent catalog mismatch: "
            + ", ".join(changed)
        )

    # The fresh target receives one metadata row during schema initialization.
    # Clear it so every copied row has the same insert semantics and a missing
    # source metadata row cannot be silently synthesized by compaction.
    target.execute("DELETE FROM coordination_metadata")
    for table in _COORDINATION_COMPACTION_TABLES:
        source_schema = _coordination_table_schema(source, table)
        target_schema = _coordination_table_schema(target, table)
        if source_schema != target_schema:
            raise RuntimeError(
                f"coordination compaction schema mismatch for table {table!r}"
            )
        cursor = source.execute(f'SELECT * FROM "{table}"')
        column_count = len(cursor.description or ())
        if column_count < 1:
            raise RuntimeError(
                f"coordination compaction table {table!r} has no columns"
            )
        insert_sql = (
            f'INSERT INTO "{table}" VALUES ('
            + ", ".join("?" for _ in range(column_count))
            + ")"
        )
        while True:
            rows = cursor.fetchmany(256)
            if not rows:
                break
            target.executemany(insert_sql, rows)


def _fsync_path(path: Path, *, directory: bool = False) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LeaseError(RuntimeError):
    """Base error for lease protocol failures."""

    code = "G_CLAIM_CONFLICT"


class LeaseConflictError(LeaseError):
    """Raised when another non-expired claim owns the task."""


class ExecutionScopeConflictError(LeaseConflictError):
    """Raised when another task revision owns the same execution scope."""

    code = "G_EXECUTION_SCOPE_CONFLICT"


class DependencyNotReadyError(LeaseConflictError):
    """Raised when a task's prerequisite receipts have not all succeeded."""

    code = "G_DEPENDENCY_NOT_READY"

    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


class LeaseExpiredError(LeaseError):
    """Raised when a caller's accepted lease is no longer current."""

    code = "G_LEASE_EXPIRED"


class StaleFencingTokenError(LeaseError):
    """Raised when execution uses a superseded fencing token."""

    code = "G_CLAIM_CONFLICT"


def canonical_profile_g_bytes(value: Any) -> bytes:
    """Encode canonical DAG-JSON-compatible bytes and reject floats."""

    def check(item: Any) -> None:
        if item is None or isinstance(item, str | bool | int):
            return
        if isinstance(item, float):
            raise ValueError("Profile G artifacts cannot contain floats")
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Profile G object keys must be strings")
            for child in item.values():
                check(child)
            return
        raise ValueError(f"unsupported Profile G value: {type(item).__name__}")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def profile_g_cid(value: Any) -> str:
    """Return CIDv1 DAG-JSON/sha2-256 without requiring an optional codec package."""

    digest = hashlib.sha256(canonical_profile_g_bytes(value)).digest()
    # CIDv1 + dag-json (0x0129 varint) + sha2-256 multihash.
    raw = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def _is_profile_g_cid(value: Any) -> bool:
    """Return whether ``value`` is canonical Profile-G CIDv1 identity text."""

    if not isinstance(value, str) or not value or value != value.lower():
        return False
    if not value.startswith("b"):
        return False
    encoded = value[1:]
    padding = "=" * ((8 - len(encoded) % 8) % 8)
    try:
        raw = base64.b32decode(encoded.upper() + padding, casefold=False)
    except (ValueError, TypeError):
        return False
    expected_prefix = b"\x01\xa9\x02\x12\x20"
    return bool(
        len(raw) == len(expected_prefix) + hashlib.sha256().digest_size
        and raw.startswith(expected_prefix)
        and "b"
        + base64.b32encode(raw).decode("ascii").rstrip("=").lower()
        == value
    )


def _content_digest(value: Any) -> str:
    """Return the explicit sha256 binding used by distributed envelopes."""

    return "sha256:" + hashlib.sha256(canonical_profile_g_bytes(value)).hexdigest()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    # The round trip detaches caller-owned mutable containers and verifies the
    # complete value against the integer-only canonical artifact contract.
    encoded = canonical_profile_g_bytes(dict(value))
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)
    return decoded


def _record_body(payload: Mapping[str, Any], *identity_fields: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in identity_fields and value not in (None, "")
    }


@dataclass(frozen=True)
class ImmutableLaneInputArtifact:
    """Content-addressed, immutable input handed to a remote lane."""

    repository_id: str
    task_cid: str
    payload: Mapping[str, Any]
    created_at_ms: int
    schema: str = DISTRIBUTED_INPUT_SCHEMA
    artifact_id: str = ""
    digest: str = ""

    def __post_init__(self) -> None:
        repository_id = _required_text(self.repository_id, "repository_id")
        task_cid = _required_text(self.task_cid, "task_cid")
        created_at_ms = _timestamp(self.created_at_ms, "created_at_ms")
        if self.schema != DISTRIBUTED_INPUT_SCHEMA:
            raise ValueError("unsupported immutable lane input schema")
        payload = _canonical_mapping(self.payload, "payload")
        body = {
            "schema": self.schema,
            "repository_id": repository_id,
            "task_cid": task_cid,
            "payload": payload,
            "created_at_ms": created_at_ms,
        }
        artifact_id = profile_g_cid(body)
        digest = _content_digest(body)
        if self.artifact_id and self.artifact_id != artifact_id:
            raise ValueError("immutable lane input artifact_id does not match content")
        if self.digest and self.digest != digest:
            raise ValueError("immutable lane input digest does not match content")
        object.__setattr__(self, "repository_id", repository_id)
        object.__setattr__(self, "task_cid", task_cid)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repository_id": self.repository_id,
            "task_cid": self.task_cid,
            "payload": dict(self.payload),
            "created_at_ms": self.created_at_ms,
            "artifact_id": self.artifact_id,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImmutableLaneInputArtifact:
        return cls(
            repository_id=value.get("repository_id", ""),
            task_cid=value.get("task_cid", ""),
            payload=value.get("payload", {}),
            created_at_ms=value.get("created_at_ms", 0),
            schema=value.get("schema", DISTRIBUTED_INPUT_SCHEMA),
            artifact_id=value.get("artifact_id", ""),
            digest=value.get("digest", ""),
        )


@dataclass(frozen=True)
class WorkerCapabilityReceipt:
    """Expiring declaration of the exact capabilities offered by one worker."""

    worker_id: str
    capabilities: tuple[str, ...]
    issued_at_ms: int
    expires_at_ms: int
    capability_revision: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = WORKER_CAPABILITY_RECEIPT_SCHEMA
    receipt_id: str = ""
    digest: str = ""

    def __post_init__(self) -> None:
        worker_id = _required_text(self.worker_id, "worker_id")
        issued = _timestamp(self.issued_at_ms, "issued_at_ms")
        expires = _timestamp(self.expires_at_ms, "expires_at_ms")
        if expires <= issued:
            raise ValueError("capability receipt must expire after it is issued")
        if self.schema != WORKER_CAPABILITY_RECEIPT_SCHEMA:
            raise ValueError("unsupported worker capability receipt schema")
        if isinstance(self.capabilities, str):
            raise ValueError("capabilities must be an iterable of strings")
        capabilities = tuple(
            sorted({_required_text(item, "capability") for item in self.capabilities})
        )
        if not capabilities:
            raise ValueError("capability receipt must declare at least one capability")
        revision = str(self.capability_revision or "")
        metadata = _canonical_mapping(self.metadata, "metadata")
        body = {
            "schema": self.schema,
            "worker_id": worker_id,
            "capabilities": list(capabilities),
            "issued_at_ms": issued,
            "expires_at_ms": expires,
            "capability_revision": revision,
            "metadata": metadata,
        }
        receipt_id = profile_g_cid(body)
        digest = _content_digest(body)
        if self.receipt_id and self.receipt_id != receipt_id:
            raise ValueError("capability receipt_id does not match content")
        if self.digest and self.digest != digest:
            raise ValueError("capability receipt digest does not match content")
        object.__setattr__(self, "worker_id", worker_id)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "capability_revision", revision)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "receipt_id", receipt_id)
        object.__setattr__(self, "digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "capabilities": list(self.capabilities),
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "capability_revision": self.capability_revision,
            "metadata": dict(self.metadata),
            "receipt_id": self.receipt_id,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerCapabilityReceipt:
        return cls(
            worker_id=value.get("worker_id", ""),
            capabilities=tuple(value.get("capabilities", ())),
            issued_at_ms=value.get("issued_at_ms", 0),
            expires_at_ms=value.get("expires_at_ms", 0),
            capability_revision=value.get("capability_revision", ""),
            metadata=value.get("metadata", {}),
            schema=value.get("schema", WORKER_CAPABILITY_RECEIPT_SCHEMA),
            receipt_id=value.get("receipt_id", ""),
            digest=value.get("digest", ""),
        )

    def validate_at(self, now_ms: int) -> None:
        now = _timestamp(now_ms, "now_ms")
        if now < self.issued_at_ms:
            raise ValueError("capability receipt is not yet valid")
        if now >= self.expires_at_ms:
            raise ValueError("capability receipt has expired")


@dataclass(frozen=True)
class WorkerEnvironmentReceipt:
    """Expiring environment identity bound to one worker capability receipt."""

    worker_id: str
    environment_id: str
    capability_receipt_id: str
    issued_at_ms: int
    expires_at_ms: int
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = WORKER_ENVIRONMENT_RECEIPT_SCHEMA
    receipt_id: str = ""
    digest: str = ""

    def __post_init__(self) -> None:
        worker_id = _required_text(self.worker_id, "worker_id")
        environment_id = _required_text(self.environment_id, "environment_id")
        capability_receipt_id = _required_text(
            self.capability_receipt_id, "capability_receipt_id"
        )
        issued = _timestamp(self.issued_at_ms, "issued_at_ms")
        expires = _timestamp(self.expires_at_ms, "expires_at_ms")
        if expires <= issued:
            raise ValueError("environment receipt must expire after it is issued")
        if self.schema != WORKER_ENVIRONMENT_RECEIPT_SCHEMA:
            raise ValueError("unsupported worker environment receipt schema")
        attributes = _canonical_mapping(self.attributes, "attributes")
        body = {
            "schema": self.schema,
            "worker_id": worker_id,
            "environment_id": environment_id,
            "capability_receipt_id": capability_receipt_id,
            "issued_at_ms": issued,
            "expires_at_ms": expires,
            "attributes": attributes,
        }
        receipt_id = profile_g_cid(body)
        digest = _content_digest(body)
        if self.receipt_id and self.receipt_id != receipt_id:
            raise ValueError("environment receipt_id does not match content")
        if self.digest and self.digest != digest:
            raise ValueError("environment receipt digest does not match content")
        object.__setattr__(self, "worker_id", worker_id)
        object.__setattr__(self, "environment_id", environment_id)
        object.__setattr__(self, "capability_receipt_id", capability_receipt_id)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "receipt_id", receipt_id)
        object.__setattr__(self, "digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "environment_id": self.environment_id,
            "capability_receipt_id": self.capability_receipt_id,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "attributes": dict(self.attributes),
            "receipt_id": self.receipt_id,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerEnvironmentReceipt:
        return cls(
            worker_id=value.get("worker_id", ""),
            environment_id=value.get("environment_id", ""),
            capability_receipt_id=value.get("capability_receipt_id", ""),
            issued_at_ms=value.get("issued_at_ms", 0),
            expires_at_ms=value.get("expires_at_ms", 0),
            attributes=value.get("attributes", {}),
            schema=value.get("schema", WORKER_ENVIRONMENT_RECEIPT_SCHEMA),
            receipt_id=value.get("receipt_id", ""),
            digest=value.get("digest", ""),
        )

    def validate_at(self, now_ms: int) -> None:
        now = _timestamp(now_ms, "now_ms")
        if now < self.issued_at_ms:
            raise ValueError("environment receipt is not yet valid")
        if now >= self.expires_at_ms:
            raise ValueError("environment receipt has expired")


def _link(value: Any) -> str:
    """Create a content link for adapter inputs that are not already artifacts."""

    return profile_g_cid(value)


def _bundle_task_statuses(bundle: Mapping[str, Any]) -> set[str]:
    """Return normalized explicit member statuses from a bundle payload."""

    tasks = bundle.get("tasks")
    if not isinstance(tasks, (list, tuple)):
        return set()
    statuses: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("status") in (None, ""):
            continue
        status = str(task["status"]).strip().lower().replace("-", "_").replace(" ", "_")
        if status in {"done", "complete"}:
            status = "completed"
        elif status in {"active"}:
            status = "in_progress"
        statuses.add(status)
    return statuses


def _reopens_blocked_bundle(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Return whether authoritative discovery reopened previously blocked work."""

    previous_statuses = _bundle_task_statuses(previous)
    current_statuses = _bundle_task_statuses(current)
    return "blocked" in previous_statuses and bool(current_statuses & READY_BUNDLE_TASK_STATUSES)


def _bundle_execution_tasks(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only members owned by this bundle execution slice.

    Member aliases are receipt authorities, so a later slice must not remap
    members completed by an earlier slice. Bundles without slice metadata retain
    the legacy all-member behavior; an explicit empty slice owns no members.
    """

    tasks = bundle.get("tasks")
    members = (
        [item for item in tasks if isinstance(item, Mapping)]
        if isinstance(tasks, (list, tuple))
        else []
    )
    if (
        "execution_slice_task_cids" not in bundle
        and "execution_slice_task_ids" not in bundle
    ):
        return members

    def values(raw: Any) -> set[str]:
        if isinstance(raw, str):
            items = raw.split(",")
        elif isinstance(raw, (list, tuple, set)):
            items = raw
        elif raw in (None, ""):
            items = ()
        else:
            items = (raw,)
        return {str(item).strip() for item in items if str(item).strip()}

    selected_cids = values(bundle.get("execution_slice_task_cids"))
    selected_ids = values(bundle.get("execution_slice_task_ids"))
    return [
        item
        for item in members
        if str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
        in selected_cids
        or str(item.get("task_id") or "").strip() in selected_ids
    ]


def _dependency_task_cids(bundle: Mapping[str, Any]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Return normalized prerequisite CIDs and their bounded source provenance.

    Objective-graph payloads have existed in a few compatible shapes.  Accepting
    those shapes here keeps the lease boundary strict without coupling it to the
    planner implementation (which would also introduce an import cycle).
    """

    found: dict[str, list[dict[str, Any]]] = {}

    def add(value: Any, source: str, *, edge: Mapping[str, Any] | None = None) -> None:
        values: list[Any]
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        elif value in (None, ""):
            values = []
        else:
            values = [value]
        for item in values:
            if isinstance(item, Mapping):
                cid = str(
                    item.get("dependency_task_cid")
                    or item.get("prerequisite_task_cid")
                    or item.get("task_cid")
                    or item.get("cid")
                    or ""
                ).strip()
            else:
                cid = str(item).strip()
            if not cid:
                continue
            provenance: dict[str, Any] = {"source": source}
            if edge is not None:
                for key in ("kind", "reason", "source_path", "source_task_cid", "target_task_cid"):
                    if edge.get(key) not in (None, ""):
                        provenance[key] = edge[key]
                edge_provenance = edge.get("provenance")
                if isinstance(edge_provenance, Mapping):
                    provenance["edge_provenance"] = dict(edge_provenance)
            records = found.setdefault(cid, [])
            if provenance not in records and len(records) < 16:
                records.append(provenance)

    # Presence matters here: the objective planner emits an explicit bundle-
    # scoped projection, and an empty projection means every prerequisite is
    # internal to this execution unit.  Falling through to member or embedded
    # dependencies would reintroduce CIDs that never receive their own lease.
    has_bundle_dependency_projection = "dependency_task_cids" in bundle
    add(bundle.get("dependency_task_cids"), "bundle.dependency_task_cids")
    embedded = bundle.get("profile_g")
    if not has_bundle_dependency_projection and isinstance(embedded, Mapping):
        embedded_task = embedded.get("task")
        if isinstance(embedded_task, Mapping):
            add(embedded_task.get("dependency_task_cids"), "bundle.profile_g.task.dependency_task_cids")
    edges = bundle.get("dependency_edges")
    if not has_bundle_dependency_projection and isinstance(edges, (list, tuple)):
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                continue
            add(
                edge.get("dependency_task_cid")
                or edge.get("prerequisite_task_cid")
                or edge.get("source_task_cid"),
                f"bundle.dependency_edges[{index}]",
                edge=edge,
            )
    # Bundle execution CIDs are the receipt authority.  Member dependencies
    # remain a compatibility fallback only when no aggregate bundle edge was
    # supplied; mixing both identities would require two receipts for one
    # logical prerequisite and could permanently block otherwise-ready work.
    tasks = bundle.get("tasks")
    if not has_bundle_dependency_projection and not found and isinstance(tasks, (list, tuple)):
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                continue
            for key in ("dependency_task_cids", "prerequisite_task_cids"):
                add(task.get(key), f"bundle.tasks[{index}].{key}")
    explicit_provenance = bundle.get("dependency_provenance")
    if not isinstance(explicit_provenance, Mapping) and isinstance(embedded, Mapping):
        explicit_provenance = embedded.get("dependency_provenance")
    if isinstance(explicit_provenance, Mapping):
        for raw_cid, raw_items in explicit_provenance.items():
            cid = str(raw_cid).strip()
            if cid not in found:
                continue
            items = raw_items if isinstance(raw_items, (list, tuple)) else [raw_items]
            for item in items:
                record = dict(item) if isinstance(item, Mapping) else {"detail": str(item)}
                record.setdefault("source", "bundle.dependency_provenance")
                if record not in found[cid] and len(found[cid]) < 16:
                    found[cid].append(record)
    return sorted(found), found


def _dependency_repair_evidence(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Return a bounded copy of planner-produced dependency repair records."""

    value = bundle.get("dependency_repair_evidence")
    embedded = bundle.get("profile_g")
    if not isinstance(value, (list, tuple)) and isinstance(embedded, Mapping):
        value = embedded.get("dependency_repair_evidence")
    records = [dict(item) for item in value or [] if isinstance(item, Mapping)] if isinstance(value, (list, tuple)) else []
    return records[:MAX_PERSISTED_DEPENDENCY_REPAIRS], len(records)


def adapt_goal_bundle(bundle: Mapping[str, Any], *, created_at_ms: int | None = None) -> dict[str, Any]:
    """Adapt one objective bundle payload into a canonical Goal/Subgoal/TaskSpec chain."""

    now = int(time.time() * 1000) if created_at_ms is None else int(created_at_ms)
    bundle_key = str(bundle.get("bundle_key") or "objective/general")
    correlation = str(bundle.get("correlation_id") or bundle_key)[:128]
    owner_did = str(bundle.get("owner_did") or "did:web:ipfs-accelerate.local")
    canonical_identity = canonical_bundle_identity(bundle)
    objective = {
        "bundle_key": bundle_key,
        "source_todo": str(bundle.get("source_todo") or ""),
        "task_ids": sorted(
            str(item.get("task_id"))
            for item in bundle.get("tasks", [])
            if isinstance(item, Mapping) and item.get("task_id")
        ),
    }
    objective_cid = _link(objective)
    policy_cid = str(bundle.get("policy_cid") or _link({"policy": "accelerator-daemon-lane-v1"}))
    goal = {
        "schema": "mcp++/profile-g/goal@1",
        "created_at_ms": now,
        "parents": [],
        "correlation_id": correlation,
        "owner_did": owner_did,
        "objective_cid": objective_cid,
        "policy_cid": policy_cid,
        "parent_goal_cids": [],
        "labels": sorted({"accelerator", "daemon-lane"}),
    }
    goal_cid = profile_g_cid(goal)
    subgoal = {
        "schema": "mcp++/profile-g/subgoal@1",
        "created_at_ms": now,
        "parents": [goal_cid],
        "correlation_id": correlation,
        "goal_cid": goal_cid,
        "parent_subgoal_cid": None,
        "objective_cid": objective_cid,
        "decomposition_method": "objective-bundle-v1",
        "decomposer_cid": _link({"adapter": "ipfs_accelerate_py.agent_supervisor"}),
        "selection_cid": None,
    }
    subgoal_cid = profile_g_cid(subgoal)
    template_cid = _link({"tool": "codex.todo_bundle", "bundle_key": bundle_key})
    plan = {
        "schema": "mcp++/profile-g/plan-branch@1",
        "created_at_ms": now,
        "parents": [subgoal_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "candidate_input_cids": [_link(dict(bundle))],
        "task_template_cids": [template_cid],
        "evaluator_cid": _link({"evaluator": "objective-bundle-priority-v1"}),
        "score_millionths": 1_000_000,
        "explanation_cid": _link({"reason": "selected generated objective bundle"}),
    }
    plan_cid = profile_g_cid(plan)
    selection = {
        "schema": "mcp++/profile-g/plan-selection@1",
        "created_at_ms": now,
        "parents": [plan_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "plan_branch_cid": plan_cid,
        "selector_did": owner_did,
        "proof_cid": str(bundle.get("proof_cid") or _link({"proof": bundle_key})),
        "policy_decision_cid": str(bundle.get("policy_decision_cid") or _link({"decision": "allow"})),
        "reason_cid": _link({"reason": "bundle emitted by accepted objective graph"}),
    }
    selection_cid = profile_g_cid(selection)
    # Subgoal.selection_cid remains null because making both immutable objects
    # point at one another would create an impossible content-addressed cycle.
    # PlanSelection is the authoritative selected-branch record.
    dependency_task_cids, dependency_provenance = _dependency_task_cids(bundle)
    dependency_repairs, dependency_repair_count = _dependency_repair_evidence(bundle)
    task = {
        "schema": "mcp++/profile-g/task@1",
        "created_at_ms": now,
        "parents": [selection_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "plan_branch_cid": plan_cid,
        "selection_cid": selection_cid,
        "interface_cid": _link({"interface": "codex.todo_bundle@1"}),
        "input_cid": _link(dict(bundle)),
        "tool": "codex.todo_bundle",
        "dependency_task_cids": dependency_task_cids,
        "idempotency_key": canonical_identity.semantic_fingerprint[:32],
        "resource_class": str(bundle.get("resource_class") or "cpu-small"),
        "deadline_ms": int(bundle.get("deadline_ms") or now + 86_400_000),
        "expected_value_millionths": int(bundle.get("expected_value_millionths") or 500_000),
        "max_attempts": _profile_g_task_spec_attempt_limit(
            bundle.get("max_attempts"),
        ),
        "execution_mode": "idempotent",
    }
    task_spec_cid = profile_g_cid(task)
    artifacts = {profile_g_cid(item): item for item in (goal, subgoal, plan, selection, task)}
    return {
        "goal": goal,
        "goal_cid": goal_cid,
        "subgoal": subgoal,
        "subgoal_cid": subgoal_cid,
        "plan_branch": plan,
        "plan_branch_cid": plan_cid,
        "selection": selection,
        "selection_cid": selection_cid,
        "task": task,
        "task_cid": task_spec_cid,
        "task_spec_cid": task_spec_cid,
        "canonical_task_key": canonical_identity.canonical_task_key,
        "canonical_task_cid": canonical_identity.canonical_task_cid,
        "dependency_provenance": dependency_provenance,
        "dependency_repair_evidence": dependency_repairs,
        "dependency_repair_evidence_count": dependency_repair_count,
        "artifacts": artifacts,
    }


def _validated_embedded_profile_g(
    bundle: Mapping[str, Any],
    embedded: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate immutable Profile-G bindings before registration trusts them."""

    from ipfs_datasets_py.logic.profile_g import (
        ProfileGError,
        validate_profile_g_artifact,
    )

    outer_limit = profile_g_task_attempt_limit(
        bundle.get("max_attempts"),
        default=3,
    )
    expected_task_limit = _profile_g_task_spec_attempt_limit(outer_limit)
    adapted = dict(embedded)
    artifacts = adapted.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("embedded Profile-G artifacts must be a mapping")

    fields = (
        ("goal", "goal_cid", "mcp++/profile-g/goal@1", "Goal"),
        ("subgoal", "subgoal_cid", "mcp++/profile-g/subgoal@1", "Subgoal"),
        (
            "plan_branch",
            "plan_branch_cid",
            "mcp++/profile-g/plan-branch@1",
            "PlanBranch",
        ),
        (
            "selection",
            "selection_cid",
            "mcp++/profile-g/plan-selection@1",
            "PlanSelection",
        ),
        ("task", "task_spec_cid", "mcp++/profile-g/task@1", "TaskSpec"),
    )
    validated_cids: dict[str, str] = {}
    for name, cid_field, schema, artifact_kind in fields:
        artifact = adapted.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"embedded Profile-G {name} artifact is required")
        if str(artifact.get("schema") or "") != schema:
            raise ValueError(f"embedded Profile-G {name} schema is invalid")
        try:
            actual_cid = validate_profile_g_artifact(
                artifact_kind,
                artifact,
            )
        except ProfileGError as exc:
            raise ValueError(
                f"embedded Profile-G {name} artifact is invalid: {exc}"
            ) from exc
        declared_cid = str(adapted.get(cid_field) or "")
        if name == "task":
            task_cid = str(adapted.get("task_cid") or "")
            if declared_cid != actual_cid or task_cid != actual_cid:
                raise ValueError("embedded Profile-G TaskSpec CID is inconsistent")
        elif declared_cid != actual_cid:
            raise ValueError(f"embedded Profile-G {name} CID is inconsistent")
        stored_artifact = artifacts.get(actual_cid)
        if not isinstance(stored_artifact, Mapping) or dict(stored_artifact) != dict(artifact):
            raise ValueError(f"embedded Profile-G {name} artifact binding is inconsistent")
        validated_cids[name] = actual_cid

    for cid, artifact in artifacts.items():
        if (
            not isinstance(artifact, Mapping)
            or str(cid) != profile_g_cid(dict(artifact))
        ):
            raise ValueError("embedded Profile-G artifact map contains an invalid CID binding")

    task = adapted["task"]
    if "max_attempts" not in task:
        raise ValueError("embedded Profile-G TaskSpec max_attempts is required")
    task_limit = profile_g_task_attempt_limit(task["max_attempts"])
    if task_limit != expected_task_limit:
        raise ValueError(
            "bundle max_attempts does not match embedded Profile-G TaskSpec"
        )

    expected_links = (
        (adapted["subgoal"], "goal_cid", validated_cids["goal"]),
        (adapted["plan_branch"], "subgoal_cid", validated_cids["subgoal"]),
        (adapted["selection"], "subgoal_cid", validated_cids["subgoal"]),
        (adapted["selection"], "plan_branch_cid", validated_cids["plan_branch"]),
        (task, "subgoal_cid", validated_cids["subgoal"]),
        (task, "plan_branch_cid", validated_cids["plan_branch"]),
        (task, "selection_cid", validated_cids["selection"]),
    )
    for artifact, field, expected in expected_links:
        if str(artifact.get(field) or "") != expected:
            raise ValueError(f"embedded Profile-G {field} link is inconsistent")

    canonical_identity = canonical_bundle_identity(bundle)
    embedded_key = str(adapted.get("canonical_task_key") or "")
    embedded_cid = str(adapted.get("canonical_task_cid") or "")
    if embedded_key != canonical_identity.canonical_task_key:
        raise ValueError("embedded canonical task key does not match bundle")
    if embedded_cid != canonical_identity.canonical_task_cid:
        raise ValueError("embedded canonical task CID does not match bundle")

    expected_objective_cid = _link(
        {
            "bundle_key": str(bundle.get("bundle_key") or "objective/general"),
            "source_todo": str(bundle.get("source_todo") or ""),
            "task_ids": sorted(
                str(item.get("task_id"))
                for item in bundle.get("tasks", [])
                if isinstance(item, Mapping) and item.get("task_id")
            ),
        }
    )
    if (
        str(adapted["goal"].get("objective_cid") or "") != expected_objective_cid
        or str(adapted["subgoal"].get("objective_cid") or "")
        != expected_objective_cid
    ):
        raise ValueError("embedded Profile-G objective does not match bundle")
    if str(task.get("idempotency_key") or "") != canonical_identity.semantic_fingerprint[:32]:
        raise ValueError("embedded Profile-G TaskSpec idempotency key does not match bundle")
    candidate_inputs = adapted["plan_branch"].get("candidate_input_cids")
    if (
        not isinstance(candidate_inputs, list)
        or candidate_inputs != [task.get("input_cid")]
    ):
        raise ValueError("embedded Profile-G input binding is inconsistent")
    return adapted


@dataclass(frozen=True)
class LeaseGrant:
    task_cid: str
    goal_cid: str
    subgoal_cid: str
    claim_cid: str
    resolution_cid: str
    claimant_did: str
    logical_epoch: int
    fencing_token: int
    lease_expires_at_ms: int
    attempt: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributedLaneDispatch:
    """A remote execution assignment bound to one accepted fencing epoch."""

    grant: LeaseGrant
    input_artifact_cid: str
    capability_receipt_cid: str
    environment_receipt_cid: str
    dispatch_cid: str
    worker_id: str
    repository_id: str
    required_capabilities: tuple[str, ...] = ()
    lease_duration_ms: int = 60_000
    cancellation_cid: str | None = None
    schema: str = DISTRIBUTED_LANE_DISPATCH_SCHEMA

    @property
    def task_cid(self) -> str:
        return self.grant.task_cid

    @property
    def logical_epoch(self) -> int:
        return self.grant.logical_epoch

    @property
    def fencing_epoch(self) -> int:
        return self.grant.logical_epoch

    @property
    def fencing_token(self) -> int:
        return self.grant.fencing_token

    @property
    def lease_expires_at_ms(self) -> int:
        return self.grant.lease_expires_at_ms

    @property
    def cancelled(self) -> bool:
        return self.cancellation_cid is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dispatch_cid": self.dispatch_cid,
            "repository_id": self.repository_id,
            "worker_id": self.worker_id,
            "task_cid": self.grant.task_cid,
            "input_artifact_cid": self.input_artifact_cid,
            "artifact_id": self.input_artifact_cid,
            "capability_receipt_cid": self.capability_receipt_cid,
            "capability_receipt_id": self.capability_receipt_cid,
            "environment_receipt_cid": self.environment_receipt_cid,
            "environment_receipt_id": self.environment_receipt_cid,
            "claim_cid": self.grant.claim_cid,
            "logical_epoch": self.grant.logical_epoch,
            "fencing_epoch": self.grant.logical_epoch,
            "fencing_token": self.grant.fencing_token,
            "lease_expires_at_ms": self.grant.lease_expires_at_ms,
            "lease_duration_ms": self.lease_duration_ms,
            "required_capabilities": list(self.required_capabilities),
            "cancellation_cid": self.cancellation_cid,
            "cancelled": self.cancelled,
            "grant": self.grant.to_dict(),
        }


@dataclass(frozen=True)
class RemoteLaneResult:
    """Canonical remote result envelope presented at the lease boundary."""

    repository_id: str
    worker_id: str
    task_cid: str
    artifact_id: str
    candidate_commit: str
    capability_receipt_id: str
    environment_receipt_id: str
    claim_cid: str
    logical_epoch: int
    fencing_token: int
    output: Mapping[str, Any]
    created_at_ms: int
    status: str = "succeeded"
    failure_class: str = "none"
    cancelled: bool = False
    request_id: str = ""
    publication_id: str = ""
    schema: str = REMOTE_LANE_RESULT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema != REMOTE_LANE_RESULT_SCHEMA:
            raise ValueError("unsupported remote lane result schema")
        text = {
            "repository_id": _required_text(self.repository_id, "repository_id"),
            "worker_id": _required_text(self.worker_id, "worker_id"),
            "task_cid": _required_text(self.task_cid, "task_cid"),
            "artifact_id": _required_text(self.artifact_id, "artifact_id"),
            "capability_receipt_id": _required_text(
                self.capability_receipt_id, "capability_receipt_id"
            ),
            "environment_receipt_id": _required_text(
                self.environment_receipt_id, "environment_receipt_id"
            ),
            "claim_cid": _required_text(self.claim_cid, "claim_cid"),
        }
        candidate_commit = str(self.candidate_commit or "").strip()
        status = str(self.status or "").strip().lower()
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("remote result status must be succeeded, failed, or cancelled")
        cancelled = bool(self.cancelled or status == "cancelled")
        if status == "succeeded" and not cancelled and not candidate_commit:
            raise ValueError("successful remote result requires candidate_commit")
        epoch = _timestamp(self.logical_epoch, "logical_epoch")
        token = _timestamp(self.fencing_token, "fencing_token")
        if epoch < 1 or token < 1:
            raise ValueError("remote result fencing values must be positive")
        created = _timestamp(self.created_at_ms, "created_at_ms")
        output = _canonical_mapping(self.output, "output")
        failure_class = str(self.failure_class or "none")[:128]
        request_id = str(self.request_id or "").strip()
        body = {
            "schema": self.schema,
            **text,
            "candidate_commit": candidate_commit,
            "logical_epoch": epoch,
            "fencing_token": token,
            "output": output,
            "created_at_ms": created,
            "status": "cancelled" if cancelled else status,
            "failure_class": failure_class,
            "cancelled": cancelled,
        }
        if not request_id:
            request_id = profile_g_cid({"remote_result_request": body})
        body["request_id"] = request_id
        publication_id = profile_g_cid(body)
        digest = _content_digest(body)
        if self.publication_id and self.publication_id != publication_id:
            raise ValueError("remote publication_id does not match content")
        if self.digest and self.digest != digest:
            raise ValueError("remote result digest does not match content")
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "candidate_commit", candidate_commit)
        object.__setattr__(self, "logical_epoch", epoch)
        object.__setattr__(self, "fencing_token", token)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "status", "cancelled" if cancelled else status)
        object.__setattr__(self, "failure_class", failure_class)
        object.__setattr__(self, "cancelled", cancelled)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "publication_id", publication_id)
        object.__setattr__(self, "digest", digest)

    @property
    def fencing_epoch(self) -> int:
        return self.logical_epoch

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repository_id": self.repository_id,
            "request_id": self.request_id,
            "publication_id": self.publication_id,
            "worker_id": self.worker_id,
            "task_cid": self.task_cid,
            "artifact_id": self.artifact_id,
            "candidate_commit": self.candidate_commit,
            "capability_receipt_id": self.capability_receipt_id,
            "environment_receipt_id": self.environment_receipt_id,
            "claim_cid": self.claim_cid,
            "logical_epoch": self.logical_epoch,
            "fencing_epoch": self.logical_epoch,
            "fencing_token": self.fencing_token,
            "output": dict(self.output),
            "created_at_ms": self.created_at_ms,
            "status": self.status,
            "failure_class": self.failure_class,
            "cancelled": self.cancelled,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteLaneResult:
        return cls(
            repository_id=value.get("repository_id", ""),
            worker_id=value.get("worker_id", ""),
            task_cid=value.get("task_cid", ""),
            artifact_id=value.get("artifact_id", value.get("input_artifact_cid", "")),
            candidate_commit=value.get("candidate_commit", ""),
            capability_receipt_id=value.get(
                "capability_receipt_id", value.get("capability_receipt_cid", "")
            ),
            environment_receipt_id=value.get(
                "environment_receipt_id", value.get("environment_receipt_cid", "")
            ),
            claim_cid=value.get("claim_cid", ""),
            logical_epoch=value.get("logical_epoch", value.get("fencing_epoch", 0)),
            fencing_token=value.get("fencing_token", 0),
            output=value.get("output", {}),
            created_at_ms=value.get("created_at_ms", 0),
            status=value.get("status", "succeeded"),
            failure_class=value.get("failure_class", "none"),
            cancelled=value.get("cancelled", False),
            request_id=value.get("request_id", ""),
            publication_id=value.get("publication_id", ""),
            schema=value.get("schema", REMOTE_LANE_RESULT_SCHEMA),
            digest=value.get("digest", ""),
        )


@dataclass(frozen=True)
class TaskLeaseState:
    """Authoritative scheduler projection for one registered task.

    ``state`` is deliberately scheduler-oriented: expired and voluntarily
    released lease records project as ``ready``.  The durable lease outcome is
    retained in ``lease_state`` and ``release_reason`` for diagnostics.
    """

    task_cid: str
    goal_cid: str
    subgoal_cid: str
    task_id: str
    bundle: dict[str, Any]
    state: str
    lease_state: str | None
    claim_cid: str | None
    resolution_cid: str | None
    claimant_did: str | None
    logical_epoch: int
    fencing_token: int
    lease_expires_at_ms: int | None
    attempt: int
    max_attempts: int
    release_reason: str | None
    retry_not_before_ms: int
    registered_at_ms: int
    updated_at_ms: int

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def to_dict(self) -> dict[str, Any]:
        payload = {definition.name: getattr(self, definition.name) for definition in fields(self)}
        payload["bundle"] = dict(self.bundle)
        return payload


class LeaseCoordinator:
    """Durable accepted-lease registry for independent daemon processes."""

    def __init__(self, path: str | Path, *, clock_ms: Callable[[], int] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            with self.path.open("rb") as stream:
                if stream.read(16) == b"SQLite format 3\0":
                    raise ValueError(
                        f"legacy SQLite coordination store requires migration: {self.path}"
                    )
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.RLock()
        self._operation_state = threading.local()
        self._connection: _DuckConnection | None = None
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        with self._database_operation():
            self._init_schema()

    @contextmanager
    def _database_operation(self) -> Iterator[None]:
        """Serialize short-lived DuckDB connections across lane processes."""

        with self._lock:
            depth = int(getattr(self._operation_state, "depth", 0))
            if depth:
                self._operation_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._operation_state.depth = depth
                return

            with _exclusive_file_lock(
                self._lock_path,
                timeout_seconds=COORDINATION_LOCK_TIMEOUT_SECONDS,
            ):
                try:
                    import duckdb
                except ImportError as exc:
                    raise RuntimeError(
                        "DuckDB is required for lease coordination"
                    ) from exc
                duckdb_connection = _connect_coordination_duckdb(duckdb, self.path)
                self._connection = _DuckConnection.wrap(
                    duckdb_connection,
                    transaction_on_context=True,
                )
                self._operation_state.depth = 1
                try:
                    yield
                finally:
                    self._operation_state.depth = 0
                    connection = self._connection
                    self._connection = None
                    if connection is not None:
                        connection.close()

    def _init_schema(self) -> None:
        assert self._connection is not None
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                  cid TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
                  created_at_ms BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                  task_cid TEXT PRIMARY KEY, goal_cid TEXT NOT NULL, subgoal_cid TEXT NOT NULL,
                  task_id TEXT NOT NULL, bundle_json TEXT NOT NULL,
                  registered_at_ms BIGINT NOT NULL DEFAULT 0,
                  updated_at_ms BIGINT NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS task_aliases (
                  alias_task_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_dependencies (
                  task_cid TEXT NOT NULL, dependency_task_cid TEXT NOT NULL,
                  provenance_json TEXT NOT NULL,
                  PRIMARY KEY(task_cid, dependency_task_cid)
                );
                CREATE TABLE IF NOT EXISTS task_dependency_repairs (
                  task_cid TEXT NOT NULL, repair_index BIGINT NOT NULL,
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY(task_cid, repair_index)
                );
                CREATE TABLE IF NOT EXISTS task_dependency_repair_state (
                  task_cid TEXT PRIMARY KEY, source_count BIGINT NOT NULL,
                  stored_count BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                  task_cid TEXT PRIMARY KEY, claim_cid TEXT NOT NULL, resolution_cid TEXT NOT NULL,
                  claimant_did TEXT NOT NULL, logical_epoch BIGINT NOT NULL,
                  fencing_token BIGINT NOT NULL, expires_at_ms BIGINT NOT NULL,
                  attempt BIGINT NOT NULL, state TEXT NOT NULL, started_at_ms BIGINT NOT NULL,
                  release_reason TEXT,
                  retry_not_before_ms BIGINT NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS token_history (
                  task_cid TEXT NOT NULL, fencing_token BIGINT NOT NULL,
                  PRIMARY KEY(task_cid, fencing_token)
                );
                CREATE TABLE IF NOT EXISTS heartbeats (
                  heartbeat_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL, claimant_did TEXT NOT NULL,
                  fencing_token BIGINT NOT NULL, observed_at_ms BIGINT NOT NULL,
                  expires_at_ms BIGINT NOT NULL, capacity_millionths BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                  receipt_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL, goal_cid TEXT NOT NULL,
                  subgoal_cid TEXT NOT NULL, claim_cid TEXT NOT NULL, fencing_token BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS distributed_inputs (
                  artifact_id TEXT PRIMARY KEY, task_cid TEXT NOT NULL,
                  repository_id TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worker_capability_receipts (
                  receipt_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL,
                  expires_at_ms BIGINT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worker_environment_receipts (
                  receipt_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL,
                  capability_receipt_id TEXT NOT NULL, expires_at_ms BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS distributed_dispatches (
                  dispatch_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL,
                  claim_cid TEXT NOT NULL, worker_id TEXT NOT NULL,
                  fencing_token BIGINT NOT NULL, cancellation_cid TEXT,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS distributed_publications (
                  publication_id TEXT PRIMARY KEY, dispatch_cid TEXT NOT NULL,
                  task_cid TEXT NOT NULL, disposition TEXT NOT NULL,
                  reason TEXT NOT NULL, created_at_ms BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coordination_metadata (
                  metadata_key TEXT PRIMARY KEY, value_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_dependencies_dependency_idx
                  ON task_dependencies(dependency_task_cid);
                CREATE INDEX IF NOT EXISTS receipts_task_order_idx
                  ON receipts(task_cid, receipt_cid);
                CREATE INDEX IF NOT EXISTS distributed_publications_task_idx
                  ON distributed_publications(task_cid, created_at_ms);
                """
            )
            # REF-036 databases remain valid after migration. Keep schema
            # evolution additive so existing DuckDB stores upgrade in place.
            task_columns = {
                str(row["name"]) for row in self._connection.execute("PRAGMA table_info(tasks)")
            }
            if "registered_at_ms" not in task_columns:
                _add_coordination_not_null_default_column(
                    self._connection,
                    table="tasks",
                    column="registered_at_ms",
                )
            if "updated_at_ms" not in task_columns:
                _add_coordination_not_null_default_column(
                    self._connection,
                    table="tasks",
                    column="updated_at_ms",
                )
            lease_columns = {
                str(row["name"]) for row in self._connection.execute("PRAGMA table_info(leases)")
            }
            if "release_reason" not in lease_columns:
                self._connection.execute("ALTER TABLE leases ADD COLUMN release_reason TEXT")
            if "retry_not_before_ms" not in lease_columns:
                _add_coordination_not_null_default_column(
                    self._connection,
                    table="leases",
                    column="retry_not_before_ms",
                )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_leases_scheduler_state "
                "ON leases(state, expires_at_ms, retry_not_before_ms)"
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO coordination_metadata VALUES(?,?)",
                (
                    "store",
                    json.dumps(
                        {
                            "schema": COORDINATION_STORE_SCHEMA,
                            "backend": "duckdb",
                        },
                        sort_keys=True,
                    ),
                ),
            )

    def close(self) -> None:
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None

    def compact(self) -> dict[str, Any]:
        """Atomically rewrite live rows into a compact DuckDB store."""

        with self._lock:
            if int(getattr(self._operation_state, "depth", 0)):
                raise RuntimeError(
                    "coordination compaction cannot run inside a database operation"
                )
            with _exclusive_file_lock(
                self._lock_path,
                timeout_seconds=COORDINATION_LOCK_TIMEOUT_SECONDS,
            ):
                source_bytes = self.path.stat().st_size
                temporary = self.path.with_name(
                    f".{self.path.name}.compact-{os.getpid()}-"
                    f"{threading.get_ident()}.tmp"
                )
                temporary_lock = temporary.with_name(f".{temporary.name}.lock")
                temporary.unlink(missing_ok=True)
                Path(f"{temporary}.wal").unlink(missing_ok=True)
                temporary_lock.unlink(missing_ok=True)
                try:
                    try:
                        import duckdb
                    except ImportError as exc:
                        raise RuntimeError(
                            "DuckDB is required for lease coordination"
                        ) from exc

                    # Build the target with the canonical additive schema.  A
                    # separate local connection avoids multi-database SQL and
                    # keeps both file identities explicit throughout copying.
                    with LeaseCoordinator(
                        temporary,
                        clock_ms=self._clock_ms,
                    ):
                        pass
                    source_connection = _connect_coordination_duckdb(
                        duckdb,
                        self.path,
                        read_only=True,
                    )
                    try:
                        target_connection = _connect_coordination_duckdb(
                            duckdb,
                            temporary,
                        )
                        try:
                            target_connection.execute("BEGIN TRANSACTION")
                            _copy_coordination_store_rows(
                                source_connection,
                                target_connection,
                            )
                            target_connection.execute(
                                "INSERT OR REPLACE INTO coordination_metadata "
                                "VALUES(?,?)",
                                (
                                    "last_compaction",
                                    json.dumps(
                                        {
                                            "compacted_at_ms": self._clock_ms(),
                                            "source_bytes": source_bytes,
                                        },
                                        sort_keys=True,
                                    ),
                                ),
                            )
                            target_connection.execute("COMMIT")
                            target_connection.execute("CHECKPOINT")
                        finally:
                            target_connection.close()
                    finally:
                        source_connection.close()
                    target_bytes = temporary.stat().st_size
                    source_mode = self.path.stat().st_mode & 0o777
                    os.chmod(temporary, source_mode)
                    _fsync_path(temporary)
                    os.replace(temporary, self.path)
                    _fsync_path(self.path.parent, directory=True)
                finally:
                    temporary.unlink(missing_ok=True)
                    Path(f"{temporary}.wal").unlink(missing_ok=True)
                    temporary_lock.unlink(missing_ok=True)
                return {
                    "source_bytes": source_bytes,
                    "target_bytes": target_bytes,
                    "reclaimed_bytes": max(0, source_bytes - target_bytes),
                    "compacted_at_ms": self._clock_ms(),
                }

    def __enter__(self) -> LeaseCoordinator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _put_artifact(self, connection: _DuckConnection, kind: str, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        cid = profile_g_cid(body)
        connection.execute(
            "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
            (cid, kind, canonical_profile_g_bytes(body).decode("utf-8"), int(body.get("created_at_ms") or self._clock_ms())),
        )
        return cid

    @_coordinator_operation
    def register_bundle(self, bundle: Mapping[str, Any], *, created_at_ms: int | None = None) -> dict[str, Any]:
        embedded = bundle.get("profile_g")
        adapted = dict(embedded) if isinstance(embedded, Mapping) and embedded.get("task_cid") else adapt_goal_bundle(bundle, created_at_ms=created_at_ms)
        canonical_identity = canonical_bundle_identity(bundle)
        canonical_task_cid = str(adapted.get("canonical_task_cid") or canonical_identity.canonical_task_cid)
        task_spec_cid = str(adapted.get("task_spec_cid") or adapted.get("task_cid") or "")
        adapted["canonical_task_key"] = str(
            adapted.get("canonical_task_key") or canonical_identity.canonical_task_key
        )
        adapted["canonical_task_cid"] = canonical_task_cid
        adapted["task_spec_cid"] = task_spec_cid
        dependency_task_cids, dependency_provenance = _dependency_task_cids(bundle)
        adapted_task = adapted.get("task")
        if "dependency_task_cids" not in bundle and isinstance(adapted_task, Mapping):
            embedded_cids, _ = _dependency_task_cids(
                {"dependency_task_cids": adapted_task.get("dependency_task_cids")}
            )
            for cid in embedded_cids:
                if cid not in dependency_task_cids:
                    dependency_task_cids.append(cid)
                    dependency_provenance.setdefault(cid, []).append(
                        {"source": "profile_g.task.dependency_task_cids"}
                    )
        dependency_task_cids.sort()
        adapted["dependency_task_cids"] = dependency_task_cids
        adapted["dependency_provenance"] = dependency_provenance
        dependency_repairs, dependency_repair_count = _dependency_repair_evidence(bundle)
        adapted["dependency_repair_evidence"] = dependency_repairs
        adapted["dependency_repair_evidence_count"] = dependency_repair_count
        # Coordination is keyed by semantic execution identity. The immutable
        # Profile-G TaskSpec CID remains available separately for provenance.
        adapted["task_cid"] = canonical_task_cid
        registered_at = self._clock_ms() if created_at_ms is None else int(created_at_ms)
        attempt_budget_reset = False
        task_id = ",".join(
            str(item.get("task_id"))
            for item in bundle.get("tasks", [])
            if isinstance(item, Mapping)
        ) or str(bundle.get("bundle_key") or adapted["task_cid"])
        bundle_json = json.dumps(dict(bundle), sort_keys=True)
        all_member_aliases = {
            str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
            for item in (
                bundle.get("tasks", [])
                if isinstance(bundle.get("tasks"), (list, tuple))
                else []
            )
            if isinstance(item, Mapping)
            and str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
        }
        execution_member_aliases = {
            str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
            for item in _bundle_execution_tasks(bundle)
            if str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
        }
        aliases = {canonical_task_cid, task_spec_cid, *execution_member_aliases}

        def sync_aliases() -> None:
            # Repair databases written before slice-scoped identities without
            # deleting an alias now owned by a different, completed slice.
            for alias in sorted(all_member_aliases - execution_member_aliases):
                self._connection.execute(
                    "DELETE FROM task_aliases WHERE alias_task_cid=? AND task_cid=?",
                    (alias, canonical_task_cid),
                )
            for alias in sorted(aliases):
                if alias:
                    self._connection.execute(
                        "INSERT OR REPLACE INTO task_aliases VALUES(?,?)",
                        (alias, canonical_task_cid),
                    )

        with self._lock, self._connection:
            previous_task = self._connection.execute(
                "SELECT bundle_json FROM tasks WHERE task_cid=?",
                (canonical_task_cid,),
            ).fetchone()
            registration_complete = self._connection.execute(
                "SELECT 1 FROM task_dependency_repair_state WHERE task_cid=?",
                (canonical_task_cid,),
            ).fetchone()
            if (
                previous_task is not None
                and registration_complete is not None
                and str(previous_task["bundle_json"]) == bundle_json
            ):
                sync_aliases()
                adapted["attempt_budget_reset"] = False
                return adapted

            for cid, artifact in adapted["artifacts"].items():
                self._connection.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                    (cid, str(artifact["schema"]), canonical_profile_g_bytes(artifact).decode("utf-8"), artifact["created_at_ms"]),
                )
            previous_bundle = (
                json.loads(str(previous_task["bundle_json"]))
                if previous_task is not None
                else {}
            )
            reopened = isinstance(previous_bundle, Mapping) and _reopens_blocked_bundle(
                previous_bundle,
                bundle,
            )
            # Keep the original immutable Goal/Subgoal provenance while
            # refreshing mutable discovery metadata on every scheduler poll.
            self._connection.execute(
                """INSERT INTO tasks(
                       task_cid, goal_cid, subgoal_cid, task_id, bundle_json,
                       registered_at_ms, updated_at_ms
                   ) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(task_cid) DO UPDATE SET
                     task_id=excluded.task_id,
                     bundle_json=excluded.bundle_json,
                     updated_at_ms=excluded.updated_at_ms""",
                (
                    canonical_task_cid,
                    adapted["goal_cid"],
                    adapted["subgoal_cid"],
                    task_id,
                    bundle_json,
                    registered_at,
                    registered_at,
                ),
            )
            if reopened:
                reset = self._connection.execute(
                    """UPDATE leases
                       SET state='released', attempt=0,
                           release_reason='requeued:bundle_status_reopened',
                           retry_not_before_ms=0
                       WHERE task_cid=?
                         AND state IN ('released','expired')
                         AND release_reason LIKE 'receipt:%:blocked'""",
                    (canonical_task_cid,),
                )
                attempt_budget_reset = reset.rowcount > 0
            # A dependency DAG names the immutable work-item CIDs while the
            # coordination lease is bundle-scoped.  Aliases bridge those two
            # identities so the slice's successful receipt unlocks only the
            # members that it executed. Earlier slice aliases must remain
            # mapped to their completed receipt authorities.
            sync_aliases()
            self._connection.execute("DELETE FROM task_dependencies WHERE task_cid=?", (canonical_task_cid,))
            for dependency_task_cid in dependency_task_cids:
                self._connection.execute(
                    "INSERT INTO task_dependencies VALUES(?,?,?)",
                    (
                        canonical_task_cid,
                        dependency_task_cid,
                        json.dumps(dependency_provenance.get(dependency_task_cid, []), sort_keys=True),
                    ),
                )
            self._connection.execute("DELETE FROM task_dependency_repairs WHERE task_cid=?", (canonical_task_cid,))
            for index, repair in enumerate(dependency_repairs):
                self._connection.execute(
                    "INSERT INTO task_dependency_repairs VALUES(?,?,?)",
                    (canonical_task_cid, index, json.dumps(repair, sort_keys=True)),
                )
            self._connection.execute(
                "INSERT OR REPLACE INTO task_dependency_repair_state VALUES(?,?,?)",
                (canonical_task_cid, dependency_repair_count, len(dependency_repairs)),
            )
        adapted["attempt_budget_reset"] = attempt_budget_reset
        return adapted

    @_coordinator_operation
    def register_bundles(
        self,
        bundles: Iterable[Mapping[str, Any]],
        *,
        created_at_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Register a discovery batch idempotently.

        This convenience API intentionally delegates one bundle at a time so
        embedded Profile G artifacts and canonical identity follow exactly the
        same compatibility path as :meth:`register_bundle`.
        """

        with self._lock, self._connection:
            return [
                self.register_bundle(bundle, created_at_ms=created_at_ms)
                for bundle in bundles
            ]

    @_coordinator_operation
    def requeue_exhausted_blocked(self, task_cid: str, *, reason: str) -> bool:
        """Reset a blocked attempt budget after authoritative work is reopened.

        This operation is deliberately narrow: it cannot disturb accepted or
        completed leases. It resets an exhausted released/expired lease when:

        * the last terminal receipt classified the work as blocked, or
        * the lease was abandoned mid-flight (``scheduler stopped`` / drain) or
          previously requeued while the authoritative board still has open
          work — otherwise a full attempt budget is burned by supervisor
          restarts and residual FVT lanes never relaunch.
        """

        normalized_reason = str(reason or "authoritative_source_reopened").strip().replace(" ", "_")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                resolved_task_cid = self._resolve_task_cid(connection, task_cid)
                if resolved_task_cid is None:
                    connection.commit()
                    return False
                row = connection.execute(
                    """SELECT t.bundle_json, l.state, l.attempt, l.release_reason
                       FROM tasks AS t
                       JOIN leases AS l ON l.task_cid=t.task_cid
                       WHERE t.task_cid=?""",
                    (resolved_task_cid,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return False
                exhausted = self._attempt_budget_exhausted(
                    row,
                    int(row["attempt"] or 0),
                )
                release_reason = str(row["release_reason"] or "")
                blocked_receipt = release_reason.startswith("receipt:") and release_reason.endswith(
                    ":blocked"
                )
                abandoned_or_requeued = (
                    release_reason == "scheduler stopped"
                    or release_reason.startswith("requeued:")
                    or release_reason.startswith("worker drained")
                    or "drained or exited" in release_reason
                )
                if (
                    row["state"] not in {"released", "expired"}
                    or not exhausted
                    or not (blocked_receipt or abandoned_or_requeued)
                ):
                    connection.commit()
                    return False
                connection.execute(
                    """UPDATE leases
                       SET state='released', attempt=0, retry_not_before_ms=0,
                           release_reason=?
                       WHERE task_cid=?""",
                    (f"requeued:{normalized_reason}"[:256], resolved_task_cid),
                )
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def requeue_completed(self, task_cid: str, *, reason: str) -> bool:
        """Reopen a completed lease when its authoritative taskboard has work.

        Successful receipts remain immutable audit records, but no longer
        satisfy dependency gates while the lease is reopened.
        """

        normalized_reason = str(reason or "authoritative_source_reopened").strip().replace(" ", "_")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                resolved_task_cid = self._resolve_task_cid(connection, task_cid)
                if resolved_task_cid is None:
                    connection.commit()
                    return False
                updated = connection.execute(
                    """UPDATE leases
                       SET state='released', attempt=0, retry_not_before_ms=0,
                           release_reason=?
                       WHERE task_cid=? AND state='completed'""",
                    (f"requeued:{normalized_reason}"[:256], resolved_task_cid),
                )
                connection.commit()
                return updated.rowcount > 0
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _resolve_task_cid(connection: _DuckConnection, task_cid: str) -> str | None:
        row = connection.execute(
            "SELECT task_cid FROM task_aliases WHERE alias_task_cid=?",
            (task_cid,),
        ).fetchone()
        if row is not None:
            return str(row[0])
        row = connection.execute("SELECT task_cid FROM tasks WHERE task_cid=?", (task_cid,)).fetchone()
        return str(row[0]) if row is not None else None

    def _dependency_cycles(
        self,
        connection: _DuckConnection,
        task_cid: str,
        *,
        max_nodes: int,
        max_cycles: int,
    ) -> tuple[list[list[str]], bool]:
        """Find a bounded set of dependency cycles reachable from ``task_cid``."""

        cycles: list[list[str]] = []
        visited: set[str] = set()
        active: list[str] = []
        active_set: set[str] = set()
        truncated = False

        def visit(current: str) -> None:
            nonlocal truncated
            if truncated or len(cycles) >= max_cycles:
                truncated = True
                return
            if len(visited) >= max_nodes and current not in visited:
                truncated = True
                return
            if current in active_set:
                start = active.index(current)
                cycle = active[start:] + [current]
                if cycle not in cycles:
                    cycles.append(cycle)
                return
            if current in visited:
                return
            visited.add(current)
            active.append(current)
            active_set.add(current)
            rows = connection.execute(
                "SELECT dependency_task_cid FROM task_dependencies WHERE task_cid=? ORDER BY dependency_task_cid",
                (current,),
            ).fetchall()
            for row in rows:
                resolved = self._resolve_task_cid(connection, str(row[0]))
                if resolved is not None:
                    visit(resolved)
            active.pop()
            active_set.remove(current)

        visit(task_cid)
        return cycles, truncated

    def _claimability(
        self,
        connection: _DuckConnection,
        task_cid: str,
        *,
        max_evidence: int,
    ) -> dict[str, Any]:
        resolved_task_cid = self._resolve_task_cid(connection, task_cid)
        if resolved_task_cid is None:
            raise KeyError(f"unknown task CID: {task_cid}")
        rows = connection.execute(
            "SELECT dependency_task_cid, provenance_json FROM task_dependencies "
            "WHERE task_cid=? ORDER BY dependency_task_cid",
            (resolved_task_cid,),
        ).fetchall()
        dependencies = [str(row["dependency_task_cid"]) for row in rows]
        provenance = {
            str(row["dependency_task_cid"]): json.loads(row["provenance_json"])
            for row in rows
        }
        satisfied: list[str] = []
        blocked: list[str] = []
        missing: list[str] = []
        evidence: list[dict[str, Any]] = []
        evidence_truncated = False

        def add_evidence(item: dict[str, Any]) -> None:
            nonlocal evidence_truncated
            if len(evidence) < max_evidence:
                evidence.append(item)
            else:
                evidence_truncated = True

        for dependency_cid in dependencies:
            receipt_task_cid = self._resolve_task_cid(connection, dependency_cid)
            if receipt_task_cid is None:
                missing.append(dependency_cid)
                blocked.append(dependency_cid)
                add_evidence(
                    {
                        "kind": "missing_dependency",
                        "dependency_task_cid": dependency_cid,
                        "provenance": provenance.get(dependency_cid, []),
                        "repair": "register the prerequisite task or repair the dependency edge",
                    }
                )
                continue
            receipt = connection.execute(
                "SELECT receipt_cid, payload_json FROM receipts WHERE task_cid=? ORDER BY rowid DESC LIMIT 1",
                (receipt_task_cid,),
            ).fetchone()
            lease = connection.execute(
                "SELECT state FROM leases WHERE task_cid=?",
                (receipt_task_cid,),
            ).fetchone()
            receipt_payload = json.loads(receipt["payload_json"]) if receipt is not None else {}
            latest_status = str(receipt_payload.get("status") or "missing")
            lease_state = str(lease["state"] or "") if lease is not None else "missing"
            if latest_status == "succeeded" and lease_state == "completed":
                satisfied.append(dependency_cid)
                continue
            blocked.append(dependency_cid)
            add_evidence(
                {
                    "kind": "prerequisite_receipt_not_succeeded",
                    "dependency_task_cid": dependency_cid,
                    "resolved_task_cid": receipt_task_cid,
                    "latest_receipt_cid": str(receipt["receipt_cid"]) if receipt is not None else None,
                    "latest_status": latest_status,
                    "lease_state": lease_state,
                    "provenance": provenance.get(dependency_cid, []),
                    "repair": "complete and merge the prerequisite successfully before claiming this task",
                }
            )

        cycles, cycle_search_truncated = self._dependency_cycles(
            connection,
            resolved_task_cid,
            max_nodes=max(64, max_evidence * 8),
            max_cycles=max_evidence,
        )
        for cycle in cycles:
            add_evidence(
                {
                    "kind": "dependency_cycle",
                    "cycle_task_cids": cycle,
                    "repair": "remove or redirect at least one cyclic prerequisite edge",
                }
            )
        planner_repairs: list[dict[str, Any]] = []
        repair_rows = connection.execute(
            "SELECT payload_json FROM task_dependency_repairs WHERE task_cid=? ORDER BY repair_index",
            (resolved_task_cid,),
        ).fetchall()
        for row in repair_rows:
            repair = json.loads(row["payload_json"])
            kind = str(repair.get("kind") or "").strip().lower().replace("-", "_")
            if kind not in STRUCTURAL_DEPENDENCY_REPAIR_KINDS:
                continue
            repair["kind"] = kind
            repair.setdefault(
                "repair",
                "repair the planner dependency metadata and regenerate the bundle schedule",
            )
            planner_repairs.append(repair)
            add_evidence(repair)
        repair_state = connection.execute(
            "SELECT source_count, stored_count FROM task_dependency_repair_state WHERE task_cid=?",
            (resolved_task_cid,),
        ).fetchone()
        persisted_repairs_truncated = bool(
            repair_state is not None and int(repair_state["source_count"]) > int(repair_state["stored_count"])
        )
        return {
            "schema": "ipfs_accelerate_py/dependency-claimability@1",
            "task_cid": resolved_task_cid,
            "claimable": not blocked and not cycles and not planner_repairs,
            "dependency_task_cids": dependencies,
            "satisfied_dependency_task_cids": satisfied,
            "blocked_dependency_task_cids": blocked,
            "missing_dependency_task_cids": missing,
            "dependency_cycles": cycles,
            "structural_dependency_repairs": planner_repairs[:max_evidence],
            "repair_evidence": evidence,
            "evidence_truncated": evidence_truncated or cycle_search_truncated or persisted_repairs_truncated,
            "planner_repair_evidence_count": int(repair_state["source_count"]) if repair_state is not None else 0,
        }

    @_coordinator_operation
    def claimability(self, task_cid: str, *, max_evidence: int = 32) -> dict[str, Any]:
        """Explain whether all prerequisite tasks have successful receipts.

        Evidence is deliberately bounded so malformed or cyclic planner input
        becomes actionable repair data instead of an unbounded walk or deadlock.
        """

        limit = int(max_evidence)
        if not 1 <= limit <= 256:
            raise ValueError("max_evidence must be in [1, 256]")
        with self._lock:
            return self._claimability(self._connection, task_cid, max_evidence=limit)

    def _expire(self, connection: _DuckConnection, task_cid: str, now: int) -> None:
        row = connection.execute(
            "SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms<=?",
            (task_cid, now),
        ).fetchone()
        if row is None:
            return
        resolution = self._resolution_payload(row, outcome="expired", now=now)
        resolution_cid = self._put_artifact(connection, "ClaimResolution", resolution)
        connection.execute(
            """UPDATE leases
               SET state='expired', resolution_cid=?, release_reason='expired'
               WHERE task_cid=? AND claim_cid=? AND fencing_token=?""",
            (resolution_cid, task_cid, row["claim_cid"], row["fencing_token"]),
        )

    @staticmethod
    def _max_attempts(task: _DuckRow | Mapping[str, Any]) -> int:
        bundle = json.loads(task["bundle_json"])
        return profile_g_task_attempt_limit(
            bundle.get("max_attempts"),
            default=3,
        )

    @classmethod
    def _attempt_budget_exhausted(
        cls,
        task: _DuckRow | Mapping[str, Any],
        attempt: int,
    ) -> bool:
        limit = cls._max_attempts(task)
        return limit > 0 and int(attempt) >= limit

    @staticmethod
    def _execution_scope(task: _DuckRow) -> str:
        """Return the stable bundle scope that must have at most one live lease."""

        bundle = json.loads(task["bundle_json"])
        return str(bundle.get("bundle_key") or "").strip()

    def _active_execution_scope_conflict(
        self,
        connection: _DuckConnection,
        task: _DuckRow,
        *,
        now: int,
    ) -> _DuckRow | None:
        """Find a live lease for another revision of this task's bundle."""

        execution_scope = self._execution_scope(task)
        if not execution_scope:
            return None
        rows = connection.execute(
            """SELECT l.*, t.bundle_json, t.task_id
               FROM leases AS l
               JOIN tasks AS t ON t.task_cid=l.task_cid
               WHERE l.task_cid<>?
                 AND l.state='accepted'
                 AND l.expires_at_ms>?""",
            (str(task["task_cid"]), now),
        ).fetchall()
        return next(
            (row for row in rows if self._execution_scope(row) == execution_scope),
            None,
        )

    def _task_projection(
        self,
        task: _DuckRow,
        lease: _DuckRow | None,
        *,
        now: int,
    ) -> TaskLeaseState:
        bundle = json.loads(task["bundle_json"])
        lease_state = str(lease["state"]) if lease is not None else None
        max_attempts = self._max_attempts(task)
        attempt = int(lease["attempt"] or 0) if lease is not None else 0
        retry_not_before = int(lease["retry_not_before_ms"] or 0) if lease is not None else 0
        if lease_state == "accepted" and int(lease["expires_at_ms"]) > now:
            state = "accepted"
        elif lease_state == "completed":
            state = "completed"
        elif self._attempt_budget_exhausted(task, attempt) or retry_not_before > now:
            state = "blocked"
        else:
            state = "ready"
        return TaskLeaseState(
            task_cid=str(task["task_cid"]),
            goal_cid=str(task["goal_cid"]),
            subgoal_cid=str(task["subgoal_cid"]),
            task_id=str(task["task_id"]),
            bundle=bundle,
            state=state,
            lease_state=lease_state,
            claim_cid=str(lease["claim_cid"]) if lease is not None else None,
            resolution_cid=str(lease["resolution_cid"]) if lease is not None else None,
            claimant_did=str(lease["claimant_did"]) if lease is not None and state == "accepted" else None,
            logical_epoch=int(lease["logical_epoch"] or 0) if lease is not None else 0,
            fencing_token=int(lease["fencing_token"] or 0) if lease is not None else 0,
            lease_expires_at_ms=(
                int(lease["expires_at_ms"]) if lease is not None and state == "accepted" else None
            ),
            attempt=attempt,
            max_attempts=max_attempts,
            release_reason=(
                str(lease["release_reason"]) if lease is not None and lease["release_reason"] else None
            ),
            retry_not_before_ms=retry_not_before,
            registered_at_ms=int(task["registered_at_ms"] or 0),
            updated_at_ms=int(task["updated_at_ms"] or 0),
        )

    @staticmethod
    def _projection_dict(state: TaskLeaseState) -> dict[str, Any]:
        result = state.to_dict()
        result["bundle_key"] = str(state.bundle.get("bundle_key") or state.task_id)
        # Common spelling used by the lane manifest.
        result["expires_at_ms"] = state.lease_expires_at_ms
        release_reason = str(state.release_reason or "")
        if release_reason.startswith("deferred:pending_acceptance:"):
            result.update(
                {
                    "acceptance_pending": True,
                    "resumable": True,
                    "deferred_reason": "pending_acceptance",
                    "pending_gate": release_reason.rsplit(":", 1)[-1],
                }
            )
            if state.state == "blocked":
                result["blocked_reason"] = "acceptance_pending_cooldown"
        return result

    def _projection_with_claimability(
        self,
        connection: _DuckConnection,
        state: TaskLeaseState,
        *,
        max_evidence: int,
    ) -> dict[str, Any]:
        """Attach bounded dependency evidence to a scheduler projection."""

        result = self._projection_dict(state)
        readiness = self._claimability(
            connection,
            state.task_cid,
            max_evidence=max_evidence,
        )
        result.update(
            {
                "claimable": bool(readiness["claimable"]),
                "dependency_task_cids": list(readiness["dependency_task_cids"]),
                "satisfied_dependency_task_cids": list(
                    readiness["satisfied_dependency_task_cids"]
                ),
                "blocked_dependency_task_cids": list(
                    readiness["blocked_dependency_task_cids"]
                ),
                "blocking_task_cids": list(readiness["blocked_dependency_task_cids"]),
                "missing_dependency_task_cids": list(
                    readiness["missing_dependency_task_cids"]
                ),
                "dependency_cycles": list(readiness["dependency_cycles"]),
                "dependency_repair_evidence": list(readiness["repair_evidence"]),
                "claimability_evidence_truncated": bool(
                    readiness["evidence_truncated"]
                ),
            }
        )
        if result["state"] == "ready" and not result["claimable"]:
            result["state"] = "blocked"
            result["blocked_reason"] = "dependency_not_ready"
        return result

    @_coordinator_operation
    def list_tasks(
        self,
        *,
        task_cids: Iterable[str] | None = None,
        now_ms: int | None = None,
        include_claimability: bool = False,
        max_claimability_evidence: int = 8,
    ) -> list[dict[str, Any]]:
        """Return a consistent live projection, optionally scoped to current tasks."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        evidence_limit = int(max_claimability_evidence)
        if not 1 <= evidence_limit <= 256:
            raise ValueError("max_claimability_evidence must be in [1, 256]")
        selected = None if task_cids is None else sorted({str(item) for item in task_cids})
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                expired = connection.execute(
                    "SELECT task_cid FROM leases WHERE state='accepted' AND expires_at_ms<=?",
                    (now,),
                ).fetchall()
                for row in expired:
                    self._expire(connection, str(row["task_cid"]), now)
                query = """SELECT t.*, l.claim_cid, l.resolution_cid, l.claimant_did,
                                  l.logical_epoch, l.fencing_token, l.expires_at_ms,
                                  l.attempt, l.state, l.started_at_ms,
                                  l.release_reason, l.retry_not_before_ms
                           FROM tasks AS t
                           LEFT JOIN leases AS l ON l.task_cid=t.task_cid"""
                if selected is None:
                    rows = connection.execute(
                        f"{query} ORDER BY t.registered_at_ms, t.task_cid"
                    ).fetchall()
                else:
                    rows = []
                    for offset in range(0, len(selected), 500):
                        chunk = selected[offset : offset + 500]
                        placeholders = ",".join("?" for _item in chunk)
                        rows.extend(
                            connection.execute(
                                f"{query} WHERE t.task_cid IN ({placeholders})",
                                chunk,
                            ).fetchall()
                        )
                    rows.sort(key=lambda row: (int(row["registered_at_ms"]), str(row["task_cid"])))
                states = [
                    self._task_projection(
                        row,
                        row if row["state"] is not None else None,
                        now=now,
                    )
                    for row in rows
                ]
                result = [
                    (
                        self._projection_with_claimability(
                            connection,
                            state,
                            max_evidence=evidence_limit,
                        )
                        if include_claimability
                        else self._projection_dict(state)
                    )
                    for state in states
                ]
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def task_state(
        self,
        task_cid: str,
        *,
        now_ms: int | None = None,
        include_claimability: bool = False,
        max_claimability_evidence: int = 8,
    ) -> dict[str, Any] | None:
        """Return the live scheduler projection for ``task_cid``."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        evidence_limit = int(max_claimability_evidence)
        if not 1 <= evidence_limit <= 256:
            raise ValueError("max_claimability_evidence must be in [1, 256]")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT * FROM tasks WHERE task_cid=?", (task_cid,)
                ).fetchone()
                if task is None:
                    connection.commit()
                    return None
                self._expire(connection, task_cid, now)
                lease = connection.execute(
                    "SELECT * FROM leases WHERE task_cid=?", (task_cid,)
                ).fetchone()
                state = self._task_projection(task, lease, now=now)
                result = (
                    self._projection_with_claimability(
                        connection,
                        state,
                        max_evidence=evidence_limit,
                    )
                    if include_claimability
                    else self._projection_dict(state)
                )
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def claim(
        self,
        task_cid: str,
        claimant_did: str,
        *,
        requested_lease_ms: int = 60_000,
        now_ms: int | None = None,
    ) -> LeaseGrant:
        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                resolved_task_cid = self._resolve_task_cid(conn, task_cid)
                if resolved_task_cid is None:
                    raise KeyError(f"unknown task CID: {task_cid}")
                task_cid = resolved_task_cid
                task = conn.execute("SELECT * FROM tasks WHERE task_cid=?", (task_cid,)).fetchone()
                assert task is not None
                readiness = self._claimability(conn, task_cid, max_evidence=32)
                if not readiness["claimable"]:
                    count = len(readiness["blocked_dependency_task_cids"])
                    cycles = len(readiness["dependency_cycles"])
                    repairs = len(readiness["structural_dependency_repairs"])
                    raise DependencyNotReadyError(
                        f"task has {count} unsatisfied prerequisite(s), {cycles} dependency cycle(s), "
                        f"and {repairs} structural dependency repair(s)",
                        evidence=readiness,
                    )
                grant = self._claim_in_transaction(
                    conn, task, claimant_did, duration=duration, now=now
                )
                conn.commit()
                return grant
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def claim_ready(
        self,
        claimant_did: str,
        *,
        requested_lease_ms: int = 60_000,
        exclude_task_cids: Iterable[str] = (),
        eligible_task_cids: Iterable[str] | None = None,
        now_ms: int | None = None,
    ) -> LeaseGrant | None:
        """Atomically select and accept the oldest ready task.

        Selection and acceptance share one ``BEGIN IMMEDIATE`` transaction;
        two scheduler processes can therefore observe the same discovery set
        without ever accepting two leases for one task. Exclusions are applied
        in Python to avoid an unbounded dynamic SQL clause.
        """

        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        excluded = {str(item) for item in exclude_task_cids}
        eligible = (
            None
            if eligible_task_cids is None
            else {str(item) for item in eligible_task_cids}
        )
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                expired = connection.execute(
                    "SELECT task_cid FROM leases WHERE state='accepted' AND expires_at_ms<=?",
                    (now,),
                ).fetchall()
                for row in expired:
                    self._expire(connection, str(row["task_cid"]), now)
                candidates = connection.execute(
                    """SELECT t.*
                       FROM tasks AS t
                       LEFT JOIN leases AS l ON l.task_cid=t.task_cid
                       WHERE l.task_cid IS NULL
                          OR (l.state IN ('released','expired')
                              AND l.retry_not_before_ms<=?)
                       ORDER BY t.registered_at_ms, t.task_cid""",
                    (now,),
                ).fetchall()
                for task in candidates:
                    candidate_cid = str(task["task_cid"])
                    if candidate_cid in excluded or (
                        eligible is not None and candidate_cid not in eligible
                    ):
                        continue
                    # Discovery order is only a scheduling hint. Re-evaluate
                    # dependency receipts in this transaction so a dynamic
                    # worker can never claim a stale or blocked plan entry.
                    readiness = self._claimability(
                        connection, candidate_cid, max_evidence=32
                    )
                    if not readiness["claimable"]:
                        continue
                    # A finite attempt budget prevents a permanently failing
                    # task from monopolizing newly idle lanes.
                    lease = connection.execute(
                        "SELECT attempt FROM leases WHERE task_cid=?", (task["task_cid"],)
                    ).fetchone()
                    if lease is not None and self._attempt_budget_exhausted(
                        task,
                        int(lease["attempt"]),
                    ):
                        continue
                    if self._active_execution_scope_conflict(
                        connection, task, now=now
                    ) is not None:
                        continue
                    grant = self._claim_in_transaction(
                        connection, task, claimant_did, duration=duration, now=now
                    )
                    connection.commit()
                    return grant
                connection.commit()
                return None
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def steal(
        self,
        task_cid: str,
        claimant_did: str,
        *,
        requested_lease_ms: int = 60_000,
        now_ms: int | None = None,
    ) -> LeaseGrant:
        """Take over an expired or released task with a new fencing token.

        An active lease is never pre-empted, even when the requesting DID is
        the current owner. This stricter behavior makes work-stealing safe to
        call from a competing idle lane.
        """

        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT * FROM tasks WHERE task_cid=?", (task_cid,)
                ).fetchone()
                if task is None:
                    raise KeyError(f"unknown task CID: {task_cid}")
                self._expire(connection, task_cid, now)
                lease = connection.execute(
                    "SELECT * FROM leases WHERE task_cid=?", (task_cid,)
                ).fetchone()
                if lease is None:
                    raise LeaseConflictError("unclaimed ready tasks must use claim or claim_ready")
                if lease["state"] == "accepted":
                    raise LeaseConflictError(f"task is leased by {lease['claimant_did']}")
                if lease["state"] not in {"expired", "released"}:
                    raise LeaseConflictError(f"task cannot be stolen from state {lease['state']}")
                grant = self._claim_in_transaction(
                    connection, task, claimant_did, duration=duration, now=now
                )
                connection.commit()
                return grant
            except Exception:
                connection.rollback()
                raise

    def _claim_in_transaction(
        self,
        connection: _DuckConnection,
        task: _DuckRow,
        claimant_did: str,
        *,
        duration: int,
        now: int,
    ) -> LeaseGrant:
        """Accept one claim inside a caller-owned immediate transaction."""

        task_cid = str(task["task_cid"])
        self._expire(connection, task_cid, now)
        active = connection.execute(
            "SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms>?",
            (task_cid, now),
        ).fetchone()
        if active is not None:
            if active["claimant_did"] == claimant_did:
                return self._grant(active, task)
            raise LeaseConflictError(f"task is leased by {active['claimant_did']}")
        scope_conflict = self._active_execution_scope_conflict(
            connection, task, now=now
        )
        if scope_conflict is not None:
            execution_scope = self._execution_scope(task)
            raise ExecutionScopeConflictError(
                f"bundle execution scope {execution_scope!r} is leased by task "
                f"{scope_conflict['task_cid']} ({scope_conflict['claimant_did']})"
            )
        prior = connection.execute(
            "SELECT * FROM leases WHERE task_cid=?", (task_cid,)
        ).fetchone()
        if prior is not None and prior["state"] == "completed":
            raise LeaseConflictError("task already has a successful terminal receipt")
        if prior is not None and self._attempt_budget_exhausted(
            task,
            int(prior["attempt"] or 0),
        ):
            raise LeaseConflictError("task attempt budget is exhausted")
        retry_not_before = int(prior["retry_not_before_ms"] or 0) if prior is not None else 0
        if retry_not_before > now:
            raise LeaseConflictError(f"task is cooling down until {retry_not_before}")
        token = int(prior["fencing_token"] or 0) + 1 if prior is not None else 1
        epoch = int(prior["logical_epoch"] or 0) + 1 if prior is not None else 1
        attempt = int(prior["attempt"] or 0) + 1 if prior is not None else 1
        claim = self._claim_payload(task, claimant_did, epoch, attempt, duration, now)
        claim_cid = self._put_artifact(connection, "TaskClaim", claim)
        expires = now + duration
        resolution = {
            "schema": "mcp++/profile-g/claim-resolution@1",
            "created_at_ms": now,
            "parents": [claim_cid],
            "correlation_id": claim["correlation_id"],
            "task_cid": task_cid,
            "logical_epoch": epoch,
            "considered_claim_cids": [claim_cid],
            "accepted_claim_cid": claim_cid,
            "outcome": "accepted",
            "fencing_token": token,
            "lease_expires_at_ms": expires,
            "attestation_cids": [],
            "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}),
            "policy_decision_cid": claim["policy_decision_cid"],
            "coordination_receipt_cid": None,
            "retry_not_before_ms": 0,
            "resolver_did": "did:web:ipfs-accelerate.local",
        }
        resolution_cid = self._put_artifact(connection, "ClaimResolution", resolution)
        connection.execute(
            """INSERT INTO leases(
                   task_cid, claim_cid, resolution_cid, claimant_did,
                   logical_epoch, fencing_token, expires_at_ms, attempt,
                   state, started_at_ms, release_reason, retry_not_before_ms
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(task_cid) DO UPDATE SET
                 claim_cid=excluded.claim_cid,
                 resolution_cid=excluded.resolution_cid,
                 claimant_did=excluded.claimant_did,
                 logical_epoch=excluded.logical_epoch,
                 fencing_token=excluded.fencing_token,
                 expires_at_ms=excluded.expires_at_ms,
                 attempt=excluded.attempt,
                 state='accepted',
                 started_at_ms=excluded.started_at_ms,
                 release_reason=NULL,
                 retry_not_before_ms=0""",
            (
                task_cid,
                claim_cid,
                resolution_cid,
                claimant_did,
                epoch,
                token,
                expires,
                attempt,
                "accepted",
                now,
                None,
                0,
            ),
        )
        connection.execute("INSERT INTO token_history VALUES(?,?)", (task_cid, token))
        return LeaseGrant(
            task_cid,
            task["goal_cid"],
            task["subgoal_cid"],
            claim_cid,
            resolution_cid,
            claimant_did,
            epoch,
            token,
            expires,
            attempt,
        )

    def _claim_payload(self, task: _DuckRow, claimant: str, epoch: int, attempt: int, duration: int, now: int) -> dict[str, Any]:
        bundle = json.loads(task["bundle_json"])
        correlation = str(bundle.get("correlation_id") or bundle.get("bundle_key") or task["task_id"])[:128]
        return {
            "schema": "mcp++/profile-g/task-claim@1", "created_at_ms": now, "parents": [],
            "correlation_id": correlation, "task_cid": task["task_cid"],
            "proposal_cid": _link({"proposal": task["task_cid"], "epoch": epoch}),
            "claimant_did": claimant, "record_cid": _link({"peer": claimant}), "logical_epoch": epoch,
            "requested_lease_ms": duration, "risk_bucket": 0, "capability_fit_millionths": 1_000_000,
            "expected_finish_ms": now + duration, "proof_cid": str(bundle.get("proof_cid") or _link({"proof": claimant})),
            "policy_decision_cid": str(bundle.get("policy_decision_cid") or _link({"decision": "allow"})), "attempt": attempt,
        }

    def _grant(self, lease: _DuckRow, task: _DuckRow | None = None) -> LeaseGrant:
        task = task or self._connection.execute("SELECT * FROM tasks WHERE task_cid=?", (lease["task_cid"],)).fetchone()
        assert task is not None
        return LeaseGrant(lease["task_cid"], task["goal_cid"], task["subgoal_cid"], lease["claim_cid"], lease["resolution_cid"], lease["claimant_did"], lease["logical_epoch"], lease["fencing_token"], lease["expires_at_ms"], lease["attempt"])

    def _current(self, connection: _DuckConnection, grant: LeaseGrant, now: int) -> _DuckRow:
        self._expire(connection, grant.task_cid, now)
        row = connection.execute("SELECT * FROM leases WHERE task_cid=?", (grant.task_cid,)).fetchone()
        if row is None or row["state"] != "accepted" or row["expires_at_ms"] <= now:
            raise LeaseExpiredError("lease has expired or was released")
        if row["claim_cid"] != grant.claim_cid or row["claimant_did"] != grant.claimant_did:
            raise StaleFencingTokenError("claim is no longer accepted")
        if row["fencing_token"] != grant.fencing_token:
            raise StaleFencingTokenError("fencing token is stale")
        return row

    @_coordinator_operation
    def validate(self, grant: LeaseGrant, *, now_ms: int | None = None) -> LeaseGrant:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(self._connection, grant, now)
                result = self._grant(row)
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    @staticmethod
    def _distributed_input(
        value: ImmutableLaneInputArtifact | Mapping[str, Any],
    ) -> ImmutableLaneInputArtifact:
        if isinstance(value, ImmutableLaneInputArtifact):
            # Reconstructing verifies that even a caller-mutated nested mapping
            # still matches the content identity originally assigned to it.
            return ImmutableLaneInputArtifact.from_dict(value.to_dict())
        if isinstance(value, Mapping):
            return ImmutableLaneInputArtifact.from_dict(value)
        raise ValueError("input_artifact must be an ImmutableLaneInputArtifact or mapping")

    @staticmethod
    def _capability_receipt(
        value: WorkerCapabilityReceipt | Mapping[str, Any],
    ) -> WorkerCapabilityReceipt:
        if isinstance(value, WorkerCapabilityReceipt):
            return WorkerCapabilityReceipt.from_dict(value.to_dict())
        if isinstance(value, Mapping):
            return WorkerCapabilityReceipt.from_dict(value)
        raise ValueError("capability_receipt must be a WorkerCapabilityReceipt or mapping")

    @staticmethod
    def _environment_receipt(
        value: WorkerEnvironmentReceipt | Mapping[str, Any],
    ) -> WorkerEnvironmentReceipt:
        if isinstance(value, WorkerEnvironmentReceipt):
            return WorkerEnvironmentReceipt.from_dict(value.to_dict())
        if isinstance(value, Mapping):
            return WorkerEnvironmentReceipt.from_dict(value)
        raise ValueError("environment_receipt must be a WorkerEnvironmentReceipt or mapping")

    @staticmethod
    def _store_distributed_records(
        connection: _DuckConnection,
        input_artifact: ImmutableLaneInputArtifact,
        capability_receipt: WorkerCapabilityReceipt,
        environment_receipt: WorkerEnvironmentReceipt,
    ) -> None:
        records = (
            (
                "distributed_inputs",
                "artifact_id",
                input_artifact.artifact_id,
                (
                    input_artifact.artifact_id,
                    input_artifact.task_cid,
                    input_artifact.repository_id,
                    canonical_profile_g_bytes(input_artifact.to_dict()).decode("utf-8"),
                ),
                "INSERT INTO distributed_inputs VALUES(?,?,?,?)",
            ),
            (
                "worker_capability_receipts",
                "receipt_id",
                capability_receipt.receipt_id,
                (
                    capability_receipt.receipt_id,
                    capability_receipt.worker_id,
                    capability_receipt.expires_at_ms,
                    canonical_profile_g_bytes(capability_receipt.to_dict()).decode("utf-8"),
                ),
                "INSERT INTO worker_capability_receipts VALUES(?,?,?,?)",
            ),
            (
                "worker_environment_receipts",
                "receipt_id",
                environment_receipt.receipt_id,
                (
                    environment_receipt.receipt_id,
                    environment_receipt.worker_id,
                    environment_receipt.capability_receipt_id,
                    environment_receipt.expires_at_ms,
                    canonical_profile_g_bytes(environment_receipt.to_dict()).decode("utf-8"),
                ),
                "INSERT INTO worker_environment_receipts VALUES(?,?,?,?,?)",
            ),
        )
        for table, id_column, identity, values, insert in records:
            existing = connection.execute(
                f"SELECT payload_json FROM {table} WHERE {id_column}=?",
                (identity,),
            ).fetchone()
            payload_json = str(values[-1])
            if existing is not None:
                if str(existing["payload_json"]) != payload_json:
                    raise ValueError(
                        f"content identity collision in {table}: {identity}"
                    )
                continue
            connection.execute(insert, values)

    @staticmethod
    def _validate_distributed_bindings(
        *,
        grant: LeaseGrant,
        input_artifact: ImmutableLaneInputArtifact,
        capability_receipt: WorkerCapabilityReceipt,
        environment_receipt: WorkerEnvironmentReceipt,
        worker_id: str,
        repository_id: str,
        required_capabilities: Iterable[str],
        now: int,
    ) -> tuple[str, ...]:
        worker = _required_text(worker_id, "worker_id")
        repository = _required_text(repository_id, "repository_id")
        if input_artifact.task_cid != grant.task_cid:
            raise ValueError("immutable input is bound to a foreign task")
        if input_artifact.repository_id != repository:
            raise ValueError("immutable input is bound to a foreign repository")
        if capability_receipt.worker_id != worker:
            raise ValueError("capability receipt is bound to a foreign worker")
        if environment_receipt.worker_id != worker:
            raise ValueError("environment receipt is bound to a foreign worker")
        if environment_receipt.capability_receipt_id != capability_receipt.receipt_id:
            raise ValueError("environment receipt is not bound to the capability receipt")
        capability_receipt.validate_at(now)
        environment_receipt.validate_at(now)
        if capability_receipt.expires_at_ms < grant.lease_expires_at_ms:
            raise ValueError("capability receipt expires before the accepted lease")
        if environment_receipt.expires_at_ms < grant.lease_expires_at_ms:
            raise ValueError("environment receipt expires before the accepted lease")
        required = tuple(
            sorted({_required_text(item, "required capability") for item in required_capabilities})
        )
        missing = sorted(set(required) - set(capability_receipt.capabilities))
        if missing:
            raise ValueError(f"worker is missing required capabilities: {', '.join(missing)}")
        return required

    @_coordinator_operation
    def register_distributed_input(
        self,
        artifact: ImmutableLaneInputArtifact | Mapping[str, Any],
    ) -> ImmutableLaneInputArtifact:
        """Persist one verified immutable remote input idempotently."""

        normalized = self._distributed_input(artifact)
        with self._lock, self._connection:
            # The paired receipt arguments are intentionally absent here; this
            # operation supports pre-staging immutable inputs before selection.
            existing = self._connection.execute(
                "SELECT payload_json FROM distributed_inputs WHERE artifact_id=?",
                (normalized.artifact_id,),
            ).fetchone()
            payload_json = canonical_profile_g_bytes(normalized.to_dict()).decode("utf-8")
            if existing is not None and str(existing["payload_json"]) != payload_json:
                raise ValueError("immutable lane input identity collision")
            if existing is None:
                self._connection.execute(
                    "INSERT INTO distributed_inputs VALUES(?,?,?,?)",
                    (
                        normalized.artifact_id,
                        normalized.task_cid,
                        normalized.repository_id,
                        payload_json,
                    ),
                )
        return normalized

    @_coordinator_operation
    def register_worker_capability_receipt(
        self,
        receipt: WorkerCapabilityReceipt | Mapping[str, Any],
        *,
        now_ms: int | None = None,
    ) -> WorkerCapabilityReceipt:
        """Persist a current, content-valid worker capability receipt."""

        normalized = self._capability_receipt(receipt)
        normalized.validate_at(self._clock_ms() if now_ms is None else int(now_ms))
        payload_json = canonical_profile_g_bytes(normalized.to_dict()).decode("utf-8")
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT payload_json FROM worker_capability_receipts WHERE receipt_id=?",
                (normalized.receipt_id,),
            ).fetchone()
            if existing is not None and str(existing["payload_json"]) != payload_json:
                raise ValueError("worker capability receipt identity collision")
            if existing is None:
                self._connection.execute(
                    "INSERT INTO worker_capability_receipts VALUES(?,?,?,?)",
                    (
                        normalized.receipt_id,
                        normalized.worker_id,
                        normalized.expires_at_ms,
                        payload_json,
                    ),
                )
        return normalized

    @_coordinator_operation
    def register_worker_environment_receipt(
        self,
        receipt: WorkerEnvironmentReceipt | Mapping[str, Any],
        *,
        capability_receipt: WorkerCapabilityReceipt | Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> WorkerEnvironmentReceipt:
        """Persist an environment receipt after checking its worker/capability binding."""

        normalized = self._environment_receipt(receipt)
        now = self._clock_ms() if now_ms is None else int(now_ms)
        normalized.validate_at(now)
        with self._lock, self._connection:
            if capability_receipt is None:
                row = self._connection.execute(
                    "SELECT payload_json FROM worker_capability_receipts WHERE receipt_id=?",
                    (normalized.capability_receipt_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("environment receipt references an unknown capability receipt")
                capability = WorkerCapabilityReceipt.from_dict(json.loads(row["payload_json"]))
            else:
                capability = self._capability_receipt(capability_receipt)
            capability.validate_at(now)
            if capability.receipt_id != normalized.capability_receipt_id:
                raise ValueError("environment receipt capability binding does not match")
            if capability.worker_id != normalized.worker_id:
                raise ValueError("environment and capability receipts name different workers")
            payload_json = canonical_profile_g_bytes(normalized.to_dict()).decode("utf-8")
            existing = self._connection.execute(
                "SELECT payload_json FROM worker_environment_receipts WHERE receipt_id=?",
                (normalized.receipt_id,),
            ).fetchone()
            if existing is not None and str(existing["payload_json"]) != payload_json:
                raise ValueError("worker environment receipt identity collision")
            if existing is None:
                self._connection.execute(
                    "INSERT INTO worker_environment_receipts VALUES(?,?,?,?,?)",
                    (
                        normalized.receipt_id,
                        normalized.worker_id,
                        normalized.capability_receipt_id,
                        normalized.expires_at_ms,
                        payload_json,
                    ),
                )
        return normalized

    @_coordinator_operation
    def dispatch_remote(
        self,
        grant: LeaseGrant,
        *,
        input_artifact: ImmutableLaneInputArtifact | Mapping[str, Any],
        capability_receipt: WorkerCapabilityReceipt | Mapping[str, Any],
        environment_receipt: WorkerEnvironmentReceipt | Mapping[str, Any],
        worker_id: str | None = None,
        repository_id: str = "",
        required_capabilities: Iterable[str] = (),
        now_ms: int | None = None,
    ) -> DistributedLaneDispatch:
        """Bind immutable work and expiring worker receipts to an accepted lease."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        artifact = self._distributed_input(input_artifact)
        capability = self._capability_receipt(capability_receipt)
        environment = self._environment_receipt(environment_receipt)
        worker = str(worker_id or capability.worker_id)
        repository = str(repository_id or artifact.repository_id)
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._current(connection, grant, now)
                required = self._validate_distributed_bindings(
                    grant=grant,
                    input_artifact=artifact,
                    capability_receipt=capability,
                    environment_receipt=environment,
                    worker_id=worker,
                    repository_id=repository,
                    required_capabilities=required_capabilities,
                    now=now,
                )
                self._store_distributed_records(
                    connection, artifact, capability, environment
                )
                body = {
                    "schema": DISTRIBUTED_LANE_DISPATCH_SCHEMA,
                    "created_at_ms": now,
                    "repository_id": repository,
                    "worker_id": worker,
                    "task_cid": grant.task_cid,
                    "input_artifact_cid": artifact.artifact_id,
                    "capability_receipt_cid": capability.receipt_id,
                    "environment_receipt_cid": environment.receipt_id,
                    "claim_cid": grant.claim_cid,
                    "logical_epoch": grant.logical_epoch,
                    "fencing_token": grant.fencing_token,
                    "lease_expires_at_ms": grant.lease_expires_at_ms,
                    "required_capabilities": list(required),
                }
                dispatch_cid = self._put_artifact(
                    connection, "DistributedLaneDispatch", body
                )
                dispatch = DistributedLaneDispatch(
                    grant=grant,
                    input_artifact_cid=artifact.artifact_id,
                    capability_receipt_cid=capability.receipt_id,
                    environment_receipt_cid=environment.receipt_id,
                    dispatch_cid=dispatch_cid,
                    worker_id=worker,
                    repository_id=repository,
                    required_capabilities=required,
                    lease_duration_ms=grant.lease_expires_at_ms - now,
                )
                payload_json = canonical_profile_g_bytes(dispatch.to_dict()).decode("utf-8")
                previous = connection.execute(
                    "SELECT payload_json FROM distributed_dispatches WHERE dispatch_cid=?",
                    (dispatch_cid,),
                ).fetchone()
                if previous is not None and str(previous["payload_json"]) != payload_json:
                    raise ValueError("distributed dispatch identity collision")
                if previous is None:
                    connection.execute(
                        "INSERT INTO distributed_dispatches VALUES(?,?,?,?,?,?,?)",
                        (
                            dispatch_cid,
                            grant.task_cid,
                            grant.claim_cid,
                            worker,
                            grant.fencing_token,
                            None,
                            payload_json,
                        ),
                    )
                connection.commit()
                return dispatch
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def renew(self, grant: LeaseGrant, *, requested_lease_ms: int = 60_000, now_ms: int | None = None) -> LeaseGrant:
        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                expires = now + duration
                claim = json.loads(conn.execute("SELECT payload_json FROM artifacts WHERE cid=?", (grant.claim_cid,)).fetchone()[0])
                event = {
                    "schema": "mcp++/profile-g/claim-resolution@1", "created_at_ms": now,
                    "parents": [row["resolution_cid"]], "correlation_id": claim["correlation_id"],
                    "task_cid": grant.task_cid, "logical_epoch": grant.logical_epoch,
                    "considered_claim_cids": [grant.claim_cid], "accepted_claim_cid": grant.claim_cid,
                    "outcome": "accepted", "fencing_token": grant.fencing_token,
                    "lease_expires_at_ms": expires, "attestation_cids": [],
                    "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}),
                    "policy_decision_cid": claim["policy_decision_cid"], "coordination_receipt_cid": None,
                    "retry_not_before_ms": 0, "resolver_did": "did:web:ipfs-accelerate.local",
                }
                renewal_cid = self._put_artifact(conn, "ClaimResolution", event)
                conn.execute("UPDATE leases SET expires_at_ms=?, resolution_cid=? WHERE task_cid=?", (expires, renewal_cid, grant.task_cid))
                conn.commit()
                return LeaseGrant(grant.task_cid, grant.goal_cid, grant.subgoal_cid, grant.claim_cid, renewal_cid, grant.claimant_did, grant.logical_epoch, grant.fencing_token, expires, grant.attempt)
            except Exception:
                conn.rollback()
                raise

    def _resolution_payload(self, row: _DuckRow, *, outcome: str, now: int) -> dict[str, Any]:
        claim = json.loads(self._connection.execute("SELECT payload_json FROM artifacts WHERE cid=?", (row["claim_cid"],)).fetchone()[0])
        return {
            "schema": "mcp++/profile-g/claim-resolution@1", "created_at_ms": now, "parents": [row["resolution_cid"]],
            "correlation_id": claim["correlation_id"], "task_cid": row["task_cid"], "logical_epoch": row["logical_epoch"],
            "considered_claim_cids": [row["claim_cid"]], "accepted_claim_cid": None, "outcome": outcome,
            "fencing_token": row["fencing_token"], "lease_expires_at_ms": None, "attestation_cids": [],
            "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}), "policy_decision_cid": claim["policy_decision_cid"],
            "coordination_receipt_cid": None, "retry_not_before_ms": 0, "resolver_did": "did:web:ipfs-accelerate.local",
        }

    @_coordinator_operation
    def release(
        self,
        grant: LeaseGrant,
        *,
        reason: str = "released",
        now_ms: int | None = None,
    ) -> str:
        """Voluntarily return accepted work to the ready pool.

        ``reason`` is scheduler metadata (for example ``drained`` or
        ``blocked``), not lease authority. The accepted claim and fencing token
        are still checked atomically before the state transition.
        """

        now = self._clock_ms() if now_ms is None else int(now_ms)
        reason = str(reason or "released")[:256]
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                cid = self._put_artifact(conn, "ClaimResolution", self._resolution_payload(row, outcome="released", now=now))
                conn.execute(
                    """UPDATE leases SET state='released', resolution_cid=?, release_reason=?
                       WHERE task_cid=? AND claim_cid=? AND fencing_token=?""",
                    (cid, reason, grant.task_cid, grant.claim_cid, grant.fencing_token),
                )
                conn.commit()
                return cid
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def defer_pending_acceptance(
        self,
        grant: LeaseGrant,
        *,
        evidence: Mapping[str, Any],
        retry_delay_ms: int = 30_000,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Return provider-review-pending work without publishing a receipt.

        This transition is deliberately narrower than :meth:`release`.  It
        accepts only exact, non-authoritative provider-review-pending evidence,
        rolls back the coordination attempt charged by the current claim, and
        applies a bounded cooldown before the task becomes claimable again.
        The resulting ``ClaimResolution`` is resumability evidence, never task
        completion authority.
        """

        if not isinstance(evidence, Mapping):
            raise ValueError("pending acceptance evidence must be a mapping")

        def exact_text_sequence(value: Any) -> tuple[str, ...]:
            if not isinstance(value, (list, tuple)):
                return ()
            if not value or any(
                not isinstance(item, str)
                or not item
                or item != item.strip()
                for item in value
            ):
                return ()
            return tuple(value)

        pending_gates = exact_text_sequence(evidence.get("pending_gates"))
        task_ids = exact_text_sequence(evidence.get("task_ids"))
        task_cids = exact_text_sequence(evidence.get("task_cids"))
        event_ids = exact_text_sequence(evidence.get("acceptance_event_ids"))
        raw_task_cids_by_id = evidence.get("task_cids_by_id")
        exact_task_cids_by_id = bool(
            isinstance(raw_task_cids_by_id, Mapping)
            and raw_task_cids_by_id
            and all(
                isinstance(task_id, str)
                and task_id == task_id.strip()
                and bool(task_id)
                and isinstance(task_cid, str)
                and task_cid == task_cid.strip()
                and _is_profile_g_cid(task_cid)
                for task_id, task_cid in raw_task_cids_by_id.items()
            )
        )
        task_cids_by_id = (
            {
                str(task_id).strip(): str(task_cid).strip()
                for task_id, task_cid in raw_task_cids_by_id.items()
            }
            if isinstance(raw_task_cids_by_id, Mapping)
            else {}
        )
        terminal_event_id = str(evidence.get("terminal_event_id") or "").strip()
        if (
            evidence.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/pending-acceptance@1"
            or evidence.get("acceptance_pending") is not True
            or evidence.get("completion_authoritative") is not False
            or evidence.get("admitted") is not False
            or pending_gates != ("provider_review",)
            or not task_ids
            or len(set(task_ids)) != len(task_ids)
            or len(task_ids) != len(task_cids)
            or len(set(task_cids)) != len(task_cids)
            or any(not _is_profile_g_cid(task_cid) for task_cid in task_cids)
            or not exact_task_cids_by_id
            or set(task_cids_by_id) != set(task_ids)
            or any(
                not task_cid or task_cids_by_id.get(task_id) != task_cid
                for task_id, task_cid in zip(task_ids, task_cids)
            )
            or len(event_ids) != len(task_ids)
            or len(set(event_ids)) != len(event_ids)
            or not terminal_event_id
            or terminal_event_id in event_ids
        ):
            raise ValueError(
                "pending acceptance deferral requires exact, durable, "
                "provider-review-only evidence"
            )
        delay = int(retry_delay_ms)
        if not 1 <= delay <= MAX_LEASE_MS:
            raise ValueError(
                f"retry_delay_ms must be in [1, {MAX_LEASE_MS}]"
            )
        normalized_evidence = _canonical_mapping(
            dict(evidence),
            "pending acceptance evidence",
        )
        now = self._clock_ms() if now_ms is None else int(now_ms)
        retry_not_before = now + delay
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                task = conn.execute(
                    "SELECT bundle_json FROM tasks WHERE task_cid=?",
                    (grant.task_cid,),
                ).fetchone()
                if task is None:
                    raise LeaseError("leased task registration is missing")
                bundle = json.loads(str(task["bundle_json"]))
                registered_slice_ids = exact_text_sequence(
                    bundle.get("execution_slice_task_ids")
                )
                registered_slice_cids = exact_text_sequence(
                    bundle.get("execution_slice_task_cids")
                )
                members = bundle.get("tasks")
                member_pairs = [
                    (
                        str(member.get("task_id") or "").strip(),
                        str(
                            member.get("canonical_task_cid")
                            or member.get("task_cid")
                            or ""
                        ).strip(),
                    )
                    for member in (
                        members if isinstance(members, (list, tuple)) else ()
                    )
                    if isinstance(member, Mapping)
                ]
                selected_id_set = set(registered_slice_ids)
                selected_cid_set = set(registered_slice_cids)
                selected_pairs = [
                    (task_id, task_cid)
                    for task_id, task_cid in member_pairs
                    if task_id in selected_id_set and task_cid in selected_cid_set
                ]
                registered_task_cids_by_id = dict(selected_pairs)
                if (
                    not registered_slice_ids
                    or len(set(registered_slice_ids)) != len(registered_slice_ids)
                    or len(registered_slice_ids) != len(registered_slice_cids)
                    or len(set(registered_slice_cids)) != len(registered_slice_cids)
                    or any(
                        not _is_profile_g_cid(task_cid)
                        for task_cid in registered_slice_cids
                    )
                    or not registered_task_cids_by_id
                    or len(selected_pairs) != len(registered_task_cids_by_id)
                    or any(
                        not task_id or not task_cid
                        for task_id, task_cid in registered_task_cids_by_id.items()
                    )
                    or set(registered_task_cids_by_id) != selected_id_set
                    or set(registered_task_cids_by_id.values()) != selected_cid_set
                    or task_cids_by_id != registered_task_cids_by_id
                ):
                    raise ValueError(
                        "pending acceptance evidence is not bound to the "
                        "leased execution slice"
                    )
                resolution = self._resolution_payload(
                    row,
                    outcome="released",
                    now=now,
                )
                resolution.update(
                    {
                        "retry_not_before_ms": retry_not_before,
                        "deferred_reason": "pending_acceptance",
                        "pending_acceptance": normalized_evidence,
                    }
                )
                resolution_cid = self._put_artifact(
                    conn,
                    "ClaimResolution",
                    resolution,
                )
                prior_attempt = max(0, int(row["attempt"] or 0) - 1)
                conn.execute(
                    """UPDATE leases
                       SET state='released', resolution_cid=?, attempt=?,
                           release_reason=?, retry_not_before_ms=?
                       WHERE task_cid=? AND claim_cid=? AND fencing_token=?""",
                    (
                        resolution_cid,
                        prior_attempt,
                        "deferred:pending_acceptance:provider_review",
                        retry_not_before,
                        grant.task_cid,
                        grant.claim_cid,
                        grant.fencing_token,
                    ),
                )
                conn.commit()
                return {
                    "resolution_cid": resolution_cid,
                    "task_cid": grant.task_cid,
                    "attempt": prior_attempt,
                    "retry_not_before_ms": retry_not_before,
                    "acceptance_pending": True,
                    "resumable": True,
                    "completion_authoritative": False,
                }
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def heartbeat(
        self,
        grant: LeaseGrant,
        *,
        capacity_millionths: int,
        ttl_ms: int = 15_000,
        now_ms: int | None = None,
        active_phase: str | None = None,
        cpu_millionths: int | None = None,
        cpu_percent: int | None = None,
        memory_percent: int | None = None,
        disk_percent: int | None = None,
        memory_used_bytes: int | None = None,
        memory_available_bytes: int | None = None,
        memory_total_bytes: int | None = None,
        disk_used_bytes: int | None = None,
        disk_available_bytes: int | None = None,
        disk_total_bytes: int | None = None,
        occupied_workers: int | None = None,
        available_workers: int | None = None,
        resource_class: str | None = None,
        provider_id: str | None = None,
        provider_capacity: Mapping[str, Any] | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish a live, fenced worker heartbeat.

        Resource measurements use integers so the resulting Profile G artifact
        remains canonical DAG-JSON.  Callers that sample fractional percentages
        should use fixed-point units (``cpu_millionths`` is one million for one
        fully occupied CPU).  Optional provider and detail mappings are retained
        in ``payload_json`` without widening the stable DuckDB table.
        """

        def optional_integer(name: str, value: int | None) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            return value

        def optional_text(name: str, value: str | None) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            return value

        now = self._clock_ms() if now_ms is None else int(now_ms)
        capacity = optional_integer("capacity_millionths", capacity_millionths)
        assert capacity is not None  # Required by the public signature.
        if not 0 <= capacity <= 1_000_000:
            raise ValueError("capacity_millionths must be in [0, 1000000]")
        ttl = optional_integer("ttl_ms", ttl_ms)
        assert ttl is not None
        measurements = {
            "cpu_millionths": optional_integer("cpu_millionths", cpu_millionths),
            "cpu_percent": optional_integer("cpu_percent", cpu_percent),
            "memory_percent": optional_integer("memory_percent", memory_percent),
            "disk_percent": optional_integer("disk_percent", disk_percent),
            "memory_used_bytes": optional_integer("memory_used_bytes", memory_used_bytes),
            "memory_available_bytes": optional_integer("memory_available_bytes", memory_available_bytes),
            "memory_total_bytes": optional_integer("memory_total_bytes", memory_total_bytes),
            "disk_used_bytes": optional_integer("disk_used_bytes", disk_used_bytes),
            "disk_available_bytes": optional_integer("disk_available_bytes", disk_available_bytes),
            "disk_total_bytes": optional_integer("disk_total_bytes", disk_total_bytes),
            "occupied_workers": optional_integer("occupied_workers", occupied_workers),
            "available_workers": optional_integer("available_workers", available_workers),
        }
        text_fields = {
            "active_phase": optional_text("active_phase", active_phase),
            "resource_class": optional_text("resource_class", resource_class),
            "provider_id": optional_text("provider_id", provider_id),
        }
        if provider_capacity is not None and not isinstance(provider_capacity, Mapping):
            raise ValueError("provider_capacity must be a mapping")
        if detail is not None and not isinstance(detail, Mapping):
            raise ValueError("detail must be a mapping")
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._current(conn, grant, now)
                payload = {"schema": "ipfs_accelerate_py/daemon-heartbeat@1", "created_at_ms": now,
                           "task_cid": grant.task_cid, "goal_cid": grant.goal_cid, "subgoal_cid": grant.subgoal_cid,
                           "claim_cid": grant.claim_cid, "claimant_did": grant.claimant_did,
                           "fencing_token": grant.fencing_token, "capacity_millionths": capacity,
                           "expires_at_ms": min(grant.lease_expires_at_ms, now + ttl)}
                payload.update({key: value for key, value in measurements.items() if value is not None})
                payload.update({key: value for key, value in text_fields.items() if value is not None})
                if provider_capacity is not None:
                    payload["provider_capacity"] = dict(provider_capacity)
                if detail is not None:
                    payload["detail"] = dict(detail)
                # Validate the complete payload before touching either artifact
                # table. This rejects nested floats and unsupported containers.
                canonical_profile_g_bytes(payload)
                cid = self._put_artifact(conn, "DaemonHeartbeat", payload)
                conn.execute("INSERT OR REPLACE INTO heartbeats VALUES(?,?,?,?,?,?,?,?)",
                             (cid, grant.task_cid, grant.claimant_did, grant.fencing_token, now, payload["expires_at_ms"], capacity, json.dumps(payload, sort_keys=True)))
                self._prune_heartbeat_history(conn, grant)
                conn.commit()
                return {**payload, "heartbeat_cid": cid}
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def latest_heartbeats(
        self,
        *,
        task_cids: Iterable[str] | None = None,
        provider_id: str | None = None,
        include_expired: bool = False,
        now_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the newest heartbeat for each task.

        Expired advertisements are excluded by default so callers cannot use a
        dead worker's resource or provider capacity for admission decisions.
        Historical inspection may opt in with ``include_expired=True``.
        ``provider_id`` matches the explicit provider telemetry identifier, not
        the lease claimant DID; legacy heartbeats therefore do not match it.
        """

        now = self._clock_ms() if now_ms is None else int(now_ms)
        selected = None if task_cids is None else {str(item) for item in task_cids}
        if selected == set():
            return []
        if provider_id is not None and not isinstance(provider_id, str):
            raise ValueError("provider_id must be a string")
        with self._lock:
            query = "SELECT * FROM heartbeats"
            clauses: list[str] = []
            parameters: list[Any] = []
            if not include_expired:
                clauses.append("expires_at_ms>?")
                parameters.append(now)
            if selected is not None:
                placeholders = ",".join("?" for _item in selected)
                clauses.append(f"task_cid IN ({placeholders})")
                parameters.extend(sorted(selected))
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY observed_at_ms DESC, heartbeat_cid DESC"
            rows = self._connection.execute(query, parameters).fetchall()

        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            task_cid = str(row["task_cid"])
            if task_cid in latest:
                continue
            try:
                payload = json.loads(str(row["payload_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if provider_id is not None and payload.get("provider_id") != provider_id:
                continue
            payload["heartbeat_cid"] = str(row["heartbeat_cid"])
            latest[task_cid] = payload
        return [latest[task_cid] for task_cid in sorted(latest)]

    @_coordinator_operation
    def latest_heartbeat(
        self,
        task_cid: str,
        *,
        include_expired: bool = False,
        now_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest current heartbeat for one task, if any."""

        items = self.latest_heartbeats(
            task_cids=(task_cid,),
            include_expired=include_expired,
            now_ms=now_ms,
        )
        return items[0] if items else None

    def _receipt_in_transaction(
        self,
        connection: _DuckConnection,
        row: _DuckRow,
        grant: LeaseGrant,
        *,
        status: str,
        output: Mapping[str, Any] | None,
        failure_class: str,
        started_at_ms: int | None,
        now: int,
    ) -> dict[str, Any]:
        """Write one terminal receipt inside a caller-owned fenced transaction."""

        if status == "succeeded" and output is None:
            raise ValueError("successful receipt requires output")
        normalized_output = (
            _canonical_mapping(output, "output") if output is not None else None
        )
        output_cid = _link(normalized_output) if normalized_output is not None else None
        claim_row = connection.execute(
            "SELECT payload_json FROM artifacts WHERE cid=?", (grant.claim_cid,)
        ).fetchone()
        if claim_row is None:
            raise ValueError("accepted claim artifact is missing")
        claim = json.loads(claim_row[0])
        payload = {
            "schema": "mcp++/profile-g/task-receipt@1",
            "created_at_ms": now,
            "parents": [row["resolution_cid"]],
            "correlation_id": claim["correlation_id"],
            "task_cid": grant.task_cid,
            "claim_cid": grant.claim_cid,
            "resolution_cid": row["resolution_cid"],
            "fencing_token": grant.fencing_token,
            "profile_b_receipt_cid": _link(
                {
                    "task": grant.task_cid,
                    "token": grant.fencing_token,
                    "finished": now,
                }
            ),
            "output_cid": output_cid,
            "status": status,
            "failure_class": failure_class,
            "attempt": grant.attempt,
            "started_at_ms": int(
                started_at_ms
                if started_at_ms is not None
                else row["started_at_ms"]
            ),
            "finished_at_ms": now,
            "resource_use_cid": _link(
                {"heartbeats": self._heartbeat_count(connection, grant)}
            ),
            "provider": "ipfs_accelerate_py",
            "provider_version": PROVIDER_VERSION,
            "next_state": "complete" if status == "succeeded" else "ready",
        }
        cid = self._put_artifact(connection, "TaskReceipt", payload)
        connection.execute(
            "INSERT INTO receipts VALUES(?,?,?,?,?,?,?)",
            (
                cid,
                grant.task_cid,
                grant.goal_cid,
                grant.subgoal_cid,
                grant.claim_cid,
                grant.fencing_token,
                json.dumps(payload, sort_keys=True),
            ),
        )
        terminal = "completed" if status == "succeeded" else "released"
        release_reason = (
            None
            if status == "succeeded"
            else f"receipt:{status}:{failure_class}"[:256]
        )
        connection.execute(
            "UPDATE leases SET state=?, release_reason=? WHERE task_cid=?",
            (terminal, release_reason, grant.task_cid),
        )
        return {
            "receipt_cid": cid,
            "goal_cid": grant.goal_cid,
            "subgoal_cid": grant.subgoal_cid,
            "receipt": payload,
        }

    @_coordinator_operation
    def receipt(
        self, grant: LeaseGrant, *, status: str, output: Mapping[str, Any] | None = None,
        failure_class: str = "none", started_at_ms: int | None = None, now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Publish a terminal receipt for a fenced execution.

        ``succeeded`` is the merge-success authority used by downstream
        dependency gates.  Callers must therefore emit it only after the task's
        outputs have merged and its required validation has passed.
        """

        now = self._clock_ms() if now_ms is None else int(now_ms)
        status = str(status)
        if status not in {"succeeded", "failed", "cancelled", "compensated"}:
            raise ValueError("invalid receipt status")
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                result = self._receipt_in_transaction(
                    conn,
                    row,
                    grant,
                    status=status,
                    output=output,
                    failure_class=failure_class,
                    started_at_ms=started_at_ms,
                    now=now,
                )
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def heartbeat_remote(
        self,
        dispatch: DistributedLaneDispatch,
        *,
        phase: str = "",
        progress_millionths: int | None = None,
        capacity_millionths: int = 0,
        ttl_ms: int = 15_000,
        requested_lease_ms: int | None = None,
        now_ms: int | None = None,
    ) -> DistributedLaneDispatch:
        """Heartbeat active remote ownership and return the current grant."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        if progress_millionths is not None and (
            isinstance(progress_millionths, bool)
            or not isinstance(progress_millionths, int)
            or not 0 <= progress_millionths <= 1_000_000
        ):
            raise ValueError("progress_millionths must be in [0, 1000000]")
        current = self.renew(
            dispatch.grant,
            requested_lease_ms=(
                dispatch.lease_duration_ms
                if requested_lease_ms is None
                else int(requested_lease_ms)
            ),
            now_ms=now,
        )
        detail: dict[str, Any] = {
            "distributed_dispatch_cid": dispatch.dispatch_cid,
            "input_artifact_cid": dispatch.input_artifact_cid,
            "capability_receipt_cid": dispatch.capability_receipt_cid,
            "environment_receipt_cid": dispatch.environment_receipt_cid,
        }
        if progress_millionths is not None:
            detail["progress_millionths"] = progress_millionths
        self.heartbeat(
            current,
            capacity_millionths=capacity_millionths,
            ttl_ms=ttl_ms,
            now_ms=now,
            active_phase=phase or "remote_execution",
            provider_id=dispatch.worker_id,
            detail=detail,
        )
        return replace(dispatch, grant=current)

    @_coordinator_operation
    def cancel_remote(
        self,
        dispatch: DistributedLaneDispatch,
        *,
        reason: str = "cancelled",
        now_ms: int | None = None,
    ) -> DistributedLaneDispatch:
        """Durably cancel a dispatch without allowing later success publication."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        cancellation_reason = str(reason or "cancelled")[:256]
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                stored = connection.execute(
                    "SELECT * FROM distributed_dispatches WHERE dispatch_cid=?",
                    (dispatch.dispatch_cid,),
                ).fetchone()
                if stored is None or str(stored["claim_cid"]) != dispatch.grant.claim_cid:
                    raise ValueError("unknown or foreign distributed dispatch")
                if stored["cancellation_cid"]:
                    connection.commit()
                    return replace(
                        dispatch, cancellation_cid=str(stored["cancellation_cid"])
                    )
                row = self._current(connection, dispatch.grant, now)
                payload = {
                    "schema": "ipfs_accelerate_py.agent_supervisor/distributed-cancellation@1",
                    "created_at_ms": now,
                    "dispatch_cid": dispatch.dispatch_cid,
                    "task_cid": dispatch.task_cid,
                    "claim_cid": dispatch.grant.claim_cid,
                    "logical_epoch": dispatch.logical_epoch,
                    "fencing_token": dispatch.fencing_token,
                    "reason": cancellation_reason,
                }
                cancellation_cid = self._put_artifact(
                    connection, "DistributedLaneCancellation", payload
                )
                resolution_cid = self._put_artifact(
                    connection,
                    "ClaimResolution",
                    self._resolution_payload(row, outcome="cancelled", now=now),
                )
                connection.execute(
                    """UPDATE leases
                       SET state='released', resolution_cid=?, release_reason=?
                       WHERE task_cid=? AND claim_cid=? AND fencing_token=?""",
                    (
                        resolution_cid,
                        f"remote_cancelled:{cancellation_reason}"[:256],
                        dispatch.task_cid,
                        dispatch.grant.claim_cid,
                        dispatch.fencing_token,
                    ),
                )
                connection.execute(
                    "UPDATE distributed_dispatches SET cancellation_cid=? WHERE dispatch_cid=?",
                    (cancellation_cid, dispatch.dispatch_cid),
                )
                connection.commit()
                return replace(dispatch, cancellation_cid=cancellation_cid)
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _quarantine_safe(value: Any, *, depth: int = 0) -> Any:
        if depth > 12:
            return "<depth-limit>"
        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            return {"invalid_float": repr(value)}
        if isinstance(value, Mapping):
            return {
                str(key): LeaseCoordinator._quarantine_safe(child, depth=depth + 1)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple, set)):
            return [
                LeaseCoordinator._quarantine_safe(child, depth=depth + 1)
                for child in list(value)[:256]
            ]
        return {"unsupported_type": type(value).__name__, "representation": repr(value)[:512]}

    def _quarantine_publication(
        self,
        connection: _DuckConnection,
        dispatch: DistributedLaneDispatch,
        raw_result: Any,
        *,
        reason: str,
        now: int,
        publication_id: str = "",
    ) -> dict[str, Any]:
        safe_result = self._quarantine_safe(raw_result)
        raw_digest = _content_digest({"result": safe_result})
        identity = publication_id or profile_g_cid(
            {
                "dispatch_cid": dispatch.dispatch_cid,
                "reason": reason,
                "raw_digest": raw_digest,
            }
        )
        payload = {
            "schema": DISTRIBUTED_QUARANTINE_SCHEMA,
            "created_at_ms": now,
            "publication_id": identity,
            "dispatch_cid": dispatch.dispatch_cid,
            "repository_id": dispatch.repository_id,
            "worker_id": dispatch.worker_id,
            "task_cid": dispatch.task_cid,
            "artifact_id": dispatch.input_artifact_cid,
            "claim_cid": dispatch.grant.claim_cid,
            "logical_epoch": dispatch.logical_epoch,
            "fencing_epoch": dispatch.logical_epoch,
            "fencing_token": dispatch.fencing_token,
            "cancelled": dispatch.cancelled,
            "reason": str(reason)[:256],
            "raw_result_digest": raw_digest,
            "raw_result": safe_result,
        }
        quarantine_cid = self._put_artifact(
            connection, "DistributedResultQuarantine", payload
        )
        disposition = {
            "accepted": False,
            "quarantined": True,
            "cancelled": dispatch.cancelled,
            "duplicate": False,
            "reason": str(reason)[:256],
            "publication_id": identity,
            "quarantine_cid": quarantine_cid,
            "distributed_publication": {
                key: value
                for key, value in payload.items()
                if key not in {"raw_result"}
            },
        }
        encoded = canonical_profile_g_bytes(disposition).decode("utf-8")
        previous = connection.execute(
            "SELECT payload_json FROM distributed_publications WHERE publication_id=?",
            (identity,),
        ).fetchone()
        if previous is None:
            connection.execute(
                "INSERT INTO distributed_publications VALUES(?,?,?,?,?,?,?)",
                (
                    identity,
                    dispatch.dispatch_cid,
                    dispatch.task_cid,
                    "quarantined",
                    str(reason)[:256],
                    now,
                    encoded,
                ),
            )
        return disposition

    @_coordinator_operation
    def publish_remote_result(
        self,
        dispatch: DistributedLaneDispatch,
        result: RemoteLaneResult | Mapping[str, Any],
        *,
        current_capability_receipt: WorkerCapabilityReceipt | Mapping[str, Any] | None = None,
        current_environment_receipt: WorkerEnvironmentReceipt | Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Accept one current fenced result or durably quarantine it.

        Replaying the same content-addressed publication is idempotent. Any
        malformed, foreign, expired, stolen, cancelled, or capability-drifted
        result remains inspectable but can never produce a terminal success
        receipt.
        """

        now = self._clock_ms() if now_ms is None else int(now_ms)
        raw_result = result.to_dict() if isinstance(result, RemoteLaneResult) else result
        try:
            normalized = (
                RemoteLaneResult.from_dict(result.to_dict())
                if isinstance(result, RemoteLaneResult)
                else RemoteLaneResult.from_dict(result)
            )
        except Exception as exc:
            with self._lock, self._connection:
                return self._quarantine_publication(
                    self._connection,
                    dispatch,
                    raw_result,
                    reason=f"malformed_result:{type(exc).__name__}",
                    now=now,
                )

        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                previous = connection.execute(
                    "SELECT payload_json FROM distributed_publications WHERE publication_id=?",
                    (normalized.publication_id,),
                ).fetchone()
                if previous is not None:
                    disposition = json.loads(str(previous["payload_json"]))
                    disposition["duplicate"] = True
                    connection.commit()
                    return disposition
                stored = connection.execute(
                    "SELECT * FROM distributed_dispatches WHERE dispatch_cid=?",
                    (dispatch.dispatch_cid,),
                ).fetchone()
                reason = ""
                if stored is None:
                    reason = "unknown_dispatch"
                elif (
                    str(stored["claim_cid"]) != dispatch.grant.claim_cid
                    or str(stored["worker_id"]) != dispatch.worker_id
                    or int(stored["fencing_token"]) != dispatch.fencing_token
                ):
                    reason = "foreign_dispatch"
                elif stored["cancellation_cid"] or dispatch.cancelled:
                    reason = "cancelled_dispatch"
                bindings = {
                    "repository_id": dispatch.repository_id,
                    "worker_id": dispatch.worker_id,
                    "task_cid": dispatch.task_cid,
                    "artifact_id": dispatch.input_artifact_cid,
                    "capability_receipt_id": dispatch.capability_receipt_cid,
                    "environment_receipt_id": dispatch.environment_receipt_cid,
                    "claim_cid": dispatch.grant.claim_cid,
                    "logical_epoch": dispatch.logical_epoch,
                    "fencing_token": dispatch.fencing_token,
                }
                for name, expected in bindings.items():
                    if getattr(normalized, name) != expected:
                        reason = reason or f"foreign_{name}"
                        break
                if current_capability_receipt is not None:
                    current_capability = self._capability_receipt(
                        current_capability_receipt
                    )
                    if current_capability.receipt_id != dispatch.capability_receipt_cid:
                        reason = reason or "capability_drift"
                if current_environment_receipt is not None:
                    current_environment = self._environment_receipt(
                        current_environment_receipt
                    )
                    if current_environment.receipt_id != dispatch.environment_receipt_cid:
                        reason = reason or "environment_drift"

                self._expire(connection, dispatch.task_cid, now)
                lease = connection.execute(
                    "SELECT * FROM leases WHERE task_cid=?", (dispatch.task_cid,)
                ).fetchone()
                if (
                    lease is None
                    or lease["state"] != "accepted"
                    or int(lease["expires_at_ms"]) <= now
                ):
                    reason = reason or "stale_or_expired_lease"
                elif (
                    str(lease["claim_cid"]) != dispatch.grant.claim_cid
                    or int(lease["logical_epoch"]) != dispatch.logical_epoch
                    or int(lease["fencing_token"]) != dispatch.fencing_token
                ):
                    reason = reason or "stale_fencing_epoch"

                capability_row = connection.execute(
                    "SELECT payload_json FROM worker_capability_receipts WHERE receipt_id=?",
                    (dispatch.capability_receipt_cid,),
                ).fetchone()
                environment_row = connection.execute(
                    "SELECT payload_json FROM worker_environment_receipts WHERE receipt_id=?",
                    (dispatch.environment_receipt_cid,),
                ).fetchone()
                if capability_row is None or environment_row is None:
                    reason = reason or "missing_worker_receipt"
                else:
                    try:
                        capability = WorkerCapabilityReceipt.from_dict(
                            json.loads(capability_row["payload_json"])
                        )
                        environment = WorkerEnvironmentReceipt.from_dict(
                            json.loads(environment_row["payload_json"])
                        )
                        capability.validate_at(now)
                        environment.validate_at(now)
                    except Exception:
                        reason = reason or "expired_or_malformed_worker_receipt"

                if reason:
                    disposition = self._quarantine_publication(
                        connection,
                        dispatch,
                        normalized.to_dict(),
                        reason=reason,
                        now=now,
                        publication_id=normalized.publication_id,
                    )
                    connection.commit()
                    return disposition

                assert lease is not None
                publication = normalized.to_dict()
                publication["schema"] = DISTRIBUTED_PUBLICATION_SCHEMA
                receipt: dict[str, Any] | None = None
                if normalized.status != "succeeded":
                    # Failure and cancellation are non-authoritative terminal
                    # outcomes. Successful publication is only a candidate and
                    # remains leased until finalize_remote_result observes the
                    # merge train's post-merge evidence gate.
                    receipt = self._receipt_in_transaction(
                        connection,
                        lease,
                        dispatch.grant,
                        status=normalized.status,
                        output=normalized.output,
                        failure_class=normalized.failure_class,
                        started_at_ms=None,
                        now=now,
                    )
                disposition = {
                    "accepted": normalized.status == "succeeded",
                    "quarantined": False,
                    "cancelled": normalized.cancelled,
                    "duplicate": False,
                    "finalized": False,
                    "reason": (
                        "cancelled"
                        if normalized.cancelled
                        else (
                            "published_pending_merge"
                            if normalized.status == "succeeded"
                            else f"accepted_{normalized.status}"
                        )
                    ),
                    "publication_id": normalized.publication_id,
                    "distributed_publication": publication,
                }
                if receipt is not None:
                    disposition["receipt"] = receipt
                connection.execute(
                    "INSERT INTO distributed_publications VALUES(?,?,?,?,?,?,?)",
                    (
                        normalized.publication_id,
                        dispatch.dispatch_cid,
                        dispatch.task_cid,
                        normalized.status,
                        disposition["reason"],
                        now,
                        canonical_profile_g_bytes(disposition).decode("utf-8"),
                    ),
                )
                connection.commit()
                return disposition
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def finalize_remote_result(
        self,
        dispatch: DistributedLaneDispatch,
        result_or_publication: RemoteLaneResult | Mapping[str, Any] | str,
        *,
        merge_result: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Create success authority only after merge and post-merge evidence pass."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        if isinstance(result_or_publication, RemoteLaneResult):
            publication_id = result_or_publication.publication_id
        elif isinstance(result_or_publication, str):
            publication_id = _required_text(result_or_publication, "publication_id")
        elif isinstance(result_or_publication, Mapping):
            publication_id = _required_text(
                result_or_publication.get("publication_id")
                or (
                    result_or_publication.get("distributed_publication", {}).get(
                        "publication_id"
                    )
                    if isinstance(
                        result_or_publication.get("distributed_publication"), Mapping
                    )
                    else ""
                ),
                "publication_id",
            )
        else:
            raise ValueError("result_or_publication must identify a remote publication")
        merge = _canonical_mapping(merge_result, "merge_result")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                stored = connection.execute(
                    "SELECT * FROM distributed_publications WHERE publication_id=?",
                    (publication_id,),
                ).fetchone()
                if stored is None:
                    raise ValueError("remote publication is not registered")
                disposition = json.loads(str(stored["payload_json"]))
                if disposition.get("finalized") is True:
                    disposition["duplicate"] = True
                    connection.commit()
                    return disposition
                if (
                    disposition.get("accepted") is not True
                    or disposition.get("quarantined") is True
                    or disposition.get("cancelled") is True
                    or str(stored["disposition"]) != "succeeded"
                ):
                    raise ValueError("only an accepted successful publication can finalize")
                publication = disposition.get("distributed_publication")
                if not isinstance(publication, Mapping):
                    raise ValueError("stored distributed publication is malformed")
                self._current(connection, dispatch.grant, now)
                if (
                    str(stored["dispatch_cid"]) != dispatch.dispatch_cid
                    or str(stored["task_cid"]) != dispatch.task_cid
                    or publication.get("claim_cid") != dispatch.grant.claim_cid
                    or publication.get("logical_epoch") != dispatch.logical_epoch
                    or publication.get("fencing_token") != dispatch.fencing_token
                ):
                    raise StaleFencingTokenError(
                        "publication is not bound to the active dispatch"
                    )
                merge_accepted = (
                    merge.get("merged") is True or merge.get("accepted") is True
                )
                expected_commit = str(publication.get("candidate_commit") or "")
                observed_candidate = str(
                    merge.get("candidate_commit")
                    or merge.get("source_commit")
                    or (
                        merge.get("distributed_publication", {}).get("candidate_commit")
                        if isinstance(merge.get("distributed_publication"), Mapping)
                        else ""
                    )
                    or ""
                )
                evidence = (
                    merge.get("post_merge_evidence")
                    or merge.get("post_merge_evidence_receipt")
                    or (
                        merge.get("validation", {}).get("post_merge_evidence_receipt")
                        if isinstance(merge.get("validation"), Mapping)
                        else None
                    )
                )
                evidence_passed = (
                    merge.get("post_merge_evidence_passed") is True
                    or (
                        isinstance(evidence, Mapping)
                        and (
                            evidence.get("passed") is True
                            or evidence.get("merge_authoritative") is True
                            or evidence.get("allowed") is True
                        )
                    )
                )
                if not merge_accepted:
                    raise ValueError("merge train did not accept the candidate")
                if not observed_candidate or observed_candidate != expected_commit:
                    raise ValueError("merge receipt is not bound to the candidate commit")
                if not evidence_passed:
                    raise ValueError("post-merge evidence gate did not pass")
                evidence_candidate = (
                    str(
                        evidence.get("candidate_commit")
                        or evidence.get("candidate_tree_id")
                        or ""
                    )
                    if isinstance(evidence, Mapping)
                    else ""
                )
                merge_candidate_tree = str(
                    merge.get("candidate_tree_id")
                    or merge.get("repository_tree")
                    or merge.get("merged_tree_id")
                    or ""
                )
                if isinstance(evidence, Mapping) and evidence_candidate:
                    bound_values = {
                        expected_commit,
                        merge_candidate_tree,
                    }
                    if evidence_candidate not in bound_values:
                        raise ValueError(
                            "post-merge evidence is bound to a foreign candidate"
                        )
                receipt_output = dict(publication.get("output") or {})
                receipt_output.update(
                    {
                        "distributed_publication": dict(publication),
                        "merge_result": merge,
                    }
                )
                lease = connection.execute(
                    "SELECT * FROM leases WHERE task_cid=?", (dispatch.task_cid,)
                ).fetchone()
                assert lease is not None
                receipt = self._receipt_in_transaction(
                    connection,
                    lease,
                    dispatch.grant,
                    status="succeeded",
                    output=receipt_output,
                    failure_class="none",
                    started_at_ms=None,
                    now=now,
                )
                disposition.update(
                    {
                        "finalized": True,
                        "duplicate": False,
                        "reason": "accepted_after_merge_evidence",
                        "merge_result": merge,
                        "receipt": receipt,
                    }
                )
                connection.execute(
                    """UPDATE distributed_publications
                       SET disposition='finalized', reason=?, payload_json=?
                       WHERE publication_id=?""",
                    (
                        disposition["reason"],
                        canonical_profile_g_bytes(disposition).decode("utf-8"),
                        publication_id,
                    ),
                )
                connection.commit()
                return disposition
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def list_distributed_publications(
        self,
        task_cid: str | None = None,
        *,
        disposition: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return durable accepted/quarantined publication decisions."""

        clauses: list[str] = []
        parameters: list[Any] = []
        if task_cid is not None:
            clauses.append("task_cid=?")
            parameters.append(str(task_cid))
        if disposition is not None:
            clauses.append("disposition=?")
            parameters.append(str(disposition))
        query = "SELECT payload_json FROM distributed_publications"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at_ms, publication_id"
        return [
            json.loads(str(row["payload_json"]))
            for row in self._connection.execute(query, parameters).fetchall()
        ]

    @staticmethod
    def _prune_heartbeat_history(
        connection: _DuckConnection,
        grant: LeaseGrant,
    ) -> None:
        stale_rows = connection.execute(
            """SELECT heartbeat_cid
               FROM (
                 SELECT heartbeat_cid,
                        row_number() OVER (
                          ORDER BY observed_at_ms DESC, heartbeat_cid DESC
                        ) AS history_rank
                 FROM heartbeats
                 WHERE task_cid=? AND fencing_token=?
               )
               WHERE history_rank>?""",
            (
                grant.task_cid,
                grant.fencing_token,
                MAX_PERSISTED_HEARTBEATS_PER_LEASE,
            ),
        ).fetchall()
        stale_cids = [(str(row[0]),) for row in stale_rows]
        if not stale_cids:
            return
        connection.executemany(
            "DELETE FROM heartbeats WHERE heartbeat_cid=?",
            stale_cids,
        )
        connection.executemany(
            "DELETE FROM artifacts "
            "WHERE cid=? AND kind='DaemonHeartbeat'",
            stale_cids,
        )

    @staticmethod
    def _heartbeat_count(connection: _DuckConnection, grant: LeaseGrant) -> int:
        return int(connection.execute("SELECT COUNT(*) FROM heartbeats WHERE task_cid=? AND fencing_token=?", (grant.task_cid, grant.fencing_token)).fetchone()[0])

    @_coordinator_operation
    def active_lease(self, task_cid: str, *, now_ms: int | None = None) -> LeaseGrant | None:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock, self._connection:
            self._expire(self._connection, task_cid, now)
            row = self._connection.execute("SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms>?", (task_cid, now)).fetchone()
            return self._grant(row) if row is not None else None

    @_coordinator_operation
    def list_receipts(self, task_cid: str) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM receipts WHERE task_cid=? ORDER BY rowid", (task_cid,)).fetchall()
        return [{"receipt_cid": row["receipt_cid"], "goal_cid": row["goal_cid"], "subgoal_cid": row["subgoal_cid"], "receipt": json.loads(row["payload_json"])} for row in rows]

    @_coordinator_operation
    def get_artifact(self, cid: str) -> dict[str, Any] | None:
        """Return a stored coordination artifact by CID."""

        row = self._connection.execute("SELECT payload_json FROM artifacts WHERE cid=?", (cid,)).fetchone()
        return json.loads(row[0]) if row is not None else None


_LEGACY_COORDINATION_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("artifacts", ("cid", "kind", "payload_json", "created_at_ms")),
    (
        "tasks",
        (
            "task_cid",
            "goal_cid",
            "subgoal_cid",
            "task_id",
            "bundle_json",
            "registered_at_ms",
            "updated_at_ms",
        ),
    ),
    ("task_aliases", ("alias_task_cid", "task_cid")),
    (
        "task_dependencies",
        ("task_cid", "dependency_task_cid", "provenance_json"),
    ),
    (
        "task_dependency_repairs",
        ("task_cid", "repair_index", "payload_json"),
    ),
    (
        "task_dependency_repair_state",
        ("task_cid", "source_count", "stored_count"),
    ),
    (
        "leases",
        (
            "task_cid",
            "claim_cid",
            "resolution_cid",
            "claimant_did",
            "logical_epoch",
            "fencing_token",
            "expires_at_ms",
            "attempt",
            "state",
            "started_at_ms",
            "release_reason",
            "retry_not_before_ms",
        ),
    ),
    ("token_history", ("task_cid", "fencing_token")),
    (
        "heartbeats",
        (
            "heartbeat_cid",
            "task_cid",
            "claimant_did",
            "fencing_token",
            "observed_at_ms",
            "expires_at_ms",
            "capacity_millionths",
            "payload_json",
        ),
    ),
    (
        "receipts",
        (
            "receipt_cid",
            "task_cid",
            "goal_cid",
            "subgoal_cid",
            "claim_cid",
            "fencing_token",
            "payload_json",
        ),
    ),
)


def migrate_sqlite_coordination_store(
    source_path: str | Path,
    target_path: str | Path,
    *,
    replace: bool = False,
    batch_size: int = 512,
    current_task_cids: Iterable[str] | None = None,
    heartbeat_history_per_lease: int = 8,
    preserve_all_history: bool = False,
) -> dict[str, Any]:
    """Atomically migrate a bounded legacy SQLite lease store into DuckDB.

    Migration is intentionally explicit. A live SQLite file is never opened as
    DuckDB, and the source remains untouched for rollback or audit. By default,
    the latest registration batch, its dependency closure, authoritative lease
    and receipt rows, and bounded heartbeat history are retained. The legacy
    source remains the cold audit archive.
    """

    source = Path(source_path)
    target = Path(target_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as stream:
        if stream.read(16) != b"SQLite format 3\0":
            raise ValueError(f"not a SQLite coordination store: {source}")
    if target.exists() and not replace:
        raise FileExistsError(target)
    limit = int(batch_size)
    if not 1 <= limit <= 10_000:
        raise ValueError("batch_size must be in [1, 10000]")
    heartbeat_limit = int(heartbeat_history_per_lease)
    if not 0 <= heartbeat_limit <= 10_000:
        raise ValueError("heartbeat_history_per_lease must be in [0, 10000]")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.migration-{threading.get_ident()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    temporary_lock = temporary.with_name(f".{temporary.name}.lock")
    counts: dict[str, int] = {}
    source_connection = sqlite3.connect(str(source))
    source_connection.row_factory = sqlite3.Row

    def table_exists(table: str) -> bool:
        return (
            source_connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            is not None
        )

    def chunks(values: Iterable[str], size: int = 400) -> Iterator[list[str]]:
        items = sorted({str(value) for value in values if str(value)})
        for offset in range(0, len(items), size):
            yield items[offset : offset + size]

    def rows_for_task_cids(
        table: str,
        columns: tuple[str, ...],
        task_ids: set[str],
    ) -> list[tuple[Any, ...]]:
        if not task_ids or not table_exists(table):
            return []
        selected = ", ".join(columns)
        rows: list[tuple[Any, ...]] = []
        for batch in chunks(task_ids):
            placeholders = ", ".join("?" for _ in batch)
            rows.extend(
                tuple(row)
                for row in source_connection.execute(
                    f"SELECT {selected} FROM {table} "
                    f"WHERE task_cid IN ({placeholders})",
                    batch,
                ).fetchall()
            )
        return rows

    def compact_historical_bundle(raw: str) -> str:
        if len(raw) <= 1_000_000:
            return raw
        try:
            bundle = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return json.dumps(
                {
                    "schema": "ipfs_accelerate_py.agent_supervisor.legacy-bundle-tombstone@1",
                    "source_bytes": len(raw),
                },
                sort_keys=True,
            )
        tasks = []
        for item in bundle.get("tasks", []) if isinstance(bundle, Mapping) else []:
            if not isinstance(item, Mapping):
                continue
            tasks.append(
                {
                    key: item[key]
                    for key in (
                        "task_id",
                        "task_cid",
                        "canonical_task_cid",
                        "status",
                        "depends_on",
                    )
                    if item.get(key) not in (None, "", [], {})
                }
            )
        return json.dumps(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.legacy-bundle-tombstone@1",
                "bundle_key": bundle.get("bundle_key")
                if isinstance(bundle, Mapping)
                else "",
                "source_todo": bundle.get("source_todo")
                if isinstance(bundle, Mapping)
                else "",
                "tasks": tasks,
                "source_bytes": len(raw),
            },
            sort_keys=True,
        )

    try:
        source_connection.execute("BEGIN")
        task_columns = dict(_LEGACY_COORDINATION_COLUMNS)["tasks"]
        retained_task_cids = {
            str(task_cid)
            for task_cid in (current_task_cids or ())
            if str(task_cid)
        }
        if table_exists("tasks"):
            if preserve_all_history:
                retained_task_cids.update(
                    str(row[0])
                    for row in source_connection.execute(
                        "SELECT task_cid FROM tasks"
                    ).fetchall()
                )
            elif not retained_task_cids:
                latest = source_connection.execute(
                    "SELECT max(updated_at_ms) FROM tasks"
                ).fetchone()[0]
                if latest is not None:
                    retained_task_cids.update(
                        str(row[0])
                        for row in source_connection.execute(
                            "SELECT task_cid FROM tasks WHERE updated_at_ms>=?",
                            (max(0, int(latest) - 60_000),),
                        ).fetchall()
                    )
        if table_exists("leases"):
            retained_task_cids.update(
                str(row[0])
                for row in source_connection.execute(
                    "SELECT task_cid FROM leases WHERE state='accepted'"
                ).fetchall()
            )

        dependency_aliases: set[str] = set()
        frontier = set(retained_task_cids)
        while frontier and table_exists("task_dependencies"):
            discovered_aliases: set[str] = set()
            for batch in chunks(frontier):
                placeholders = ", ".join("?" for _ in batch)
                discovered_aliases.update(
                    str(row[0])
                    for row in source_connection.execute(
                        "SELECT dependency_task_cid FROM task_dependencies "
                        f"WHERE task_cid IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
            dependency_aliases.update(discovered_aliases)
            resolved: set[str] = set()
            if discovered_aliases and table_exists("task_aliases"):
                for batch in chunks(discovered_aliases):
                    placeholders = ", ".join("?" for _ in batch)
                    resolved.update(
                        str(row[0])
                        for row in source_connection.execute(
                            "SELECT task_cid FROM task_aliases "
                            f"WHERE alias_task_cid IN ({placeholders})",
                            batch,
                        ).fetchall()
                    )
            if discovered_aliases and table_exists("tasks"):
                for batch in chunks(discovered_aliases):
                    placeholders = ", ".join("?" for _ in batch)
                    resolved.update(
                        str(row[0])
                        for row in source_connection.execute(
                            "SELECT task_cid FROM tasks "
                            f"WHERE task_cid IN ({placeholders})",
                            batch,
                        ).fetchall()
                    )
            frontier = resolved - retained_task_cids
            retained_task_cids.update(resolved)

        table_rows: dict[str, list[tuple[Any, ...]]] = {}
        raw_task_rows = rows_for_task_cids(
            "tasks", task_columns, retained_task_cids
        )
        seed_task_cids = {
            str(task_cid)
            for task_cid in (current_task_cids or retained_task_cids)
            if str(task_cid)
        }
        task_rows: list[tuple[Any, ...]] = []
        artifact_cids: set[str] = set()
        required_aliases = set(retained_task_cids) | dependency_aliases
        for row in raw_task_rows:
            values = list(row)
            task_cid = str(values[0])
            artifact_cids.update((str(values[1]), str(values[2])))
            if task_cid not in seed_task_cids:
                values[4] = compact_historical_bundle(str(values[4]))
            try:
                bundle = json.loads(str(values[4]))
            except (TypeError, ValueError, json.JSONDecodeError):
                bundle = {}
            profile = (
                bundle.get("profile_g")
                if isinstance(bundle, Mapping)
                and isinstance(bundle.get("profile_g"), Mapping)
                else {}
            )
            for key in (
                "goal_cid",
                "subgoal_cid",
                "plan_branch_cid",
                "selection_cid",
                "task_spec_cid",
            ):
                if profile.get(key):
                    artifact_cids.add(str(profile[key]))
                    required_aliases.add(str(profile[key]))
            artifacts = profile.get("artifacts")
            if isinstance(artifacts, Mapping):
                artifact_cids.update(str(cid) for cid in artifacts)
            for item in bundle.get("tasks", []) if isinstance(bundle, Mapping) else []:
                if not isinstance(item, Mapping):
                    continue
                for key in ("canonical_task_cid", "task_cid"):
                    if item.get(key):
                        required_aliases.add(str(item[key]))
            task_rows.append(tuple(values))
        table_rows["tasks"] = task_rows

        alias_columns = dict(_LEGACY_COORDINATION_COLUMNS)["task_aliases"]
        aliases = rows_for_task_cids(
            "task_aliases", alias_columns, retained_task_cids
        )
        if not preserve_all_history:
            aliases = [
                row
                for row in aliases
                if str(row[0]) in required_aliases or str(row[0]) == str(row[1])
            ]
        table_rows["task_aliases"] = aliases
        artifact_cids.update(str(row[0]) for row in aliases)

        for table in (
            "task_dependencies",
            "task_dependency_repairs",
            "task_dependency_repair_state",
            "leases",
            "token_history",
            "receipts",
        ):
            columns = dict(_LEGACY_COORDINATION_COLUMNS)[table]
            table_rows[table] = rows_for_task_cids(
                table, columns, retained_task_cids
            )
        for row in table_rows["leases"]:
            artifact_cids.update((str(row[1]), str(row[2])))
        for row in table_rows["receipts"]:
            artifact_cids.update((str(row[0]), str(row[4])))

        heartbeat_columns = dict(_LEGACY_COORDINATION_COLUMNS)["heartbeats"]
        heartbeats: list[tuple[Any, ...]] = []
        if retained_task_cids and table_exists("heartbeats"):
            for batch in chunks(retained_task_cids):
                placeholders = ", ".join("?" for _ in batch)
                selected = ", ".join(f"h.{column}" for column in heartbeat_columns)
                query = (
                    f"SELECT {selected} FROM ("
                    "SELECT h.*, row_number() OVER ("
                    "PARTITION BY h.task_cid, h.fencing_token "
                    "ORDER BY h.observed_at_ms DESC, h.heartbeat_cid DESC"
                    ") AS history_rank FROM heartbeats h "
                    f"WHERE h.task_cid IN ({placeholders})"
                    ") h LEFT JOIN leases l ON l.task_cid=h.task_cid "
                    "WHERE h.history_rank<=? OR "
                    "(l.state='accepted' AND l.fencing_token=h.fencing_token)"
                )
                heartbeats.extend(
                    tuple(row)
                    for row in source_connection.execute(
                        query,
                        [*batch, heartbeat_limit],
                    ).fetchall()
                )
        table_rows["heartbeats"] = heartbeats
        artifact_cids.update(str(row[0]) for row in heartbeats)

        artifact_columns = dict(_LEGACY_COORDINATION_COLUMNS)["artifacts"]
        artifacts: list[tuple[Any, ...]] = []
        artifact_row_count = 0
        if table_exists("artifacts"):
            artifact_row_count = int(
                source_connection.execute(
                    "SELECT count(*) FROM artifacts"
                ).fetchone()[0]
            )
        retain_all_artifacts = preserve_all_history or (
            artifact_row_count <= SMALL_STORE_FULL_ARTIFACT_LIMIT
        )
        if retain_all_artifacts and artifact_row_count:
            artifacts = [
                tuple(row)
                for row in source_connection.execute(
                    "SELECT cid, kind, payload_json, created_at_ms FROM artifacts"
                ).fetchall()
            ]
        elif artifact_cids and artifact_row_count:
            for batch in chunks(artifact_cids):
                placeholders = ", ".join("?" for _ in batch)
                artifacts.extend(
                    tuple(row)
                    for row in source_connection.execute(
                        "SELECT cid, kind, payload_json, created_at_ms "
                        f"FROM artifacts WHERE cid IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
        table_rows["artifacts"] = artifacts

        with LeaseCoordinator(temporary) as coordinator:
            with coordinator._database_operation():
                connection = coordinator._connection
                assert connection is not None
                connection.execute("BEGIN TRANSACTION")
                try:
                    for table, columns in _LEGACY_COORDINATION_COLUMNS:
                        selected = ", ".join(columns)
                        placeholders = ", ".join("?" for _ in columns)
                        copied = 0
                        rows = table_rows.get(table, [])
                        for offset in range(0, len(rows), limit):
                            batch = rows[offset : offset + limit]
                            connection.executemany(
                                f"INSERT OR REPLACE INTO {table} "
                                f"({selected}) VALUES ({placeholders})",
                                batch,
                            )
                            copied += len(batch)
                        counts[table] = copied
                    connection.execute(
                        "INSERT OR REPLACE INTO coordination_metadata VALUES(?,?)",
                        (
                            "legacy_sqlite_migration",
                            json.dumps(
                                {
                                    "schema": COORDINATION_STORE_SCHEMA,
                                    "source_path": str(source.resolve()),
                                    "source_bytes": source.stat().st_size,
                                    "migrated_at_ms": int(time.time() * 1000),
                                    "preserve_all_history": bool(
                                        preserve_all_history
                                    ),
                                    "retained_task_count": len(
                                        retained_task_cids
                                    ),
                                    "heartbeat_history_per_lease": heartbeat_limit,
                                    "retained_all_artifacts": retain_all_artifacts,
                                    "row_counts": counts,
                                },
                                sort_keys=True,
                            ),
                        ),
                    )
                    connection.commit()
                    connection.execute("CHECKPOINT")
                except Exception:
                    connection.rollback()
                    raise
        source_connection.rollback()
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        source_connection.close()
        temporary_lock.unlink(missing_ok=True)

    return {
        "schema": COORDINATION_STORE_SCHEMA,
        "source_path": str(source),
        "target_path": str(target),
        "source_bytes": source.stat().st_size,
        "target_bytes": target.stat().st_size,
        "retained_task_count": len(retained_task_cids),
        "heartbeat_history_per_lease": heartbeat_limit,
        "preserve_all_history": bool(preserve_all_history),
        "retained_all_artifacts": retain_all_artifacts,
        "row_counts": counts,
    }


class DistributedSingleFlightError(RuntimeError):
    """Base failure for the durable semantic-key single-flight protocol."""


class DistributedSingleFlightTimeout(DistributedSingleFlightError, TimeoutError):
    """One member's deadline elapsed while another member retained ownership."""


class DistributedSingleFlightCancelled(DistributedSingleFlightError):
    """One member stopped waiting without cancelling the shared computation."""


class DistributedSingleFlightExecutionError(DistributedSingleFlightError):
    """A leader published one bounded, fail-closed execution outcome."""

    def __init__(
        self,
        reason_code: str,
        *,
        outcome: "SingleFlightOutcome | None" = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.outcome = outcome


class StaleSingleFlightLeaseError(DistributedSingleFlightError):
    """A released, expired, superseded, or foreign lease attempted mutation."""


def _single_flight_identity(value: Any) -> tuple[str, str]:
    """Return a stable ``(key_id, namespace)`` without importing cache code."""

    if isinstance(value, str):
        key_id = value.strip()
        namespace = ""
    elif isinstance(value, Mapping):
        key_id = str(value.get("key_id") or value.get("semantic_key") or "").strip()
        namespace = str(value.get("namespace") or "").strip()
    else:
        key_id = str(getattr(value, "key_id", "") or "").strip()
        kind = getattr(value, "namespace", "")
        namespace = str(getattr(kind, "value", kind) or "").strip()
    if not key_id:
        raise ValueError("single-flight semantic key identity is required")
    if len(key_id.encode("utf-8")) > 4_096:
        raise ValueError("single-flight semantic key identity is too large")
    if len(namespace.encode("utf-8")) > 256:
        raise ValueError("single-flight namespace is too large")
    return key_id, namespace


def _single_flight_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "single-flight outcomes must contain canonical JSON values"
        ) from exc


def _single_flight_cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    is_set = getattr(cancel_event, "is_set", None)
    if not callable(is_set):
        raise ValueError("cancel_event must provide is_set()")
    return bool(is_set())


@dataclass(frozen=True)
class SingleFlightLeaseGrant:
    """One fenced generation, projected differently to its owner and followers."""

    key_id: str
    namespace: str
    owner_id: str
    lease_id: str
    fencing_token: int
    acquired_at_ms: int
    heartbeat_at_ms: int
    expires_at_ms: int
    acquired: bool
    completed: bool = False

    @property
    def is_owner(self) -> bool:
        return self.acquired

    @property
    def is_leader(self) -> bool:
        return self.acquired

    def to_dict(self) -> dict[str, Any]:
        # A follower never receives the owner's unguessable publication token.
        return asdict(self)


@dataclass(frozen=True)
class SingleFlightAttestation:
    """Coordinator-authenticated binding for one bounded flight outcome."""

    key_id: str
    namespace: str
    owner_id: str
    fencing_token: int
    outcome_digest: str
    attestation_id: str
    schema: str = SINGLE_FLIGHT_ATTESTATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SingleFlightOutcome:
    """The sole bounded outcome followers may consume for one fence."""

    key_id: str
    namespace: str
    owner_id: str
    fencing_token: int
    status: str
    value: Any
    created_at_ms: int
    expires_at_ms: int
    outcome_digest: str
    attestation: SingleFlightAttestation
    schema: str = SINGLE_FLIGHT_OUTCOME_SCHEMA

    @property
    def successful(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "key_id": self.key_id,
            "namespace": self.namespace,
            "owner_id": self.owner_id,
            "fencing_token": self.fencing_token,
            "status": self.status,
            "value": self.value,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "outcome_digest": self.outcome_digest,
            "attestation": self.attestation.to_dict(),
        }


@dataclass(frozen=True)
class DistributedSingleFlightResult:
    """Member projection of a verified owner-attested outcome."""

    outcome: SingleFlightOutcome
    owner: bool

    @property
    def shared(self) -> bool:
        return not self.owner

    @property
    def value(self) -> Any:
        return self.outcome.value

    @property
    def fencing_token(self) -> int:
        return self.outcome.fencing_token

    @property
    def attestation(self) -> SingleFlightAttestation:
        return self.outcome.attestation


class DistributedSingleFlightCoordinator:
    """SQLite-backed semantic-key leases for threads, processes, and shared hosts.

    A shared filesystem path is sufficient for optional multi-host operation;
    callers may instead pass the same ``attestation_secret`` to coordinators
    backed by a replicated/transport adapter in the future.  SQLite
    ``BEGIN IMMEDIATE`` selects exactly one owner for each generation.  The
    unguessable lease ID plus monotonically increasing fence prevents stale or
    foreign publishers, and an HMAC binds the only follower-visible outcome.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        lease_seconds: float = DEFAULT_SINGLE_FLIGHT_LEASE_SECONDS,
        outcome_ttl_seconds: float = DEFAULT_SINGLE_FLIGHT_OUTCOME_TTL_SECONDS,
        poll_interval_seconds: float = DEFAULT_SINGLE_FLIGHT_POLL_SECONDS,
        max_outcome_bytes: int = DEFAULT_SINGLE_FLIGHT_MAX_OUTCOME_BYTES,
        clock_ms: Callable[[], int] | None = None,
        attestation_secret: bytes | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("lease_seconds", lease_seconds),
            ("outcome_ttl_seconds", outcome_ttl_seconds),
            ("poll_interval_seconds", poll_interval_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(max_outcome_bytes, bool)
            or not isinstance(max_outcome_bytes, int)
            or max_outcome_bytes < 1_024
        ):
            raise ValueError("max_outcome_bytes must be at least 1024")
        self.lease_seconds = float(lease_seconds)
        self.outcome_ttl_seconds = float(outcome_ttl_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.max_outcome_bytes = max_outcome_bytes
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._thread_lock = threading.RLock()
        self._secret_path = self.path.with_name(f".{self.path.name}.attestation-key")
        self._secret = self._load_secret(attestation_secret)
        self._init_store()

    def _load_secret(self, supplied: bytes | None) -> bytes:
        if supplied is not None:
            if not isinstance(supplied, bytes) or len(supplied) < 16:
                raise ValueError("attestation_secret must contain at least 16 bytes")
            return supplied
        try:
            descriptor = os.open(
                self._secret_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            try:
                os.write(descriptor, secrets.token_bytes(32))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        secret = self._secret_path.read_bytes()
        if len(secret) < 16 or len(secret) > 4_096:
            raise ValueError("single-flight attestation key is invalid")
        return secret

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_store(self) -> None:
        with self._thread_lock:
            connection = self._connect()
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS single_flight_fences(
                      key_id TEXT PRIMARY KEY,
                      namespace TEXT NOT NULL,
                      fencing_token INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS single_flight_leases(
                      key_id TEXT PRIMARY KEY,
                      namespace TEXT NOT NULL,
                      owner_id TEXT NOT NULL,
                      lease_id TEXT NOT NULL,
                      fencing_token INTEGER NOT NULL,
                      acquired_at_ms INTEGER NOT NULL,
                      heartbeat_at_ms INTEGER NOT NULL,
                      expires_at_ms INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS single_flight_outcomes(
                      key_id TEXT PRIMARY KEY,
                      namespace TEXT NOT NULL,
                      owner_id TEXT NOT NULL,
                      fencing_token INTEGER NOT NULL,
                      status TEXT NOT NULL,
                      outcome_json TEXT NOT NULL,
                      outcome_digest TEXT NOT NULL,
                      attestation_id TEXT NOT NULL,
                      created_at_ms INTEGER NOT NULL,
                      expires_at_ms INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS single_flight_metadata(
                      metadata_key TEXT PRIMARY KEY,
                      value_json TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT OR REPLACE INTO single_flight_metadata VALUES(?,?)",
                    (
                        "store",
                        json.dumps(
                            {"schema": SINGLE_FLIGHT_STORE_SCHEMA},
                            sort_keys=True,
                        ),
                    ),
                )
            finally:
                connection.close()

    @staticmethod
    def _owner_id(owner_id: str | None) -> str:
        owner = (
            owner_id
            or f"pid:{os.getpid()}:thread:{threading.get_ident()}"
        ).strip()
        if not owner or len(owner.encode("utf-8")) > 1_024:
            raise ValueError("owner_id must be nonempty and bounded")
        return owner

    @staticmethod
    def _lease_ms(lease_seconds: float) -> int:
        return max(1, int(lease_seconds * 1000))

    def _attestation_id(
        self,
        *,
        key_id: str,
        namespace: str,
        owner_id: str,
        fencing_token: int,
        outcome_digest: str,
    ) -> str:
        content = _single_flight_json_bytes(
            {
                "schema": SINGLE_FLIGHT_ATTESTATION_SCHEMA,
                "key_id": key_id,
                "namespace": namespace,
                "owner_id": owner_id,
                "fencing_token": fencing_token,
                "outcome_digest": outcome_digest,
            }
        )
        digest = hmac.new(self._secret, content, hashlib.sha256).hexdigest()
        return f"single-flight-attestation:hmac-sha256:{digest}"

    def acquire(
        self,
        key: Any,
        *,
        owner_id: str | None = None,
        lease_seconds: float | None = None,
    ) -> SingleFlightLeaseGrant:
        key_id, namespace = _single_flight_identity(key)
        owner = self._owner_id(owner_id)
        duration = self.lease_seconds if lease_seconds is None else lease_seconds
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise ValueError("lease_seconds must be positive")
        now = self._clock_ms()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            fence_row = connection.execute(
                "SELECT namespace, fencing_token FROM single_flight_fences WHERE key_id=?",
                (key_id,),
            ).fetchone()
            if (
                fence_row is not None
                and namespace
                and str(fence_row["namespace"])
                and str(fence_row["namespace"]) != namespace
            ):
                raise DistributedSingleFlightError(
                    "semantic key is already bound to another namespace"
                )
            outcome = connection.execute(
                """
                SELECT * FROM single_flight_outcomes
                WHERE key_id=? AND expires_at_ms>?
                """,
                (key_id, now),
            ).fetchone()
            if outcome is not None:
                connection.commit()
                return SingleFlightLeaseGrant(
                    key_id=key_id,
                    namespace=str(outcome["namespace"]),
                    owner_id=str(outcome["owner_id"]),
                    lease_id="",
                    fencing_token=int(outcome["fencing_token"]),
                    acquired_at_ms=int(outcome["created_at_ms"]),
                    heartbeat_at_ms=int(outcome["created_at_ms"]),
                    expires_at_ms=int(outcome["expires_at_ms"]),
                    acquired=False,
                    completed=True,
                )
            lease = connection.execute(
                "SELECT * FROM single_flight_leases WHERE key_id=?",
                (key_id,),
            ).fetchone()
            if lease is not None and int(lease["expires_at_ms"]) > now:
                connection.commit()
                return SingleFlightLeaseGrant(
                    key_id=key_id,
                    namespace=str(lease["namespace"]),
                    owner_id=str(lease["owner_id"]),
                    lease_id="",
                    fencing_token=int(lease["fencing_token"]),
                    acquired_at_ms=int(lease["acquired_at_ms"]),
                    heartbeat_at_ms=int(lease["heartbeat_at_ms"]),
                    expires_at_ms=int(lease["expires_at_ms"]),
                    acquired=False,
                )
            prior_fence = int(fence_row["fencing_token"]) if fence_row else 0
            if lease is not None:
                prior_fence = max(prior_fence, int(lease["fencing_token"]))
            token = prior_fence + 1
            bound_namespace = (
                namespace
                or (str(fence_row["namespace"]) if fence_row is not None else "")
            )
            lease_id = secrets.token_hex(32)
            expires = now + self._lease_ms(float(duration))
            connection.execute(
                """
                INSERT INTO single_flight_fences VALUES(?,?,?)
                ON CONFLICT(key_id) DO UPDATE SET
                  namespace=excluded.namespace,
                  fencing_token=excluded.fencing_token
                """,
                (key_id, bound_namespace, token),
            )
            connection.execute(
                """
                INSERT INTO single_flight_leases VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(key_id) DO UPDATE SET
                  namespace=excluded.namespace,
                  owner_id=excluded.owner_id,
                  lease_id=excluded.lease_id,
                  fencing_token=excluded.fencing_token,
                  acquired_at_ms=excluded.acquired_at_ms,
                  heartbeat_at_ms=excluded.heartbeat_at_ms,
                  expires_at_ms=excluded.expires_at_ms
                """,
                (
                    key_id,
                    bound_namespace,
                    owner,
                    lease_id,
                    token,
                    now,
                    now,
                    expires,
                ),
            )
            connection.execute(
                "DELETE FROM single_flight_outcomes WHERE key_id=?",
                (key_id,),
            )
            connection.commit()
            return SingleFlightLeaseGrant(
                key_id=key_id,
                namespace=bound_namespace,
                owner_id=owner,
                lease_id=lease_id,
                fencing_token=token,
                acquired_at_ms=now,
                heartbeat_at_ms=now,
                expires_at_ms=expires,
                acquired=True,
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    acquire_lease = acquire

    def heartbeat(
        self,
        grant: SingleFlightLeaseGrant,
        *,
        lease_seconds: float | None = None,
    ) -> SingleFlightLeaseGrant:
        if not grant.acquired or not grant.lease_id:
            raise StaleSingleFlightLeaseError(
                "only the current single-flight owner may heartbeat"
            )
        duration = self.lease_seconds if lease_seconds is None else lease_seconds
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise ValueError("lease_seconds must be positive")
        now = self._clock_ms()
        expires = now + self._lease_ms(float(duration))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE single_flight_leases
                SET heartbeat_at_ms=?, expires_at_ms=?
                WHERE key_id=? AND namespace=? AND owner_id=? AND lease_id=?
                  AND fencing_token=? AND expires_at_ms>?
                """,
                (
                    now,
                    expires,
                    grant.key_id,
                    grant.namespace,
                    grant.owner_id,
                    grant.lease_id,
                    grant.fencing_token,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise StaleSingleFlightLeaseError(
                    "single-flight lease is expired or fenced"
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return SingleFlightLeaseGrant(
            **{
                **grant.to_dict(),
                "heartbeat_at_ms": now,
                "expires_at_ms": expires,
            }
        )

    renew = heartbeat
    renew_lease = heartbeat

    def release(self, grant: SingleFlightLeaseGrant) -> bool:
        if not grant.acquired or not grant.lease_id:
            return False
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                DELETE FROM single_flight_leases
                WHERE key_id=? AND namespace=? AND owner_id=? AND lease_id=?
                  AND fencing_token=?
                """,
                (
                    grant.key_id,
                    grant.namespace,
                    grant.owner_id,
                    grant.lease_id,
                    grant.fencing_token,
                ),
            )
            return cursor.rowcount == 1
        finally:
            connection.close()

    release_lease = release

    def publish(
        self,
        grant: SingleFlightLeaseGrant,
        value: Any,
        *,
        status: str = "ok",
        outcome_ttl_seconds: float | None = None,
    ) -> SingleFlightOutcome:
        if not grant.acquired or not grant.lease_id:
            raise StaleSingleFlightLeaseError(
                "only the current single-flight owner may publish"
            )
        if status not in {"ok", "error"}:
            raise ValueError("single-flight status must be ok or error")
        ttl = (
            self.outcome_ttl_seconds
            if outcome_ttl_seconds is None
            else outcome_ttl_seconds
        )
        if (
            isinstance(ttl, bool)
            or not isinstance(ttl, (int, float))
            or ttl <= 0
        ):
            raise ValueError("outcome_ttl_seconds must be positive")
        now = self._clock_ms()
        expires = now + self._lease_ms(float(ttl))
        envelope = {
            "schema": SINGLE_FLIGHT_OUTCOME_SCHEMA,
            "key_id": grant.key_id,
            "namespace": grant.namespace,
            "owner_id": grant.owner_id,
            "fencing_token": grant.fencing_token,
            "status": status,
            "value": value,
            "created_at_ms": now,
            "expires_at_ms": expires,
        }
        encoded = _single_flight_json_bytes(envelope)
        if len(encoded) > self.max_outcome_bytes:
            raise DistributedSingleFlightError(
                "single-flight outcome exceeds max_outcome_bytes"
            )
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        attestation_id = self._attestation_id(
            key_id=grant.key_id,
            namespace=grant.namespace,
            owner_id=grant.owner_id,
            fencing_token=grant.fencing_token,
            outcome_digest=digest,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM single_flight_leases
                WHERE key_id=? AND namespace=? AND owner_id=? AND lease_id=?
                  AND fencing_token=? AND expires_at_ms>?
                """,
                (
                    grant.key_id,
                    grant.namespace,
                    grant.owner_id,
                    grant.lease_id,
                    grant.fencing_token,
                    now,
                ),
            ).fetchone()
            if active is None:
                raise StaleSingleFlightLeaseError(
                    "cannot publish from an expired, stale, or foreign lease"
                )
            connection.execute(
                """
                INSERT INTO single_flight_outcomes VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key_id) DO UPDATE SET
                  namespace=excluded.namespace,
                  owner_id=excluded.owner_id,
                  fencing_token=excluded.fencing_token,
                  status=excluded.status,
                  outcome_json=excluded.outcome_json,
                  outcome_digest=excluded.outcome_digest,
                  attestation_id=excluded.attestation_id,
                  created_at_ms=excluded.created_at_ms,
                  expires_at_ms=excluded.expires_at_ms
                """,
                (
                    grant.key_id,
                    grant.namespace,
                    grant.owner_id,
                    grant.fencing_token,
                    status,
                    encoded.decode("utf-8"),
                    digest,
                    attestation_id,
                    now,
                    expires,
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        outcome = self.read_outcome(
            grant.key_id,
            fencing_token=grant.fencing_token,
        )
        if outcome is None:  # pragma: no cover - transaction guarantees this
            raise DistributedSingleFlightError(
                "published single-flight outcome was not readable"
            )
        return outcome

    publish_outcome = publish

    def read_outcome(
        self,
        key: Any,
        *,
        fencing_token: int | None = None,
    ) -> SingleFlightOutcome | None:
        key_id, requested_namespace = _single_flight_identity(key)
        now = self._clock_ms()
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM single_flight_outcomes
                WHERE key_id=? AND expires_at_ms>?
                """,
                (key_id, now),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        if (
            fencing_token is not None
            and int(row["fencing_token"]) != fencing_token
        ):
            return None
        if requested_namespace and str(row["namespace"]) != requested_namespace:
            raise DistributedSingleFlightError(
                "single-flight outcome namespace binding mismatch"
            )
        try:
            raw_outcome = str(row["outcome_json"])
            if len(raw_outcome.encode("utf-8")) > self.max_outcome_bytes:
                raise ValueError("outcome exceeds max_outcome_bytes")
            payload = json.loads(raw_outcome)
            if not isinstance(payload, Mapping):
                raise ValueError("outcome is not an object")
            encoded = _single_flight_json_bytes(payload)
            digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
            expected_attestation = self._attestation_id(
                key_id=str(row["key_id"]),
                namespace=str(row["namespace"]),
                owner_id=str(row["owner_id"]),
                fencing_token=int(row["fencing_token"]),
                outcome_digest=digest,
            )
            if (
                payload.get("schema") != SINGLE_FLIGHT_OUTCOME_SCHEMA
                or payload.get("key_id") != row["key_id"]
                or payload.get("namespace") != row["namespace"]
                or payload.get("owner_id") != row["owner_id"]
                or payload.get("fencing_token") != row["fencing_token"]
                or payload.get("status") != row["status"]
                or payload.get("created_at_ms") != row["created_at_ms"]
                or payload.get("expires_at_ms") != row["expires_at_ms"]
                or digest != row["outcome_digest"]
                or expected_attestation != row["attestation_id"]
            ):
                raise ValueError("outcome binding, digest, or attestation mismatch")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DistributedSingleFlightExecutionError(
                "single_flight_outcome_rejected"
            ) from exc
        attestation = SingleFlightAttestation(
            key_id=str(row["key_id"]),
            namespace=str(row["namespace"]),
            owner_id=str(row["owner_id"]),
            fencing_token=int(row["fencing_token"]),
            outcome_digest=str(row["outcome_digest"]),
            attestation_id=str(row["attestation_id"]),
        )
        return SingleFlightOutcome(
            key_id=str(row["key_id"]),
            namespace=str(row["namespace"]),
            owner_id=str(row["owner_id"]),
            fencing_token=int(row["fencing_token"]),
            status=str(row["status"]),
            value=payload.get("value"),
            created_at_ms=int(row["created_at_ms"]),
            expires_at_ms=int(row["expires_at_ms"]),
            outcome_digest=str(row["outcome_digest"]),
            attestation=attestation,
        )

    def verify_outcome(self, outcome: SingleFlightOutcome) -> bool:
        if not isinstance(outcome, SingleFlightOutcome):
            return False
        try:
            stored = self.read_outcome(
                {
                    "key_id": outcome.key_id,
                    "namespace": outcome.namespace,
                },
                fencing_token=outcome.fencing_token,
            )
        except DistributedSingleFlightError:
            return False
        return stored == outcome

    def discard_outcome(
        self,
        key: Any,
        *,
        fencing_token: int,
    ) -> bool:
        """Conditionally remove an unusable rendezvous result.

        This does not alter a live lease or its fence.  It is used by cache
        followers when the referenced cache record has independently expired
        or failed exact-key validation.
        """

        key_id, namespace = _single_flight_identity(key)
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                DELETE FROM single_flight_outcomes
                WHERE key_id=? AND namespace=? AND fencing_token=?
                """,
                (key_id, namespace, fencing_token),
            )
            return cursor.rowcount == 1
        finally:
            connection.close()

    def _generation_active(self, grant: SingleFlightLeaseGrant) -> bool:
        now = self._clock_ms()
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT fencing_token, expires_at_ms
                FROM single_flight_leases WHERE key_id=?
                """,
                (grant.key_id,),
            ).fetchone()
        finally:
            connection.close()
        return bool(
            row is not None
            and int(row["fencing_token"]) == grant.fencing_token
            and int(row["expires_at_ms"]) > now
        )

    def active_lease_count(self, *, namespace: str | None = None) -> int:
        """Return the current durable flight count for bounded observability."""

        now = self._clock_ms()
        connection = self._connect()
        try:
            if namespace is None:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM single_flight_leases
                    WHERE expires_at_ms>?
                    """,
                    (now,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM single_flight_leases
                    WHERE namespace=? AND expires_at_ms>?
                    """,
                    (str(namespace), now),
                ).fetchone()
        finally:
            connection.close()
        return int(row["count"]) if row is not None else 0

    def wait_for_outcome(
        self,
        grant: SingleFlightLeaseGrant,
        *,
        timeout_seconds: float,
        cancel_event: Any = None,
        deadline_monotonic: float | None = None,
    ) -> SingleFlightOutcome | None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        member_deadline = time.monotonic() + float(timeout_seconds)
        if deadline_monotonic is not None:
            member_deadline = min(member_deadline, float(deadline_monotonic))
        while True:
            if _single_flight_cancelled(cancel_event):
                raise DistributedSingleFlightCancelled(
                    "single_flight_member_cancelled"
                )
            remaining = member_deadline - time.monotonic()
            if remaining <= 0:
                raise DistributedSingleFlightTimeout(
                    "single_flight_member_deadline"
                )
            outcome = self.read_outcome(
                {
                    "key_id": grant.key_id,
                    "namespace": grant.namespace,
                },
                fencing_token=grant.fencing_token,
            )
            if outcome is not None:
                return outcome
            if not self._generation_active(grant):
                return None
            time.sleep(min(self.poll_interval_seconds, remaining))

    def coordinate(
        self,
        key: Any,
        execute: Callable[[], Any],
        *,
        owner_id: str | None = None,
        lease_seconds: float | None = None,
        timeout_seconds: float = 60.0,
        deadline_monotonic: float | None = None,
        cancel_event: Any = None,
        outcome_ttl_seconds: float | None = None,
    ) -> DistributedSingleFlightResult:
        """Execute one owner and give each live member the same attested outcome."""

        if not callable(execute):
            raise ValueError("execute must be callable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        member_deadline = time.monotonic() + float(timeout_seconds)
        if deadline_monotonic is not None:
            member_deadline = min(member_deadline, float(deadline_monotonic))
        duration = self.lease_seconds if lease_seconds is None else float(lease_seconds)
        while True:
            if _single_flight_cancelled(cancel_event):
                raise DistributedSingleFlightCancelled(
                    "single_flight_member_cancelled"
                )
            if time.monotonic() >= member_deadline:
                raise DistributedSingleFlightTimeout(
                    "single_flight_member_deadline"
                )
            grant = self.acquire(
                key,
                owner_id=owner_id,
                lease_seconds=duration,
            )
            if not grant.acquired:
                outcome = self.wait_for_outcome(
                    grant,
                    timeout_seconds=max(
                        self.poll_interval_seconds,
                        member_deadline - time.monotonic(),
                    ),
                    cancel_event=cancel_event,
                    deadline_monotonic=member_deadline,
                )
                if outcome is None:
                    continue
                if not outcome.successful:
                    reason = (
                        str(outcome.value.get("reason_code"))
                        if isinstance(outcome.value, Mapping)
                        else "single_flight_execution_failed"
                    )
                    raise DistributedSingleFlightExecutionError(
                        reason,
                        outcome=outcome,
                    )
                return DistributedSingleFlightResult(outcome, owner=False)

            # Cancellation and deadlines are member-specific: an owner that
            # has not begun user work simply relinquishes the generation so a
            # live follower can take over.  It does not publish cancellation.
            if _single_flight_cancelled(cancel_event):
                self.release(grant)
                raise DistributedSingleFlightCancelled(
                    "single_flight_member_cancelled"
                )
            if time.monotonic() >= member_deadline:
                self.release(grant)
                raise DistributedSingleFlightTimeout(
                    "single_flight_member_deadline"
                )

            heartbeat_stop = threading.Event()
            heartbeat_failures: list[BaseException] = []

            def maintain_lease() -> None:
                interval = max(0.01, duration / 3.0)
                while not heartbeat_stop.wait(interval):
                    try:
                        self.heartbeat(grant, lease_seconds=duration)
                    except BaseException as exc:
                        heartbeat_failures.append(exc)
                        return

            heartbeat_thread = threading.Thread(
                target=maintain_lease,
                name=f"cache-flight-heartbeat-{grant.fencing_token}",
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                value = execute()
                member_cancelled = _single_flight_cancelled(cancel_event)
                member_expired = time.monotonic() >= member_deadline
                heartbeat_stop.set()
                heartbeat_thread.join()
                if heartbeat_failures:
                    raise StaleSingleFlightLeaseError(
                        "single-flight owner lost its heartbeat fence"
                    ) from heartbeat_failures[0]
                outcome = self.publish(
                    grant,
                    value,
                    status="ok",
                    outcome_ttl_seconds=outcome_ttl_seconds,
                )
                # A member-specific cancellation/deadline does not discard
                # useful completed work or poison other followers.  Publish
                # under the still-live fence, then honor only this member's
                # terminal state.
                if member_cancelled:
                    raise DistributedSingleFlightCancelled(
                        "single_flight_member_cancelled"
                    )
                if member_expired:
                    raise DistributedSingleFlightTimeout(
                        "single_flight_member_deadline"
                    )
                return DistributedSingleFlightResult(outcome, owner=True)
            except (
                DistributedSingleFlightCancelled,
                DistributedSingleFlightTimeout,
            ):
                raise
            except BaseException:
                heartbeat_stop.set()
                heartbeat_thread.join()
                if not heartbeat_failures:
                    try:
                        self.publish(
                            grant,
                            {"reason_code": "single_flight_execution_failed"},
                            status="error",
                            outcome_ttl_seconds=outcome_ttl_seconds,
                        )
                    except BaseException:
                        pass
                raise
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join()
                self.release(grant)

    single_flight = coordinate
    execute_single_flight = coordinate
    run_single_flight = coordinate

    def purge_expired(self) -> dict[str, int]:
        now = self._clock_ms()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outcomes = connection.execute(
                "DELETE FROM single_flight_outcomes WHERE expires_at_ms<=?",
                (now,),
            ).rowcount
            leases = connection.execute(
                "DELETE FROM single_flight_leases WHERE expires_at_ms<=?",
                (now,),
            ).rowcount
            connection.commit()
            return {"leases": leases, "outcomes": outcomes}
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


# Compatibility spellings for callers focused on the lease primitive rather
# than its cache-facing use.
SingleFlightLeaseCoordinator = DistributedSingleFlightCoordinator
SingleFlightCoordinator = DistributedSingleFlightCoordinator
SingleFlightResult = DistributedSingleFlightResult
SingleFlightTimeout = DistributedSingleFlightTimeout
SingleFlightCancelled = DistributedSingleFlightCancelled
SingleFlightExecutionError = DistributedSingleFlightExecutionError


@dataclass(frozen=True)
class LeasedQueuedTask:
    """A queue task paired with the only grant that authorizes its execution."""

    task: Any
    grant: LeaseGrant


class LeaseQueueBridge:
    """Bind the legacy DuckDB queue claim lifecycle to Profile G leases.

    Queue ownership alone is never execution authority.  A bridge claim is
    returned only after the embedded canonical TaskSpec has an accepted lease;
    a conflicting queue claim is immediately returned to the queue.
    """

    def __init__(
        self,
        queue: Any,
        coordinator: LeaseCoordinator,
        *,
        worker_id: str,
        claimant_did: str,
        lease_ms: int = 60_000,
    ) -> None:
        self.queue = queue
        self.coordinator = coordinator
        self.worker_id = worker_id
        self.claimant_did = claimant_did
        self.lease_ms = lease_ms

    def claim_next(self, *, supported_task_types: list[str] | None = None) -> LeasedQueuedTask | None:
        task = self.queue.claim_next(worker_id=self.worker_id, supported_task_types=supported_task_types)
        if task is None:
            return None
        payload = task.payload if isinstance(task.payload, Mapping) else {}
        try:
            adapted = self.coordinator.register_bundle(payload)
            grant = self.coordinator.claim(
                adapted["task_cid"],
                self.claimant_did,
                requested_lease_ms=self.lease_ms,
            )
        except Exception:
            self.queue.release(task_id=task.task_id, worker_id=self.worker_id, reason="profile-g lease not accepted")
            raise
        return LeasedQueuedTask(task=task, grant=grant)

    def renew(self, leased: LeasedQueuedTask) -> LeasedQueuedTask:
        return LeasedQueuedTask(
            task=leased.task,
            grant=self.coordinator.renew(leased.grant, requested_lease_ms=self.lease_ms),
        )

    def release(self, leased: LeasedQueuedTask, *, reason: str = "released") -> bool:
        self.coordinator.release(leased.grant, reason=reason)
        return bool(self.queue.release(task_id=leased.task.task_id, worker_id=self.worker_id, reason=reason))

    def complete(
        self,
        leased: LeasedQueuedTask,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        failure_class: str = "none",
    ) -> dict[str, Any]:
        receipt = self.coordinator.receipt(
            leased.grant,
            status=status,
            output=output,
            failure_class=failure_class,
        )
        queue_status = "completed" if status == "succeeded" else "failed"
        self.queue.complete(
            task_id=leased.task.task_id,
            status=queue_status,
            result={"profile_g": receipt},
            error=None if status == "succeeded" else failure_class,
        )
        return receipt


__all__ = [
    "DEFAULT_SINGLE_FLIGHT_LEASE_SECONDS",
    "DEFAULT_SINGLE_FLIGHT_MAX_OUTCOME_BYTES",
    "DEFAULT_SINGLE_FLIGHT_OUTCOME_TTL_SECONDS",
    "DEFAULT_SINGLE_FLIGHT_POLL_SECONDS",
    "DISTRIBUTED_INPUT_SCHEMA",
    "DISTRIBUTED_LANE_DISPATCH_SCHEMA",
    "DISTRIBUTED_PUBLICATION_SCHEMA",
    "DISTRIBUTED_QUARANTINE_SCHEMA",
    "DependencyNotReadyError",
    "DistributedLaneDispatch",
    "DistributedSingleFlightCancelled",
    "DistributedSingleFlightCoordinator",
    "DistributedSingleFlightError",
    "DistributedSingleFlightExecutionError",
    "DistributedSingleFlightResult",
    "DistributedSingleFlightTimeout",
    "LeaseConflictError", "LeaseCoordinator", "LeaseError", "LeaseExpiredError", "LeaseGrant",
    "LeaseQueueBridge", "LeasedQueuedTask",
    "ImmutableLaneInputArtifact",
    "MAX_LEASE_MS", "MIN_LEASE_MS",
    "REMOTE_LANE_RESULT_SCHEMA",
    "RemoteLaneResult",
    "SINGLE_FLIGHT_ATTESTATION_SCHEMA",
    "SINGLE_FLIGHT_OUTCOME_SCHEMA",
    "SINGLE_FLIGHT_STORE_SCHEMA",
    "SingleFlightAttestation",
    "SingleFlightCancelled",
    "SingleFlightCoordinator",
    "SingleFlightExecutionError",
    "SingleFlightLeaseCoordinator",
    "SingleFlightLeaseGrant",
    "SingleFlightOutcome",
    "SingleFlightResult",
    "SingleFlightTimeout",
    "StaleFencingTokenError",
    "StaleSingleFlightLeaseError",
    "TaskLeaseState", "adapt_goal_bundle",
    "WORKER_CAPABILITY_RECEIPT_SCHEMA",
    "WORKER_ENVIRONMENT_RECEIPT_SCHEMA",
    "WorkerCapabilityReceipt",
    "WorkerEnvironmentReceipt",
    "canonical_profile_g_bytes", "profile_g_cid",
]
