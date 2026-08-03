"""Transactional, directly queryable DuckDB projection for supervisor tasks.

The module deliberately has no import-time DuckDB dependency.  A database is
installed through a temporary file and an atomic rename, while updates use a
short transaction protected by the repository's process-shared DuckDB lock.
Every public read validates the small schema envelope and every write checks a
durable writer fence and compare-and-swap revision.

Flexible records are stored as canonical JSON.  The lossless
``formal_plan_input_*`` tables are part of the schema so the independent
:class:`FormalPlanCompiler` reader can reproduce the original plan rather than
trusting this projection's denormalized task rows.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .duckdb_state import exclusive_file_lock
from ..planning.formal_plan_compiler import (
    CompilationStatus,
    FORMAL_PLAN_INPUT_SCHEMA,
    FormalPlanCompiler,
    prompt_goal_graph_to_formal_input,
)
from ..proof.formal_verification_contracts import canonical_json, content_identity


DUCKDB_TASK_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-task-source@1"
)
DUCKDB_TASK_SOURCE_SCHEMA_VERSION: Final = 1
WORKFLOW_SCHEMA: Final = DUCKDB_TASK_SOURCE_SCHEMA
WORKFLOW_SCHEMA_VERSION: Final = DUCKDB_TASK_SOURCE_SCHEMA_VERSION
SCHEMA_VERSION: Final = DUCKDB_TASK_SOURCE_SCHEMA_VERSION
TASK_SOURCE_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-snapshot@1"
)
TASK_SOURCE_PAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-page@1"
)
TASK_SOURCE_EVENT_PAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-event-page@1"
)
TASK_SOURCE_CAS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-cas@1"
)
TASK_SOURCE_INTEGRITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-integrity@1"
)
TASK_SOURCE_MIGRATION_PREVIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-migration-preview@1"
)
TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-source-migration-receipt@1"
)
MATERIALIZATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-materialization-receipt@1"
)
DERIVED_RUNTIME_SOURCE_ROLE: Final = "derived_runtime"
DERIVED_RUNTIME_MATERIALIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-derived-runtime-materialization@1"
)

# Operator-protected seed anchors. Derived runtime materialization refuses to
# install a population whose task outputs target these paths.
DEFAULT_DERIVED_PROTECTED_ANCHORS: Final[tuple[str, ...]] = (
    "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md",
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json",
    "docs/architecture/agent_supervisor_planner_doctor_threat_model.md",
    "config/agent_supervisor_planner_doctor_authority_policy.json",
    "config/agent_supervisor_planner_doctor_authority_policy.seal.json",
    "test/api/test_agent_supervisor_planner_doctor_authority_policy.py",
    "config/agent_supervisor_planner_doctor_benchmark.json",
    "config/agent_supervisor_planner_doctor_benchmark.seal.json",
    "docs/architecture/agent_supervisor_planner_doctor_benchmark.md",
    "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json",
    "test/api/test_agent_supervisor_planner_doctor_benchmark_contract.py",
)

DEFAULT_QUERY_LIMIT: Final = 100
MAX_QUERY_LIMIT: Final = 1_000
MAX_TASKS: Final = 8_192
MAX_GOALS: Final = 4_096
MAX_EDGES: Final = 65_536
MAX_EVENTS_PER_PAGE: Final = 1_000
MAX_EVENTS: Final = 100_000
MAX_JSON_BYTES: Final = 2 * 1024 * 1024
MAX_IDENTIFIER_BYTES: Final = 512
MAX_WATCH_SECONDS: Final = 30.0

_CURSOR_VERSION: Final = 1
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_TERMINAL_STATUSES: Final = frozenset(
    {"completed", "cancelled", "skipped", "failed", "quarantined"}
)
_COMPLETED_STATUSES: Final = frozenset({"completed", "skipped"})
_READY_STATUSES: Final = frozenset(
    {"proposed", "admitted", "pending", "ready", "retrying"}
)
_TASK_STATUSES: Final = frozenset(
    {
        *_READY_STATUSES,
        *_TERMINAL_STATUSES,
        "claimed",
        "in_progress",
        "blocked",
    }
)
_MUTABLE_IDENTITY_FIELDS: Final = frozenset(
    {
        "status",
        "created_at",
        "created_at_ms",
        "updated_at",
        "updated_at_ms",
        "revision",
        "task_revision",
        "completion",
        "completion_receipt",
        "receipt",
    }
)
_IDENTITY_ALIAS_FIELDS: Final = frozenset(
    {
        "cid",
        "content_id",
        "canonical_id",
        "canonical_task_cid",
        "task_cid",
        "task_id",
        "id",
    }
)

_TABLE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "workflow_metadata": ("key", "value", "value_json"),
    "artifacts": (
        "cid",
        "media_type",
        "byte_length",
        "digest",
        "storage_uri",
        "provenance_json",
    ),
    "goals": (
        "goal_cid",
        "goal_alias",
        "parent_goal_cid",
        "ordinal",
        "title",
        "body_json",
    ),
    "tasks": (
        "task_cid",
        "task_alias",
        "goal_cid",
        "ordinal",
        "status",
        "revision",
        "identity_json",
        "body_json",
    ),
    "task_dependencies": (
        "task_cid",
        "dependency_task_cid",
        "kind",
    ),
    "task_outputs": ("task_cid", "ordinal", "path", "effect_json"),
    "task_validations": (
        "task_cid",
        "ordinal",
        "argv_json",
        "policy_json",
    ),
    "task_acceptance": (
        "task_cid",
        "ordinal",
        "criterion",
        "evidence_policy_json",
    ),
    "task_events": (
        "event_cid",
        "sequence",
        "revision",
        "task_cid",
        "event_type",
        "body_json",
    ),
    "materialization_receipts": (
        "receipt_cid",
        "plan_root_cid",
        "revision",
        "body_json",
    ),
    "formal_plan_input_records": ("section", "record_id", "payload_json"),
    "formal_plan_input_metadata": ("field_name", "field_value"),
    "schema_migration_receipts": (
        "receipt_cid",
        "from_version",
        "to_version",
        "revision",
        "body_json",
    ),
}

_QUERY_TABLES: Final = frozenset(
    {
        "artifacts",
        "goals",
        "tasks",
        "task_dependencies",
        "task_outputs",
        "task_validations",
        "task_acceptance",
        "task_events",
        "materialization_receipts",
        "formal_plan_input_records",
        "formal_plan_input_metadata",
    }
)

_SCHEMA_SQL: Final = """
CREATE TABLE workflow_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL,
    value_json VARCHAR NOT NULL
);
CREATE TABLE artifacts (
    cid VARCHAR PRIMARY KEY,
    media_type VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    digest VARCHAR NOT NULL,
    storage_uri VARCHAR NOT NULL,
    provenance_json VARCHAR NOT NULL
);
CREATE TABLE goals (
    goal_cid VARCHAR PRIMARY KEY,
    goal_alias VARCHAR NOT NULL UNIQUE,
    parent_goal_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL UNIQUE,
    title VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE TABLE tasks (
    task_cid VARCHAR PRIMARY KEY,
    task_alias VARCHAR NOT NULL UNIQUE,
    goal_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL UNIQUE,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    identity_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE TABLE task_dependencies (
    task_cid VARCHAR NOT NULL,
    dependency_task_cid VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    PRIMARY KEY(task_cid, dependency_task_cid, kind)
);
CREATE TABLE task_outputs (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    effect_json VARCHAR NOT NULL,
    PRIMARY KEY(task_cid, ordinal),
    UNIQUE(task_cid, path)
);
CREATE TABLE task_validations (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    argv_json VARCHAR NOT NULL,
    policy_json VARCHAR NOT NULL,
    PRIMARY KEY(task_cid, ordinal)
);
CREATE TABLE task_acceptance (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    criterion VARCHAR NOT NULL,
    evidence_policy_json VARCHAR NOT NULL,
    PRIMARY KEY(task_cid, ordinal)
);
CREATE TABLE task_events (
    event_cid VARCHAR PRIMARY KEY,
    sequence BIGINT NOT NULL UNIQUE,
    revision BIGINT NOT NULL,
    task_cid VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE TABLE materialization_receipts (
    receipt_cid VARCHAR PRIMARY KEY,
    plan_root_cid VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE TABLE formal_plan_input_records (
    section VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL,
    PRIMARY KEY(section, record_id)
);
CREATE TABLE formal_plan_input_metadata (
    field_name VARCHAR PRIMARY KEY,
    field_value VARCHAR NOT NULL
);
CREATE TABLE schema_migration_receipts (
    receipt_cid VARCHAR PRIMARY KEY,
    from_version INTEGER NOT NULL,
    to_version INTEGER NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
"""


class DuckDBTaskSourceError(RuntimeError):
    """Base class for fail-closed task-source errors."""


class DuckDBUnavailableError(DuckDBTaskSourceError):
    """DuckDB was requested but is not installed."""


class TaskSourceIntegrityError(DuckDBTaskSourceError):
    """The database is corrupt, partial, foreign, or internally inconsistent."""


class TaskSourceConflictError(DuckDBTaskSourceError):
    """A CAS revision, cursor, writer, or event identity is stale."""


class TaskSourceBoundsError(DuckDBTaskSourceError, ValueError):
    """A caller attempted an unbounded or over-budget operation."""


class TaskSourceInjectionError(DuckDBTaskSourceError, ValueError):
    """An untrusted SQL identifier or unsafe persisted value was rejected."""


class UnsupportedSchemaMigrationError(DuckDBTaskSourceError):
    """The requested schema migration is outside the closed compatibility map."""


class _RecordMapping(Mapping[str, Any]):
    """Let result records support both attribute and mapping-style consumers."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class WriterFence(_RecordMapping):
    writer_id: str
    fencing_token: int
    revision: int

    @property
    def fence(self) -> int:
        return self.fencing_token

    def to_dict(self) -> dict[str, Any]:
        return {
            "writer_id": self.writer_id,
            "fencing_token": self.fencing_token,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class TaskRecord(_RecordMapping):
    task_cid: str
    task_alias: str
    goal_cid: str
    ordinal: int
    status: str
    revision: int
    body: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()

    @property
    def task_id(self) -> str:
        return self.task_alias

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_cid": self.task_cid,
            "task_alias": self.task_alias,
            "task_id": self.task_alias,
            "goal_cid": self.goal_cid,
            "ordinal": self.ordinal,
            "status": self.status,
            "revision": self.revision,
            "dependencies": list(self.dependencies),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class TaskSourceSnapshot(_RecordMapping):
    source_schema: str
    schema_version: int
    plan_root_cid: str
    repository_tree_id: str
    projection_cid: str
    formal_plan_id: str
    source_identity: str
    revision: int
    event_cursor: int
    goal_count: int
    task_count: int
    dependency_count: int
    terminal: bool

    @property
    def cursor(self) -> int:
        return self.event_cursor

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_SNAPSHOT_SCHEMA,
            "source_schema": self.source_schema,
            "schema_version": self.schema_version,
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "projection_cid": self.projection_cid,
            "formal_plan_id": self.formal_plan_id,
            "source_identity": self.source_identity,
            "revision": self.revision,
            "event_cursor": self.event_cursor,
            "cursor": self.event_cursor,
            "goal_count": self.goal_count,
            "task_count": self.task_count,
            "dependency_count": self.dependency_count,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class TaskPage(_RecordMapping):
    tasks: tuple[TaskRecord, ...]
    revision: int
    next_cursor: str = ""

    @property
    def records(self) -> tuple[TaskRecord, ...]:
        return self.tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_PAGE_SCHEMA,
            "tasks": [item.to_dict() for item in self.tasks],
            "records": [item.to_dict() for item in self.tasks],
            "revision": self.revision,
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True)
class EventPage(_RecordMapping):
    events: tuple[Mapping[str, Any], ...]
    cursor: int
    revision: int
    timed_out: bool = False

    @property
    def next_cursor(self) -> int:
        return self.cursor

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_EVENT_PAGE_SCHEMA,
            "events": [dict(item) for item in self.events],
            "cursor": self.cursor,
            "next_cursor": self.cursor,
            "revision": self.revision,
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True)
class CASResult(_RecordMapping):
    task: TaskRecord
    previous_status: str
    revision: int
    event_cursor: int
    changed: bool
    receipt_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_CAS_SCHEMA,
            "task": self.task.to_dict(),
            "previous_status": self.previous_status,
            "revision": self.revision,
            "event_cursor": self.event_cursor,
            "changed": self.changed,
            "receipt_cid": self.receipt_cid,
        }


@dataclass(frozen=True)
class IntegrityReport(_RecordMapping):
    valid: bool
    plan_root_cid: str
    projection_cid: str
    revision: int
    event_cursor: int
    formal_plan_id: str
    source_identity: str
    checked_tables: tuple[str, ...]
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_INTEGRITY_SCHEMA,
            "valid": self.valid,
            "plan_root_cid": self.plan_root_cid,
            "projection_cid": self.projection_cid,
            "revision": self.revision,
            "event_cursor": self.event_cursor,
            "formal_plan_id": self.formal_plan_id,
            "source_identity": self.source_identity,
            "checked_tables": list(self.checked_tables),
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class MigrationPreview(_RecordMapping):
    preview_id: str
    database_identity: str
    from_version: int
    to_version: int
    base_revision: int
    statement_digests: tuple[str, ...]
    rollback_identity: str
    changed: bool
    supported: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_MIGRATION_PREVIEW_SCHEMA,
            "preview_id": self.preview_id,
            "database_identity": self.database_identity,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "base_revision": self.base_revision,
            "statement_digests": list(self.statement_digests),
            "rollback_identity": self.rollback_identity,
            "changed": self.changed,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class MigrationReceipt(_RecordMapping):
    receipt_cid: str
    preview_id: str
    from_version: int
    to_version: int
    revision: int
    changed: bool
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA,
            "receipt_cid": self.receipt_cid,
            "preview_id": self.preview_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "revision": self.revision,
            "changed": self.changed,
            "rolled_back": self.rolled_back,
        }


def duckdb_available() -> bool:
    """Return availability without importing DuckDB when this module loads."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _duckdb_module() -> Any:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise DuckDBUnavailableError(
            "DuckDB task storage is unavailable; install the optional duckdb "
            "dependency or select another task source"
        ) from exc
    return duckdb


def _canonical(value: Any, *, noun: str = "record") -> str:
    try:
        encoded = canonical_json(value)
    except Exception as exc:
        raise ValueError(f"{noun} must contain canonical JSON values") from exc
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise TaskSourceBoundsError(f"{noun} exceeds the JSON persistence bound")
    return encoded


def _decode_canonical(value: Any, *, noun: str) -> Any:
    if not isinstance(value, str):
        raise TaskSourceIntegrityError(f"{noun} is not stored as JSON text")
    if len(value.encode("utf-8")) > MAX_JSON_BYTES:
        raise TaskSourceIntegrityError(f"{noun} exceeds the JSON persistence bound")
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise TaskSourceIntegrityError(f"{noun} contains malformed JSON") from exc
    if _canonical(decoded, noun=noun) != value:
        raise TaskSourceIntegrityError(f"{noun} is not canonical JSON")
    return decoded


def _identifier(value: Any, *, noun: str) -> str:
    selected = str(value or "").strip()
    if (
        not selected
        or len(selected.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or not _SAFE_IDENTIFIER.fullmatch(selected)
    ):
        raise TaskSourceInjectionError(
            f"{noun} must be a bounded, single-line opaque identifier"
        )
    return selected


def _status(value: Any) -> str:
    selected = str(getattr(value, "value", value) or "").strip().lower()
    if selected not in _TASK_STATUSES:
        raise ValueError(f"unsupported task status {selected!r}")
    return selected


def _task_key(value: Any, *, noun: str = "task identifier") -> str:
    if isinstance(value, TaskRecord):
        value = value.task_cid
    elif isinstance(value, Mapping):
        value = _first(value, "task_cid", "task_alias", "task_id", "id")
    return _identifier(value, noun=noun)


def _positive_limit(limit: int, *, maximum: int = MAX_QUERY_LIMIT) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
        raise TaskSourceBoundsError(f"limit must be between 1 and {maximum}")
    return limit


def _as_mapping(value: Any, *, noun: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        converter = getattr(value, "to_dict", None)
        if not callable(converter):
            converter = getattr(value, "to_record", None)
        if not callable(converter):
            raise TypeError(f"{noun} must be a mapping or expose to_dict()/to_record()")
        converted = converter()
        if not isinstance(converted, Mapping):
            raise TypeError(f"{noun} conversion must produce a mapping")
        result = dict(converted)
    normalized = json.loads(_canonical(result, noun=noun))
    if not isinstance(normalized, dict):
        raise TypeError(f"{noun} must be a JSON object")
    return normalized


def _values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _first(record: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _without_fields(value: Any, excluded: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_fields(member, excluded)
            for key, member in value.items()
            if str(key) not in excluded
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_without_fields(member, excluded) for member in value]
    return value


def _task_identity_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    value = _without_fields(
        record, _MUTABLE_IDENTITY_FIELDS | _IDENTITY_ALIAS_FIELDS
    )
    if not isinstance(value, dict):
        raise TypeError("task identity payload must be an object")
    return value


def _cursor_encode(plan_root: str, revision: int, offset: int) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "plan_root_cid": plan_root,
        "revision": revision,
        "offset": offset,
    }
    raw = _canonical(payload, noun="task cursor").encode("utf-8")
    digest = hashlib.sha256(b"duckdb-task-cursor-v1\0" + raw).hexdigest()
    envelope = _canonical(
        {"payload": payload, "digest": digest}, noun="task cursor envelope"
    ).encode("utf-8")
    return base64.urlsafe_b64encode(envelope).decode("ascii").rstrip("=")


def _cursor_decode(cursor: str, plan_root: str, revision: int) -> int:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 2_048:
        raise TaskSourceConflictError("task cursor is malformed")
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        envelope = json.loads(raw)
        payload = envelope["payload"]
        digest = envelope["digest"]
    except Exception as exc:
        raise TaskSourceConflictError("task cursor is malformed") from exc
    canonical_payload = _canonical(payload, noun="task cursor").encode("utf-8")
    expected = hashlib.sha256(
        b"duckdb-task-cursor-v1\0" + canonical_payload
    ).hexdigest()
    if not isinstance(digest, str) or not hmac.compare_digest(digest, expected):
        raise TaskSourceConflictError("task cursor identity does not match")
    if (
        payload.get("v") != _CURSOR_VERSION
        or payload.get("plan_root_cid") != plan_root
        or payload.get("revision") != revision
    ):
        raise TaskSourceConflictError("task cursor is stale or foreign")
    offset = payload.get("offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise TaskSourceConflictError("task cursor offset is invalid")
    return offset


def _configure_connection(connection: Any) -> None:
    """Disable DuckDB's external/autoload surfaces for this local store."""

    for statement in (
        "SET enable_external_access=false",
        "SET autoinstall_known_extensions=false",
        "SET autoload_known_extensions=false",
        "SET threads=1",
        "SET memory_limit='256MB'",
    ):
        connection.execute(statement)


def _connect(path: Path, *, read_only: bool) -> Any:
    duckdb = _duckdb_module()
    try:
        connection = duckdb.connect(str(path), read_only=read_only)
        _configure_connection(connection)
        return connection
    except Exception as exc:
        raise TaskSourceIntegrityError(f"could not open DuckDB task source: {exc}") from exc


def _table_names(connection: Any) -> set[str]:
    return {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}


def _table_columns(connection: Any, table: str) -> tuple[str, ...]:
    # ``table`` only comes from the module's closed constant map.
    cursor = connection.execute(f'SELECT * FROM "{table}" LIMIT 0')
    return tuple(str(item[0]) for item in cursor.description or ())


def _metadata(connection: Any) -> dict[str, tuple[str, str]]:
    rows = connection.execute(
        "SELECT key, value, value_json FROM workflow_metadata ORDER BY key"
    ).fetchall()
    result: dict[str, tuple[str, str]] = {}
    for key, value, value_json in rows:
        selected = str(key)
        if selected in result:
            raise TaskSourceIntegrityError("workflow metadata contains duplicate keys")
        result[selected] = (str(value), str(value_json))
    return result


def _meta_value(metadata: Mapping[str, tuple[str, str]], key: str) -> str:
    try:
        value, value_json = metadata[key]
    except KeyError as exc:
        raise TaskSourceIntegrityError(
            f"workflow metadata is missing required key {key!r}"
        ) from exc
    decoded = _decode_canonical(value_json, noun=f"workflow metadata {key}")
    if decoded != value:
        raise TaskSourceIntegrityError(
            f"workflow metadata {key!r} text and JSON values disagree"
        )
    return value


def _set_metadata(connection: Any, key: str, value: Any) -> None:
    selected = str(value)
    encoded = _canonical(selected, noun=f"workflow metadata {key}")
    connection.execute(
        """
        INSERT INTO workflow_metadata(key, value, value_json) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value,
            value_json=excluded.value_json
        """,
        [key, selected, encoded],
    )


def _section_bundle(source: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "objectives": ("objectives", "objective_records", "goals"),
        "tasks": ("tasks", "task_records", "taskboard", "taskboard_records"),
        "ast": ("ast", "ast_records", "ast_scopes"),
        "policies": (
            "policies",
            "policy_records",
            "proof_policies",
            "proof_policy",
        ),
        "leases": ("leases", "lease_records"),
        "evidence": ("evidence", "evidence_records"),
    }
    bundle: dict[str, Any] = {"repository_tree_id": str(source.get("repository_tree_id") or "")}
    for section, names in aliases.items():
        records: list[dict[str, Any]] = []
        for name in names:
            if name not in source:
                continue
            for item in _values(source[name]):
                records.append(_as_mapping(item, noun=f"{section} record"))
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            identity = content_identity(record)
            previous = unique.get(identity)
            if previous is not None and previous != record:
                raise TaskSourceIntegrityError(
                    f"conflicting canonical {section} record {identity}"
                )
            unique[identity] = record
        bundle[section] = [unique[key] for key in sorted(unique)]
    return bundle


def _source_and_projection(
    source: Any,
    *,
    repository_tree_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the compiler source and optional richer prompt graph record."""

    graph_record: dict[str, Any] | None = None
    try:
        from ..prompt.prompt_workflow import PromptGoalGraph
    except ImportError:
        PromptGoalGraph = None  # type: ignore[assignment,misc]

    if PromptGoalGraph is not None and isinstance(source, PromptGoalGraph):
        tree_id = _identifier(repository_tree_id, noun="repository_tree_id")
        graph_record = _as_mapping(source, noun="prompt goal graph")
        graph_record["content_id"] = source.plan_root_cid
        formal = prompt_goal_graph_to_formal_input(
            source, repository_tree_id=tree_id
        )
        return _as_mapping(formal, noun="formal plan input"), graph_record

    record = _as_mapping(source, noun="formal plan input")
    if (
        PromptGoalGraph is not None
        and record.get("schema") == getattr(PromptGoalGraph, "SCHEMA", object())
    ):
        graph = PromptGoalGraph.from_dict(record)
        tree_id = _identifier(repository_tree_id, noun="repository_tree_id")
        formal = prompt_goal_graph_to_formal_input(
            graph, repository_tree_id=tree_id
        )
        return _as_mapping(formal, noun="formal plan input"), record
    return record, graph_record


def _goal_rows(
    bundle: Mapping[str, Any],
    graph: Mapping[str, Any] | None,
) -> list[tuple[str, str, str, int, str, str]]:
    records: list[dict[str, Any]] = []
    if graph is not None:
        records = [
            _as_mapping(item, noun="goal record")
            for item in _values(graph.get("goals"))
        ]
    else:
        for objective in bundle["objectives"]:
            records.append(dict(objective))
            parent = _first(objective, "goal_cid", "canonical_goal_id", "goal_id", "id")
            for raw_subgoal in _values(objective.get("subgoals")):
                subgoal = _as_mapping(raw_subgoal, noun="subgoal record")
                subgoal.setdefault("parent_goal_cid", parent)
                records.append(subgoal)
    if not records:
        raise TaskSourceIntegrityError("formal plan input has no goals")
    if len(records) > MAX_GOALS:
        raise TaskSourceBoundsError("goal population exceeds the storage bound")

    result: list[tuple[str, str, str, int, str, str]] = []
    cids: set[str] = set()
    aliases: set[str] = set()
    for ordinal, record in enumerate(
        sorted(
            records,
            key=lambda item: _first(
                item,
                "goal_cid",
                "subgoal_cid",
                "content_id",
                "cid",
                "goal_id",
                "subgoal_id",
                "id",
            ),
        )
    ):
        cid = _identifier(
            _first(
                record,
                "goal_cid",
                "subgoal_cid",
                "content_id",
                "cid",
                "canonical_goal_id",
                "canonical_subgoal_id",
                "goal_id",
                "subgoal_id",
                "id",
            )
            or content_identity(record),
            noun="goal_cid",
        )
        alias = _identifier(
            _first(record, "goal_key", "goal_id", "subgoal_id", "id") or cid,
            noun="goal_alias",
        )
        if cid in cids or alias in aliases:
            raise TaskSourceIntegrityError("goal CIDs and aliases must be unique")
        cids.add(cid)
        aliases.add(alias)
        parent = _first(
            record, "parent_goal_cid", "parent_id", "parent_goal_id"
        )
        if parent:
            parent = _identifier(parent, noun="parent_goal_cid")
        result.append(
            (
                cid,
                alias,
                parent,
                ordinal,
                str(record.get("title") or record.get("objective") or ""),
                _canonical(record, noun="goal body"),
            )
        )
    cid_set = {row[0] for row in result}
    if any(alias != cid and alias in cid_set for cid, alias, *_rest in result):
        raise TaskSourceIntegrityError(
            "goal alias collides with another goal CID"
        )
    for cid, _alias, parent, *_rest in result:
        if parent and parent not in cid_set:
            raise TaskSourceIntegrityError(
                f"goal {cid!r} references unknown parent {parent!r}"
            )
        if parent == cid:
            raise TaskSourceIntegrityError("goal cannot parent itself")
    _assert_acyclic({row[0]: (() if not row[2] else (row[2],)) for row in result}, "goal")
    return result


def _task_rows(
    bundle: Mapping[str, Any],
    graph: Mapping[str, Any] | None,
) -> tuple[
    list[tuple[str, str, str, int, str, int, str, str]],
    dict[str, dict[str, Any]],
]:
    records = (
        [_as_mapping(item, noun="task record") for item in _values(graph.get("tasks"))]
        if graph is not None
        else [dict(item) for item in bundle["tasks"]]
    )
    if not records:
        raise TaskSourceIntegrityError("formal plan input has no tasks")
    if len(records) > MAX_TASKS:
        raise TaskSourceBoundsError("task population exceeds the storage bound")

    # Prompt workflow records can prove their supplied CID excludes status.
    try:
        from ..prompt.prompt_workflow import PromptTaskRecord
    except ImportError:
        PromptTaskRecord = None  # type: ignore[assignment,misc]

    result: list[tuple[str, str, str, int, str, int, str, str]] = []
    by_cid: dict[str, dict[str, Any]] = {}
    aliases: set[str] = set()
    immutable_ids: dict[str, str] = {}
    sort_key = lambda item: _first(  # noqa: E731
        item,
        "task_cid",
        "canonical_task_cid",
        "content_id",
        "cid",
        "task_id",
        "task_key",
        "id",
    )
    for ordinal, record in enumerate(sorted(records, key=sort_key)):
        if (
            PromptTaskRecord is not None
            and record.get("schema") == getattr(PromptTaskRecord, "SCHEMA", object())
        ):
            parsed = PromptTaskRecord.from_dict(record)
            claimed = _first(record, "task_cid", "content_id")
            if claimed and claimed != parsed.task_cid:
                raise TaskSourceIntegrityError(
                    "prompt task CID does not match its status-independent identity"
                )
        cid = _identifier(
            _first(
                record,
                "task_cid",
                "canonical_task_cid",
                "content_id",
                "cid",
                "task_id",
                "id",
            )
            or content_identity(_task_identity_payload(record)),
            noun="task_cid",
        )
        alias = _identifier(
            _first(record, "task_key", "task_id", "id") or cid,
            noun="task_alias",
        )
        goal = _identifier(
            _first(record, "goal_cid", "subgoal_cid", "goal_id", "subgoal_id"),
            noun="task goal_cid",
        )
        if cid in by_cid or alias in aliases:
            raise TaskSourceIntegrityError("task CIDs and aliases must be unique")
        aliases.add(alias)
        identity = _task_identity_payload(record)
        immutable_id = content_identity(identity)
        other = immutable_ids.get(immutable_id)
        if other is not None and other != cid:
            raise TaskSourceIntegrityError(
                "task identities differ only by mutable status or aliases"
            )
        immutable_ids[immutable_id] = cid
        selected_status = _status(record.get("status") or "pending")
        body = dict(record)
        for name in _MUTABLE_IDENTITY_FIELDS:
            body.pop(name, None)
        identity_json = _canonical(identity, noun="task identity")
        body_json = _canonical(body, noun="task body")
        result.append(
            (
                cid,
                alias,
                goal,
                ordinal,
                selected_status,
                1,
                identity_json,
                body_json,
            )
        )
        by_cid[cid] = record
    return result, by_cid


def _assert_acyclic(edges: Mapping[str, Sequence[str]], noun: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise TaskSourceIntegrityError(f"{noun} graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in edges.get(node, ()):
            if dependency not in edges:
                raise TaskSourceIntegrityError(
                    f"{noun} graph references unknown node {dependency!r}"
                )
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(edges):
        visit(node)


def _dependencies(
    tasks: Mapping[str, Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    graph: dict[str, tuple[str, ...]] = {}
    for cid, record in tasks.items():
        raw = (
            record.get("dependency_task_cids")
            or record.get("depends_on")
            or record.get("dependencies")
            or record.get("blocking_task_cids")
            or ()
        )
        selected: list[str] = []
        for dependency in _values(raw):
            if isinstance(dependency, Mapping):
                dependency = _first(
                    dependency, "task_cid", "dependency_task_cid", "task_id", "id"
                )
            key = str(dependency or "").strip()
            resolved = aliases.get(key, key)
            resolved = _identifier(resolved, noun="dependency_task_cid")
            if resolved == cid:
                raise TaskSourceIntegrityError("task cannot depend on itself")
            if resolved not in tasks:
                raise TaskSourceIntegrityError(
                    f"task {cid!r} references unknown dependency {resolved!r}"
                )
            selected.append(resolved)
        dependencies = tuple(sorted(set(selected)))
        graph[cid] = dependencies
        result.extend((cid, dependency, "requires") for dependency in dependencies)
    if len(result) > MAX_EDGES:
        raise TaskSourceBoundsError("task dependency population exceeds its bound")
    _assert_acyclic(graph, "task dependency")
    return result


def _nested_rows(
    tasks: Mapping[str, Mapping[str, Any]],
) -> tuple[
    list[tuple[str, int, str, str]],
    list[tuple[str, int, str, str]],
    list[tuple[str, int, str, str]],
]:
    outputs: list[tuple[str, int, str, str]] = []
    validations: list[tuple[str, int, str, str]] = []
    acceptance: list[tuple[str, int, str, str]] = []
    for cid in sorted(tasks):
        record = tasks[cid]
        raw_outputs = record.get("outputs") or record.get("effects") or ()
        output_paths: set[str] = set()
        for ordinal, raw in enumerate(_values(raw_outputs)):
            item = (
                _as_mapping(raw, noun="task output")
                if isinstance(raw, Mapping) or hasattr(raw, "to_dict")
                else {"value": raw}
            )
            path = str(item.get("path") or item.get("fluent_id") or "")
            if path in output_paths:
                raise TaskSourceIntegrityError(
                    f"task {cid!r} contains duplicate output path {path!r}"
                )
            output_paths.add(path)
            outputs.append((cid, ordinal, path, _canonical(item, noun="task output")))

        raw_validations = record.get("validations") or record.get(
            "validation_commands"
        ) or ()
        for ordinal, raw in enumerate(_values(raw_validations)):
            if isinstance(raw, Mapping) or hasattr(raw, "to_dict"):
                item = _as_mapping(raw, noun="task validation")
                argv = item.get("argv")
                policy = {key: value for key, value in item.items() if key != "argv"}
            else:
                text = str(raw)
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    decoded = [text]
                if isinstance(decoded, Mapping):
                    item = dict(decoded)
                    argv = item.get("argv", [])
                    policy = {key: value for key, value in item.items() if key != "argv"}
                else:
                    argv = decoded if isinstance(decoded, list) else [decoded]
                    policy = {}
            validations.append(
                (
                    cid,
                    ordinal,
                    _canonical(argv, noun="validation argv"),
                    _canonical(policy, noun="validation policy"),
                )
            )

        raw_acceptance = record.get("acceptance") or record.get(
            "acceptance_criteria"
        ) or ()
        for ordinal, raw in enumerate(_values(raw_acceptance)):
            if isinstance(raw, Mapping) or hasattr(raw, "to_dict"):
                item = _as_mapping(raw, noun="acceptance criterion")
                criterion = str(
                    item.get("criterion")
                    or item.get("statement")
                    or item.get("id")
                    or item.get("criterion_key")
                    or ""
                )
                evidence = {
                    key: value
                    for key, value in item.items()
                    if key not in {"criterion", "statement"}
                }
            else:
                criterion = str(raw)
                evidence = {}
            acceptance.append(
                (
                    cid,
                    ordinal,
                    criterion,
                    _canonical(evidence, noun="acceptance evidence policy"),
                )
            )
    return outputs, validations, acceptance


def _artifact_rows(bundle: Mapping[str, Any]) -> list[tuple[str, str, int, str, str, str]]:
    records = [*bundle["ast"], *bundle["evidence"]]
    result: dict[str, tuple[str, str, int, str, str, str]] = {}
    for record in records:
        cid = _first(
            record,
            "artifact_cid",
            "evidence_cid",
            "symbol_cid",
            "scope_cid",
            "ast_cid",
            "content_id",
            "cid",
        )
        if not cid:
            continue
        cid = _identifier(cid, noun="artifact cid")
        digest = str(record.get("digest") or "")
        if not digest and ":sha256:" in cid:
            digest = cid.rsplit(":sha256:", 1)[-1]
        row = (
            cid,
            str(record.get("media_type") or "application/json"),
            int(record.get("byte_length") or 0),
            digest,
            str(record.get("storage_uri") or ""),
            _canonical(record, noun="artifact provenance"),
        )
        previous = result.get(cid)
        if previous is not None and previous != row:
            raise TaskSourceIntegrityError(f"conflicting artifact record {cid!r}")
        result[cid] = row
    return [result[key] for key in sorted(result)]


def _projection_identity(connection: Any) -> str:
    payload: dict[str, Any] = {
        "schema": DUCKDB_TASK_SOURCE_SCHEMA,
        "schema_version": DUCKDB_TASK_SOURCE_SCHEMA_VERSION,
    }
    for table in (
        "artifacts",
        "goals",
        "tasks",
        "task_dependencies",
        "task_outputs",
        "task_validations",
        "task_acceptance",
        "formal_plan_input_records",
        "formal_plan_input_metadata",
    ):
        columns = _TABLE_COLUMNS[table]
        identity_columns = (
            tuple(name for name in columns if name not in {"status", "revision"})
            if table == "tasks"
            else columns
        )
        selected = ", ".join(f'"{name}"' for name in identity_columns)
        order = ", ".join(f'"{name}"' for name in columns[:2])
        rows = connection.execute(
            f'SELECT {selected} FROM "{table}" ORDER BY {order}'
        ).fetchall()
        payload[table] = [list(row) for row in rows]
    return content_identity(payload)


def _insert_many(connection: Any, table: str, rows: Sequence[Sequence[Any]]) -> None:
    if not rows:
        return
    columns = _TABLE_COLUMNS[table]
    placeholders = ", ".join("?" for _ in columns)
    names = ", ".join(f'"{name}"' for name in columns)
    connection.executemany(
        f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
        rows,
    )


def _database_identity(metadata: Mapping[str, tuple[str, str]]) -> str:
    return content_identity(
        {
            "schema": _meta_value(metadata, "source_schema"),
            "schema_version": _meta_value(metadata, "schema_version"),
            "plan_root_cid": _meta_value(metadata, "plan_root_cid"),
            "projection_cid": _meta_value(metadata, "projection_cid"),
            "source_identity": _meta_value(metadata, "source_identity"),
        }
    )


def _validate_dynamic_state(
    connection: Any,
    metadata: Mapping[str, tuple[str, str]],
    initial_statuses: Mapping[str, Any],
) -> None:
    """Validate the bounded mutable state independently of static projection."""

    task_rows = connection.execute(
        "SELECT task_cid, status, revision FROM tasks ORDER BY task_cid"
    ).fetchall()
    if len(task_rows) > MAX_TASKS:
        raise TaskSourceIntegrityError("task population exceeds its state bound")
    current_tasks = {
        str(task_cid): (str(status), int(revision))
        for task_cid, status, revision in task_rows
    }
    if set(initial_statuses) != set(current_tasks):
        raise TaskSourceIntegrityError(
            "initial task status population does not match tasks"
        )
    for task_cid, status_value in initial_statuses.items():
        try:
            _status(status_value)
        except ValueError as exc:
            raise TaskSourceIntegrityError(
                f"task {task_cid!r} has an invalid initial status"
            ) from exc

    event_rows = connection.execute(
        "SELECT sequence, revision, task_cid, event_type, body_json "
        "FROM task_events ORDER BY sequence"
    ).fetchall()
    if len(event_rows) > MAX_EVENTS:
        raise TaskSourceIntegrityError("event population exceeds its state bound")
    sequences = [int(row[0]) for row in event_rows]
    if sequences != list(range(1, len(sequences) + 1)):
        raise TaskSourceIntegrityError("event sequence is not contiguous")
    cursor = int(_meta_value(metadata, "event_sequence"))
    global_revision = int(_meta_value(metadata, "revision"))
    if cursor != len(event_rows) or global_revision != len(event_rows) + 1:
        raise TaskSourceIntegrityError(
            "workflow revision/cursor does not match committed events"
        )
    status_chains: dict[str, list[Mapping[str, Any]]] = {
        task_cid: [] for task_cid in current_tasks
    }
    for sequence, revision, task_cid, event_type, body_json in event_rows:
        if int(revision) != int(sequence) + 1:
            raise TaskSourceIntegrityError(
                "event revisions are not contiguous and monotonic"
            )
        selected_task = str(task_cid)
        if selected_task not in current_tasks:
            raise TaskSourceIntegrityError("event references an unknown task")
        body = _decode_canonical(body_json, noun=f"event {sequence}")
        if not isinstance(body, Mapping):
            raise TaskSourceIntegrityError("event body must be a JSON object")
        if str(event_type) == "status_changed":
            status_chains[selected_task].append(body)
    for task_cid, chain in status_chains.items():
        expected_status = str(initial_statuses[task_cid])
        expected_revision = 1
        for event in chain:
            expected_revision += 1
            if (
                event.get("task_cid") != task_cid
                or event.get("previous_status") != expected_status
                or event.get("task_revision") != expected_revision
            ):
                raise TaskSourceIntegrityError(
                    f"task {task_cid!r} status event chain is corrupt"
                )
            try:
                expected_status = _status(event.get("status"))
            except ValueError as exc:
                raise TaskSourceIntegrityError(
                    f"task {task_cid!r} status event is invalid"
                ) from exc
        if current_tasks[task_cid] != (expected_status, expected_revision):
            raise TaskSourceIntegrityError(
                f"task {task_cid!r} status/revision has no valid event history"
            )


class DuckDBTaskSource:
    """One versioned, fenced DuckDB task-source projection."""

    compatibility_matrix: Final[Mapping[int, tuple[int, ...]]] = {1: (1,)}

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        expected_plan_root_cid: str = "",
        expected_repository_tree_id: str = "",
        writer_id: str = "local",
        fencing_token: int = 1,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self.database_path = Path(database_path).absolute()
        self.path = self.database_path
        self.expected_plan_root_cid = (
            _identifier(expected_plan_root_cid, noun="expected_plan_root_cid")
            if expected_plan_root_cid
            else ""
        )
        self.expected_repository_tree_id = (
            _identifier(
                expected_repository_tree_id,
                noun="expected_repository_tree_id",
            )
            if expected_repository_tree_id
            else ""
        )
        self.writer_id = _identifier(writer_id, noun="writer_id")
        if (
            isinstance(fencing_token, bool)
            or not isinstance(fencing_token, int)
            or fencing_token < 1
        ):
            raise ValueError("fencing_token must be a positive integer")
        self.fencing_token = fencing_token
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._lock_path = self.database_path.with_name(
            f".{self.database_path.name}.lock"
        )
        self._installing_path = self.database_path.with_name(
            f".{self.database_path.name}.installing"
        )
        self._recover_atomic_install()

    @staticmethod
    def available() -> bool:
        return duckdb_available()

    is_available = available

    @property
    def exists(self) -> bool:
        return self.database_path.is_file()

    def _recover_atomic_install(self) -> None:
        if not self._installing_path.exists():
            return
        with exclusive_file_lock(
            self._lock_path, timeout_seconds=self.lock_timeout_seconds
        ):
            if not self._installing_path.exists():
                return
            if self.database_path.exists():
                try:
                    self._installing_path.unlink()
                except OSError as exc:
                    raise TaskSourceIntegrityError(
                        "could not discard stale atomic-install file"
                    ) from exc
                return
            try:
                connection = _connect(self._installing_path, read_only=True)
                try:
                    metadata = self._validate_connection(
                        connection, require_complete=True, check_projection=True
                    )
                    if _meta_value(metadata, "installation_state") != "complete":
                        raise TaskSourceIntegrityError(
                            "atomic-install file is incomplete"
                        )
                finally:
                    connection.close()
            except Exception:
                try:
                    self._installing_path.unlink()
                except OSError:
                    pass
                return
            os.replace(self._installing_path, self.database_path)
            self._fsync_parent()

    def _fsync_parent(self) -> None:
        try:
            descriptor = os.open(self.database_path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def _read_connection(self) -> Iterator[tuple[Any, dict[str, tuple[str, str]]]]:
        if not self.database_path.is_file():
            raise TaskSourceIntegrityError("DuckDB task source is not installed")
        connection = _connect(self.database_path, read_only=True)
        try:
            metadata = self._validate_connection(
                connection, require_complete=True, check_projection=True
            )
            yield connection, metadata
        finally:
            connection.close()

    @contextmanager
    def _write_connection(
        self,
        *,
        writer_id: str | None = None,
        fencing_token: int | None = None,
    ) -> Iterator[tuple[Any, dict[str, tuple[str, str]]]]:
        selected_writer = _identifier(
            writer_id or self.writer_id, noun="writer_id"
        )
        selected_fence = self.fencing_token if fencing_token is None else fencing_token
        if (
            isinstance(selected_fence, bool)
            or not isinstance(selected_fence, int)
            or selected_fence < 1
        ):
            raise ValueError("fencing_token must be a positive integer")
        with exclusive_file_lock(
            self._lock_path, timeout_seconds=self.lock_timeout_seconds
        ):
            connection = _connect(self.database_path, read_only=False)
            try:
                connection.execute("BEGIN TRANSACTION")
                metadata = self._validate_connection(
                    connection, require_complete=True, check_projection=True
                )
                current_writer = _meta_value(metadata, "writer_id")
                current_fence = int(_meta_value(metadata, "writer_fence"))
                if selected_fence < current_fence:
                    raise TaskSourceConflictError("writer fencing token is stale")
                if selected_fence == current_fence and selected_writer != current_writer:
                    raise TaskSourceConflictError(
                        "writer fencing token belongs to another writer"
                    )
                if selected_fence > current_fence:
                    _set_metadata(connection, "writer_id", selected_writer)
                    _set_metadata(connection, "writer_fence", selected_fence)
                    metadata = _metadata(connection)
                yield connection, metadata
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                connection.close()

    def _validate_connection(
        self,
        connection: Any,
        *,
        require_complete: bool,
        check_projection: bool,
    ) -> dict[str, tuple[str, str]]:
        tables = _table_names(connection)
        expected_tables = set(_TABLE_COLUMNS)
        missing = expected_tables - tables
        if missing:
            raise TaskSourceIntegrityError(
                "DuckDB task source is partial; missing tables: "
                + ", ".join(sorted(missing))
            )
        unexpected = tables - expected_tables
        if unexpected:
            raise TaskSourceIntegrityError(
                "DuckDB task source contains unexpected tables: "
                + ", ".join(sorted(unexpected))
            )
        for table, columns in _TABLE_COLUMNS.items():
            actual = _table_columns(connection, table)
            if actual != columns:
                raise TaskSourceIntegrityError(
                    f"DuckDB task-source table {table!r} has an incompatible schema"
                )
        metadata = _metadata(connection)
        if _meta_value(metadata, "source_schema") != DUCKDB_TASK_SOURCE_SCHEMA:
            raise TaskSourceIntegrityError("unsupported DuckDB task-source schema")
        try:
            version = int(_meta_value(metadata, "schema_version"))
            revision = int(_meta_value(metadata, "revision"))
            event_sequence = int(_meta_value(metadata, "event_sequence"))
            writer_fence = int(_meta_value(metadata, "writer_fence"))
        except ValueError as exc:
            raise TaskSourceIntegrityError(
                "numeric workflow metadata is malformed"
            ) from exc
        if version != DUCKDB_TASK_SOURCE_SCHEMA_VERSION:
            raise TaskSourceIntegrityError(
                f"unsupported DuckDB task-source schema version {version}"
            )
        if revision < 1 or event_sequence < 0 or writer_fence < 1:
            raise TaskSourceIntegrityError(
                "workflow revision, cursor, or writer fence is invalid"
            )
        state = _meta_value(metadata, "installation_state")
        if require_complete and state != "complete":
            raise TaskSourceIntegrityError("DuckDB task source is only partially installed")
        initial_statuses = _decode_canonical(
            _meta_value(metadata, "initial_task_statuses"),
            noun="initial task statuses",
        )
        if not isinstance(initial_statuses, Mapping):
            raise TaskSourceIntegrityError(
                "initial task statuses must be a JSON object"
            )
        _validate_dynamic_state(connection, metadata, initial_statuses)
        plan_root = _meta_value(metadata, "plan_root_cid")
        repository_tree = _meta_value(metadata, "repository_tree_id")
        for key, value in (
            ("plan_root_cid", plan_root),
            ("repository_tree_id", repository_tree),
            ("writer_id", _meta_value(metadata, "writer_id")),
            ("formal_plan_id", _meta_value(metadata, "formal_plan_id")),
            ("source_identity", _meta_value(metadata, "source_identity")),
            ("materialization_receipt_cid", _meta_value(metadata, "materialization_receipt_cid")),
        ):
            try:
                _identifier(value, noun=key)
            except TaskSourceInjectionError as exc:
                raise TaskSourceIntegrityError(
                    f"workflow metadata {key!r} is malformed"
                ) from exc
        if self.expected_plan_root_cid and plan_root != self.expected_plan_root_cid:
            raise TaskSourceIntegrityError("DuckDB task source has a foreign plan root")
        if (
            self.expected_repository_tree_id
            and repository_tree != self.expected_repository_tree_id
        ):
            raise TaskSourceIntegrityError(
                "DuckDB task source has a foreign repository root"
            )
        if check_projection:
            claimed = _meta_value(metadata, "projection_cid")
            if claimed != _projection_identity(connection):
                raise TaskSourceIntegrityError(
                    "DuckDB task-source projection identity does not match"
                )
        return metadata

    def materialize(
        self,
        source: Any,
        *,
        repository_tree_id: str = "",
        plan_root_cid: str = "",
        receipt: Mapping[str, Any] | None = None,
        expected_absent: bool = False,
        writer_id: str | None = None,
        fencing_token: int | None = None,
        fault_injector: Any | None = None,
    ) -> Mapping[str, Any]:
        """Atomically install one admitted formal input or prompt goal graph.

        Replaying the identical plan is a read-only no-op.  An existing
        different plan is never replaced by this method.
        """

        _duckdb_module()
        formal_source, graph = _source_and_projection(
            source, repository_tree_id=repository_tree_id
        )
        bundle = _section_bundle(formal_source)
        source_tree_id = str(bundle.get("repository_tree_id") or "")
        if (
            repository_tree_id
            and source_tree_id
            and repository_tree_id != source_tree_id
        ):
            raise TaskSourceIntegrityError(
                "supplied repository root disagrees with formal plan input"
            )
        tree_id = (
            repository_tree_id
            or source_tree_id
        )
        tree_id = _identifier(tree_id, noun="repository_tree_id")
        bundle["repository_tree_id"] = tree_id
        # Check projection identities before formal compilation.  Otherwise a
        # duplicated status-derived task can first manifest as a secondary
        # formal effect/dependency collision and obscure the identity failure.
        _task_rows(bundle, graph)
        compile_result = FormalPlanCompiler().compile(formal_source)
        if (
            compile_result.status is not CompilationStatus.COMPILED
            or compile_result.plan is None
        ):
            diagnostics = "; ".join(item.message for item in compile_result.issues[:5])
            raise TaskSourceIntegrityError(
                "formal plan input did not compile successfully"
                + (f": {diagnostics}" if diagnostics else "")
            )
        claimed_root = (
            str(graph.get("content_id") or "")
            if graph is not None
            else str(
                formal_source.get("plan_root_cid")
                or formal_source.get("plan_cid")
                or ""
            )
        )
        if plan_root_cid and claimed_root and plan_root_cid != claimed_root:
            raise TaskSourceIntegrityError(
                "supplied plan root disagrees with the canonical source root"
            )
        root = plan_root_cid or claimed_root or compile_result.plan_id
        root = _identifier(root, noun="plan_root_cid")
        if self.expected_plan_root_cid and root != self.expected_plan_root_cid:
            raise TaskSourceIntegrityError("materialization has a foreign plan root")
        if (
            self.expected_repository_tree_id
            and tree_id != self.expected_repository_tree_id
        ):
            raise TaskSourceIntegrityError(
                "materialization has a foreign repository root"
            )

        with exclusive_file_lock(
            self._lock_path, timeout_seconds=self.lock_timeout_seconds
        ):
            if self.database_path.exists():
                if expected_absent:
                    raise TaskSourceConflictError("DuckDB task source already exists")
                with self._read_connection() as (_connection, metadata):
                    if (
                        _meta_value(metadata, "plan_root_cid") != root
                        or _meta_value(metadata, "source_identity")
                        != compile_result.source_identity
                    ):
                        raise TaskSourceConflictError(
                            "existing task source contains a different task population"
                        )
                    existing = self.snapshot()
                    return {
                        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
                        "receipt_cid": _meta_value(
                            metadata, "materialization_receipt_cid"
                        ),
                        "plan_root_cid": root,
                        "projection_cid": existing.projection_cid,
                        "revision": existing.revision,
                        "changed": False,
                        "replayed": True,
                    }
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            if self._installing_path.exists():
                self._installing_path.unlink()
            selected_writer = _identifier(
                writer_id or self.writer_id, noun="writer_id"
            )
            selected_fence = (
                self.fencing_token if fencing_token is None else fencing_token
            )
            if (
                isinstance(selected_fence, bool)
                or not isinstance(selected_fence, int)
                or selected_fence < 1
            ):
                raise ValueError("fencing_token must be a positive integer")
            connection = _connect(self._installing_path, read_only=False)
            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(_SCHEMA_SQL)
                initial_metadata = {
                    "source_schema": DUCKDB_TASK_SOURCE_SCHEMA,
                    "schema_version": str(DUCKDB_TASK_SOURCE_SCHEMA_VERSION),
                    "installation_state": "preparing",
                    "plan_root_cid": root,
                    "repository_tree_id": tree_id,
                    "projection_cid": "pending",
                    "formal_plan_id": compile_result.plan_id,
                    "source_identity": compile_result.source_identity,
                    "revision": "1",
                    "event_sequence": "0",
                    "writer_id": selected_writer,
                    "writer_fence": str(selected_fence),
                    "materialization_receipt_cid": "pending",
                    "initial_task_statuses": "pending",
                }
                for key, value in initial_metadata.items():
                    _set_metadata(connection, key, value)

                goal_rows = _goal_rows(bundle, graph)
                task_rows, task_records = _task_rows(bundle, graph)
                goal_cids = {row[0] for row in goal_rows}
                goal_aliases = {row[1]: row[0] for row in goal_rows}
                normalized_tasks: dict[str, dict[str, Any]] = {}
                normalized_task_rows = []
                task_aliases: dict[str, str] = {}
                for row in task_rows:
                    cid, alias, goal, *rest = row
                    resolved_goal = goal_aliases.get(goal, goal)
                    if resolved_goal not in goal_cids:
                        raise TaskSourceIntegrityError(
                            f"task {cid!r} references unknown goal {goal!r}"
                        )
                    normalized_task_rows.append(
                        (cid, alias, resolved_goal, *rest)
                    )
                    normalized_tasks[cid] = task_records[cid]
                    task_aliases[alias] = cid
                    if alias != cid and alias in task_records:
                        raise TaskSourceIntegrityError(
                            "task alias collides with another task CID"
                        )
                dependencies = _dependencies(normalized_tasks, task_aliases)
                outputs, validations, acceptance = _nested_rows(normalized_tasks)
                artifacts = _artifact_rows(bundle)
                _set_metadata(
                    connection,
                    "initial_task_statuses",
                    _canonical(
                        {row[0]: row[4] for row in normalized_task_rows},
                        noun="initial task statuses",
                    ),
                )
                formal_rows = [
                    (
                        section,
                        content_identity(record),
                        _canonical(record, noun=f"formal {section} record"),
                    )
                    for section in (
                        "objectives",
                        "tasks",
                        "ast",
                        "policies",
                        "leases",
                        "evidence",
                    )
                    for record in bundle[section]
                ]
                _insert_many(connection, "artifacts", artifacts)
                _insert_many(connection, "goals", goal_rows)
                _insert_many(connection, "tasks", normalized_task_rows)
                _insert_many(connection, "task_dependencies", dependencies)
                _insert_many(connection, "task_outputs", outputs)
                _insert_many(connection, "task_validations", validations)
                _insert_many(connection, "task_acceptance", acceptance)
                _insert_many(connection, "formal_plan_input_records", formal_rows)
                _insert_many(
                    connection,
                    "formal_plan_input_metadata",
                    (
                        ("repository_tree_id", tree_id),
                        ("schema", FORMAL_PLAN_INPUT_SCHEMA),
                        ("source_identity", compile_result.source_identity),
                    ),
                )
                if callable(fault_injector):
                    fault_injector("rows_written")

                independently_compiled = FormalPlanCompiler().compile_duckdb(connection)
                if (
                    independently_compiled.status is not CompilationStatus.COMPILED
                    or independently_compiled.plan_id != compile_result.plan_id
                    or independently_compiled.source_identity
                    != compile_result.source_identity
                ):
                    raise TaskSourceIntegrityError(
                        "independent DuckDB recompilation disagrees with original plan"
                    )
                projection_cid = _projection_identity(connection)
                receipt_body = {
                    "schema": MATERIALIZATION_RECEIPT_SCHEMA,
                    "plan_root_cid": root,
                    "projection_cid": projection_cid,
                    "formal_plan_id": compile_result.plan_id,
                    "source_identity": compile_result.source_identity,
                    "repository_tree_id": tree_id,
                    "revision": 1,
                    "goal_count": len(goal_rows),
                    "task_count": len(task_rows),
                    "dependency_count": len(dependencies),
                    "receipt": dict(receipt or {}),
                }
                receipt_cid = content_identity(receipt_body)
                _insert_many(
                    connection,
                    "materialization_receipts",
                    ((receipt_cid, root, 1, _canonical(receipt_body)),),
                )
                _set_metadata(connection, "projection_cid", projection_cid)
                _set_metadata(
                    connection, "materialization_receipt_cid", receipt_cid
                )
                _set_metadata(connection, "installation_state", "complete")
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                connection.close()
                try:
                    self._installing_path.unlink()
                except OSError:
                    pass
                raise
            else:
                connection.close()
            if callable(fault_injector):
                fault_injector("before_install")
            os.replace(self._installing_path, self.database_path)
            self._fsync_parent()
        return {
            **receipt_body,
            "receipt_cid": receipt_cid,
            "changed": True,
            "replayed": False,
        }

    install = materialize

    def materialize_derived_runtime(
        self,
        source: Any,
        *,
        formal_plan_id: str,
        source_identity: str,
        parallel_plan_digest: str,
        admission_receipt_cid: str,
        repository_tree_id: str = "",
        plan_root_cid: str = "",
        receipt: Mapping[str, Any] | None = None,
        protected_anchors: Sequence[str] = DEFAULT_DERIVED_PROTECTED_ANCHORS,
        writer_id: str | None = None,
        fencing_token: int | None = None,
        fault_injector: Any | None = None,
    ) -> Mapping[str, Any]:
        """Admit compiled derived work into this separate DuckDB source.

        Generated work may enter only after independent formal-plan
        compilation, structural admission, and parallel-plan compilation.
        The four gate identities are re-checked against a fresh formal
        recompile before install.  Identical population replay is a no-op.
        Seed-board anchors cannot appear as task outputs, and the projection
        is labeled ``source_role=derived_runtime`` so it cannot be confused
        with the operator-owned seed board.
        """

        if not (
            str(formal_plan_id or "").strip()
            and str(source_identity or "").strip()
            and str(parallel_plan_digest or "").strip()
            and str(admission_receipt_cid or "").strip()
        ):
            raise TaskSourceIntegrityError(
                "derived runtime materialization requires independent "
                "plan/admission/parallel compilation identities"
            )
        formal_plan_id = _identifier(formal_plan_id, noun="formal_plan_id")
        source_identity = _identifier(source_identity, noun="source_identity")
        parallel_plan_digest = _identifier(
            parallel_plan_digest, noun="parallel_plan_digest"
        )
        admission_receipt_cid = _identifier(
            admission_receipt_cid, noun="admission_receipt_cid"
        )

        # Reject candidate self-authorization / completion claims on the source.
        if isinstance(source, Mapping):
            for forbidden in (
                "completion_authority",
                "mutation_authority",
                "seed_board_edit",
                "mutate_seed_board",
                "threshold_lower_authority",
                "self_authorization",
                "mark_complete",
            ):
                if source.get(forbidden) not in (None, False, "", 0):
                    raise TaskSourceIntegrityError(
                        f"derived source cannot claim {forbidden}"
                    )
            if source.get("source_role") not in (
                None,
                "",
                DERIVED_RUNTIME_SOURCE_ROLE,
            ):
                raise TaskSourceIntegrityError(
                    "derived source_role must be derived_runtime"
                )

        formal_source, graph = _source_and_projection(
            source, repository_tree_id=repository_tree_id
        )
        # Refuse populations that edit protected anchors.
        bundle = _section_bundle(formal_source)
        _task_rows(bundle, graph)
        for record in bundle.get("tasks") or ():
            if not isinstance(record, Mapping):
                continue
            for effect in record.get("effects") or ():
                if not isinstance(effect, Mapping):
                    continue
                path = str(effect.get("path") or "").replace("\\", "/").strip("/")
                for anchor in protected_anchors:
                    target = str(anchor).replace("\\", "/").strip("/")
                    if path and (
                        path == target or path.startswith(target + "/")
                    ):
                        raise TaskSourceIntegrityError(
                            "derived runtime population targets a protected "
                            f"anchor path: {path}"
                        )

        compile_result = FormalPlanCompiler().compile(formal_source)
        if (
            compile_result.status is not CompilationStatus.COMPILED
            or compile_result.plan is None
        ):
            diagnostics = "; ".join(
                item.message for item in compile_result.issues[:5]
            )
            raise TaskSourceIntegrityError(
                "derived runtime formal plan input did not compile successfully"
                + (f": {diagnostics}" if diagnostics else "")
            )
        if (
            compile_result.plan_id != formal_plan_id
            or compile_result.source_identity != source_identity
        ):
            raise TaskSourceIntegrityError(
                "derived runtime formal plan identity does not match the "
                "independent compilation gates"
            )

        derived_receipt = {
            **dict(receipt or {}),
            "schema": DERIVED_RUNTIME_MATERIALIZATION_SCHEMA,
            "source_role": DERIVED_RUNTIME_SOURCE_ROLE,
            "mutates_seed_board": False,
            "completion_authority": False,
            "mutation_authority": False,
            "seed_board_edit": False,
            "threshold_lower_authority": False,
            "self_authorization": False,
            "formal_plan_id": formal_plan_id,
            "source_identity": source_identity,
            "parallel_plan_digest": parallel_plan_digest,
            "admission_receipt_cid": admission_receipt_cid,
        }
        result = self.materialize(
            source,
            repository_tree_id=repository_tree_id,
            plan_root_cid=plan_root_cid,
            receipt=derived_receipt,
            writer_id=writer_id,
            fencing_token=fencing_token,
            fault_injector=fault_injector,
        )
        # Stamp durable role metadata when the database was installed or
        # already held the identical population (replay no-op).
        if self.database_path.exists():
            with exclusive_file_lock(
                self._lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                connection = _connect(self.database_path, read_only=False)
                try:
                    connection.execute("BEGIN TRANSACTION")
                    metadata = _metadata(connection)
                    live_writer = _meta_value(metadata, "writer_id")
                    live_fence_raw = _meta_value(metadata, "writer_fence")
                    try:
                        live_fence = int(live_fence_raw)
                    except (TypeError, ValueError):
                        live_fence = 0
                    selected_writer = _identifier(
                        writer_id or self.writer_id, noun="writer_id"
                    )
                    selected_fence = (
                        self.fencing_token
                        if fencing_token is None
                        else fencing_token
                    )
                    if live_writer and live_writer != selected_writer:
                        raise TaskSourceConflictError(
                            "stale writer cannot stamp derived runtime metadata"
                        )
                    if (
                        live_fence
                        and isinstance(selected_fence, int)
                        and selected_fence < live_fence
                    ):
                        raise TaskSourceConflictError(
                            "stale fencing token cannot stamp derived runtime metadata"
                        )
                    for key, value in (
                        ("source_role", DERIVED_RUNTIME_SOURCE_ROLE),
                        ("mutates_seed_board", "false"),
                        ("parallel_plan_digest", parallel_plan_digest),
                        ("admission_receipt_cid", admission_receipt_cid),
                        ("derived_runtime", "true"),
                    ):
                        _set_metadata(connection, key, value)
                    connection.execute("COMMIT")
                except BaseException:
                    try:
                        connection.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                finally:
                    connection.close()
        return {
            **dict(result),
            "source_role": DERIVED_RUNTIME_SOURCE_ROLE,
            "mutates_seed_board": False,
            "formal_plan_id": formal_plan_id,
            "source_identity": source_identity,
            "parallel_plan_digest": parallel_plan_digest,
            "admission_receipt_cid": admission_receipt_cid,
            "derived_runtime": True,
        }

    @classmethod
    def from_formal_plan_input(
        cls,
        database_path: str | os.PathLike[str],
        source: Any,
        **kwargs: Any,
    ) -> "DuckDBTaskSource":
        materialize_keys = {
            "repository_tree_id",
            "plan_root_cid",
            "receipt",
            "expected_absent",
            "fault_injector",
        }
        constructor = {
            key: value for key, value in kwargs.items() if key not in materialize_keys
        }
        materialize = {
            key: value for key, value in kwargs.items() if key in materialize_keys
        }
        result = cls(database_path, **constructor)
        result.materialize(source, **materialize)
        return result

    def snapshot(self) -> TaskSourceSnapshot:
        with self._read_connection() as (connection, metadata):
            goal_count = int(connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0])
            task_count = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
            dependency_count = int(
                connection.execute("SELECT COUNT(*) FROM task_dependencies").fetchone()[0]
            )
            nonterminal = int(
                connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status NOT IN (?, ?, ?, ?, ?)",
                    sorted(_TERMINAL_STATUSES),
                ).fetchone()[0]
            )
            return TaskSourceSnapshot(
                source_schema=_meta_value(metadata, "source_schema"),
                schema_version=int(_meta_value(metadata, "schema_version")),
                plan_root_cid=_meta_value(metadata, "plan_root_cid"),
                repository_tree_id=_meta_value(metadata, "repository_tree_id"),
                projection_cid=_meta_value(metadata, "projection_cid"),
                formal_plan_id=_meta_value(metadata, "formal_plan_id"),
                source_identity=_meta_value(metadata, "source_identity"),
                revision=int(_meta_value(metadata, "revision")),
                event_cursor=int(_meta_value(metadata, "event_sequence")),
                goal_count=goal_count,
                task_count=task_count,
                dependency_count=dependency_count,
                terminal=nonterminal == 0,
            )

    def _task_records(self, connection: Any, rows: Sequence[Sequence[Any]]) -> tuple[TaskRecord, ...]:
        if not rows:
            return ()
        cids = [str(row[0]) for row in rows]
        placeholders = ", ".join("?" for _ in cids)
        dependency_rows = connection.execute(
            "SELECT task_cid, dependency_task_cid FROM task_dependencies "
            f"WHERE task_cid IN ({placeholders}) "
            "ORDER BY task_cid, dependency_task_cid",
            cids,
        ).fetchall()
        dependencies: dict[str, list[str]] = {cid: [] for cid in cids}
        for task_cid, dependency in dependency_rows:
            dependencies[str(task_cid)].append(str(dependency))
        result: list[TaskRecord] = []
        for (
            task_cid,
            task_alias,
            goal_cid,
            ordinal,
            status,
            revision,
            _identity_json,
            body_json,
        ) in rows:
            body = _decode_canonical(body_json, noun=f"task {task_cid} body")
            if not isinstance(body, Mapping):
                raise TaskSourceIntegrityError("task body must be a JSON object")
            result.append(
                TaskRecord(
                    task_cid=str(task_cid),
                    task_alias=str(task_alias),
                    goal_cid=str(goal_cid),
                    ordinal=int(ordinal),
                    status=str(status),
                    revision=int(revision),
                    body=dict(body),
                    dependencies=tuple(dependencies[str(task_cid)]),
                )
            )
        return tuple(result)

    def list_tasks(
        self,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        selected_limit = _positive_limit(limit)
        with self._read_connection() as (connection, metadata):
            revision = int(_meta_value(metadata, "revision"))
            plan_root = _meta_value(metadata, "plan_root_cid")
            offset = _cursor_decode(cursor, plan_root, revision) if cursor else 0
            statuses: tuple[str, ...]
            if status is None:
                statuses = ()
            elif isinstance(status, str):
                statuses = (_status(status),)
            else:
                statuses = tuple(sorted({_status(item) for item in status}))
                if not statuses:
                    raise ValueError("status filter must not be empty")
            parameters: list[Any] = []
            where = ""
            if statuses:
                where = "WHERE status IN (" + ", ".join("?" for _ in statuses) + ")"
                parameters.extend(statuses)
            parameters.extend((selected_limit + 1, offset))
            rows = connection.execute(
                "SELECT task_cid, task_alias, goal_cid, ordinal, status, "
                "revision, identity_json, body_json FROM tasks "
                f"{where} ORDER BY ordinal, task_cid LIMIT ? OFFSET ?",
                parameters,
            ).fetchall()
            has_more = len(rows) > selected_limit
            rows = rows[:selected_limit]
            tasks = self._task_records(connection, rows)
            next_cursor = (
                _cursor_encode(plan_root, revision, offset + len(rows))
                if has_more
                else ""
            )
            return TaskPage(tasks=tasks, revision=revision, next_cursor=next_cursor)

    def get_task(
        self, task_cid_or_alias: str | TaskRecord | Mapping[str, Any]
    ) -> TaskRecord | None:
        key = _task_key(task_cid_or_alias)
        with self._read_connection() as (connection, _metadata_value):
            rows = connection.execute(
                "SELECT task_cid, task_alias, goal_cid, ordinal, status, "
                "revision, identity_json, body_json FROM tasks "
                "WHERE task_cid = ? OR task_alias = ? ORDER BY task_cid LIMIT 2",
                [key, key],
            ).fetchall()
            if len(rows) > 1:
                raise TaskSourceIntegrityError("task CID/alias lookup is ambiguous")
            records = self._task_records(connection, rows)
            return records[0] if records else None

    get = get_task

    def ready_tasks(
        self,
        completed_ids: Iterable[str] = (),
        blocked_ids: Iterable[str] = (),
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        selected_limit = _positive_limit(limit)
        completed = {
            _identifier(item, noun="completed task identifier")
            for item in completed_ids
        }
        blocked = {
            _identifier(item, noun="blocked task identifier") for item in blocked_ids
        }
        if completed & blocked:
            raise ValueError("completed_ids and blocked_ids must be disjoint")
        with self._read_connection() as (connection, metadata):
            rows = connection.execute(
                "SELECT task_cid, task_alias, goal_cid, ordinal, status, "
                "revision, identity_json, body_json FROM tasks ORDER BY ordinal, task_cid"
            ).fetchall()
            if len(rows) > MAX_TASKS:
                raise TaskSourceBoundsError("task population exceeds readiness bound")
            tasks = self._task_records(connection, rows)
            by_id = {item.task_cid: item for item in tasks}
            aliases = {item.task_alias: item.task_cid for item in tasks}
            resolved_completed = {aliases.get(item, item) for item in completed}
            resolved_blocked = {aliases.get(item, item) for item in blocked}
            unknown = (resolved_completed | resolved_blocked) - set(by_id)
            if unknown:
                raise TaskSourceIntegrityError(
                    "readiness input references unknown tasks: "
                    + ", ".join(sorted(unknown))
                )
            durable_completed = {
                item.task_cid
                for item in tasks
                if item.status in _COMPLETED_STATUSES
            }
            durable_blocked = {
                item.task_cid for item in tasks if item.status == "blocked"
            }
            satisfied = durable_completed | resolved_completed
            unavailable = durable_blocked | resolved_blocked
            ready = tuple(
                item
                for item in tasks
                if item.status in _READY_STATUSES
                and item.task_cid not in unavailable
                and set(item.dependencies).issubset(satisfied)
                and not set(item.dependencies).intersection(unavailable)
            )[:selected_limit]
            return TaskPage(
                tasks=ready,
                revision=int(_meta_value(metadata, "revision")),
            )

    readiness = ready_tasks

    def acquire_writer(
        self,
        writer_id: str,
        *,
        expected_fencing_token: int | None = None,
    ) -> WriterFence:
        selected = _identifier(writer_id, noun="writer_id")
        with exclusive_file_lock(
            self._lock_path, timeout_seconds=self.lock_timeout_seconds
        ):
            connection = _connect(self.database_path, read_only=False)
            try:
                connection.execute("BEGIN TRANSACTION")
                metadata = self._validate_connection(
                    connection, require_complete=True, check_projection=True
                )
                current = int(_meta_value(metadata, "writer_fence"))
                if (
                    expected_fencing_token is not None
                    and expected_fencing_token != current
                ):
                    raise TaskSourceConflictError("writer fence CAS is stale")
                new_fence = current + 1
                _set_metadata(connection, "writer_id", selected)
                _set_metadata(connection, "writer_fence", new_fence)
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                connection.close()
        self.writer_id = selected
        self.fencing_token = new_fence
        return WriterFence(
            writer_id=selected,
            fencing_token=new_fence,
            revision=int(_meta_value(metadata, "revision")),
        )

    def compare_and_set_status(
        self,
        task_cid_or_alias: str | TaskRecord | Mapping[str, Any],
        expected_revision: int,
        status: str,
        receipt: Mapping[str, Any] | None = None,
        *,
        writer_id: str | None = None,
        fencing_token: int | None = None,
    ) -> CASResult:
        key = _task_key(task_cid_or_alias)
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 1
        ):
            raise ValueError("expected_revision must be a positive integer")
        new_status = _status(status)
        receipt_record = _as_mapping(receipt or {}, noun="CAS receipt")
        with self._write_connection(
            writer_id=writer_id, fencing_token=fencing_token
        ) as (connection, metadata):
            rows = connection.execute(
                "SELECT task_cid, task_alias, goal_cid, ordinal, status, "
                "revision, identity_json, body_json FROM tasks "
                "WHERE task_cid = ? OR task_alias = ? ORDER BY task_cid LIMIT 2",
                [key, key],
            ).fetchall()
            if not rows:
                raise KeyError(key)
            if len(rows) > 1:
                raise TaskSourceIntegrityError("task CID/alias lookup is ambiguous")
            row = rows[0]
            task_cid = str(row[0])
            previous_status = str(row[4])
            current_task_revision = int(row[5])
            if current_task_revision != expected_revision:
                raise TaskSourceConflictError("task revision CAS is stale")
            if previous_status == new_status:
                task = self._task_records(connection, rows)[0]
                return CASResult(
                    task=task,
                    previous_status=previous_status,
                    revision=int(_meta_value(metadata, "revision")),
                    event_cursor=int(_meta_value(metadata, "event_sequence")),
                    changed=False,
                    receipt_cid="",
                )
            global_revision = int(_meta_value(metadata, "revision")) + 1
            event_sequence = int(_meta_value(metadata, "event_sequence")) + 1
            if event_sequence > MAX_EVENTS:
                raise TaskSourceBoundsError(
                    "event population exceeds the durable state bound"
                )
            new_task_revision = current_task_revision + 1
            event_body = {
                "schema": TASK_SOURCE_CAS_SCHEMA,
                "task_cid": task_cid,
                "previous_status": previous_status,
                "status": new_status,
                "task_revision": new_task_revision,
                "revision": global_revision,
                "receipt": receipt_record,
            }
            event_cid = content_identity(event_body)
            receipt_cid = content_identity(
                {
                    "namespace": "task-status-receipt",
                    "event_cid": event_cid,
                    "receipt": receipt_record,
                }
            )
            connection.execute(
                "UPDATE tasks SET status = ?, revision = ? "
                "WHERE task_cid = ? AND revision = ?",
                [new_status, new_task_revision, task_cid, current_task_revision],
            )
            if connection.execute(
                "SELECT revision FROM tasks WHERE task_cid = ?", [task_cid]
            ).fetchone()[0] != new_task_revision:
                raise TaskSourceConflictError("task revision changed concurrently")
            _insert_many(
                connection,
                "task_events",
                (
                    (
                        event_cid,
                        event_sequence,
                        global_revision,
                        task_cid,
                        "status_changed",
                        _canonical(event_body, noun="status event"),
                    ),
                ),
            )
            _set_metadata(connection, "revision", global_revision)
            _set_metadata(connection, "event_sequence", event_sequence)
            updated_rows = connection.execute(
                "SELECT task_cid, task_alias, goal_cid, ordinal, status, "
                "revision, identity_json, body_json FROM tasks WHERE task_cid = ?",
                [task_cid],
            ).fetchall()
            task = self._task_records(connection, updated_rows)[0]
            return CASResult(
                task=task,
                previous_status=previous_status,
                revision=global_revision,
                event_cursor=event_sequence,
                changed=True,
                receipt_cid=receipt_cid,
            )

    cas_status = compare_and_set_status

    def append_event(
        self,
        event: Mapping[str, Any],
        lease: Mapping[str, Any] | None = None,
        fence: int | None = None,
        *,
        writer_id: str | None = None,
    ) -> Mapping[str, Any]:
        body = _as_mapping(event, noun="task event")
        task_key = _identifier(
            _first(body, "task_cid", "task_id"), noun="event task identifier"
        )
        event_type = _identifier(
            _first(body, "event_type", "type", "kind"), noun="event_type"
        )
        if event_type == "status_changed":
            raise ValueError(
                "status_changed is reserved for compare_and_set_status"
            )
        lease_record = _as_mapping(lease or {}, noun="event lease")
        selected_fence = fence
        if selected_fence is None and lease_record:
            raw_fence = lease_record.get("fencing_token", lease_record.get("fence"))
            if raw_fence is not None:
                selected_fence = int(raw_fence)
        selected_fence = self.fencing_token if selected_fence is None else selected_fence
        with self._write_connection(
            writer_id=writer_id, fencing_token=selected_fence
        ) as (connection, metadata):
            task_row = connection.execute(
                "SELECT task_cid FROM tasks WHERE task_cid = ? OR task_alias = ? "
                "ORDER BY task_cid LIMIT 2",
                [task_key, task_key],
            ).fetchall()
            if len(task_row) != 1:
                raise TaskSourceIntegrityError(
                    "event must reference exactly one existing task"
                )
            task_cid = str(task_row[0][0])
            current_revision = int(_meta_value(metadata, "revision"))
            sequence = int(_meta_value(metadata, "event_sequence")) + 1
            if sequence > MAX_EVENTS:
                raise TaskSourceBoundsError(
                    "event population exceeds the durable state bound"
                )
            persisted = {
                **body,
                "task_cid": task_cid,
                "event_type": event_type,
                "lease": lease_record,
            }
            event_cid = str(body.get("event_cid") or content_identity(persisted))
            event_cid = _identifier(event_cid, noun="event_cid")
            existing = connection.execute(
                "SELECT sequence, revision, task_cid, event_type, body_json "
                "FROM task_events WHERE event_cid = ?",
                [event_cid],
            ).fetchone()
            encoded = _canonical(persisted, noun="task event")
            if existing is not None:
                if (
                    str(existing[2]) != task_cid
                    or str(existing[3]) != event_type
                    or str(existing[4]) != encoded
                ):
                    raise TaskSourceConflictError(
                        "event CID was replayed with different content"
                    )
                return {
                    "event_cid": event_cid,
                    "sequence": int(existing[0]),
                    "revision": int(existing[1]),
                    "changed": False,
                }
            new_revision = current_revision + 1
            _insert_many(
                connection,
                "task_events",
                (
                    (
                        event_cid,
                        sequence,
                        new_revision,
                        task_cid,
                        event_type,
                        encoded,
                    ),
                ),
            )
            _set_metadata(connection, "revision", new_revision)
            _set_metadata(connection, "event_sequence", sequence)
            return {
                "event_cid": event_cid,
                "sequence": sequence,
                "revision": new_revision,
                "changed": True,
            }

    def events(self, cursor: int = 0, limit: int = DEFAULT_QUERY_LIMIT) -> EventPage:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("event cursor must be a non-negative integer")
        selected_limit = _positive_limit(limit, maximum=MAX_EVENTS_PER_PAGE)
        with self._read_connection() as (connection, metadata):
            current = int(_meta_value(metadata, "event_sequence"))
            if cursor > current:
                raise TaskSourceConflictError("event cursor is ahead of the source")
            rows = connection.execute(
                "SELECT event_cid, sequence, revision, task_cid, event_type, body_json "
                "FROM task_events WHERE sequence > ? ORDER BY sequence LIMIT ?",
                [cursor, selected_limit],
            ).fetchall()
            events: list[dict[str, Any]] = []
            next_cursor = cursor
            for event_cid, sequence, revision, task_cid, event_type, body_json in rows:
                body = _decode_canonical(body_json, noun=f"event {event_cid}")
                events.append(
                    {
                        "event_cid": str(event_cid),
                        "sequence": int(sequence),
                        "revision": int(revision),
                        "task_cid": str(task_cid),
                        "event_type": str(event_type),
                        "body": body,
                    }
                )
                next_cursor = int(sequence)
            return EventPage(
                events=tuple(events),
                cursor=next_cursor,
                revision=int(_meta_value(metadata, "revision")),
            )

    def watch(
        self,
        cursor: int = 0,
        timeout: float = 0.0,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> EventPage:
        if timeout < 0 or timeout > MAX_WATCH_SECONDS:
            raise TaskSourceBoundsError(
                f"watch timeout must be between 0 and {MAX_WATCH_SECONDS:g} seconds"
            )
        deadline = time.monotonic() + float(timeout)
        while True:
            page = self.events(cursor=cursor, limit=limit)
            if page.events:
                return page
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return EventPage(
                    events=(),
                    cursor=page.cursor,
                    revision=page.revision,
                    timed_out=True,
                )
            time.sleep(min(0.05, remaining))

    def query(
        self,
        table: str,
        *,
        cursor: int = 0,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> tuple[Mapping[str, Any], ...]:
        """Run one closed, parameter-free bounded projection query.

        Table names are selected from a constant allowlist and never accepted
        as SQL fragments.  Values are returned as plain mappings.
        """

        if table not in _QUERY_TABLES:
            raise TaskSourceInjectionError("query table is not in the closed allowlist")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("query cursor must be a non-negative integer")
        selected_limit = _positive_limit(limit)
        columns = _TABLE_COLUMNS[table]
        selected = ", ".join(f'"{name}"' for name in columns)
        order = ", ".join(f'"{name}"' for name in columns[:2])
        with self._read_connection() as (connection, _metadata_value):
            rows = connection.execute(
                f'SELECT {selected} FROM "{table}" '
                f"ORDER BY {order} LIMIT ? OFFSET ?",
                [selected_limit, cursor],
            ).fetchall()
        return tuple(dict(zip(columns, row)) for row in rows)

    def recompile_formal_plan(self) -> Any:
        with self._read_connection() as (connection, metadata):
            result = FormalPlanCompiler().compile_duckdb(connection)
            if (
                result.status is not CompilationStatus.COMPILED
                or result.plan_id != _meta_value(metadata, "formal_plan_id")
                or result.source_identity != _meta_value(metadata, "source_identity")
            ):
                raise TaskSourceIntegrityError(
                    "independent formal plan recompilation disagrees with metadata"
                )
            return result

    recompile = recompile_formal_plan

    def validate_integrity(self) -> IntegrityReport:
        with self._read_connection() as (connection, metadata):
            task_rows = connection.execute(
                "SELECT task_cid, task_alias, goal_cid, ordinal, status, "
                "revision, identity_json, body_json FROM tasks ORDER BY task_cid"
            ).fetchall()
            if len(task_rows) > MAX_TASKS:
                raise TaskSourceIntegrityError("task population exceeds integrity bound")
            goal_rows = connection.execute(
                "SELECT goal_cid, parent_goal_cid, body_json FROM goals ORDER BY goal_cid"
            ).fetchall()
            if len(goal_rows) > MAX_GOALS:
                raise TaskSourceIntegrityError("goal population exceeds integrity bound")
            goal_ids = {str(row[0]) for row in goal_rows}
            goal_graph: dict[str, tuple[str, ...]] = {}
            for goal_cid, parent, body_json in goal_rows:
                _decode_canonical(body_json, noun=f"goal {goal_cid} body")
                goal_graph[str(goal_cid)] = (() if not parent else (str(parent),))
            _assert_acyclic(goal_graph, "goal")
            task_ids = {str(row[0]) for row in task_rows}
            aliases = {str(row[1]) for row in task_rows}
            if len(task_ids) != len(task_rows) or len(aliases) != len(task_rows):
                raise TaskSourceIntegrityError("task keys or aliases are duplicated")
            task_graph: dict[str, list[str]] = {item: [] for item in task_ids}
            global_revision = int(_meta_value(metadata, "revision"))
            initial_statuses = _decode_canonical(
                _meta_value(metadata, "initial_task_statuses"),
                noun="initial task statuses",
            )
            if (
                not isinstance(initial_statuses, Mapping)
                or set(initial_statuses) != task_ids
            ):
                raise TaskSourceIntegrityError(
                    "initial task status population does not match tasks"
                )
            for task_cid, status_value in initial_statuses.items():
                try:
                    _status(status_value)
                except ValueError as exc:
                    raise TaskSourceIntegrityError(
                        f"task {task_cid!r} has an invalid initial status"
                    ) from exc
            for row in task_rows:
                task_cid, _alias, goal_cid, _ordinal, status, revision, identity_json, body_json = row
                if str(goal_cid) not in goal_ids:
                    raise TaskSourceIntegrityError(
                        f"task {task_cid!r} references unknown goal"
                    )
                _status(status)
                if int(revision) < 1:
                    raise TaskSourceIntegrityError("task revision is invalid")
                if int(revision) > global_revision:
                    raise TaskSourceIntegrityError(
                        "task revision exceeds the source revision"
                    )
                identity = _decode_canonical(
                    identity_json, noun=f"task {task_cid} identity"
                )
                body = _decode_canonical(body_json, noun=f"task {task_cid} body")
                if not isinstance(identity, Mapping) or not isinstance(body, Mapping):
                    raise TaskSourceIntegrityError(
                        "task identity and body must be JSON objects"
                    )
                if _task_identity_payload(body) != identity:
                    raise TaskSourceIntegrityError(
                        f"task {task_cid!r} immutable identity changed"
                    )
            dependency_rows = connection.execute(
                "SELECT task_cid, dependency_task_cid FROM task_dependencies "
                "ORDER BY task_cid, dependency_task_cid"
            ).fetchall()
            if len(dependency_rows) > MAX_EDGES:
                raise TaskSourceIntegrityError(
                    "task dependency population exceeds integrity bound"
                )
            for task_cid, dependency in dependency_rows:
                if str(task_cid) not in task_ids or str(dependency) not in task_ids:
                    raise TaskSourceIntegrityError(
                        "task dependency references an unknown task"
                    )
                task_graph[str(task_cid)].append(str(dependency))
            _assert_acyclic(task_graph, "task dependency")
            for table in (
                "task_outputs",
                "task_validations",
                "task_acceptance",
                "task_events",
            ):
                rows = connection.execute(
                    f'SELECT DISTINCT task_cid FROM "{table}"'
                ).fetchall()
                unknown = {str(row[0]) for row in rows} - task_ids
                if unknown:
                    raise TaskSourceIntegrityError(
                        f"{table} references an unknown task"
                    )
            plan_root = _meta_value(metadata, "plan_root_cid")
            receipt_roots = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT plan_root_cid FROM materialization_receipts"
                ).fetchall()
            }
            if receipt_roots != {plan_root}:
                raise TaskSourceIntegrityError(
                    "materialization receipts have a missing or foreign plan root"
                )
            for table, json_columns in {
                "artifacts": ("provenance_json",),
                "task_outputs": ("effect_json",),
                "task_validations": ("argv_json", "policy_json"),
                "task_acceptance": ("evidence_policy_json",),
                "task_events": ("body_json",),
                "materialization_receipts": ("body_json",),
                "formal_plan_input_records": ("payload_json",),
                "schema_migration_receipts": ("body_json",),
            }.items():
                columns = _TABLE_COLUMNS[table]
                selected = ", ".join(f'"{name}"' for name in json_columns)
                rows = connection.execute(f'SELECT {selected} FROM "{table}"').fetchall()
                for row in rows:
                    for name, value in zip(json_columns, row):
                        _decode_canonical(value, noun=f"{table}.{name}")
            event_rows = connection.execute(
                "SELECT sequence, revision, task_cid, event_type, body_json "
                "FROM task_events ORDER BY sequence"
            ).fetchall()
            sequences = [int(row[0]) for row in event_rows]
            if sequences != list(range(1, len(sequences) + 1)):
                raise TaskSourceIntegrityError("event sequence is not contiguous")
            if sequences and sequences[-1] != int(
                _meta_value(metadata, "event_sequence")
            ):
                raise TaskSourceIntegrityError("event cursor metadata is stale")
            if not sequences and int(_meta_value(metadata, "event_sequence")) != 0:
                raise TaskSourceIntegrityError("event cursor metadata is corrupt")
            if global_revision != len(event_rows) + 1:
                raise TaskSourceIntegrityError(
                    "source revision is not monotonic with committed events"
                )
            if any(int(row[1]) != int(row[0]) + 1 for row in event_rows):
                raise TaskSourceIntegrityError(
                    "event revisions are not contiguous and monotonic"
                )
            status_chains: dict[str, list[Mapping[str, Any]]] = {
                task_cid: [] for task_cid in task_ids
            }
            for _sequence, _revision, task_cid, event_type, body_json in event_rows:
                if str(event_type) != "status_changed":
                    continue
                body = _decode_canonical(
                    body_json, noun=f"status event for {task_cid}"
                )
                if not isinstance(body, Mapping):
                    raise TaskSourceIntegrityError(
                        "status event body must be an object"
                    )
                status_chains[str(task_cid)].append(body)
            current_rows = {str(row[0]): row for row in task_rows}
            for task_cid, chain in status_chains.items():
                expected_status = str(initial_statuses[task_cid])
                expected_task_revision = 1
                for event in chain:
                    expected_task_revision += 1
                    if (
                        event.get("task_cid") != task_cid
                        or event.get("previous_status") != expected_status
                        or event.get("task_revision") != expected_task_revision
                    ):
                        raise TaskSourceIntegrityError(
                            f"task {task_cid!r} status event chain is corrupt"
                        )
                    try:
                        expected_status = _status(event.get("status"))
                    except ValueError as exc:
                        raise TaskSourceIntegrityError(
                            f"task {task_cid!r} status event is invalid"
                        ) from exc
                current = current_rows[task_cid]
                if (
                    str(current[4]) != expected_status
                    or int(current[5]) != expected_task_revision
                ):
                    raise TaskSourceIntegrityError(
                        f"task {task_cid!r} status/revision has no valid event history"
                    )
            result = FormalPlanCompiler().compile_duckdb(connection)
            if (
                result.status is not CompilationStatus.COMPILED
                or result.plan_id != _meta_value(metadata, "formal_plan_id")
                or result.source_identity != _meta_value(metadata, "source_identity")
            ):
                raise TaskSourceIntegrityError(
                    "independent formal plan recompilation disagrees with source"
                )
            return IntegrityReport(
                valid=True,
                plan_root_cid=_meta_value(metadata, "plan_root_cid"),
                projection_cid=_meta_value(metadata, "projection_cid"),
                revision=global_revision,
                event_cursor=int(_meta_value(metadata, "event_sequence")),
                formal_plan_id=result.plan_id,
                source_identity=result.source_identity,
                checked_tables=tuple(sorted(_TABLE_COLUMNS)),
            )

    integrity = validate_integrity

    def preview_migration(
        self, target_version: int = DUCKDB_TASK_SOURCE_SCHEMA_VERSION
    ) -> MigrationPreview:
        if isinstance(target_version, bool) or not isinstance(target_version, int):
            raise ValueError("target_version must be an integer")
        with self._read_connection() as (_connection, metadata):
            current = int(_meta_value(metadata, "schema_version"))
            revision = int(_meta_value(metadata, "revision"))
            database_identity = _database_identity(metadata)
        supported = target_version in self.compatibility_matrix.get(current, ())
        statements: tuple[str, ...] = ()
        material = {
            "schema": TASK_SOURCE_MIGRATION_PREVIEW_SCHEMA,
            "database_identity": database_identity,
            "from_version": current,
            "to_version": target_version,
            "base_revision": revision,
            "statement_digests": list(statements),
        }
        rollback_identity = content_identity(
            {"namespace": "duckdb-schema-rollback", **material}
        )
        return MigrationPreview(
            preview_id=content_identity(material),
            database_identity=database_identity,
            from_version=current,
            to_version=target_version,
            base_revision=revision,
            statement_digests=statements,
            rollback_identity=rollback_identity,
            changed=current != target_version,
            supported=supported,
        )

    migration_preview = preview_migration

    def migrate(
        self,
        preview: MigrationPreview,
        *,
        writer_id: str | None = None,
        fencing_token: int | None = None,
        fault_injector: Any | None = None,
    ) -> MigrationReceipt:
        if not isinstance(preview, MigrationPreview):
            raise TypeError("preview must be a MigrationPreview")
        if not preview.supported:
            raise UnsupportedSchemaMigrationError(
                f"schema migration {preview.from_version}->{preview.to_version} "
                "is not supported"
            )
        with self._write_connection(
            writer_id=writer_id, fencing_token=fencing_token
        ) as (connection, metadata):
            if (
                _database_identity(metadata) != preview.database_identity
                or int(_meta_value(metadata, "revision")) != preview.base_revision
                or int(_meta_value(metadata, "schema_version"))
                != preview.from_version
            ):
                raise TaskSourceConflictError("schema migration preview is stale")
            if callable(fault_injector):
                fault_injector("before_migration")
            # Version 1 is intentionally the only compatibility target today.
            # Keeping the migration inside this transaction means any future
            # registered statements inherit rollback-on-error semantics.
            changed = preview.changed
            if changed:
                raise UnsupportedSchemaMigrationError(
                    "no registered statements exist for this schema migration"
                )
            body = {
                "schema": TASK_SOURCE_MIGRATION_RECEIPT_SCHEMA,
                "preview_id": preview.preview_id,
                "from_version": preview.from_version,
                "to_version": preview.to_version,
                "revision": preview.base_revision,
                "changed": False,
                "rollback_identity": preview.rollback_identity,
            }
            receipt_cid = content_identity(body)
            existing = connection.execute(
                "SELECT body_json FROM schema_migration_receipts WHERE receipt_cid = ?",
                [receipt_cid],
            ).fetchone()
            encoded = _canonical(body, noun="schema migration receipt")
            if existing is None:
                _insert_many(
                    connection,
                    "schema_migration_receipts",
                    (
                        (
                            receipt_cid,
                            preview.from_version,
                            preview.to_version,
                            preview.base_revision,
                            encoded,
                        ),
                    ),
                )
            elif str(existing[0]) != encoded:
                raise TaskSourceConflictError(
                    "schema migration receipt identity collision"
                )
            return MigrationReceipt(
                receipt_cid=receipt_cid,
                preview_id=preview.preview_id,
                from_version=preview.from_version,
                to_version=preview.to_version,
                revision=preview.base_revision,
                changed=False,
            )

    apply_migration = migrate

    def rollback_migration(
        self,
        receipt: MigrationReceipt,
        *,
        writer_id: str | None = None,
        fencing_token: int | None = None,
    ) -> MigrationReceipt:
        if not isinstance(receipt, MigrationReceipt):
            raise TypeError("receipt must be a MigrationReceipt")
        if receipt.changed:
            raise UnsupportedSchemaMigrationError(
                "committed cross-version rollback requires a registered "
                "version-specific rollback"
            )
        # A no-op migration has no bytes to restore.  Return an identity-bound
        # rollback receipt after revalidating the source and writer fence.
        with self._write_connection(
            writer_id=writer_id, fencing_token=fencing_token
        ) as (_connection, metadata):
            if int(_meta_value(metadata, "schema_version")) != receipt.to_version:
                raise TaskSourceConflictError("migration rollback source is stale")
            body = {
                **receipt.to_dict(),
                "rolled_back": True,
            }
            return MigrationReceipt(
                receipt_cid=content_identity(body),
                preview_id=receipt.preview_id,
                from_version=receipt.to_version,
                to_version=receipt.from_version,
                revision=int(_meta_value(metadata, "revision")),
                changed=False,
                rolled_back=True,
            )

    def recover(self) -> IntegrityReport:
        """Recover an interrupted initial install, then verify durable state."""

        self._recover_atomic_install()
        return self.validate_integrity()

    def plan_revision_projection_cid(self) -> str:
        """Return the exact content-addressed DuckDB projection CID."""

        if not self.database_path.exists():
            return ""
        snapshot = self.snapshot()
        return str(snapshot.projection_cid)

    def apply_plan_revision(
        self,
        *,
        revision: Any = None,
        admission: Any = None,
        goal_graph: Any = None,
        aliases: Mapping[str, str] | None = None,
        repository_tree_id: str = "",
        retained_task_cids: Sequence[str] = (),
        claimed_task_cids: Sequence[str] = (),
        deferred_item_keys: Sequence[str] = (),
        origin: str = "create",
        delta: Any = None,
        store_continuation: Any | None = None,
        idempotency_key: str = "",
        fencing_token: int | None = None,
        receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply one create/steer plan revision onto this DuckDB projection.

        Create installs an admitted formal plan/graph.  Steer refuses to rewrite
        claimed or accepted task identity payloads and only admits additive
        population growth when a full candidate graph is supplied.  Continuation
        state is written to the optional plan-revision store (CAS), never kept
        only in process dictionaries.
        """

        del aliases  # DuckDB materialization derives aliases from the graph.
        source = goal_graph if goal_graph is not None else admission
        if source is None:
            raise TaskSourceIntegrityError(
                "apply_plan_revision requires a goal graph or formal plan input"
            )
        tree_id = repository_tree_id
        if store_continuation is not None and idempotency_key:
            putter = getattr(store_continuation, "put_continuation", None)
            if callable(putter):
                existing: dict[str, Any] = {}
                loader = getattr(store_continuation, "load_continuation", None)
                if callable(loader):
                    prior = loader(idempotency_key)
                    if isinstance(prior, Mapping):
                        existing = dict(prior)
                existing.update(
                    {
                        "duckdb_pending": {
                            "database_path": str(self.database_path),
                            "origin": str(origin),
                            "plan_root_cid": str(
                                getattr(revision, "plan_root_cid", "") or ""
                            ),
                        }
                    }
                )
                putter(idempotency_key, existing)

        claimed = {str(item) for item in claimed_task_cids}
        retained = {str(item) for item in retained_task_cids}
        protected = claimed | retained

        if self.database_path.exists():
            current = self.snapshot()
            if origin == "create" or str(origin).endswith("create"):
                # Create is install-once; identical replay is a no-op.
                result = self.materialize(
                    source,
                    repository_tree_id=tree_id,
                    plan_root_cid="",
                    receipt=receipt,
                    fencing_token=fencing_token,
                )
                return {
                    "projection_cid": str(result.get("projection_cid") or ""),
                    "receipt_cid": str(result.get("receipt_cid") or ""),
                    "plan_root_cid": str(result.get("plan_root_cid") or ""),
                    "changed": bool(result.get("changed")),
                    "replayed": bool(result.get("replayed")),
                    "deferred_item_keys": list(deferred_item_keys),
                }

            # Steer: verify protected task identities cannot change and refuse
            # drops.  Additive population growth is applied by materializing a
            # candidate into a temp database and swapping under store backups.
            candidate_root = str(getattr(revision, "plan_root_cid", "") or "")
            formal_source, graph = _source_and_projection(
                source, repository_tree_id=tree_id
            )
            bundle = _section_bundle(formal_source)
            task_rows, _task_records = _task_rows(bundle, graph)
            candidate_cids = {str(row[0]) for row in task_rows}
            current_tasks = {
                str(task.task_cid): task for task in self.list_tasks(limit=MAX_TASKS)
            }
            current_cids = set(current_tasks)
            if protected - candidate_cids:
                missing = sorted(protected - candidate_cids)
                raise TaskSourceConflictError(
                    "claimed/accepted tasks missing from candidate plan: "
                    + ", ".join(missing)
                )
            if current_cids - candidate_cids:
                raise TaskSourceConflictError(
                    "steer apply would drop existing DuckDB tasks"
                )
            if candidate_cids == current_cids and (
                not candidate_root or candidate_root == current.plan_root_cid
            ):
                return {
                    "projection_cid": current.projection_cid,
                    "receipt_cid": "",
                    "plan_root_cid": current.plan_root_cid,
                    "changed": False,
                    "replayed": True,
                    "deferred_item_keys": list(deferred_item_keys),
                    "delta_cid": str(getattr(delta, "delta_cid", "") or ""),
                }

            # After lifecycle events exist, a full candidate reinstall would
            # destroy status history.  Refuse rather than rewrite claimed work.
            if int(current.event_cursor) > 0:
                raise TaskSourceConflictError(
                    "DuckDB plan revision cannot reinstall after lifecycle "
                    "events; claimed/accepted history must remain durable"
                )
            # Verify protected task identities against the live rows before any
            # candidate install.
            with self._read_connection() as (live_connection, _metadata):
                live_identity_rows = live_connection.execute(
                    "SELECT task_cid, identity_json FROM tasks ORDER BY task_cid"
                ).fetchall()
            live_identities = {
                str(task_cid): _decode_canonical(
                    identity_json, noun=f"task {task_cid} identity"
                )
                for task_cid, identity_json in live_identity_rows
            }
            candidate_identities = {
                str(row[0]): _decode_canonical(
                    row[6], noun=f"task {row[0]} identity"
                )
                for row in task_rows
            }
            for task_cid in protected:
                if task_cid not in live_identities:
                    continue
                if live_identities[task_cid] != candidate_identities.get(task_cid):
                    raise TaskSourceConflictError(
                        f"claimed/accepted task {task_cid!r} identity "
                        "would change under steer apply"
                    )

            # Build a candidate database beside the live path, then atomically
            # replace under the live lock.  Safe only while no lifecycle events
            # have been recorded (all tasks still at initial revision).
            candidate_path = self.database_path.with_name(
                f".{self.database_path.name}.plan-revision-candidate"
            )
            if candidate_path.exists():
                candidate_path.unlink()
            candidate = DuckDBTaskSource(
                candidate_path,
                writer_id=self.writer_id,
                fencing_token=(
                    self.fencing_token if fencing_token is None else fencing_token
                ),
                lock_timeout_seconds=self.lock_timeout_seconds,
            )
            try:
                materialize_result = candidate.materialize(
                    source,
                    repository_tree_id=tree_id,
                    plan_root_cid="",
                    receipt={
                        **dict(receipt or {}),
                        "plan_revision_cid": str(
                            getattr(revision, "revision_cid", "")
                            or getattr(revision, "content_id", "")
                            or ""
                        ),
                        "plan_revision_plan_root_cid": candidate_root,
                        "origin": str(origin),
                        "deferred_item_keys": list(deferred_item_keys),
                    },
                    fencing_token=fencing_token,
                )
                with exclusive_file_lock(
                    self._lock_path, timeout_seconds=self.lock_timeout_seconds
                ):
                    os.replace(candidate_path, self.database_path)
                    self._fsync_parent()
                return {
                    "projection_cid": str(
                        materialize_result.get("projection_cid") or ""
                    ),
                    "receipt_cid": str(materialize_result.get("receipt_cid") or ""),
                    "plan_root_cid": str(
                        materialize_result.get("plan_root_cid") or candidate_root
                    ),
                    "changed": True,
                    "replayed": False,
                    "deferred_item_keys": list(deferred_item_keys),
                    "delta_cid": str(getattr(delta, "delta_cid", "") or ""),
                }
            except Exception:
                try:
                    if candidate_path.exists():
                        candidate_path.unlink()
                except OSError:
                    pass
                raise

        # DuckDB materialize binds the graph/formal plan root.  The plan
        # revision root may be an admitted-plan envelope CID and must not be
        # forced as the projection plan_root_cid.
        result = self.materialize(
            source,
            repository_tree_id=tree_id,
            plan_root_cid="",
            receipt={
                **dict(receipt or {}),
                "plan_revision_cid": str(
                    getattr(revision, "revision_cid", "")
                    or getattr(revision, "content_id", "")
                    or ""
                ),
                "plan_revision_plan_root_cid": str(
                    getattr(revision, "plan_root_cid", "") or ""
                ),
                "deferred_item_keys": list(deferred_item_keys),
                "origin": str(origin),
            },
            fencing_token=fencing_token,
        )
        if store_continuation is not None and idempotency_key:
            putter = getattr(store_continuation, "put_continuation", None)
            if callable(putter):
                existing = {}
                loader = getattr(store_continuation, "load_continuation", None)
                if callable(loader):
                    prior = loader(idempotency_key)
                    if isinstance(prior, Mapping):
                        existing = dict(prior)
                existing["duckdb_committed"] = {
                    "projection_cid": str(result.get("projection_cid") or ""),
                    "receipt_cid": str(result.get("receipt_cid") or ""),
                }
                putter(idempotency_key, existing)
        return {
            "projection_cid": str(result.get("projection_cid") or ""),
            "receipt_cid": str(result.get("receipt_cid") or ""),
            "plan_root_cid": str(result.get("plan_root_cid") or ""),
            "changed": bool(result.get("changed")),
            "replayed": bool(result.get("replayed")),
            "deferred_item_keys": list(deferred_item_keys),
        }


def materialize_duckdb_task_source(
    database_path: str | os.PathLike[str],
    source: Any,
    **kwargs: Any,
) -> DuckDBTaskSource:
    """Functional constructor used by callers that prefer one-step setup."""

    return DuckDBTaskSource.from_formal_plan_input(database_path, source, **kwargs)


__all__ = [
    "CASResult",
    "DEFAULT_DERIVED_PROTECTED_ANCHORS",
    "DEFAULT_QUERY_LIMIT",
    "DERIVED_RUNTIME_MATERIALIZATION_SCHEMA",
    "DERIVED_RUNTIME_SOURCE_ROLE",
    "DUCKDB_TASK_SOURCE_SCHEMA",
    "DUCKDB_TASK_SOURCE_SCHEMA_VERSION",
    "DuckDBTaskSource",
    "DuckDBTaskSourceError",
    "DuckDBUnavailableError",
    "EventPage",
    "IntegrityReport",
    "MAX_QUERY_LIMIT",
    "MigrationPreview",
    "MigrationReceipt",
    "TaskPage",
    "TaskRecord",
    "TaskSourceBoundsError",
    "TaskSourceConflictError",
    "TaskSourceInjectionError",
    "TaskSourceIntegrityError",
    "TaskSourceSnapshot",
    "UnsupportedSchemaMigrationError",
    "SCHEMA_VERSION",
    "WORKFLOW_SCHEMA",
    "WORKFLOW_SCHEMA_VERSION",
    "WriterFence",
    "duckdb_available",
    "materialize_duckdb_task_source",
]
