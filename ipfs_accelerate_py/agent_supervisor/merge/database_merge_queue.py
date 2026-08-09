"""Integrate validation, merge queue, publication, and task completion.

DQP-019 / DatabaseMergeQueue@1, ValidationRun@1
===============================================

:class:`DatabaseMergeQueue` is the durable authority for validation starts and
results, merge claims/attempts/outcomes, conflict resolution, publication, and
task completion under shared task/attempt/worktree/fence coordinates.

Authority rules (fail-closed)
-----------------------------
* A task completes only after an **accepted merge** and **current validation
  evidence** commit together in one transaction.
* Stale worktree or fence results are rejected on every protected write.
* JSON stage receipts and queue files are optional projections; they cannot
  settle work or grant completion.
* Merge claims are serialized per target scope; fair priority aging prevents
  permanent starvation of older lower-priority entries.
* Partial publication is recorded but never equivalent to completion.

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
from typing import Any, ClassVar, Final

from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes
from .merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
    MergeQueueFenceError,
)

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_MERGE_QUEUE_INTERFACE: Final[str] = "DatabaseMergeQueue@1"
VALIDATION_RUN_INTERFACE: Final[str] = "ValidationRun@1"
MERGE_ATTEMPT_INTERFACE: Final[str] = "MergeAttempt@1"
COMPLETION_RECEIPT_INTERFACE: Final[str] = "CompletionReceipt@1"

DATABASE_MERGE_QUEUE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-merge-queue@1"
)
VALIDATION_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/validation-run@1"
)
VALIDATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/validation-result@1"
)
MERGE_QUEUE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-queue-entry@1"
)
MERGE_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-attempt@1"
)
COMPLETION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/completion-receipt@1"
)
DOMAIN_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-recovery-event@1"
)
JSON_RECEIPT_MIRROR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-json-receipt-mirror@1"
)

DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_PRIORITY_AGING_MS: Final[int] = 300_000
MAX_PAYLOAD_BYTES: Final[int] = 262_144
MAX_VALIDATION_COMMANDS: Final[int] = 64

_PRIORITY_ORDER: Final[dict[str, int]] = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
}

ClockMs = Callable[[], int]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseMergeError(RuntimeError):
    """Base fail-closed error for database merge/validation authority."""

    code = "DQP_MERGE_ERROR"


class DatabaseMergeNotOpenError(DatabaseMergeError):
    code = "DQP_MERGE_NOT_OPEN"


class DatabaseMergeConflictError(DatabaseMergeError):
    code = "DQP_MERGE_CONFLICT"


class DatabaseMergeStaleError(DatabaseMergeError, MergeQueueFenceError):
    """Stale worktree, fence, claim, or validation result."""

    code = "DQP_MERGE_STALE"


class DatabaseMergeNotReadyError(DatabaseMergeError):
    """Completion or publication blocked by missing prerequisites."""

    code = "DQP_MERGE_NOT_READY"


class DatabaseMergeBoundsError(DatabaseMergeError, ValueError):
    code = "DQP_MERGE_BOUNDS"


class DuckDBUnavailableError(DatabaseMergeError):
    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class MergeEntryStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    VALIDATING = "validating"
    MERGING = "merging"
    ACCEPTED = "accepted"
    PUBLISHED = "published"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    CANCELLED = "cancelled"


class ValidationStatus(str, Enum):
    STARTED = "started"
    PASSED = "passed"
    FAILED = "failed"
    STALE = "stale"
    SUPERSEDED = "superseded"


class MergeAttemptStatus(str, Enum):
    STARTED = "started"
    REBASED = "rebased"
    CONFLICT = "conflict"
    ACCEPTED = "accepted"
    PARTIAL_PUBLISH = "partial_publish"
    PUBLISHED = "published"
    FAILED = "failed"
    ABORTED = "aborted"


class CompletionStatus(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"


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
        raise DatabaseMergeError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseMergeError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseMergeBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseMergeBoundsError(f"{name} must be a positive integer")
    return value


def _normalise_priority(value: Any) -> str:
    priority = str(value or "P2").strip().upper()
    return priority if priority in _PRIORITY_ORDER else "P2"


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


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise DatabaseMergeBoundsError(f"{name} exceeds the {max_bytes}-byte bound")
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


def _enum_value(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).strip().lower())
    except Exception as exc:
        raise DatabaseMergeError(f"invalid {name}: {value!r}") from exc


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationRun:
    """One fenced validation execution bound to task/attempt/worktree."""

    INTERFACE: ClassVar[str] = VALIDATION_RUN_INTERFACE
    SCHEMA: ClassVar[str] = VALIDATION_RUN_SCHEMA

    run_id: str
    task_cid: str
    attempt_id: str
    worktree_id: str
    fencing_token: int
    fence_epoch: int
    status: ValidationStatus
    started_at_ms: int
    commands: tuple[str, ...] = ()
    evidence_digest: str = ""
    result_summary: str = ""
    finished_at_ms: int | None = None
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
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
            "status",
            _enum_value(self.status, ValidationStatus, "status"),
        )
        object.__setattr__(
            self, "started_at_ms", _nonneg_int(int(self.started_at_ms), "started_at_ms")
        )
        commands = tuple(
            _text(item, "command") for item in (self.commands or ())
        )
        if len(commands) > MAX_VALIDATION_COMMANDS:
            raise DatabaseMergeBoundsError("too many validation commands")
        object.__setattr__(self, "commands", commands)
        object.__setattr__(
            self,
            "evidence_digest",
            _text(self.evidence_digest, "evidence_digest", required=False),
        )
        object.__setattr__(
            self,
            "result_summary",
            _text(self.result_summary, "result_summary", required=False),
        )
        if self.finished_at_ms is not None:
            object.__setattr__(
                self,
                "finished_at_ms",
                _nonneg_int(int(self.finished_at_ms), "finished_at_ms"),
            )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def current_and_passed(self) -> bool:
        return self.status is ValidationStatus.PASSED and bool(self.evidence_digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "run_id": self.run_id,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "worktree_id": self.worktree_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "status": self.status.value
            if isinstance(self.status, ValidationStatus)
            else str(self.status),
            "started_at_ms": int(self.started_at_ms),
            "finished_at_ms": self.finished_at_ms,
            "commands": list(self.commands),
            "evidence_digest": self.evidence_digest,
            "result_summary": self.result_summary,
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class MergeQueueEntry:
    """One merge-queue candidate under fenced task/worktree ownership."""

    SCHEMA: ClassVar[str] = MERGE_QUEUE_ENTRY_SCHEMA

    entry_id: str
    task_cid: str
    attempt_id: str
    worktree_id: str
    branch_name: str
    commit_sha: str
    priority: str
    status: MergeEntryStatus
    fencing_token: int
    fence_epoch: int
    enqueued_at_ms: int
    target_repository_id: str = ""
    target_branch: str = ""
    claim_token: str = ""
    claim_generation: int = 0
    consumer_id: str = ""
    claimed_at_ms: int = 0
    attempt_number: int = 1
    failure_count: int = 0
    current_validation_run_id: str = ""
    current_merge_attempt_id: str = ""
    accepted_merge_attempt_id: str = ""
    completion_receipt_id: str = ""
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _text(self.entry_id, "entry_id"))
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "branch_name", _text(self.branch_name, "branch_name")
        )
        object.__setattr__(
            self, "commit_sha", _text(self.commit_sha, "commit_sha")
        )
        object.__setattr__(self, "priority", _normalise_priority(self.priority))
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, MergeEntryStatus, "status"),
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
            "enqueued_at_ms",
            _nonneg_int(int(self.enqueued_at_ms), "enqueued_at_ms"),
        )
        object.__setattr__(
            self,
            "target_repository_id",
            _text(self.target_repository_id, "target_repository_id", required=False),
        )
        object.__setattr__(
            self,
            "target_branch",
            _text(self.target_branch, "target_branch", required=False),
        )
        object.__setattr__(
            self,
            "claim_token",
            _text(self.claim_token, "claim_token", required=False),
        )
        object.__setattr__(
            self,
            "claim_generation",
            _nonneg_int(int(self.claim_generation), "claim_generation"),
        )
        object.__setattr__(
            self,
            "consumer_id",
            _text(self.consumer_id, "consumer_id", required=False),
        )
        object.__setattr__(
            self,
            "claimed_at_ms",
            _nonneg_int(int(self.claimed_at_ms), "claimed_at_ms"),
        )
        object.__setattr__(
            self,
            "attempt_number",
            _positive_int(int(self.attempt_number), "attempt_number"),
        )
        object.__setattr__(
            self,
            "failure_count",
            _nonneg_int(int(self.failure_count), "failure_count"),
        )
        object.__setattr__(
            self,
            "current_validation_run_id",
            _text(
                self.current_validation_run_id,
                "current_validation_run_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "current_merge_attempt_id",
            _text(
                self.current_merge_attempt_id,
                "current_merge_attempt_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "accepted_merge_attempt_id",
            _text(
                self.accepted_merge_attempt_id,
                "accepted_merge_attempt_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "completion_receipt_id",
            _text(
                self.completion_receipt_id,
                "completion_receipt_id",
                required=False,
            ),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def has_target_binding(self) -> bool:
        return bool(self.target_repository_id and self.target_branch)

    @property
    def target_scope(self) -> str:
        if self.has_target_binding:
            return f"{self.target_repository_id}:{self.target_branch}"
        return "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entry_id": self.entry_id,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "worktree_id": self.worktree_id,
            "branch_name": self.branch_name,
            "commit_sha": self.commit_sha,
            "priority": self.priority,
            "status": self.status.value
            if isinstance(self.status, MergeEntryStatus)
            else str(self.status),
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "enqueued_at_ms": int(self.enqueued_at_ms),
            "target_repository_id": self.target_repository_id,
            "target_branch": self.target_branch,
            "target_binding_schema": (
                MERGE_TARGET_BINDING_SCHEMA if self.has_target_binding else ""
            ),
            "claim_token": self.claim_token,
            "claim_generation": int(self.claim_generation),
            "consumer_id": self.consumer_id,
            "claimed_at_ms": int(self.claimed_at_ms),
            "attempt_number": int(self.attempt_number),
            "failure_count": int(self.failure_count),
            "current_validation_run_id": self.current_validation_run_id,
            "current_merge_attempt_id": self.current_merge_attempt_id,
            "accepted_merge_attempt_id": self.accepted_merge_attempt_id,
            "completion_receipt_id": self.completion_receipt_id,
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class MergeAttempt:
    """One merge attempt outcome under the owning claim fence."""

    INTERFACE: ClassVar[str] = MERGE_ATTEMPT_INTERFACE
    SCHEMA: ClassVar[str] = MERGE_ATTEMPT_SCHEMA

    attempt_id: str
    entry_id: str
    task_cid: str
    worktree_id: str
    fencing_token: int
    fence_epoch: int
    claim_token: str
    claim_generation: int
    status: MergeAttemptStatus
    started_at_ms: int
    base_commit: str = ""
    result_commit: str = ""
    conflict_paths: tuple[str, ...] = ()
    finished_at_ms: int | None = None
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "entry_id", _text(self.entry_id, "entry_id"))
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
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
            self, "claim_token", _text(self.claim_token, "claim_token")
        )
        object.__setattr__(
            self,
            "claim_generation",
            _positive_int(int(self.claim_generation), "claim_generation"),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, MergeAttemptStatus, "status"),
        )
        object.__setattr__(
            self, "started_at_ms", _nonneg_int(int(self.started_at_ms), "started_at_ms")
        )
        object.__setattr__(
            self,
            "base_commit",
            _text(self.base_commit, "base_commit", required=False),
        )
        object.__setattr__(
            self,
            "result_commit",
            _text(self.result_commit, "result_commit", required=False),
        )
        object.__setattr__(
            self,
            "conflict_paths",
            tuple(
                _text(item, "conflict_path") for item in (self.conflict_paths or ())
            ),
        )
        if self.finished_at_ms is not None:
            object.__setattr__(
                self,
                "finished_at_ms",
                _nonneg_int(int(self.finished_at_ms), "finished_at_ms"),
            )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def accepted(self) -> bool:
        return self.status in {
            MergeAttemptStatus.ACCEPTED,
            MergeAttemptStatus.PUBLISHED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": self.attempt_id,
            "entry_id": self.entry_id,
            "task_cid": self.task_cid,
            "worktree_id": self.worktree_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "claim_token": self.claim_token,
            "claim_generation": int(self.claim_generation),
            "status": self.status.value
            if isinstance(self.status, MergeAttemptStatus)
            else str(self.status),
            "started_at_ms": int(self.started_at_ms),
            "finished_at_ms": self.finished_at_ms,
            "base_commit": self.base_commit,
            "result_commit": self.result_commit,
            "conflict_paths": list(self.conflict_paths),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class CompletionReceipt:
    """Authoritative task completion bound to merge + validation evidence."""

    INTERFACE: ClassVar[str] = COMPLETION_RECEIPT_INTERFACE
    SCHEMA: ClassVar[str] = COMPLETION_RECEIPT_SCHEMA

    receipt_id: str
    task_cid: str
    entry_id: str
    validation_run_id: str
    merge_attempt_id: str
    worktree_id: str
    fencing_token: int
    fence_epoch: int
    evidence_digest: str
    result_commit: str
    completed_at_ms: int
    status: CompletionStatus = CompletionStatus.COMPLETED
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, "receipt_id"))
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(self, "entry_id", _text(self.entry_id, "entry_id"))
        object.__setattr__(
            self,
            "validation_run_id",
            _text(self.validation_run_id, "validation_run_id"),
        )
        object.__setattr__(
            self,
            "merge_attempt_id",
            _text(self.merge_attempt_id, "merge_attempt_id"),
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
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
            self, "evidence_digest", _text(self.evidence_digest, "evidence_digest")
        )
        object.__setattr__(
            self, "result_commit", _text(self.result_commit, "result_commit")
        )
        object.__setattr__(
            self,
            "completed_at_ms",
            _nonneg_int(int(self.completed_at_ms), "completed_at_ms"),
        )
        object.__setattr__(
            self,
            "status",
            _enum_value(self.status, CompletionStatus, "status"),
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
            "receipt_id": self.receipt_id,
            "task_cid": self.task_cid,
            "entry_id": self.entry_id,
            "validation_run_id": self.validation_run_id,
            "merge_attempt_id": self.merge_attempt_id,
            "worktree_id": self.worktree_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "evidence_digest": self.evidence_digest,
            "result_commit": self.result_commit,
            "completed_at_ms": int(self.completed_at_ms),
            "completed_at": _utc_iso_from_ms(self.completed_at_ms)
            if self.completed_at_ms
            else "",
            "status": self.status.value
            if isinstance(self.status, CompletionStatus)
            else str(self.status),
            "body": dict(self.body),
        }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS merge_recovery_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS merge_queue_entries (
    entry_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    commit_sha VARCHAR NOT NULL,
    priority VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    enqueued_at_ms BIGINT NOT NULL,
    target_repository_id VARCHAR NOT NULL DEFAULT '',
    target_branch VARCHAR NOT NULL DEFAULT '',
    target_scope VARCHAR NOT NULL DEFAULT 'default',
    claim_token VARCHAR NOT NULL DEFAULT '',
    claim_generation BIGINT NOT NULL DEFAULT 0,
    consumer_id VARCHAR NOT NULL DEFAULT '',
    claimed_at_ms BIGINT NOT NULL DEFAULT 0,
    attempt_number BIGINT NOT NULL DEFAULT 1,
    failure_count BIGINT NOT NULL DEFAULT 0,
    current_validation_run_id VARCHAR NOT NULL DEFAULT '',
    current_merge_attempt_id VARCHAR NOT NULL DEFAULT '',
    accepted_merge_attempt_id VARCHAR NOT NULL DEFAULT '',
    completion_receipt_id VARCHAR NOT NULL DEFAULT '',
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    updated_at_ms BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS merge_queue_entries_status_idx
    ON merge_queue_entries(status, enqueued_at_ms);
CREATE INDEX IF NOT EXISTS merge_queue_entries_task_idx
    ON merge_queue_entries(task_cid, status);
CREATE INDEX IF NOT EXISTS merge_queue_entries_scope_idx
    ON merge_queue_entries(target_scope, status);
CREATE UNIQUE INDEX IF NOT EXISTS merge_queue_entries_dedupe_uidx
    ON merge_queue_entries(task_cid, commit_sha, target_scope);

CREATE TABLE IF NOT EXISTS validation_runs (
    run_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    commands_json VARCHAR NOT NULL DEFAULT '[]',
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    result_summary VARCHAR NOT NULL DEFAULT '',
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS validation_runs_task_idx
    ON validation_runs(task_cid, status, started_at_ms);
CREATE INDEX IF NOT EXISTS validation_runs_entry_idx
    ON validation_runs(entry_id, started_at_ms);

CREATE TABLE IF NOT EXISTS validation_results (
    result_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS validation_results_run_idx
    ON validation_results(run_id);

CREATE TABLE IF NOT EXISTS merge_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    claim_token VARCHAR NOT NULL,
    claim_generation BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    base_commit VARCHAR NOT NULL DEFAULT '',
    result_commit VARCHAR NOT NULL DEFAULT '',
    conflict_paths_json VARCHAR NOT NULL DEFAULT '[]',
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS merge_attempts_entry_idx
    ON merge_attempts(entry_id, started_at_ms);
CREATE INDEX IF NOT EXISTS merge_attempts_task_idx
    ON merge_attempts(task_cid, status);

CREATE TABLE IF NOT EXISTS completion_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    entry_id VARCHAR NOT NULL,
    validation_run_id VARCHAR NOT NULL,
    merge_attempt_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    result_commit VARCHAR NOT NULL,
    completed_at_ms BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS completion_receipts_task_uidx
    ON completion_receipts(task_cid);
CREATE INDEX IF NOT EXISTS completion_receipts_entry_idx
    ON completion_receipts(entry_id);

CREATE TABLE IF NOT EXISTS domain_events (
    event_id VARCHAR PRIMARY KEY,
    stream_key VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    sequence_no BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS domain_events_stream_seq_uidx
    ON domain_events(stream_key, sequence_no);
CREATE INDEX IF NOT EXISTS domain_events_type_idx
    ON domain_events(event_type, observed_at_ms);

CREATE TABLE IF NOT EXISTS json_receipt_mirrors (
    mirror_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    payload_digest VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    grants_completion BOOLEAN NOT NULL DEFAULT FALSE,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS json_receipt_mirrors_entry_idx
    ON json_receipt_mirrors(entry_id, recorded_at_ms);

CREATE TABLE IF NOT EXISTS recovery_actions (
    action_id VARCHAR PRIMARY KEY,
    action_kind VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    entry_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL DEFAULT 0,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL DEFAULT '',
    retry_budget BIGINT NOT NULL DEFAULT 0,
    retry_count BIGINT NOT NULL DEFAULT 0,
    recorded_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    reason VARCHAR NOT NULL DEFAULT '',
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS recovery_actions_idempotency_idx
    ON recovery_actions(idempotency_key);
CREATE INDEX IF NOT EXISTS recovery_actions_task_idx
    ON recovery_actions(task_cid, recorded_at_ms);
CREATE INDEX IF NOT EXISTS recovery_actions_kind_idx
    ON recovery_actions(action_kind, status);
"""


# ---------------------------------------------------------------------------
# DatabaseMergeQueue
# ---------------------------------------------------------------------------


class DatabaseMergeQueue:
    """DuckDB-backed validation + merge + completion authority.

    Interface: ``DatabaseMergeQueue@1`` with projected records
    ``ValidationRun@1``, merge attempts, and completion receipts.
    """

    INTERFACE: ClassVar[str] = DATABASE_MERGE_QUEUE_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_MERGE_QUEUE_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        priority_aging_ms: int = DEFAULT_PRIORITY_AGING_MS,
        max_processing_per_scope: int = 1,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseMergeQueue; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._max_attempts = _positive_int(int(max_attempts), "max_attempts")
        self._priority_aging_ms = _nonneg_int(
            int(priority_aging_ms), "priority_aging_ms"
        )
        self._max_processing_per_scope = _positive_int(
            int(max_processing_per_scope), "max_processing_per_scope"
        )
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

    @property
    def max_attempts(self) -> int:
        return int(self._max_attempts)

    def open(self) -> "DatabaseMergeQueue":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                for key, value in (
                    ("interface", DATABASE_MERGE_QUEUE_INTERFACE),
                    ("schema", DATABASE_MERGE_QUEUE_SCHEMA),
                    ("authority", "database"),
                    (
                        "json_receipts_grant_completion",
                        "false",
                    ),
                ):
                    connection.execute(
                        """
                        INSERT INTO merge_recovery_metadata(key, value)
                        VALUES (?, ?)
                        ON CONFLICT (key) DO UPDATE SET value = excluded.value
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

    def __enter__(self) -> "DatabaseMergeQueue":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseMergeNotOpenError("DatabaseMergeQueue is not open")
        return self._connection

    def _begin(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        connection.execute("BEGIN TRANSACTION")

    def _rollback_if_open(self, connection: Any) -> None:
        try:
            if getattr(connection, "in_transaction", False):
                connection.execute("ROLLBACK")
                return
            rollback = getattr(connection, "rollback", None)
            if callable(rollback):
                rollback()
        except Exception:
            pass

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            connection.execute("COMMIT")
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    def _now_ms(self) -> int:
        return int(self._clock_ms())

    def authority_policy(self) -> dict[str, Any]:
        return {
            "semantic_authority": "database",
            "byte_authority": "git",
            "json_receipts_grant_completion": False,
            "queue_files_grant_completion": False,
            "completion_requires": (
                "accepted_merge",
                "current_validation_evidence",
            ),
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
        }

    # -- row mappers ---------------------------------------------------------

    def _entry_from_row(self, row: Any) -> MergeQueueEntry:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        return MergeQueueEntry(
            entry_id=str(_row_get(mapping, "entry_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            attempt_id=str(_row_get(mapping, "attempt_id", default="")),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            branch_name=str(_row_get(mapping, "branch_name", default="")),
            commit_sha=str(_row_get(mapping, "commit_sha", default="")),
            priority=str(_row_get(mapping, "priority", default="P2")),
            status=str(_row_get(mapping, "status", default="pending")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            enqueued_at_ms=int(_row_get(mapping, "enqueued_at_ms", default=0) or 0),
            target_repository_id=str(
                _row_get(mapping, "target_repository_id", default="") or ""
            ),
            target_branch=str(_row_get(mapping, "target_branch", default="") or ""),
            claim_token=str(_row_get(mapping, "claim_token", default="") or ""),
            claim_generation=int(
                _row_get(mapping, "claim_generation", default=0) or 0
            ),
            consumer_id=str(_row_get(mapping, "consumer_id", default="") or ""),
            claimed_at_ms=int(_row_get(mapping, "claimed_at_ms", default=0) or 0),
            attempt_number=int(_row_get(mapping, "attempt_number", default=1) or 1),
            failure_count=int(_row_get(mapping, "failure_count", default=0) or 0),
            current_validation_run_id=str(
                _row_get(mapping, "current_validation_run_id", default="") or ""
            ),
            current_merge_attempt_id=str(
                _row_get(mapping, "current_merge_attempt_id", default="") or ""
            ),
            accepted_merge_attempt_id=str(
                _row_get(mapping, "accepted_merge_attempt_id", default="") or ""
            ),
            completion_receipt_id=str(
                _row_get(mapping, "completion_receipt_id", default="") or ""
            ),
            revision=int(_row_get(mapping, "revision", default=1) or 1),
            body=body if isinstance(body, Mapping) else {},
        )

    def _validation_from_row(self, row: Any) -> ValidationRun:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        commands_raw = _row_get(mapping, "commands_json", default="[]")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        try:
            commands = json.loads(str(commands_raw or "[]"))
        except json.JSONDecodeError:
            commands = []
        finished = _row_get(mapping, "finished_at_ms", default=None)
        return ValidationRun(
            run_id=str(_row_get(mapping, "run_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            attempt_id=str(_row_get(mapping, "attempt_id", default="")),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            status=str(_row_get(mapping, "status", default="started")),
            started_at_ms=int(_row_get(mapping, "started_at_ms", default=0) or 0),
            commands=tuple(commands) if isinstance(commands, list) else (),
            evidence_digest=str(
                _row_get(mapping, "evidence_digest", default="") or ""
            ),
            result_summary=str(
                _row_get(mapping, "result_summary", default="") or ""
            ),
            finished_at_ms=None if finished is None else int(finished),
            revision=int(_row_get(mapping, "revision", default=1) or 1),
            body=body if isinstance(body, Mapping) else {},
        )

    def _merge_attempt_from_row(self, row: Any) -> MergeAttempt:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        paths_raw = _row_get(mapping, "conflict_paths_json", default="[]")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        try:
            paths = json.loads(str(paths_raw or "[]"))
        except json.JSONDecodeError:
            paths = []
        finished = _row_get(mapping, "finished_at_ms", default=None)
        return MergeAttempt(
            attempt_id=str(_row_get(mapping, "attempt_id", default="")),
            entry_id=str(_row_get(mapping, "entry_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            claim_token=str(_row_get(mapping, "claim_token", default="") or ""),
            claim_generation=int(
                _row_get(mapping, "claim_generation", default=0) or 0
            ),
            status=str(_row_get(mapping, "status", default="started")),
            started_at_ms=int(_row_get(mapping, "started_at_ms", default=0) or 0),
            base_commit=str(_row_get(mapping, "base_commit", default="") or ""),
            result_commit=str(_row_get(mapping, "result_commit", default="") or ""),
            conflict_paths=tuple(paths) if isinstance(paths, list) else (),
            finished_at_ms=None if finished is None else int(finished),
            revision=int(_row_get(mapping, "revision", default=1) or 1),
            body=body if isinstance(body, Mapping) else {},
        )

    def _completion_from_row(self, row: Any) -> CompletionReceipt:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        return CompletionReceipt(
            receipt_id=str(_row_get(mapping, "receipt_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            entry_id=str(_row_get(mapping, "entry_id", default="")),
            validation_run_id=str(
                _row_get(mapping, "validation_run_id", default="") or ""
            ),
            merge_attempt_id=str(
                _row_get(mapping, "merge_attempt_id", default="") or ""
            ),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            evidence_digest=str(
                _row_get(mapping, "evidence_digest", default="") or ""
            ),
            result_commit=str(_row_get(mapping, "result_commit", default="") or ""),
            completed_at_ms=int(
                _row_get(mapping, "completed_at_ms", default=0) or 0
            ),
            status=str(_row_get(mapping, "status", default="completed")),
            body=body if isinstance(body, Mapping) else {},
        )

    def _load_entry(self, connection: Any, entry_id: str) -> MergeQueueEntry | None:
        row = connection.execute(
            "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
            [entry_id],
        ).fetchone()
        return None if row is None else self._entry_from_row(row)

    def _load_validation(
        self, connection: Any, run_id: str
    ) -> ValidationRun | None:
        row = connection.execute(
            "SELECT * FROM validation_runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        return None if row is None else self._validation_from_row(row)

    def _load_merge_attempt(
        self, connection: Any, attempt_id: str
    ) -> MergeAttempt | None:
        row = connection.execute(
            "SELECT * FROM merge_attempts WHERE attempt_id = ?",
            [attempt_id],
        ).fetchone()
        return None if row is None else self._merge_attempt_from_row(row)

    def _append_event(
        self,
        connection: Any,
        *,
        stream_key: str,
        event_type: str,
        body: Mapping[str, Any],
        now_ms: int,
    ) -> None:
        seq_row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence_no), 0) AS max_seq
            FROM domain_events WHERE stream_key = ?
            """,
            [stream_key],
        ).fetchone()
        seq_mapping = _row_mapping(seq_row)
        next_seq = int(_row_get(seq_mapping, "max_seq", default=0) or 0) + 1
        connection.execute(
            """
            INSERT INTO domain_events(
                event_id, stream_key, event_type, sequence_no,
                observed_at_ms, body_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                _new_id("event"),
                stream_key,
                event_type,
                next_seq,
                now_ms,
                _canonical_json(
                    {
                        "schema": DOMAIN_EVENT_SCHEMA,
                        **dict(body),
                    }
                ),
            ],
        )

    def _require_claim_match(
        self,
        entry: MergeQueueEntry,
        *,
        claim_token: str,
        claim_generation: int,
        consumer_id: str = "",
        operation: str,
        allow_failed: bool = False,
    ) -> None:
        allowed = {
            MergeEntryStatus.CLAIMED,
            MergeEntryStatus.VALIDATING,
            MergeEntryStatus.MERGING,
            MergeEntryStatus.ACCEPTED,
            MergeEntryStatus.PUBLISHED,
        }
        if allow_failed:
            allowed.add(MergeEntryStatus.FAILED)
        if entry.status not in allowed:
            raise DatabaseMergeStaleError(
                f"{operation} rejected: entry {entry.entry_id} is {entry.status.value}"
            )
        if (
            not entry.claim_token
            or entry.claim_token != claim_token
            or int(entry.claim_generation) != int(claim_generation)
        ):
            raise DatabaseMergeStaleError(
                f"{operation} rejected for entry {entry.entry_id}: "
                "claim token or generation is stale"
            )
        if consumer_id and entry.consumer_id and entry.consumer_id != consumer_id:
            raise DatabaseMergeStaleError(
                f"{operation} rejected for entry {entry.entry_id}: claim owner mismatch"
            )

    def _require_fence_match(
        self,
        entry: MergeQueueEntry,
        *,
        worktree_id: str,
        fencing_token: int,
        fence_epoch: int,
        operation: str,
    ) -> None:
        if entry.worktree_id != worktree_id:
            raise DatabaseMergeStaleError(
                f"{operation} rejected: worktree {worktree_id} is stale for "
                f"entry {entry.entry_id}"
            )
        if (
            int(entry.fencing_token) != int(fencing_token)
            or int(entry.fence_epoch) != int(fence_epoch)
        ):
            raise DatabaseMergeStaleError(
                f"{operation} rejected: fence token/epoch is stale for "
                f"entry {entry.entry_id}"
            )

    def _fairness_key(
        self, entry: MergeQueueEntry, now_ms: int
    ) -> tuple[int, int, str]:
        base = _PRIORITY_ORDER.get(entry.priority, _PRIORITY_ORDER["P2"])
        if self._priority_aging_ms > 0:
            promotions = int(
                max(0, now_ms - entry.enqueued_at_ms) / self._priority_aging_ms
            )
            effective = max(0, base - promotions)
        else:
            effective = base
        return effective, int(entry.enqueued_at_ms), entry.entry_id

    # -- enqueue / claim -----------------------------------------------------

    def enqueue(
        self,
        *,
        task_cid: str,
        attempt_id: str,
        worktree_id: str,
        branch_name: str,
        commit_sha: str,
        fencing_token: int,
        fence_epoch: int,
        priority: str = "P2",
        target_repository_id: str = "",
        target_branch: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeQueueEntry:
        """Atomically enqueue or return the existing task-and-commit entry."""

        task = _text(task_cid, "task_cid")
        attempt = _text(attempt_id, "attempt_id")
        worktree = _text(worktree_id, "worktree_id")
        branch = _text(branch_name, "branch_name")
        commit = _text(commit_sha, "commit_sha")
        token = _positive_int(int(fencing_token), "fencing_token")
        epoch = _positive_int(int(fence_epoch), "fence_epoch")
        prio = _normalise_priority(priority)
        repo = _text(target_repository_id, "target_repository_id", required=False)
        tbranch = _text(target_branch, "target_branch", required=False)
        if bool(repo) != bool(tbranch):
            raise DatabaseMergeError(
                "target_repository_id and target_branch must be supplied together"
            )
        scope = f"{repo}:{tbranch}" if repo else "default"
        payload = _bounded_mapping(body, name="body")
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    """
                    SELECT * FROM merge_queue_entries
                    WHERE task_cid = ? AND commit_sha = ? AND target_scope = ?
                    """,
                    [task, commit, scope],
                ).fetchone()
                if existing is not None:
                    entry = self._entry_from_row(existing)
                    self._commit_if_idle(connection)
                    return entry
                entry_id = _new_id("merge")
                connection.execute(
                    """
                    INSERT INTO merge_queue_entries(
                        entry_id, task_cid, attempt_id, worktree_id, branch_name,
                        commit_sha, priority, status, fencing_token, fence_epoch,
                        enqueued_at_ms, target_repository_id, target_branch,
                        target_scope, revision, body_json, updated_at_ms
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?
                    )
                    """,
                    [
                        entry_id,
                        task,
                        attempt,
                        worktree,
                        branch,
                        commit,
                        prio,
                        MergeEntryStatus.PENDING.value,
                        token,
                        epoch,
                        now,
                        repo,
                        tbranch,
                        scope,
                        _canonical_json(payload),
                        now,
                    ],
                )
                self._append_event(
                    connection,
                    stream_key=f"task:{task}",
                    event_type="merge_enqueued",
                    body={
                        "entry_id": entry_id,
                        "task_cid": task,
                        "commit_sha": commit,
                        "target_scope": scope,
                    },
                    now_ms=now,
                )
                entry = self._load_entry(connection, entry_id)
                assert entry is not None
                self._commit_if_idle(connection)
                return entry
            except Exception:
                self._rollback_if_open(connection)
                raise

    def claim(
        self,
        *,
        consumer_id: str,
        target_repository_id: str = "",
        target_branch: str = "",
        limit: int = 1,
    ) -> tuple[MergeQueueEntry, ...]:
        """Claim ready pending entries with serialized per-scope fairness."""

        consumer = _text(consumer_id, "consumer_id")
        bound = _positive_int(int(limit), "limit")
        repo = _text(target_repository_id, "target_repository_id", required=False)
        tbranch = _text(target_branch, "target_branch", required=False)
        if bool(repo) != bool(tbranch):
            raise DatabaseMergeError(
                "target_repository_id and target_branch must be supplied together"
            )
        scope_filter = f"{repo}:{tbranch}" if repo else ""
        now = self._now_ms()
        claimed: list[MergeQueueEntry] = []
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                rows = connection.execute(
                    """
                    SELECT * FROM merge_queue_entries
                    WHERE status = ?
                    """,
                    [MergeEntryStatus.PENDING.value],
                ).fetchall()
                entries = [self._entry_from_row(row) for row in rows]
                if scope_filter:
                    entries = [
                        item for item in entries if item.target_scope == scope_filter
                    ]
                entries.sort(key=lambda item: self._fairness_key(item, now))
                for candidate in entries:
                    if len(claimed) >= bound:
                        break
                    active = connection.execute(
                        """
                        SELECT COUNT(*) AS active_count FROM merge_queue_entries
                        WHERE target_scope = ?
                          AND status IN (?, ?, ?, ?, ?)
                        """,
                        [
                            candidate.target_scope,
                            MergeEntryStatus.CLAIMED.value,
                            MergeEntryStatus.VALIDATING.value,
                            MergeEntryStatus.MERGING.value,
                            MergeEntryStatus.ACCEPTED.value,
                            MergeEntryStatus.PUBLISHED.value,
                        ],
                    ).fetchone()
                    active_mapping = _row_mapping(active)
                    active_count = int(
                        _row_get(active_mapping, "active_count", default=0) or 0
                    )
                    if active_count >= self._max_processing_per_scope:
                        continue
                    claim_token = uuid.uuid4().hex
                    updated = connection.execute(
                        """
                        UPDATE merge_queue_entries
                        SET status = ?, claim_token = ?,
                            claim_generation = claim_generation + 1,
                            consumer_id = ?, claimed_at_ms = ?,
                            updated_at_ms = ?, revision = revision + 1
                        WHERE entry_id = ? AND status = ?
                        """,
                        [
                            MergeEntryStatus.CLAIMED.value,
                            claim_token,
                            consumer,
                            now,
                            now,
                            candidate.entry_id,
                            MergeEntryStatus.PENDING.value,
                        ],
                    )
                    if getattr(updated, "rowcount", 1) == 0:
                        continue
                    entry = self._load_entry(connection, candidate.entry_id)
                    if entry is None or entry.status is not MergeEntryStatus.CLAIMED:
                        continue
                    self._append_event(
                        connection,
                        stream_key=f"task:{entry.task_cid}",
                        event_type="merge_claimed",
                        body={
                            "entry_id": entry.entry_id,
                            "consumer_id": consumer,
                            "claim_generation": entry.claim_generation,
                        },
                        now_ms=now,
                    )
                    claimed.append(entry)
                self._commit_if_idle(connection)
                return tuple(claimed)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_entry(self, entry_id: str) -> MergeQueueEntry | None:
        eid = _text(entry_id, "entry_id")
        with self._lock:
            connection = self._require()
            return self._load_entry(connection, eid)

    def list_entries(
        self,
        *,
        status: MergeEntryStatus | str | None = None,
        task_cid: str = "",
        limit: int = 100,
    ) -> tuple[MergeQueueEntry, ...]:
        bound = _positive_int(int(limit), "limit")
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            status_value = (
                status.value
                if isinstance(status, MergeEntryStatus)
                else str(status).strip().lower()
            )
            clauses.append("status = ?")
            params.append(status_value)
        if task_cid:
            clauses.append("task_cid = ?")
            params.append(_text(task_cid, "task_cid"))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                f"""
                SELECT * FROM merge_queue_entries
                {where}
                ORDER BY enqueued_at_ms ASC, entry_id ASC
                LIMIT {bound}
                """,
                params,
            ).fetchall()
            return tuple(self._entry_from_row(row) for row in rows)

    # -- validation ----------------------------------------------------------

    def start_validation(
        self,
        *,
        entry_id: str,
        claim_token: str,
        claim_generation: int,
        commands: Sequence[str],
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> ValidationRun:
        """Start a validation run under the current claim fence."""

        eid = _text(entry_id, "entry_id")
        token = _text(claim_token, "claim_token")
        generation = _positive_int(int(claim_generation), "claim_generation")
        command_list = tuple(_text(item, "command") for item in commands)
        if not command_list:
            raise DatabaseMergeError("validation commands are required")
        if len(command_list) > MAX_VALIDATION_COMMANDS:
            raise DatabaseMergeBoundsError("too many validation commands")
        payload = _bounded_mapping(body, name="body")
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, eid)
                if entry is None:
                    raise DatabaseMergeError(f"unknown merge entry {eid}")
                self._require_claim_match(
                    entry,
                    claim_token=token,
                    claim_generation=generation,
                    consumer_id=consumer_id,
                    operation="start_validation",
                )
                # Supersede prior unfinished runs for this entry.
                connection.execute(
                    """
                    UPDATE validation_runs
                    SET status = ?, revision = revision + 1
                    WHERE entry_id = ? AND status = ?
                    """,
                    [
                        ValidationStatus.SUPERSEDED.value,
                        eid,
                        ValidationStatus.STARTED.value,
                    ],
                )
                run_id = _new_id("validation")
                connection.execute(
                    """
                    INSERT INTO validation_runs(
                        run_id, entry_id, task_cid, attempt_id, worktree_id,
                        fencing_token, fence_epoch, status, started_at_ms,
                        commands_json, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    [
                        run_id,
                        eid,
                        entry.task_cid,
                        entry.attempt_id,
                        entry.worktree_id,
                        entry.fencing_token,
                        entry.fence_epoch,
                        ValidationStatus.STARTED.value,
                        now,
                        _canonical_json(list(command_list)),
                        _canonical_json(payload),
                    ],
                )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?, current_validation_run_id = ?,
                        revision = revision + 1, updated_at_ms = ?
                    WHERE entry_id = ?
                    """,
                    [MergeEntryStatus.VALIDATING.value, run_id, now, eid],
                )
                self._append_event(
                    connection,
                    stream_key=f"task:{entry.task_cid}",
                    event_type="validation_started",
                    body={"entry_id": eid, "run_id": run_id},
                    now_ms=now,
                )
                run = self._load_validation(connection, run_id)
                assert run is not None
                self._commit_if_idle(connection)
                return run
            except Exception:
                self._rollback_if_open(connection)
                raise

    def finish_validation(
        self,
        *,
        run_id: str,
        claim_token: str,
        claim_generation: int,
        passed: bool,
        evidence_digest: str = "",
        result_summary: str = "",
        worktree_id: str | None = None,
        fencing_token: int | None = None,
        fence_epoch: int | None = None,
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> ValidationRun:
        """Record validation outcome; reject stale worktree/fence results."""

        rid = _text(run_id, "run_id")
        token = _text(claim_token, "claim_token")
        generation = _positive_int(int(claim_generation), "claim_generation")
        payload = _bounded_mapping(body, name="body")
        now = self._now_ms()
        committed_stale_error: DatabaseMergeStaleError | None = None
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                run = self._load_validation(connection, rid)
                if run is None:
                    raise DatabaseMergeError(f"unknown validation run {rid}")
                if run.status is not ValidationStatus.STARTED:
                    raise DatabaseMergeStaleError(
                        f"validation run {rid} is not active ({run.status.value})"
                    )
                entry_row = connection.execute(
                    "SELECT entry_id FROM validation_runs WHERE run_id = ?",
                    [rid],
                ).fetchone()
                entry_id = str(
                    _row_get(_row_mapping(entry_row), "entry_id", default="")
                )
                entry = self._load_entry(connection, entry_id)
                if entry is None:
                    raise DatabaseMergeError(
                        f"validation run {rid} has no merge entry"
                    )
                self._require_claim_match(
                    entry,
                    claim_token=token,
                    claim_generation=generation,
                    consumer_id=consumer_id,
                    operation="finish_validation",
                )
                observed_worktree = (
                    entry.worktree_id
                    if worktree_id is None
                    else _text(worktree_id, "worktree_id")
                )
                observed_token = (
                    entry.fencing_token
                    if fencing_token is None
                    else _positive_int(int(fencing_token), "fencing_token")
                )
                observed_epoch = (
                    entry.fence_epoch
                    if fence_epoch is None
                    else _positive_int(int(fence_epoch), "fence_epoch")
                )
                stale = False
                try:
                    self._require_fence_match(
                        entry,
                        worktree_id=observed_worktree,
                        fencing_token=observed_token,
                        fence_epoch=observed_epoch,
                        operation="finish_validation",
                    )
                    if (
                        run.worktree_id != observed_worktree
                        or int(run.fencing_token) != int(observed_token)
                        or int(run.fence_epoch) != int(observed_epoch)
                    ):
                        raise DatabaseMergeStaleError(
                            f"validation run {rid} fence/worktree is stale"
                        )
                except DatabaseMergeStaleError:
                    stale = True

                if stale:
                    connection.execute(
                        """
                        UPDATE validation_runs
                        SET status = ?, finished_at_ms = ?, revision = revision + 1,
                            result_summary = ?
                        WHERE run_id = ?
                        """,
                        [
                            ValidationStatus.STALE.value,
                            now,
                            "stale worktree or fence",
                            rid,
                        ],
                    )
                    self._append_event(
                        connection,
                        stream_key=f"task:{entry.task_cid}",
                        event_type="validation_stale",
                        body={"run_id": rid, "entry_id": entry.entry_id},
                        now_ms=now,
                    )
                    self._commit_if_idle(connection)
                    committed_stale_error = DatabaseMergeStaleError(
                        f"finish_validation rejected: worktree or fence is stale "
                        f"for run {rid}"
                    )
                else:
                    if passed:
                        digest = _text(evidence_digest, "evidence_digest")
                        status = ValidationStatus.PASSED
                        entry_status = MergeEntryStatus.CLAIMED
                    else:
                        digest = _text(
                            evidence_digest, "evidence_digest", required=False
                        )
                        status = ValidationStatus.FAILED
                        entry_status = MergeEntryStatus.FAILED

                    connection.execute(
                        """
                        UPDATE validation_runs
                        SET status = ?, finished_at_ms = ?, evidence_digest = ?,
                            result_summary = ?, revision = revision + 1,
                            body_json = ?
                        WHERE run_id = ?
                        """,
                        [
                            status.value,
                            now,
                            digest,
                            _text(result_summary, "result_summary", required=False),
                            _canonical_json(payload),
                            rid,
                        ],
                    )
                    connection.execute(
                        """
                        INSERT INTO validation_results(
                            result_id, run_id, entry_id, task_cid, status,
                            evidence_digest, recorded_at_ms, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            _new_id("vresult"),
                            rid,
                            entry.entry_id,
                            entry.task_cid,
                            status.value,
                            digest,
                            now,
                            _canonical_json(
                                {
                                    "schema": VALIDATION_RESULT_SCHEMA,
                                    "summary": result_summary,
                                }
                            ),
                        ],
                    )
                    connection.execute(
                        """
                        UPDATE merge_queue_entries
                        SET status = ?, revision = revision + 1, updated_at_ms = ?,
                            failure_count = failure_count + ?
                        WHERE entry_id = ?
                        """,
                        [
                            entry_status.value,
                            now,
                            0 if passed else 1,
                            entry.entry_id,
                        ],
                    )
                    self._append_event(
                        connection,
                        stream_key=f"task:{entry.task_cid}",
                        event_type=(
                            "validation_passed" if passed else "validation_failed"
                        ),
                        body={
                            "run_id": rid,
                            "entry_id": entry.entry_id,
                            "evidence_digest": digest,
                        },
                        now_ms=now,
                    )
                    finished = self._load_validation(connection, rid)
                    assert finished is not None
                    self._commit_if_idle(connection)
                    return finished
            except Exception:
                self._rollback_if_open(connection)
                raise
        if committed_stale_error is not None:
            raise committed_stale_error
        raise DatabaseMergeError("finish_validation failed without a result")

    def get_validation_run(self, run_id: str) -> ValidationRun | None:
        rid = _text(run_id, "run_id")
        with self._lock:
            connection = self._require()
            return self._load_validation(connection, rid)

    def current_validation(
        self, entry_id: str
    ) -> ValidationRun | None:
        eid = _text(entry_id, "entry_id")
        with self._lock:
            connection = self._require()
            entry = self._load_entry(connection, eid)
            if entry is None or not entry.current_validation_run_id:
                return None
            return self._load_validation(
                connection, entry.current_validation_run_id
            )

    # -- merge attempts ------------------------------------------------------

    def begin_merge_attempt(
        self,
        *,
        entry_id: str,
        claim_token: str,
        claim_generation: int,
        base_commit: str = "",
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        eid = _text(entry_id, "entry_id")
        token = _text(claim_token, "claim_token")
        generation = _positive_int(int(claim_generation), "claim_generation")
        payload = _bounded_mapping(body, name="body")
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, eid)
                if entry is None:
                    raise DatabaseMergeError(f"unknown merge entry {eid}")
                self._require_claim_match(
                    entry,
                    claim_token=token,
                    claim_generation=generation,
                    consumer_id=consumer_id,
                    operation="begin_merge_attempt",
                )
                attempt_id = _new_id("mattempt")
                connection.execute(
                    """
                    INSERT INTO merge_attempts(
                        attempt_id, entry_id, task_cid, worktree_id,
                        fencing_token, fence_epoch, claim_token, claim_generation,
                        status, started_at_ms, base_commit, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    [
                        attempt_id,
                        eid,
                        entry.task_cid,
                        entry.worktree_id,
                        entry.fencing_token,
                        entry.fence_epoch,
                        token,
                        generation,
                        MergeAttemptStatus.STARTED.value,
                        now,
                        _text(base_commit, "base_commit", required=False),
                        _canonical_json(payload),
                    ],
                )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?, current_merge_attempt_id = ?,
                        revision = revision + 1, updated_at_ms = ?
                    WHERE entry_id = ?
                    """,
                    [MergeEntryStatus.MERGING.value, attempt_id, now, eid],
                )
                self._append_event(
                    connection,
                    stream_key=f"task:{entry.task_cid}",
                    event_type="merge_attempt_started",
                    body={"entry_id": eid, "attempt_id": attempt_id},
                    now_ms=now,
                )
                attempt = self._load_merge_attempt(connection, attempt_id)
                assert attempt is not None
                self._commit_if_idle(connection)
                return attempt
            except Exception:
                self._rollback_if_open(connection)
                raise

    def record_rebase(
        self,
        *,
        attempt_id: str,
        claim_token: str,
        claim_generation: int,
        result_commit: str,
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        return self._transition_merge_attempt(
            attempt_id=attempt_id,
            claim_token=claim_token,
            claim_generation=claim_generation,
            status=MergeAttemptStatus.REBASED,
            result_commit=result_commit,
            consumer_id=consumer_id,
            body=body,
            event_type="merge_rebased",
        )

    def record_conflict(
        self,
        *,
        attempt_id: str,
        claim_token: str,
        claim_generation: int,
        conflict_paths: Sequence[str],
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        return self._transition_merge_attempt(
            attempt_id=attempt_id,
            claim_token=claim_token,
            claim_generation=claim_generation,
            status=MergeAttemptStatus.CONFLICT,
            conflict_paths=conflict_paths,
            consumer_id=consumer_id,
            body=body,
            event_type="merge_conflict",
            fail_entry=True,
        )

    def accept_merge(
        self,
        *,
        attempt_id: str,
        claim_token: str,
        claim_generation: int,
        result_commit: str,
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        return self._transition_merge_attempt(
            attempt_id=attempt_id,
            claim_token=claim_token,
            claim_generation=claim_generation,
            status=MergeAttemptStatus.ACCEPTED,
            result_commit=result_commit,
            consumer_id=consumer_id,
            body=body,
            event_type="merge_accepted",
            mark_accepted=True,
        )

    def record_partial_publish(
        self,
        *,
        attempt_id: str,
        claim_token: str,
        claim_generation: int,
        result_commit: str = "",
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        return self._transition_merge_attempt(
            attempt_id=attempt_id,
            claim_token=claim_token,
            claim_generation=claim_generation,
            status=MergeAttemptStatus.PARTIAL_PUBLISH,
            result_commit=result_commit,
            consumer_id=consumer_id,
            body=body,
            event_type="merge_partial_publish",
            entry_status=MergeEntryStatus.PUBLISHED,
        )

    def publish_merge(
        self,
        *,
        attempt_id: str,
        claim_token: str,
        claim_generation: int,
        result_commit: str,
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        return self._transition_merge_attempt(
            attempt_id=attempt_id,
            claim_token=claim_token,
            claim_generation=claim_generation,
            status=MergeAttemptStatus.PUBLISHED,
            result_commit=result_commit,
            consumer_id=consumer_id,
            body=body,
            event_type="merge_published",
            mark_accepted=True,
            entry_status=MergeEntryStatus.PUBLISHED,
        )

    def _transition_merge_attempt(
        self,
        *,
        attempt_id: str,
        claim_token: str,
        claim_generation: int,
        status: MergeAttemptStatus,
        result_commit: str = "",
        conflict_paths: Sequence[str] = (),
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
        event_type: str,
        mark_accepted: bool = False,
        fail_entry: bool = False,
        entry_status: MergeEntryStatus | None = None,
    ) -> MergeAttempt:
        aid = _text(attempt_id, "attempt_id")
        token = _text(claim_token, "claim_token")
        generation = _positive_int(int(claim_generation), "claim_generation")
        payload = _bounded_mapping(body, name="body")
        paths = tuple(_text(item, "conflict_path") for item in conflict_paths)
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                attempt = self._load_merge_attempt(connection, aid)
                if attempt is None:
                    raise DatabaseMergeError(f"unknown merge attempt {aid}")
                entry = self._load_entry(connection, attempt.entry_id)
                if entry is None:
                    raise DatabaseMergeError(
                        f"merge attempt {aid} has no entry"
                    )
                self._require_claim_match(
                    entry,
                    claim_token=token,
                    claim_generation=generation,
                    consumer_id=consumer_id,
                    operation=event_type,
                )
                self._require_fence_match(
                    entry,
                    worktree_id=attempt.worktree_id,
                    fencing_token=attempt.fencing_token,
                    fence_epoch=attempt.fence_epoch,
                    operation=event_type,
                )
                if (
                    attempt.claim_token != token
                    or int(attempt.claim_generation) != generation
                ):
                    raise DatabaseMergeStaleError(
                        f"{event_type} rejected: merge attempt claim is stale"
                    )
                commit_text = _text(
                    result_commit or attempt.result_commit,
                    "result_commit",
                    required=status
                    in {
                        MergeAttemptStatus.ACCEPTED,
                        MergeAttemptStatus.PUBLISHED,
                        MergeAttemptStatus.REBASED,
                    },
                )
                connection.execute(
                    """
                    UPDATE merge_attempts
                    SET status = ?, finished_at_ms = ?, result_commit = ?,
                        conflict_paths_json = ?, revision = revision + 1,
                        body_json = ?
                    WHERE attempt_id = ?
                    """,
                    [
                        status.value,
                        now,
                        commit_text,
                        _canonical_json(list(paths)),
                        _canonical_json(payload),
                        aid,
                    ],
                )
                next_entry_status = entry_status
                if fail_entry:
                    next_entry_status = MergeEntryStatus.FAILED
                elif mark_accepted:
                    next_entry_status = next_entry_status or MergeEntryStatus.ACCEPTED
                if next_entry_status is not None:
                    if mark_accepted:
                        connection.execute(
                            """
                            UPDATE merge_queue_entries
                            SET status = ?,
                                accepted_merge_attempt_id = ?,
                                failure_count = failure_count + ?,
                                revision = revision + 1,
                                updated_at_ms = ?
                            WHERE entry_id = ?
                            """,
                            [
                                next_entry_status.value,
                                aid,
                                1 if fail_entry else 0,
                                now,
                                entry.entry_id,
                            ],
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE merge_queue_entries
                            SET status = ?,
                                failure_count = failure_count + ?,
                                revision = revision + 1,
                                updated_at_ms = ?
                            WHERE entry_id = ?
                            """,
                            [
                                next_entry_status.value,
                                1 if fail_entry else 0,
                                now,
                                entry.entry_id,
                            ],
                        )
                self._append_event(
                    connection,
                    stream_key=f"task:{entry.task_cid}",
                    event_type=event_type,
                    body={
                        "entry_id": entry.entry_id,
                        "attempt_id": aid,
                        "status": status.value,
                        "result_commit": commit_text,
                        "conflict_paths": list(paths),
                    },
                    now_ms=now,
                )
                updated = self._load_merge_attempt(connection, aid)
                assert updated is not None
                self._commit_if_idle(connection)
                return updated
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_merge_attempt(self, attempt_id: str) -> MergeAttempt | None:
        aid = _text(attempt_id, "attempt_id")
        with self._lock:
            connection = self._require()
            return self._load_merge_attempt(connection, aid)

    # -- completion (atomic merge + validation) ------------------------------

    def complete_task(
        self,
        *,
        entry_id: str,
        claim_token: str,
        claim_generation: int,
        consumer_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> CompletionReceipt:
        """Complete a task only when accepted merge and current validation exist.

        Both prerequisites are verified and the completion receipt is written in
        one transaction. Neither a JSON receipt nor a queue file can settle work.
        """

        eid = _text(entry_id, "entry_id")
        token = _text(claim_token, "claim_token")
        generation = _positive_int(int(claim_generation), "claim_generation")
        payload = _bounded_mapping(body, name="body")
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, eid)
                if entry is None:
                    raise DatabaseMergeError(f"unknown merge entry {eid}")
                if entry.status is MergeEntryStatus.COMPLETED and entry.completion_receipt_id:
                    existing = connection.execute(
                        "SELECT * FROM completion_receipts WHERE receipt_id = ?",
                        [entry.completion_receipt_id],
                    ).fetchone()
                    if existing is not None:
                        self._commit_if_idle(connection)
                        return self._completion_from_row(existing)
                self._require_claim_match(
                    entry,
                    claim_token=token,
                    claim_generation=generation,
                    consumer_id=consumer_id,
                    operation="complete_task",
                )
                if not entry.accepted_merge_attempt_id:
                    raise DatabaseMergeNotReadyError(
                        "completion requires an accepted merge attempt"
                    )
                merge_attempt = self._load_merge_attempt(
                    connection, entry.accepted_merge_attempt_id
                )
                if merge_attempt is None or not merge_attempt.accepted:
                    raise DatabaseMergeNotReadyError(
                        "accepted merge attempt is missing or not accepted"
                    )
                if (
                    merge_attempt.worktree_id != entry.worktree_id
                    or int(merge_attempt.fencing_token) != int(entry.fencing_token)
                    or int(merge_attempt.fence_epoch) != int(entry.fence_epoch)
                ):
                    raise DatabaseMergeStaleError(
                        "accepted merge attempt fence/worktree is stale"
                    )
                if not entry.current_validation_run_id:
                    raise DatabaseMergeNotReadyError(
                        "completion requires current validation evidence"
                    )
                validation = self._load_validation(
                    connection, entry.current_validation_run_id
                )
                if validation is None or not validation.current_and_passed:
                    raise DatabaseMergeNotReadyError(
                        "current validation evidence is missing or not passed"
                    )
                if (
                    validation.worktree_id != entry.worktree_id
                    or int(validation.fencing_token) != int(entry.fencing_token)
                    or int(validation.fence_epoch) != int(entry.fence_epoch)
                ):
                    raise DatabaseMergeStaleError(
                        "validation evidence fence/worktree is stale"
                    )
                if not merge_attempt.result_commit:
                    raise DatabaseMergeNotReadyError(
                        "accepted merge requires a result commit"
                    )
                # Partial publish alone cannot complete.
                if merge_attempt.status is MergeAttemptStatus.PARTIAL_PUBLISH:
                    raise DatabaseMergeNotReadyError(
                        "partial publication cannot settle task completion"
                    )
                receipt_id = _new_id("completion")
                connection.execute(
                    """
                    INSERT INTO completion_receipts(
                        receipt_id, task_cid, entry_id, validation_run_id,
                        merge_attempt_id, worktree_id, fencing_token, fence_epoch,
                        evidence_digest, result_commit, completed_at_ms, status,
                        body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt_id,
                        entry.task_cid,
                        entry.entry_id,
                        validation.run_id,
                        merge_attempt.attempt_id,
                        entry.worktree_id,
                        entry.fencing_token,
                        entry.fence_epoch,
                        validation.evidence_digest,
                        merge_attempt.result_commit,
                        now,
                        CompletionStatus.COMPLETED.value,
                        _canonical_json(payload),
                    ],
                )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?, completion_receipt_id = ?,
                        claim_token = '', consumer_id = '',
                        claim_generation = claim_generation + 1,
                        claimed_at_ms = 0,
                        revision = revision + 1, updated_at_ms = ?
                    WHERE entry_id = ?
                      AND claim_token = ?
                      AND claim_generation = ?
                    """,
                    [
                        MergeEntryStatus.COMPLETED.value,
                        receipt_id,
                        now,
                        eid,
                        token,
                        generation,
                    ],
                )
                self._append_event(
                    connection,
                    stream_key=f"task:{entry.task_cid}",
                    event_type="task_completed",
                    body={
                        "entry_id": eid,
                        "receipt_id": receipt_id,
                        "validation_run_id": validation.run_id,
                        "merge_attempt_id": merge_attempt.attempt_id,
                        "evidence_digest": validation.evidence_digest,
                        "result_commit": merge_attempt.result_commit,
                    },
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM completion_receipts WHERE receipt_id = ?",
                    [receipt_id],
                ).fetchone()
                assert row is not None
                receipt = self._completion_from_row(row)
                self._commit_if_idle(connection)
                return receipt
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_completion(self, task_cid: str) -> CompletionReceipt | None:
        task = _text(task_cid, "task_cid")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM completion_receipts WHERE task_cid = ?",
                [task],
            ).fetchone()
            return None if row is None else self._completion_from_row(row)

    def is_task_complete(self, task_cid: str) -> bool:
        """Return whether the database holds an authoritative completion receipt.

        JSON mirrors and queue files are ignored.
        """

        return self.get_completion(task_cid) is not None

    # -- optional JSON projection (never authority) --------------------------

    def project_json_receipt(
        self,
        *,
        entry_id: str,
        path: str | Path,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write an optional human-readable receipt that cannot grant completion."""

        eid = _text(entry_id, "entry_id")
        target = Path(path)
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            entry = self._load_entry(connection, eid)
            if entry is None:
                raise DatabaseMergeError(f"unknown merge entry {eid}")
            body = {
                "schema": JSON_RECEIPT_MIRROR_SCHEMA,
                "entry_id": eid,
                "task_cid": entry.task_cid,
                "status": entry.status.value,
                "grants_completion": False,
                "authority": "database",
                "payload": dict(payload or entry.to_dict()),
            }
            encoded = _canonical_json(body)
            digest = _sha256_hex(encoded.encode("utf-8"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(encoded + "\n", encoding="utf-8")
            self._begin(connection)
            try:
                connection.execute(
                    """
                    INSERT INTO json_receipt_mirrors(
                        mirror_id, entry_id, path, payload_digest,
                        recorded_at_ms, grants_completion, body_json
                    ) VALUES (?, ?, ?, ?, ?, FALSE, ?)
                    """,
                    [
                        _new_id("mirror"),
                        eid,
                        str(target),
                        digest,
                        now,
                        encoded,
                    ],
                )
                self._commit_if_idle(connection)
            except Exception:
                self._rollback_if_open(connection)
                raise
            return {
                "path": str(target),
                "payload_digest": digest,
                "grants_completion": False,
                "entry_id": eid,
            }

    def import_json_receipt_cannot_complete(
        self,
        *,
        task_cid: str,
        path: str | Path,
    ) -> dict[str, Any]:
        """Demonstrate that a JSON receipt path cannot settle completion."""

        task = _text(task_cid, "task_cid")
        target = Path(path)
        payload: dict[str, Any] = {}
        if target.is_file():
            try:
                loaded = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping):
                    payload = dict(loaded)
            except (OSError, json.JSONDecodeError):
                payload = {}
        complete = self.is_task_complete(task)
        return {
            "task_cid": task,
            "path": str(target),
            "json_claims_complete": bool(
                payload.get("status") in {"completed", "complete"}
                or payload.get("completed") is True
            ),
            "database_complete": complete,
            "settled": complete,
            "json_grants_completion": False,
        }

    # -- recovery support used by DatabaseRecovery ---------------------------

    def requeue_for_retry(
        self,
        *,
        entry_id: str,
        reason: str = "",
        claim_token: str = "",
        claim_generation: int = 0,
        force: bool = False,
    ) -> MergeQueueEntry:
        """Return entry to pending when retries remain, else quarantine."""

        eid = _text(entry_id, "entry_id")
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, eid)
                if entry is None:
                    raise DatabaseMergeError(f"unknown merge entry {eid}")
                if entry.status is MergeEntryStatus.COMPLETED:
                    self._commit_if_idle(connection)
                    return entry
                if entry.status is MergeEntryStatus.QUARANTINED:
                    self._commit_if_idle(connection)
                    return entry
                if not force:
                    self._require_claim_match(
                        entry,
                        claim_token=_text(claim_token, "claim_token"),
                        claim_generation=_positive_int(
                            int(claim_generation), "claim_generation"
                        ),
                        operation="requeue_for_retry",
                        allow_failed=True,
                    )
                next_attempt = int(entry.attempt_number) + 1
                failure_count = int(entry.failure_count) + 1
                if next_attempt > self._max_attempts:
                    status = MergeEntryStatus.QUARANTINED
                else:
                    status = MergeEntryStatus.PENDING
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?, attempt_number = ?, failure_count = ?,
                        claim_token = '', consumer_id = '',
                        claim_generation = claim_generation + 1,
                        claimed_at_ms = 0,
                        accepted_merge_attempt_id = '',
                        current_merge_attempt_id = '',
                        current_validation_run_id = '',
                        revision = revision + 1, updated_at_ms = ?,
                        body_json = ?
                    WHERE entry_id = ?
                    """,
                    [
                        status.value,
                        next_attempt if status is MergeEntryStatus.PENDING else entry.attempt_number,
                        failure_count,
                        now,
                        _canonical_json(
                            {
                                **dict(entry.body),
                                "last_retry_reason": _text(
                                    reason, "reason", required=False
                                ),
                            }
                        ),
                        eid,
                    ],
                )
                self._append_event(
                    connection,
                    stream_key=f"task:{entry.task_cid}",
                    event_type=(
                        "merge_quarantined"
                        if status is MergeEntryStatus.QUARANTINED
                        else "merge_requeued"
                    ),
                    body={
                        "entry_id": eid,
                        "reason": reason,
                        "attempt_number": next_attempt,
                        "status": status.value,
                    },
                    now_ms=now,
                )
                updated = self._load_entry(connection, eid)
                assert updated is not None
                self._commit_if_idle(connection)
                return updated
            except Exception:
                self._rollback_if_open(connection)
                raise

    def quarantine_entry(
        self,
        *,
        entry_id: str,
        reason: str = "",
        claim_token: str = "",
        claim_generation: int = 0,
        force: bool = False,
    ) -> MergeQueueEntry:
        eid = _text(entry_id, "entry_id")
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, eid)
                if entry is None:
                    raise DatabaseMergeError(f"unknown merge entry {eid}")
                if entry.status is MergeEntryStatus.QUARANTINED:
                    self._commit_if_idle(connection)
                    return entry
                if entry.status is MergeEntryStatus.COMPLETED:
                    raise DatabaseMergeConflictError(
                        "completed entries cannot be quarantined"
                    )
                if not force:
                    self._require_claim_match(
                        entry,
                        claim_token=_text(claim_token, "claim_token"),
                        claim_generation=_positive_int(
                            int(claim_generation), "claim_generation"
                        ),
                        operation="quarantine_entry",
                        allow_failed=True,
                    )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?, claim_token = '', consumer_id = '',
                        claim_generation = claim_generation + 1,
                        claimed_at_ms = 0,
                        revision = revision + 1, updated_at_ms = ?,
                        body_json = ?
                    WHERE entry_id = ?
                    """,
                    [
                        MergeEntryStatus.QUARANTINED.value,
                        now,
                        _canonical_json(
                            {
                                **dict(entry.body),
                                "quarantine_reason": _text(
                                    reason, "reason", required=False
                                ),
                            }
                        ),
                        eid,
                    ],
                )
                self._append_event(
                    connection,
                    stream_key=f"task:{entry.task_cid}",
                    event_type="merge_quarantined",
                    body={"entry_id": eid, "reason": reason},
                    now_ms=now,
                )
                updated = self._load_entry(connection, eid)
                assert updated is not None
                self._commit_if_idle(connection)
                return updated
            except Exception:
                self._rollback_if_open(connection)
                raise

    def release_stale_claim(
        self,
        *,
        entry_id: str,
        expected_claim_token: str,
        expected_claim_generation: int,
        reason: str = "crash_recovery",
    ) -> MergeQueueEntry:
        """Release a claim only when the expected fence still matches (CAS)."""

        eid = _text(entry_id, "entry_id")
        token = _text(expected_claim_token, "expected_claim_token")
        generation = _positive_int(
            int(expected_claim_generation), "expected_claim_generation"
        )
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, eid)
                if entry is None:
                    raise DatabaseMergeError(f"unknown merge entry {eid}")
                if entry.status is MergeEntryStatus.COMPLETED:
                    self._commit_if_idle(connection)
                    return entry
                if (
                    entry.claim_token != token
                    or int(entry.claim_generation) != generation
                ):
                    raise DatabaseMergeStaleError(
                        f"release_stale_claim rejected: claim fence is stale for {eid}"
                    )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?, claim_token = '', consumer_id = '',
                        claim_generation = claim_generation + 1,
                        claimed_at_ms = 0,
                        revision = revision + 1, updated_at_ms = ?
                    WHERE entry_id = ?
                      AND claim_token = ?
                      AND claim_generation = ?
                    """,
                    [
                        MergeEntryStatus.PENDING.value,
                        now,
                        eid,
                        token,
                        generation,
                    ],
                )
                self._append_event(
                    connection,
                    stream_key=f"task:{entry.task_cid}",
                    event_type="merge_claim_released",
                    body={"entry_id": eid, "reason": reason},
                    now_ms=now,
                )
                updated = self._load_entry(connection, eid)
                assert updated is not None
                self._commit_if_idle(connection)
                return updated
            except Exception:
                self._rollback_if_open(connection)
                raise

    def domain_events(
        self,
        *,
        stream_key: str = "",
        event_type: str = "",
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        bound = _positive_int(int(limit), "limit")
        clauses: list[str] = []
        params: list[Any] = []
        if stream_key:
            clauses.append("stream_key = ?")
            params.append(_text(stream_key, "stream_key"))
        if event_type:
            clauses.append("event_type = ?")
            params.append(_text(event_type, "event_type"))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                f"""
                SELECT * FROM domain_events
                {where}
                ORDER BY observed_at_ms ASC, sequence_no ASC
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
                        "event_id": str(_row_get(mapping, "event_id", default="")),
                        "stream_key": str(
                            _row_get(mapping, "stream_key", default="")
                        ),
                        "event_type": str(
                            _row_get(mapping, "event_type", default="")
                        ),
                        "sequence_no": int(
                            _row_get(mapping, "sequence_no", default=0) or 0
                        ),
                        "observed_at_ms": int(
                            _row_get(mapping, "observed_at_ms", default=0) or 0
                        ),
                        "body": body if isinstance(body, Mapping) else {},
                    }
                )
            return tuple(events)

    # -- recovery action storage (shared with DatabaseRecovery) --------------

    def record_recovery_action(
        self,
        *,
        action_kind: str,
        status: str,
        task_cid: str = "",
        entry_id: str = "",
        worktree_id: str = "",
        fencing_token: int = 0,
        fence_epoch: int = 0,
        idempotency_key: str = "",
        retry_budget: int = 0,
        retry_count: int = 0,
        reason: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one recovery action row (idempotent under ``idempotency_key``)."""

        kind = _text(action_kind, "action_kind")
        status_text = _text(status, "status")
        key = _text(idempotency_key, "idempotency_key", required=False)
        payload = _bounded_mapping(body, name="body")
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                if key:
                    existing = connection.execute(
                        """
                        SELECT * FROM recovery_actions
                        WHERE idempotency_key = ?
                        """,
                        [key],
                    ).fetchone()
                    if existing is not None:
                        self._commit_if_idle(connection)
                        return self._recovery_action_dict(existing)
                action_id = _new_id("recovery")
                connection.execute(
                    """
                    INSERT INTO recovery_actions(
                        action_id, action_kind, task_cid, entry_id, worktree_id,
                        fencing_token, fence_epoch, status, idempotency_key,
                        retry_budget, retry_count, recorded_at_ms, finished_at_ms,
                        reason, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    [
                        action_id,
                        kind,
                        _text(task_cid, "task_cid", required=False),
                        _text(entry_id, "entry_id", required=False),
                        _text(worktree_id, "worktree_id", required=False),
                        _nonneg_int(int(fencing_token), "fencing_token"),
                        _nonneg_int(int(fence_epoch), "fence_epoch"),
                        status_text,
                        key,
                        _nonneg_int(int(retry_budget), "retry_budget"),
                        _nonneg_int(int(retry_count), "retry_count"),
                        now,
                        now,
                        _text(reason, "reason", required=False),
                        _canonical_json(payload),
                    ],
                )
                row = connection.execute(
                    "SELECT * FROM recovery_actions WHERE action_id = ?",
                    [action_id],
                ).fetchone()
                assert row is not None
                self._commit_if_idle(connection)
                return self._recovery_action_dict(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_recovery_action(self, action_id: str) -> dict[str, Any] | None:
        aid = _text(action_id, "action_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM recovery_actions WHERE action_id = ?",
                [aid],
            ).fetchone()
            return None if row is None else self._recovery_action_dict(row)

    def get_recovery_by_idempotency(
        self, idempotency_key: str
    ) -> dict[str, Any] | None:
        key = _text(idempotency_key, "idempotency_key")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM recovery_actions WHERE idempotency_key = ?",
                [key],
            ).fetchone()
            return None if row is None else self._recovery_action_dict(row)

    def list_recovery_actions(
        self,
        *,
        task_cid: str = "",
        entry_id: str = "",
        action_kind: str = "",
        status: str = "",
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        bound = _positive_int(int(limit), "limit")
        clauses: list[str] = []
        params: list[Any] = []
        if task_cid:
            clauses.append("task_cid = ?")
            params.append(_text(task_cid, "task_cid"))
        if entry_id:
            clauses.append("entry_id = ?")
            params.append(_text(entry_id, "entry_id"))
        if action_kind:
            clauses.append("action_kind = ?")
            params.append(_text(action_kind, "action_kind"))
        if status:
            clauses.append("status = ?")
            params.append(_text(status, "status"))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                f"""
                SELECT * FROM recovery_actions
                {where}
                ORDER BY recorded_at_ms ASC, action_id ASC
                LIMIT {bound}
                """,
                params,
            ).fetchall()
            return tuple(self._recovery_action_dict(row) for row in rows)

    def _recovery_action_dict(self, row: Any) -> dict[str, Any]:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        finished = _row_get(mapping, "finished_at_ms", default=None)
        return {
            "action_id": str(_row_get(mapping, "action_id", default="")),
            "action_kind": str(_row_get(mapping, "action_kind", default="")),
            "status": str(_row_get(mapping, "status", default="")),
            "task_cid": str(_row_get(mapping, "task_cid", default="") or ""),
            "entry_id": str(_row_get(mapping, "entry_id", default="") or ""),
            "worktree_id": str(_row_get(mapping, "worktree_id", default="") or ""),
            "fencing_token": int(
                _row_get(mapping, "fencing_token", default=0) or 0
            ),
            "fence_epoch": int(_row_get(mapping, "fence_epoch", default=0) or 0),
            "idempotency_key": str(
                _row_get(mapping, "idempotency_key", default="") or ""
            ),
            "retry_budget": int(_row_get(mapping, "retry_budget", default=0) or 0),
            "retry_count": int(_row_get(mapping, "retry_count", default=0) or 0),
            "recorded_at_ms": int(
                _row_get(mapping, "recorded_at_ms", default=0) or 0
            ),
            "finished_at_ms": None if finished is None else int(finished),
            "reason": str(_row_get(mapping, "reason", default="") or ""),
            "revision": int(_row_get(mapping, "revision", default=1) or 1),
            "body": body if isinstance(body, Mapping) else {},
        }


def open_database_merge_queue(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    priority_aging_ms: int = DEFAULT_PRIORITY_AGING_MS,
    max_processing_per_scope: int = 1,
) -> DatabaseMergeQueue:
    """Open a :class:`DatabaseMergeQueue` on ``database_path``."""

    return DatabaseMergeQueue(
        database_path,
        clock_ms=clock_ms,
        max_attempts=max_attempts,
        priority_aging_ms=priority_aging_ms,
        max_processing_per_scope=max_processing_per_scope,
    ).open()


__all__ = [
    "DATABASE_MERGE_QUEUE_INTERFACE",
    "VALIDATION_RUN_INTERFACE",
    "MERGE_ATTEMPT_INTERFACE",
    "COMPLETION_RECEIPT_INTERFACE",
    "DATABASE_MERGE_QUEUE_SCHEMA",
    "VALIDATION_RUN_SCHEMA",
    "MERGE_QUEUE_ENTRY_SCHEMA",
    "MERGE_ATTEMPT_SCHEMA",
    "COMPLETION_RECEIPT_SCHEMA",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_PRIORITY_AGING_MS",
    "MergeEntryStatus",
    "ValidationStatus",
    "MergeAttemptStatus",
    "CompletionStatus",
    "ValidationRun",
    "MergeQueueEntry",
    "MergeAttempt",
    "CompletionReceipt",
    "DatabaseMergeQueue",
    "DatabaseMergeError",
    "DatabaseMergeNotOpenError",
    "DatabaseMergeConflictError",
    "DatabaseMergeStaleError",
    "DatabaseMergeNotReadyError",
    "DatabaseMergeBoundsError",
    "DuckDBUnavailableError",
    "duckdb_available",
    "open_database_merge_queue",
]
