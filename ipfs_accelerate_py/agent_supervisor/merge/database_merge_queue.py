"""Transactional validation, merge queue, and settlement authority.

DQP-019 / DatabaseMergeQueue@1, ValidationRun@1
===============================================

:class:`DatabaseMergeQueue` is the durable authority for merge-queue entries,
validation runs/results, merge attempts/outcomes, and atomic settlement. A
task settles only when an **accepted merge attempt** and a **current passed
validation run** commit together under matching worktree, fence, and claim
coordinates. JSON stage receipts may be projected for humans but cannot settle
work alone.

Authority rules (fail-closed)
-----------------------------
* One exclusive merge claim is active per repository/target binding.
* Fair selection prefers higher priority, then earlier enqueue order.
* Stale fencing tokens, fence epochs, worktree identities, or claim tokens are
  rejected on every protected write.
* Validation failure, conflict, partial publish, or crash leave the entry
  non-settled and queryable.
* Retry budgets are enforced; exhaustion quarantines the entry.
* Settlement never reads or trusts an external JSON receipt path as authority.

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

from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_MERGE_QUEUE_INTERFACE: Final[str] = "DatabaseMergeQueue@1"
VALIDATION_RUN_INTERFACE: Final[str] = "ValidationRun@1"
MERGE_ATTEMPT_INTERFACE: Final[str] = "MergeAttempt@1"
SETTLEMENT_RECEIPT_INTERFACE: Final[str] = "SettlementReceipt@1"

DATABASE_MERGE_QUEUE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-merge-queue@1"
)
VALIDATION_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/validation-run@1"
)
MERGE_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-attempt@1"
)
SETTLEMENT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/settlement-receipt@1"
)
MERGE_QUEUE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-queue-entry@1"
)
MERGE_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-queue-event@1"
)

DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_MAX_PROCESSING: Final[int] = 1
MAX_PAYLOAD_BYTES: Final[int] = 262_144
_PRIORITY_ORDER: Final[dict[str, int]] = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS merge_queue_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS merge_queue_entries (
    entry_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    target_branch VARCHAR NOT NULL,
    source_branch VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    commit_sha VARCHAR NOT NULL DEFAULT '',
    priority VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    enqueued_at_ms BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL,
    claimed_at_ms BIGINT NOT NULL DEFAULT 0,
    consumer_id VARCHAR NOT NULL DEFAULT '',
    claim_token VARCHAR NOT NULL DEFAULT '',
    claim_generation BIGINT NOT NULL DEFAULT 0,
    fencing_token BIGINT NOT NULL DEFAULT 0,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    attempt_count BIGINT NOT NULL DEFAULT 0,
    failure_count BIGINT NOT NULL DEFAULT 0,
    failure_reason VARCHAR NOT NULL DEFAULT '',
    retry_not_before_ms BIGINT NOT NULL DEFAULT 0,
    current_validation_run_id VARCHAR NOT NULL DEFAULT '',
    current_merge_attempt_id VARCHAR NOT NULL DEFAULT '',
    settlement_id VARCHAR NOT NULL DEFAULT '',
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS merge_queue_entries_task_repo_uidx
    ON merge_queue_entries(repository_id, target_branch, task_cid, commit_sha);
CREATE INDEX IF NOT EXISTS merge_queue_entries_status_idx
    ON merge_queue_entries(repository_id, target_branch, status, ordinal);
CREATE INDEX IF NOT EXISTS merge_queue_entries_claim_idx
    ON merge_queue_entries(claim_token, claim_generation);

CREATE TABLE IF NOT EXISTS validation_runs (
    run_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    claim_token VARCHAR NOT NULL,
    claim_generation BIGINT NOT NULL,
    command_digest VARCHAR NOT NULL DEFAULT '',
    status VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT NOT NULL DEFAULT 0,
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    outcome VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS validation_runs_entry_idx
    ON validation_runs(entry_id, started_at_ms);
CREATE INDEX IF NOT EXISTS validation_runs_task_idx
    ON validation_runs(task_cid, started_at_ms);

CREATE TABLE IF NOT EXISTS validation_results (
    result_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS validation_results_run_ordinal_uidx
    ON validation_results(run_id, ordinal);

CREATE TABLE IF NOT EXISTS merge_attempts (
    merge_attempt_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    claim_token VARCHAR NOT NULL,
    claim_generation BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL DEFAULT '',
    result_commit_id VARCHAR NOT NULL DEFAULT '',
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS merge_attempts_entry_idx
    ON merge_attempts(entry_id, started_at_ms);
CREATE INDEX IF NOT EXISTS merge_attempts_task_idx
    ON merge_attempts(task_cid, started_at_ms);

CREATE TABLE IF NOT EXISTS settlement_receipts (
    settlement_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    merge_attempt_id VARCHAR NOT NULL,
    validation_run_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    result_commit_id VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    settled_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS settlement_receipts_entry_uidx
    ON settlement_receipts(entry_id);
CREATE INDEX IF NOT EXISTS settlement_receipts_task_idx
    ON settlement_receipts(task_cid, settled_at_ms);

CREATE TABLE IF NOT EXISTS merge_queue_events (
    event_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS merge_queue_events_entry_idx
    ON merge_queue_events(entry_id, observed_at_ms);
"""

ClockMs = Callable[[], int]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseMergeQueueError(RuntimeError):
    """Base fail-closed error for the database merge queue."""

    code = "DQP_MERGE_QUEUE_ERROR"


class DatabaseMergeQueueNotOpenError(DatabaseMergeQueueError):
    """Operation requires an open merge queue."""

    code = "DQP_MERGE_QUEUE_NOT_OPEN"


class DatabaseMergeQueueConflictError(DatabaseMergeQueueError):
    """Exclusive claim, identity, or settlement conflict."""

    code = "DQP_MERGE_QUEUE_CONFLICT"


class DatabaseMergeQueueStaleFenceError(DatabaseMergeQueueError):
    """Stale claim, fencing token, fence epoch, or worktree identity."""

    code = "DQP_MERGE_QUEUE_STALE_FENCE"


class DatabaseMergeQueueBoundsError(DatabaseMergeQueueError, ValueError):
    """Payload, attempt, or capacity bound exceeded."""

    code = "DQP_MERGE_QUEUE_BOUNDS"


class DatabaseMergeQueueNotReadyError(DatabaseMergeQueueError):
    """Settlement or claim blocked by missing evidence."""

    code = "DQP_MERGE_QUEUE_NOT_READY"


class DuckDBUnavailableError(DatabaseMergeQueueError):
    """Optional DuckDB dependency is not installed."""

    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class EntryStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    VALIDATING = "validating"
    MERGING = "merging"
    ACCEPTED = "accepted"
    FAILED = "failed"
    CONFLICT = "conflict"
    QUARANTINED = "quarantined"
    SETTLED = "settled"
    CANCELLED = "cancelled"


class ValidationStatus(str, Enum):
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"


class MergeAttemptStatus(str, Enum):
    RUNNING = "running"
    ACCEPTED = "accepted"
    FAILED = "failed"
    CONFLICT = "conflict"
    REBASE_REQUIRED = "rebase_required"
    PARTIAL_PUBLISH = "partial_publish"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"


class MergeOutcome(str, Enum):
    ACCEPTED = "accepted"
    FAILED = "failed"
    CONFLICT = "conflict"
    REBASE_REQUIRED = "rebase_required"
    PARTIAL_PUBLISH = "partial_publish"
    INTERRUPTED = "interrupted"


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
        raise DatabaseMergeQueueError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseMergeQueueError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseMergeQueueBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseMergeQueueBoundsError(f"{name} must be a positive integer")
    return value


def _normalise_priority(value: Any) -> str:
    priority = str(value or "P2").strip().upper()
    return priority if priority in _PRIORITY_ORDER else "P2"


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
        raise DatabaseMergeQueueBoundsError(
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


def _command_digest(argv: Sequence[str] | None) -> str:
    return _sha256_hex(
        _canonical_json({"argv": [str(item) for item in (argv or ())]}).encode("utf-8")
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeQueueEntry:
    """One durable merge-queue entry under database authority."""

    INTERFACE: ClassVar[str] = DATABASE_MERGE_QUEUE_INTERFACE
    SCHEMA: ClassVar[str] = MERGE_QUEUE_ENTRY_SCHEMA

    entry_id: str
    repository_id: str
    target_branch: str
    source_branch: str
    task_cid: str
    worktree_id: str
    priority: str
    status: EntryStatus
    ordinal: int
    enqueued_at_ms: int
    updated_at_ms: int
    revision: int
    commit_sha: str = ""
    claimed_at_ms: int = 0
    consumer_id: str = ""
    claim_token: str = ""
    claim_generation: int = 0
    fencing_token: int = 0
    fence_epoch: int = 0
    attempt_count: int = 0
    failure_count: int = 0
    failure_reason: str = ""
    retry_not_before_ms: int = 0
    current_validation_run_id: str = ""
    current_merge_attempt_id: str = ""
    settlement_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _text(self.entry_id, "entry_id"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "target_branch", _text(self.target_branch, "target_branch")
        )
        object.__setattr__(
            self, "source_branch", _text(self.source_branch, "source_branch")
        )
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "priority", _normalise_priority(self.priority))
        status = self.status
        if not isinstance(status, EntryStatus):
            status = EntryStatus(str(status).strip().lower())
            object.__setattr__(self, "status", status)
        object.__setattr__(self, "ordinal", _nonneg_int(int(self.ordinal), "ordinal"))
        object.__setattr__(
            self, "enqueued_at_ms", _nonneg_int(int(self.enqueued_at_ms), "enqueued_at_ms")
        )
        object.__setattr__(
            self, "updated_at_ms", _nonneg_int(int(self.updated_at_ms), "updated_at_ms")
        )
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self, "commit_sha", _text(self.commit_sha, "commit_sha", required=False)
        )
        object.__setattr__(
            self, "claimed_at_ms", _nonneg_int(int(self.claimed_at_ms), "claimed_at_ms")
        )
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id", required=False)
        )
        object.__setattr__(
            self, "claim_token", _text(self.claim_token, "claim_token", required=False)
        )
        object.__setattr__(
            self,
            "claim_generation",
            _nonneg_int(int(self.claim_generation), "claim_generation"),
        )
        object.__setattr__(
            self,
            "fencing_token",
            _nonneg_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _nonneg_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self, "attempt_count", _nonneg_int(int(self.attempt_count), "attempt_count")
        )
        object.__setattr__(
            self, "failure_count", _nonneg_int(int(self.failure_count), "failure_count")
        )
        object.__setattr__(
            self,
            "failure_reason",
            _text(self.failure_reason, "failure_reason", required=False),
        )
        object.__setattr__(
            self,
            "retry_not_before_ms",
            _nonneg_int(int(self.retry_not_before_ms), "retry_not_before_ms"),
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
            "settlement_id",
            _text(self.settlement_id, "settlement_id", required=False),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def active_claim(self) -> bool:
        return self.status in {
            EntryStatus.CLAIMED,
            EntryStatus.VALIDATING,
            EntryStatus.MERGING,
            EntryStatus.ACCEPTED,
        } and bool(self.claim_token)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "entry_id": self.entry_id,
            "repository_id": self.repository_id,
            "target_branch": self.target_branch,
            "source_branch": self.source_branch,
            "task_cid": self.task_cid,
            "worktree_id": self.worktree_id,
            "commit_sha": self.commit_sha,
            "priority": self.priority,
            "status": self.status.value,
            "ordinal": int(self.ordinal),
            "enqueued_at_ms": int(self.enqueued_at_ms),
            "updated_at_ms": int(self.updated_at_ms),
            "enqueued_at": _utc_iso_from_ms(self.enqueued_at_ms),
            "updated_at": _utc_iso_from_ms(self.updated_at_ms),
            "claimed_at_ms": int(self.claimed_at_ms),
            "consumer_id": self.consumer_id,
            "claim_token": self.claim_token,
            "claim_generation": int(self.claim_generation),
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "attempt_count": int(self.attempt_count),
            "failure_count": int(self.failure_count),
            "failure_reason": self.failure_reason,
            "retry_not_before_ms": int(self.retry_not_before_ms),
            "current_validation_run_id": self.current_validation_run_id,
            "current_merge_attempt_id": self.current_merge_attempt_id,
            "settlement_id": self.settlement_id,
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class ValidationRun:
    """One hermetic validation run bound to a merge claim fence."""

    INTERFACE: ClassVar[str] = VALIDATION_RUN_INTERFACE
    SCHEMA: ClassVar[str] = VALIDATION_RUN_SCHEMA

    run_id: str
    entry_id: str
    task_cid: str
    worktree_id: str
    fencing_token: int
    fence_epoch: int
    claim_token: str
    claim_generation: int
    status: ValidationStatus
    started_at_ms: int
    attempt_id: str = ""
    command_digest: str = ""
    finished_at_ms: int = 0
    evidence_digest: str = ""
    outcome: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
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
        object.__setattr__(self, "claim_token", _text(self.claim_token, "claim_token"))
        object.__setattr__(
            self,
            "claim_generation",
            _positive_int(int(self.claim_generation), "claim_generation"),
        )
        status = self.status
        if not isinstance(status, ValidationStatus):
            status = ValidationStatus(str(status).strip().lower())
            object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "started_at_ms", _nonneg_int(int(self.started_at_ms), "started_at_ms")
        )
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", required=False)
        )
        object.__setattr__(
            self,
            "command_digest",
            _text(self.command_digest, "command_digest", required=False),
        )
        object.__setattr__(
            self,
            "finished_at_ms",
            _nonneg_int(int(self.finished_at_ms), "finished_at_ms"),
        )
        object.__setattr__(
            self,
            "evidence_digest",
            _text(self.evidence_digest, "evidence_digest", required=False),
        )
        object.__setattr__(
            self, "outcome", _text(self.outcome, "outcome", required=False)
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def passed(self) -> bool:
        return self.status is ValidationStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "run_id": self.run_id,
            "entry_id": self.entry_id,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "worktree_id": self.worktree_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "claim_token": self.claim_token,
            "claim_generation": int(self.claim_generation),
            "command_digest": self.command_digest,
            "status": self.status.value,
            "started_at_ms": int(self.started_at_ms),
            "finished_at_ms": int(self.finished_at_ms),
            "started_at": _utc_iso_from_ms(self.started_at_ms),
            "finished_at": (
                _utc_iso_from_ms(self.finished_at_ms) if self.finished_at_ms else ""
            ),
            "evidence_digest": self.evidence_digest,
            "outcome": self.outcome,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class MergeAttempt:
    """One merge attempt bound to a claimed entry fence."""

    INTERFACE: ClassVar[str] = MERGE_ATTEMPT_INTERFACE
    SCHEMA: ClassVar[str] = MERGE_ATTEMPT_SCHEMA

    merge_attempt_id: str
    entry_id: str
    task_cid: str
    worktree_id: str
    fencing_token: int
    fence_epoch: int
    claim_token: str
    claim_generation: int
    status: MergeAttemptStatus
    started_at_ms: int
    outcome: str = ""
    result_commit_id: str = ""
    finished_at_ms: int = 0
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "merge_attempt_id", _text(self.merge_attempt_id, "merge_attempt_id")
        )
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
        object.__setattr__(self, "claim_token", _text(self.claim_token, "claim_token"))
        object.__setattr__(
            self,
            "claim_generation",
            _positive_int(int(self.claim_generation), "claim_generation"),
        )
        status = self.status
        if not isinstance(status, MergeAttemptStatus):
            status = MergeAttemptStatus(str(status).strip().lower())
            object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "started_at_ms", _nonneg_int(int(self.started_at_ms), "started_at_ms")
        )
        object.__setattr__(
            self, "outcome", _text(self.outcome, "outcome", required=False)
        )
        object.__setattr__(
            self,
            "result_commit_id",
            _text(self.result_commit_id, "result_commit_id", required=False),
        )
        object.__setattr__(
            self,
            "finished_at_ms",
            _nonneg_int(int(self.finished_at_ms), "finished_at_ms"),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def accepted(self) -> bool:
        return self.status is MergeAttemptStatus.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "merge_attempt_id": self.merge_attempt_id,
            "entry_id": self.entry_id,
            "task_cid": self.task_cid,
            "worktree_id": self.worktree_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "claim_token": self.claim_token,
            "claim_generation": int(self.claim_generation),
            "status": self.status.value,
            "outcome": self.outcome,
            "result_commit_id": self.result_commit_id,
            "started_at_ms": int(self.started_at_ms),
            "finished_at_ms": int(self.finished_at_ms),
            "started_at": _utc_iso_from_ms(self.started_at_ms),
            "finished_at": (
                _utc_iso_from_ms(self.finished_at_ms) if self.finished_at_ms else ""
            ),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class SettlementReceipt:
    """Atomic settlement of accepted merge + current validation evidence."""

    INTERFACE: ClassVar[str] = SETTLEMENT_RECEIPT_INTERFACE
    SCHEMA: ClassVar[str] = SETTLEMENT_RECEIPT_SCHEMA

    settlement_id: str
    entry_id: str
    task_cid: str
    merge_attempt_id: str
    validation_run_id: str
    worktree_id: str
    fencing_token: int
    fence_epoch: int
    result_commit_id: str
    evidence_digest: str
    settled_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "settlement_id", _text(self.settlement_id, "settlement_id")
        )
        object.__setattr__(self, "entry_id", _text(self.entry_id, "entry_id"))
        object.__setattr__(self, "task_cid", _text(self.task_cid, "task_cid"))
        object.__setattr__(
            self, "merge_attempt_id", _text(self.merge_attempt_id, "merge_attempt_id")
        )
        object.__setattr__(
            self, "validation_run_id", _text(self.validation_run_id, "validation_run_id")
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
            self, "result_commit_id", _text(self.result_commit_id, "result_commit_id")
        )
        object.__setattr__(
            self, "evidence_digest", _text(self.evidence_digest, "evidence_digest")
        )
        object.__setattr__(
            self, "settled_at_ms", _nonneg_int(int(self.settled_at_ms), "settled_at_ms")
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
            "settlement_id": self.settlement_id,
            "entry_id": self.entry_id,
            "task_cid": self.task_cid,
            "merge_attempt_id": self.merge_attempt_id,
            "validation_run_id": self.validation_run_id,
            "worktree_id": self.worktree_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "result_commit_id": self.result_commit_id,
            "evidence_digest": self.evidence_digest,
            "settled_at_ms": int(self.settled_at_ms),
            "settled_at": _utc_iso_from_ms(self.settled_at_ms),
            "body": dict(self.body),
            "authority": "database",
            "json_receipt_authority": "none",
        }


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class DatabaseMergeQueue:
    """DuckDB-backed validation, merge, and settlement authority.

    Interface: ``DatabaseMergeQueue@1`` with projected records
    ``ValidationRun@1``, ``MergeAttempt@1``, and ``SettlementReceipt@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_MERGE_QUEUE_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_MERGE_QUEUE_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_processing: int = DEFAULT_MAX_PROCESSING,
        priority_aging_ms: int = 300_000,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseMergeQueue; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._max_attempts = _positive_int(int(max_attempts), "max_attempts")
        self._max_processing = _positive_int(int(max_processing), "max_processing")
        self._priority_aging_ms = _nonneg_int(
            int(priority_aging_ms), "priority_aging_ms"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True
        self._ordinal = 0

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

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
                ):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO merge_queue_metadata(key, value)
                        VALUES (?, ?)
                        """,
                        [key, value],
                    )
                row = connection.execute(
                    "SELECT COALESCE(MAX(ordinal), 0) AS max_ordinal FROM merge_queue_entries"
                ).fetchone()
                mapping = _row_mapping(row)
                self._ordinal = int(_row_get(mapping, "max_ordinal", "0", default=0) or 0)
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

    def authority_policy(self) -> dict[str, str]:
        """Return the fail-closed authority split for settlement."""

        return {
            "semantic_authority": "database",
            "settlement_authority": "database",
            "json_receipt_authority": "none",
            "byte_authority": "git",
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
        }

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseMergeQueueNotOpenError("DatabaseMergeQueue is not open")
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

    def _next_ordinal(self) -> int:
        self._ordinal += 1
        return self._ordinal

    def _record_event(
        self,
        connection: Any,
        *,
        entry_id: str,
        event_type: str,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> None:
        now = self._now_ms() if now_ms is None else int(now_ms)
        connection.execute(
            """
            INSERT INTO merge_queue_events(
                event_id, entry_id, event_type, observed_at_ms, body_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                _new_id("event"),
                entry_id,
                event_type,
                now,
                _canonical_json(_bounded_mapping(body, name="event body")),
            ],
        )

    # -- enqueue / claim -----------------------------------------------------

    def enqueue(
        self,
        *,
        repository_id: str,
        target_branch: str,
        source_branch: str,
        task_cid: str,
        worktree_id: str,
        commit_sha: str = "",
        priority: str = "P2",
        fencing_token: int = 1,
        fence_epoch: int = 1,
        body: Mapping[str, Any] | None = None,
    ) -> MergeQueueEntry:
        """Enqueue or return the existing task/commit binding for one target."""

        repo = _text(repository_id, "repository_id")
        target = _text(target_branch, "target_branch")
        source = _text(source_branch, "source_branch")
        task = _text(task_cid, "task_cid")
        worktree = _text(worktree_id, "worktree_id")
        commit = _text(commit_sha, "commit_sha", required=False)
        prio = _normalise_priority(priority)
        fence = _positive_int(int(fencing_token), "fencing_token")
        epoch = _positive_int(int(fence_epoch), "fence_epoch")
        payload = _bounded_mapping(body, name="body")
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    """
                    SELECT * FROM merge_queue_entries
                    WHERE repository_id = ? AND target_branch = ?
                      AND task_cid = ? AND commit_sha = ?
                    """,
                    [repo, target, task, commit],
                ).fetchone()
                if existing is not None:
                    self._commit_if_idle(connection)
                    return self._entry_from_row(existing)

                entry_id = _new_id("entry")
                ordinal = self._next_ordinal()
                connection.execute(
                    """
                    INSERT INTO merge_queue_entries(
                        entry_id, repository_id, target_branch, source_branch,
                        task_cid, worktree_id, commit_sha, priority, status,
                        ordinal, enqueued_at_ms, updated_at_ms, fencing_token,
                        fence_epoch, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    [
                        entry_id,
                        repo,
                        target,
                        source,
                        task,
                        worktree,
                        commit,
                        prio,
                        EntryStatus.PENDING.value,
                        ordinal,
                        now,
                        now,
                        fence,
                        epoch,
                        _canonical_json(payload),
                    ],
                )
                self._record_event(
                    connection,
                    entry_id=entry_id,
                    event_type="enqueued",
                    body={"task_cid": task, "priority": prio},
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
                    [entry_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._entry_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def claim_next(
        self,
        *,
        repository_id: str,
        target_branch: str,
        consumer_id: str,
        limit: int = 1,
    ) -> tuple[MergeQueueEntry, ...]:
        """Atomically claim a fair, capacity-bounded batch for one target."""

        repo = _text(repository_id, "repository_id")
        target = _text(target_branch, "target_branch")
        consumer = _text(consumer_id, "consumer_id")
        requested = _positive_int(int(limit), "limit")
        now = self._now_ms()
        claimed: list[MergeQueueEntry] = []

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                active_rows = connection.execute(
                    """
                    SELECT entry_id FROM merge_queue_entries
                    WHERE repository_id = ? AND target_branch = ?
                      AND status IN (
                          'claimed', 'validating', 'merging', 'accepted'
                      )
                    """,
                    [repo, target],
                ).fetchall()
                capacity = max(0, self._max_processing - len(active_rows))
                claim_count = min(requested, capacity)
                if claim_count <= 0:
                    self._commit_if_idle(connection)
                    return ()

                pending_rows = connection.execute(
                    """
                    SELECT * FROM merge_queue_entries
                    WHERE repository_id = ? AND target_branch = ?
                      AND status = 'pending' AND retry_not_before_ms <= ?
                    """,
                    [repo, target, now],
                ).fetchall()
                selected = sorted(
                    pending_rows, key=lambda row: self._fairness_key(row, now)
                )[:claim_count]
                for row in selected:
                    mapping = _row_mapping(row)
                    entry_id = str(_row_get(mapping, "entry_id", default=""))
                    claim_token = uuid.uuid4().hex
                    connection.execute(
                        """
                        UPDATE merge_queue_entries
                        SET status = 'claimed',
                            claimed_at_ms = ?,
                            consumer_id = ?,
                            claim_token = ?,
                            claim_generation = claim_generation + 1,
                            retry_not_before_ms = 0,
                            updated_at_ms = ?,
                            revision = revision + 1
                        WHERE entry_id = ? AND status = 'pending'
                        """,
                        [now, consumer, claim_token, now, entry_id],
                    )
                    claimed_row = connection.execute(
                        "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
                        [entry_id],
                    ).fetchone()
                    if claimed_row is None:
                        continue
                    entry = self._entry_from_row(claimed_row)
                    if (
                        entry.status is not EntryStatus.CLAIMED
                        or entry.claim_token != claim_token
                        or entry.consumer_id != consumer
                    ):
                        # Lost the race or update did not apply.
                        continue
                    self._record_event(
                        connection,
                        entry_id=entry.entry_id,
                        event_type="claimed",
                        body={
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

    def _fairness_key(self, row: Any, now_ms: int) -> tuple[int, int, int, str]:
        mapping = _row_mapping(row)
        base = _PRIORITY_ORDER.get(
            str(_row_get(mapping, "priority", default="P2")), _PRIORITY_ORDER["P2"]
        )
        enqueued = int(_row_get(mapping, "enqueued_at_ms", default=0) or 0)
        if self._priority_aging_ms > 0:
            promotions = int(max(0, now_ms - enqueued) / self._priority_aging_ms)
            effective = max(0, base - promotions)
        else:
            effective = base
        ordinal = int(_row_get(mapping, "ordinal", default=0) or 0)
        entry_id = str(_row_get(mapping, "entry_id", default=""))
        return effective, enqueued, ordinal, entry_id

    # -- claim fence helpers -------------------------------------------------

    def _load_entry(self, connection: Any, entry_id: str) -> MergeQueueEntry:
        row = connection.execute(
            "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
            [entry_id],
        ).fetchone()
        if row is None:
            raise DatabaseMergeQueueConflictError(f"unknown entry {entry_id}")
        return self._entry_from_row(row)

    def _require_claim(
        self,
        entry: MergeQueueEntry,
        *,
        claim_token: str,
        claim_generation: int,
        worktree_id: str,
        fencing_token: int,
        fence_epoch: int,
        operation: str,
        allow_statuses: set[EntryStatus] | None = None,
    ) -> None:
        allowed = allow_statuses or {
            EntryStatus.CLAIMED,
            EntryStatus.VALIDATING,
            EntryStatus.MERGING,
            EntryStatus.ACCEPTED,
        }
        if entry.status not in allowed:
            raise DatabaseMergeQueueStaleFenceError(
                f"{operation} rejected: entry status is {entry.status.value}"
            )
        if (
            not claim_token
            or entry.claim_token != claim_token
            or entry.claim_generation != claim_generation
        ):
            raise DatabaseMergeQueueStaleFenceError(
                f"{operation} rejected: claim token or generation is stale"
            )
        if entry.worktree_id != worktree_id:
            raise DatabaseMergeQueueStaleFenceError(
                f"{operation} rejected: worktree identity is stale"
            )
        if (
            entry.fencing_token != fencing_token
            or entry.fence_epoch != fence_epoch
        ):
            raise DatabaseMergeQueueStaleFenceError(
                f"{operation} rejected: fencing token or epoch is stale"
            )

    def owns_claim(
        self,
        entry: MergeQueueEntry,
        *,
        claim_token: str = "",
        claim_generation: int | None = None,
        worktree_id: str = "",
        fencing_token: int | None = None,
        fence_epoch: int | None = None,
    ) -> bool:
        """Return whether the supplied coordinates still own the entry claim."""

        token = claim_token or entry.claim_token
        generation = (
            entry.claim_generation
            if claim_generation is None
            else int(claim_generation)
        )
        worktree = worktree_id or entry.worktree_id
        fence = entry.fencing_token if fencing_token is None else int(fencing_token)
        epoch = entry.fence_epoch if fence_epoch is None else int(fence_epoch)
        with self._lock:
            connection = self._require()
            current = self._load_entry(connection, entry.entry_id)
        try:
            self._require_claim(
                current,
                claim_token=token,
                claim_generation=generation,
                worktree_id=worktree,
                fencing_token=fence,
                fence_epoch=epoch,
                operation="owns_claim",
            )
            return True
        except DatabaseMergeQueueStaleFenceError:
            return False

    # -- validation ----------------------------------------------------------

    def start_validation(
        self,
        entry: MergeQueueEntry,
        *,
        argv: Sequence[str] | None = None,
        attempt_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> ValidationRun:
        """Start a validation run under the current claim fence."""

        now = self._now_ms()
        payload = _bounded_mapping(body, name="validation body")
        argv_list = [str(item) for item in (argv or ())]
        digest = _command_digest(argv_list)

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_entry(connection, entry.entry_id)
                self._require_claim(
                    current,
                    claim_token=entry.claim_token,
                    claim_generation=entry.claim_generation,
                    worktree_id=entry.worktree_id,
                    fencing_token=entry.fencing_token,
                    fence_epoch=entry.fence_epoch,
                    operation="start_validation",
                    allow_statuses={EntryStatus.CLAIMED, EntryStatus.VALIDATING},
                )
                run_id = _new_id("validation")
                connection.execute(
                    """
                    INSERT INTO validation_runs(
                        run_id, entry_id, task_cid, attempt_id, worktree_id,
                        fencing_token, fence_epoch, claim_token, claim_generation,
                        command_digest, status, started_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        run_id,
                        current.entry_id,
                        current.task_cid,
                        _text(attempt_id, "attempt_id", required=False),
                        current.worktree_id,
                        current.fencing_token,
                        current.fence_epoch,
                        current.claim_token,
                        current.claim_generation,
                        digest,
                        ValidationStatus.RUNNING.value,
                        now,
                        _canonical_json({"argv": argv_list, **payload}),
                    ],
                )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = 'validating',
                        current_validation_run_id = ?,
                        updated_at_ms = ?,
                        revision = revision + 1
                    WHERE entry_id = ?
                    """,
                    [run_id, now, current.entry_id],
                )
                self._record_event(
                    connection,
                    entry_id=current.entry_id,
                    event_type="validation_started",
                    body={"run_id": run_id, "command_digest": digest},
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM validation_runs WHERE run_id = ?",
                    [run_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._validation_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def finish_validation(
        self,
        run: ValidationRun,
        *,
        outcome: str,
        evidence_digest: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> ValidationRun:
        """Finish a validation run; only matching claim fences may mutate it."""

        outcome_text = str(outcome or "").strip().lower()
        if outcome_text not in {"passed", "failed", "error", "interrupted"}:
            raise DatabaseMergeQueueBoundsError(
                f"validation outcome {outcome!r} is not in the closed set"
            )
        status = ValidationStatus(outcome_text)
        digest = _text(evidence_digest, "evidence_digest", required=False)
        if status is ValidationStatus.PASSED and not digest:
            raise DatabaseMergeQueueBoundsError(
                "passed validation requires a non-empty evidence_digest"
            )
        payload = _bounded_mapping(body, name="validation result body")
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, run.entry_id)
                self._require_claim(
                    entry,
                    claim_token=run.claim_token,
                    claim_generation=run.claim_generation,
                    worktree_id=run.worktree_id,
                    fencing_token=run.fencing_token,
                    fence_epoch=run.fence_epoch,
                    operation="finish_validation",
                    allow_statuses={EntryStatus.VALIDATING, EntryStatus.CLAIMED},
                )
                existing = connection.execute(
                    "SELECT * FROM validation_runs WHERE run_id = ?",
                    [run.run_id],
                ).fetchone()
                if existing is None:
                    raise DatabaseMergeQueueConflictError(
                        f"unknown validation run {run.run_id}"
                    )
                existing_run = self._validation_from_row(existing)
                if existing_run.status is not ValidationStatus.RUNNING:
                    if (
                        existing_run.status is status
                        and existing_run.evidence_digest == digest
                    ):
                        self._commit_if_idle(connection)
                        return existing_run
                    raise DatabaseMergeQueueConflictError(
                        f"validation run {run.run_id} is already terminal"
                    )
                if (
                    existing_run.claim_token != run.claim_token
                    or existing_run.claim_generation != run.claim_generation
                    or existing_run.worktree_id != run.worktree_id
                    or existing_run.fencing_token != run.fencing_token
                    or existing_run.fence_epoch != run.fence_epoch
                ):
                    raise DatabaseMergeQueueStaleFenceError(
                        "finish_validation rejected: run fence is stale"
                    )

                result_id = _new_id("validation-result")
                connection.execute(
                    """
                    UPDATE validation_runs
                    SET status = ?, finished_at_ms = ?, evidence_digest = ?,
                        outcome = ?, body_json = ?
                    WHERE run_id = ? AND status = 'running'
                    """,
                    [
                        status.value,
                        now,
                        digest,
                        outcome_text,
                        _canonical_json({**dict(existing_run.body), **payload}),
                        run.run_id,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO validation_results(
                        result_id, run_id, entry_id, task_cid, ordinal, outcome,
                        evidence_digest, recorded_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    [
                        result_id,
                        run.run_id,
                        entry.entry_id,
                        entry.task_cid,
                        outcome_text,
                        digest,
                        now,
                        _canonical_json(payload),
                    ],
                )
                if status is ValidationStatus.PASSED:
                    connection.execute(
                        """
                        UPDATE merge_queue_entries
                        SET status = 'claimed',
                            current_validation_run_id = ?,
                            updated_at_ms = ?,
                            revision = revision + 1
                        WHERE entry_id = ?
                        """,
                        [run.run_id, now, entry.entry_id],
                    )
                else:
                    connection.execute(
                        """
                        UPDATE merge_queue_entries
                        SET status = 'failed',
                            failure_count = failure_count + 1,
                            failure_reason = ?,
                            updated_at_ms = ?,
                            revision = revision + 1
                        WHERE entry_id = ?
                        """,
                        [f"validation_{outcome_text}", now, entry.entry_id],
                    )
                self._record_event(
                    connection,
                    entry_id=entry.entry_id,
                    event_type="validation_finished",
                    body={
                        "run_id": run.run_id,
                        "outcome": outcome_text,
                        "evidence_digest": digest,
                    },
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM validation_runs WHERE run_id = ?",
                    [run.run_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._validation_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- merge attempts ------------------------------------------------------

    def start_merge_attempt(
        self,
        entry: MergeQueueEntry,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        """Start a merge attempt under the current claim fence."""

        now = self._now_ms()
        payload = _bounded_mapping(body, name="merge attempt body")

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_entry(connection, entry.entry_id)
                # VALIDATING is claim-active: reject with NotReady (not StaleFence)
                # when evidence is still running or missing.
                self._require_claim(
                    current,
                    claim_token=entry.claim_token,
                    claim_generation=entry.claim_generation,
                    worktree_id=entry.worktree_id,
                    fencing_token=entry.fencing_token,
                    fence_epoch=entry.fence_epoch,
                    operation="start_merge_attempt",
                    allow_statuses={
                        EntryStatus.CLAIMED,
                        EntryStatus.VALIDATING,
                        EntryStatus.MERGING,
                    },
                )
                # Require a current passed validation before merge starts.
                if not current.current_validation_run_id:
                    raise DatabaseMergeQueueNotReadyError(
                        "merge attempt requires a current passed validation run"
                    )
                validation = self.get_validation_run(current.current_validation_run_id)
                if validation is None or not validation.passed:
                    raise DatabaseMergeQueueNotReadyError(
                        "merge attempt requires a passed validation run"
                    )
                if (
                    validation.worktree_id != current.worktree_id
                    or validation.fencing_token != current.fencing_token
                    or validation.fence_epoch != current.fence_epoch
                    or validation.claim_token != current.claim_token
                    or validation.claim_generation != current.claim_generation
                ):
                    raise DatabaseMergeQueueStaleFenceError(
                        "merge attempt rejected: validation fence is stale"
                    )

                attempt_id = _new_id("merge-attempt")
                connection.execute(
                    """
                    INSERT INTO merge_attempts(
                        merge_attempt_id, entry_id, task_cid, worktree_id,
                        fencing_token, fence_epoch, claim_token, claim_generation,
                        status, started_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        attempt_id,
                        current.entry_id,
                        current.task_cid,
                        current.worktree_id,
                        current.fencing_token,
                        current.fence_epoch,
                        current.claim_token,
                        current.claim_generation,
                        MergeAttemptStatus.RUNNING.value,
                        now,
                        _canonical_json(payload),
                    ],
                )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = 'merging',
                        current_merge_attempt_id = ?,
                        attempt_count = attempt_count + 1,
                        updated_at_ms = ?,
                        revision = revision + 1
                    WHERE entry_id = ?
                    """,
                    [attempt_id, now, current.entry_id],
                )
                self._record_event(
                    connection,
                    entry_id=current.entry_id,
                    event_type="merge_started",
                    body={"merge_attempt_id": attempt_id},
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM merge_attempts WHERE merge_attempt_id = ?",
                    [attempt_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._attempt_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def finish_merge_attempt(
        self,
        attempt: MergeAttempt,
        *,
        outcome: str | MergeOutcome,
        result_commit_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeAttempt:
        """Finish a merge attempt with a closed-set outcome."""

        if isinstance(outcome, MergeOutcome):
            outcome_value = outcome
        else:
            outcome_value = MergeOutcome(str(outcome).strip().lower())
        status = MergeAttemptStatus(outcome_value.value)
        commit = _text(result_commit_id, "result_commit_id", required=False)
        if status is MergeAttemptStatus.ACCEPTED and not commit:
            raise DatabaseMergeQueueBoundsError(
                "accepted merge requires a non-empty result_commit_id"
            )
        payload = _bounded_mapping(body, name="merge finish body")
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                entry = self._load_entry(connection, attempt.entry_id)
                self._require_claim(
                    entry,
                    claim_token=attempt.claim_token,
                    claim_generation=attempt.claim_generation,
                    worktree_id=attempt.worktree_id,
                    fencing_token=attempt.fencing_token,
                    fence_epoch=attempt.fence_epoch,
                    operation="finish_merge_attempt",
                    allow_statuses={EntryStatus.MERGING, EntryStatus.CLAIMED},
                )
                existing = connection.execute(
                    "SELECT * FROM merge_attempts WHERE merge_attempt_id = ?",
                    [attempt.merge_attempt_id],
                ).fetchone()
                if existing is None:
                    raise DatabaseMergeQueueConflictError(
                        f"unknown merge attempt {attempt.merge_attempt_id}"
                    )
                existing_attempt = self._attempt_from_row(existing)
                if existing_attempt.status is not MergeAttemptStatus.RUNNING:
                    if (
                        existing_attempt.status is status
                        and existing_attempt.result_commit_id == commit
                    ):
                        self._commit_if_idle(connection)
                        return existing_attempt
                    raise DatabaseMergeQueueConflictError(
                        f"merge attempt {attempt.merge_attempt_id} is already terminal"
                    )
                if (
                    existing_attempt.claim_token != attempt.claim_token
                    or existing_attempt.claim_generation != attempt.claim_generation
                    or existing_attempt.worktree_id != attempt.worktree_id
                    or existing_attempt.fencing_token != attempt.fencing_token
                    or existing_attempt.fence_epoch != attempt.fence_epoch
                ):
                    raise DatabaseMergeQueueStaleFenceError(
                        "finish_merge_attempt rejected: attempt fence is stale"
                    )

                connection.execute(
                    """
                    UPDATE merge_attempts
                    SET status = ?, outcome = ?, result_commit_id = ?,
                        finished_at_ms = ?, body_json = ?
                    WHERE merge_attempt_id = ? AND status = 'running'
                    """,
                    [
                        status.value,
                        outcome_value.value,
                        commit,
                        now,
                        _canonical_json({**dict(existing_attempt.body), **payload}),
                        attempt.merge_attempt_id,
                    ],
                )

                if status is MergeAttemptStatus.ACCEPTED:
                    entry_status = EntryStatus.ACCEPTED.value
                    failure_reason = ""
                elif status is MergeAttemptStatus.CONFLICT:
                    entry_status = EntryStatus.CONFLICT.value
                    failure_reason = "merge_conflict"
                elif status is MergeAttemptStatus.REBASE_REQUIRED:
                    entry_status = EntryStatus.FAILED.value
                    failure_reason = "rebase_required"
                elif status is MergeAttemptStatus.PARTIAL_PUBLISH:
                    entry_status = EntryStatus.FAILED.value
                    failure_reason = "partial_publish"
                else:
                    entry_status = EntryStatus.FAILED.value
                    failure_reason = outcome_value.value

                if status is MergeAttemptStatus.ACCEPTED:
                    connection.execute(
                        """
                        UPDATE merge_queue_entries
                        SET status = ?,
                            current_merge_attempt_id = ?,
                            failure_reason = '',
                            updated_at_ms = ?,
                            revision = revision + 1
                        WHERE entry_id = ?
                        """,
                        [
                            entry_status,
                            attempt.merge_attempt_id,
                            now,
                            entry.entry_id,
                        ],
                    )
                else:
                    connection.execute(
                        """
                        UPDATE merge_queue_entries
                        SET status = ?,
                            failure_count = failure_count + 1,
                            failure_reason = ?,
                            updated_at_ms = ?,
                            revision = revision + 1
                        WHERE entry_id = ?
                        """,
                        [entry_status, failure_reason, now, entry.entry_id],
                    )
                self._record_event(
                    connection,
                    entry_id=entry.entry_id,
                    event_type="merge_finished",
                    body={
                        "merge_attempt_id": attempt.merge_attempt_id,
                        "outcome": outcome_value.value,
                        "result_commit_id": commit,
                    },
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM merge_attempts WHERE merge_attempt_id = ?",
                    [attempt.merge_attempt_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._attempt_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- settlement / retry / quarantine ------------------------------------

    def settle(
        self,
        entry: MergeQueueEntry,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> SettlementReceipt:
        """Atomically settle accepted merge + current validation evidence.

        JSON receipt paths are ignored. Settlement authority is exclusively the
        database rows for the current claim fence, accepted merge attempt, and
        passed validation run.
        """

        payload = _bounded_mapping(body, name="settlement body")
        # Explicitly refuse external JSON receipt authority.
        if "json_receipt_path" in payload or "queue_file" in payload:
            raise DatabaseMergeQueueError(
                "JSON receipt or queue file alone cannot settle work"
            )
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_entry(connection, entry.entry_id)
                if current.status is EntryStatus.SETTLED and current.settlement_id:
                    existing = self.get_settlement(current.settlement_id)
                    if existing is not None:
                        self._commit_if_idle(connection)
                        return existing

                # Claim-active intermediate states keep the fence but are not
                # ready to settle until accepted merge + passed validation exist.
                self._require_claim(
                    current,
                    claim_token=entry.claim_token,
                    claim_generation=entry.claim_generation,
                    worktree_id=entry.worktree_id,
                    fencing_token=entry.fencing_token,
                    fence_epoch=entry.fence_epoch,
                    operation="settle",
                    allow_statuses={
                        EntryStatus.CLAIMED,
                        EntryStatus.VALIDATING,
                        EntryStatus.MERGING,
                        EntryStatus.ACCEPTED,
                    },
                )
                if not current.current_merge_attempt_id:
                    raise DatabaseMergeQueueNotReadyError(
                        "settlement requires an accepted merge attempt"
                    )
                if not current.current_validation_run_id:
                    raise DatabaseMergeQueueNotReadyError(
                        "settlement requires a current passed validation run"
                    )

                attempt_row = connection.execute(
                    "SELECT * FROM merge_attempts WHERE merge_attempt_id = ?",
                    [current.current_merge_attempt_id],
                ).fetchone()
                if attempt_row is None:
                    raise DatabaseMergeQueueNotReadyError(
                        "settlement requires a durable merge attempt row"
                    )
                attempt = self._attempt_from_row(attempt_row)
                if not attempt.accepted:
                    raise DatabaseMergeQueueNotReadyError(
                        "settlement requires an accepted merge outcome"
                    )
                if (
                    attempt.worktree_id != current.worktree_id
                    or attempt.fencing_token != current.fencing_token
                    or attempt.fence_epoch != current.fence_epoch
                    or attempt.claim_token != current.claim_token
                    or attempt.claim_generation != current.claim_generation
                ):
                    raise DatabaseMergeQueueStaleFenceError(
                        "settlement rejected: merge attempt fence is stale"
                    )

                validation_row = connection.execute(
                    "SELECT * FROM validation_runs WHERE run_id = ?",
                    [current.current_validation_run_id],
                ).fetchone()
                if validation_row is None:
                    raise DatabaseMergeQueueNotReadyError(
                        "settlement requires a durable validation run row"
                    )
                validation = self._validation_from_row(validation_row)
                if not validation.passed or not validation.evidence_digest:
                    raise DatabaseMergeQueueNotReadyError(
                        "settlement requires current passed validation evidence"
                    )
                if (
                    validation.worktree_id != current.worktree_id
                    or validation.fencing_token != current.fencing_token
                    or validation.fence_epoch != current.fence_epoch
                    or validation.claim_token != current.claim_token
                    or validation.claim_generation != current.claim_generation
                ):
                    raise DatabaseMergeQueueStaleFenceError(
                        "settlement rejected: validation fence is stale"
                    )

                settlement_id = _new_id("settlement")
                connection.execute(
                    """
                    INSERT INTO settlement_receipts(
                        settlement_id, entry_id, task_cid, merge_attempt_id,
                        validation_run_id, worktree_id, fencing_token, fence_epoch,
                        result_commit_id, evidence_digest, settled_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        settlement_id,
                        current.entry_id,
                        current.task_cid,
                        attempt.merge_attempt_id,
                        validation.run_id,
                        current.worktree_id,
                        current.fencing_token,
                        current.fence_epoch,
                        attempt.result_commit_id,
                        validation.evidence_digest,
                        now,
                        _canonical_json(
                            {
                                "authority": "database",
                                "json_receipt_authority": "none",
                                **payload,
                            }
                        ),
                    ],
                )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = 'settled',
                        settlement_id = ?,
                        claim_token = '',
                        claim_generation = claim_generation + 1,
                        consumer_id = '',
                        claimed_at_ms = 0,
                        updated_at_ms = ?,
                        revision = revision + 1
                    WHERE entry_id = ?
                      AND claim_token = ?
                      AND claim_generation = ?
                    """,
                    [
                        settlement_id,
                        now,
                        current.entry_id,
                        current.claim_token,
                        current.claim_generation,
                    ],
                )
                self._record_event(
                    connection,
                    entry_id=current.entry_id,
                    event_type="settled",
                    body={
                        "settlement_id": settlement_id,
                        "merge_attempt_id": attempt.merge_attempt_id,
                        "validation_run_id": validation.run_id,
                    },
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM settlement_receipts WHERE settlement_id = ?",
                    [settlement_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._settlement_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def requeue(
        self,
        entry: MergeQueueEntry,
        *,
        reason: str = "",
        delay_ms: int = 0,
        body: Mapping[str, Any] | None = None,
    ) -> MergeQueueEntry:
        """Release a claim for retry, or quarantine after attempt exhaustion."""

        now = self._now_ms()
        delay = _nonneg_int(int(delay_ms), "delay_ms")
        payload = _bounded_mapping(body, name="requeue body")
        reason_text = _text(reason, "reason", required=False)

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_entry(connection, entry.entry_id)
                if current.status in {
                    EntryStatus.SETTLED,
                    EntryStatus.QUARANTINED,
                    EntryStatus.CANCELLED,
                }:
                    self._commit_if_idle(connection)
                    return current
                self._require_claim(
                    current,
                    claim_token=entry.claim_token,
                    claim_generation=entry.claim_generation,
                    worktree_id=entry.worktree_id,
                    fencing_token=entry.fencing_token,
                    fence_epoch=entry.fence_epoch,
                    operation="requeue",
                    allow_statuses={
                        EntryStatus.CLAIMED,
                        EntryStatus.VALIDATING,
                        EntryStatus.MERGING,
                        EntryStatus.FAILED,
                        EntryStatus.CONFLICT,
                        EntryStatus.ACCEPTED,
                    },
                )
                next_failure = current.failure_count + (
                    0
                    if current.status
                    in {EntryStatus.FAILED, EntryStatus.CONFLICT}
                    else 1
                )
                exhausted = (
                    current.attempt_count >= self._max_attempts
                    or next_failure >= self._max_attempts
                )
                if exhausted:
                    status = EntryStatus.QUARANTINED.value
                    retry_not_before = 0
                    event_type = "quarantined"
                else:
                    status = EntryStatus.PENDING.value
                    retry_not_before = now + delay
                    event_type = "requeued"
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?,
                        failure_count = ?,
                        failure_reason = ?,
                        claim_token = '',
                        claim_generation = claim_generation + 1,
                        consumer_id = '',
                        claimed_at_ms = 0,
                        retry_not_before_ms = ?,
                        updated_at_ms = ?,
                        revision = revision + 1,
                        body_json = ?
                    WHERE entry_id = ?
                    """,
                    [
                        status,
                        next_failure if not exhausted else max(next_failure, current.failure_count),
                        reason_text or current.failure_reason,
                        retry_not_before,
                        now,
                        _canonical_json({**dict(current.body), **payload}),
                        current.entry_id,
                    ],
                )
                self._record_event(
                    connection,
                    entry_id=current.entry_id,
                    event_type=event_type,
                    body={"reason": reason_text, "exhausted": exhausted},
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
                    [current.entry_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._entry_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def quarantine(
        self,
        entry: MergeQueueEntry,
        *,
        reason: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> MergeQueueEntry:
        """Force an entry into quarantine under the current claim fence."""

        now = self._now_ms()
        reason_text = _text(reason, "reason", required=False)
        payload = _bounded_mapping(body, name="quarantine body")

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_entry(connection, entry.entry_id)
                if current.status is EntryStatus.QUARANTINED:
                    self._commit_if_idle(connection)
                    return current
                if current.claim_token:
                    self._require_claim(
                        current,
                        claim_token=entry.claim_token,
                        claim_generation=entry.claim_generation,
                        worktree_id=entry.worktree_id,
                        fencing_token=entry.fencing_token,
                        fence_epoch=entry.fence_epoch,
                        operation="quarantine",
                        allow_statuses={
                            EntryStatus.CLAIMED,
                            EntryStatus.VALIDATING,
                            EntryStatus.MERGING,
                            EntryStatus.FAILED,
                            EntryStatus.CONFLICT,
                            EntryStatus.ACCEPTED,
                            EntryStatus.PENDING,
                        },
                    )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = 'quarantined',
                        failure_reason = ?,
                        claim_token = '',
                        claim_generation = claim_generation + 1,
                        consumer_id = '',
                        claimed_at_ms = 0,
                        updated_at_ms = ?,
                        revision = revision + 1,
                        body_json = ?
                    WHERE entry_id = ?
                    """,
                    [
                        reason_text or current.failure_reason or "quarantined",
                        now,
                        _canonical_json({**dict(current.body), **payload}),
                        current.entry_id,
                    ],
                )
                self._record_event(
                    connection,
                    entry_id=current.entry_id,
                    event_type="quarantined",
                    body={"reason": reason_text},
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
                    [current.entry_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._entry_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def recover_stale_claim(
        self,
        entry_id: str,
        *,
        reason: str = "crash",
    ) -> MergeQueueEntry:
        """Release a crashed claim without trusting the former worker fence."""

        now = self._now_ms()
        reason_text = _text(reason, "reason", required=False) or "crash"
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_entry(connection, entry_id)
                if current.status not in {
                    EntryStatus.CLAIMED,
                    EntryStatus.VALIDATING,
                    EntryStatus.MERGING,
                    EntryStatus.ACCEPTED,
                }:
                    self._commit_if_idle(connection)
                    return current
                # Interrupt in-flight validation/merge rows for queryability.
                if current.current_validation_run_id:
                    connection.execute(
                        """
                        UPDATE validation_runs
                        SET status = 'interrupted',
                            outcome = 'interrupted',
                            finished_at_ms = ?
                        WHERE run_id = ? AND status = 'running'
                        """,
                        [now, current.current_validation_run_id],
                    )
                if current.current_merge_attempt_id:
                    connection.execute(
                        """
                        UPDATE merge_attempts
                        SET status = 'interrupted',
                            outcome = 'interrupted',
                            finished_at_ms = ?
                        WHERE merge_attempt_id = ? AND status = 'running'
                        """,
                        [now, current.current_merge_attempt_id],
                    )
                exhausted = current.attempt_count >= self._max_attempts
                status = (
                    EntryStatus.QUARANTINED.value
                    if exhausted
                    else EntryStatus.PENDING.value
                )
                connection.execute(
                    """
                    UPDATE merge_queue_entries
                    SET status = ?,
                        failure_reason = ?,
                        claim_token = '',
                        claim_generation = claim_generation + 1,
                        consumer_id = '',
                        claimed_at_ms = 0,
                        updated_at_ms = ?,
                        revision = revision + 1
                    WHERE entry_id = ?
                    """,
                    [status, reason_text, now, current.entry_id],
                )
                self._record_event(
                    connection,
                    entry_id=current.entry_id,
                    event_type="claim_recovered",
                    body={"reason": reason_text, "exhausted": exhausted},
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
                    [current.entry_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._entry_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- queries -------------------------------------------------------------

    def get_entry(self, entry_id: str) -> MergeQueueEntry | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM merge_queue_entries WHERE entry_id = ?",
                [_text(entry_id, "entry_id")],
            ).fetchone()
        return None if row is None else self._entry_from_row(row)

    def get_validation_run(self, run_id: str) -> ValidationRun | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM validation_runs WHERE run_id = ?",
                [_text(run_id, "run_id")],
            ).fetchone()
        return None if row is None else self._validation_from_row(row)

    def get_merge_attempt(self, merge_attempt_id: str) -> MergeAttempt | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM merge_attempts WHERE merge_attempt_id = ?",
                [_text(merge_attempt_id, "merge_attempt_id")],
            ).fetchone()
        return None if row is None else self._attempt_from_row(row)

    def get_settlement(self, settlement_id: str) -> SettlementReceipt | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM settlement_receipts WHERE settlement_id = ?",
                [_text(settlement_id, "settlement_id")],
            ).fetchone()
        return None if row is None else self._settlement_from_row(row)

    def list_entries(
        self,
        *,
        repository_id: str = "",
        target_branch: str = "",
        status: str | EntryStatus | None = None,
        task_cid: str = "",
    ) -> tuple[MergeQueueEntry, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if repository_id:
            clauses.append("repository_id = ?")
            params.append(_text(repository_id, "repository_id"))
        if target_branch:
            clauses.append("target_branch = ?")
            params.append(_text(target_branch, "target_branch"))
        if status is not None:
            status_value = (
                status.value if isinstance(status, EntryStatus) else str(status)
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
                ORDER BY ordinal ASC
                """,
                params,
            ).fetchall()
        return tuple(self._entry_from_row(row) for row in rows)

    def events(self, *, entry_id: str = "") -> tuple[dict[str, Any], ...]:
        with self._lock:
            connection = self._require()
            if entry_id:
                rows = connection.execute(
                    """
                    SELECT * FROM merge_queue_events
                    WHERE entry_id = ?
                    ORDER BY observed_at_ms ASC, event_id ASC
                    """,
                    [_text(entry_id, "entry_id")],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM merge_queue_events
                    ORDER BY observed_at_ms ASC, event_id ASC
                    """
                ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            mapping = _row_mapping(row)
            body_raw = _row_get(mapping, "body_json", default="{}")
            try:
                body = json.loads(body_raw or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                body = {}
            events.append(
                {
                    "schema": MERGE_EVENT_SCHEMA,
                    "event_id": str(_row_get(mapping, "event_id", default="")),
                    "entry_id": str(_row_get(mapping, "entry_id", default="")),
                    "event_type": str(_row_get(mapping, "event_type", default="")),
                    "observed_at_ms": int(
                        _row_get(mapping, "observed_at_ms", default=0) or 0
                    ),
                    "body": body if isinstance(body, Mapping) else {},
                }
            )
        return tuple(events)

    # -- row mappers ---------------------------------------------------------

    def _entry_from_row(self, row: Any) -> MergeQueueEntry:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return MergeQueueEntry(
            entry_id=str(_row_get(mapping, "entry_id", default="")),
            repository_id=str(_row_get(mapping, "repository_id", default="")),
            target_branch=str(_row_get(mapping, "target_branch", default="")),
            source_branch=str(_row_get(mapping, "source_branch", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            commit_sha=str(_row_get(mapping, "commit_sha", default="") or ""),
            priority=str(_row_get(mapping, "priority", default="P2")),
            status=EntryStatus(str(_row_get(mapping, "status", default="pending"))),
            ordinal=int(_row_get(mapping, "ordinal", default=0) or 0),
            enqueued_at_ms=int(_row_get(mapping, "enqueued_at_ms", default=0) or 0),
            updated_at_ms=int(_row_get(mapping, "updated_at_ms", default=0) or 0),
            claimed_at_ms=int(_row_get(mapping, "claimed_at_ms", default=0) or 0),
            consumer_id=str(_row_get(mapping, "consumer_id", default="") or ""),
            claim_token=str(_row_get(mapping, "claim_token", default="") or ""),
            claim_generation=int(
                _row_get(mapping, "claim_generation", default=0) or 0
            ),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            attempt_count=int(_row_get(mapping, "attempt_count", default=0) or 0),
            failure_count=int(_row_get(mapping, "failure_count", default=0) or 0),
            failure_reason=str(
                _row_get(mapping, "failure_reason", default="") or ""
            ),
            retry_not_before_ms=int(
                _row_get(mapping, "retry_not_before_ms", default=0) or 0
            ),
            current_validation_run_id=str(
                _row_get(mapping, "current_validation_run_id", default="") or ""
            ),
            current_merge_attempt_id=str(
                _row_get(mapping, "current_merge_attempt_id", default="") or ""
            ),
            settlement_id=str(
                _row_get(mapping, "settlement_id", default="") or ""
            ),
            revision=int(_row_get(mapping, "revision", default=1) or 1),
            body=body if isinstance(body, Mapping) else {},
        )

    def _validation_from_row(self, row: Any) -> ValidationRun:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return ValidationRun(
            run_id=str(_row_get(mapping, "run_id", default="")),
            entry_id=str(_row_get(mapping, "entry_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            attempt_id=str(_row_get(mapping, "attempt_id", default="") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            claim_token=str(_row_get(mapping, "claim_token", default="")),
            claim_generation=int(
                _row_get(mapping, "claim_generation", default=0) or 0
            ),
            command_digest=str(
                _row_get(mapping, "command_digest", default="") or ""
            ),
            status=ValidationStatus(
                str(_row_get(mapping, "status", default="running"))
            ),
            started_at_ms=int(_row_get(mapping, "started_at_ms", default=0) or 0),
            finished_at_ms=int(
                _row_get(mapping, "finished_at_ms", default=0) or 0
            ),
            evidence_digest=str(
                _row_get(mapping, "evidence_digest", default="") or ""
            ),
            outcome=str(_row_get(mapping, "outcome", default="") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    def _attempt_from_row(self, row: Any) -> MergeAttempt:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return MergeAttempt(
            merge_attempt_id=str(
                _row_get(mapping, "merge_attempt_id", default="")
            ),
            entry_id=str(_row_get(mapping, "entry_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            claim_token=str(_row_get(mapping, "claim_token", default="")),
            claim_generation=int(
                _row_get(mapping, "claim_generation", default=0) or 0
            ),
            status=MergeAttemptStatus(
                str(_row_get(mapping, "status", default="running"))
            ),
            outcome=str(_row_get(mapping, "outcome", default="") or ""),
            result_commit_id=str(
                _row_get(mapping, "result_commit_id", default="") or ""
            ),
            started_at_ms=int(_row_get(mapping, "started_at_ms", default=0) or 0),
            finished_at_ms=int(
                _row_get(mapping, "finished_at_ms", default=0) or 0
            ),
            body=body if isinstance(body, Mapping) else {},
        )

    def _settlement_from_row(self, row: Any) -> SettlementReceipt:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return SettlementReceipt(
            settlement_id=str(_row_get(mapping, "settlement_id", default="")),
            entry_id=str(_row_get(mapping, "entry_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="")),
            merge_attempt_id=str(
                _row_get(mapping, "merge_attempt_id", default="")
            ),
            validation_run_id=str(
                _row_get(mapping, "validation_run_id", default="")
            ),
            worktree_id=str(_row_get(mapping, "worktree_id", default="")),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            result_commit_id=str(
                _row_get(mapping, "result_commit_id", default="")
            ),
            evidence_digest=str(
                _row_get(mapping, "evidence_digest", default="")
            ),
            settled_at_ms=int(_row_get(mapping, "settled_at_ms", default=0) or 0),
            body=body if isinstance(body, Mapping) else {},
        )


def open_database_merge_queue(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_processing: int = DEFAULT_MAX_PROCESSING,
    priority_aging_ms: int = 300_000,
) -> DatabaseMergeQueue:
    """Open a :class:`DatabaseMergeQueue` on ``database_path``."""

    return DatabaseMergeQueue(
        database_path,
        clock_ms=clock_ms,
        max_attempts=max_attempts,
        max_processing=max_processing,
        priority_aging_ms=priority_aging_ms,
    ).open()


__all__ = [
    "DATABASE_MERGE_QUEUE_INTERFACE",
    "VALIDATION_RUN_INTERFACE",
    "MERGE_ATTEMPT_INTERFACE",
    "SETTLEMENT_RECEIPT_INTERFACE",
    "DATABASE_MERGE_QUEUE_SCHEMA",
    "VALIDATION_RUN_SCHEMA",
    "MERGE_ATTEMPT_SCHEMA",
    "SETTLEMENT_RECEIPT_SCHEMA",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_PROCESSING",
    "EntryStatus",
    "ValidationStatus",
    "MergeAttemptStatus",
    "MergeOutcome",
    "MergeQueueEntry",
    "ValidationRun",
    "MergeAttempt",
    "SettlementReceipt",
    "DatabaseMergeQueue",
    "DatabaseMergeQueueError",
    "DatabaseMergeQueueNotOpenError",
    "DatabaseMergeQueueConflictError",
    "DatabaseMergeQueueStaleFenceError",
    "DatabaseMergeQueueBoundsError",
    "DatabaseMergeQueueNotReadyError",
    "DuckDBUnavailableError",
    "duckdb_available",
    "open_database_merge_queue",
]
