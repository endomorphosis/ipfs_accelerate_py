"""Durable, deduplicating merge queue for implementation lanes.

The queue is deliberately process safe. Producers may be independent daemon
processes, but only one consumer can atomically claim a request. DuckDB is the
authoritative index and small JSON files are retained as human-readable stage
receipts.  A request is idempotent when both its canonical task identity and
source commit match an existing request, including a completed or quarantined
request.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from ..task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBRow,
    connect_duckdb_with_policy,
    exclusive_file_lock,
    initialize_duckdb_database,
    open_duckdb_connection,
)

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_ACTIVE_STATES = ("pending", "processing")
_COMMIT_METADATA_KEYS = (
    "commit_sha",
    "source_commit",
    "implementation_commit",
    "candidate_commit",
    "head_sha",
    "commit",
)
_CANONICAL_METADATA_KEYS = (
    "canonical_task_key",
    "canonical_task_id",
    "canonical_task_cid",
    "task_cid",
)
_LEGACY_JSON_IMPORT_MARKER = "merge_queue:legacy_json_import@1"
MERGE_QUEUE_THROUGHPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-queue-throughput@1"
)
MERGE_TARGET_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
)
MERGE_QUEUE_SETTLEMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-queue-settlement@1"
)
FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "merge-queue-false-positive-completion-reopen@1"
)
_FALSE_POSITIVE_COMPLETION_RECEIPT_FIELDS = frozenset(
    {
        "already_merged",
        "canonical_task_id",
        "commit_sha",
        "distributed_publication_admission",
        "finished_at",
        "integrated",
        "merge_commit",
        "merged",
        "mutation_short_circuited",
        "reason",
        "request_id",
        "started_at",
        "status",
        "target_branch",
        "target_commit",
        "task_id",
    }
)
_FALSE_POSITIVE_COMPLETION_ADMISSION_FIELDS = frozenset(
    {"admitted", "distributed", "request_id", "schema", "status"}
)
_DISTRIBUTED_LANE_ADMISSION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/distributed-lane-admission@1"
)
_FALSE_POSITIVE_COMPLETION_REOPEN_FIELDS = frozenset(
    {
        "schema",
        "reason",
        "train_receipt_id",
        "previous_target_commit",
        "previous_claim_generation",
        "reopened_at",
    }
)
_FALSE_POSITIVE_COMPLETION_REOPEN_METADATA_KEY = (
    "false_positive_completion_reopen"
)
MAX_MERGE_QUEUE_DEFERRAL_SECONDS = 3600.0
MAX_MERGE_QUEUE_RECORDED_DEFERRALS = 32
# A healthy index of a few dozen receipts is well under 1 MiB. Larger files
# are leftover row-group bloat; opening one under the 256 MB DuckDB cap
# leaves no room for a startup write-commit and abort the lane.
MERGE_QUEUE_BLOAT_REBUILD_BYTES = 8 * 1024 * 1024
_MERGE_REQUEST_COPY_COLUMNS = (
    "request_id",
    "branch_name",
    "task_id",
    "priority",
    "lane_id",
    "enqueued_at",
    "attempt",
    "metadata_json",
    "commit_sha",
    "canonical_task_id",
    "canonical_task_key",
    "dedupe_key",
    "status",
    "claimed_at",
    "consumer_id",
    "failure_count",
    "failure_reason",
    "claim_token",
    "claim_generation",
    "retry_not_before",
    "finished_at",
    "updated_at",
)
_MERGE_QUEUE_SCHEMA_SQL = """
                CREATE TABLE IF NOT EXISTS merge_requests (
                    request_id TEXT PRIMARY KEY,
                    branch_name TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    enqueued_at DOUBLE NOT NULL,
                    attempt INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    canonical_task_id TEXT NOT NULL,
                    canonical_task_key TEXT NOT NULL,
                    dedupe_key TEXT,
                    status TEXT NOT NULL,
                    claimed_at DOUBLE NOT NULL DEFAULT 0,
                    consumer_id TEXT NOT NULL DEFAULT '',
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT NOT NULL DEFAULT '',
                    claim_token TEXT NOT NULL DEFAULT '',
                    claim_generation BIGINT NOT NULL DEFAULT 0,
                    retry_not_before DOUBLE NOT NULL DEFAULT 0,
                    finished_at DOUBLE NOT NULL DEFAULT 0,
                    updated_at DOUBLE NOT NULL
                );
                ALTER TABLE merge_requests
                  ADD COLUMN IF NOT EXISTS claim_token TEXT DEFAULT '';
                ALTER TABLE merge_requests
                  ADD COLUMN IF NOT EXISTS claim_generation BIGINT DEFAULT 0;
                ALTER TABLE merge_requests
                  ADD COLUMN IF NOT EXISTS retry_not_before DOUBLE DEFAULT 0;
                UPDATE merge_requests
                  SET claim_token=COALESCE(claim_token, ''),
                      claim_generation=COALESCE(claim_generation, 0),
                      retry_not_before=COALESCE(retry_not_before, 0)
                  WHERE claim_token IS NULL OR claim_generation IS NULL
                     OR retry_not_before IS NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS merge_requests_dedupe
                  ON merge_requests(dedupe_key);
                CREATE INDEX IF NOT EXISTS merge_requests_stage_order
                  ON merge_requests(status, enqueued_at);
                CREATE INDEX IF NOT EXISTS merge_requests_retry_eligibility
                  ON merge_requests(status, retry_not_before, enqueued_at);
                """

_MERGE_QUEUE_SETTLEMENT_STATES = (
    "pending",
    "processing",
    "completed",
    "quarantined",
    "cancelled",
)
_MERGE_QUEUE_SETTLEMENT_MAX_ACTIVE_IDS = 1024
_MERGE_QUEUE_SETTLEMENT_MAX_METADATA_BYTES = 64 * 1024
_MERGE_QUEUE_SETTLEMENT_MAX_STORE_METADATA_ROWS = 256
_MERGE_QUEUE_SETTLEMENT_LOCK_TIMEOUT_SECONDS = 5.0
_MERGE_QUEUE_SETTLEMENT_COLUMNS = {
    "agent_supervisor_store_metadata": (
        ("key", "VARCHAR", "NO"),
        ("value", "VARCHAR", "NO"),
    ),
    "merge_requests": (
        ("request_id", "VARCHAR", "NO"),
        ("branch_name", "VARCHAR", "NO"),
        ("task_id", "VARCHAR", "NO"),
        ("priority", "VARCHAR", "NO"),
        ("lane_id", "VARCHAR", "NO"),
        ("enqueued_at", "DOUBLE", "NO"),
        ("attempt", "INTEGER", "NO"),
        ("metadata_json", "VARCHAR", "NO"),
        ("commit_sha", "VARCHAR", "NO"),
        ("canonical_task_id", "VARCHAR", "NO"),
        ("canonical_task_key", "VARCHAR", "NO"),
        ("dedupe_key", "VARCHAR", "YES"),
        ("status", "VARCHAR", "NO"),
        ("claimed_at", "DOUBLE", "NO"),
        ("consumer_id", "VARCHAR", "NO"),
        ("failure_count", "INTEGER", "NO"),
        ("failure_reason", "VARCHAR", "NO"),
        ("claim_token", "VARCHAR", "NO"),
        ("claim_generation", "BIGINT", "NO"),
        ("retry_not_before", "DOUBLE", "NO"),
        ("finished_at", "DOUBLE", "NO"),
        ("updated_at", "DOUBLE", "NO"),
    ),
}


class MergeQueueFullError(RuntimeError):
    """Raised when accepting another active request would exceed queue capacity."""


class MergeQueueFenceError(RuntimeError):
    """Raised when stale or non-owning work tries to mutate a claimed request."""


class MergeQueueIntegrityError(RuntimeError):
    """Raised when durable queue identities disagree with legacy projections."""


def _settlement_canonical_bytes(value: Mapping[str, Any] | list[Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _settlement_content_id(value: Mapping[str, Any] | list[Any]) -> str:
    return "sha256:" + hashlib.sha256(_settlement_canonical_bytes(value)).hexdigest()


def _finite_nonnegative_timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        timestamp = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return timestamp if math.isfinite(timestamp) and timestamp >= 0 else None


def _settlement_regular_file_identity(path: Path, *, label: str) -> dict[str, int]:
    try:
        details = path.lstat()
    except OSError as exc:
        raise MergeQueueIntegrityError(
            f"merge queue settlement {label} is unavailable"
        ) from exc
    if not stat.S_ISREG(details.st_mode):
        raise MergeQueueIntegrityError(
            f"merge queue settlement {label} must be a regular file"
        )
    return {
        "device": int(details.st_dev),
        "inode": int(details.st_ino),
        "size_bytes": int(details.st_size),
        "modified_ns": int(details.st_mtime_ns),
        "changed_ns": int(details.st_ctime_ns),
    }


def _merge_queue_settlement_lock_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lock_timeout_seconds must be a finite number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0 or timeout > 5.0:
        raise ValueError("lock_timeout_seconds must be between 0 and 5 seconds")
    return timeout


@contextmanager
def _locked_existing_merge_queue_store(
    lock_path: Path,
    *,
    timeout_seconds: float,
) -> Iterator[None]:
    """Lock an existing queue without creating or modifying its lock file."""

    expected = _settlement_regular_file_identity(lock_path, label="lock")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise MergeQueueIntegrityError(
            "merge queue settlement lock could not be opened"
        ) from exc
    acquired = False
    deadline = time.monotonic() + timeout_seconds
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_dev) != expected["device"]
            or int(opened.st_ino) != expected["inode"]
        ):
            raise MergeQueueIntegrityError(
                "merge queue settlement lock identity changed"
            )
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise MergeQueueIntegrityError(
                        "merge queue settlement lock is busy"
                    ) from None
                time.sleep(0.01)
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _strict_settlement_metadata(value: str) -> dict[str, Any]:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate metadata key: {key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    decoded = json.loads(
        value,
        object_pairs_hook=build_object,
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise ValueError("merge request metadata must be a JSON object")
    return decoded


def _read_merge_queue_settlement_under_lock(
    queue_dir: Path | str,
    *,
    target_repository_id: str,
    target_branch: str,
    max_active_ids: int = 256,
    expected_database: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Read one content-addressed settlement while the queue lock is held.

    This is the operator/goal-authority read path.  It opens only a pre-existing
    local DuckDB store and lock file, takes one bounded read-only transaction,
    and never constructs ``MergeQueue`` or runs queue setup/reconciliation.
    ``settled`` is true only when no pending or processing rows exist. Every
    active row must carry the exact repository and branch binding requested by
    the caller; a mixed, legacy, malformed, or unreadable queue is rejected.
    """

    if (
        not isinstance(target_repository_id, str)
        or not target_repository_id
        or target_repository_id != target_repository_id.strip()
        or "\x00" in target_repository_id
    ):
        raise ValueError("target_repository_id must be an exact non-empty string")
    if (
        not isinstance(target_branch, str)
        or not target_branch
        or target_branch != target_branch.strip()
        or "\x00" in target_branch
    ):
        raise ValueError("target_branch must be an exact non-empty string")
    if (
        type(max_active_ids) is not int
        or max_active_ids < 1
        or max_active_ids > _MERGE_QUEUE_SETTLEMENT_MAX_ACTIVE_IDS
    ):
        raise ValueError(
            "max_active_ids must be an integer between 1 and "
            f"{_MERGE_QUEUE_SETTLEMENT_MAX_ACTIVE_IDS}"
        )

    directory = Path(queue_dir)
    try:
        directory_details = directory.lstat()
    except OSError as exc:
        raise MergeQueueIntegrityError(
            "merge queue settlement directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(directory_details.st_mode):
        raise MergeQueueIntegrityError(
            "merge queue settlement directory must be an existing real directory"
        )
    database_path = directory / "merge_queue.duckdb"
    wal_path = directory / "merge_queue.duckdb.wal"
    initial_database = _settlement_regular_file_identity(
        database_path,
        label="database",
    )
    if initial_database["size_bytes"] <= 0:
        raise MergeQueueIntegrityError(
            "merge queue settlement database must not be empty"
        )
    if wal_path.exists() or wal_path.is_symlink():
        raise MergeQueueIntegrityError(
            "merge queue settlement database has an outstanding write-ahead log"
        )
    if expected_database is not None and dict(expected_database) != initial_database:
        raise MergeQueueIntegrityError(
            "merge queue settlement database identity changed before lock"
        )

    status_counts: dict[str, int] = {
        state: 0 for state in _MERGE_QUEUE_SETTLEMENT_STATES
    }
    active_requests: list[dict[str, str]] = []
    store_metadata: list[list[str]] = []
    row_count = 0
    max_updated_at = 0.0
    max_claim_generation = 0
    final_database: dict[str, int]
    connection: Any | None = None
    transaction_open = False
    try:
        def read_locked_snapshot() -> None:
            nonlocal connection
            nonlocal final_database
            nonlocal max_claim_generation
            nonlocal max_updated_at
            nonlocal row_count
            nonlocal transaction_open

            locked_database = _settlement_regular_file_identity(
                database_path,
                label="database",
            )
            if locked_database != initial_database:
                raise MergeQueueIntegrityError(
                    "merge queue settlement database identity changed before read"
                )
            if wal_path.exists() or wal_path.is_symlink():
                raise MergeQueueIntegrityError(
                    "merge queue settlement database has an outstanding write-ahead log"
                )
            try:
                import duckdb

                connection = connect_duckdb_with_policy(
                    duckdb,
                    database_path,
                    read_only=True,
                )
                connection.execute("BEGIN TRANSACTION")
                transaction_open = True

                table_rows = connection.execute(
                    """
                    SELECT table_name, table_type
                    FROM information_schema.tables
                    WHERE table_schema = 'main'
                      AND table_name IN (
                          'agent_supervisor_store_metadata',
                          'merge_requests'
                      )
                    ORDER BY table_name
                    """
                ).fetchall()
                expected_tables = [
                    ("agent_supervisor_store_metadata", "BASE TABLE"),
                    ("merge_requests", "BASE TABLE"),
                ]
                if [tuple(row) for row in table_rows] != expected_tables:
                    raise MergeQueueIntegrityError(
                        "merge queue settlement database tables are missing or malformed"
                    )

                column_rows = connection.execute(
                    """
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'main'
                      AND table_name IN (
                          'agent_supervisor_store_metadata',
                          'merge_requests'
                      )
                    ORDER BY table_name, ordinal_position
                    """
                ).fetchall()
                observed_columns: dict[str, list[tuple[str, str, str]]] = {
                    table_name: []
                    for table_name in _MERGE_QUEUE_SETTLEMENT_COLUMNS
                }
                for table_name, column_name, data_type, is_nullable in column_rows:
                    name = str(table_name)
                    if name not in observed_columns:
                        raise MergeQueueIntegrityError(
                            "merge queue settlement database columns are malformed"
                        )
                    observed_columns[name].append(
                        (str(column_name), str(data_type), str(is_nullable))
                    )
                if any(
                    tuple(observed_columns[table_name]) != expected
                    for table_name, expected in _MERGE_QUEUE_SETTLEMENT_COLUMNS.items()
                ):
                    raise MergeQueueIntegrityError(
                        "merge queue settlement database columns are missing or malformed"
                    )

                metadata_rows = connection.execute(
                    """
                    SELECT
                        CASE WHEN octet_length(encode(key)) <= 1024 THEN key END,
                        octet_length(encode(key)),
                        CASE WHEN octet_length(encode(value)) <= ? THEN value END,
                        octet_length(encode(value))
                    FROM agent_supervisor_store_metadata
                    ORDER BY key
                    LIMIT ?
                    """,
                    (
                        _MERGE_QUEUE_SETTLEMENT_MAX_METADATA_BYTES,
                        _MERGE_QUEUE_SETTLEMENT_MAX_STORE_METADATA_ROWS + 1,
                    ),
                ).fetchall()
                if len(metadata_rows) > _MERGE_QUEUE_SETTLEMENT_MAX_STORE_METADATA_ROWS:
                    raise MergeQueueIntegrityError(
                        "merge queue settlement store metadata exceeds the read bound"
                    )
                for key, key_size, value, value_size in metadata_rows:
                    if (
                        key is None
                        or value is None
                        or int(key_size) <= 0
                        or int(key_size) > 1024
                        or int(value_size) > _MERGE_QUEUE_SETTLEMENT_MAX_METADATA_BYTES
                    ):
                        raise MergeQueueIntegrityError(
                            "merge queue settlement store metadata is malformed"
                        )
                    store_metadata.append([str(key), str(value)])

                status_rows = connection.execute(
                    """
                    SELECT
                        CASE WHEN octet_length(encode(status)) <= 32 THEN status END,
                        octet_length(encode(status)),
                        count(*)
                    FROM merge_requests
                    GROUP BY status
                    ORDER BY status
                    LIMIT ?
                    """,
                    (len(_MERGE_QUEUE_SETTLEMENT_STATES) + 1,),
                ).fetchall()
                if len(status_rows) > len(_MERGE_QUEUE_SETTLEMENT_STATES):
                    raise MergeQueueIntegrityError(
                        "merge queue settlement contains unknown states"
                    )
                for status_value, status_size, count in status_rows:
                    if (
                        status_value is None
                        or int(status_size) <= 0
                        or str(status_value) not in status_counts
                        or int(count) < 0
                    ):
                        raise MergeQueueIntegrityError(
                            "merge queue settlement contains an unknown state"
                        )
                    status_counts[str(status_value)] = int(count)

                summary = connection.execute(
                    """
                    SELECT
                        count(*),
                        count(DISTINCT request_id),
                        COALESCE(max(updated_at), 0.0),
                        COALESCE(max(claim_generation), 0)
                    FROM merge_requests
                    """
                ).fetchone()
                if summary is None:
                    raise MergeQueueIntegrityError(
                        "merge queue settlement summary is unavailable"
                    )
                row_count = int(summary[0])
                distinct_request_ids = int(summary[1])
                max_updated_at = float(summary[2])
                max_claim_generation = int(summary[3])
                if (
                    row_count < 0
                    or distinct_request_ids != row_count
                    or sum(status_counts.values()) != row_count
                    or not math.isfinite(max_updated_at)
                    or max_updated_at < 0
                    or max_claim_generation < 0
                ):
                    raise MergeQueueIntegrityError(
                        "merge queue settlement summary is malformed"
                    )

                active_rows = connection.execute(
                    """
                    SELECT
                        CASE WHEN octet_length(encode(request_id)) <= 512 THEN request_id END,
                        octet_length(encode(request_id)),
                        status,
                        CASE WHEN octet_length(encode(metadata_json)) <= ?
                             THEN metadata_json END,
                        octet_length(encode(metadata_json))
                    FROM merge_requests
                    WHERE status IN ('pending', 'processing')
                    ORDER BY request_id
                    LIMIT ?
                    """,
                    (
                        _MERGE_QUEUE_SETTLEMENT_MAX_METADATA_BYTES,
                        max_active_ids + 1,
                    ),
                ).fetchall()
                expected_active_count = (
                    status_counts["pending"] + status_counts["processing"]
                )
                if (
                    expected_active_count > max_active_ids
                    or len(active_rows) != expected_active_count
                ):
                    raise MergeQueueIntegrityError(
                        "merge queue settlement active request bound was exceeded"
                    )
                for (
                    request_id,
                    request_id_size,
                    status_value,
                    metadata_json,
                    metadata_size,
                ) in active_rows:
                    if (
                        request_id is None
                        or int(request_id_size) <= 0
                        or int(request_id_size) > 512
                        or metadata_json is None
                        or int(metadata_size) <= 0
                        or int(metadata_size)
                        > _MERGE_QUEUE_SETTLEMENT_MAX_METADATA_BYTES
                    ):
                        raise MergeQueueIntegrityError(
                            "merge queue settlement active request is malformed"
                        )
                    status_text = str(status_value)
                    if status_text not in _ACTIVE_STATES:
                        raise MergeQueueIntegrityError(
                            "merge queue settlement active request state is malformed"
                        )
                    try:
                        metadata = _strict_settlement_metadata(str(metadata_json))
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise MergeQueueIntegrityError(
                            "merge queue settlement active request metadata is malformed"
                        ) from exc
                    if (
                        metadata.get("target_binding_schema")
                        != MERGE_TARGET_BINDING_SCHEMA
                        or metadata.get("target_repository_id")
                        != target_repository_id
                        or metadata.get("target_branch") != target_branch
                    ):
                        raise MergeQueueIntegrityError(
                            "merge queue settlement active request target is unbound or differs"
                        )
                    active_requests.append(
                        {
                            "request_id": str(request_id),
                            "status": status_text,
                        }
                    )

                connection.execute("COMMIT")
                transaction_open = False
            except BaseException:
                if connection is not None and transaction_open:
                    try:
                        connection.execute("ROLLBACK")
                    except Exception:
                        pass
                    transaction_open = False
                raise
            finally:
                if connection is not None:
                    connection.close()
                    connection = None

            final_database = _settlement_regular_file_identity(
                database_path,
                label="database",
            )
            if final_database != locked_database:
                raise MergeQueueIntegrityError(
                    "merge queue settlement database changed during read"
                )
            if wal_path.exists() or wal_path.is_symlink():
                raise MergeQueueIntegrityError(
                    "merge queue settlement database changed during read"
                )

        read_locked_snapshot()
    except MergeQueueIntegrityError:
        raise
    except Exception as exc:
        raise MergeQueueIntegrityError(
            "merge queue settlement store could not be read"
        ) from exc

    store_metadata_cid = _settlement_content_id(store_metadata)
    metadata_by_key = {key: value for key, value in store_metadata}
    database_identity = {
        "path": str(database_path.resolve(strict=True)),
        **final_database,
    }
    snapshot = {
        "database": database_identity,
        "store_metadata_cid": store_metadata_cid,
        "store_metadata_rows": len(store_metadata),
        "store_id": metadata_by_key.get("store_id"),
        "store_generation": metadata_by_key.get("store_generation"),
        "row_count": row_count,
        "max_updated_at": max_updated_at,
        "max_claim_generation": max_claim_generation,
        "status_counts": status_counts,
        "active_requests": active_requests,
    }
    receipt: dict[str, Any] = {
        "schema": MERGE_QUEUE_SETTLEMENT_SCHEMA,
        "settled": not active_requests,
        "target": {
            "binding_schema": MERGE_TARGET_BINDING_SCHEMA,
            "repository_id": target_repository_id,
            "branch": target_branch,
        },
        "database": database_identity,
        "store": {
            "metadata_cid": store_metadata_cid,
            "metadata_rows": len(store_metadata),
            "store_id": metadata_by_key.get("store_id"),
            "generation": metadata_by_key.get("store_generation"),
        },
        "row_count": row_count,
        "max_updated_at": max_updated_at,
        "max_claim_generation": max_claim_generation,
        "status_counts": status_counts,
        "active_count": len(active_requests),
        "active_request_ids": [
            request["request_id"] for request in active_requests
        ],
        "snapshot_cid": _settlement_content_id(snapshot),
    }
    receipt["receipt_cid"] = _settlement_content_id(receipt)
    return receipt


@contextmanager
def hold_merge_queue_settlement(
    queue_dir: Path | str,
    *,
    target_repository_id: str,
    target_branch: str,
    max_active_ids: int = 256,
    lock_timeout_seconds: float = _MERGE_QUEUE_SETTLEMENT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Yield a queue settlement receipt while retaining its writer lock.

    The guard is the authorization form of the settlement API: callers may
    perform their compare-and-swap while the yielded receipt remains protected
    from queue writers.  The database identity is checked again before the
    lock is released, including when the guarded operation raises.
    """

    timeout = _merge_queue_settlement_lock_timeout(lock_timeout_seconds)
    directory = Path(queue_dir)
    try:
        directory_details = directory.lstat()
    except OSError as exc:
        raise MergeQueueIntegrityError(
            "merge queue settlement directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(directory_details.st_mode):
        raise MergeQueueIntegrityError(
            "merge queue settlement directory must be an existing real directory"
        )
    database_path = directory / "merge_queue.duckdb"
    lock_path = directory / ".merge_queue.duckdb.lock"
    wal_path = directory / "merge_queue.duckdb.wal"
    initial_database = _settlement_regular_file_identity(
        database_path,
        label="database",
    )
    if initial_database["size_bytes"] <= 0:
        raise MergeQueueIntegrityError(
            "merge queue settlement database must not be empty"
        )
    if wal_path.exists() or wal_path.is_symlink():
        raise MergeQueueIntegrityError(
            "merge queue settlement database has an outstanding write-ahead log"
        )

    with _locked_existing_merge_queue_store(
        lock_path,
        timeout_seconds=timeout,
    ):
        receipt = _read_merge_queue_settlement_under_lock(
            directory,
            target_repository_id=target_repository_id,
            target_branch=target_branch,
            max_active_ids=max_active_ids,
            expected_database=initial_database,
        )
        expected_database = {
            key: receipt["database"][key]
            for key in (
                "device",
                "inode",
                "size_bytes",
                "modified_ns",
                "changed_ns",
            )
        }
        try:
            yield receipt
        finally:
            observed_database = _settlement_regular_file_identity(
                database_path,
                label="database",
            )
            if observed_database != expected_database:
                raise MergeQueueIntegrityError(
                    "merge queue settlement database changed while guarded"
                )
            if wal_path.exists() or wal_path.is_symlink():
                raise MergeQueueIntegrityError(
                    "merge queue settlement database changed while guarded"
                )


def read_merge_queue_settlement(
    queue_dir: Path | str,
    *,
    target_repository_id: str,
    target_branch: str,
    max_active_ids: int = 256,
    lock_timeout_seconds: float = _MERGE_QUEUE_SETTLEMENT_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Return a convenience snapshot without retaining the queue guard.

    Use :func:`hold_merge_queue_settlement` when a receipt authorizes a later
    mutation; the context manager keeps the queue immutable through that
    mutation and closes the read-to-CAS race.
    """

    with hold_merge_queue_settlement(
        queue_dir,
        target_repository_id=target_repository_id,
        target_branch=target_branch,
        max_active_ids=max_active_ids,
        lock_timeout_seconds=lock_timeout_seconds,
    ) as receipt:
        return receipt


@dataclass(frozen=True)
class MergeRequest:
    """One immutable merge candidate and its durable queue state."""

    request_id: str
    branch_name: str
    task_id: str
    priority: str
    lane_id: str
    enqueued_at: float
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None
    commit_sha: str = ""
    canonical_task_id: str = ""
    canonical_task_key: str = ""
    status: str = "pending"
    claimed_at: float = 0.0
    consumer_id: str = ""
    failure_count: int = 0
    failure_reason: str = ""
    claim_token: str = ""
    claim_generation: int = 0
    retry_not_before: float = 0.0

    @property
    def canonical_identity(self) -> str:
        """Return the strongest task identity supplied by the producer."""

        return self.canonical_task_key or self.canonical_task_id or self.task_id

    @property
    def target_repository_id(self) -> str:
        """Return the physical repository this request may mutate."""

        return str(self.metadata.get("target_repository_id") or "").strip()

    @property
    def target_branch(self) -> str:
        """Return the exact local branch this request may mutate."""

        return str(self.metadata.get("target_branch") or "").strip()

    @property
    def has_target_binding(self) -> bool:
        """Return whether the request carries a complete versioned binding."""

        return bool(
            self.metadata.get("target_binding_schema")
            == MERGE_TARGET_BINDING_SCHEMA
            and self.target_repository_id
            and self.target_branch
        )

    @property
    def dedupe_key(self) -> str:
        """Return the stable task-and-commit idempotency key, when available."""

        if not self.commit_sha:
            return ""
        identity = self.canonical_identity.strip().casefold()
        commit = self.commit_sha.strip().casefold()
        parts = [identity, commit]
        if self.has_target_binding:
            parts.extend(
                (
                    self.target_repository_id,
                    self.target_branch,
                )
            )
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "branch_name": self.branch_name,
            "task_id": self.task_id,
            "priority": self.priority,
            "lane_id": self.lane_id,
            "enqueued_at": self.enqueued_at,
            "attempt": self.attempt,
            "metadata": dict(self.metadata),
            "commit_sha": self.commit_sha,
            "canonical_task_id": self.canonical_task_id,
            "canonical_task_key": self.canonical_task_key,
            "status": self.status,
            "claimed_at": self.claimed_at,
            "consumer_id": self.consumer_id,
            "failure_count": self.failure_count,
            "failure_reason": self.failure_reason,
            "claim_token": self.claim_token,
            "claim_generation": self.claim_generation,
            "retry_not_before": self.retry_not_before,
            "dedupe_key": self.dedupe_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, file_path: Optional[Path] = None) -> "MergeRequest":
        metadata_value = data.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        commit_sha = str(data.get("commit_sha") or "")
        if not commit_sha:
            commit_sha = _first_metadata_value(metadata, _COMMIT_METADATA_KEYS)
        canonical_task_key = str(data.get("canonical_task_key") or "")
        canonical_task_id = str(data.get("canonical_task_id") or "")
        if not canonical_task_key:
            canonical_task_key = _first_metadata_value(metadata, ("canonical_task_key",))
        if not canonical_task_id:
            canonical_task_id = _first_metadata_value(
                metadata, ("canonical_task_id", "canonical_task_cid", "task_cid")
            )
        return cls(
            request_id=str(data.get("request_id") or ""),
            branch_name=str(data.get("branch_name") or data.get("branch") or ""),
            task_id=str(data.get("task_id") or ""),
            priority=_normalise_priority(str(data.get("priority") or "P2")),
            lane_id=str(data.get("lane_id") or ""),
            enqueued_at=_safe_float(data.get("enqueued_at"), 0.0),
            attempt=max(1, _safe_int(data.get("attempt"), 1)),
            metadata=metadata,
            file_path=file_path,
            commit_sha=commit_sha,
            canonical_task_id=canonical_task_id,
            canonical_task_key=canonical_task_key,
            status=str(data.get("status") or "pending"),
            claimed_at=_safe_float(data.get("claimed_at"), 0.0),
            consumer_id=str(data.get("consumer_id") or ""),
            failure_count=max(0, _safe_int(data.get("failure_count"), 0)),
            failure_reason=str(data.get("failure_reason") or ""),
            claim_token=str(data.get("claim_token") or ""),
            claim_generation=max(0, _safe_int(data.get("claim_generation"), 0)),
            retry_not_before=max(
                0.0,
                _safe_float(data.get("retry_not_before"), 0.0),
            ),
        )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_priority(value: str) -> str:
    priority = value.strip().upper()
    return priority if priority in _PRIORITY_ORDER else "P2"


def _first_metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON receipt without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


class MergeQueue:
    """DuckDB-backed priority queue with atomic claims and bounded retries.

    ``priority_aging_seconds`` promotes an old request by one priority tier for
    every elapsed interval.  This keeps P0 ahead under ordinary load while
    guaranteeing that a continuously busy high-priority tier cannot starve an
    older request forever.
    """

    def __init__(
        self,
        queue_dir: Path | str,
        *,
        max_age_seconds: float = 3600,
        max_queue_size: int = 100,
        max_processing: int | None = None,
        max_worktree_bytes: int | None = None,
        worktree_usage: Callable[[], int] | None = None,
        priority_aging_seconds: float = 300,
        max_attempts: int = 3,
        clock: Callable[[], float] | None = None,
        target_repository_id: str = "",
        target_branch: str = "",
        require_target_binding: bool = False,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.processing_dir = self.queue_dir / "processing"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"  # compatibility projection
        self.quarantine_dir = self.queue_dir / "quarantine"
        self.cancelled_dir = self.queue_dir / "cancelled"
        self.database_path = self.queue_dir / "merge_queue.duckdb"
        self._legacy_database_path = self.queue_dir / "merge_queue.sqlite3"
        self.max_age_seconds = max(0.0, float(max_age_seconds))
        self.max_queue_size = max(1, int(max_queue_size))
        self.max_processing = max(
            1,
            int(
                max_processing
                if max_processing is not None
                else self.max_queue_size
            ),
        )
        self.max_worktree_bytes = (
            None
            if max_worktree_bytes is None
            else max(0, int(max_worktree_bytes))
        )
        self._worktree_usage = worktree_usage
        self.priority_aging_seconds = max(0.0, float(priority_aging_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self._clock = clock or time.time
        self.target_repository_id = ""
        self.target_branch = ""
        self.require_target_binding = False
        self.bind_target(
            target_repository_id,
            target_branch,
            required=require_target_binding,
        )
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.quarantine_dir,
            self.cancelled_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._import_legacy_files()
        self._compact_if_bloated()

    def bind_target(
        self,
        target_repository_id: str,
        target_branch: str,
        *,
        required: bool = True,
    ) -> None:
        """Bind this producer/consumer view to one repository and target ref.

        Binding is process-local while every enqueued request persists the
        versioned values. Existing unbound legacy rows remain in the database
        but are invisible to a required bound consumer.
        """

        repository_id = str(target_repository_id or "").strip()
        branch = str(target_branch or "").strip()
        if bool(repository_id) != bool(branch):
            raise ValueError(
                "target_repository_id and target_branch must be supplied together"
            )
        if required and not repository_id:
            raise ValueError("a required merge target binding must not be empty")
        if (
            self.target_repository_id
            and repository_id
            and self.target_repository_id != repository_id
        ):
            raise ValueError("merge queue target repository binding changed")
        if self.target_branch and branch and self.target_branch != branch:
            raise ValueError("merge queue target branch binding changed")
        if repository_id:
            self.target_repository_id = repository_id
            self.target_branch = branch
        self.require_target_binding = bool(
            self.require_target_binding or required
        )

    def _connect(self) -> DuckDBConnection:
        return open_duckdb_connection(self.database_path)

    def _init_database(self) -> None:
        if self._merge_requests_table_exists():
            # A write-transaction COMMIT against a bloated file sits on the
            # 256 MB DuckDB cap and abort-kills the lane (FATAL unique revert).
            return
        initialize_duckdb_database(
            self.database_path,
            legacy_sqlite_path=self._legacy_database_path,
            table_names=("merge_requests",),
            value_transform=lambda table, column, value: (
                None
                if table == "merge_requests"
                and column == "dedupe_key"
                and not str(value or "")
                else value
            ),
            schema_sql=_MERGE_QUEUE_SCHEMA_SQL,
        )

    def _import_legacy_files(self) -> None:
        """Import legacy JSON queue files once, preserving their original stage."""

        stage_dirs = (
            ("pending", self.pending_dir),
            ("processing", self.processing_dir),
            ("completed", self.completed_dir),
            ("quarantined", self.failed_dir),
            ("quarantined", self.quarantine_dir),
            ("cancelled", self.cancelled_dir),
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM merge_requests LIMIT 1"
            ).fetchone()
            if existing is not None:
                # The durable index is already populated. Re-reading every
                # stage receipt on constructor start pins the whole table
                # again and OOMs a 256MB-capped DuckDB on a large queue.
                return
            connection.execute("BEGIN IMMEDIATE")
            try:
                metadata_rows = connection.execute(
                    "SELECT key, value FROM agent_supervisor_store_metadata"
                ).fetchall()
                import_marker_values = [
                    str(row["value"])
                    for row in metadata_rows
                    if str(row["key"]) == _LEGACY_JSON_IMPORT_MARKER
                ]
                if import_marker_values:
                    if import_marker_values != ["complete"]:
                        raise MergeQueueIntegrityError(
                            "legacy merge receipt import marker is invalid"
                        )
                    connection.commit()
                    return

                existing_by_request_id: dict[str, str] = {}
                existing_by_dedupe_key: dict[str, str] = {}
                for row in connection.execute(
                    "SELECT request_id, dedupe_key FROM merge_requests"
                ).fetchall():
                    request_id = str(row["request_id"])
                    dedupe_key = str(row["dedupe_key"] or "")
                    if (
                        request_id in existing_by_request_id
                        and existing_by_request_id[request_id] != dedupe_key
                    ):
                        raise MergeQueueIntegrityError(
                            "merge queue contains conflicting dedupe identities "
                            f"for request_id {request_id!r}"
                        )
                    existing_by_request_id[request_id] = dedupe_key
                    if not dedupe_key:
                        continue
                    previous_request_id = existing_by_dedupe_key.get(dedupe_key)
                    if (
                        previous_request_id is not None
                        and previous_request_id != request_id
                    ):
                        raise MergeQueueIntegrityError(
                            "merge queue contains conflicting request identities "
                            f"for dedupe_key {dedupe_key!r}"
                        )
                    existing_by_dedupe_key[dedupe_key] = request_id

                for status, directory in stage_dirs:
                    for path in sorted(
                        directory.glob("*.json"),
                        key=lambda candidate: candidate.name,
                    ):
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            request = MergeRequest.from_dict(payload, file_path=path)
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            continue
                        if not request.request_id:
                            continue
                        request = replace(request, status=status)
                        request_id = request.request_id
                        dedupe_key = request.dedupe_key
                        if request_id in existing_by_request_id:
                            authoritative_dedupe_key = (
                                existing_by_request_id[request_id]
                            )
                            if (
                                authoritative_dedupe_key
                                and dedupe_key
                                and authoritative_dedupe_key != dedupe_key
                            ):
                                raise MergeQueueIntegrityError(
                                    "legacy merge receipt conflicts with the "
                                    "authoritative dedupe identity for "
                                    f"request_id {request_id!r}"
                                )
                            continue
                        if (
                            dedupe_key
                            and dedupe_key in existing_by_dedupe_key
                        ):
                            raise MergeQueueIntegrityError(
                                "legacy merge receipt conflicts with the "
                                "authoritative request identity for "
                                f"dedupe_key {dedupe_key!r}"
                            )
                        self._insert(connection, request, ignore=False)
                        existing_by_request_id[request_id] = dedupe_key
                        if dedupe_key:
                            existing_by_dedupe_key[dedupe_key] = request_id
                connection.execute(
                    """
                    INSERT INTO agent_supervisor_store_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    (_LEGACY_JSON_IMPORT_MARKER, "complete"),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _merge_requests_table_exists(self) -> bool:
        if not self.database_path.is_file() or self.database_path.stat().st_size <= 0:
            return False
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'main'
                      AND table_name = 'merge_requests'
                    LIMIT 1
                    """
                ).fetchone()
        except Exception:
            return False
        return row is not None

    def _stage_request_ids(self) -> list[str]:
        request_ids: list[str] = []
        seen: set[str] = set()
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.quarantine_dir,
            self.cancelled_dir,
        ):
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                request_id = path.stem.strip()
                if not request_id or request_id in seen:
                    continue
                seen.add(request_id)
                request_ids.append(request_id)
        return request_ids

    def _compact_if_bloated(self) -> None:
        try:
            size = self.database_path.stat().st_size
        except OSError:
            return
        if size < MERGE_QUEUE_BLOAT_REBUILD_BYTES:
            return
        self._rebuild_store_from_live_rows()

    def _rebuild_store_from_live_rows(self) -> None:
        """Rewrite a bloated DuckDB file from the live receipt-backed rows."""

        request_ids = self._stage_request_ids()
        if not request_ids:
            return
        columns = _MERGE_REQUEST_COPY_COLUMNS
        select_sql = (
            "SELECT "
            + ", ".join(columns)
            + " FROM merge_requests WHERE request_id = ?"
        )
        rows: list[tuple[Any, ...]] = []
        meta_rows: list[tuple[Any, ...]] = []
        with self._connect() as connection:
            for request_id in request_ids:
                row = connection.execute(select_sql, (request_id,)).fetchone()
                if row is None:
                    continue
                rows.append(tuple(row[column] for column in columns))
            try:
                meta = connection.execute(
                    "SELECT key, value FROM agent_supervisor_store_metadata"
                ).fetchall()
                meta_rows = [(row["key"], row["value"]) for row in meta]
            except Exception:
                meta_rows = []
        if not rows:
            return
        rebuilt = self.database_path.with_name(f"{self.database_path.name}.rebuild")
        if rebuilt.exists():
            rebuilt.unlink()
        initialize_duckdb_database(
            rebuilt,
            table_names=("merge_requests",),
            schema_sql=_MERGE_QUEUE_SCHEMA_SQL,
        )
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = (
            "INSERT INTO merge_requests ("
            + ", ".join(columns)
            + f") VALUES ({placeholders})"
        )
        with open_duckdb_connection(rebuilt) as connection:
            connection.execute("BEGIN TRANSACTION")
            for row in rows:
                connection.execute(insert_sql, row)
            for key, value in meta_rows:
                connection.execute(
                    """INSERT INTO agent_supervisor_store_metadata(key, value)
                       VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                    (key, value),
                )
            connection.commit()
            connection.execute("CHECKPOINT")
        lock_path = self.database_path.with_name(f".{self.database_path.name}.lock")
        with exclusive_file_lock(lock_path, timeout_seconds=60.0):
            wal = Path(str(self.database_path) + ".wal")
            os.replace(rebuilt, self.database_path)
            try:
                if wal.exists():
                    wal.unlink()
            except OSError:
                pass
            try:
                os.chmod(self.database_path, 0o600)
            except OSError:
                pass

    def _insert(
        self,
        connection: DuckDBConnection,
        request: MergeRequest,
        *,
        ignore: bool,
    ) -> None:
        if _FALSE_POSITIVE_COMPLETION_REOPEN_METADATA_KEY in request.metadata:
            raise MergeQueueIntegrityError(
                "false-positive completion reopen metadata is queue-reserved"
            )
        if ignore:
            # INSERT OR IGNORE still appends then reverts on unique conflict.
            # DuckDB treats that revert as FATAL and abort-kills the process.
            existing = connection.execute(
                "SELECT 1 FROM merge_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing is not None:
                return
            if request.dedupe_key:
                existing = connection.execute(
                    "SELECT 1 FROM merge_requests WHERE dedupe_key = ?",
                    (request.dedupe_key,),
                ).fetchone()
                if existing is not None:
                    return
        connection.execute(
            """INSERT INTO merge_requests (
                request_id, branch_name, task_id, priority, lane_id, enqueued_at,
                attempt, metadata_json, commit_sha, canonical_task_id,
                canonical_task_key, dedupe_key, status, claimed_at, consumer_id,
                failure_count, failure_reason, claim_token, claim_generation,
                retry_not_before, finished_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request.request_id,
                request.branch_name,
                request.task_id,
                request.priority,
                request.lane_id,
                request.enqueued_at,
                request.attempt,
                json.dumps(request.metadata, sort_keys=True, separators=(",", ":"), default=str),
                request.commit_sha,
                request.canonical_task_id,
                request.canonical_task_key,
                request.dedupe_key or None,
                request.status,
                request.claimed_at,
                request.consumer_id,
                request.failure_count,
                request.failure_reason,
                request.claim_token,
                request.claim_generation,
                request.retry_not_before,
                0.0,
                self._clock(),
            ),
        )

    @staticmethod
    def _find_by_dedupe_key(
        connection: DuckDBConnection,
        dedupe_key: str,
    ) -> DuckDBRow | None:
        """Find one dedupe identity without trusting a secondary ART index."""

        expected = str(dedupe_key or "")
        if not expected:
            return None
        matches = [
            row
            for row in connection.execute(
                "SELECT * FROM merge_requests"
            ).fetchall()
            if str(row["dedupe_key"] or "") == expected
        ]
        if len(matches) > 1:
            raise MergeQueueIntegrityError(
                "merge queue contains multiple requests for dedupe_key "
                f"{expected!r}"
            )
        return matches[0] if matches else None

    def enqueue(
        self,
        *,
        branch_name: str,
        task_id: str,
        priority: str = "P2",
        lane_id: str = "",
        attempt: int = 1,
        metadata: dict[str, Any] | None = None,
        commit_sha: str = "",
        canonical_task_id: str = "",
        canonical_task_key: str = "",
        canonical_task_cid: str = "",
        target_repository_id: str = "",
        target_branch: str = "",
    ) -> MergeRequest:
        """Atomically enqueue or return the existing task-and-commit request."""

        if not str(branch_name).strip():
            raise ValueError("branch_name must not be empty")
        if not str(task_id).strip():
            raise ValueError("task_id must not be empty")
        metadata_dict = dict(metadata or {})
        if _FALSE_POSITIVE_COMPLETION_REOPEN_METADATA_KEY in metadata_dict:
            raise ValueError(
                "false-positive completion reopen metadata is queue-reserved"
            )
        declared_repository_id = str(
            target_repository_id
            or metadata_dict.get("target_repository_id")
            or ""
        ).strip()
        declared_branch = str(
            target_branch or metadata_dict.get("target_branch") or ""
        ).strip()
        if self.target_repository_id:
            if (
                declared_repository_id
                and declared_repository_id != self.target_repository_id
            ):
                raise ValueError(
                    "request target repository differs from the queue binding"
                )
            if declared_branch and declared_branch != self.target_branch:
                raise ValueError(
                    "request target branch differs from the queue binding"
                )
            declared_repository_id = self.target_repository_id
            declared_branch = self.target_branch
        if bool(declared_repository_id) != bool(declared_branch):
            raise ValueError(
                "request target_repository_id and target_branch must be "
                "supplied together"
            )
        if self.require_target_binding and not declared_repository_id:
            raise ValueError("bound merge queue refuses an unbound request")
        if declared_repository_id:
            supplied_schema = str(
                metadata_dict.get("target_binding_schema") or ""
            ).strip()
            if supplied_schema and supplied_schema != MERGE_TARGET_BINDING_SCHEMA:
                raise ValueError("request merge target binding schema changed")
            metadata_dict.update(
                {
                    "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
                    "target_repository_id": declared_repository_id,
                    "target_branch": declared_branch,
                }
            )
        commit_sha = str(commit_sha or _first_metadata_value(metadata_dict, _COMMIT_METADATA_KEYS)).strip()
        canonical_task_key = str(
            canonical_task_key
            or _first_metadata_value(metadata_dict, ("canonical_task_key",))
        ).strip()
        canonical_task_id = str(
            canonical_task_id
            or canonical_task_cid
            or _first_metadata_value(metadata_dict, ("canonical_task_id", "canonical_task_cid", "task_cid"))
        ).strip()
        now = self._clock()
        request = MergeRequest(
            request_id=f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:12]}",
            branch_name=str(branch_name).strip(),
            task_id=str(task_id).strip(),
            priority=_normalise_priority(priority),
            lane_id=str(lane_id or os.getpid()),
            enqueued_at=now,
            attempt=max(1, int(attempt)),
            metadata=metadata_dict,
            commit_sha=commit_sha,
            canonical_task_id=canonical_task_id,
            canonical_task_key=canonical_task_key,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if request.dedupe_key:
                    row = self._find_by_dedupe_key(
                        connection,
                        request.dedupe_key,
                    )
                    if row is not None:
                        connection.commit()
                        return self._request_from_row(row)
                active_rows = connection.execute(
                    """SELECT metadata_json FROM merge_requests
                       WHERE status IN ('pending','processing')"""
                ).fetchall()
                active_count = sum(
                    self._metadata_matches_target(row["metadata_json"])
                    for row in active_rows
                )
                if active_count >= self.max_queue_size:
                    connection.rollback()
                    raise MergeQueueFullError(
                        f"merge queue capacity {self.max_queue_size} has been reached"
                    )
                # Never INSERT a row whose request_id or dedupe_key already
                # exists. DuckDB INSERT OR IGNORE still appends-then-reverts
                # and abort-kills the process on that unique-index revert.
                self._insert(connection, request, ignore=True)
                stored = connection.execute(
                    "SELECT * FROM merge_requests WHERE request_id = ?",
                    (request.request_id,),
                ).fetchone()
                if stored is None and request.dedupe_key:
                    stored = connection.execute(
                        "SELECT * FROM merge_requests WHERE dedupe_key = ?",
                        (request.dedupe_key,),
                    ).fetchone()
                if stored is None:
                    connection.rollback()
                    raise RuntimeError(
                        "merge queue insert was ignored without an existing row"
                    )
                connection.commit()
                request = self._request_from_row(stored)
            except Exception:
                connection.rollback()
                if not request.dedupe_key:
                    raise
                row = self._find_by_dedupe_key(
                    connection,
                    request.dedupe_key,
                )
                if row is None:
                    raise
                return self._request_from_row(row)
        receipt_path = self._write_stage_receipt(request)
        return replace(request, file_path=receipt_path)

    def _metadata_matches_target(self, value: Any) -> bool:
        """Return whether one durable row belongs to this consumer view."""

        if not self.target_repository_id:
            return not self.require_target_binding
        try:
            metadata = (
                json.loads(value or "{}")
                if not isinstance(value, Mapping)
                else value
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(metadata, Mapping):
            return False
        return bool(
            metadata.get("target_binding_schema")
            == MERGE_TARGET_BINDING_SCHEMA
            and str(metadata.get("target_repository_id") or "").strip()
            == self.target_repository_id
            and str(metadata.get("target_branch") or "").strip()
            == self.target_branch
        )

    def _target_binding_sql(self) -> tuple[str, tuple[str, ...]]:
        """Return an exact SQL target fence for bounded stage snapshots."""

        if not self.target_repository_id:
            return "", ()
        return (
            " AND json_extract_string(metadata_json, "
            "'$.target_binding_schema') = ?"
            " AND json_extract_string(metadata_json, "
            "'$.target_repository_id') = ?"
            " AND json_extract_string(metadata_json, '$.target_branch') = ?",
            (
                MERGE_TARGET_BINDING_SCHEMA,
                self.target_repository_id,
                self.target_branch,
            ),
        )

    def _require_row_target(
        self,
        row: DuckDBRow,
        *,
        operation: str,
        request_id: str,
    ) -> None:
        """Fence mutations attempted through a foreign bound queue view."""

        if not self._metadata_matches_target(row["metadata_json"]):
            raise MergeQueueFenceError(
                f"{operation} rejected for request {request_id}: "
                "request target differs from the queue binding"
            )

    @staticmethod
    def _requires_explicit_false_positive_recovery(value: Any) -> bool:
        """Keep queue-authored recovery rows out of generic dequeue paths."""

        try:
            metadata = (
                json.loads(value or "{}")
                if not isinstance(value, Mapping)
                else value
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(metadata, Mapping)
            and _FALSE_POSITIVE_COMPLETION_REOPEN_METADATA_KEY in metadata
        )

    def dequeue(self, consumer_id: str = "") -> Optional[MergeRequest]:
        """Atomically claim the fairest pending request for one consumer."""

        claimed = self.dequeue_many(1, consumer_id=consumer_id)
        return claimed[0] if claimed else None

    def claim_pending_request(
        self,
        request: MergeRequest | str,
        *,
        consumer_id: str = "",
    ) -> Optional[MergeRequest]:
        """Atomically claim one exact pending request.

        A request-specific recovery consumer must not accidentally dequeue a
        different board's work after reconstructing the original request's
        completion authority.  This path retains the ordinary processing and
        worktree-capacity fences while binding the claim to one request id.
        """

        request_id = (
            request.request_id
            if isinstance(request, MergeRequest)
            else str(request)
        )
        if not request_id:
            return None
        self._purge_stale()
        consumer = str(consumer_id or os.getpid())
        now = self._clock()
        claimed_row: DuckDBRow | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM merge_requests WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                self._require_row_target(
                    row,
                    operation="claim_pending_request",
                    request_id=request_id,
                )
                if (
                    str(row["status"]) != "pending"
                    or float(row["retry_not_before"] or 0) > now
                ):
                    connection.commit()
                    return None

                processing_rows = connection.execute(
                    "SELECT metadata_json FROM merge_requests "
                    "WHERE status='processing'"
                ).fetchall()
                if self.target_repository_id or self.require_target_binding:
                    processing_rows = [
                        item
                        for item in processing_rows
                        if self._metadata_matches_target(item["metadata_json"])
                    ]
                if len(processing_rows) >= self.max_processing:
                    connection.commit()
                    return None
                reserved_bytes = sum(
                    self._worktree_bytes_from_metadata_json(
                        item["metadata_json"]
                    )
                    for item in processing_rows
                )
                observed_bytes = self._observed_worktree_bytes()
                requested_bytes = self._worktree_bytes_from_metadata_json(
                    row["metadata_json"]
                )
                if (
                    self.max_worktree_bytes is not None
                    and (
                        self.max_worktree_bytes <= 0
                        or max(reserved_bytes, observed_bytes)
                        + requested_bytes
                        > self.max_worktree_bytes
                    )
                ):
                    connection.commit()
                    return None

                claim_token = uuid.uuid4().hex
                updated = connection.execute(
                    """UPDATE merge_requests
                       SET status='processing', claimed_at=?, consumer_id=?,
                           claim_token=?, claim_generation=claim_generation + 1,
                           retry_not_before=0, updated_at=?
                       WHERE request_id=? AND status='pending'
                         AND retry_not_before <= ?""",
                    (
                        now,
                        consumer,
                        claim_token,
                        now,
                        request_id,
                        now,
                    ),
                )
                if updated.rowcount == 1:
                    claimed_row = connection.execute(
                        "SELECT * FROM merge_requests WHERE request_id=?",
                        (request_id,),
                    ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if claimed_row is None:
            return None
        claimed = self._request_from_row(claimed_row)
        receipt_path = self._write_stage_receipt(claimed)
        return replace(claimed, file_path=receipt_path)

    def dequeue_many(
        self,
        limit: int,
        consumer_id: str = "",
    ) -> tuple[MergeRequest, ...]:
        """Atomically claim a bounded, deterministically ordered preflight batch.

        ``max_processing`` is the merge-debt/backpressure fence.  Batch
        producers cannot reserve more worktrees or validation capacity than
        the configured number of in-flight requests, even when multiple
        processes race to claim work.
        """

        requested = int(limit)
        if requested <= 0:
            return ()
        self._purge_stale()
        consumer = str(consumer_id or os.getpid())
        now = self._clock()
        claimed_rows: list[DuckDBRow] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                processing_rows = connection.execute(
                    "SELECT metadata_json FROM merge_requests WHERE status='processing'"
                ).fetchall()
                if self.target_repository_id or self.require_target_binding:
                    processing_rows = [
                        row
                        for row in processing_rows
                        if self._metadata_matches_target(row["metadata_json"])
                    ]
                processing = len(processing_rows)
                capacity = max(0, self.max_processing - processing)
                claim_count = min(requested, capacity)
                if claim_count <= 0:
                    connection.commit()
                    return ()
                reserved_bytes = sum(
                    self._worktree_bytes_from_metadata_json(row["metadata_json"])
                    for row in processing_rows
                )
                observed_bytes = self._observed_worktree_bytes()
                worktree_bytes = max(reserved_bytes, observed_bytes)
                rows = connection.execute(
                    """SELECT * FROM merge_requests
                       WHERE status = 'pending' AND retry_not_before <= ?""",
                    (now,),
                ).fetchall()
                if self.target_repository_id or self.require_target_binding:
                    rows = [
                        row
                        for row in rows
                        if self._metadata_matches_target(row["metadata_json"])
                    ]
                rows = [
                    row
                    for row in rows
                    if not self._requires_explicit_false_positive_recovery(
                        row["metadata_json"]
                    )
                ]
                if not rows:
                    connection.commit()
                    return ()
                selected: list[DuckDBRow] = []
                for row in sorted(rows, key=lambda item: self._fairness_key(item, now)):
                    if len(selected) >= claim_count:
                        break
                    estimate = self._worktree_bytes_from_metadata_json(
                        row["metadata_json"]
                    )
                    if (
                        self.max_worktree_bytes is not None
                        and (
                            self.max_worktree_bytes <= 0
                            or worktree_bytes + estimate > self.max_worktree_bytes
                        )
                    ):
                        continue
                    selected.append(row)
                    worktree_bytes += estimate
                for row in selected:
                    claim_token = uuid.uuid4().hex
                    updated = connection.execute(
                        """UPDATE merge_requests
                           SET status='processing', claimed_at=?, consumer_id=?,
                               claim_token=?, claim_generation=claim_generation + 1,
                               retry_not_before=0, updated_at=?
                           WHERE request_id=? AND status='pending'""",
                        (
                            now,
                            consumer,
                            claim_token,
                            now,
                            row["request_id"],
                        ),
                    )
                    if updated.rowcount != 1:
                        continue
                    claimed_row = connection.execute(
                        "SELECT * FROM merge_requests WHERE request_id=?",
                        (row["request_id"],),
                    ).fetchone()
                    if claimed_row is not None:
                        claimed_rows.append(claimed_row)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        claimed: list[MergeRequest] = []
        for row in claimed_rows:
            request = self._request_from_row(row)
            receipt_path = self._write_stage_receipt(request)
            claimed.append(replace(request, file_path=receipt_path))
        return tuple(claimed)

    def _worktree_bytes_from_metadata_json(self, value: Any) -> int:
        """Read a reservation estimate, conservatively bounding unknown work."""

        try:
            metadata = json.loads(value or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return self.max_worktree_bytes or 0
        if not isinstance(metadata, Mapping):
            return self.max_worktree_bytes or 0
        for key in (
            "worktree_bytes",
            "estimated_worktree_bytes",
            "worktree_disk_bytes",
        ):
            if key not in metadata:
                continue
            return max(0, _safe_int(metadata.get(key), 0))
        # Once a disk limit is requested, an unestimated worktree reserves the
        # whole budget.  This admits it serially without allowing missing
        # producer metadata to defeat the bound.
        return self.max_worktree_bytes or 0

    def _observed_worktree_bytes(self) -> int:
        """Return observed worktree use, failing closed when a configured probe fails."""

        if self._worktree_usage is None:
            return 0
        try:
            return max(0, int(self._worktree_usage()))
        except Exception:
            return self.max_worktree_bytes or 0

    def _fairness_key(self, row: DuckDBRow, now: float) -> tuple[int, float, str]:
        base = _PRIORITY_ORDER.get(str(row["priority"]), _PRIORITY_ORDER["P2"])
        if self.priority_aging_seconds > 0:
            promotions = int(max(0.0, now - float(row["enqueued_at"])) / self.priority_aging_seconds)
            effective = max(0, base - promotions)
        else:
            effective = base
        return effective, float(row["enqueued_at"]), str(row["request_id"])

    def _claim_matches(
        self,
        row: DuckDBRow,
        request: MergeRequest,
        *,
        consumer_id: str = "",
    ) -> bool:
        """Compare all durable claim coordinates, including ownership."""

        expected_consumer = str(consumer_id or request.consumer_id)
        claimed_at = _safe_float(
            row["claimed_at"] or row["enqueued_at"],
            0.0,
        )
        expired = (
            self.max_age_seconds > 0
            and self._clock() - claimed_at > self.max_age_seconds
        )
        return (
            str(row["status"]) == "processing"
            and not expired
            and bool(request.claim_token)
            and str(row["claim_token"] or "") == request.claim_token
            and int(row["claim_generation"] or 0) == request.claim_generation
            and str(row["consumer_id"] or "") == request.consumer_id
            and (not consumer_id or str(row["consumer_id"] or "") == expected_consumer)
        )

    def owns_claim(
        self,
        request: MergeRequest,
        *,
        consumer_id: str = "",
    ) -> bool:
        """Return whether ``request`` still owns the current processing fence.

        Merge workers should call this immediately before any target mutation.
        The subsequent terminal queue transition performs the same comparison
        atomically, so an expired, cancelled, or recovered claim fails closed.
        """

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request.request_id,),
            ).fetchone()
        return (
            row is not None
            and self._metadata_matches_target(row["metadata_json"])
            and self._claim_matches(row, request, consumer_id=consumer_id)
        )

    def _require_claim(
        self,
        row: DuckDBRow,
        request: MergeRequest,
        *,
        operation: str,
        allow_pending: bool = False,
    ) -> None:
        self._require_row_target(
            row,
            operation=operation,
            request_id=request.request_id,
        )
        status = str(row["status"])
        if allow_pending and status == "pending" and not request.claim_token:
            return
        if not self._claim_matches(row, request):
            raise MergeQueueFenceError(
                f"{operation} rejected for request {request.request_id}: "
                "claim token, generation, owner, or state is stale"
            )

    def complete(self, request: MergeRequest, metadata: Mapping[str, Any] | None = None) -> None:
        """Mark a claimed request complete; duplicate completion is harmless."""

        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return
            self._require_row_target(
                row,
                operation="complete",
                request_id=request.request_id,
            )
            if str(row["status"]) == "completed":
                connection.commit()
                return
            self._require_claim(row, request, operation="complete")
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata["completion"] = dict(metadata)
            connection.execute(
                """UPDATE merge_requests SET status='completed', metadata_json=?,
                   finished_at=?, updated_at=?, consumer_id='', claimed_at=0,
                   claim_token='', claim_generation=claim_generation + 1,
                   retry_not_before=0
                   WHERE request_id=? AND status='processing'
                     AND claim_token=? AND claim_generation=? AND consumer_id=?""",
                (
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now,
                    now,
                    request.request_id,
                    request.claim_token,
                    request.claim_generation,
                    request.consumer_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        self._write_stage_receipt(self._request_from_row(row))
        self._prune_receipts(self.completed_dir, keep=50)

    def reopen_false_positive_completion(
        self,
        request: MergeRequest,
        *,
        completion_receipt: Mapping[str, Any],
    ) -> MergeRequest | None:
        """Reopen one exactly proved false ``already_merged`` completion.

        This is deliberately narrower than an operator retry.  It accepts only
        the train receipt emitted by the metadata-only declared-output
        shortcut, only for a target-bound completed row with no callback
        completion contract, and records the receipt digest before returning
        the row to ``pending``.  The caller must separately prove from Git that
        the candidate's declared outputs are not actually present on target.

        Replaying the same receipt after this transition is observation-only;
        it never regresses a later pending, processing, quarantined, or
        completed generation.
        """

        if not isinstance(completion_receipt, Mapping):
            raise MergeQueueIntegrityError(
                "false-positive completion receipt must be a mapping"
            )
        receipt = dict(completion_receipt)
        receipt_fields = set(receipt)
        content_addressed = receipt_fields == (
            _FALSE_POSITIVE_COMPLETION_RECEIPT_FIELDS | {"receipt_id"}
        )
        if (
            receipt_fields != _FALSE_POSITIVE_COMPLETION_RECEIPT_FIELDS
            and not content_addressed
        ):
            raise MergeQueueIntegrityError(
                "false-positive completion receipt has an unknown shape"
            )
        receipt_content = (
            {key: value for key, value in receipt.items() if key != "receipt_id"}
            if content_addressed
            else receipt
        )
        try:
            receipt_id = _settlement_content_id(receipt_content)
        except (TypeError, ValueError) as exc:
            raise MergeQueueIntegrityError(
                "false-positive completion receipt is not canonical JSON"
            ) from exc
        if content_addressed and receipt.get("receipt_id") != receipt_id:
            raise MergeQueueIntegrityError(
                "false-positive completion receipt content identity changed"
            )
        admission = receipt.get("distributed_publication_admission")
        target_commit = str(receipt.get("target_commit") or "")
        merge_commit = str(receipt.get("merge_commit") or "")
        started_at = receipt.get("started_at")
        finished_at = receipt.get("finished_at")
        started_timestamp = _finite_nonnegative_timestamp(started_at)
        finished_timestamp = _finite_nonnegative_timestamp(finished_at)
        timestamps_valid = bool(
            started_timestamp is not None
            and finished_timestamp is not None
            and finished_timestamp >= started_timestamp
        )
        if (
            receipt.get("request_id") != request.request_id
            or receipt.get("status") != "already_merged"
            or receipt.get("reason") != "declared_outputs_already_on_target"
            or receipt.get("mutation_short_circuited") is not True
            or receipt.get("already_merged") is not True
            or receipt.get("integrated") is not True
            or receipt.get("merged") is not False
            or receipt.get("commit_sha") != request.commit_sha
            or receipt.get("canonical_task_id") != request.canonical_identity
            or receipt.get("task_id") != request.task_id
            or receipt.get("target_branch") != request.target_branch
            or len(target_commit) != 40
            or any(character not in "0123456789abcdef" for character in target_commit)
            or merge_commit != target_commit
            or not timestamps_valid
            or not isinstance(admission, Mapping)
            or set(admission) != _FALSE_POSITIVE_COMPLETION_ADMISSION_FIELDS
            or admission.get("schema") != _DISTRIBUTED_LANE_ADMISSION_SCHEMA
            or admission.get("status") != "local"
            or admission.get("admitted") is not True
            or admission.get("distributed") is not False
            or admission.get("request_id") != request.request_id
        ):
            raise MergeQueueIntegrityError(
                "false-positive completion receipt does not match its request"
            )
        if (
            self.require_target_binding is not True
            or not request.has_target_binding
            or request.target_repository_id != self.target_repository_id
            or request.target_branch != self.target_branch
        ):
            raise MergeQueueIntegrityError(
                "false-positive completion reopen requires the exact queue target"
            )

        now = self._clock()
        if _finite_nonnegative_timestamp(now) is None:
            raise MergeQueueIntegrityError(
                "false-positive completion reopen clock is invalid"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request.request_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="reopen false-positive completion",
                request_id=request.request_id,
            )
            current = self._request_from_row(row)
            prior_reopen = current.metadata.get(
                _FALSE_POSITIVE_COMPLETION_REOPEN_METADATA_KEY
            )
            if (
                isinstance(prior_reopen, Mapping)
                and prior_reopen.get("train_receipt_id") == receipt_id
            ):
                prior_generation = prior_reopen.get("previous_claim_generation")
                reopened_at = prior_reopen.get("reopened_at")
                reopened_timestamp = _finite_nonnegative_timestamp(reopened_at)
                exact_prior_reopen = bool(
                    set(prior_reopen) == _FALSE_POSITIVE_COMPLETION_REOPEN_FIELDS
                    and prior_reopen.get("schema")
                    == FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA
                    and prior_reopen.get("reason")
                    == "declared_outputs_not_on_target"
                    and prior_reopen.get("previous_target_commit")
                    == target_commit
                    and not isinstance(prior_generation, bool)
                    and isinstance(prior_generation, int)
                    and prior_generation == request.claim_generation
                    and reopened_timestamp is not None
                    and current.claim_generation >= prior_generation + 1
                    and (
                        current.claim_generation == prior_generation + 1
                        or current.status
                        in {"processing", "quarantined", "completed", "cancelled"}
                    )
                    and (
                        current.claim_generation != prior_generation + 1
                        or current.status == "pending"
                    )
                )
                if not exact_prior_reopen:
                    connection.rollback()
                    raise MergeQueueIntegrityError(
                        "false-positive completion reopen lineage is malformed"
                    )
                connection.commit()
                return current
            immutable_snapshot_matches = bool(
                current.status == "completed"
                and request.status == "completed"
                and current.request_id == request.request_id
                and current.branch_name == request.branch_name
                and current.task_id == request.task_id
                and current.commit_sha == request.commit_sha
                and current.canonical_task_id == request.canonical_task_id
                and current.canonical_task_key == request.canonical_task_key
                and current.claim_generation == request.claim_generation
                and current.metadata == request.metadata
            )
            if not immutable_snapshot_matches:
                connection.rollback()
                raise MergeQueueFenceError(
                    "false-positive completion reopen rejected a stale row snapshot"
                )
            if "completion" in current.metadata:
                connection.rollback()
                raise MergeQueueIntegrityError(
                    "false-positive completion reopen rejected a callback completion"
                )
            metadata = dict(current.metadata)
            metadata[_FALSE_POSITIVE_COMPLETION_REOPEN_METADATA_KEY] = {
                "schema": FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA,
                "reason": "declared_outputs_not_on_target",
                "train_receipt_id": receipt_id,
                "previous_target_commit": target_commit,
                "previous_claim_generation": int(current.claim_generation),
                "reopened_at": now,
            }
            connection.execute(
                """UPDATE merge_requests SET status='pending', metadata_json=?,
                   claimed_at=0, consumer_id='', claim_token='',
                   claim_generation=claim_generation + 1,
                   retry_not_before=0, finished_at=0, updated_at=?
                   WHERE request_id=? AND status='completed'
                     AND claim_generation=?""",
                (
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    now,
                    request.request_id,
                    int(current.claim_generation),
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request.request_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        reopened = self._request_from_row(row)
        if (
            reopened.status != "pending"
            or reopened.claim_generation != current.claim_generation + 1
        ):
            raise MergeQueueFenceError(
                "false-positive completion reopen lost its completed-row fence"
            )
        self._write_stage_receipt(reopened)
        return reopened

    def fail(
        self,
        request: MergeRequest,
        reason: str = "",
        *,
        retryable: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Record a failure, optionally retrying within the configured bound.

        Terminal failures and exhausted retries are placed in quarantine and
        return the durable receipt path.  A scheduled retry returns ``None``.
        """

        if retryable:
            result = self.requeue(request, reason=reason, metadata=metadata)
            return result if isinstance(result, Path) else None
        return self.quarantine(request, reason=reason, metadata=metadata)

    def requeue(
        self,
        request: MergeRequest,
        reason: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MergeRequest | Path | None:
        """Retry one request once, or quarantine it after ``max_attempts``."""

        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="requeue",
                request_id=request.request_id,
            )
            if str(row["status"]) in {"completed", "quarantined"}:
                connection.commit()
                resolved = self._request_from_row(row)
                if resolved.status == "quarantined":
                    return self._stage_path(resolved)
                return resolved
            self._require_claim(row, request, operation="requeue")
            next_attempt = max(int(row["attempt"]), int(row["failure_count"]) + 1) + 1
            failure_count = int(row["failure_count"]) + 1
            terminal = next_attempt > self.max_attempts
            status = "quarantined" if terminal else "pending"
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata.setdefault("failure_metadata", []).append(dict(metadata))
            connection.execute(
                """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                   failure_reason=?, metadata_json=?, claimed_at=0, consumer_id='',
                   claim_token='', claim_generation=claim_generation + 1,
                   retry_not_before=0, finished_at=?, updated_at=? WHERE request_id=?
                     AND status='processing' AND claim_token=?
                     AND claim_generation=? AND consumer_id=?""",
                (
                    status,
                    next_attempt,
                    failure_count,
                    str(reason),
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now if terminal else 0.0,
                    now,
                    request.request_id,
                    request.claim_token,
                    request.claim_generation,
                    request.consumer_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        updated = self._request_from_row(row)
        path = self._write_stage_receipt(updated)
        return path if terminal else updated

    def defer(
        self,
        request: MergeRequest,
        reason: str = "",
        *,
        delay_seconds: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> MergeRequest | None:
        """Release a claim into a durable cooldown without consuming a retry.

        Deferral is for external contention that prevented a merge attempt
        from starting.  The request remains pending and retains its attempt
        and failure counters; ``retry_not_before`` prevents immediate reclaim.
        """

        delay = float(delay_seconds)
        if not math.isfinite(delay):
            raise ValueError("merge queue deferral delay must be finite")
        if delay > MAX_MERGE_QUEUE_DEFERRAL_SECONDS:
            raise ValueError(
                "merge queue deferral delay exceeds the durable cooldown limit"
            )
        delay = max(0.0, delay)
        now = self._clock()
        retry_not_before = now + delay
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request.request_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="defer",
                request_id=request.request_id,
            )
            self._require_claim(row, request, operation="defer")
            request_metadata = json.loads(row["metadata_json"] or "{}")
            deferral: dict[str, Any] = {
                "at": now,
                "reason": str(reason),
                "retry_not_before": retry_not_before,
            }
            if metadata:
                deferral["metadata"] = dict(metadata)
            deferrals = request_metadata.setdefault("deferrals", [])
            if not isinstance(deferrals, list):
                deferrals = []
            deferrals.append(deferral)
            request_metadata["deferrals"] = deferrals[
                -MAX_MERGE_QUEUE_RECORDED_DEFERRALS:
            ]
            connection.execute(
                """UPDATE merge_requests SET status='pending',
                   metadata_json=?, claimed_at=0,
                   consumer_id='', claim_token='',
                   claim_generation=claim_generation + 1,
                   retry_not_before=?, finished_at=0, updated_at=?
                   WHERE request_id=? AND status='processing'
                     AND claim_token=? AND claim_generation=? AND consumer_id=?""",
                (
                    json.dumps(
                        request_metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    retry_not_before,
                    now,
                    request.request_id,
                    request.claim_token,
                    request.claim_generation,
                    request.consumer_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request.request_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        deferred = self._request_from_row(row)
        receipt_path = self._write_stage_receipt(deferred)
        return replace(deferred, file_path=receipt_path)

    def quarantine(
        self,
        request: MergeRequest,
        reason: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Terminally quarantine one request and materialize its receipt."""

        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="quarantine",
                request_id=request.request_id,
            )
            if str(row["status"]) == "quarantined":
                connection.commit()
                return self._stage_path(self._request_from_row(row))
            self._require_claim(
                row,
                request,
                operation="quarantine",
                allow_pending=True,
            )
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata["quarantine"] = dict(metadata)
            connection.execute(
                """UPDATE merge_requests SET status='quarantined', failure_count=?,
                   failure_reason=?, metadata_json=?, claimed_at=0, consumer_id='',
                   claim_token='', claim_generation=claim_generation + 1,
                   retry_not_before=0, finished_at=?, updated_at=? WHERE request_id=?""",
                (
                    int(row["failure_count"]) + 1,
                    str(reason or row["failure_reason"]),
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now,
                    now,
                    request.request_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._write_stage_receipt(self._request_from_row(row))

    def cancel(
        self,
        request: MergeRequest | str,
        reason: str = "cancelled",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MergeRequest | None:
        """Durably cancel pending work or an exactly fenced processing claim.

        A request id is sufficient for work which has not been claimed.  Once
        processing begins, callers must pass the exact :class:`MergeRequest`
        returned by ``dequeue``; this prevents an operator or stale worker from
        cancelling a newer owner's claim accidentally.
        """

        supplied = request if isinstance(request, MergeRequest) else None
        request_id = supplied.request_id if supplied is not None else str(request)
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="cancel",
                request_id=request_id,
            )
            status = str(row["status"])
            if status == "cancelled":
                connection.commit()
                return self._request_from_row(row)
            if status in {"completed", "quarantined"}:
                connection.commit()
                return self._request_from_row(row)
            if status == "processing":
                if supplied is None:
                    connection.rollback()
                    raise MergeQueueFenceError(
                        f"cancel rejected for request {request_id}: "
                        "a processing request requires its current claim"
                    )
                self._require_claim(row, supplied, operation="cancel")
            request_metadata = json.loads(row["metadata_json"] or "{}")
            cancellation = {"at": now, "reason": str(reason or "cancelled")}
            if metadata:
                cancellation["metadata"] = dict(metadata)
            request_metadata["cancellation"] = cancellation
            connection.execute(
                """UPDATE merge_requests SET status='cancelled', failure_reason=?,
                   metadata_json=?, claimed_at=0, consumer_id='', claim_token='',
                   claim_generation=claim_generation + 1,
                   retry_not_before=0, finished_at=?, updated_at=?
                   WHERE request_id=? AND status IN ('pending','processing')""",
                (
                    str(reason or "cancelled"),
                    json.dumps(
                        request_metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    now,
                    now,
                    request_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        cancelled = self._request_from_row(row)
        receipt_path = self._write_stage_receipt(cancelled)
        return replace(cancelled, file_path=receipt_path)

    def revive_quarantined(
        self,
        request: MergeRequest | str,
        reason: str = "",
        *,
        reset_failures: bool = False,
    ) -> MergeRequest | None:
        """Return a quarantined request to pending after operator review.

        The operation is atomic and idempotent.  A revival record is retained
        in request metadata so administrative recovery does not erase why the
        candidate was quarantined.  ``reset_failures`` is intended for false
        positives such as a host suspension while a request was still pending.
        """

        request_id = request.request_id if isinstance(request, MergeRequest) else str(request)
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="revive",
                request_id=request_id,
            )
            if str(row["status"]) != "quarantined":
                connection.commit()
                return self._request_from_row(row)

            request_metadata = json.loads(row["metadata_json"] or "{}")
            request_metadata.setdefault("revivals", []).append(
                {
                    "at": now,
                    "reason": str(reason),
                    "previous_enqueued_at": float(row["enqueued_at"]),
                    "previous_failure_count": int(row["failure_count"]),
                    "previous_failure_reason": str(row["failure_reason"]),
                }
            )
            failure_count = 0 if reset_failures else int(row["failure_count"])
            attempt = 1 if reset_failures else int(row["attempt"])
            connection.execute(
                """UPDATE merge_requests SET status='pending', enqueued_at=?, attempt=?,
                   failure_count=?, failure_reason='', metadata_json=?, claimed_at=0,
                   consumer_id='', claim_token='',
                   claim_generation=claim_generation + 1,
                   retry_not_before=0, finished_at=0, updated_at=?
                   WHERE request_id=?""",
                (
                    now,
                    attempt,
                    failure_count,
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now,
                    request_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        revived = self._request_from_row(row)
        receipt_path = self._write_stage_receipt(revived)
        return replace(revived, file_path=receipt_path)

    def quarantined_requests(
        self,
        *,
        limit: int = 32,
        after_request_id: str | None = None,
    ) -> tuple[MergeRequest, ...]:
        """Return a bounded deterministic snapshot of target-bound quarantines.

        This is intentionally read-only.  A merge train may use the snapshot
        to prove that an immutable candidate was integrated before a worker
        crashed, then revive that one request through
        :meth:`revive_quarantined`.  Foreign target rows and malformed target
        metadata are never exposed through a bound queue view.
        """

        requested = max(0, min(int(limit), 256))
        if requested == 0:
            return ()
        target_sql, target_parameters = self._target_binding_sql()
        recovery_order = after_request_id is not None
        cursor = str(after_request_id or "")
        cursor_sql = " AND request_id > ?" if cursor else ""
        cursor_parameters = (cursor,) if cursor else ()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM merge_requests "
                "WHERE status='quarantined'"
                + target_sql
                + cursor_sql
                + (
                    " ORDER BY request_id LIMIT ?"
                    if recovery_order
                    else " ORDER BY finished_at, request_id LIMIT ?"
                ),
                (*target_parameters, *cursor_parameters, requested),
            ).fetchall()
        return tuple(
            self._request_from_row(row)
            for row in rows
            if self._metadata_matches_target(row["metadata_json"])
        )[:requested]

    def completed_requests(
        self,
        *,
        limit: int = 32,
        completion_schema: str = "",
        completion_reason: str = "",
        before_request_id: str = "",
    ) -> tuple[MergeRequest, ...]:
        """Return a bounded target-bound completion snapshot.

        Recovery consumers may filter the nested completion contract before
        ``LIMIT`` and paginate by the immutable time-prefixed request id.  All
        pages use that same keyset order; mutable or out-of-order completion
        timestamps therefore cannot hide rows after the first page.
        """

        requested = max(0, min(int(limit), 256))
        if requested == 0:
            return ()
        target_sql, target_parameters = self._target_binding_sql()
        schema = str(completion_schema or "")
        reason = str(completion_reason or "")
        cursor = str(before_request_id or "")
        completion_sql = ""
        completion_parameters: tuple[str, ...] = ()
        if schema:
            completion_sql += (
                " AND json_extract_string(metadata_json, "
                "'$.completion.schema') = ?"
            )
            completion_parameters += (schema,)
        if reason:
            completion_sql += (
                " AND json_extract_string(metadata_json, "
                "'$.completion.reason') = ?"
            )
            completion_parameters += (reason,)
        cursor_sql = " AND request_id < ?" if cursor else ""
        cursor_parameters = (cursor,) if cursor else ()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM merge_requests "
                "WHERE status='completed'"
                + target_sql
                + completion_sql
                + cursor_sql
                + " ORDER BY request_id DESC LIMIT ?",
                (
                    *target_parameters,
                    *completion_parameters,
                    *cursor_parameters,
                    requested,
                ),
            ).fetchall()
        return tuple(
            self._request_from_row(row)
            for row in rows
            if self._metadata_matches_target(row["metadata_json"])
        )[:requested]

    def pending_requests(
        self,
        *,
        limit: int = 32,
        after_request_id: str | None = None,
    ) -> tuple[MergeRequest, ...]:
        """Return a bounded fair-order snapshot of target-bound pending work."""

        requested = max(0, min(int(limit), 256))
        if requested == 0:
            return ()
        now = self._clock()
        target_sql, target_parameters = self._target_binding_sql()
        recovery_order = after_request_id is not None
        cursor = str(after_request_id or "")
        cursor_sql = " AND request_id > ?" if cursor else ""
        cursor_parameters = (cursor,) if cursor else ()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM merge_requests "
                "WHERE status='pending' AND retry_not_before <= ?"
                + target_sql
                + cursor_sql
                + (" ORDER BY request_id LIMIT ?" if recovery_order else " LIMIT ?"),
                (
                    now,
                    *target_parameters,
                    *cursor_parameters,
                    requested,
                ),
            ).fetchall()
        matching = [
            row
            for row in rows
            if self._metadata_matches_target(row["metadata_json"])
        ]
        ordered = (
            matching
            if recovery_order
            else sorted(
                matching,
                key=lambda item: self._fairness_key(item, now),
            )
        )
        return tuple(self._request_from_row(row) for row in ordered[:requested])

    def processing_requests(
        self,
        *,
        limit: int = 32,
        after_request_id: str | None = None,
    ) -> tuple[MergeRequest, ...]:
        """Return a bounded oldest-first target-bound processing snapshot.

        Visibility is not abandonment authority.  A caller may use this
        snapshot to discover an exact request id after a crash, but only the
        merge train consumer lease may recover a processing claim.
        """

        requested = max(0, min(int(limit), 256))
        if requested == 0:
            return ()
        target_sql, target_parameters = self._target_binding_sql()
        recovery_order = after_request_id is not None
        cursor = str(after_request_id or "")
        cursor_sql = " AND request_id > ?" if cursor else ""
        cursor_parameters = (cursor,) if cursor else ()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM merge_requests "
                "WHERE status='processing'"
                + target_sql
                + cursor_sql
                + (
                    " ORDER BY request_id LIMIT ?"
                    if recovery_order
                    else " ORDER BY claimed_at, request_id LIMIT ?"
                ),
                (*target_parameters, *cursor_parameters, requested),
            ).fetchall()
        return tuple(
            self._request_from_row(row)
            for row in rows
            if self._metadata_matches_target(row["metadata_json"])
        )[:requested]

    def get(self, request_id: str) -> MergeRequest | None:
        """Return the current durable request by id."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        return self._request_from_row(row) if row is not None else None

    def active_canonical_task_ids(self) -> set[str]:
        """Return content identities currently waiting for merge or being merged."""

        return self._canonical_task_ids_for_statuses(_ACTIVE_STATES)

    def completed_canonical_task_ids(self) -> set[str]:
        """Return content identities with a successful terminal merge receipt."""

        return self._canonical_task_ids_for_statuses(("completed",))

    def completed_task_cid_bindings(self) -> dict[str, set[str]]:
        """Return CID-bound primary and bundle members from completed receipts.

        Only versioned queue requests whose primary row identity agrees with
        the metadata binding participate.  This lets another daemon lane
        repair a task board after callback completion without treating a
        display ID or queue admission as proof of completion.
        """

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT task_id, canonical_task_id, metadata_json
                   FROM merge_requests
                   WHERE status='completed'"""
            ).fetchall()
        bindings: dict[str, set[str]] = {}
        for row in rows:
            raw_metadata = row["metadata_json"] or "{}"
            if not self._metadata_matches_target(raw_metadata):
                continue
            try:
                metadata = json.loads(raw_metadata)
            except (TypeError, json.JSONDecodeError):
                continue
            if (
                not isinstance(metadata, dict)
                or metadata.get("schema")
                != "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
            ):
                continue
            raw_bindings = metadata.get("completion_task_cids")
            if not isinstance(raw_bindings, dict) or not raw_bindings:
                continue
            primary_task_id = str(row["task_id"] or "")
            primary_task_cid = str(row["canonical_task_id"] or "")
            if (
                not primary_task_id
                or not primary_task_cid
                or str(raw_bindings.get(primary_task_id) or "")
                != primary_task_cid
            ):
                continue
            for task_id, task_cid in raw_bindings.items():
                normalized_id = str(task_id).strip()
                normalized_cid = str(task_cid).strip()
                if normalized_id and normalized_cid:
                    bindings.setdefault(normalized_id, set()).add(
                        normalized_cid
                    )
        return bindings

    def _canonical_task_ids_for_statuses(self, statuses: tuple[str, ...]) -> set[str]:
        normalized = tuple(
            dict.fromkeys(
                str(status).strip() for status in statuses if str(status).strip()
            )
        )
        if not normalized:
            return set()
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT canonical_task_id, metadata_json
                    FROM merge_requests
                    WHERE status IN ({placeholders}) AND canonical_task_id != ''""",
                normalized,
            ).fetchall()
        return {
            str(row["canonical_task_id"])
            for row in rows
            if self._metadata_matches_target(row["metadata_json"])
        }

    def pending_count(self) -> int:
        return self._count("pending")

    def processing_count(self) -> int:
        return self._count("processing")

    def _count(self, status: str) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT metadata_json FROM merge_requests WHERE status=?",
                (status,),
            ).fetchall()
        return sum(
            self._metadata_matches_target(row["metadata_json"])
            for row in rows
        )

    def has_pending_for_task(
        self,
        task_id: str,
        *,
        commit_sha: str | None = None,
    ) -> bool:
        """Return whether a task (and optionally commit) is active."""

        identity = str(task_id).strip().casefold()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT task_id, canonical_task_id, canonical_task_key,
                          commit_sha, metadata_json
                   FROM merge_requests WHERE status IN ('pending','processing')"""
            ).fetchall()
        for row in rows:
            if not self._metadata_matches_target(row["metadata_json"]):
                continue
            identities = {
                str(row["task_id"]).casefold(),
                str(row["canonical_task_id"]).casefold(),
                str(row["canonical_task_key"]).casefold(),
            }
            if identity not in identities:
                continue
            if commit_sha is None or str(row["commit_sha"]).casefold() == str(commit_sha).casefold():
                return True
        return False

    def _purge_stale(self) -> int:
        """Recover abandoned consumer claims that exceeded their lease bound.

        Pending requests have no consumer lease and therefore do not expire.
        Queue capacity and explicit cancellation bound their lifetime.  This
        distinction also keeps a suspended host from quarantining valid work.
        """

        if self.max_age_seconds <= 0:
            return 0
        now = self._clock()
        changed: list[MergeRequest] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM merge_requests WHERE status='processing'"
            ).fetchall()
            for row in rows:
                if not self._metadata_matches_target(row["metadata_json"]):
                    continue
                reference_time = float(row["claimed_at"] or row["enqueued_at"])
                if now - reference_time <= self.max_age_seconds:
                    continue
                attempt = int(row["attempt"])
                failure_count = int(row["failure_count"])
                if attempt < self.max_attempts:
                    new_status = "pending"
                    new_attempt = attempt + 1
                    failure_count += 1
                    reason = "consumer claim expired; request recovered"
                    finished_at = 0.0
                else:
                    new_status = "quarantined"
                    new_attempt = attempt
                    failure_count += 1
                    reason = "processing request exceeded max age"
                    finished_at = now
                connection.execute(
                    """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                       failure_reason=?, claimed_at=0, consumer_id='', claim_token='',
                       claim_generation=claim_generation + 1,
                       retry_not_before=0, finished_at=?,
                       updated_at=? WHERE request_id=?""",
                    (
                        new_status,
                        new_attempt,
                        failure_count,
                        reason,
                        finished_at,
                        now,
                        row["request_id"],
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM merge_requests WHERE request_id=?", (row["request_id"],)
                ).fetchone()
                if updated is not None:
                    changed.append(self._request_from_row(updated))
            connection.commit()
        for request in changed:
            self._write_stage_receipt(request)
        return len(changed)

    def recover_abandoned_train_claims(self) -> int:
        """Recover claims left by a crashed process-safe merge train.

        Callers must hold the merge train's repo-wide consumer lock. Once that
        lock is acquired, no live ``merge-train:*`` consumer can still own a
        processing row, so waiting for the general queue age timeout only
        wastes throughput. Claims from other queue consumers are untouched.
        """

        now = self._clock()
        changed: list[MergeRequest] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM merge_requests WHERE status='processing' AND consumer_id LIKE 'merge-train:%'"
            ).fetchall()
            for row in rows:
                if not self._metadata_matches_target(row["metadata_json"]):
                    continue
                attempt = int(row["attempt"])
                failure_count = int(row["failure_count"]) + 1
                if attempt < self.max_attempts:
                    status = "pending"
                    next_attempt = attempt + 1
                    finished_at = 0.0
                    reason = "merge train consumer exited; claim recovered"
                else:
                    status = "quarantined"
                    next_attempt = attempt
                    finished_at = now
                    reason = "merge train consumer exited on final attempt"
                connection.execute(
                    """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                       failure_reason=?, claimed_at=0, consumer_id='', claim_token='',
                       claim_generation=claim_generation + 1,
                       retry_not_before=0, finished_at=?,
                       updated_at=? WHERE request_id=? AND status='processing'""",
                    (
                        status,
                        next_attempt,
                        failure_count,
                        reason,
                        finished_at,
                        now,
                        row["request_id"],
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM merge_requests WHERE request_id=?", (row["request_id"],)
                ).fetchone()
                if updated is not None:
                    changed.append(self._request_from_row(updated))
            connection.commit()
        for request in changed:
            self._write_stage_receipt(request)
        return len(changed)

    def status(self) -> dict[str, Any]:
        """Return an authoritative stage summary suitable for daemon status."""

        with self._connect() as connection:
            stage_rows = connection.execute(
                """SELECT status, enqueued_at, finished_at, metadata_json
                   FROM merge_requests"""
            ).fetchall()
            stage_rows = [
                row
                for row in stage_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
            counts: dict[str, int] = {}
            for row in stage_rows:
                status = str(row["status"])
                counts[status] = counts.get(status, 0) + 1
            timing_rows = connection.execute(
                """SELECT enqueued_at, finished_at, metadata_json
                   FROM merge_requests
                   WHERE status='completed' AND finished_at > 0
                   ORDER BY finished_at"""
            ).fetchall()
            timing_rows = [
                row
                for row in timing_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
            processing_rows = connection.execute(
                "SELECT metadata_json FROM merge_requests WHERE status='processing'"
            ).fetchall()
            processing_rows = [
                row
                for row in processing_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
            pending_rows = connection.execute(
                "SELECT metadata_json FROM merge_requests WHERE status='pending'"
            ).fetchall()
            pending_rows = [
                row
                for row in pending_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
        completed_span = (
            max(
                0.0,
                float(timing_rows[-1]["finished_at"])
                - float(timing_rows[0]["enqueued_at"]),
            )
            if timing_rows
            else 0.0
        )
        active = counts.get("pending", 0) + counts.get("processing", 0)
        merge_debt = counts.get("processing", 0)
        reserved_worktree_bytes = sum(
            self._worktree_bytes_from_metadata_json(row["metadata_json"])
            for row in processing_rows
        )
        observed_worktree_bytes = self._observed_worktree_bytes()
        worktree_bytes_in_use = max(
            reserved_worktree_bytes,
            observed_worktree_bytes,
        )
        disk_backpressure = (
            self.max_worktree_bytes is not None
            and (
                worktree_bytes_in_use >= self.max_worktree_bytes
                or any(
                    worktree_bytes_in_use
                    + self._worktree_bytes_from_metadata_json(row["metadata_json"])
                    > self.max_worktree_bytes
                    for row in pending_rows
                )
            )
        )
        return {
            "pending": counts.get("pending", 0),
            "processing": merge_debt,
            "completed": counts.get("completed", 0),
            "failed": counts.get("quarantined", 0),
            "quarantined": counts.get("quarantined", 0),
            "cancelled": counts.get("cancelled", 0),
            "total": sum(counts.values()),
            "queue_dir": str(self.queue_dir),
            "database_path": str(self.database_path),
            "target_repository_id": self.target_repository_id,
            "target_branch": self.target_branch,
            "target_binding_required": self.require_target_binding,
            "max_attempts": self.max_attempts,
            "max_queue_size": self.max_queue_size,
            "max_processing": self.max_processing,
            "merge_debt": merge_debt,
            "max_worktree_bytes": self.max_worktree_bytes,
            "reserved_worktree_bytes": reserved_worktree_bytes,
            "observed_worktree_bytes": observed_worktree_bytes,
            "worktree_bytes_in_use": worktree_bytes_in_use,
            "disk_backpressure": disk_backpressure,
            "backpressure": (
                active >= self.max_queue_size
                or merge_debt >= self.max_processing
                or disk_backpressure
            ),
            "throughput": {
                "schema": MERGE_QUEUE_THROUGHPUT_SCHEMA,
                "lane": "merge-queue-persistence",
                "accepted_count": len(timing_rows),
                "elapsed_seconds": completed_span,
                "accepted_per_second": (
                    len(timing_rows) / completed_span
                    if completed_span > 0
                    else 0.0
                ),
            },
        }

    def _request_from_row(self, row: DuckDBRow) -> MergeRequest:
        status = str(row["status"])
        payload = {
            "request_id": row["request_id"],
            "branch_name": row["branch_name"],
            "task_id": row["task_id"],
            "priority": row["priority"],
            "lane_id": row["lane_id"],
            "enqueued_at": row["enqueued_at"],
            "attempt": row["attempt"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "commit_sha": row["commit_sha"],
            "canonical_task_id": row["canonical_task_id"],
            "canonical_task_key": row["canonical_task_key"],
            "status": status,
            "claimed_at": row["claimed_at"],
            "consumer_id": row["consumer_id"],
            "failure_count": row["failure_count"],
            "failure_reason": row["failure_reason"],
            "claim_token": row["claim_token"],
            "claim_generation": row["claim_generation"],
            "retry_not_before": row["retry_not_before"],
        }
        request = MergeRequest.from_dict(payload)
        return replace(request, file_path=self._stage_path(request))

    def _stage_path(self, request: MergeRequest) -> Path:
        stage_dir = {
            "pending": self.pending_dir,
            "processing": self.processing_dir,
            "completed": self.completed_dir,
            "quarantined": self.quarantine_dir,
            "cancelled": self.cancelled_dir,
        }.get(request.status, self.failed_dir)
        return stage_dir / f"{request.request_id}.json"

    def _write_stage_receipt(self, request: MergeRequest) -> Path:
        destination = self._stage_path(request)
        payload = request.to_dict()
        if request.status == "quarantined":
            payload.update(
                {
                    "receipt_type": "merge_quarantine",
                    "quarantined_at": self._clock(),
                    "receipt_id": hashlib.sha256(
                        f"{request.request_id}\0{request.failure_reason}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        elif request.status == "cancelled":
            payload.update(
                {
                    "receipt_type": "merge_cancellation",
                    "cancelled_at": self._clock(),
                    "receipt_id": hashlib.sha256(
                        (
                            f"{request.request_id}\0{request.failure_reason}"
                            f"\0{request.claim_generation}"
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
        _atomic_write_json(destination, payload)
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.quarantine_dir,
            self.cancelled_dir,
        ):
            candidate = directory / destination.name
            if candidate == destination:
                continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        return destination

    @staticmethod
    def _prune_receipts(directory: Path, *, keep: int) -> None:
        paths = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime)
        for path in paths[:-keep]:
            try:
                path.unlink()
            except OSError:
                pass


__all__ = [
    "FALSE_POSITIVE_COMPLETION_REOPEN_SCHEMA",
    "MergeQueue",
    "MergeQueueFullError",
    "MergeQueueFenceError",
    "MergeQueueIntegrityError",
    "MergeRequest",
    "_PRIORITY_ORDER",
]
