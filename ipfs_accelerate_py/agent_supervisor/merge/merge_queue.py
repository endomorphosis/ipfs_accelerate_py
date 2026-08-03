"""Durable, deduplicating merge queue for implementation lanes.

The queue is deliberately process safe. Producers may be independent daemon
processes, but only one consumer can atomically claim a request. DuckDB is the
authoritative index and small JSON files are retained as human-readable stage
receipts.  A request is idempotent when both its canonical task identity and
source commit match an existing request, including a completed or quarantined
request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBRow,
    initialize_duckdb_database,
    open_duckdb_connection,
)
from ..proof.formal_verification_contracts import content_identity


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
MERGE_QUEUE_THROUGHPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-queue-throughput@1"
)
MERGE_TARGET_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
)
SUBMODULE_INTEGRATION_RECOVERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/submodule-integration-recovery@1"
)
POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-review-denial-tombstone@2"
)
_FULL_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_GIT_TREE_ID = re.compile(r"^git-tree:[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MAX_DENIAL_RECORD_BYTES = 16 * 1024


class MergeQueueFullError(RuntimeError):
    """Raised when accepting another active request would exceed queue capacity."""


class MergeQueueFenceError(RuntimeError):
    """Raised when stale or non-owning work tries to mutate a claimed request."""


class MergeQueueIntegrityError(RuntimeError):
    """Raised when permanent queue authority is malformed or conflicting."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise MergeQueueIntegrityError(
            "post-merge denial record is not canonical JSON"
        ) from exc
    if len(encoded.encode("utf-8")) > _MAX_DENIAL_RECORD_BYTES:
        raise MergeQueueIntegrityError(
            "post-merge denial record exceeds its persistence bound"
        )
    return encoded


def _post_merge_review_terminal_key_material(
    record: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "target_repository_id": str(
            record.get("target_repository_id") or ""
        ),
        "target_branch": str(record.get("target_branch") or ""),
        "task_id": str(record.get("task_id") or ""),
        "canonical_task_key": str(
            record.get("canonical_task_key") or ""
        ),
        "canonical_task_cid": str(
            record.get("canonical_task_cid") or ""
        ),
        "task_binding_id": str(record.get("task_binding_id") or ""),
        "implementation_commit": str(
            record.get("implementation_commit") or ""
        ),
    }


def _validated_post_merge_review_denial(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return one exact, content-addressed terminal denial record."""

    if not isinstance(value, Mapping):
        raise MergeQueueIntegrityError(
            "post-merge denial record must be an object"
        )
    record = dict(value)
    expected_fields = {
        "schema",
        "terminal_key_id",
        "denial_id",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "review_attempt",
        "implementation_attempt",
        "target_implementation_attempt",
        "implementation_commit",
        "merge_commit",
        "repository_tree_id",
        "review_receipt_id",
        "review_request_id",
        "review_response_id",
        "diff_binding_id",
        "implementer_provenance_id",
        "correction_origin_stream_id",
        "correction_authorized",
        "decision",
        "source_finding_count",
        "included_finding_count",
        "truncated",
        "findings",
        "repository_write_authorized",
        "proof_authoritative",
        "completion_authoritative",
    }
    if set(record) != expected_fields:
        raise MergeQueueIntegrityError(
            "post-merge denial record schema fields changed"
        )
    required_text = (
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "review_receipt_id",
        "review_request_id",
        "review_response_id",
        "diff_binding_id",
        "implementer_provenance_id",
        "correction_origin_stream_id",
    )
    if (
        record.get("schema")
        != POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
        or record.get("decision") != "changes_required"
        or any(not isinstance(record.get(name), str) or not record[name]
               for name in required_text)
        or not _FULL_GIT_OBJECT_ID.fullmatch(
            str(record.get("implementation_commit") or "")
        )
        or not _FULL_GIT_OBJECT_ID.fullmatch(
            str(record.get("merge_commit") or "")
        )
        or not _GIT_TREE_ID.fullmatch(
            str(record.get("repository_tree_id") or "")
        )
        or record.get("repository_write_authorized") is not False
        or record.get("proof_authoritative") is not False
        or record.get("completion_authoritative") is not False
        or not isinstance(record.get("correction_authorized"), bool)
        or not isinstance(record.get("truncated"), bool)
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial record binding is invalid"
        )
    integer_fields = (
        "review_attempt",
        "implementation_attempt",
        "target_implementation_attempt",
        "source_finding_count",
        "included_finding_count",
    )
    for name in integer_fields:
        item = record.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise MergeQueueIntegrityError(
                f"post-merge denial {name} must be a positive integer"
            )
    if (
        record["target_implementation_attempt"]
        != record["implementation_attempt"] + 1
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial target attempt is not exact-next"
        )
    findings = record.get("findings")
    if (
        not isinstance(findings, list)
        or not 1 <= len(findings) <= 4
        or record["included_finding_count"] != len(findings)
        or record["source_finding_count"] < len(findings)
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial finding projection is invalid"
        )
    for finding in findings:
        if (
            not isinstance(finding, Mapping)
            or set(finding)
            != {
                "finding_id",
                "source_ordinal",
                "code",
                "severity",
                "summary",
            }
        ):
            raise MergeQueueIntegrityError(
                "post-merge denial finding schema is invalid"
            )
        material = dict(finding)
        finding_id = str(material.pop("finding_id", "") or "")
        if (
            finding_id != content_identity(material)
            or isinstance(finding.get("source_ordinal"), bool)
            or not isinstance(finding.get("source_ordinal"), int)
            or int(finding["source_ordinal"]) < 1
            or finding.get("severity")
            not in {"blocker", "high", "medium", "low", "info"}
            or not isinstance(finding.get("code"), str)
            or not finding["code"]
            or not isinstance(finding.get("summary"), str)
            or not finding["summary"]
        ):
            raise MergeQueueIntegrityError(
                "post-merge denial finding identity is invalid"
            )
    terminal_key_id = str(record.get("terminal_key_id") or "")
    if terminal_key_id != content_identity(
        _post_merge_review_terminal_key_material(record)
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial terminal key identity is invalid"
        )
    denial_material = dict(record)
    denial_id = str(denial_material.pop("denial_id", "") or "")
    if denial_id != content_identity(denial_material):
        raise MergeQueueIntegrityError(
            "post-merge denial content identity is invalid"
        )
    return record, _canonical_json(record)


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
            schema_sql="""
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
                    finished_at DOUBLE NOT NULL DEFAULT 0,
                    updated_at DOUBLE NOT NULL
                );
                ALTER TABLE merge_requests
                  ADD COLUMN IF NOT EXISTS claim_token TEXT DEFAULT '';
                ALTER TABLE merge_requests
                  ADD COLUMN IF NOT EXISTS claim_generation BIGINT DEFAULT 0;
                UPDATE merge_requests
                  SET claim_token=COALESCE(claim_token, ''),
                      claim_generation=COALESCE(claim_generation, 0)
                  WHERE claim_token IS NULL OR claim_generation IS NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS merge_requests_dedupe
                  ON merge_requests(dedupe_key);
                CREATE INDEX IF NOT EXISTS merge_requests_stage_order
                  ON merge_requests(status, enqueued_at);
                CREATE TABLE IF NOT EXISTS post_merge_review_denials (
                    terminal_key_id TEXT PRIMARY KEY,
                    denial_id TEXT NOT NULL UNIQUE,
                    target_repository_id TEXT NOT NULL,
                    target_branch TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    canonical_task_key TEXT NOT NULL,
                    canonical_task_cid TEXT NOT NULL,
                    task_binding_id TEXT NOT NULL,
                    implementation_commit TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at DOUBLE NOT NULL
                );
                CREATE INDEX IF NOT EXISTS post_merge_review_denials_target
                  ON post_merge_review_denials(
                    target_repository_id,
                    target_branch,
                    task_id
                  );
                """,
        )

    def record_post_merge_review_denial(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit one permanent, exact-candidate review-denial tombstone."""

        record, canonical = _validated_post_merge_review_denial(value)
        if (
            self.target_repository_id
            and (
                record["target_repository_id"]
                != self.target_repository_id
                or record["target_branch"] != self.target_branch
            )
        ):
            raise MergeQueueFenceError(
                "post-merge denial target differs from queue binding"
            )
        if self.require_target_binding and not self.target_repository_id:
            raise MergeQueueFenceError(
                "bound merge queue lacks a target for denial authority"
            )
        terminal_key_id = str(record["terminal_key_id"])
        denial_id = str(record["denial_id"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    """SELECT terminal_key_id, denial_id, record_json
                       FROM post_merge_review_denials
                       WHERE terminal_key_id=? OR denial_id=?""",
                    (terminal_key_id, denial_id),
                ).fetchall()
                if existing:
                    if len(existing) != 1:
                        raise MergeQueueIntegrityError(
                            "conflicting post-merge denial authority exists"
                        )
                    try:
                        existing_decoded = json.loads(
                            str(existing[0]["record_json"])
                        )
                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise MergeQueueIntegrityError(
                            "existing post-merge denial authority is malformed"
                        ) from exc
                    existing_record, existing_canonical = (
                        _validated_post_merge_review_denial(
                            existing_decoded
                        )
                    )
                    if (
                        str(existing[0]["record_json"])
                        != existing_canonical
                        or str(existing[0]["terminal_key_id"])
                        != str(existing_record["terminal_key_id"])
                        or str(existing[0]["denial_id"])
                        != str(existing_record["denial_id"])
                    ):
                        raise MergeQueueIntegrityError(
                            "existing post-merge denial authority changed"
                        )
                    if (
                        str(existing_record["terminal_key_id"])
                        != terminal_key_id
                    ):
                        raise MergeQueueIntegrityError(
                            "post-merge denial identity crosses terminal keys"
                        )
                    if (
                        str(existing[0]["denial_id"]) == denial_id
                        and existing_canonical == canonical
                    ):
                        connection.commit()
                        return existing_record
                    candidates: list[
                        tuple[tuple[int, str], dict[str, Any]]
                    ] = []
                    for candidate in (existing_record, record):
                        representative = dict(candidate)
                        representative.pop("denial_id", None)
                        representative.pop(
                            "correction_authorized",
                            None,
                        )
                        # Same-terminal records can legitimately differ after
                        # another lane reviewed the same immutable
                        # implementation against a later target HEAD. Prefer
                        # the strictly authorized origin record, then converge
                        # ties by canonical content regardless of migration
                        # order. Authorization remains attached to its own
                        # verified origin payload.
                        candidates.append(
                            (
                                (
                                    0
                                    if candidate[
                                        "correction_authorized"
                                    ]
                                    else 1,
                                    _canonical_json(representative),
                                ),
                                dict(candidate),
                            )
                        )
                    selected = dict(min(candidates, key=lambda item: item[0])[1])
                    selected.pop("denial_id", None)
                    selected["denial_id"] = content_identity(selected)
                    selected_record, selected_canonical = (
                        _validated_post_merge_review_denial(selected)
                    )
                    if selected_canonical == existing_canonical:
                        connection.commit()
                        return existing_record
                    connection.execute(
                        """UPDATE post_merge_review_denials
                           SET denial_id=?, record_json=?
                           WHERE terminal_key_id=?""",
                        (
                            selected_record["denial_id"],
                            selected_canonical,
                            terminal_key_id,
                        ),
                    )
                    connection.commit()
                    return selected_record
                connection.execute(
                    """INSERT INTO post_merge_review_denials (
                         terminal_key_id, denial_id,
                         target_repository_id, target_branch, task_id,
                         canonical_task_key, canonical_task_cid,
                         task_binding_id, implementation_commit,
                         record_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        terminal_key_id,
                        denial_id,
                        record["target_repository_id"],
                        record["target_branch"],
                        record["task_id"],
                        record["canonical_task_key"],
                        record["canonical_task_cid"],
                        record["task_binding_id"],
                        record["implementation_commit"],
                        canonical,
                        self._clock(),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return record

    def verified_post_merge_review_denials(
        self,
    ) -> tuple[dict[str, Any], ...]:
        """Return all permanent denial tombstones or fail on any corruption."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT terminal_key_id, denial_id,
                          target_repository_id, target_branch, task_id,
                          canonical_task_key, canonical_task_cid,
                          task_binding_id, implementation_commit,
                          record_json
                   FROM post_merge_review_denials
                   ORDER BY created_at, terminal_key_id"""
            ).fetchall()
        verified: list[dict[str, Any]] = []
        for row in rows:
            try:
                decoded = json.loads(str(row["record_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise MergeQueueIntegrityError(
                    "post-merge denial registry contains malformed JSON"
                ) from exc
            record, canonical = _validated_post_merge_review_denial(decoded)
            row_bindings = {
                "terminal_key_id": str(row["terminal_key_id"]),
                "denial_id": str(row["denial_id"]),
                "target_repository_id": str(
                    row["target_repository_id"]
                ),
                "target_branch": str(row["target_branch"]),
                "task_id": str(row["task_id"]),
                "canonical_task_key": str(
                    row["canonical_task_key"]
                ),
                "canonical_task_cid": str(
                    row["canonical_task_cid"]
                ),
                "task_binding_id": str(row["task_binding_id"]),
                "implementation_commit": str(
                    row["implementation_commit"]
                ),
            }
            if (
                canonical != str(row["record_json"])
                or any(record[name] != item for name, item in row_bindings.items())
            ):
                raise MergeQueueIntegrityError(
                    "post-merge denial registry row binding changed"
                )
            if (
                self.target_repository_id
                and (
                    record["target_repository_id"]
                    != self.target_repository_id
                    or record["target_branch"] != self.target_branch
                )
            ):
                continue
            verified.append(record)
        return tuple(verified)

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
            connection.execute("BEGIN IMMEDIATE")
            try:
                for status, directory in stage_dirs:
                    for path in directory.glob("*.json"):
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            request = MergeRequest.from_dict(payload, file_path=path)
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            continue
                        if not request.request_id:
                            continue
                        request = replace(request, status=status)
                        self._insert(connection, request, ignore=True)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _insert(
        self,
        connection: DuckDBConnection,
        request: MergeRequest,
        *,
        ignore: bool,
    ) -> None:
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        connection.execute(
            f"""{verb} INTO merge_requests (
                request_id, branch_name, task_id, priority, lane_id, enqueued_at,
                attempt, metadata_json, commit_sha, canonical_task_id,
                canonical_task_key, dedupe_key, status, claimed_at, consumer_id,
                failure_count, failure_reason, claim_token, claim_generation,
                finished_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                0.0,
                self._clock(),
            ),
        )

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
                    row = connection.execute(
                        "SELECT * FROM merge_requests WHERE dedupe_key = ?",
                        (request.dedupe_key,),
                    ).fetchone()
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
                self._insert(connection, request, ignore=False)
                connection.commit()
            except Exception:
                connection.rollback()
                if not request.dedupe_key:
                    raise
                row = connection.execute(
                    "SELECT * FROM merge_requests WHERE dedupe_key = ?", (request.dedupe_key,)
                ).fetchone()
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

    def dequeue(self, consumer_id: str = "") -> Optional[MergeRequest]:
        """Atomically claim the fairest pending request for one consumer."""

        claimed = self.dequeue_many(1, consumer_id=consumer_id)
        return claimed[0] if claimed else None

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
                    "SELECT * FROM merge_requests WHERE status = 'pending'"
                ).fetchall()
                if self.target_repository_id or self.require_target_binding:
                    rows = [
                        row
                        for row in rows
                        if self._metadata_matches_target(row["metadata_json"])
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
                               updated_at=?
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
                   claim_token='', claim_generation=claim_generation + 1
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
                   finished_at=?, updated_at=? WHERE request_id=?
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
                   finished_at=?, updated_at=? WHERE request_id=?""",
                (
                    max(1, int(row["failure_count"])),
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
                   claim_generation=claim_generation + 1, finished_at=?, updated_at=?
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

    @staticmethod
    def _normalized_submodule_recovery_targets(
        request_metadata: Mapping[str, Any],
        approved_submodule_integrations: Mapping[str, str],
    ) -> list[dict[str, str]]:
        """Return exact task-bound child postimages or reject the whole grant."""

        raw_task_binding = request_metadata.get(
            "task_owned_submodule_integration_binding"
        )
        raw_targets = (
            raw_task_binding.get("targets")
            if isinstance(raw_task_binding, Mapping)
            else None
        )
        bound_paths = {
            str(target.get("path") or "").strip("/")
            for target in raw_targets or ()
            if isinstance(target, Mapping)
        }
        if (
            not isinstance(approved_submodule_integrations, Mapping)
            or not approved_submodule_integrations
            or len(approved_submodule_integrations) > 256
        ):
            raise ValueError(
                "approved submodule integrations must be a non-empty mapping"
            )
        normalized_targets: list[dict[str, str]] = []
        for raw_path, raw_commit in sorted(
            approved_submodule_integrations.items(),
            key=lambda item: str(item[0]),
        ):
            path = str(raw_path or "").strip("/")
            commit = str(raw_commit or "").strip()
            path_parts = Path(path).parts
            if (
                not path
                or path != str(raw_path or "")
                or path not in bound_paths
                or Path(path).is_absolute()
                or any(part in {"", ".", ".."} for part in path_parts)
            ):
                raise ValueError(
                    f"submodule recovery path is not task-bound: {raw_path!r}"
                )
            if (
                len(commit) not in {40, 64}
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in commit
                )
            ):
                raise ValueError(
                    f"submodule recovery commit is invalid for {path}"
                )
            normalized_targets.append(
                {
                    "path": path,
                    "integrated_target": commit.lower(),
                }
            )
        return normalized_targets

    def revive_quarantined(
        self,
        request: MergeRequest | str,
        reason: str = "",
        *,
        reset_failures: bool = False,
        approved_submodule_integrations: Mapping[str, str] | None = None,
    ) -> MergeRequest | None:
        """Return a quarantined request to pending after operator review.

        The operation is atomic and idempotent.  A revival record is retained
        in request metadata so administrative recovery does not erase why the
        candidate was quarantined.  ``reset_failures`` is intended for false
        positives such as a host suspension while a request was still pending.
        ``approved_submodule_integrations`` binds recovery to exact reviewed
        child postimages; it never authorizes an arbitrary descendant.
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
            status = str(row["status"])
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if status != "quarantined":
                if (
                    status in _ACTIVE_STATES
                    and approved_submodule_integrations is not None
                ):
                    if not str(reason or "").strip():
                        connection.rollback()
                        raise ValueError(
                            "submodule integration recovery requires an operator reason"
                        )
                    normalized_targets = (
                        self._normalized_submodule_recovery_targets(
                            request_metadata,
                            approved_submodule_integrations,
                        )
                    )
                    existing = request_metadata.get(
                        "operator_submodule_integration_recovery"
                    )
                    row_generation = int(row["claim_generation"] or 0)
                    existing_generation = (
                        existing.get(
                            "revival_generation"
                            if status == "pending"
                            else "claim_generation"
                        )
                        if isinstance(existing, Mapping)
                        else None
                    )
                    same_active_grant = bool(
                        isinstance(existing, Mapping)
                        and existing.get("schema")
                        == SUBMODULE_INTEGRATION_RECOVERY_SCHEMA
                        and existing.get("request_id") == request_id
                        and existing.get("implementation_commit")
                        == str(row["commit_sha"] or "")
                        and existing.get("target_repository_id")
                        == str(
                            request_metadata.get("target_repository_id")
                            or ""
                        ).strip()
                        and existing.get("target_branch")
                        == str(
                            request_metadata.get("target_branch") or ""
                        ).strip()
                        and existing.get("targets") == normalized_targets
                        and not isinstance(existing_generation, bool)
                        and isinstance(existing_generation, int)
                        and existing_generation == row_generation
                    )
                    if not same_active_grant:
                        connection.rollback()
                        raise MergeQueueFenceError(
                            f"revive rejected for active request {request_id}: "
                            "submodule recovery approval differs from the "
                            "current generation-bound grant"
                        )
                connection.commit()
                return self._request_from_row(row)

            # Recovery is a single-revival capability. Preserve its audit copy
            # in ``revivals`` but never carry the live top-level grant into a
            # later quarantine/revival cycle.
            request_metadata.pop(
                "operator_submodule_integration_recovery",
                None,
            )
            recovery_binding: dict[str, Any] | None = None
            if approved_submodule_integrations is not None:
                if not str(reason or "").strip():
                    connection.rollback()
                    raise ValueError(
                        "submodule integration recovery requires an operator reason"
                    )
                try:
                    normalized_targets = (
                        self._normalized_submodule_recovery_targets(
                            request_metadata,
                            approved_submodule_integrations,
                        )
                    )
                except ValueError:
                    connection.rollback()
                    raise
                quarantine_generation = int(row["claim_generation"] or 0)
                revival_generation = quarantine_generation + 1
                recovery_binding = {
                    "schema": SUBMODULE_INTEGRATION_RECOVERY_SCHEMA,
                    "approved_at": now,
                    "reason": str(reason).strip(),
                    "request_id": request_id,
                    "implementation_commit": str(row["commit_sha"] or ""),
                    "target_repository_id": str(
                        request_metadata.get("target_repository_id") or ""
                    ).strip(),
                    "target_branch": str(
                        request_metadata.get("target_branch") or ""
                    ).strip(),
                    "quarantine_generation": quarantine_generation,
                    "revival_generation": revival_generation,
                    "claim_generation": revival_generation + 1,
                    "targets": normalized_targets,
                }
                request_metadata[
                    "operator_submodule_integration_recovery"
                ] = recovery_binding
            revival = {
                "at": now,
                "reason": str(reason),
                "previous_enqueued_at": float(row["enqueued_at"]),
                "previous_failure_count": int(row["failure_count"]),
                "previous_failure_reason": str(row["failure_reason"]),
            }
            if recovery_binding is not None:
                revival["submodule_integration_recovery"] = recovery_binding
            request_metadata.setdefault("revivals", []).append(revival)
            failure_count = 0 if reset_failures else int(row["failure_count"])
            attempt = 1 if reset_failures else int(row["attempt"])
            connection.execute(
                """UPDATE merge_requests SET status='pending', enqueued_at=?, attempt=?,
                   failure_count=?, failure_reason='', metadata_json=?, claimed_at=0,
                   consumer_id='', claim_token='',
                   claim_generation=claim_generation + 1,
                   finished_at=0, updated_at=? WHERE request_id=?""",
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
                    failure_count = max(1, failure_count)
                    reason = "processing request exceeded max age"
                    finished_at = now
                connection.execute(
                    """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                       failure_reason=?, claimed_at=0, consumer_id='', claim_token='',
                       claim_generation=claim_generation + 1, finished_at=?,
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
                       claim_generation=claim_generation + 1, finished_at=?,
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
    "MergeQueue",
    "MergeQueueFullError",
    "MergeQueueFenceError",
    "MergeRequest",
    "_PRIORITY_ORDER",
]
