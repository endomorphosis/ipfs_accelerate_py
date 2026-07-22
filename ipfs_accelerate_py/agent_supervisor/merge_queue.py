"""Durable, deduplicating merge queue for implementation lanes.

The queue is deliberately process safe.  Producers may be independent daemon
processes, but only one consumer can atomically claim a request.  SQLite is the
authoritative index and small JSON files are retained as human-readable stage
receipts.  A request is idempotent when both its canonical task identity and
source commit match an existing request, including a completed or quarantined
request.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


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


class MergeQueueFullError(RuntimeError):
    """Raised when accepting another active request would exceed queue capacity."""


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

    @property
    def canonical_identity(self) -> str:
        """Return the strongest task identity supplied by the producer."""

        return self.canonical_task_key or self.canonical_task_id or self.task_id

    @property
    def dedupe_key(self) -> str:
        """Return the stable task-and-commit idempotency key, when available."""

        if not self.commit_sha:
            return ""
        identity = self.canonical_identity.strip().casefold()
        commit = self.commit_sha.strip().casefold()
        return hashlib.sha256(f"{identity}\0{commit}".encode("utf-8")).hexdigest()

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
    """SQLite-backed priority queue with atomic claims and bounded retries.

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
        priority_aging_seconds: float = 300,
        max_attempts: int = 3,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.processing_dir = self.queue_dir / "processing"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"  # compatibility projection
        self.quarantine_dir = self.queue_dir / "quarantine"
        self.database_path = self.queue_dir / "merge_queue.sqlite3"
        self.max_age_seconds = max(0.0, float(max_age_seconds))
        self.max_queue_size = max(1, int(max_queue_size))
        self.priority_aging_seconds = max(0.0, float(priority_aging_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self._clock = clock or time.time
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.quarantine_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._import_legacy_files()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS merge_requests (
                    request_id TEXT PRIMARY KEY,
                    branch_name TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    enqueued_at REAL NOT NULL,
                    attempt INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    canonical_task_id TEXT NOT NULL,
                    canonical_task_key TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    claimed_at REAL NOT NULL DEFAULT 0,
                    consumer_id TEXT NOT NULL DEFAULT '',
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT NOT NULL DEFAULT '',
                    finished_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS merge_requests_dedupe
                  ON merge_requests(dedupe_key) WHERE dedupe_key <> '';
                CREATE INDEX IF NOT EXISTS merge_requests_stage_order
                  ON merge_requests(status, enqueued_at);
                """
            )

    def _import_legacy_files(self) -> None:
        """Import pre-SQLite queue files once, preserving their original stage."""

        stage_dirs = (
            ("pending", self.pending_dir),
            ("processing", self.processing_dir),
            ("completed", self.completed_dir),
            ("quarantined", self.failed_dir),
            ("quarantined", self.quarantine_dir),
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

    def _insert(self, connection: sqlite3.Connection, request: MergeRequest, *, ignore: bool) -> None:
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        connection.execute(
            f"""{verb} INTO merge_requests (
                request_id, branch_name, task_id, priority, lane_id, enqueued_at,
                attempt, metadata_json, commit_sha, canonical_task_id,
                canonical_task_key, dedupe_key, status, claimed_at, consumer_id,
                failure_count, failure_reason, finished_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                request.dedupe_key,
                request.status,
                request.claimed_at,
                request.consumer_id,
                request.failure_count,
                request.failure_reason,
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
    ) -> MergeRequest:
        """Atomically enqueue or return the existing task-and-commit request."""

        if not str(branch_name).strip():
            raise ValueError("branch_name must not be empty")
        if not str(task_id).strip():
            raise ValueError("task_id must not be empty")
        metadata_dict = dict(metadata or {})
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
                active_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM merge_requests WHERE status IN ('pending','processing')"
                    ).fetchone()[0]
                )
                if active_count >= self.max_queue_size:
                    connection.rollback()
                    raise MergeQueueFullError(
                        f"merge queue capacity {self.max_queue_size} has been reached"
                    )
                self._insert(connection, request, ignore=False)
                connection.commit()
            except sqlite3.IntegrityError:
                connection.rollback()
                if not request.dedupe_key:
                    raise
                row = connection.execute(
                    "SELECT * FROM merge_requests WHERE dedupe_key = ?", (request.dedupe_key,)
                ).fetchone()
                if row is None:
                    raise
                return self._request_from_row(row)
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
        receipt_path = self._write_stage_receipt(request)
        return replace(request, file_path=receipt_path)

    def dequeue(self, consumer_id: str = "") -> Optional[MergeRequest]:
        """Atomically claim the fairest pending request for one consumer."""

        self._purge_stale()
        consumer = str(consumer_id or os.getpid())
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT * FROM merge_requests WHERE status = 'pending'"
                ).fetchall()
                if not rows:
                    connection.commit()
                    return None
                row = min(rows, key=lambda item: self._fairness_key(item, now))
                updated = connection.execute(
                    """UPDATE merge_requests
                       SET status='processing', claimed_at=?, consumer_id=?, updated_at=?
                       WHERE request_id=? AND status='pending'""",
                    (now, consumer, now, row["request_id"]),
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    return None
                row = connection.execute(
                    "SELECT * FROM merge_requests WHERE request_id=?", (row["request_id"],)
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        assert row is not None
        claimed = self._request_from_row(row)
        receipt_path = self._write_stage_receipt(claimed)
        return replace(claimed, file_path=receipt_path)

    def _fairness_key(self, row: sqlite3.Row, now: float) -> tuple[int, float, str]:
        base = _PRIORITY_ORDER.get(str(row["priority"]), _PRIORITY_ORDER["P2"])
        if self.priority_aging_seconds > 0:
            promotions = int(max(0.0, now - float(row["enqueued_at"])) / self.priority_aging_seconds)
            effective = max(0, base - promotions)
        else:
            effective = base
        return effective, float(row["enqueued_at"]), str(row["request_id"])

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
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata["completion"] = dict(metadata)
            connection.execute(
                """UPDATE merge_requests SET status='completed', metadata_json=?,
                   finished_at=?, updated_at=?, consumer_id='', claimed_at=0
                   WHERE request_id=?""",
                (
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
            if str(row["status"]) in {"completed", "quarantined"}:
                connection.commit()
                resolved = self._request_from_row(row)
                if resolved.status == "quarantined":
                    return self._stage_path(resolved)
                return resolved
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
                   finished_at=?, updated_at=? WHERE request_id=?""",
                (
                    status,
                    next_attempt,
                    failure_count,
                    str(reason),
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now if terminal else 0.0,
                    now,
                    request.request_id,
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
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata["quarantine"] = dict(metadata)
            connection.execute(
                """UPDATE merge_requests SET status='quarantined', failure_count=?,
                   failure_reason=?, metadata_json=?, claimed_at=0, consumer_id='',
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
                f"""SELECT DISTINCT canonical_task_id
                    FROM merge_requests
                    WHERE status IN ({placeholders}) AND canonical_task_id != ''""",
                normalized,
            ).fetchall()
        return {str(row["canonical_task_id"]) for row in rows}

    def pending_count(self) -> int:
        return self._count("pending")

    def processing_count(self) -> int:
        return self._count("processing")

    def _count(self, status: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM merge_requests WHERE status=?", (status,)
                ).fetchone()[0]
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
                """SELECT task_id, canonical_task_id, canonical_task_key, commit_sha
                   FROM merge_requests WHERE status IN ('pending','processing')"""
            ).fetchall()
        for row in rows:
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
        """Recover abandoned claims and quarantine requests beyond their bounds."""

        if self.max_age_seconds <= 0:
            return 0
        now = self._clock()
        changed: list[MergeRequest] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM merge_requests WHERE status IN ('pending','processing')"
            ).fetchall()
            for row in rows:
                status = str(row["status"])
                reference_time = float(row["claimed_at"] or row["enqueued_at"])
                if now - reference_time <= self.max_age_seconds:
                    continue
                attempt = int(row["attempt"])
                failure_count = int(row["failure_count"])
                if status == "processing" and attempt < self.max_attempts:
                    new_status = "pending"
                    new_attempt = attempt + 1
                    failure_count += 1
                    reason = "consumer claim expired; request recovered"
                    finished_at = 0.0
                else:
                    new_status = "quarantined"
                    new_attempt = attempt
                    failure_count = max(1, failure_count)
                    reason = f"{status} request exceeded max age"
                    finished_at = now
                connection.execute(
                    """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                       failure_reason=?, claimed_at=0, consumer_id='', finished_at=?,
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
                       failure_reason=?, claimed_at=0, consumer_id='', finished_at=?,
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
            counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM merge_requests GROUP BY status"
                ).fetchall()
            }
        return {
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("quarantined", 0),
            "quarantined": counts.get("quarantined", 0),
            "total": sum(counts.values()),
            "queue_dir": str(self.queue_dir),
            "database_path": str(self.database_path),
            "max_attempts": self.max_attempts,
        }

    def _request_from_row(self, row: sqlite3.Row) -> MergeRequest:
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
        }
        request = MergeRequest.from_dict(payload)
        return replace(request, file_path=self._stage_path(request))

    def _stage_path(self, request: MergeRequest) -> Path:
        stage_dir = {
            "pending": self.pending_dir,
            "processing": self.processing_dir,
            "completed": self.completed_dir,
            "quarantined": self.quarantine_dir,
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
        _atomic_write_json(destination, payload)
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.quarantine_dir,
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
    "MergeRequest",
    "_PRIORITY_ORDER",
]
