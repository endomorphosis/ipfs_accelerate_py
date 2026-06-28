"""FIFO merge queue for multi-lane supervisor coordination.

Instead of all lanes competing for a single lock simultaneously (causing thrashing),
lanes enqueue their merge requests and a single consumer processes them in order.

This eliminates the thundering herd problem where N lanes all wake up and
try to acquire the merge lock at the same time after each iteration.

Architecture:
    - Each lane writes a merge request file to a shared queue directory
    - Requests are named with timestamps for natural FIFO ordering
    - The merge consumer (whichever lane holds the lock) drains pending requests
    - Lanes that don't hold the lock simply enqueue and move on
    - Stale requests (older than max_age) are automatically purged

Usage:
    queue = MergeQueue(queue_dir=state_root / "merge_queue", repo_root=repo_root)

    # Lane enqueues a merge request
    queue.enqueue(branch_name="impl/task-42-attempt-1", task_id="PORTAL-042", priority="P1")

    # Merge consumer processes next request
    request = queue.dequeue()
    if request:
        # ... perform merge ...
        queue.complete(request)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class MergeRequest:
    """A queued merge request from a lane."""

    request_id: str
    branch_name: str
    task_id: str
    priority: str
    lane_id: str
    enqueued_at: float
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "branch_name": self.branch_name,
            "task_id": self.task_id,
            "priority": self.priority,
            "lane_id": self.lane_id,
            "enqueued_at": self.enqueued_at,
            "attempt": self.attempt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, file_path: Optional[Path] = None) -> "MergeRequest":
        return cls(
            request_id=str(data.get("request_id", "")),
            branch_name=str(data.get("branch_name", "")),
            task_id=str(data.get("task_id", "")),
            priority=str(data.get("priority", "P2")),
            lane_id=str(data.get("lane_id", "")),
            enqueued_at=float(data.get("enqueued_at", 0.0)),
            attempt=int(data.get("attempt", 1)),
            metadata=dict(data.get("metadata", {})),
            file_path=file_path,
        )


# Priority ordering for merge queue (lower = higher priority)
_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


class MergeQueue:
    """File-based FIFO merge queue with priority support.

    Merge requests are stored as individual JSON files in a directory.
    This avoids the need for a database or message broker while still
    providing atomic enqueue/dequeue semantics via filesystem operations.
    """

    def __init__(
        self,
        queue_dir: Path,
        *,
        max_age_seconds: float = 3600,
        max_queue_size: int = 100,
    ) -> None:
        self.queue_dir = queue_dir
        self.pending_dir = queue_dir / "pending"
        self.processing_dir = queue_dir / "processing"
        self.completed_dir = queue_dir / "completed"
        self.failed_dir = queue_dir / "failed"
        self.max_age_seconds = max_age_seconds
        self.max_queue_size = max_queue_size

        # Ensure directories exist
        for d in (self.pending_dir, self.processing_dir, self.completed_dir, self.failed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        *,
        branch_name: str,
        task_id: str,
        priority: str = "P2",
        lane_id: str = "",
        attempt: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> MergeRequest:
        """Add a merge request to the queue."""
        now = time.time()
        # Generate unique request ID using timestamp + PID for uniqueness
        request_id = f"{now:.6f}-{os.getpid()}-{task_id}"

        request = MergeRequest(
            request_id=request_id,
            branch_name=branch_name,
            task_id=task_id,
            priority=priority,
            lane_id=lane_id or str(os.getpid()),
            enqueued_at=now,
            attempt=attempt,
            metadata=metadata or {},
        )

        # Write atomically
        file_name = f"{_PRIORITY_ORDER.get(priority, 9):01d}-{now:.6f}-{task_id}.json"
        file_path = self.pending_dir / file_name
        tmp_path = file_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(request.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp_path.rename(file_path)

        return request

    def dequeue(self) -> Optional[MergeRequest]:
        """Get the next merge request (highest priority, oldest first).

        Moves the request from pending to processing directory.
        Returns None if queue is empty.
        """
        self._purge_stale()

        # List pending requests sorted by filename (priority prefix + timestamp)
        pending = sorted(self.pending_dir.glob("*.json"))
        if not pending:
            return None

        # Try to claim the first one (atomic rename)
        for request_file in pending:
            processing_path = self.processing_dir / request_file.name
            try:
                request_file.rename(processing_path)
            except (FileNotFoundError, OSError):
                # Another lane claimed it first
                continue

            try:
                data = json.loads(processing_path.read_text(encoding="utf-8"))
                return MergeRequest.from_dict(data, file_path=processing_path)
            except (json.JSONDecodeError, OSError):
                # Corrupted request - move to failed
                try:
                    processing_path.rename(self.failed_dir / request_file.name)
                except OSError:
                    pass
                continue

        return None

    def complete(self, request: MergeRequest) -> None:
        """Mark a merge request as completed."""
        if request.file_path and request.file_path.exists():
            completed_path = self.completed_dir / request.file_path.name
            try:
                request.file_path.rename(completed_path)
            except OSError:
                pass

        # Prune old completed entries (keep last 50)
        completed = sorted(self.completed_dir.glob("*.json"))
        for old in completed[:-50]:
            try:
                old.unlink()
            except OSError:
                pass

    def fail(self, request: MergeRequest, reason: str = "") -> None:
        """Mark a merge request as failed."""
        if request.file_path and request.file_path.exists():
            # Add failure info
            try:
                data = json.loads(request.file_path.read_text(encoding="utf-8"))
                data["failure_reason"] = reason
                data["failed_at"] = time.time()
                failed_path = self.failed_dir / request.file_path.name
                failed_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                request.file_path.unlink()
            except OSError:
                try:
                    request.file_path.rename(self.failed_dir / request.file_path.name)
                except OSError:
                    pass

    def requeue(self, request: MergeRequest) -> None:
        """Put a failed request back in the queue for retry."""
        if request.file_path and request.file_path.exists():
            pending_path = self.pending_dir / request.file_path.name
            try:
                request.file_path.rename(pending_path)
            except OSError:
                pass

    def pending_count(self) -> int:
        """Return number of pending merge requests."""
        return len(list(self.pending_dir.glob("*.json")))

    def processing_count(self) -> int:
        """Return number of currently processing requests."""
        return len(list(self.processing_dir.glob("*.json")))

    def has_pending_for_task(self, task_id: str) -> bool:
        """Check if a merge is already queued for this task."""
        for f in self.pending_dir.glob("*.json"):
            if task_id in f.name:
                return True
        for f in self.processing_dir.glob("*.json"):
            if task_id in f.name:
                return True
        return False

    def _purge_stale(self) -> int:
        """Remove requests older than max_age_seconds."""
        now = time.time()
        purged = 0

        # Purge stale pending requests
        for f in self.pending_dir.glob("*.json"):
            try:
                stat = f.stat()
                if now - stat.st_mtime > self.max_age_seconds:
                    f.rename(self.failed_dir / f.name)
                    purged += 1
            except OSError:
                pass

        # Purge stale processing requests (likely from crashed lanes)
        for f in self.processing_dir.glob("*.json"):
            try:
                stat = f.stat()
                if now - stat.st_mtime > self.max_age_seconds:
                    # Requeue or fail based on age
                    if now - stat.st_mtime > self.max_age_seconds * 2:
                        f.rename(self.failed_dir / f.name)
                    else:
                        f.rename(self.pending_dir / f.name)
                    purged += 1
            except OSError:
                pass

        # Enforce max queue size
        pending = sorted(self.pending_dir.glob("*.json"))
        if len(pending) > self.max_queue_size:
            for excess in pending[self.max_queue_size:]:
                try:
                    excess.rename(self.failed_dir / excess.name)
                    purged += 1
                except OSError:
                    pass

        return purged

    def status(self) -> dict[str, Any]:
        """Return queue status summary."""
        return {
            "pending": self.pending_count(),
            "processing": self.processing_count(),
            "completed": len(list(self.completed_dir.glob("*.json"))),
            "failed": len(list(self.failed_dir.glob("*.json"))),
            "queue_dir": str(self.queue_dir),
        }
