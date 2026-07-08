"""Persistent priority queue for task selection state.

Survives supervisor restarts without losing task ordering, penalties, or
selection history. Stored as a compact JSON file alongside the daemon state.

Key features:
- Persists selection penalties (merge failures, no-change outcomes) across restarts
- Tracks attempt counts per task to avoid repeating failed tasks indefinitely
- Stores last-selected timestamp to implement fair round-robin within priority tiers
- Auto-compacts stale entries for completed tasks
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskQueueEntry:
    """Priority queue entry for a single task."""

    task_id: str
    priority: str = "P2"
    track: str = ""
    selection_penalty: int = 0
    attempt_count: int = 0
    last_selected_at: float = 0.0
    last_completed_at: float = 0.0
    consecutive_failures: int = 0
    consecutive_no_change: int = 0
    merge_failure_count: int = 0
    cooldown_until: float = 0.0
    notes: str = ""

    def record_selection(self) -> None:
        self.last_selected_at = time.time()
        self.attempt_count += 1

    def record_success(self) -> None:
        self.last_completed_at = time.time()
        self.consecutive_failures = 0
        self.consecutive_no_change = 0
        self.selection_penalty = 0
        self.cooldown_until = 0.0

    def record_failure(self, reason: str = "") -> None:
        self.consecutive_failures += 1
        # Exponential cooldown: 5min * 2^(failures-1), max 4 hours
        cooldown = min(300 * (2 ** (self.consecutive_failures - 1)), 14400)
        self.cooldown_until = time.time() + cooldown
        self.selection_penalty = min(self.consecutive_failures * 100, 5000)
        self.notes = reason

    def record_no_change(self) -> None:
        self.consecutive_no_change += 1
        # Back off from tasks that produce no changes
        cooldown = min(600 * self.consecutive_no_change, 7200)
        self.cooldown_until = time.time() + cooldown

    def record_merge_failure(self) -> None:
        self.merge_failure_count += 1
        self.selection_penalty += 500

    def is_cooled_down(self) -> bool:
        """Return True if the task is still in cooldown."""
        return self.cooldown_until > time.time()

    def effective_penalty(self) -> int:
        """Return the effective selection penalty including cooldown state."""
        if self.is_cooled_down():
            return self.selection_penalty + 10000  # Very high penalty during cooldown
        return self.selection_penalty

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "priority": self.priority,
            "track": self.track,
            "selection_penalty": self.selection_penalty,
            "attempt_count": self.attempt_count,
            "last_selected_at": self.last_selected_at,
            "last_completed_at": self.last_completed_at,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_no_change": self.consecutive_no_change,
            "merge_failure_count": self.merge_failure_count,
            "cooldown_until": self.cooldown_until,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskQueueEntry":
        return cls(
            task_id=str(data.get("task_id", "")),
            priority=str(data.get("priority", "P2")),
            track=str(data.get("track", "")),
            selection_penalty=int(data.get("selection_penalty", 0)),
            attempt_count=int(data.get("attempt_count", 0)),
            last_selected_at=float(data.get("last_selected_at", 0.0)),
            last_completed_at=float(data.get("last_completed_at", 0.0)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            consecutive_no_change=int(data.get("consecutive_no_change", 0)),
            merge_failure_count=int(data.get("merge_failure_count", 0)),
            cooldown_until=float(data.get("cooldown_until", 0.0)),
            notes=str(data.get("notes", "")),
        )


@dataclass
class PersistentTaskQueue:
    """Persistent priority queue that survives restarts."""

    entries: dict[str, TaskQueueEntry] = field(default_factory=dict)
    _path: Path | None = None
    _dirty: bool = False
    _last_save_time: float = 0.0
    _save_interval: float = 30.0  # Save at most every 30 seconds

    def get_or_create(self, task_id: str, *, priority: str = "P2", track: str = "") -> TaskQueueEntry:
        if task_id not in self.entries:
            self.entries[task_id] = TaskQueueEntry(task_id=task_id, priority=priority, track=track)
            self._dirty = True
        entry = self.entries[task_id]
        if priority and entry.priority != priority:
            entry.priority = priority
            self._dirty = True
        if track and entry.track != track:
            entry.track = track
            self._dirty = True
        return entry

    def record_selection(self, task_id: str) -> None:
        entry = self.get_or_create(task_id)
        entry.record_selection()
        self._dirty = True
        self._maybe_save()

    def record_success(self, task_id: str) -> None:
        entry = self.get_or_create(task_id)
        entry.record_success()
        self._dirty = True
        self._maybe_save()

    def record_failure(self, task_id: str, reason: str = "") -> None:
        entry = self.get_or_create(task_id)
        entry.record_failure(reason)
        self._dirty = True
        self._maybe_save()

    def record_no_change(self, task_id: str) -> None:
        entry = self.get_or_create(task_id)
        entry.record_no_change()
        self._dirty = True
        self._maybe_save()

    def record_merge_failure(self, task_id: str) -> None:
        entry = self.get_or_create(task_id)
        entry.record_merge_failure()
        self._dirty = True
        self._maybe_save()

    def get_penalty(self, task_id: str) -> int:
        """Get the effective penalty for a task (used in sort key)."""
        if task_id not in self.entries:
            return 0
        return self.entries[task_id].effective_penalty()

    def is_cooled_down(self, task_id: str) -> bool:
        """Check if a task is in cooldown."""
        if task_id not in self.entries:
            return False
        return self.entries[task_id].is_cooled_down()

    def compact(self, active_task_ids: set[str]) -> int:
        """Remove entries for tasks no longer in the active backlog.

        Returns the number of entries removed.
        """
        stale_ids = [tid for tid in self.entries if tid not in active_task_ids]
        for tid in stale_ids:
            del self.entries[tid]
        if stale_ids:
            self._dirty = True
            self._maybe_save()
        return len(stale_ids)

    def summary(self) -> dict[str, Any]:
        cooled = sum(1 for e in self.entries.values() if e.is_cooled_down())
        penalized = sum(1 for e in self.entries.values() if e.selection_penalty > 0)
        return {
            "total_entries": len(self.entries),
            "cooled_down": cooled,
            "penalized": penalized,
            "total_attempts": sum(e.attempt_count for e in self.entries.values()),
            "total_failures": sum(e.consecutive_failures for e in self.entries.values()),
        }

    def _maybe_save(self) -> None:
        """Save if dirty and enough time has elapsed since last save."""
        if not self._dirty or self._path is None:
            return
        now = time.time()
        if now - self._last_save_time < self._save_interval:
            return
        self.save()

    def save(self) -> None:
        """Force save to disk."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "schema": "persistent_task_queue_v1",
                "updated_at": time.time(),
                "entry_count": len(self.entries),
                "entries": {tid: entry.to_dict() for tid, entry in self.entries.items()},
            }
            # Atomic write via temp file
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            tmp_path.replace(self._path)
            self._dirty = False
            self._last_save_time = time.time()
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path, *, save_interval: float = 30.0) -> "PersistentTaskQueue":
        queue = cls(_path=path, _save_interval=save_interval)
        if not path.exists():
            return queue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for task_id, entry_data in data.get("entries", {}).items():
                queue.entries[task_id] = TaskQueueEntry.from_dict(entry_data)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return queue
