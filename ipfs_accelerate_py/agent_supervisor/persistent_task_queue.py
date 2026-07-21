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

from .task_identity import TaskIdentity


@dataclass
class TaskQueueEntry:
    """Priority queue entry for a single task."""

    task_id: str
    priority: str = "P2"
    track: str = ""
    canonical_task_cid: str = ""
    canonical_task_key: str = ""
    aliases: list[str] = field(default_factory=list)
    provenance: list[dict[str, str]] = field(default_factory=list)
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
            "canonical_task_cid": self.canonical_task_cid,
            "canonical_task_key": self.canonical_task_key,
            "aliases": list(self.aliases),
            "provenance": list(self.provenance),
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
            canonical_task_cid=str(data.get("canonical_task_cid", "")),
            canonical_task_key=str(data.get("canonical_task_key", "")),
            aliases=[str(item) for item in data.get("aliases", []) if str(item)],
            provenance=[
                {str(key): str(value) for key, value in item.items()}
                for item in data.get("provenance", [])
                if isinstance(item, dict)
            ],
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
    aliases: dict[str, str] = field(default_factory=dict)
    _path: Path | None = None
    _dirty: bool = False
    _last_save_time: float = 0.0
    _save_interval: float = 30.0  # Save at most every 30 seconds

    @property
    def dirty(self) -> bool:
        return self._dirty

    def resolve_key(self, task_ref: str) -> str:
        if task_ref in self.entries:
            return task_ref
        return self.aliases.get(task_ref, task_ref)

    @staticmethod
    def _merge_entries(target: TaskQueueEntry, source: TaskQueueEntry) -> None:
        target.selection_penalty = max(target.selection_penalty, source.selection_penalty)
        target.attempt_count = max(target.attempt_count, source.attempt_count)
        target.last_selected_at = max(target.last_selected_at, source.last_selected_at)
        target.last_completed_at = max(target.last_completed_at, source.last_completed_at)
        target.consecutive_failures = max(target.consecutive_failures, source.consecutive_failures)
        target.consecutive_no_change = max(target.consecutive_no_change, source.consecutive_no_change)
        target.merge_failure_count = max(target.merge_failure_count, source.merge_failure_count)
        target.cooldown_until = max(target.cooldown_until, source.cooldown_until)
        target.notes = target.notes or source.notes
        target.aliases = list(dict.fromkeys([*target.aliases, *source.aliases, source.task_id]))
        for item in source.provenance:
            if item not in target.provenance:
                target.provenance.append(item)

    def register_task(
        self,
        identity: TaskIdentity,
        *,
        priority: str = "P2",
        track: str = "",
    ) -> TaskQueueEntry:
        """Register or idempotently migrate one board-local alias."""

        changed = False
        canonical_key = identity.canonical_task_cid
        candidate_keys = [
            canonical_key,
            identity.display_task_id,
        ]
        existing_keys = [key for key in dict.fromkeys(candidate_keys) if key in self.entries]
        if canonical_key in self.entries:
            entry = self.entries[canonical_key]
        elif existing_keys:
            legacy_key = existing_keys.pop(0)
            entry = self.entries.pop(legacy_key)
            self.entries[canonical_key] = entry
            changed = True
        else:
            entry = TaskQueueEntry(
                task_id=identity.display_task_id or canonical_key,
                priority=priority,
                track=track,
            )
            self.entries[canonical_key] = entry
            changed = True

        for duplicate_key in existing_keys:
            if duplicate_key == canonical_key:
                continue
            self._merge_entries(entry, self.entries.pop(duplicate_key))
            changed = True

        before_entry = entry.to_dict()
        entry.canonical_task_cid = canonical_key
        entry.canonical_task_key = identity.canonical_task_key
        if priority:
            entry.priority = priority
        if track:
            entry.track = track
        aliases = [identity.display_task_id, identity.namespaced_alias]
        for alias in aliases:
            if alias and alias not in entry.aliases:
                entry.aliases.append(alias)
        if identity.display_task_id:
            existing_target = self.aliases.get(identity.display_task_id)
            existing_entry = self.entries.get(existing_target or "")
            same_provenance = existing_entry is not None and any(
                item.get("board_namespace") == identity.board_namespace
                and item.get("display_task_id") == identity.display_task_id
                for item in existing_entry.provenance
            )
            if existing_target is None or existing_target == canonical_key or same_provenance:
                if existing_target != canonical_key:
                    self.aliases[identity.display_task_id] = canonical_key
                    changed = True
        if identity.namespaced_alias:
            if self.aliases.get(identity.namespaced_alias) != canonical_key:
                self.aliases[identity.namespaced_alias] = canonical_key
                changed = True
        provenance = {
            "board_namespace": identity.board_namespace,
            "display_task_id": identity.display_task_id,
            "source_path": identity.source_path,
        }
        if provenance not in entry.provenance:
            entry.provenance.append(provenance)
        self._dirty = self._dirty or changed or entry.to_dict() != before_entry
        return entry

    def get_or_create(self, task_id: str, *, priority: str = "P2", track: str = "") -> TaskQueueEntry:
        key = self.resolve_key(task_id)
        if key not in self.entries:
            self.entries[key] = TaskQueueEntry(task_id=task_id, priority=priority, track=track)
            self._dirty = True
        entry = self.entries[key]
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
        key = self.resolve_key(task_id)
        if key not in self.entries:
            return 0
        return self.entries[key].effective_penalty()

    def is_cooled_down(self, task_id: str) -> bool:
        """Check if a task is in cooldown."""
        key = self.resolve_key(task_id)
        if key not in self.entries:
            return False
        return self.entries[key].is_cooled_down()

    def compact(self, active_task_ids: set[str]) -> int:
        """Remove entries for tasks no longer in the active backlog.

        Returns the number of entries removed.
        """
        active_keys = {self.resolve_key(task_id) for task_id in active_task_ids}
        stale_ids = [tid for tid in self.entries if tid not in active_keys]
        for tid in stale_ids:
            del self.entries[tid]
        if stale_ids:
            stale_set = set(stale_ids)
            self.aliases = {
                alias: target
                for alias, target in self.aliases.items()
                if target not in stale_set
            }
        if stale_ids:
            self._dirty = True
            self._maybe_save()
        return len(stale_ids)

    def summary(self) -> dict[str, Any]:
        cooled = sum(1 for e in self.entries.values() if e.is_cooled_down())
        penalized = sum(1 for e in self.entries.values() if e.selection_penalty > 0)
        return {
            "total_entries": len(self.entries),
            "alias_count": len(self.aliases),
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
                "schema": "persistent_task_queue_v2",
                "updated_at": time.time(),
                "entry_count": len(self.entries),
                "entries": {tid: entry.to_dict() for tid, entry in self.entries.items()},
                "aliases": dict(sorted(self.aliases.items())),
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
            for stored_key, entry_data in data.get("entries", {}).items():
                entry = TaskQueueEntry.from_dict(entry_data)
                key = entry.canonical_task_cid or str(stored_key)
                if key in queue.entries:
                    queue._merge_entries(queue.entries[key], entry)
                else:
                    queue.entries[key] = entry
            raw_aliases = data.get("aliases")
            if isinstance(raw_aliases, dict):
                queue.aliases.update(
                    {
                        str(alias): str(target)
                        for alias, target in raw_aliases.items()
                        if str(alias) and str(target)
                    }
                )
            for key, entry in queue.entries.items():
                for alias in entry.aliases:
                    queue.aliases.setdefault(alias, key)
        except (AttributeError, json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
        return queue
