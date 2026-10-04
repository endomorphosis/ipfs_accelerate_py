"""Merge checkpoint persistence for crash recovery.

When merging multiple submodules, the supervisor needs to track which
submodules have been successfully merged so that if it crashes mid-merge,
it can resume from the last checkpoint instead of starting over or leaving
the repository in an inconsistent state.

Usage:
    checkpoint = MergeCheckpoint.create(
        checkpoint_dir=state_dir / "merge_checkpoints",
        branch_name="implementation/task-42-attempt-1",
        task_id="PORTAL-042",
    )

    for submodule in submodules_to_merge:
        result = merge_submodule(submodule)
        checkpoint.record_submodule(submodule, result)

    checkpoint.complete()

On restart:
    checkpoint = MergeCheckpoint.resume(checkpoint_dir, branch_name)
    if checkpoint:
        # Skip already-merged submodules
        remaining = checkpoint.pending_submodules(all_submodules)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MergeCheckpoint:
    """Tracks progress of a multi-submodule merge for crash recovery."""

    branch_name: str
    task_id: str
    attempt: int
    started_at: float
    merged_submodules: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_submodules: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed: bool = False
    completed_at: float = 0.0
    _path: Path | None = None

    def record_submodule(self, submodule_path: str, result: dict[str, Any]) -> None:
        """Record the merge result for a submodule."""
        if result.get("merged"):
            self.merged_submodules[submodule_path] = result
        else:
            self.failed_submodules[submodule_path] = result
        self._save()

    def is_already_merged(self, submodule_path: str) -> bool:
        """Check if a submodule was already successfully merged in this checkpoint."""
        return submodule_path in self.merged_submodules

    def pending_submodules(self, all_submodules: list[str]) -> list[str]:
        """Return submodules that haven't been merged yet."""
        done = set(self.merged_submodules.keys()) | set(self.failed_submodules.keys())
        return [s for s in all_submodules if s not in done]

    def complete(self) -> None:
        """Mark the merge as fully completed and clean up."""
        self.completed = True
        self.completed_at = time.time()
        self._save()
        # Remove checkpoint file on successful completion
        if self._path and self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                pass

    def abort(self, reason: str = "") -> None:
        """Mark as aborted (too many failures, etc.)."""
        self.completed = True
        self.completed_at = time.time()
        self.failed_submodules["_abort"] = {"reason": reason, "at": time.time()}
        self._save()

    def summary(self) -> dict[str, Any]:
        return {
            "branch_name": self.branch_name,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "merged_count": len(self.merged_submodules),
            "failed_count": len(self.failed_submodules),
            "completed": self.completed,
            "started_at": self.started_at,
        }

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "schema": "merge_checkpoint_v1",
                "branch_name": self.branch_name,
                "task_id": self.task_id,
                "attempt": self.attempt,
                "started_at": self.started_at,
                "merged_submodules": self.merged_submodules,
                "failed_submodules": self.failed_submodules,
                "completed": self.completed,
                "completed_at": self.completed_at,
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            pass

    @classmethod
    def create(
        cls,
        *,
        checkpoint_dir: Path,
        branch_name: str,
        task_id: str,
        attempt: int = 1,
    ) -> "MergeCheckpoint":
        """Create a new merge checkpoint."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_name = branch_name.replace("/", "-").replace("\\", "-")[:100]
        path = checkpoint_dir / f"{safe_name}.json"
        checkpoint = cls(
            branch_name=branch_name,
            task_id=task_id,
            attempt=attempt,
            started_at=time.time(),
            _path=path,
        )
        checkpoint._save()
        return checkpoint

    @classmethod
    def resume(cls, checkpoint_dir: Path, branch_name: str) -> "MergeCheckpoint | None":
        """Try to resume an existing checkpoint for a branch.

        Returns None if no checkpoint exists or it's already completed.
        """
        safe_name = branch_name.replace("/", "-").replace("\\", "-")[:100]
        path = checkpoint_dir / f"{safe_name}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        if data.get("completed"):
            # Already completed - clean up stale file
            try:
                path.unlink()
            except OSError:
                pass
            return None

        checkpoint = cls(
            branch_name=data.get("branch_name", branch_name),
            task_id=str(data.get("task_id", "")),
            attempt=int(data.get("attempt", 1)),
            started_at=float(data.get("started_at", 0.0)),
            merged_submodules=data.get("merged_submodules", {}),
            failed_submodules=data.get("failed_submodules", {}),
            completed=False,
            _path=path,
        )
        return checkpoint

    @classmethod
    def cleanup_stale(cls, checkpoint_dir: Path, *, max_age_seconds: float = 7200) -> int:
        """Remove checkpoint files older than max_age that were never completed.

        Returns number of stale checkpoints removed.
        """
        if not checkpoint_dir.exists():
            return 0
        now = time.time()
        removed = 0
        for f in checkpoint_dir.glob("*.json"):
            try:
                stat = f.stat()
                if now - stat.st_mtime > max_age_seconds:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        return removed
