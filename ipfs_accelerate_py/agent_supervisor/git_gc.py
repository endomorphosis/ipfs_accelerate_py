"""Automatic git garbage collection for long-running supervisor worktrees.

Over hours/days of operation, git repositories accumulate:
- Loose objects from abandoned worktree branches
- Dangling commits from force-deleted branches
- Stale pack files from repeated fetch/merge cycles
- Reflogs that grow unbounded

This module provides periodic, non-blocking garbage collection that runs
during supervisor idle time without disrupting active implementations.

Usage:
    from ipfs_accelerate_py.agent_supervisor.git_gc import GitGarbageCollector

    gc = GitGarbageCollector(repo_root=Path("."))
    result = gc.run_if_needed()

Environment variables:
    IPFS_ACCELERATE_AGENT_GC_INTERVAL_SECONDS: Min seconds between GC runs (default: 14400 / 4h)
    IPFS_ACCELERATE_AGENT_GC_AGGRESSIVE_INTERVAL_SECONDS: Min seconds between aggressive GC (default: 86400 / 24h)
    IPFS_ACCELERATE_AGENT_GC_MAX_LOOSE_OBJECTS: Trigger GC when loose objects exceed this (default: 5000)
    IPFS_ACCELERATE_AGENT_GC_REFLOG_EXPIRE_DAYS: Expire reflog entries older than this (default: 7)
    IPFS_ACCELERATE_AGENT_GC_WORKTREE_ROOT: Root path containing worktrees to also GC
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GC_INTERVAL_ENV = "IPFS_ACCELERATE_AGENT_GC_INTERVAL_SECONDS"
_GC_AGGRESSIVE_INTERVAL_ENV = "IPFS_ACCELERATE_AGENT_GC_AGGRESSIVE_INTERVAL_SECONDS"
_GC_MAX_LOOSE_ENV = "IPFS_ACCELERATE_AGENT_GC_MAX_LOOSE_OBJECTS"
_GC_REFLOG_EXPIRE_ENV = "IPFS_ACCELERATE_AGENT_GC_REFLOG_EXPIRE_DAYS"
_GC_WORKTREE_ROOT_ENV = "IPFS_ACCELERATE_AGENT_GC_WORKTREE_ROOT"

DEFAULT_GC_INTERVAL = 14400  # 4 hours
DEFAULT_GC_AGGRESSIVE_INTERVAL = 86400  # 24 hours
DEFAULT_MAX_LOOSE_OBJECTS = 5000
DEFAULT_REFLOG_EXPIRE_DAYS = 7


def _run_git(args: list[str], *, cwd: Path, timeout: float = 300) -> subprocess.CompletedProcess:
    """Run Git with a timeout that also terminates descendant processes."""
    command = ["git"] + args
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def count_loose_objects(repo_root: Path) -> int:
    """Count loose git objects in the repository."""
    result = _run_git(["count-objects", "-v"], cwd=repo_root)
    if result.returncode != 0:
        return 0
    for line in result.stdout.splitlines():
        if line.startswith("count:"):
            try:
                return int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
    return 0


def repo_disk_usage(repo_root: Path) -> dict[str, Any]:
    """Get disk usage statistics for the git repository."""
    result = _run_git(["count-objects", "-v", "-H"], cwd=repo_root)
    stats: dict[str, Any] = {"raw_output": result.stdout if result.returncode == 0 else ""}

    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().replace(" ", "_").replace("-", "_")
                value = parts[1].strip()
                stats[key] = value

    return stats


@dataclass
class GCState:
    """Tracks when GC operations were last performed."""

    last_gc_time: float = 0.0
    last_aggressive_gc_time: float = 0.0
    last_prune_time: float = 0.0
    last_reflog_expire_time: float = 0.0
    last_repack_time: float = 0.0
    total_gc_runs: int = 0
    total_objects_freed: int = 0
    _path: Path | None = None

    def save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_gc_time": self.last_gc_time,
                "last_aggressive_gc_time": self.last_aggressive_gc_time,
                "last_prune_time": self.last_prune_time,
                "last_reflog_expire_time": self.last_reflog_expire_time,
                "last_repack_time": self.last_repack_time,
                "total_gc_runs": self.total_gc_runs,
                "total_objects_freed": self.total_objects_freed,
            }
            self._path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path) -> "GCState":
        state = cls(_path=path)
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state.last_gc_time = float(data.get("last_gc_time", 0.0))
            state.last_aggressive_gc_time = float(data.get("last_aggressive_gc_time", 0.0))
            state.last_prune_time = float(data.get("last_prune_time", 0.0))
            state.last_reflog_expire_time = float(data.get("last_reflog_expire_time", 0.0))
            state.last_repack_time = float(data.get("last_repack_time", 0.0))
            state.total_gc_runs = int(data.get("total_gc_runs", 0))
            state.total_objects_freed = int(data.get("total_objects_freed", 0))
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return state


@dataclass
class GitGarbageCollector:
    """Automatic git garbage collection for long-running supervisor repos."""

    repo_root: Path
    state_path: Path | None = None
    worktree_root: Path | None = None
    gc_interval: float = field(default_factory=lambda: float(os.environ.get(_GC_INTERVAL_ENV, str(DEFAULT_GC_INTERVAL))))
    aggressive_interval: float = field(default_factory=lambda: float(os.environ.get(_GC_AGGRESSIVE_INTERVAL_ENV, str(DEFAULT_GC_AGGRESSIVE_INTERVAL))))
    max_loose_objects: int = field(default_factory=lambda: int(os.environ.get(_GC_MAX_LOOSE_ENV, str(DEFAULT_MAX_LOOSE_OBJECTS))))
    reflog_expire_days: int = field(default_factory=lambda: int(os.environ.get(_GC_REFLOG_EXPIRE_ENV, str(DEFAULT_REFLOG_EXPIRE_DAYS))))
    _state: GCState | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.state_path is None:
            self.state_path = self.repo_root / "data" / "agent_supervisor" / "gc_state.json"
        if self.worktree_root is None:
            env_root = os.environ.get(_GC_WORKTREE_ROOT_ENV)
            if env_root:
                self.worktree_root = Path(env_root)
        self._state = GCState.load(self.state_path)

    @property
    def state(self) -> GCState:
        if self._state is None:
            self._state = GCState.load(self.state_path)
        return self._state

    def needs_gc(self) -> bool:
        """Check if GC should run based on time interval and object count."""
        now = time.time()
        if now - self.state.last_gc_time < self.gc_interval:
            return False

        # Also check loose object count
        loose_count = count_loose_objects(self.repo_root)
        if loose_count >= self.max_loose_objects:
            return True

        # Time-based trigger
        return True

    def needs_aggressive_gc(self) -> bool:
        """Check if aggressive GC should run (less frequent, more thorough)."""
        now = time.time()
        return now - self.state.last_aggressive_gc_time >= self.aggressive_interval

    def run_if_needed(self) -> dict[str, Any]:
        """Run GC if conditions are met. Returns summary of actions taken."""
        if not self.needs_gc():
            return {"ran": False, "reason": "not_needed"}

        return self.run()

    def run(self, *, aggressive: bool = False) -> dict[str, Any]:
        """Run garbage collection operations."""
        if aggressive or self.needs_aggressive_gc():
            return self._run_aggressive()
        return self._run_standard()

    def _run_standard(self) -> dict[str, Any]:
        """Standard GC: prune worktrees, expire reflogs, auto gc."""
        results: dict[str, Any] = {
            "type": "standard",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
        }

        loose_before = count_loose_objects(self.repo_root)
        results["loose_objects_before"] = loose_before

        # Step 1: Prune worktrees
        step = self._prune_worktrees()
        results["steps"].append(step)

        # Step 2: Expire old reflogs
        step = self._expire_reflogs()
        results["steps"].append(step)

        # Step 3: Standard git gc (auto mode - only runs if needed)
        step = self._run_git_gc(auto=True)
        results["steps"].append(step)

        # Step 4: GC submodules if configured
        if self.worktree_root:
            step = self._gc_submodule_repos()
            results["steps"].append(step)

        loose_after = count_loose_objects(self.repo_root)
        results["loose_objects_after"] = loose_after
        results["objects_freed"] = max(0, loose_before - loose_after)
        results["finished_at"] = datetime.now(timezone.utc).isoformat()

        # Update state
        self.state.last_gc_time = time.time()
        self.state.total_gc_runs += 1
        self.state.total_objects_freed += results["objects_freed"]
        self.state.save()

        logger.info(
            "Git GC completed: freed %d objects (%d -> %d loose)",
            results["objects_freed"],
            loose_before,
            loose_after,
        )
        return results

    def _run_aggressive(self) -> dict[str, Any]:
        """Aggressive GC: full repack, prune all, expire all reflogs."""
        results: dict[str, Any] = {
            "type": "aggressive",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
        }

        loose_before = count_loose_objects(self.repo_root)
        results["loose_objects_before"] = loose_before

        # Step 1: Prune worktrees
        step = self._prune_worktrees()
        results["steps"].append(step)

        # Step 2: Expire ALL reflogs (aggressive)
        step = self._expire_reflogs(expire_all=True)
        results["steps"].append(step)

        # Step 3: Aggressive git gc
        step = self._run_git_gc(aggressive=True)
        results["steps"].append(step)

        # Step 4: Repack
        step = self._repack()
        results["steps"].append(step)

        # Step 5: Prune unreachable objects
        step = self._prune_objects()
        results["steps"].append(step)

        # Step 6: GC submodules
        if self.worktree_root:
            step = self._gc_submodule_repos()
            results["steps"].append(step)

        loose_after = count_loose_objects(self.repo_root)
        results["loose_objects_after"] = loose_after
        results["objects_freed"] = max(0, loose_before - loose_after)
        results["finished_at"] = datetime.now(timezone.utc).isoformat()

        # Update state
        now = time.time()
        self.state.last_gc_time = now
        self.state.last_aggressive_gc_time = now
        self.state.last_repack_time = now
        self.state.total_gc_runs += 1
        self.state.total_objects_freed += results["objects_freed"]
        self.state.save()

        logger.info(
            "Aggressive GC completed: freed %d objects (%d -> %d loose)",
            results["objects_freed"],
            loose_before,
            loose_after,
        )
        return results

    def _prune_worktrees(self) -> dict[str, Any]:
        """Prune stale worktree references."""
        result = _run_git(["worktree", "prune"], cwd=self.repo_root)
        self.state.last_prune_time = time.time()
        return {
            "step": "worktree_prune",
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip()[:500],
        }

    def _expire_reflogs(self, *, expire_all: bool = False) -> dict[str, Any]:
        """Expire old reflog entries."""
        if expire_all:
            args = ["reflog", "expire", "--expire=now", "--all"]
        else:
            expire_time = f"{self.reflog_expire_days}.days.ago"
            args = ["reflog", "expire", f"--expire={expire_time}", "--all"]

        result = _run_git(args, cwd=self.repo_root)
        self.state.last_reflog_expire_time = time.time()
        return {
            "step": "reflog_expire",
            "expire_all": expire_all,
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip()[:500],
        }

    def _run_git_gc(self, *, auto: bool = False, aggressive: bool = False) -> dict[str, Any]:
        """Run git gc with specified mode."""
        args = ["gc", "--quiet"]
        if auto:
            args.append("--auto")
        if aggressive:
            args.append("--aggressive")

        try:
            result = _run_git(args, cwd=self.repo_root, timeout=600)
        except subprocess.TimeoutExpired:
            return {"step": "git_gc", "error": "timeout", "mode": "aggressive" if aggressive else "auto" if auto else "standard"}

        return {
            "step": "git_gc",
            "mode": "aggressive" if aggressive else "auto" if auto else "standard",
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip()[:500],
        }

    def _repack(self) -> dict[str, Any]:
        """Repack objects into fewer, larger pack files."""
        # -a: pack all objects, -d: remove redundant packs, --depth=250: deeper delta chains
        result = _run_git(["repack", "-a", "-d", "--depth=250", "--window=250"], cwd=self.repo_root, timeout=600)
        self.state.last_repack_time = time.time()
        return {
            "step": "repack",
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip()[:500],
        }

    def _prune_objects(self) -> dict[str, Any]:
        """Prune unreachable objects."""
        result = _run_git(["prune", "--expire=now"], cwd=self.repo_root)
        return {
            "step": "prune_objects",
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip()[:500],
        }

    def _gc_submodule_repos(self) -> dict[str, Any]:
        """Run light GC on submodule repositories."""
        results: list[dict[str, Any]] = []

        # Find submodules
        sm_result = _run_git(["submodule", "foreach", "--quiet", "echo $sm_path"], cwd=self.repo_root)
        if sm_result.returncode != 0:
            return {"step": "submodule_gc", "error": "list_failed", "submodules": []}

        for sm_path in sm_result.stdout.strip().splitlines():
            sm_path = sm_path.strip()
            if not sm_path:
                continue
            full_path = self.repo_root / sm_path
            if not full_path.exists() or not (full_path / ".git").exists():
                continue

            gc_result = _run_git(["gc", "--auto", "--quiet"], cwd=full_path, timeout=120)
            results.append({
                "submodule": sm_path,
                "returncode": gc_result.returncode,
            })

        return {
            "step": "submodule_gc",
            "count": len(results),
            "submodules": results[:20],
        }

    def disk_usage_report(self) -> dict[str, Any]:
        """Generate a disk usage report for the repository."""
        stats = repo_disk_usage(self.repo_root)
        stats["gc_state"] = {
            "last_gc": self.state.last_gc_time,
            "last_aggressive": self.state.last_aggressive_gc_time,
            "total_runs": self.state.total_gc_runs,
            "total_freed": self.state.total_objects_freed,
        }
        return stats
