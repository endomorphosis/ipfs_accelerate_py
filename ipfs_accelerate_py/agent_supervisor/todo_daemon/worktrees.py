"""Reusable Git worktree ownership and cleanup helpers for todo daemons."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from .engine import CommandResult, run_command
from .git_utils import (
    git_worktree_paths_from_porcelain as _shared_git_worktree_paths_from_porcelain,
    paths_from_git_status_porcelain as _shared_paths_from_git_status_porcelain,
    untracked_paths_from_git_status_porcelain as _shared_untracked_paths_from_git_status_porcelain,
)


CommandRunner = Callable[..., CommandResult]
OwnerAlivePredicate = Callable[[int, Path, Path], bool]
TraceResultFormatter = Callable[[CommandResult, int], Any]
WorktreeOwnerWriter = Callable[[Path], None]
WorktreePrepare = Callable[[Path], Any]


WORKTREE_POOL_SCHEMA = "agent-supervisor-worktree-pool-v1"


def _run_command_with_timeout(
    run_command_fn: CommandRunner,
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> CommandResult:
    normalized_timeout = max(1, int(timeout_seconds))
    try:
        return run_command_fn(tuple(command), cwd=cwd, timeout_seconds=normalized_timeout)
    except TypeError as exc:
        if "timeout_seconds" not in str(exc):
            raise
        return run_command_fn(tuple(command), cwd=cwd, timeout=normalized_timeout)


def _trace_key(label: Optional[str], name: str) -> str:
    return name if not label else f"{label}_{name}"


def _compact_trace_result(result: CommandResult, limit: int) -> dict[str, Any]:
    return result.compact(limit=limit)


def git_status_paths(stdout: str) -> list[str]:
    """Return paths from ``git status --porcelain`` output."""

    return _shared_paths_from_git_status_porcelain(stdout)


def untracked_paths_from_git_status(stdout: str) -> list[str]:
    """Return untracked paths from ``git status --porcelain`` output."""

    return _shared_untracked_paths_from_git_status_porcelain(stdout)


def git_worktree_paths_from_porcelain(stdout: str) -> list[Path]:
    """Return registered Git worktree paths from porcelain output."""

    return _shared_git_worktree_paths_from_porcelain(stdout)


def normalize_worktree_path(path: str | Path) -> str:
    """Return a slash-normalized worktree-relative path string."""

    return str(path).replace("\\", "/").strip()


def unique_worktree_paths(paths: Sequence[str | Path]) -> list[str]:
    """Return non-empty worktree paths, slash-normalized and deduplicated in order."""

    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = normalize_worktree_path(path)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def repo_relative_worktree_path(path: str | Path, *, repo_root: Path) -> str:
    """Return ``path`` relative to ``repo_root`` when possible, normalized for Git pathspecs."""

    candidate = Path(path)
    absolute_candidate = candidate if candidate.is_absolute() else repo_root / candidate
    try:
        return absolute_candidate.relative_to(repo_root).as_posix()
    except ValueError:
        return normalize_worktree_path(candidate.as_posix())


def worktree_path_allowed(path: str | Path, *, allowed_prefixes: Sequence[str]) -> bool:
    """Return whether a normalized worktree path is inside one of the allowed prefixes."""

    normalized = normalize_worktree_path(path)
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)


def resolve_worktree_file_edit_path(
    root: Path,
    path: str | Path,
    *,
    allowed_prefixes: Sequence[str],
    error_prefix: str = "Worktree edit",
) -> Path:
    """Resolve a complete-file edit path under ``root`` after traversal and allowlist checks."""

    raw_path = str(path)
    normalized = normalize_worktree_path(raw_path)
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"{error_prefix} path is unsafe: {raw_path!r}")
    if not worktree_path_allowed(normalized, allowed_prefixes=allowed_prefixes):
        raise ValueError(f"{error_prefix} path is outside daemon allowlist: {raw_path!r}")
    return root / normalized


def disallowed_worktree_paths(
    paths: Sequence[str | Path],
    *,
    allowed_prefixes: Sequence[str],
    ignored_paths: Sequence[str | Path] = (),
) -> list[str]:
    """Return changed worktree paths outside the daemon's write allowlist."""

    ignored = set(unique_worktree_paths(ignored_paths))
    disallowed: list[str] = []
    for path in unique_worktree_paths(paths):
        if path in ignored:
            continue
        if worktree_path_allowed(path, allowed_prefixes=allowed_prefixes):
            continue
        disallowed.append(path)
    return disallowed


def dirty_worktree_paths(
    *,
    repo_root: Path,
    paths: Sequence[str | Path],
    timeout_seconds: int = 60,
    run_command_fn: CommandRunner = run_command,
) -> list[str]:
    """Return dirty Git status paths for a normalized path subset."""

    normalized_paths = unique_worktree_paths(paths)
    if not normalized_paths:
        return []
    status = _run_command_with_timeout(
        run_command_fn,
        ("git", "status", "--porcelain", "--", *normalized_paths),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    if not status.ok:
        return []
    return git_status_paths(status.stdout)


def worktree_diff(
    *,
    worktree_path: Path,
    paths: Sequence[str | Path],
    raw_trace: Optional[dict[str, Any]] = None,
    label: str = "worktree",
    timeout_seconds: int = 60,
    run_command_fn: CommandRunner = run_command,
    trace_result_formatter: TraceResultFormatter = _compact_trace_result,
) -> str:
    """Return a binary Git diff for a normalized worktree path subset.

    Untracked files are staged with intent-to-add before diffing so callers can
    harvest new complete-file changes without accepting the whole worktree.
    """

    normalized_paths = unique_worktree_paths(paths)
    if not normalized_paths:
        if raw_trace is not None:
            raw_trace[_trace_key(label, "status")] = {"skipped": True, "reason": "no_paths"}
            raw_trace[_trace_key(label, "untracked_paths")] = []
            raw_trace[_trace_key(label, "git_diff")] = {"skipped": True, "reason": "no_paths"}
        return ""

    status_result = _run_command_with_timeout(
        run_command_fn,
        ("git", "status", "--porcelain", "--", *normalized_paths),
        cwd=worktree_path,
        timeout_seconds=timeout_seconds,
    )
    if raw_trace is not None:
        raw_trace[_trace_key(label, "status")] = trace_result_formatter(status_result, 12000)

    untracked_paths = untracked_paths_from_git_status(status_result.stdout)
    if raw_trace is not None:
        raw_trace[_trace_key(label, "untracked_paths")] = untracked_paths

    if untracked_paths:
        add_intent = _run_command_with_timeout(
            run_command_fn,
            ("git", "add", "-N", "--", *untracked_paths),
            cwd=worktree_path,
            timeout_seconds=timeout_seconds,
        )
        if raw_trace is not None:
            raw_trace[_trace_key(label, "git_add_intent_to_add")] = trace_result_formatter(
                add_intent,
                12000,
            )

    diff_result = _run_command_with_timeout(
        run_command_fn,
        ("git", "diff", "--binary", "--", *normalized_paths),
        cwd=worktree_path,
        timeout_seconds=timeout_seconds,
    )
    if raw_trace is not None:
        raw_trace[_trace_key(label, "git_diff")] = trace_result_formatter(diff_result, 20000)
    return diff_result.stdout if diff_result.ok else ""


def worktree_file_edits(
    worktree_path: Path,
    changed_files: Sequence[str | Path],
    *,
    allowed_prefixes: Sequence[str],
) -> list[dict[str, str]]:
    """Read complete UTF-8 file edits from an isolated worktree for allowed paths."""

    edits: list[dict[str, str]] = []
    for path_text in unique_worktree_paths(changed_files):
        if not worktree_path_allowed(path_text, allowed_prefixes=allowed_prefixes):
            continue
        path = worktree_path / path_text
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        edits.append({"path": path_text, "content": content})
    return edits


def write_worktree_file_edits_to_root(
    root: Path,
    edits: Sequence[Mapping[str, Any]],
    *,
    allowed_prefixes: Sequence[str],
    error_prefix: str = "Worktree edit",
) -> None:
    """Write complete file edits into ``root`` after allowlist and traversal checks."""

    for edit in edits:
        path = resolve_worktree_file_edit_path(
            root,
            str(edit.get("path", "")),
            allowed_prefixes=allowed_prefixes,
            error_prefix=error_prefix,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(edit.get("content", "")), encoding="utf-8")


def pid_is_alive(pid: int) -> bool:
    """Return whether ``pid`` appears live and signalable."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_command_line(pid: int) -> str:
    """Return a process command line from procfs when available."""

    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def pid_looks_like_worktree_owner(
    pid: int,
    *,
    repo_root: Path,
    worktree_path: Path,
    daemon_process_fragment: str = "",
    daemon_repo_hint_fragment: str = "--repo-root",
    worker_process_fragment: str = "codex",
) -> bool:
    """Return whether a live process plausibly owns a daemon worktree."""

    if not pid_is_alive(pid):
        return False
    command_line = pid_command_line(pid)
    if not command_line:
        return True
    normalized_repo = str(repo_root.resolve())
    normalized_worktree = str(worktree_path.resolve())
    if daemon_process_fragment and daemon_process_fragment in command_line:
        return normalized_repo in command_line or daemon_repo_hint_fragment in command_line
    if worker_process_fragment and worker_process_fragment in command_line and normalized_worktree in command_line:
        return True
    return False


def owner_pid_from_worktree(path: Path, owner: Mapping[str, Any]) -> Optional[int]:
    """Return the owner pid from metadata or a trailing ``_<pid>`` worktree name."""

    try:
        pid = int(owner.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid > 0:
        return pid
    match = re.search(r"_(\d+)$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning ``{}`` on missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_worktree_owner_file(
    path: Path,
    *,
    schema: str,
    repo_root: Path,
    pid: Optional[int] = None,
    attempt: int = 0,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write a reusable daemon worktree-owner metadata file."""

    payload: dict[str, Any] = {
        "schema": schema,
        "pid": os.getpid() if pid is None else int(pid),
        "attempt": int(attempt),
        "repo_root": str(repo_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_at_epoch": time.time(),
    }
    if extra:
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


@dataclass
class GitWorktreeSession:
    """State for a managed detached Git worktree lifecycle."""

    repo_root: Path
    path: Path
    metadata_rel: str
    owner_rel: str
    raw_trace: dict[str, Any] = field(default_factory=dict)
    add_result: Optional[CommandResult] = None

    @property
    def ready(self) -> bool:
        """Return whether the detached worktree was created successfully."""

        return bool(self.add_result and self.add_result.ok)


@dataclass
class WorktreeLease:
    """An exclusive task-local checkout borrowed from :class:`WorktreePool`.

    Pool bookkeeping deliberately lives outside ``path``.  The checkout can
    therefore be staged with ``git add -A`` without committing lease metadata.
    A lease must be released before the checkout can be handed to another
    task.  ``release(reusable=True)`` still discards a dirty checkout rather
    than silently erasing task output.
    """

    pool: "WorktreePool" = field(repr=False)
    path: Path
    cache_key: str
    base_ref: str
    base_commit: str
    branch_name: str
    dependency_paths: tuple[str, ...]
    reused: bool
    setup_seconds: float
    estimated_seconds_saved: float
    entry_id: str
    invalidation_reasons: tuple[str, ...] = ()
    acquired_at_epoch: float = field(default_factory=time.time)
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def cache_hit(self) -> bool:
        """Return whether setup was served from a previously prepared entry."""

        return self.reused

    @property
    def metadata(self) -> dict[str, Any]:
        """Return stable, event-log-friendly reuse measurements."""

        return {
            "cache_key": self.cache_key,
            "base_ref": self.base_ref,
            "base_commit": self.base_commit,
            "branch": self.branch_name,
            "worktree_path": str(self.path),
            "dependency_paths": list(self.dependency_paths),
            "reused": self.reused,
            "cache_hit": self.cache_hit,
            "setup_seconds": round(self.setup_seconds, 6),
            "estimated_seconds_saved": round(self.estimated_seconds_saved, 6),
            "setup_time_saved_seconds": round(self.estimated_seconds_saved, 6),
            "invalidation_reason": self.invalidation_reasons[-1] if self.invalidation_reasons else "",
            "invalidation_reasons": list(self.invalidation_reasons),
            "entry_id": self.entry_id,
        }

    def release(self, *, reusable: bool = True) -> dict[str, Any]:
        """Return this checkout to the pool, or discard it when unsafe."""

        if self._released:
            return {"released": False, "reason": "already_released", **self.metadata}
        self._released = True
        return self.pool.release(self, reusable=reusable)

    def __enter__(self) -> "WorktreeLease":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release(reusable=exc_type is None)


class WorktreePool:
    """Pool prepared Git worktrees without sharing task-local mutations.

    Entries are keyed by an explicit dependency/setup key and the resolved base
    commit.  Warm acquisition is allowed only for a registered, recursively
    clean checkout whose dependency HEADs match the recorded prepared state.
    Atomic sidecar lock files make a checkout exclusive across daemon processes.

    The pool intentionally does not infer a cache key.  Callers should include
    every input that affects preparation (for example submodule gitlinks,
    lockfiles, platform and dependency setup version) in ``cache_key``.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        worktree_root: Path,
        run_command_fn: CommandRunner = run_command,
        max_entries: int = 4,
        command_timeout_seconds: int = 120,
        state_dirname: str = ".pool-state",
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.worktree_root = worktree_root.resolve()
        self.run_command_fn = run_command_fn
        self.max_entries = max(1, int(max_entries))
        self.command_timeout_seconds = max(1, int(command_timeout_seconds))
        self.state_root = self.worktree_root / state_dirname
        common_dir_result = _run_command_with_timeout(
            run_command_fn,
            ("git", "rev-parse", "--git-common-dir"),
            cwd=self.repo_root,
            timeout_seconds=self.command_timeout_seconds,
        )
        common_dir_text = common_dir_result.stdout.strip() if common_dir_result.ok else ""
        common_dir = Path(common_dir_text) if common_dir_text else self.repo_root / ".git"
        self.repo_common_dir = (common_dir if common_dir.is_absolute() else self.repo_root / common_dir).resolve()
        self._metrics: dict[str, Any] = {
            "acquisitions": 0,
            "cold_acquisitions": 0,
            "warm_acquisitions": 0,
            "rejected_entries": 0,
            "released_entries": 0,
            "discarded_entries": 0,
            "setup_seconds": 0.0,
            "estimated_seconds_saved": 0.0,
            "rejection_reasons": {},
        }

    @property
    def metrics(self) -> dict[str, Any]:
        """Return a copy of measured process-local pool activity."""

        result = dict(self._metrics)
        result["rejection_reasons"] = dict(self._metrics["rejection_reasons"])
        result["setup_seconds"] = round(float(result["setup_seconds"]), 6)
        result["estimated_seconds_saved"] = round(float(result["estimated_seconds_saved"]), 6)
        attempted = int(result["acquisitions"])
        result["warm_hit_rate"] = round(int(result["warm_acquisitions"]) / attempted, 6) if attempted else 0.0
        result["idle_entries"] = sum(
            1 for state in self._states() if state.get("state") == "idle" and not self._lock_path(state).exists()
        )
        return result

    def acquire(
        self,
        *,
        cache_key: str,
        base_ref: str = "HEAD",
        branch_name: str = "",
        dependency_paths: Sequence[str | Path] = (),
        prepare: Optional[WorktreePrepare] = None,
        activate: Optional[WorktreePrepare] = None,
        worktree_path: Optional[Path] = None,
    ) -> WorktreeLease:
        """Exclusively acquire a clean prepared worktree.

        ``prepare`` is run only on the cold path, after Git creates the main
        checkout.  ``activate`` runs on both paths after binding the task branch
        and is intended for inexpensive task-specific submodule branch setup.
        Both callbacks must finish with all repositories clean.
        """

        normalized_key = str(cache_key).strip()
        if not normalized_key:
            raise ValueError("worktree pool cache_key must not be empty")
        dependencies = tuple(unique_worktree_paths(dependency_paths))
        for dependency in dependencies:
            candidate = Path(dependency)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"worktree dependency path is unsafe: {dependency!r}")
        base_commit = self._rev_parse(self.repo_root, base_ref)
        if not base_commit:
            raise RuntimeError(f"cannot resolve worktree pool base ref {base_ref!r}")
        requested_path = worktree_path.resolve() if worktree_path is not None else None
        if requested_path is not None:
            try:
                requested_path.relative_to(self.worktree_root)
            except ValueError as exc:
                raise ValueError("pooled worktree path must be inside worktree_root") from exc

        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._metrics["acquisitions"] += 1
        acquired_started = time.monotonic()
        invalidation_reasons: list[str] = []
        for state in self._states():
            if not self._state_matches(state, cache_key=normalized_key, base_commit=base_commit, dependencies=dependencies):
                continue
            lock_path = self._try_claim(state)
            if lock_path is None:
                continue
            if state.get("state") == "initializing":
                invalidation_reasons.append("stale_initializing_entry")
                self._reject_and_discard(state, reason="stale_initializing_entry", lock_path=lock_path)
                continue
            valid, reason = self._validate_idle_entry(state)
            if not valid:
                invalidation_reasons.append(reason)
                self._reject_and_discard(state, reason=reason, lock_path=lock_path)
                continue
            path = Path(str(state["path"]))
            if requested_path is not None and path.resolve() != requested_path:
                if requested_path.exists():
                    invalidation_reasons.append("requested_path_exists")
                    self._reject_and_discard(state, reason="requested_path_exists", lock_path=lock_path)
                    continue
                requested_path.parent.mkdir(parents=True, exist_ok=True)
                move = self._run(
                    ("git", "worktree", "move", str(path), str(requested_path)),
                    cwd=self.repo_root,
                )
                if not move.ok:
                    invalidation_reasons.append("worktree_move_failed")
                    self._reject_and_discard(state, reason="worktree_move_failed", lock_path=lock_path)
                    continue
                path = requested_path
                state["path"] = str(path)
                self._write_state(state)
            bind = self._bind_task_branch(path, branch_name=branch_name, base_commit=base_commit)
            if not bind.ok:
                invalidation_reasons.append("branch_bind_failed")
                self._reject_and_discard(state, reason="branch_bind_failed", lock_path=lock_path)
                continue
            try:
                if activate is not None:
                    activate(path)
            except BaseException:
                self._discard_state(state)
                self._remove_lock(lock_path)
                raise
            active_clean, active_reason = self._repositories_clean(path, dependencies)
            expected_dependency_heads = {
                str(key): str(value) for key, value in dict(state.get("dependency_heads") or {}).items()
            }
            if not active_clean or self._dependency_heads(path, dependencies) != expected_dependency_heads:
                rejection_reason = active_reason if not active_clean else "dependency_head_mismatch_after_activate"
                invalidation_reasons.append(rejection_reason)
                self._reject_and_discard(state, reason=rejection_reason, lock_path=lock_path)
                continue
            elapsed = time.monotonic() - acquired_started
            estimated_saved = max(0.0, float(state.get("cold_setup_seconds") or 0.0) - elapsed)
            state.update(
                {
                    "state": "leased",
                    "branch": branch_name,
                    "lease_pid": os.getpid(),
                    "leased_at_epoch": time.time(),
                    "last_used_at_epoch": time.time(),
                    "use_count": int(state.get("use_count") or 0) + 1,
                }
            )
            self._write_state(state)
            self._metrics["warm_acquisitions"] += 1
            self._metrics["setup_seconds"] += elapsed
            self._metrics["estimated_seconds_saved"] += estimated_saved
            return self._lease_from_state(
                state,
                base_ref=base_ref,
                branch_name=branch_name,
                reused=True,
                setup_seconds=elapsed,
                estimated_seconds_saved=estimated_saved,
                invalidation_reasons=tuple(invalidation_reasons),
            )

        return self._create_cold_entry(
            cache_key=normalized_key,
            base_ref=base_ref,
            base_commit=base_commit,
            branch_name=branch_name,
            dependencies=dependencies,
            prepare=prepare,
            activate=activate,
            requested_path=requested_path,
            started=acquired_started,
            invalidation_reasons=tuple(invalidation_reasons),
        )

    @contextmanager
    def lease(self, **kwargs: Any) -> Iterator[WorktreeLease]:
        """Context-manager form of :meth:`acquire`."""

        borrowed = self.acquire(**kwargs)
        try:
            yield borrowed
        except BaseException:
            borrowed.release(reusable=False)
            raise
        else:
            borrowed.release(reusable=True)

    def release(self, lease: WorktreeLease, *, reusable: bool = True) -> dict[str, Any]:
        """Release an exclusive lease, retaining it only after safe scrubbing."""

        state = self._read_state(lease.entry_id)
        lock_path = self.state_root / f"{lease.entry_id}.lock"
        if not state or str(state.get("lease_token")) != lease.entry_id:
            self._remove_lock(lock_path)
            return {"released": False, "reason": "lease_state_missing", **lease.metadata}
        if not reusable:
            discard = self._discard_state(state)
            self._remove_lock(lock_path)
            self._metrics["discarded_entries"] += 1
            return {"released": True, "pooled": False, "reason": "reuse_disabled", "discard": discard, **lease.metadata}

        clean, reason = self._repositories_clean(lease.path, lease.dependency_paths)
        if not clean:
            # Never reset an uncommitted task workspace merely to obtain a pool
            # hit.  Removing the managed checkout is deterministic and prevents
            # accidental cross-task mutation sharing.
            discard = self._discard_state(state)
            self._remove_lock(lock_path)
            self._record_rejection(reason)
            self._metrics["discarded_entries"] += 1
            return {"released": True, "pooled": False, "reason": reason, "discard": discard, **lease.metadata}

        restored, reason = self._restore_prepared_state(state)
        if not restored:
            discard = self._discard_state(state)
            self._remove_lock(lock_path)
            self._record_rejection(reason)
            self._metrics["discarded_entries"] += 1
            return {"released": True, "pooled": False, "reason": reason, "discard": discard, **lease.metadata}

        state.update(
            {
                "state": "idle",
                "branch": "",
                "lease_pid": 0,
                "released_at_epoch": time.time(),
                "last_used_at_epoch": time.time(),
            }
        )
        self._write_state(state)
        self._remove_lock(lock_path)
        self._metrics["released_entries"] += 1
        self._prune_excess_idle(exclude_entry_id=lease.entry_id)
        return {"released": True, "pooled": True, "reason": "clean_prepared_workspace", **lease.metadata}

    def invalidate(self, *, cache_key: Optional[str] = None) -> dict[str, Any]:
        """Discard idle entries, optionally limited to one setup cache key."""

        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for state in self._states():
            if cache_key is not None and str(state.get("cache_key")) != str(cache_key):
                continue
            lock_path = self._try_claim(state)
            if lock_path is None:
                skipped.append({"path": str(state.get("path") or ""), "reason": "leased"})
                continue
            removed.append(self._discard_state(state))
            self._remove_lock(lock_path)
        return {"removed": removed, "skipped": skipped}

    def _create_cold_entry(
        self,
        *,
        cache_key: str,
        base_ref: str,
        base_commit: str,
        branch_name: str,
        dependencies: tuple[str, ...],
        prepare: Optional[WorktreePrepare],
        activate: Optional[WorktreePrepare],
        requested_path: Optional[Path],
        started: float,
        invalidation_reasons: tuple[str, ...],
    ) -> WorktreeLease:
        digest = hashlib.sha256(f"{cache_key}\0{base_commit}".encode("utf-8")).hexdigest()[:12]
        entry_id = f"{digest}-{uuid.uuid4().hex[:12]}"
        path = (requested_path or (self.worktree_root / f"workspace-{entry_id}")).resolve()
        try:
            path.relative_to(self.worktree_root)
        except ValueError as exc:
            raise ValueError("pooled worktree path must be inside worktree_root") from exc
        if path.exists():
            raise FileExistsError(f"pooled worktree path already exists: {path}")
        lock_path = self.state_root / f"{entry_id}.lock"
        self._create_lock(lock_path)
        state: dict[str, Any] = {
            "schema": WORKTREE_POOL_SCHEMA,
            "lease_token": entry_id,
            "path": str(path),
            "repo_root": str(self.repo_root),
            "repo_common_dir": str(self.repo_common_dir),
            "cache_key": cache_key,
            "base_commit": base_commit,
            "dependency_paths": list(dependencies),
            "state": "initializing",
            "lease_pid": os.getpid(),
            "created_at_epoch": time.time(),
            "last_used_at_epoch": time.time(),
            "use_count": 1,
        }
        self._write_state(state)
        add_command = ["git", "worktree", "add"]
        if branch_name:
            add_command.extend(["-b", branch_name])
        else:
            add_command.append("--detach")
        add_command.extend([str(path), base_commit])
        add = self._run(add_command, cwd=self.repo_root)
        if not add.ok:
            self._discard_state(state)
            self._remove_lock(lock_path)
            raise RuntimeError(f"failed to create pooled worktree: {add.stderr or add.stdout}")
        try:
            if prepare is not None:
                prepare(path)
            if activate is not None:
                activate(path)
            clean, reason = self._repositories_clean(path, dependencies)
            if not clean:
                raise RuntimeError(f"prepared worktree is not reusable: {reason}")
            dependency_heads = self._dependency_heads(path, dependencies)
            if len(dependency_heads) != len(dependencies):
                raise RuntimeError("prepared worktree dependency is missing or has no resolvable HEAD")
        except BaseException:
            self._discard_state(state)
            self._remove_lock(lock_path)
            raise
        elapsed = time.monotonic() - started
        state.update(
            {
                "state": "leased",
                "branch": branch_name,
                "dependency_heads": dependency_heads,
                "cold_setup_seconds": elapsed,
            }
        )
        self._write_state(state)
        self._metrics["cold_acquisitions"] += 1
        self._metrics["setup_seconds"] += elapsed
        return self._lease_from_state(
            state,
            base_ref=base_ref,
            branch_name=branch_name,
            reused=False,
            setup_seconds=elapsed,
            estimated_seconds_saved=0.0,
            invalidation_reasons=invalidation_reasons,
        )

    def _lease_from_state(
        self,
        state: Mapping[str, Any],
        *,
        base_ref: str,
        branch_name: str,
        reused: bool,
        setup_seconds: float,
        estimated_seconds_saved: float,
        invalidation_reasons: tuple[str, ...] = (),
    ) -> WorktreeLease:
        return WorktreeLease(
            pool=self,
            path=Path(str(state["path"])),
            cache_key=str(state["cache_key"]),
            base_ref=base_ref,
            base_commit=str(state["base_commit"]),
            branch_name=branch_name,
            dependency_paths=tuple(str(item) for item in state.get("dependency_paths") or ()),
            reused=reused,
            setup_seconds=setup_seconds,
            estimated_seconds_saved=estimated_seconds_saved,
            entry_id=str(state["lease_token"]),
            invalidation_reasons=invalidation_reasons,
        )

    def _state_matches(
        self,
        state: Mapping[str, Any],
        *,
        cache_key: str,
        base_commit: str,
        dependencies: tuple[str, ...],
    ) -> bool:
        return (
            state.get("schema") == WORKTREE_POOL_SCHEMA
            and state.get("state") in {"idle", "leased", "initializing"}
            and str(state.get("repo_root")) == str(self.repo_root)
            and str(state.get("repo_common_dir")) == str(self.repo_common_dir)
            and str(state.get("cache_key")) == cache_key
            and str(state.get("base_commit")) == base_commit
            and tuple(str(item) for item in state.get("dependency_paths") or ()) == dependencies
        )

    def _validate_idle_entry(self, state: Mapping[str, Any]) -> tuple[bool, str]:
        path = Path(str(state.get("path") or ""))
        if not path.is_dir():
            return False, "workspace_missing"
        registered = self._run(("git", "worktree", "list", "--porcelain"), cwd=self.repo_root)
        registered_paths = {
            str(candidate.resolve()) for candidate in git_worktree_paths_from_porcelain(registered.stdout)
        }
        if not registered.ok or str(path.resolve()) not in registered_paths:
            return False, "worktree_not_registered"
        if self._rev_parse(path, "HEAD") != str(state.get("base_commit") or ""):
            return False, "base_commit_mismatch"
        clean, reason = self._repositories_clean(
            path,
            tuple(str(item) for item in state.get("dependency_paths") or ()),
        )
        if not clean:
            return False, reason
        expected_heads = {
            str(key): str(value) for key, value in dict(state.get("dependency_heads") or {}).items()
        }
        if self._dependency_heads(path, tuple(expected_heads)) != expected_heads:
            return False, "dependency_head_mismatch"
        return True, "ready"

    def _repositories_clean(self, path: Path, dependencies: Sequence[str]) -> tuple[bool, str]:
        status = self._run(("git", "status", "--porcelain", "--untracked-files=all"), cwd=path)
        if not status.ok:
            return False, "worktree_status_failed"
        if status.stdout.strip():
            return False, "dirty_worktree"
        for relative in dependencies:
            target = path / relative
            if not target.is_dir() or not self._rev_parse(target, "HEAD"):
                return False, f"dependency_missing:{relative}"
            dependency_status = self._run(
                ("git", "status", "--porcelain", "--untracked-files=all"),
                cwd=target,
            )
            if not dependency_status.ok:
                return False, f"dependency_status_failed:{relative}"
            if dependency_status.stdout.strip():
                return False, f"dirty_dependency:{relative}"
        return True, "clean"

    def _dependency_heads(self, path: Path, dependencies: Sequence[str]) -> dict[str, str]:
        heads: dict[str, str] = {}
        for relative in dependencies:
            head = self._rev_parse(path / relative, "HEAD")
            if head:
                heads[str(relative)] = head
        return heads

    def _restore_prepared_state(self, state: Mapping[str, Any]) -> tuple[bool, str]:
        path = Path(str(state["path"]))
        dependency_heads = {
            str(key): str(value) for key, value in dict(state.get("dependency_heads") or {}).items()
        }
        # Restore children before the parent because the task branch may have
        # changed a gitlink.  -ffd removes task-local untracked context but keeps
        # ignored dependency caches such as node_modules.
        for relative, head in sorted(dependency_heads.items(), key=lambda item: item[0].count("/"), reverse=True):
            target = path / relative
            for command in (
                ("git", "switch", "--detach", head),
                ("git", "reset", "--hard", head),
                ("git", "clean", "-ffd"),
            ):
                if not self._run(command, cwd=target).ok:
                    return False, f"dependency_restore_failed:{relative}"
        base_commit = str(state["base_commit"])
        for command in (
            ("git", "switch", "--detach", base_commit),
            ("git", "reset", "--hard", base_commit),
            ("git", "clean", "-ffd"),
        ):
            if not self._run(command, cwd=path).ok:
                return False, "base_restore_failed"
        clean, reason = self._repositories_clean(path, tuple(dependency_heads))
        if not clean:
            return False, reason
        if self._dependency_heads(path, tuple(dependency_heads)) != dependency_heads:
            return False, "dependency_head_mismatch_after_restore"
        return True, "restored"

    def _bind_task_branch(self, path: Path, *, branch_name: str, base_commit: str) -> CommandResult:
        if not branch_name:
            return self._run(("git", "switch", "--detach", base_commit), cwd=path)
        return self._run(("git", "switch", "-C", branch_name, base_commit), cwd=path)

    def _states(self) -> list[dict[str, Any]]:
        if not self.state_root.exists():
            return []
        states: list[dict[str, Any]] = []
        for path in sorted(self.state_root.glob("*.json")):
            state = read_json_object(path)
            # Bind the token to its sidecar filename.  Besides ignoring partial
            # or foreign JSON, this prevents a corrupted token from selecting
            # an arbitrary path when state and lock files are removed.
            if (
                state.get("schema") == WORKTREE_POOL_SCHEMA
                and str(state.get("lease_token") or "") == path.stem
            ):
                states.append(state)
        return states

    def _state_path(self, entry_id: str) -> Path:
        return self.state_root / f"{entry_id}.json"

    def _lock_path(self, state: Mapping[str, Any]) -> Path:
        return self.state_root / f"{state.get('lease_token', '')}.lock"

    def _read_state(self, entry_id: str) -> dict[str, Any]:
        return read_json_object(self._state_path(entry_id))

    def _write_state(self, state: Mapping[str, Any]) -> None:
        entry_id = str(state["lease_token"])
        path = self._state_path(entry_id)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(dict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _create_lock(self, lock_path: Path) -> None:
        payload = json.dumps({"pid": os.getpid(), "created_at_epoch": time.time()}).encode("utf-8")
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    def _try_claim(self, state: Mapping[str, Any]) -> Optional[Path]:
        lock_path = self._lock_path(state)
        if state.get("state") != "idle":
            try:
                state_owner_pid = int(state.get("lease_pid") or 0)
            except (TypeError, ValueError):
                state_owner_pid = 0
            if state_owner_pid and pid_is_alive(state_owner_pid):
                return None
        try:
            self._create_lock(lock_path)
            return lock_path
        except FileExistsError:
            lock = read_json_object(lock_path)
            try:
                owner_pid = int(lock.get("pid") or 0)
            except (TypeError, ValueError):
                owner_pid = 0
            if owner_pid and pid_is_alive(owner_pid):
                return None
            # A dead claimant never authorizes immediate reuse.  Reclaiming the
            # lock merely permits the normal clean/stale validation below.
            self._remove_lock(lock_path)
            try:
                self._create_lock(lock_path)
                return lock_path
            except FileExistsError:
                return None

    @staticmethod
    def _remove_lock(lock_path: Path) -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _reject_and_discard(self, state: Mapping[str, Any], *, reason: str, lock_path: Path) -> None:
        self._record_rejection(reason)
        self._discard_state(state)
        self._remove_lock(lock_path)

    def _record_rejection(self, reason: str) -> None:
        self._metrics["rejected_entries"] += 1
        reasons = self._metrics["rejection_reasons"]
        reasons[reason] = int(reasons.get(reason) or 0) + 1

    def _discard_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        raw_path = str(state.get("path") or "").strip()
        path = Path(raw_path) if raw_path else self.worktree_root
        try:
            resolved = path.resolve()
            resolved.relative_to(self.worktree_root)
            safe_path = resolved != self.worktree_root
        except (OSError, RuntimeError, ValueError):
            safe_path = False
        if not safe_path:
            entry_id = str(state.get("lease_token") or "")
            if re.fullmatch(r"[A-Za-z0-9._-]+", entry_id):
                self._state_path(entry_id).unlink(missing_ok=True)
            return {
                "path": raw_path,
                "removed": False,
                "reason": "unsafe_or_invalid_worktree_path",
            }
        remove = self._run(("git", "worktree", "remove", "--force", str(path)), cwd=self.repo_root)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        try:
            self._state_path(str(state.get("lease_token") or "")).unlink()
        except FileNotFoundError:
            pass
        return {"path": str(path), "removed": not path.exists(), "git_remove": remove.compact(limit=2000)}

    def _prune_excess_idle(self, *, exclude_entry_id: str) -> None:
        idle = sorted(
            (state for state in self._states() if state.get("state") == "idle"),
            key=lambda state: float(state.get("last_used_at_epoch") or 0.0),
        )
        while len(idle) > self.max_entries:
            state = idle.pop(0)
            if str(state.get("lease_token")) == exclude_entry_id and idle:
                state = idle.pop(0)
            lock_path = self._try_claim(state)
            if lock_path is None:
                continue
            self._discard_state(state)
            self._remove_lock(lock_path)

    def _rev_parse(self, cwd: Path, ref: str) -> str:
        result = self._run(("git", "rev-parse", "--verify", f"{ref}^{{commit}}"), cwd=cwd)
        return result.stdout.strip() if result.ok else ""

    def _run(self, command: Sequence[str], *, cwd: Path) -> CommandResult:
        return _run_command_with_timeout(
            self.run_command_fn,
            command,
            cwd=cwd,
            timeout_seconds=self.command_timeout_seconds,
        )


@contextmanager
def managed_git_worktree(
    *,
    repo_root: Path,
    worktree_path: Path,
    metadata_rel: str,
    owner_rel: str,
    trace_context: Optional[Mapping[str, Any]] = None,
    run_command_fn: CommandRunner = run_command,
    owner_writer: Optional[WorktreeOwnerWriter] = None,
    add_timeout_seconds: int = 60,
    remove_timeout_seconds: int = 60,
    prune_on_exit: bool = True,
) -> Iterator[GitWorktreeSession]:
    """Create a detached Git worktree and always remove/prune it on exit."""

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    raw_trace: dict[str, Any] = dict(trace_context or {})
    raw_trace.update(
        {
            "worktree_path": str(worktree_path),
            "metadata_path": metadata_rel,
            "owner_path": owner_rel,
        }
    )
    session = GitWorktreeSession(
        repo_root=repo_root,
        path=worktree_path,
        metadata_rel=metadata_rel,
        owner_rel=owner_rel,
        raw_trace=raw_trace,
    )
    try:
        add_result = run_command_fn(
            ("git", "worktree", "add", "--detach", str(worktree_path), "HEAD"),
            cwd=repo_root,
            timeout_seconds=max(1, int(add_timeout_seconds)),
        )
        session.add_result = add_result
        raw_trace["worktree_add"] = add_result.compact(limit=12000)
        if add_result.ok and owner_writer is not None:
            owner_writer(worktree_path / owner_rel)
        yield session
    finally:
        remove_result = run_command_fn(
            ("git", "worktree", "remove", "--force", str(worktree_path)),
            cwd=repo_root,
            timeout_seconds=max(1, int(remove_timeout_seconds)),
        )
        raw_trace["worktree_remove"] = remove_result.compact(limit=12000)
        if not remove_result.ok and worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        if prune_on_exit:
            prune_result = run_command_fn(
                ("git", "worktree", "prune", "--expire", "now"),
                cwd=repo_root,
                timeout_seconds=max(1, int(remove_timeout_seconds)),
            )
            raw_trace["worktree_prune_after_remove"] = prune_result.compact(limit=12000)


def cleanup_stale_daemon_worktrees(
    *,
    repo_root: Path,
    worktree_root: Path,
    stale_after_seconds: int,
    owner_filename: str,
    patterns: Sequence[str] = ("cycle_*", "repair_*"),
    run_command_fn: CommandRunner = run_command,
    owner_alive: Optional[OwnerAlivePredicate] = None,
    now_epoch: Optional[float] = None,
) -> dict[str, Any]:
    """Remove daemon-created worktrees whose owner is gone and whose age is stale."""

    stale_after = max(1, int(stale_after_seconds))
    result: dict[str, Any] = {
        "valid": True,
        "worktree_root": str(worktree_root),
        "stale_after_seconds": stale_after,
        "patterns": list(patterns),
        "removed": [],
        "skipped": [],
        "errors": [],
    }
    prune_before = run_command_fn(
        ("git", "worktree", "prune", "--expire", "now"),
        cwd=repo_root,
        timeout_seconds=60,
    )
    result["prune_before"] = prune_before.compact(limit=12000)
    if not worktree_root.exists():
        return result

    root_resolved = worktree_root.resolve()
    list_result = run_command_fn(
        ("git", "worktree", "list", "--porcelain"),
        cwd=repo_root,
        timeout_seconds=60,
    )
    result["worktree_list"] = list_result.compact(limit=12000)
    registered_paths = {str(path) for path in git_worktree_paths_from_porcelain(list_result.stdout)}
    now = time.time() if now_epoch is None else float(now_epoch)

    candidates: list[Path] = []
    seen_candidates: set[Path] = set()
    for pattern in patterns:
        for candidate in worktree_root.glob(pattern):
            resolved_candidate = candidate.resolve()
            if resolved_candidate in seen_candidates:
                continue
            seen_candidates.add(resolved_candidate)
            candidates.append(candidate)

    for candidate in sorted(candidates):
        if not candidate.exists():
            continue
        try:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root_resolved):
                result["skipped"].append({"path": str(candidate), "reason": "outside_worktree_root"})
                continue
            if not candidate.is_dir():
                result["skipped"].append({"path": str(candidate), "reason": "not_directory"})
                continue
            owner = read_json_object(candidate / owner_filename)
            owner_pid = owner_pid_from_worktree(candidate, owner)
            owner_is_alive = bool(owner_pid and owner_alive is not None and owner_alive(owner_pid, repo_root, candidate))
            try:
                created_at = float(owner.get("created_at_epoch") or candidate.stat().st_mtime)
            except (OSError, TypeError, ValueError):
                created_at = candidate.stat().st_mtime
            age_seconds = max(0.0, now - created_at)
            if owner_is_alive:
                result["skipped"].append(
                    {
                        "path": str(candidate),
                        "reason": "owner_pid_alive",
                        "owner_pid": owner_pid,
                        "age_seconds": round(age_seconds, 3),
                    }
                )
                continue
            if age_seconds < stale_after:
                result["skipped"].append(
                    {
                        "path": str(candidate),
                        "reason": "not_stale_yet",
                        "owner_pid": owner_pid,
                        "age_seconds": round(age_seconds, 3),
                    }
                )
                continue

            registered = str(resolved) in registered_paths
            if registered:
                remove_result = run_command_fn(
                    ("git", "worktree", "remove", "--force", str(resolved)),
                    cwd=repo_root,
                    timeout_seconds=60,
                )
                if not remove_result.ok and candidate.exists():
                    shutil.rmtree(candidate, ignore_errors=True)
            else:
                shutil.rmtree(candidate, ignore_errors=True)
                remove_result = CommandResult(
                    ("shutil.rmtree", str(resolved)),
                    0 if not candidate.exists() else 1,
                    "",
                    "" if not candidate.exists() else "directory still exists after rmtree",
                )
            record = {
                "path": str(candidate),
                "registered": registered,
                "owner_pid": owner_pid,
                "age_seconds": round(age_seconds, 3),
                "remove": remove_result.compact(limit=12000),
            }
            if remove_result.ok:
                result["removed"].append(record)
            else:
                result["valid"] = False
                result["errors"].append(record)
        except Exception as exc:
            result["valid"] = False
            result["errors"].append({"path": str(candidate), "exception": f"{type(exc).__name__}: {exc}"})

    prune_after = run_command_fn(
        ("git", "worktree", "prune", "--expire", "now"),
        cwd=repo_root,
        timeout_seconds=60,
    )
    result["prune_after"] = prune_after.compact(limit=12000)
    if not prune_after.ok:
        result["valid"] = False
    return result
