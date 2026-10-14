"""Serialize native Git mutations by physical checkout, independently of boards.

The durable board lease continues to decide mutation authority. This advisory
guard only prevents two authorized workers from editing the same Git index at
once. Its inode is never removed or used to reclaim a foreign Git lock.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import threading
import time
import uuid
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - fail closed without the POSIX backend
    fcntl = None

LOCK_NAME = "agent-checkout-git-mutation.guard"
PENDING_CUSTODY_NAME = "agent-git-contention-custody.pending.json"
MAX_INDEX_BYTES = 64 * 1024 * 1024
_registry_lock = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}
_held = threading.local()


def _after_fork() -> None:
    global _registry_lock, _thread_locks, _held
    _registry_lock = threading.Lock()
    _thread_locks = {}
    _held = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class CheckoutGitMutationBlocked(RuntimeError):
    def __init__(self, reason: str, *, paths: tuple[str, ...] = ()):
        super().__init__(reason)
        self.reason = reason
        self.paths = paths


def git_checkout_directory(repo: Path) -> Path:
    """Resolve the per-worktree Git directory without an optional index write."""

    if any(os.environ.get(key) for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")):
        raise CheckoutGitMutationBlocked("checkout_git_environment_override")
    result = subprocess.run(
        ["git", "--no-optional-locks", "rev-parse", "--absolute-git-dir"],
        cwd=repo, text=True, capture_output=True, check=False, timeout=10,
    )
    if result.returncode or not result.stdout.strip():
        raise CheckoutGitMutationBlocked("checkout_git_directory_unavailable")
    path = Path(result.stdout.strip()).resolve(strict=True)
    if not path.is_dir():
        raise CheckoutGitMutationBlocked("checkout_git_directory_unavailable")
    return path


def _foreign_git_locks(directory: Path) -> tuple[str, ...]:
    return tuple(str(directory / name) for name in ("index.lock", "HEAD.lock")
                 if os.path.lexists(directory / name))


@contextmanager
def checkout_git_mutation(repo: Path, *, timeout_seconds: float = 30.0) -> Iterator[Path]:
    """Hold one reentrant checkout guard; preserve every unknown lock inode."""

    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("checkout Git timeout must be finite and nonnegative")
    directory = git_checkout_directory(Path(repo))
    key = str(directory)
    depths = getattr(_held, "depths", None)
    if depths is None:
        depths = _held.depths = {}
    if depths.get(key, 0):
        depths[key] += 1
        try:
            yield directory
        finally:
            depths[key] -= 1
        return
    if fcntl is None:
        raise CheckoutGitMutationBlocked("checkout_git_advisory_backend_unavailable")
    with _registry_lock:
        thread_lock = _thread_locks.setdefault(key, threading.RLock())
    started = time.monotonic()
    if not thread_lock.acquire(timeout=timeout):
        raise CheckoutGitMutationBlocked("checkout_git_mutation_busy")
    descriptor = None
    locked = False
    body_entered = False
    path = directory / LOCK_NAME
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise CheckoutGitMutationBlocked("checkout_git_guard_invalid", paths=(str(path),))
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise CheckoutGitMutationBlocked("checkout_git_mutation_busy", paths=(str(path),))
                time.sleep(min(0.025, remaining))
        current = path.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise CheckoutGitMutationBlocked("checkout_git_guard_replaced", paths=(str(path),))
        foreign = _foreign_git_locks(directory)
        if foreign:
            raise CheckoutGitMutationBlocked("checkout_git_foreign_lock_present", paths=foreign)
        if os.path.lexists(directory / PENDING_CUSTODY_NAME):
            raise CheckoutGitMutationBlocked(
                "checkout_git_retained_contention_requires_recovery",
                paths=(str(directory / PENDING_CUSTODY_NAME),),
            )
        depths[key] = 1
        body_entered = True
        try:
            yield directory
        finally:
            depths.pop(key, None)
    except OSError as exc:
        if body_entered:
            raise
        raise CheckoutGitMutationBlocked("checkout_git_guard_io_error", paths=(str(path),)) from exc
    finally:
        if descriptor is not None:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        thread_lock.release()


def serialize_checkout_git(*, repo_parameter: str | None = None):
    """Fence a daemon operation without changing its board authority checks."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def guarded(self, *args, **kwargs):
            arguments = signature.bind(self, *args, **kwargs).arguments
            repo = arguments[repo_parameter] if repo_parameter else self.repo_root
            try:
                with checkout_git_mutation(Path(repo)):
                    return function(self, *args, **kwargs)
            except CheckoutGitMutationBlocked as exc:
                result = {
                    **dict(arguments.get("failure_fields") or {}),
                    "attempted": False, "merged": False, "committed": False,
                    "updated": False, "durable": False, "blocked": True,
                    "infrastructure_failure": True, "provider_dispatched": False,
                    "reason": exc.reason, "lock_paths": list(exc.paths),
                    "operation": function.__name__,
                }
                task_id = arguments.get("task_id") or arguments.get("primary_task_id")
                if task_id:
                    result["task_id"] = str(task_id)
                if exc.reason == "checkout_git_retained_contention_requires_recovery":
                    result["checkout_mutation_recovery_required"] = True
                self._record_event("checkout_git_mutation_deferred", result)
                return result

        return guarded
    return decorate


def git_lock_contention(result: subprocess.CompletedProcess) -> bool:
    """Classify only Git's explicit lock-file exclusion failure."""

    message = str(result.stderr or "").casefold()
    return bool(result.returncode and ".lock" in message and (
        ("unable to create" in message and "file exists" in message)
        or ("cannot lock ref" in message and "file exists" in message)
        or "another git process" in message
    ))


def capture_checkout_index(repo: Path) -> dict[str, Any]:
    """Capture exact index bytes for custody, never for automatic restoration."""

    directory = git_checkout_directory(repo)
    path = directory / "index"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return {"present": False, "payload": b"", "sha256": None}
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_INDEX_BYTES:
            raise CheckoutGitMutationBlocked("checkout_git_index_custody_bound")
        payload = stream.read(MAX_INDEX_BYTES + 1)
        after = os.fstat(stream.fileno())
        observed = path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(payload) > MAX_INDEX_BYTES or identity(before) != identity(after) or identity(after) != identity(observed):
            raise CheckoutGitMutationBlocked("checkout_git_index_changed_during_custody")
    return {"present": True, "payload": payload, "sha256": hashlib.sha256(payload).hexdigest()}


def retain_git_contention_custody(repo: Path, *, before_index: dict[str, Any],
                                  task_id: str, branch: str, before_head: str,
                                  result: subprocess.CompletedProcess) -> dict[str, Any]:
    """Retain partial Git state as an owned observation without deleting data."""

    directory = git_checkout_directory(repo)
    # Publish the refusal boundary first: even a subsequent disk/read failure
    # must leave later workers unable to interpret partial stages as a legal
    # content conflict or delete the retained checkout.
    marker = directory / PENDING_CUSTODY_NAME
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(json.dumps({"schema": "agent-git-contention-custody-pending/v1",
                                 "pid": os.getpid(), "task_id": task_id,
                                 "branch": branch, "before_head": before_head,
                                 "before_index_sha256": before_index["sha256"],
                                 "provider_dispatched": False,
                                 "completion_authority": False}, sort_keys=True).encode())
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    try:
        after_index = capture_checkout_index(repo)
    except (OSError, CheckoutGitMutationBlocked) as exc:
        after_index = {"present": None, "payload": b"", "sha256": None,
                       "capture_error": str(exc)[:512]}
    custody_root = directory / "agent-git-contention-custody"
    custody_root.mkdir(mode=0o700, exist_ok=True)
    root_fd = os.open(custody_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    receipt_name = uuid.uuid4().hex
    os.mkdir(receipt_name, mode=0o700, dir_fd=root_fd)
    receipt_dir = custody_root / receipt_name
    receipt_fd = os.open(receipt_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    identities = {}
    try:
        for label, snapshot in (("before", before_index), ("after", after_index)):
            payload = snapshot["payload"]
            fd = os.open(label + ".index", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=receipt_fd)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            identities[label] = {k: snapshot[k] for k in ("present", "sha256")}
            identities[label]["bytes"] = len(payload)
            if "capture_error" in snapshot:
                identities[label]["capture_error"] = snapshot["capture_error"]
    except BaseException:
        os.close(receipt_fd)
        os.close(root_fd)
        raise
    status = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain=v1"],
                            cwd=repo, text=True, capture_output=True, check=False)
    receipt = {
        "schema": "agent-checkout-git-contention-custody/v1", "pid": os.getpid(),
        "task_id": task_id, "branch": branch, "before_head": before_head,
        "index_snapshots": identities, "returncode": result.returncode,
        "stderr": str(result.stderr or "")[-4000:], "stdout": str(result.stdout or "")[-4000:],
        "status_returncode": status.returncode, "status": status.stdout[-16000:],
        "data_restored_or_deleted": False, "foreign_git_lock_removed": False,
        "provider_dispatched": False, "completion_authority": False,
        "disposition": "retained_unreviewed_git_state_requires_retry_or_owned_recovery",
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path = receipt_dir / "receipt.json"
    digest = hashlib.sha256(payload).hexdigest()
    try:
        fd = os.open("receipt.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=receipt_fd)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(receipt_fd)
        os.fsync(root_fd)
    finally:
        os.close(receipt_fd)
        os.close(root_fd)
    return {"path": str(path), "sha256": digest,
            "index_changed": before_index["sha256"] != after_index["sha256"],
            "data_restored_or_deleted": False, "completion_authority": False}
