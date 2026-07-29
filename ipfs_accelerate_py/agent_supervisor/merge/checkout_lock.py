"""Reusable checkout mutation lock helpers for autonomous agent supervisors."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised only on non-Windows hosts
    msvcrt = None  # type: ignore[assignment]

from ..proof.formal_verification_contracts import content_identity


DEFAULT_CHECKOUT_MUTATION_LOCK_NAME = "implementation-main-merge.lock"
PROTECTED_PATH_MAINTENANCE_LOCK_NAME = (
    "implementation-protected-path-maintenance.lock"
)
DEFAULT_MERGE_TRAIN_DIRECTORY_NAME = "agent-merge-trains"
BACKLOG_REFINERY_AUTHOR_EMAIL = "accelerator-backlog-refinery@example.invalid"
GENERATED_PROTECTED_BOARD_COMMIT_MARKER = (
    "[agent-supervisor:generated-protected-board]"
)


def generated_protected_board_commit_subject(subject: str) -> str:
    """Tag a supervisor-generated protected-board commit for fence validation."""

    normalized = str(subject or "").strip()
    if normalized.endswith(GENERATED_PROTECTED_BOARD_COMMIT_MARKER):
        return normalized
    return f"{normalized} {GENERATED_PROTECTED_BOARD_COMMIT_MARKER}".strip()


@contextmanager
def serialized_lock_update(lock_path: Path) -> Iterator[None]:
    """Serialize create/inspect/replace operations for one durable lock path.

    The durable JSON lock remains the long-lived ownership record.  This
    adjacent advisory guard is held only while that record is inspected or
    replaced, closing the stale-owner check/unlink race between the supervisor
    and its managed implementation daemon.
    """

    if fcntl is None and msvcrt is None:
        raise RuntimeError(
            "durable lock replacement requires an advisory file-lock backend"
        )
    guard_path = lock_path.with_name(f".{lock_path.name}.update.lock")
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(guard_path, flags, 0o600)
    locked = False
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            assert msvcrt is not None
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            while True:
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {
                        errno.EACCES,
                        errno.EAGAIN,
                        errno.EDEADLK,
                    }:
                        raise
                    time.sleep(0.05)
        locked = True
        yield
    finally:
        try:
            if locked:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    assert msvcrt is not None
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


def git_common_dir(repo_root: Path) -> Path:
    """Return the repository common git directory for a checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return repo_root / ".git"
    stdout = result.stdout or ""
    if result.returncode != 0 or not stdout.strip():
        return repo_root / ".git"
    path = Path(stdout.strip())
    return path if path.is_absolute() else repo_root / path


def checkout_mutation_lock_path(
    repo_root: Path,
    *,
    lock_name: str = DEFAULT_CHECKOUT_MUTATION_LOCK_NAME,
) -> Path:
    """Return a repo-wide lock path for parent checkout mutations."""

    return git_common_dir(repo_root) / lock_name


def checkout_repository_id(repo_root: Path) -> str:
    """Return a stable local identity shared by all worktrees of one Git repo."""

    common_dir = git_common_dir(repo_root)
    try:
        identity_source = str(common_dir.resolve())
    except (OSError, RuntimeError):
        identity_source = str(common_dir)
    return (
        "repository:"
        + content_identity(
            {
                "kind": "local-git-common-directory",
                "path": identity_source,
            }
        )
    )


def merge_target_queue_dir(
    repo_root: Path,
    target_branch: str,
) -> Path:
    """Return the queue namespace for one physical repository and target ref.

    Older supervisors used one queue directly under ``agent-merge-train``.
    Keeping target-scoped queues in a new directory prevents a still-running
    legacy consumer from claiming requests produced by upgraded daemons.
    """

    branch = str(target_branch or "").strip()
    if not branch:
        raise ValueError("merge target branch must not be empty")
    repository_id = checkout_repository_id(repo_root)
    binding = f"{repository_id}\0{branch}".encode("utf-8")
    digest = hashlib.sha256(binding).hexdigest()[:20]
    safe_branch = "".join(
        character if character.isalnum() or character in "-._" else "-"
        for character in branch
    ).strip("-") or "target"
    return (
        git_common_dir(repo_root)
        / DEFAULT_MERGE_TRAIN_DIRECTORY_NAME
        / f"{safe_branch[:48]}-{digest}"
    )


def checkout_lock_metadata(
    *,
    kind: str,
    repo_root: Path,
    task_id: str = "",
    branch: str = "",
    attempt: int = 0,
    owner_script: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable metadata for a checkout mutation lock."""

    payload: dict[str, Any] = {
        "kind": kind,
        "pid": os.getpid(),
        "owner_script": owner_script if owner_script is not None else Path(sys.argv[0]).name,
        "repo_root": str(repo_root.resolve()),
        "task_id": task_id,
        "attempt": int(attempt or 0),
        "branch": branch,
    }
    if extra:
        payload.update(extra)
    return payload


def checkout_lock_owner_is_active(
    metadata: dict[str, Any],
    *,
    expected_kind: str,
    expected_repo_root: Path | None = None,
    process_command_line: Any,
    process_is_running: Any,
) -> bool:
    """Return whether lock metadata still belongs to a live compatible process."""

    kind = str(metadata.get("kind") or "")
    if kind and kind != expected_kind:
        return False
    repo_root = str(metadata.get("repo_root") or "")
    if expected_repo_root is not None and repo_root:
        try:
            if Path(repo_root).resolve() != expected_repo_root.resolve():
                return False
        except OSError:
            return False
    try:
        pid = int(metadata.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if not process_is_running(pid):
        return False
    owner_script = str(metadata.get("owner_script") or "")
    command_line = process_command_line(pid)
    if owner_script and owner_script not in command_line:
        # Module launches (``python -m package.worker``) expose the module
        # name, not the source filename, in argv.
        owner_module_stem = Path(owner_script).stem
        if not owner_module_stem or owner_module_stem not in command_line:
            return False
    return True
