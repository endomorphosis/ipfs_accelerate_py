"""Reusable checkout mutation lock helpers for autonomous agent supervisors."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

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
DEFAULT_MERGE_TRAIN_DIRECTORY_NAME = "agent-merge-trains"
DEFAULT_OBJECTIVE_ADMISSION_LOCK_DIRECTORY_NAME = (
    "agent-objective-admission-locks"
)
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
def serialized_lock_update(
    lock_path: Path,
    *,
    timeout_seconds: float | None = None,
    poll_seconds: float = 0.05,
) -> Iterator[None]:
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
    deadline = (
        None
        if timeout_seconds is None
        else time.monotonic() + max(0.0, float(timeout_seconds))
    )
    try:
        if fcntl is not None:
            if deadline is None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                "timed out serializing durable lock update"
                            )
                        time.sleep(
                            min(max(0.001, float(poll_seconds)), remaining)
                        )
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
                    if (
                        deadline is not None
                        and deadline - time.monotonic() <= 0
                    ):
                        raise TimeoutError(
                            "timed out serializing durable lock update"
                        )
                    sleep_seconds = max(0.001, float(poll_seconds))
                    if deadline is not None:
                        sleep_seconds = min(
                            sleep_seconds,
                            max(0.0, deadline - time.monotonic()),
                        )
                    if sleep_seconds <= 0:
                        raise TimeoutError(
                            "timed out serializing durable lock update"
                        )
                    time.sleep(sleep_seconds)
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


def objective_admission_lock_path(objective_path: Path) -> Path:
    """Return a durable lock outside a Git worktree when one is available.

    Objective admission locks must survive for the lifetime of cooperating
    processes: unlinking an advisory lock after use creates an inode race.
    Keeping that persistent coordination file beside a tracked objective,
    however, makes an otherwise clean integration checkout permanently dirty.
    A repository's common Git directory is shared by the processes operating
    on that checkout while remaining outside its status surface.

    Git-less callers retain the historical adjacent-lock fallback.
    """

    objective = Path(objective_path).resolve()
    cwd = objective.parent
    while not cwd.exists() and cwd.parent != cwd:
        cwd = cwd.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "--git-common-dir"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        result = None
    lines = (
        [line.strip() for line in (result.stdout or "").splitlines()]
        if result is not None and result.returncode == 0
        else []
    )
    if len(lines) >= 2:
        top = Path(lines[0]).resolve()
        try:
            objective.relative_to(top)
        except ValueError:
            pass
        else:
            common_dir = Path(lines[1])
            if not common_dir.is_absolute():
                common_dir = (cwd / common_dir).resolve()
            safe_name = "".join(
                character
                if character.isalnum() or character in "-._"
                else "-"
                for character in objective.name
            ).strip("-") or "objective"
            binding = f"{top}\0{objective}".encode("utf-8")
            digest = hashlib.sha256(binding).hexdigest()[:20]
            return (
                common_dir
                / DEFAULT_OBJECTIVE_ADMISSION_LOCK_DIRECTORY_NAME
                / f"{safe_name[:64]}-{digest}.lock"
            )
    return objective.with_name(f".{objective.name}.admission.lock")


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
    if not str(payload.get("lease_id") or ""):
        payload["lease_id"] = content_identity(
            {
                "kind": "checkout-mutation-lease",
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "repo_root": payload["repo_root"],
                "task_id": task_id,
                "attempt": int(attempt or 0),
                "branch": branch,
                "operation": str(payload.get("operation") or ""),
                "issued_ns": time.time_ns(),
            }
        )
    return payload


@dataclass(frozen=True)
class CheckoutMutationLease:
    """A fully published checkout lock bound to one filesystem identity."""

    lock_path: Path
    metadata: Mapping[str, Any]
    device: int
    inode: int

    @property
    def lease_id(self) -> str:
        return str(self.metadata.get("lease_id") or "")


def _read_checkout_lock(
    lock_path: Path,
) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    """Read one lock with a stable inode identity or return inconclusive."""

    try:
        before = lock_path.stat(follow_symlinks=False)
        raw = lock_path.read_text(encoding="utf-8")
        after = lock_path.stat(follow_symlinks=False)
    except (OSError, UnicodeError):
        return None, None
    before_identity = (int(before.st_dev), int(before.st_ino))
    if before_identity != (int(after.st_dev), int(after.st_ino)):
        return None, None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None, before_identity
    if not isinstance(payload, dict):
        return None, before_identity
    return payload, before_identity


def _try_publish_checkout_mutation_lease(
    lock_path: Path,
    metadata: Mapping[str, Any],
) -> tuple[
    CheckoutMutationLease | None,
    str,
    dict[str, Any] | None,
]:
    """Atomically publish complete lock metadata without an empty-file window."""

    normalized = dict(metadata)
    lease_id = str(normalized.get("lease_id") or "")
    if not lease_id:
        raise ValueError("checkout mutation lock metadata requires lease_id")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{lock_path.name}.",
        suffix=".pending",
        dir=lock_path.parent,
    )
    temp_path = Path(temp_name)
    try:
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            fchmod(temp_fd, 0o600)
        data = json.dumps(
            normalized,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        offset = 0
        while offset < len(data):
            written = os.write(temp_fd, data[offset:])
            if written <= 0:
                raise OSError(
                    "short write while publishing checkout mutation lease"
                )
            offset += written
        os.fsync(temp_fd)
        # Capture the identity from the file descriptor we exclusively own.
        # Looking up the destination after publication could instead bind the
        # lease to a replacement installed by another process.
        published_stat = os.fstat(temp_fd)
        published_identity = (
            int(published_stat.st_dev),
            int(published_stat.st_ino),
        )
        try:
            os.link(temp_path, lock_path)
        except FileExistsError:
            existing, _identity = _read_checkout_lock(lock_path)
            return None, "lock_exists", existing
        except OSError:
            return None, "lock_publication_failed", None
        try:
            destination_stat = lock_path.stat(follow_symlinks=False)
        except OSError:
            return None, "lock_publication_lost", None
        if (
            int(destination_stat.st_dev),
            int(destination_stat.st_ino),
        ) != published_identity:
            return None, "lock_publication_lost", None
        return (
            CheckoutMutationLease(
                lock_path=lock_path,
                metadata=normalized,
                device=published_identity[0],
                inode=published_identity[1],
            ),
            "acquired",
            None,
        )
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        temp_path.unlink(missing_ok=True)


def acquire_checkout_mutation_lease(
    lock_path: Path,
    metadata: Mapping[str, Any],
    *,
    owner_active: Callable[[dict[str, Any]], bool],
    timeout_seconds: float = 0.0,
    poll_seconds: float = 0.05,
) -> tuple[
    CheckoutMutationLease | None,
    str,
    dict[str, Any] | None,
    float,
]:
    """Acquire an atomic lease, waiting boundedly and clearing valid stale owners."""

    started = time.monotonic()
    deadline = started + max(0.0, float(timeout_seconds))
    last_reason = "lock_unavailable"
    last_owner: dict[str, Any] | None = None
    cleared_owner: dict[str, Any] | None = None
    while True:
        lease, reason, existing = _try_publish_checkout_mutation_lease(
            lock_path,
            metadata,
        )
        if lease is not None:
            return (
                lease,
                "acquired",
                cleared_owner,
                max(0.0, time.monotonic() - started),
            )
        last_reason = reason
        last_owner = existing
        if (
            reason == "lock_exists"
            and existing
            and remove_inactive_checkout_mutation_lock(
                lock_path,
                expected_metadata=existing,
                owner_active=owner_active,
                timeout_seconds=min(
                    1.0,
                    max(0.0, deadline - time.monotonic()),
                ),
            )
        ):
            cleared_owner = dict(existing)
            continue
        remaining = deadline - time.monotonic()
        if reason != "lock_exists" or remaining <= 0:
            return (
                None,
                last_reason,
                last_owner,
                max(0.0, time.monotonic() - started),
            )
        time.sleep(
            min(
                max(0.001, float(poll_seconds)),
                remaining,
            )
        )


def remove_inactive_checkout_mutation_lock(
    lock_path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    owner_active: Callable[[dict[str, Any]], bool],
    timeout_seconds: float = 1.0,
) -> bool:
    """Remove only the same fully published lock after rechecking its owner."""

    expected = dict(expected_metadata)
    expected_lease_id = str(expected.get("lease_id") or "")
    if not expected:
        # An empty or malformed record can be an in-flight pre-lease publisher.
        # Valid, nonempty legacy JSON remains reclaimable for upgrade safety.
        return False
    try:
        with serialized_lock_update(
            lock_path,
            timeout_seconds=timeout_seconds,
        ):
            current, identity = _read_checkout_lock(lock_path)
            if (
                current is None
                or identity is None
                or current != expected
                or (
                    expected_lease_id
                    and str(current.get("lease_id") or "")
                    != expected_lease_id
                )
            ):
                return False
            try:
                if owner_active(current):
                    return False
            except Exception:
                # Owner validation is a safety boundary.  An inconclusive
                # liveness probe must preserve the incumbent record.
                return False
            confirmed, confirmed_identity = _read_checkout_lock(lock_path)
            if confirmed != current or confirmed_identity != identity:
                return False
            lock_path.unlink()
            return True
    except (FileNotFoundError, TimeoutError):
        return False


def release_checkout_mutation_lease(
    lease: CheckoutMutationLease,
    *,
    timeout_seconds: float = 1.0,
) -> bool:
    """Release only the exact inode and lease identity acquired by the caller."""

    try:
        with serialized_lock_update(
            lease.lock_path,
            timeout_seconds=timeout_seconds,
        ):
            current, identity = _read_checkout_lock(lease.lock_path)
            if (
                current is None
                or identity != (lease.device, lease.inode)
                or str(current.get("lease_id") or "") != lease.lease_id
            ):
                return False
            lease.lock_path.unlink()
            return True
    except (FileNotFoundError, TimeoutError):
        return False


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
