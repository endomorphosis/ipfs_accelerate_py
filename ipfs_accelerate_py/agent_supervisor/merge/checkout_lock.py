"""Reusable checkout mutation lock helpers for autonomous agent supervisors."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

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
CRASH_FENCE_RECONCILIATION_LOCK_NAME = (
    "implementation-protected-path-crash-fence-recon.lock"
)
# Exclusive crash-fence mutations must finish quickly: expensive path/objective
# scans stay outside this critical section, and only revalidation + fence
# writes run while the lease is held.
DEFAULT_CHECKOUT_MAINTENANCE_MAX_HOLD_SECONDS = 2.0
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


def crash_fence_reconciliation_lock_path(repo_root: Path) -> Path:
    """Return the short exclusive lock used for crash-fence mutations."""

    return checkout_mutation_lock_path(
        repo_root,
        lock_name=CRASH_FENCE_RECONCILIATION_LOCK_NAME,
    )


def durable_input_generation(path: Path) -> dict[str, Any]:
    """Capture an immutable-input identity used to revalidate before mutation.

    Callers scan expensive state outside the exclusive lease, then re-read this
    generation under the critical section. Any mismatch aborts the mutation so
    concurrent maintenance or a peer reconciler cannot produce a false clear or
    false incident from a stale plan.
    """

    try:
        if not path.exists():
            return {
                "path": str(path),
                "state": "missing",
            }
    except OSError as exc:
        return {
            "path": str(path),
            "state": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[-500:],
        }
    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        return {
            "path": str(path),
            "state": "missing",
        }
    except OSError as exc:
        return {
            "path": str(path),
            "state": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[-500:],
        }
    digest = ""
    size = int(getattr(stat_result, "st_size", 0) or 0)
    if stat_module_is_regular_file(stat_result) and size <= 1_048_576:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            return {
                "path": str(path),
                "state": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[-500:],
            }
        digest = hashlib.sha256(payload).hexdigest()
    return {
        "path": str(path),
        "state": "present",
        "device": int(getattr(stat_result, "st_dev", 0) or 0),
        "inode": int(getattr(stat_result, "st_ino", 0) or 0),
        "size": size,
        "mtime_ns": int(getattr(stat_result, "st_mtime_ns", 0) or 0),
        "sha256": digest,
    }


def stat_module_is_regular_file(stat_result: os.stat_result) -> bool:
    """Return whether a stat result describes a regular file."""

    import stat as stat_module

    return stat_module.S_ISREG(int(stat_result.st_mode))


def generations_match(
    expected: Mapping[str, Any] | None,
    observed: Mapping[str, Any] | None,
) -> bool:
    """Return whether two durable-input generations are identical."""

    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        return False
    return dict(expected) == dict(observed)


@dataclass
class CheckoutMaintenanceLease:
    """Bounded exclusive lease for short checkout-mutation critical sections.

    Repository-global maintenance and crash-fence reconciliation both need
    mutual exclusion, but expensive scans must not keep the exclusive section
    open. Callers scan immutable inputs first, then enter
    :meth:`exclusive_section` only for revalidation and fenced mutation.
    """

    lock_path: Path
    metadata: dict[str, Any]
    max_hold_seconds: float = DEFAULT_CHECKOUT_MAINTENANCE_MAX_HOLD_SECONDS
    _acquired_at_monotonic: float | None = field(default=None, init=False, repr=False)
    _hold_seconds: float | None = field(default=None, init=False, repr=False)
    _lease_id: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.lock_path = Path(self.lock_path)
        self.metadata = dict(self.metadata)
        self.max_hold_seconds = float(self.max_hold_seconds)
        if self.max_hold_seconds <= 0:
            raise ValueError("max_hold_seconds must be positive")
        lease_id = str(self.metadata.get("lease_id") or "").strip()
        if not lease_id:
            seed = (
                f"{os.getpid()}:{time.time_ns()}:{self.lock_path}:"
                f"{self.metadata.get('kind') or 'checkout-maintenance'}"
            )
            lease_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()
            self.metadata["lease_id"] = lease_id
        self._lease_id = lease_id
        self.metadata.setdefault("kind", "checkout-maintenance")
        self.metadata.setdefault("pid", os.getpid())
        self.metadata.setdefault("owner_script", Path(sys.argv[0]).name)

    @property
    def lease_id(self) -> str:
        return self._lease_id

    @property
    def hold_seconds(self) -> float | None:
        if self._hold_seconds is not None:
            return self._hold_seconds
        if self._acquired_at_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self._acquired_at_monotonic)

    def is_held(self) -> bool:
        return self._acquired_at_monotonic is not None and self._hold_seconds is None

    def ensure_within_hold_bound(self) -> float:
        """Return current hold seconds or raise when the exclusive bound is exceeded."""

        held = self.hold_seconds
        if held is None:
            raise RuntimeError("checkout maintenance lease is not held")
        if held > self.max_hold_seconds:
            raise RuntimeError(
                "checkout maintenance lease hold exceeded bound: "
                f"{held:.6f}s > {self.max_hold_seconds:.6f}s"
            )
        return held

    def _publish_lease(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.lock_path.with_name(
            f".{self.lock_path.name}.{self._lease_id}.tmp"
        )
        data = (
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        fd: int | None = None
        try:
            fd = os.open(
                temporary_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(fd, "wb") as stream:
                fd = None
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, self.lock_path)
            except FileExistsError:
                return False
            return True
        finally:
            if fd is not None:
                os.close(fd)
            temporary_path.unlink(missing_ok=True)

    def _load_existing(self) -> dict[str, Any] | None:
        try:
            if not self.lock_path.exists():
                return None
        except OSError:
            return None
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {"kind": "malformed", "path": str(self.lock_path)}
        if not isinstance(payload, dict):
            return {"kind": "malformed", "path": str(self.lock_path)}
        return payload

    def try_acquire(
        self,
        *,
        owner_is_active: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Attempt a non-blocking exclusive acquire with bounded metadata."""

        if self.is_held():
            return True, {
                "blocked": False,
                "reason": "checkout_maintenance_lease_already_held",
                "lease_id": self._lease_id,
                "lock_path": str(self.lock_path),
            }
        with serialized_lock_update(self.lock_path):
            for _ in range(2):
                if self._publish_lease():
                    self._acquired_at_monotonic = time.monotonic()
                    self._hold_seconds = None
                    return True, {
                        "blocked": False,
                        "reason": "checkout_maintenance_lease_acquired",
                        "lease_id": self._lease_id,
                        "lock_path": str(self.lock_path),
                        "max_hold_seconds": self.max_hold_seconds,
                    }
                existing = self._load_existing()
                if existing is None:
                    try:
                        self.lock_path.unlink()
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        return False, {
                            "blocked": True,
                            "reason": "checkout_maintenance_lease_cleanup_failed",
                            "lock_path": str(self.lock_path),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    continue
                if str(existing.get("kind") or "") == "malformed":
                    # Atomic publication cannot create a partial owned lease.
                    # Treat malformed state as unknown ownership and require
                    # explicit recovery instead of deleting a potentially live
                    # coordination record.
                    return False, {
                        "blocked": True,
                        "reason": "checkout_maintenance_lease_malformed",
                        "lock_path": str(self.lock_path),
                    }
                active = True
                if owner_is_active is not None:
                    try:
                        active = bool(owner_is_active(existing))
                    except Exception:
                        active = True
                if active:
                    return False, {
                        "blocked": True,
                        "reason": "checkout_maintenance_lease_active",
                        "lock_path": str(self.lock_path),
                        "lock_owner_pid": int(existing.get("pid") or 0),
                        "lock_owner_lease_id": str(
                            existing.get("lease_id") or ""
                        ),
                    }
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    return False, {
                        "blocked": True,
                        "reason": "checkout_maintenance_lease_cleanup_failed",
                        "lock_path": str(self.lock_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        return False, {
            "blocked": True,
            "reason": "checkout_maintenance_lease_unavailable",
            "lock_path": str(self.lock_path),
        }

    def release(self) -> dict[str, Any]:
        """Release this exclusive lease when still owned by ``lease_id``."""

        hold = self.hold_seconds
        if self._acquired_at_monotonic is not None and self._hold_seconds is None:
            self._hold_seconds = max(
                0.0, time.monotonic() - self._acquired_at_monotonic
            )
            hold = self._hold_seconds
        self._acquired_at_monotonic = None
        try:
            with serialized_lock_update(self.lock_path):
                existing = self._load_existing()
                if existing is None:
                    return {
                        "released": True,
                        "reason": "checkout_maintenance_lease_absent",
                        "hold_seconds": hold,
                        "max_hold_seconds": self.max_hold_seconds,
                    }
                if str(existing.get("lease_id") or "") != self._lease_id:
                    return {
                        "released": False,
                        "reason": "checkout_maintenance_lease_replaced",
                        "hold_seconds": hold,
                        "max_hold_seconds": self.max_hold_seconds,
                    }
                self.lock_path.unlink(missing_ok=True)
        except (OSError, RuntimeError) as exc:
            return {
                "released": False,
                "reason": "checkout_maintenance_lease_release_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "hold_seconds": hold,
                "max_hold_seconds": self.max_hold_seconds,
            }
        return {
            "released": True,
            "reason": "checkout_maintenance_lease_released",
            "hold_seconds": hold,
            "max_hold_seconds": self.max_hold_seconds,
            "within_bound": (
                hold is not None and hold <= self.max_hold_seconds
            ),
        }

    @contextmanager
    def exclusive_section(
        self,
        *,
        owner_is_active: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Hold the exclusive lease only for the critical section body.

        The yielded timing dict is updated in ``finally`` so callers that keep a
        reference observe final hold seconds even when the body returns early.
        """

        acquired, guard = self.try_acquire(owner_is_active=owner_is_active)
        if not acquired:
            raise RuntimeError(
                str(guard.get("reason") or "checkout_maintenance_lease_unavailable")
            )
        timing: dict[str, Any] = {
            "lease_id": self._lease_id,
            "lock_path": str(self.lock_path),
            "max_hold_seconds": self.max_hold_seconds,
            "hold_seconds": None,
            "within_bound": None,
        }
        body_error: BaseException | None = None
        try:
            yield timing
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            release_info = self.release()
            hold = release_info.get("hold_seconds")
            timing["hold_seconds"] = hold
            within_bound = (
                hold is not None and float(hold) <= self.max_hold_seconds
            )
            timing["within_bound"] = within_bound
            timing["release"] = release_info
            if body_error is None and hold is not None and not within_bound:
                # Surface bound violations after a clean body so callers cannot
                # miss an over-long exclusive section.
                raise RuntimeError(
                    "checkout maintenance lease hold exceeded bound: "
                    f"{float(hold):.6f}s > {self.max_hold_seconds:.6f}s"
                )

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
    if owner_script and command_line and owner_script not in command_line:
        # Module launches (``python -m package.worker``) expose the module
        # name, not the source filename, in argv.
        owner_module_stem = Path(owner_script).stem
        if not owner_module_stem or owner_module_stem not in command_line:
            return False
    return True
