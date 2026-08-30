"""Reusable supervisor status and watchdog helpers for todo daemons."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shlex
import stat as stat_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..proof.formal_verification_contracts import content_identity
from .core import (
    ManagedDaemonSpec,
    child_pids,
    first_present,
    now_utc,
    parse_timestamp,
    pid_alive,
    process_args,
    read_json,
    terminate_pid_tree,
    write_json,
)

JsonDict = dict[str, Any]

_MERGE_RESOLVER_MODULES = frozenset(
    {
        "ipfs_accelerate_py.agent_supervisor.integrations.llm_merge_resolver_fallback",
        "ipfs_accelerate_py.agent_supervisor.llm_router_merge_resolver",
    }
)
_MERGE_RESOLVER_SCRIPTS = frozenset(
    {
        "llm_merge_resolver_fallback.py",
        "llm_router_merge_resolver.py",
    }
)
_PYTHON_AGENT_WORKER_MODULES = _MERGE_RESOLVER_MODULES | frozenset(
    {
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
    }
)
_PYTHON_AGENT_WORKER_SCRIPTS = _MERGE_RESOLVER_SCRIPTS | frozenset(
    {"grok_cli_runner.py"}
)

_SEALED_RUNNER_ROUTE_FLAG = "--agent-implementation-route-json"
_SEALED_RUNNER_MEMFD_TARGET = (
    "/memfd:ipfs-accelerate-accepted-control-plane (deleted)"
)
_SHA256_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SEALED_RUNNER_PATH_RE = re.compile(r"/proc/self/fd/([0-9]+)")
_PYTHON_EXECUTABLE_RE = re.compile(
    r"python(?:[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?",
    re.IGNORECASE,
)
_SEALED_RUNNER_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_PROTECTED_ATTEMPT_LATCH_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-implementation-attempt-latch@1"
)
_PROTECTED_ATTEMPT_LATCH_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "attempt",
        "task_revision_cid",
        "board_namespace",
        "route_id",
        "invocation_id",
        "logical_attempt_id",
        "worktree_id",
        "provider_attempt_store",
        "provider_attempt_store_identity",
        "latch_id",
    }
)
_PROVIDER_RUNNER_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.provider-runner-birth@1"
)
_PROVIDER_RUNNER_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "attempt",
        "task_revision_cid",
        "workspace_path",
        "latch_id",
        "route_id",
        "invocation_id",
        "logical_attempt_id",
        "worktree_id",
        "owner_pid",
        "owner_start_ticks",
        "pid",
        "start_ticks",
        "argv_sha256",
        "executable_device",
        "executable_inode",
        "descriptor_number",
        "descriptor_device",
        "descriptor_inode",
        "descriptor_size",
        "descriptor_seals",
        "archive_sha256",
        "receipt_id",
    }
)
_ORDINARY_PROVIDER_RUNNER_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "ordinary-provider-runner-birth@1"
)
_ORDINARY_PROVIDER_RUNNER_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "attempt",
        "task_revision_cid",
        "workspace_path",
        "owner_pid",
        "owner_start_ticks",
        "pid",
        "start_time_ticks",
        "boot_id",
        "process_group_id",
        "session_id",
        "argv_sha256",
        "receipt_id",
    }
)

DEFAULT_WORKTREE_PHASES = frozenset(
    {
        "implementing",
        "merge_resolver",
        "requesting_worktree_edit",
        "retrying_worktree_edit",
        "repairing_failed_worktree_edit",
        "repairing_failed_tests_before_rollback",
    }
)


def _aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class HeartbeatSnapshot:
    """Parsed daemon heartbeat state."""

    heartbeat_at: Optional[datetime]
    age_seconds: Optional[float]
    pid: Any
    pid_alive: bool
    stale_after_seconds: float

    @property
    def fresh(self) -> bool:
        return self.age_seconds is not None and self.age_seconds <= self.stale_after_seconds

    @property
    def stale(self) -> bool:
        return self.age_seconds is not None and self.age_seconds > self.stale_after_seconds

    def to_payload(self, *, prefix: str = "heartbeat") -> JsonDict:
        return {
            f"{prefix}_at": None if self.heartbeat_at is None else self.heartbeat_at.isoformat(),
            f"{prefix}_age_seconds": None if self.age_seconds is None else round(self.age_seconds, 3),
            "daemon_pid": self.pid,
            "daemon_pid_alive": self.pid_alive,
            f"{prefix}_stale_after_seconds": self.stale_after_seconds,
            f"{prefix}_fresh": self.fresh,
            f"{prefix}_stale": self.stale,
        }


def heartbeat_snapshot(
    status: Mapping[str, Any],
    *,
    stale_after_seconds: float,
    pid_keys: Sequence[str] = ("heartbeat_pid", "pid"),
    timestamp_keys: Sequence[str] = ("heartbeat_at", "updated_at"),
    now: Optional[datetime] = None,
) -> HeartbeatSnapshot:
    """Return parsed heartbeat age, freshness, and process liveness."""

    heartbeat_at = None
    for key in timestamp_keys:
        heartbeat_at = _aware_utc(parse_timestamp(status.get(key)))
        if heartbeat_at is not None:
            break
    now_at = _aware_utc(now) or now_utc()
    age_seconds = None if heartbeat_at is None else max(0.0, (now_at - heartbeat_at).total_seconds())
    pid = None
    for key in pid_keys:
        pid = status.get(key)
        if pid:
            break
    return HeartbeatSnapshot(
        heartbeat_at=heartbeat_at,
        age_seconds=age_seconds,
        pid=pid,
        pid_alive=pid_alive(pid) if pid else False,
        stale_after_seconds=float(stale_after_seconds),
    )


def read_heartbeat_snapshot(
    path: Optional[Path],
    *,
    stale_after_seconds: float,
    pid_keys: Sequence[str] = ("heartbeat_pid", "pid"),
    timestamp_keys: Sequence[str] = ("heartbeat_at", "updated_at"),
    now: Optional[datetime] = None,
) -> HeartbeatSnapshot:
    """Read a status file and return parsed heartbeat state."""

    return heartbeat_snapshot(
        read_json(path),
        stale_after_seconds=stale_after_seconds,
        pid_keys=pid_keys,
        timestamp_keys=timestamp_keys,
        now=now,
    )


def heartbeat_is_stale(
    path: Optional[Path],
    *,
    stale_after_seconds: float,
    now: Optional[datetime] = None,
) -> bool:
    """Return whether a status file heartbeat is present and stale."""

    return read_heartbeat_snapshot(path, stale_after_seconds=stale_after_seconds, now=now).stale


def descendant_processes(root_pid: Any) -> list[JsonDict]:
    """Return descendant processes for a root pid using the shared process primitives."""

    if isinstance(root_pid, bool):
        return []
    try:
        root = int(root_pid)
    except (TypeError, ValueError):
        return []
    if root <= 1:
        return []
    stack = list(child_pids(root))
    seen: set[int] = set()
    found: list[JsonDict] = []
    while stack:
        pid = stack.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        found.append(
            {
                "pid": pid,
                "cmdline": process_args(pid),
                # ``ps`` text is useful diagnostics, but it is not an exact
                # executable identity: quoting is lost and long sealed route
                # arguments may be truncated.  Keep the procfs argv as the
                # authority used by the sealed-runner liveness check below.
                "argv": _process_command_argv(pid),
                "start_ticks": _process_start_ticks(pid),
            }
        )
        stack.extend(child_pids(pid))
    return found


def _process_command_argv(pid: Any) -> tuple[str, ...] | None:
    """Read one exact Linux argv without accepting lossy process-table text."""

    if isinstance(pid, bool):
        return None
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return None
    if process_id <= 0:
        return None
    try:
        raw = (Path("/proc") / str(process_id) / "cmdline").read_bytes()
    except OSError:
        return None
    if not raw or not raw.endswith(b"\0"):
        return None
    try:
        argv = tuple(
            item.decode("utf-8")
            for item in raw.split(b"\0")[:-1]
        )
    except UnicodeError:
        return None
    if not argv or any(not item or "\0" in item for item in argv):
        return None
    return argv


def _process_start_ticks(pid: Any) -> int | None:
    """Read Linux process start ticks, which remain stable across exec."""

    if isinstance(pid, bool):
        return None
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return None
    if process_id <= 0:
        return None
    try:
        raw = (Path("/proc") / str(process_id) / "stat").read_text(
            encoding="utf-8"
        )
        # comm is parenthesized and may itself contain spaces or ``)``.  The
        # last close-paren precedes field 3; starttime is field 22.
        close = raw.rindex(")")
        fields = raw[close + 1 :].strip().split()
        start_ticks = int(fields[19])
    except (IndexError, OSError, UnicodeError, ValueError):
        return None
    return start_ticks if start_ticks > 0 else None


def _process_birth_factors(
    pid: Any,
) -> tuple[int, int, int, int] | None:
    """Read parent, process-group, session, and start ticks atomically.

    ``None`` is positive evidence that the process no longer executes.  An
    unreadable or malformed record raises ``OSError`` so recovery never turns
    inspection failure into authority to signal a numeric PID.
    """

    if isinstance(pid, bool):
        raise OSError("boolean process id is invalid")
    try:
        process_id = int(pid)
    except (TypeError, ValueError) as exc:
        raise OSError("process id is invalid") from exc
    if process_id <= 0:
        raise OSError("process id is invalid")
    process_root = Path("/proc") / str(process_id)
    try:
        raw = (process_root / "stat").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        try:
            process_root.stat()
        except FileNotFoundError:
            return None
        except OSError:
            raise
        raise OSError("process stat is unavailable") from exc
    except (OSError, UnicodeError):
        raise
    try:
        fields = raw[raw.rindex(")") + 1 :].strip().split()
        state = fields[0]
        parent_pid = int(fields[1])
        process_group = int(fields[2])
        session = int(fields[3])
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise OSError("process birth record is malformed") from exc
    if state == "Z":
        return None
    if (
        parent_pid < 0
        or process_group <= 0
        or session <= 0
        or start_ticks <= 0
    ):
        raise OSError("process birth record is malformed")
    return parent_pid, process_group, session, start_ticks


def _system_boot_id() -> str:
    """Return the current boot identity or raise when it is unknowable."""

    try:
        value = (
            Path("/proc/sys/kernel/random/boot_id")
            .read_text(encoding="ascii")
            .strip()
        )
    except UnicodeError as exc:
        raise OSError("system boot identity is malformed") from exc
    if (
        not value
        or len(value) > 128
        or any(character in value for character in "\0\n\r")
    ):
        raise OSError("system boot identity is unavailable")
    return value


def _ordinary_provider_runner_observation(
    pid: int,
) -> tuple[str, tuple[int, int, int, int] | None, str | None]:
    """Observe one ordinary child without collapsing inspection failure.

    The boot identity brackets the procfs reads.  A missing process is a
    positive observation; a live process whose argv cannot be read is not.
    """

    boot_before = _system_boot_id()
    birth = _process_birth_factors(pid)
    argv_digest: str | None = None
    if birth is not None:
        argv = _process_command_argv(pid)
        if argv is None:
            raise OSError("ordinary provider runner argv is unavailable")
        argv_digest = _argv_sha256(argv)
        if not argv_digest:
            raise OSError("ordinary provider runner argv is malformed")
    boot_after = _system_boot_id()
    if boot_after != boot_before:
        raise OSError("system boot identity drifted during inspection")
    return boot_before, birth, argv_digest


def _single_argv_value(argv: Sequence[str], flag: str) -> str | None:
    positions = [index for index, item in enumerate(argv) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return None
    value = argv[positions[0] + 1]
    if not value or value.startswith("--") or any(
        character in value for character in "\0\n\r"
    ):
        return None
    return value


def _normalized_absolute_path(value: object) -> str:
    if not isinstance(value, str) or not value or any(
        character in value for character in "\0\n\r"
    ):
        return ""
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        return ""
    return value


def _active_implementation_identity(
    status: Mapping[str, Any],
) -> tuple[str, int, str, str] | None:
    task_id = status.get("active_task_id")
    attempt = status.get("active_attempt")
    task_revision_cid = status.get("active_task_cid")
    workspace = _normalized_absolute_path(status.get("active_worktree_path"))
    if (
        status.get("implementation_in_progress") is not True
        or not isinstance(task_id, str)
        or not task_id
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
        or not isinstance(task_revision_cid, str)
        or not task_revision_cid
        or not workspace
    ):
        return None
    return task_id, attempt, task_revision_cid, workspace


def _argv_sha256(argv: Sequence[str]) -> str:
    try:
        encoded = b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
    except (AttributeError, UnicodeError):
        return ""
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _process_parent_pid(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rindex(")") + 1 :].strip().split()
        parent = int(fields[1])
    except (IndexError, OSError, UnicodeError, ValueError):
        return None
    return parent if parent > 0 else None


def validated_protected_attempt_latch(
    status: Mapping[str, Any],
    *,
    task_id: str,
    attempt: int,
    task_revision_cid: str,
) -> Mapping[str, Any] | None:
    """Return the exact self-addressed latch for one active attempt."""

    attempts = status.get("protected_implementation_attempts")
    key = content_identity(
        {
            "task_id": task_id,
            "attempt": attempt,
            "task_revision_cid": task_revision_cid,
        }
    )
    latch = attempts.get(key) if isinstance(attempts, Mapping) else None
    if (
        not isinstance(latch, Mapping)
        or set(latch) != _PROTECTED_ATTEMPT_LATCH_FIELDS
        or latch.get("schema") != _PROTECTED_ATTEMPT_LATCH_SCHEMA
        or latch.get("task_id") != task_id
        or isinstance(latch.get("attempt"), bool)
        or not isinstance(latch.get("attempt"), int)
        or latch.get("attempt", 0) < 1
        or latch.get("attempt") != attempt
        or latch.get("task_revision_cid") != task_revision_cid
        or any(
            not isinstance(latch.get(name), str) or not latch.get(name)
            for name in (
                "board_namespace",
                "route_id",
                "invocation_id",
                "logical_attempt_id",
                "worktree_id",
                "provider_attempt_store",
                "provider_attempt_store_identity",
                "latch_id",
            )
        )
    ):
        return None
    body = {name: latch[name] for name in latch if name != "latch_id"}
    return latch if latch.get("latch_id") == content_identity(body) else None


def _positive_int(value: object, *, minimum: int = 1) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _provider_runner_receipt_matches(
    item: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    daemon_pid: int,
    argv: Sequence[str],
    descriptor_number: int,
) -> bool:
    receipt = status.get("active_provider_runner")
    active = _active_implementation_identity(status)
    if (
        active is None
        or not isinstance(receipt, Mapping)
        or set(receipt) != _PROVIDER_RUNNER_RECEIPT_FIELDS
        or receipt.get("schema") != _PROVIDER_RUNNER_RECEIPT_SCHEMA
    ):
        return False
    task_id, attempt, revision, workspace = active
    pid = item.get("pid")
    start_ticks = item.get("start_ticks")
    latch = validated_protected_attempt_latch(
        status,
        task_id=task_id,
        attempt=attempt,
        task_revision_cid=revision,
    )


    numeric_fields = {
        name: _positive_int(
            receipt.get(name),
            minimum=3 if name == "descriptor_number" else 1,
        )
        for name in (
            "owner_pid",
            "owner_start_ticks",
            "pid",
            "start_ticks",
            "executable_device",
            "executable_inode",
            "descriptor_number",
            "descriptor_device",
            "descriptor_inode",
            "descriptor_size",
        )
    }
    seals = receipt.get("descriptor_seals")
    required_seals = (
        fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SEAL
    )
    if (
        latch is None
        or any(value is None for value in numeric_fields.values())
        or numeric_fields["descriptor_size"] > _SEALED_RUNNER_MAX_ARCHIVE_BYTES
        or isinstance(seals, bool)
        or not isinstance(seals, int)
        or seals < 0
        or seals & required_seals != required_seals
        or receipt.get("task_id") != task_id
        or isinstance(receipt.get("attempt"), bool)
        or not isinstance(receipt.get("attempt"), int)
        or receipt.get("attempt") != attempt
        or receipt.get("task_revision_cid") != revision
        or receipt.get("workspace_path") != workspace
        or receipt.get("latch_id") != latch.get("latch_id")
        or receipt.get("route_id") != latch.get("route_id")
        or receipt.get("invocation_id") != latch.get("invocation_id")
        or receipt.get("logical_attempt_id")
        != latch.get("logical_attempt_id")
        or receipt.get("worktree_id") != latch.get("worktree_id")
        or receipt.get("pid") != pid
        or receipt.get("start_ticks") != start_ticks
        or receipt.get("owner_pid") != daemon_pid
        or receipt.get("owner_start_ticks") != _process_start_ticks(daemon_pid)
        or receipt.get("argv_sha256") != _argv_sha256(argv)
        or receipt.get("descriptor_number") != descriptor_number
        or _SHA256_ID_RE.fullmatch(str(receipt.get("argv_sha256") or ""))
        is None
        or _SHA256_ID_RE.fullmatch(str(receipt.get("archive_sha256") or ""))
        is None
        or receipt.get("receipt_id")
        != content_identity(
            {
                key: receipt[key]
                for key in _PROVIDER_RUNNER_RECEIPT_FIELDS
                if key != "receipt_id"
            }
        )
        or _process_parent_pid(numeric_fields["pid"]) != daemon_pid
    ):
        return False
    fd_path = Path("/proc") / str(pid) / "fd" / str(descriptor_number)
    owner_fd_path = (
        Path("/proc") / str(daemon_pid) / "fd" / str(descriptor_number)
    )
    exe_path = Path("/proc") / str(pid) / "exe"
    try:
        target_before = os.readlink(fd_path)
        owner_target_before = os.readlink(owner_fd_path)
        fd_before = os.stat(fd_path)
        owner_fd_before = os.stat(owner_fd_path)
        exe_before = os.stat(exe_path)
        fd = os.open(fd_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            seals = int(fcntl.fcntl(fd, fcntl.F_GET_SEALS))
            opened = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError:
        return False
    if (
        target_before != _SEALED_RUNNER_MEMFD_TARGET
        or owner_target_before != target_before
        or not stat_module.S_ISREG(fd_before.st_mode)
        or fd_before.st_uid != os.geteuid()
        or fd_before.st_nlink != 0
        or (fd_before.st_dev, fd_before.st_ino, fd_before.st_size)
        != (
            receipt.get("descriptor_device"),
            receipt.get("descriptor_inode"),
            receipt.get("descriptor_size"),
        )
        or (owner_fd_before.st_dev, owner_fd_before.st_ino, owner_fd_before.st_size)
        != (fd_before.st_dev, fd_before.st_ino, fd_before.st_size)
        or (opened.st_dev, opened.st_ino, opened.st_size)
        != (fd_before.st_dev, fd_before.st_ino, fd_before.st_size)
        or seals != receipt.get("descriptor_seals")
        or (exe_before.st_dev, exe_before.st_ino)
        != (
            receipt.get("executable_device"),
            receipt.get("executable_inode"),
        )
    ):
        return False
    try:
        parent_after = _process_parent_pid(int(pid))
        owner_start_after = _process_start_ticks(daemon_pid)
        start_after = _process_start_ticks(pid)
        argv_after = _process_command_argv(pid)
        target_after = os.readlink(fd_path)
        owner_target_after = os.readlink(owner_fd_path)
        fd_after = os.stat(fd_path)
        owner_fd_after = os.stat(owner_fd_path)
        exe_after = os.stat(exe_path)
    except OSError:
        return False
    return bool(
        parent_after == daemon_pid
        and owner_start_after == receipt.get("owner_start_ticks")
        and start_after == start_ticks
        and argv_after == tuple(argv)
        and target_after == target_before
        and owner_target_after == owner_target_before
        and (fd_after.st_dev, fd_after.st_ino, fd_after.st_size)
        == (fd_before.st_dev, fd_before.st_ino, fd_before.st_size)
        and (owner_fd_after.st_dev, owner_fd_after.st_ino, owner_fd_after.st_size)
        == (owner_fd_before.st_dev, owner_fd_before.st_ino, owner_fd_before.st_size)
        and (exe_after.st_dev, exe_after.st_ino)
        == (exe_before.st_dev, exe_before.st_ino)
    )


def _validated_ordinary_provider_runner_receipt(
    status: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str]:
    """Validate one self-addressed, task-bound ordinary birth receipt."""

    receipt = status.get("active_provider_runner")
    if not isinstance(receipt, Mapping) or not receipt:
        return None, "ordinary_provider_runner_receipt_absent"
    if receipt.get("schema") != _ORDINARY_PROVIDER_RUNNER_RECEIPT_SCHEMA:
        return None, "ordinary_provider_runner_receipt_not_applicable"
    active = _active_implementation_identity(status)
    if (
        active is None
        or set(receipt) != _ORDINARY_PROVIDER_RUNNER_RECEIPT_FIELDS
    ):
        return None, "ordinary_provider_runner_receipt_shape_invalid"
    task_id, attempt, revision, workspace = active
    numeric = {
        name: _positive_int(receipt.get(name))
        for name in (
            "owner_pid",
            "owner_start_ticks",
            "pid",
            "start_time_ticks",
            "process_group_id",
            "session_id",
        )
    }
    boot_id = receipt.get("boot_id")
    if (
        any(value is None for value in numeric.values())
        or numeric["pid"] <= 1
        or numeric["process_group_id"] != numeric["pid"]
        or numeric["session_id"] != numeric["pid"]
        or not isinstance(boot_id, str)
        or not boot_id
        or len(boot_id) > 128
        or any(character in boot_id for character in "\0\n\r")
        or receipt.get("task_id") != task_id
        or isinstance(receipt.get("attempt"), bool)
        or not isinstance(receipt.get("attempt"), int)
        or receipt.get("attempt") != attempt
        or receipt.get("task_revision_cid") != revision
        or receipt.get("workspace_path") != workspace
        or _SHA256_ID_RE.fullmatch(
            str(receipt.get("argv_sha256") or "")
        )
        is None
        or receipt.get("receipt_id")
        != content_identity(
            {
                key: receipt[key]
                for key in _ORDINARY_PROVIDER_RUNNER_RECEIPT_FIELDS
                if key != "receipt_id"
            }
        )
    ):
        return None, "ordinary_provider_runner_receipt_binding_invalid"
    return receipt, "ordinary_provider_runner_receipt_valid"


def _ordinary_provider_runner_receipt_matches(
    item: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    daemon_pid: int,
    argv: Sequence[str],
) -> bool:
    """Match an ordinary worker using exact persisted birth evidence."""

    receipt, reason = _validated_ordinary_provider_runner_receipt(status)
    if receipt is None or reason != "ordinary_provider_runner_receipt_valid":
        return False
    pid = item.get("pid")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 1
        or receipt.get("pid") != pid
        or item.get("start_ticks") != receipt.get("start_time_ticks")
        or receipt.get("owner_pid") != daemon_pid
        or receipt.get("owner_start_ticks")
        != _process_start_ticks(daemon_pid)
        or receipt.get("argv_sha256") != _argv_sha256(argv)
    ):
        return False
    try:
        before = _ordinary_provider_runner_observation(pid)
        after = _ordinary_provider_runner_observation(pid)
    except OSError:
        return False
    expected = (
        str(receipt["boot_id"]),
        (
            daemon_pid,
            int(receipt["process_group_id"]),
            int(receipt["session_id"]),
            int(receipt["start_time_ticks"]),
        ),
        str(receipt["argv_sha256"]),
    )
    return bool(before == expected and after == expected)


def fence_ordinary_provider_runner(
    status: Mapping[str, Any],
    *,
    grace_seconds: float = 1.0,
) -> JsonDict:
    """Fence one exact ordinary provider birth without scanning by argv.

    An absent or sealed receipt is outside this helper's authority.  A
    malformed ordinary receipt or an inspection failure is an explicit unsafe
    result; callers must not clear the attempt or restart work past it.
    """

    raw_receipt = status.get("active_provider_runner")
    if not raw_receipt:
        return {
            "applicable": False,
            "safe_to_restart": True,
            "fenced": False,
            "reason": "ordinary_provider_runner_receipt_absent",
        }
    if (
        isinstance(raw_receipt, Mapping)
        and raw_receipt.get("schema") == _PROVIDER_RUNNER_RECEIPT_SCHEMA
    ):
        return {
            "applicable": False,
            "safe_to_restart": True,
            "fenced": False,
            "reason": "sealed_provider_runner_receipt_not_applicable",
        }
    if (
        not isinstance(raw_receipt, Mapping)
        or raw_receipt.get("schema")
        != _ORDINARY_PROVIDER_RUNNER_RECEIPT_SCHEMA
    ):
        active_attempt = bool(
            status.get("implementation_in_progress") is True
            or _active_implementation_identity(status) is not None
        )
        return {
            "applicable": active_attempt,
            "safe_to_restart": not active_attempt,
            "fenced": False,
            "reason": (
                "active_provider_runner_receipt_schema_unknown"
                if active_attempt
                else "ordinary_provider_runner_receipt_not_applicable"
            ),
        }
    receipt, reason = _validated_ordinary_provider_runner_receipt(status)
    if receipt is None:
        return {
            "applicable": True,
            "safe_to_restart": False,
            "fenced": False,
            "reason": reason,
        }
    pid = int(receipt["pid"])
    try:
        before = _ordinary_provider_runner_observation(pid)
        rechecked = _ordinary_provider_runner_observation(pid)
    except OSError:
        return {
            "applicable": True,
            "safe_to_restart": False,
            "fenced": False,
            "pid": pid,
            "reason": "ordinary_provider_runner_liveness_unknown",
        }
    if rechecked != before:
        return {
            "applicable": True,
            "safe_to_restart": False,
            "fenced": False,
            "pid": pid,
            "reason": "ordinary_provider_runner_recheck_drifted",
        }
    boot_id, birth, argv_digest = before
    recorded_boot = str(receipt["boot_id"])
    # A different start or boot proves that this numeric PID no longer names
    # the recorded process.  Never signal the replacement.
    if boot_id != recorded_boot:
        return {
            "applicable": True,
            "safe_to_restart": True,
            "fenced": False,
            "pid": pid,
            "pid_reused": True,
            "reason": "ordinary_provider_runner_recorded_birth_dead",
        }
    parent_pid = 0
    if birth is not None:
        parent_pid, process_group, session, start_ticks = birth
        if start_ticks != int(receipt["start_time_ticks"]):
            return {
                "applicable": True,
                "safe_to_restart": True,
                "fenced": False,
                "pid": pid,
                "pid_reused": True,
                "reason": "ordinary_provider_runner_recorded_birth_dead",
            }
        if (
            process_group != int(receipt["process_group_id"])
            or session != int(receipt["session_id"])
            or argv_digest != receipt["argv_sha256"]
        ):
            return {
                "applicable": True,
                "safe_to_restart": False,
                "fenced": False,
                "pid": pid,
                "reason": "ordinary_provider_runner_exact_identity_mismatch",
            }
    fenced = terminate_pid_tree(
        pid,
        grace_seconds=max(0.0, float(grace_seconds)),
        freeze_first=True,
        require_gone=True,
        owned_process_group_id=int(receipt["process_group_id"]),
        expected_root_start_time_ticks=int(receipt["start_time_ticks"]),
        strict_timeout_seconds=max(0.2, float(grace_seconds)),
    )
    try:
        remaining_boot, remaining, _remaining_argv = (
            _ordinary_provider_runner_observation(pid)
        )
    except OSError:
        return {
            "applicable": True,
            "safe_to_restart": False,
            "fenced": False,
            "pid": pid,
            "reason": "ordinary_provider_runner_post_fence_unknown",
        }
    exact_still_alive = bool(
        remaining is not None
        and remaining[3] == receipt.get("start_time_ticks")
        and remaining_boot == recorded_boot
    )
    safe = bool(fenced and not exact_still_alive)
    return {
        "applicable": True,
        "safe_to_restart": safe,
        "fenced": safe,
        "pid": pid,
        "parent_pid_before_fence": parent_pid,
        "reason": (
            "ordinary_provider_runner_exact_birth_fenced"
            if safe
            else "ordinary_provider_runner_exact_birth_fence_failed"
        ),
    }


def _sealed_agent_worker_process(
    item: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    *,
    daemon_pid: int,
) -> bool:
    """Recognize only a task-bound accepted-generation sealed runner."""

    if status is None:
        return False
    active = _active_implementation_identity(status)
    argv = item.get("argv")
    pid = item.get("pid")
    if (
        active is None
        or not isinstance(argv, (tuple, list))
        or any(not isinstance(value, str) or not value for value in argv)
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or len(argv) < 8
        or _PYTHON_EXECUTABLE_RE.fullmatch(os.path.basename(argv[0])) is None
        or argv[1] != "-I"
        or argv.count("-I") != 1
    ):
        return False
    descriptor_match = _SEALED_RUNNER_PATH_RE.fullmatch(argv[2])
    if descriptor_match is None:
        return False
    descriptor_number = int(descriptor_match.group(1))
    workspace_flag = _single_argv_value(argv, "--workspace")
    task_id, attempt, revision, active_workspace = active
    if (
        argv.count("--workspace") != 1
        or _normalized_absolute_path(workspace_flag) != active_workspace
    ):
        return False
    route_count = argv.count(_SEALED_RUNNER_ROUTE_FLAG)
    if route_count != 1 or "--agent-implementation-recovery-json" in argv:
        return False

    return _provider_runner_receipt_matches(
        item,
        status,
        daemon_pid=daemon_pid,
        argv=argv,
        descriptor_number=descriptor_number,
    )


def _recognized_agent_worker_process(
    item: Mapping[str, Any],
    status: Mapping[str, Any] | None,
    *,
    daemon_pid: int,
) -> bool:
    """Recognize a worker without letting argv bypass an active receipt."""

    if _sealed_agent_worker_process(
        item,
        status,
        daemon_pid=daemon_pid,
    ):
        return True
    command_match = _is_agent_worker_command(
        str(item.get("cmdline") or "")
    )
    if not command_match:
        return False
    if status is None or _active_implementation_identity(status) is None:
        # Legacy diagnostics and non-task phases retain their historical
        # recognition.  Once an exact implementation attempt is active, argv
        # alone can no longer establish provider liveness.
        return True
    if str(status.get("active_phase") or "") != "implementing":
        return True
    argv = item.get("argv")
    if (
        not isinstance(argv, (tuple, list))
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        return False
    return _ordinary_provider_runner_receipt_matches(
        item,
        status,
        daemon_pid=daemon_pid,
        argv=argv,
    )


def active_codex_exec_workers(
    root_pid: Any,
    current_status: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Return active task-bound agent workers below a daemon pid."""

    workers: list[JsonDict] = []
    if isinstance(root_pid, bool):
        return workers
    try:
        daemon_pid = int(root_pid)
    except (TypeError, ValueError):
        return workers
    if daemon_pid <= 1:
        return workers
    for item in descendant_processes(daemon_pid):
        if _recognized_agent_worker_process(
            item,
            current_status,
            daemon_pid=daemon_pid,
        ):
            workers.append(item)
    return workers


def _is_agent_worker_command(cmdline: str) -> bool:
    try:
        tokens = shlex.split(cmdline)
    except ValueError:
        tokens = cmdline.split()
    if not tokens:
        return False

    executable = os.path.basename(tokens[0]).lower()
    lowered = [token.lower() for token in tokens]
    if executable == "codex":
        # Codex accepts global safety/configuration options before the
        # subcommand (for example ``codex --ask-for-approval never
        # --disable browser_use -c web_search=\"disabled\" exec ...``).
        # Requiring ``exec`` to be argv[1] makes the watchdog miss those
        # workers and recycle a healthy implementation lane.
        return "exec" in lowered[1:]
    if executable == "copilot":
        return True
    if executable == "grok":
        # Grok's CLI runs the implementation prompt directly without a
        # subcommand.  Treating it as an ordinary descendant makes the
        # watchdog report a healthy worker as missing and can trigger false
        # worktree-without-worker recovery.
        return True
    if executable == "node" and len(tokens) > 1:
        wrapped_executable = os.path.basename(tokens[1]).lower()
        if wrapped_executable in {"copilot", "grok"}:
            return True
        if wrapped_executable == "codex":
            return "exec" in lowered[2:]
        return False
    if executable in {"bash", "sh"} and len(tokens) > 1:
        return os.path.basename(tokens[1]).lower() == "llm_merge_resolver_fallback.sh"
    if executable in _MERGE_RESOLVER_SCRIPTS:
        return True
    if not executable.startswith("python"):
        return False

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-m":
            return (
                index + 1 < len(tokens)
                and tokens[index + 1].lower() in _PYTHON_AGENT_WORKER_MODULES
            )
        if token.startswith("-"):
            index += 1
            continue
        return os.path.basename(token).lower() in _PYTHON_AGENT_WORKER_SCRIPTS
    return False


def worktree_phase_worker_status(
    current: Mapping[str, Any],
    daemon_pid: Any = None,
    threshold_seconds: float = 0.0,
    *,
    phases: frozenset[str] = DEFAULT_WORKTREE_PHASES,
    now: Optional[datetime] = None,
) -> JsonDict:
    """Report whether a worktree-edit phase appears stuck without a worker."""

    phase = str(first_present(current.get("active_phase"), current.get("phase")) or "")
    phase_detail = str(
        first_present(
            current.get("active_phase_detail"),
            current.get("phase_detail"),
        )
        or ""
    )
    started_value = first_present(
        current.get("active_phase_started_at"),
        current.get("phase_started_at"),
        current.get("active_phase_updated_at"),
        current.get("phase_updated_at"),
    )
    started = _aware_utc(parse_timestamp(started_value))
    tracking_generation = content_identity(
        {
            "phase": phase,
            "phase_detail": phase_detail,
            "phase_started_at": (
                "" if started is None else started.isoformat()
            ),
        }
    )
    if phase not in phases:
        return {
            "required": False,
            "phase": phase,
            "tracking_generation": tracking_generation,
        }
    now_at = _aware_utc(now) or now_utc()
    age = None if started is None else max(0.0, (now_at - started).total_seconds())
    root_pid = daemon_pid or current.get("heartbeat_pid") or current.get("pid")
    try:
        daemon_pid_value = int(root_pid)
    except (TypeError, ValueError):
        daemon_pid_value = 0
    descendants = descendant_processes(root_pid)
    workers = [
        item
        for item in descendants
        if _recognized_agent_worker_process(
            item,
            current,
            daemon_pid=daemon_pid_value,
        )
    ]
    stalled = bool(age is not None and threshold_seconds > 0 and age >= threshold_seconds and not workers)
    return {
        "required": True,
        "phase": phase,
        "tracking_generation": tracking_generation,
        "phase_age_seconds": None if age is None else round(age, 3),
        "threshold_seconds": float(threshold_seconds),
        "active_worker_pids": [item.get("pid") for item in workers],
        "active_worker_count": len(workers),
        "descendant_count": len(descendants),
        "stalled_without_active_worker": stalled,
    }


@dataclass(frozen=True)
class SupervisorStatusContext:
    """Reusable context for rendering supervisor status payloads."""

    spec: ManagedDaemonSpec
    schema: str = ""
    static_fields: Mapping[str, Any] = field(default_factory=dict)

    def payload(
        self,
        status: str,
        *,
        run_id: str = "",
        log_path: str = "",
        daemon_pid: Any = None,
        restart_count: int = 0,
        last_exit_code: Any = None,
        supervisor_pid: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> JsonDict:
        return build_supervisor_status_payload(
            self.spec,
            status=status,
            schema=self.schema,
            static_fields=self.static_fields,
            run_id=run_id,
            log_path=log_path,
            daemon_pid=daemon_pid,
            restart_count=restart_count,
            last_exit_code=last_exit_code,
            supervisor_pid=supervisor_pid,
            extra=extra,
        )

    def write(self, status: str, **kwargs: Any) -> JsonDict:
        payload = self.payload(status, **kwargs)
        path = self.spec.resolve(self.spec.supervisor_status_path)
        assert path is not None
        write_json(path, payload)
        return payload


def build_supervisor_status_payload(
    spec: ManagedDaemonSpec,
    *,
    status: str,
    schema: str = "",
    static_fields: Optional[Mapping[str, Any]] = None,
    run_id: str = "",
    log_path: str = "",
    daemon_pid: Any = None,
    restart_count: int = 0,
    last_exit_code: Any = None,
    supervisor_pid: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Build the common supervisor JSON status payload."""

    payload: JsonDict = {
        "schema": schema or f"{spec.schema}.supervisor",
        "status": status,
        "updated_at": now_utc().isoformat(),
        "repo_root": str(spec.repo_root),
        "supervisor_pid": os.getpid() if supervisor_pid is None else supervisor_pid,
        "daemon_pid": daemon_pid,
        "restart_count": int(restart_count),
        "run_id": run_id,
        "log_path": log_path,
        "current_status_path": spec.repo_relative(spec.status_path),
        "progress_path": spec.repo_relative(spec.progress_path),
        "child_pid_path": spec.repo_relative(spec.child_pid_path),
        "supervisor_lock_path": spec.repo_relative(spec.supervisor_lock_path),
    }
    if last_exit_code is not None:
        payload["last_exit_code"] = last_exit_code
    if static_fields:
        payload.update(dict(static_fields))
    if extra:
        payload.update(dict(extra))
    return payload
