"""Reusable supervisor status and watchdog helpers for todo daemons."""

from __future__ import annotations

import fcntl
import hashlib
import json
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
_ORDINARY_GROK_ORPHAN_FENCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "ordinary-grok-orphan-container-fence@1"
)
_ORDINARY_GROK_TEMP_ROOT = Path("/tmp")
_ORDINARY_GROK_LEASE_RE = re.compile(
    r"asref-grok-container-[A-Za-z0-9._-]{1,128}"
)
_ORDINARY_GROK_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
_ORDINARY_GROK_CONTAINER_NAME_RE = re.compile(
    r"ipfs-accelerate-grok-([0-9]+)-[0-9a-f]{32}"
)
_ORDINARY_GROK_INSPECTION_MAX_BYTES = 256 * 1024
_ORDINARY_GROK_MAX_TEMP_ENTRIES = 4096
_ORDINARY_GROK_MAX_LEASE_ENTRIES = 256
_ORDINARY_GROK_MAX_MASK_ENTRIES = 1024

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


def _ordinary_grok_docker_binary() -> str:
    """Resolve the trusted local Docker CLI without consulting a mutable tag."""

    for candidate in (Path("/usr/bin/docker"), Path("/usr/local/bin/docker")):
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            continue
        if (
            resolved in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
            and resolved.is_file()
            and os.access(resolved, os.X_OK)
            and metadata.st_uid == 0
            and not metadata.st_mode & 0o022
        ):
            return str(resolved)
    return ""


def _ordinary_grok_docker_query(
    command: Sequence[str],
) -> tuple[int, bytes, bytes]:
    """Run one bounded Docker control query with the canonical environment."""

    from ..runtime import grok_cli_runner

    return grok_cli_runner._bounded_docker_query(command, timeout=10.0)


def _ordinary_grok_file_identity(path: Path) -> tuple[int, ...]:
    metadata = os.lstat(path)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
    )


def _ordinary_grok_lease_records() -> list[JsonDict]:
    """Read only uid-owned, private legacy Grok lease records."""

    temporary_root = _ORDINARY_GROK_TEMP_ROOT
    if not temporary_root.is_absolute():
        raise ValueError("ordinary Grok temporary root is invalid")
    try:
        entries = []
        with os.scandir(temporary_root) as iterator:
            for inspected_count, entry in enumerate(iterator, start=1):
                if inspected_count > _ORDINARY_GROK_MAX_TEMP_ENTRIES:
                    raise ValueError(
                        "ordinary Grok temporary-root enumeration is oversized"
                    )
                if _ORDINARY_GROK_LEASE_RE.fullmatch(entry.name) is not None:
                    entries.append(entry)
                    if len(entries) > _ORDINARY_GROK_MAX_LEASE_ENTRIES:
                        raise ValueError(
                            "ordinary Grok lease enumeration is oversized"
                        )
    except OSError as exc:
        raise ValueError("ordinary Grok lease enumeration failed") from exc
    entries.sort(key=lambda item: item.name)
    records: list[JsonDict] = []
    for entry in entries:
        lease_root = temporary_root / entry.name
        try:
            root_stat = os.lstat(lease_root)
        except OSError as exc:
            raise ValueError("ordinary Grok lease inspection failed") from exc
        if root_stat.st_uid != os.geteuid():
            continue
        docker_config = lease_root / "docker-config"
        cidfile = lease_root / "container.cid"
        mask_root = lease_root / "provider-masks"
        try:
            config_stat = os.lstat(docker_config)
            cid_stat = os.lstat(cidfile)
            mask_stat = os.lstat(mask_root)
        except OSError as exc:
            raise ValueError("ordinary Grok owned lease is incomplete") from exc
        if (
            not stat_module.S_ISDIR(root_stat.st_mode)
            or stat_module.S_IMODE(root_stat.st_mode) != 0o700
            or not stat_module.S_ISDIR(config_stat.st_mode)
            or config_stat.st_uid != os.geteuid()
            or stat_module.S_IMODE(config_stat.st_mode) != 0o700
            or not stat_module.S_ISREG(cid_stat.st_mode)
            or cid_stat.st_uid != os.geteuid()
            or cid_stat.st_nlink != 1
            # Docker creates cidfiles as 0664 on this host.  The containing
            # uid-owned 0700 directory is the privacy boundary; reject only
            # executable or special-mode cidfiles here.
            or stat_module.S_IMODE(cid_stat.st_mode) & 0o7111
            or not stat_module.S_ISDIR(mask_stat.st_mode)
            or mask_stat.st_uid != os.geteuid()
            or stat_module.S_IMODE(mask_stat.st_mode) != 0o700
        ):
            raise ValueError("ordinary Grok owned lease is not private")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        try:
            descriptor = os.open(cidfile, flags)
            try:
                opened_stat = os.fstat(descriptor)
                encoded_cid = os.read(descriptor, 65)
                trailing = os.read(descriptor, 1)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise ValueError("ordinary Grok cidfile is unreadable") from exc
        try:
            container_id = encoded_cid.decode("ascii")
        except UnicodeError as exc:
            raise ValueError("ordinary Grok cidfile is malformed") from exc
        if (
            trailing
            or _ORDINARY_GROK_CONTAINER_ID_RE.fullmatch(container_id) is None
            or (
                opened_stat.st_dev,
                opened_stat.st_ino,
                opened_stat.st_uid,
                opened_stat.st_mode,
                opened_stat.st_nlink,
                opened_stat.st_size,
            )
            != (
                cid_stat.st_dev,
                cid_stat.st_ino,
                cid_stat.st_uid,
                cid_stat.st_mode,
                cid_stat.st_nlink,
                cid_stat.st_size,
            )
        ):
            raise ValueError("ordinary Grok cidfile is malformed")
        mask_sources: list[str] = []
        try:
            mask_entries = []
            with os.scandir(mask_root) as iterator:
                for mask_count, mask_entry in enumerate(iterator, start=1):
                    if mask_count > _ORDINARY_GROK_MAX_MASK_ENTRIES:
                        raise ValueError(
                            "ordinary Grok provider masks are oversized"
                        )
                    mask_entries.append(mask_entry)
        except OSError as exc:
            raise ValueError("ordinary Grok provider masks are unreadable") from exc
        mask_entries.sort(key=lambda item: item.name)
        for mask_entry in mask_entries:
            mask_path = mask_root / mask_entry.name
            try:
                metadata = os.lstat(mask_path)
            except OSError as exc:
                raise ValueError("ordinary Grok provider mask drifted") from exc
            if (
                not mask_entry.name.isdecimal()
                or metadata.st_uid != os.geteuid()
                or stat_module.S_IMODE(metadata.st_mode) != 0
                or not (
                    stat_module.S_ISREG(metadata.st_mode)
                    or stat_module.S_ISDIR(metadata.st_mode)
                )
            ):
                raise ValueError("ordinary Grok provider mask drifted")
            mask_sources.append(str(mask_path))
        cas_marker = lease_root / "cas-owned"
        try:
            os.lstat(cas_marker)
            cas_owned = True
        except FileNotFoundError:
            cas_owned = False
        except OSError as exc:
            raise ValueError("ordinary Grok CAS marker is unreadable") from exc
        records.append(
            {
                "lease_root": str(lease_root),
                "docker_config": str(docker_config),
                "container_id": container_id,
                "cas_owned": cas_owned,
                "mask_sources": mask_sources,
                "filesystem_identity": [
                    list(_ordinary_grok_file_identity(path))
                    for path in (lease_root, docker_config, cidfile, mask_root)
                    + tuple(Path(item) for item in mask_sources)
                ],
            }
        )
    return records


def _ordinary_grok_strict_json(value: bytes) -> object:
    def closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    try:
        return json.loads(
            value,
            object_pairs_hook=closed_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Docker inspection JSON is invalid") from exc


def _ordinary_grok_container_inspection(
    prefix: Sequence[str], container_id: str
) -> Mapping[str, Any] | None:
    returncode, stdout, stderr = _ordinary_grok_docker_query(
        [*prefix, "container", "inspect", container_id]
    )
    if any(
        len(value) > _ORDINARY_GROK_INSPECTION_MAX_BYTES
        for value in (stdout, stderr)
    ):
        raise ValueError("Docker inspection was oversized")
    if returncode != 0:
        if _ordinary_grok_container_absent(prefix, container_id):
            return None
        raise ValueError("Docker inspection failed")
    payload = _ordinary_grok_strict_json(stdout)
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], Mapping)
    ):
        raise ValueError("Docker inspection shape is invalid")
    return payload[0]


def _ordinary_grok_container_absent(
    prefix: Sequence[str], container_id: str
) -> bool:
    returncode, stdout, stderr = _ordinary_grok_docker_query(
        [
            *prefix,
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            f"id={container_id}",
            "--format",
            "{{.ID}}",
        ]
    )
    if (
        returncode != 0
        or len(stdout) > _ORDINARY_GROK_INSPECTION_MAX_BYTES
        or len(stderr) > _ORDINARY_GROK_INSPECTION_MAX_BYTES
    ):
        raise ValueError("Docker absence verification failed")
    observed = stdout.decode("ascii", errors="strict").split()
    if any(
        _ORDINARY_GROK_CONTAINER_ID_RE.fullmatch(item) is None
        for item in observed
    ):
        raise ValueError("Docker absence response is invalid")
    return not observed


def _ordinary_grok_inspection_projection(
    inspection: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    expected_name: re.Pattern[str],
    workspace: str,
) -> JsonDict:
    config = inspection.get("Config")
    mounts = inspection.get("Mounts")
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    name = inspection.get("Name")
    container_id = record["container_id"]
    image_id = inspection.get("Image")
    if (
        inspection.get("Id") != container_id
        or not isinstance(name, str)
        or expected_name.fullmatch(name.removeprefix("/")) is None
        or _SHA256_ID_RE.fullmatch(str(image_id or "")) is None
        or not isinstance(config, Mapping)
        or config.get("Image") != image_id
        or not isinstance(labels, Mapping)
        or labels.get("ipfs_accelerate.grok_isolation") != "true"
        or labels.get("ipfs_accelerate.codex_fallback_isolation") is not None
        or not isinstance(mounts, list)
    ):
        raise ValueError("ordinary Grok container identity drifted")
    normalized: list[JsonDict] = []
    for mount in mounts:
        if not isinstance(mount, Mapping):
            raise ValueError("ordinary Grok container mount is invalid")
        item = {
            "Type": mount.get("Type"),
            "Source": mount.get("Source"),
            "Destination": mount.get("Destination"),
            "RW": mount.get("RW"),
        }
        if not isinstance(item["Source"], str):
            raise ValueError("ordinary Grok container mount is invalid")
        normalized.append(item)
    workspace_mounts = [
        item
        for item in normalized
        if item == {
            "Type": "bind",
            "Source": workspace,
            "Destination": workspace,
            "RW": True,
        }
    ]
    expected_masks = set(record["mask_sources"])
    observed_masks = {
        str(item["Source"])
        for item in normalized
        if Path(str(item["Source"])).parent.name == "provider-masks"
    }
    admitted_masks = {
        str(item["Source"])
        for item in normalized
        if str(item["Source"]) in expected_masks
        and item["Type"] == "bind"
        and item["RW"] is False
    }
    if (
        len(workspace_mounts) != 1
        or not expected_masks
        or observed_masks != expected_masks
        or admitted_masks != expected_masks
    ):
        raise ValueError("ordinary Grok container mount authority drifted")
    return {
        "container_id": container_id,
        "container_name": name.removeprefix("/"),
        "image_id": image_id,
        "mounts": sorted(
            normalized,
            key=lambda item: (
                str(item["Source"]),
                str(item["Destination"]),
            ),
        ),
    }


def _ordinary_grok_inspection_mentions_workspace(
    inspection: Mapping[str, Any],
    workspace: str,
) -> bool:
    mounts = inspection.get("Mounts")
    if not isinstance(mounts, list):
        return False
    return any(
        isinstance(mount, Mapping)
        and mount.get("Type") == "bind"
        and mount.get("Source") == workspace
        and mount.get("Destination") == workspace
        for mount in mounts
    )


def _ordinary_runner_numeric_pid_absent(
    runner_receipt: Mapping[str, Any],
) -> bool:
    """Prove twice that the recorded numeric PID is absent on the same boot."""

    pid = int(runner_receipt["pid"])
    expected_boot = str(runner_receipt["boot_id"])
    first = _ordinary_provider_runner_observation(pid)
    second = _ordinary_provider_runner_observation(pid)
    return bool(
        first == second
        and first[0] == expected_boot
        and first[1] is None
        and first[2] is None
    )


def _ordinary_grok_orphan_receipt(
    runner_receipt: Mapping[str, Any],
    *,
    safe: bool,
    removed: bool,
    reason: str,
    detail: Mapping[str, Any] | None = None,
) -> JsonDict:
    body: JsonDict = {
        "schema": _ORDINARY_GROK_ORPHAN_FENCE_SCHEMA,
        "task_id": runner_receipt["task_id"],
        "attempt": runner_receipt["attempt"],
        "task_revision_cid": runner_receipt["task_revision_cid"],
        "workspace_path": runner_receipt["workspace_path"],
        "runner_pid": runner_receipt["pid"],
        "runner_receipt_id": runner_receipt["receipt_id"],
        "safe_to_restart": safe,
        "removed": removed,
        "reason": reason,
        "detail": dict(detail or {}),
    }
    return {**body, "receipt_id": content_identity(body)}


def _fence_ordinary_grok_orphan_container(
    status: Mapping[str, Any],
    *,
    host_birth_dead_or_fenced: bool,
) -> JsonDict:
    """Remove one exact non-CAS legacy Grok orphan after host fencing.

    This is deliberately bounded legacy recovery.  The v1 ordinary birth
    receipt predates Docker lease binding, so a deleted private lease is
    indistinguishable from a native ordinary runner and is treated as absent.
    Future births must persist the backend, CID, and lease identity instead of
    widening this compatibility path.
    """

    runner_receipt, reason = _validated_ordinary_provider_runner_receipt(status)
    if runner_receipt is None or not host_birth_dead_or_fenced:
        unavailable_receipt = {
            "task_id": "",
            "attempt": 0,
            "task_revision_cid": "",
            "workspace_path": "",
            "pid": 0,
            "receipt_id": "",
        }
        return _ordinary_grok_orphan_receipt(
            runner_receipt or unavailable_receipt,
            safe=False,
            removed=False,
            reason=(
                reason
                if runner_receipt is None
                else "ordinary_runner_host_fence_unproven"
            ),
        )
    try:
        records = _ordinary_grok_lease_records()
        # An ordinary receipt carries no Docker authority.  In the absence of
        # even one private Grok lease, keep plain ordinary runners hermetic and
        # do not make Docker availability a new completion dependency.
        if not records:
            return _ordinary_grok_orphan_receipt(
                runner_receipt,
                safe=True,
                removed=False,
                reason="ordinary_grok_orphan_private_lease_absent",
            )
        if not _ordinary_runner_numeric_pid_absent(runner_receipt):
            raise ValueError("ordinary runner numeric PID is no longer absent")
        docker = _ordinary_grok_docker_binary()
        if not docker:
            raise ValueError("Docker unavailable")
        from ..runtime import grok_cli_runner

        expected_name = re.compile(
            rf"ipfs-accelerate-grok-{int(runner_receipt['pid'])}-[0-9a-f]{{32}}"
        )
        candidates: list[tuple[JsonDict, JsonDict, list[str]]] = []
        for record in records:
            prefix = [
                docker,
                f"--host={grok_cli_runner._DOCKER_LOCAL_HOST}",
                "--config",
                str(record["docker_config"]),
            ]
            inspected = _ordinary_grok_container_inspection(
                prefix, str(record["container_id"])
            )
            if inspected is None:
                continue
            if not _ordinary_grok_inspection_mentions_workspace(
                inspected,
                str(runner_receipt["workspace_path"]),
            ):
                continue
            # The ordinary recovery route never adopts or removes a CAS-owned
            # lease, even if its marker is malformed or dangling.  An
            # unrelated CAS lease does not block this exact workspace fence.
            if record["cas_owned"]:
                raise ValueError(
                    "ordinary Grok workspace is owned by CAS authority"
                )
            projection = _ordinary_grok_inspection_projection(
                inspected,
                record=record,
                expected_name=expected_name,
                workspace=str(runner_receipt["workspace_path"]),
            )
            candidates.append((record, projection, prefix))
        if not candidates:
            return _ordinary_grok_orphan_receipt(
                runner_receipt,
                safe=True,
                removed=False,
                reason="ordinary_grok_orphan_container_absent",
            )
        if len(candidates) != 1:
            raise ValueError("ordinary Grok orphan candidate is ambiguous")
        record, projection, prefix = candidates[0]
        current_records = {
            item["lease_root"]: item for item in _ordinary_grok_lease_records()
        }
        if current_records.get(record["lease_root"]) != record:
            raise ValueError("ordinary Grok lease drifted before cleanup")
        if not _ordinary_runner_numeric_pid_absent(runner_receipt):
            raise ValueError("ordinary runner numeric PID was reused before cleanup")
        reinspection = _ordinary_grok_container_inspection(
            prefix, str(record["container_id"])
        )
        if reinspection is None or _ordinary_grok_inspection_projection(
            reinspection,
            record=record,
            expected_name=expected_name,
            workspace=str(runner_receipt["workspace_path"]),
        ) != projection:
            raise ValueError("ordinary Grok container drifted before cleanup")
        returncode, _stdout, _stderr = _ordinary_grok_docker_query(
            [*prefix, "rm", "--force", str(record["container_id"])]
        )
        absent = _ordinary_grok_container_absent(
            prefix, str(record["container_id"])
        )
        absent_recheck = _ordinary_grok_container_absent(
            prefix, str(record["container_id"])
        )
        pid_absent = _ordinary_runner_numeric_pid_absent(runner_receipt)
        if returncode != 0 or not absent or not absent_recheck or not pid_absent:
            raise ValueError("ordinary Grok orphan removal could not be verified")
        return _ordinary_grok_orphan_receipt(
            runner_receipt,
            safe=True,
            removed=True,
            reason="ordinary_grok_orphan_container_removed",
            detail={
                "container_id": record["container_id"],
                "container_name": projection["container_name"],
                "image_id": projection["image_id"],
                "lease_root": record["lease_root"],
                "provider_mask_sources": list(record["mask_sources"]),
            },
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return _ordinary_grok_orphan_receipt(
            runner_receipt,
            safe=False,
            removed=False,
            reason="ordinary_grok_orphan_container_fence_unproven",
            detail={"error": str(exc)[:512]},
        )


def _with_ordinary_grok_orphan_fence(
    status: Mapping[str, Any], host_result: JsonDict
) -> JsonDict:
    container_fence = _fence_ordinary_grok_orphan_container(
        status,
        host_birth_dead_or_fenced=bool(host_result.get("safe_to_restart")),
    )
    result = {**host_result, "container_fence": container_fence}
    if not container_fence["safe_to_restart"]:
        result["host_fenced"] = bool(host_result.get("fenced"))
        result["safe_to_restart"] = False
        result["fenced"] = False
        result["reason"] = "ordinary_grok_orphan_container_fence_unproven"
    return result


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
    host_result = {
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
    return (
        _with_ordinary_grok_orphan_fence(status, host_result)
        if safe
        else host_result
    )


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
