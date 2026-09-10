"""Kernel boundary for processes that retain state-authority credentials.

Implementation providers run under the same host account as the supervisor in
the current deployment profile.  Environment scrubbing alone is therefore not
enough: a same-UID child can ordinarily read a dumpable parent's
``/proc/<pid>/environ``.  Trusted control processes call this module before
they spawn provider code.  Linux then denies same-UID process introspection,
while ordinary provider children receive no state credential in their own
environment.

This is an isolation boundary, not an authorization decision.  Typed owner
commands and canonical repository validation remain mandatory.
"""

from __future__ import annotations

import array
import ctypes
import hashlib
import json
import os
import re
import secrets
import select
import signal
import socket
import stat
import struct
import sys
import threading
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn

PR_GET_DUMPABLE: Final = 3
PR_SET_DUMPABLE: Final = 4
PR_SET_PDEATHSIG: Final = 1
_CAP_SYS_PTRACE: Final = 19
_YAMA_PTRACE_SCOPE_PATH: Final = Path("/proc/sys/kernel/yama/ptrace_scope")
_LINUX_PROCESS_SCOPE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/linux-process-scope@1"
)
_LINUX_PROCESS_SCOPE_FIELDS: Final = frozenset(
    {
        "schema",
        "pid",
        "start_time_ticks",
        "boot_id",
        "pid_namespace_device",
        "pid_namespace_inode",
        "cgroup_relative_path",
        "cgroup_mount_device",
        "cgroup_mount_inode",
        "cgroup_directory_device",
        "cgroup_directory_inode",
        "cgroup_events_device",
        "cgroup_events_inode",
        "identity_id",
    }
)
_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES: Final = 16 * 1024
_LINUX_PROCESS_SCOPE_MAX_MOUNTINFO_BYTES: Final = 512 * 1024
_LINUX_PROCESS_SCOPE_MAX_PROC_ENTRIES: Final = 262_144
_PROC_ROOT: Final = Path("/proc")
_BOOT_ID_PATH: Final = Path("/proc/sys/kernel/random/boot_id")
_MOUNTINFO_PATH: Final = Path("/proc/self/mountinfo")
_BOOT_ID_RE: Final = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
STATE_AUTHORITY_CREDENTIAL_NAMES: Final = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
    }
)

STATE_AUTHORITY_DESCRIPTOR_ENV_NAMES: Final = frozenset(
    {"IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"}
)
STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET"
)
STATE_AUTHORITY_HANDOFF_ADDRESS_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_ADDRESS"
)
STATE_AUTHORITY_HANDOFF_PARENT_PID_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_PID"
)
STATE_AUTHORITY_HANDOFF_PARENT_START_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_START"
)
STATE_AUTHORITY_HANDOFF_BOOT_ID_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_BOOT_ID"
)
STATE_AUTHORITY_HANDOFF_PARENT_LOSS_POLICY_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_LOSS_POLICY"
)
STATE_AUTHORITY_PARENT_LOSS_TERMINATE: Final = "terminate_with_parent"
STATE_AUTHORITY_PARENT_LOSS_DETACHED: Final = "independent_detached"
STATE_AUTHORITY_PARENT_LOSS_POLICIES: Final = frozenset(
    {
        STATE_AUTHORITY_PARENT_LOSS_TERMINATE,
        STATE_AUTHORITY_PARENT_LOSS_DETACHED,
    }
)
STATE_AUTHORITY_HANDOFF_ENV_NAMES: Final = frozenset(
    {
        STATE_AUTHORITY_HANDOFF_ADDRESS_ENV,
        STATE_AUTHORITY_HANDOFF_PARENT_PID_ENV,
        STATE_AUTHORITY_HANDOFF_PARENT_START_ENV,
        STATE_AUTHORITY_HANDOFF_BOOT_ID_ENV,
        STATE_AUTHORITY_HANDOFF_PARENT_LOSS_POLICY_ENV,
    }
)
_CAPTURED_STATE_AUTHORITY_CREDENTIALS: dict[str, str] = {}
_CAPTURED_STATE_AUTHORITY_LOCK = threading.RLock()


class StateAuthorityProcessIsolationError(RuntimeError):
    """A credential-bearing process could not establish its kernel boundary."""


def _scope_error(
    message: str,
    cause: BaseException | None = None,
) -> NoReturn:
    error = StateAuthorityProcessIsolationError(message)
    if cause is None:
        raise error
    raise error from cause


def _scope_integer(value: object, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _scope_error(f"Linux process scope {name} is not an integer")
    if value <= 0 or value > maximum:
        _scope_error(f"Linux process scope {name} is out of bounds")
    return value


def _canonical_cgroup_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        _scope_error("Linux process scope cgroup path is invalid")
    if value == ".":
        return value
    if (
        value.startswith("/")
        or value.endswith("/")
        or not value.isascii()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _scope_error("Linux process scope cgroup path is not relative")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        _scope_error("Linux process scope cgroup path escapes its mount")
    if str(PurePosixPath(*parts)) != value:
        _scope_error("Linux process scope cgroup path is noncanonical")
    return value


def _scope_identity(body: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_linux_process_scope(record: Mapping[str, object]) -> dict[str, object]:
    """Return the canonical immutable Linux process/cgroup scope record."""

    if not isinstance(record, Mapping) or set(record) != _LINUX_PROCESS_SCOPE_FIELDS:
        _scope_error("Linux process scope record shape is invalid")
    schema = record.get("schema")
    if schema != _LINUX_PROCESS_SCOPE_SCHEMA:
        _scope_error("Linux process scope schema is invalid")
    boot_id = record.get("boot_id")
    if not isinstance(boot_id, str) or _BOOT_ID_RE.fullmatch(boot_id) is None:
        _scope_error("Linux process scope boot identity is invalid")
    body: dict[str, object] = {
        "schema": schema,
        "pid": _scope_integer(record.get("pid"), "pid", maximum=2**31 - 1),
        "start_time_ticks": _scope_integer(
            record.get("start_time_ticks"),
            "start_time_ticks",
            maximum=2**63 - 1,
        ),
        "boot_id": boot_id,
        "pid_namespace_device": _scope_integer(
            record.get("pid_namespace_device"),
            "pid_namespace_device",
            maximum=2**64 - 1,
        ),
        "pid_namespace_inode": _scope_integer(
            record.get("pid_namespace_inode"),
            "pid_namespace_inode",
            maximum=2**64 - 1,
        ),
        "cgroup_relative_path": _canonical_cgroup_relative_path(
            record.get("cgroup_relative_path")
        ),
        "cgroup_mount_device": _scope_integer(
            record.get("cgroup_mount_device"),
            "cgroup_mount_device",
            maximum=2**64 - 1,
        ),
        "cgroup_mount_inode": _scope_integer(
            record.get("cgroup_mount_inode"),
            "cgroup_mount_inode",
            maximum=2**64 - 1,
        ),
        "cgroup_directory_device": _scope_integer(
            record.get("cgroup_directory_device"),
            "cgroup_directory_device",
            maximum=2**64 - 1,
        ),
        "cgroup_directory_inode": _scope_integer(
            record.get("cgroup_directory_inode"),
            "cgroup_directory_inode",
            maximum=2**64 - 1,
        ),
        "cgroup_events_device": _scope_integer(
            record.get("cgroup_events_device"),
            "cgroup_events_device",
            maximum=2**64 - 1,
        ),
        "cgroup_events_inode": _scope_integer(
            record.get("cgroup_events_inode"),
            "cgroup_events_inode",
            maximum=2**64 - 1,
        ),
    }
    identity_id = record.get("identity_id")
    if identity_id != _scope_identity(body):
        _scope_error("Linux process scope identity is invalid")
    return {**body, "identity_id": identity_id}


def _read_scope_descriptor(descriptor: int, *, maximum_bytes: int) -> str:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _scope_error("Linux process scope control file is not regular")
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = os.read(descriptor, maximum_bytes + 1)
        after = os.fstat(descriptor)
    except OSError as exc:
        _scope_error("Linux process scope control file is unavailable", exc)
    if len(payload) > maximum_bytes:
        _scope_error("Linux process scope control file is oversized")
    if (before.st_dev, before.st_ino, before.st_mode) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
    ):
        _scope_error("Linux process scope control file changed while read")
    try:
        return payload.decode("ascii")
    except UnicodeError as exc:
        _scope_error("Linux process scope control file is malformed", exc)


def _read_scope_path(path: Path, *, maximum_bytes: int) -> str:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        return _read_scope_descriptor(descriptor, maximum_bytes=maximum_bytes)
    except OSError as exc:
        _scope_error("Linux process scope control path is unavailable", exc)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_scope_at(directory_descriptor: int, name: str, *, maximum_bytes: int) -> str:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        return _read_scope_descriptor(descriptor, maximum_bytes=maximum_bytes)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_scope_directory(path: Path) -> int:
    if not path.is_absolute():
        _scope_error("Linux process scope directory is not absolute")
    descriptor = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for part in path.parts[1:]:
            if not part or part in {".", ".."}:
                _scope_error("Linux process scope directory path is invalid")
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_scope_relative_directory(root_descriptor: int, relative_path: str) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        if relative_path != ".":
            for part in relative_path.split("/"):
                child = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _decode_mountinfo_path(value: str) -> Path:
    escapes = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    decoded = re.sub(
        r"\\(040|011|012|134)",
        lambda match: escapes[match.group(1)],
        value,
    )
    path = Path(decoded)
    if not path.is_absolute() or "\x00" in decoded:
        _scope_error("Linux process scope cgroup mount path is invalid")
    return path


def _cgroup2_mount_path() -> Path:
    mountinfo = _read_scope_path(
        _MOUNTINFO_PATH,
        maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_MOUNTINFO_BYTES,
    )
    mounts: list[Path] = []
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) <= separator + 2 or fields[separator + 1] != "cgroup2":
            continue
        if fields[3] != "/":
            _scope_error("Linux process scope requires the root cgroup-v2 mount")
        mounts.append(_decode_mountinfo_path(fields[4]))
    if len(mounts) != 1:
        _scope_error("Linux process scope requires one cgroup-v2 mount")
    return mounts[0]


def _parse_scope_stat(raw: str) -> int:
    closing = raw.rfind(")")
    if closing < 2:
        _scope_error("Linux process scope stat record is malformed")
    fields = raw[closing + 2 :].split()
    try:
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as exc:
        _scope_error("Linux process scope stat record is malformed", exc)
    if start_ticks <= 0 or start_ticks > 2**63 - 1:
        _scope_error("Linux process scope start identity is invalid")
    return start_ticks


def _parse_scope_cgroup(raw: str) -> str:
    lines = [line for line in raw.splitlines() if line]
    if len(lines) != 1 or not lines[0].startswith("0::"):
        _scope_error("Linux process scope requires unified cgroup-v2 membership")
    absolute = lines[0][3:]
    if not absolute.startswith("/"):
        _scope_error("Linux process scope cgroup membership is malformed")
    relative = absolute[1:] or "."
    return _canonical_cgroup_relative_path(relative)


def _pid_namespace_identity(process_descriptor: int) -> tuple[int, int]:
    namespace_descriptor = -1
    try:
        namespace_descriptor = os.open(
            "ns/pid",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            dir_fd=process_descriptor,
        )
        metadata = os.fstat(namespace_descriptor)
    except OSError as exc:
        _scope_error("Linux process scope PID namespace is unavailable", exc)
    finally:
        if namespace_descriptor >= 0:
            os.close(namespace_descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_dev <= 0 or metadata.st_ino <= 0:
        _scope_error("Linux process scope PID namespace is invalid")
    return metadata.st_dev, metadata.st_ino


def _open_process_directory(pid: int) -> int:
    proc_descriptor = _open_scope_directory(_PROC_ROOT)
    try:
        descriptor = os.open(
            str(pid),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=proc_descriptor,
        )
    finally:
        os.close(proc_descriptor)
    return descriptor


def _current_boot_id() -> str:
    boot_id = _read_scope_path(_BOOT_ID_PATH, maximum_bytes=128).strip()
    if _BOOT_ID_RE.fullmatch(boot_id) is None:
        _scope_error("Linux process scope boot identity is unavailable")
    return boot_id


def _open_cgroup_scope(relative_path: str) -> tuple[int, int, int]:
    mount_descriptor = _open_scope_directory(_cgroup2_mount_path())
    try:
        cgroup_descriptor = _open_scope_relative_directory(
            mount_descriptor,
            relative_path,
        )
    except BaseException:
        os.close(mount_descriptor)
        raise
    events_descriptor = -1
    try:
        events_descriptor = os.open(
            "cgroup.events",
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=cgroup_descriptor,
        )
        if not stat.S_ISREG(os.fstat(events_descriptor).st_mode):
            _scope_error("Linux process scope cgroup.events is invalid")
    except BaseException:
        if events_descriptor >= 0:
            os.close(events_descriptor)
        os.close(cgroup_descriptor)
        os.close(mount_descriptor)
        raise
    return mount_descriptor, cgroup_descriptor, events_descriptor


def capture_linux_process_scope(pid: int) -> dict[str, object]:
    """Capture one exact Linux PID namespace and cgroup-v2 execution scope."""

    if not sys.platform.startswith("linux"):
        _scope_error("Linux process scope requires Linux")
    pid = _scope_integer(pid, "pid", maximum=2**31 - 1)
    process_descriptor = mount_descriptor = cgroup_descriptor = events_descriptor = -1
    try:
        process_descriptor = _open_process_directory(pid)
        start_ticks = _parse_scope_stat(
            _read_scope_at(
                process_descriptor,
                "stat",
                maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES,
            )
        )
        namespace_device, namespace_inode = _pid_namespace_identity(process_descriptor)
        relative_path = _parse_scope_cgroup(
            _read_scope_at(
                process_descriptor,
                "cgroup",
                maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES,
            )
        )
        boot_id = _current_boot_id()
        mount_descriptor, cgroup_descriptor, events_descriptor = _open_cgroup_scope(
            relative_path
        )
        mount_metadata = os.fstat(mount_descriptor)
        cgroup_metadata = os.fstat(cgroup_descriptor)
        events_metadata = os.fstat(events_descriptor)
        if (
            _parse_scope_stat(
                _read_scope_at(
                    process_descriptor,
                    "stat",
                    maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES,
                )
            )
            != start_ticks
            or _pid_namespace_identity(process_descriptor)
            != (namespace_device, namespace_inode)
            or _parse_scope_cgroup(
                _read_scope_at(
                    process_descriptor,
                    "cgroup",
                    maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES,
                )
            )
            != relative_path
            or _current_boot_id() != boot_id
        ):
            _scope_error("Linux process scope changed while captured")
        body: dict[str, object] = {
            "schema": _LINUX_PROCESS_SCOPE_SCHEMA,
            "pid": pid,
            "start_time_ticks": start_ticks,
            "boot_id": boot_id,
            "pid_namespace_device": namespace_device,
            "pid_namespace_inode": namespace_inode,
            "cgroup_relative_path": relative_path,
            "cgroup_mount_device": mount_metadata.st_dev,
            "cgroup_mount_inode": mount_metadata.st_ino,
            "cgroup_directory_device": cgroup_metadata.st_dev,
            "cgroup_directory_inode": cgroup_metadata.st_ino,
            "cgroup_events_device": events_metadata.st_dev,
            "cgroup_events_inode": events_metadata.st_ino,
        }
        return validate_linux_process_scope(
            {**body, "identity_id": _scope_identity(body)}
        )
    except OSError as exc:
        _scope_error("Linux process scope capture failed", exc)
    finally:
        for descriptor in (
            events_descriptor,
            cgroup_descriptor,
            mount_descriptor,
            process_descriptor,
        ):
            if descriptor >= 0:
                os.close(descriptor)


def _pidfd_reports_exit(pid: int) -> bool | None:
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None:
        return None
    descriptor = -1
    try:
        descriptor = pidfd_open(pid, 0)
        poller = select.poll()
        poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
        return bool(poller.poll(0))
    except ProcessLookupError:
        return True
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _exact_scope_init_state(record: Mapping[str, object]) -> str:
    descriptor = -1
    try:
        descriptor = _open_process_directory(int(record["pid"]))
    except FileNotFoundError:
        return "gone"
    except OSError:
        return "unknown"
    try:
        start_ticks = _parse_scope_stat(
            _read_scope_at(
                descriptor,
                "stat",
                maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES,
            )
        )
        namespace = _pid_namespace_identity(descriptor)
        cgroup = _parse_scope_cgroup(
            _read_scope_at(
                descriptor,
                "cgroup",
                maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES,
            )
        )
        if (
            start_ticks != record["start_time_ticks"]
            or namespace
            != (record["pid_namespace_device"], record["pid_namespace_inode"])
            or cgroup != record["cgroup_relative_path"]
        ):
            return "mismatch"
        exited = _pidfd_reports_exit(int(record["pid"]))
        if exited is None:
            return "unknown"
        if exited:
            return "gone"
        if (
            _parse_scope_stat(
                _read_scope_at(
                    descriptor,
                    "stat",
                    maximum_bytes=_LINUX_PROCESS_SCOPE_MAX_PROC_BYTES,
                )
            )
            != record["start_time_ticks"]
            or _pid_namespace_identity(descriptor) != namespace
        ):
            return "mismatch"
        return "alive"
    except (OSError, StateAuthorityProcessIsolationError):
        return "unknown"
    finally:
        os.close(descriptor)


def _scanner_pid_namespace_identity() -> tuple[int, int] | None:
    descriptor = -1
    try:
        # ``/proc/self`` is a magic symlink; the scope opener deliberately
        # rejects symlinks.  Resolve it by numeric PID through the same
        # descriptor-relative process opener used for every candidate.
        descriptor = _open_process_directory(os.getpid())
        return _pid_namespace_identity(descriptor)
    except (OSError, StateAuthorityProcessIsolationError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _scope_pid_namespace_depth(process_descriptor: int) -> int:
    status = _read_scope_at(
        process_descriptor,
        "status",
        maximum_bytes=128 * 1024,
    )
    values = []
    for line in status.splitlines():
        name, separator, raw_value = line.partition(":")
        if separator and name == "NSpid":
            values.append(raw_value.split())
    if len(values) != 1 or not values[0]:
        _scope_error("Linux process scope NSpid record is incomplete")
    if any(not item.isascii() or not item.isdecimal() or int(item) <= 0 for item in values[0]):
        _scope_error("Linux process scope NSpid record is malformed")
    return len(values[0])


def _scope_namespace_has_members(record: Mapping[str, object]) -> bool | None:
    numeric: list[str] = []
    try:
        with os.scandir(_PROC_ROOT) as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    continue
                numeric.append(entry.name)
                if len(numeric) > _LINUX_PROCESS_SCOPE_MAX_PROC_ENTRIES:
                    return None
    except OSError:
        return None
    expected = (record["pid_namespace_device"], record["pid_namespace_inode"])
    scanner_namespace = _scanner_pid_namespace_identity()
    target_is_nested = scanner_namespace is not None and scanner_namespace != expected
    for pid_name in numeric:
        descriptor = -1
        try:
            descriptor = _open_process_directory(int(pid_name))
            # A process with only one NSpid value is in the scanner's PID
            # namespace and therefore cannot inhabit a recorded descendant
            # namespace.  This safely avoids permission-dependent namespace
            # opens for unrelated host services; every nested candidate is
            # still compared by namespace identity independently of cgroup
            # membership.  An unreadable extant candidate remains unknown.
            if target_is_nested and _scope_pid_namespace_depth(descriptor) == 1:
                continue
            if _pid_namespace_identity(descriptor) == expected:
                return True
        except FileNotFoundError:
            continue
        except (OSError, StateAuthorityProcessIsolationError):
            try:
                metadata = os.stat(
                    _PROC_ROOT / pid_name,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(metadata.st_mode):
                    return None
            except OSError:
                pass
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return False


def _cgroup_events_populated(events_descriptor: int) -> int:
    raw = _read_scope_descriptor(events_descriptor, maximum_bytes=4096)
    values: dict[str, str] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] in values:
            _scope_error("Linux process scope cgroup.events is malformed")
        values[fields[0]] = fields[1]
    if values.get("populated") not in {"0", "1"}:
        _scope_error("Linux process scope populated state is invalid")
    return int(values["populated"])


def linux_process_scope_quiescent(record: Mapping[str, object]) -> bool:
    """Return true only after the exact PID namespace and cgroup are empty."""

    try:
        canonical = validate_linux_process_scope(record)
        if not sys.platform.startswith("linux") or _current_boot_id() != canonical["boot_id"]:
            return False
        init_state = _exact_scope_init_state(canonical)
        if init_state != "gone":
            return False
        namespace_has_members = _scope_namespace_has_members(canonical)
        if namespace_has_members is not False:
            return False
        mount_descriptor = _open_scope_directory(_cgroup2_mount_path())
        try:
            mount_metadata = os.fstat(mount_descriptor)
            if (
                mount_metadata.st_dev != canonical["cgroup_mount_device"]
                or mount_metadata.st_ino != canonical["cgroup_mount_inode"]
            ):
                return False
            try:
                cgroup_descriptor = _open_scope_relative_directory(
                    mount_descriptor,
                    str(canonical["cgroup_relative_path"]),
                )
            except FileNotFoundError:
                return True
            try:
                cgroup_metadata = os.fstat(cgroup_descriptor)
                if (
                    cgroup_metadata.st_dev
                    != canonical["cgroup_directory_device"]
                    or cgroup_metadata.st_ino
                    != canonical["cgroup_directory_inode"]
                ):
                    return False
                events_descriptor = os.open(
                    "cgroup.events",
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=cgroup_descriptor,
                )
                try:
                    events_metadata = os.fstat(events_descriptor)
                    if (
                        events_metadata.st_dev
                        != canonical["cgroup_events_device"]
                        or events_metadata.st_ino
                        != canonical["cgroup_events_inode"]
                    ):
                        return False
                    return _cgroup_events_populated(events_descriptor) == 0
                finally:
                    os.close(events_descriptor)
            finally:
                os.close(cgroup_descriptor)
        finally:
            os.close(mount_descriptor)
    except (OSError, StateAuthorityProcessIsolationError):
        return False


def _process_birth(pid: int) -> tuple[int, int, str]:
    """Return exact parent/start/boot identity from Linux procfs."""

    try:
        raw = (Path("/proc") / str(int(pid)) / "stat").read_text(
            encoding="ascii"
        )
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        parent_pid = int(fields[1])
        start_ticks = int(fields[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, IndexError, UnicodeError, ValueError) as exc:
        raise StateAuthorityProcessIsolationError(
            "state-authority process birth is unavailable"
        ) from exc
    if closing < 0 or parent_pid < 0 or start_ticks <= 0 or not boot_id:
        raise StateAuthorityProcessIsolationError(
            "state-authority process birth is invalid"
        )
    return parent_pid, start_ticks, boot_id


def state_authority_process_birth(
    pid: int | None = None,
) -> tuple[int, int, str]:
    """Expose one exact procfs birth tuple for a sealed parent binding."""

    return _process_birth(os.getpid() if pid is None else int(pid))


def arm_state_authority_parent_death_signal(
    *,
    expected_parent_pid: int,
    expected_parent_start_time_ticks: int,
    expected_boot_id: str,
) -> None:
    """Arm SIGTERM-on-parent-loss before a child can acquire authority."""

    if not sys.platform.startswith("linux"):
        raise StateAuthorityProcessIsolationError(
            "state-authority parent-loss fence requires Linux"
        )
    parent_pid, _own_start, boot_id = _process_birth(os.getpid())
    if parent_pid != int(expected_parent_pid) or boot_id != expected_boot_id:
        raise StateAuthorityProcessIsolationError(
            "state-authority delegate parent changed before arming"
        )
    _grandparent, parent_start, parent_boot = _process_birth(parent_pid)
    if (
        parent_start != int(expected_parent_start_time_ticks)
        or parent_boot != expected_boot_id
    ):
        raise StateAuthorityProcessIsolationError(
            "state-authority delegate parent birth differs"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise StateAuthorityProcessIsolationError(
            f"PR_SET_PDEATHSIG failed with errno {error_number}"
        )
    parent_after, _start_after, boot_after = _process_birth(os.getpid())
    if parent_after != parent_pid or boot_after != expected_boot_id:
        raise StateAuthorityProcessIsolationError(
            "state-authority delegate parent changed while arming"
        )


def make_state_authority_process_nondumpable() -> None:
    """Unconditionally establish and verify the Linux non-dumpable boundary."""

    if not sys.platform.startswith("linux"):
        raise StateAuthorityProcessIsolationError(
            "state authority requires a qualified Linux non-dumpable process"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise StateAuthorityProcessIsolationError(
            f"PR_SET_DUMPABLE failed with errno {error_number}"
        )
    if libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise StateAuthorityProcessIsolationError(
            "state-authority process remained dumpable"
        )


def _read_bounded_proc_text(path: Path, *, maximum_bytes: int) -> str:
    """Read one procfs control file without following a substituted name."""

    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise StateAuthorityProcessIsolationError(
                "state-authority ptrace prerequisite is not a regular proc file"
            )
        payload = os.read(descriptor, maximum_bytes + 1)
    except OSError as exc:
        raise StateAuthorityProcessIsolationError(
            "state-authority ptrace prerequisite is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise StateAuthorityProcessIsolationError(
            "state-authority ptrace prerequisite is oversized"
        )
    try:
        return payload.decode("ascii")
    except UnicodeError as exc:
        raise StateAuthorityProcessIsolationError(
            "state-authority ptrace prerequisite is malformed"
        ) from exc


def _same_user_namespace_ptrace_capability_pids() -> tuple[int, ...]:
    """Return same-UID peers that can override Yama for this user namespace."""

    try:
        own_namespace = os.stat("/proc/self/ns/user").st_ino
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        raise StateAuthorityProcessIsolationError(
            "state-authority process capability inventory is unavailable"
        ) from exc
    capability_mask = 1 << _CAP_SYS_PTRACE
    offenders: list[int] = []
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            process_metadata = os.stat(entry, follow_symlinks=False)
            if process_metadata.st_uid != os.geteuid():
                continue
            status = _read_bounded_proc_text(
                entry / "status",
                maximum_bytes=128 * 1024,
            )
        except FileNotFoundError:
            continue
        except StateAuthorityProcessIsolationError:
            # A process can disappear between directory, namespace, and status
            # observations.  Prove that disappearance before treating it as a
            # benign race; an extant unreadable peer remains fail-closed.
            if not entry.exists():
                continue
            raise
        except OSError as exc:
            if not entry.exists():
                continue
            raise StateAuthorityProcessIsolationError(
                "same-UID ptrace capability could not be qualified"
            ) from exc
        capability_values: dict[str, int] = {}
        try:
            for line in status.splitlines():
                key, separator, value = line.partition(":")
                if separator and key in {"CapInh", "CapPrm", "CapEff", "CapAmb"}:
                    capability_values[key] = int(value.strip(), 16)
        except ValueError as exc:
            raise StateAuthorityProcessIsolationError(
                "same-UID ptrace capability record is malformed"
            ) from exc
        if set(capability_values) != {"CapInh", "CapPrm", "CapEff", "CapAmb"}:
            raise StateAuthorityProcessIsolationError(
                "same-UID ptrace capability record is incomplete"
            )
        if not any(
            value & capability_mask for value in capability_values.values()
        ):
            continue
        try:
            namespace_metadata = os.stat(entry / "ns" / "user")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            if not entry.exists():
                continue
            raise StateAuthorityProcessIsolationError(
                "capable same-UID peer namespace is unavailable"
            ) from exc
        if namespace_metadata.st_ino == own_namespace:
            offenders.append(pid)
        # Capabilities in a descendant user namespace do not grant ptrace
        # authority over this namespace (for example rootless Docker).
    return tuple(sorted(set(offenders)))


def require_state_authority_handoff_ptrace_protection() -> None:
    """Fail closed unless the post-exec handoff has a qualified ptrace gate.

    ``execve`` resets the ordinary non-dumpable flag.  Yama mode 1 or stricter
    prevents an untrusted same-UID sibling from attaching during the small
    exec-to-``PR_SET_DUMPABLE`` interval.  A peer holding ``CAP_SYS_PTRACE`` in
    the same user namespace can override that policy, so such a launch is
    rejected as well.  The check is repeated immediately before delivery.
    """

    if not sys.platform.startswith("linux"):
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff requires qualified Linux ptrace policy"
        )
    raw_scope = _read_bounded_proc_text(
        _YAMA_PTRACE_SCOPE_PATH,
        maximum_bytes=32,
    ).strip()
    try:
        scope = int(raw_scope)
    except ValueError as exc:
        raise StateAuthorityProcessIsolationError(
            "state-authority ptrace scope is malformed"
        ) from exc
    if scope not in {1, 2, 3}:
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff requires Yama ptrace_scope >= 1"
        )
    offenders = _same_user_namespace_ptrace_capability_pids()
    if offenders:
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff has a same-namespace CAP_SYS_PTRACE peer"
        )


def env_secret_handle_target(secret_handle: str) -> str:
    """Return the environment variable named by an ``env://`` secret handle."""

    handle = str(secret_handle or "").strip()
    if not handle.startswith("env://"):
        return ""
    target = handle[len("env://") :].strip()
    if not target or not target.isidentifier():
        return ""
    return target


def forward_env_secret_handle_credentials(
    child_environment: MutableMapping[str, str],
    *,
    secret_handle: str,
    source_environment: Mapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Copy an already-admitted ``env://`` credential into a trusted child.

    This never mints a token.  Provider children must still go through
    ``provider_subprocess_environment``, which scrubs these names.
    """

    target = env_secret_handle_target(secret_handle)
    if not target:
        return child_environment
    source = os.environ if source_environment is None else source_environment
    value = state_authority_credential(target, environment=source)
    if value:
        child_environment[target] = value
    return child_environment


def state_authority_credential(
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a trusted process credential without exposing it to children."""

    source = os.environ if environment is None else environment
    value = str(source.get(name, "") or "").strip()
    if value:
        return value
    if name not in STATE_AUTHORITY_CREDENTIAL_NAMES:
        return ""
    with _CAPTURED_STATE_AUTHORITY_LOCK:
        return str(_CAPTURED_STATE_AUTHORITY_CREDENTIALS.get(name, "") or "")


def state_authority_credentials_present(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an admitted raw state credential is present."""

    source = os.environ if environment is None else environment
    if any(
        bool(str(source.get(name, "") or "").strip())
        for name in (*STATE_AUTHORITY_CREDENTIAL_NAMES, *STATE_AUTHORITY_HANDOFF_ENV_NAMES)
    ):
        return True
    if environment is not None:
        return False
    with _CAPTURED_STATE_AUTHORITY_LOCK:
        return any(_CAPTURED_STATE_AUTHORITY_CREDENTIALS.values())


def establish_state_authority_process_boundary() -> bool:
    """Make the current process non-dumpable before it mints a credential."""

    make_state_authority_process_nondumpable()
    return True


def state_authority_pass_fds(
    environment: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Return validated inherited descriptors for trusted control children."""

    source = os.environ if environment is None else environment
    broker_socket_present = bool(
        str(source.get(STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV, "") or "").strip()
    )
    broker_descriptor_present = bool(
        str(
            source.get(
                "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
                "",
            )
            or ""
        ).strip()
    )
    if broker_socket_present != broker_descriptor_present:
        raise StateAuthorityProcessIsolationError(
            "state-authority broker binding is incomplete"
        )
    descriptors: list[int] = []
    for name in STATE_AUTHORITY_DESCRIPTOR_ENV_NAMES:
        raw = str(source.get(name, "") or "").strip()
        if not raw:
            continue
        if not raw.isascii() or not raw.isdecimal():
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor binding is invalid"
            )
        descriptor = int(raw)
        if descriptor < 3 or descriptor > 1_048_576:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor binding is invalid"
            )
        try:
            observed = os.fstat(descriptor)
        except OSError as exc:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or not 32 <= observed.st_size <= 256
        ):
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is not a bounded owner memfd"
            )
        if not sys.platform.startswith("linux"):
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor requires a qualified Linux memfd"
            )
        import fcntl

        required_seals = (
            int(getattr(fcntl, "F_SEAL_SEAL", 0x0001))
            | int(getattr(fcntl, "F_SEAL_SHRINK", 0x0002))
            | int(getattr(fcntl, "F_SEAL_GROW", 0x0004))
            | int(getattr(fcntl, "F_SEAL_WRITE", 0x0008))
        )
        try:
            observed_seals = int(
                fcntl.fcntl(
                    descriptor,
                    int(getattr(fcntl, "F_GET_SEALS", 1034)),
                )
            )
        except OSError as exc:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is not sealed"
            ) from exc
        if observed_seals & required_seals != required_seals:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is not sealed"
            )
        descriptors.append(descriptor)
    return tuple(sorted(set(descriptors)))


class StateAuthorityChildHandoff:
    """One-shot post-exec SCM_RIGHTS delivery to one exact direct child."""

    def __init__(
        self,
        *,
        listener: socket.socket | None,
        source_fd: int = -1,
        address: str = "",
        parent_loss_policy: str = "",
    ) -> None:
        self._listener = listener
        self._source_fd = source_fd
        self._address = address
        self._parent_loss_policy = parent_loss_policy
        self._closed = False

    @property
    def pass_fds(self) -> tuple[int, ...]:
        # The reusable secret is deliberately absent throughout exec.
        return ()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()

    @staticmethod
    def _assert_execution_identity(
        *,
        pid: int,
        executable_descriptor: int,
        child_executable_descriptor: int,
        expected_argv: tuple[str, ...],
    ) -> None:
        """Bind an authority redemption to the retained executable and argv."""

        if executable_descriptor < 3 or not expected_argv:
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff execution binding is invalid"
            )
        try:
            executable_before = os.fstat(executable_descriptor)
            observed_before = os.fstat(child_executable_descriptor)
            expected = b"\0".join(os.fsencode(item) for item in expected_argv) + b"\0"
            if len(expected) > 1024 * 1024 or any(
                b"\0" in os.fsencode(item) for item in expected_argv
            ):
                raise ValueError("child argv is not bounded")
            command_descriptor = os.open(
                f"/proc/{int(pid)}/cmdline",
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                observed_command = bytearray()
                while len(observed_command) <= len(expected):
                    chunk = os.read(
                        command_descriptor,
                        min(64 * 1024, len(expected) + 1 - len(observed_command)),
                    )
                    if not chunk:
                        break
                    observed_command.extend(chunk)
            finally:
                os.close(command_descriptor)
            observed_after = os.fstat(child_executable_descriptor)
            executable_after = os.fstat(executable_descriptor)
        except (OSError, TypeError, ValueError) as exc:
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff child execution is unavailable"
            ) from exc
        expected_identity = (
            int(executable_before.st_dev),
            int(executable_before.st_ino),
            int(executable_before.st_mode),
            int(executable_before.st_uid),
            int(executable_before.st_gid),
            int(executable_before.st_nlink),
            int(executable_before.st_size),
        )
        observed_before_identity = (
            int(observed_before.st_dev),
            int(observed_before.st_ino),
            int(observed_before.st_mode),
            int(observed_before.st_uid),
            int(observed_before.st_gid),
            int(observed_before.st_nlink),
            int(observed_before.st_size),
        )
        observed_after_identity = (
            int(observed_after.st_dev),
            int(observed_after.st_ino),
            int(observed_after.st_mode),
            int(observed_after.st_uid),
            int(observed_after.st_gid),
            int(observed_after.st_nlink),
            int(observed_after.st_size),
        )
        if (
            not stat.S_ISREG(executable_before.st_mode)
            or executable_before.st_uid != 0
            or executable_before.st_nlink != 1
            or stat.S_IMODE(executable_before.st_mode) & 0o022
            or (
                int(executable_after.st_dev),
                int(executable_after.st_ino),
                int(executable_after.st_mode),
                int(executable_after.st_uid),
                int(executable_after.st_gid),
                int(executable_after.st_nlink),
                int(executable_after.st_size),
            )
            != expected_identity
            or observed_before_identity != expected_identity
            or observed_after_identity != expected_identity
            or bytes(observed_command) != expected
        ):
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff child executable or argv differs"
            )

    def deliver(
        self,
        process: Any,
        *,
        timeout_seconds: float = 10.0,
        expected_executable_descriptor: int | None = None,
        expected_argv: Sequence[str] | None = None,
    ) -> None:
        """Deliver only after the exact child proves post-exec hardening."""

        listener = self._listener
        if listener is None:
            self.close()
            return
        if (expected_executable_descriptor is None) != (expected_argv is None):
            self.close()
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff execution binding is incomplete"
            )
        canonical_expected_argv = (
            tuple(str(item) for item in expected_argv)
            if expected_argv is not None
            else None
        )
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        try:
            require_state_authority_handoff_ptrace_protection()
            expected_pid = int(process.pid)
            expected_parent, expected_start, expected_boot = _process_birth(
                expected_pid
            )
            if expected_parent != os.getpid():
                raise StateAuthorityProcessIsolationError(
                    "state-authority handoff child is not a direct exact birth"
                )
            listener.settimeout(max(0.05, deadline - time.monotonic()))
            while time.monotonic() < deadline:
                try:
                    channel, _address = listener.accept()
                except TimeoutError as exc:
                    raise StateAuthorityProcessIsolationError(
                        "state-authority handoff child did not redeem"
                    ) from exc
                child_execution_descriptors: list[int] = []
                try:
                    channel.settimeout(
                        max(0.05, deadline - time.monotonic())
                    )
                    credentials = channel.getsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_PEERCRED,
                        struct.calcsize("3i"),
                    )
                    peer_pid, peer_uid, _peer_gid = struct.unpack(
                        "3i", credentials
                    )
                    if peer_pid != expected_pid or peer_uid != os.geteuid():
                        continue
                    payload = bytearray()
                    first_read = True
                    while b"\n" not in payload and len(payload) <= 1024:
                        if first_read:
                            first_read = False
                            chunk, ancillary, flags, _peer = channel.recvmsg(
                                1024,
                                socket.CMSG_SPACE(array.array("i").itemsize),
                                getattr(socket, "MSG_CMSG_CLOEXEC", 0),
                            )
                            if flags & (
                                getattr(socket, "MSG_CTRUNC", 0)
                                | getattr(socket, "MSG_TRUNC", 0)
                            ):
                                raise StateAuthorityProcessIsolationError(
                                    "state-authority child execution descriptor was truncated"
                                )
                            for level, kind, descriptor_payload in ancillary:
                                if not (
                                    level == socket.SOL_SOCKET
                                    and kind == socket.SCM_RIGHTS
                                ):
                                    raise StateAuthorityProcessIsolationError(
                                        "unknown state-authority child ancillary data"
                                    )
                                received = array.array("i")
                                received.frombytes(
                                    descriptor_payload[
                                        : len(descriptor_payload)
                                        - (
                                            len(descriptor_payload)
                                            % received.itemsize
                                        )
                                    ]
                                )
                                child_execution_descriptors.extend(
                                    int(item) for item in received
                                )
                        else:
                            chunk = channel.recv(1024)
                        if not chunk:
                            break
                        payload.extend(chunk)
                    request = json.loads(bytes(payload).split(b"\n", 1)[0])
                    canonical_request = json.dumps(
                        {
                            "pid": expected_pid,
                            "parent_pid": os.getpid(),
                            "start_time_ticks": expected_start,
                            "boot_id": expected_boot,
                            "address": self._address,
                            "parent_loss_policy": self._parent_loss_policy,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8") + b"\n"
                    if (
                        not isinstance(request, dict)
                        or request
                        != {
                            "pid": expected_pid,
                            "parent_pid": os.getpid(),
                            "start_time_ticks": expected_start,
                            "boot_id": expected_boot,
                            "address": self._address,
                            "parent_loss_policy": self._parent_loss_policy,
                        }
                        or bytes(payload) != canonical_request
                    ):
                        continue
                    current_parent, current_start, current_boot = _process_birth(
                        expected_pid
                    )
                    if (
                        current_parent != os.getpid()
                        or current_start != expected_start
                        or current_boot != expected_boot
                    ):
                        raise StateAuthorityProcessIsolationError(
                            "state-authority handoff child birth changed"
                        )
                    if expected_executable_descriptor is not None:
                        assert canonical_expected_argv is not None
                        if len(child_execution_descriptors) != 1:
                            raise StateAuthorityProcessIsolationError(
                                "state-authority child execution descriptor is absent"
                            )
                        self._assert_execution_identity(
                            pid=expected_pid,
                            executable_descriptor=int(
                                expected_executable_descriptor
                            ),
                            child_executable_descriptor=(
                                child_execution_descriptors[0]
                            ),
                            expected_argv=canonical_expected_argv,
                        )
                    elif child_execution_descriptors:
                        raise StateAuthorityProcessIsolationError(
                            "unexpected state-authority child execution descriptor"
                        )
                    # Recheck at the final transfer boundary.  A degraded host
                    # policy or newly capable same-namespace peer is never
                    # allowed to turn the authenticated PID into authority.
                    final_parent, final_start, final_boot = _process_birth(
                        expected_pid
                    )
                    if (
                        final_parent != os.getpid()
                        or final_start != expected_start
                        or final_boot != expected_boot
                    ):
                        raise StateAuthorityProcessIsolationError(
                            "state-authority handoff child birth changed before transfer"
                        )
                    require_state_authority_handoff_ptrace_protection()
                    rights = array.array("i", [self._source_fd])
                    channel.sendmsg(
                        [b"F"],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                rights.tobytes(),
                            )
                        ],
                    )
                    if channel.recv(1) != b"A":
                        raise StateAuthorityProcessIsolationError(
                            "state-authority handoff child did not acknowledge"
                        )
                    return
                except (
                    json.JSONDecodeError,
                    OSError,
                    UnicodeError,
                    ValueError,
                ):
                    if int(getattr(process, "poll")() or 0) != 0:
                        raise StateAuthorityProcessIsolationError(
                            "state-authority handoff child exited"
                        ) from None
                finally:
                    for descriptor in child_execution_descriptors:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                    channel.close()
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff deadline expired"
            )
        finally:
            self.close()


def prepare_state_authority_child_handoff(
    child_environment: MutableMapping[str, str],
    *,
    parent_loss_policy: str | None = None,
) -> StateAuthorityChildHandoff:
    """Replace inherited broker authority with a post-hardening one-shot."""

    descriptors = state_authority_pass_fds(child_environment)
    if not descriptors:
        return StateAuthorityChildHandoff(listener=None)
    if parent_loss_policy not in STATE_AUTHORITY_PARENT_LOSS_POLICIES:
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff parent-loss policy is required"
        )
    if len(descriptors) != 1:
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff requires one sealed descriptor"
        )
    require_state_authority_handoff_ptrace_protection()
    make_state_authority_process_nondumpable()
    _parent, parent_start, boot_id = _process_birth(os.getpid())
    address = "ipfs-accelerate-state-handoff-" + secrets.token_hex(24)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind("\0" + address)
        listener.listen(4)
    except BaseException:
        listener.close()
        raise
    child_environment.pop(
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
        None,
    )
    child_environment.update(
        {
            STATE_AUTHORITY_HANDOFF_ADDRESS_ENV: address,
            STATE_AUTHORITY_HANDOFF_PARENT_PID_ENV: str(os.getpid()),
            STATE_AUTHORITY_HANDOFF_PARENT_START_ENV: str(parent_start),
            STATE_AUTHORITY_HANDOFF_BOOT_ID_ENV: boot_id,
            STATE_AUTHORITY_HANDOFF_PARENT_LOSS_POLICY_ENV: parent_loss_policy,
        }
    )
    return StateAuthorityChildHandoff(
        listener=listener,
        source_fd=descriptors[0],
        address=address,
        parent_loss_policy=parent_loss_policy,
    )


def receive_state_authority_child_handoff(
    environment: MutableMapping[str, str] | None = None,
) -> bool:
    """Redeem a sealed broker descriptor only after becoming non-dumpable."""

    target = os.environ if environment is None else environment
    values = {
        name: str(target.get(name, "") or "").strip()
        for name in STATE_AUTHORITY_HANDOFF_ENV_NAMES
    }
    if not any(values.values()):
        return False
    if not all(values.values()):
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff binding is incomplete"
        )
    make_state_authority_process_nondumpable()
    try:
        expected_parent = int(values[STATE_AUTHORITY_HANDOFF_PARENT_PID_ENV])
        expected_parent_start = int(
            values[STATE_AUTHORITY_HANDOFF_PARENT_START_ENV]
        )
    except ValueError as exc:
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff parent identity is invalid"
        ) from exc
    parent_loss_policy = values[
        STATE_AUTHORITY_HANDOFF_PARENT_LOSS_POLICY_ENV
    ]
    if parent_loss_policy not in STATE_AUTHORITY_PARENT_LOSS_POLICIES:
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff parent-loss policy is invalid"
        )
    if parent_loss_policy == STATE_AUTHORITY_PARENT_LOSS_TERMINATE:
        arm_state_authority_parent_death_signal(
            expected_parent_pid=expected_parent,
            expected_parent_start_time_ticks=expected_parent_start,
            expected_boot_id=values[STATE_AUTHORITY_HANDOFF_BOOT_ID_ENV],
        )
    parent_pid, start_ticks, boot_id = _process_birth(os.getpid())
    if (
        parent_pid != expected_parent
        or boot_id != values[STATE_AUTHORITY_HANDOFF_BOOT_ID_ENV]
    ):
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff parent or boot changed"
        )
    _grandparent, parent_start, parent_boot = _process_birth(parent_pid)
    if (
        parent_start != expected_parent_start
        or parent_boot != boot_id
    ):
        raise StateAuthorityProcessIsolationError(
            "state-authority handoff parent birth changed"
        )
    address = values[STATE_AUTHORITY_HANDOFF_ADDRESS_ENV]
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    received_fds: list[int] = []
    try:
        channel.settimeout(10.0)
        channel.connect("\0" + address)
        credentials = channel.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        peer_pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
        if peer_pid != expected_parent or peer_uid != os.geteuid():
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff peer is not the exact parent"
            )
        request = json.dumps(
            {
                "pid": os.getpid(),
                "parent_pid": parent_pid,
                "start_time_ticks": start_ticks,
                "boot_id": boot_id,
                "address": address,
                "parent_loss_policy": parent_loss_policy,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        channel.sendall(request)
        data, ancillary, flags, _peer = channel.recvmsg(
            1,
            socket.CMSG_SPACE(array.array("i").itemsize),
        )
        if data != b"F" or flags & getattr(socket, "MSG_CTRUNC", 0):
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff descriptor was truncated"
            )
        for level, kind, payload in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                descriptors = array.array("i")
                descriptors.frombytes(
                    payload[: len(payload) - (len(payload) % descriptors.itemsize)]
                )
                received_fds.extend(int(item) for item in descriptors)
        if len(received_fds) != 1:
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff delivered the wrong descriptor count"
            )
        descriptor = received_fds[0]
        os.set_inheritable(descriptor, False)
        target[
            "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"
        ] = str(descriptor)
        if state_authority_pass_fds(target) != (descriptor,):
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff descriptor did not validate"
            )
        for name in STATE_AUTHORITY_HANDOFF_ENV_NAMES:
            target.pop(name, None)
        channel.sendall(b"A")
        return True
    except BaseException:
        for descriptor in received_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass
        target.pop(
            "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
            None,
        )
        raise
    finally:
        channel.close()



def capture_state_authority_credentials() -> bool:
    """Harden, retain credentials in memory, and remove them from ``environ``.

    Ordinary subprocess APIs inherit ``os.environ`` when no explicit mapping is
    supplied.  Capturing at each trusted module entry therefore makes every
    unclassified child token-free by default.  The few sealed authority hops
    re-add the captured value through ``forward_env_secret_handle_credentials``.
    """

    present = {
        name: str(os.environ.get(name, "") or "").strip()
        for name in STATE_AUTHORITY_CREDENTIAL_NAMES
        if str(os.environ.get(name, "") or "").strip()
    }
    if not present:
        return False
    establish_state_authority_process_boundary()
    with _CAPTURED_STATE_AUTHORITY_LOCK:
        for name, value in present.items():
            prior = _CAPTURED_STATE_AUTHORITY_CREDENTIALS.get(name, "")
            if prior and prior != value:
                raise StateAuthorityProcessIsolationError(
                    "state-authority credential changed within one process"
                )
        _CAPTURED_STATE_AUTHORITY_CREDENTIALS.update(present)
        for name in present:
            os.environ.pop(name, None)
    return True


def clear_captured_state_authority_credentials() -> None:
    """Drop in-memory credentials. Tests and process teardown only."""

    with _CAPTURED_STATE_AUTHORITY_LOCK:
        _CAPTURED_STATE_AUTHORITY_CREDENTIALS.clear()


def harden_state_authority_process(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Make a credential-bearing Linux process non-dumpable, or fail closed.

    Returns ``False`` when no credential is present, so ordinary provider-free
    imports and hermetic tests retain their normal process behavior.
    """

    source = os.environ if environment is None else environment
    if not state_authority_credentials_present(source):
        return False
    establish_state_authority_process_boundary()
    if any(
        bool(str(source.get(name, "") or "").strip())
        for name in STATE_AUTHORITY_HANDOFF_ENV_NAMES
    ):
        if not isinstance(source, MutableMapping):
            raise StateAuthorityProcessIsolationError(
                "state-authority handoff environment is immutable"
            )
        receive_state_authority_child_handoff(source)
    return True


__all__ = (
    "PR_GET_DUMPABLE",
    "PR_SET_PDEATHSIG",
    "PR_SET_DUMPABLE",
    "STATE_AUTHORITY_CREDENTIAL_NAMES",
    "STATE_AUTHORITY_DESCRIPTOR_ENV_NAMES",
    "STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV",
    "STATE_AUTHORITY_HANDOFF_PARENT_LOSS_POLICY_ENV",
    "STATE_AUTHORITY_PARENT_LOSS_DETACHED",
    "STATE_AUTHORITY_PARENT_LOSS_POLICIES",
    "STATE_AUTHORITY_PARENT_LOSS_TERMINATE",
    "StateAuthorityProcessIsolationError",
    "capture_state_authority_credentials",
    "clear_captured_state_authority_credentials",
    "establish_state_authority_process_boundary",
    "env_secret_handle_target",
    "forward_env_secret_handle_credentials",
    "harden_state_authority_process",
    "state_authority_credential",
    "StateAuthorityChildHandoff",
    "arm_state_authority_parent_death_signal",
    "capture_linux_process_scope",
    "linux_process_scope_quiescent",
    "make_state_authority_process_nondumpable",
    "prepare_state_authority_child_handoff",
    "receive_state_authority_child_handoff",
    "require_state_authority_handoff_ptrace_protection",
    "state_authority_credentials_present",
    "state_authority_pass_fds",
    "state_authority_process_birth",
    "validate_linux_process_scope",
)
