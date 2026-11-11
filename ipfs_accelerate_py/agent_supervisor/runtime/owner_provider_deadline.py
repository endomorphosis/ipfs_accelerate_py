"""Carry an original provider deadline to its exact owner wrapper.

The sealed descriptor authenticates a restriction from the direct parent. It
does not carry task, resource, dispatch, retry or completion authority. The
owner consumes and closes it before launching an isolated worker.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

OWNER_WRAPPER = "/opt/ipfs-supervisor/bin/router-worker"
DEADLINE_FD_ENV = "IPFS_ACCELERATE_PROVIDER_DEADLINE_FD"
DEADLINE_SHA256_ENV = "IPFS_ACCELERATE_PROVIDER_DEADLINE_SHA256"
SCHEMA = "owner-provider-deadline-handoff@1"
MAX_HANDOFF_BYTES = 8192
MAX_DEADLINE_NS = (1 << 63) - 1
_FIELDS = frozenset({"schema", "absolute_deadline_ns", "issuer_pid",
    "issuer_start_ticks", "issuer_uid", "issuer_euid", "boot_id",
    "command_sha256", "workspace", "namespaces"})
_SEALS = sum(getattr(fcntl, name, fallback) for name, fallback in (
    ("F_SEAL_SEAL", 1), ("F_SEAL_SHRINK", 2), ("F_SEAL_GROW", 4),
    ("F_SEAL_WRITE", 8)))


class OwnerProviderDeadlineError(ValueError):
    """The exact parent deadline restriction could not be authenticated."""


def _need(condition, message):
    if not condition:
        raise OwnerProviderDeadlineError(message)


def owner_provider_deadline_selected(command):
    """Select only the literal installed owner wrapper, without path aliases."""
    return (type(command) in (list, tuple) and bool(command)
        and type(command[0]) is str and command[0] == OWNER_WRAPPER)


def _command(command):
    _need(owner_provider_deadline_selected(command)
        and 1 <= len(command) <= 256
        and all(type(part) is str and 0 < len(part.encode("utf-8")) <= 8192
            and "\x00" not in part for part in command),
        "bounded literal owner wrapper command required")
    result = list(command)
    _need(sum(len(part.encode("utf-8")) + 1 for part in result) <= 65536,
        "owner wrapper command exceeds its byte bound")
    return result


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _deadline_ns(value):
    _need(type(value) in (int, float) and value > 0
        and (type(value) is int or math.isfinite(value)),
        "finite positive original provider deadline required")
    # Floor the exact binary float, rather than rounding a float product up.
    numerator, denominator = (value, 1) if type(value) is int else value.as_integer_ratio()
    result = numerator * 1_000_000_000 // denominator
    _need(0 < result <= MAX_DEADLINE_NS,
        "bounded original provider deadline required")
    _remaining(result)
    return result


def _remaining(deadline):
    _need(type(deadline) is int and 0 < deadline <= MAX_DEADLINE_NS,
        "bounded integer provider deadline required")
    if time.monotonic_ns() >= deadline:
        raise TimeoutError("original provider deadline exhausted")


def _boot_id():
    with open("/proc/sys/kernel/random/boot_id", "r", encoding="ascii") as stream:
        value = stream.read(65).strip()
    _need(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is not None,
        "exact kernel boot identity required")
    return value


def _process(pid):
    _need(type(pid) is int and 0 < pid < (1 << 31), "exact issuer PID required")
    with open(f"/proc/{pid}/stat", "rb") as stream:
        raw = stream.read(16385)
    end = raw.rfind(b")")
    fields = raw[end + 2:].split()
    _need(len(raw) <= 16384 and end > 0 and len(fields) >= 20
        and raw.split(b" ", 1)[0] == str(pid).encode("ascii")
        and fields[0] not in (b"Z", b"X", b"x"),
        "live bounded issuer process birth required")
    start = int(fields[19])
    _need(start > 0, "positive issuer process birth required")
    with open(f"/proc/{pid}/status", "r", encoding="ascii") as stream:
        status = stream.read(65537)
    _need(len(status) <= 65536, "bounded issuer process status required")
    uids = [line.split()[1:] for line in status.splitlines() if line.startswith("Uid:")]
    _need(len(uids) == 1 and len(uids[0]) == 4,
        "exact issuer real and effective UID required")
    real_uid, effective_uid = int(uids[0][0]), int(uids[0][1])
    _need(real_uid >= 0 and effective_uid >= 0, "nonnegative issuer UID required")
    return start, real_uid, effective_uid


def _namespaces(pid):
    result = {}
    for name in ("pid", "mnt", "time"):
        try:
            identity = os.readlink(f"/proc/{pid}/ns/{name}")
        except FileNotFoundError:
            if name == "time":
                continue
            raise
        _need(type(identity) is str and 0 < len(identity) <= 256
            and re.fullmatch(name + r":\[[0-9]+\]", identity) is not None,
            "exact kernel namespace identity required")
        result[name] = identity
    return result


def _workspace(workspace, *, current=False):
    path = Path(workspace)
    _need(path.is_absolute() and len(os.fsencode(path)) <= 4096
        and path.resolve(strict=True) == path and (not current or Path.cwd() == path),
        "exact canonical owner workspace required")
    info = path.lstat()
    _need(stat.S_ISDIR(info.st_mode), "actual owner workspace directory required")
    return {"path": str(path), "device": info.st_dev, "inode": info.st_ino}


def _actual_command(command):
    with open("/proc/self/cmdline", "rb") as stream:
        raw = stream.read(65537)
    _need(len(raw) <= 65536 and raw.endswith(b"\x00"),
        "bounded actual owner wrapper command required")
    observed = [part.decode("utf-8") for part in raw[:-1].split(b"\x00")]
    _need(len(observed) >= len(command) and observed[-len(command):] == command,
        "actual owner wrapper argv differs from its deadline binding")
    prefix = observed[:-len(command)]
    _need(not prefix or (len(prefix) in (1, 2)
        and Path(prefix[0]).is_absolute()
        and re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(prefix[0]).name) is not None
        and (len(prefix) == 1 or prefix[1] == "-I")),
        "actual owner wrapper interpreter differs from its literal entry")


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class OwnerProviderDeadlineHandoff:
    environment: dict[str, str]
    pass_fds: tuple[int, ...]


@contextmanager
def prepare_owner_provider_deadline_handoff(*, command, workspace, absolute_deadline):
    """Yield one sealed, owner-only direct-child deadline descriptor."""
    literal = _command(command)
    deadline = _deadline_ns(absolute_deadline)
    pid = os.getpid()
    birth = _process(pid)
    _need(birth[1] == birth[2] == os.getuid() == os.geteuid(),
        "one exact issuer UID required")
    value = {"schema": SCHEMA, "absolute_deadline_ns": deadline,
        "issuer_pid": pid, "issuer_start_ticks": birth[0],
        "issuer_uid": birth[1], "issuer_euid": birth[2],
        "boot_id": _boot_id(), "command_sha256": hashlib.sha256(_wire(literal)).hexdigest(),
        "workspace": _workspace(workspace), "namespaces": _namespaces(pid)}
    raw = _wire(value)
    _need(len(raw) <= MAX_HANDOFF_BYTES, "owner deadline handoff exceeds its byte bound")
    flags = getattr(os, "MFD_CLOEXEC", 1) | getattr(os, "MFD_ALLOW_SEALING", 2)
    if hasattr(os, "memfd_create"):
        descriptor = os.memfd_create("owner-provider-deadline", flags)
    else:
        from ..validation.validation_runtime import _validation_memfd_create
        descriptor = _validation_memfd_create("owner-provider-deadline", flags)
    try:
        _need(type(descriptor) is int and descriptor >= 3,
            "private owner deadline descriptor required")
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _need(written > 0, "owner deadline descriptor write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        fcntl.fcntl(descriptor, getattr(fcntl, "F_ADD_SEALS", 1033), _SEALS)
        _need(fcntl.fcntl(descriptor, getattr(fcntl, "F_GET_SEALS", 1034)) & _SEALS == _SEALS,
            "owner deadline descriptor lacks required seals")
        _need(_process(pid) == birth and _workspace(workspace) == value["workspace"]
            and _namespaces(pid) == value["namespaces"] and _boot_id() == value["boot_id"],
            "owner deadline issuer changed during handoff preparation")
        _remaining(deadline)
        yield OwnerProviderDeadlineHandoff(environment={DEADLINE_FD_ENV: str(descriptor),
            DEADLINE_SHA256_ENV: hashlib.sha256(raw).hexdigest()}, pass_fds=(descriptor,))
    finally:
        os.close(descriptor)


def consume_owner_provider_deadline_handoff(*, command, workspace, environment=None):
    """Consume a sealed restriction from the actual direct parent, or return None.

    Both markers are removed before validation. A canonical descriptor number
    is closed even when the digest marker is missing or the payload refuses.
    """
    source = os.environ if environment is None else environment
    _need(isinstance(source, MutableMapping), "mutable owner deadline environment required")
    fd_text = source.pop(DEADLINE_FD_ENV, None)
    digest = source.pop(DEADLINE_SHA256_ENV, None)
    if fd_text is None and digest is None:
        return None
    _need(type(fd_text) is str and re.fullmatch(r"[1-9][0-9]{0,9}", fd_text) is not None
        and 3 <= int(fd_text) < (1 << 31), "canonical inherited owner deadline descriptor required")
    descriptor = int(fd_text)
    try:
        _need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "complete owner deadline descriptor and digest required")
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o400
            and before.st_nlink == 0 and before.st_uid == os.getuid() == os.geteuid()
            and 0 < before.st_size <= MAX_HANDOFF_BYTES,
            "sealed owner-only bounded anonymous deadline descriptor required")
        _need(fcntl.fcntl(descriptor, getattr(fcntl, "F_GET_SEALS", 1034)) & _SEALS == _SEALS,
            "inherited owner deadline descriptor lacks required seals")
        raw = os.pread(descriptor, MAX_HANDOFF_BYTES + 1, 0)
        _need(len(raw) == before.st_size
            and _file_identity(os.fstat(descriptor)) == _file_identity(before)
            and hashlib.sha256(raw).hexdigest() == digest,
            "inherited owner deadline descriptor bytes differ")
        def unique(pairs):
            result = {}
            for key, item in pairs:
                _need(key not in result, "duplicate owner deadline field refused")
                result[key] = item
            return result
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(OwnerProviderDeadlineError(
                "nonfinite owner deadline value refused")))
        _need(type(value) is dict and set(value) == _FIELDS and value["schema"] == SCHEMA
            and _wire(value) == raw, "closed canonical owner deadline handoff required")
        for key in ("absolute_deadline_ns", "issuer_pid", "issuer_start_ticks", "issuer_uid", "issuer_euid"):
            _need(type(value[key]) is int, "literal integer owner deadline identity required")
        literal = _command(command)
        _actual_command(literal)
        _need(value["command_sha256"] == hashlib.sha256(_wire(literal)).hexdigest(),
            "owner deadline literal command binding differs")
        _need(type(value["workspace"]) is dict and set(value["workspace"]) == {"path", "device", "inode"}
            and type(value["workspace"]["path"]) is str
            and all(type(value["workspace"][key]) is int for key in ("device", "inode"))
            and value["workspace"] == _workspace(workspace, current=True),
            "owner deadline workspace binding differs")
        pid = os.getppid()
        birth = _process(pid)
        namespaces = _namespaces(pid)
        _need(value["issuer_pid"] == pid and value["issuer_start_ticks"] == birth[0]
            and value["issuer_uid"] == birth[1] == os.getuid()
            and value["issuer_euid"] == birth[2] == os.geteuid()
            and birth[1] == birth[2] and value["boot_id"] == _boot_id(),
            "owner deadline issuer differs from the actual direct parent")
        _need(type(value["namespaces"]) is dict and value["namespaces"] == namespaces
            and namespaces == _namespaces(os.getpid()),
            "owner deadline process, mount or time namespace differs")
        deadline = value["absolute_deadline_ns"]
        _remaining(deadline)
        _need(os.getppid() == pid and _process(pid) == birth
            and _namespaces(pid) == namespaces and _namespaces(os.getpid()) == namespaces
            and _workspace(workspace, current=True) == value["workspace"] and _boot_id() == value["boot_id"]
            and _file_identity(os.fstat(descriptor)) == _file_identity(before)
            and fcntl.fcntl(descriptor, getattr(fcntl, "F_GET_SEALS", 1034)) & _SEALS == _SEALS,
            "owner deadline identity changed during consumption")
        _actual_command(literal)
        _remaining(deadline)
        return deadline
    finally:
        os.close(descriptor)


__all__ = ["OWNER_WRAPPER", "DEADLINE_FD_ENV", "DEADLINE_SHA256_ENV", "SCHEMA",
    "MAX_HANDOFF_BYTES", "MAX_DEADLINE_NS", "OwnerProviderDeadlineError",
    "OwnerProviderDeadlineHandoff", "owner_provider_deadline_selected",
    "prepare_owner_provider_deadline_handoff", "consume_owner_provider_deadline_handoff"]
