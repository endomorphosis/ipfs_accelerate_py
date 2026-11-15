"""Owner-private census of prepared, born and retired coding children.

These immutable observations do not admit a child, retry, or completion. Only
the live owner handle can append events; serialized event files are never
adopted. A partial write is retained as incomplete evidence, never overwritten.
Cleanup and broker closure are exact caller observations, not inferred custody.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
import time
import uuid

SCHEMA = "owner-worker-attempt-event@1"
MAX_PROMPT_BYTES = 256000
MAX_EVENT_BYTES = 8192
MAX_DEADLINE_NS = (1 << 63) - 1
COMMON_FIELDS = frozenset({"schema", "event", "attempt_id", "purpose",
    "native_prompt_sha256", "workspace_identity_sha256", "boundary_manifest_sha256",
    "absolute_deadline_ns", "owner_pid", "owner_start_ticks", "observation_only",
    "execution_authority", "completion_authority", "retry_authority"})
SPAWNED_FIELDS = COMMON_FIELDS | {"prepared_sha256", "child_pid", "child_start_ticks"}
RETIRED_FIELDS = SPAWNED_FIELDS | {"spawned_sha256", "waited", "child_returncode",
    "cleanup_succeeded", "brokers_closed"}
_TOKEN = object()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    _need(len(raw) <= MAX_EVENT_BYTES, "worker attempt event exceeds its byte bound")
    return raw


def _directory_identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _file_identity(info):
    return (_directory_identity(info), info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns)


def _birth(pid):
    """Observe an unreaped real child, including a child that exited promptly.

    A zombie still has an unambiguous birth and direct parent until wait reaps
    it. The ordinary live-service identity helper intentionally excludes it.
    """
    _need(type(pid) is int and 0 < pid < (1 << 31), "exact process PID required")
    path = Path("/proc") / str(pid) / "stat"
    def read():
        with path.open("rb") as stream:
            raw = stream.read(16385)
        end = raw.rfind(b")")
        fields = raw[end + 2:].split()
        _need(len(raw) <= 16384 and end > 0 and len(fields) >= 20
            and raw.split(b" ", 1)[0] == str(pid).encode("ascii"),
            "bounded kernel process birth required")
        parent, ticks = int(fields[1]), int(fields[19])
        _need(parent >= 0 and ticks > 0, "positive kernel process birth required")
        return pid, parent, ticks
    first = read()
    _need(read() == first, "kernel process birth changed")
    return first


def _deadline_ns(value):
    # Refuse unrepresentable deadlines rather than silently shortening or
    # resetting the original allowance. The nanosecond floor is conservative.
    _need(type(value) in (int, float) and 0 < value <= MAX_DEADLINE_NS / 1_000_000_000,
        "finite bounded absolute worker deadline required")
    result = int(value * 1_000_000_000)
    _need(0 < result <= MAX_DEADLINE_NS, "bounded absolute worker deadline required")
    _remaining(result)
    return result


def _remaining(deadline):
    if time.monotonic_ns() >= deadline:
        raise TimeoutError("original worker attempt deadline exhausted")


def _prompt_digest(input_fd):
    _need(type(input_fd) is int and input_fd >= 0, "exact sealed input descriptor required")
    seals = sum(getattr(fcntl, name, fallback) for name, fallback in (
        ("F_SEAL_SEAL", 1), ("F_SEAL_SHRINK", 2), ("F_SEAL_GROW", 4), ("F_SEAL_WRITE", 8)))
    _need(fcntl.fcntl(input_fd, getattr(fcntl, "F_GET_SEALS", 1034)) & seals == seals,
        "worker attempt requires sealed original stdin")
    before = os.fstat(input_fd)
    _need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= MAX_PROMPT_BYTES,
        "bounded sealed original stdin required")
    raw = os.pread(input_fd, before.st_size + 1, 0)
    _need(len(raw) == before.st_size and _file_identity(os.fstat(input_fd)) == _file_identity(before)
        and fcntl.fcntl(input_fd, getattr(fcntl, "F_GET_SEALS", 1034)) & seals == seals,
        "sealed original stdin changed")
    return _sha(raw)


@dataclass(frozen=True, slots=True)
class _Event:
    name: str
    raw: bytes
    identity: tuple


@dataclass(slots=True)
class _State:
    prepared: _Event | None = None
    spawned: _Event | None = None
    process: subprocess.Popen | None = None
    child: tuple | None = None
    spent: bool = False
    retired: bool = False
    retired_event: _Event | None = None
    lock: object = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True, init=False)
class OwnerWorkerAttempt:
    """One owner-issued observation handle; never a launch capability."""
    _token: object
    _common: bytes
    _boundary: Path
    _workspace: Path
    _workspace_identity: tuple
    _parent: Path
    _parent_identity: tuple
    _state: _State

    def __init__(self, *, token, common, boundary, workspace, workspace_identity,
            parent, parent_identity):
        _need(token is _TOKEN and type(self) is OwnerWorkerAttempt,
            "worker attempt must be prepared by its live owner")
        for key, value in dict(_token=token, _common=_wire(common), _boundary=boundary,
                _workspace=workspace, _workspace_identity=workspace_identity,
                _parent=parent, _parent_identity=parent_identity, _state=_State()).items():
            object.__setattr__(self, key, value)

    @property
    def attempt_id(self):
        return json.loads(self._common)["attempt_id"]

    def _current(self, *, deadline):
        from . import source_formalization_context as guard
        _need(type(self) is OwnerWorkerAttempt and self._token is _TOKEN,
            "exact owner worker attempt required")
        common = json.loads(self._common)
        value, digest = guard._owner_boundary(self._boundary)
        _need(digest == common["boundary_manifest_sha256"]
            and value["owner_uid"] == os.getuid() == os.geteuid() == os.getgid(),
            "worker attempt owner boundary changed")
        owner = _birth(os.getpid())
        _need((owner[0], owner[2]) == (common["owner_pid"], common["owner_start_ticks"]),
            "worker attempt belongs to another owner birth")
        _need(self._workspace.resolve(strict=True) == self._workspace
            and stat.S_ISDIR(self._workspace.lstat().st_mode)
            and _directory_identity(self._workspace.lstat()) == self._workspace_identity,
            "worker attempt workspace identity changed")
        guard._protected_directory(self._parent, owner=value["owner_uid"], group=value["owner_uid"])
        _need(_directory_identity(self._parent.lstat()) == self._parent_identity,
            "worker attempt observation directory changed")
        if deadline:
            _remaining(common["absolute_deadline_ns"])
        return common

    def _check_event(self, descriptor, event):
        before = os.stat(event.name, dir_fd=descriptor, follow_symlinks=False)
        _need(_file_identity(before) == event.identity and stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o400 and before.st_nlink == 1,
            "immutable worker attempt event identity changed")
        fd = os.open(event.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=descriptor)
        try:
            _need(_file_identity(os.fstat(fd)) == event.identity,
                "immutable worker attempt event was replaced")
            raw = os.read(fd, MAX_EVENT_BYTES + 1)
            _need(raw == event.raw and _file_identity(os.fstat(fd)) == event.identity
                and _file_identity(os.stat(event.name, dir_fd=descriptor, follow_symlinks=False)) == event.identity,
                "immutable worker attempt event bytes changed")
        finally:
            os.close(fd)

    def _publish(self, value, *, deadline):
        OwnerWorkerAttempt._current(self, deadline=deadline)
        parent_fd = os.open(self._parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            _need(_directory_identity(os.fstat(parent_fd)) == self._parent_identity,
                "worker attempt directory was replaced")
            prior = tuple(event for event in (self._state.prepared, self._state.spawned) if event is not None)
            for event in prior:
                OwnerWorkerAttempt._check_event(self, parent_fd, event)
            raw = _wire(value)
            name = "worker-attempt-" + value["attempt_id"] + "-" + value["event"] + ".json"
            # Preserve partial files on errors. Neither a retry nor a different
            # owner may adopt them as a successfully published observation.
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o400, dir_fd=parent_fd)
            try:
                remaining = memoryview(raw)
                while remaining:
                    count = os.write(fd, remaining)
                    _need(count > 0, "worker attempt event write made no progress")
                    remaining = remaining[count:]
                os.fchmod(fd, 0o400)
                os.fsync(fd)
                info = os.fstat(fd)
                _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o400
                    and info.st_uid == os.getuid() and info.st_gid == os.getgid()
                    and info.st_nlink == 1 and info.st_size == len(raw),
                    "private worker attempt event publication changed")
                event = _Event(name, raw, _file_identity(info))
            finally:
                os.close(fd)
            os.fsync(parent_fd)
            for previous in (*prior, event):
                OwnerWorkerAttempt._check_event(self, parent_fd, previous)
            OwnerWorkerAttempt._current(self, deadline=deadline)
            _need(_directory_identity(os.fstat(parent_fd)) == self._parent_identity,
                "worker attempt directory changed during publication")
            return event
        finally:
            os.close(parent_fd)

    def note_spawned(self, process):
        """Record exactly one actual direct child without invoking a provider."""
        with self._state.lock:
            _need(not self._state.spent and not self._state.retired and self._state.prepared is not None,
                "worker attempt birth observation is already consumed")
            _need(type(process) is subprocess.Popen and process._child_created is True,
                "real Popen child required")
            # Recognition of an actual Popen is already enough to preclude a
            # later false no-spawn record, even if kernel observation fails.
            self._state.spent = True
            self._state.process = process
            _need(process.returncode is None, "unreaped real Popen child required")
            child = _birth(process.pid)
            _need(child[1] == os.getpid(), "worker attempt child belongs to another owner")
            # A failed birth write must remain unknown, never become a false
            # no-spawn retirement. The actual child is the caller's to clean up.
            self._state.child = child
            common = OwnerWorkerAttempt._current(self, deadline=True)
            value = dict(common, event="spawned", prepared_sha256=_sha(self._state.prepared.raw),
                child_pid=child[0], child_start_ticks=child[2])
            event = OwnerWorkerAttempt._publish(self, value, deadline=True)
            _need(_birth(process.pid) == child and process.returncode is None,
                "worker attempt child birth changed during publication")
            self._state.spawned = event

    def finish(self, *, waited, cleanup_succeeded, brokers_closed):
        """Persist retirement even after expiry; flags never grant settlement.

        A no-spawn retirement is valid only if the caller never attempted a
        launch. A Popen error or interruption before assigning its return value
        leaves the prepared chain unknown; the monitor must not call finish.
        """
        _need(all(type(flag) is bool for flag in (waited, cleanup_succeeded, brokers_closed)),
            "exact boolean worker retirement observations required")
        with self._state.lock:
            _need(not self._state.retired and self._state.prepared is not None,
                "worker attempt retirement is already consumed")
            state = self._state
            _need((state.process is None) == (state.spawned is None),
                "worker attempt birth publication is incomplete")
            returncode = None
            if waited:
                _need(type(state.process) is subprocess.Popen
                    and type(state.process.returncode) is int,
                    "waited worker attempt requires a settled real child")
                returncode = state.process.returncode
                _need(subprocess.Popen.poll(state.process) == returncode,
                    "worker attempt waited outcome changed")
            common = OwnerWorkerAttempt._current(self, deadline=False)
            child = state.child
            value = dict(common, event="retired", prepared_sha256=_sha(state.prepared.raw),
                spawned_sha256=None if state.spawned is None else _sha(state.spawned.raw),
                child_pid=None if child is None else child[0], child_start_ticks=None if child is None else child[2],
                waited=waited, child_returncode=returncode, cleanup_succeeded=cleanup_succeeded,
                brokers_closed=brokers_closed)
            state.retired = True
            state.retired_event = OwnerWorkerAttempt._publish(self, value, deadline=False)
            return json.loads(_wire(value))


def prepare_owner_worker_attempt(*, boundary, workspace, input_fd, absolute_deadline):
    """Capture sealed original stdin and publish one fresh owner attempt ID."""
    from . import source_formalization_context as guard
    deadline = _deadline_ns(absolute_deadline)
    boundary = Path(boundary)
    workspace = Path(workspace)
    _need(boundary.is_absolute() and boundary.resolve(strict=True) == boundary
        and workspace.is_absolute() and workspace.resolve(strict=True) == workspace
        and stat.S_ISDIR(workspace.lstat().st_mode), "absolute real worker paths required")
    value, boundary_sha = guard._owner_boundary(boundary)
    _need(value["owner_uid"] == os.getuid() == os.geteuid() == os.getgid(),
        "worker attempt requires the exact owner identity")
    parent = boundary.parent / "worker-observations"
    guard._protected_directory(parent, owner=value["owner_uid"], group=value["owner_uid"])
    owner = _birth(os.getpid())
    common = dict(schema=SCHEMA, event="prepared", attempt_id=uuid.uuid4().hex, purpose="coding",
        native_prompt_sha256=_prompt_digest(input_fd), workspace_identity_sha256=_sha(str(workspace).encode("utf-8")),
        boundary_manifest_sha256=boundary_sha, absolute_deadline_ns=deadline,
        owner_pid=owner[0], owner_start_ticks=owner[2], observation_only=True,
        execution_authority=False, completion_authority=False, retry_authority=False)
    result = OwnerWorkerAttempt(token=_TOKEN, common=common, boundary=boundary, workspace=workspace,
        workspace_identity=_directory_identity(workspace.lstat()), parent=parent,
        parent_identity=_directory_identity(parent.lstat()))
    result._state.prepared = OwnerWorkerAttempt._publish(result, common, deadline=True)
    return result


__all__ = ["SCHEMA", "COMMON_FIELDS", "SPAWNED_FIELDS", "RETIRED_FIELDS", "OwnerWorkerAttempt",
    "prepare_owner_worker_attempt"]
