"""Fail-closed storage admission and non-authoritative emergency telemetry."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator
import uuid

MINIMUM_FREE_BYTES = 64 * 1024 * 1024
_SPACE_ERRNOS = {errno.ENOSPC, errno.EDQUOT}
_ACTIVE: ContextVar[StorageGuard | None] = ContextVar("supervisor_storage_guard", default=None)
_SCHEMA = "ipfs_accelerate_py/agent-supervisor/storage-pressure@1"


def is_storage_pressure(error: BaseException) -> bool:
    return isinstance(error, OSError) and error.errno in _SPACE_ERRNOS


def _directory(path: Path) -> Path:
    candidate = path.resolve()
    if not candidate.is_dir():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _durable_probe(directory: Path) -> None:
    first = directory / f".supervisor-capacity-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    second = first.with_suffix(".committed")
    try:
        fd = os.open(first, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"storage-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(first, second)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for path in (first, second):
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                if not is_storage_pressure(error):
                    raise


def check_storage_capacity(paths: Iterable[Path], *, minimum_free_bytes: int = MINIMUM_FREE_BYTES,
                           durable: bool = True) -> dict[str, Any]:
    """Probe actual target filesystems, including resolved output symlinks.

    Only ENOSPC/EDQUOT become pressure results. Permission and unrelated I/O
    failures remain errors. A free-space estimate alone cannot clear pressure.
    """
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 1:
        raise ValueError("minimum_free_bytes must be positive")
    observations = []
    directories = list(dict.fromkeys(_directory(Path(path)) for path in paths))
    try:
        for directory in directories:
            info = os.statvfs(directory)
            available = int(info.f_bavail) * int(info.f_frsize)
            observations.append({"path": str(directory), "free_bytes": available,
                                 "minimum_free_bytes": minimum_free_bytes})
            if available < minimum_free_bytes:
                return {"ok": False, "storage_pressure": True, "reason": "insufficient_free_bytes",
                        "observations": observations, "work_complete": False, "completion_authority": False}
        if durable:
            for directory in directories:
                _durable_probe(directory)
    except OSError as error:
        if not is_storage_pressure(error):
            raise
        return {"ok": False, "storage_pressure": True, "reason": errno.errorcode[error.errno],
                "errno": error.errno, "observations": observations,
                "work_complete": False, "completion_authority": False}
    return {"ok": True, "storage_pressure": False, "reason": "", "observations": observations,
            "durable_probe": durable, "work_complete": False, "completion_authority": False}


def _runtime_directory() -> Path:
    # The user runtime is tmpfs and independent of a full persistent volume.
    base = Path(f"/run/user/{os.getuid()}")
    info = base.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError("A private uid-owned runtime directory is required")
    directory = base / "ipfs-agent-supervisor-storage"
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError("Unsafe emergency telemetry directory")
    return directory


def _emergency_path(primary_path: Path) -> Path:
    identity = hashlib.sha256(os.fsencode(primary_path.resolve())).hexdigest()
    return _runtime_directory() / f"{identity}.json"


def _process_identity(pid: int) -> tuple[int, str, str] | None:
    try:
        directory = Path("/proc") / str(pid)
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return directory.stat().st_uid, fields[19], hashlib.sha256((directory / "cmdline").read_bytes()).hexdigest()
    except (OSError, ValueError, IndexError):
        return None


def publish_emergency_status(primary_path: Path, payload: dict[str, Any], *,
                             error: BaseException | None = None) -> dict[str, Any] | None:
    """Publish explicit degraded observation, never task or completion evidence.

    Failure of this secondary channel must not unwind an already-owned command.
    Consumers must independently verify the expected owner and process role.
    """
    try:
        identity = _process_identity(os.getpid())
        if identity is None:
            return None
        now = int(time.time() * 1000)
        degraded = {**payload, "state": "degraded", "phase": "storage_pressure",
                    "storage_original_phase": payload.get("phase", ""), "storage_pressure": True,
                    "backpressure": True, "action": "storage_wait", "heartbeat_at_ms": now, "updated_at_ms": now,
                    "work_complete": False, "completion_authority": False, "gui_acceptance": False}
        record = {"schema": _SCHEMA, "canonical_path": str(primary_path.resolve()), "uid": identity[0],
                  "pid": os.getpid(), "start_ticks": identity[1], "command_sha256": identity[2],
                  "observed_at_ms": now, "errno": getattr(error, "errno", None), "payload": degraded,
                  "work_complete": False, "completion_authority": False}
        path = _emergency_path(primary_path)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return record
    except (OSError, ValueError, TypeError):
        return None


def read_emergency_status(primary_path: Path, *, expected_pid: int, identity_check: Callable[[int], bool],
                          max_age_seconds: float = 120) -> dict[str, Any] | None:
    """Read telemetry only for an independently expected current role/owner."""
    try:
        if type(expected_pid) is not int or expected_pid <= 0 or not 0 < max_age_seconds <= 600:
            return None
        path = _emergency_path(primary_path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 1_000_000:
            return None
        value = json.loads(path.read_text())
        current = _process_identity(expected_pid)
        if current is None or current[0] != os.getuid() or not identity_check(expected_pid):
            return None
        if (value.get("schema") != _SCHEMA or value.get("canonical_path") != str(primary_path.resolve())
                or value.get("pid") != expected_pid or value.get("uid") != current[0]
                or value.get("start_ticks") != current[1] or value.get("command_sha256") != current[2]):
            return None
        age = time.time() * 1000 - float(value["observed_at_ms"])
        if not -5000 <= age <= max_age_seconds * 1000:
            return None
        payload = value.get("payload")
        if not isinstance(payload, dict) or payload.get("storage_pressure") is not True:
            return None
        if payload.get("state") != "degraded" or payload.get("work_complete") is not False or payload.get("completion_authority") is not False:
            return None
        return value
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _clear_emergency(primary_path: Path) -> None:
    try:
        path = _emergency_path(primary_path)
        value = json.loads(path.read_text())
        identity = _process_identity(os.getpid())
        if identity and value.get("pid") == os.getpid() and value.get("start_ticks") == identity[1]:
            path.unlink()
    except (OSError, ValueError, TypeError):
        pass


def clear_emergency_status(primary_path: Path) -> None:
    """Clear only this process's marker after its real durable recovery."""
    _clear_emergency(primary_path)


class StorageGuard:
    """Keep owned work alive, but defer new work and durable commits under pressure."""
    def __init__(self, status_path: Path, paths: Iterable[Path], *, check_continue: Callable[[], None],
                 minimum_free_bytes: int = MINIMUM_FREE_BYTES, retry_seconds: float = 5):
        self.status_path = status_path.resolve()
        self.paths = list(paths)
        self.check_continue = check_continue
        self.minimum_free_bytes = minimum_free_bytes
        self.retry_seconds = max(0.01, min(float(retry_seconds), 5))
        self.pressure: dict[str, Any] | None = None
        self.last_payload: dict[str, Any] = {}

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def admission(self) -> dict[str, Any]:
        self.check_continue()
        result = check_storage_capacity(self.paths, minimum_free_bytes=self.minimum_free_bytes)
        if not result["ok"]:
            self.pressure = result
        return result

    def _publish(self, error: BaseException | None = None) -> None:
        publish_emergency_status(self.status_path, self.last_payload, error=error)

    def persist(self, path: Path, text: str, writer: Callable[[Path, str], None]) -> None:
        if path.resolve() == self.status_path:
            payload = json.loads(text)
            self.last_payload = payload
            capacity = check_storage_capacity(self.paths, minimum_free_bytes=self.minimum_free_bytes,
                                              durable=self.pressure is not None)
            if not capacity["ok"]:
                self.pressure = capacity
            if self.pressure and not capacity["ok"]:
                payload = {**payload, "state": "degraded", "phase": "storage_pressure",
                           "storage_original_phase": payload.get("phase", ""), "storage_pressure": True,
                           "backpressure": True, "storage_reason": capacity["reason"]}
            try:
                writer(path, json.dumps(payload, indent=2) + "\n")
            except OSError as error:
                if not is_storage_pressure(error):
                    raise
                self.pressure = {"ok": False, "reason": errno.errorcode[error.errno], "errno": error.errno}
                self._publish(error)
                return
            if capacity["ok"]:
                self.pressure = None
                _clear_emergency(self.status_path)
            else:
                self._publish()
            return
        # Retain the producer's real return/evidence on this call stack.
        self.retry_io(lambda: writer(path, text))

    def retry_io(self, operation: Callable[[], Any]) -> Any:
        """Retry a pre-admission or durability operation without rerunning work."""
        while True:
            self.check_continue()
            if self.pressure is None or self.admission()["ok"]:
                try:
                    return operation()
                except OSError as error:
                    if not is_storage_pressure(error):
                        raise
                    self.pressure = {"ok": False, "reason": errno.errorcode[error.errno], "errno": error.errno}
                    self._publish(error)
            else:
                self._publish()
            time.sleep(self.retry_seconds)


def persist_text(path: Path, text: str, writer: Callable[[Path, str], None]) -> None:
    guard = _ACTIVE.get()
    if guard is None:
        writer(path, text)
    else:
        guard.persist(path, text, writer)


@contextmanager
def open_command_log(path: Path, header: str) -> Iterator[Any]:
    """Do not spawn a command until its real output stream can be opened."""
    def open_once():
        handle = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            handle.write(header)
            handle.flush()
            return handle
        except BaseException:
            if handle is not None:
                try:
                    handle.close()
                except OSError as error:
                    if not is_storage_pressure(error):
                        raise
            raise
    guard = _ACTIVE.get()
    handle = guard.retry_io(open_once) if guard else open_once()
    try:
        yield handle
    finally:
        handle.close()
