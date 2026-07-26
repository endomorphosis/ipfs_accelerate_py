"""Durable taskboard updates and event-driven runtime wakeup primitives.

The watcher in this module is deliberately only a *hint* source.  Correctness
comes from comparing bounded, canonical metadata snapshots after every native
notification and after a low-frequency safety timeout.  This keeps duplicate
or coalesced filesystem notifications harmless and preserves liveness on
filesystems which do not implement native change notification.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import select
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, TextIO

from .control_contracts import EventCursor


PATH_METADATA_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/path-metadata@1"
)
PROJECTION_DELTA_CHECKPOINT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/projection-delta-checkpoint@1"
)
EVENT_DRIVEN_RUNTIME_REQUIREMENT_ID: Final = (
    "asi-117:event-driven-delta-checkpoint-runtime"
)
DEFAULT_METADATA_ENTRY_LIMIT: Final = 256
DEFAULT_METADATA_DEPTH_LIMIT: Final = 6
DEFAULT_SAFETY_INTERVAL_SECONDS: Final = 300.0
_MAX_CHECKPOINT_BYTES: Final = 1024 * 1024


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("runtime metadata must contain canonical JSON values") from exc


def _content_id(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:{hashlib.sha256(_canonical_json_bytes(value)).hexdigest()}"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = -1
        if directory >= 0:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def locked_taskboard(path: Path) -> Iterator[TextIO]:
    """Lock a taskboard's inode while a scanner performs read-modify-write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def replace_locked_taskboard(stream: TextIO, text: str) -> bool:
    """Replace a locked taskboard only when its exact text changed.

    ``False`` means the current bytes already matched.  That path deliberately
    performs neither a write nor an fsync, which makes idempotent refill and
    drained-board passes observable as true no-ops.
    """

    stream.seek(0)
    if stream.read() == text:
        return False
    stream.seek(0)
    stream.truncate()
    stream.write(text)
    stream.flush()
    os.fsync(stream.fileno())
    return True


@dataclass(frozen=True)
class PathMetadata:
    """A bounded canonical metadata snapshot for one file or directory."""

    path: str
    exists: bool
    kind: str
    device: int = 0
    inode: int = 0
    mode: int = 0
    size: int = 0
    mtime_ns: int = 0
    ctime_ns: int = 0
    entry_count: int = 0
    entries_scanned: int = 0
    entries_truncated: bool = False
    entries_digest: str = ""
    error: str = ""
    schema: str = PATH_METADATA_SCHEMA
    metadata_id: str = ""

    def __post_init__(self) -> None:
        if self.schema != PATH_METADATA_SCHEMA:
            raise ValueError("unsupported path metadata schema")
        if not self.path:
            raise ValueError("path metadata requires a path")
        if self.kind not in {"missing", "file", "directory", "symlink", "other", "error"}:
            raise ValueError("unsupported path metadata kind")
        for name in (
            "device",
            "inode",
            "mode",
            "size",
            "mtime_ns",
            "ctime_ns",
            "entry_count",
            "entries_scanned",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected = _content_id("path-metadata", self._identity_payload())
        if self.metadata_id and self.metadata_id != expected:
            raise ValueError("path metadata identity does not match")
        object.__setattr__(self, "metadata_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "exists": self.exists,
            "kind": self.kind,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "entry_count": self.entry_count,
            "entries_scanned": self.entries_scanned,
            "entries_truncated": self.entries_truncated,
            "entries_digest": self.entries_digest,
            "error": self.error,
        }

    @property
    def cursor(self) -> str:
        """Return the canonical cursor for change comparisons."""

        return self.metadata_id

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "metadata_id": self.metadata_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PathMetadata":
        return cls(
            schema=str(value.get("schema") or ""),
            path=str(value.get("path") or ""),
            exists=bool(value.get("exists")),
            kind=str(value.get("kind") or ""),
            device=int(value.get("device") or 0),
            inode=int(value.get("inode") or 0),
            mode=int(value.get("mode") or 0),
            size=int(value.get("size") or 0),
            mtime_ns=int(value.get("mtime_ns") or 0),
            ctime_ns=int(value.get("ctime_ns") or 0),
            entry_count=int(value.get("entry_count") or 0),
            entries_scanned=int(value.get("entries_scanned") or 0),
            entries_truncated=bool(value.get("entries_truncated")),
            entries_digest=str(value.get("entries_digest") or ""),
            error=str(value.get("error") or ""),
            metadata_id=str(value.get("metadata_id") or ""),
        )

    @classmethod
    def capture(
        cls,
        path: Path | str,
        *,
        max_entries: int = DEFAULT_METADATA_ENTRY_LIMIT,
        max_depth: int = DEFAULT_METADATA_DEPTH_LIMIT,
    ) -> "PathMetadata":
        """Capture stable stat metadata and a bounded directory-tree digest."""

        target = Path(path)
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        display_path = str(target.absolute())
        try:
            stat_result = target.lstat()
        except FileNotFoundError:
            return cls(path=display_path, exists=False, kind="missing")
        except OSError as exc:
            return cls(
                path=display_path,
                exists=False,
                kind="error",
                error=f"{type(exc).__name__}:{exc.errno or 0}",
            )

        if target.is_symlink():
            kind = "symlink"
        elif target.is_file():
            kind = "file"
        elif target.is_dir():
            kind = "directory"
        else:
            kind = "other"

        entries: list[tuple[str, int, int, int, int, int, str]] = []
        entry_count = 0
        truncated = False
        if kind == "directory":
            pending: deque[tuple[Path, str, int]] = deque([(target, "", 0)])
            while pending:
                directory, relative_parent, depth = pending.popleft()
                try:
                    child_stream = os.scandir(directory)
                except OSError as exc:
                    entries.append(
                        (
                            relative_parent,
                            0,
                            0,
                            0,
                            0,
                            0,
                            f"error:{exc.errno or 0}",
                        )
                    )
                    continue
                with child_stream:
                    for child in child_stream:
                        entry_count += 1
                        if len(entries) >= max_entries:
                            truncated = True
                            break
                        relative = (
                            f"{relative_parent}/{child.name}"
                            if relative_parent
                            else child.name
                        )
                        try:
                            child_stat = child.stat(follow_symlinks=False)
                            if child.is_symlink():
                                child_kind = "symlink"
                            elif child.is_dir(follow_symlinks=False):
                                child_kind = "directory"
                            elif child.is_file(follow_symlinks=False):
                                child_kind = "file"
                            else:
                                child_kind = "other"
                            entries.append(
                                (
                                    relative,
                                    int(child_stat.st_mode),
                                    int(child_stat.st_size),
                                    int(child_stat.st_mtime_ns),
                                    int(child_stat.st_ctime_ns),
                                    int(child_stat.st_ino),
                                    child_kind,
                                )
                            )
                            if (
                                child_kind == "directory"
                                and depth < max_depth
                                and len(entries) < max_entries
                            ):
                                pending.append(
                                    (Path(child.path), relative, depth + 1)
                                )
                        except OSError as exc:
                            entries.append(
                                (
                                    relative,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    f"error:{exc.errno or 0}",
                                )
                            )
                if truncated:
                    break
            entries.sort(key=lambda item: item[0])
        entries_digest = (
            "sha256:" + hashlib.sha256(_canonical_json_bytes(entries)).hexdigest()
            if entries
            else ""
        )
        return cls(
            path=display_path,
            exists=True,
            kind=kind,
            device=int(stat_result.st_dev),
            inode=int(stat_result.st_ino),
            mode=int(stat_result.st_mode),
            size=int(stat_result.st_size),
            mtime_ns=int(stat_result.st_mtime_ns),
            ctime_ns=int(stat_result.st_ctime_ns),
            entry_count=entry_count,
            entries_scanned=len(entries),
            entries_truncated=truncated,
            entries_digest=entries_digest,
        )


def path_metadata(
    path: Path | str,
    *,
    max_entries: int = DEFAULT_METADATA_ENTRY_LIMIT,
    max_depth: int = DEFAULT_METADATA_DEPTH_LIMIT,
) -> PathMetadata:
    """Compatibility-friendly function form of :meth:`PathMetadata.capture`."""

    return PathMetadata.capture(
        path,
        max_entries=max_entries,
        max_depth=max_depth,
    )


metadata_snapshot = path_metadata


def task_ids_from_artifact_names(
    directory: Path,
    *,
    task_prefix: str,
) -> set[str]:
    """Recover allocated display IDs from durable discovery filenames."""

    if not directory.exists():
        return set()
    normalized = task_prefix.rstrip("-") + "-"
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?P<number>\d+)(?!\d)",
        flags=re.IGNORECASE,
    )
    task_ids: set[str] = set()
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        for match in pattern.finditer(path.name):
            number = int(match.group("number"))
            task_ids.add(f"{normalized}{number:03d}")
    return task_ids


def _projection_delta(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    changed = {
        key: current[key]
        for key in sorted(current)
        if key not in previous or previous[key] != current[key]
    }
    removed = [key for key in sorted(previous) if key not in current]
    return {"set": changed, "remove": removed}


@dataclass(frozen=True)
class ProjectionCheckpointResult:
    """Result of one conditional projection/checkpoint materialization."""

    changed: bool
    projection_changed: bool
    cursor_changed: bool
    checkpoint_id: str
    projection_id: str
    delta: Mapping[str, Any] = field(default_factory=dict)
    write_count: int = 0

    @property
    def written(self) -> bool:
        return self.write_count > 0


class ProjectionDeltaCheckpointStore:
    """Atomically bind a projection and canonical event cursor.

    The durable record contains the recoverable current projection plus the
    exact top-level delta from the preceding checkpoint.  Identical projection
    and cursor inputs return without opening a temporary output or fsyncing.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        max_bytes: int = _MAX_CHECKPOINT_BYTES,
    ) -> None:
        self.path = Path(path)
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._thread_lock = threading.RLock()
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._thread_lock:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, Any] | None:
        try:
            raw = self.path.read_bytes()
        except (FileNotFoundError, OSError):
            return None
        if len(raw) > self.max_bytes:
            raise ValueError("projection checkpoint exceeds persistence bound")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("projection checkpoint is malformed") from exc
        if not isinstance(value, dict):
            raise ValueError("projection checkpoint must be an object")
        if value.get("schema") != PROJECTION_DELTA_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported projection checkpoint schema")
        expected = _content_id(
            "projection-checkpoint",
            {key: item for key, item in value.items() if key != "checkpoint_id"},
        )
        if value.get("checkpoint_id") != expected:
            raise ValueError("projection checkpoint identity does not match")
        projection = value.get("projection")
        if not isinstance(projection, dict):
            raise ValueError("projection checkpoint projection must be an object")
        cursor = value.get("cursor")
        if not isinstance(cursor, Mapping):
            raise ValueError("projection checkpoint cursor must be an object")
        EventCursor.from_dict(cursor)
        return value

    def load(self) -> tuple[dict[str, Any], EventCursor] | None:
        # Writers publish complete checkpoints with ``os.replace``.  A reader
        # therefore sees either the previous complete file or the next one and
        # does not need to create/acquire the sidecar lock.  Keeping this path
        # lock-free is important for read-only daemon startup: merely
        # inspecting an absent checkpoint must not create repository state.
        record = self._read_unlocked()
        if record is None:
            return None
        return (
            dict(record["projection"]),
            EventCursor.from_dict(record["cursor"]),
        )

    def load_record(self) -> dict[str, Any] | None:
        record = self._read_unlocked()
        return None if record is None else dict(record)

    def materialize(
        self,
        projection: Mapping[str, Any],
        cursor: EventCursor | Mapping[str, Any] | str,
    ) -> ProjectionCheckpointResult:
        """Persist one changed projection/cursor pair and its compact delta."""

        if not isinstance(projection, Mapping):
            raise TypeError("projection must be a mapping")
        canonical_projection = json.loads(_canonical_json_bytes(projection))
        if not isinstance(canonical_projection, dict):
            raise TypeError("projection must be a JSON object")
        if isinstance(cursor, str):
            canonical_cursor = EventCursor.from_token(cursor)
        elif isinstance(cursor, EventCursor):
            canonical_cursor = cursor
        elif isinstance(cursor, Mapping):
            canonical_cursor = EventCursor.from_dict(cursor)
        else:
            raise TypeError("cursor must be an EventCursor, record, or token")

        with self._guard():
            previous = self._read_unlocked()
            previous_projection = (
                dict(previous.get("projection") or {}) if previous else {}
            )
            previous_cursor = (
                EventCursor.from_dict(previous["cursor"]) if previous else None
            )
            if previous_cursor is not None:
                if canonical_cursor.stream_id != previous_cursor.stream_id:
                    raise ValueError(
                        "projection checkpoint cursor belongs to a different stream"
                    )
                if canonical_cursor.position < previous_cursor.position:
                    raise ValueError(
                        "projection checkpoint cursor cannot move backwards"
                    )
                if (
                    canonical_cursor.position == previous_cursor.position
                    and canonical_cursor != previous_cursor
                ):
                    raise ValueError(
                        "projection checkpoint cursor anchor changed at the same position"
                    )
            projection_changed = (
                previous is None or previous_projection != canonical_projection
            )
            cursor_changed = (
                previous_cursor is None or previous_cursor != canonical_cursor
            )
            delta = _projection_delta(
                previous_projection,
                canonical_projection,
            )
            projection_id = _content_id(
                "runtime-projection", canonical_projection
            )
            if not projection_changed and not cursor_changed:
                return ProjectionCheckpointResult(
                    changed=False,
                    projection_changed=False,
                    cursor_changed=False,
                    checkpoint_id=str(previous.get("checkpoint_id") or ""),
                    projection_id=projection_id,
                    delta=delta,
                    write_count=0,
                )
            generation = int(previous.get("generation") or 0) + 1 if previous else 1
            record: dict[str, Any] = {
                "schema": PROJECTION_DELTA_CHECKPOINT_SCHEMA,
                "generation": generation,
                "projection_id": projection_id,
                "projection": canonical_projection,
                "cursor": canonical_cursor.to_record(),
                "delta": delta,
            }
            record["checkpoint_id"] = _content_id(
                "projection-checkpoint", record
            )
            encoded = _canonical_json_bytes(record) + b"\n"
            if len(encoded) > self.max_bytes:
                raise ValueError("projection checkpoint exceeds persistence bound")
            _atomic_write(self.path, encoded)
            return ProjectionCheckpointResult(
                changed=True,
                projection_changed=projection_changed,
                cursor_changed=cursor_changed,
                checkpoint_id=record["checkpoint_id"],
                projection_id=projection_id,
                delta=delta,
                write_count=1,
            )

    # Concise aliases for callers which model checkpoints as conditional saves.
    checkpoint = materialize
    store = materialize


class NativeWatcherUnavailable(OSError):
    """Native directory notification is unavailable for the selected targets."""


class BlockingTimerWatcher:
    """Blocking pipe/timer fallback with no periodic spin loop."""

    backend = "blocking_timer"
    native = False

    def __init__(self, _paths: Iterable[Path | str] = ()) -> None:
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._read_fd, False)
        os.set_blocking(self._write_fd, False)
        self._closed = False

    def notify(self) -> None:
        if self._closed:
            return
        try:
            os.write(self._write_fd, b"\0")
        except BlockingIOError:
            pass
        except OSError:
            if not self._closed:
                raise

    wake = notify

    def wait(self, timeout: float | None) -> bool:
        if self._closed:
            return False
        selected_timeout = None if timeout is None else max(0.0, float(timeout))
        try:
            ready, _writable, _errors = select.select(
                [self._read_fd], [], [], selected_timeout
            )
        except InterruptedError:
            return True
        if not ready:
            return False
        while True:
            try:
                if not os.read(self._read_fd, 4096):
                    break
            except BlockingIOError:
                break
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._read_fd)
        os.close(self._write_fd)

    def __enter__(self) -> "BlockingTimerWatcher":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class LinuxDirectoryWatcher:
    """Small stdlib/ctypes inotify adapter used only as a wakeup hint."""

    backend = "inotify"
    native = True

    _IN_NONBLOCK = getattr(os, "O_NONBLOCK", 0x800)
    _IN_CLOEXEC = getattr(os, "O_CLOEXEC", 0x80000)
    _IN_MASK = (
        0x00000002  # IN_MODIFY
        | 0x00000004  # IN_ATTRIB
        | 0x00000008  # IN_CLOSE_WRITE
        | 0x00000040  # IN_MOVED_FROM
        | 0x00000080  # IN_MOVED_TO
        | 0x00000100  # IN_CREATE
        | 0x00000200  # IN_DELETE
        | 0x00000400  # IN_DELETE_SELF
        | 0x00000800  # IN_MOVE_SELF
    )

    def __init__(
        self,
        paths: Iterable[Path | str],
        *,
        max_directories: int = DEFAULT_METADATA_ENTRY_LIMIT,
    ) -> None:
        if os.name != "posix" or not hasattr(ctypes, "CDLL"):
            raise NativeWatcherUnavailable(
                errno.ENOSYS, "native directory notification is unavailable"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        initializer = getattr(libc, "inotify_init1", None)
        add_watch = getattr(libc, "inotify_add_watch", None)
        if initializer is None or add_watch is None:
            raise NativeWatcherUnavailable(
                errno.ENOSYS, "inotify is unavailable"
            )
        initializer.argtypes = [ctypes.c_int]
        initializer.restype = ctypes.c_int
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        fd = initializer(self._IN_NONBLOCK | self._IN_CLOEXEC)
        if fd < 0:
            error_number = ctypes.get_errno()
            raise NativeWatcherUnavailable(
                error_number, os.strerror(error_number)
            )
        self._fd = fd
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._read_fd, False)
        os.set_blocking(self._write_fd, False)
        self._closed = False
        self._watch_descriptors: dict[int, Path] = {}
        try:
            directories = self._watch_directories(
                paths, max_directories=max_directories
            )
            if not directories:
                directories = [Path.cwd()]
            for directory in directories:
                descriptor = add_watch(
                    self._fd,
                    os.fsencode(directory),
                    self._IN_MASK,
                )
                if descriptor < 0:
                    error_number = ctypes.get_errno()
                    raise NativeWatcherUnavailable(
                        error_number,
                        f"cannot watch {directory}: {os.strerror(error_number)}",
                    )
                self._watch_descriptors[int(descriptor)] = directory
        except Exception:
            self.close()
            raise

    @staticmethod
    def _nearest_existing_directory(path: Path) -> Path:
        candidate = path if path.is_dir() else path.parent
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    @classmethod
    def _watch_directories(
        cls,
        paths: Iterable[Path | str],
        *,
        max_directories: int,
    ) -> list[Path]:
        directories: list[Path] = []
        seen: set[Path] = set()
        pending: deque[Path] = deque(
            cls._nearest_existing_directory(Path(path).absolute())
            for path in paths
        )
        while pending and len(directories) < max_directories:
            directory = pending.popleft()
            try:
                key = directory.resolve()
            except OSError:
                key = directory.absolute()
            if key in seen or not directory.is_dir():
                continue
            seen.add(key)
            directories.append(directory)
            try:
                child_stream = os.scandir(directory)
            except OSError:
                continue
            with child_stream:
                for item in child_stream:
                    if len(directories) + len(pending) >= max_directories:
                        break
                    try:
                        is_directory = item.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_directory:
                        pending.append(Path(item.path))
        return directories

    def notify(self) -> None:
        if self._closed:
            return
        try:
            os.write(self._write_fd, b"\0")
        except BlockingIOError:
            pass

    wake = notify

    def wait(self, timeout: float | None) -> bool:
        if self._closed:
            return False
        selected_timeout = None if timeout is None else max(0.0, float(timeout))
        try:
            ready, _writable, _errors = select.select(
                [self._fd, self._read_fd], [], [], selected_timeout
            )
        except InterruptedError:
            return True
        if not ready:
            return False
        if self._read_fd in ready:
            try:
                while os.read(self._read_fd, 4096):
                    pass
            except BlockingIOError:
                pass
        if self._fd in ready:
            try:
                while os.read(self._fd, 64 * 1024):
                    pass
            except BlockingIOError:
                pass
        return True

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for descriptor in (self._fd, self._read_fd, self._write_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __enter__(self) -> "LinuxDirectoryWatcher":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def create_directory_watcher(
    paths: Iterable[Path | str],
    *,
    prefer_native: bool = True,
) -> LinuxDirectoryWatcher | BlockingTimerWatcher:
    """Create a native watcher, safely falling back to a blocking pipe/timer."""

    selected_paths = tuple(Path(path) for path in paths)
    if prefer_native and selected_paths:
        try:
            return LinuxDirectoryWatcher(selected_paths)
        except (NativeWatcherUnavailable, OSError, ValueError):
            pass
    return BlockingTimerWatcher(selected_paths)


# Public compatibility aliases use capability-oriented rather than OS names.
NativeDirectoryWatcher = LinuxDirectoryWatcher
FallbackDirectoryWatcher = BlockingTimerWatcher
DeltaCheckpointStore = ProjectionDeltaCheckpointStore
ProjectionDeltaStore = ProjectionDeltaCheckpointStore
DirectoryWatcher = LinuxDirectoryWatcher


class RuntimeWakeKind(str, Enum):
    TASK_BOARD = "task_board"
    OBJECTIVE = "objective"
    REPOSITORY = "repository"
    CHILD_PROCESS = "child_process"
    LEASE = "lease"
    VALIDATION = "validation"
    PROVIDER_CAPACITY = "provider_capacity"
    POLICY = "policy"
    OBSERVATION_WINDOW = "observation_window"


RUNTIME_WAKE_KINDS: Final = tuple(item.value for item in RuntimeWakeKind)


@dataclass(frozen=True)
class RuntimeWakeEvent:
    """One coalesced wake decision backed by canonical source cursors."""

    kinds: tuple[RuntimeWakeKind, ...]
    metadata: tuple[PathMetadata, ...] = ()
    semantic_cursors: Mapping[str, str] = field(default_factory=dict)
    reason: str = "notification"
    safety_timer: bool = False

    @property
    def kind(self) -> RuntimeWakeKind:
        return self.kinds[0]

    @property
    def cursor_ids(self) -> tuple[str, ...]:
        values = [item.cursor for item in self.metadata]
        values.extend(
            self.semantic_cursors[key] for key in sorted(self.semantic_cursors)
        )
        return tuple(values)


class RuntimeWakeCoordinator:
    """Block for semantic/native wakeups and reconcile them with metadata.

    ``wait`` is two-phase: it does not advance acknowledged cursors.
    Call :meth:`acknowledge` only after the corresponding projection delta is
    durable.  A failure before acknowledgement therefore replays the wake.
    """

    def __init__(
        self,
        targets: Mapping[
            RuntimeWakeKind | str,
            Path | str | Iterable[Path | str],
        ] | None = None,
        *,
        safety_interval_seconds: float = DEFAULT_SAFETY_INTERVAL_SECONDS,
        prefer_native: bool = True,
        watcher: Any | None = None,
        clock: Any = time.monotonic,
        metadata_entry_limit: int = DEFAULT_METADATA_ENTRY_LIMIT,
        metadata_depth_limit: int = DEFAULT_METADATA_DEPTH_LIMIT,
    ) -> None:
        if safety_interval_seconds <= 0:
            raise ValueError("safety_interval_seconds must be positive")
        self.safety_interval_seconds = float(safety_interval_seconds)
        self.clock = clock
        self.metadata_entry_limit = int(metadata_entry_limit)
        self.metadata_depth_limit = int(metadata_depth_limit)
        self._targets: dict[RuntimeWakeKind, tuple[Path, ...]] = {}
        for raw_kind, raw_paths in (targets or {}).items():
            kind = self._coerce_kind(raw_kind)
            if isinstance(raw_paths, (str, Path)):
                paths = (Path(raw_paths),)
            else:
                paths = tuple(Path(path) for path in raw_paths)
            self._targets[kind] = paths
        all_paths = tuple(
            path for paths in self._targets.values() for path in paths
        )
        self.watcher = watcher or create_directory_watcher(
            all_paths, prefer_native=prefer_native
        )
        self._acknowledged_metadata = self._capture_all()
        self._pending_hints: dict[RuntimeWakeKind, str] = {}
        self._acknowledged_semantic: dict[RuntimeWakeKind, str] = {}
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _coerce_kind(value: RuntimeWakeKind | str) -> RuntimeWakeKind:
        if isinstance(value, RuntimeWakeKind):
            return value
        return RuntimeWakeKind(str(value))

    @property
    def backend(self) -> str:
        return str(getattr(self.watcher, "backend", "custom"))

    @property
    def native(self) -> bool:
        return bool(getattr(self.watcher, "native", False))

    def _capture_all(
        self,
    ) -> dict[RuntimeWakeKind, tuple[PathMetadata, ...]]:
        return {
            kind: tuple(
                PathMetadata.capture(
                    path,
                    max_entries=self.metadata_entry_limit,
                    max_depth=self.metadata_depth_limit,
                )
                for path in paths
            )
            for kind, paths in self._targets.items()
        }

    def notify(
        self,
        kind: RuntimeWakeKind | str,
        *,
        revision: str = "",
    ) -> None:
        """Publish a coalescible hint and interrupt the blocking waiter."""

        selected = self._coerce_kind(kind)
        token = str(revision).strip()
        if not token:
            token = _content_id(
                "wake-hint",
                {
                    "kind": selected.value,
                    "ordinal": time.monotonic_ns(),
                },
            )
        with self._lock:
            self._pending_hints[selected] = token
        self.watcher.notify()

    publish = notify
    signal = notify

    def register_child_process(
        self,
        process: Any,
        *,
        revision: str = "",
    ) -> threading.Thread:
        """Wake when a ``subprocess.Popen``-compatible child exits."""

        def await_child() -> None:
            returncode = process.wait()
            token = revision or f"pid:{getattr(process, 'pid', 0)}:rc:{returncode}"
            self.notify(RuntimeWakeKind.CHILD_PROCESS, revision=token)

        thread = threading.Thread(
            target=await_child,
            name="agent-supervisor-child-wake",
            daemon=True,
        )
        thread.start()
        return thread

    def _candidate(
        self,
        *,
        safety_timer: bool,
    ) -> RuntimeWakeEvent | None:
        current = self._capture_all()
        changed_kinds = {
            kind
            for kind, snapshots in current.items()
            if snapshots != self._acknowledged_metadata.get(kind, ())
        }
        with self._lock:
            hints = dict(self._pending_hints)
        semantic: dict[str, str] = {}
        stale_semantic: list[RuntimeWakeKind] = []
        for kind, token in hints.items():
            # File-backed hints are verified against canonical metadata, so a
            # duplicate native notification cannot trigger an expensive pass.
            if kind in self._targets:
                continue
            if self._acknowledged_semantic.get(kind) != token:
                changed_kinds.add(kind)
                semantic[kind.value] = token
            else:
                stale_semantic.append(kind)
        if stale_semantic:
            with self._lock:
                for kind in stale_semantic:
                    if self._pending_hints.get(kind) == hints[kind]:
                        self._pending_hints.pop(kind, None)
        if safety_timer:
            changed_kinds.add(RuntimeWakeKind.OBSERVATION_WINDOW)
        if not changed_kinds:
            # Discard only proven-spurious file hints.  Semantic hints remain
            # pending until their wake has been durably acknowledged.
            with self._lock:
                for kind in tuple(self._pending_hints):
                    if kind in self._targets:
                        self._pending_hints.pop(kind, None)
            return None
        selected_metadata = tuple(
            snapshot
            for kind in sorted(changed_kinds, key=lambda item: item.value)
            for snapshot in current.get(kind, ())
        )
        return RuntimeWakeEvent(
            kinds=tuple(sorted(changed_kinds, key=lambda item: item.value)),
            metadata=selected_metadata,
            semantic_cursors=semantic,
            reason="safety_timer" if safety_timer else "notification",
            safety_timer=safety_timer,
        )

    def wait(self, timeout: float | None = None) -> RuntimeWakeEvent:
        """Block until a meaningful wake or the safety observation window."""

        if self._closed:
            raise RuntimeError("runtime wake coordinator is closed")
        maximum = (
            self.safety_interval_seconds
            if timeout is None
            else min(self.safety_interval_seconds, max(0.0, float(timeout)))
        )
        deadline = self.clock() + maximum
        while True:
            with self._lock:
                has_pending = bool(self._pending_hints)
            if has_pending:
                candidate = self._candidate(safety_timer=False)
                if candidate is not None:
                    return candidate
            remaining = max(0.0, deadline - self.clock())
            if remaining <= 0:
                candidate = self._candidate(safety_timer=True)
                assert candidate is not None
                return candidate
            notified = bool(self.watcher.wait(remaining))
            if not notified:
                candidate = self._candidate(safety_timer=True)
                assert candidate is not None
                return candidate
            candidate = self._candidate(safety_timer=False)
            if candidate is not None:
                return candidate

    wait_for_wake = wait

    def acknowledge(self, event: RuntimeWakeEvent) -> None:
        """Advance only cursors represented by a successfully applied wake."""

        if not isinstance(event, RuntimeWakeEvent):
            raise TypeError("event must be a RuntimeWakeEvent")
        by_path = {item.path: item for item in event.metadata}
        with self._lock:
            for kind in event.kinds:
                if kind in self._targets:
                    previous = self._acknowledged_metadata.get(kind, ())
                    self._acknowledged_metadata[kind] = tuple(
                        by_path.get(item.path, item) for item in previous
                    )
                    self._pending_hints.pop(kind, None)
                token = event.semantic_cursors.get(kind.value)
                if token:
                    self._acknowledged_semantic[kind] = token
                    if self._pending_hints.get(kind) == token:
                        self._pending_hints.pop(kind, None)

    ack = acknowledge

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.watcher.close()

    def __enter__(self) -> "RuntimeWakeCoordinator":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
