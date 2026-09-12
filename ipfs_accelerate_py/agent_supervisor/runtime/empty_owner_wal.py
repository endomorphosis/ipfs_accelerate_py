"""Preserve an empty, stopped-owner WAL before the R23 absence baseline.

This is file-custody recovery, never database repair or task-settlement evidence.
Every invocation requires fresh native source/endpoint gates, four retained
startup flocks, and an actual canonical-inode OFD writer fence. Journal JSON
records observations; it cannot replace any of those live prerequisites.
"""
from __future__ import annotations

from contextlib import ExitStack
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import time
from typing import Any, Callable, Mapping, Sequence
import uuid


class EmptyOwnerWalError(RuntimeError):
    """Startup remains refused; any preserved original must stay in custody."""


_MAX_RECORD = 32768
_MAX_ENTRIES = 128
_MAX_ENTRY_FILES = 64
_EMPTY_HASH = hashlib.sha256(b"").hexdigest()
_FLAGS = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOATIME
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _RENAMEAT2.restype = ctypes.c_int


def _fail(reason: str) -> None:
    raise EmptyOwnerWalError("empty owner WAL preservation: " + reason)


def _stat(info: os.stat_result) -> dict[str, int]:
    return {"device": info.st_dev, "inode": info.st_ino, "mode": info.st_mode,
            "uid": info.st_uid, "nlink": info.st_nlink, "size": info.st_size,
            "atime_ns": info.st_atime_ns, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns}


def _directory(info: os.stat_result) -> dict[str, int]:
    return {"device": info.st_dev, "inode": info.st_ino, "mode": info.st_mode, "uid": info.st_uid}


def _named(fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _regular(info: os.stat_result, *, modes: set[int], maximum: int) -> None:
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) not in modes
            or not 0 <= info.st_size <= maximum):
        _fail("file identity is unsafe")


def _open_file(stack: ExitStack, parent: int, name: str, *, modes: set[int],
               maximum: int, writable: bool = False) -> tuple[int, dict[str, int]]:
    before = _named(parent, name)
    if before is None:
        _fail("required file is absent")
    _regular(before, modes=modes, maximum=maximum)
    fd = os.open(name, (os.O_RDWR if writable else os.O_RDONLY) | _FLAGS, dir_fd=parent)
    stack.callback(os.close, fd)
    opened = os.fstat(fd)
    _regular(opened, modes=modes, maximum=maximum)
    named = _named(parent, name)
    if named is None or _stat(before) != _stat(opened) or _stat(opened) != _stat(named):
        _fail("file changed during open")
    return fd, _stat(opened)


def _require_file(fd: int, parent: int, name: str, expected: Mapping[str, Any]) -> None:
    named = _named(parent, name)
    wanted = {key: expected[key] for key in _stat(os.fstat(fd))}
    if named is None or _stat(os.fstat(fd)) != wanted or _stat(named) != wanted:
        _fail("retained file or name changed")


def _hash(fd: int) -> str:
    before = os.fstat(fd)
    if not 0 <= before.st_size <= 8 * 1024**3:
        _fail("file hash size exceeds bound")
    digest = hashlib.sha256()
    offset = 0
    deadline = time.monotonic() + 60.0
    while offset < before.st_size:
        if time.monotonic() >= deadline:
            _fail("file hash deadline exceeded")
        chunk = os.pread(fd, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            _fail("file shortened during hashing")
        digest.update(chunk)
        offset += len(chunk)
    if _stat(os.fstat(fd)) != _stat(before):
        _fail("file changed during hashing")
    return digest.hexdigest()


def _encode(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _cid(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail("duplicate journal key")
        value[key] = item
    return value


def _read(parent: int, name: str, *, durable: bool = False) -> dict[str, Any]:
    with ExitStack() as stack:
        fd, expected = _open_file(stack, parent, name, modes={0o600}, maximum=_MAX_RECORD)
        raw = os.pread(fd, _MAX_RECORD + 1, 0)
        _require_file(fd, parent, name, expected)
        if durable:
            os.fsync(fd)
            _require_file(fd, parent, name, expected)
        value = json.loads(raw, object_pairs_hook=_pairs)
        if not isinstance(value, dict) or _encode(value) != raw:
            _fail("noncanonical journal record")
        return value


def _rename(source_fd: int, source: str, target_fd: int, target: str) -> None:
    if _RENAMEAT2 is None:
        _fail("no-replace rename is unavailable")
    if _RENAMEAT2(source_fd, os.fsencode(source), target_fd, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _publish(parent: int, name: str, value: Mapping[str, Any]) -> None:
    """Publish without replacement; interrupted partial files stay diagnostic."""
    raw = _encode(value)
    if len(raw) > _MAX_RECORD:
        _fail("journal record is too large")
    if _named(parent, name) is not None:
        if _read(parent, name, durable=True) != value:
            _fail("journal record already differs")
        os.fsync(parent)
        return
    pending = ".pending-" + uuid.uuid4().hex
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FLAGS, 0o600, dir_fd=parent)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(fd, raw[offset:])
            if count <= 0:
                _fail("journal write made no progress")
            offset += count
        os.fsync(fd)
        _rename(parent, pending, parent, name)
        os.fsync(parent)
    finally:
        os.close(fd)


def _open_directory(stack: ExitStack, parent: int, name: str, *, create: bool) -> tuple[int, dict[str, int]]:
    if create and _named(parent, name) is None:
        os.mkdir(name, 0o700, dir_fd=parent)
        os.fsync(parent)
    before = _named(parent, name)
    if (before is None or not stat.S_ISDIR(before.st_mode) or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o700):
        _fail("journal directory is unsafe")
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | _FLAGS, dir_fd=parent)
    stack.callback(os.close, fd)
    expected = _directory(before)
    if _directory(os.fstat(fd)) != expected or _directory(os.stat(name, dir_fd=parent, follow_symlinks=False)) != expected:
        _fail("journal directory changed")
    return fd, expected


def _require_directory(fd: int, parent: int, name: str, expected: Mapping[str, int]) -> None:
    named = _named(parent, name)
    if named is None or _directory(named) != expected or _directory(os.fstat(fd)) != expected:
        _fail("journal directory binding changed")


def _require_locks(parent: int, database_name: str, descriptors: Sequence[int]) -> None:
    names = [f".{database_name}{suffix}" for suffix in
             (".state-owner.lock", ".migration.lock", ".intent.lock", ".lock")]
    if len(descriptors) != 4 or len(set(descriptors)) != 4:
        _fail("four distinct retained startup lock descriptors are required")
    for fd, name in zip(descriptors, names):
        opened = os.fstat(fd)
        _regular(opened, modes={0o600}, maximum=65536)
        named = _named(parent, name)
        if named is None or _stat(named) != _stat(opened):
            _fail("startup lock name changed")
        key = f"{os.major(opened.st_dev):02x}:{os.minor(opened.st_dev):02x}:{opened.st_ino}"
        lines = Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
        if not any(line.split()[2:] == ["FLOCK", "ADVISORY", "WRITE", str(os.getpid()), key, "0", "EOF"]
                   for line in lines if line.startswith("lock:")):
            _fail("startup lock descriptor does not hold the exclusive flock")


def _require_database_fence(fd: int) -> None:
    opened = os.fstat(fd)
    key = f"{os.major(opened.st_dev):02x}:{os.minor(opened.st_dev):02x}:{opened.st_ino}"
    lines = Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
    if not any(line.split()[2:] == ["OFDLCK", "ADVISORY", "WRITE", "-1", key, "0", "EOF"]
               for line in lines if line.startswith("lock:")):
        _fail("canonical database writer fence is no longer held")


def _source(source: Mapping[str, Any]) -> dict[str, str]:
    if (set(source) != {"candidate_head", "candidate_tree", "store_id"}
            or any(type(source[key]) is not str or not re.fullmatch(r"[0-9a-f]{40}", source[key])
                   for key in ("candidate_head", "candidate_tree"))
            or source["store_id"] != "data/aseh/control.duckdb"):
        _fail("native source identity is invalid")
    return dict(source)


def _observation(value: Any, *, modes: set[int], maximum: int) -> dict[str, Any]:
    keys = {"device", "inode", "mode", "uid", "nlink", "size", "atime_ns", "mtime_ns", "ctime_ns", "sha256"}
    if (not isinstance(value, dict) or set(value) != keys
            or any(type(value[key]) is not int or value[key] < 0 for key in keys - {"sha256"})
            or not stat.S_ISREG(value["mode"]) or stat.S_IMODE(value["mode"]) not in modes
            or value["uid"] != os.geteuid() or value["nlink"] != 1 or value["size"] > maximum
            or type(value["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        _fail("journal file observation is invalid")
    return value


def preserve_empty_owner_wal(*, directory_fd: int, directory_path: Path,
                             database_name: str, lock_descriptors: Sequence[int],
                             source_identity: Mapping[str, Any], native_guard: Callable[[], None]) -> dict[str, Any]:
    """Preserve only an empty WAL, or finish its interrupted custody audit.

    ``native_guard`` rechecks the current native source witness, inert server,
    owner namespace and listener absence. It is invoked before and after each
    publication/move. The function never starts an owner or opens DuckDB.
    """
    try:
        return _preserve(directory_fd=directory_fd, directory_path=directory_path,
                         database_name=database_name, lock_descriptors=lock_descriptors,
                         source_identity=source_identity, native_guard=native_guard)
    except EmptyOwnerWalError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise EmptyOwnerWalError("empty owner WAL preservation: observation or durable audit failed") from exc


def _preserve(*, directory_fd: int, directory_path: Path, database_name: str,
              lock_descriptors: Sequence[int], source_identity: Mapping[str, Any],
              native_guard: Callable[[], None]) -> dict[str, Any]:
    if database_name != "control.duckdb" or not directory_path.is_absolute() or not callable(native_guard):
        _fail("native scope is invalid")
    source = _source(source_identity)
    parent = _directory(os.fstat(directory_fd))
    if not stat.S_ISDIR(parent["mode"]) or parent["uid"] != os.geteuid():
        _fail("canonical parent is unsafe")
    wal_name = database_name + ".wal"
    journal_name = "." + database_name + ".empty-wal-preservation"

    def gate() -> None:
        if (_directory(os.lstat(directory_path)) != parent
                or _directory(os.fstat(directory_fd)) != parent):
            _fail("canonical parent binding changed")
        _require_locks(directory_fd, database_name, lock_descriptors)
        for suffix in (".wal.checkpoint", ".wal.recovery"):
            if _named(directory_fd, database_name + suffix) is not None:
                _fail("additional WAL recovery artifact is present")
        native_guard()

    gate()
    if _named(directory_fd, wal_name) is None and _named(directory_fd, journal_name) is None:
        return {"preserved": False, "reason": "wal_absent", "completion_authority": False}
    with ExitStack() as stack:
        database_fd, db_stat = _open_file(stack, directory_fd, database_name,
            modes={0o600, 0o664}, maximum=8 * 1024**3, writable=True)
        try:
            fcntl.fcntl(database_fd, fcntl.F_OFD_SETLK,
                        struct.pack("hhqqi4x", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        except OSError as exc:
            if exc.errno in {errno.EAGAIN, errno.EACCES}:
                _fail("canonical database writer fence is contended")
            raise
        _require_database_fence(database_fd)
        database = {**db_stat, "sha256": _hash(database_fd)}
        _require_file(database_fd, directory_fd, database_name, database)
        gate()
        # Validate the initial WAL before creating any recovery artifacts.
        if _named(directory_fd, wal_name) is not None:
            _open_file(stack, directory_fd, wal_name, modes={0o600, 0o644, 0o664}, maximum=0)
        journal_fd, journal_stat = _open_directory(stack, directory_fd, journal_name, create=True)

        def journal_gate() -> None:
            gate()
            _require_database_fence(database_fd)
            _require_file(database_fd, directory_fd, database_name, database)
            _require_directory(journal_fd, directory_fd, journal_name, journal_stat)

        names = os.listdir(journal_fd)
        if len(names) > _MAX_ENTRIES or any(not re.fullmatch(r"[0-9a-f]{64}", name) for name in names):
            _fail("journal population is unknown or exceeds its bound")
        pending: list[tuple[str, int, dict[str, int], dict[str, Any]]] = []
        for name in sorted(names):
            entry_fd, entry_stat = _open_directory(stack, journal_fd, name, create=False)
            files = os.listdir(entry_fd)
            if len(files) > _MAX_ENTRY_FILES or any(
                item not in {"prepared.json", "original.wal", "completed.json"}
                and not re.fullmatch(r"\.pending-[0-9a-f]{32}", item) for item in files):
                _fail("journal entry population is unknown or exceeds its bound")
            for item in files:
                if item.startswith(".pending-"):
                    with ExitStack() as diagnostic_stack:
                        _open_file(diagnostic_stack, entry_fd, item, modes={0o600}, maximum=_MAX_RECORD)
            if "prepared.json" not in files:
                if set(files) - {item for item in files if item.startswith(".pending-")}:
                    _fail("journal entry lacks prepared evidence")
                continue  # Failed pre-publication bytes are diagnostic only.
            prepared = _read(entry_fd, "prepared.json", durable=True)
            if (set(prepared) != {"schema", "source", "parent", "database", "wal"}
                    or prepared["schema"] != "aseh-empty-wal-prepared@1" or _cid(prepared) != name
                    or prepared["parent"] != parent):
                _fail("prepared journal binding differs")
            _source(prepared["source"])
            _observation(prepared["database"], modes={0o600, 0o664}, maximum=8 * 1024**3)
            wal = _observation(prepared["wal"], modes={0o600, 0o644, 0o664}, maximum=0)
            if wal["sha256"] != _EMPTY_HASH:
                _fail("prepared WAL is not empty")
            if "completed.json" in files:
                complete = _read(entry_fd, "completed.json", durable=True)
                if (set(complete) != {"schema", "prepared_sha256", "archived_wal", "completion_authority"}
                        or complete["schema"] != "aseh-empty-wal-completed@1"
                        or complete["prepared_sha256"] != name or complete["completion_authority"] is not False):
                    _fail("completed journal binding differs")
                archived = _observation(complete["archived_wal"], modes={0o600, 0o644, 0o664}, maximum=0)
                if {k: v for k, v in archived.items() if k != "ctime_ns"} != {k: v for k, v in wal.items() if k != "ctime_ns"}:
                    _fail("archived WAL differs from original")
                with ExitStack() as archive_stack:
                    archive_fd, _ = _open_file(archive_stack, entry_fd, "original.wal", modes={0o600, 0o644, 0o664}, maximum=0)
                    _require_file(archive_fd, entry_fd, "original.wal", archived)
                    os.fsync(archive_fd)
                os.fsync(entry_fd)
            else:
                pending.append((name, entry_fd, entry_stat, prepared))
        if len(pending) > 1:
            _fail("multiple unfinished preservation entries")
        journal_gate()
        if not pending and _named(directory_fd, wal_name) is None:
            os.fsync(journal_fd)
            os.fsync(directory_fd)
            journal_gate()
            return {"preserved": False, "reason": "wal_already_preserved", "completion_authority": False}
        if pending:
            name, entry_fd, entry_stat, prepared = pending[0]
            if prepared["source"] != source or prepared["database"] != database:
                _fail("unfinished preservation source or database changed")
        else:
            wal_fd, wal_stat = _open_file(stack, directory_fd, wal_name, modes={0o600, 0o644, 0o664}, maximum=0)
            prepared = {"schema": "aseh-empty-wal-prepared@1", "source": source,
                        "parent": parent, "database": database, "wal": {**wal_stat, "sha256": _EMPTY_HASH}}
            name = _cid(prepared)
            if len(names) >= _MAX_ENTRIES and name not in names:
                _fail("journal entry limit reached")
            entry_fd, entry_stat = _open_directory(stack, journal_fd, name, create=True)
            journal_gate()
            _require_file(wal_fd, directory_fd, wal_name, prepared["wal"])
            os.fsync(wal_fd)
            os.fsync(directory_fd)
            _publish(entry_fd, "prepared.json", prepared)

        def entry_gate() -> None:
            journal_gate()
            _require_directory(entry_fd, journal_fd, name, entry_stat)
            if _read(entry_fd, "prepared.json") != prepared:
                _fail("prepared evidence changed")

        entry_gate()
        original = prepared["wal"]
        if _named(entry_fd, "original.wal") is None:
            wal_fd, _ = _open_file(stack, directory_fd, wal_name, modes={0o600, 0o644, 0o664}, maximum=0)
            _require_file(wal_fd, directory_fd, wal_name, original)
            entry_gate()
            _require_file(wal_fd, directory_fd, wal_name, original)
            _rename(directory_fd, wal_name, entry_fd, "original.wal")
            # A race can only leave preserved evidence: never remove a raced
            # replacement or continue startup after a mismatched retirement.
            archived = {**_stat(os.fstat(wal_fd)), "sha256": _hash(wal_fd)}
            if {k: v for k, v in archived.items() if k != "ctime_ns"} != {k: v for k, v in original.items() if k != "ctime_ns"}:
                _fail("WAL changed at retirement")
            _require_file(wal_fd, entry_fd, "original.wal", archived)
            os.fsync(wal_fd)
            os.fsync(entry_fd)
            os.fsync(directory_fd)
        else:
            wal_fd, wal_stat = _open_file(stack, entry_fd, "original.wal", modes={0o600, 0o644, 0o664}, maximum=0)
            archived = {**wal_stat, "sha256": _hash(wal_fd)}
            if {k: v for k, v in archived.items() if k != "ctime_ns"} != {k: v for k, v in original.items() if k != "ctime_ns"}:
                _fail("retained WAL differs from original")
        entry_gate()
        if _named(directory_fd, wal_name) is not None:
            _fail("canonical WAL is not absent after preservation")
        _require_file(wal_fd, entry_fd, "original.wal", archived)
        if _hash(database_fd) != database["sha256"]:
            _fail("canonical database bytes changed")
        # Repeat durability barriers when resuming an interrupted audit too.
        os.fsync(wal_fd)
        os.fsync(entry_fd)
        os.fsync(directory_fd)
        complete = {"schema": "aseh-empty-wal-completed@1", "prepared_sha256": name,
                    "archived_wal": archived, "completion_authority": False}
        _publish(entry_fd, "completed.json", complete)
        entry_gate()
        if _read(entry_fd, "completed.json") != complete:
            _fail("completed evidence changed")
        _require_file(wal_fd, entry_fd, "original.wal", archived)
        if _named(directory_fd, wal_name) is not None:
            _fail("canonical WAL reappeared after audit")
        return {"preserved": True, "prepared_sha256": name, "completion_authority": False}
