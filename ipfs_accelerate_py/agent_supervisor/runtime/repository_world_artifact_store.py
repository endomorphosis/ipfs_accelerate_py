"""Retain bounded historical world inputs in an explicitly selected owner store.

These artifacts contain task and source context. Only their closed references
belong in public invocation receipts. Storage neither issues live packets nor
attests provider delivery, and never overwrites an existing invocation file.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import time

from .repository_world_invocation_replay import (
    MAX_BYTES, freeze_repository_world_invocation,
)


REFERENCE_SCHEMA = "repository-world-invocation-artifact-reference@1"
MAX_ARTIFACTS = 16
MAX_STORE_BYTES = 16 * 1024 * 1024
LOCK_TIMEOUT_SECONDS = .25
_NAME = re.compile(r"[0-9a-f]{32}\.json")


def _path(value) -> Path:
    path = Path(value)
    if (not path.is_absolute() or ".." in path.parts or path == Path("/")
            or "\x00" in str(path) or len(str(path).encode("utf-8")) > 4096):
        raise ValueError("absolute bounded world artifact directory required")
    return path


def _open_directory(path: Path) -> int:
    """Open every component without following links, including parent links."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                                      | os.O_CLOEXEC, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _private_directory(info):
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("world artifact directory requires owner mode 0700")


def _private_file(info, *, allow_empty=False):
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600
            or not (0 if allow_empty else 1) <= info.st_size <= MAX_BYTES):
        raise ValueError("world artifact requires one bounded owner mode 0600 regular file")


def prepare_repository_world_artifact_directory(directory: Path, *, workspace: Path) -> tuple[int, int]:
    """Create only the final directory under an existing parent; pin its inode.

An existing directory must already be private. Unsafe paths and permissions are
refused without modification. No artifact content is written during admission.
"""
    directory, workspace = _path(directory), _path(workspace)
    if directory.is_relative_to(workspace) or workspace.is_relative_to(directory):
        raise ValueError("world artifact directory must be separate from the source workspace")
    parent = _open_directory(directory.parent)
    descriptor = None
    try:
        try:
            os.mkdir(directory.name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
        descriptor = os.open(directory.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                             | os.O_CLOEXEC, dir_fd=parent)
        info = os.fstat(descriptor)
        _private_directory(info)
        return info.st_dev, info.st_ino
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


@contextmanager
def _locked_directory(directory, expected_identity):
    if (type(expected_identity) is not tuple or len(expected_identity) != 2
            or any(type(value) is not int for value in expected_identity)):
        raise ValueError("world artifact directory identity required")
    descriptor = _open_directory(_path(directory))
    try:
        info = os.fstat(descriptor)
        _private_directory(info)
        if (info.st_dev, info.st_ino) != expected_identity:
            raise ValueError("world artifact directory changed after admission")
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("world artifact directory is busy") from None
                time.sleep(.01)
        yield descriptor
    finally:
        os.close(descriptor)


def _capacity(descriptor, added_bytes):
    total, count = 0, 0
    with os.scandir(descriptor) as entries:
        for entry in entries:
            count += 1
            if count >= MAX_ARTIFACTS or _NAME.fullmatch(entry.name) is None:
                raise ValueError("world artifact inventory exceeds its bound or has unknown entries")
            info = entry.stat(follow_symlinks=False)
            # A failed exclusive write can leave an empty unreferenced file.
            # Count its slot without preventing later captures after recovery.
            # Completed artifacts still require nonempty bytes below.
            _private_file(info, allow_empty=True)
            total += info.st_size
            if total + added_bytes > MAX_STORE_BYTES:
                raise ValueError("world artifact store byte quota exceeded")
    if added_bytes > MAX_STORE_BYTES:
        raise ValueError("world artifact store byte quota exceeded")


def store_repository_world_invocation_artifact(*, directory: Path,
        expected_identity: tuple[int, int], packet_artifact: bytes,
        expected_packet_sha256: str, invocation_receipt: Mapping) -> dict:
    """Join a frozen packet to its finalized receipt and retain exact bytes.

The packet must have been frozen before dispatch. Finalization uses historical
reconstruction only, so a worker edit or closed source owner does not erase the
recorded input. The caller keeps both this reference and the separate receipt.
"""
    if not isinstance(invocation_receipt, Mapping):
        raise ValueError("world invocation receipt required")
    invocation_id = invocation_receipt.get("invocation_id")
    if type(invocation_id) is not str or re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None:
        raise ValueError("world invocation identity required")
    raw = freeze_repository_world_invocation(packet_artifact=packet_artifact,
        expected_packet_sha256=expected_packet_sha256, invocation_receipt=invocation_receipt)
    if not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("world invocation artifact exceeds its byte bound")
    name = invocation_id + ".json"
    with _locked_directory(directory, expected_identity) as directory_fd:
        _capacity(directory_fd, len(raw))
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                             | os.O_CLOEXEC | os.O_NONBLOCK, 0o600, dir_fd=directory_fd)
        try:
            os.fchmod(descriptor, 0o600)
            pending = memoryview(raw)
            while pending:
                written = os.write(descriptor, pending)
                if written <= 0:
                    raise OSError("world artifact write made no progress")
                pending = pending[written:]
            info = os.fstat(descriptor)
            _private_file(info)
            if info.st_size != len(raw):
                raise ValueError("world invocation artifact size changed during capture")
            os.fsync(descriptor)
            os.fsync(directory_fd)
        finally:
            # A failed exclusive write may leave an unreferenced partial file.
            # Never unlink an entry that another writer could have substituted.
            os.close(descriptor)
    return dict(schema=REFERENCE_SCHEMA, invocation_id=invocation_id,
        artifact_sha256=hashlib.sha256(raw).hexdigest(), artifact_bytes=len(raw),
        packet_artifact_sha256=expected_packet_sha256)
