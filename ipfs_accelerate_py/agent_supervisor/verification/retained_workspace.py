"""Bounded full-tree evidence for retained callback workspaces.

This is source evidence only, never a historical callback receipt or acceptance.
The byte serialization matches the legacy retained-workspace fingerprint.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat
import time

MAX_CONTENT_BYTES = 4 * 1024 * 1024 * 1024
MAX_ENTRIES = 200_000
MAX_SECONDS = 120.0
CHUNK_BYTES = 1024 * 1024


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def fingerprint_retained_tree(
    workspace: Path, *, max_bytes: int = MAX_CONTENT_BYTES,
    max_entries: int = MAX_ENTRIES, max_seconds: float = MAX_SECONDS,
) -> dict[str, str | int]:
    """Read every byte with bounded memory, work and a final mutation check.

    Do not follow symlinks, skip ignored files, or reuse unverified cached hashes.
    Callers must also hold their native source/callback custody through use of
    the result; this scan cannot establish exclusive ownership of a workspace.
    """
    if (type(max_bytes) is not int or max_bytes < 1
            or type(max_entries) is not int or max_entries < 1
            or isinstance(max_seconds, bool)
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("invalid retained workspace fingerprint limits")
    workspace = Path(workspace)
    deadline = time.monotonic() + max_seconds

    def check_time() -> None:
        if time.monotonic() >= deadline:
            raise RuntimeError("retained workspace exceeds fingerprint time budget")

    def walk_error(error: OSError) -> None:
        raise error

    try:
        root_before = workspace.lstat()
        if not stat.S_ISDIR(root_before.st_mode):
            raise RuntimeError("retained workspace is not a directory")
        # A pinned root descriptor also prevents a replaced root from redirecting
        # the walk. fwalk pins each directory; file opens are relative to it.
        root_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if _identity(os.fstat(root_fd)) != _identity(root_before):
                raise RuntimeError("retained workspace changed during fingerprint")

            def scan(read_content: bool) -> tuple[str, str, int, int]:
                content = hashlib.sha256()
                metadata = hashlib.sha256()
                entries = total = 0
                for directory, directories, files, directory_fd in os.fwalk(
                    ".", dir_fd=root_fd, follow_symlinks=False, onerror=walk_error,
                ):
                    check_time()
                    directories.sort()
                    files.sort()
                    for name in (*directories, *files):
                        check_time()
                        entries += 1
                        if entries > max_entries:
                            raise RuntimeError("retained workspace exceeds fingerprint entry budget")
                        relative = (Path(directory) / name).as_posix()
                        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        prefix = relative.encode("utf-8", errors="surrogateescape") + b"\0"
                        metadata.update(prefix + repr(_identity(before)).encode("ascii") + b"\0")
                        content.update(prefix + str(before.st_mode).encode("ascii") + b"\0")
                        if stat.S_ISLNK(before.st_mode):
                            target = os.readlink(name, dir_fd=directory_fd).encode(
                                "utf-8", errors="surrogateescape")
                            total += len(target)
                            content.update(b"L" + target)
                        elif stat.S_ISDIR(before.st_mode):
                            content.update(b"D")
                        elif stat.S_ISREG(before.st_mode):
                            total += before.st_size
                            if total > max_bytes:
                                raise RuntimeError("retained workspace exceeds fingerprint byte budget")
                            content.update(b"F")
                            if read_content:
                                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                             dir_fd=directory_fd)
                                with os.fdopen(fd, "rb") as handle:
                                    if _identity(os.fstat(handle.fileno())) != _identity(before):
                                        raise RuntimeError("retained workspace changed during fingerprint")
                                    remaining = before.st_size
                                    while remaining:
                                        check_time()
                                        chunk = handle.read(min(CHUNK_BYTES, remaining))
                                        if not chunk:
                                            raise RuntimeError("retained workspace changed during fingerprint")
                                        remaining -= len(chunk)
                                        content.update(chunk)
                                    if (handle.read(1) or
                                            _identity(os.fstat(handle.fileno())) != _identity(before)):
                                        raise RuntimeError("retained workspace changed during fingerprint")
                        else:
                            raise RuntimeError("retained workspace contains an unsupported special file")
                        if total > max_bytes:
                            raise RuntimeError("retained workspace exceeds fingerprint byte budget")
                        if _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)) != _identity(before):
                            raise RuntimeError("retained workspace changed during fingerprint")
                        content.update(b"\0")
                return content.hexdigest(), metadata.hexdigest(), entries, total

            digest, metadata, entries, total = scan(True)
            _, after_metadata, after_entries, after_total = scan(False)
            check_time()
            if ((metadata, entries, total) != (after_metadata, after_entries, after_total)
                    or _identity(workspace.lstat()) != _identity(root_before)
                    or _identity(os.fstat(root_fd)) != _identity(root_before)):
                raise RuntimeError("retained workspace changed during fingerprint")
            return {"content_digest": "sha256:" + digest,
                    "entry_count": entries, "content_bytes": total}
        finally:
            os.close(root_fd)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("retained workspace fingerprint failed") from exc
