#!/usr/bin/env python3
"""Copy a checkpointed SAWM replica without closing a writer-process DB fd.

POSIX locks belong to the process and closing any descriptor for that inode
releases them. Only this credential-free child opens/closes the canonical file.
The parent must already have serialized mutations and checkpointed the writer.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

MAX_PACKET = 8192
MAX_DATABASE = 8 * 1024**3
COPY_TIMEOUT_SECONDS = 60


class ReplicaCopyUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("bounded native replica copy unavailable")


def _identity(observed):
    if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid():
        raise ReplicaCopyUnavailable()
    return [observed.st_dev, observed.st_ino, observed.st_size,
            observed.st_mtime_ns, observed.st_ctime_ns, observed.st_uid]


def _copy(request):
    if (not isinstance(request, dict)
            or set(request) != {"source", "target", "source_identity", "nonce"}
            or not isinstance(request["nonce"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", request["nonce"])
            or not isinstance(request["source_identity"], list)
            or len(request["source_identity"]) != 6
            or any(type(value) is not int or value < 0 for value in request["source_identity"])):
        raise ReplicaCopyUnavailable()
    source, target = Path(request["source"]), Path(request["target"])
    if (not source.is_absolute() or not target.is_absolute() or source == target
            or source.parent != target.parent
            or source.resolve() != source or target.resolve() != target):
        raise ReplicaCopyUnavailable()
    before = request["source_identity"]
    if _identity(source.lstat()) != before or not 0 < before[2] <= MAX_DATABASE:
        raise ReplicaCopyUnavailable()
    temporary = target.with_name(f".{target.name}.{request['nonce']}.tmp")
    source_fd = target_fd = -1
    temporary_identity = None
    size = 0
    digest = hashlib.sha256()
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        if _identity(os.fstat(source_fd)) != before:
            raise ReplicaCopyUnavailable()
        target_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        opened = os.fstat(target_fd)
        temporary_identity = (opened.st_dev, opened.st_ino)
        while size < before[2]:
            chunk = os.read(source_fd, min(1024 * 1024, before[2] - size))
            if not chunk:
                raise ReplicaCopyUnavailable()
            digest.update(chunk)
            size += len(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(target_fd, remaining)
                if written <= 0:
                    raise ReplicaCopyUnavailable()
                remaining = remaining[written:]
        if (os.read(source_fd, 1) or _identity(os.fstat(source_fd)) != before
                or _identity(source.lstat()) != before):
            raise ReplicaCopyUnavailable()
        os.fsync(target_fd)
        os.close(target_fd)
        target_fd = -1
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {"nonce": request["nonce"], "sha256": digest.hexdigest(), "size_bytes": size,
                "source_identity": before, "target_identity": _identity(target.lstat())}
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if target_fd >= 0:
            os.close(target_fd)
        if temporary_identity is not None:
            try:
                observed = temporary.lstat()
                if (observed.st_dev, observed.st_ino) == temporary_identity:
                    temporary.unlink()
            except OSError:
                pass


def copy_replica(source: Path, target: Path):
    """Parent half: stat only; never acquire a canonical database descriptor."""
    source, target = Path(source).absolute(), Path(target).absolute()
    if (source.resolve() != source or target.resolve() != target or source == target
            or source.parent != target.parent):
        raise ReplicaCopyUnavailable()
    before = _identity(source.lstat())
    request = {"source": str(source), "target": str(target),
               "source_identity": before, "nonce": uuid.uuid4().hex}
    packet = json.dumps(request, sort_keys=True).encode()
    if len(packet) > MAX_PACKET:
        raise ReplicaCopyUnavailable()
    try:
        with tempfile.TemporaryFile() as output:
            # -I -S and the minimal environment exclude provider credentials,
            # PYTHONPATH/startup hooks and package imports from this file copier.
            child = subprocess.Popen([sys.executable, "-I", "-S", str(Path(__file__).resolve())],
                stdin=subprocess.PIPE, stdout=output, stderr=subprocess.DEVNULL,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"}, close_fds=True)
            try:
                child.communicate(packet, timeout=COPY_TIMEOUT_SECONDS)
            except BaseException:
                child.kill()
                child.wait()
                raise
            output.seek(0)
            raw = output.read(MAX_PACKET + 1)
        if child.returncode or len(raw) > MAX_PACKET:
            raise ReplicaCopyUnavailable()
        receipt = json.loads(raw)
        if (not isinstance(receipt, dict)
                or set(receipt) != {"nonce", "sha256", "size_bytes", "source_identity", "target_identity"}
                or receipt["nonce"] != request["nonce"]
                or receipt["source_identity"] != before or _identity(source.lstat()) != before
                or receipt["target_identity"] != _identity(target.lstat())
                or type(receipt["size_bytes"]) is not int or receipt["size_bytes"] != before[2]
                or not isinstance(receipt["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])):
            raise ReplicaCopyUnavailable()
        return {"authority": "non_authoritative_read_replica", "path": str(target),
                "source_database_path": str(source), "sha256": receipt["sha256"],
                "size_bytes": receipt["size_bytes"]}
    except Exception:
        raise ReplicaCopyUnavailable() from None


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_PACKET + 1)
        if len(raw) > MAX_PACKET:
            raise ReplicaCopyUnavailable()
        result = _copy(json.loads(raw))
        sys.stdout.write(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
