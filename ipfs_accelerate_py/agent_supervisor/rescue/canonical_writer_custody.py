"""Bounded metadata-only observation of canonical DuckDB writer custody."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_MAX_KERNEL_LOCK_BYTES = 1024 * 1024


def valid_birth(birth: Any) -> bool:
    return (
        isinstance(birth, Mapping)
        and type(birth.get("pid")) is int
        and birth["pid"] > 1
        and type(birth.get("start_time_ticks")) is int
        and birth["start_time_ticks"] > 0
        and isinstance(birth.get("boot_id"), str)
        and bool(birth["boot_id"])
    )


def read_kernel_locks(*, maximum_bytes: int = _MAX_KERNEL_LOCK_BYTES) -> str:
    """Read a bounded kernel snapshot without opening the canonical store."""
    descriptor = os.open(
        "/proc/locks", os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        chunks = []
        size = 0
        while size <= maximum_bytes:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - size))
            if not chunk:
                return b"".join(chunks).decode("ascii")
            chunks.append(chunk)
            size += len(chunk)
        raise ValueError("kernel lock snapshot exceeds bound")
    finally:
        os.close(descriptor)


def observe_canonical_writer_lock(
    database: Path,
    birth: Mapping[str, Any],
    *,
    process_identity: Callable[[int], Mapping[str, Any]],
    birth_matches: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
    kernel_locks: Callable[[], str] = read_kernel_locks,
    namespaces: Callable[[int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Observe exact live PID/birth/inode custody; uncertainty never requests restart."""
    unknown = {
        "verified": False,
        "reason": "canonical_writer_lock_observation_unavailable",
    }
    try:
        if (
            not valid_birth(birth)
            or not database.is_absolute()
            or database.resolve(strict=True) != database
        ):
            return unknown
        if not birth_matches(process_identity(birth["pid"]), birth):
            return {
                "verified": False,
                "reason": "native_identity_changed_during_writer_lock_probe",
            }
        namespace_before = namespaces(birth["pid"]) if namespaces else None
        before = database.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
            return unknown
        expected = (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino)
        held = False
        for line in kernel_locks().splitlines():
            fields = line.split()
            if len(fields) > 1 and fields[1] == "->":
                fields.pop(1)
                # Blocked requests are not held locks, but still validate the
                # complete snapshot before reporting an absent writer lock.
                blocked = True
            else:
                blocked = False
            if len(fields) != 8 or not fields[0].endswith(":"):
                return unknown
            major, minor, inode = fields[5].split(":")
            lock_inode = (int(major, 16), int(minor, 16), int(inode))
            pid = int(fields[4])
            if (
                not blocked
                and fields[1:4] == ["POSIX", "ADVISORY", "WRITE"]
                and pid == birth["pid"]
                and lock_inode == expected
                and fields[6:] == ["0", "EOF"]
            ):
                held = True
        namespace_after = namespaces(birth["pid"]) if namespaces else None
        after = database.lstat()
        if (before.st_dev, before.st_ino, before.st_mode, before.st_uid) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
        ) or database.resolve(strict=True) != database:
            return {
                "verified": False,
                "reason": "canonical_database_changed_during_writer_lock_probe",
            }
        if not birth_matches(process_identity(birth["pid"]), birth):
            return {
                "verified": False,
                "reason": "native_identity_changed_during_writer_lock_probe",
            }
        if namespaces is not None:
            namespace_verified = bool(
                namespace_before == namespace_after
                and isinstance(namespace_after, Mapping)
                and namespace_after.get("verified") is True
            )
            if not held and not namespace_verified:
                return {
                    "verified": False,
                    "reason": "canonical_writer_lock_namespace_unverified",
                    "namespace_verified": False,
                }
            return {
                "verified": True,
                "held": held,
                "namespace_verified": namespace_verified,
            }
        return {"verified": True, "held": held}
    except (OSError, ValueError, TypeError, KeyError):
        return unknown


def observe_owner_namespaces(
    pid: int, *, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    """Namespace-inaccessible absence is unknown; this creates no signal authority."""
    if type(pid) is not int or pid <= 1:
        return {"verified": False}
    try:
        identities = {}
        for kind in ("pid", "mnt"):
            owner = os.readlink(proc_root / str(pid) / "ns" / kind)
            current = os.readlink(proc_root / "self" / "ns" / kind)
            prefix = f"{kind}:["
            if any(
                not value.startswith(prefix)
                or not value.endswith("]")
                or not value[len(prefix) : -1].isdigit()
                for value in (owner, current)
            ):
                return {"verified": False}
            if owner != current:
                return {"verified": False}
            identities[kind] = owner
        return {"verified": True, "identities": identities}
    except OSError:
        return {"verified": False}
