"""Metadata-validated file observations shared through the existing Quack owner.

Routine observations recheck metadata on every use and never renew their TTL
on a hit. They are derived cache data, not proof of current bytes. Strict
callers always read bytes. Missing configuration uses bounded local hashing;
a configured but unavailable owner is an error, not a hash-storm fallback.
"""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipfs_accelerate_py._hash_resources import hashing_worker_slot

from ..task_sources.hash_observations import (
    DEFAULT_TTL_SECONDS,
    IDENTITY_PROFILE,
    IDENTITY_SCHEMA,
    MAX_TTL_SECONDS,
    identity_key,
)

_connections: dict[tuple[object, ...], Any] = {}
_connections_lock = threading.Lock()


class SharedHashError(RuntimeError):
    """Shared observation failed or the file changed during observation."""


@dataclass(frozen=True)
class FileHashObservation:
    sha256: str
    cache_hit: bool
    freshness: str
    bytes_read: int


def _ttl(value: float | None) -> float:
    selected = float(os.environ.get("IPFS_HASH_CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS)
                     if value is None else value)
    if not math.isfinite(selected) or not (selected == 0 or 0.001 <= selected <= MAX_TTL_SECONDS):
        raise ValueError("hash observation TTL must be 0 or between 0.001 and 604800 seconds")
    return selected


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(metadata, name)) for name in (
        "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
        "st_size", "st_mtime_ns", "st_ctime_ns",
    ))


def _file_identity(descriptor: int, metadata: os.stat_result) -> dict[str, Any]:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    fields = dict(line.split(":", 1) for line in
                  Path(f"/proc/self/fdinfo/{descriptor}").read_text(encoding="ascii").splitlines()
                  if ":" in line)
    mount = fields["mnt_id"].strip()
    return {
        "schema": IDENTITY_SCHEMA, "algorithm": "sha256", "profile": IDENTITY_PROFILE,
        "host_boot_id": boot, "mount_id": mount,
        **{name: int(getattr(metadata, "st_" + name)) for name in (
            "dev", "ino", "mode", "uid", "gid", "nlink", "size", "mtime_ns", "ctime_ns",
        )},
    }


def _close_connections() -> None:
    with _connections_lock:
        entries = list(_connections.values())
        _connections.clear()
    for connection in entries:
        try:
            connection.close()
        except Exception:
            pass


def _after_fork() -> None:
    global _connections_lock
    _connections_lock = threading.Lock()
    # Close the child's descriptor copies without sending a protocol close on
    # the parent's still-live session or acquiring an inherited request mutex.
    for connection in _connections.values():
        try:
            connection._socket.close()
        except OSError:
            pass
    _connections.clear()


atexit.register(_close_connections)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class _DefaultHashConnection:
    def hash_observation(self, request: dict[str, Any]) -> dict[str, Any]:
        # Serialize only tiny owner requests, not byte hashing. This also makes
        # credential renewal safe when several local workers share one session.
        with _connections_lock:
            connection = _default_hash_connection_unlocked()
            if connection is None:
                raise SharedHashError("shared owner configuration disappeared")
            try:
                return connection.hash_observation(request)
            except Exception:
                # Transport loss can hide an already committed claim/complete.
                # Evict the broken session, but never replay a mutation here.
                # A subsequent call may reconnect; owner leases fence any
                # uncertain producer outcome without an independent hash storm.
                for key, cached in tuple(_connections.items()):
                    if cached is connection:
                        del _connections[key]
                connection._closed = True
                connection._socket.close()
                raise


def default_hash_connection() -> Any | None:
    configured = [bool(os.environ.get(name, "").strip()) for name in (
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
    )]
    # A store ID alone is also used by non-authority/read-only clients. Only
    # broker handoff fields opt a process into shared hashing authority.
    if not any(configured[1:]):
        return None
    if not all(configured):
        raise SharedHashError("shared hash owner handoff is incomplete")
    return _DefaultHashConnection()


def _default_hash_connection_unlocked() -> Any | None:
    """Use the attached owner's existing sealed grant-broker handoff.

    Supervisors must attach to the same owner/store to share observations.
    A caller may instead supply an already authenticated common-owner client.
    No token is written to the environment, argv, or a file.
    """
    from ..task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
        TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
        TypedStateOwnerConnection,
        kernel_process_birth_id,
        request_hash_observation_credential,
        typed_owner_socket_path,
    )
    store = os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "").strip()
    broker = os.environ.get(TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV, "").strip()
    secret = os.environ.get(TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV, "").strip()
    if not store or not broker or not secret:
        return None
    socket_path = typed_owner_socket_path(store)
    key = (os.getpid(), store, str(socket_path), broker, secret)
    cached = _connections.get(key)
    if cached is not None:
        if int(cached.grant.get("expires_at", 0)) > int(time.time() * 1000) + 5000:
            return cached
        _connections.pop(key).close()
    client_id = f"hash-observer:{os.getpid()}"
    birth = kernel_process_birth_id()
    token = request_hash_observation_credential(
        store_id=store, client_id=client_id, process_birth_id=birth,
    )
    if not token:
        raise SharedHashError("configured hash owner did not issue a credential")
    connection = TypedStateOwnerConnection(
        socket_path=socket_path, token=token, store_id=store,
        client_id=client_id, process_birth_id=birth, timeout_seconds=5,
    )
    if len(_connections) >= 32:
        old_key = next(iter(_connections))
        _connections.pop(old_key).close()
    _connections[key] = connection
    return connection


def hash_descriptor(
    descriptor: int, *, connection: Any | None = None, ttl_seconds: float | None = None,
    strict: bool = False, timeout_seconds: float = 60.0, max_bytes: int | None = None,
) -> FileHashObservation:
    """Hash one held regular file, coalescing routine misses through its owner."""
    ttl = _ttl(ttl_seconds)
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("hash timeout must be finite and positive")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("maximum hash bytes must be a nonnegative integer")
    deadline = time.monotonic() + timeout
    retained = os.dup(descriptor)
    try:
        before = os.fstat(retained)
        if not stat.S_ISREG(before.st_mode):
            raise SharedHashError("only regular files have shared hash observations")
        if max_bytes is not None and before.st_size > max_bytes:
            raise SharedHashError("file exceeds the caller's hash byte limit")
        witness = _stat_identity(before)

        def unchanged() -> None:
            if (_stat_identity(os.fstat(retained)) != witness
                    or _stat_identity(os.fstat(descriptor)) != witness):
                raise SharedHashError("file changed during hash observation")

        def measure() -> str:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("hash observation deadline expired")
            with hashing_worker_slot(timeout=remaining):
                unchanged()
                digest = hashlib.sha256()
                offset = 0
                while offset < before.st_size:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("hash observation deadline expired")
                    chunk = os.pread(retained, min(1024 * 1024, before.st_size - offset), offset)
                    if not chunk:
                        raise SharedHashError("file was truncated during hashing")
                    digest.update(chunk)
                    offset += len(chunk)
                unchanged()
                return digest.hexdigest()

        if strict or ttl == 0:
            return FileHashObservation(measure(), False, "fresh-bytes", before.st_size)
        owner = connection if connection is not None else default_hash_connection()
        if owner is None:
            return FileHashObservation(measure(), False, "fresh-bytes", before.st_size)
        identity = _file_identity(retained, before)
        key = identity_key(identity)
        base = {"key": key, "identity": identity}
        while True:
            unchanged()
            if time.monotonic() >= deadline:
                raise TimeoutError("shared hash owner remained busy until deadline")
            response = owner.hash_observation({
                "action": "claim", **base, "ttl_ms": max(1, int(ttl * 1000)),
                # The canonical owner wire format intentionally forbids floats.
                "lease_ms": max(1, int(min(300.0, timeout) * 1000)),
            })
            status = response.get("status")
            if status == "hit":
                digest = response.get("sha256")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise SharedHashError("owner returned an invalid SHA-256 observation")
                unchanged()
                return FileHashObservation(digest, True, "metadata-and-ttl", 0)
            if status == "busy":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("shared hash observation deadline expired")
                time.sleep(min(0.05, remaining))
                continue
            if status != "claimed":
                raise SharedHashError("owner returned an invalid claim response")
            claim = {**base, **{name: response[name] for name in ("generation", "lease_token", "fence")}}
            try:
                digest = measure()
                unchanged()
                completed = owner.hash_observation({"action": "complete", **claim, "sha256": digest})
                if completed.get("status") not in {"completed", "hit"}:
                    raise SharedHashError("owner did not accept the hash completion")
                unchanged()
                return FileHashObservation(digest, False, "fresh-bytes", before.st_size)
            except BaseException:
                try:
                    owner.hash_observation({"action": "abort", **claim})
                except Exception:
                    pass
                raise
    finally:
        os.close(retained)


def hash_file(path: Path, **kwargs: Any) -> FileHashObservation:
    """Open a file without following its final symlink and check replacement."""
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = _stat_identity(os.fstat(descriptor))
        result = hash_descriptor(descriptor, **kwargs)
        if _stat_identity(os.stat(path, follow_symlinks=False)) != before:
            raise SharedHashError("file path was replaced during hash observation")
        return result
    finally:
        os.close(descriptor)
