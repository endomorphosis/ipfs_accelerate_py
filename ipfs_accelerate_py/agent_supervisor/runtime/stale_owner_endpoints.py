"""Retire inert typed endpoints under an exclusive stopped-owner namespace.

This repairs bootstrap files only. Published status supplies a conservative
dead-birth prerequisite, never task, database, or completion authority.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import stat
import time
from pathlib import Path
from typing import Any

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def reclaim_stale_typed_owner_endpoints(
    *,
    server: Any,
    owner_handle: Any,
    state_directory_fd: int,
) -> dict[str, Any]:
    """Require exact dead birth, secure inodes, and repeated listener refusal.

    The caller retains its anchored state directory and all native startup
    locks throughout this call. An absent marker alone proves nothing: an
    earlier interrupted cleanup may already have removed it. Keep the old
    status projection until normal startup so partial endpoint retirement is
    safely repeatable. Never read or log the bootstrap token.
    """
    from .quack_state_server import (
        QUACK_STATE_SERVER_SCHEMA,
        STATE_SERVER_IDENTITY_SCHEMA,
        QuackStateServerOwnershipError,
        _assert_owner_lock_name_matches_descriptor,
        _mutation_duplicate_guard,
        read_locked_owner_marker,
    )
    from ..task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
    )

    def reject(reason: str) -> None:
        raise QuackStateServerOwnershipError("stale typed endpoint recovery: " + reason)

    def locked_namespace() -> None:
        owner_handle.assert_canonical_parent()
        _assert_owner_lock_name_matches_descriptor(
            directory_descriptor=owner_handle.directory_fileno(),
            lock_name=server.owner_lock_path().name,
            lock_descriptor=owner_handle.fileno(),
        )
        if Path(server.config.database_path).parent != owner_handle.directory_path:
            reject("database scope differs")
        if (
            read_locked_owner_marker(owner_handle, server.owner_marker_path())
            is not None
        ):
            reject("owner marker remains")
        opened = os.fstat(state_directory_fd)
        named = os.lstat(server.config.state_dir)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid)
            != (named.st_dev, named.st_ino, named.st_mode, named.st_uid)
        ):
            reject("state directory changed or is unsafe")

    socket_path = server.typed_command_socket_path()
    paths = (
        socket_path,
        socket_path.parent / TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
        server.typed_command_token_path(),
    )
    state_dir = Path(server.config.state_dir).resolve(strict=True)
    if any(path.parent.resolve(strict=True) != state_dir for path in paths):
        reject("endpoint scope differs")
    if len({path.name for path in paths}) != 3:
        reject("endpoint names overlap")
    locked_namespace()

    def endpoint_observation() -> dict[str, tuple[int, ...]]:
        observed = {}
        for ordinal, path in enumerate(paths):
            try:
                info = os.stat(
                    path.name, dir_fd=state_directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                continue
            valid_type = (
                stat.S_ISSOCK(info.st_mode)
                if ordinal < 2
                else stat.S_ISREG(info.st_mode)
            )
            if (
                not valid_type
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or (ordinal == 2 and info.st_size != 64)
            ):
                reject("endpoint inode is unsafe")
            observed[path.name] = _identity(info)
        return observed

    observed = endpoint_observation()
    if not observed:
        return {"reclaimed": False, "reason": "endpoints_absent"}
    if (
        getattr(server.lifecycle, "value", "") not in {"created", "failed"}
        or server.identity is not None
    ):
        reject("owner lifecycle is not inert")
    status_path = server.status_path()
    if status_path.parent.resolve(strict=True) != state_dir:
        reject("status scope differs")
    status_fd = os.open(
        status_path.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=state_directory_fd,
    )
    try:
        info = os.fstat(status_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or not 0 < info.st_size <= 65536
        ):
            reject("previous status is unsafe")
        status_identity = _identity(info)
        raw = os.read(status_fd, 65537)
        if (
            len(raw) != info.st_size
            or _identity(os.fstat(status_fd)) != status_identity
        ):
            reject("previous status changed")
        payload = json.loads(raw, object_pairs_hook=_mutation_duplicate_guard)
    finally:
        os.close(status_fd)
    identity = payload.get("identity") or {}
    birth_data = identity.get("process_birth") or {}
    if (
        set(birth_data) != {"pid", "start_time_ticks", "boot_id", "parent_pid"}
        or any(
            type(birth_data.get(key)) is not int or birth_data[key] <= 0
            for key in ("pid", "start_time_ticks", "parent_pid")
        )
        or type(birth_data.get("boot_id")) is not str
        or not birth_data["boot_id"]
    ):
        reject("previous process birth is incomplete")
    birth = ProcessBirthIdentity.from_dict(birth_data)
    material = (
        f"{birth.pid}:{birth.start_time_ticks}:{birth.boot_id}:{birth.parent_pid}"
    )
    birth_id = "birth:" + hashlib.sha256(material.encode()).hexdigest()[:32]
    if (
        payload.get("schema") != QUACK_STATE_SERVER_SCHEMA
        or payload.get("database_path") != str(server.config.database_path)
        or payload.get("state_dir") != str(server.config.state_dir)
        or payload.get("store_id") != server.config.store_id
        or identity.get("schema") != STATE_SERVER_IDENTITY_SCHEMA
        or identity.get("store_id") != server.config.store_id
        or identity.get("process_birth_id") != birth_id
        or not identity.get("server_id")
        or not identity.get("database_uuid")
        or type(identity.get("generation")) is not int
        or identity["generation"] < 1
        or server.config.host != "127.0.0.1"
        or not 0 < server.config.port < 65536
        or identity.get("listen_uri") != f"quack:127.0.0.1:{server.config.port}"
    ):
        reject("previous status binding differs")

    def recheck(expected: dict[str, tuple[int, ...]]) -> None:
        locked_namespace()
        if owner_liveness(birth) is not OwnerLiveness.DEAD:
            reject("previous owner is live or unknown")
        if (
            _identity(
                os.stat(
                    status_path.name, dir_fd=state_directory_fd, follow_symlinks=False
                )
            )
            != status_identity
        ):
            reject("previous status changed")
        if endpoint_observation() != expected:
            reject("endpoint set changed")

    recheck(observed)
    # Use the retained directory descriptor to avoid long Unix socket paths or
    # a replaceable /proc/self/cwd alias. Refuse a live or indeterminate peer.
    for sample in range(2):
        targets = [(socket.AF_INET, (server.config.host, server.config.port))]
        targets.extend(
            (socket.AF_UNIX, f"/proc/self/fd/{state_directory_fd}/{p.name}")
            for p in paths[:2]
            if p.name in observed
        )
        for family, address in targets:
            with socket.socket(family, socket.SOCK_STREAM) as channel:
                channel.settimeout(0.25)
                if channel.connect_ex(address) != errno.ECONNREFUSED:
                    reject("listener is live or unknown")
        recheck(observed)
        if sample == 0:
            time.sleep(0.05)
    reclaimed = []
    for path in paths:
        if path.name in observed:
            recheck(observed)
            os.unlink(path.name, dir_fd=state_directory_fd)
            del observed[path.name]
            reclaimed.append(path.name)
    os.fsync(state_directory_fd)
    recheck(observed)
    return {
        "reclaimed": True,
        "reason": "dead_owner_inert_endpoints",
        "paths": reclaimed,
        "server_id": identity["server_id"],
        "generation": identity["generation"],
        "process_birth": birth.to_dict(),
        "listener_absence_samples": 2,
        "task_completion_authority": False,
    }
