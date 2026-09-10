"""Short-lived derived sessions following the managed fleet owner deployment."""

from __future__ import annotations

import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable

from ..task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_FILENAME, compact_default_owner_socket_path,
)
from .quack_fleet_topology import SCHEMA, attach_typed_instance


def _read_regular(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("fleet client configuration must be a regular file")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("fleet client configuration exceeds its bound")
        return raw
    finally:
        os.close(descriptor)


def fleet_connection_factory(deployment_path: Path, *, repository_id: str,
                             client_id: str, timeout_seconds: float) -> Callable[[], Any]:
    path = Path(deployment_path).expanduser()
    if not path.is_absolute():
        raise ValueError("fleet deployment path must be absolute")
    for name, value in (("repository_id", repository_id), ("client_id", client_id)):
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError(f"invalid derived fleet {name}")
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 30):
        raise ValueError("derived fleet timeout must be between zero and 30 seconds")

    def connect():
        # Re-read both files after every idle interval or owner restart. Keep
        # source/semantic authority out of this disposable cache connection.
        deployment = json.loads(_read_regular(path, 8 * 1024 * 1024))
        if not isinstance(deployment, dict) or deployment.get("schema") != SCHEMA:
            raise ValueError("compiled fleet deployment required")
        instance = deployment["instances"]["derived_coordination"]
        if instance.get("managed_by_fleet") is not True:
            raise ValueError("derived coordination must use its managed owner")
        state, database = Path(instance["state_dir"]), Path(instance["database_path"])
        if not state.is_absolute() or not database.is_absolute():
            raise ValueError("derived owner paths must be absolute")
        token = _read_regular(state / "derived-coordination.token", 4096).decode().strip()
        socket = compact_default_owner_socket_path(state / TYPED_STATE_OWNER_SOCKET_FILENAME,
                                                  identity=database)
        return attach_typed_instance(deployment, "derived_coordination", socket_path=socket,
            token=token, client_id=client_id,
            process_birth_id=f"birth:derived-fleet:{os.getpid()}:{time.time_ns()}",
            derived_repository_id=repository_id, timeout_seconds=timeout_seconds)

    return connect
