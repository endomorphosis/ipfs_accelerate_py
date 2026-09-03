"""Fail-closed exclusive Quack owner death recovery.

A leftover marker or ready-looking status is never treated as live.
Recovery is admitted only when process birth is dead or absent and the
listen port is closed. Unknown liveness and a bound port fail closed.
This module does not open DuckDB and does not start a server.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
)
from .quack_state_server import OwnerMarker, reclaim_stale_owner_marker

EXCLUSIVE_OWNER_RECOVERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/exclusive-owner-death-recovery@1"
)
ARCHIVE_SUFFIX_PREFIX = ".pre-relaunch-"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _port_open(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.3)
    try:
        return sock.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        sock.close()


def observe_exclusive_owner_liveness(
    *,
    marker_path: Path,
    listen_host: str = "127.0.0.1",
    listen_port: int | None = None,
    liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    port_probe: Callable[[str, int], bool] | None = None,
) -> dict[str, Any]:
    """Classify exclusive-owner liveness without treating a leftover marker as live."""

    probe = liveness or (lambda birth: owner_liveness(birth))
    payload = _read_json(Path(marker_path))
    marker: OwnerMarker | None = None
    birth_state = "absent"
    server_id = ""
    if payload is not None:
        try:
            marker = OwnerMarker.from_dict(payload)
        except (TypeError, ValueError, KeyError):
            birth_state = "unknown"
        else:
            server_id = marker.server_id
            observed = probe(marker.process_birth)
            if observed is OwnerLiveness.ALIVE:
                birth_state = "alive"
            elif observed is OwnerLiveness.UNKNOWN:
                birth_state = "unknown"
            else:
                birth_state = "dead"

    port_open: bool | None = None
    if listen_port is not None:
        checker = port_probe or _port_open
        port_open = bool(checker(str(listen_host), int(listen_port)))

    if birth_state == "alive" and port_open is False:
        # Process exists; a closed port is not proof it can be reclaimed.
        state = "unknown"
        reason = "owner_alive_listen_port_closed"
    elif birth_state == "alive":
        state = "alive"
        reason = "owner_alive"
    elif birth_state == "unknown":
        state = "unknown"
        reason = "owner_liveness_unknown"
    elif port_open is True:
        state = "unknown"
        reason = "listen_port_held"
    elif birth_state == "dead":
        state = "dead"
        reason = "owner_dead"
    else:
        state = "absent"
        reason = "owner_absent"

    return {
        "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
        "state": state,
        "reason": reason,
        "birth_state": birth_state,
        "port_open": port_open,
        "server_id": server_id,
        "marker_path": str(marker_path),
        "has_marker": marker is not None,
    }


def admit_dead_exclusive_owner_recovery(
    *,
    marker_path: Path,
    lock_path: Path,
    listen_host: str = "127.0.0.1",
    listen_port: int | None = None,
    live_runtime: Path | None = None,
    liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    port_probe: Callable[[str, int], bool] | None = None,
) -> dict[str, Any]:
    """Admit rematerialize/relaunch only after the exclusive owner is proved dead.

    Reclaims a stale marker. Does not start a server. Does not open DuckDB.
    """

    observation = observe_exclusive_owner_liveness(
        marker_path=marker_path,
        listen_host=listen_host,
        listen_port=listen_port,
        liveness=liveness,
        port_probe=port_probe,
    )
    if observation["state"] in {"alive", "unknown"}:
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "admitted": False,
            "reclaimed": False,
            "relaunch": False,
            "restore_from": "",
            "reason": observation["reason"],
            "observation": observation,
        }

    reclaim = reclaim_stale_owner_marker(
        marker_path=Path(marker_path),
        lock_path=Path(lock_path),
        liveness=liveness,
    )
    if reclaim.get("reason") in {"lock_held", "owner_alive", "owner_liveness_unknown"}:
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "admitted": False,
            "reclaimed": bool(reclaim.get("reclaimed")),
            "relaunch": False,
            "restore_from": "",
            "reason": str(reclaim.get("reason") or "reclaim_refused"),
            "observation": observation,
            "reclaim": reclaim,
        }

    restore_from = ""
    if live_runtime is not None and not Path(live_runtime).exists():
        restore_from = str(
            newest_archived_exclusive_owner_runtime(Path(live_runtime)) or ""
        )

    return {
        "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
        "admitted": True,
        "reclaimed": bool(reclaim.get("reclaimed")),
        "relaunch": True,
        "restore_from": restore_from,
        "reason": observation["reason"],
        "observation": observation,
        "reclaim": reclaim,
    }


def newest_archived_exclusive_owner_runtime(live_runtime: Path) -> Path | None:
    """Return the newest ``*.pre-relaunch-*`` sibling of ``live_runtime``."""

    parent = Path(live_runtime).parent
    name = Path(live_runtime).name
    prefix = name + ARCHIVE_SUFFIX_PREFIX
    archives = [
        path
        for path in parent.glob(prefix + "*")
        if path.is_dir() and not path.is_symlink()
    ]
    if not archives:
        return None
    return max(archives, key=lambda path: path.stat().st_mtime)


def restore_archived_exclusive_owner_runtime(
    live_runtime: Path,
    *,
    archive: Path | None = None,
) -> dict[str, Any]:
    """Move one archived runtime back to the live path.

    Refuses if the live path already exists. Does not open DuckDB.
    """

    destination = Path(live_runtime)
    if destination.exists():
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "restored": False,
            "reason": "live_runtime_present",
            "live_runtime": str(destination),
        }
    source = Path(archive) if archive is not None else newest_archived_exclusive_owner_runtime(
        destination
    )
    if source is None:
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "restored": False,
            "reason": "no_archive",
            "live_runtime": str(destination),
        }
    if not source.is_dir() or source.is_symlink():
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "restored": False,
            "reason": "archive_not_directory",
            "live_runtime": str(destination),
            "archive": str(source),
        }
    expected_parent = destination.parent.resolve()
    try:
        source_parent = source.resolve().parent
    except OSError:
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "restored": False,
            "reason": "archive_unresolvable",
            "live_runtime": str(destination),
            "archive": str(source),
        }
    if source_parent != expected_parent:
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "restored": False,
            "reason": "archive_outside_parent",
            "live_runtime": str(destination),
            "archive": str(source),
        }
    if ARCHIVE_SUFFIX_PREFIX not in source.name:
        return {
            "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
            "restored": False,
            "reason": "archive_name_invalid",
            "live_runtime": str(destination),
            "archive": str(source),
        }
    os.rename(source, destination)
    return {
        "schema": EXCLUSIVE_OWNER_RECOVERY_SCHEMA,
        "restored": True,
        "reason": "restored_archive",
        "live_runtime": str(destination),
        "archive": str(source),
    }


__all__ = (
    "ARCHIVE_SUFFIX_PREFIX",
    "EXCLUSIVE_OWNER_RECOVERY_SCHEMA",
    "admit_dead_exclusive_owner_recovery",
    "newest_archived_exclusive_owner_runtime",
    "observe_exclusive_owner_liveness",
    "restore_archived_exclusive_owner_runtime",
)
