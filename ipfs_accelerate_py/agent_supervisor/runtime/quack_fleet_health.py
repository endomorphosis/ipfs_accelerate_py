"""Probe the dedicated Quack gateways and restart only repeatedly failed owners."""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..rescue.live_board_probe import birth_matches, process_identity
from ..task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_FILENAME,
    TypedStateOwnerRemoteError,
    compact_default_owner_socket_path,
)
from .quack_fleet_topology import ROLES, attach_typed_instance

SERVICES = {role: f"ipfs-quack-fleet-{role.replace('_', '-')}.service" for role in ROLES}
_MAX_KERNEL_LOCK_BYTES = 1024 * 1024


def _valid_birth(birth: Any) -> bool:
    return (isinstance(birth, Mapping) and type(birth.get("pid")) is int and birth["pid"] > 1
            and type(birth.get("start_time_ticks")) is int and birth["start_time_ticks"] > 0
            and isinstance(birth.get("boot_id"), str) and bool(birth["boot_id"]))


def _read_owner_status(state: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads((state / "quack-state-server.status.json").read_text())
        return value if isinstance(value, Mapping) else None
    except (OSError, ValueError):
        return None


def _read_kernel_locks() -> str:
    """Read a bounded kernel snapshot without opening the canonical store."""
    descriptor = os.open("/proc/locks", os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        chunks = []
        size = 0
        while size <= _MAX_KERNEL_LOCK_BYTES:
            chunk = os.read(descriptor, min(65536, _MAX_KERNEL_LOCK_BYTES + 1 - size))
            if not chunk:
                return b"".join(chunks).decode("ascii")
            chunks.append(chunk)
            size += len(chunk)
        raise ValueError("kernel lock snapshot exceeds bound")
    finally:
        os.close(descriptor)


def _canonical_writer_lock(database: Path, birth: Mapping[str, Any]) -> dict[str, Any]:
    """Observe exact live PID/birth/inode custody; uncertainty never requests restart."""
    unknown = {"verified": False, "reason": "canonical_writer_lock_observation_unavailable"}
    try:
        if (not _valid_birth(birth)
                or not database.is_absolute() or database.resolve(strict=True) != database):
            return unknown
        if not birth_matches(process_identity(birth["pid"]), birth):
            return {"verified": False, "reason": "native_identity_changed_during_writer_lock_probe"}
        before = database.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
            return unknown
        expected = (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino)
        held = False
        for line in _read_kernel_locks().splitlines():
            fields = line.split()
            if len(fields) > 1 and fields[1] == "->":
                fields.pop(1)
                # Blocked requests are not held locks, but still validate the
                # complete snapshot before treating absence as restart evidence.
                blocked = True
            else:
                blocked = False
            if len(fields) != 8 or not fields[0].endswith(":"):
                return unknown
            major, minor, inode = fields[5].split(":")
            lock_inode = (int(major, 16), int(minor, 16), int(inode))
            pid = int(fields[4])
            if (not blocked and fields[1:4] == ["POSIX", "ADVISORY", "WRITE"]
                    and pid == birth["pid"] and lock_inode == expected and fields[6:] == ["0", "EOF"]):
                held = True
        after = database.lstat()
        if ((before.st_dev, before.st_ino, before.st_mode, before.st_uid)
                != (after.st_dev, after.st_ino, after.st_mode, after.st_uid)
                or database.resolve(strict=True) != database):
            return {"verified": False, "reason": "canonical_database_changed_during_writer_lock_probe"}
        if not birth_matches(process_identity(birth["pid"]), birth):
            return {"verified": False, "reason": "native_identity_changed_during_writer_lock_probe"}
        return {"verified": True, "held": held}
    except (OSError, ValueError, TypeError, KeyError):
        return unknown


def probe_owner(deployment: Mapping[str, Any], role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError("unknown dedicated owner role")
    instance = deployment["instances"][role]
    state = Path(instance["state_dir"])
    client = None
    try:
        owner = _read_owner_status(state)
        if owner is None:
            return {"healthy": False, "reason": "native_owner_status_unverified", "restartable": False}
        identity = owner.get("identity", {})
        birth = identity.get("process_birth", {}) if isinstance(identity, Mapping) else None
        if owner.get("lifecycle") != "ready":
            return {"healthy": False, "reason": "native_owner_not_ready_or_alive"}
        if not _valid_birth(birth):
            return {"healthy": False, "reason": "native_owner_birth_unverified", "restartable": False}
        actual_birth = process_identity(birth["pid"])
        if not birth_matches(actual_birth, birth):
            if not actual_birth:
                try:
                    os.stat(f"/proc/{birth['pid']}")
                except FileNotFoundError:
                    return {"healthy": False, "reason": "native_owner_not_ready_or_alive"}
                except OSError:
                    pass
            return {"healthy": False, "reason": "native_owner_birth_unverified", "restartable": False}
        aggregate = role == "aggregate_control"
        token = state / ("fleet-observation-read.token" if aggregate else "derived-coordination.token")
        socket = compact_default_owner_socket_path(state / TYPED_STATE_OWNER_SOCKET_FILENAME, identity=instance["database_path"])
        client = attach_typed_instance(deployment, role, socket_path=socket, token=token.read_text().strip(),
                    client_id=f"fleet:health:{role}", process_birth_id=f"birth:fleet-health:{os.getpid()}:{time.time_ns()}",
                    derived_repository_id="fleet:health" if not aggregate else "", fleet_observation_read=aggregate,
                    timeout_seconds=5)
        generation = client.load_generation()
        verified_kinds = []
        if not aggregate:
            client.derived_coordination({"operation": "list_references",
                                         "repository_id": "fleet:health", "tree_id": "fleet:health"})
            expected = deployment.get("derived_coordination", {})
            if expected.get("artifact_kinds"):
                try:
                    capabilities = client.derived_coordination({"operation": "capabilities",
                        "repository_id": "fleet:health"})["result"]
                except TypedStateOwnerRemoteError as error:
                    if error.error_code == "operation_failed" and error.error_type == "ValueError":
                        return {"healthy": False, "reason": "derived_capability_mismatch", "restartable": False}
                    raise
                required = set(expected["artifact_kinds"])
                if (capabilities.get("artifact_schema") != expected.get("artifact_schema")
                    or not required.issubset(capabilities.get("artifact_kinds", []))
                    or not {"record_artifact", "lookup_artifact", "list_artifacts"}.issubset(capabilities.get("operations", []))):
                    return {"healthy": False, "reason": "derived_capability_mismatch", "restartable": False}
                verified_kinds = sorted(required)
        after = _read_owner_status(state)
        if after is None or not isinstance(after.get("identity"), Mapping):
            return {"healthy": False, "reason": "native_owner_status_unverified", "restartable": False}
        if (after.get("lifecycle") != "ready" or after["identity"].get("process_birth_id") != identity.get("process_birth_id")
            or not birth_matches(process_identity(birth.get("pid")), birth)
            or client.session.server_id != identity.get("server_id") or generation.generation != identity.get("generation")
            or generation.database_uuid != identity.get("database_uuid")):
            return {"healthy": False, "reason": "native_identity_changed_during_health_query", "restartable": False}
        writer_lock = _canonical_writer_lock(Path(instance["database_path"]), birth)
        if not writer_lock["verified"]:
            return {"healthy": False, "reason": writer_lock["reason"], "restartable": False}
        if not writer_lock["held"]:
            return {"healthy": False, "reason": "canonical_writer_lock_missing", "restartable": True}
        return {"healthy": True, "reason": "authenticated_typed_generation_read",
                "canonical_writer_lock_verified": True,
                "verified_artifact_kinds": verified_kinds,
                "derived_service_verified": not aggregate, "generation": generation.generation,
                "database_uuid": generation.database_uuid, "process_birth_id": identity["process_birth_id"]}
    except Exception as error:  # noqa: BLE001 - one failed native probe becomes bounded recovery evidence
        return {"healthy": False, "reason": f"typed_gateway_unavailable:{type(error).__name__}"}
    finally:
        if client is not None:
            client.close()


def recovery_decision(previous: Mapping[str, Any], probe: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    failures = 0 if probe.get("healthy") is True else int(previous.get("consecutive_failures", 0)) + 1
    restarts = [stamp for stamp in previous.get("restart_times", []) if 0 <= now - stamp <= 3600]
    restart = (probe.get("restartable") is not False and failures >= 2
               and len(restarts) < 3 and (not restarts or now - max(restarts) >= 300))
    return {"consecutive_failures": failures, "restart_times": restarts, "restart_requested": restart, "probe": dict(probe), "observed_at": now}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    deployment = json.loads(args.deployment.read_text())
    root = args.deployment.parent
    state_path = root / "gateway-health.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    holds = [str(root / name) for name in ("HOLD", "OPERATOR_STOP") if os.path.lexists(root / name)]
    if holds:
        print(json.dumps({"held": True, "hold_files": holds}))
        return 0
    result = {}
    for role in ROLES:
        decision = recovery_decision(previous.get(role, {}), probe_owner(deployment, role), now=time.time())
        if args.apply and decision["restart_requested"]:
            # A hold can arrive while a bounded native query is in flight.
            # Recheck at every service-effect boundary, including the second role.
            holds = [str(root / name) for name in ("HOLD", "OPERATOR_STOP") if os.path.lexists(root / name)]
            if holds:
                decision.update(restart_requested=False, restart_deferred="operator_hold", hold_files=holds)
                result[role] = decision
                continue
            # The role->unit map is closed; inventory data cannot select a
            # taskboard owner, arbitrary service, shell command or process.
            # An unknown timeout still consumes this bounded control attempt;
            # it never proves that systemd did not begin stopping the owner.
            decision["restart_times"].append(time.time())
            try:
                completed = subprocess.run(["systemctl", "--user", "restart", SERVICES[role]], timeout=150, check=False, capture_output=True)
                decision["restart_returncode"] = completed.returncode
                decision["restart_outcome"] = "completed"
            except subprocess.TimeoutExpired:
                decision.update(restart_returncode=None, restart_outcome="timeout_unknown")
            except OSError as error:
                decision.update(restart_returncode=None, restart_outcome="launch_failed", restart_error_type=type(error).__name__)
        result[role] = decision
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, state_path)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
