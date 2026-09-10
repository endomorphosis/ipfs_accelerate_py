"""Probe the dedicated Quack gateways and restart only repeatedly failed owners."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..rescue.live_board_probe import birth_matches, process_identity
from ..task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_FILENAME,
    compact_default_owner_socket_path,
)
from .quack_fleet_topology import ROLES, attach_typed_instance

SERVICES = {role: f"ipfs-quack-fleet-{role.replace('_', '-')}.service" for role in ROLES}


def probe_owner(deployment: Mapping[str, Any], role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError("unknown dedicated owner role")
    instance = deployment["instances"][role]
    state = Path(instance["state_dir"])
    client = None
    try:
        owner = json.loads((state / "quack-state-server.status.json").read_text())
        identity = owner.get("identity", {})
        birth = identity.get("process_birth", {})
        if owner.get("lifecycle") != "ready" or not birth_matches(process_identity(birth.get("pid")), birth):
            return {"healthy": False, "reason": "native_owner_not_ready_or_alive"}
        aggregate = role == "aggregate_control"
        token = state / ("fleet-observation-read.token" if aggregate else "derived-coordination.token")
        socket = compact_default_owner_socket_path(state / TYPED_STATE_OWNER_SOCKET_FILENAME, identity=instance["database_path"])
        client = attach_typed_instance(deployment, role, socket_path=socket, token=token.read_text().strip(),
                    client_id=f"fleet:health:{role}", process_birth_id=f"birth:fleet-health:{os.getpid()}:{time.time_ns()}",
                    derived_repository_id="fleet:health" if not aggregate else "", fleet_observation_read=aggregate,
                    timeout_seconds=5)
        generation = client.load_generation()
        if not aggregate:
            client.derived_coordination({"operation": "list_references",
                                         "repository_id": "fleet:health", "tree_id": "fleet:health"})
        after = json.loads((state / "quack-state-server.status.json").read_text())
        if (after.get("lifecycle") != "ready" or after.get("identity", {}).get("process_birth_id") != identity.get("process_birth_id")
            or not birth_matches(process_identity(birth.get("pid")), birth)
            or client.session.server_id != identity.get("server_id") or generation.generation != identity.get("generation")
            or generation.database_uuid != identity.get("database_uuid")):
            return {"healthy": False, "reason": "native_identity_changed_during_health_query"}
        return {"healthy": True, "reason": "authenticated_typed_generation_read",
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
    restart = failures >= 2 and len(restarts) < 3 and (not restarts or now - max(restarts) >= 300)
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
            # The role->unit map is closed; inventory data cannot select a
            # taskboard owner, arbitrary service, shell command or process.
            completed = subprocess.run(["systemctl", "--user", "restart", SERVICES[role]], timeout=150, check=False, capture_output=True)
            decision["restart_times"].append(time.time())
            decision["restart_returncode"] = completed.returncode
        result[role] = decision
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, state_path)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
