"""Real-process helper for the CASF executor bootstrap qualification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    provider_subprocess_environment,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientError,
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
    request_state_owner_bootstrap,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-fd", type=int, required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--claim", action="store_true")
    parser.add_argument("--task-cid", default="task:casf-executor-e2e")
    parser.add_argument("--task-revision", type=int, default=1)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    arguments = parser.parse_args()

    credentials = request_state_owner_bootstrap(
        arguments.bootstrap_fd,
        client_id=arguments.client_id,
        store_id=arguments.store_id,
    )
    credentials.install_environment()
    route_policy_id = credentials.execution_route_policy.policy_id
    route_aliases = {
        entry.task_alias for entry in credentials.execution_route_policy.entries
    }

    owner_connections: list[TypedStateOwnerConnection] = []

    def connection_factory(_endpoint: Any) -> TypedStateOwnerConnection:
        connection = TypedStateOwnerConnection(
            socket_path=Path(credentials.socket_path),
            token=credentials.token,
            client_id=credentials.client_id,
            process_birth_id=credentials.process_birth_id,
            store_id=credentials.store_id,
        )
        owner_connections.append(connection)
        return connection

    client = QuackStateClient(
        owner_id=credentials.client_id,
        store_id=credentials.store_id,
        process_birth_id=credentials.process_birth_id,
        connection_factory=connection_factory,
    )
    try:
        client.attach(credentials.endpoint, server_id=credentials.server_id)
        task_source = TypedDatabaseTaskSource(client, owns_client=False)
        metadata = client.execute("whoami_metadata")
        result: dict[str, Any] = {
            "pid": os.getpid(),
            "attached": bool(metadata),
            "claimed": False,
            "route_policy_in_argv": any(
                route_policy_id in argument
                or any(alias in argument for alias in route_aliases)
                for argument in sys.argv
            ),
            "route_policy_in_environment": any(
                route_policy_id in value
                or any(alias in value for alias in route_aliases)
                for value in os.environ.values()
            ),
            "granted_operations": sorted(
                owner_connections[-1].grant.get("allowed_operations") or ()
            ),
            "granted_command_operations": sorted(
                owner_connections[-1].grant.get("allowed_command_operations")
                or ()
            ),
        }
        try:
            client.execute("count_tasks")
        except (QuackClientError, TypedStateOwnerError):
            result["unrelated_read_denied"] = True
        else:
            result["unrelated_read_denied"] = False
        if arguments.claim:
            claim = task_source.compare_and_set_status(
                arguments.task_cid,
                arguments.task_revision,
                "in_progress",
                {
                    "operation": "database_claim",
                    "claim_phase_schema": (
                        TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                    ),
                    "claim_process_attestation": dict(
                        task_source.claim_process_attestation()
                    ),
                    "claim_id": f"claim:casf-executor-e2e:{os.getpid()}",
                    "attempt_id": f"attempt:casf-executor-e2e:{os.getpid()}",
                    "attempt_number": 1,
                    "lease_id": f"lease:casf-executor-e2e:{os.getpid()}",
                    "owner_session_id": "session:casf-executor-e2e",
                    "fencing_token": 1,
                    "fence_epoch": 1,
                    "claimed_from_revision": arguments.task_revision,
                },
            )
            result["claimed"] = bool(claim.changed)
        provider_environment = provider_subprocess_environment(os.environ)
        result["provider_received_token"] = (
            TYPED_STATE_OWNER_TOKEN_ENV in provider_environment
        )
        result["provider_received_socket"] = (
            TYPED_STATE_OWNER_SOCKET_ENV in provider_environment
        )
        arguments.output.write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        client.close()
    if arguments.hold_seconds > 0:
        time.sleep(arguments.hold_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
