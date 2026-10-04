"""Real-process helper for the CASF executor bootstrap qualification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
)


class _ReservationCaptured(BaseException):
    """Stop the helper after the authoritative reservation CAS."""


def _json_value(value: Any) -> Any:
    """Copy immutable owner projections into plain JSON containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


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
    parser.add_argument(
        "--daemon-action",
        choices=("reserve", "recover", "claim"),
    )
    parser.add_argument("--database", type=Path)
    parser.add_argument("--coordination-path", type=Path)
    parser.add_argument("--execution-path", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--owner-session-id", default="session:casf-executor-e2e")
    parser.add_argument("--task-prefix", default="DOEP-")
    parser.add_argument("--clock-offset-ms", type=int, default=0)
    parser.add_argument("--accept-fixture-callback-proof", action="store_true")
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

    client = QuackStateClient(
        owner_id=credentials.client_id,
        store_id=credentials.store_id,
        process_birth_id=credentials.process_birth_id,
    )
    try:
        client.attach(credentials.endpoint, server_id=credentials.server_id)
        connection_adapter = getattr(client, "_adapter", None)
        owner_connection = getattr(connection_adapter, "raw", None)
        if not isinstance(owner_connection, TypedStateOwnerConnection):
            raise AssertionError("helper did not attach the exact typed owner")
        task_source = TypedDatabaseTaskSource(
            client,
            owns_client=False,
            execution_route_policy=credentials.execution_route_policy,
        )
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
                owner_connection.grant.get("allowed_operations") or ()
            ),
            "granted_command_operations": sorted(
                owner_connection.grant.get("allowed_command_operations")
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
        if arguments.daemon_action is not None:
            required_paths = (
                arguments.database,
                arguments.coordination_path,
                arguments.execution_path,
                arguments.repo_root,
            )
            if any(path is None for path in required_paths):
                parser.error(
                    "--daemon-action requires --database, --coordination-path, "
                    "--execution-path, and --repo-root"
                )
            provider_calls: list[str] = []

            def reject_provider(attempt: DatabaseTaskAttempt) -> Any:
                provider_calls.append(attempt.attempt_id)
                raise AssertionError("bootstrap recovery helper dispatched provider")

            def child_clock() -> int:
                return int(time.time() * 1_000) + arguments.clock_offset_ms

            task_source._clock_ms = child_clock  # type: ignore[attr-defined]
            daemon = DatabaseImplementationDaemon(
                database_path=arguments.database,
                coordination_path=arguments.coordination_path,
                execution_path=arguments.execution_path,
                owner_session_id=arguments.owner_session_id,
                process_instance_id=credentials.process_birth_id,
                authority_mode="quack",
                task_source_kind="duckdb",
                quack_uri=credentials.endpoint,
                task_source=task_source,
                close_task_source=False,
                state_owner_bootstrap_credentials=credentials,
                install_schema=False,
                repo_root=arguments.repo_root,
                merge_target_ref="HEAD",
                task_prefix=arguments.task_prefix,
                provider_fn=reject_provider,
                post_merge_recovery_fn=lambda: None,
                require_real_execution=True,
                max_task_attempts=10,
                lease_ms=5_000,
                clock_ms=child_clock,
            ).open()
            try:
                if arguments.accept_fixture_callback_proof:
                    daemon._verified_post_merge_callback_integration_receipt = (
                        lambda raw, **_kwargs: dict(raw)
                    )
                    daemon._merge_repo_root = arguments.repo_root
                    daemon._merge_target_branch = "main"
                if arguments.daemon_action == "reserve":
                    captured: dict[str, Any] = {}

                    def capture_reservation(
                        attempt: DatabaseTaskAttempt,
                        *,
                        reservation_receipt: Mapping[str, Any],
                    ) -> None:
                        captured["attempt"] = attempt
                        captured["reservation_receipt"] = dict(
                            reservation_receipt
                        )
                        raise _ReservationCaptured

                    daemon._promote_typed_attempt_admission = capture_reservation
                    try:
                        daemon.claim_next()
                    except _ReservationCaptured:
                        pass
                    attempt = captured.get("attempt")
                    if not isinstance(attempt, DatabaseTaskAttempt):
                        record = task_source.get_task(arguments.task_cid)
                        ready = task_source.ready_tasks(limit=10)
                        result["reserve_probe"] = {
                            "automatic_claim_exclusions": sorted(
                                daemon._automatic_claim_exclusions()
                            ),
                            "callback_retry_within_budget": (
                                None
                                if record is None
                                else daemon._callback_no_effect_retry_claim_is_within_budget(
                                    record
                                )
                            ),
                            "ready_task_cids": [
                                item.task_cid for item in ready.tasks
                            ],
                            "coordination": (
                                daemon.coordinator.coordination_registry_projection()
                            ),
                        }
                    else:
                        record = task_source.get_task(attempt.task_cid)
                        result["daemon_attempt"] = attempt.to_dict()
                    result["daemon_record"] = (
                        None if record is None else record.to_dict()
                    )
                    if isinstance(attempt, DatabaseTaskAttempt):
                        result["reservation_receipt"] = captured[
                            "reservation_receipt"
                        ]
                elif arguments.daemon_action == "recover":
                    before_recovery = task_source.get_task(arguments.task_cid)
                    before_receipt = (
                        before_recovery.body.get("completion_receipt")
                        if before_recovery is not None
                        else None
                    )
                    result["recovery_probe"] = {
                        "shared_claim_binding": (
                            None
                            if before_recovery is None
                            else (
                                dict(binding)
                                if (
                                    binding := daemon
                                    ._shared_claim_binding_for_this_owner(
                                        before_recovery
                                    )
                                )
                                is not None
                                else None
                            )
                        ),
                        "historic_liveness": (
                            None
                            if not isinstance(before_receipt, Mapping)
                            else str(
                                daemon._typed_historic_claim_liveness(
                                    before_receipt
                                )
                            )
                        ),
                    }
                    dead_recoveries = [
                        dict(item)
                        for item in daemon._recover_lost_typed_claim_reservations()
                    ]
                    result["dead_recoveries"] = dead_recoveries
                    recovered = task_source.get_task(arguments.task_cid)
                    result["daemon_record"] = (
                        None if recovered is None else recovered.to_dict()
                    )
                    result["daemon_history"] = (
                        task_source.task_revision_history_projection(
                            arguments.task_cid
                        )
                    )
                    queue_entry = task_source.get_queue_entry(
                        arguments.task_cid
                    )
                    result["daemon_queue_entry"] = (
                        None
                        if queue_entry is None
                        else queue_entry.to_dict()
                    )
                else:
                    attempt = daemon.claim_next()
                    if attempt is None:
                        raise AssertionError("daemon did not claim a task")
                    admitted = task_source.get_task(attempt.task_cid)
                    result["daemon_attempt"] = attempt.to_dict()
                    result["daemon_record"] = (
                        None if admitted is None else admitted.to_dict()
                    )
                result["provider_calls"] = provider_calls
            finally:
                daemon.close()
        provider_environment = provider_subprocess_environment(os.environ)
        result["provider_received_token"] = (
            TYPED_STATE_OWNER_TOKEN_ENV in provider_environment
        )
        result["provider_received_socket"] = (
            TYPED_STATE_OWNER_SOCKET_ENV in provider_environment
        )
        arguments.output.write_text(
            json.dumps(_json_value(result), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if arguments.hold_seconds > 0:
            time.sleep(arguments.hold_seconds)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
