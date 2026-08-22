"""Real-PID qualification of the sealed CASF coordinator production chain.

Only :class:`QuackStateServer` opens the real migrated DuckDB.  Both the
bootstrap authority and the child coordinator use named operations through
the typed Unix-socket state-owner boundary.  The coordinator credentials are
PID/process-birth bound and cross the process boundary only through a private
inherited pipe.
"""

# Python 3.8 compatibility intentionally uses ``datetime.timezone.utc``.
# ruff: noqa: UP017

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.federation.bootstrap_runtime import (
    admit_bootstrap_federation,
)
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationLifecycleState,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
    process_birth_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    QuackStateServerReadyError,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionConflictKind,
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
    TransportMode,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS,
    SUPERVISOR_RUNTIME_CHILD_ALLOWED_OPERATIONS,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerConnection,
    TypedStateOwnerRemoteError,
    build_control_plane_operation_catalog,
    catalog_fingerprint,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability,
    _migrate,
    _profile,
)

_ROOT = Path(__file__).resolve().parents[3]
_T = TypeVar("_T")


def _eventually(
    observe: Callable[[], _T | None],
    *,
    timeout_seconds: float,
    failure: str,
) -> _T:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        observed = observe()
        if observed is not None:
            return observed
        time.sleep(0.025)
    pytest.fail(failure)


def _status_when(
    path: Path,
    process: subprocess.Popen[bytes],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    def observe() -> dict[str, Any] | None:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            try:
                terminal_status = path.read_text(encoding="utf-8")
            except OSError:
                terminal_status = "<unavailable>"
            pytest.fail(
                "coordinator exited before publishing the expected state: "
                f"returncode={process.returncode}, "
                f"stdout={stdout.decode(errors='replace')!r}, "
                f"stderr={stderr.decode(errors='replace')!r}, "
                f"status={terminal_status}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return payload if predicate(payload) else None

    return _eventually(
        observe,
        timeout_seconds=timeout_seconds,
        failure="coordinator did not publish the expected bounded status",
    )


def _write_pipe(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(descriptor, payload[offset:])


def test_failure_observation_is_bounded_and_message_free() -> None:
    remote = TypedStateOwnerRemoteError(
        "authorization_denied",
        "TypedStateOwnerAuthorizationError",
    )
    try:
        raise TransactionError(
            "driver text that must not be persisted",
            kind=TransactionConflictKind.UNKNOWN,
        ) from remote
    except TransactionError as error:
        observation = supervisor_runtime._failure_observation(error)

    assert observation == {
        "chain": [
            {
                "error_class": "TransactionError",
                "kind": "unknown",
            },
            {
                "error_class": "TypedStateOwnerRemoteError",
                "error_code": "authorization_denied",
                "error_type": "TypedStateOwnerAuthorizationError",
            },
        ]
    }
    encoded = json.dumps(observation)
    assert "driver text" not in encoded
    assert "typed state-owner" not in encoded


@pytest.mark.timeout(45)
def test_authenticated_bootstrap_routes_to_real_pid_coordinator_and_stops_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise bootstrap -> outbox -> wait -> take -> ack -> STOPPED."""

    server = build_server(
        database_path=tmp_path / "control.duckdb",
        state_dir=tmp_path / "owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="casf-supervisor-runtime-e2e-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    central_client_id = "client:casf-supervisor-runtime-e2e"
    catalog = build_control_plane_operation_catalog()
    central_token = server.issue_typed_client_grant(
        client_id=central_client_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=tuple(catalog),
        allowed_command_operations=(
            "federation.create",
            "budget.reserve",
            "budget.release",
            "supervisor.register",
            "subagent.register",
            "subscription.register",
            "event.route.persist",
            "event.outbox.disposition",
        ),
        peer_pid=os.getpid(),
        ttl_seconds=120.0,
    )

    def central_connection(_endpoint: Any) -> TypedStateOwnerConnection:
        return TypedStateOwnerConnection(
            socket_path=server.typed_command_socket_path(),
            token=central_token,
            client_id=central_client_id,
            process_birth_id=identity.process_birth_id,
            store_id=identity.store_id,
        )

    client = QuackStateClient(
        owner_id=central_client_id,
        store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
        expected_identity=identity.store_identity(),
        connection_factory=central_connection,
    )
    process: subprocess.Popen[bytes] | None = None
    write_descriptor = -1
    runtime_token = ""
    event_token = ""
    admission: Any = None
    try:
        session = client.attach(identity.listen_uri, server_id=identity.server_id)
        assert session.transport_mode is TransportMode.QUACK
        repository = server.bind_federation_repository(
            client,
            require_quack_authority=True,
        )
        generation = client.load_generation()
        admission = admit_bootstrap_federation(
            repository,
            profile=_profile(),
            repository_id="repository:ipfs_accelerate_py",
            repository_tree_id="tree:casf-supervisor-runtime-e2e-v1",
            plan_root_ref="plan-root:casf-supervisor-runtime-e2e-v1",
            operation_catalog_ref=catalog_fingerprint(catalog),
            control_plane_generation=generation.generation,
            fencing_epoch=generation.fence_epoch,
            ready_task_refs=("CASF-013",),
            authentication_key=b"casf-supervisor-runtime-e2e-key",
            now=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        assert admission.federation_receipt.outcome == "accepted"
        assert any(
            item.startswith("authentication:casf-local-bootstrap:")
            for item in admission.federation_receipt.evidence_refs
        )

        pump = server.start_federation_outbox_worker(
            health_deadline_seconds=2.0,
        )
        assert pump["available"] is True
        assert pump["polling"] is False
        assert pump["thread_alive"] is True
        assert pump["initial_event_count"] >= 1
        assert pump["initial_delivery_count"] >= 1

        queued = repository.load_deliverable_deliveries(
            admission.subscription.subscription_id,
            admission.subscription.revision,
            tenant_id=admission.subscription.tenant_id,
            federation_id=admission.subscription.federation_id,
            maximum=admission.subscription.maximum_batch,
            expected_fencing_epoch=admission.fencing_epoch,
        )
        assert queued
        routed = queued[0]
        routed_event = routed.delivery.decision.representative_event
        assert routed_event.supervisor_id == admission.supervisor.record_id
        assert routed_event.event_type.value == "SUPERVISOR_HEALTH_CHANGED"

        read_descriptor, write_descriptor = os.pipe()
        os.set_inheritable(read_descriptor, True)
        child_environment = dict(os.environ)
        child_environment.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
        child_environment.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
        child_code = (
            "import sys; "
            "from ipfs_accelerate_py.agent_supervisor.federation import "
            "supervisor_runtime as runtime; "
            "runtime.HEARTBEAT_SECONDS = 0.05; "
            "raise SystemExit(runtime.run_supervisor_runtime(int(sys.argv[1])))"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", child_code, str(read_descriptor)],
            cwd=_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_environment,
            pass_fds=(read_descriptor,),
            start_new_session=True,
        )
        os.close(read_descriptor)

        birth = _eventually(
            lambda: read_process_birth(process.pid),
            timeout_seconds=3.0,
            failure="coordinator process birth did not become observable",
        )
        birth_id = process_birth_id(birth)
        common_grant = {
            "peer_pid": process.pid,
            "process_birth_id": birth_id,
            "tenant_id": admission.federation_identity.binding.tenant_id,
            "federation_id": admission.federation_identity.record_id,
            "ttl_seconds": 120.0,
        }
        runtime_token = server.issue_typed_client_grant(
            client_id="casf-supervisor-runtime:" + admission.supervisor.record_id,
            allowed_operations=tuple(SUPERVISOR_RUNTIME_CHILD_ALLOWED_OPERATIONS),
            allowed_command_operations=(
                "supervisor.runtime.attest",
                "supervisor.transition",
            ),
            entity_scopes={"supervisor_id": admission.supervisor.record_id},
            **common_grant,
        )
        event_token = server.issue_typed_client_grant(
            client_id="casf-supervisor-events:" + admission.supervisor.record_id,
            allowed_operations=tuple(SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS),
            allowed_command_operations=(
                "event.delivery.record",
                "event.acknowledge",
            ),
            entity_scopes={
                "subscription_id": admission.subscription.subscription_id,
            },
            **common_grant,
        )
        gateway = server._command_gateway
        assert gateway is not None
        assert gateway._grants[runtime_token].allowed_operations == (
            SUPERVISOR_RUNTIME_CHILD_ALLOWED_OPERATIONS
        )
        assert gateway._grants[event_token].allowed_operations == (
            SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS
        )
        assert "casf_select_active_admission_budget_usage" not in gateway._grants[
            runtime_token
        ].allowed_operations
        assert "casf_select_active_admission_budget_usage" not in gateway._grants[
            event_token
        ].allowed_operations
        assert "event.delivery.fail" not in gateway._grants[
            event_token
        ].allowed_command_operations
        observed_ack_capacity_wakes: list[tuple[int, int, int]] = []
        observed_semantic_commits: dict[str, list[tuple[Any, Any]]] = {}
        original_observer = gateway._commit_observer
        assert original_observer is not None

        def observe_child_commit(command, manifest):
            operation = str(command.parameters.get("operation") or "")
            if operation in {
                "supervisor.runtime.attest",
                "supervisor.transition",
                "event.delivery.record",
                "event.acknowledge",
            }:
                observed_semantic_commits.setdefault(operation, []).append(
                    (command, deepcopy(tuple(manifest)))
                )
            before = int(
                server.outbox_worker_capability()["notification_generation"]
            )
            original_observer(command, manifest)
            after = int(
                server.outbox_worker_capability()["notification_generation"]
            )
            if operation == "event.acknowledge":
                sequences = [
                    int(bound.get("global_sequence") or 0)
                    for name, bound in manifest
                    if name == "casf_insert_event_acknowledgement"
                ]
                observed_ack_capacity_wakes.append((max(sequences), before, after))

        gateway._commit_observer = observe_child_commit
        status_path = (tmp_path / "runtime" / "supervisor-status.json").resolve()
        task_state_path = (tmp_path / "runtime" / "task-state.json").resolve()
        bundle = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "causal-federation-runtime-credentials@1"
            ),
            "endpoint": identity.listen_uri,
            "socket_path": str(server.typed_command_socket_path()),
            "store_id": identity.store_id,
            "server_id": identity.server_id,
            "process_birth_id": birth_id,
            "runtime_token": runtime_token,
            "event_token": event_token,
            "tenant_id": admission.federation_identity.binding.tenant_id,
            "federation_id": admission.federation_identity.record_id,
            "supervisor_id": admission.supervisor.record_id,
            "subscription_id": admission.subscription.subscription_id,
            "consumer_id": admission.subscription.consumer_id,
            "fencing_epoch": admission.fencing_epoch,
            "task_count": 1,
            "completed_count": 0,
            "ready_count": 1,
            "status_path": str(status_path),
            "task_state_path": str(task_state_path),
        }
        encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
        assert len(encoded) <= 65_536

        # Neither grant is observable through process arguments or environment.
        cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
        environ = Path(f"/proc/{process.pid}/environ").read_bytes()
        for credential in (runtime_token, event_token):
            assert credential.encode() not in cmdline
            assert credential.encode() not in environ
        assert (TYPED_STATE_OWNER_TOKEN_ENV + "=").encode() not in environ

        _write_pipe(write_descriptor, encoded)
        os.close(write_descriptor)
        write_descriptor = -1

        idle = _status_when(
            status_path,
            process,
            lambda item: (
                item.get("lifecycle_state")
                == FederationLifecycleState.IDLE.value
                and item.get("event_wait_qualified") is True
                and item.get("server_owned_event_wait") is True
            ),
            timeout_seconds=12.0,
        )
        assert idle["supervisor_pid"] == process.pid
        processed = _status_when(
            status_path,
            process,
            lambda item: (
                item.get("lifecycle_state")
                == FederationLifecycleState.IDLE.value
                and int(item.get("events_processed") or 0) >= 1
                and int(item.get("event_cursor") or 0)
                >= routed_event.global_sequence
                and int(item.get("wait_calls") or 0) >= 1
                and int(item.get("last_batch_size") or 0) >= 1
                and int(item.get("heartbeat_count") or 0) >= 3
            ),
            timeout_seconds=12.0,
        )

        def lifecycle_events() -> dict[str, dict[str, Any]] | None:
            selected: dict[str, dict[str, Any]] = {}
            for command, manifest in observed_semantic_commits.get(
                "supervisor.transition", ()
            ):
                state = str(command.parameters.get("requested_state") or "")
                if state not in {
                    FederationLifecycleState.STARTING.value,
                    FederationLifecycleState.IDLE.value,
                }:
                    continue
                event = next(
                    (
                        dict(bound)
                        for name, bound in manifest
                        if name == "casf_insert_domain_event"
                    ),
                    None,
                )
                if event is not None:
                    selected[state] = event
            return selected if len(selected) == 2 else None

        transition_events = _eventually(
            lifecycle_events,
            timeout_seconds=3.0,
            failure="coordinator did not commit both lifecycle events",
        )
        transition_event_ids = {
            str(event["event_id"]) for event in transition_events.values()
        }
        transition_watermark = max(
            int(event["global_sequence"]) for event in transition_events.values()
        )

        def child_event_ids(operation: str) -> set[str]:
            return {
                str(command.parameters.get("event_id") or "")
                for command, _manifest in observed_semantic_commits.get(operation, ())
            }

        lifecycle_drained = _status_when(
            status_path,
            process,
            lambda item: (
                item.get("lifecycle_state")
                == FederationLifecycleState.IDLE.value
                and item.get("error_class") == ""
                and int(item.get("event_cursor") or 0) >= transition_watermark
                and transition_event_ids.issubset(
                    child_event_ids("event.delivery.record")
                )
                and transition_event_ids.issubset(
                    child_event_ids("event.acknowledge")
                )
            ),
            timeout_seconds=12.0,
        )
        owner_projection = _eventually(
            lambda: (
                projection
                if int(projection["outbox_worker"]["watermark"])
                >= transition_watermark
                and int(projection["outbox_worker"]["committed_sequence"])
                >= transition_watermark
                else None
            )
            if (
                projection := json.loads(
                    server.status_path().read_text(encoding="utf-8")
                )
            )
            else None,
            timeout_seconds=3.0,
            failure="owner status did not publish the drained event watermark",
        )
        assert processed["runtime_process_birth_id"] == birth_id
        assert lifecycle_drained["events_processed"] >= processed["events_processed"]
        assert owner_projection["outbox_worker"]["thread_alive"] is True
        assert owner_projection["outbox_worker"]["last_error_type"] == ""
        assert observed_ack_capacity_wakes
        assert all(
            sequence > 0 and after > before
            for sequence, before, after in observed_ack_capacity_wakes
        )
        live_cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
        live_environ = Path(f"/proc/{process.pid}/environ").read_bytes()
        for credential in (runtime_token, event_token):
            assert credential.encode() not in live_cmdline
            assert credential.encode() not in live_environ

        cursor = repository.get_cursor(
            tenant_id=admission.subscription.tenant_id,
            federation_id=admission.subscription.federation_id,
            consumer_id=admission.subscription.consumer_id,
            subscription_id=admission.subscription.subscription_id,
        )
        assert cursor.global_sequence >= processed["event_cursor"]
        assert cursor.global_sequence >= routed_event.global_sequence
        assert cursor.revision >= 2

        attempt_number = routed.delivery.attempt_number + 1
        attempt_id = "delivery-attempt:" + content_identity(
            {
                "delivery_id": routed.delivery.delivery_id,
                "attempt_number": attempt_number,
            }
        )
        attempt_rows = client.execute(
            "casf_select_delivery_for_ack",
            {
                "attempt_id": attempt_id,
                "tenant_id": admission.subscription.tenant_id,
                "federation_id": admission.subscription.federation_id,
                "event_id": routed_event.event_id,
                "subscription_id": admission.subscription.subscription_id,
                "subscription_revision": admission.subscription.revision,
                "consumer_id": admission.subscription.consumer_id,
                "fencing_epoch": admission.fencing_epoch,
            },
        )
        assert len(attempt_rows) == 1
        assert attempt_rows[0]["attempt_id"] == attempt_id
        assert attempt_rows[0]["status"] == "acknowledged"
        assert attempt_rows[0]["attempt_number"] == attempt_number
        assert attempt_rows[0]["fencing_epoch"] == admission.fencing_epoch
        assert attempt_rows[0]["delivery_id"] == routed.delivery.delivery_id
        assert attempt_rows[0]["queue_status"] == "acknowledged"

        health_rows = client.execute(
            "casf_select_supervisor_bootstrap_health",
            {
                "tenant_id": admission.subscription.tenant_id,
                "federation_id": admission.subscription.federation_id,
                "supervisor_id": admission.supervisor.record_id,
                "subscription_id": admission.subscription.subscription_id,
                "consumer_id": admission.subscription.consumer_id,
                "event_id": processed["first_event_id"],
                "acknowledgement_id": processed["first_acknowledgement_id"],
                "delivery_attempt_id": processed["first_delivery_attempt_id"],
                "observed_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
        )
        assert len(health_rows) == 1
        health = health_rows[0]
        assert health["lifecycle_state"] == FederationLifecycleState.IDLE.value
        assert health["process_id"] == process.pid
        assert health["acknowledged_event_id"] == processed["first_event_id"]
        assert health["acknowledgement_id"] == processed["first_acknowledgement_id"]
        assert health["delivery_attempt_id"] == processed["first_delivery_attempt_id"]
        assert health["delivery_attempt_status"] == "acknowledged"
        assert health["delivery_queue_status"] == "acknowledged"
        assert health["cursor_global_sequence"] >= health["acknowledged_global_sequence"]
        # The deliberately accelerated heartbeat can commit the next health
        # event while this snapshot is read.  The field is the total pending
        # range (including events after the bootstrap acknowledgement), not a
        # falsely truncated bootstrap-only count.
        assert int(health["pending_required_deliveries"]) >= 0

        for projection_path in (status_path, task_state_path):
            projection = projection_path.read_bytes()
            assert runtime_token.encode() not in projection
            assert event_token.encode() not in projection

        # A router-thread failure after the exact first acknowledgement must
        # not be masked by the still-live coordinator process and runtime
        # lease.  Inject at the next bounded owner wait, after the successful
        # production chain above has been proven.
        worker = server._outbox_worker
        assert worker is not None
        assert server.outbox_worker_capability()["last_error_type"] == ""

        def fail_next_wait(*, deadline_monotonic: float):
            del deadline_monotonic
            raise RuntimeError("injected owner outbox failure")

        monkeypatch.setattr(worker, "wait_and_drain", fail_next_wait)
        failed_worker = _eventually(
            lambda: (
                dict(server.outbox_worker_capability())
                if server.outbox_worker_capability()["available"] is False
                else None
            ),
            timeout_seconds=5.0,
            failure="owner did not publish the injected outbox-worker failure",
        )
        assert failed_worker["last_error_type"] == "RuntimeError"
        owner_status = _eventually(
            lambda: (
                status
                if (
                    status := json.loads(
                        server.status_path().read_text(encoding="utf-8")
                    )
                )["outbox_worker"]["available"]
                is False
                else None
            ),
            timeout_seconds=2.0,
            failure="owner status projection did not publish outbox failure",
        )
        assert owner_status["outbox_worker"]["available"] is False
        assert owner_status["outbox_worker"]["last_error_type"] == "RuntimeError"
        with pytest.raises(QuackStateServerReadyError, match="outbox worker"):
            server.ready()

        process.send_signal(signal.SIGTERM)
        server.cancel_event_wait(admission.subscription.consumer_id)
        stdout, stderr = process.communicate(timeout=10.0)
        assert process.returncode == 0, stderr.decode(errors="replace")
        assert runtime_token.encode() not in stdout + stderr
        assert event_token.encode() not in stdout + stderr
        stopped = json.loads(status_path.read_text(encoding="utf-8"))
        assert stopped["lifecycle_state"] == FederationLifecycleState.STOPPED.value
        assert stopped["events_processed"] >= processed["events_processed"]
        assert stopped["error_class"] == ""
        assert server.outbox_worker_capability()["last_error_type"] == "RuntimeError"

        # The production child traversed each owner-side semantic command
        # path above.  Reuse those exact committed manifests to prove that a
        # compromised holder cannot change the lifecycle event, manufacture
        # a delivery, or skip the cursor.  Each forgery is rejected by the
        # state owner before a commit can become durable.
        def captured(operation: str) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
            command, manifest = observed_semantic_commits[operation][0]
            return command, [
                (name, dict(bound)) for name, bound in deepcopy(manifest)
            ]

        transition_command, transition_manifest = captured("supervisor.transition")
        transition_update = next(
            bound
            for name, bound in transition_manifest
            if name == "casf_update_supervisor_lifecycle"
        )
        transition_event = next(
            bound
            for name, bound in transition_manifest
            if name == "casf_insert_domain_event"
        )
        transition_authority = {
            "operation": "supervisor.transition",
            "scope": {
                field: transition_command.parameters[field]
                for field in ("tenant_id", "federation_id", "supervisor_id")
            },
            "supervisor": {
                "revision": int(transition_update["expected_revision"]),
                "fencing_epoch": int(
                    transition_update["expected_fencing_epoch"]
                ),
            },
            "target_state": str(transition_update["lifecycle_state"]),
        }
        transition_event["event_type"] = "TASK_COMPLETED"
        with pytest.raises(
            TypedStateOwnerAuthorizationError,
            match="semantic mutation differs",
        ):
            gateway._validate_semantic_manifest(
                transition_command,
                transition_manifest,
                transition_authority,
            )

        delivery_command, delivery_manifest = captured("event.delivery.record")
        delivery_attempt = next(
            bound
            for name, bound in delivery_manifest
            if name == "casf_insert_delivery_attempt"
        )
        delivery_queue = next(
            bound
            for name, bound in delivery_manifest
            if name == "casf_mark_queue_delivered"
        )
        delivery_authority = {
            "operation": "event.delivery.record",
            "scope": {
                field: delivery_command.parameters[field]
                for field in (
                    "tenant_id",
                    "federation_id",
                    "event_id",
                    "subscription_id",
                    "consumer_id",
                )
            },
            "attempt_id": delivery_attempt["attempt_id"],
            "outbox_id": delivery_attempt["outbox_id"],
            "delivery_id": delivery_attempt["delivery_id"],
            "subscription_revision": int(
                delivery_attempt["subscription_revision"]
            ),
            "attempt_number": int(delivery_attempt["attempt_number"]),
            "prior_attempt_number": int(delivery_queue["prior_attempt_number"]),
            "queue_revision": int(delivery_queue["expected_revision"]),
        }
        delivery_attempt["event_id"] = "event:forged"
        with pytest.raises(
            TypedStateOwnerAuthorizationError,
            match="semantic mutation differs",
        ):
            gateway._validate_semantic_manifest(
                delivery_command,
                delivery_manifest,
                delivery_authority,
            )

        acknowledgement_command, acknowledgement_manifest = captured(
            "event.acknowledge"
        )
        acknowledgement = next(
            bound
            for name, bound in acknowledgement_manifest
            if name == "casf_insert_event_acknowledgement"
        )
        acknowledgement_queue = next(
            bound
            for name, bound in acknowledgement_manifest
            if name == "casf_mark_queue_acknowledged"
        )
        acknowledgement_cursor = next(
            bound
            for name, bound in acknowledgement_manifest
            if name == "casf_advance_consumer_cursor"
        )
        acknowledgement_authority = {
            "operation": "event.acknowledge",
            "scope": {
                field: acknowledgement_command.parameters[field]
                for field in (
                    "tenant_id",
                    "federation_id",
                    "event_id",
                    "subscription_id",
                    "consumer_id",
                )
            },
            "subscription_revision": int(
                acknowledgement["subscription_revision"]
            ),
            "cursor_revision": int(acknowledgement_cursor["expected_revision"]),
            "attempt_id": acknowledgement["delivery_attempt_id"],
            "delivery_id": acknowledgement_queue["delivery_id"],
            "queue_revision": int(acknowledgement_queue["expected_revision"]),
            "event_sequence": int(acknowledgement["global_sequence"]),
            "acknowledgement_id": acknowledgement["acknowledgement_id"],
            "disposition": "processed",
        }
        acknowledgement_cursor["upper_global_sequence"] = (
            int(acknowledgement["global_sequence"]) + 1
        )
        with pytest.raises(
            TypedStateOwnerAuthorizationError,
            match="semantic mutation differs",
        ):
            gateway._validate_semantic_manifest(
                acknowledgement_command,
                acknowledgement_manifest,
                acknowledgement_authority,
            )
    finally:
        if write_descriptor >= 0:
            os.close(write_descriptor)
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            if admission is not None:
                server.cancel_event_wait(admission.subscription.consumer_id)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        client.close()
        server.stop()
