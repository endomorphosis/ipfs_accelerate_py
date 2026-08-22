"""Bounded event-wait runtime for the first-tranche CASF coordinator.

The runtime has no database path and no SQL surface.  It receives two
state-owner-issued credentials through a private inherited pipe: one exact
supervisor lifecycle capability and one exact subscription capability.  It
uses separate typed connections so a blocking wait never holds the runtime
lease channel.

This coordinator intentionally does not execute plan tasks.  It proves the
production trigger/outbox/route/wait/ack path and remains safely idle until a
later tranche admits a task executor.
"""

# Python 3.8 compatibility requires ``datetime.timezone.utc``.
# ruff: noqa: UP017

from __future__ import annotations

import json
import os
import signal
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient
from ..task_sources.typed_state_owner import TypedStateOwnerConnection
from .contracts import FederationContractError, FederationLifecycleState, utc_now
from .durable_event_router import DurableEventRouter
from .events import EventAcknowledgement, EventWaitRequest
from .registry import FederationStateRepository

MAX_CREDENTIAL_BUNDLE_BYTES = 65_536
WAIT_DEADLINE_SECONDS = 10
HEARTBEAT_SECONDS = 20


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_pipe_bundle(descriptor: int) -> dict[str, Any]:
    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 3:
        raise FederationContractError("credential pipe descriptor is invalid")
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise FederationContractError("credential pipe is unavailable") from exc
    if not stat.S_ISFIFO(metadata.st_mode) and not stat.S_ISSOCK(metadata.st_mode):
        raise FederationContractError("credentials must arrive over a private pipe")
    chunks: list[bytes] = []
    observed = 0
    try:
        while True:
            chunk = os.read(descriptor, min(8_192, MAX_CREDENTIAL_BUNDLE_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > MAX_CREDENTIAL_BUNDLE_BYTES:
                raise FederationContractError("credential bundle exceeds its bound")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FederationContractError("credential bundle is malformed") from exc
    if not isinstance(payload, dict):
        raise FederationContractError("credential bundle must be an object")
    return payload


@dataclass(frozen=True)
class SupervisorRuntimeCredentials:
    """Private, non-serializable runtime authority received from the owner."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation-runtime-credentials@1"
    )

    endpoint: str
    socket_path: Path
    store_id: str
    server_id: str
    process_birth_id: str
    runtime_token: str
    event_token: str
    tenant_id: str
    federation_id: str
    supervisor_id: str
    subscription_id: str
    consumer_id: str
    fencing_epoch: int
    task_count: int
    completed_count: int
    ready_count: int
    status_path: Path
    task_state_path: Path

    @classmethod
    def from_pipe(cls, descriptor: int) -> SupervisorRuntimeCredentials:
        payload = _read_pipe_bundle(descriptor)
        fields = {
            "schema",
            "endpoint",
            "socket_path",
            "store_id",
            "server_id",
            "process_birth_id",
            "runtime_token",
            "event_token",
            "tenant_id",
            "federation_id",
            "supervisor_id",
            "subscription_id",
            "consumer_id",
            "fencing_epoch",
            "task_count",
            "completed_count",
            "ready_count",
            "status_path",
            "task_state_path",
        }
        unknown = set(payload) - fields
        if unknown or set(payload) != fields:
            raise FederationContractError(
                "credential bundle fields differ from the closed runtime schema"
            )
        if payload.get("schema") != cls.SCHEMA:
            raise FederationContractError("credential bundle schema differs")
        text_names = (
            "endpoint",
            "store_id",
            "server_id",
            "process_birth_id",
            "runtime_token",
            "event_token",
            "tenant_id",
            "federation_id",
            "supervisor_id",
            "subscription_id",
            "consumer_id",
        )
        values = {name: str(payload.get(name) or "").strip() for name in text_names}
        if any(not value or len(value) > 4_096 for value in values.values()):
            raise FederationContractError("credential bundle identity is invalid")
        if len(values["runtime_token"]) < 16 or len(values["event_token"]) < 16:
            raise FederationContractError("runtime credential is unavailable")
        integers: dict[str, int] = {}
        for name in ("fencing_epoch", "task_count", "completed_count", "ready_count"):
            raw = payload.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise FederationContractError("runtime projection bound is invalid")
            integers[name] = raw
        if integers["fencing_epoch"] < 1:
            raise FederationContractError("runtime fencing epoch is invalid")
        if integers["completed_count"] > integers["task_count"]:
            raise FederationContractError("runtime completed count exceeds population")
        socket_path = Path(str(payload["socket_path"])).resolve(strict=False)
        status_path = Path(str(payload["status_path"])).resolve(strict=False)
        task_state_path = Path(str(payload["task_state_path"])).resolve(strict=False)
        if not socket_path.is_absolute() or not status_path.is_absolute():
            raise FederationContractError("runtime paths must be owner-resolved")
        if status_path.parent != task_state_path.parent:
            raise FederationContractError("runtime projections must share one state directory")
        return cls(
            endpoint=values["endpoint"],
            socket_path=socket_path,
            store_id=values["store_id"],
            server_id=values["server_id"],
            process_birth_id=values["process_birth_id"],
            runtime_token=values["runtime_token"],
            event_token=values["event_token"],
            tenant_id=values["tenant_id"],
            federation_id=values["federation_id"],
            supervisor_id=values["supervisor_id"],
            subscription_id=values["subscription_id"],
            consumer_id=values["consumer_id"],
            fencing_epoch=integers["fencing_epoch"],
            task_count=integers["task_count"],
            completed_count=integers["completed_count"],
            ready_count=integers["ready_count"],
            status_path=status_path,
            task_state_path=task_state_path,
        )


def _client(
    credentials: SupervisorRuntimeCredentials,
    *,
    client_id: str,
    token: str,
) -> QuackStateClient:
    def connection_factory(_endpoint: Any) -> TypedStateOwnerConnection:
        return TypedStateOwnerConnection(
            socket_path=credentials.socket_path,
            token=token,
            client_id=client_id,
            process_birth_id=credentials.process_birth_id,
            store_id=credentials.store_id,
            timeout_seconds=30.0,
        )

    client = QuackStateClient(
        owner_id=client_id,
        store_id=credentials.store_id,
        process_birth_id=credentials.process_birth_id,
        connection_factory=connection_factory,
    )
    client.attach(credentials.endpoint, server_id=credentials.server_id)
    return client


def _write_runtime_projection(
    credentials: SupervisorRuntimeCredentials,
    *,
    lifecycle_state: str,
    lifecycle_revision: int,
    event_cursor: int,
    events_processed: int,
    wait_calls: int,
    heartbeat_count: int,
    last_batch_size: int,
    last_event_id: str = "",
    last_acknowledgement_id: str = "",
    last_delivery_attempt_id: str = "",
    first_event_id: str = "",
    first_acknowledgement_id: str = "",
    first_delivery_attempt_id: str = "",
    error_class: str = "",
) -> None:
    observed = utc_now()
    task_state = {
        "schema": "CASFEventSupervisorTaskProjection@1",
        "task_count": credentials.task_count,
        "completed_count": credentials.completed_count,
        "eligible_ready_count": credentials.ready_count,
        "blocked_count": 0,
        "external_reserved_count": 0,
        "active_task_id": "",
        "implementation_in_progress": False,
        "task_execution_admitted": False,
        "event_cursor": event_cursor,
        "source": "sealed_bootstrap_projection",
    }
    task_state["projection_cid"] = content_identity(task_state)
    status = {
        "schema": "CASFEventSupervisorStatus@1",
        "status": lifecycle_state.lower(),
        "lifecycle_state": lifecycle_state,
        "lifecycle_revision": lifecycle_revision,
        "updated_at": observed,
        "supervisor_pid": os.getpid(),
        "supervisor_pid_alive": True,
        "runtime_process_birth_id": credentials.process_birth_id,
        "tenant_id": credentials.tenant_id,
        "federation_id": credentials.federation_id,
        "supervisor_id": credentials.supervisor_id,
        "subscription_id": credentials.subscription_id,
        "consumer_id": credentials.consumer_id,
        "fencing_epoch": credentials.fencing_epoch,
        "active_worker_count": 0,
        "registered_logical_subagents": 1,
        "active_subagent_processes": 0,
        "stalled_without_active_worker": False,
        "backpressure": False,
        "backpressure_reasons": [],
        "server_owned_event_wait": True,
        "event_wait_transport": "typed_state_owner_bounded_long_wait",
        "event_wait_adaptive_polling": False,
        "event_wait_qualified": True,
        "event_cursor": event_cursor,
        "events_processed": events_processed,
        "wait_calls": wait_calls,
        "heartbeat_count": heartbeat_count,
        "last_batch_size": last_batch_size,
        "last_event_id": last_event_id,
        "last_acknowledgement_id": last_acknowledgement_id,
        "last_delivery_attempt_id": last_delivery_attempt_id,
        "first_event_id": first_event_id,
        "first_acknowledgement_id": first_acknowledgement_id,
        "first_delivery_attempt_id": first_delivery_attempt_id,
        "idle_task_board_scans": 0,
        "idle_model_calls": 0,
        "idle_context_rebuilds": 0,
        "idle_activity_counter_source": "declared_by_bounded_runtime",
        "task_execution_admitted": False,
        "execution_scope": "first_tranche_event_coordination_only",
        "error_class": error_class,
        "current_status_path": str(credentials.task_state_path),
    }
    _atomic_json(credentials.task_state_path, task_state)
    _atomic_json(credentials.status_path, status)


def run_supervisor_runtime(descriptor: int) -> int:
    """Run one bounded coordinator until a termination signal is received."""

    credentials = SupervisorRuntimeCredentials.from_pipe(descriptor)
    stopping = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    runtime_client: QuackStateClient | None = None
    event_client: QuackStateClient | None = None
    runtime_repository: FederationStateRepository | None = None
    lifecycle_revision = 1
    event_cursor = 0
    events_processed = 0
    wait_calls = 0
    heartbeat_count = 0
    last_batch_size = 0
    last_event_id = ""
    last_acknowledgement_id = ""
    last_delivery_attempt_id = ""
    first_event_id = ""
    first_acknowledgement_id = ""
    first_delivery_attempt_id = ""
    try:
        runtime_client = _client(
            credentials,
            client_id="casf-supervisor-runtime:" + credentials.supervisor_id,
            token=credentials.runtime_token,
        )
        event_client = _client(
            credentials,
            client_id="casf-supervisor-events:" + credentials.supervisor_id,
            token=credentials.event_token,
        )
        runtime_repository = FederationStateRepository(
            runtime_client,
            require_quack_authority=True,
        )
        event_repository = FederationStateRepository(
            event_client,
            require_quack_authority=True,
        )
        runtime_repository.attest_supervisor_runtime(
            supervisor_id=credentials.supervisor_id,
            tenant_id=credentials.tenant_id,
            federation_id=credentials.federation_id,
            expected_revision=lifecycle_revision,
            expected_fencing_epoch=credentials.fencing_epoch,
            idempotency_key="runtime-attest:initial:" + credentials.process_birth_id,
        )
        starting = runtime_repository.transition_supervisor(
            supervisor_id=credentials.supervisor_id,
            tenant_id=credentials.tenant_id,
            federation_id=credentials.federation_id,
            requested_state=FederationLifecycleState.STARTING,
            expected_revision=lifecycle_revision,
            expected_fencing_epoch=credentials.fencing_epoch,
            active_effects=0,
            active_attempts=0,
            idempotency_key="runtime-transition:starting:" + credentials.process_birth_id,
        )
        lifecycle_revision = int(starting["revision"])
        idle = runtime_repository.transition_supervisor(
            supervisor_id=credentials.supervisor_id,
            tenant_id=credentials.tenant_id,
            federation_id=credentials.federation_id,
            requested_state=FederationLifecycleState.IDLE,
            expected_revision=lifecycle_revision,
            expected_fencing_epoch=credentials.fencing_epoch,
            active_effects=0,
            active_attempts=0,
            idempotency_key="runtime-transition:idle:" + credentials.process_birth_id,
        )
        lifecycle_revision = int(idle["revision"])
        router = DurableEventRouter(event_repository)
        subscription = router.restore_subscription(
            tenant_id=credentials.tenant_id,
            federation_id=credentials.federation_id,
            subscription_id=credentials.subscription_id,
        )
        cursor = event_repository.get_cursor(
            tenant_id=credentials.tenant_id,
            federation_id=credentials.federation_id,
            consumer_id=credentials.consumer_id,
            subscription_id=credentials.subscription_id,
        )
        event_cursor = cursor.global_sequence
        capability = event_client.event_wait_capability()
        if (
            capability.get("event_driven_qualified") is not True
            or capability.get("adaptive_polling") is not False
        ):
            raise FederationContractError("typed remote event wait is not qualified")
        _write_runtime_projection(
            credentials,
            lifecycle_state=FederationLifecycleState.IDLE.value,
            lifecycle_revision=lifecycle_revision,
            event_cursor=event_cursor,
            events_processed=events_processed,
            wait_calls=wait_calls,
            heartbeat_count=heartbeat_count,
            last_batch_size=last_batch_size,
        )
        last_heartbeat = time.monotonic()
        while not stopping.is_set():
            deadline = datetime.now(timezone.utc) + timedelta(
                seconds=WAIT_DEADLINE_SECONDS
            )
            request = EventWaitRequest(
                consumer_id=credentials.consumer_id,
                after_cursor=event_cursor,
                subscription_id=credentials.subscription_id,
                subscription_revision=subscription.revision,
                deadline=deadline.isoformat().replace("+00:00", "Z"),
                maximum_events=subscription.maximum_batch,
            )
            batch = event_client.wait_for_events(request)
            wait_calls += 1
            last_batch_size = len(batch.events)
            projection_changed = False
            if batch.events:
                deliveries = router.take(
                    credentials.subscription_id,
                    tenant_id=credentials.tenant_id,
                    federation_id=credentials.federation_id,
                    maximum=min(len(batch.events), subscription.maximum_batch),
                    expected_fencing_epoch=credentials.fencing_epoch,
                    recorded_at=utc_now(),
                )
                if not deliveries:
                    raise FederationContractError(
                        "event wait woke without a durable deliverable record"
                    )
                for exposed in deliveries:
                    event = exposed.queued.delivery.decision.representative_event
                    acknowledgement = EventAcknowledgement(
                        acknowledgement_id=(
                            "acknowledgement:"
                            + content_identity(
                                {
                                    "supervisor_id": credentials.supervisor_id,
                                    "event_id": event.event_id,
                                    "attempt_id": exposed.attempt.attempt_id,
                                }
                            )
                        ),
                        event_id=event.event_id,
                        consumer_id=credentials.consumer_id,
                        subscription_id=credentials.subscription_id,
                        subscription_revision=subscription.revision,
                        global_sequence=event.global_sequence,
                        processed_effect_ref="effect:observed:" + event.event_cid,
                        recorded_at=utc_now(),
                    )
                    cursor = event_repository.acknowledge_event(
                        acknowledgement,
                        tenant_id=credentials.tenant_id,
                        federation_id=credentials.federation_id,
                        delivery_attempt_id=exposed.attempt.attempt_id,
                        expected_cursor_revision=cursor.revision,
                        expected_fencing_epoch=credentials.fencing_epoch,
                        idempotency_key="acknowledge:" + acknowledgement.cid,
                    )
                    event_cursor = cursor.global_sequence
                    events_processed += 1
                    last_event_id = event.event_id
                    last_acknowledgement_id = acknowledgement.acknowledgement_id
                    last_delivery_attempt_id = exposed.attempt.attempt_id
                    if not first_event_id:
                        first_event_id = event.event_id
                        first_acknowledgement_id = acknowledgement.acknowledgement_id
                        first_delivery_attempt_id = exposed.attempt.attempt_id
                projection_changed = True
            if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                heartbeat_count += 1
                runtime_repository.attest_supervisor_runtime(
                    supervisor_id=credentials.supervisor_id,
                    tenant_id=credentials.tenant_id,
                    federation_id=credentials.federation_id,
                    expected_revision=lifecycle_revision,
                    expected_fencing_epoch=credentials.fencing_epoch,
                    idempotency_key=(
                        f"runtime-heartbeat:{credentials.process_birth_id}:"
                        f"{heartbeat_count}"
                    ),
                )
                last_heartbeat = time.monotonic()
                projection_changed = True
            if projection_changed:
                _write_runtime_projection(
                    credentials,
                    lifecycle_state=FederationLifecycleState.IDLE.value,
                    lifecycle_revision=lifecycle_revision,
                    event_cursor=event_cursor,
                    events_processed=events_processed,
                    wait_calls=wait_calls,
                    heartbeat_count=heartbeat_count,
                    last_batch_size=last_batch_size,
                    last_event_id=last_event_id,
                    last_acknowledgement_id=last_acknowledgement_id,
                    last_delivery_attempt_id=last_delivery_attempt_id,
                    first_event_id=first_event_id,
                    first_acknowledgement_id=first_acknowledgement_id,
                    first_delivery_attempt_id=first_delivery_attempt_id,
                )

        stopped = runtime_repository.transition_supervisor(
            supervisor_id=credentials.supervisor_id,
            tenant_id=credentials.tenant_id,
            federation_id=credentials.federation_id,
            requested_state=FederationLifecycleState.STOPPED,
            expected_revision=lifecycle_revision,
            expected_fencing_epoch=credentials.fencing_epoch,
            active_effects=0,
            active_attempts=0,
            idempotency_key="runtime-transition:stopped:" + credentials.process_birth_id,
        )
        lifecycle_revision = int(stopped["revision"])
        _write_runtime_projection(
            credentials,
            lifecycle_state=FederationLifecycleState.STOPPED.value,
            lifecycle_revision=lifecycle_revision,
            event_cursor=event_cursor,
            events_processed=events_processed,
            wait_calls=wait_calls,
            heartbeat_count=heartbeat_count,
            last_batch_size=last_batch_size,
            last_event_id=last_event_id,
            last_acknowledgement_id=last_acknowledgement_id,
            last_delivery_attempt_id=last_delivery_attempt_id,
            first_event_id=first_event_id,
            first_acknowledgement_id=first_acknowledgement_id,
            first_delivery_attempt_id=first_delivery_attempt_id,
        )
        return 0
    except BaseException as exc:
        # A local projection is never the lifecycle authority.  When the
        # owner connection is still usable, fence the failure into DuckDB and
        # its transactional event/outbox before publishing diagnostics.  A
        # hard crash or owner loss remains a typed CASF-029 recovery gap.
        if runtime_repository is not None:
            try:
                failed = runtime_repository.transition_supervisor(
                    supervisor_id=credentials.supervisor_id,
                    tenant_id=credentials.tenant_id,
                    federation_id=credentials.federation_id,
                    requested_state=FederationLifecycleState.FAILED,
                    expected_revision=lifecycle_revision,
                    expected_fencing_epoch=credentials.fencing_epoch,
                    active_effects=0,
                    active_attempts=0,
                    idempotency_key=(
                        "runtime-transition:failed:"
                        + credentials.process_birth_id
                        + ":"
                        + type(exc).__name__
                    ),
                )
                lifecycle_revision = int(failed["revision"])
            except BaseException:
                # Preserve the originating error class.  Owner-side expired
                # lease reconciliation is intentionally not simulated here.
                pass
        _write_runtime_projection(
            credentials,
            lifecycle_state=FederationLifecycleState.FAILED.value,
            lifecycle_revision=lifecycle_revision,
            event_cursor=event_cursor,
            events_processed=events_processed,
            wait_calls=wait_calls,
            heartbeat_count=heartbeat_count,
            last_batch_size=last_batch_size,
            last_event_id=last_event_id,
            last_acknowledgement_id=last_acknowledgement_id,
            last_delivery_attempt_id=last_delivery_attempt_id,
            first_event_id=first_event_id,
            first_acknowledgement_id=first_acknowledgement_id,
            first_delivery_attempt_id=first_delivery_attempt_id,
            error_class=type(exc).__name__,
        )
        return 1
    finally:
        if event_client is not None:
            event_client.close()
        if runtime_client is not None:
            runtime_client.close()


__all__ = [
    "HEARTBEAT_SECONDS",
    "MAX_CREDENTIAL_BUNDLE_BYTES",
    "SupervisorRuntimeCredentials",
    "WAIT_DEADLINE_SECONDS",
    "run_supervisor_runtime",
]
