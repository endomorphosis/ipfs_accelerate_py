"""Cross-process qualification for the typed remote event-wait boundary."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.event_wait import (
    StateOwnerEventWait,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    DomainEvent,
    EventClass,
    EventEffectClass,
    EventWaitRequest,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox import (
    EventDraft,
    materialize_event,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    TypedStateOwnerRemoteError,
)
from test.api.causal_federation.test_typed_state_owner import _gateway, _install

TENANT_ID = "tenant:remote-wait"
FEDERATION_ID = "federation:remote-wait"
CONSUMER_ID = "consumer:remote-wait"
SUBSCRIPTION_ID = "subscription:remote-wait"
CLIENT_ID = "client:remote-wait"
PROCESS_BIRTH_ID = "process-birth:remote-wait"


def _deadline(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat().replace(
        "+00:00", "Z"
    )


class _RaceWindowEventSource:
    """Expose the exact query/register race without introducing polling."""

    def __init__(self, store_generation: int) -> None:
        self._store_generation = store_generation
        self._events: tuple[DomainEvent, ...] = ()
        self._lock = threading.Lock()
        self.first_empty_query = threading.Event()
        self.release_first_query = threading.Event()
        self.query_count = 0

    def store_generation(self) -> int:
        return self._store_generation

    def events_for_subscription(
        self,
        *,
        consumer_id: str,
        subscription_id: str,
        subscription_revision: int,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[DomainEvent, ...]:
        assert consumer_id == CONSUMER_ID
        assert subscription_id == SUBSCRIPTION_ID
        assert subscription_revision == 1
        assert after_cursor == 0
        assert maximum_events == 4
        with self._lock:
            self.query_count += 1
            ordinal = self.query_count
            snapshot = self._events
        if ordinal == 1 and not snapshot:
            self.first_empty_query.set()
            assert self.release_first_query.wait(5.0)
        return snapshot

    def publish(self, event: DomainEvent) -> None:
        with self._lock:
            self._events = (event,)


def _remote_wait_child(
    control: Any,
    results: Any,
    socket_path: str,
) -> None:
    """Receive the credential only through ``control`` and exercise Quack."""

    results.send(("booted", os.getpid()))
    token = control.recv_bytes().decode("ascii")
    token_in_argv = any(token in item for item in sys.argv)
    token_in_environment = any(token in value for value in os.environ.values())
    connection: TypedStateOwnerConnection | None = None
    client: QuackStateClient | None = None
    try:
        connection = TypedStateOwnerConnection(
            socket_path=Path(socket_path),
            token=token,
            client_id=CLIENT_ID,
            process_birth_id=PROCESS_BIRTH_ID,
            store_id="control.duckdb",
            timeout_seconds=8.0,
        )
        client = QuackStateClient(
            owner_id=CLIENT_ID,
            store_id="control.duckdb",
            process_birth_id=PROCESS_BIRTH_ID,
            connect_timeout_seconds=8.0,
            connection_factory=lambda _endpoint: connection,
        )
        session = client.attach(
            "quack:127.0.0.1:7777",
            server_id="server:typed-owner-test",
        )

        scoped_denials: list[str] = []
        for consumer_id, subscription_id in (
            ("consumer:outside-grant", SUBSCRIPTION_ID),
            (CONSUMER_ID, "subscription:outside-grant"),
        ):
            try:
                client.wait_for_events(
                    EventWaitRequest(
                        consumer_id=consumer_id,
                        after_cursor=0,
                        subscription_id=subscription_id,
                        subscription_revision=1,
                        deadline=_deadline(5.0),
                        maximum_events=4,
                    )
                )
            except TypedStateOwnerRemoteError as exc:
                scoped_denials.append(exc.error_code)

        deadline_denial = ""
        try:
            client.wait_for_events(
                EventWaitRequest(
                    consumer_id=CONSUMER_ID,
                    after_cursor=0,
                    subscription_id=SUBSCRIPTION_ID,
                    subscription_revision=1,
                    deadline=_deadline(120.0),
                    maximum_events=4,
                )
            )
        except TypedStateOwnerRemoteError as exc:
            deadline_denial = exc.error_code

        capability = client.event_wait_capability()
        results.send(
            (
                "ready",
                {
                    "transport_mode": session.transport_mode.value,
                    "token_in_argv": token_in_argv,
                    "token_in_environment": token_in_environment,
                    "scope_denials": scoped_denials,
                    "deadline_denial": deadline_denial,
                    "capability": capability,
                },
            )
        )
        assert control.recv() == "start-wait"
        request = EventWaitRequest(
            consumer_id=CONSUMER_ID,
            after_cursor=0,
            subscription_id=SUBSCRIPTION_ID,
            subscription_revision=1,
            deadline=_deadline(5.0),
            maximum_events=4,
        )
        started = time.monotonic()
        batch = client.wait_for_events(request)
        results.send(("completed", batch.to_dict(), time.monotonic() - started))
    except BaseException as exc:  # pragma: no cover - parent reports exact failure
        results.send(("error", type(exc).__name__, str(exc)))
    finally:
        if client is not None:
            client.close()
        elif connection is not None:
            connection.close()


def _receive(connection: Any, expected: str, timeout: float = 10.0) -> tuple[Any, ...]:
    assert connection.poll(timeout), f"timed out waiting for child message {expected!r}"
    message = connection.recv()
    assert message[0] != "error", message
    assert message[0] == expected, message
    return message


def _event() -> DomainEvent:
    event, _outbox = materialize_event(
        EventDraft(
            event_type=EventClass.TASK_READY,
            stream_id="stream:remote-wait",
            causal_parent_ids=(),
            correlation_id="correlation:remote-wait",
            causation_id="causation:remote-wait",
            tenant_id=TENANT_ID,
            federation_id=FEDERATION_ID,
            supervisor_id="supervisor:remote-wait",
            task_id="task:remote-wait",
            repository_id="repository:remote-wait",
            tree_id="tree:remote-wait",
            payload_ref="artifact:remote-wait",
            changed_fact_refs=("fact:remote-wait",),
            effect_class=EventEffectClass.AUTHORITATIVE_STATE,
            deduplication_key="deduplication:remote-wait",
        ),
        stream_sequence=1,
        global_sequence=1,
        recorded_at="2026-08-22T00:00:00Z",
    )
    return event


@pytest.mark.skipif(not hasattr(os, "getpid"), reason="requires process identity")
def test_cross_process_typed_remote_wait_is_scoped_blocking_and_lossless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TYPED_STATE_OWNER_TOKEN_ENV, raising=False)
    monkeypatch.delenv(TYPED_STATE_OWNER_SOCKET_ENV, raising=False)
    database = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(database)
    gateway, owner_connection = _gateway(database, socket_path)
    generation = int(gateway.identity["generation"])
    source = _RaceWindowEventSource(generation)
    owner_wait = StateOwnerEventWait(source)
    handler_grants: list[tuple[str, str, int]] = []

    def wait_handler(request: EventWaitRequest, grant: Any) -> Any:
        handler_grants.append((grant.tenant_id, grant.federation_id, grant.peer_pid))
        assert grant.tenant_id == TENANT_ID
        assert grant.federation_id == FEDERATION_ID
        return owner_wait.wait_for_events(request)

    gateway.bind_event_wait_handlers(
        wait=wait_handler,
        cancel=lambda consumer_id, _grant: owner_wait.cancel(consumer_id),
        clear_cancellation=lambda consumer_id, _grant: owner_wait.clear_cancel(
            consumer_id
        ),
    )

    context = multiprocessing.get_context("spawn")
    parent_control, child_control = context.Pipe(duplex=True)
    result_receive, result_send = context.Pipe(duplex=False)
    process = context.Process(
        target=_remote_wait_child,
        args=(child_control, result_send, str(socket_path)),
    )
    process.start()
    notifier: threading.Thread | None = None
    try:
        booted = _receive(result_receive, "booted")
        assert booted[1] == process.pid
        token, grant = gateway.issue_grant(
            client_id=CLIENT_ID,
            process_birth_id=PROCESS_BIRTH_ID,
            allowed_operations=(
                "whoami_metadata",
                "load_store_generation",
                "event.wait",
            ),
            tenant_id=TENANT_ID,
            federation_id=FEDERATION_ID,
            entity_scopes={
                "consumer_id": CONSUMER_ID,
                "subscription_id": SUBSCRIPTION_ID,
            },
            peer_pid=process.pid,
            ttl_seconds=30.0,
        )
        assert grant.peer_pid == process.pid
        assert token.encode("ascii") not in Path(f"/proc/{process.pid}/cmdline").read_bytes()
        assert token.encode("ascii") not in Path(f"/proc/{process.pid}/environ").read_bytes()
        with pytest.raises((OSError, TypedStateOwnerError)):
            TypedStateOwnerConnection(
                socket_path=socket_path,
                token=token,
                client_id=CLIENT_ID,
                process_birth_id=PROCESS_BIRTH_ID,
                store_id="control.duckdb",
            )
        parent_control.send_bytes(token.encode("ascii"))

        ready = _receive(result_receive, "ready")
        evidence = ready[1]
        assert evidence["transport_mode"] == "quack"
        assert evidence["token_in_argv"] is False
        assert evidence["token_in_environment"] is False
        assert evidence["scope_denials"] == [
            "authorization_denied",
            "authorization_denied",
        ]
        assert evidence["deadline_denial"] == "authorization_denied"
        assert evidence["capability"] == {
            "available": True,
            "interface": "TypedStateOwnerEventWait@1",
            "client_interface": "QuackStateClientEventWait@1",
            "transport": "typed_state_owner_bounded_long_wait",
            "server_owned": True,
            "blocking_condition": True,
            "adaptive_polling": False,
            "event_driven_qualified": True,
        }
        assert handler_grants == []

        parent_control.send("start-wait")
        assert source.first_empty_query.wait(5.0)
        source.publish(_event())
        notifier = threading.Thread(
            target=owner_wait.notify_committed,
            args=(1,),
            name="remote-wait-race-notifier",
        )
        notifier.start()
        source.release_first_query.set()

        completed = _receive(result_receive, "completed")
        batch = completed[1]
        assert completed[2] < 4.0
        assert batch["timed_out"] is False
        assert batch["cancelled"] is False
        assert batch["server_shutdown"] is False
        assert batch["next_cursor"] == 1
        assert [item["event_id"] for item in batch["events"]] == [_event().event_id]
        assert source.query_count == 2
        assert owner_wait.query_count == 2
        assert owner_wait.wakeup_count == 1
        assert handler_grants == [(TENANT_ID, FEDERATION_ID, process.pid)]
    finally:
        source.release_first_query.set()
        owner_wait.shutdown()
        if notifier is not None:
            notifier.join(timeout=5.0)
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        parent_control.close()
        child_control.close()
        result_receive.close()
        result_send.close()
        gateway.stop()
        owner_connection.close()
    assert process.exitcode == 0
