"""Runtime qualification for CASF's owner-bound typed event wait."""

# Python 3.8 compatibility intentionally uses ``timezone.utc`` instead of the
# Python 3.11-only ``datetime.UTC`` alias.
# ruff: noqa: UP017

from __future__ import annotations

import ast
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    DomainEvent,
    EventBatch,
    EventClass,
    EventEffectClass,
    EventWaitRequest,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox import (
    EventDraft,
    materialize_event,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    QuackStateServer,
    QuackStateServerControlError,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    MigrationRunReport,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    ParsedVersion,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientError,
    QuackStateClient,
    TransportMode,
)

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / "scripts" / "run_agent_supervisor_causal_event_federation.py"
_DIGEST = "sha256:" + ("ab" * 32)
_DATABASE_UUID = "123e4567-e89b-12d3-a456-426614174000"


def _deadline(seconds: float) -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return value.isoformat().replace("+00:00", "Z")


def _request(*, seconds: float = 1.0, consumer_id: str = "consumer:test") -> EventWaitRequest:
    return EventWaitRequest(
        consumer_id=consumer_id,
        after_cursor=0,
        subscription_id="subscription:test",
        subscription_revision=1,
        deadline=_deadline(seconds),
        maximum_events=8,
    )


def _event(sequence: int = 1) -> DomainEvent:
    event, _outbox = materialize_event(
        EventDraft(
            event_type=EventClass.TASK_READY,
            stream_id="stream:test",
            causal_parent_ids=(),
            correlation_id="correlation:test",
            causation_id="causation:test",
            tenant_id="tenant:test",
            federation_id="federation:test",
            supervisor_id="supervisor:test",
            task_id="task:test",
            repository_id="repository:test",
            tree_id="tree:test",
            payload_ref="payload:test",
            changed_fact_refs=("fact:test",),
            effect_class=EventEffectClass.AUTHORITATIVE_STATE,
            deduplication_key=f"dedupe:{sequence}",
        ),
        stream_sequence=sequence,
        global_sequence=sequence,
        recorded_at="2030-01-01T00:00:00Z",
    )
    return event


class _MemoryEventSource:
    def __init__(self, *, store_generation: int = 1) -> None:
        self._lock = threading.Lock()
        self._events: list[DomainEvent] = []
        self._store_generation = store_generation
        self.queries = 0
        self.first_query = threading.Event()

    def events_for_subscription(
        self,
        *,
        consumer_id: str,
        subscription_id: str,
        subscription_revision: int,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[DomainEvent, ...]:
        assert consumer_id
        assert subscription_id == "subscription:test"
        assert subscription_revision == 1
        with self._lock:
            self.queries += 1
            self.first_query.set()
            return tuple(
                item
                for item in self._events
                if item.global_sequence > after_cursor
            )[:maximum_events]

    def store_generation(self) -> int:
        return self._store_generation

    def append(self, event: DomainEvent) -> None:
        with self._lock:
            self._events.append(event)


class _Result:
    def __init__(self, columns: tuple[str, ...] = (), rows: tuple[tuple[Any, ...], ...] = ()):
        self.description = tuple((name,) for name in columns)
        self._rows = list(rows)
        self._offset = 0

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._offset >= len(self._rows):
            return None
        row = self._rows[self._offset]
        self._offset += 1
        return row

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows[self._offset :]
        self._offset = len(self._rows)
        return rows


class _HarnessConnection:
    """One in-memory SQL-shaped owner connection; it opens no database file."""

    def __init__(self) -> None:
        self.closed = False
        self.generation = 0
        self.schema_revision = 2
        self.fence_epoch = 1
        self.revision = 0
        self.birth_id = "birth:owner"
        self.metadata = {
            "database_uuid": _DATABASE_UUID,
            "schema_version": "2",
            "schema_fingerprint": _DIGEST,
            "application_version": "test",
            "tool_version": "test",
        }

    def execute(self, sql: str, parameters: Any = None) -> _Result:
        normalized = " ".join(str(sql).strip().upper().split())
        values = list(parameters or ())
        if "COALESCE(MAX(GENERATION)" in normalized:
            return _Result(("generation",), ((self.generation,),))
        if (
            normalized.startswith("SELECT GENERATION, SCHEMA_REVISION")
            and "FROM STORE_GENERATIONS" in normalized
        ):
            if self.generation < 1:
                return _Result()
            return _Result(
                (
                    "generation",
                    "schema_revision",
                    "fence_epoch",
                    "revision",
                    "database_uuid",
                    "birth_id",
                ),
                (
                    (
                        self.generation,
                        self.schema_revision,
                        self.fence_epoch,
                        self.revision,
                        _DATABASE_UUID,
                        self.birth_id,
                    ),
                ),
            )
        if "FROM CONTROL_PLANE_METADATA" in normalized and "KEY IN" in normalized:
            rows = tuple((key, value) for key, value in sorted(self.metadata.items()))
            return _Result(("key", "value"), rows)
        if "FROM CONTROL_PLANE_METADATA" in normalized and "WHERE KEY =" in normalized:
            key = str(values[0]) if values else ""
            value = self.metadata.get(key)
            return _Result(("value",), (() if value is None else ((value,),)))
        if normalized.startswith("INSERT INTO STORE_GENERATIONS"):
            self.generation = int(values[0])
            self.schema_revision = int(values[1])
            self.fence_epoch = int(values[2])
            self.revision = int(values[3])
            self.birth_id = str(values[5])
            return _Result()
        if normalized.startswith("SELECT 1"):
            return _Result(("live",), ((1,),))
        return _Result()

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _BorrowedConnection:
    """Prevent a client view from closing the server-owned connection."""

    def __init__(self, connection: _HarnessConnection) -> None:
        self._connection = connection

    def execute(self, sql: str, parameters: Any = None) -> _Result:
        return self._connection.execute(sql, parameters)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        return None


def _capability() -> QuackCapabilityReport:
    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=QuackCapabilityStatus.COMPATIBLE,
        profile=profile,
        duckdb_importable=True,
        duckdb_version="1.5.5",
        duckdb_version_parsed=ParsedVersion(1, 5, 5, raw="1.5.5"),
        platform_name="Linux",
        platform_machine="test",
        extension=ExtensionObservation(
            name="quack",
            installed=True,
            loaded=True,
            install_path="/qualified/quack.duckdb_extension",
            extension_version="test",
        ),
        extension_fingerprint=_DIGEST,
        observed_functions=("quack_serve", "quack_query"),
        observed_surfaces=profile.required_surfaces,
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


def _migration() -> MigrationRunReport:
    return MigrationRunReport(
        from_version=2,
        to_version=2,
        receipts=(),
        schema_fingerprint=_DIGEST,
        catalog_fingerprint=_DIGEST,
        changed=False,
    )


def _server(tmp_path: Path) -> tuple[QuackStateServer, _HarnessConnection]:
    connection = _HarnessConnection()
    server = build_server(
        database_path=tmp_path / "control.duckdb",
        state_dir=tmp_path / "owner",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: _capability(),
        migrate=lambda _path: _migration(),
        connection_factory=lambda _path: connection,
        process_birth_factory=lambda: ProcessBirthIdentity(
            pid=4242,
            start_time_ticks=7,
            boot_id="boot:test",
            parent_pid=1,
        ),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    server.start()
    return server, connection


def _client(server: QuackStateServer, connection: _HarnessConnection) -> QuackStateClient:
    identity = server.identity
    assert identity is not None
    client = QuackStateClient(
        owner_id="owner:event-wait-test",
        store_id=identity.store_id,
        expected_identity=identity.store_identity(),
        connection_factory=lambda _endpoint: _BorrowedConnection(connection),
    )
    client.attach(
        identity.listen_uri,
        mode=TransportMode.QUACK,
        server_id=identity.server_id,
    )
    return client


def test_owner_boundary_has_no_lost_wakeup_and_one_shared_condition(
    tmp_path: Path,
) -> None:
    server, connection = _server(tmp_path)
    source = _MemoryEventSource()
    first_capability = server.bind_event_source(source)
    second_capability = server.bind_event_source(source)
    client = _client(server, connection)
    client.bind_event_wait_source(source, owner_boundary=server)
    results: list[EventBatch] = []
    worker = threading.Thread(target=lambda: results.append(client.wait_for_events(_request())))
    try:
        worker.start()
        assert source.first_query.wait(timeout=1)
        source.append(_event())
        assert server.notify_committed_event(1) is True
        worker.join(timeout=1)

        assert not worker.is_alive()
        assert results[0].events == (_event(),)
        assert results[0].next_cursor == 1
        assert source.queries == 2
        assert first_capability["interface"] == "StateOwnerEventWait@1"
        assert second_capability["interface"] == first_capability["interface"]
        capability = client.event_wait_capability()
        assert capability["transport"] == "owner_local_condition"
        assert capability["event_driven_qualified"] is False
        assert capability["notification_generation"] == 1
        with pytest.raises(QuackStateServerControlError, match="different event source"):
            server.bind_event_source(_MemoryEventSource())
    finally:
        client.close()
        server.stop()


def test_owner_idle_deadline_performs_one_query_and_zero_periodic_wakeups(
    tmp_path: Path,
) -> None:
    server, connection = _server(tmp_path)
    source = _MemoryEventSource()
    server.bind_event_source(source)
    client = _client(server, connection)
    client.bind_event_wait_source(source, owner_boundary=server)
    try:
        batch = client.wait_for_events(_request(seconds=0.05))

        assert batch.timed_out is True
        assert source.queries == 1
        capability = server.event_wait_capability()
        assert capability["query_count"] == 1
        assert capability["wakeup_count"] == 0
        assert capability["idle_repeated_database_scans"] is False
    finally:
        client.close()
        server.stop()


def test_event_wait_rejects_a_source_from_another_store_generation(
    tmp_path: Path,
) -> None:
    server, connection = _server(tmp_path)
    source = _MemoryEventSource(store_generation=2)
    server.bind_event_source(source)
    client = _client(server, connection)
    client.bind_event_wait_source(source, owner_boundary=server)
    try:
        with pytest.raises(
            QuackStateServerControlError,
            match="store generation differs",
        ):
            client.wait_for_events(_request(seconds=0.01))
    finally:
        client.close()
        server.stop()


def test_owner_cancellation_and_shutdown_reach_blocked_clients(tmp_path: Path) -> None:
    server, connection = _server(tmp_path)
    source = _MemoryEventSource()
    server.bind_event_source(source)
    client = _client(server, connection)
    client.bind_event_wait_source(source, owner_boundary=server)
    cancelled: list[EventBatch] = []
    first = threading.Thread(
        target=lambda: cancelled.append(client.wait_for_events(_request(seconds=2)))
    )
    first.start()
    assert source.first_query.wait(timeout=1)
    client.cancel_event_wait("consumer:test")
    first.join(timeout=1)
    assert not first.is_alive()
    assert cancelled[0].cancelled is True

    client.clear_event_wait_cancellation("consumer:test")
    source.first_query.clear()
    shutdown: list[EventBatch] = []
    second = threading.Thread(
        target=lambda: shutdown.append(client.wait_for_events(_request(seconds=2)))
    )
    second.start()
    assert source.first_query.wait(timeout=1)
    server.stop()
    second.join(timeout=1)
    try:
        assert not second.is_alive()
        assert shutdown[0].server_shutdown is True
    finally:
        client.close()


def test_remote_quack_fallback_is_bounded_backing_off_and_unqualified(
    tmp_path: Path,
) -> None:
    server, connection = _server(tmp_path)
    source = _MemoryEventSource()
    client = _client(server, connection)
    client.bind_event_wait_source(
        source,
        minimum_interval_seconds=0.01,
        maximum_interval_seconds=0.02,
        backoff_multiplier=2.0,
    )
    try:
        capability = client.event_wait_capability()
        assert capability["transport"] == "quack_adaptive_long_poll"
        assert capability["bounded"] is True
        assert capability["backs_off_when_idle"] is True
        assert capability["event_driven_qualification"] is False
        assert capability["event_driven_qualified"] is False

        batch = client.wait_for_events(_request(seconds=0.06))
        assert batch.timed_out is True
        assert 2 <= source.queries <= 10
        with pytest.raises(QuackClientError, match="cancellation is unavailable"):
            client.cancel_event_wait("consumer:test")
    finally:
        client.close()
        server.stop()


def test_canonical_repository_binding_checks_owner_identity_and_installs_hook(
    tmp_path: Path,
) -> None:
    server, connection = _server(tmp_path)
    client = _client(server, connection)
    try:
        repository = server.bind_federation_repository(client)

        assert repository.store_generation() == 1
        assert client.event_wait_capability()["transport"] == "owner_local_condition"
        assert server.notify_committed_event(4) is True
        assert server.event_wait_capability()["notification_generation"] == 1
    finally:
        client.close()
        server.stop()


def test_operator_uses_default_owner_transport_connection_and_blocking_event() -> None:
    tree = ast.parse(OPERATOR.read_text(encoding="utf-8"))
    state_owner = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == "state_owner"
    )
    build_calls = [
        node
        for node in ast.walk(state_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_server"
    ]
    assert len(build_calls) == 1
    keyword_names = {item.arg for item in build_calls[0].keywords}
    assert "transport" not in keyword_names
    assert "connection_factory" not in keyword_names
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "threading"
        and node.func.attr == "Event"
        for node in ast.walk(state_owner)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "stopping"
        and node.func.attr == "wait"
        for node in ast.walk(state_owner)
    )
    ready_call = next(
        node
        for node in ast.walk(state_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "server"
        and node.func.attr == "ready"
    )
    worker_call = next(
        node
        for node in ast.walk(state_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "server"
        and node.func.attr == "start_federation_outbox_worker"
    )
    spawn_call = next(
        node
        for node in ast.walk(state_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_spawn_event_supervisor"
    )
    health_calls = sorted(
        (
            node
            for node in ast.walk(state_owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_state_owner_outbox_health"
        ),
        key=lambda node: node.lineno,
    )
    assert len(health_calls) >= 3
    assert ready_call.lineno < worker_call.lineno < spawn_call.lineno
    assert worker_call.lineno < health_calls[0].lineno < spawn_call.lineno
    assert spawn_call.lineno < health_calls[1].lineno
