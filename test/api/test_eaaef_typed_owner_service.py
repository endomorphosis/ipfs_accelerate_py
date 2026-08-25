from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_bootstrap_gateway as runtime,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    eaaef_typed_owner_service as owner_service,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    install_eaaef_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
    quack_daemon_operation_intent,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerGateway,
    TypedStateOwnerRemoteError,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability as _quack_capability,
)
from test.api.test_eaaef_bootstrap_gateway_launch import (
    NOW_MS,
    _client,
    _signed_capability,
)
from test.api.test_eaaef_lane_gateway_runtime import (
    _admission_bundle,
    _envelope,
)
from test.api.test_eaaef_quack_command_fabric import _provision_operational


def _intent(
    capability: dict[str, object], *, limit: int
) -> dict[str, object]:
    return dict(
        quack_daemon_operation_intent(
            gateway_binding_cid=str(capability["gateway_binding_cid"]),
            operational_capability_cid=str(capability["capability_cid"]),
            operation="task.ready",
            arguments={"limit": limit},
        )
    )


def _owner_gateway(
    connection: object,
    capability: dict[str, object],
    socket_path: Path,
) -> TypedStateOwnerGateway:
    return TypedStateOwnerGateway(
        connection=connection,
        socket_path=socket_path,
        store_id=str(capability["store_id"]),
        identity={
            "server_id": "server:eaaef-owner-test",
            "store_id": capability["store_id"],
            "database_uuid": "12345678-1234-4234-8234-123456789abc",
            "generation": capability["owner_generation"],
            "fence_epoch": capability["fence_epoch"],
        },
    )


def _bind_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    owner_service.EAAEFTypedOwnerCommandService,
    object,
    object,
    dict[str, object],
    dict[str, object],
]:
    admission, capability, context = _admission_bundle(
        tmp_path / "authority"
    )
    database = tmp_path / "operational.duckdb"
    _provision_operational(database, capability)
    import duckdb

    connection = duckdb.connect(str(database))
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    gateway = _owner_gateway(connection, capability, tmp_path / "owner.sock")
    service = owner_service._bind_eaaef_typed_owner_command_service_from_gateway(  # noqa: SLF001
        owner_gateway=gateway,
        admission=admission,
    )
    return service, connection, admission, capability, context


def test_owner_local_service_rolls_back_replays_and_ignores_stale_authority_for_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, connection, _admission, capability, context = _bind_service(
        tmp_path, monkeypatch
    )
    try:
        invalid = _envelope(
            _intent(capability, limit=0),
            capability,
            context,
            serial=1,
        )
        with pytest.raises(Exception, match="limit must be"):
            service.submit_authorized_operation(invalid)
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events "
            "WHERE event_type='authorized_state_command_receipt'"
        ).fetchone()[0] == 0

        envelope = _envelope(
            _intent(capability, limit=2),
            capability,
            context,
            serial=1,
        )
        receipt = service.submit_authorized_operation(envelope)
        result = json.loads(str(receipt["result_json"]))
        assert [row["task_cid"] for row in result["value"]["tasks"]] == [
            "task:eaaef:1"
        ]
        assert service.submit_authorized_operation(envelope) == receipt
        assert service.lookup_authorized_operation_receipt(envelope) == receipt
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events "
            "WHERE event_type='authorized_state_command_receipt'"
        ).fetchone()[0] == 1

        evidence = service.evidence()
        assert evidence["borrows_open_connection"] is True
        assert evidence["shares_owner_transaction_lock"] is True
        assert evidence["opens_database"] is False
        assert evidence["local_sidecar_enabled"] is False
        assert evidence["downstream_catalog_enabled"] is False
        assert evidence["production_admitted"] is False
        assert (  # noqa: SLF001 - same-boundary qualification
            service._connection is service._owner_gateway._connection
        )
        assert (  # noqa: SLF001 - same-boundary qualification
            service._transaction_lock
            is service._owner_gateway._transaction_lock
        )
        with pytest.raises(TypeError):
            owner_service.bind_eaaef_typed_owner_command_service(
                owner_gateway=service._owner_gateway,  # noqa: SLF001
                admission=_admission,
            )
        with pytest.raises(TypeError):
            owner_service.bind_eaaef_typed_owner_command_service(
                connection=connection,
                transaction_lock=threading.Lock(),
                admission=_admission,
            )

        merge_path = context["artifact_paths"]["merge"]
        merge_path.write_text("{}", encoding="ascii")
        assert service.lookup_authorized_operation_receipt(envelope) == receipt
        fresh = _envelope(
            _intent(capability, limit=1),
            capability,
            context,
            serial=2,
        )
        with pytest.raises(
            owner_service.EAAEFTypedOwnerServiceError,
            match="stale or changed",
        ):
            service.submit_authorized_operation(fresh)
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events "
            "WHERE event_type='authorized_state_command_receipt'"
        ).fetchone()[0] == 1
    finally:
        service.close()
        connection.close()


def test_real_quack_owner_socket_drives_gateway_without_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability_bundle = _signed_capability(
        owner_bindings={
            "store_id": "eaaef-real-owner-wire-v1",
            "owner_generation": 1,
            "fence_epoch": 1,
        }
    )
    admission, capability, context = _admission_bundle(
        tmp_path / "authority",
        capability_bundle=capability_bundle,
    )
    database = tmp_path / "operational.duckdb"
    _provision_operational(database, capability)
    with open_duckdb_connection(database) as seed:
        seed.execute("DELETE FROM state_servers")
        seed.execute("DELETE FROM server_epochs")
        seed.execute("DELETE FROM store_generations")
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    monkeypatch.setattr(
        runtime.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources."
        "eaaef_borrowed_transaction.time.time_ns",
        lambda: NOW_MS * 1_000_000,
    )

    server = build_server(
        database_path=database,
        state_dir=tmp_path / "owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id=str(capability["store_id"]),
        transport=FakeQuackTransport(),
        capability_probe=_quack_capability,
        migrate=lambda path: install_eaaef_operational_schema(
            path,
            application_version="eaaef-fabric-test",
            tool_version="1.5.5",
            owner_id="eaaef-real-owner-wire-test",
        ),
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    server.clock = lambda: 4.0
    identity = server.start()
    assert identity.generation == identity.fence_epoch == 1
    service = owner_service.bind_eaaef_typed_owner_command_service(
        owner_server=server,
        admission=admission,
    )
    with pytest.raises(
        owner_service.EAAEFTypedOwnerServiceError,
        match="already bound",
    ):
        owner_service.bind_eaaef_typed_owner_command_service(
            owner_server=server,
            admission=admission,
        )
    client_id = "eaaef-real-wire-client"
    process_birth_id = "birth:eaaef-real-wire-client"
    token = server.issue_typed_client_grant(
        client_id=client_id,
        process_birth_id=process_birth_id,
        allowed_operations=tuple(
            sorted(owner_service.EAAEF_TYPED_OWNER_COMMAND_OPERATIONS)
        ),
        peer_pid=os.getpid(),
    )
    owner_connection = TypedStateOwnerConnection(
        socket_path=server.typed_command_socket_path(),
        token=token,
        client_id=client_id,
        process_birth_id=process_birth_id,
        store_id=str(capability["store_id"]),
    )
    transport = runtime.bind_eaaef_typed_owner_command_transport(
        owner_connection=owner_connection,
        admission=admission,
        maximum_wait_ms=10,
        poll_interval_ms=1,
    )
    authorization_client = _client(
        capability,
        context,
        clock_ms=lambda: NOW_MS,
    )

    def authorize(_self: object, intent: object) -> object:
        return _envelope(
            dict(intent),
            capability,
            context,
            serial=1,
            expected_revision=0,
        )

    monkeypatch.setattr(type(authorization_client), "authorize", authorize)
    journal_parent = tmp_path / "journal"
    journal_parent.mkdir(mode=0o700)
    journal = runtime.open_eaaef_exact_envelope_journal(
        journal_parent,
        admission=admission,
    )
    gateway = runtime.create_eaaef_bootstrap_command_gateway(
        admission=admission,
        authorization_client=authorization_client,
        transport=transport,
        journal=journal,
    )
    retained_envelope = _envelope(
        _intent(capability, limit=2),
        capability,
        context,
        serial=1,
        expected_revision=0,
    )
    try:
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            owner_connection._request(  # noqa: SLF001 - malformed-wire probe
                "eaaef.command.lookup",
                envelope=retained_envelope.to_dict(),
                merge_admission_cid=service.admission_cid,
                operational_capability_cid=(
                    service.operational_capability_cid
                ),
                unexpected="must-be-rejected",
            )
        assert denied.value.error_code == "protocol_denied"
        with pytest.raises(TypedStateOwnerRemoteError) as incomplete:
            owner_connection._request(  # noqa: SLF001 - malformed-wire probe
                "eaaef.command.lookup",
                envelope=retained_envelope.to_dict(),
                merge_admission_cid=service.admission_cid,
            )
        assert incomplete.value.error_code == "protocol_denied"
        assert gateway.evidence()["transport"] == "typed_state_owner"
        assert (
            owner_service.EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER
            in gateway.evidence()["production_blockers"]
        )
        with pytest.raises(QuackDaemonGatewayError, match="production no-go"):
            gateway.require_production_admission()
        gateway.attach()
        page = gateway.task_source.ready_tasks(limit=2)
        assert isinstance(page.tasks, tuple)
        receipt = transport.lookup_receipt(retained_envelope)
        assert receipt is not None
        result = json.loads(str(receipt["result_json"]))
        assert result["daemon_operation"] == "task.ready"
        assert result["value"]["revision"] == 0

        idle_client_id = "eaaef-idle-transaction-client"
        idle_birth_id = "birth:eaaef-idle-transaction-client"
        idle_token = server.issue_typed_client_grant(
            client_id=idle_client_id,
            process_birth_id=idle_birth_id,
            allowed_operations=("load_store_generation",),
            allowed_command_operations=("task.status.cas",),
            peer_pid=os.getpid(),
        )
        idle_connection = TypedStateOwnerConnection(
            socket_path=server.typed_command_socket_path(),
            token=idle_token,
            client_id=idle_client_id,
            process_birth_id=idle_birth_id,
            store_id=str(capability["store_id"]),
        )
        head = idle_connection.execute_operation(
            "load_store_generation"
        ).fetchone()
        assert head is not None
        idle_connection.prepare_command(
            StateCommand(
                command_id="command:eaaef-idle-stop",
                command_kind=CommandKind.CLAIM,
                store_id=str(capability["store_id"]),
                session_id=idle_connection.session_id,
                expected_generation=int(head[0]),
                expected_revision=int(head[3]),
                fence_epoch=int(head[2]),
                idempotency_key="idempotency:eaaef-idle-stop",
                parameters={
                    "operation": "task.status.cas",
                    "task_cid": "task:eaaef-idle-stop",
                    "expected_task_revision": 0,
                    "status": "claimed",
                },
            )
        )
        idle_connection.execute("BEGIN TRANSACTION")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                stopped = executor.submit(server.stop).result(timeout=10)
            assert stopped["stopped"] is True
        finally:
            idle_connection.close()
    finally:
        gateway.close()
        owner_connection.close()
        server.stop()
    with pytest.raises(
        owner_service.EAAEFTypedOwnerServiceError,
        match="closed",
    ):
        service.lookup_authorized_operation_receipt(retained_envelope)


def test_owner_lock_keeps_real_connection_transaction_concurrency_at_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, capability, context = _admission_bundle(
        tmp_path / "authority"
    )
    database = tmp_path / "operational.duckdb"
    _provision_operational(database, capability)
    import duckdb

    open_count = 0
    native_connect = duckdb.connect

    def counted_connect(*args: object, **kwargs: object) -> object:
        nonlocal open_count
        open_count += 1
        return native_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", counted_connect)
    raw = duckdb.connect(str(database))
    assert open_count == 1

    class GuardedOwnerConnection:
        def __init__(self, connection: object) -> None:
            self._connection = connection
            self._guard = threading.Lock()
            self._active = 0
            self.maximum_active = 0

        def execute(self, statement: str, *args: object) -> object:
            begins = statement.strip().upper() == "BEGIN TRANSACTION"
            if begins:
                with self._guard:
                    self._active += 1
                    self.maximum_active = max(
                        self.maximum_active, self._active
                    )
                time.sleep(0.01)
            try:
                return self._connection.execute(statement, *args)
            except BaseException:
                if begins:
                    with self._guard:
                        self._active -= 1
                raise

        def commit(self) -> None:
            try:
                self._connection.commit()
            finally:
                with self._guard:
                    self._active -= 1

        def rollback(self) -> None:
            try:
                self._connection.rollback()
            finally:
                with self._guard:
                    if self._active:
                        self._active -= 1

    guarded = GuardedOwnerConnection(raw)
    monkeypatch.setattr(
        owner_service.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )
    gateway = _owner_gateway(guarded, capability, tmp_path / "owner.sock")
    service = owner_service._bind_eaaef_typed_owner_command_service_from_gateway(  # noqa: SLF001
        owner_gateway=gateway,
        admission=admission,
    )
    envelope = _envelope(
        _intent(capability, limit=1),
        capability,
        context,
        serial=1,
    )
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = tuple(
                executor.map(
                    lambda _index: service.lookup_authorized_operation_receipt(
                        envelope
                    ),
                    range(16),
                )
            )
        assert results == (None,) * 16
        assert guarded.maximum_active == 1
        assert open_count == 1
    finally:
        service.close()
        raw.close()
