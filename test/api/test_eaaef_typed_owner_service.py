from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_bootstrap_gateway as runtime,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    eaaef_typed_owner_service as owner_service,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    AuthorizedStateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandFabric,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
    quack_daemon_operation_intent,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
)
from test.api.test_eaaef_bootstrap_gateway_launch import (
    NOW_MS,
    _client,
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
    service = owner_service.bind_eaaef_typed_owner_command_service(
        connection=connection,
        transaction_lock=threading.Lock(),
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


def test_typed_transport_drives_gateway_without_database_or_quack_client_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, connection, admission, capability, context = _bind_service(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        runtime.time,
        "time_ns",
        lambda: NOW_MS * 1_000_000,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("typed EAAEF transport attempted a legacy client or database open")

    monkeypatch.setattr(runtime.QuackCommandClient, "append", forbidden)
    monkeypatch.setattr(runtime.QuackReadClient, "list_recent_receipts", forbidden)
    monkeypatch.setattr(QuackCommandFabric, "__init__", forbidden)
    import duckdb

    monkeypatch.setattr(duckdb, "connect", forbidden)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "open_duckdb_connection",
        forbidden,
    )

    owner_connection = object.__new__(TypedStateOwnerConnection)

    def owner_request(action: str, **fields: object) -> dict[str, object]:
        assert fields["merge_admission_cid"] == admission["merge_admission_cid"]
        assert (
            fields["operational_capability_cid"]
            == admission["operational_capability_cid"]
        )
        envelope = AuthorizedStateCommand.from_dict(fields["envelope"])
        if action == owner_service.EAAEF_TYPED_OWNER_COMMAND_SUBMIT_OPERATION:
            receipt = service.submit_authorized_operation(envelope)
        elif action == owner_service.EAAEF_TYPED_OWNER_COMMAND_LOOKUP_OPERATION:
            receipt = service.lookup_authorized_operation_receipt(envelope)
        else:
            raise AssertionError("unexpected typed-owner service operation")
        return {"receipt": receipt}

    owner_connection._request = owner_request  # type: ignore[method-assign]  # noqa: SLF001
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
    try:
        assert gateway.evidence()["transport"] == "typed_state_owner"
        assert (
            owner_service.EAAEF_TYPED_OWNER_TRANSPORT_PRODUCTION_BLOCKER
            in gateway.evidence()["production_blockers"]
        )
        with pytest.raises(QuackDaemonGatewayError, match="production no-go"):
            gateway.require_production_admission()
        gateway.attach()
        page = gateway.task_source.ready_tasks(limit=2)
        assert [row.task_cid for row in page.tasks] == ["task:eaaef:1"]
        assert transport.lookup_receipt(
            _envelope(
                _intent(capability, limit=2),
                capability,
                context,
                serial=1,
            )
        ) is not None
    finally:
        gateway.close()
        service.close()
        connection.close()


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
    service = owner_service.bind_eaaef_typed_owner_command_service(
        connection=guarded,
        transaction_lock=threading.Lock(),
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
