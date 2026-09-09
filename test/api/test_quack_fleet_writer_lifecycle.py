"""Actual native grants remain bounded across polling, renewal and reconnect."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_observer import (
    FleetObserver,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServer,
    ServerLifecycle,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
)
from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from test.api.test_quack_fleet_observation import sample


@pytest.fixture
def native(tmp_path):
    database = tmp_path / "control.duckdb"
    _install(database)
    gateway, connection = _gateway(database, tmp_path / "owner.sock")
    # Exercise real server issuance/renewal/revocation against the live native
    # typed gateway without requiring a second TCP listener in the test.
    server = object.__new__(QuackStateServer)
    server._lock = threading.RLock()
    server._lifecycle = ServerLifecycle.READY
    server._command_gateway = gateway
    server._identity = SimpleNamespace(
        **gateway.identity, listen_uri="quack:127.0.0.1:7777"
    )
    server.config = SimpleNamespace(
        typed_command_socket_path_override=gateway.socket_path
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py/taskboard-fleet-inventory@1",
                "boards": [{"id": "spar"}],
            }
        )
    )
    observer = FleetObserver(server, inventory, tmp_path / "view.json")
    try:
        yield observer, server, gateway, connection
    finally:
        observer._retire_writer()
        gateway.stop()
        connection.close()


def test_native_completed_poll_batches_reuse_one_grant_and_connection(native):
    observer, _, gateway, connection = native
    observer._polls = SimpleNamespace(step=lambda *_: [sample()], close=lambda: None)
    clients = []
    for _ in range(25):
        assert observer.cycle()["sources"]["spar"]["available"]
        clients.append(observer._writer_client)
    assert len({id(client) for client in clients}) == 1
    assert len(gateway._grants) == 1
    assert not gateway._revoked_grants
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind='fleet_source_observation'"
        ).fetchone()[0]
        == 25
    )
    # This observer issues diagnostic writes only, never task mutation.
    assert connection.execute("SELECT status FROM tasks").fetchone()[0] == "ready"


def test_due_grant_uses_native_renewal_without_new_capability(native, monkeypatch):
    observer, server, gateway, _ = native
    client = observer._client()
    grant = observer._writer_grant
    token = observer._writer_token
    observer._writer_grant = replace(grant, expires_at=int(time.time() * 1000) + 50_000)
    calls = []
    actual = server.renew_typed_client_grant

    def renew(grant_id, **kwargs):
        calls.append(grant_id)
        return actual(grant_id, **kwargs)

    monkeypatch.setattr(server, "renew_typed_client_grant", renew)
    assert observer._client() is client
    assert observer._client() is client
    assert calls == [grant.grant_id]
    assert observer._writer_token == token and len(gateway._grants) == 1
    assert observer._writer_grant.allowed_operations == grant.allowed_operations
    assert (
        observer._writer_grant.allowed_command_operations
        == grant.allowed_command_operations
    )
    assert observer._writer_grant.peer_start_time_ticks == grant.peer_start_time_ticks
    assert observer._writer_grant.expires_at > int(time.time() * 1000) + 500_000


def test_expired_local_grant_retires_and_reconnects_once(native):
    observer, _, gateway, _ = native
    old = observer._client()
    grant_id = observer._writer_grant.grant_id
    expired_at = int(time.time() * 1000) - 1
    observer._writer_grant = replace(
        observer._writer_grant, issued_at=expired_at - 600_000, expires_at=expired_at
    )
    new = observer._client()
    assert new is not old and not old.attached
    assert observer._writer_grant.grant_id != grant_id
    assert len(gateway._grants) == 1
    assert gateway._revoked_grants == {grant_id}
    assert observer._client() is new


def test_native_revocation_is_not_hidden_by_cached_writer(native):
    observer, server, gateway, _ = native
    observer._client()
    grant_id = observer._writer_grant.grant_id
    server.revoke_typed_client_grant(grant_id)
    observer._close_writer_client()
    # Reconnect verifies the cached grant with the native owner. One failure
    # ends this cycle; the subsequent cycle may request one new exact grant.
    with pytest.raises(Exception, match="cannot be renewed"):
        observer._client()
    assert observer._writer_client is None and observer._writer_grant is None
    assert len(gateway._grants) == 0
    assert observer._client().attached
    assert len(gateway._grants) == 1


def test_transport_reconnect_reuses_valid_grant_after_failed_attach(
    native, monkeypatch
):
    observer, _, gateway, _ = native
    actual = QuackStateClient.attach
    attempts = []

    def attach(client, *args, **kwargs):
        attempts.append(client)
        if len(attempts) == 1:
            raise RuntimeError("bounded simulated connection failure")
        return actual(client, *args, **kwargs)

    monkeypatch.setattr(QuackStateClient, "attach", attach)
    with pytest.raises(RuntimeError, match="connection failure"):
        observer._client()
    original_grant = observer._writer_grant.grant_id
    assert observer._writer_client is None and len(gateway._grants) == 1
    assert observer._client().attached
    assert observer._writer_grant.grant_id == original_grant
    assert len(gateway._grants) == 1 and not gateway._revoked_grants
    assert not attempts[0].attached


def test_owner_binding_change_never_returns_cached_connection(native, monkeypatch):
    observer, server, _, _ = native
    old = observer._client()
    identity = server._identity
    server._identity = SimpleNamespace(
        **{**vars(identity), "generation": identity.generation + 1}
    )

    def refused(**_):
        raise RuntimeError("replacement gateway must admit its own grant")

    monkeypatch.setattr(server, "issue_typed_client_grant_record", refused)
    with pytest.raises(RuntimeError, match="replacement gateway"):
        observer._client()
    assert not old.attached
    assert observer._writer_client is None and observer._writer_grant is None


def test_failed_cycle_closes_connection_and_later_cycle_reuses_grant(
    native, monkeypatch
):
    from ipfs_accelerate_py.agent_supervisor.federation.fleet_observation import (
        FleetObservationStore,
    )

    observer, _, gateway, _ = native
    observer._polls = SimpleNamespace(step=lambda *_: [sample()], close=lambda: None)
    observer.cycle()
    grant_id = observer._writer_grant.grant_id
    old = observer._writer_client
    with monkeypatch.context() as patch:
        patch.setattr(
            FleetObservationStore,
            "record",
            lambda *args: (_ for _ in ()).throw(RuntimeError("write failed")),
        )
        with pytest.raises(RuntimeError, match="write failed"):
            observer.cycle()
    assert not old.attached and observer._writer_client is None
    assert observer.cycle()["sources"]["spar"]["available"]
    assert observer._writer_grant.grant_id == grant_id
    assert len(gateway._grants) == 1


def test_observer_exit_closes_client_and_retires_its_one_grant(native):
    observer, _, gateway, _ = native
    client = observer._client()
    observer.stop_event.set()
    observer._run()
    assert not client.attached
    assert observer._writer_client is None and observer._writer_grant is None
    assert len(gateway._grants) == 0
    assert len(gateway._revoked_grants) == 1
