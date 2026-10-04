"""Shared hash observations use the existing real owner socket and DuckDB."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    ControlPlaneMigrationRunner,
    MigrationCatalog,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_datasets_authoritative_operational_schema,
    load_datasets_authoritative_operational_catalog,
    verify_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.hash_observations import (
    IDENTITY_PROFILE,
    IDENTITY_SCHEMA,
    identity_key,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    HASH_OBSERVATION_SERVICE_OPERATION,
    SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS,
    TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TypedStateOwnerConnection,
    TypedStateOwnerRemoteError,
    kernel_process_birth_id,
    request_hash_observation_credential,
)
from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from test.api.test_agent_supervisor_quack_state_server import _real_database_server


def _request(action: str = "claim", **fields: object) -> dict:
    identity = {
        "schema": IDENTITY_SCHEMA,
        "algorithm": "sha256",
        "profile": IDENTITY_PROFILE,
        "host_boot_id": "boot-test",
        "mount_id": "mount-test",
        "dev": 1,
        "ino": 2,
        "mode": 0o100644,
        "uid": 1000,
        "gid": 1000,
        "nlink": 1,
        "size": 10,
        "mtime_ns": 123,
        "ctime_ns": 123,
    }
    return {"action": action, "key": identity_key(identity),
            "identity": identity, **fields}


@pytest.fixture
def owner(tmp_path: Path, request):
    database = tmp_path / "control.duckdb"
    _install(database)
    if getattr(request, "param", "") == "legacy":
        with open_duckdb_connection(database) as legacy:
            legacy.execute("DROP TABLE hash_observations")
    gateway, connection = _gateway(database, tmp_path / "owner.sock")
    clients = []

    def connect(client_id: str, *, operations=(HASH_OBSERVATION_SERVICE_OPERATION,)):
        birth = kernel_process_birth_id()
        token, grant = gateway.issue_grant(
            client_id=client_id,
            process_birth_id=birth,
            peer_pid=os.getpid(),
            allowed_operations=operations,
        )
        client = TypedStateOwnerConnection(
            socket_path=gateway.socket_path,
            token=token,
            client_id=client_id,
            process_birth_id=birth,
            store_id=gateway.store_id,
        )
        clients.append(client)
        return client, grant

    yield gateway, connection, connect
    for client in clients:
        client.close()
    gateway.stop()
    connection.close()


def test_two_supervisor_sessions_share_claim_and_completed_observation(owner) -> None:
    gateway, connection, connect = owner
    first, _ = connect("hash-supervisor-a")
    second, _ = connect("hash-supervisor-b")
    claim = first.hash_observation(_request())
    assert claim["status"] == "claimed"
    assert second.hash_observation(_request())["status"] == "busy"
    completion = _request(
        "complete", sha256="a" * 64,
        **{key: claim[key] for key in ("generation", "lease_token", "fence")},
    )
    with pytest.raises(TypedStateOwnerRemoteError):
        second.hash_observation(completion)
    assert first.hash_observation(completion)["status"] == "hit"
    assert second.hash_observation(_request())["sha256"] == "a" * 64
    assert connection.execute("SELECT count(*) FROM hash_observations").fetchone()[0] == 1


def test_hash_service_does_not_widen_status_event_or_sql_authority(owner) -> None:
    _, connection, connect = owner
    assert HASH_OBSERVATION_SERVICE_OPERATION not in SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS
    read_client, _ = connect("hash-read-client", operations=("whoami_metadata",))
    with pytest.raises(TypedStateOwnerRemoteError) as denied:
        read_client.hash_observation(_request())
    assert denied.value.error_code == "authorization_denied"
    hash_client, _ = connect("hash-only-client")
    with pytest.raises(TypedStateOwnerRemoteError):
        hash_client.hash_observation(_request(sql="DELETE FROM tasks"))
    with pytest.raises(TypedStateOwnerRemoteError):
        hash_client._request("execute", operation="txn_cas_task_status", parameters=[])
    assert connection.execute("SELECT count(*) FROM hash_observations").fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM tasks").fetchone()[0] == 1


def test_revoked_hash_grant_cannot_publish_pending_claim(owner) -> None:
    gateway, connection, connect = owner
    client, grant = connect("hash-revoked-client")
    claim = client.hash_observation(_request())
    gateway.revoke_grant(grant.grant_id)
    with pytest.raises(TypedStateOwnerRemoteError) as denied:
        client.hash_observation(_request(
            "complete", sha256="b" * 64,
            **{key: claim[key] for key in ("generation", "lease_token", "fence")},
        ))
    assert denied.value.error_code == "authorization_denied"
    assert connection.execute("SELECT state FROM hash_observations").fetchone()[0] == "claimed"


@pytest.mark.parametrize("owner", ["legacy"], indirect=True)
def test_unmigrated_owner_refuses_cache_without_creating_tables(owner) -> None:
    _, connection, connect = owner
    client, _ = connect("hash-unmigrated-client")
    with pytest.raises(TypedStateOwnerRemoteError) as unavailable:
        client.hash_observation(_request())
    assert unavailable.value.error_code == "protocol_denied"
    assert connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'hash_observations'"
    ).fetchone()[0] == 0


def test_broker_mints_only_hash_service_grant(tmp_path: Path, monkeypatch) -> None:
    server = _real_database_server(tmp_path)
    client = None
    try:
        server.start()
        handoff = server.start_supervisor_grant_broker()
        for name in (TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
                     TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
                     TYPED_STATE_OWNER_SOCKET_ENV):
            monkeypatch.setenv(name, handoff[name])
        birth = kernel_process_birth_id()
        token = request_hash_observation_credential(
            store_id=server.config.store_id,
            client_id="hash-broker-client",
            process_birth_id=birth,
        )
        client = TypedStateOwnerConnection(
            socket_path=server.typed_command_socket_path(),
            token=token,
            client_id="hash-broker-client",
            process_birth_id=birth,
            store_id=server.config.store_id,
        )
        assert client.grant["allowed_operations"] == [HASH_OBSERVATION_SERVICE_OPERATION]
        assert client.grant["allowed_command_operations"] == []
        assert client.grant["allowed_database_task_commands"] == []
        assert client.hash_observation(_request())["status"] == "claimed"
    finally:
        if client is not None:
            client.close()
        server.stop()


def test_operational_profile_upgrade_preserves_base_receipt(tmp_path: Path) -> None:
    database = tmp_path / "operational.duckdb"
    catalog = load_datasets_authoritative_operational_catalog()
    original = catalog.get(1)
    runner = ControlPlaneMigrationRunner.for_database(
        database,
        catalog=MigrationCatalog.from_migrations((original,)),
        application_version="test", tool_version="test", owner_id="hash-upgrade-test",
    )
    first = runner.apply()
    upgraded = install_datasets_authoritative_operational_schema(
        database,
        application_version="test", tool_version="test", owner_id="hash-upgrade-test",
    )
    assert first.to_version == 1
    assert upgraded.from_version == 1 and upgraded.to_version == 2
    verified = verify_datasets_authoritative_operational_schema(database)
    assert verified["migration_checksum"] == original.checksum
    assert verified["extension_tables_ok"] == ["hash_observations"]
