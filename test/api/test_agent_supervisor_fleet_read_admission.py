"""Independent aggregate readers cannot acquire observation or board writes."""

import os

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerError,
)


def test_fleet_read_bootstrap_is_peer_bound_readonly_and_revocable(tmp_path):
    db = tmp_path / "control.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    token = gateway.configure_fleet_observation_reads()
    client = TypedStateOwnerConnection(
        socket_path=gateway.socket_path,
        token=token,
        client_id="fleet-reader",
        process_birth_id="birth:fleet-reader",
        store_id="control.duckdb",
        fleet_observation_read=True,
    )
    try:
        assert client.grant["peer_pid"] == os.getpid()
        assert not client.grant["allowed_command_operations"]
        assert client.grant["expires_at"] - client.grant["issued_at"] == 120000
        assert (
            client.execute_operation(
                "fleet_select_source_observation", ["fleet-source:spar"]
            ).fetchall()
            == []
        )
        for operation in ["fleet_insert_observation_artifact", "txn_cas_task_status"]:
            with pytest.raises(TypedStateOwnerError):
                client.execute_operation(
                    operation, [None] * gateway.catalog[operation].parameter_count
                )
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=gateway.socket_path,
                token=token,
                client_id="fleet-reader",
                process_birth_id="birth:fleet-reader",
                store_id="control.duckdb",
                derived_repository_id="repo:one",
            )
        gateway.revoke_grant(client.grant["grant_id"])
        with pytest.raises(TypedStateOwnerError):
            client.execute_operation(
                "fleet_select_source_observation", ["fleet-source:spar"]
            )
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_fleet_bootstrap_requires_distinct_credential_and_single_admission_mode(
    tmp_path,
):
    db = tmp_path / "control.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    status = gateway.configure_status_bootstrap()
    gateway.configure_fleet_observation_reads()
    try:
        for status_bootstrap in [False, True]:
            with pytest.raises(TypedStateOwnerError):
                TypedStateOwnerConnection(
                    socket_path=gateway.socket_path,
                    token=status,
                    client_id="fleet-reader",
                    process_birth_id="birth:fleet-reader",
                    store_id="control.duckdb",
                    fleet_observation_read=True,
                    status_bootstrap=status_bootstrap,
                )
    finally:
        gateway.stop()
        connection.close()
