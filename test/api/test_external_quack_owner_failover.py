"""EAAEF-093: bind qualification to the real fenced Quack state owner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
    BoundedQuackTransport,
    ExternalQuackOwner,
    ExternalQuackOwnerNotReady,
    RemoteSqlRefusedError,
    RetiredInMemoryOwnerError,
    StaleOwnerError,
    issue_envelope,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    InProcessQuackTransport,
    QuackStateServer,
    QuackStateServerNotRunningError,
    QuackStateServerOwnershipError,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerProtocolError,
)

BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
SHARD_ID = "eaaef-093-disposable-shard"
STORE_ID = "eaaef-093-control"


def _server(root: Path) -> QuackStateServer:
    return build_server(
        database_path=root / "control.duckdb",
        state_dir=root / "owner",
        port=0,
        repository_id="repository:eaaef-093-test",
        store_id=STORE_ID,
        secret_handle="handle:eaaef-093-test-owner",
    )


def _owner(server: QuackStateServer) -> ExternalQuackOwner:
    owner = server.bind_external_quack_owner(
        board_namespace=BOARD_NAMESPACE,
        shard_id=SHARD_ID,
    )
    assert isinstance(owner, ExternalQuackOwner)
    return owner


def test_process_local_owner_and_transport_are_retired() -> None:
    with pytest.raises(TypeError, match="issued only by a READY QuackStateServer"):
        ExternalQuackOwner("owner-a", shard_id=SHARD_ID)
    with pytest.raises(RetiredInMemoryOwnerError) as envelope:
        issue_envelope(operation="put")
    assert envelope.value.reason_code == "in_memory_owner_retired"
    with pytest.raises(RetiredInMemoryOwnerError):
        BoundedQuackTransport()


def test_ready_real_owner_issues_resource_free_fail_closed_facade(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    with pytest.raises(QuackStateServerNotRunningError):
        server.bind_external_quack_owner(
            board_namespace=BOARD_NAMESPACE,
            shard_id=SHARD_ID,
        )

    identity = server.start()
    try:
        owner = _owner(server)
        lease = owner.lease()
        assert isinstance(server.transport, InProcessQuackTransport)
        assert owner.owner_id == identity.server_id
        assert owner.board_namespace == BOARD_NAMESPACE
        assert owner.epoch == identity.generation
        assert owner.fence == identity.fence_epoch
        assert owner.bound_port > 0
        assert owner.listen_uri == identity.listen_uri
        assert owner.operational_table_exposed is False
        assert owner.production_admitted is False
        assert owner.assert_current(lease) == lease
        assert not hasattr(owner, "database_path")
        assert not hasattr(owner, "connection")
        assert not hasattr(owner, "execute")
        assert not hasattr(owner, "execute_sql")

        evidence = owner.evidence()
        assert evidence["backing_owner_interface"] == "QuackStateServer@1"
        assert evidence["opens_database"] is False
        assert evidence["creates_dispatcher"] is False
        assert evidence["local_sidecar_writes"] is False
        assert evidence["direct_task_source"] is False
        assert evidence["arbitrary_sql_enabled"] is False
        assert evidence["production_admitted"] is False
        assert evidence["production_blockers"] == [
            EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER
        ]

        with pytest.raises(
            QuackDaemonGatewayError,
            match=EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
        ):
            owner.daemon_gateway()
    finally:
        server.stop()


def test_second_real_owner_is_refused_before_database_open(tmp_path: Path) -> None:
    first = _server(tmp_path)
    first_identity = first.start()
    try:
        owner = _owner(first)
        assert owner.lease().server_id == first_identity.server_id
        second = _server(tmp_path)
        with pytest.raises(
            QuackStateServerOwnershipError,
            match="second state-owner refused",
        ):
            second.start()
        assert second._connection is None  # noqa: SLF001 - exact owner boundary
        assert first.ready()["ready"] is True
        assert owner.assert_current(owner.lease()).server_id == first_identity.server_id
    finally:
        first.stop()


def test_restart_advances_generation_and_rejects_stale_lease_and_token(
    tmp_path: Path,
) -> None:
    first = _server(tmp_path)
    first_identity = first.start()
    first_owner = _owner(first)
    stale = first_owner.lease()
    client_id = "eaaef-093-prior-generation-client"
    process_birth_id = "birth:eaaef-093-prior-generation-client"
    stale_token = first.issue_typed_client_grant(
        client_id=client_id,
        process_birth_id=process_birth_id,
        allowed_operations=("load_store_generation",),
        peer_pid=os.getpid(),
    )
    first.stop()
    with pytest.raises(ExternalQuackOwnerNotReady):
        first_owner.lease()

    second = _server(tmp_path)
    second_identity = second.start()
    try:
        successor = _owner(second)
        current = successor.assert_successor(stale)
        assert current.generation > first_identity.generation
        assert current.fence_epoch > first_identity.fence_epoch
        assert current.server_id == second_identity.server_id
        assert current.server_id != stale.server_id
        with pytest.raises(StaleOwnerError, match="stale owner") as rejected:
            successor.assert_current(stale)
        assert rejected.value.reason_code == "stale_owner"

        # The restarted gateway owns a fresh in-memory grant set.  A token
        # minted by the prior server generation is rejected during handshake;
        # authentication failures deliberately close the channel silently.
        with pytest.raises(TypedStateOwnerProtocolError, match="channel closed"):
            TypedStateOwnerConnection(
                socket_path=second.typed_command_socket_path(),
                token=stale_token,
                client_id=client_id,
                process_birth_id=process_birth_id,
                store_id=STORE_ID,
            )
    finally:
        second.stop()


def test_owner_gateway_rejects_sql_and_all_unqualified_dispatch(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    server.start()
    try:
        owner = _owner(server)
        for operation in (
            "sql.execute",
            "execute_sql",
            "remote_update_sql",
            "UPDATE tasks SET status='claimed'",
        ):
            with pytest.raises(RemoteSqlRefusedError) as rejected:
                owner.require_operation(operation)
            assert rejected.value.reason_code == "remote_sql_refused"

        with pytest.raises(
            QuackDaemonGatewayError,
            match=EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
        ):
            owner.require_operation("task.ready")
        with pytest.raises(
            QuackDaemonGatewayError,
            match=EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
        ):
            owner.daemon_gateway()
    finally:
        server.stop()
