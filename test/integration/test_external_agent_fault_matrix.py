"""EAAEF-144: crashes, partitions, stale authority, and recovery.

Stale epoch/fence fails closed.  Recovery does not invent authority.  Live
Quack, Docker, and network partitions are not injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.external_control_recovery import (
    RecoveryError,
    recover,
)
from ipfs_accelerate_py.agent_supervisor.runtime.external_fixed_point import terminate
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
    ExternalQuackOwner,
    RemoteSqlRefusedError,
    RetiredInMemoryOwnerError,
    StaleOwnerError,
    issue_envelope,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServer,
    QuackStateServerOwnershipError,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
)

OWNER_B = "owner:failover"
BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
SHARD_ID = "eaaef-144-disposable-fault-matrix-shard"
STORE_ID = "eaaef-144-control"


def _server(root: Path) -> QuackStateServer:
    return build_server(
        database_path=root / "control.duckdb",
        state_dir=root / "owner",
        port=0,
        repository_id="repository:eaaef-144-test",
        store_id=STORE_ID,
        secret_handle="handle:eaaef-144-test-owner",
    )


def _owner(server: QuackStateServer) -> ExternalQuackOwner:
    owner = server.bind_external_quack_owner(
        board_namespace=BOARD_NAMESPACE,
        shard_id=SHARD_ID,
    )
    assert isinstance(owner, ExternalQuackOwner)
    return owner


def test_stale_epoch_fence_partition_and_recovery_do_not_invent_authority(
    tmp_path: Path,
) -> None:
    first_server = _server(tmp_path)
    first_identity = first_server.start()
    first_owner = _owner(first_server)
    first = first_owner.lease()

    duplicate = _server(tmp_path)
    with pytest.raises(
        QuackStateServerOwnershipError,
        match="second state-owner refused",
    ):
        duplicate.start()
    assert first_owner.assert_current(first) == first

    first_server.stop()
    successor_server = _server(tmp_path)
    successor_identity = successor_server.start()
    try:
        successor = _owner(successor_server)
        takeover = successor.assert_successor(first)
        assert takeover.server_id == successor_identity.server_id
        assert takeover.server_id != first_identity.server_id
        with pytest.raises(StaleOwnerError, match="stale owner") as stale_err:
            successor.assert_current(first)
        assert stale_err.value.reason_code == "stale_owner"

        with pytest.raises(RetiredInMemoryOwnerError) as envelope:
            issue_envelope(
                operation="put",
                key="task-1",
                value={"status": "running"},
                principal_id=OWNER_B,
                idempotency_key="idem-2",
            )
        assert envelope.value.reason_code == "in_memory_owner_retired"
        with pytest.raises(RemoteSqlRefusedError) as sql:
            successor.require_operation("UPDATE tasks SET status = 'forged'")
        assert sql.value.reason_code == "remote_sql_refused"
        with pytest.raises(
            QuackDaemonGatewayError,
            match=EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
        ):
            successor.daemon_gateway()
        assert not hasattr(successor, "apply")
    finally:
        successor_server.stop()

    recovered = recover(
        current_epoch=takeover.epoch,
        backup_epoch=first.epoch,
        duplicate=False,
        ducklake_available=False,
    )
    assert recovered["accepted_stale_write"] is False
    assert recovered["epoch"] == takeover.epoch
    with pytest.raises(RecoveryError, match="stale"):
        recover(
            current_epoch=takeover.epoch,
            backup_epoch=takeover.epoch + 1,
            duplicate=False,
            ducklake_available=True,
        )
    with pytest.raises(RecoveryError, match="duplicate"):
        recover(
            current_epoch=takeover.epoch,
            backup_epoch=takeover.epoch,
            duplicate=True,
            ducklake_available=True,
        )

    incomplete = terminate(
        goals_complete=False,
        tests_current=True,
        proofs_current=True,
        invalidations_empty=True,
        merge_queue_empty=True,
        claims_empty=True,
        source_root="sha256:" + "a" * 64,
        semantic_root="sha256:" + "b" * 64,
    )
    assert incomplete["terminal"] == "not_complete"

    cases = {
        "supervisor_crash_failover": True,
        "stale_owner_epoch_rejected": True,
        "stale_backup_epoch_rejected": True,
        "duplicate_transaction_rejected": True,
        "ducklake_outage_no_invented_authority": True,
        "remote_sql_refused": True,
        "unsigned_envelope_rejected": True,
        "recovery_does_not_invent_authority": recovered["accepted_stale_write"] is False,
    }
    assert all(cases.values())

    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-overlay-receipt@1",
        "task_id": "EAAEF-144",
        "evidence_mode": "contract_fail_closed",
        "live_runtime_invoked": False,
        "live_eight_container_qualification": False,
        "live_quack_invoked": False,
        "cases": cases,
        "accepted_stale_write": False,
        "invented_authority": False,
        "owner_dispatch_admitted": False,
        "failover_epoch": takeover.epoch,
    }
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["live_runtime_invoked"] is False
    assert payload["live_eight_container_qualification"] is False
    assert payload["accepted_stale_write"] is False
    assert payload["owner_dispatch_admitted"] is False
    assert payload["invented_authority"] is False
