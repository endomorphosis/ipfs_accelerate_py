"""EAAEF-097: DuckDB/Quack owner is sole authority; DuckLake lag never is."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.external_control_recovery import (
    RecoveryError,
    recover,
)
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
    LIVE_QUACK_PORT,
    ExternalQuackOwner,
    ExternalQuackOwnerNotReady,
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KIT_ROOT = _REPO_ROOT / "ipfs_kit_py"
if str(_KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIT_ROOT))

RECEIPT_PATH = (
    _REPO_ROOT
    / "docs/architecture/external_agent_autonomous_execution_fabric/receipts/control_plane.json"
)

BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
SHARD_ID = "eaaef-097-disposable-test-shard"
STORE_ID = "eaaef-097-control"


def _server(root: Path) -> QuackStateServer:
    return build_server(
        database_path=root / "control.duckdb",
        state_dir=root / "owner",
        port=0,
        repository_id="repository:eaaef-097-test",
        store_id=STORE_ID,
        secret_handle="handle:eaaef-097-test-owner",
    )


def _owner(server: QuackStateServer) -> ExternalQuackOwner:
    owner = server.bind_external_quack_owner(
        board_namespace=BOARD_NAMESPACE,
        shard_id=SHARD_ID,
    )
    assert isinstance(owner, ExternalQuackOwner)
    return owner


def test_sole_ready_owner_issues_fail_closed_facade(tmp_path: Path) -> None:
    server = _server(tmp_path)
    identity = server.start()
    try:
        owner = _owner(server)
        lease = owner.lease()
        assert lease.owner_id == identity.server_id
        assert lease.generation == identity.generation
        assert lease.fence_epoch == identity.fence_epoch
        assert owner.assert_current(lease) == lease
        assert owner.bound_port > 0
        assert owner.operational_table_exposed is False
        assert owner.production_admitted is False
        assert LIVE_QUACK_PORT == 19495

        duplicate = _server(tmp_path)
        with pytest.raises(
            QuackStateServerOwnershipError,
            match="second state-owner refused",
        ):
            duplicate.start()
        assert owner.assert_current(lease) == lease

        with pytest.raises(RemoteSqlRefusedError) as sql:
            owner.require_operation("UPDATE tasks SET status = 'hijacked'")
        assert sql.value.reason_code == "remote_sql_refused"
        with pytest.raises(
            QuackDaemonGatewayError,
            match=EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
        ):
            owner.daemon_gateway()
    finally:
        server.stop()


def test_stale_fence_epoch_fails(tmp_path: Path) -> None:
    first = _server(tmp_path)
    first_identity = first.start()
    first_owner = _owner(first)
    stale = first_owner.lease()
    first.stop()
    with pytest.raises(ExternalQuackOwnerNotReady):
        first_owner.lease()

    successor_server = _server(tmp_path)
    successor_identity = successor_server.start()
    try:
        successor = _owner(successor_server)
        current = successor.assert_successor(stale)
        assert current.server_id == successor_identity.server_id
        assert current.server_id != first_identity.server_id
        assert current.generation > stale.generation
        assert current.fence_epoch > stale.fence_epoch
        with pytest.raises(StaleOwnerError, match="stale owner") as rejected:
            successor.assert_current(stale)
        assert rejected.value.reason_code == "stale_owner"
    finally:
        successor_server.stop()


def test_retired_envelope_retries_cannot_invent_success(tmp_path: Path) -> None:
    server = _server(tmp_path)
    server.start()
    try:
        owner = _owner(server)
        before = owner.lease()
        for _ in range(2):
            with pytest.raises(RetiredInMemoryOwnerError) as rejected:
                issue_envelope(
                    operation="put",
                    key="task-1",
                    value={"status": "claimed"},
                    principal_id="principal:worker",
                    idempotency_key="idem-1",
                )
            assert rejected.value.reason_code == "in_memory_owner_retired"
        assert owner.assert_current(before) == before
        assert not hasattr(owner, "apply")
        assert not hasattr(owner, "get")
    finally:
        server.stop()


def test_ducklake_loss_and_lag_never_change_authority(tmp_path: Path) -> None:
    server = _server(tmp_path)
    server.start()
    owner = _owner(server)
    lease = owner.lease()

    try:
        recovered = recover(
            current_epoch=lease.epoch,
            backup_epoch=lease.epoch,
            duplicate=False,
            ducklake_available=False,
        )
        assert recovered["accepted_stale_write"] is False
        assert recovered["ducklake_available"] is False
        with pytest.raises(RecoveryError):
            recover(
                current_epoch=lease.epoch,
                backup_epoch=lease.epoch + 8,
                duplicate=False,
                ducklake_available=True,
            )

        from ipfs_datasets_py.ducklake.external_agent_history import (
            HistoryAuthorityError,
            HistoryCursor,
            project_outbox,
        )

        cursor = HistoryCursor(
            outbox_ordinal=1,
            owner_epoch=lease.epoch,
            fence=lease.fence,
            source_digest="sha256:" + ("c" * 64),
            owner_id=lease.owner_id,
            shard_id=lease.shard_id,
        )
        epoch = project_outbox(
            cursor,
            (
                {
                    "kind": "event",
                    "event_id": "evt-1",
                    "run_id": "run-1",
                    "task_id": "EAAEF-097",
                    "payload": {"status": "projected"},
                },
            ),
            epoch_id="epoch-lag",
        )
        assert epoch.grants_current_authority is False
        with pytest.raises(HistoryAuthorityError):
            epoch.grant_claim("task-1")
        with pytest.raises(HistoryAuthorityError):
            epoch.grant_lease("task-1")
        with pytest.raises(HistoryAuthorityError):
            epoch.grant_fence("task-1")
        with pytest.raises(HistoryAuthorityError):
            epoch.grant_merge_authority("task-1")

        from ipfs_kit_py.external_agent_history.publication import authority_from_lag

        assert authority_from_lag({"behind_ordinals": 99, "available": False}) is False
        assert owner.assert_current(lease) == lease
        evidence = owner.evidence()
        assert evidence["local_sidecar_writes"] is False
        assert evidence["production_admitted"] is False
        assert owner.bound_port > 0
    finally:
        server.stop()


def test_qualification_receipt_is_contract_fail_closed() -> None:
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "qualification-receipt@1"
    assert payload.get("task_id") == "EAAEF-097" or payload.get("task_alias") == "EAAEF-097"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["live_runtime_invoked"] is False
    assert payload.get("live_quack_contacted", False) is False
    assert int(payload.get("live_quack_port", LIVE_QUACK_PORT)) == 19495
