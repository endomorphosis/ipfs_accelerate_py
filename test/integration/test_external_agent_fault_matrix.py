"""EAAEF-144: source-level owner and recovery fail-closed contract.

This harness observes owner contention, restart fencing, typed recovery errors,
and retired dispatch rejection.  It does not claim to inject provider, prover,
container, DuckLake, or network failures.  The board receipt validator requires
case-specific observations for those production claims.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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

BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
SHARD_ID = "eaaef-144-disposable-fault-matrix-shard"
STORE_ID = "eaaef-144-control"
RECEIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/architecture/external_agent_autonomous_execution_fabric/receipts/fault_matrix.json"
)
REQUIRED_OBSERVED_FAULTS = frozenset(
    {
        "provider_outage",
        "prover_outage",
        "quack_owner_crash",
        "supervisor_crash",
        "worker_crash",
        "network_partition",
        "ducklake_outage",
        "duplicate_transaction",
        "conflict",
        "stale_root",
        "stale_plan",
        "resource_exhaustion",
        "no_progress",
    }
)


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


def _validate_current_receipt(payload: object) -> None:
    assert isinstance(payload, Mapping)
    assert payload.get("schema") == "qualification-receipt@1"
    assert payload.get("task_id") == "EAAEF-144"
    assert payload.get("evidence_mode") != "contract_fail_closed"
    assert "in_memory_ExternalQuackOwner" not in json.dumps(payload, sort_keys=True)

    cases = payload.get("cases")
    assert isinstance(cases, list), (
        "fault cases must be observation records, not hard-coded booleans"
    )
    observed_case_ids: set[str] = set()
    for case in cases:
        assert isinstance(case, Mapping)
        case_id = str(case.get("case_id") or "")
        assert case_id
        assert case.get("observed") is True
        assert str(case.get("outcome") or "")
        assert str(case.get("evidence_cid") or "")
        observed_case_ids.add(case_id)
    assert REQUIRED_OBSERVED_FAULTS <= observed_case_ids
    assert payload.get("observed_case_count") == len(cases)
    assert payload.get("live_runtime_invoked") is True
    assert payload.get("live_quack_invoked") is True
    assert payload.get("accepted_stale_write") is False
    assert payload.get("invented_authority") is False


def test_owner_contention_stale_fence_and_recovery_fail_closed(
    tmp_path: Path,
) -> None:
    first_server = _server(tmp_path)
    first_identity = first_server.start()
    try:
        first_owner = _owner(first_server)
        first = first_owner.lease()

        duplicate = _server(tmp_path)
        try:
            with pytest.raises(
                QuackStateServerOwnershipError,
                match="second state-owner refused",
            ):
                duplicate.start()
        finally:
            duplicate.stop()
        assert first_owner.assert_current(first) == first
    finally:
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
                principal_id=successor_identity.server_id,
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


def test_board_declared_qualification_receipt_is_current() -> None:
    assert RECEIPT_PATH.is_file(), f"EAAEF-144 board-declared receipt is missing: {RECEIPT_PATH}"
    _validate_current_receipt(json.loads(RECEIPT_PATH.read_text(encoding="utf-8")))
