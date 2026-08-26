"""EAAEF-097: DuckDB/Quack owner is sole authority; DuckLake lag never is."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.external_control_recovery import (
    RecoveryError,
    recover,
)
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
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
REQUIRED_PRODUCTION_OBSERVATIONS = frozenset(
    {
        "sole_owner_command_apply",
        "second_owner_refusal",
        "stale_fence_rejection",
        "idempotent_retries",
        "ducklake_authority_separation",
    }
)


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


def _fixture_cid(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _is_evidence_cid(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _validate_current_receipt(payload: object) -> None:
    assert isinstance(payload, Mapping)
    assert payload.get("schema") == "qualification-receipt@1"
    assert payload.get("task_id") == "EAAEF-097"

    encoded = json.dumps(payload, sort_keys=True)
    assert "in_memory_ExternalQuackOwner" not in encoded
    assert "sole_owner_applies_envelopes" not in encoded
    assert "sole_owner_apply" not in encoded
    assert payload.get("evidence_mode") == "observed_live"
    assert payload.get("live_runtime_invoked") is True
    assert payload.get("live_quack_contacted") is True

    owner_evidence = payload.get("owner_evidence")
    assert isinstance(owner_evidence, Mapping)
    assert owner_evidence.get("interface") == "ExternalQuackOwner@1"
    assert owner_evidence.get("backing_owner_interface") == "QuackStateServer@1"
    assert str(owner_evidence.get("server_id") or "").startswith("server:")
    assert _is_evidence_cid(owner_evidence.get("lease_cid"))
    assert owner_evidence.get("production_admitted") is True
    assert list(owner_evidence.get("production_blockers") or ()) == []

    observations = payload.get("observations")
    assert isinstance(observations, list)
    observed_operations: set[str] = set()
    for observation in observations:
        assert isinstance(observation, Mapping)
        operation = str(observation.get("operation") or "")
        assert operation
        assert observation.get("observed") is True
        assert observation.get("outcome") == "passed"
        assert _is_evidence_cid(observation.get("evidence_cid"))
        observed_operations.add(operation)
    assert len(observations) == len(REQUIRED_PRODUCTION_OBSERVATIONS)
    assert observed_operations == REQUIRED_PRODUCTION_OBSERVATIONS

    assert payload.get("observed_server_id") == owner_evidence["server_id"]
    assert payload.get("owner_dispatch_admitted") is True
    assert list(payload.get("production_blockers") or ()) == []
    assert payload.get("terminal") == "completed"
    assert payload.get("qualification_status") == "qualified"
    assert payload.get("result") == "pass"


def _truthful_production_receipt_fixture() -> dict[str, object]:
    server_id = "server:observed-production-owner"
    return {
        "schema": "qualification-receipt@1",
        "task_id": "EAAEF-097",
        "evidence_mode": "observed_live",
        "live_runtime_invoked": True,
        "live_quack_contacted": True,
        "observed_server_id": server_id,
        "owner_dispatch_admitted": True,
        "production_blockers": [],
        "terminal": "completed",
        "qualification_status": "qualified",
        "result": "pass",
        "owner_evidence": {
            "interface": "ExternalQuackOwner@1",
            "backing_owner_interface": "QuackStateServer@1",
            "server_id": server_id,
            "lease_cid": _fixture_cid("eaaef-097-production-owner-lease"),
            "production_admitted": True,
            "production_blockers": [],
        },
        "observations": [
            {
                "operation": operation,
                "observed": True,
                "outcome": "passed",
                "evidence_cid": _fixture_cid(f"eaaef-097:{operation}"),
            }
            for operation in sorted(REQUIRED_PRODUCTION_OBSERVATIONS)
        ],
    }


def test_receipt_validator_accepts_only_observed_production_fixture() -> None:
    truthful = _truthful_production_receipt_fixture()
    _validate_current_receipt(truthful)

    legacy = json.loads(json.dumps(truthful))
    legacy["owner"] = "in_memory_ExternalQuackOwner"
    with pytest.raises(AssertionError):
        _validate_current_receipt(legacy)

    assertion_only = json.loads(json.dumps(truthful))
    assertion_only["observations"] = ["sole_owner_applies_envelopes"]
    with pytest.raises(AssertionError):
        _validate_current_receipt(assertion_only)


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

        duplicate = _server(tmp_path)
        try:
            with pytest.raises(
                QuackStateServerOwnershipError,
                match="second state-owner refused",
            ):
                duplicate.start()
        finally:
            duplicate.stop()
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
    try:
        first_owner = _owner(first)
        stale = first_owner.lease()
    finally:
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
    try:
        owner = _owner(server)
        lease = owner.lease()
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


def test_board_declared_qualification_receipt_is_current() -> None:
    assert RECEIPT_PATH.is_file(), f"EAAEF-097 board-declared receipt is missing: {RECEIPT_PATH}"
    _validate_current_receipt(json.loads(RECEIPT_PATH.read_text(encoding="utf-8")))
