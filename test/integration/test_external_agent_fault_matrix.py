"""EAAEF-144: crashes, partitions, stale authority, and recovery.

Stale epoch/fence fails closed.  Recovery does not invent authority.  Live
Quack, Docker, and network partitions are not injected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.external_control_recovery import (
    RecoveryError,
    recover,
)
from ipfs_accelerate_py.agent_supervisor.runtime.external_fixed_point import terminate
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    DuplicateOwnerError,
    ExternalQuackOwner,
    RemoteSqlRefusedError,
    StaleOwnerError,
    UnsignedEnvelopeError,
    issue_envelope,
)


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "fault_matrix.json"
)

OWNER_A = "owner:primary"
OWNER_B = "owner:failover"


def _put(key: str, status: str, idempotency_key: str) -> dict[str, object]:
    return issue_envelope(
        operation="put",
        key=key,
        value={"status": status},
        principal_id="principal:worker",
        idempotency_key=idempotency_key,
    )


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_stale_epoch_fence_partition_and_recovery_do_not_invent_authority() -> None:
    owner = ExternalQuackOwner(OWNER_A, shard_id="fault-matrix-shard")
    first = owner.lease()
    owner.apply(_put("task-1", "claimed", "idem-1"), owner_id=OWNER_A, epoch=first.epoch)

    with pytest.raises(DuplicateOwnerError, match="second owner"):
        owner.claim(OWNER_B, epoch=first.epoch)

    takeover = owner.failover(OWNER_B)
    with pytest.raises(StaleOwnerError, match="stale owner") as stale_err:
        owner.apply(
            _put("task-1", "hijacked", "idem-stale"),
            owner_id=OWNER_A,
            epoch=first.epoch,
        )
    assert stale_err.value.reason_code == "stale_owner"
    assert owner.get("task-1")["status"] == "claimed"

    owner.apply(
        _put("task-1", "running", "idem-2"),
        owner_id=OWNER_B,
        epoch=takeover.epoch,
    )
    assert owner.get("task-1")["status"] == "running"

    with pytest.raises(UnsignedEnvelopeError, match="missing"):
        owner.apply({}, owner_id=OWNER_B, epoch=takeover.epoch)
    with pytest.raises(RemoteSqlRefusedError):
        owner.remote_update_sql("UPDATE tasks SET status = 'forged'")

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

    payload = _write_receipt(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-overlay-receipt@1",
            "task_id": "EAAEF-144",
            "evidence_mode": "contract_fail_closed",
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "live_quack_invoked": False,
            "cases": cases,
            "accepted_stale_write": False,
            "invented_authority": False,
            "failover_epoch": takeover.epoch,
        }
    )
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert saved["accepted_stale_write"] is False
    assert payload["invented_authority"] is False
