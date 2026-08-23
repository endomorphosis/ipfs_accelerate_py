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
    INITIAL_EPOCH,
    LIVE_QUACK_PORT,
    DuplicateOwnerError,
    ExternalQuackOwner,
    RemoteSqlRefusedError,
    StaleOwnerError,
    issue_envelope,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_KIT_ROOT = _REPO_ROOT / "ipfs_kit_py"
if str(_KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIT_ROOT))

RECEIPT_PATH = (
    _REPO_ROOT
    / "docs/architecture/external_agent_autonomous_execution_fabric/receipts/control_plane.json"
)

OWNER_A = "owner-a"
OWNER_B = "owner-b"


def _owner() -> ExternalQuackOwner:
    return ExternalQuackOwner(OWNER_A, shard_id="disposable-test-shard")


def _put(
    key: str = "task-1",
    *,
    status: str = "claimed",
    idempotency_key: str = "idem-1",
) -> dict[str, object]:
    return issue_envelope(
        operation="put",
        key=key,
        value={"status": status},
        principal_id="principal:worker",
        idempotency_key=idempotency_key,
    )


def test_sole_owner_applies_envelopes() -> None:
    owner = _owner()
    lease = owner.lease()
    assert owner.epoch == INITIAL_EPOCH == 1
    assert lease.owner_id == OWNER_A
    receipt = owner.apply(_put(), owner_id=OWNER_A, epoch=lease.epoch)
    assert receipt["status"] == "applied"
    assert owner.get("task-1")["status"] == "claimed"
    assert owner.bound_port is None
    assert owner.transport.bound_port is None
    assert LIVE_QUACK_PORT == 19495
    with pytest.raises(DuplicateOwnerError):
        owner.claim(OWNER_B, epoch=1)
    with pytest.raises(RemoteSqlRefusedError):
        owner.remote_update_sql("UPDATE tasks SET status = 'hijacked'")


def test_stale_fence_epoch_fails() -> None:
    owner = _owner()
    stale = owner.lease()
    owner.apply(_put(), owner_id=stale.owner_id, epoch=stale.epoch)
    owner.failover(OWNER_B)
    with pytest.raises(StaleOwnerError, match="stale owner"):
        owner.apply(
            _put(status="running", idempotency_key="idem-stale"),
            owner_id=stale.owner_id,
            epoch=stale.epoch,
        )
    assert owner.get("task-1")["status"] == "claimed"
    owner.apply(
        _put(status="running", idempotency_key="idem-2"),
        owner_id=OWNER_B,
        epoch=2,
    )
    assert owner.get("task-1")["status"] == "running"


def test_retries_are_idempotent() -> None:
    owner = _owner()
    envelope = _put()
    first = owner.apply(envelope, owner_id=OWNER_A, epoch=1)
    replay = owner.apply(envelope, owner_id=OWNER_A, epoch=1)
    assert dict(first) == dict(replay)
    assert owner.get("task-1")["status"] == "claimed"


def test_ducklake_loss_and_lag_never_change_authority() -> None:
    owner = _owner()
    owner.apply(_put(), owner_id=OWNER_A, epoch=1)
    before = dict(owner.get("task-1") or {})
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
    assert owner.get("task-1") == before
    assert owner.lease().owner_id == OWNER_A
    assert owner.lease().epoch == 1
    assert owner.lease().fence == 1
    assert owner.bound_port is None


def test_qualification_receipt_is_contract_fail_closed() -> None:
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "qualification-receipt@1"
    assert payload.get("task_id") == "EAAEF-097" or payload.get("task_alias") == "EAAEF-097"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["live_runtime_invoked"] is False
    assert payload.get("live_quack_contacted", False) is False
    assert int(payload.get("live_quack_port", LIVE_QUACK_PORT)) == 19495
