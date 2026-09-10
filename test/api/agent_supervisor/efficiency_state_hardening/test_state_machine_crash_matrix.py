"""Crash/restart matrix for the existing durable owner-recovery authority."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, TypeVar

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import EventCursor
from ipfs_accelerate_py.agent_supervisor.control.task_transition_service import (
    TaskLeaseFence,
    TaskTransitionService,
    TransitionLeaseError,
)
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    OwnerRestartSnapshot,
    RecoveryFault,
    SupervisorRecovery,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


_ResultT = TypeVar("_ResultT")


def _snapshot() -> OwnerRestartSnapshot:
    return OwnerRestartSnapshot(
        repository_id="repository:state-machine-matrix",
        tree_id="tree:state-machine-matrix",
        generation=3,
        cursor=EventCursor(
            stream_id="events:state-machine-matrix",
            position=11,
            last_event_id="event:11",
            snapshot_id="tree:state-machine-matrix",
        ),
        task_state={"task:alpha": {"revision": 4, "status": "reconciliation_pending"}},
        event_state={"head_event_id": "event:11", "event_count": 11},
        lease_state={
            "lease_id": "lease:owner-one",
            "owner_session_id": "owner:one",
            "fencing_epoch": 7,
            "claim_revision": 4,
        },
        idempotency_state={"provider:alpha": "effect-receipt:alpha"},
        reconciliation_state={
            "provider:alpha": {
                "status": "provider_outcome_unknown",
                "effect_observed": True,
                "receipt_observed": False,
            }
        },
        owner_session_id="owner:one",
        fencing_epoch=7,
        authenticated=True,
        authentication_subject_id="supervisor:alpha",
        authentication_binding_id="grant-binding:durable",
    )


class _DurableOwner:
    """Serializable stand-in for the already-existing owner authority."""

    def __init__(self) -> None:
        self.snapshot = _snapshot()
        self.takeovers = 0

    def rebuild(self, expected: OwnerRestartSnapshot) -> OwnerRestartSnapshot:
        assert expected.snapshot_id == self.snapshot.snapshot_id
        return OwnerRestartSnapshot.from_dict(self.snapshot.to_dict())

    def authenticated(self, snapshot: OwnerRestartSnapshot) -> bool:
        return (
            snapshot.authenticated
            and snapshot.authentication_subject_id == "supervisor:alpha"
            and snapshot.authentication_binding_id == "grant-binding:durable"
        )

    def take_over(self, expected: OwnerRestartSnapshot, owner: str) -> OwnerRestartSnapshot:
        self.takeovers += 1
        self.snapshot = replace(
            expected,
            generation=expected.generation + 1,
            owner_session_id=owner,
            fencing_epoch=expected.fencing_epoch + 1,
            lease_state={
                **expected.lease_state,
                "lease_id": f"lease:{owner}",
                "owner_session_id": owner,
                "fencing_epoch": expected.fencing_epoch + 1,
                "claim_revision": int(expected.lease_state["claim_revision"]) + 1,
            },
            state_root="",
            snapshot_id="",
        )
        return self.snapshot


def _recovery(tmp_path: Path, owner: _DurableOwner) -> SupervisorRecovery:
    recovery = SupervisorRecovery(tmp_path / "recovery")
    recovery.checkpoint_owner_restart(owner.snapshot, accepted_merged_tree_evidence=("receipt:accepted",))
    return recovery


@pytest.mark.parametrize("fault", tuple(RecoveryFault), ids=lambda fault: fault.value)
def test_every_crash_boundary_restarts_from_the_same_durable_truth(
    tmp_path: Path, fault: RecoveryFault
) -> None:
    owner = _DurableOwner()
    _recovery(tmp_path, owner)
    expected = owner.snapshot

    receipt = SupervisorRecovery(tmp_path / "recovery").recover_owner_restart(
        incident_id=f"matrix:{fault.value}",
        fault=fault,
        repository_id=expected.repository_id,
        tree_id=expected.tree_id,
        owner_session_id=expected.owner_session_id,
        rebuild=owner.rebuild,
        verify_authenticated=owner.authenticated,
    )

    assert receipt.resulting_state_root == expected.state_root
    assert receipt.resulting_snapshot_id == expected.snapshot_id
    assert receipt.resulting_fencing_epoch == expected.fencing_epoch
    assert owner.snapshot.reconciliation_state == expected.reconciliation_state
    assert owner.snapshot.idempotency_state == expected.idempotency_state

    # Replaying the identical crash delivery is a durable no-op, which covers
    # duplicate delivery at each crash boundary.
    replay = SupervisorRecovery(tmp_path / "recovery").recover_owner_restart(
        incident_id=f"matrix:{fault.value}",
        fault=fault,
        repository_id=expected.repository_id,
        tree_id=expected.tree_id,
        owner_session_id=expected.owner_session_id,
        rebuild=owner.rebuild,
        verify_authenticated=owner.authenticated,
    )
    assert replay.receipt_id == receipt.receipt_id


def test_owner_loss_takes_over_once_advances_fence_and_never_erases_unknown_effect(tmp_path: Path) -> None:
    owner = _DurableOwner()
    recovery = _recovery(tmp_path, owner)
    before = owner.snapshot
    receipt = recovery.recover_owner_restart(
        incident_id="matrix:owner-loss",
        fault=RecoveryFault.STALE_LEASE,
        repository_id=before.repository_id,
        tree_id=before.tree_id,
        owner_session_id="owner:two",
        rebuild=owner.rebuild,
        verify_authenticated=owner.authenticated,
        owner_lost=True,
        takeover=owner.take_over,
        current_fencing_token=8,
    )
    assert receipt.takeover is True
    assert receipt.previous_fencing_epoch == 7
    assert receipt.resulting_fencing_epoch == 8
    assert owner.snapshot.reconciliation_state == before.reconciliation_state
    assert owner.snapshot.idempotency_state == before.idempotency_state

    duplicate = SupervisorRecovery(tmp_path / "recovery").recover_owner_restart(
        incident_id="matrix:owner-loss",
        fault=RecoveryFault.STALE_LEASE,
        repository_id=before.repository_id,
        tree_id=before.tree_id,
        owner_session_id="owner:two",
        rebuild=owner.rebuild,
        verify_authenticated=owner.authenticated,
        owner_lost=True,
        takeover=owner.take_over,
        current_fencing_token=8,
    )
    assert duplicate.receipt_id == receipt.receipt_id
    assert owner.takeovers == 1


class _CurrentLeaseAuthority:
    """One deterministic serialization boundary for an event/projection check."""

    def __init__(self, lease: TaskLeaseFence) -> None:
        self.lease = lease

    def execute_fenced(
        self, lease: TaskLeaseFence, callback: Callable[[], _ResultT]
    ) -> _ResultT:
        if lease != self.lease:
            raise TransitionLeaseError("stale lease/fence cannot execute or complete")
        return callback()


def test_transition_event_and_materialized_task_projection_are_equivalent(tmp_path: Path) -> None:
    repository = IntentRepository(tmp_path / "intent.duckdb")
    repository.upsert_goal(
        goal_cid="goal:matrix", goal_alias="GOAL-MATRIX", objective_id="objective:matrix", title="matrix"
    )
    repository.upsert_task(
        task_cid="task:matrix", task_alias="TASK-MATRIX", goal_cid="goal:matrix", status="ready"
    )
    lease = TaskLeaseFence(
        task_cid="task:matrix",
        owner_session_id="owner:matrix",
        lease_id="lease:matrix",
        fencing_token=1,
        fence_epoch=1,
        claim_revision=1,
    )
    service = TaskTransitionService(repository, lease_fence_authority=_CurrentLeaseAuthority(lease))
    result = service.transition(
        StateCommand(
            command_id="command:matrix",
            command_kind=CommandKind.APPEND,
            store_id="store:matrix",
            session_id=lease.owner_session_id,
            expected_generation=1,
            expected_revision=1,
            fence_epoch=lease.fence_epoch,
            idempotency_key="idempotency:matrix",
            parameters={
                "task_cid": lease.task_cid,
                "new_status": "in_progress",
                "lease_id": lease.lease_id,
                "fencing_token": lease.fencing_token,
                "claim_revision": lease.claim_revision,
            },
        )
    )
    task = repository.get_task(lease.task_cid)
    events = repository.list_events(after_global_sequence=result.receipt.global_sequence - 1, limit=1)
    assert task is not None and len(events) == 1
    event_body = events[0]["body"]["body"]
    assert event_body["task_cid"] == lease.task_cid
    assert event_body["revision"] == task["revision"] == result.revision
    assert task["status"] == result.status == "in_progress"
    repository.assert_projection_matches_events()
