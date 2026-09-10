"""Owner-loss recovery keeps the existing authority's durable truth intact."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import EventCursor
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    OwnerRestartSnapshot,
    RecoveryFault,
    RecoveryIntegrityError,
    SupervisorRecovery,
)


def _snapshot(*, owner: str = "owner:one", fence: int = 7, generation: int = 1) -> OwnerRestartSnapshot:
    """A realistic owner projection including an unresolved external effect."""

    return OwnerRestartSnapshot(
        repository_id="repository:owner-restart",
        tree_id="tree:owner-restart",
        generation=generation,
        cursor=EventCursor(
            stream_id="events:owner-restart",
            position=19,
            last_event_id="event:19",
            snapshot_id="tree:owner-restart",
        ),
        task_state={
            "task:alpha": {"revision": 4, "status": "reconciliation_pending"},
        },
        event_state={"head_event_id": "event:19", "event_count": 19},
        lease_state={
            "lease_id": f"lease:{owner}",
            "owner_session_id": owner,
            "fencing_epoch": fence,
            "claim_revision": 3,
        },
        idempotency_state={"provider:alpha": "effect-receipt:alpha"},
        reconciliation_state={
            "provider:alpha": {
                "status": "provider_outcome_unknown",
                "effect_observed": True,
                "receipt_observed": False,
            }
        },
        owner_session_id=owner,
        fencing_epoch=fence,
        authenticated=True,
        authentication_subject_id="supervisor:alpha",
        authentication_binding_id="grant-binding:durable",
    )


class _ExistingOwnerAuthority:
    """A test double for the already-existing typed owner/transition authority."""

    def __init__(self, snapshot: OwnerRestartSnapshot) -> None:
        self.snapshot = snapshot
        self.authentication_checks: list[str] = []
        self.takeovers = 0

    def rebuild(self, expected: OwnerRestartSnapshot) -> OwnerRestartSnapshot:
        assert expected.snapshot_id == self.snapshot.snapshot_id
        # Simulate serialization across the dead process boundary.
        return OwnerRestartSnapshot.from_dict(self.snapshot.to_dict())

    def take_over(self, expected: OwnerRestartSnapshot, owner: str) -> OwnerRestartSnapshot:
        assert expected.snapshot_id == self.snapshot.snapshot_id
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

    def authenticated(self, snapshot: OwnerRestartSnapshot) -> bool:
        self.authentication_checks.append(snapshot.authentication_binding_id)
        # The authority sees only a durable binding ID, never a rematerialized
        # credential or a direct database repair request.
        return (
            snapshot.authenticated
            and snapshot.authentication_subject_id == "supervisor:alpha"
            and snapshot.authentication_binding_id == "grant-binding:durable"
        )


def _checkpoint(root: Path, authority: _ExistingOwnerAuthority) -> SupervisorRecovery:
    recovery = SupervisorRecovery(root / "recovery")
    recovery.checkpoint_owner_restart(
        authority.snapshot,
        accepted_merged_tree_evidence=("receipt:accepted-merge",),
    )
    return recovery


def test_crash_boundary_and_repeated_restart_reconstruct_identical_state_root(
    tmp_path: Path,
) -> None:
    authority = _ExistingOwnerAuthority(_snapshot())
    _checkpoint(tmp_path, authority)
    expected_root = authority.snapshot.state_root
    expected_snapshot = authority.snapshot.snapshot_id

    # Every process gets a fresh recovery object, as it would after an owner
    # crash between any durable checkpoint and the next authenticated request.
    receipts = []
    for number in range(3):
        recovery = SupervisorRecovery(tmp_path / "recovery")
        receipts.append(
            recovery.recover_owner_restart(
                incident_id=f"restart:crash-boundary:{number}",
                fault=RecoveryFault.PROCESS_CRASH,
                repository_id="repository:owner-restart",
                tree_id="tree:owner-restart",
                owner_session_id="owner:one",
                rebuild=authority.rebuild,
                verify_authenticated=authority.authenticated,
            )
        )

    assert {receipt.resulting_state_root for receipt in receipts} == {expected_root}
    assert {receipt.resulting_snapshot_id for receipt in receipts} == {expected_snapshot}
    assert {receipt.resulting_fencing_epoch for receipt in receipts} == {7}
    assert authority.takeovers == 0
    assert authority.authentication_checks == ["grant-binding:durable"] * 3


def test_owner_takeover_advances_fence_and_preserves_unknown_effect_truth(tmp_path: Path) -> None:
    authority = _ExistingOwnerAuthority(_snapshot())
    recovery = _checkpoint(tmp_path, authority)
    before = authority.snapshot

    receipt = recovery.recover_owner_restart(
        incident_id="restart:owner-lost",
        fault=RecoveryFault.STALE_LEASE,
        repository_id=before.repository_id,
        tree_id=before.tree_id,
        owner_session_id="owner:two",
        rebuild=authority.rebuild,
        verify_authenticated=authority.authenticated,
        owner_lost=True,
        takeover=authority.take_over,
        current_fencing_token=8,
    )

    assert receipt.takeover is True
    assert receipt.previous_fencing_epoch == 7
    assert receipt.resulting_fencing_epoch == 8
    assert authority.snapshot.owner_session_id == "owner:two"
    assert authority.snapshot.task_state == before.task_state
    assert authority.snapshot.event_state == before.event_state
    assert authority.snapshot.idempotency_state == before.idempotency_state
    assert authority.snapshot.reconciliation_state == before.reconciliation_state
    base_receipt = recovery.receipt("owner-restart-base:restart:owner-lost")
    assert base_receipt is not None and base_receipt.stale_actor_fenced

    # Replaying the same crash report after another process restart adopts the
    # immutable receipt; it cannot take over a second time.
    replayed = SupervisorRecovery(tmp_path / "recovery").recover_owner_restart(
        incident_id="restart:owner-lost",
        fault=RecoveryFault.STALE_LEASE,
        repository_id=before.repository_id,
        tree_id=before.tree_id,
        owner_session_id="owner:two",
        rebuild=authority.rebuild,
        verify_authenticated=authority.authenticated,
        owner_lost=True,
        takeover=authority.take_over,
        current_fencing_token=8,
    )
    assert replayed.receipt_id == receipt.receipt_id
    assert authority.takeovers == 1

    restarted = SupervisorRecovery(tmp_path / "recovery")
    resumed = restarted.recover_owner_restart(
        incident_id="restart:post-takeover",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id=before.repository_id,
        tree_id=before.tree_id,
        owner_session_id="owner:two",
        rebuild=authority.rebuild,
        verify_authenticated=authority.authenticated,
    )
    assert resumed.resulting_state_root == authority.snapshot.state_root
    assert resumed.resulting_fencing_epoch == 8
    assert authority.takeovers == 1


def test_restart_rejects_rebuild_that_loses_unknown_outcome_or_needs_credentials(
    tmp_path: Path,
) -> None:
    authority = _ExistingOwnerAuthority(_snapshot())
    recovery = _checkpoint(tmp_path, authority)

    def lost_unknown_effect(expected: OwnerRestartSnapshot) -> OwnerRestartSnapshot:
        return replace(
            expected,
            reconciliation_state={},
            state_root="",
            snapshot_id="",
        )

    with pytest.raises(RecoveryIntegrityError, match="state root"):
        recovery.recover_owner_restart(
            incident_id="restart:lost-unknown-effect",
            fault=RecoveryFault.PROCESS_CRASH,
            repository_id=authority.snapshot.repository_id,
            tree_id=authority.snapshot.tree_id,
            owner_session_id="owner:one",
            rebuild=lost_unknown_effect,
            verify_authenticated=authority.authenticated,
        )

    with pytest.raises(RecoveryIntegrityError, match="credential material"):
        replace(
            authority.snapshot,
            task_state={"task:alpha": {"secret": "must-not-persist"}},
            state_root="",
            snapshot_id="",
        )
