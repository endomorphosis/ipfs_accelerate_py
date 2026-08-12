"""Tests for control-plane checkpoint/backup/restore (DQP-033).

Acceptance:

* Restore reproduces store/schema/event/task/lease roots and invalidates
  pre-rotation writers
* No accepted state is lost in the declared crash matrix
* Backup success is independently verified
* Direct-file maintenance cannot occur while server ownership is live/unknown

Evidence subset: crash before/after checkpoint, corrupt copy, disk full,
partial restore, schema version, server stopped, stale client, backup age.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup import (
    BACKUP_STATUS_VERIFIED,
    CONTROL_PLANE_BACKUP_INTERFACE,
    DECLARED_CRASH_SCENARIOS,
    RESTORE_OUTCOME_REHEARSAL,
    RESTORE_OUTCOME_SUCCESS,
    RESTORE_RECEIPT_INTERFACE,
    STORE_GENERATION_ROTATION_INTERFACE,
    AuthorityRoots,
    BackupSnapshot,
    ControlPlaneBackup,
    ControlPlaneBackupCorruptionError,
    ControlPlaneBackupOwnershipError,
    ControlPlaneBackupVerificationError,
    CrashScenario,
    OwnershipState,
    RestoreReceipt,
    StoreGenerationRotation,
    build_control_plane_backup,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    OwnerMarker,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    DEFAULT_MAINTENANCE_SCOPE,
    MAINTENANCE_LEASE_ACTIVE,
    acquire_maintenance_lease,
    release_maintenance_lease,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane backup hermetic tests",
)

_UUID = "123e4567-e89b-12d3-a456-426614174000"
_BIRTH = "birth:server-backup-1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _install(db: Path) -> None:
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="backup-test",
    )


def _seed_generation(
    db: Path,
    *,
    generation: int = 1,
    fence_epoch: int = 1,
    revision: int = 0,
    database_uuid: str = _UUID,
    birth_id: str = _BIRTH,
) -> None:
    with open_duckdb_connection(db) as connection:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at
            ) VALUES (?, 1, ?, ?, ?, ?, ?)
            """,
            [
                generation,
                fence_epoch,
                revision,
                database_uuid,
                birth_id,
                "1970-01-01T00:00:00Z",
            ],
        )


def _seed_population(db: Path) -> dict[str, Any]:
    """Seed accepted authority state: tasks, lease, domain event."""

    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "goal:root",
                "G-ROOT",
                "objective:backup",
                "",
                1,
                "Root",
                "open",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                0,
                "{}",
            ],
        )
        task_cids = []
        for index in range(3):
            task_cid = f"task:cid:{index + 1:03d}"
            task_cids.append(task_cid)
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    task_cid,
                    f"T-{index + 1:03d}",
                    "goal:root",
                    "",
                    "objective:backup",
                    index + 1,
                    "ready" if index else "claimed",
                    0,
                    "P0",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "{}",
                    json.dumps({"accepted": True, "ordinal": index + 1}),
                ],
            )
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt, state,
                started_at_ms, release_reason, retry_not_before_ms,
                owner_session_id, fence_epoch, revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                task_cids[0],
                "claim:001",
                "resolution:001",
                "did:claimant:1",
                1,
                7,
                9_999_999_999,
                1,
                "held",
                0,
                None,
                0,
                "session:lease-owner",
                1,
                0,
            ],
        )
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "event:001",
                "stream:tasks",
                1,
                1,
                "task.claimed",
                task_cids[0],
                "attempt:001",
                "session:lease-owner",
                "1970-01-01T00:00:01Z",
                json.dumps({"accepted": True}),
            ],
        )
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "event:002",
                "stream:tasks",
                2,
                2,
                "task.progress",
                task_cids[0],
                "attempt:001",
                "session:lease-owner",
                "1970-01-01T00:00:02Z",
                json.dumps({"phase": "implement"}),
            ],
        )
    return {"task_cids": task_cids}


def _prepare_db(tmp_path: Path) -> tuple[Path, Path, Path]:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    db = state_dir / "control.duckdb"
    backup_root = tmp_path / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    _install(db)
    _seed_generation(db)
    _seed_population(db)
    return db, backup_root, state_dir


def _lease(db: Path):
    return acquire_maintenance_lease(
        db,
        owner_session_id="session:backup-operator",
        process_birth_id="birth:backup-operator",
        scope=DEFAULT_MAINTENANCE_SCOPE,
    )


def _service(
    db: Path,
    backup_root: Path,
    state_dir: Path,
    *,
    liveness: OwnerLiveness | None = None,
    encryption_key_handle: str = "",
) -> ControlPlaneBackup:
    probe = None
    if liveness is not None:
        probe = lambda _birth: liveness  # noqa: E731
    return build_control_plane_backup(
        database_path=db,
        backup_root=backup_root,
        state_dir=state_dir,
        owner_liveness_probe=probe,
        encryption_key_handle=encryption_key_handle,
    )


def _write_owner_marker(
    state_dir: Path,
    db: Path,
    *,
    pid: int = 1,
    server_id: str = "server:live",
) -> Path:
    marker_path = state_dir / f"{db.name}.state-owner.json"
    marker = OwnerMarker(
        server_id=server_id,
        process_birth=ProcessBirthIdentity(
            pid=pid,
            start_time_ticks=100,
            boot_id="boot-test",
            parent_pid=0,
        ),
        database_path=str(db),
        started_at="1970-01-01T00:00:00Z",
        fence_token="fence-test",
        generation=1,
    )
    marker_path.write_text(
        json.dumps(marker.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return marker_path


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert CONTROL_PLANE_BACKUP_INTERFACE == "ControlPlaneBackup@1"
    assert RESTORE_RECEIPT_INTERFACE == "RestoreReceipt@1"
    assert STORE_GENERATION_ROTATION_INTERFACE == "StoreGenerationRotation@1"
    assert CrashScenario.CRASH_BEFORE_CHECKPOINT.value in DECLARED_CRASH_SCENARIOS
    assert set(DECLARED_CRASH_SCENARIOS) == {
        "crash_before_checkpoint",
        "crash_after_checkpoint",
        "corrupt_copy",
        "disk_full",
        "partial_restore",
        "schema_version",
        "server_stopped",
        "stale_client",
        "backup_age",
    }


# ---------------------------------------------------------------------------
# Ownership fence
# ---------------------------------------------------------------------------


def test_direct_file_maintenance_refused_when_owner_live(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    _write_owner_marker(state_dir, db, pid=os.getpid())
    service = _service(db, backup_root, state_dir, liveness=OwnerLiveness.ALIVE)
    observation = service.observe_ownership()
    assert observation.state is OwnershipState.LIVE
    assert not observation.admits_direct_file_maintenance
    with pytest.raises(ControlPlaneBackupOwnershipError, match="live"):
        service.assert_direct_file_maintenance_admitted()
    lease = _lease(db)
    try:
        with pytest.raises(ControlPlaneBackupOwnershipError):
            service.create_backup(maintenance_lease=lease)
    finally:
        release_maintenance_lease(db, lease)


def test_direct_file_maintenance_refused_when_owner_unknown(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    _write_owner_marker(state_dir, db, pid=999999)
    service = _service(db, backup_root, state_dir, liveness=OwnerLiveness.UNKNOWN)
    lease = _lease(db)
    try:
        with pytest.raises(ControlPlaneBackupOwnershipError, match="unknown"):
            service.create_backup(
                maintenance_lease=lease,
                require_maintenance_lease=True,
            )
    finally:
        release_maintenance_lease(db, lease)

def test_direct_file_maintenance_admitted_when_owner_absent_or_dead(
    tmp_path: Path,
) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    observation = service.observe_ownership()
    assert observation.state is OwnershipState.ABSENT
    assert observation.admits_direct_file_maintenance

    _write_owner_marker(state_dir, db, pid=1)
    dead_service = _service(
        db, backup_root, state_dir, liveness=OwnerLiveness.DEAD
    )
    dead_obs = dead_service.observe_ownership()
    assert dead_obs.state is OwnershipState.DEAD
    assert dead_obs.admits_direct_file_maintenance


def test_backup_requires_active_maintenance_lease(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    with pytest.raises(ControlPlaneBackupOwnershipError, match="maintenance lease"):
        service.create_backup(require_maintenance_lease=True)


# ---------------------------------------------------------------------------
# Checkpoint / roots / backup verification
# ---------------------------------------------------------------------------


def test_checkpoint_and_capture_roots(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    receipt = service.checkpoint()
    assert receipt["checkpointed"] is True
    roots = service.capture_roots()
    assert isinstance(roots, AuthorityRoots)
    assert roots.task_count == 3
    assert roots.lease_count == 1
    assert roots.event_watermark == 2
    assert roots.generation == 1
    assert roots.database_uuid == _UUID
    assert roots.store_root
    assert roots.schema_root
    assert roots.event_root
    assert roots.task_root
    assert roots.lease_root
    # Stable across re-capture.
    again = service.capture_roots()
    assert roots.matches(again)


def test_create_backup_independently_verified(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    lease = _lease(db)
    try:
        pre_roots = service.capture_roots()
        snapshot = service.create_backup(maintenance_lease=lease)
        assert snapshot.status == BACKUP_STATUS_VERIFIED
        assert snapshot.backup_id.startswith("backup:")
        assert Path(snapshot.body_path).is_file()
        assert Path(snapshot.manifest_path).is_file()
        assert snapshot.roots.matches(pre_roots)
        assert snapshot.independent_verification.get("verified") is True

        # Independent re-verification does not trust the create path alone.
        verification = service.verify_backup(snapshot)
        assert verification.verified is True
        assert verification.openable is True
        assert verification.roots is not None
        assert verification.roots.matches(pre_roots)

        # Row recorded in backup_snapshots.
        with open_duckdb_connection(db) as connection:
            row = connection.execute(
                "SELECT backup_id, status, artifact_digest FROM backup_snapshots "
                "WHERE backup_id = ?",
                [snapshot.backup_id],
            ).fetchone()
            assert row is not None
    finally:
        release_maintenance_lease(db, lease)


def test_backup_with_encryption_handle_is_digest_bound(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(
        db,
        backup_root,
        state_dir,
        encryption_key_handle="handle:backup-key:v1",
    )
    lease = _lease(db)
    try:
        snapshot = service.create_backup(maintenance_lease=lease)
        assert snapshot.encryption_bound is True
        assert snapshot.encryption_handle == "handle:backup-key:v1"
        verification = service.verify_backup(snapshot)
        assert verification.verified is True
    finally:
        release_maintenance_lease(db, lease)


def test_corrupt_backup_fails_independent_verification(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    lease = _lease(db)
    try:
        snapshot = service.create_backup(maintenance_lease=lease)
        body = Path(snapshot.body_path)
        original = body.read_bytes()
        body.write_bytes(b"\x00CORRUPT" + original[8:])
        verification = service.verify_backup(snapshot)
        assert verification.verified is False
        with pytest.raises(ControlPlaneBackupCorruptionError):
            service.probe_corruption(snapshot)
    finally:
        release_maintenance_lease(db, lease)


# ---------------------------------------------------------------------------
# Restore + generation rotation
# ---------------------------------------------------------------------------


def test_restore_reproduces_roots_and_invalidates_pre_rotation_writers(
    tmp_path: Path,
) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    lease = _lease(db)
    try:
        accepted = service.capture_roots()
        snapshot = service.create_backup(maintenance_lease=lease)

        # Mutate live state after backup (should be wiped by restore).
        with open_duckdb_connection(db) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'lost' WHERE task_cid = ?",
                ["task:cid:002"],
            )
            connection.execute(
                """
                INSERT INTO domain_events (
                    event_id, stream_id, sequence, global_sequence, event_type,
                    task_cid, attempt_id, session_id, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    "event:lost",
                    "stream:tasks",
                    99,
                    99,
                    "task.lost",
                    "task:cid:002",
                    "",
                    "",
                    "1970-01-01T00:00:99Z",
                    "{}",
                ],
            )
        diverged = service.capture_roots()
        assert not diverged.matches(accepted)

        receipt = service.restore(
            snapshot,
            maintenance_lease=lease,
            rotate_generation=True,
            rehearsal=False,
        )
        assert isinstance(receipt, RestoreReceipt)
        assert receipt.outcome == RESTORE_OUTCOME_SUCCESS
        assert receipt.writers_invalidated is True
        assert receipt.rotation is not None
        assert receipt.rotation.INTERFACE == STORE_GENERATION_ROTATION_INTERFACE
        assert receipt.rotation.invalidates(accepted.generation, accepted.fence_epoch)
        # Content roots reproduced; generation advanced.
        assert receipt.roots.matches(accepted, ignore_generation=True)
        assert receipt.roots.generation == accepted.generation + 1
        assert receipt.roots.fence_epoch == accepted.fence_epoch + 1
        assert receipt.roots.task_count == 3
        assert receipt.roots.lease_count == 1
        assert receipt.roots.event_watermark == 2
        assert receipt.roots.task_root == accepted.task_root
        assert receipt.roots.lease_root == accepted.lease_root
        assert receipt.roots.event_root == accepted.event_root
        assert receipt.roots.schema_root == accepted.schema_root

        # Pre-rotation writer generation is rejected.
        pre_writer_generation = accepted.generation
        pre_writer_fence = accepted.fence_epoch
        assert receipt.rotation.invalidates(pre_writer_generation, pre_writer_fence)
        assert not receipt.rotation.invalidates(
            receipt.roots.generation, receipt.roots.fence_epoch
        )

        # Receipt persisted.
        with open_duckdb_connection(db) as connection:
            row = connection.execute(
                "SELECT outcome FROM restore_receipts WHERE receipt_id = ?",
                [receipt.receipt_id],
            ).fetchone()
            assert row is not None
    finally:
        release_maintenance_lease(db, lease)


def test_restore_rehearsal_does_not_mutate_live_database(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    lease = _lease(db)
    try:
        accepted = service.capture_roots()
        snapshot = service.create_backup(maintenance_lease=lease)
        receipt = service.restore(
            snapshot,
            maintenance_lease=lease,
            rotate_generation=True,
            rehearsal=True,
        )
        assert receipt.outcome == RESTORE_OUTCOME_REHEARSAL
        assert receipt.rehearsal is True
        assert receipt.writers_invalidated is True
        live = service.capture_roots()
        assert live.matches(accepted)
        assert live.generation == accepted.generation
    finally:
        release_maintenance_lease(db, lease)


def test_restore_refuses_unverified_backup(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    lease = _lease(db)
    try:
        snapshot = service.create_backup(maintenance_lease=lease)
        Path(snapshot.body_path).write_bytes(b"not-a-database")
        with pytest.raises(ControlPlaneBackupVerificationError):
            service.restore(snapshot, maintenance_lease=lease)
    finally:
        release_maintenance_lease(db, lease)


def test_store_generation_rotation_strictly_increases(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        StoreGenerationRotation(
            rotation_id="r1",
            store_id="control.duckdb",
            database_uuid=_UUID,
            previous_generation=2,
            new_generation=2,
            previous_fence_epoch=1,
            new_fence_epoch=2,
            schema_revision=1,
            birth_id="birth:x",
            rotated_at="1970-01-01T00:00:00Z",
            reason="test",
        )


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_retention_prunes_old_backups(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    clock = {"now": 1_700_000_000.0}

    def _clock() -> float:
        return float(clock["now"])

    service = ControlPlaneBackup(
        database_path=db,
        backup_root=backup_root,
        state_dir=state_dir,
        clock=_clock,
    )
    lease = _lease(db)
    try:
        ids: list[str] = []
        for index in range(4):
            clock["now"] = 1_700_000_000.0 + index * 100
            snap = service.create_backup(
                maintenance_lease=lease,
                backup_id=f"backup:ret-{index}",
            )
            ids.append(snap.backup_id)
        manifest = service.apply_retention(keep_count=2, max_age_seconds=10_000)
        assert len(manifest.retained) == 2
        assert len(manifest.pruned) == 2
        remaining = {item.backup_id for item in service.list_backups()}
        assert remaining == set(manifest.retained)
        assert ids[-1] in remaining
        assert ids[0] not in remaining
    finally:
        release_maintenance_lease(db, lease)


# ---------------------------------------------------------------------------
# Crash matrix
# ---------------------------------------------------------------------------


def test_declared_crash_matrix_preserves_accepted_state(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    lease = _lease(db)
    try:
        accepted = service.capture_roots()
        report = service.evaluate_crash_matrix(
            accepted_roots=accepted,
            maintenance_lease=lease,
        )
        assert report.accepted_state_preserved is True
        for name in DECLARED_CRASH_SCENARIOS:
            assert name in report.scenarios
            assert report.scenarios[name].get("ok") is True, name
        # Live accepted state still present after matrix side effects.
        live = service.capture_roots()
        assert live.matches(accepted)
        payload = report.to_dict()
        assert payload["accepted_state_preserved"] is True
        assert set(payload["declared_scenarios"]) == set(DECLARED_CRASH_SCENARIOS)
    finally:
        release_maintenance_lease(db, lease)


def test_backup_snapshot_round_trip_dict(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    service = _service(db, backup_root, state_dir)
    lease = _lease(db)
    try:
        snapshot = service.create_backup(maintenance_lease=lease)
        restored = BackupSnapshot.from_dict(snapshot.to_dict())
        assert restored.backup_id == snapshot.backup_id
        assert restored.artifact_digest == snapshot.artifact_digest
        assert restored.roots.matches(snapshot.roots)
        receipt = service.restore(
            restored,
            maintenance_lease=lease,
            rotate_generation=True,
            rehearsal=True,
        )
        again = RestoreReceipt.from_dict(receipt.to_dict())
        assert again.receipt_id == receipt.receipt_id
        assert again.writers_invalidated is True
    finally:
        release_maintenance_lease(db, lease)


def test_lock_held_blocks_direct_file_maintenance(tmp_path: Path) -> None:
    db, backup_root, state_dir = _prepare_db(tmp_path)
    lock_path = state_dir / f"{db.name}.state-owner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        service = _service(db, backup_root, state_dir)
        observation = service.observe_ownership()
        assert observation.state is OwnershipState.LOCK_HELD
        with pytest.raises(ControlPlaneBackupOwnershipError):
            service.assert_direct_file_maintenance_admitted()
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
