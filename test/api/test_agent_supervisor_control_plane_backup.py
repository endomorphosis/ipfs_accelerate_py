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
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup import (
    CONTROL_PLANE_BACKUP_INTERFACE,
    DECLARED_CRASH_SCENARIOS,
    OWNERSHIP_LIVE,
    OWNERSHIP_UNKNOWN,
    RESTORE_RECEIPT_INTERFACE,
    STORE_GENERATION_ROTATION_INTERFACE,
    BackupSnapshot,
    ControlPlaneBackup,
    ControlPlaneBackupGenerationError,
    ControlPlaneBackupIntegrityError,
    ControlPlaneBackupOwnershipError,
    RestoreReceipt,
    StoreGenerationRotation,
    assert_direct_file_maintenance_allowed,
    compute_snapshot_roots,
    duckdb_available,
    inspect_server_ownership,
    open_control_plane_backup,
    owner_marker_path_for,
    seed_control_plane_for_backup,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import OwnerMarker
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    StoreGeneration,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane backup hermetic tests",
)

_UUID = "123e4567-e89b-12d3-a456-426614174000"


def _seed(tmp_path: Path, name: str = "control.duckdb") -> Path:
    db = tmp_path / name
    seed_control_plane_for_backup(db, database_uuid=_UUID)
    return db


def _service(db: Path, tmp_path: Path, **kwargs: object) -> ControlPlaneBackup:
    return open_control_plane_backup(
        db,
        backup_root=tmp_path / "backups",
        skip_ownership_check=True,
        **kwargs,  # type: ignore[arg-type]
    )


def _write_owner_marker(
    db: Path,
    *,
    pid: int = 1,
    start_time_ticks: int = 100,
) -> Path:
    marker_path = owner_marker_path_for(db)
    marker = OwnerMarker(
        server_id="server:test",
        process_birth=ProcessBirthIdentity(
            pid=pid,
            start_time_ticks=start_time_ticks,
            boot_id="boot-test",
            parent_pid=0,
        ),
        database_path=str(db),
        started_at="1970-01-01T00:00:00Z",
        fence_token="fence-test",
        generation=1,
    )
    marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
    return marker_path


# ---------------------------------------------------------------------------
# Interface / cold import
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert CONTROL_PLANE_BACKUP_INTERFACE == "ControlPlaneBackup@1"
    assert RESTORE_RECEIPT_INTERFACE == "RestoreReceipt@1"
    assert STORE_GENERATION_ROTATION_INTERFACE == "StoreGenerationRotation@1"
    assert ControlPlaneBackup.INTERFACE == CONTROL_PLANE_BACKUP_INTERFACE
    assert RestoreReceipt.INTERFACE == RESTORE_RECEIPT_INTERFACE
    assert StoreGenerationRotation.INTERFACE == STORE_GENERATION_ROTATION_INTERFACE


def test_cold_import_has_no_side_effects() -> None:
    # Importing the module must not open databases or touch the filesystem.
    import ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup as mod

    assert mod.CONTROL_PLANE_BACKUP_VERSION == 1
    assert "crash_before_checkpoint" in mod.DECLARED_CRASH_SCENARIOS


# ---------------------------------------------------------------------------
# Roots / checkpoint / backup verification
# ---------------------------------------------------------------------------


def test_compute_snapshot_roots_stable(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    first = compute_snapshot_roots(db)
    second = compute_snapshot_roots(db)
    assert first.matches(second)
    assert first.task_count >= 1
    assert first.lease_count >= 1
    assert first.event_watermark >= 1
    assert first.store_root.startswith("b")
    assert first.schema_root.startswith("b")
    assert first.event_root.startswith("b")
    assert first.task_root.startswith("b")
    assert first.lease_root.startswith("b")
    assert first.accepted_root.startswith("b")


def test_checkpoint_and_independently_verified_backup(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    ckpt = svc.checkpoint()
    assert ckpt.checkpointed is True
    assert ckpt.roots is not None
    assert ckpt.roots.task_count >= 1

    snapshot = svc.create_backup(label="primary")
    assert snapshot.status == "verified"
    assert snapshot.artifact_digest.startswith("sha256:")
    assert Path(str(snapshot.body["artifact_path"])).is_file()

    # Independent verification recomputes digests/roots without trusting creator.
    verification = svc.verify_backup(snapshot)
    assert verification.verified is True
    assert verification.roots_match is True
    assert verification.recomputed_digest == snapshot.artifact_digest
    assert verification.recomputed_digest == verification.artifact_digest


def test_corrupt_backup_fails_independent_verification(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    snapshot = svc.create_backup(label="to-corrupt")
    artifact = Path(str(snapshot.body["artifact_path"]))
    raw = bytearray(artifact.read_bytes())
    raw[min(128, len(raw) - 1)] ^= 0xFF
    artifact.write_bytes(bytes(raw))

    verification = svc.verify_backup(snapshot)
    assert verification.verified is False


def test_encrypted_backup_is_digest_bound(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    key = b"test-backup-key-32-bytes-padded!!"
    svc = _service(db, tmp_path, encryption_key=key)
    snapshot = svc.create_backup(encrypt=True, label="enc")
    assert snapshot.encrypted is True
    assert snapshot.cleartext_digest.startswith("sha256:")
    assert snapshot.artifact_digest != snapshot.cleartext_digest

    verification = svc.verify_backup(snapshot, encryption_key=key)
    assert verification.verified is True

    # Wrong key fails closed.
    bad = svc.verify_backup(snapshot, encryption_key=b"wrong-key-material-here!!!!")
    assert bad.verified is False


# ---------------------------------------------------------------------------
# Restore + generation rotation
# ---------------------------------------------------------------------------


def test_restore_reproduces_roots_and_invalidates_pre_rotation_writers(
    tmp_path: Path,
) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    before = compute_snapshot_roots(db)
    snapshot = svc.create_backup(label="restore-me")

    with open_duckdb_connection(db) as connection:
        row = connection.execute(
            """
            SELECT generation, schema_revision, fence_epoch, revision,
                   database_uuid, birth_id
            FROM store_generations
            ORDER BY generation DESC LIMIT 1
            """
        ).fetchone()
    pre_writer = StoreGeneration(
        store_id="control.duckdb",
        generation=int(row[0] if not hasattr(row, "keys") else row["generation"]),
        schema_revision=int(
            row[1] if not hasattr(row, "keys") else row["schema_revision"]
        ),
        fence_epoch=int(row[2] if not hasattr(row, "keys") else row["fence_epoch"]),
        revision=int(row[3] if not hasattr(row, "keys") else row["revision"]),
        database_uuid=str(
            row[4] if not hasattr(row, "keys") else row["database_uuid"]
        ),
        birth_id=str(row[5] if not hasattr(row, "keys") else row["birth_id"] or ""),
    )

    # Mutate live store after backup so restore must come from the artifact.
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_cid, objective_id,
                ordinal, status, revision, priority, created_at, updated_at,
                identity_json, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "task:cid:post-backup",
                "T-POST",
                "goal:root",
                "",
                "objective:backup",
                99,
                "ready",
                0,
                "P0",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                "{}",
                "{}",
            ],
        )
        connection.execute("CHECKPOINT")
    mutated = compute_snapshot_roots(db)
    assert mutated.task_root != before.task_root

    target = tmp_path / "restored.duckdb"
    receipt = svc.restore(snapshot, target_path=target, rotate_generation=True)
    assert receipt.outcome == "success"
    assert receipt.rotated_generation is not None
    assert receipt.rotated_generation == before.generation + 1

    restored = compute_snapshot_roots(target)
    # Domain/accepted roots match the backup (not the mutated live store).
    assert restored.event_root == before.event_root
    assert restored.task_root == before.task_root
    assert restored.lease_root == before.lease_root
    assert restored.schema_root == before.schema_root
    assert restored.accepted_root == before.accepted_root
    assert restored.generation == before.generation + 1
    assert restored.store_root != before.store_root  # generation advanced

    # Pre-rotation writer is refused against the restored/rotated store.
    restored_svc = _service(target, tmp_path / "restored_backups")
    with pytest.raises(ControlPlaneBackupGenerationError):
        restored_svc.assert_writer_admitted(pre_writer)

    rotation = StoreGenerationRotation(
        rotation_id="rotation:test",
        store_id="control.duckdb",
        database_uuid=before.database_uuid,
        previous_generation=before.generation,
        new_generation=restored.generation,
        previous_fence_epoch=pre_writer.fence_epoch,
        new_fence_epoch=pre_writer.fence_epoch + 1,
        schema_revision=before.schema_revision,
        previous_revision=pre_writer.revision,
        new_revision=0,
        birth_id="birth:test",
        rotated_at="1970-01-01T00:00:00Z",
    )
    assert rotation.invalidates(pre_writer) is True


def test_restore_without_rotation_keeps_generation(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    before = compute_snapshot_roots(db)
    snapshot = svc.create_backup()
    target = tmp_path / "plain-restore.duckdb"
    receipt = svc.restore(snapshot, target_path=target, rotate_generation=False)
    assert receipt.rotated_generation is None
    restored = compute_snapshot_roots(target)
    assert restored.matches(before)
    assert restored.store_root == before.store_root
    assert restored.accepted_root == before.accepted_root


def test_generation_rotation_invalidates_writers(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    with open_duckdb_connection(db) as connection:
        row = connection.execute(
            """
            SELECT generation, schema_revision, fence_epoch, revision,
                   database_uuid, birth_id
            FROM store_generations ORDER BY generation DESC LIMIT 1
            """
        ).fetchone()
    previous = StoreGeneration(
        store_id="control.duckdb",
        generation=int(row[0]),
        schema_revision=int(row[1]),
        fence_epoch=int(row[2]),
        revision=int(row[3]),
        database_uuid=str(row[4]),
        birth_id=str(row[5] or ""),
    )
    rotation = svc.rotate_generation(reason="test")
    assert rotation.new_generation == previous.generation + 1
    assert rotation.invalidates(previous) is True
    with pytest.raises(ControlPlaneBackupGenerationError):
        svc.assert_writer_admitted(previous)
    live = svc.assert_writer_admitted(rotation.new_store_generation())
    assert live.generation == rotation.new_generation


# ---------------------------------------------------------------------------
# Ownership fencing
# ---------------------------------------------------------------------------


def test_direct_file_maintenance_refused_when_ownership_live(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    _write_owner_marker(db, pid=os_getpid_safe())

    def always_alive(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return OwnerLiveness.ALIVE

    status = inspect_server_ownership(db, liveness=always_alive)
    assert status["status"] == OWNERSHIP_LIVE
    assert status["allow_direct_file"] is False

    with pytest.raises(ControlPlaneBackupOwnershipError):
        assert_direct_file_maintenance_allowed(db, liveness=always_alive)

    svc = ControlPlaneBackup(
        db,
        backup_root=tmp_path / "backups",
        skip_ownership_check=False,
        liveness=always_alive,
    )
    with pytest.raises(ControlPlaneBackupOwnershipError):
        svc.checkpoint()
    with pytest.raises(ControlPlaneBackupOwnershipError):
        svc.create_backup()


def test_direct_file_maintenance_refused_when_ownership_unknown(
    tmp_path: Path,
) -> None:
    db = _seed(tmp_path)
    _write_owner_marker(db)

    def always_unknown(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return OwnerLiveness.UNKNOWN

    status = inspect_server_ownership(db, liveness=always_unknown)
    assert status["status"] == OWNERSHIP_UNKNOWN
    assert status["allow_direct_file"] is False

    svc = ControlPlaneBackup(
        db,
        backup_root=tmp_path / "backups",
        liveness=always_unknown,
    )
    with pytest.raises(ControlPlaneBackupOwnershipError):
        svc.create_backup()


def test_direct_file_maintenance_allowed_when_owner_stale_or_absent(
    tmp_path: Path,
) -> None:
    db = _seed(tmp_path)

    # Absent marker: allowed.
    status = inspect_server_ownership(db)
    assert status["allow_direct_file"] is True

    _write_owner_marker(db)

    def always_dead(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return OwnerLiveness.DEAD

    status = inspect_server_ownership(db, liveness=always_dead)
    assert status["allow_direct_file"] is True

    svc = ControlPlaneBackup(
        db,
        backup_root=tmp_path / "backups",
        liveness=always_dead,
    )
    snapshot = svc.create_backup(label="stale-owner")
    assert snapshot.status == "verified"


def os_getpid_safe() -> int:
    import os

    return int(os.getpid())


# ---------------------------------------------------------------------------
# Retention + crash matrix
# ---------------------------------------------------------------------------


def test_retention_prunes_old_backups_without_losing_live_state(
    tmp_path: Path,
) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    before = compute_snapshot_roots(db)
    first = svc.create_backup(label="one")
    second = svc.create_backup(label="two")
    assert Path(str(first.body["artifact_path"])).is_file()
    assert Path(str(second.body["artifact_path"])).is_file()

    manifest = svc.apply_retention(keep_last=1, max_age_seconds=10**9)
    assert second.backup_id in manifest.retained
    assert first.backup_id in manifest.pruned
    assert not Path(str(first.body["artifact_path"])).exists()
    assert Path(str(second.body["artifact_path"])).is_file()

    after = compute_snapshot_roots(db)
    assert after.accepted_root == before.accepted_root


def test_declared_crash_matrix_preserves_accepted_state(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    report = svc.run_crash_matrix(work_dir=tmp_path / "crash_matrix")
    assert report.accepted_state_lost is False
    for name in DECLARED_CRASH_SCENARIOS:
        assert name in report.scenarios
        assert report.scenarios[name]["accepted_state_preserved"] is True

    # Stale-client scenario must refuse pre-rotation writers.
    stale = report.scenarios["stale_client"]
    assert stale["pre_rotation_writer_refused"] is True
    assert stale["invalidates_previous"] is True

    # Corrupt copy must fail closed.
    assert report.scenarios["corrupt_copy"]["verification_failed_closed"] is True

    # Partial restore must fail closed.
    assert report.scenarios["partial_restore"]["failed_closed"] is True


def test_backup_rows_recorded_when_possible(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    snapshot = svc.create_backup(label="row")
    with open_duckdb_connection(db) as connection:
        row = connection.execute(
            "SELECT backup_id, status, artifact_digest FROM backup_snapshots "
            "WHERE backup_id = ?",
            [snapshot.backup_id],
        ).fetchone()
    assert row is not None
    # DuckDB may return tuple or mapping.
    values = list(row) if not hasattr(row, "keys") else [row[k] for k in row.keys()]
    assert snapshot.backup_id in values
    assert "verified" in values


def test_unverified_backup_restore_refused(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    snapshot = svc.create_backup()
    artifact = Path(str(snapshot.body["artifact_path"]))
    artifact.write_bytes(b"not-a-duckdb-file")
    target = tmp_path / "should-not-exist.duckdb"
    with pytest.raises(ControlPlaneBackupIntegrityError):
        svc.restore(snapshot, target_path=target, rotate_generation=False)
    assert not target.exists() or target.stat().st_size != len(b"not-a-duckdb-file")


def test_snapshot_from_metadata_roundtrip(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    svc = _service(db, tmp_path)
    snapshot = svc.create_backup()
    meta_path = Path(str(snapshot.body["meta_path"]))
    loaded = svc.verify_backup(meta_path)
    assert loaded.verified is True
    assert isinstance(snapshot, BackupSnapshot)
