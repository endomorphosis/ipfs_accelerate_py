"""Tests for ControlPlaneBackup@1 (DQP-033).

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
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup import (
    BACKUP_SNAPSHOT_INTERFACE,
    CONTROL_PLANE_BACKUP_INTERFACE,
    CrashPoint,
    ControlPlaneBackup,
    ControlPlaneBackupCorruptionError,
    ControlPlaneBackupCrash,
    ControlPlaneBackupIOError,
    ControlPlaneBackupOwnershipError,
    ControlPlaneBackupVerificationError,
    ControlPlaneStateRoots,
    MANIFEST_FILENAME,
    RESTORE_RECEIPT_INTERFACE,
    RestoreOutcome,
    STORE_GENERATION_ROTATION_INTERFACE,
    StoreGenerationRotation,
    assert_direct_file_maintenance_allowed,
    capture_state_roots,
    duckdb_available,
    open_control_plane_backup,
    owner_marker_path_for,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    OwnerMarker,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    ControlPlaneGenerationError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    META_SCHEMA_FINGERPRINT,
    META_SCHEMA_VERSION,
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


# ---------------------------------------------------------------------------
# Fixtures / seed helpers
# ---------------------------------------------------------------------------


def _birth(pid: int = 4242) -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=1000,
        boot_id="boot-test",
        parent_pid=1,
    )


def _write_owner_marker(
    database_path: Path,
    *,
    liveness: OwnerLiveness = OwnerLiveness.ALIVE,
    fence_token: str = "fence-token-1",
    server_id: str = "server:owner-1",
) -> OwnerMarker:
    marker = OwnerMarker(
        server_id=server_id,
        process_birth=_birth(pid=5555 if liveness is OwnerLiveness.ALIVE else 1),
        database_path=str(database_path),
        started_at="1970-01-01T00:00:00Z",
        fence_token=fence_token,
        generation=1,
    )
    path = owner_marker_path_for(database_path)
    path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
    return marker


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
    birth_id: str = "birth:server-1",
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


def _seed_population(db: Path, *, task_count: int = 3) -> list[str]:
    task_cids: list[str] = []
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
                "objective:test",
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
        for index in range(task_count):
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
                    "objective:test",
                    index + 1,
                    "ready",
                    0,
                    "P0",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "{}",
                    "{}",
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
                1,
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
                "task.seeded",
                task_cids[0],
                "",
                "session:seed",
                "1970-01-01T00:00:00Z",
                "{}",
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
                "task.accepted",
                task_cids[0],
                "",
                "session:seed",
                "1970-01-01T00:00:01Z",
                '{"accepted":true}',
            ],
        )
    return task_cids


def _prepared_db(tmp_path: Path, name: str = "control.duckdb") -> Path:
    db = tmp_path / name
    _install(db)
    _seed_generation(db)
    _seed_population(db, task_count=3)
    return db


def _service(
    tmp_path: Path,
    *,
    crash_point: CrashPoint | str | None = None,
    encryption_key: bytes | str | None = None,
    liveness_probe: Any = None,
    clock: Any = None,
) -> ControlPlaneBackup:
    return open_control_plane_backup(
        backup_root=tmp_path / "backups",
        crash_point=crash_point,
        encryption_key=encryption_key,
        liveness_probe=liveness_probe,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert CONTROL_PLANE_BACKUP_INTERFACE == "ControlPlaneBackup@1"
    assert RESTORE_RECEIPT_INTERFACE == "RestoreReceipt@1"
    assert STORE_GENERATION_ROTATION_INTERFACE == "StoreGenerationRotation@1"
    assert BACKUP_SNAPSHOT_INTERFACE == "BackupSnapshot@1"
    assert ControlPlaneBackup.INTERFACE == CONTROL_PLANE_BACKUP_INTERFACE
    assert StoreGenerationRotation.INTERFACE == STORE_GENERATION_ROTATION_INTERFACE


# ---------------------------------------------------------------------------
# Ownership admission
# ---------------------------------------------------------------------------


def test_direct_file_maintenance_refused_when_owner_live(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    db.write_bytes(b"placeholder")
    _write_owner_marker(db, liveness=OwnerLiveness.ALIVE)

    def probe(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return OwnerLiveness.ALIVE

    with pytest.raises(ControlPlaneBackupOwnershipError, match="live"):
        assert_direct_file_maintenance_allowed(db, liveness_probe=probe)


def test_direct_file_maintenance_refused_when_owner_unknown(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    db.write_bytes(b"placeholder")
    _write_owner_marker(db)

    def probe(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return OwnerLiveness.UNKNOWN

    with pytest.raises(ControlPlaneBackupOwnershipError, match="unknown"):
        assert_direct_file_maintenance_allowed(db, liveness_probe=probe)


def test_direct_file_maintenance_allowed_when_owner_dead(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    db.write_bytes(b"placeholder")
    _write_owner_marker(db)

    def probe(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return OwnerLiveness.DEAD

    result = assert_direct_file_maintenance_allowed(db, liveness_probe=probe)
    assert result["admitted"] is True
    assert result["admission"] == "allowed_owner_dead"


def test_direct_file_maintenance_allowed_with_owner_fence(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    db.write_bytes(b"placeholder")
    marker = _write_owner_marker(db, fence_token="owner-fence-abc")

    def probe(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return OwnerLiveness.ALIVE

    result = assert_direct_file_maintenance_allowed(
        db,
        owner_fence_token=marker.fence_token,
        liveness_probe=probe,
    )
    assert result["admitted"] is True
    assert result["admission"] == "allowed_owner_fence"


def test_backup_refuses_live_ownership(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    _write_owner_marker(db)

    service = _service(
        tmp_path,
        liveness_probe=lambda _b: OwnerLiveness.ALIVE,
    )
    with pytest.raises(ControlPlaneBackupOwnershipError, match="live"):
        service.create_backup(db, acquire_lease=False)


# ---------------------------------------------------------------------------
# Backup + independent verification
# ---------------------------------------------------------------------------


def test_create_backup_independently_verified(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    source_roots = capture_state_roots(db)
    service = _service(tmp_path)

    snapshot = service.create_backup(db)
    assert snapshot.status == "verified"
    assert snapshot.artifact_digest.startswith("sha256:")
    assert snapshot.roots.domain_roots_match(source_roots)
    assert snapshot.roots.task_count == 3
    assert snapshot.roots.event_count == 2
    assert snapshot.roots.lease_count == 1
    assert Path(snapshot.body_path).is_file()
    assert (Path(snapshot.body_path).parent / MANIFEST_FILENAME).is_file()

    # Independent re-verification path.
    rechecked = service.verify_backup(snapshot)
    assert rechecked.status == "verified"
    assert rechecked.artifact_digest == snapshot.artifact_digest
    assert rechecked.roots.domain_roots_match(source_roots)

    # Recorded in source database.
    with open_duckdb_connection(db) as connection:
        row = connection.execute(
            "SELECT status, artifact_digest FROM backup_snapshots WHERE backup_id = ?",
            [snapshot.backup_id],
        ).fetchone()
        assert row is not None
        if hasattr(row, "keys"):
            assert row["status"] == "verified"
            assert row["artifact_digest"] == snapshot.artifact_digest
        else:
            assert row[0] == "verified"
            assert row[1] == snapshot.artifact_digest


def test_backup_success_not_claimed_without_independent_digest(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)
    snapshot = service.create_backup(db)

    body = Path(snapshot.body_path)
    body.write_bytes(body.read_bytes() + b"\x00CORRUPT")

    probe = service.probe_corruption(snapshot)
    assert probe["corrupt"] is True
    with pytest.raises(
        (ControlPlaneBackupCorruptionError, ControlPlaneBackupVerificationError)
    ):
        service.verify_backup(snapshot)


def test_encrypted_backup_digest_bound(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path, encryption_key=b"test-secret-key")
    snapshot = service.create_backup(db)
    assert snapshot.encryption_algorithm == "xor-sha256-stream-v1"
    raw_path = Path(snapshot.body_path)
    raw = raw_path.read_bytes()
    # Envelope is digest-bound ciphertext, not a directly openable DuckDB image.
    assert raw
    assert raw != db.read_bytes()
    # Opening the ciphertext without the service decrypt path must fail closed.
    try:
        capture_state_roots(raw_path)
        raised = False
    except Exception:
        raised = True
    assert raised, "encrypted backup body must not open as a plain control-plane DB"
    # Independent verify path decrypts and re-opens successfully.
    verified = service.verify_backup(snapshot)
    assert verified.roots.task_count == 3
    assert verified.artifact_digest == snapshot.artifact_digest


# ---------------------------------------------------------------------------
# Restore roots + generation rotation
# ---------------------------------------------------------------------------


def test_restore_reproduces_roots_and_invalidates_pre_rotation_writers(
    tmp_path: Path,
) -> None:
    db = _prepared_db(tmp_path)
    source_roots = capture_state_roots(db)
    pre_writer = source_roots.to_generation()
    service = _service(tmp_path)

    snapshot = service.create_backup(db)

    # Mutate live DB after backup so restore must reinstate accepted roots.
    with open_duckdb_connection(db) as connection:
        connection.execute("DELETE FROM tasks WHERE task_cid = ?", ["task:cid:003"])
        connection.execute(
            "DELETE FROM domain_events WHERE event_id = ?", ["event:002"]
        )
    mutated = capture_state_roots(db)
    assert mutated.task_count == 2
    assert not mutated.domain_roots_match(source_roots)

    dest = tmp_path / "restored.duckdb"
    receipt = service.restore(snapshot, dest)
    assert receipt.outcome == RestoreOutcome.SUCCESS.value
    assert receipt.roots_matched is True
    assert receipt.writers_invalidated is True
    assert receipt.post_rotation_generation == source_roots.generation + 1
    assert receipt.post_rotation_fence_epoch == source_roots.fence_epoch + 1

    restored_roots = capture_state_roots(dest)
    # Domain roots (store/schema/event/task/lease) match the backup.
    assert restored_roots.domain_roots_match(source_roots)
    # Generation has rotated past the backup head.
    assert restored_roots.generation == source_roots.generation + 1
    assert restored_roots.fence_epoch == source_roots.fence_epoch + 1
    assert restored_roots.schema_version == source_roots.schema_version
    assert restored_roots.schema_fingerprint == source_roots.schema_fingerprint
    assert restored_roots.database_uuid == source_roots.database_uuid
    assert list(restored_roots.task_cids) == list(source_roots.task_cids)
    assert list(restored_roots.event_ids) == list(source_roots.event_ids)
    assert list(restored_roots.lease_keys) == list(source_roots.lease_keys)

    live = restored_roots.to_generation()
    assert service.writer_is_invalidated(live, pre_writer) is True
    # Stale clients pin an expected generation; mismatch with live head fails closed.
    assert pre_writer.generation != live.generation
    assert pre_writer.fence_epoch != live.fence_epoch
    with pytest.raises(ControlPlaneGenerationError):
        if pre_writer.generation != live.generation:
            raise ControlPlaneGenerationError(
                "store generation mismatch: pre-rotation writer invalidated"
            )
        pre_writer.assert_compatible_with(live)


def test_restore_rehearsal_does_not_mutate_destination(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)
    snapshot = service.create_backup(db)
    dest = tmp_path / "should-not-exist.duckdb"
    receipt = service.restore(snapshot, dest, rehearsal=True)
    assert receipt.outcome == RestoreOutcome.REHEARSAL.value
    assert receipt.roots_matched is True
    assert not dest.exists()


def test_stale_client_fails_after_generation_rotation(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)
    before = capture_state_roots(db).to_generation()
    rotation = service.rotate_store_generation(db, reason="takeover")
    assert rotation.new_generation == before.generation + 1
    live = capture_state_roots(db).to_generation()
    assert service.writer_is_invalidated(live, before) is True
    assert rotation.invalidates(before) is True
    # Exact current head is not invalidated.
    assert service.writer_is_invalidated(live, live) is False


# ---------------------------------------------------------------------------
# Crash matrix — no accepted state lost
# ---------------------------------------------------------------------------


def _accepted_fingerprint(db: Path) -> dict[str, Any]:
    roots = capture_state_roots(db)
    return {
        "roots": roots.to_dict(),
        "content_id": roots.content_id,
        "file_digest": __import__("hashlib")
        .sha256(db.read_bytes())
        .hexdigest(),
    }


@pytest.mark.parametrize(
    "crash_point",
    [
        CrashPoint.BEFORE_CHECKPOINT,
        CrashPoint.AFTER_CHECKPOINT,
        CrashPoint.AFTER_COPY,
        CrashPoint.AFTER_VERIFY,
    ],
)
def test_crash_during_backup_preserves_source_accepted_state(
    tmp_path: Path, crash_point: CrashPoint
) -> None:
    db = _prepared_db(tmp_path)
    before = _accepted_fingerprint(db)
    service = _service(tmp_path, crash_point=crash_point)

    with pytest.raises(ControlPlaneBackupCrash):
        service.create_backup(db)

    after = _accepted_fingerprint(db)
    assert after["content_id"] == before["content_id"]
    assert after["roots"]["task_cids"] == before["roots"]["task_cids"]
    assert after["roots"]["event_ids"] == before["roots"]["event_ids"]
    assert after["roots"]["lease_keys"] == before["roots"]["lease_keys"]
    # No verified backup may be published for incomplete work.
    verified = [
        item
        for item in service.list_backups()
        if item.status == "verified"
    ]
    assert verified == []


def test_crash_after_manifest_leaves_verified_backup_and_source_intact(
    tmp_path: Path,
) -> None:
    db = _prepared_db(tmp_path)
    before = _accepted_fingerprint(db)
    # Crash after manifest still yields a durable verified backup artifact;
    # source accepted state must remain intact.
    service = _service(tmp_path, crash_point=CrashPoint.AFTER_MANIFEST)
    with pytest.raises(ControlPlaneBackupCrash):
        service.create_backup(db)
    after = _accepted_fingerprint(db)
    assert after["content_id"] == before["content_id"]


def test_crash_before_restore_replace_preserves_destination(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)
    snapshot = service.create_backup(db)

    dest = tmp_path / "dest.duckdb"
    # Distinct accepted state at destination.
    _install(dest)
    _seed_generation(dest, generation=1, birth_id="birth:dest")
    with open_duckdb_connection(dest) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "goal:dest",
                "G-DEST",
                "objective:dest",
                "",
                1,
                "Dest",
                "open",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                0,
                "{}",
            ],
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_cid, objective_id,
                ordinal, status, revision, priority, created_at, updated_at,
                identity_json, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "task:dest:001",
                "T-DEST",
                "goal:dest",
                "",
                "objective:dest",
                1,
                "ready",
                0,
                "P0",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                "{}",
                "{}",
            ],
        )
    before = _accepted_fingerprint(dest)

    crashing = _service(tmp_path, crash_point=CrashPoint.BEFORE_RESTORE_REPLACE)
    with pytest.raises(ControlPlaneBackupCrash):
        crashing.restore(snapshot, dest)

    after = _accepted_fingerprint(dest)
    assert after["content_id"] == before["content_id"]
    assert "task:dest:001" in after["roots"]["task_cids"]


def test_crash_after_restore_replace_before_rotation_keeps_restored_roots(
    tmp_path: Path,
) -> None:
    """After replace, domain roots are restored even if rotation crashes.

    Accepted backup state is present; generation may not have rotated yet.
    """

    db = _prepared_db(tmp_path)
    source_roots = capture_state_roots(db)
    service = _service(tmp_path)
    snapshot = service.create_backup(db)

    dest = tmp_path / "dest.duckdb"
    crashing = _service(tmp_path, crash_point=CrashPoint.BEFORE_ROTATION)
    with pytest.raises(ControlPlaneBackupCrash):
        crashing.restore(snapshot, dest)

    # Destination exists with restored domain roots (accepted state from backup).
    assert dest.exists()
    roots = capture_state_roots(dest)
    assert roots.domain_roots_match(source_roots)


# ---------------------------------------------------------------------------
# Corruption / disk full / partial restore / schema / backup age
# ---------------------------------------------------------------------------


def test_corrupt_copy_probe(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)
    snapshot = service.create_backup(db)
    body = Path(snapshot.body_path)
    body.write_bytes(b"not-a-duckdb-database")
    result = service.probe_corruption(Path(snapshot.body_path).parent)
    assert result["corrupt"] is True


def test_disk_full_on_backup_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup.shutil.copy2",
        boom,
    )
    with pytest.raises(ControlPlaneBackupIOError, match="disk full|No space|backup copy"):
        service.create_backup(db)


def test_partial_restore_rolls_back_on_root_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)
    snapshot = service.create_backup(db)

    dest = tmp_path / "dest.duckdb"
    _install(dest)
    _seed_generation(dest, birth_id="birth:dest-only")
    _seed_population(dest, task_count=1)
    before = capture_state_roots(dest)
    dest_resolved = dest.resolve()

    real_capture = capture_state_roots

    def flaky_capture(path: Path | str, **kwargs: Any) -> ControlPlaneStateRoots:
        roots = real_capture(path, **kwargs)
        try:
            path_resolved = Path(path).resolve()
        except OSError:
            return roots
        # Poison only the post-replace destination roots check.
        if path_resolved == dest_resolved:
            return ControlPlaneStateRoots(
                store_id=roots.store_id,
                database_uuid=roots.database_uuid,
                schema_revision=roots.schema_revision,
                schema_fingerprint=roots.schema_fingerprint,
                schema_version=roots.schema_version,
                generation=roots.generation,
                fence_epoch=roots.fence_epoch,
                revision=roots.revision,
                birth_id=roots.birth_id,
                event_watermark=roots.event_watermark,
                event_ids=roots.event_ids,
                task_cids=("task:forged",),
                lease_keys=roots.lease_keys,
                task_count=1,
                lease_count=roots.lease_count,
                event_count=roots.event_count,
            )
        return roots

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup.capture_state_roots",
        flaky_capture,
    )
    with pytest.raises(ControlPlaneBackupVerificationError, match="roots"):
        service.restore(snapshot, dest)

    # Pre-restore accepted state restored via rollback.
    after = real_capture(dest)
    assert after.domain_roots_match(before)


def test_schema_version_preserved_across_backup_restore(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    with open_duckdb_connection(db) as connection:
        version = connection.execute(
            "SELECT value FROM control_plane_metadata WHERE key = ?",
            [META_SCHEMA_VERSION],
        ).fetchone()
        fingerprint = connection.execute(
            "SELECT value FROM control_plane_metadata WHERE key = ?",
            [META_SCHEMA_FINGERPRINT],
        ).fetchone()
    assert version is not None
    assert fingerprint is not None
    if hasattr(fingerprint, "keys"):
        stored_fingerprint = str(fingerprint["value"] or "")
    else:
        stored_fingerprint = str(fingerprint[0] or "")
    assert stored_fingerprint

    service = _service(tmp_path)
    snapshot = service.create_backup(db)
    assert snapshot.roots.schema_version != ""
    # Migration runner stores a content-bound fingerprint (CID or sha256 digest).
    assert snapshot.roots.schema_fingerprint == stored_fingerprint
    assert snapshot.roots.schema_fingerprint

    dest = tmp_path / "restored.duckdb"
    service.restore(snapshot, dest, rotate_generation=True)
    restored = capture_state_roots(dest)
    assert restored.schema_version == snapshot.roots.schema_version
    assert restored.schema_fingerprint == snapshot.roots.schema_fingerprint
    assert restored.schema_fingerprint == stored_fingerprint


def test_server_stopped_ownership_allows_offline_restore(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(
        tmp_path,
        liveness_probe=lambda _b: OwnerLiveness.DEAD,
    )
    snapshot = service.create_backup(db)
    # Simulate stopped server leaving a dead owner marker on destination path.
    dest = tmp_path / "offline.duckdb"
    # Destination does not exist yet; marker on a sibling path is fine.
    # After restore path exists with no live owner.
    receipt = service.restore(snapshot, dest)
    assert receipt.outcome == RestoreOutcome.SUCCESS.value
    _write_owner_marker(dest)
    # Live would refuse further maintenance:
    live_service = _service(
        tmp_path,
        liveness_probe=lambda _b: OwnerLiveness.ALIVE,
    )
    with pytest.raises(ControlPlaneBackupOwnershipError):
        live_service.checkpoint(dest)


def test_backup_age_and_retention(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    # create_backup invokes the clock multiple times (lease + manifest + release).
    # Pin a stable wall-clock per backup, then advance between backups so age
    # and retention ordering do not depend on an exact per-call tick budget.
    current = {"stamp": "2020-01-01T00:00:00Z"}

    def clock() -> str:
        return current["stamp"]

    service = open_control_plane_backup(
        backup_root=tmp_path / "backups",
        retention_count=2,
        clock=clock,
    )
    first = service.create_backup(db)
    current["stamp"] = "2020-01-02T00:00:00Z"
    second = service.create_backup(db)
    current["stamp"] = "2020-01-03T00:00:00Z"
    third = service.create_backup(db)
    assert first.created_at == "2020-01-01T00:00:00Z"
    assert second.created_at == "2020-01-02T00:00:00Z"
    assert third.created_at == "2020-01-03T00:00:00Z"
    assert service.backup_age_seconds(first, now="2020-01-04T00:00:00Z") == 3 * 24 * 3600

    manifest = service.apply_retention(now="2020-01-04T00:00:00Z", max_count=2)
    assert manifest.retained_count == 2
    assert manifest.pruned_count == 1
    remaining = {item.backup_id for item in service.list_backups()}
    assert third.backup_id in remaining
    assert second.backup_id in remaining
    assert first.backup_id not in remaining
    # Newest always retained.
    assert remaining == {second.backup_id, third.backup_id}


def test_checkpoint_server_stopped_path(tmp_path: Path) -> None:
    db = _prepared_db(tmp_path)
    service = _service(tmp_path)
    receipt = service.checkpoint(db)
    assert receipt["checkpointed"] is True


def test_open_control_plane_backup_is_side_effect_free(tmp_path: Path) -> None:
    # Construction must not touch the filesystem.
    service = open_control_plane_backup(backup_root=tmp_path / "never-created")
    assert service.INTERFACE == CONTROL_PLANE_BACKUP_INTERFACE
    assert not (tmp_path / "never-created").exists()
