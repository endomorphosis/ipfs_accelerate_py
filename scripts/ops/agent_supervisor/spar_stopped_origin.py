"""Distinct stopped-state origin installer; live-capture admission is unchanged.

Only StoppedQueueCapture from the fresh fenced producer enters this transaction.
Original queue bytes, cursor positions, signatures and unknown claims are kept.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from . import spar_merge_owner as role

SCHEMA = "spar/native-stopped-queue-origin@1"
REQUIRED_MARKER = "native-stopped-profile-required.json"
_FIELDS = {"schema", "database_uuid", "database_path", "manifest", "capture",
           "receipt_imports", "cursor_imports", "cursor_source_bytes"}


def required_profile(queue_root, requested):
    """The durable marker requires validation; it never supplies admission."""
    from .spar_merge_owner_handoff import STOPPED_PROFILE
    marker = Path(queue_root) / REQUIRED_MARKER
    if not (marker.exists() or marker.is_symlink()):
        return requested
    legacy = Path(queue_root) / "native-legacy-profile-required.json"
    if legacy.exists() or legacy.is_symlink() or requested not in (None, "", STOPPED_PROFILE):
        raise role.SparMergeOwnerError("stopped native origin requirement conflicts with requested profile")
    return STOPPED_PROFILE


def validate_record(record, *, database):
    from .spar_stopped_capture import SCHEMA as CAPTURE_SCHEMA

    role._closed(record, _FIELDS)
    if record["schema"] != SCHEMA or record["database_path"] != str(database):
        raise role.SparMergeOwnerError("native migrated origin path or schema differs")
    manifest = role.validate_manifest(record["manifest"])
    capture = record["capture"]
    if (
        type(capture) is not dict
        or capture.get("schema") != CAPTURE_SCHEMA
        or capture.get("manifest") != manifest
        or capture.get("queue_root") != str(database.parent)
        or capture.get("capture_coherent") is not True
        or capture.get("consumer_processes_closed") is not True
        or type(capture.get("preserved_inventory_cid")) is not str
        or any(capture.get(key) is not False for key in (
            "callback_settled", "signing_authority", "source_admitted", "completion_authority"))
        or type(record["receipt_imports"]) is not list
        or type(record["cursor_imports"]) is not list
    ):
        raise role.SparMergeOwnerError("native migrated origin provenance differs")
    from .spar_stopped_capture import (
        ADMISSION_SCHEMA, KEEPER_SCHEMA, PROCESS_FIELDS, validate_task_observation,
    )
    admission = capture.get("task_admission")
    if type(admission) is not dict or admission.get("schema") != ADMISSION_SCHEMA:
        raise role.SparMergeOwnerError("fresh stopped task admission missing")
    observed = admission.get("observation")
    owner = admission.get("closed_owner")
    if type(observed) is not dict or type(owner) is not dict:
        raise role.SparMergeOwnerError("fresh stopped canonical observation missing")
    reconstructed = validate_task_observation(
        observed, owner=owner, bootstrap=observed.get("bootstrap"),
        source=capture.get("source"), task_files=admission.get("canonical_task_files"))
    if admission != reconstructed or capture.get("owner_identity") != admission["owner_identity"]:
        raise role.SparMergeOwnerError("fresh stopped task admission seal differs")
    succession = capture.get("keeper_succession", {})
    closure = capture.get("closure", {})
    from .spar_legacy_capture import SENTINEL_CODE
    keeper = succession.get("keeper")
    if (succession.get("schema") != KEEPER_SCHEMA
        or type(keeper) is not dict or set(keeper) != PROCESS_FIELDS
        or succession.get("old_controller_pidfd_exit") is not True
        or succession.get("old_helper_pidfd_exit") is not True
        or succession.get("keeper_code_sha256") != hashlib.sha256(SENTINEL_CODE.encode()).hexdigest()
        or succession.get("callback_settled") is not False
        or succession.get("task_or_store_authority") is not False
        or closure.get("native_actors_closed") is not True
        or closure.get("workflow_keeper") != keeper
        or closure.get("observed_cgroup_members") != [keeper["pid"]]):
        raise role.SparMergeOwnerError("fresh stopped keeper succession differs")
    receipts = record["receipt_imports"]
    expected = manifest["receipt_imports"]
    if len(receipts) != len(expected):
        raise role.SparMergeOwnerError("native migrated receipt population differs")
    for receipt, spec in zip(receipts, expected):
        role._closed(receipt, {"receipt_key", "revision", "receipt_cid", "receipt"})
        if (
            any(receipt[key] != spec[key] for key in ("receipt_key", "revision", "receipt_cid"))
            or role._cid(receipt["receipt"]) != receipt["receipt_cid"]
        ):
            raise role.SparMergeOwnerError("native migrated receipt content differs")
    cursors = record["cursor_imports"]
    sources = record["cursor_source_bytes"]
    if type(sources) is not list or len(cursors) != len(manifest["cursor_imports"]) or len(sources) != len(cursors):
        raise role.SparMergeOwnerError("native migrated cursor population differs")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    scopes = {role.recovery_scope_cid(
        store_id=manifest["store_id"], repository_id=manifest["repository_id"],
        target_branch=manifest["target_branch"], scope_binding=scope): scope
        for scope in manifest["scope_bindings"]}
    for cursor, spec, source in zip(cursors, manifest["cursor_imports"], sources):
        role._closed(cursor, {"scope_cid", "cursors", "state_cid"})
        if cursor["scope_cid"] != spec["scope_cid"] or cursor["state_cid"] != role._cid(cursor["cursors"]):
            raise role.SparMergeOwnerError("native migrated cursor content differs")
        role._closed(source, {"path", "hex"})
        entry = entries[spec["path"]]
        if (source["path"] != spec["path"] or type(source["hex"]) is not str
            or len(source["hex"]) != entry["size_bytes"] * 2):
            raise role.SparMergeOwnerError("native migrated cursor source differs")
        try:
            raw = bytes.fromhex(source["hex"])
        except ValueError as exc:
            raise role.SparMergeOwnerError("native migrated cursor source encoding differs") from exc
        if raw.hex() != source["hex"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise role.SparMergeOwnerError("native migrated cursor source digest differs")
        body = role._decode(raw)
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import _POST_MERGE_RECOVERY_CURSOR_SCHEMA
        from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
        scope = scopes[spec["scope_cid"]]
        if (body.get("schema") != _POST_MERGE_RECOVERY_CURSOR_SCHEMA
            or body.get("target_repository_id") != manifest["repository_id"]
            or body.get("target_branch") != manifest["target_branch"]
            or body.get("attempt_root") != scope["attempt_root"]
            or body.get("state_id") != content_identity({key: value for key, value in body.items() if key != "state_id"})
            or body.get("cursors") != cursor["cursors"]):
            raise role.SparMergeOwnerError("native migrated cursor differs from captured source bytes")
    return manifest


def _sync_directory(directory):
    descriptor = role._open_directory(directory)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_stopped_queue(captured, prepared):
    """Retain the exact candidate inode across validation and canonical copy."""
    from .spar_stopped_capture import StoppedQueueCapture
    if type(captured) is not StoppedQueueCapture or type(prepared) is not role.PreparedQueueStore:
        raise role.SparMergeOwnerError("native retained capture and prepared clone required")
    receipt = captured.require_current()
    root = Path(receipt["queue_root"])
    if (dict(prepared.manifest) != receipt["manifest"]
        or prepared.database_path.is_relative_to(root)
        or captured.path.is_relative_to(root)
        or prepared.database_path == captured.path / receipt["manifest"]["database"]):
        raise role.SparMergeOwnerError("native installation inputs do not bind a distinct preserved clone")
    # Opening then closing any descriptor of an already-owned POSIX-lock inode
    # would release this process's writer lock. The veto must precede the open.
    role._refuse_observed_input_locks({
        "candidate": role._file_identity(prepared.database_path.lstat())})
    descriptor = role._open_regular(prepared.database_path.parent, prepared.database_path.name)
    try:
        return _install_stopped_queue(captured, prepared, descriptor)
    finally:
        os.close(descriptor)


def _install_stopped_queue(captured, prepared, descriptor):
    """One local held transaction; neither parameter can be an audit JSON object."""
    from .spar_stopped_capture import StoppedQueueCapture
    from .spar_merge_owner_handoff import ORIGIN_TABLE

    if type(captured) is not StoppedQueueCapture or type(prepared) is not role.PreparedQueueStore:
        raise role.SparMergeOwnerError("native retained capture and prepared clone required")
    receipt = captured.require_current()
    manifest = receipt["manifest"]
    queue_root = Path(receipt["queue_root"])
    database = queue_root / "merge_queue.duckdb"
    candidate = prepared.database_path
    if (
        dict(prepared.manifest) != manifest
        or candidate.is_relative_to(queue_root)
        or captured.path.is_relative_to(queue_root)
        or candidate == captured.path / manifest["database"]
    ):
        raise role.SparMergeOwnerError("native installation inputs do not bind a distinct preserved clone")
    baseline = role._json(prepared.preserved_inventory)
    if (len(baseline) > role.MAX_INVENTORY_DIGEST_BYTES
        or "sha256:" + hashlib.sha256(baseline).hexdigest() != receipt["preserved_inventory_cid"]):
        raise role.SparMergeOwnerError("prepared preservation inventory differs from native capture")
    record = {"schema": SCHEMA, "database_uuid": prepared.database_uuid,
              "database_path": str(database), "manifest": manifest, "capture": receipt,
              "receipt_imports": list(prepared.receipt_imports), "cursor_imports": list(prepared.cursor_imports),
              "cursor_source_bytes": []}
    entries = {entry["path"]: entry for entry in manifest["files"]}
    for spec in manifest["cursor_imports"]:
        raw = role.copy_entry(captured.path, entries[spec["path"]], None)
        record["cursor_source_bytes"].append({"path": spec["path"], "hex": raw.hex()})
    validate_record(record, database=database)
    def retained_identity(*, writer=False):
        retained = os.fstat(descriptor)
        current = candidate.lstat()
        if role._file_identity(retained) != role._file_identity(current):
            raise role.SparMergeOwnerError("prepared native database pathname changed from retained inode")
        if writer and not role._writer_lock_held(candidate):
            raise role.SparMergeOwnerError("prepared native database writer is not bound to retained inode")
        return role._file_identity(retained)

    role._refuse_observed_input_locks({"candidate": retained_identity()})
    role.verify_installed_schema(candidate)
    retained_identity()
    with role.open_duckdb_connection(candidate, prefer_quack=False) as connection:
        retained_identity(writer=True)
        before = role.inventory(connection)
        role.require_preserved(prepared.preserved_inventory, before)
        if role._owner_metadata(connection)["database_uuid"] != prepared.database_uuid or ORIGIN_TABLE in before:
            raise role.SparMergeOwnerError("prepared origin is substituted or already initialized")
        connection.execute("BEGIN TRANSACTION")
        connection.execute("CREATE TABLE " + ORIGIN_TABLE + " (origin_cid VARCHAR PRIMARY KEY, origin_json VARCHAR NOT NULL)")
        connection.execute("INSERT INTO " + ORIGIN_TABLE + " VALUES (?,?)", [role._cid(record), role._json(record).decode()])
        connection.commit()
        connection.execute("CHECKPOINT")
        after = role.inventory(connection)
        role.require_preserved(before, {key: value for key, value in after.items() if key != ORIGIN_TABLE})
        validated_identity = retained_identity(writer=True)
    if candidate.with_suffix(candidate.suffix + ".wal").exists():
        raise role.SparMergeOwnerError("prepared native candidate still has a WAL")
    body = {"schema": "spar/native-stopped-profile-requirement@1", "origin_cid": role._cid(record),
            "capture_cid": role._cid(receipt), "database_path": str(database),
            "captured_path": str(captured.path), "callback_settled": False, "completion_authority": False}
    # Ensure the complete capture directory entries are durable before any
    # canonical change. Input copies were individually fsynced by copy_entry.
    for directory, _, _ in os.walk(captured.path.parent, topdown=False):
        _sync_directory(Path(directory))
    _sync_directory(captured.path.parent.parent)
    captured.require_current()
    marker = queue_root / REQUIRED_MARKER
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        content = role._json(body)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    _sync_directory(queue_root)
    # From this point all failures deliberately keep the requirement marker.
    # Future startup must validate an installed origin, never reopen legacy mode.
    session = captured._session

    def gate():
        if session._closed_gate() != receipt["closure"]:
            raise role.SparMergeOwnerError("native unit or closure changed during origin installation")

    gate()
    stat_before = candidate.lstat()
    if retained_identity() != validated_identity:
        raise role.SparMergeOwnerError("prepared native database changed after validation")
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    if retained_identity() != validated_identity:
        raise role.SparMergeOwnerError("prepared native database changed")
    staging = queue_root / ".native-stopped-database.prepared"
    role.copy_entry(candidate.parent, {"path": candidate.name, "size_bytes": stat_before.st_size,
                    "sha256": digest.hexdigest()}, staging)
    staging_entry = {"path": staging.name, "size_bytes": stat_before.st_size,
                     "sha256": digest.hexdigest()}
    session._retain_replacement_queue(staging)
    gate()
    role.copy_entry(queue_root, staging_entry, None, digest_only=True)
    # Verify the entire original tree except our two newly created records.
    original = role.file_inventory(queue_root)
    for name in (REQUIRED_MARKER, staging.name):
        original.pop(name, None)
    if original != captured._identities:
        raise role.SparMergeOwnerError("native queue changed before origin replacement")
    wal = database.with_suffix(database.suffix + ".wal")
    if manifest["wal"] is not None:
        retired_wal = candidate.parent / "retired-canonical-merge-queue.wal"
        if retired_wal.exists():
            raise role.SparMergeOwnerError("retired canonical WAL path already exists")
        os.rename(wal, retired_wal)
        _sync_directory(candidate.parent)
        _sync_directory(queue_root)
    gate()
    role.copy_entry(queue_root, staging_entry, None, digest_only=True)
    os.replace(staging, database)
    installed_info = database.lstat()
    session._installed_queue_inode = (installed_info.st_dev, installed_info.st_ino)
    _sync_directory(queue_root)
    gate()
    return {"installed": True, "origin_cid": role._cid(record), "database_uuid": prepared.database_uuid,
            "database_path": str(database), "callback_settled": False,
            "source_admitted": False, "completion_authority": False}


STOPPED_PROFILE = "native-stopped-capture@1"

def load_stopped_origin(database, *, repository_id, target_branch, store_id, scopes, profile=STOPPED_PROFILE,
                 launch_context=None):
    from .spar_merge_owner_handoff import ORIGIN_TABLE
    if profile != STOPPED_PROFILE:
        raise role.SparMergeOwnerError("explicit stopped-state profile required")
    # A positive observed lock veto precedes every canonical DB open. Absence
    # does not certify legacy closure; only this role's canonical origin is read.
    info = database.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise role.SparMergeOwnerError(
            "native queue database is not an owned regular file"
        )
    role._refuse_observed_input_locks({"database": role._file_identity(info)})
    with role.open_duckdb_connection(database) as connection:
        names = {
            r[0]
            for r in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog=current_database() AND table_schema='main'"
            ).fetchall()
        }
        if ORIGIN_TABLE not in names:
            raise role.SparMergeOwnerError(
                "existing queue has no native fresh origin; legacy capture admission required"
            )
        columns = connection.execute(
            "SELECT column_name,data_type,is_nullable FROM information_schema.columns WHERE table_catalog=current_database() AND table_schema='main' AND table_name=? ORDER BY ordinal_position",
            [ORIGIN_TABLE],
        ).fetchall()
        if [tuple(row[i] for i in range(3)) for row in columns] != [
            ("origin_cid", "VARCHAR", "NO"),
            ("origin_json", "VARCHAR", "NO"),
        ]:
            raise role.SparMergeOwnerError("native fresh origin schema differs")
        rows = connection.execute(
            "SELECT origin_cid,origin_json FROM " + ORIGIN_TABLE + " LIMIT 2"
        ).fetchall()
        if len(rows) != 1:
            raise role.SparMergeOwnerError(
                "native fresh origin is missing or ambiguous"
            )
        record = role._decode(str(rows[0][1]).encode())
        manifest = validate_record(record, database=database)
        if (
            rows[0][0] != role._cid(record)
            or record["schema"] != SCHEMA
            or record["database_path"] != str(database)
        ):
            raise role.SparMergeOwnerError("native fresh origin identity differs")
        metadata = role._owner_metadata(connection)
        if metadata is None or metadata["database_uuid"] != record["database_uuid"]:
            raise role.SparMergeOwnerError(
                "native fresh UUID differs from preserved origin"
            )
        expected = {
            "repository_id": repository_id,
            "target_branch": target_branch,
            "store_id": store_id,
        }
        if any(manifest[key] != value for key, value in expected.items()):
            raise role.SparMergeOwnerError(
                "current native source namespace differs from fresh origin; explicit migration required"
            )
        transition = None
        if manifest["scope_bindings"] != scopes:
            if launch_context is None:
                raise role.SparMergeOwnerError(
                    "current native source namespace differs from fresh origin; explicit migration required")
            from .spar_legacy_launch_transition import qualify_transition
            try:
                transition = qualify_transition(origin=record, scopes=scopes, **launch_context)
            except role.SparMergeOwnerError as exc:
                raise role.SparMergeOwnerError(
                    "current native source namespace differs: " + str(exc)) from exc
        elif "legacy_merge_recovery_migrations" in names:
            migrated = connection.execute(
                "SELECT payload_cid FROM legacy_merge_recovery_migrations WHERE migration_id=?",
                ["spar-native-launch:" + role._cid(manifest)]).fetchall()
            if migrated:
                raise role.SparMergeOwnerError(
                    "native launch already transitioned away from the captured configuration")
        baseline = role.inventory(connection)
    role.verify_installed_schema(database)
    return role.PreparedQueueStore(
        database, manifest, record["database_uuid"], baseline,
        tuple(record.get("receipt_imports", ())), tuple(record.get("cursor_imports", ())),
        transition,
    )

