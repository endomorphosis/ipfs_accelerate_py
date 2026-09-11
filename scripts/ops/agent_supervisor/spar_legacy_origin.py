"""Install a retained coherent SPAR capture as a durable native queue origin.

The caller must retain the actual capture session across this transaction.
Any interrupted transaction keeps a permanent native-profile requirement;
there is no legacy filesystem fallback, automatic rollback, claim release,
acceptance, signing operation, or source adoption here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import spar_merge_owner as role

SCHEMA = "spar/native-legacy-queue-origin@1"
REQUIRED_MARKER = "native-legacy-profile-required.json"
_FIELDS = {"schema", "database_uuid", "database_path", "manifest", "capture",
           "receipt_imports", "cursor_imports"}


def required_profile(queue_root, requested):
    """The durable marker requires validation; it never supplies admission."""
    from .spar_merge_owner_handoff import LEGACY_PROFILE
    marker = Path(queue_root) / REQUIRED_MARKER
    return LEGACY_PROFILE if not requested and (marker.exists() or marker.is_symlink()) else requested


def validate_record(record, *, database):
    from .spar_legacy_capture import SCHEMA as CAPTURE_SCHEMA

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
        or any(capture.get(key) is not False for key in (
            "callback_settled", "signing_authority", "source_admitted", "completion_authority"))
        or type(record["receipt_imports"]) is not list
        or type(record["cursor_imports"]) is not list
    ):
        raise role.SparMergeOwnerError("native migrated origin provenance differs")
    closure = capture.get("closure", {})
    if closure.get("pidfd_exit_observed") is not True or closure.get("native_actors_closed") is not True:
        raise role.SparMergeOwnerError("native migrated origin lacks positive process closure")
    if closure.get("cgroup_empty_observed") is True:
        if closure.get("workflow_sentinel") is not None:
            raise role.SparMergeOwnerError("native migrated cgroup closure is contradictory")
    else:
        from .spar_legacy_capture import SENTINEL_CODE
        sentinel = closure.get("workflow_sentinel")
        if (type(sentinel) is not dict
            or sentinel.get("role") != "workflow_sentinel_without_native_state_access"
            or sentinel.get("code_sha256") != hashlib.sha256(SENTINEL_CODE.encode()).hexdigest()
            or type(sentinel.get("birth")) is not dict
            or sentinel.get("observed_cgroup_members") != [sentinel["birth"].get("pid")]):
            raise role.SparMergeOwnerError("native migrated retained population is unqualified")
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
    if len(cursors) != len(manifest["cursor_imports"]):
        raise role.SparMergeOwnerError("native migrated cursor population differs")
    for cursor, spec in zip(cursors, manifest["cursor_imports"]):
        role._closed(cursor, {"scope_cid", "cursors", "state_cid"})
        if cursor["scope_cid"] != spec["scope_cid"] or cursor["state_cid"] != role._cid(cursor["cursors"]):
            raise role.SparMergeOwnerError("native migrated cursor content differs")
    return manifest


def _sync_directory(directory):
    descriptor = role._open_directory(directory)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_captured_queue(captured, prepared):
    """One local held transaction; neither parameter can be an audit JSON object."""
    from .spar_legacy_capture import CoherentLegacyCapture
    from .spar_merge_owner_handoff import ORIGIN_TABLE

    if type(captured) is not CoherentLegacyCapture or type(prepared) is not role.PreparedQueueStore:
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
    record = {"schema": SCHEMA, "database_uuid": prepared.database_uuid,
              "database_path": str(database), "manifest": manifest, "capture": receipt,
              "receipt_imports": list(prepared.receipt_imports), "cursor_imports": list(prepared.cursor_imports)}
    validate_record(record, database=database)
    role._refuse_observed_input_locks({"candidate": role._file_identity(candidate.lstat())})
    role.verify_installed_schema(candidate)
    with role.open_duckdb_connection(candidate, prefer_quack=False) as connection:
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
        validated_identity = role._file_identity(candidate.lstat())
    if candidate.with_suffix(candidate.suffix + ".wal").exists():
        raise role.SparMergeOwnerError("prepared native candidate still has a WAL")
    body = {"schema": "spar/native-legacy-profile-requirement@1", "origin_cid": role._cid(record),
            "capture_cid": role._cid(receipt), "database_path": str(database),
            "captured_path": str(captured.path), "callback_settled": False, "completion_authority": False}
    # Ensure the complete capture directory entries are durable before any
    # canonical change. Input copies were individually fsynced by copy_entry.
    for directory, _, _ in os.walk(captured.path, topdown=False):
        _sync_directory(Path(directory))
    _sync_directory(captured.path.parent)
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
    if role._file_identity(stat_before) != validated_identity:
        raise role.SparMergeOwnerError("prepared native database changed after validation")
    digest = hashlib.sha256()
    descriptor = role._open_regular(candidate.parent, candidate.name)
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if role._file_identity(os.fstat(descriptor)) != role._file_identity(stat_before):
            raise role.SparMergeOwnerError("prepared native database changed")
    finally:
        os.close(descriptor)
    staging = queue_root / ".native-legacy-database.prepared"
    role.copy_entry(candidate.parent, {"path": candidate.name, "size_bytes": stat_before.st_size,
                    "sha256": digest.hexdigest()}, staging)
    staging_entry = {"path": staging.name, "size_bytes": stat_before.st_size,
                     "sha256": digest.hexdigest()}
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
    _sync_directory(queue_root)
    gate()
    return {"installed": True, "origin_cid": role._cid(record), "database_uuid": prepared.database_uuid,
            "database_path": str(database), "callback_settled": False,
            "source_admitted": False, "completion_authority": False}
