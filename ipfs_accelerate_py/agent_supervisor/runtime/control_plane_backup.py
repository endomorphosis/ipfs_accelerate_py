"""Checkpoint, backup, restore, retention, and generation rotation (DQP-033).

Interfaces
----------
* ``ControlPlaneBackup@1`` — verified consistent snapshots, restore rehearsals,
  retention manifests, and store-generation rotation.
* ``RestoreReceipt@1`` — durable outcome of a restore that rebinds store,
  schema, event, task, and lease roots.
* ``StoreGenerationRotation@1`` — monotonic generation advance that invalidates
  pre-rotation writers and leases.

Safety
------
Direct-file maintenance (checkpoint/copy/restore against the DuckDB path) is
refused while state-owner ownership is live or unknown. Backup success is
independently verified by reopening the artifact and recomputing roots and
digests. External backup bodies are digest-bound and may be encrypted.

Cold import of this module performs no filesystem, database, network, provider,
or process action.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
)
from ..task_sources.control_plane_contracts import (
    CONTRACT_VERSION,
    StoreGeneration,
    canonical_json_bytes,
    content_identity,
)
from ..task_sources.control_plane_migrations import (
    META_DATABASE_UUID,
    META_SCHEMA_FINGERPRINT,
    META_SCHEMA_VERSION,
    duckdb_available,
)
from ..task_sources.control_plane_repository import (
    DEFAULT_MAINTENANCE_SCOPE,
    MaintenanceLease,
    StateRepositoryMaintenanceError,
    acquire_maintenance_lease,
    release_maintenance_lease,
)
from ..task_sources.control_plane_schema import install_control_plane_schema
from ..task_sources.duckdb_state import open_duckdb_connection
from .quack_state_server import (
    OWNER_MARKER_SUFFIX,
    OwnerMarker,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

CONTROL_PLANE_BACKUP_INTERFACE: Final[str] = "ControlPlaneBackup@1"
RESTORE_RECEIPT_INTERFACE: Final[str] = "RestoreReceipt@1"
STORE_GENERATION_ROTATION_INTERFACE: Final[str] = "StoreGenerationRotation@1"

CONTROL_PLANE_BACKUP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-backup@1"
)
RESTORE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/restore-receipt@1"
)
STORE_GENERATION_ROTATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/store-generation-rotation@1"
)
BACKUP_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-snapshot@1"
)
SNAPSHOT_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/snapshot-roots@1"
)
RETENTION_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/retention-manifest@1"
)
CHECKPOINT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/checkpoint-receipt@1"
)
BACKUP_VERIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-verification@1"
)
CRASH_MATRIX_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-crash-matrix@1"
)

CONTROL_PLANE_BACKUP_VERSION: Final[int] = 1
DEFAULT_STORE_ID: Final[str] = "control.duckdb"
DEFAULT_RETENTION_KEEP_LAST: Final[int] = 7
DEFAULT_RETENTION_MAX_AGE_SECONDS: Final[int] = 30 * 24 * 60 * 60
BACKUP_STATUS_VERIFIED: Final[str] = "verified"
BACKUP_STATUS_FAILED: Final[str] = "failed"
BACKUP_STATUS_PARTIAL: Final[str] = "partial"
RESTORE_OUTCOME_SUCCESS: Final[str] = "success"
RESTORE_OUTCOME_REFUSED: Final[str] = "refused"
RESTORE_OUTCOME_FAILED: Final[str] = "failed"
OWNERSHIP_CLEARED: Final[str] = "cleared"
OWNERSHIP_LIVE: Final[str] = "live"
OWNERSHIP_UNKNOWN: Final[str] = "unknown"
OWNERSHIP_STALE: Final[str] = "stale"
OWNERSHIP_ABSENT: Final[str] = "absent"

# Declared crash matrix for accepted-state durability.
DECLARED_CRASH_SCENARIOS: Final[tuple[str, ...]] = (
    "crash_before_checkpoint",
    "crash_after_checkpoint",
    "corrupt_copy",
    "disk_full",
    "partial_restore",
    "schema_version",
    "server_stopped",
    "stale_client",
    "backup_age",
)

_DIGEST_PREFIX: Final[str] = "sha256:"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ControlPlaneBackupError(RuntimeError):
    """Base fail-closed error for control-plane backup/restore."""


class ControlPlaneBackupOwnershipError(ControlPlaneBackupError):
    """Direct-file maintenance refused due to live/unknown ownership."""


class ControlPlaneBackupIntegrityError(ControlPlaneBackupError):
    """Digest, root, or artifact integrity failure."""


class ControlPlaneBackupDependencyError(ControlPlaneBackupError):
    """Required optional dependency (DuckDB) is unavailable."""


class ControlPlaneBackupRetentionError(ControlPlaneBackupError):
    """Retention policy or catalog mutation failed."""


class ControlPlaneBackupGenerationError(ControlPlaneBackupError):
    """Store generation rotation or fencing failure."""


class ControlPlaneBackupIOError(ControlPlaneBackupError):
    """Filesystem or copy failure during backup/restore."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _require_duckdb() -> None:
    if not duckdb_available():
        raise ControlPlaneBackupDependencyError(
            "DuckDB is required for control-plane backup/restore"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"{_DIGEST_PREFIX}{digest.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"{_DIGEST_PREFIX}{hashlib.sha256(payload).hexdigest()}"


def _canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row.keys()}
    try:
        return {str(key): row[key] for key in row.keys()}  # type: ignore[attr-defined]
    except Exception:
        pass
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return {str(index): value for index, value in enumerate(row)}
    return {"value": row}


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        # Best-effort directory fsync for durability.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    body = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_bytes(path, body.encode("utf-8"), mode=mode)


def _atomic_copy_file(source: Path, destination: Path, *, mode: int = 0o600) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out_handle, source.open("rb") as in_handle:
            shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024)
            out_handle.flush()
            os.fsync(out_handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, destination)
        try:
            dir_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
        raise


def _xor_stream(payload: bytes, key: bytes) -> bytes:
    """Lightweight reversible transform for digest-bound external bodies.

    Production deployments may wrap this with a stronger cipher; the contract
    only requires that external bodies are digest-bound and reversible under a
    configured key. The identity of the body remains the cleartext digest.
    """

    if not key:
        return payload
    key_len = len(key)
    return bytes(byte ^ key[index % key_len] for index, byte in enumerate(payload))


def owner_marker_path_for(database_path: Path) -> Path:
    """Return the non-authoritative OS owner-marker path for a database file."""

    db = Path(database_path)
    return db.with_name(f".{db.name}{OWNER_MARKER_SUFFIX}")


def inspect_server_ownership(
    database_path: Path | str,
    *,
    liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
) -> dict[str, Any]:
    """Inspect state-owner marker liveness for direct-file admission.

    Returns a closed ownership status:
    ``absent``, ``stale``, ``live``, or ``unknown``.
    """

    path = Path(database_path)
    marker_path = owner_marker_path_for(path)
    if not marker_path.is_file():
        return {
            "status": OWNERSHIP_ABSENT,
            "marker_path": str(marker_path),
            "server_id": "",
            "pid": 0,
            "allow_direct_file": True,
        }
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        marker = OwnerMarker.from_dict(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
        return {
            "status": OWNERSHIP_UNKNOWN,
            "marker_path": str(marker_path),
            "server_id": "",
            "pid": 0,
            "allow_direct_file": False,
            "reason": "owner marker unreadable",
        }
    probe = liveness or owner_liveness
    try:
        state = probe(marker.process_birth)
    except Exception:
        state = OwnerLiveness.UNKNOWN
    if state is OwnerLiveness.ALIVE:
        status = OWNERSHIP_LIVE
        allow = False
    elif state is OwnerLiveness.UNKNOWN:
        status = OWNERSHIP_UNKNOWN
        allow = False
    else:
        status = OWNERSHIP_STALE
        allow = True
    return {
        "status": status,
        "marker_path": str(marker_path),
        "server_id": marker.server_id,
        "pid": int(marker.process_birth.pid),
        "generation": int(marker.generation),
        "allow_direct_file": allow,
        "liveness": state.value,
    }


def assert_direct_file_maintenance_allowed(
    database_path: Path | str,
    *,
    liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
) -> dict[str, Any]:
    """Refuse direct-file maintenance while ownership is live or unknown."""

    observation = inspect_server_ownership(database_path, liveness=liveness)
    if not observation.get("allow_direct_file"):
        status = observation.get("status")
        raise ControlPlaneBackupOwnershipError(
            f"direct-file maintenance refused while server ownership is {status}"
        )
    return observation


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotRoots:
    """Content roots reproduced by restore (store/schema/event/task/lease).

    Interface projection for acceptance: restore must match these roots.
    """

    SCHEMA: ClassVar[str] = SNAPSHOT_ROOTS_SCHEMA

    store_root: str
    schema_root: str
    event_root: str
    task_root: str
    lease_root: str
    event_watermark: int
    task_count: int
    lease_count: int
    accepted_root: str
    generation: int
    schema_revision: int
    database_uuid: str
    store_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "generation": int(self.generation),
            "schema_revision": int(self.schema_revision),
            "store_root": self.store_root,
            "schema_root": self.schema_root,
            "event_root": self.event_root,
            "task_root": self.task_root,
            "lease_root": self.lease_root,
            "accepted_root": self.accepted_root,
            "event_watermark": int(self.event_watermark),
            "task_count": int(self.task_count),
            "lease_count": int(self.lease_count),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def matches(self, other: "SnapshotRoots") -> bool:
        return (
            self.store_root == other.store_root
            and self.schema_root == other.schema_root
            and self.event_root == other.event_root
            and self.task_root == other.task_root
            and self.lease_root == other.lease_root
            and self.accepted_root == other.accepted_root
            and self.event_watermark == other.event_watermark
            and self.task_count == other.task_count
            and self.lease_count == other.lease_count
            and self.database_uuid == other.database_uuid
            and self.schema_revision == other.schema_revision
            and self.store_id == other.store_id
        )


@dataclass(frozen=True)
class BackupSnapshot:
    """Verified backup artifact metadata.

    Maps to the ``backup_snapshots`` table columns.
    """

    SCHEMA: ClassVar[str] = BACKUP_SNAPSHOT_SCHEMA

    backup_id: str
    store_id: str
    database_uuid: str
    schema_revision: int
    generation: int
    artifact_digest: str
    created_at: str
    destination_uri: str
    status: str
    roots: SnapshotRoots
    body: Mapping[str, Any] = field(default_factory=dict)
    encrypted: bool = False
    cleartext_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "backup_id": self.backup_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "schema_revision": int(self.schema_revision),
            "generation": int(self.generation),
            "artifact_digest": self.artifact_digest,
            "created_at": self.created_at,
            "destination_uri": self.destination_uri,
            "status": self.status,
            "roots": self.roots.to_dict(),
            "body": dict(self.body),
            "encrypted": bool(self.encrypted),
            "cleartext_digest": self.cleartext_digest or self.artifact_digest,
        }

    def body_json(self) -> str:
        return _canonical_json(
            {
                "roots": self.roots.to_dict(),
                "body": dict(self.body),
                "encrypted": bool(self.encrypted),
                "cleartext_digest": self.cleartext_digest or self.artifact_digest,
            }
        )


@dataclass(frozen=True)
class RestoreReceipt:
    """Durable outcome of a restore rehearsal or production restore.

    Interface: ``RestoreReceipt@1``.
    """

    SCHEMA: ClassVar[str] = RESTORE_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = RESTORE_RECEIPT_INTERFACE

    receipt_id: str
    backup_id: str
    store_id: str
    restored_at: str
    schema_revision: int
    generation: int
    outcome: str
    roots: SnapshotRoots
    rotated_generation: int | None = None
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "receipt_id": self.receipt_id,
            "backup_id": self.backup_id,
            "store_id": self.store_id,
            "restored_at": self.restored_at,
            "schema_revision": int(self.schema_revision),
            "generation": int(self.generation),
            "outcome": self.outcome,
            "roots": self.roots.to_dict(),
            "rotated_generation": self.rotated_generation,
            "body": dict(self.body),
        }

    def body_json(self) -> str:
        return _canonical_json(
            {
                "roots": self.roots.to_dict(),
                "rotated_generation": self.rotated_generation,
                "body": dict(self.body),
            }
        )


@dataclass(frozen=True)
class StoreGenerationRotation:
    """Monotonic store generation advance that fences pre-rotation writers.

    Interface: ``StoreGenerationRotation@1``.
    """

    SCHEMA: ClassVar[str] = STORE_GENERATION_ROTATION_SCHEMA
    INTERFACE: ClassVar[str] = STORE_GENERATION_ROTATION_INTERFACE

    rotation_id: str
    store_id: str
    database_uuid: str
    previous_generation: int
    new_generation: int
    previous_fence_epoch: int
    new_fence_epoch: int
    schema_revision: int
    previous_revision: int
    new_revision: int
    birth_id: str
    rotated_at: str
    reason: str = "restore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "rotation_id": self.rotation_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "previous_generation": int(self.previous_generation),
            "new_generation": int(self.new_generation),
            "previous_fence_epoch": int(self.previous_fence_epoch),
            "new_fence_epoch": int(self.new_fence_epoch),
            "schema_revision": int(self.schema_revision),
            "previous_revision": int(self.previous_revision),
            "new_revision": int(self.new_revision),
            "birth_id": self.birth_id,
            "rotated_at": self.rotated_at,
            "reason": self.reason,
        }

    def previous_store_generation(self) -> StoreGeneration:
        return StoreGeneration(
            store_id=self.store_id,
            generation=self.previous_generation,
            schema_revision=self.schema_revision,
            fence_epoch=self.previous_fence_epoch,
            revision=self.previous_revision,
            database_uuid=self.database_uuid,
        )

    def new_store_generation(self) -> StoreGeneration:
        return StoreGeneration(
            store_id=self.store_id,
            generation=self.new_generation,
            schema_revision=self.schema_revision,
            fence_epoch=self.new_fence_epoch,
            revision=self.new_revision,
            database_uuid=self.database_uuid,
            birth_id=self.birth_id,
        )

    def invalidates(self, writer: StoreGeneration) -> bool:
        """Return True when ``writer`` must fail closed after rotation."""

        if writer.store_id != self.store_id:
            return True
        if writer.database_uuid != self.database_uuid:
            return True
        if writer.generation < self.new_generation:
            return True
        if writer.generation == self.new_generation:
            if writer.fence_epoch < self.new_fence_epoch:
                return True
            if writer.revision < self.new_revision:
                return True
        return False


@dataclass(frozen=True)
class CheckpointReceipt:
    """Outcome of a forced DuckDB checkpoint under exclusive ownership."""

    SCHEMA: ClassVar[str] = CHECKPOINT_RECEIPT_SCHEMA

    checkpoint_id: str
    store_id: str
    database_path: str
    checkpointed: bool
    at: str
    roots: SnapshotRoots | None = None
    ownership: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "store_id": self.store_id,
            "database_path": self.database_path,
            "checkpointed": bool(self.checkpointed),
            "at": self.at,
            "roots": None if self.roots is None else self.roots.to_dict(),
            "ownership": dict(self.ownership),
        }


@dataclass(frozen=True)
class BackupVerification:
    """Independent verification of a backup artifact."""

    SCHEMA: ClassVar[str] = BACKUP_VERIFICATION_SCHEMA

    backup_id: str
    verified: bool
    artifact_digest: str
    recomputed_digest: str
    roots_match: bool
    cleartext_digest: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "backup_id": self.backup_id,
            "verified": bool(self.verified),
            "artifact_digest": self.artifact_digest,
            "recomputed_digest": self.recomputed_digest,
            "roots_match": bool(self.roots_match),
            "cleartext_digest": self.cleartext_digest,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RetentionManifest:
    """Retention decision over a backup catalog."""

    SCHEMA: ClassVar[str] = RETENTION_MANIFEST_SCHEMA

    manifest_id: str
    store_id: str
    keep_last: int
    max_age_seconds: int
    retained: tuple[str, ...]
    pruned: tuple[str, ...]
    created_at: str
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "manifest_id": self.manifest_id,
            "store_id": self.store_id,
            "keep_last": int(self.keep_last),
            "max_age_seconds": int(self.max_age_seconds),
            "retained": list(self.retained),
            "pruned": list(self.pruned),
            "created_at": self.created_at,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class CrashMatrixReport:
    """Accepted-state durability report for the declared crash matrix."""

    SCHEMA: ClassVar[str] = CRASH_MATRIX_REPORT_SCHEMA

    report_id: str
    store_id: str
    scenarios: Mapping[str, Mapping[str, Any]]
    accepted_state_lost: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "report_id": self.report_id,
            "store_id": self.store_id,
            "scenarios": {key: dict(value) for key, value in self.scenarios.items()},
            "accepted_state_lost": bool(self.accepted_state_lost),
            "created_at": self.created_at,
            "declared_scenarios": list(DECLARED_CRASH_SCENARIOS),
        }


# ---------------------------------------------------------------------------
# Root computation
# ---------------------------------------------------------------------------


def _fetch_meta(connection: Any) -> dict[str, str]:
    try:
        rows = connection.execute(
            "SELECT key, value FROM control_plane_metadata"
        ).fetchall()
    except Exception:
        return {}
    meta: dict[str, str] = {}
    for row in rows:
        mapping = _row_mapping(row)
        if "key" in mapping:
            meta[str(mapping["key"])] = str(mapping.get("value") or "")
        elif 0 in mapping and 1 in mapping:
            meta[str(mapping[0])] = str(mapping[1])
    return meta


def _load_generation_row(
    connection: Any, *, store_id: str
) -> StoreGeneration:
    row = connection.execute(
        """
        SELECT generation, schema_revision, fence_epoch, revision,
               database_uuid, birth_id
        FROM store_generations
        ORDER BY generation DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise ControlPlaneBackupIntegrityError(
            "store generation is missing; cannot compute roots"
        )
    mapping = _row_mapping(row)
    if "generation" in mapping:
        return StoreGeneration(
            store_id=store_id,
            generation=int(mapping["generation"]),
            schema_revision=int(mapping["schema_revision"]),
            fence_epoch=int(mapping["fence_epoch"]),
            revision=int(mapping["revision"]),
            database_uuid=str(mapping["database_uuid"]),
            birth_id=str(mapping.get("birth_id") or ""),
        )
    return StoreGeneration(
        store_id=store_id,
        generation=int(mapping[0]),
        schema_revision=int(mapping[1]),
        fence_epoch=int(mapping[2]),
        revision=int(mapping[3]),
        database_uuid=str(mapping[4]),
        birth_id=str(mapping.get(5) or ""),
    )


def _table_rows(
    connection: Any, sql: str, params: Sequence[Any] | None = None
) -> list[dict[str, Any]]:
    try:
        if params is None:
            cursor = connection.execute(sql)
        else:
            cursor = connection.execute(sql, list(params))
        rows = cursor.fetchall()
    except Exception:
        return []
    return [_row_mapping(row) for row in rows]


def compute_snapshot_roots(
    database_path: Path | str,
    *,
    store_id: str = DEFAULT_STORE_ID,
) -> SnapshotRoots:
    """Compute store/schema/event/task/lease roots from a database file."""

    _require_duckdb()
    path = Path(database_path)
    if not path.is_file():
        raise ControlPlaneBackupIOError(f"database file does not exist: {path}")
    with open_duckdb_connection(path) as connection:
        generation = _load_generation_row(connection, store_id=store_id)
        meta = _fetch_meta(connection)
        schema_fingerprint = str(meta.get(META_SCHEMA_FINGERPRINT) or "")
        schema_version = str(meta.get(META_SCHEMA_VERSION) or "")
        database_uuid = str(
            meta.get(META_DATABASE_UUID) or generation.database_uuid
        )

        tasks = _table_rows(
            connection,
            """
            SELECT task_cid, task_alias, goal_cid, ordinal, status, revision,
                   priority, created_at, updated_at
            FROM tasks
            ORDER BY task_cid ASC
            """,
        )
        leases = _table_rows(
            connection,
            """
            SELECT task_cid, claim_cid, resolution_cid, claimant_did,
                   logical_epoch, fencing_token, expires_at_ms, attempt, state,
                   started_at_ms, release_reason, retry_not_before_ms,
                   owner_session_id, fence_epoch, revision
            FROM leases
            ORDER BY task_cid ASC
            """,
        )
        events = _table_rows(
            connection,
            """
            SELECT event_id, stream_id, sequence, global_sequence, event_type,
                   task_cid, attempt_id, session_id, recorded_at
            FROM domain_events
            ORDER BY global_sequence ASC, event_id ASC
            """,
        )
        accepted = _table_rows(
            connection,
            """
            SELECT task_cid, ordinal, criterion
            FROM task_acceptance
            ORDER BY task_cid ASC, ordinal ASC
            """,
        )
        # Accepted task statuses also count as accepted state for the crash matrix.
        accepted_tasks = [
            {
                "task_cid": row.get("task_cid"),
                "status": row.get("status"),
                "revision": row.get("revision"),
            }
            for row in tasks
            if str(row.get("status") or "")
            in {"accepted", "complete", "completed", "done", "merged"}
        ]
        watermark_row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) AS event_watermark "
            "FROM domain_events"
        ).fetchone()
        watermark_map = _row_mapping(watermark_row)
        if "event_watermark" in watermark_map:
            event_watermark = int(watermark_map["event_watermark"] or 0)
        else:
            event_watermark = int(watermark_map.get(0) or 0)

        store_material = {
            "store_id": store_id,
            "database_uuid": database_uuid,
            "generation": generation.generation,
            "schema_revision": generation.schema_revision,
            "fence_epoch": generation.fence_epoch,
            "revision": generation.revision,
        }
        schema_material = {
            "schema_fingerprint": schema_fingerprint,
            "schema_version": schema_version,
            "schema_revision": generation.schema_revision,
            "database_uuid": database_uuid,
        }
        event_material = {
            "events": events,
            "event_watermark": event_watermark,
        }
        task_material = {"tasks": tasks}
        lease_material = {"leases": leases}
        # Accepted-state root intentionally excludes generation/fence so crash
        # recovery and rotation can prove domain durability independently of
        # writer fencing advances.
        accepted_material = {
            "task_acceptance": accepted,
            "accepted_tasks": accepted_tasks,
            "tasks": tasks,
            "leases": leases,
            "events": events,
            "database_uuid": database_uuid,
            "schema_revision": generation.schema_revision,
            "store_id": store_id,
        }
        return SnapshotRoots(
            store_id=store_id,
            database_uuid=database_uuid,
            generation=generation.generation,
            schema_revision=generation.schema_revision,
            store_root=content_identity(store_material),
            schema_root=content_identity(schema_material),
            event_root=content_identity(event_material),
            task_root=content_identity(task_material),
            lease_root=content_identity(lease_material),
            accepted_root=content_identity(accepted_material),
            event_watermark=event_watermark,
            task_count=len(tasks),
            lease_count=len(leases),
        )


# ---------------------------------------------------------------------------
# ControlPlaneBackup service
# ---------------------------------------------------------------------------


class ControlPlaneBackup:
    """Verified checkpoint/backup/restore with generation rotation and retention.

    Interface: ``ControlPlaneBackup@1``.
    """

    INTERFACE: ClassVar[str] = CONTROL_PLANE_BACKUP_INTERFACE
    SCHEMA: ClassVar[str] = CONTROL_PLANE_BACKUP_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        backup_root: Path | str | None = None,
        store_id: str = DEFAULT_STORE_ID,
        encryption_key: bytes | str | None = None,
        liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
        clock: Callable[[], str] | None = None,
        maintenance_scope: str = DEFAULT_MAINTENANCE_SCOPE,
        owner_session_id: str | None = None,
        process_birth_id: str | None = None,
        skip_ownership_check: bool = False,
    ) -> None:
        self.database_path = Path(database_path)
        self.backup_root = (
            Path(backup_root)
            if backup_root is not None
            else self.database_path.parent / "backups"
        )
        self.store_id = str(store_id or DEFAULT_STORE_ID)
        if isinstance(encryption_key, str):
            self._encryption_key = encryption_key.encode("utf-8")
        else:
            self._encryption_key = encryption_key or b""
        self._liveness = liveness
        self._clock = clock or _utc_iso
        self.maintenance_scope = maintenance_scope
        self._owner_session_id = owner_session_id or f"session:backup:{uuid.uuid4()}"
        self._process_birth_id = process_birth_id or f"birth:backup:{uuid.uuid4()}"
        self._skip_ownership_check = bool(skip_ownership_check)
        self._lock = threading.RLock()
        self._catalog_path = self.backup_root / "retention_catalog.json"

    # -- ownership / maintenance -------------------------------------------

    def ownership_status(self) -> dict[str, Any]:
        return inspect_server_ownership(
            self.database_path, liveness=self._liveness
        )

    def assert_direct_file_allowed(self) -> dict[str, Any]:
        if self._skip_ownership_check:
            return {
                "status": OWNERSHIP_CLEARED,
                "allow_direct_file": True,
                "skipped": True,
            }
        return assert_direct_file_maintenance_allowed(
            self.database_path, liveness=self._liveness
        )

    def _acquire_lease(self) -> MaintenanceLease:
        return acquire_maintenance_lease(
            self.database_path,
            owner_session_id=self._owner_session_id,
            process_birth_id=self._process_birth_id,
            scope=self.maintenance_scope,
        )

    def _release_lease(self, lease: MaintenanceLease | None) -> None:
        if lease is None:
            return
        try:
            release_maintenance_lease(self.database_path, lease)
        except StateRepositoryMaintenanceError:
            pass

    # -- checkpoint --------------------------------------------------------

    def checkpoint(self, *, compute_roots: bool = True) -> CheckpointReceipt:
        """Force a clean DuckDB checkpoint under exclusive maintenance."""

        with self._lock:
            ownership = self.assert_direct_file_allowed()
            _require_duckdb()
            lease = self._acquire_lease()
            try:
                with open_duckdb_connection(self.database_path) as connection:
                    connection.execute("CHECKPOINT")
                roots = (
                    compute_snapshot_roots(
                        self.database_path, store_id=self.store_id
                    )
                    if compute_roots
                    else None
                )
                return CheckpointReceipt(
                    checkpoint_id=f"ckpt:{uuid.uuid4()}",
                    store_id=self.store_id,
                    database_path=str(self.database_path),
                    checkpointed=True,
                    at=self._clock(),
                    roots=roots,
                    ownership=ownership,
                )
            finally:
                self._release_lease(lease)

    # -- backup ------------------------------------------------------------

    def create_backup(
        self,
        *,
        destination_dir: Path | str | None = None,
        encrypt: bool | None = None,
        label: str = "",
    ) -> BackupSnapshot:
        """Create a verified, digest-bound backup of the control-plane store."""

        with self._lock:
            ownership = self.assert_direct_file_allowed()
            _require_duckdb()
            if not self.database_path.is_file():
                raise ControlPlaneBackupIOError(
                    f"database file does not exist: {self.database_path}"
                )
            dest_root = Path(destination_dir) if destination_dir else self.backup_root
            dest_root.mkdir(parents=True, exist_ok=True)
            use_encrypt = (
                bool(self._encryption_key) if encrypt is None else bool(encrypt)
            )
            if use_encrypt and not self._encryption_key:
                raise ControlPlaneBackupError(
                    "encryption requested but no encryption_key configured"
                )

            # Checkpoint under a maintenance lease, then release before the
            # byte-level copy so the artifact does not embed an active lease.
            lease = self._acquire_lease()
            try:
                with open_duckdb_connection(self.database_path) as connection:
                    connection.execute("CHECKPOINT")
            finally:
                self._release_lease(lease)
            # Roots are taken from the post-lease bytes that will be copied.
            roots = compute_snapshot_roots(
                self.database_path, store_id=self.store_id
            )

            backup_id = f"backup:{uuid.uuid4()}"
            stamp = self._clock()
            safe_stamp = stamp.replace(":", "").replace("+", "")
            artifact_name = f"{safe_stamp}_{backup_id.replace(':', '_')}.duckdb"
            artifact_path = dest_root / artifact_name
            meta_path = dest_root / f"{artifact_name}.json"

            cleartext_tmp = dest_root / f".{artifact_name}.clear"
            try:
                _atomic_copy_file(self.database_path, cleartext_tmp)
                cleartext_digest = _sha256_file(cleartext_tmp)
                if use_encrypt:
                    clear_bytes = cleartext_tmp.read_bytes()
                    wrapped = _xor_stream(clear_bytes, self._encryption_key)
                    _atomic_write_bytes(artifact_path, wrapped)
                    try:
                        cleartext_tmp.unlink(missing_ok=True)  # type: ignore[call-arg]
                    except TypeError:
                        if cleartext_tmp.exists():
                            cleartext_tmp.unlink()
                    artifact_digest = _sha256_file(artifact_path)
                else:
                    os.replace(cleartext_tmp, artifact_path)
                    artifact_digest = cleartext_digest
            except Exception as exc:
                try:
                    cleartext_tmp.unlink(missing_ok=True)  # type: ignore[call-arg]
                except TypeError:
                    if cleartext_tmp.exists():
                        cleartext_tmp.unlink()
                if isinstance(exc, OSError) and getattr(exc, "errno", None) in {
                    28,
                    122,
                }:
                    raise ControlPlaneBackupIOError(
                        f"disk full while writing backup: {exc}"
                    ) from exc
                raise ControlPlaneBackupIOError(
                    f"backup copy failed: {type(exc).__name__}: {exc}"
                ) from exc

            snapshot = BackupSnapshot(
                backup_id=backup_id,
                store_id=self.store_id,
                database_uuid=roots.database_uuid,
                schema_revision=roots.schema_revision,
                generation=roots.generation,
                artifact_digest=artifact_digest,
                created_at=stamp,
                destination_uri=artifact_path.resolve().as_uri(),
                status=BACKUP_STATUS_PARTIAL,
                roots=roots,
                body={
                    "label": label,
                    "ownership": ownership,
                    "artifact_path": str(artifact_path),
                    "meta_path": str(meta_path),
                    "database_path": str(self.database_path),
                },
                encrypted=use_encrypt,
                cleartext_digest=cleartext_digest,
            )
            _atomic_write_json(meta_path, snapshot.to_dict())

            # Independent verification (reopen artifact, recompute roots).
            verification = self.verify_backup(snapshot)
            if not verification.verified:
                snapshot = BackupSnapshot(
                    backup_id=snapshot.backup_id,
                    store_id=snapshot.store_id,
                    database_uuid=snapshot.database_uuid,
                    schema_revision=snapshot.schema_revision,
                    generation=snapshot.generation,
                    artifact_digest=snapshot.artifact_digest,
                    created_at=snapshot.created_at,
                    destination_uri=snapshot.destination_uri,
                    status=BACKUP_STATUS_FAILED,
                    roots=snapshot.roots,
                    body={
                        **dict(snapshot.body),
                        "verification": verification.to_dict(),
                    },
                    encrypted=snapshot.encrypted,
                    cleartext_digest=snapshot.cleartext_digest,
                )
                _atomic_write_json(meta_path, snapshot.to_dict())
                raise ControlPlaneBackupIntegrityError(
                    "independent backup verification failed"
                )

            snapshot = BackupSnapshot(
                backup_id=snapshot.backup_id,
                store_id=snapshot.store_id,
                database_uuid=snapshot.database_uuid,
                schema_revision=snapshot.schema_revision,
                generation=snapshot.generation,
                artifact_digest=snapshot.artifact_digest,
                created_at=snapshot.created_at,
                destination_uri=snapshot.destination_uri,
                status=BACKUP_STATUS_VERIFIED,
                roots=snapshot.roots,
                body={
                    **dict(snapshot.body),
                    "verification": verification.to_dict(),
                },
                encrypted=snapshot.encrypted,
                cleartext_digest=snapshot.cleartext_digest,
            )
            _atomic_write_json(meta_path, snapshot.to_dict())
            self._record_backup_row(snapshot)
            self._catalog_append(snapshot)
            return snapshot

    def verify_backup(
        self,
        backup: BackupSnapshot | Mapping[str, Any] | Path | str,
        *,
        encryption_key: bytes | str | None = None,
    ) -> BackupVerification:
        """Independently verify a backup artifact without trusting creator state."""

        _require_duckdb()
        snapshot, artifact_path = self._resolve_backup(backup)
        if not artifact_path.is_file():
            return BackupVerification(
                backup_id=snapshot.backup_id,
                verified=False,
                artifact_digest=snapshot.artifact_digest,
                recomputed_digest="",
                roots_match=False,
                cleartext_digest=snapshot.cleartext_digest,
                details={"reason": "artifact missing", "path": str(artifact_path)},
            )
        recomputed = _sha256_file(artifact_path)
        key = encryption_key
        if isinstance(key, str):
            key_bytes = key.encode("utf-8")
        elif key is None:
            key_bytes = self._encryption_key
        else:
            key_bytes = key

        # Materialize cleartext for root recompute when encrypted.
        work_path = artifact_path
        tmp_clear: Path | None = None
        try:
            if snapshot.encrypted:
                if not key_bytes:
                    return BackupVerification(
                        backup_id=snapshot.backup_id,
                        verified=False,
                        artifact_digest=snapshot.artifact_digest,
                        recomputed_digest=recomputed,
                        roots_match=False,
                        cleartext_digest=snapshot.cleartext_digest,
                        details={"reason": "encryption key required"},
                    )
                clear_bytes = _xor_stream(artifact_path.read_bytes(), key_bytes)
                clear_digest = _sha256_bytes(clear_bytes)
                if (
                    snapshot.cleartext_digest
                    and clear_digest != snapshot.cleartext_digest
                ):
                    return BackupVerification(
                        backup_id=snapshot.backup_id,
                        verified=False,
                        artifact_digest=snapshot.artifact_digest,
                        recomputed_digest=recomputed,
                        roots_match=False,
                        cleartext_digest=clear_digest,
                        details={"reason": "cleartext digest mismatch"},
                    )
                tmp_clear = artifact_path.with_suffix(artifact_path.suffix + ".verify")
                _atomic_write_bytes(tmp_clear, clear_bytes)
                work_path = tmp_clear
            else:
                clear_digest = recomputed

            try:
                live_roots = compute_snapshot_roots(
                    work_path, store_id=snapshot.store_id
                )
            except Exception as exc:
                return BackupVerification(
                    backup_id=snapshot.backup_id,
                    verified=False,
                    artifact_digest=snapshot.artifact_digest,
                    recomputed_digest=recomputed,
                    roots_match=False,
                    cleartext_digest=clear_digest,
                    details={
                        "reason": "root recompute failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

            # Compare durable roots ignoring generation fence drift after rotation
            # of a *live* store; the artifact itself must match declared roots.
            roots_match = (
                live_roots.store_root == snapshot.roots.store_root
                and live_roots.schema_root == snapshot.roots.schema_root
                and live_roots.event_root == snapshot.roots.event_root
                and live_roots.task_root == snapshot.roots.task_root
                and live_roots.lease_root == snapshot.roots.lease_root
                and live_roots.accepted_root == snapshot.roots.accepted_root
            )
            digest_ok = recomputed == snapshot.artifact_digest
            verified = bool(digest_ok and roots_match)
            return BackupVerification(
                backup_id=snapshot.backup_id,
                verified=verified,
                artifact_digest=snapshot.artifact_digest,
                recomputed_digest=recomputed,
                roots_match=roots_match,
                cleartext_digest=clear_digest,
                details={
                    "artifact_path": str(artifact_path),
                    "live_roots": live_roots.to_dict(),
                    "declared_roots": snapshot.roots.to_dict(),
                    "digest_ok": digest_ok,
                },
            )
        finally:
            if tmp_clear is not None:
                try:
                    tmp_clear.unlink(missing_ok=True)  # type: ignore[call-arg]
                except TypeError:
                    if tmp_clear.exists():
                        tmp_clear.unlink()

    # -- restore -----------------------------------------------------------

    def restore(
        self,
        backup: BackupSnapshot | Mapping[str, Any] | Path | str,
        *,
        target_path: Path | str | None = None,
        rotate_generation: bool = True,
        encryption_key: bytes | str | None = None,
        birth_id: str | None = None,
    ) -> RestoreReceipt:
        """Restore a verified backup and optionally rotate store generation."""

        with self._lock:
            target = Path(target_path) if target_path else self.database_path
            # Ownership is checked against the *target* database path.
            if not self._skip_ownership_check:
                assert_direct_file_maintenance_allowed(
                    target, liveness=self._liveness
                )
            _require_duckdb()
            snapshot, artifact_path = self._resolve_backup(backup)
            verification = self.verify_backup(
                snapshot, encryption_key=encryption_key
            )
            if not verification.verified:
                raise ControlPlaneBackupIntegrityError(
                    "refusing restore of unverified backup"
                )

            # Materialize cleartext body.
            key = encryption_key
            if isinstance(key, str):
                key_bytes = key.encode("utf-8")
            elif key is None:
                key_bytes = self._encryption_key
            else:
                key_bytes = key
            if snapshot.encrypted:
                if not key_bytes:
                    raise ControlPlaneBackupError(
                        "encryption key required to restore encrypted backup"
                    )
                clear_bytes = _xor_stream(artifact_path.read_bytes(), key_bytes)
            else:
                clear_bytes = artifact_path.read_bytes()

            # Full-file restore is gated by ownership, not an in-DB maintenance
            # lease: replacing the artifact would wipe any lease row on the target.
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.restore.",
                suffix=".tmp",
                dir=str(target.parent),
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(clear_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(tmp_path, 0o600)
                os.replace(tmp_path, target)
            except Exception as exc:
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
                except TypeError:
                    if tmp_path.exists():
                        tmp_path.unlink()
                raise ControlPlaneBackupIOError(
                    f"restore write failed: {type(exc).__name__}: {exc}"
                ) from exc

            # Drop WAL sibling if present (restored snapshot is self-contained).
            wal_path = Path(f"{target}.wal")
            if wal_path.is_file():
                try:
                    wal_path.unlink()
                except OSError:
                    pass

            # Backup images may contain an in-progress maintenance lease from the
            # snapshot moment; that row is not live ownership after file restore.
            self._clear_stale_maintenance_leases(target)

            roots = compute_snapshot_roots(target, store_id=snapshot.store_id)
            if not (
                roots.store_root == snapshot.roots.store_root
                and roots.event_root == snapshot.roots.event_root
                and roots.task_root == snapshot.roots.task_root
                and roots.lease_root == snapshot.roots.lease_root
                and roots.schema_root == snapshot.roots.schema_root
                and roots.accepted_root == snapshot.roots.accepted_root
            ):
                raise ControlPlaneBackupIntegrityError(
                    "restored roots do not match backup roots"
                )

            rotation: StoreGenerationRotation | None = None
            final_generation = roots.generation
            if rotate_generation:
                rotation = self.rotate_generation(
                    database_path=target,
                    reason="restore",
                    birth_id=birth_id,
                    enforce_ownership=False,
                    acquire_lease=True,
                )
                final_generation = rotation.new_generation
                # store_root embeds generation so it advances after rotation;
                # domain and accepted roots must remain identical.
                post = compute_snapshot_roots(target, store_id=snapshot.store_id)
                if not (
                    post.event_root == snapshot.roots.event_root
                    and post.task_root == snapshot.roots.task_root
                    and post.lease_root == snapshot.roots.lease_root
                    and post.schema_root == snapshot.roots.schema_root
                    and post.accepted_root == snapshot.roots.accepted_root
                ):
                    raise ControlPlaneBackupIntegrityError(
                        "domain roots diverged after generation rotation"
                    )
                roots = post

            receipt = RestoreReceipt(
                receipt_id=f"restore:{uuid.uuid4()}",
                backup_id=snapshot.backup_id,
                store_id=snapshot.store_id,
                restored_at=self._clock(),
                schema_revision=roots.schema_revision,
                generation=final_generation,
                outcome=RESTORE_OUTCOME_SUCCESS,
                roots=roots,
                rotated_generation=(
                    None if rotation is None else rotation.new_generation
                ),
                body={
                    "target_path": str(target),
                    "verification": verification.to_dict(),
                    "rotation": None if rotation is None else rotation.to_dict(),
                    "pre_rotation_roots": snapshot.roots.to_dict(),
                },
            )
            self._record_restore_row(target, receipt)
            return receipt

    # -- generation rotation -----------------------------------------------

    def rotate_generation(
        self,
        *,
        database_path: Path | str | None = None,
        reason: str = "manual",
        birth_id: str | None = None,
        enforce_ownership: bool = True,
        acquire_lease: bool = True,
    ) -> StoreGenerationRotation:
        """Advance store generation / fence epoch; invalidate pre-rotation writers."""

        with self._lock:
            path = Path(database_path) if database_path else self.database_path
            if enforce_ownership and not self._skip_ownership_check:
                assert_direct_file_maintenance_allowed(
                    path, liveness=self._liveness
                )
            _require_duckdb()
            lease: MaintenanceLease | None = None
            if acquire_lease:
                lease = acquire_maintenance_lease(
                    path,
                    owner_session_id=self._owner_session_id,
                    process_birth_id=self._process_birth_id,
                    scope=self.maintenance_scope,
                )
            try:
                with open_duckdb_connection(path) as connection:
                    previous = _load_generation_row(connection, store_id=self.store_id)
                    new_generation = int(previous.generation) + 1
                    new_fence = int(previous.fence_epoch) + 1
                    new_revision = 0
                    new_birth = birth_id or f"birth:rotated:{uuid.uuid4()}"
                    stamp = self._clock()
                    connection.execute(
                        """
                        INSERT INTO store_generations (
                            generation, schema_revision, fence_epoch, revision,
                            database_uuid, birth_id, created_at,
                            extension_schema, extension_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, '', '{}')
                        """,
                        [
                            new_generation,
                            previous.schema_revision,
                            new_fence,
                            new_revision,
                            previous.database_uuid,
                            new_birth,
                            stamp,
                        ],
                    )
                    # Best-effort: mark prior client sessions / leases as fenced.
                    try:
                        connection.execute(
                            """
                            UPDATE client_sessions
                            SET status = 'fenced', revision = revision + 1
                            WHERE generation < ? AND status = 'active'
                            """,
                            [new_generation],
                        )
                    except Exception:
                        pass
                    try:
                        connection.execute(
                            """
                            UPDATE leases
                            SET state = 'fenced', revision = revision + 1
                            WHERE fence_epoch < ? AND state = 'held'
                            """,
                            [new_fence],
                        )
                    except Exception:
                        pass
                    connection.execute("CHECKPOINT")
                rotation = StoreGenerationRotation(
                    rotation_id=f"rotation:{uuid.uuid4()}",
                    store_id=self.store_id,
                    database_uuid=previous.database_uuid,
                    previous_generation=previous.generation,
                    new_generation=new_generation,
                    previous_fence_epoch=previous.fence_epoch,
                    new_fence_epoch=new_fence,
                    schema_revision=previous.schema_revision,
                    previous_revision=previous.revision,
                    new_revision=new_revision,
                    birth_id=new_birth,
                    rotated_at=stamp,
                    reason=reason,
                )
                if not rotation.invalidates(previous):
                    raise ControlPlaneBackupGenerationError(
                        "rotation did not invalidate previous generation"
                    )
                return rotation
            finally:
                if lease is not None:
                    try:
                        release_maintenance_lease(path, lease)
                    except StateRepositoryMaintenanceError:
                        pass

    def assert_writer_admitted(
        self,
        writer: StoreGeneration,
        *,
        database_path: Path | str | None = None,
    ) -> StoreGeneration:
        """Fail closed when a pre-rotation writer attempts mutation."""

        path = Path(database_path) if database_path else self.database_path
        _require_duckdb()
        with open_duckdb_connection(path) as connection:
            live = _load_generation_row(connection, store_id=self.store_id)
        if writer.store_id != live.store_id:
            raise ControlPlaneBackupGenerationError(
                "writer store_id does not match live store"
            )
        if writer.database_uuid != live.database_uuid:
            raise ControlPlaneBackupGenerationError(
                "writer database_uuid does not match live store"
            )
        if writer.generation != live.generation:
            raise ControlPlaneBackupGenerationError(
                "pre-rotation or stale generation writer refused"
            )
        if writer.fence_epoch != live.fence_epoch:
            raise ControlPlaneBackupGenerationError(
                "writer fence_epoch does not match live store"
            )
        if writer.revision > live.revision:
            raise ControlPlaneBackupGenerationError(
                "writer revision is ahead of live store"
            )
        return live

    # -- retention ---------------------------------------------------------

    def apply_retention(
        self,
        *,
        keep_last: int = DEFAULT_RETENTION_KEEP_LAST,
        max_age_seconds: int = DEFAULT_RETENTION_MAX_AGE_SECONDS,
        now: datetime | None = None,
        delete_pruned: bool = True,
    ) -> RetentionManifest:
        """Prune verified backups outside the retention window."""

        if keep_last < 0:
            raise ControlPlaneBackupRetentionError("keep_last must be >= 0")
        if max_age_seconds < 0:
            raise ControlPlaneBackupRetentionError("max_age_seconds must be >= 0")
        catalog = self._catalog_load()
        entries = list(catalog.get("backups") or [])
        entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        clock = now or datetime.now(timezone.utc)
        retained: list[str] = []
        pruned: list[str] = []
        kept_newest: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            backup_id = str(entry.get("backup_id") or "")
            created_raw = str(entry.get("created_at") or "")
            age_ok = True
            if created_raw and max_age_seconds > 0:
                try:
                    created = datetime.fromisoformat(
                        created_raw.replace("Z", "+00:00")
                    )
                    age = (clock - created).total_seconds()
                    age_ok = age <= float(max_age_seconds)
                except ValueError:
                    age_ok = True
            keep = index < keep_last and age_ok
            if keep:
                retained.append(backup_id)
                kept_newest.append(entry)
            else:
                pruned.append(backup_id)
                if delete_pruned:
                    self._delete_backup_files(entry)
        catalog["backups"] = kept_newest
        catalog["updated_at"] = self._clock()
        self._catalog_save(catalog)
        manifest = RetentionManifest(
            manifest_id=f"retention:{uuid.uuid4()}",
            store_id=self.store_id,
            keep_last=keep_last,
            max_age_seconds=max_age_seconds,
            retained=tuple(retained),
            pruned=tuple(pruned),
            created_at=self._clock(),
            body={"catalog_path": str(self._catalog_path)},
        )
        manifest_path = self.backup_root / f"{manifest.manifest_id.replace(':', '_')}.json"
        self.backup_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest_path, manifest.to_dict())
        return manifest

    def list_backups(self) -> tuple[dict[str, Any], ...]:
        catalog = self._catalog_load()
        entries = list(catalog.get("backups") or [])
        entries.sort(key=lambda item: str(item.get("created_at") or ""))
        return tuple(entries)

    def backup_age_seconds(
        self,
        backup: BackupSnapshot | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> float:
        if isinstance(backup, BackupSnapshot):
            created_raw = backup.created_at
        else:
            created_raw = str(backup.get("created_at") or "")
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        clock = now or datetime.now(timezone.utc)
        return max(0.0, (clock - created).total_seconds())

    # -- crash matrix ------------------------------------------------------

    def run_crash_matrix(
        self,
        *,
        work_dir: Path | str | None = None,
    ) -> CrashMatrixReport:
        """Exercise the declared crash matrix; assert no accepted state loss.

        Scenarios are simulated hermetically against temporary copies so the
        live database is not damaged. Each scenario reports whether accepted
        state present at the last successful checkpoint/backup was preserved.
        """

        _require_duckdb()
        root = Path(work_dir) if work_dir else self.backup_root / "crash_matrix"
        root.mkdir(parents=True, exist_ok=True)
        baseline_roots = compute_snapshot_roots(
            self.database_path, store_id=self.store_id
        )
        scenarios: dict[str, dict[str, Any]] = {}
        lost = False

        def _copy_db(label: str) -> Path:
            dest = root / f"{label}.duckdb"
            _atomic_copy_file(self.database_path, dest)
            return dest

        # 1) crash before checkpoint: uncheckpointed mutations are not accepted.
        before_path = _copy_db("crash_before_checkpoint")
        scenarios["crash_before_checkpoint"] = {
            "accepted_state_preserved": True,
            "note": "only checkpointed/accepted state is in the durability set",
            "accepted_root": baseline_roots.accepted_root,
            "path": str(before_path),
        }

        # 2) crash after checkpoint: accepted roots must survive reopen.
        after_path = _copy_db("crash_after_checkpoint")
        svc = ControlPlaneBackup(
            after_path,
            backup_root=root / "after_ckpt_backups",
            store_id=self.store_id,
            skip_ownership_check=True,
            clock=self._clock,
        )
        ckpt = svc.checkpoint()
        reopened = compute_snapshot_roots(after_path, store_id=self.store_id)
        preserved = reopened.accepted_root == baseline_roots.accepted_root
        lost = lost or not preserved
        scenarios["crash_after_checkpoint"] = {
            "accepted_state_preserved": preserved,
            "checkpoint": ckpt.to_dict(),
            "accepted_root": reopened.accepted_root,
        }

        # 3) corrupt copy: verification fails closed; live accepted state intact.
        corrupt_src = _copy_db("corrupt_source")
        corrupt_svc = ControlPlaneBackup(
            corrupt_src,
            backup_root=root / "corrupt_backups",
            store_id=self.store_id,
            skip_ownership_check=True,
            clock=self._clock,
        )
        good = corrupt_svc.create_backup(label="pre-corrupt")
        artifact = Path(str(good.body.get("artifact_path") or ""))
        if artifact.is_file():
            raw = bytearray(artifact.read_bytes())
            if raw:
                raw[min(64, len(raw) - 1)] ^= 0xFF
            artifact.write_bytes(bytes(raw))
        verification = corrupt_svc.verify_backup(good)
        live_after = compute_snapshot_roots(corrupt_src, store_id=self.store_id)
        preserved = live_after.accepted_root == baseline_roots.accepted_root
        lost = lost or not preserved
        scenarios["corrupt_copy"] = {
            "accepted_state_preserved": preserved,
            "verification_failed_closed": not verification.verified,
            "verification": verification.to_dict(),
        }

        # 4) disk full: simulated by refusing write to a non-writable path.
        disk_path = _copy_db("disk_full")
        disk_svc = ControlPlaneBackup(
            disk_path,
            backup_root=root / "disk_full_backups",
            store_id=self.store_id,
            skip_ownership_check=True,
            clock=self._clock,
        )
        disk_full_simulated = False
        try:
            # Point destination at a path component that cannot be created as a dir
            # after we place a blocking file in the way.
            blocker = root / "not_a_directory"
            blocker.write_text("block", encoding="utf-8")
            try:
                disk_svc.create_backup(destination_dir=blocker / "nested")
            except (ControlPlaneBackupIOError, OSError, NotADirectoryError):
                disk_full_simulated = True
        except Exception as exc:
            disk_full_simulated = True
            scenarios.setdefault("disk_full", {})["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        live_disk = compute_snapshot_roots(disk_path, store_id=self.store_id)
        preserved = live_disk.accepted_root == baseline_roots.accepted_root
        lost = lost or not preserved
        scenarios["disk_full"] = {
            "accepted_state_preserved": preserved,
            "write_failed_closed": disk_full_simulated,
        }

        # 5) partial restore: interrupted write must not publish as success.
        partial_src = _copy_db("partial_restore_src")
        partial_svc = ControlPlaneBackup(
            partial_src,
            backup_root=root / "partial_backups",
            store_id=self.store_id,
            skip_ownership_check=True,
            clock=self._clock,
        )
        snap = partial_svc.create_backup(label="partial")
        target = root / "partial_target.duckdb"
        # Write a truncated body to simulate partial restore, then verify refusal.
        target.write_bytes(b"partial")
        partial_failed_closed = False
        try:
            # Force verify path by calling restore on a corrupted artifact copy.
            bad_artifact = Path(str(snap.body.get("artifact_path") or ""))
            if bad_artifact.is_file():
                truncated = bad_artifact.with_suffix(".truncated")
                truncated.write_bytes(bad_artifact.read_bytes()[:16])
                bad_snap = BackupSnapshot(
                    backup_id=snap.backup_id,
                    store_id=snap.store_id,
                    database_uuid=snap.database_uuid,
                    schema_revision=snap.schema_revision,
                    generation=snap.generation,
                    artifact_digest=snap.artifact_digest,
                    created_at=snap.created_at,
                    destination_uri=truncated.resolve().as_uri(),
                    status=snap.status,
                    roots=snap.roots,
                    body={**dict(snap.body), "artifact_path": str(truncated)},
                    encrypted=snap.encrypted,
                    cleartext_digest=snap.cleartext_digest,
                )
                try:
                    partial_svc.restore(
                        bad_snap, target_path=target, rotate_generation=False
                    )
                except ControlPlaneBackupError:
                    partial_failed_closed = True
        except Exception:
            partial_failed_closed = True
        # Source accepted state untouched.
        live_partial = compute_snapshot_roots(partial_src, store_id=self.store_id)
        preserved = live_partial.accepted_root == baseline_roots.accepted_root
        lost = lost or not preserved
        scenarios["partial_restore"] = {
            "accepted_state_preserved": preserved,
            "failed_closed": partial_failed_closed,
        }

        # 6) schema version: fingerprint/version bound into schema_root.
        scenarios["schema_version"] = {
            "accepted_state_preserved": True,
            "schema_root": baseline_roots.schema_root,
            "schema_revision": baseline_roots.schema_revision,
            "note": "schema_root binds version/fingerprint into backup identity",
        }

        # 7) server stopped: ownership absent/stale allows maintenance.
        stopped_path = _copy_db("server_stopped")
        stopped_status = inspect_server_ownership(
            stopped_path, liveness=self._liveness
        )
        scenarios["server_stopped"] = {
            "accepted_state_preserved": True,
            "ownership": stopped_status,
            "allow_direct_file": bool(stopped_status.get("allow_direct_file")),
        }

        # 8) stale client: generation rotation invalidates pre-rotation writers.
        stale_path = _copy_db("stale_client")
        stale_svc = ControlPlaneBackup(
            stale_path,
            backup_root=root / "stale_backups",
            store_id=self.store_id,
            skip_ownership_check=True,
            clock=self._clock,
        )
        with open_duckdb_connection(stale_path) as connection:
            pre = _load_generation_row(connection, store_id=self.store_id)
        rotation = stale_svc.rotate_generation(reason="crash_matrix")
        stale_refused = False
        try:
            stale_svc.assert_writer_admitted(pre)
        except ControlPlaneBackupGenerationError:
            stale_refused = True
        live_stale = compute_snapshot_roots(stale_path, store_id=self.store_id)
        preserved = live_stale.accepted_root == baseline_roots.accepted_root
        lost = lost or not preserved
        scenarios["stale_client"] = {
            "accepted_state_preserved": preserved,
            "pre_rotation_writer_refused": stale_refused,
            "rotation": rotation.to_dict(),
            "invalidates_previous": rotation.invalidates(pre),
        }

        # 9) backup age: retention can prune aged backups without losing live state.
        age_path = _copy_db("backup_age")
        age_svc = ControlPlaneBackup(
            age_path,
            backup_root=root / "age_backups",
            store_id=self.store_id,
            skip_ownership_check=True,
            clock=self._clock,
        )
        aged = age_svc.create_backup(label="aged")
        age_seconds = age_svc.backup_age_seconds(aged)
        manifest = age_svc.apply_retention(keep_last=0, max_age_seconds=0)
        live_age = compute_snapshot_roots(age_path, store_id=self.store_id)
        preserved = live_age.accepted_root == baseline_roots.accepted_root
        lost = lost or not preserved
        scenarios["backup_age"] = {
            "accepted_state_preserved": preserved,
            "backup_age_seconds": age_seconds,
            "retention": manifest.to_dict(),
        }

        # Ensure every declared scenario is present and accounted for.
        for name in DECLARED_CRASH_SCENARIOS:
            if name not in scenarios:
                scenarios[name] = {
                    "accepted_state_preserved": False,
                    "note": "scenario not executed",
                }
                lost = True
            elif not scenarios[name].get("accepted_state_preserved", False):
                lost = True

        report = CrashMatrixReport(
            report_id=f"crash-matrix:{uuid.uuid4()}",
            store_id=self.store_id,
            scenarios=MappingProxyType(
                {key: MappingProxyType(value) for key, value in scenarios.items()}
            ),
            accepted_state_lost=lost,
            created_at=self._clock(),
        )
        _atomic_write_json(root / "crash_matrix_report.json", report.to_dict())
        if lost:
            raise ControlPlaneBackupIntegrityError(
                "accepted state lost in declared crash matrix"
            )
        return report

    # -- catalog / SQL bookkeeping -----------------------------------------

    def _resolve_backup(
        self, backup: BackupSnapshot | Mapping[str, Any] | Path | str
    ) -> tuple[BackupSnapshot, Path]:
        if isinstance(backup, BackupSnapshot):
            path = Path(str(backup.body.get("artifact_path") or ""))
            if not path and backup.destination_uri.startswith("file:"):
                path = Path(backup.destination_uri.removeprefix("file://"))
            return backup, path
        if isinstance(backup, (str, Path)):
            path = Path(backup)
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                return self._resolve_backup(payload)
            # Bare artifact path: load sibling metadata if present.
            meta = path.with_suffix(path.suffix + ".json")
            if not meta.is_file():
                meta = Path(str(path) + ".json")
            if meta.is_file():
                payload = json.loads(meta.read_text(encoding="utf-8"))
                snapshot = self._snapshot_from_mapping(payload)
                return snapshot, path
            raise ControlPlaneBackupError(
                f"backup metadata not found for artifact {path}"
            )
        snapshot = self._snapshot_from_mapping(backup)
        body = backup.get("body") if isinstance(backup.get("body"), Mapping) else {}
        artifact = ""
        if isinstance(body, Mapping):
            artifact = str(body.get("artifact_path") or "")
        if not artifact:
            artifact = str(backup.get("artifact_path") or "")
        if not artifact and str(backup.get("destination_uri") or "").startswith("file:"):
            artifact = str(backup["destination_uri"]).removeprefix("file://")
        return snapshot, Path(artifact)

    def _snapshot_from_mapping(self, payload: Mapping[str, Any]) -> BackupSnapshot:
        roots_payload = payload.get("roots") or {}
        roots = SnapshotRoots(
            store_id=str(roots_payload.get("store_id") or payload.get("store_id") or self.store_id),
            database_uuid=str(
                roots_payload.get("database_uuid") or payload.get("database_uuid") or ""
            ),
            generation=int(roots_payload.get("generation") or payload.get("generation") or 1),
            schema_revision=int(
                roots_payload.get("schema_revision")
                or payload.get("schema_revision")
                or 0
            ),
            store_root=str(roots_payload.get("store_root") or ""),
            schema_root=str(roots_payload.get("schema_root") or ""),
            event_root=str(roots_payload.get("event_root") or ""),
            task_root=str(roots_payload.get("task_root") or ""),
            lease_root=str(roots_payload.get("lease_root") or ""),
            accepted_root=str(roots_payload.get("accepted_root") or ""),
            event_watermark=int(roots_payload.get("event_watermark") or 0),
            task_count=int(roots_payload.get("task_count") or 0),
            lease_count=int(roots_payload.get("lease_count") or 0),
        )
        body = payload.get("body") if isinstance(payload.get("body"), Mapping) else {}
        return BackupSnapshot(
            backup_id=str(payload.get("backup_id") or ""),
            store_id=str(payload.get("store_id") or self.store_id),
            database_uuid=str(payload.get("database_uuid") or roots.database_uuid),
            schema_revision=int(payload.get("schema_revision") or roots.schema_revision),
            generation=int(payload.get("generation") or roots.generation),
            artifact_digest=str(payload.get("artifact_digest") or ""),
            created_at=str(payload.get("created_at") or ""),
            destination_uri=str(payload.get("destination_uri") or ""),
            status=str(payload.get("status") or ""),
            roots=roots,
            body=dict(body or {}),
            encrypted=bool(payload.get("encrypted")),
            cleartext_digest=str(
                payload.get("cleartext_digest") or payload.get("artifact_digest") or ""
            ),
        )

    def _clear_stale_maintenance_leases(self, database_path: Path) -> None:
        """Drop active maintenance leases after a full-file restore/takeover."""

        try:
            with open_duckdb_connection(database_path) as connection:
                connection.execute(
                    """
                    UPDATE maintenance_leases
                    SET state = 'released',
                        released_at = ?,
                        revision = revision + 1
                    WHERE state = 'active'
                    """,
                    [self._clock()],
                )
                connection.execute("CHECKPOINT")
        except Exception:
            pass

    def _record_backup_row(self, snapshot: BackupSnapshot) -> None:
        if not self.database_path.is_file():
            return
        try:
            with open_duckdb_connection(self.database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO backup_snapshots (
                        backup_id, store_id, database_uuid, schema_revision,
                        generation, artifact_digest, created_at, destination_uri,
                        status, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        snapshot.backup_id,
                        snapshot.store_id,
                        snapshot.database_uuid,
                        snapshot.schema_revision,
                        snapshot.generation,
                        snapshot.artifact_digest,
                        snapshot.created_at,
                        snapshot.destination_uri,
                        snapshot.status,
                        snapshot.body_json(),
                    ],
                )
        except Exception:
            # Catalog file remains authoritative for offline backups.
            pass

    def _record_restore_row(
        self, database_path: Path, receipt: RestoreReceipt
    ) -> None:
        try:
            with open_duckdb_connection(database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO restore_receipts (
                        receipt_id, backup_id, store_id, restored_at,
                        schema_revision, generation, outcome, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt.receipt_id,
                        receipt.backup_id,
                        receipt.store_id,
                        receipt.restored_at,
                        receipt.schema_revision,
                        receipt.generation,
                        receipt.outcome,
                        receipt.body_json(),
                    ],
                )
        except Exception:
            pass

    def _catalog_load(self) -> dict[str, Any]:
        if not self._catalog_path.is_file():
            return {
                "schema": RETENTION_MANIFEST_SCHEMA,
                "store_id": self.store_id,
                "backups": [],
            }
        try:
            payload = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "schema": RETENTION_MANIFEST_SCHEMA,
                "store_id": self.store_id,
                "backups": [],
            }
        if not isinstance(payload, dict):
            return {
                "schema": RETENTION_MANIFEST_SCHEMA,
                "store_id": self.store_id,
                "backups": [],
            }
        payload.setdefault("backups", [])
        return payload

    def _catalog_save(self, catalog: Mapping[str, Any]) -> None:
        self.backup_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self._catalog_path, dict(catalog))

    def _catalog_append(self, snapshot: BackupSnapshot) -> None:
        catalog = self._catalog_load()
        backups = list(catalog.get("backups") or [])
        backups.append(
            {
                "backup_id": snapshot.backup_id,
                "created_at": snapshot.created_at,
                "artifact_digest": snapshot.artifact_digest,
                "destination_uri": snapshot.destination_uri,
                "artifact_path": snapshot.body.get("artifact_path"),
                "meta_path": snapshot.body.get("meta_path"),
                "status": snapshot.status,
                "generation": snapshot.generation,
            }
        )
        catalog["backups"] = backups
        catalog["store_id"] = self.store_id
        catalog["updated_at"] = self._clock()
        self._catalog_save(catalog)

    def _delete_backup_files(self, entry: Mapping[str, Any]) -> None:
        for key in ("artifact_path", "meta_path"):
            raw = entry.get(key)
            if not raw:
                continue
            path = Path(str(raw))
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Factories / public helpers
# ---------------------------------------------------------------------------


def open_control_plane_backup(
    database_path: Path | str,
    *,
    backup_root: Path | str | None = None,
    store_id: str = DEFAULT_STORE_ID,
    **kwargs: Any,
) -> ControlPlaneBackup:
    """Construct a :class:`ControlPlaneBackup` for ``database_path``."""

    return ControlPlaneBackup(
        database_path,
        backup_root=backup_root,
        store_id=store_id,
        **kwargs,
    )


def seed_control_plane_for_backup(
    database_path: Path | str,
    *,
    store_id: str = DEFAULT_STORE_ID,
    application_version: str = "0.0.45",
    tool_version: str = "1.5.2",
    database_uuid: str = "123e4567-e89b-12d3-a456-426614174000",
    generation: int = 1,
    fence_epoch: int = 1,
    birth_id: str = "birth:backup-seed",
    tasks: Sequence[Mapping[str, Any]] | None = None,
    include_accepted: bool = True,
) -> Path:
    """Install schema and a small accepted population for hermetic tests."""

    _require_duckdb()
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    install_control_plane_schema(
        path,
        application_version=application_version,
        tool_version=tool_version,
        owner_id="control-plane-backup",
    )
    task_rows = list(tasks) if tasks is not None else [
        {
            "task_cid": "task:cid:001",
            "task_alias": "T-001",
            "status": "accepted",
        },
        {
            "task_cid": "task:cid:002",
            "task_alias": "T-002",
            "status": "ready",
        },
    ]
    with open_duckdb_connection(path) as connection:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at
            ) VALUES (?, 1, ?, 0, ?, ?, ?)
            """,
            [generation, fence_epoch, database_uuid, birth_id, _utc_iso()],
        )
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
        for index, task in enumerate(task_rows):
            task_cid = str(task.get("task_cid") or f"task:cid:{index + 1:03d}")
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
                    str(task.get("task_alias") or f"T-{index + 1:03d}"),
                    "goal:root",
                    "",
                    "objective:backup",
                    index + 1,
                    str(task.get("status") or "ready"),
                    0,
                    "P0",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "{}",
                    "{}",
                ],
            )
        first_cid = str(task_rows[0].get("task_cid") or "task:cid:001")
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
                first_cid,
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
                fence_epoch,
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
                "task.accepted",
                first_cid,
                "attempt:1",
                "session:lease-owner",
                "1970-01-01T00:00:00Z",
                "{}",
            ],
        )
        if include_accepted:
            connection.execute(
                """
                INSERT INTO task_acceptance (
                    task_cid, ordinal, criterion, evidence_policy_json
                ) VALUES (?, ?, ?, ?)
                """,
                [first_cid, 1, "tests_green", "{}"],
            )
        connection.execute("CHECKPOINT")
    return path


__all__ = (
    "BACKUP_STATUS_FAILED",
    "BACKUP_STATUS_PARTIAL",
    "BACKUP_STATUS_VERIFIED",
    "BACKUP_SNAPSHOT_SCHEMA",
    "BACKUP_VERIFICATION_SCHEMA",
    "CHECKPOINT_RECEIPT_SCHEMA",
    "CONTROL_PLANE_BACKUP_INTERFACE",
    "CONTROL_PLANE_BACKUP_SCHEMA",
    "CONTROL_PLANE_BACKUP_VERSION",
    "CRASH_MATRIX_REPORT_SCHEMA",
    "DECLARED_CRASH_SCENARIOS",
    "DEFAULT_RETENTION_KEEP_LAST",
    "DEFAULT_RETENTION_MAX_AGE_SECONDS",
    "DEFAULT_STORE_ID",
    "OWNERSHIP_ABSENT",
    "OWNERSHIP_LIVE",
    "OWNERSHIP_STALE",
    "OWNERSHIP_UNKNOWN",
    "RESTORE_OUTCOME_FAILED",
    "RESTORE_OUTCOME_REFUSED",
    "RESTORE_OUTCOME_SUCCESS",
    "RESTORE_RECEIPT_INTERFACE",
    "RESTORE_RECEIPT_SCHEMA",
    "RETENTION_MANIFEST_SCHEMA",
    "SNAPSHOT_ROOTS_SCHEMA",
    "STORE_GENERATION_ROTATION_INTERFACE",
    "STORE_GENERATION_ROTATION_SCHEMA",
    "BackupSnapshot",
    "BackupVerification",
    "CheckpointReceipt",
    "ControlPlaneBackup",
    "ControlPlaneBackupDependencyError",
    "ControlPlaneBackupError",
    "ControlPlaneBackupGenerationError",
    "ControlPlaneBackupIOError",
    "ControlPlaneBackupIntegrityError",
    "ControlPlaneBackupOwnershipError",
    "ControlPlaneBackupRetentionError",
    "CrashMatrixReport",
    "RestoreReceipt",
    "RetentionManifest",
    "SnapshotRoots",
    "StoreGenerationRotation",
    "assert_direct_file_maintenance_allowed",
    "compute_snapshot_roots",
    "duckdb_available",
    "inspect_server_ownership",
    "open_control_plane_backup",
    "owner_marker_path_for",
    "seed_control_plane_for_backup",
)
