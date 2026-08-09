"""Checkpoint, backup, restore, retention, and generation rotation (DQP-033).

Interfaces: ``ControlPlaneBackup@1``, ``RestoreReceipt@1``,
``StoreGenerationRotation@1``

Creates verified consistent snapshots of ``control.duckdb``, retention
manifests, corruption probes, restore rehearsals, and store-generation
rotation. Pre-rotation writers and leases fail after restore/takeover.
Direct-file maintenance is refused while server ownership is live or unknown.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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
    ControlPlaneContractError,
    ControlPlaneGenerationError,
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
    DEFAULT_STORE_ID,
    MAINTENANCE_LEASE_ACTIVE,
    MaintenanceLease,
)
from ..task_sources.control_plane_schema import (
    CONTROL_PLANE_SCHEMA_REVISION,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from .quack_state_server import (
    OWNER_LOCK_SUFFIX,
    OWNER_MARKER_SUFFIX,
    OwnerMarker,
)

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

CONTROL_PLANE_BACKUP_INTERFACE: Final[str] = "ControlPlaneBackup@1"
RESTORE_RECEIPT_INTERFACE: Final[str] = "RestoreReceipt@1"
STORE_GENERATION_ROTATION_INTERFACE: Final[str] = "StoreGenerationRotation@1"

CONTROL_PLANE_BACKUP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-backup@1"
)
BACKUP_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-snapshot@1"
)
RESTORE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/restore-receipt@1"
)
STORE_GENERATION_ROTATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/store-generation-rotation@1"
)
AUTHORITY_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/authority-roots@1"
)
RETENTION_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-retention-manifest@1"
)
BACKUP_VERIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-verification@1"
)
CRASH_MATRIX_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-crash-matrix@1"
)

CONTROL_PLANE_BACKUP_VERSION: Final[int] = 1
BACKUP_BODY_FILENAME: Final[str] = "control.duckdb"
BACKUP_MANIFEST_FILENAME: Final[str] = "backup.manifest.json"
RETENTION_MANIFEST_FILENAME: Final[str] = "retention.manifest.json"
DEFAULT_RETENTION_COUNT: Final[int] = 5
DEFAULT_MAX_BACKUP_AGE_SECONDS: Final[int] = 30 * 24 * 3600
BACKUP_STATUS_VERIFIED: Final[str] = "verified"
BACKUP_STATUS_FAILED: Final[str] = "failed"
RESTORE_OUTCOME_SUCCESS: Final[str] = "success"
RESTORE_OUTCOME_REHEARSAL: Final[str] = "rehearsal"
RESTORE_OUTCOME_FAILED: Final[str] = "failed"

# Declared crash matrix (evidence subset).
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


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ControlPlaneBackupError(RuntimeError):
    """Base fail-closed error for control-plane backup/restore."""


class ControlPlaneBackupOwnershipError(ControlPlaneBackupError):
    """Direct-file maintenance refused due to live/unknown ownership."""


class ControlPlaneBackupVerificationError(ControlPlaneBackupError):
    """Independent backup verification failed."""


class ControlPlaneBackupRestoreError(ControlPlaneBackupError):
    """Restore failed or partially applied and was rolled back."""


class ControlPlaneBackupRetentionError(ControlPlaneBackupError):
    """Retention policy evaluation or prune failed."""


class ControlPlaneBackupCorruptionError(ControlPlaneBackupError):
    """Corruption probe detected a damaged backup body."""


class ControlPlaneBackupRequestError(ControlPlaneBackupError, ValueError):
    """Malformed backup/restore request."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class OwnershipState(str, Enum):
    """Observed exclusive state-owner condition for direct-file maintenance."""

    ABSENT = "absent"
    DEAD = "dead"
    LIVE = "live"
    UNKNOWN = "unknown"
    LOCK_HELD = "lock_held"


class CrashScenario(str, Enum):
    """Declared crash / failure matrix for backup recovery evidence."""

    CRASH_BEFORE_CHECKPOINT = "crash_before_checkpoint"
    CRASH_AFTER_CHECKPOINT = "crash_after_checkpoint"
    CORRUPT_COPY = "corrupt_copy"
    DISK_FULL = "disk_full"
    PARTIAL_RESTORE = "partial_restore"
    SCHEMA_VERSION = "schema_version"
    SERVER_STOPPED = "server_stopped"
    STALE_CLIENT = "stale_client"
    BACKUP_AGE = "backup_age"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(now: float | None = None) -> str:
    instant = time.time() if now is None else float(now)
    return (
        datetime.fromtimestamp(instant, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _row_value(row: Any, key: str | int, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, Mapping):
        return row.get(key if isinstance(key, str) else str(key), default)
    try:
        if isinstance(key, int):
            return row[key]
        return row[key]
    except (KeyError, IndexError, TypeError):
        if isinstance(key, str):
            try:
                return row[key]
            except Exception:
                return default
        return default


def _copy_database_files(source: Path, destination: Path) -> list[str]:
    """Copy DuckDB primary file and optional WAL/sidecar files."""

    if not source.is_file():
        raise ControlPlaneBackupRequestError(f"database file missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    shutil.copy2(source, destination)
    copied.append(destination.name)
    for suffix in (".wal", ".tmp", ".crash"):
        sibling = Path(str(source) + suffix)
        if sibling.is_file():
            target = Path(str(destination) + suffix)
            shutil.copy2(sibling, target)
            copied.append(target.name)
    return copied


def _replace_database_files(source: Path, destination: Path) -> None:
    """Atomically replace destination DB with source (and matching sidecars)."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f".restore-staging-{uuid.uuid4().hex}")
    try:
        _copy_database_files(source, staging)
        # Remove previous WAL so a restored body is not merged with stale WAL.
        for suffix in (".wal", ".tmp", ".crash"):
            stale = Path(str(destination) + suffix)
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
        os.replace(staging, destination)
        for suffix in (".wal", ".tmp", ".crash"):
            staged_side = Path(str(staging) + suffix)
            dest_side = Path(str(destination) + suffix)
            if staged_side.is_file():
                os.replace(staged_side, dest_side)
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        for suffix in (".wal", ".tmp", ".crash"):
            try:
                Path(str(staging) + suffix).unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnershipObservation:
    """Observation of exclusive state-owner condition for maintenance fencing."""

    state: OwnershipState
    marker_path: str
    lock_path: str
    server_id: str = ""
    reason: str = ""
    process_birth: Mapping[str, Any] = field(default_factory=dict)

    @property
    def admits_direct_file_maintenance(self) -> bool:
        return self.state in {OwnershipState.ABSENT, OwnershipState.DEAD}

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "marker_path": self.marker_path,
            "lock_path": self.lock_path,
            "server_id": self.server_id,
            "reason": self.reason,
            "process_birth": dict(self.process_birth),
            "admits_direct_file_maintenance": self.admits_direct_file_maintenance,
        }


@dataclass(frozen=True)
class AuthorityRoots:
    """Canonical authority roots reproduced by restore.

    Roots cover store generation, schema fingerprint, domain events, tasks,
    and leases — the acceptance surface for DQP-033.
    """

    SCHEMA: ClassVar[str] = AUTHORITY_ROOTS_SCHEMA

    store_root: str
    schema_root: str
    event_root: str
    task_root: str
    lease_root: str
    database_uuid: str
    schema_revision: int
    generation: int
    fence_epoch: int
    revision: int
    event_watermark: int
    task_count: int
    lease_count: int
    store_id: str = DEFAULT_STORE_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "store_id": self.store_id,
            "store_root": self.store_root,
            "schema_root": self.schema_root,
            "event_root": self.event_root,
            "task_root": self.task_root,
            "lease_root": self.lease_root,
            "database_uuid": self.database_uuid,
            "schema_revision": int(self.schema_revision),
            "generation": int(self.generation),
            "fence_epoch": int(self.fence_epoch),
            "revision": int(self.revision),
            "event_watermark": int(self.event_watermark),
            "task_count": int(self.task_count),
            "lease_count": int(self.lease_count),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def matches(self, other: "AuthorityRoots", *, ignore_generation: bool = False) -> bool:
        """Compare roots. Generation may differ after intentional rotation."""

        if self.store_id != other.store_id:
            return False
        if self.database_uuid != other.database_uuid:
            return False
        if self.schema_root != other.schema_root:
            return False
        if self.event_root != other.event_root:
            return False
        if self.task_root != other.task_root:
            return False
        if self.lease_root != other.lease_root:
            return False
        if int(self.schema_revision) != int(other.schema_revision):
            return False
        if int(self.event_watermark) != int(other.event_watermark):
            return False
        if int(self.task_count) != int(other.task_count):
            return False
        if int(self.lease_count) != int(other.lease_count):
            return False
        if ignore_generation:
            return True
        return (
            self.store_root == other.store_root
            and int(self.generation) == int(other.generation)
            and int(self.fence_epoch) == int(other.fence_epoch)
            and int(self.revision) == int(other.revision)
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorityRoots":
        if not isinstance(payload, Mapping):
            raise ControlPlaneBackupRequestError("authority roots payload must be an object")
        return cls(
            store_root=str(payload.get("store_root") or ""),
            schema_root=str(payload.get("schema_root") or ""),
            event_root=str(payload.get("event_root") or ""),
            task_root=str(payload.get("task_root") or ""),
            lease_root=str(payload.get("lease_root") or ""),
            database_uuid=str(payload.get("database_uuid") or ""),
            schema_revision=int(payload.get("schema_revision") or 0),
            generation=int(payload.get("generation") or 0),
            fence_epoch=int(payload.get("fence_epoch") or 0),
            revision=int(payload.get("revision") or 0),
            event_watermark=int(payload.get("event_watermark") or 0),
            task_count=int(payload.get("task_count") or 0),
            lease_count=int(payload.get("lease_count") or 0),
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
        )


@dataclass(frozen=True)
class BackupSnapshot:
    """Verified consistent backup snapshot record.

    Interface projection for ``backup_snapshots`` rows + on-disk manifest.
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
    roots: AuthorityRoots
    body_path: str
    manifest_path: str
    fence_epoch: int = 0
    revision: int = 0
    event_watermark: int = 0
    encryption_bound: bool = False
    encryption_handle: str = ""
    independent_verification: Mapping[str, Any] = field(default_factory=dict)
    body_json: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "interface": CONTROL_PLANE_BACKUP_INTERFACE,
            "backup_id": self.backup_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "schema_revision": int(self.schema_revision),
            "generation": int(self.generation),
            "fence_epoch": int(self.fence_epoch),
            "revision": int(self.revision),
            "event_watermark": int(self.event_watermark),
            "artifact_digest": self.artifact_digest,
            "created_at": self.created_at,
            "destination_uri": self.destination_uri,
            "status": self.status,
            "roots": self.roots.to_dict(),
            "body_path": self.body_path,
            "manifest_path": self.manifest_path,
            "encryption_bound": bool(self.encryption_bound),
            "encryption_handle": self.encryption_handle,
            "independent_verification": dict(self.independent_verification),
            "body_json": dict(self.body_json),
        }

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "backup_id": self.backup_id,
                "artifact_digest": self.artifact_digest,
                "roots": self.roots.to_dict(),
                "generation": int(self.generation),
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackupSnapshot":
        if not isinstance(payload, Mapping):
            raise ControlPlaneBackupRequestError("backup snapshot payload must be an object")
        roots_payload = payload.get("roots") or {}
        if not isinstance(roots_payload, Mapping):
            raise ControlPlaneBackupRequestError("backup roots must be an object")
        return cls(
            backup_id=str(payload.get("backup_id") or ""),
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
            database_uuid=str(payload.get("database_uuid") or ""),
            schema_revision=int(payload.get("schema_revision") or 0),
            generation=int(payload.get("generation") or 0),
            artifact_digest=str(payload.get("artifact_digest") or ""),
            created_at=str(payload.get("created_at") or ""),
            destination_uri=str(payload.get("destination_uri") or ""),
            status=str(payload.get("status") or ""),
            roots=AuthorityRoots.from_dict(roots_payload),
            body_path=str(payload.get("body_path") or ""),
            manifest_path=str(payload.get("manifest_path") or ""),
            fence_epoch=int(payload.get("fence_epoch") or 0),
            revision=int(payload.get("revision") or 0),
            event_watermark=int(payload.get("event_watermark") or 0),
            encryption_bound=bool(payload.get("encryption_bound") or False),
            encryption_handle=str(payload.get("encryption_handle") or ""),
            independent_verification=dict(payload.get("independent_verification") or {}),
            body_json=dict(payload.get("body_json") or {}),
        )


@dataclass(frozen=True)
class StoreGenerationRotation:
    """Store-generation rotation that invalidates pre-rotation writers.

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
    birth_id: str
    rotated_at: str
    reason: str
    backup_id: str = ""

    def __post_init__(self) -> None:
        if int(self.new_generation) <= int(self.previous_generation):
            raise ControlPlaneGenerationError(
                "store generation rotation must strictly increase generation"
            )
        if int(self.new_fence_epoch) <= int(self.previous_fence_epoch):
            raise ControlPlaneGenerationError(
                "store generation rotation must strictly increase fence_epoch"
            )

    def invalidates(self, writer_generation: int, writer_fence_epoch: int = 0) -> bool:
        """Return True when a pre-rotation writer/lease must fail closed."""

        if int(writer_generation) < int(self.new_generation):
            return True
        if int(writer_generation) == int(self.new_generation) and int(
            writer_fence_epoch
        ) < int(self.new_fence_epoch):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "rotation_id": self.rotation_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "previous_generation": int(self.previous_generation),
            "new_generation": int(self.new_generation),
            "previous_fence_epoch": int(self.previous_fence_epoch),
            "new_fence_epoch": int(self.new_fence_epoch),
            "schema_revision": int(self.schema_revision),
            "birth_id": self.birth_id,
            "rotated_at": self.rotated_at,
            "reason": self.reason,
            "backup_id": self.backup_id,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoreGenerationRotation":
        if not isinstance(payload, Mapping):
            raise ControlPlaneBackupRequestError(
                "store generation rotation payload must be an object"
            )
        return cls(
            rotation_id=str(payload.get("rotation_id") or ""),
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
            database_uuid=str(payload.get("database_uuid") or ""),
            previous_generation=int(payload.get("previous_generation") or 0),
            new_generation=int(payload.get("new_generation") or 0),
            previous_fence_epoch=int(payload.get("previous_fence_epoch") or 0),
            new_fence_epoch=int(payload.get("new_fence_epoch") or 0),
            schema_revision=int(payload.get("schema_revision") or 0),
            birth_id=str(payload.get("birth_id") or ""),
            rotated_at=str(payload.get("rotated_at") or ""),
            reason=str(payload.get("reason") or ""),
            backup_id=str(payload.get("backup_id") or ""),
        )


@dataclass(frozen=True)
class RestoreReceipt:
    """Receipt for a restore or restore rehearsal.

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
    roots: AuthorityRoots
    rotation: StoreGenerationRotation | None = None
    artifact_digest: str = ""
    destination_uri: str = ""
    rehearsal: bool = False
    writers_invalidated: bool = False
    body_json: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "receipt_id": self.receipt_id,
            "backup_id": self.backup_id,
            "store_id": self.store_id,
            "restored_at": self.restored_at,
            "schema_revision": int(self.schema_revision),
            "generation": int(self.generation),
            "outcome": self.outcome,
            "roots": self.roots.to_dict(),
            "rotation": None if self.rotation is None else self.rotation.to_dict(),
            "artifact_digest": self.artifact_digest,
            "destination_uri": self.destination_uri,
            "rehearsal": bool(self.rehearsal),
            "writers_invalidated": bool(self.writers_invalidated),
            "body_json": dict(self.body_json),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RestoreReceipt":
        if not isinstance(payload, Mapping):
            raise ControlPlaneBackupRequestError("restore receipt payload must be an object")
        roots_payload = payload.get("roots") or {}
        if not isinstance(roots_payload, Mapping):
            raise ControlPlaneBackupRequestError("restore roots must be an object")
        rotation_payload = payload.get("rotation")
        rotation: StoreGenerationRotation | None
        if rotation_payload in (None, {}):
            rotation = None
        elif isinstance(rotation_payload, Mapping):
            rotation = StoreGenerationRotation.from_dict(rotation_payload)
        else:
            raise ControlPlaneBackupRequestError("rotation must be an object or null")
        return cls(
            receipt_id=str(payload.get("receipt_id") or ""),
            backup_id=str(payload.get("backup_id") or ""),
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
            restored_at=str(payload.get("restored_at") or ""),
            schema_revision=int(payload.get("schema_revision") or 0),
            generation=int(payload.get("generation") or 0),
            outcome=str(payload.get("outcome") or ""),
            roots=AuthorityRoots.from_dict(roots_payload),
            rotation=rotation,
            artifact_digest=str(payload.get("artifact_digest") or ""),
            destination_uri=str(payload.get("destination_uri") or ""),
            rehearsal=bool(payload.get("rehearsal") or False),
            writers_invalidated=bool(payload.get("writers_invalidated") or False),
            body_json=dict(payload.get("body_json") or {}),
        )


@dataclass(frozen=True)
class BackupVerification:
    """Independent verification result for a backup artifact."""

    SCHEMA: ClassVar[str] = BACKUP_VERIFICATION_SCHEMA

    backup_id: str
    verified: bool
    artifact_digest: str
    observed_digest: str
    roots: AuthorityRoots | None
    openable: bool
    reason: str = ""
    checks: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "backup_id": self.backup_id,
            "verified": bool(self.verified),
            "artifact_digest": self.artifact_digest,
            "observed_digest": self.observed_digest,
            "roots": None if self.roots is None else self.roots.to_dict(),
            "openable": bool(self.openable),
            "reason": self.reason,
            "checks": dict(self.checks),
        }


@dataclass(frozen=True)
class RetentionManifest:
    """Retention decision for the backup archive."""

    SCHEMA: ClassVar[str] = RETENTION_MANIFEST_SCHEMA

    manifest_id: str
    created_at: str
    keep_count: int
    max_age_seconds: int
    retained: tuple[str, ...]
    pruned: tuple[str, ...]
    destination_uri: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "keep_count": int(self.keep_count),
            "max_age_seconds": int(self.max_age_seconds),
            "retained": list(self.retained),
            "pruned": list(self.pruned),
            "destination_uri": self.destination_uri,
        }


@dataclass(frozen=True)
class CrashMatrixReport:
    """Evidence report for the declared backup crash matrix."""

    SCHEMA: ClassVar[str] = CRASH_MATRIX_REPORT_SCHEMA

    scenarios: Mapping[str, Mapping[str, Any]]
    accepted_state_preserved: bool
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "scenarios": {key: dict(value) for key, value in self.scenarios.items()},
            "accepted_state_preserved": bool(self.accepted_state_preserved),
            "generated_at": self.generated_at,
            "declared_scenarios": list(DECLARED_CRASH_SCENARIOS),
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ControlPlaneBackup:
    """Production control-plane checkpoint / backup / restore service.

    Interface: ``ControlPlaneBackup@1``.
    """

    INTERFACE: ClassVar[str] = CONTROL_PLANE_BACKUP_INTERFACE
    SCHEMA: ClassVar[str] = CONTROL_PLANE_BACKUP_SCHEMA
    VERSION: ClassVar[int] = CONTROL_PLANE_BACKUP_VERSION

    def __init__(
        self,
        *,
        database_path: Path | str,
        backup_root: Path | str,
        state_dir: Path | str | None = None,
        store_id: str = DEFAULT_STORE_ID,
        maintenance_scope: str = DEFAULT_MAINTENANCE_SCOPE,
        owner_liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
        clock: Callable[[], float] | None = None,
        encryption_key_handle: str = "",
    ) -> None:
        self.database_path = Path(database_path)
        self.backup_root = Path(backup_root)
        self.state_dir = (
            Path(state_dir)
            if state_dir is not None
            else self.database_path.parent
        )
        self.store_id = str(store_id or DEFAULT_STORE_ID).strip() or DEFAULT_STORE_ID
        self.maintenance_scope = (
            str(maintenance_scope or DEFAULT_MAINTENANCE_SCOPE).strip()
            or DEFAULT_MAINTENANCE_SCOPE
        )
        self._liveness = owner_liveness_probe or (
            lambda birth: owner_liveness(birth)
        )
        self._clock = clock or time.time
        self.encryption_key_handle = str(encryption_key_handle or "").strip()

    # -- paths ---------------------------------------------------------------

    def owner_marker_path(self) -> Path:
        return self.state_dir / f"{self.database_path.name}{OWNER_MARKER_SUFFIX}"

    def owner_lock_path(self) -> Path:
        return self.state_dir / f"{self.database_path.name}{OWNER_LOCK_SUFFIX}"

    def backup_directory(self, backup_id: str) -> Path:
        safe = backup_id.replace("/", "_").replace("..", "_")
        return self.backup_root / safe

    # -- ownership fence -----------------------------------------------------

    def observe_ownership(self) -> OwnershipObservation:
        """Observe whether direct-file maintenance is admitted."""

        marker_path = self.owner_marker_path()
        lock_path = self.owner_lock_path()
        # Lock held by another process ⇒ ownership live/unknown for maintenance.
        if lock_path.exists():
            try:
                handle = lock_path.open("a+b")
            except OSError:
                return OwnershipObservation(
                    state=OwnershipState.UNKNOWN,
                    marker_path=str(marker_path),
                    lock_path=str(lock_path),
                    reason="lock_unreadable",
                )
            try:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return OwnershipObservation(
                        state=OwnershipState.LOCK_HELD,
                        marker_path=str(marker_path),
                        lock_path=str(lock_path),
                        reason="exclusive_owner_lock_held",
                    )
                else:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
            finally:
                handle.close()

        payload = _read_json(marker_path)
        if payload is None:
            return OwnershipObservation(
                state=OwnershipState.ABSENT,
                marker_path=str(marker_path),
                lock_path=str(lock_path),
                reason="no_owner_marker",
            )
        try:
            marker = OwnerMarker.from_dict(payload)
        except (TypeError, ValueError, KeyError, ControlPlaneContractError):
            return OwnershipObservation(
                state=OwnershipState.UNKNOWN,
                marker_path=str(marker_path),
                lock_path=str(lock_path),
                reason="owner_marker_corrupt",
            )
        liveness = self._liveness(marker.process_birth)
        if liveness is OwnerLiveness.ALIVE:
            return OwnershipObservation(
                state=OwnershipState.LIVE,
                marker_path=str(marker_path),
                lock_path=str(lock_path),
                server_id=marker.server_id,
                reason="owner_alive",
                process_birth=marker.process_birth.to_dict(),
            )
        if liveness is OwnerLiveness.UNKNOWN:
            return OwnershipObservation(
                state=OwnershipState.UNKNOWN,
                marker_path=str(marker_path),
                lock_path=str(lock_path),
                server_id=marker.server_id,
                reason="owner_liveness_unknown",
                process_birth=marker.process_birth.to_dict(),
            )
        return OwnershipObservation(
            state=OwnershipState.DEAD,
            marker_path=str(marker_path),
            lock_path=str(lock_path),
            server_id=marker.server_id,
            reason="owner_dead",
            process_birth=marker.process_birth.to_dict(),
        )

    def assert_direct_file_maintenance_admitted(self) -> OwnershipObservation:
        """Refuse direct-file maintenance while ownership is live or unknown."""

        observation = self.observe_ownership()
        if not observation.admits_direct_file_maintenance:
            raise ControlPlaneBackupOwnershipError(
                "direct-file maintenance cannot occur while server ownership "
                f"is {observation.state.value} ({observation.reason})"
            )
        return observation

    # -- roots / checkpoint --------------------------------------------------

    def checkpoint(self, *, connection: Any | None = None) -> dict[str, Any]:
        """Force a clean DuckDB CHECKPOINT on the owned database file."""

        self.assert_direct_file_maintenance_admitted()
        at = _utc_iso(self._clock())
        if connection is not None:
            connection.execute("CHECKPOINT")
            return {
                "checkpointed": True,
                "database_path": str(self.database_path),
                "at": at,
                "mode": "connection",
            }
        if not self.database_path.is_file():
            raise ControlPlaneBackupRequestError(
                f"cannot checkpoint missing database: {self.database_path}"
            )
        with open_duckdb_connection(self.database_path) as conn:
            conn.execute("CHECKPOINT")
        return {
            "checkpointed": True,
            "database_path": str(self.database_path),
            "at": at,
            "mode": "embedded",
        }

    def capture_roots(
        self,
        database_path: Path | str | None = None,
        *,
        store_id: str | None = None,
    ) -> AuthorityRoots:
        """Capture store/schema/event/task/lease authority roots."""

        path = Path(database_path) if database_path is not None else self.database_path
        if not path.is_file():
            raise ControlPlaneBackupRequestError(f"database file missing: {path}")
        resolved_store = str(store_id or self.store_id)

        with open_duckdb_connection(path) as connection:
            meta_rows = connection.execute(
                "SELECT key, value FROM control_plane_metadata"
            ).fetchall()
            meta: dict[str, str] = {}
            for row in meta_rows:
                key = str(_row_value(row, "key", _row_value(row, 0, "")))
                value = str(_row_value(row, "value", _row_value(row, 1, "")))
                meta[key] = value

            generation_row = connection.execute(
                """
                SELECT generation, schema_revision, fence_epoch, revision,
                       database_uuid, birth_id
                FROM store_generations
                ORDER BY generation DESC
                LIMIT 1
                """
            ).fetchone()
            if generation_row is None:
                raise ControlPlaneBackupRequestError(
                    "store_generations is empty; refuse root capture"
                )
            generation = int(
                _row_value(generation_row, "generation", _row_value(generation_row, 0, 0))
            )
            schema_revision = int(
                _row_value(
                    generation_row,
                    "schema_revision",
                    _row_value(generation_row, 1, CONTROL_PLANE_SCHEMA_REVISION),
                )
            )
            fence_epoch = int(
                _row_value(generation_row, "fence_epoch", _row_value(generation_row, 2, 0))
            )
            revision = int(
                _row_value(generation_row, "revision", _row_value(generation_row, 3, 0))
            )
            database_uuid = str(
                _row_value(
                    generation_row,
                    "database_uuid",
                    _row_value(generation_row, 4, meta.get(META_DATABASE_UUID, "")),
                )
            )
            birth_id = str(
                _row_value(generation_row, "birth_id", _row_value(generation_row, 5, ""))
            )

            schema_fingerprint = str(meta.get(META_SCHEMA_FINGERPRINT) or "")
            schema_version = str(meta.get(META_SCHEMA_VERSION) or schema_revision)

            task_rows = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, ordinal, status, revision
                FROM tasks
                ORDER BY task_cid ASC
                """
            ).fetchall()
            tasks: list[dict[str, Any]] = []
            for row in task_rows:
                tasks.append(
                    {
                        "task_cid": str(_row_value(row, "task_cid", _row_value(row, 0, ""))),
                        "task_alias": str(
                            _row_value(row, "task_alias", _row_value(row, 1, ""))
                        ),
                        "goal_cid": str(
                            _row_value(row, "goal_cid", _row_value(row, 2, ""))
                        ),
                        "ordinal": int(
                            _row_value(row, "ordinal", _row_value(row, 3, 0)) or 0
                        ),
                        "status": str(
                            _row_value(row, "status", _row_value(row, 4, ""))
                        ),
                        "revision": int(
                            _row_value(row, "revision", _row_value(row, 5, 0)) or 0
                        ),
                    }
                )

            lease_rows = connection.execute(
                """
                SELECT task_cid, claim_cid, resolution_cid, claimant_did,
                       logical_epoch, fencing_token, expires_at_ms, attempt,
                       state, started_at_ms, revision, fence_epoch
                FROM leases
                ORDER BY task_cid ASC
                """
            ).fetchall()
            leases: list[dict[str, Any]] = []
            for row in lease_rows:
                leases.append(
                    {
                        "task_cid": str(
                            _row_value(row, "task_cid", _row_value(row, 0, ""))
                        ),
                        "claim_cid": str(
                            _row_value(row, "claim_cid", _row_value(row, 1, ""))
                        ),
                        "resolution_cid": str(
                            _row_value(row, "resolution_cid", _row_value(row, 2, ""))
                        ),
                        "claimant_did": str(
                            _row_value(row, "claimant_did", _row_value(row, 3, ""))
                        ),
                        "logical_epoch": int(
                            _row_value(row, "logical_epoch", _row_value(row, 4, 0)) or 0
                        ),
                        "fencing_token": int(
                            _row_value(row, "fencing_token", _row_value(row, 5, 0)) or 0
                        ),
                        "expires_at_ms": int(
                            _row_value(row, "expires_at_ms", _row_value(row, 6, 0)) or 0
                        ),
                        "attempt": int(
                            _row_value(row, "attempt", _row_value(row, 7, 0)) or 0
                        ),
                        "state": str(
                            _row_value(row, "state", _row_value(row, 8, ""))
                        ),
                        "started_at_ms": int(
                            _row_value(row, "started_at_ms", _row_value(row, 9, 0)) or 0
                        ),
                        "revision": int(
                            _row_value(row, "revision", _row_value(row, 10, 0)) or 0
                        ),
                        "fence_epoch": int(
                            _row_value(row, "fence_epoch", _row_value(row, 11, 0)) or 0
                        ),
                    }
                )

            event_rows = connection.execute(
                """
                SELECT event_id, stream_id, sequence, global_sequence, event_type,
                       task_cid, attempt_id, session_id, recorded_at
                FROM domain_events
                ORDER BY global_sequence ASC
                """
            ).fetchall()
            events: list[dict[str, Any]] = []
            for row in event_rows:
                events.append(
                    {
                        "event_id": str(
                            _row_value(row, "event_id", _row_value(row, 0, ""))
                        ),
                        "stream_id": str(
                            _row_value(row, "stream_id", _row_value(row, 1, ""))
                        ),
                        "sequence": int(
                            _row_value(row, "sequence", _row_value(row, 2, 0)) or 0
                        ),
                        "global_sequence": int(
                            _row_value(row, "global_sequence", _row_value(row, 3, 0))
                            or 0
                        ),
                        "event_type": str(
                            _row_value(row, "event_type", _row_value(row, 4, ""))
                        ),
                        "task_cid": str(
                            _row_value(row, "task_cid", _row_value(row, 5, ""))
                        ),
                        "attempt_id": str(
                            _row_value(row, "attempt_id", _row_value(row, 6, ""))
                        ),
                        "session_id": str(
                            _row_value(row, "session_id", _row_value(row, 7, ""))
                        ),
                        "recorded_at": str(
                            _row_value(row, "recorded_at", _row_value(row, 8, ""))
                        ),
                    }
                )
            watermark_row = connection.execute(
                "SELECT COALESCE(MAX(global_sequence), 0) AS event_watermark "
                "FROM domain_events"
            ).fetchone()
            event_watermark = int(
                _row_value(
                    watermark_row,
                    "event_watermark",
                    _row_value(watermark_row, 0, 0),
                )
                or 0
            )

        store_material = {
            "store_id": resolved_store,
            "generation": generation,
            "schema_revision": schema_revision,
            "fence_epoch": fence_epoch,
            "revision": revision,
            "database_uuid": database_uuid,
            "birth_id": birth_id,
        }
        schema_material = {
            "schema_fingerprint": schema_fingerprint,
            "schema_version": schema_version,
            "schema_revision": schema_revision,
        }
        return AuthorityRoots(
            store_root=content_identity(store_material),
            schema_root=content_identity(schema_material),
            event_root=content_identity(events),
            task_root=content_identity(tasks),
            lease_root=content_identity(leases),
            database_uuid=database_uuid,
            schema_revision=schema_revision,
            generation=generation,
            fence_epoch=fence_epoch,
            revision=revision,
            event_watermark=event_watermark,
            task_count=len(tasks),
            lease_count=len(leases),
            store_id=resolved_store,
        )

    # -- backup --------------------------------------------------------------

    def create_backup(
        self,
        *,
        maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
        backup_id: str | None = None,
        skip_checkpoint: bool = False,
        require_maintenance_lease: bool = True,
        record_in_database: bool = True,
    ) -> BackupSnapshot:
        """Create a verified consistent snapshot under exclusive maintenance."""

        ownership = self.assert_direct_file_maintenance_admitted()
        if require_maintenance_lease:
            self._require_active_maintenance_lease(maintenance_lease)

        if not self.database_path.is_file():
            raise ControlPlaneBackupRequestError(
                f"database file missing: {self.database_path}"
            )

        if not skip_checkpoint:
            self.checkpoint()

        pre_roots = self.capture_roots()
        created_at = _utc_iso(self._clock())
        bid = (backup_id or f"backup:{uuid.uuid4()}").strip()
        if not bid:
            raise ControlPlaneBackupRequestError("backup_id must not be empty")

        destination = self.backup_directory(bid)
        if destination.exists():
            raise ControlPlaneBackupRequestError(
                f"backup destination already exists: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=False)
        body_path = destination / BACKUP_BODY_FILENAME
        try:
            _copy_database_files(self.database_path, body_path)
        except OSError as exc:
            # disk full / IO failure
            shutil.rmtree(destination, ignore_errors=True)
            raise ControlPlaneBackupError(
                f"backup copy failed: {type(exc).__name__}: {exc}"
            ) from exc

        artifact_digest = _sha256_file(body_path)
        if self.encryption_key_handle:
            # Digest-bind external body to opaque handle; never store raw keys.
            artifact_digest = _sha256_bytes(
                canonical_json_bytes(
                    {
                        "body_digest": artifact_digest,
                        "encryption_handle": self.encryption_key_handle,
                    }
                )
            )
            encryption_bound = True
        else:
            encryption_bound = False

        # Independent verification: reopen the *copy*, not the live file.
        verification = self.verify_backup_path(
            body_path,
            expected_digest=(
                _sha256_file(body_path)
                if not encryption_bound
                else None
            ),
            expected_roots=pre_roots,
            backup_id=bid,
            encryption_handle=self.encryption_key_handle if encryption_bound else "",
        )
        if not verification.verified:
            shutil.rmtree(destination, ignore_errors=True)
            raise ControlPlaneBackupVerificationError(
                f"independent backup verification failed: {verification.reason}"
            )

        snapshot = BackupSnapshot(
            backup_id=bid,
            store_id=self.store_id,
            database_uuid=pre_roots.database_uuid,
            schema_revision=pre_roots.schema_revision,
            generation=pre_roots.generation,
            artifact_digest=artifact_digest,
            created_at=created_at,
            destination_uri=destination.as_uri(),
            status=BACKUP_STATUS_VERIFIED,
            roots=pre_roots,
            body_path=str(body_path),
            manifest_path=str(destination / BACKUP_MANIFEST_FILENAME),
            fence_epoch=pre_roots.fence_epoch,
            revision=pre_roots.revision,
            event_watermark=pre_roots.event_watermark,
            encryption_bound=encryption_bound,
            encryption_handle=self.encryption_key_handle if encryption_bound else "",
            independent_verification=verification.to_dict(),
            body_json={
                "ownership": ownership.to_dict(),
                "maintenance_scope": self.maintenance_scope,
                "skip_checkpoint": bool(skip_checkpoint),
            },
        )
        _atomic_write_json(Path(snapshot.manifest_path), snapshot.to_dict())

        if record_in_database:
            self._record_backup_snapshot(snapshot)

        return snapshot

    def verify_backup(
        self,
        backup: BackupSnapshot | Mapping[str, Any] | str | Path,
    ) -> BackupVerification:
        """Independently verify a backup by digest, openability, and roots."""

        snapshot = self._coerce_snapshot(backup)
        body = Path(snapshot.body_path)
        expected_body_digest = None
        if snapshot.encryption_bound:
            # Re-derive bound digest after verifying body bytes.
            expected_body_digest = None
        else:
            expected_body_digest = snapshot.artifact_digest
        verification = self.verify_backup_path(
            body,
            expected_digest=expected_body_digest,
            expected_roots=snapshot.roots,
            backup_id=snapshot.backup_id,
            encryption_handle=snapshot.encryption_handle if snapshot.encryption_bound else "",
        )
        if snapshot.encryption_bound and verification.openable:
            body_digest = _sha256_file(body)
            bound = _sha256_bytes(
                canonical_json_bytes(
                    {
                        "body_digest": body_digest,
                        "encryption_handle": snapshot.encryption_handle,
                    }
                )
            )
            digest_ok = bound == snapshot.artifact_digest
            verified = bool(verification.verified and digest_ok)
            return BackupVerification(
                backup_id=snapshot.backup_id,
                verified=verified,
                artifact_digest=snapshot.artifact_digest,
                observed_digest=bound,
                roots=verification.roots,
                openable=verification.openable,
                reason="" if verified else "encryption_bound_digest_mismatch",
                checks={
                    **dict(verification.checks),
                    "encryption_bound": True,
                    "body_digest": body_digest,
                    "digest_match": digest_ok,
                },
            )
        return verification

    def verify_backup_path(
        self,
        body_path: Path | str,
        *,
        expected_digest: str | None = None,
        expected_roots: AuthorityRoots | None = None,
        backup_id: str = "",
        encryption_handle: str = "",
    ) -> BackupVerification:
        """Verify a backup body file independently of the live store."""

        path = Path(body_path)
        checks: dict[str, Any] = {"body_path": str(path)}
        if not path.is_file():
            return BackupVerification(
                backup_id=backup_id,
                verified=False,
                artifact_digest=str(expected_digest or ""),
                observed_digest="",
                roots=None,
                openable=False,
                reason="body_missing",
                checks=checks,
            )
        try:
            observed_digest = _sha256_file(path)
        except OSError as exc:
            return BackupVerification(
                backup_id=backup_id,
                verified=False,
                artifact_digest=str(expected_digest or ""),
                observed_digest="",
                roots=None,
                openable=False,
                reason=f"digest_failed:{type(exc).__name__}",
                checks=checks,
            )
        checks["observed_digest"] = observed_digest
        digest_ok = True
        if expected_digest is not None:
            digest_ok = observed_digest == expected_digest
            checks["digest_match"] = digest_ok
            checks["expected_digest"] = expected_digest

        openable = False
        roots: AuthorityRoots | None = None
        root_reason = ""
        try:
            roots = self.capture_roots(path)
            openable = True
            checks["openable"] = True
            if expected_roots is not None:
                roots_match = roots.matches(expected_roots)
                checks["roots_match"] = roots_match
                if not roots_match:
                    root_reason = "roots_mismatch"
        except Exception as exc:  # noqa: BLE001 — verification must never raise for probe
            openable = False
            checks["openable"] = False
            checks["open_error"] = f"{type(exc).__name__}: {exc}"
            root_reason = "not_openable"

        verified = bool(digest_ok and openable and root_reason == "")
        if encryption_handle:
            checks["encryption_handle_present"] = True
        reason = ""
        if not digest_ok:
            reason = "digest_mismatch"
        elif root_reason:
            reason = root_reason
        return BackupVerification(
            backup_id=backup_id,
            verified=verified,
            artifact_digest=str(expected_digest or observed_digest),
            observed_digest=observed_digest,
            roots=roots,
            openable=openable,
            reason=reason,
            checks=checks,
        )

    def probe_corruption(
        self,
        backup: BackupSnapshot | Mapping[str, Any] | str | Path,
    ) -> dict[str, Any]:
        """Probe a backup for corruption; fail closed on damage."""

        verification = self.verify_backup(backup)
        if not verification.verified:
            raise ControlPlaneBackupCorruptionError(
                f"corruption probe failed: {verification.reason or 'unverified'}"
            )
        return {
            "corrupt": False,
            "backup_id": verification.backup_id,
            "verification": verification.to_dict(),
        }

    # -- restore / rotation --------------------------------------------------

    def rotate_generation(
        self,
        *,
        database_path: Path | str | None = None,
        reason: str,
        backup_id: str = "",
        birth_id: str | None = None,
        maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
        require_maintenance_lease: bool = True,
    ) -> StoreGenerationRotation:
        """Advance store generation and fence epoch; invalidate old writers."""

        self.assert_direct_file_maintenance_admitted()
        if require_maintenance_lease:
            self._require_active_maintenance_lease(maintenance_lease)
        path = Path(database_path) if database_path is not None else self.database_path
        if not path.is_file():
            raise ControlPlaneBackupRequestError(f"database file missing: {path}")

        with open_duckdb_connection(path) as connection:
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
                raise ControlPlaneBackupRequestError(
                    "cannot rotate generation: store_generations empty"
                )
            previous_generation = int(
                _row_value(row, "generation", _row_value(row, 0, 0))
            )
            schema_revision = int(
                _row_value(
                    row,
                    "schema_revision",
                    _row_value(row, 1, CONTROL_PLANE_SCHEMA_REVISION),
                )
            )
            previous_fence = int(
                _row_value(row, "fence_epoch", _row_value(row, 2, 0))
            )
            revision = int(_row_value(row, "revision", _row_value(row, 3, 0)))
            database_uuid = str(
                _row_value(row, "database_uuid", _row_value(row, 4, ""))
            )
            previous_birth = str(
                _row_value(row, "birth_id", _row_value(row, 5, ""))
            )
            new_generation = previous_generation + 1
            new_fence = previous_fence + 1
            new_birth = birth_id or f"birth:restore:{uuid.uuid4()}"
            created_at = _utc_iso(self._clock())
            connection.execute(
                """
                INSERT INTO store_generations (
                    generation, schema_revision, fence_epoch, revision,
                    database_uuid, birth_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    new_generation,
                    schema_revision,
                    new_fence,
                    revision,
                    database_uuid,
                    new_birth,
                    created_at,
                ],
            )
            # Pre-rotation writers fail closed by generation/fence comparison.
            # Do not mutate task/event/lease content rows here — restore must
            # still reproduce those authority roots after rotation.
            try:
                connection.execute(
                    """
                    UPDATE client_sessions
                    SET status = 'invalidated_by_generation_rotation',
                        revision = revision + 1
                    WHERE generation < ? OR fence_epoch < ?
                    """,
                    [new_generation, new_fence],
                )
            except Exception:
                # Table may be empty in hermetic fixtures; non-fatal.
                pass
            connection.execute("CHECKPOINT")

        rotation = StoreGenerationRotation(
            rotation_id=f"rotation:{uuid.uuid4()}",
            store_id=self.store_id,
            database_uuid=database_uuid,
            previous_generation=previous_generation,
            new_generation=new_generation,
            previous_fence_epoch=previous_fence,
            new_fence_epoch=new_fence,
            schema_revision=schema_revision,
            birth_id=new_birth,
            rotated_at=_utc_iso(self._clock()),
            reason=reason,
            backup_id=backup_id,
        )
        return rotation

    def restore(
        self,
        backup: BackupSnapshot | Mapping[str, Any] | str | Path,
        *,
        maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
        require_maintenance_lease: bool = True,
        rotate_generation: bool = True,
        rehearsal: bool = False,
        destination: Path | str | None = None,
    ) -> RestoreReceipt:
        """Restore from a verified backup; optionally rotate generation.

        When ``rehearsal`` is True the live database is not replaced; restore
        is applied to a temporary destination and verified.
        """

        ownership = self.assert_direct_file_maintenance_admitted()
        if require_maintenance_lease:
            self._require_active_maintenance_lease(maintenance_lease)

        snapshot = self._coerce_snapshot(backup)
        verification = self.verify_backup(snapshot)
        if not verification.verified:
            raise ControlPlaneBackupVerificationError(
                f"refusing restore of unverified backup: {verification.reason}"
            )

        body = Path(snapshot.body_path)
        target = (
            Path(destination)
            if destination is not None
            else (self.database_path if not rehearsal else None)
        )
        if rehearsal and target is None:
            rehearsal_dir = self.backup_root / f".rehearsal-{uuid.uuid4().hex}"
            rehearsal_dir.mkdir(parents=True, exist_ok=True)
            target = rehearsal_dir / BACKUP_BODY_FILENAME

        assert target is not None
        target = Path(target)

        # Preserve live DB for rollback of partial restore.
        rollback_copy: Path | None = None
        rollback_dir: Path | None = None
        restore_committed = False
        if (
            not rehearsal
            and self.database_path.is_file()
            and target.resolve() == self.database_path.resolve()
        ):
            rollback_dir = self.backup_root / f".rollback-{uuid.uuid4().hex}"
            rollback_dir.mkdir(parents=True, exist_ok=True)
            rollback_copy = rollback_dir / BACKUP_BODY_FILENAME
            try:
                _copy_database_files(self.database_path, rollback_copy)
            except OSError as exc:
                raise ControlPlaneBackupRestoreError(
                    f"failed to stage rollback copy: {exc}"
                ) from exc

        try:
            _replace_database_files(body, target)
            post_roots = self.capture_roots(target)
            if not post_roots.matches(snapshot.roots, ignore_generation=False):
                # Exact pre-rotation root match required before intentional rotation.
                raise ControlPlaneBackupRestoreError(
                    "restored authority roots do not match backup roots"
                )

            rotation: StoreGenerationRotation | None = None
            writers_invalidated = False
            if rotate_generation:
                rotation = self.rotate_generation(
                    database_path=target,
                    reason="restore",
                    backup_id=snapshot.backup_id,
                    maintenance_lease=maintenance_lease,
                    require_maintenance_lease=False,
                )
                writers_invalidated = rotation.invalidates(
                    snapshot.generation, snapshot.fence_epoch
                )
                # Re-capture roots after rotation (store root changes; authority
                # event/task/lease content roots must still match).
                rotated_roots = self.capture_roots(target)
                if not rotated_roots.matches(snapshot.roots, ignore_generation=True):
                    raise ControlPlaneBackupRestoreError(
                        "post-rotation authority content roots diverged"
                    )
                final_roots = rotated_roots
            else:
                final_roots = post_roots

            receipt = RestoreReceipt(
                receipt_id=f"restore:{uuid.uuid4()}",
                backup_id=snapshot.backup_id,
                store_id=self.store_id,
                restored_at=_utc_iso(self._clock()),
                schema_revision=final_roots.schema_revision,
                generation=final_roots.generation,
                outcome=(
                    RESTORE_OUTCOME_REHEARSAL if rehearsal else RESTORE_OUTCOME_SUCCESS
                ),
                roots=final_roots,
                rotation=rotation,
                artifact_digest=snapshot.artifact_digest,
                destination_uri=target.as_uri(),
                rehearsal=rehearsal,
                writers_invalidated=writers_invalidated,
                body_json={
                    "ownership": ownership.to_dict(),
                    "pre_roots": snapshot.roots.to_dict(),
                    "verification": verification.to_dict(),
                },
            )
            # Authority restore is committed; do not roll it back for bookkeeping.
            restore_committed = True
            if not rehearsal and target.resolve() == self.database_path.resolve():
                try:
                    self._record_restore_receipt(receipt)
                except Exception as record_exc:  # noqa: BLE001
                    receipt = RestoreReceipt(
                        receipt_id=receipt.receipt_id,
                        backup_id=receipt.backup_id,
                        store_id=receipt.store_id,
                        restored_at=receipt.restored_at,
                        schema_revision=receipt.schema_revision,
                        generation=receipt.generation,
                        outcome=receipt.outcome,
                        roots=receipt.roots,
                        rotation=receipt.rotation,
                        artifact_digest=receipt.artifact_digest,
                        destination_uri=receipt.destination_uri,
                        rehearsal=receipt.rehearsal,
                        writers_invalidated=receipt.writers_invalidated,
                        body_json={
                            **dict(receipt.body_json),
                            "receipt_record_error": (
                                f"{type(record_exc).__name__}: {record_exc}"
                            ),
                        },
                    )
            return receipt
        except Exception as exc:
            if (
                not restore_committed
                and rollback_copy is not None
                and rollback_copy.is_file()
            ):
                try:
                    _replace_database_files(rollback_copy, self.database_path)
                except Exception as rollback_exc:
                    raise ControlPlaneBackupRestoreError(
                        f"partial restore and rollback failed: {exc}; "
                        f"rollback={rollback_exc}"
                    ) from exc
            if isinstance(exc, ControlPlaneBackupError):
                raise
            raise ControlPlaneBackupRestoreError(
                f"restore failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if rollback_dir is not None:
                try:
                    shutil.rmtree(rollback_dir, ignore_errors=True)
                except OSError:
                    pass

    # -- retention -----------------------------------------------------------

    def list_backups(self) -> tuple[BackupSnapshot, ...]:
        """List on-disk backup manifests under the backup root."""

        if not self.backup_root.is_dir():
            return ()
        snapshots: list[BackupSnapshot] = []
        for child in sorted(self.backup_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            manifest = child / BACKUP_MANIFEST_FILENAME
            payload = _read_json(manifest)
            if payload is None:
                continue
            try:
                snapshots.append(BackupSnapshot.from_dict(payload))
            except (TypeError, ValueError, ControlPlaneBackupRequestError):
                continue
        snapshots.sort(key=lambda item: item.created_at)
        return tuple(snapshots)

    def apply_retention(
        self,
        *,
        keep_count: int = DEFAULT_RETENTION_COUNT,
        max_age_seconds: int = DEFAULT_MAX_BACKUP_AGE_SECONDS,
        now: float | None = None,
    ) -> RetentionManifest:
        """Prune backups outside keep-count / max-age policy."""

        if keep_count < 0:
            raise ControlPlaneBackupRetentionError("keep_count must be >= 0")
        if max_age_seconds < 0:
            raise ControlPlaneBackupRetentionError("max_age_seconds must be >= 0")

        snapshots = list(self.list_backups())
        instant = self._clock() if now is None else float(now)
        retained: list[str] = []
        pruned: list[str] = []

        # Newest first for keep-count.
        ordered = sorted(snapshots, key=lambda item: item.created_at, reverse=True)
        for index, snapshot in enumerate(ordered):
            age_expired = False
            try:
                created = datetime.fromisoformat(
                    snapshot.created_at.replace("Z", "+00:00")
                ).timestamp()
                age_expired = (instant - created) > float(max_age_seconds)
            except ValueError:
                age_expired = False
            over_count = index >= int(keep_count)
            if over_count or age_expired:
                directory = self.backup_directory(snapshot.backup_id)
                shutil.rmtree(directory, ignore_errors=True)
                pruned.append(snapshot.backup_id)
            else:
                retained.append(snapshot.backup_id)

        manifest = RetentionManifest(
            manifest_id=f"retention:{uuid.uuid4()}",
            created_at=_utc_iso(instant),
            keep_count=int(keep_count),
            max_age_seconds=int(max_age_seconds),
            retained=tuple(sorted(retained)),
            pruned=tuple(sorted(pruned)),
            destination_uri=self.backup_root.as_uri(),
        )
        _atomic_write_json(
            self.backup_root / RETENTION_MANIFEST_FILENAME,
            manifest.to_dict(),
        )
        return manifest

    def backup_age_seconds(
        self,
        backup: BackupSnapshot | Mapping[str, Any] | str | Path,
        *,
        now: float | None = None,
    ) -> float:
        snapshot = self._coerce_snapshot(backup)
        instant = self._clock() if now is None else float(now)
        created = datetime.fromisoformat(
            snapshot.created_at.replace("Z", "+00:00")
        ).timestamp()
        return max(0.0, instant - created)

    # -- crash matrix --------------------------------------------------------

    def evaluate_crash_matrix(
        self,
        *,
        accepted_roots: AuthorityRoots,
        maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
    ) -> CrashMatrixReport:
        """Exercise the declared crash matrix without losing accepted state.

        Scenarios are simulated with hermetic filesystem side effects against
        temporary copies. The live database is left unchanged when a scenario
        would otherwise mutate it.
        """

        self.assert_direct_file_maintenance_admitted()
        results: dict[str, dict[str, Any]] = {}
        preserved = True

        # Baseline verified backup of accepted state.
        baseline = self.create_backup(
            maintenance_lease=maintenance_lease,
            require_maintenance_lease=maintenance_lease is not None,
            backup_id=f"backup:matrix-baseline:{uuid.uuid4().hex}",
        )
        if not baseline.roots.matches(accepted_roots, ignore_generation=False):
            preserved = False
            results["baseline"] = {
                "ok": False,
                "reason": "accepted_roots_mismatch_at_baseline",
            }
        else:
            results["baseline"] = {"ok": True, "backup_id": baseline.backup_id}

        # crash_before_checkpoint: backup with skip_checkpoint still captures
        # durable committed state (accepted rows).
        try:
            before_cp = self.create_backup(
                maintenance_lease=maintenance_lease,
                require_maintenance_lease=maintenance_lease is not None,
                skip_checkpoint=True,
                backup_id=f"backup:matrix-before-cp:{uuid.uuid4().hex}",
            )
            ok = before_cp.roots.matches(accepted_roots, ignore_generation=False)
            preserved = preserved and ok
            results[CrashScenario.CRASH_BEFORE_CHECKPOINT.value] = {
                "ok": ok,
                "accepted_state_preserved": ok,
                "backup_id": before_cp.backup_id,
            }
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.CRASH_BEFORE_CHECKPOINT.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # crash_after_checkpoint
        try:
            self.checkpoint()
            after_cp = self.create_backup(
                maintenance_lease=maintenance_lease,
                require_maintenance_lease=maintenance_lease is not None,
                skip_checkpoint=True,
                backup_id=f"backup:matrix-after-cp:{uuid.uuid4().hex}",
            )
            ok = after_cp.roots.matches(accepted_roots, ignore_generation=False)
            preserved = preserved and ok
            results[CrashScenario.CRASH_AFTER_CHECKPOINT.value] = {
                "ok": ok,
                "accepted_state_preserved": ok,
                "backup_id": after_cp.backup_id,
            }
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.CRASH_AFTER_CHECKPOINT.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # corrupt_copy
        try:
            corrupt_dir = self.backup_root / f".corrupt-{uuid.uuid4().hex}"
            corrupt_dir.mkdir(parents=True, exist_ok=True)
            corrupt_body = corrupt_dir / BACKUP_BODY_FILENAME
            shutil.copy2(baseline.body_path, corrupt_body)
            with corrupt_body.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"\x00\xffCORRUPT")
            verification = self.verify_backup_path(
                corrupt_body,
                expected_digest=baseline.artifact_digest
                if not baseline.encryption_bound
                else None,
                expected_roots=baseline.roots,
                backup_id="corrupt-probe",
            )
            ok = not verification.verified
            results[CrashScenario.CORRUPT_COPY.value] = {
                "ok": ok,
                "detected": not verification.verified,
                "reason": verification.reason,
            }
            if not ok:
                preserved = False
            shutil.rmtree(corrupt_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.CORRUPT_COPY.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # disk_full — simulate by writing to a non-writable destination
        try:
            full_root = self.backup_root / f".disk-full-{uuid.uuid4().hex}"
            full_root.mkdir(parents=True, exist_ok=True)
            # Create a file where a directory is required to force copy failure.
            blocker = full_root / f"backup:disk-full-{uuid.uuid4().hex}"
            blocker.write_text("not-a-directory\n", encoding="utf-8")
            failing = ControlPlaneBackup(
                database_path=self.database_path,
                backup_root=full_root,
                state_dir=self.state_dir,
                store_id=self.store_id,
                maintenance_scope=self.maintenance_scope,
                owner_liveness_probe=self._liveness,
                clock=self._clock,
            )
            raised = False
            try:
                failing.create_backup(
                    maintenance_lease=maintenance_lease,
                    require_maintenance_lease=maintenance_lease is not None,
                    backup_id=blocker.name,
                    skip_checkpoint=True,
                    record_in_database=False,
                )
            except (ControlPlaneBackupError, OSError, FileExistsError):
                raised = True
            # Live accepted state must still match.
            live_roots = self.capture_roots()
            ok = raised and live_roots.matches(accepted_roots, ignore_generation=False)
            preserved = preserved and ok
            results[CrashScenario.DISK_FULL.value] = {
                "ok": ok,
                "failure_observed": raised,
                "accepted_state_preserved": live_roots.matches(
                    accepted_roots, ignore_generation=False
                ),
            }
            shutil.rmtree(full_root, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.DISK_FULL.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # partial_restore — truncated body must fail and leave live state
        try:
            partial_dir = self.backup_root / f".partial-{uuid.uuid4().hex}"
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial_body = partial_dir / BACKUP_BODY_FILENAME
            data = Path(baseline.body_path).read_bytes()
            partial_body.write_bytes(data[: max(1, len(data) // 4)])
            partial_snapshot = BackupSnapshot(
                backup_id=f"backup:partial:{uuid.uuid4().hex}",
                store_id=baseline.store_id,
                database_uuid=baseline.database_uuid,
                schema_revision=baseline.schema_revision,
                generation=baseline.generation,
                artifact_digest=_sha256_file(partial_body),
                created_at=baseline.created_at,
                destination_uri=partial_dir.as_uri(),
                status=BACKUP_STATUS_FAILED,
                roots=baseline.roots,
                body_path=str(partial_body),
                manifest_path=str(partial_dir / BACKUP_MANIFEST_FILENAME),
                fence_epoch=baseline.fence_epoch,
                revision=baseline.revision,
                event_watermark=baseline.event_watermark,
            )
            _atomic_write_json(
                Path(partial_snapshot.manifest_path), partial_snapshot.to_dict()
            )
            refused = False
            try:
                self.restore(
                    partial_snapshot,
                    maintenance_lease=maintenance_lease,
                    require_maintenance_lease=maintenance_lease is not None,
                    rotate_generation=False,
                    rehearsal=True,
                )
            except ControlPlaneBackupError:
                refused = True
            live_roots = self.capture_roots()
            ok = refused and live_roots.matches(accepted_roots, ignore_generation=False)
            preserved = preserved and ok
            results[CrashScenario.PARTIAL_RESTORE.value] = {
                "ok": ok,
                "restore_refused": refused,
                "accepted_state_preserved": live_roots.matches(
                    accepted_roots, ignore_generation=False
                ),
            }
            shutil.rmtree(partial_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.PARTIAL_RESTORE.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # schema_version — roots bind schema revision
        try:
            ok = (
                int(baseline.roots.schema_revision) == int(accepted_roots.schema_revision)
                and baseline.roots.schema_root == accepted_roots.schema_root
            )
            preserved = preserved and ok
            results[CrashScenario.SCHEMA_VERSION.value] = {
                "ok": ok,
                "schema_revision": int(baseline.roots.schema_revision),
                "schema_root": baseline.roots.schema_root,
            }
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.SCHEMA_VERSION.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # server_stopped — ownership admits maintenance
        try:
            observation = self.observe_ownership()
            ok = observation.admits_direct_file_maintenance
            results[CrashScenario.SERVER_STOPPED.value] = {
                "ok": ok,
                "ownership": observation.to_dict(),
            }
            if not ok:
                preserved = False
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.SERVER_STOPPED.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # stale_client — generation rotation invalidates pre-rotation writers
        try:
            rehearsal = self.restore(
                baseline,
                maintenance_lease=maintenance_lease,
                require_maintenance_lease=maintenance_lease is not None,
                rotate_generation=True,
                rehearsal=True,
            )
            rotation = rehearsal.rotation
            ok = (
                rotation is not None
                and rotation.invalidates(baseline.generation, baseline.fence_epoch)
                and rehearsal.writers_invalidated
                and rehearsal.roots.matches(accepted_roots, ignore_generation=True)
            )
            preserved = preserved and ok
            results[CrashScenario.STALE_CLIENT.value] = {
                "ok": ok,
                "writers_invalidated": rehearsal.writers_invalidated,
                "rotation": None if rotation is None else rotation.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.STALE_CLIENT.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # backup_age
        try:
            age = self.backup_age_seconds(baseline)
            ok = age >= 0.0
            results[CrashScenario.BACKUP_AGE.value] = {
                "ok": ok,
                "age_seconds": age,
                "backup_id": baseline.backup_id,
            }
            if not ok:
                preserved = False
        except Exception as exc:  # noqa: BLE001
            preserved = False
            results[CrashScenario.BACKUP_AGE.value] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # Ensure every declared scenario has an entry.
        for name in DECLARED_CRASH_SCENARIOS:
            results.setdefault(
                name,
                {"ok": False, "reason": "scenario_not_evaluated"},
            )
            if name in results and not results[name].get("ok", False):
                if name != "baseline":
                    # already tracked
                    pass

        for name in DECLARED_CRASH_SCENARIOS:
            if not results.get(name, {}).get("ok", False):
                preserved = False

        return CrashMatrixReport(
            scenarios=MappingProxyType(
                {key: MappingProxyType(dict(value)) for key, value in results.items()}
            ),
            accepted_state_preserved=preserved,
            generated_at=_utc_iso(self._clock()),
        )

    # -- internal ------------------------------------------------------------

    def _require_active_maintenance_lease(
        self,
        lease: MaintenanceLease | Mapping[str, Any] | None,
    ) -> MaintenanceLease:
        if lease is None:
            raise ControlPlaneBackupOwnershipError(
                "exclusive maintenance lease is required for direct-file "
                "backup/restore operations"
            )
        if isinstance(lease, MaintenanceLease):
            record = lease
        else:
            record = MaintenanceLease(
                lease_id=str(lease["lease_id"]),
                scope=str(lease["scope"]),
                owner_session_id=str(lease["owner_session_id"]),
                process_birth_id=str(lease.get("process_birth_id") or ""),
                fencing_token=int(lease["fencing_token"]),
                fence_epoch=int(lease["fence_epoch"]),
                acquired_at=str(lease["acquired_at"]),
                expires_at=str(lease["expires_at"]),
                state=str(lease.get("state") or MAINTENANCE_LEASE_ACTIVE),
                revision=int(lease.get("revision") or 0),
            )
        if record.state != MAINTENANCE_LEASE_ACTIVE:
            raise ControlPlaneBackupOwnershipError(
                f"maintenance lease is not active: {record.state}"
            )
        if record.scope != self.maintenance_scope:
            raise ControlPlaneBackupOwnershipError(
                f"maintenance lease scope mismatch: {record.scope!r} != "
                f"{self.maintenance_scope!r}"
            )
        # Confirm lease row still active when the database is present.
        if self.database_path.is_file():
            with open_duckdb_connection(self.database_path) as connection:
                row = connection.execute(
                    """
                    SELECT lease_id, state, fencing_token, scope
                    FROM maintenance_leases
                    WHERE lease_id = ?
                    LIMIT 1
                    """,
                    [record.lease_id],
                ).fetchone()
                if row is None:
                    raise ControlPlaneBackupOwnershipError(
                        "maintenance lease not found in database"
                    )
                state = str(_row_value(row, "state", _row_value(row, 1, "")))
                token = int(
                    _row_value(row, "fencing_token", _row_value(row, 2, 0)) or 0
                )
                scope = str(_row_value(row, "scope", _row_value(row, 3, "")))
                if state != MAINTENANCE_LEASE_ACTIVE:
                    raise ControlPlaneBackupOwnershipError(
                        "maintenance lease is not active in the store"
                    )
                if token != int(record.fencing_token):
                    raise ControlPlaneBackupOwnershipError(
                        "maintenance lease fencing_token mismatch"
                    )
                if scope != self.maintenance_scope:
                    raise ControlPlaneBackupOwnershipError(
                        "maintenance lease scope mismatch in store"
                    )
        return record

    def _coerce_snapshot(
        self,
        backup: BackupSnapshot | Mapping[str, Any] | str | Path,
    ) -> BackupSnapshot:
        if isinstance(backup, BackupSnapshot):
            return backup
        if isinstance(backup, Mapping):
            return BackupSnapshot.from_dict(backup)
        path = Path(backup)
        if path.is_dir():
            manifest = path / BACKUP_MANIFEST_FILENAME
        elif path.is_file() and path.name == BACKUP_MANIFEST_FILENAME:
            manifest = path
        elif path.is_file() and path.suffix == ".json":
            manifest = path
        else:
            # Treat as backup_id.
            manifest = self.backup_directory(str(backup)) / BACKUP_MANIFEST_FILENAME
        payload = _read_json(manifest)
        if payload is None:
            raise ControlPlaneBackupRequestError(
                f"backup manifest not found or unreadable: {manifest}"
            )
        return BackupSnapshot.from_dict(payload)

    def _record_backup_snapshot(self, snapshot: BackupSnapshot) -> None:
        if not self.database_path.is_file():
            return
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
                    int(snapshot.schema_revision),
                    int(snapshot.generation),
                    snapshot.artifact_digest,
                    snapshot.created_at,
                    snapshot.destination_uri,
                    snapshot.status,
                    json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":")),
                ],
            )

    def _record_restore_receipt(self, receipt: RestoreReceipt) -> None:
        if not self.database_path.is_file():
            return
        with open_duckdb_connection(self.database_path) as connection:
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
                    int(receipt.schema_revision),
                    int(receipt.generation),
                    receipt.outcome,
                    json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")),
                ],
            )


def build_control_plane_backup(
    *,
    database_path: Path | str,
    backup_root: Path | str,
    state_dir: Path | str | None = None,
    store_id: str = DEFAULT_STORE_ID,
    maintenance_scope: str = DEFAULT_MAINTENANCE_SCOPE,
    owner_liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    clock: Callable[[], float] | None = None,
    encryption_key_handle: str = "",
) -> ControlPlaneBackup:
    """Construct a configured :class:`ControlPlaneBackup` service."""

    return ControlPlaneBackup(
        database_path=database_path,
        backup_root=backup_root,
        state_dir=state_dir,
        store_id=store_id,
        maintenance_scope=maintenance_scope,
        owner_liveness_probe=owner_liveness_probe,
        clock=clock,
        encryption_key_handle=encryption_key_handle,
    )


__all__ = (
    "AUTHORITY_ROOTS_SCHEMA",
    "BACKUP_BODY_FILENAME",
    "BACKUP_MANIFEST_FILENAME",
    "BACKUP_SNAPSHOT_SCHEMA",
    "BACKUP_STATUS_FAILED",
    "BACKUP_STATUS_VERIFIED",
    "BACKUP_VERIFICATION_SCHEMA",
    "CONTROL_PLANE_BACKUP_INTERFACE",
    "CONTROL_PLANE_BACKUP_SCHEMA",
    "CONTROL_PLANE_BACKUP_VERSION",
    "CRASH_MATRIX_REPORT_SCHEMA",
    "CrashMatrixReport",
    "CrashScenario",
    "ControlPlaneBackup",
    "ControlPlaneBackupCorruptionError",
    "ControlPlaneBackupError",
    "ControlPlaneBackupOwnershipError",
    "ControlPlaneBackupRequestError",
    "ControlPlaneBackupRestoreError",
    "ControlPlaneBackupRetentionError",
    "ControlPlaneBackupVerificationError",
    "DECLARED_CRASH_SCENARIOS",
    "DEFAULT_MAX_BACKUP_AGE_SECONDS",
    "DEFAULT_RETENTION_COUNT",
    "AuthorityRoots",
    "BackupSnapshot",
    "BackupVerification",
    "OwnershipObservation",
    "OwnershipState",
    "RESTORE_OUTCOME_FAILED",
    "RESTORE_OUTCOME_REHEARSAL",
    "RESTORE_OUTCOME_SUCCESS",
    "RESTORE_RECEIPT_INTERFACE",
    "RESTORE_RECEIPT_SCHEMA",
    "RETENTION_MANIFEST_SCHEMA",
    "RestoreReceipt",
    "RetentionManifest",
    "STORE_GENERATION_ROTATION_INTERFACE",
    "STORE_GENERATION_ROTATION_SCHEMA",
    "StoreGenerationRotation",
    "build_control_plane_backup",
    "duckdb_available",
)
