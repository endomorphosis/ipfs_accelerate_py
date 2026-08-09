"""Checkpoint, backup, restore, retention, and store-generation rotation.

Interfaces: ``ControlPlaneBackup@1``, ``RestoreReceipt@1``,
``StoreGenerationRotation@1``

Creates independently verified consistent snapshots of ``control.duckdb``,
applies retention manifests, probes corruption, rehearses restore, and rotates
store generation so pre-rotation writers fail closed. Direct-file maintenance
is refused while state-owner ownership is live or unknown unless the caller
proves ownership with a matching fence token.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final
from urllib.parse import unquote, urlparse

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    current_process_birth,
    owner_liveness,
)
from ..task_sources.control_plane_contracts import (
    CONTRACT_VERSION,
    ControlPlaneGenerationError,
    StoreGeneration,
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
    MAINTENANCE_LEASE_ACTIVE,
    MAINTENANCE_LEASE_RELEASED,
    MaintenanceLease,
    StateRepositoryMaintenanceError,
    acquire_maintenance_lease,
    release_maintenance_lease,
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
# Contract identity
# ---------------------------------------------------------------------------

CONTROL_PLANE_BACKUP_INTERFACE: Final[str] = "ControlPlaneBackup@1"
RESTORE_RECEIPT_INTERFACE: Final[str] = "RestoreReceipt@1"
STORE_GENERATION_ROTATION_INTERFACE: Final[str] = "StoreGenerationRotation@1"
BACKUP_SNAPSHOT_INTERFACE: Final[str] = "BackupSnapshot@1"
RETENTION_MANIFEST_INTERFACE: Final[str] = "RetentionManifest@1"
STATE_ROOTS_INTERFACE: Final[str] = "ControlPlaneStateRoots@1"

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
RETENTION_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/retention-manifest@1"
)
STATE_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-state-roots@1"
)
BACKUP_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/backup-manifest@1"
)

CONTROL_PLANE_BACKUP_VERSION: Final[int] = 1
DEFAULT_STORE_ID: Final[str] = "control.duckdb"
DEFAULT_RETENTION_COUNT: Final[int] = 8
DEFAULT_RETENTION_MAX_AGE_SECONDS: Final[int] = 30 * 24 * 3600
MANIFEST_FILENAME: Final[str] = "backup.manifest.json"
BODY_FILENAME: Final[str] = "control.duckdb"

_DIGEST_RE_PREFIX: Final[str] = "sha256:"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ControlPlaneBackupError(RuntimeError):
    """Base fail-closed error for control-plane backup/restore."""


class ControlPlaneBackupOwnershipError(ControlPlaneBackupError):
    """Direct-file maintenance refused due to live/unknown ownership."""


class ControlPlaneBackupVerificationError(ControlPlaneBackupError):
    """Independent backup verification failed."""


class ControlPlaneBackupCorruptionError(ControlPlaneBackupError):
    """Backup body is corrupt or digests do not match."""


class ControlPlaneBackupIOError(ControlPlaneBackupError):
    """Filesystem or database I/O failed (including simulated disk full)."""


class ControlPlaneBackupLeaseError(ControlPlaneBackupError):
    """Maintenance lease precondition failed."""


class ControlPlaneBackupCrash(ControlPlaneBackupError):
    """Injected crash for the declared crash matrix (tests only)."""


class ControlPlaneBackupStateError(ControlPlaneBackupError):
    """Store state is missing, inconsistent, or refused rotation."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class BackupStatus(str, Enum):
    """Closed backup lifecycle statuses."""

    PENDING = "pending"
    VERIFIED = "verified"
    CORRUPT = "corrupt"
    RETAINED = "retained"
    PRUNED = "pruned"
    FAILED = "failed"


class RestoreOutcome(str, Enum):
    """Closed restore outcomes."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    REHEARSAL = "rehearsal"


class CrashPoint(str, Enum):
    """Declared crash-injection points for the recovery matrix."""

    BEFORE_CHECKPOINT = "before_checkpoint"
    AFTER_CHECKPOINT = "after_checkpoint"
    AFTER_COPY = "after_copy"
    AFTER_VERIFY = "after_verify"
    AFTER_MANIFEST = "after_manifest"
    BEFORE_RESTORE_REPLACE = "before_restore_replace"
    AFTER_RESTORE_REPLACE = "after_restore_replace"
    AFTER_ROOTS_CHECK = "after_roots_check"
    BEFORE_ROTATION = "before_rotation"
    AFTER_ROTATION = "after_rotation"


class OwnershipAdmission(str, Enum):
    """Result of direct-file maintenance ownership admission."""

    ALLOWED_NO_MARKER = "allowed_no_marker"
    ALLOWED_OWNER_DEAD = "allowed_owner_dead"
    ALLOWED_OWNER_FENCE = "allowed_owner_fence"
    REFUSED_OWNER_ALIVE = "refused_owner_alive"
    REFUSED_OWNER_UNKNOWN = "refused_owner_unknown"
    REFUSED_LOCK_HELD = "refused_lock_held"


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


def _sha256_bytes(payload: bytes) -> str:
    return f"{_DIGEST_RE_PREFIX}{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return f"{_DIGEST_RE_PREFIX}{digest.hexdigest()}"


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(tmp), flags, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    text = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    _atomic_write_bytes(path, text.encode("utf-8") + b"\n", mode=mode)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ControlPlaneBackupIOError(f"failed to read {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ControlPlaneBackupCorruptionError(
            f"malformed JSON at {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ControlPlaneBackupCorruptionError(f"JSON object required at {path}")
    return value


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row.keys()}
    if hasattr(row, "keys"):
        return {str(key): row[key] for key in row.keys()}
    # Fallback positional is not used; DuckDB rows expose keys.
    return {}


def owner_marker_path_for(database_path: Path | str) -> Path:
    db = Path(database_path)
    return db.with_name(f".{db.name}{OWNER_MARKER_SUFFIX}")


def owner_lock_path_for(database_path: Path | str) -> Path:
    db = Path(database_path)
    return db.with_name(f".{db.name}{OWNER_LOCK_SUFFIX}")


def assert_direct_file_maintenance_allowed(
    database_path: Path | str,
    *,
    owner_fence_token: str | None = None,
    liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    require_absent_lock: bool = False,
) -> dict[str, Any]:
    """Refuse direct-file maintenance while ownership is live or unknown.

    Owner-driven maintenance may proceed when ``owner_fence_token`` matches the
    live marker fence. External maintenance is allowed only when no marker
    exists or the recorded process birth is proved dead.
    """

    path = Path(database_path)
    marker_path = owner_marker_path_for(path)
    lock_path = owner_lock_path_for(path)
    probe = liveness_probe or (lambda birth: owner_liveness(birth))

    if require_absent_lock and lock_path.exists():
        # Best-effort: if the lock file exists we still try non-blocking flock.
        import fcntl

        handle = lock_path.open("a+b")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ControlPlaneBackupOwnershipError(
                    "direct-file maintenance refused: exclusive owner lock is held"
                ) from exc
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    payload = _read_json(marker_path)
    if payload is None:
        return {
            "admitted": True,
            "admission": OwnershipAdmission.ALLOWED_NO_MARKER.value,
            "marker_path": str(marker_path),
        }

    try:
        marker = OwnerMarker.from_dict(payload)
    except (TypeError, ValueError, KeyError) as exc:
        # Corrupt marker is treated as unknown ownership → fail closed.
        raise ControlPlaneBackupOwnershipError(
            "direct-file maintenance refused: owner marker is corrupt/unknown"
        ) from exc

    if owner_fence_token and owner_fence_token == marker.fence_token:
        return {
            "admitted": True,
            "admission": OwnershipAdmission.ALLOWED_OWNER_FENCE.value,
            "server_id": marker.server_id,
            "marker_path": str(marker_path),
        }

    state = probe(marker.process_birth)
    if state is OwnerLiveness.ALIVE:
        raise ControlPlaneBackupOwnershipError(
            "direct-file maintenance refused: state-owner ownership is live"
        )
    if state is OwnerLiveness.UNKNOWN:
        raise ControlPlaneBackupOwnershipError(
            "direct-file maintenance refused: state-owner ownership is unknown"
        )
    return {
        "admitted": True,
        "admission": OwnershipAdmission.ALLOWED_OWNER_DEAD.value,
        "server_id": marker.server_id,
        "marker_path": str(marker_path),
    }


def _derive_stream_key(secret: bytes, *, length: int) -> bytes:
    material = b""
    counter = 0
    while len(material) < length:
        material += hashlib.sha256(secret + counter.to_bytes(4, "big")).digest()
        counter += 1
    return material[:length]


def _encrypt_body(plaintext: bytes, encryption_key: bytes | None) -> tuple[bytes, str]:
    """Return (ciphertext_or_plain, algorithm). Digest-bound envelope only."""

    if encryption_key is None:
        return plaintext, "none"
    key = hashlib.sha256(b"control-plane-backup-v1:" + encryption_key).digest()
    stream = _derive_stream_key(key, length=len(plaintext))
    cipher = bytes(a ^ b for a, b in zip(plaintext, stream))
    return cipher, "xor-sha256-stream-v1"


def _decrypt_body(ciphertext: bytes, encryption_key: bytes | None, algorithm: str) -> bytes:
    if algorithm in ("", "none"):
        return ciphertext
    if algorithm != "xor-sha256-stream-v1":
        raise ControlPlaneBackupVerificationError(
            f"unsupported backup encryption algorithm: {algorithm!r}"
        )
    if encryption_key is None:
        raise ControlPlaneBackupVerificationError(
            "encryption key required to decrypt backup body"
        )
    key = hashlib.sha256(b"control-plane-backup-v1:" + encryption_key).digest()
    stream = _derive_stream_key(key, length=len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlPlaneStateRoots:
    """Canonical store/schema/event/task/lease roots for restore verification.

    Interface projection used by backup and restore receipts.
    """

    SCHEMA: ClassVar[str] = STATE_ROOTS_SCHEMA
    INTERFACE: ClassVar[str] = STATE_ROOTS_INTERFACE

    store_id: str
    database_uuid: str
    schema_revision: int
    schema_fingerprint: str
    schema_version: str
    generation: int
    fence_epoch: int
    revision: int
    birth_id: str
    event_watermark: int
    event_ids: tuple[str, ...]
    task_cids: tuple[str, ...]
    lease_keys: tuple[str, ...]
    task_count: int
    lease_count: int
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "schema_revision": int(self.schema_revision),
            "schema_fingerprint": self.schema_fingerprint,
            "schema_version": self.schema_version,
            "generation": int(self.generation),
            "fence_epoch": int(self.fence_epoch),
            "revision": int(self.revision),
            "birth_id": self.birth_id,
            "event_watermark": int(self.event_watermark),
            "event_ids": list(self.event_ids),
            "task_cids": list(self.task_cids),
            "lease_keys": list(self.lease_keys),
            "task_count": int(self.task_count),
            "lease_count": int(self.lease_count),
            "event_count": int(self.event_count),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_generation(self, *, store_id: str | None = None) -> StoreGeneration:
        return StoreGeneration(
            store_id=store_id or self.store_id,
            generation=int(self.generation),
            schema_revision=int(self.schema_revision),
            fence_epoch=int(self.fence_epoch),
            revision=int(self.revision),
            database_uuid=self.database_uuid,
            birth_id=self.birth_id,
        )

    def domain_roots_match(self, other: "ControlPlaneStateRoots") -> bool:
        """Compare store/schema/event/task/lease roots (ignore generation head)."""

        return (
            self.store_id == other.store_id
            and self.database_uuid == other.database_uuid
            and self.schema_revision == other.schema_revision
            and self.schema_fingerprint == other.schema_fingerprint
            and self.schema_version == other.schema_version
            and self.event_watermark == other.event_watermark
            and self.event_ids == other.event_ids
            and self.task_cids == other.task_cids
            and self.lease_keys == other.lease_keys
            and self.task_count == other.task_count
            and self.lease_count == other.lease_count
            and self.event_count == other.event_count
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlPlaneStateRoots":
        return cls(
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
            database_uuid=str(payload.get("database_uuid") or ""),
            schema_revision=int(payload.get("schema_revision") or 0),
            schema_fingerprint=str(payload.get("schema_fingerprint") or ""),
            schema_version=str(payload.get("schema_version") or ""),
            generation=int(payload.get("generation") or 1),
            fence_epoch=int(payload.get("fence_epoch") or 0),
            revision=int(payload.get("revision") or 0),
            birth_id=str(payload.get("birth_id") or ""),
            event_watermark=int(payload.get("event_watermark") or 0),
            event_ids=tuple(str(item) for item in payload.get("event_ids") or ()),
            task_cids=tuple(str(item) for item in payload.get("task_cids") or ()),
            lease_keys=tuple(str(item) for item in payload.get("lease_keys") or ()),
            task_count=int(payload.get("task_count") or 0),
            lease_count=int(payload.get("lease_count") or 0),
            event_count=int(payload.get("event_count") or 0),
        )


@dataclass(frozen=True)
class BackupSnapshot:
    """Verified backup snapshot identity and destination binding.

    Interface: ``BackupSnapshot@1`` (persisted as ``backup_snapshots`` rows).
    """

    SCHEMA: ClassVar[str] = BACKUP_SNAPSHOT_SCHEMA
    INTERFACE: ClassVar[str] = BACKUP_SNAPSHOT_INTERFACE

    backup_id: str
    store_id: str
    database_uuid: str
    schema_revision: int
    generation: int
    artifact_digest: str
    created_at: str
    destination_uri: str
    status: str
    roots: ControlPlaneStateRoots
    body_path: str = ""
    encryption_algorithm: str = "none"
    verified_at: str = ""
    age_seconds: int = 0

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
            "body_path": self.body_path,
            "encryption_algorithm": self.encryption_algorithm,
            "verified_at": self.verified_at,
            "age_seconds": int(self.age_seconds),
        }

    @property
    def content_id(self) -> str:
        material = {
            key: value
            for key, value in self.to_dict().items()
            if key not in {"age_seconds"}
        }
        return content_identity(material)

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    def body_json(self) -> str:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackupSnapshot":
        roots_payload = payload.get("roots") or {}
        if not isinstance(roots_payload, Mapping):
            raise ControlPlaneBackupError("backup roots must be an object")
        return cls(
            backup_id=str(payload.get("backup_id") or ""),
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
            database_uuid=str(payload.get("database_uuid") or ""),
            schema_revision=int(payload.get("schema_revision") or 0),
            generation=int(payload.get("generation") or 1),
            artifact_digest=str(payload.get("artifact_digest") or ""),
            created_at=str(payload.get("created_at") or ""),
            destination_uri=str(payload.get("destination_uri") or ""),
            status=str(payload.get("status") or BackupStatus.PENDING.value),
            roots=ControlPlaneStateRoots.from_dict(roots_payload),
            body_path=str(payload.get("body_path") or ""),
            encryption_algorithm=str(payload.get("encryption_algorithm") or "none"),
            verified_at=str(payload.get("verified_at") or ""),
            age_seconds=int(payload.get("age_seconds") or 0),
        )


@dataclass(frozen=True)
class RestoreReceipt:
    """Authoritative restore outcome bound to a backup and post-rotation generation.

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
    roots_matched: bool
    pre_rotation_generation: int
    post_rotation_generation: int
    post_rotation_fence_epoch: int
    writers_invalidated: bool
    destination_path: str
    artifact_digest: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "receipt_id": self.receipt_id,
            "backup_id": self.backup_id,
            "store_id": self.store_id,
            "restored_at": self.restored_at,
            "schema_revision": int(self.schema_revision),
            "generation": int(self.generation),
            "outcome": self.outcome,
            "roots_matched": bool(self.roots_matched),
            "pre_rotation_generation": int(self.pre_rotation_generation),
            "post_rotation_generation": int(self.post_rotation_generation),
            "post_rotation_fence_epoch": int(self.post_rotation_fence_epoch),
            "writers_invalidated": bool(self.writers_invalidated),
            "destination_path": self.destination_path,
            "artifact_digest": self.artifact_digest,
            "details": dict(self.details),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    def body_json(self) -> str:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RestoreReceipt":
        details = payload.get("details") or {}
        if not isinstance(details, Mapping):
            details = {}
        return cls(
            receipt_id=str(payload.get("receipt_id") or ""),
            backup_id=str(payload.get("backup_id") or ""),
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
            restored_at=str(payload.get("restored_at") or ""),
            schema_revision=int(payload.get("schema_revision") or 0),
            generation=int(payload.get("generation") or 1),
            outcome=str(payload.get("outcome") or RestoreOutcome.FAILED.value),
            roots_matched=bool(payload.get("roots_matched")),
            pre_rotation_generation=int(payload.get("pre_rotation_generation") or 0),
            post_rotation_generation=int(payload.get("post_rotation_generation") or 0),
            post_rotation_fence_epoch=int(payload.get("post_rotation_fence_epoch") or 0),
            writers_invalidated=bool(payload.get("writers_invalidated")),
            destination_path=str(payload.get("destination_path") or ""),
            artifact_digest=str(payload.get("artifact_digest") or ""),
            details=dict(details),
        )


@dataclass(frozen=True)
class StoreGenerationRotation:
    """Record of a store-generation rotation that fences pre-rotation writers.

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
    revision: int
    birth_id: str
    rotated_at: str
    reason: str = "restore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "rotation_id": self.rotation_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "previous_generation": int(self.previous_generation),
            "new_generation": int(self.new_generation),
            "previous_fence_epoch": int(self.previous_fence_epoch),
            "new_fence_epoch": int(self.new_fence_epoch),
            "schema_revision": int(self.schema_revision),
            "revision": int(self.revision),
            "birth_id": self.birth_id,
            "rotated_at": self.rotated_at,
            "reason": self.reason,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_generation(self) -> StoreGeneration:
        return StoreGeneration(
            store_id=self.store_id,
            generation=int(self.new_generation),
            schema_revision=int(self.schema_revision),
            fence_epoch=int(self.new_fence_epoch),
            revision=int(self.revision),
            database_uuid=self.database_uuid,
            birth_id=self.birth_id,
        )

    def invalidates(self, writer_generation: StoreGeneration) -> bool:
        """Return True when a pre-rotation writer must fail closed."""

        if writer_generation.store_id != self.store_id:
            return True
        if writer_generation.database_uuid != self.database_uuid:
            return True
        if writer_generation.generation < self.new_generation:
            return True
        if (
            writer_generation.generation == self.new_generation
            and writer_generation.fence_epoch < self.new_fence_epoch
        ):
            return True
        return False


@dataclass(frozen=True)
class RetentionManifest:
    """Retention policy application receipt over verified backups."""

    SCHEMA: ClassVar[str] = RETENTION_MANIFEST_SCHEMA
    INTERFACE: ClassVar[str] = RETENTION_MANIFEST_INTERFACE

    manifest_id: str
    store_id: str
    created_at: str
    max_count: int
    max_age_seconds: int
    retained_backup_ids: tuple[str, ...]
    pruned_backup_ids: tuple[str, ...]
    retained_count: int
    pruned_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "manifest_id": self.manifest_id,
            "store_id": self.store_id,
            "created_at": self.created_at,
            "max_count": int(self.max_count),
            "max_age_seconds": int(self.max_age_seconds),
            "retained_backup_ids": list(self.retained_backup_ids),
            "pruned_backup_ids": list(self.pruned_backup_ids),
            "retained_count": int(self.retained_count),
            "pruned_count": int(self.pruned_count),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


# ---------------------------------------------------------------------------
# State root capture / checkpoint
# ---------------------------------------------------------------------------


def capture_state_roots(
    database_path: Path | str,
    *,
    store_id: str = DEFAULT_STORE_ID,
) -> ControlPlaneStateRoots:
    """Capture store/schema/event/task/lease roots from an embedded database."""

    if not duckdb_available():
        raise ControlPlaneBackupError("DuckDB is required to capture state roots")
    path = Path(database_path)
    if not path.is_file():
        raise ControlPlaneBackupStateError(f"database not found: {path}")

    with open_duckdb_connection(path) as connection:
        return _capture_state_roots_on_connection(connection, store_id=store_id)


def _meta_get(connection: Any, key: str) -> str:
    result = connection.execute(
        "SELECT value FROM control_plane_metadata WHERE key = ?",
        [key],
    )
    row = result.fetchone()
    if row is None:
        return ""
    if isinstance(row, Mapping):
        return str(row.get("value") or "")
    return str(row[0] or "")


def _capture_state_roots_on_connection(
    connection: Any,
    *,
    store_id: str,
) -> ControlPlaneStateRoots:
    gen_row = connection.execute(
        """
        SELECT generation, schema_revision, fence_epoch, revision,
               database_uuid, birth_id
        FROM store_generations
        ORDER BY generation DESC
        LIMIT 1
        """
    ).fetchone()
    if gen_row is None:
        raise ControlPlaneBackupStateError("store generation row is missing")
    gen = _row_mapping(gen_row)
    if not gen and not isinstance(gen_row, Mapping):
        gen = {
            "generation": gen_row[0],
            "schema_revision": gen_row[1],
            "fence_epoch": gen_row[2],
            "revision": gen_row[3],
            "database_uuid": gen_row[4],
            "birth_id": gen_row[5],
        }

    database_uuid = str(gen.get("database_uuid") or _meta_get(connection, META_DATABASE_UUID))
    schema_fingerprint = _meta_get(connection, META_SCHEMA_FINGERPRINT)
    schema_version = _meta_get(connection, META_SCHEMA_VERSION)

    event_rows = connection.execute(
        """
        SELECT event_id FROM domain_events
        ORDER BY global_sequence ASC, event_id ASC
        """
    ).fetchall()
    event_ids: list[str] = []
    for row in event_rows:
        mapped = _row_mapping(row)
        if mapped:
            event_ids.append(str(mapped.get("event_id") or ""))
        else:
            event_ids.append(str(row[0]))
    event_ids = [item for item in event_ids if item]
    watermark_row = connection.execute(
        "SELECT COALESCE(MAX(global_sequence), 0) AS event_watermark FROM domain_events"
    ).fetchone()
    if watermark_row is None:
        event_watermark = 0
    elif isinstance(watermark_row, Mapping):
        event_watermark = int(watermark_row.get("event_watermark") or 0)
    else:
        event_watermark = int(watermark_row[0] or 0)

    task_rows = connection.execute(
        "SELECT task_cid FROM tasks ORDER BY task_cid ASC"
    ).fetchall()
    task_cids: list[str] = []
    for row in task_rows:
        mapped = _row_mapping(row)
        if mapped:
            task_cids.append(str(mapped.get("task_cid") or ""))
        else:
            task_cids.append(str(row[0]))
    task_cids = [item for item in task_cids if item]

    lease_rows = connection.execute(
        """
        SELECT task_cid, claim_cid, fencing_token, fence_epoch, state
        FROM leases
        ORDER BY task_cid ASC, claim_cid ASC
        """
    ).fetchall()
    lease_keys: list[str] = []
    for row in lease_rows:
        mapped = _row_mapping(row)
        if mapped:
            lease_keys.append(
                f"{mapped.get('task_cid')}:{mapped.get('claim_cid')}:"
                f"{mapped.get('fencing_token')}:{mapped.get('fence_epoch')}:"
                f"{mapped.get('state')}"
            )
        else:
            lease_keys.append(
                f"{row[0]}:{row[1]}:{row[2]}:{row[3]}:{row[4]}"
            )

    return ControlPlaneStateRoots(
        store_id=store_id,
        database_uuid=database_uuid,
        schema_revision=int(gen.get("schema_revision") or CONTROL_PLANE_SCHEMA_REVISION),
        schema_fingerprint=schema_fingerprint,
        schema_version=str(schema_version),
        generation=int(gen.get("generation") or 1),
        fence_epoch=int(gen.get("fence_epoch") or 0),
        revision=int(gen.get("revision") or 0),
        birth_id=str(gen.get("birth_id") or ""),
        event_watermark=int(event_watermark),
        event_ids=tuple(event_ids),
        task_cids=tuple(task_cids),
        lease_keys=tuple(lease_keys),
        task_count=len(task_cids),
        lease_count=len(lease_keys),
        event_count=len(event_ids),
    )


def checkpoint_database(database_path: Path | str) -> dict[str, Any]:
    """Force a clean DuckDB CHECKPOINT on ``database_path``."""

    if not duckdb_available():
        raise ControlPlaneBackupError("DuckDB is required to checkpoint")
    path = Path(database_path)
    with open_duckdb_connection(path) as connection:
        connection.execute("CHECKPOINT")
    return {
        "checkpointed": True,
        "database_path": str(path),
        "at": _utc_iso(),
    }


# ---------------------------------------------------------------------------
# ControlPlaneBackup service
# ---------------------------------------------------------------------------


class ControlPlaneBackup:
    """Production control-plane checkpoint/backup/restore/retention service.

    Interface: ``ControlPlaneBackup@1``.
    """

    INTERFACE: ClassVar[str] = CONTROL_PLANE_BACKUP_INTERFACE
    SCHEMA: ClassVar[str] = CONTROL_PLANE_BACKUP_SCHEMA
    VERSION: ClassVar[int] = CONTROL_PLANE_BACKUP_VERSION

    def __init__(
        self,
        *,
        store_id: str = DEFAULT_STORE_ID,
        backup_root: Path | str | None = None,
        retention_count: int = DEFAULT_RETENTION_COUNT,
        retention_max_age_seconds: int = DEFAULT_RETENTION_MAX_AGE_SECONDS,
        encryption_key: bytes | str | None = None,
        liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
        clock: Callable[[], str] | None = None,
        crash_point: CrashPoint | str | None = None,
        process_birth_factory: Callable[[], ProcessBirthIdentity] | None = None,
    ) -> None:
        self.store_id = str(store_id or DEFAULT_STORE_ID)
        self.backup_root = Path(backup_root) if backup_root is not None else None
        self.retention_count = max(1, int(retention_count))
        self.retention_max_age_seconds = max(0, int(retention_max_age_seconds))
        if isinstance(encryption_key, str):
            self._encryption_key: bytes | None = encryption_key.encode("utf-8")
        else:
            self._encryption_key = encryption_key
        self._liveness_probe = liveness_probe
        self._clock = clock or _utc_iso
        self._crash_point = (
            None
            if crash_point is None
            else CrashPoint(str(crash_point))
        )
        self._process_birth_factory = process_birth_factory or current_process_birth
        self._lock = threading.RLock()

    # -- crash injection ---------------------------------------------------

    def _maybe_crash(self, point: CrashPoint) -> None:
        if self._crash_point is not None and self._crash_point is point:
            raise ControlPlaneBackupCrash(f"injected crash at {point.value}")

    # -- ownership ---------------------------------------------------------

    def assert_maintenance_allowed(
        self,
        database_path: Path | str,
        *,
        owner_fence_token: str | None = None,
    ) -> dict[str, Any]:
        return assert_direct_file_maintenance_allowed(
            database_path,
            owner_fence_token=owner_fence_token,
            liveness_probe=self._liveness_probe,
        )

    # -- checkpoint / roots ------------------------------------------------

    def checkpoint(
        self,
        database_path: Path | str,
        *,
        owner_fence_token: str | None = None,
    ) -> dict[str, Any]:
        """Checkpoint under admitted direct-file or owner-fence authority."""

        with self._lock:
            self.assert_maintenance_allowed(
                database_path, owner_fence_token=owner_fence_token
            )
            self._maybe_crash(CrashPoint.BEFORE_CHECKPOINT)
            receipt = checkpoint_database(database_path)
            self._maybe_crash(CrashPoint.AFTER_CHECKPOINT)
            return receipt

    def capture_roots(self, database_path: Path | str) -> ControlPlaneStateRoots:
        return capture_state_roots(database_path, store_id=self.store_id)

    # -- backup ------------------------------------------------------------

    def create_backup(
        self,
        database_path: Path | str,
        *,
        destination_dir: Path | str | None = None,
        owner_fence_token: str | None = None,
        maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
        acquire_lease: bool = True,
        record_in_database: bool = True,
        label: str = "",
    ) -> BackupSnapshot:
        """Create a verified consistent snapshot of the control-plane database.

        Steps:
        1. Admit direct-file maintenance (or owner fence).
        2. Optionally acquire a maintenance lease.
        3. Checkpoint, capture roots, copy body, independently verify.
        4. Write digest-bound manifest; optionally record ``backup_snapshots``.
        """

        with self._lock:
            path = Path(database_path)
            if not path.is_file():
                raise ControlPlaneBackupStateError(f"database not found: {path}")

            admission = self.assert_maintenance_allowed(
                path, owner_fence_token=owner_fence_token
            )

            dest_root = Path(destination_dir) if destination_dir else self.backup_root
            if dest_root is None:
                raise ControlPlaneBackupError(
                    "destination_dir or backup_root is required for create_backup"
                )
            dest_root.mkdir(parents=True, exist_ok=True)

            lease: MaintenanceLease | None = None
            lease_owned = False
            if maintenance_lease is not None:
                if isinstance(maintenance_lease, MaintenanceLease):
                    lease = maintenance_lease
                else:
                    lease = MaintenanceLease(
                        lease_id=str(maintenance_lease["lease_id"]),
                        scope=str(maintenance_lease["scope"]),
                        owner_session_id=str(maintenance_lease["owner_session_id"]),
                        process_birth_id=str(maintenance_lease["process_birth_id"]),
                        fencing_token=int(maintenance_lease["fencing_token"]),
                        fence_epoch=int(maintenance_lease["fence_epoch"]),
                        acquired_at=str(maintenance_lease["acquired_at"]),
                        expires_at=str(maintenance_lease["expires_at"]),
                        state=str(
                            maintenance_lease.get("state") or MAINTENANCE_LEASE_ACTIVE
                        ),
                        revision=int(maintenance_lease.get("revision") or 0),
                    )
                if not lease.active:
                    raise ControlPlaneBackupLeaseError(
                        "provided maintenance lease is not active"
                    )
            elif acquire_lease and owner_fence_token is None:
                birth = self._process_birth_factory()
                try:
                    lease = acquire_maintenance_lease(
                        path,
                        owner_session_id=f"session:backup:{uuid.uuid4()}",
                        process_birth_id=(
                            f"birth:{birth.pid}:{birth.start_time_ticks}"
                        ),
                        scope=DEFAULT_MAINTENANCE_SCOPE,
                        clock=self._clock,
                    )
                    lease_owned = True
                except StateRepositoryMaintenanceError as exc:
                    raise ControlPlaneBackupLeaseError(str(exc)) from exc

            backup_id = f"backup:{uuid.uuid4()}"
            backup_dir = dest_root / backup_id.replace(":", "_")
            backup_dir.mkdir(parents=True, exist_ok=False)
            body_path = backup_dir / BODY_FILENAME
            manifest_path = backup_dir / MANIFEST_FILENAME

            try:
                self._maybe_crash(CrashPoint.BEFORE_CHECKPOINT)
                checkpoint_database(path)
                self._maybe_crash(CrashPoint.AFTER_CHECKPOINT)

                roots = capture_state_roots(path, store_id=self.store_id)

                # Copy under exclusive open so writers cannot interleave.
                with open_duckdb_connection(path) as connection:
                    connection.execute("CHECKPOINT")
                    # Force close before file copy of the on-disk image.
                try:
                    shutil.copy2(path, body_path)
                except OSError as exc:
                    raise ControlPlaneBackupIOError(
                        f"backup copy failed (disk full or I/O error): {exc}"
                    ) from exc
                self._maybe_crash(CrashPoint.AFTER_COPY)

                # Snapshots must not carry a live maintenance lease into restore.
                self._release_active_maintenance_leases(body_path)

                plaintext = body_path.read_bytes()
                cipher, algorithm = _encrypt_body(plaintext, self._encryption_key)
                if algorithm != "none":
                    # Replace body with encrypted envelope payload.
                    _atomic_write_bytes(body_path, cipher)
                    artifact_digest = _sha256_bytes(cipher)
                else:
                    artifact_digest = _sha256_file(body_path)

                # Independent verification: re-hash and open roots from a
                # temporary decrypted copy — never trust the write path alone.
                verify_digest = _sha256_file(body_path)
                if verify_digest != artifact_digest:
                    raise ControlPlaneBackupVerificationError(
                        "independent digest verification failed after copy"
                    )
                verified_roots = self._verify_body_roots(
                    body_path,
                    expected_digest=artifact_digest,
                    encryption_algorithm=algorithm,
                    expected_roots=roots,
                )
                if not verified_roots.domain_roots_match(roots):
                    raise ControlPlaneBackupVerificationError(
                        "backup roots diverge from source after independent open"
                    )
                self._maybe_crash(CrashPoint.AFTER_VERIFY)

                created_at = self._clock()
                snapshot = BackupSnapshot(
                    backup_id=backup_id,
                    store_id=self.store_id,
                    database_uuid=roots.database_uuid,
                    schema_revision=roots.schema_revision,
                    generation=roots.generation,
                    artifact_digest=artifact_digest,
                    created_at=created_at,
                    destination_uri=backup_dir.resolve().as_uri(),
                    status=BackupStatus.VERIFIED.value,
                    roots=roots,
                    body_path=str(body_path),
                    encryption_algorithm=algorithm,
                    verified_at=created_at,
                )
                manifest = {
                    "schema": BACKUP_MANIFEST_SCHEMA,
                    "contract_version": CONTRACT_VERSION,
                    "backup": snapshot.to_record(),
                    "admission": admission,
                    "label": str(label or ""),
                    "lease_id": lease.lease_id if lease is not None else "",
                    "body_filename": BODY_FILENAME,
                    "encryption_algorithm": algorithm,
                }
                _atomic_write_json(manifest_path, manifest)
                self._maybe_crash(CrashPoint.AFTER_MANIFEST)

                if record_in_database:
                    self._record_backup_snapshot(path, snapshot)

                return snapshot
            except ControlPlaneBackupCrash:
                # Leave partial artifacts for crash-matrix inspection; do not
                # claim verified status.
                raise
            except Exception:
                # Best-effort cleanup of incomplete backup directory.
                if backup_dir.exists() and not (backup_dir / MANIFEST_FILENAME).exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                raise
            finally:
                if lease_owned and lease is not None:
                    try:
                        release_maintenance_lease(path, lease, clock=self._clock)
                    except StateRepositoryMaintenanceError:
                        pass

    def _release_active_maintenance_leases(self, database_path: Path) -> None:
        """Mark any active maintenance leases released inside a snapshot body."""

        with open_duckdb_connection(database_path) as connection:
            connection.execute(
                """
                UPDATE maintenance_leases
                SET state = ?, released_at = ?, revision = revision + 1
                WHERE state = ?
                """,
                [
                    MAINTENANCE_LEASE_RELEASED,
                    self._clock(),
                    MAINTENANCE_LEASE_ACTIVE,
                ],
            )
            connection.execute("CHECKPOINT")

    def _verify_body_roots(
        self,
        body_path: Path,
        *,
        expected_digest: str,
        encryption_algorithm: str,
        expected_roots: ControlPlaneStateRoots | None = None,
    ) -> ControlPlaneStateRoots:
        actual_digest = _sha256_file(body_path)
        if actual_digest != expected_digest:
            raise ControlPlaneBackupCorruptionError(
                "backup body digest mismatch during independent verification"
            )
        raw = body_path.read_bytes()
        plaintext = _decrypt_body(raw, self._encryption_key, encryption_algorithm)
        # Materialize a temporary database image for DuckDB open.
        with tempfile.TemporaryDirectory(prefix="cp-backup-verify-") as tmp:
            tmp_db = Path(tmp) / "verify.duckdb"
            tmp_db.write_bytes(plaintext)
            try:
                roots = capture_state_roots(tmp_db, store_id=self.store_id)
            except Exception as exc:
                raise ControlPlaneBackupCorruptionError(
                    f"backup body cannot be opened as control-plane database: {exc}"
                ) from exc
        if expected_roots is not None and not roots.domain_roots_match(expected_roots):
            raise ControlPlaneBackupVerificationError(
                "verified backup roots do not match captured source roots"
            )
        return roots

    def _record_backup_snapshot(
        self, database_path: Path, snapshot: BackupSnapshot
    ) -> None:
        with open_duckdb_connection(database_path) as connection:
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
                    snapshot.body_json(),
                ],
            )

    def verify_backup(
        self,
        backup: BackupSnapshot | Path | str | Mapping[str, Any],
    ) -> BackupSnapshot:
        """Independently re-verify a backup body and manifest."""

        snapshot, body_path = self._resolve_backup(backup)
        roots = self._verify_body_roots(
            body_path,
            expected_digest=snapshot.artifact_digest,
            encryption_algorithm=snapshot.encryption_algorithm,
            expected_roots=snapshot.roots,
        )
        verified_at = self._clock()
        return BackupSnapshot(
            backup_id=snapshot.backup_id,
            store_id=snapshot.store_id,
            database_uuid=snapshot.database_uuid,
            schema_revision=snapshot.schema_revision,
            generation=snapshot.generation,
            artifact_digest=snapshot.artifact_digest,
            created_at=snapshot.created_at,
            destination_uri=snapshot.destination_uri,
            status=BackupStatus.VERIFIED.value,
            roots=roots,
            body_path=str(body_path),
            encryption_algorithm=snapshot.encryption_algorithm,
            verified_at=verified_at,
            age_seconds=self.backup_age_seconds(snapshot, now=verified_at),
        )

    def probe_corruption(
        self,
        backup: BackupSnapshot | Path | str | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Probe a backup for corruption; never raises on expected corruption."""

        try:
            verified = self.verify_backup(backup)
            return {
                "corrupt": False,
                "backup_id": verified.backup_id,
                "artifact_digest": verified.artifact_digest,
                "status": verified.status,
            }
        except (
            ControlPlaneBackupCorruptionError,
            ControlPlaneBackupVerificationError,
            ControlPlaneBackupError,
        ) as exc:
            snapshot_id = ""
            try:
                snapshot, _body = self._resolve_backup(backup)
                snapshot_id = snapshot.backup_id
            except Exception:
                snapshot_id = ""
            return {
                "corrupt": True,
                "backup_id": snapshot_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    def backup_age_seconds(
        self,
        backup: BackupSnapshot | Mapping[str, Any],
        *,
        now: str | None = None,
    ) -> int:
        snapshot = (
            backup
            if isinstance(backup, BackupSnapshot)
            else BackupSnapshot.from_dict(backup)
        )
        created = _parse_iso(snapshot.created_at)
        current = _parse_iso(now or self._clock())
        if created is None or current is None:
            return 0
        return max(0, int((current - created).total_seconds()))

    def _resolve_backup(
        self,
        backup: BackupSnapshot | Path | str | Mapping[str, Any],
    ) -> tuple[BackupSnapshot, Path]:
        if isinstance(backup, BackupSnapshot):
            body = Path(backup.body_path) if backup.body_path else None
            if body is None or not body.is_file():
                # Try destination_uri directory.
                dest = backup.destination_uri
                if dest.startswith("file:"):
                    parsed = urlparse(dest)
                    body = Path(unquote(parsed.path)) / BODY_FILENAME
                else:
                    raise ControlPlaneBackupError(
                        "backup body_path is missing and destination_uri is not a file URI"
                    )
            if not body.is_file():
                raise ControlPlaneBackupError(f"backup body not found: {body}")
            return backup, body

        if isinstance(backup, Mapping):
            snapshot = BackupSnapshot.from_dict(backup)
            return self._resolve_backup(snapshot)

        path = Path(backup)
        if path.is_dir():
            manifest_path = path / MANIFEST_FILENAME
            body_path = path / BODY_FILENAME
        elif path.name == MANIFEST_FILENAME:
            manifest_path = path
            body_path = path.parent / BODY_FILENAME
        elif path.name == BODY_FILENAME:
            body_path = path
            manifest_path = path.parent / MANIFEST_FILENAME
        else:
            # Treat as backup directory path string.
            manifest_path = path / MANIFEST_FILENAME
            body_path = path / BODY_FILENAME

        manifest = _read_json(manifest_path)
        if manifest is None:
            raise ControlPlaneBackupError(f"backup manifest not found: {manifest_path}")
        backup_payload = manifest.get("backup")
        if not isinstance(backup_payload, Mapping):
            raise ControlPlaneBackupCorruptionError("manifest missing backup object")
        snapshot = BackupSnapshot.from_dict(backup_payload)
        if not body_path.is_file():
            raise ControlPlaneBackupError(f"backup body not found: {body_path}")
        return snapshot, body_path

    # -- restore / rotation ------------------------------------------------

    def restore(
        self,
        backup: BackupSnapshot | Path | str | Mapping[str, Any],
        destination_path: Path | str,
        *,
        owner_fence_token: str | None = None,
        maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
        acquire_lease: bool = True,
        rotate_generation: bool = True,
        rehearsal: bool = False,
        record_receipt: bool = True,
    ) -> RestoreReceipt:
        """Restore a verified backup and optionally rotate store generation.

        Restored domain roots (store/schema/event/task/lease) must match the
        backup. Pre-rotation writers are invalidated when generation rotates.
        """

        with self._lock:
            dest = Path(destination_path)
            snapshot = self.verify_backup(backup)
            body_path = Path(snapshot.body_path)

            admission = self.assert_maintenance_allowed(
                dest if dest.exists() else body_path,
                owner_fence_token=owner_fence_token,
            )
            # Always admit against destination if it exists.
            if dest.exists():
                self.assert_maintenance_allowed(
                    dest, owner_fence_token=owner_fence_token
                )

            lease: MaintenanceLease | None = None
            lease_owned = False

            # Materialize plaintext restore image.
            raw = body_path.read_bytes()
            plaintext = _decrypt_body(
                raw, self._encryption_key, snapshot.encryption_algorithm
            )

            if rehearsal:
                with tempfile.TemporaryDirectory(prefix="cp-restore-rehearsal-") as tmp:
                    rehearsal_db = Path(tmp) / "rehearsal.duckdb"
                    rehearsal_db.write_bytes(plaintext)
                    roots = capture_state_roots(rehearsal_db, store_id=self.store_id)
                    matched = roots.domain_roots_match(snapshot.roots)
                    if not matched:
                        raise ControlPlaneBackupVerificationError(
                            "restore rehearsal roots do not match backup"
                        )
                    receipt = RestoreReceipt(
                        receipt_id=f"restore:{uuid.uuid4()}",
                        backup_id=snapshot.backup_id,
                        store_id=self.store_id,
                        restored_at=self._clock(),
                        schema_revision=roots.schema_revision,
                        generation=roots.generation,
                        outcome=RestoreOutcome.REHEARSAL.value,
                        roots_matched=True,
                        pre_rotation_generation=roots.generation,
                        post_rotation_generation=roots.generation,
                        post_rotation_fence_epoch=roots.fence_epoch,
                        writers_invalidated=False,
                        destination_path=str(rehearsal_db),
                        artifact_digest=snapshot.artifact_digest,
                        details={"admission": admission, "rehearsal": True},
                    )
                    return receipt

            self._maybe_crash(CrashPoint.BEFORE_RESTORE_REPLACE)

            # Preserve prior accepted state until replace succeeds.
            dest.parent.mkdir(parents=True, exist_ok=True)
            backup_of_dest: Path | None = None
            if dest.exists():
                backup_of_dest = dest.with_suffix(
                    dest.suffix + f".pre-restore.{uuid.uuid4().hex}"
                )
                shutil.copy2(dest, backup_of_dest)

            tmp_dest = dest.with_name(
                f".{dest.name}.restore-tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            try:
                _atomic_write_bytes(tmp_dest, plaintext)
                os.replace(str(tmp_dest), str(dest))
            except OSError as exc:
                try:
                    tmp_dest.unlink()
                except FileNotFoundError:
                    pass
                if backup_of_dest is not None and backup_of_dest.exists():
                    # Leave original in place on failure.
                    pass
                raise ControlPlaneBackupIOError(
                    f"restore replace failed (disk full or I/O error): {exc}"
                ) from exc

            self._maybe_crash(CrashPoint.AFTER_RESTORE_REPLACE)

            # Lease on restored DB for exclusive post-restore work.
            if maintenance_lease is not None:
                if isinstance(maintenance_lease, MaintenanceLease):
                    lease = maintenance_lease
                else:
                    lease = MaintenanceLease(
                        lease_id=str(maintenance_lease["lease_id"]),
                        scope=str(maintenance_lease["scope"]),
                        owner_session_id=str(maintenance_lease["owner_session_id"]),
                        process_birth_id=str(maintenance_lease["process_birth_id"]),
                        fencing_token=int(maintenance_lease["fencing_token"]),
                        fence_epoch=int(maintenance_lease["fence_epoch"]),
                        acquired_at=str(maintenance_lease["acquired_at"]),
                        expires_at=str(maintenance_lease["expires_at"]),
                        state=str(
                            maintenance_lease.get("state") or MAINTENANCE_LEASE_ACTIVE
                        ),
                        revision=int(maintenance_lease.get("revision") or 0),
                    )
            elif acquire_lease and owner_fence_token is None:
                birth = self._process_birth_factory()
                try:
                    lease = acquire_maintenance_lease(
                        dest,
                        owner_session_id=f"session:restore:{uuid.uuid4()}",
                        process_birth_id=(
                            f"birth:{birth.pid}:{birth.start_time_ticks}"
                        ),
                        scope=DEFAULT_MAINTENANCE_SCOPE,
                        clock=self._clock,
                    )
                    lease_owned = True
                except StateRepositoryMaintenanceError as exc:
                    raise ControlPlaneBackupLeaseError(str(exc)) from exc

            try:
                roots = capture_state_roots(dest, store_id=self.store_id)
                matched = roots.domain_roots_match(snapshot.roots)
                if not matched:
                    # Partial restore: roll back if we have a pre-image.
                    if backup_of_dest is not None and backup_of_dest.exists():
                        os.replace(str(backup_of_dest), str(dest))
                    raise ControlPlaneBackupVerificationError(
                        "restored roots do not match backup store/schema/"
                        "event/task/lease roots"
                    )
                self._maybe_crash(CrashPoint.AFTER_ROOTS_CHECK)

                pre_generation = roots.generation
                pre_fence = roots.fence_epoch
                post_generation = pre_generation
                post_fence = pre_fence
                writers_invalidated = False
                rotation: StoreGenerationRotation | None = None

                if rotate_generation:
                    self._maybe_crash(CrashPoint.BEFORE_ROTATION)
                    rotation = self.rotate_store_generation(
                        dest,
                        reason="restore",
                        owner_fence_token=owner_fence_token,
                        acquire_lease=False,
                    )
                    post_generation = rotation.new_generation
                    post_fence = rotation.new_fence_epoch
                    writers_invalidated = True
                    self._maybe_crash(CrashPoint.AFTER_ROTATION)

                # Drop pre-restore safety copy only after success.
                if backup_of_dest is not None and backup_of_dest.exists():
                    try:
                        backup_of_dest.unlink()
                    except OSError:
                        pass

                receipt = RestoreReceipt(
                    receipt_id=f"restore:{uuid.uuid4()}",
                    backup_id=snapshot.backup_id,
                    store_id=self.store_id,
                    restored_at=self._clock(),
                    schema_revision=roots.schema_revision,
                    generation=post_generation,
                    outcome=RestoreOutcome.SUCCESS.value,
                    roots_matched=True,
                    pre_rotation_generation=pre_generation,
                    post_rotation_generation=post_generation,
                    post_rotation_fence_epoch=post_fence,
                    writers_invalidated=writers_invalidated,
                    destination_path=str(dest),
                    artifact_digest=snapshot.artifact_digest,
                    details={
                        "admission": admission,
                        "rotation": rotation.to_dict() if rotation else {},
                        "pre_fence_epoch": pre_fence,
                    },
                )
                if record_receipt:
                    self._record_restore_receipt(dest, receipt)
                return receipt
            finally:
                if lease_owned and lease is not None:
                    try:
                        release_maintenance_lease(dest, lease, clock=self._clock)
                    except StateRepositoryMaintenanceError:
                        pass

    def rotate_store_generation(
        self,
        database_path: Path | str,
        *,
        reason: str = "restore",
        owner_fence_token: str | None = None,
        acquire_lease: bool = True,
        birth_id: str | None = None,
    ) -> StoreGenerationRotation:
        """Advance store generation and fence epoch; invalidate prior writers."""

        with self._lock:
            path = Path(database_path)
            self.assert_maintenance_allowed(
                path, owner_fence_token=owner_fence_token
            )
            lease: MaintenanceLease | None = None
            lease_owned = False
            if acquire_lease and owner_fence_token is None:
                birth = self._process_birth_factory()
                try:
                    lease = acquire_maintenance_lease(
                        path,
                        owner_session_id=f"session:rotation:{uuid.uuid4()}",
                        process_birth_id=(
                            f"birth:{birth.pid}:{birth.start_time_ticks}"
                        ),
                        scope=DEFAULT_MAINTENANCE_SCOPE,
                        clock=self._clock,
                    )
                    lease_owned = True
                except StateRepositoryMaintenanceError as exc:
                    raise ControlPlaneBackupLeaseError(str(exc)) from exc
            try:
                return self._rotate_generation_unlocked(
                    path,
                    reason=reason,
                    birth_id=birth_id,
                )
            finally:
                if lease_owned and lease is not None:
                    try:
                        release_maintenance_lease(path, lease, clock=self._clock)
                    except StateRepositoryMaintenanceError:
                        pass

    def _rotate_generation_unlocked(
        self,
        path: Path,
        *,
        reason: str,
        birth_id: str | None,
    ) -> StoreGenerationRotation:
        if not duckdb_available():
            raise ControlPlaneBackupError("DuckDB is required for generation rotation")
        rotated_at = self._clock()
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
                raise ControlPlaneBackupStateError("store generation row is missing")
            mapped = _row_mapping(row)
            if not mapped and not isinstance(row, Mapping):
                mapped = {
                    "generation": row[0],
                    "schema_revision": row[1],
                    "fence_epoch": row[2],
                    "revision": row[3],
                    "database_uuid": row[4],
                    "birth_id": row[5],
                }
            previous_generation = int(mapped["generation"])
            previous_fence = int(mapped["fence_epoch"])
            schema_revision = int(mapped["schema_revision"])
            revision = int(mapped["revision"])
            database_uuid = str(mapped["database_uuid"])
            prev_birth = str(mapped.get("birth_id") or "")
            new_generation = previous_generation + 1
            new_fence = previous_fence + 1
            new_birth = birth_id or f"birth:rotation:{new_generation}:{uuid.uuid4().hex}"
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
                    rotated_at,
                ],
            )
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
            revision=revision,
            birth_id=new_birth,
            rotated_at=rotated_at,
            reason=reason,
        )
        # Prove pre-rotation generation is incompatible as a continuing writer.
        previous = StoreGeneration(
            store_id=self.store_id,
            generation=previous_generation,
            schema_revision=schema_revision,
            fence_epoch=previous_fence,
            revision=revision,
            database_uuid=database_uuid,
            birth_id=prev_birth,
        )
        if not rotation.invalidates(previous):
            raise ControlPlaneBackupStateError(
                "generation rotation failed to invalidate pre-rotation writers"
            )
        return rotation

    def _record_restore_receipt(
        self, database_path: Path, receipt: RestoreReceipt
    ) -> None:
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
                    int(receipt.schema_revision),
                    int(receipt.generation),
                    receipt.outcome,
                    receipt.body_json(),
                ],
            )

    def assert_writer_invalidated(
        self,
        database_path: Path | str,
        writer_generation: StoreGeneration,
    ) -> None:
        """Fail closed when a pre-rotation writer generation is still accepted."""

        live = capture_state_roots(database_path, store_id=self.store_id).to_generation()
        if live.compatible_with(writer_generation) and (
            writer_generation.generation == live.generation
            and writer_generation.fence_epoch == live.fence_epoch
        ):
            # Exact match is current writer — not pre-rotation.
            return
        # Pre-rotation writers must not be compatible as continuing heads when
        # their generation lags the live head.
        if writer_generation.generation < live.generation:
            try:
                writer_generation.assert_compatible_with(live)
            except ControlPlaneGenerationError:
                return
            # compatible_with is directional: live.compatible_with(writer) is the
            # "may continue" check used by clients holding writer_generation.
            if not writer_generation.compatible_with(live):
                return
            # Writer claims it can continue — that is a failure after rotation.
            raise ControlPlaneGenerationError(
                "pre-rotation writer was not invalidated after generation rotation"
            )
        if writer_generation.generation == live.generation:
            if writer_generation.fence_epoch < live.fence_epoch:
                return
            if writer_generation.revision < live.revision:
                return
        # Otherwise ensure client-style check fails.
        if not writer_generation.compatible_with(live):
            return
        if (
            writer_generation.generation == live.generation
            and writer_generation.fence_epoch == live.fence_epoch
            and writer_generation.revision == live.revision
        ):
            return
        raise ControlPlaneGenerationError(
            "pre-rotation writer still appears compatible with live generation"
        )

    def writer_is_invalidated(
        self,
        live: StoreGeneration,
        writer_generation: StoreGeneration,
    ) -> bool:
        """Return True when ``writer_generation`` cannot continue after rotation."""

        if writer_generation.store_id != live.store_id:
            return True
        if writer_generation.database_uuid != live.database_uuid:
            return True
        # Clients holding an older generation cannot continue.
        if writer_generation.generation < live.generation:
            return True
        if (
            writer_generation.generation == live.generation
            and writer_generation.fence_epoch < live.fence_epoch
        ):
            return True
        return False

    # -- retention ---------------------------------------------------------

    def apply_retention(
        self,
        backup_root: Path | str | None = None,
        *,
        max_count: int | None = None,
        max_age_seconds: int | None = None,
        now: str | None = None,
    ) -> RetentionManifest:
        """Prune old verified backups; always retain the newest verified one."""

        with self._lock:
            root = Path(backup_root) if backup_root else self.backup_root
            if root is None:
                raise ControlPlaneBackupError("backup_root is required for retention")
            root.mkdir(parents=True, exist_ok=True)
            count_limit = max(1, int(max_count if max_count is not None else self.retention_count))
            age_limit = int(
                max_age_seconds
                if max_age_seconds is not None
                else self.retention_max_age_seconds
            )
            current = now or self._clock()

            snapshots: list[BackupSnapshot] = []
            for child in sorted(root.iterdir() if root.exists() else []):
                if not child.is_dir():
                    continue
                manifest_path = child / MANIFEST_FILENAME
                if not manifest_path.is_file():
                    continue
                try:
                    snapshot, _body = self._resolve_backup(child)
                except ControlPlaneBackupError:
                    continue
                age = self.backup_age_seconds(snapshot, now=current)
                snapshots.append(
                    BackupSnapshot(
                        backup_id=snapshot.backup_id,
                        store_id=snapshot.store_id,
                        database_uuid=snapshot.database_uuid,
                        schema_revision=snapshot.schema_revision,
                        generation=snapshot.generation,
                        artifact_digest=snapshot.artifact_digest,
                        created_at=snapshot.created_at,
                        destination_uri=snapshot.destination_uri,
                        status=snapshot.status,
                        roots=snapshot.roots,
                        body_path=snapshot.body_path,
                        encryption_algorithm=snapshot.encryption_algorithm,
                        verified_at=snapshot.verified_at,
                        age_seconds=age,
                    )
                )

            # Newest first.
            snapshots.sort(key=lambda item: item.created_at, reverse=True)
            retained: list[str] = []
            pruned: list[str] = []
            for index, snapshot in enumerate(snapshots):
                keep = True
                if index >= count_limit:
                    keep = False
                if age_limit > 0 and snapshot.age_seconds > age_limit and index > 0:
                    # Never prune the newest solely for age.
                    keep = False
                if index == 0:
                    keep = True
                if keep:
                    retained.append(snapshot.backup_id)
                else:
                    pruned.append(snapshot.backup_id)
                    self._prune_backup_dir(snapshot)

            return RetentionManifest(
                manifest_id=f"retention:{uuid.uuid4()}",
                store_id=self.store_id,
                created_at=current,
                max_count=count_limit,
                max_age_seconds=age_limit,
                retained_backup_ids=tuple(retained),
                pruned_backup_ids=tuple(pruned),
                retained_count=len(retained),
                pruned_count=len(pruned),
            )

    def _prune_backup_dir(self, snapshot: BackupSnapshot) -> None:
        body = Path(snapshot.body_path) if snapshot.body_path else None
        if body is not None and body.exists():
            directory = body.parent
            shutil.rmtree(directory, ignore_errors=True)
            return
        dest = snapshot.destination_uri
        if dest.startswith("file:"):
            parsed = urlparse(dest)
            directory = Path(unquote(parsed.path))
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)

    def list_backups(
        self, backup_root: Path | str | None = None
    ) -> tuple[BackupSnapshot, ...]:
        root = Path(backup_root) if backup_root else self.backup_root
        if root is None or not root.exists():
            return ()
        items: list[BackupSnapshot] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if not (child / MANIFEST_FILENAME).is_file():
                continue
            try:
                snapshot, _body = self._resolve_backup(child)
            except ControlPlaneBackupError:
                continue
            items.append(snapshot)
        items.sort(key=lambda item: item.created_at)
        return tuple(items)


def _parse_iso(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def open_control_plane_backup(
    *,
    store_id: str = DEFAULT_STORE_ID,
    backup_root: Path | str | None = None,
    retention_count: int = DEFAULT_RETENTION_COUNT,
    retention_max_age_seconds: int = DEFAULT_RETENTION_MAX_AGE_SECONDS,
    encryption_key: bytes | str | None = None,
    liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    clock: Callable[[], str] | None = None,
    crash_point: CrashPoint | str | None = None,
) -> ControlPlaneBackup:
    """Construct a :class:`ControlPlaneBackup` service (no I/O)."""

    return ControlPlaneBackup(
        store_id=store_id,
        backup_root=backup_root,
        retention_count=retention_count,
        retention_max_age_seconds=retention_max_age_seconds,
        encryption_key=encryption_key,
        liveness_probe=liveness_probe,
        clock=clock,
        crash_point=crash_point,
    )


__all__ = [
    "BACKUP_MANIFEST_SCHEMA",
    "BACKUP_SNAPSHOT_INTERFACE",
    "BACKUP_SNAPSHOT_SCHEMA",
    "BODY_FILENAME",
    "CONTROL_PLANE_BACKUP_INTERFACE",
    "CONTROL_PLANE_BACKUP_SCHEMA",
    "CONTROL_PLANE_BACKUP_VERSION",
    "CrashPoint",
    "BackupSnapshot",
    "BackupStatus",
    "ControlPlaneBackup",
    "ControlPlaneBackupCorruptionError",
    "ControlPlaneBackupCrash",
    "ControlPlaneBackupError",
    "ControlPlaneBackupIOError",
    "ControlPlaneBackupLeaseError",
    "ControlPlaneBackupOwnershipError",
    "ControlPlaneBackupStateError",
    "ControlPlaneBackupVerificationError",
    "ControlPlaneStateRoots",
    "DEFAULT_RETENTION_COUNT",
    "DEFAULT_RETENTION_MAX_AGE_SECONDS",
    "DEFAULT_STORE_ID",
    "MANIFEST_FILENAME",
    "OwnershipAdmission",
    "RESTORE_RECEIPT_INTERFACE",
    "RESTORE_RECEIPT_SCHEMA",
    "RETENTION_MANIFEST_INTERFACE",
    "RETENTION_MANIFEST_SCHEMA",
    "RestoreOutcome",
    "RestoreReceipt",
    "STATE_ROOTS_INTERFACE",
    "STATE_ROOTS_SCHEMA",
    "STORE_GENERATION_ROTATION_INTERFACE",
    "STORE_GENERATION_ROTATION_SCHEMA",
    "StoreGenerationRotation",
    "RetentionManifest",
    "assert_direct_file_maintenance_allowed",
    "capture_state_roots",
    "checkpoint_database",
    "duckdb_available",
    "open_control_plane_backup",
    "owner_lock_path_for",
    "owner_marker_path_for",
]
