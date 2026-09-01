"""Attempt-local Portal execution for database-authoritative task claims.

``DatabaseImplementationDaemon`` owns the durable claim and completion state.
``PortalImplementationDaemon`` owns the already-landed implementation pipeline
(provider routing, isolated worktrees, validation, proof gates, and merge
reconciliation).  This module joins those authorities without allowing the
Portal daemon to mutate the canonical task board: each database attempt gets a
single-task Markdown *projection* below its private state directory.

The projection is deliberately disposable and non-authoritative.  Its
immutable fields are sealed before provider execution; only its status line
may change.  A database phase may consume the result only after the projected
task has a matching durable Portal completion event.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import shlex
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..merge.checkout_lock import checkout_repository_id
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.task_identity import canonical_task_identity

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@2"
)
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@3"
)
DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@1"
)
DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@2"
)
DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@3"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@2"
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_OUTPUT_PATH_FIELDS: Final[tuple[str, ...]] = (
    "path",
    "output",
    "artifact_id",
    "fluent_id",
)
_DECLARED_OUTPUT_EFFECT_FIELDS: Final[frozenset[str]] = frozenset(
    {"effect_id", "declared_path", "effect"}
)
_MAX_ACCEPTED_SOURCE_EVENT_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_ACCEPTED_SOURCE_EVENT_LINES: Final[int] = 65_536
_MAX_ATTEMPT_CONTROL_BYTES: Final[int] = 4 * 1024 * 1024


class DatabasePortalBridgeError(RuntimeError):
    """A database claim could not obtain trustworthy Portal evidence."""


class DatabasePortalBridgeDeferred(DatabasePortalBridgeError):
    """Portal execution made bounded progress but is not yet acceptable."""


@dataclass(frozen=True)
class DatabasePortalAttemptPaths:
    """Private, non-authoritative paths for one database task attempt."""

    root: Path
    task_projection: Path
    binding: Path
    state: Path
    strategy: Path
    events: Path
    implementation_logs: Path


PortalDaemonFactory = Callable[[DatabasePortalAttemptPaths, str], Any]
PriorAttemptAuthority = Callable[
    [Any, Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]

CROSS_ATTEMPT_LIFECYCLE_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-cross-attempt-lifecycle-authority@1"
)
CROSS_ATTEMPT_LIFECYCLE_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-cross-attempt-lifecycle-recovery@1"
)
CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME: Final[str] = (
    "cross-attempt-lifecycle-recovery.json"
)
CROSS_ATTEMPT_DECLARED_OUTPUT_PRESERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-cross-attempt-declared-output-preservation@1"
)
CROSS_ATTEMPT_PROTECTED_STATE_CLEARANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-cross-attempt-protected-state-clearance@1"
)
CROSS_ATTEMPT_PROTECTED_STATE_RETIREMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-cross-attempt-protected-state-retirement@1"
)
CROSS_ATTEMPT_PROTECTED_STATE_ADOPTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-cross-attempt-protected-state-adoption@1"
)
_DECLARED_OUTPUT_BLOB_PREFIX: Final[str] = (
    "cross-attempt-declared-output-blob-"
)
_DECLARED_OUTPUT_PRESERVATION_PREFIX: Final[str] = (
    "cross-attempt-declared-output-preservation-"
)
_PROTECTED_STATE_MARKER_BLOB_PREFIX: Final[str] = (
    "cross-attempt-protected-state-marker-"
)
_PROTECTED_STATE_CLEARANCE_PREFIX: Final[str] = (
    "cross-attempt-protected-state-clearance-"
)
_PROTECTED_STATE_RETIREMENT_PREFIX: Final[str] = (
    "cross-attempt-protected-state-retirement-"
)
_PROTECTED_STATE_ADOPTION_PREFIX: Final[str] = (
    "cross-attempt-protected-state-adoption-"
)
_MAX_PRESERVED_DECLARED_OUTPUT_FILES: Final[int] = 64
_MAX_PRESERVED_DECLARED_OUTPUT_FILE_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_PRESERVED_DECLARED_OUTPUT_TOTAL_BYTES: Final[int] = 16 * 1024 * 1024
_SUPERSEDED_LIFECYCLE_TERMINAL_REASON: Final[str] = (
    "superseded_database_attempt_preserved"
)
_SENSITIVE_DECLARED_OUTPUT_PATH = re.compile(
    r"(?i)(?:^|/)(?:\.env(?:\.|$)|\.ssh(?:/|$)|secrets?(?:/|$)|"
    r"credentials?(?:/|$)|private[-_]?keys?(?:/|$)|[^/]*\.(?:pem|key|p12|pfx))"
)
_ACTIVE_PROTECTED_STATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "recorded_at",
        "task_id",
        "attempt",
        "workspace_path",
        "ephemeral_worktree",
        "protected_paths",
        "snapshot",
    }
)
_PREPARED_PROTECTED_CLEARANCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_id",
        "attempt",
        "current_attempt_id",
        "prior_attempt_id",
        "current_binding_id",
        "prior_binding_id",
        "lifecycle_record_id",
        "lifecycle_authority_id",
        "lifecycle_transition_basis_id",
        "prior_lifecycle_state",
        "prior_lifecycle_fence",
        "expected_terminal_lifecycle_fence",
        "terminal_reason",
        "database_authority_id",
        "portal_state_binding_id",
        "active_marker_sha256",
        "active_marker_blob_filename",
        "active_marker",
        "protected_path_proof",
        "preservation",
        "workspace_status_id",
        "clearance_phase",
        "active_marker_retired",
        "retirement_operation",
        "worktree_deleted",
        "provider_dispatched",
        "mutation_authority",
        "merge_authority",
        "task_completion_authority",
        "worker_self_approval",
        "normal_validation_required",
        "clearance_id",
        "receipt_id",
    }
)
_PROTECTED_RETIREMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "prepared_clearance",
        "prepared_clearance_filename",
        "retired_marker_filename",
        "active_marker_sha256",
        "clearance_phase",
        "active_marker_retired",
        "retirement_operation",
        "worktree_deleted",
        "provider_dispatched",
        "mutation_authority",
        "merge_authority",
        "task_completion_authority",
        "worker_self_approval",
        "normal_validation_required",
        "retirement_id",
        "receipt_id",
    }
)
_PROTECTED_ADOPTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_id",
        "successor_attempt_id",
        "successor_binding_id",
        "abandoned_attempt_id",
        "abandoned_binding_id",
        "prior_attempt_id",
        "prior_binding_id",
        "lifecycle_recovery_id",
        "lifecycle_recovery_receipt_id",
        "lifecycle_recovery_phase",
        "observed_lifecycle_state",
        "observed_lifecycle_fence",
        "observed_lifecycle_authority_id",
        "successor_database_authority",
        "protected_artifacts",
        "adoption_operation",
        "provider_dispatched",
        "mutation_authority",
        "merge_authority",
        "task_completion_authority",
        "worker_self_approval",
        "normal_validation_required",
        "adoption_id",
        "receipt_id",
    }
)
_ATTEMPT_DIRECTORY = re.compile(r"[0-9a-f]{24}")
_BINDING_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "goal_cid",
        "plan_cid",
        "task_revision",
        "fencing_token",
        "fence_epoch",
        "lease_id",
        "task_body_digest",
        "projection_seed_digest",
        "projection_immutable_digest",
        "authoritative_task_store",
        "projection_authority",
        "binding_id",
    }
)
_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        *_BINDING_FIELDS_V1,
        "control_binding_id",
        "control_task_projection_cid",
        "control_expected_revision",
        "control_portal_binding_basis_cid",
    }
)
_PRIOR_AUTHORITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "authorized",
        "task_cid",
        "task_alias",
        "current_attempt_id",
        "prior_attempt_id",
        "current_attempt_number",
        "prior_attempt_number",
        "current_binding_id",
        "prior_binding_id",
        "current_fencing_token",
        "prior_fencing_token",
        "current_control_binding_id",
        "prior_control_binding_id",
        "current_control_task_projection_cid",
        "prior_control_task_projection_cid",
        "current_control_expected_revision",
        "prior_control_expected_revision",
        "prior_execution_status",
        "prior_claim_state",
        "prior_coordination_status",
        "legacy_current_binding",
        "legacy_prior_binding",
        "mutation_authority",
        "completion_authority",
    }
)
_RECOVERY_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "phase",
        "task_cid",
        "task_alias",
        "current_attempt_id",
        "prior_attempt_id",
        "current_attempt_number",
        "prior_attempt_number",
        "current_binding_id",
        "prior_binding_id",
        "current_fencing_token",
        "prior_fencing_token",
        "lifecycle_record_id",
        "lifecycle_authority_id",
        "lifecycle_transition_basis_id",
        "prior_lifecycle_state",
        "prior_lifecycle_fence",
        "expected_terminal_lifecycle_fence",
        "terminal_lifecycle_authority_id",
        "terminal_reason",
        "database_authority",
        "portal_state_binding",
        "preservation",
        "worktree_deleted",
        "provider_dispatched",
        "task_completion_authority",
        "recovery_id",
        "receipt_id",
    }
)
_RECOVERY_ID_EXCLUDED_FIELDS: Final[frozenset[str]] = frozenset(
    {"phase", "terminal_lifecycle_authority_id", "recovery_id", "receipt_id"}
)
_ATTEMPT_DIRECTORY_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "attempt_root_device",
        "attempt_root_inode",
        "attempt_root_mode",
        "attempt_directory_device",
        "attempt_directory_inode",
        "attempt_directory_mode",
    }
)
_PORTAL_STATE_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "task_id",
        "canonical_task_cid",
        "canonical_task_key",
        "projection_identity_id",
        "portal_state_id",
        "implementation_lock_id",
        "active_attempt",
        "active_worktree_path",
        "active_branch",
        "attempt_directory_identity",
    }
)
_PRESERVATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "workspace_path",
        "branch",
        "head",
        "tree",
        "workspace_device",
        "workspace_inode",
        "workspace_mode",
        "process_inventory",
        "container_inventory",
        "preservation_mode",
    }
)
_CONTROL_CLAIM_BINDING_SCHEMA_V1: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-claim-binding@1"
)
_CONTROL_CLAIM_BINDING_SCHEMA_V2: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-claim-binding@2"
)
_CONTROL_PORTAL_BASIS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-binding-basis@1"
)
_CONTROL_CLAIM_BINDING_FIELDS_V2: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_cid",
        "claim_id",
        "attempt_id",
        "attempt_number",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "control_expected_status",
        "control_expected_revision",
        "control_task_projection_cid",
        "database_portal_binding_basis",
        "database_portal_binding_basis_cid",
        "binding_id",
    }
)
_CONTROL_PORTAL_BASIS_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_alias",
        "task_revision",
        "goal_cid",
        "plan_cid",
        "task_body_digest",
        "control_task_projection_cid",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def _attempt_key(attempt_id: Any) -> str:
    selected = str(attempt_id or "")
    if not selected:
        raise DatabasePortalBridgeError("database Portal attempt identity is empty")
    return hashlib.sha256(selected.encode("utf-8")).hexdigest()[:24]


def _canonical_transition_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source transition is not canonical JSON"
        ) from exc


def _reject_duplicate_event_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatabasePortalBridgeError(
                "Portal accepted-source event repeats a JSON key"
            )
        result[key] = value
    return result


def _reject_duplicate_control_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatabasePortalBridgeError(
                "database Portal control record repeats a JSON key"
            )
        result[key] = value
    return result


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DatabasePortalBridgeError(
            f"could not read Portal attempt artifact {path.name!r}"
        ) from exc


def _accepted_source_events(
    path: Path,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    """Read a bounded regular event log without following a link."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source events are unreadable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > _MAX_ACCEPTED_SOURCE_EVENT_BYTES
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source event log is not a bounded regular file"
            )
        payload = bytearray()
        while len(payload) <= _MAX_ACCEPTED_SOURCE_EVENT_BYTES:
            block = os.read(
                descriptor,
                min(
                    65_536,
                    _MAX_ACCEPTED_SOURCE_EVENT_BYTES + 1 - len(payload),
                ),
            )
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source events are unreadable"
        ) from exc
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or len(payload) != before.st_size
        or len(payload) > _MAX_ACCEPTED_SOURCE_EVENT_BYTES
    ):
        raise DatabasePortalBridgeError(
            "Portal accepted-source event log changed while read"
        )
    try:
        text = bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatabasePortalBridgeError(
            "Portal accepted-source events are not UTF-8"
        ) from exc
    lines = text.splitlines()
    if len(lines) > _MAX_ACCEPTED_SOURCE_EVENT_LINES:
        raise DatabasePortalBridgeError(
            "Portal accepted-source event log exceeds its line bound"
        )
    records: list[Mapping[str, Any]] = []
    for line in lines:
        if not line:
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_event_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {value}")
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "Portal accepted-source event log contains invalid JSON"
            ) from exc
        if not isinstance(record, Mapping):
            raise DatabasePortalBridgeError(
                "Portal accepted-source event is not an object"
            )
        records.append(record)
    return tuple(records), _sha256_bytes(bytes(payload))


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    sealed_directory_identity: Mapping[str, Any],
) -> None:
    """Durably replace one direct child of an already sealed attempt directory."""

    try:
        expected_device = sealed_directory_identity[
            "attempt_directory_device"
        ]
        expected_inode = sealed_directory_identity[
            "attempt_directory_inode"
        ]
    except (KeyError, TypeError) as exc:
        raise DatabasePortalBridgeError(
            "database Portal atomic write lacks a sealed directory identity"
        ) from exc
    if (
        type(expected_device) is not int
        or type(expected_inode) is not int
        or expected_device < 0
        or expected_inode < 1
    ):
        raise DatabasePortalBridgeError(
            "database Portal atomic write has an invalid sealed directory identity"
        )
    target_name = path.name
    if not target_name or target_name in {".", ".."}:
        raise DatabasePortalBridgeError(
            "database Portal atomic write target is not a direct child"
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_only is None:
        raise DatabasePortalBridgeError(
            "database Portal atomic write requires no-follow directory access"
        )

    directory_descriptor = -1
    temporary_descriptor = -1
    temporary_name = ""
    cleanup_failure: OSError | None = None

    def require_safe_target() -> None:
        try:
            target_identity = os.stat(
                target_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal atomic write target is unreadable"
            ) from exc
        if not stat.S_ISREG(target_identity.st_mode):
            raise DatabasePortalBridgeError(
                "database Portal atomic write target is a symlink or nonregular file"
            )

    try:
        try:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY
                | directory_only
                | nofollow
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal atomic write parent is not the sealed directory"
            ) from exc
        observed_directory = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(observed_directory.st_mode)
            or observed_directory.st_dev != expected_device
            or observed_directory.st_ino != expected_inode
        ):
            raise DatabasePortalBridgeError(
                "database Portal atomic write parent identity changed after seal"
            )
        require_safe_target()

        temporary_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(16):
            candidate = f".{target_name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_descriptor = os.open(
                    candidate,
                    temporary_flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal atomic write temporary could not be created"
                ) from exc
            temporary_name = candidate
            break
        if temporary_descriptor < 0:
            raise DatabasePortalBridgeError(
                "database Portal atomic write temporary name bound exhausted"
            )

        temporary_identity = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(temporary_identity.st_mode)
            or temporary_identity.st_nlink != 1
        ):
            raise DatabasePortalBridgeError(
                "database Portal atomic write temporary is not a private regular file"
            )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(temporary_descriptor, remaining)
            if written < 1:
                raise OSError("short database Portal atomic write")
            remaining = remaining[written:]
        os.fsync(temporary_descriptor)
        durable_temporary = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(durable_temporary.st_mode)
            or durable_temporary.st_dev != temporary_identity.st_dev
            or durable_temporary.st_ino != temporary_identity.st_ino
            or durable_temporary.st_nlink != 1
            or durable_temporary.st_size != len(payload)
        ):
            raise DatabasePortalBridgeError(
                "database Portal atomic write temporary identity changed"
            )
        os.close(temporary_descriptor)
        temporary_descriptor = -1

        named_temporary = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named_temporary.st_mode)
            or named_temporary.st_dev != durable_temporary.st_dev
            or named_temporary.st_ino != durable_temporary.st_ino
            or named_temporary.st_nlink != 1
            or named_temporary.st_size != len(payload)
        ):
            raise DatabasePortalBridgeError(
                "database Portal atomic write temporary name changed"
            )
        require_safe_target()
        before_replace = os.fstat(directory_descriptor)
        if (
            before_replace.st_dev != expected_device
            or before_replace.st_ino != expected_inode
        ):
            raise DatabasePortalBridgeError(
                "database Portal atomic write parent identity changed before replace"
            )
        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        published = os.stat(
            target_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(published.st_mode)
            or published.st_dev != durable_temporary.st_dev
            or published.st_ino != durable_temporary.st_ino
            or published.st_nlink != 1
            or published.st_size != len(payload)
        ):
            raise DatabasePortalBridgeError(
                "database Portal atomic write publication identity changed"
            )
        os.fsync(directory_descriptor)
    except DatabasePortalBridgeError:
        raise
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal atomic write failed"
        ) from exc
    finally:
        if temporary_descriptor >= 0:
            try:
                os.close(temporary_descriptor)
            except OSError as exc:
                cleanup_failure = exc
        if temporary_name and directory_descriptor >= 0:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_failure = cleanup_failure or exc
            else:
                try:
                    os.fsync(directory_descriptor)
                except OSError as exc:
                    cleanup_failure = cleanup_failure or exc
        if directory_descriptor >= 0:
            try:
                os.close(directory_descriptor)
            except OSError as exc:
                cleanup_failure = cleanup_failure or exc
        if cleanup_failure is not None:
            raise DatabasePortalBridgeError(
                "database Portal atomic write cleanup was not durable"
            ) from cleanup_failure


def _stable_regular_descriptor_bytes(
    descriptor: int,
    *,
    noun: str,
) -> tuple[bytes, os.stat_result]:
    """Read one already-open bounded regular file without changing its offset."""

    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > _MAX_ATTEMPT_CONTROL_BYTES
        ):
            raise DatabasePortalBridgeError(
                f"{noun} is not a bounded single-link regular file"
            )
        payload = bytearray()
        while len(payload) <= _MAX_ATTEMPT_CONTROL_BYTES:
            block = os.read(
                descriptor,
                min(
                    65_536,
                    _MAX_ATTEMPT_CONTROL_BYTES + 1 - len(payload),
                ),
            )
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DatabasePortalBridgeError(f"{noun} is unreadable") from exc
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or len(payload) != before.st_size
        or len(payload) > _MAX_ATTEMPT_CONTROL_BYTES
    ):
        raise DatabasePortalBridgeError(f"{noun} changed while read")
    return bytes(payload), after


def _stable_regular_bytes(path: Path, *, noun: str) -> tuple[bytes, os.stat_result]:
    """Read one bounded, single-link regular file and retain its exact identity."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DatabasePortalBridgeError(f"{noun} is unreadable") from exc
    try:
        return _stable_regular_descriptor_bytes(descriptor, noun=noun)
    finally:
        os.close(descriptor)


def _stable_regular_bytes_at(
    directory_descriptor: int,
    name: str,
    *,
    noun: str,
) -> tuple[bytes, os.stat_result]:
    """Read one direct child relative to a pinned directory descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise DatabasePortalBridgeError(f"{noun} is unreadable") from exc
    try:
        return _stable_regular_descriptor_bytes(descriptor, noun=noun)
    finally:
        os.close(descriptor)


def _publish_immutable_file(
    path: Path,
    payload: bytes,
    *,
    sealed_directory_identity: Mapping[str, Any],
) -> None:
    """Publish a content-addressed direct child without replacing prior evidence."""

    try:
        expected_device = sealed_directory_identity["attempt_directory_device"]
        expected_inode = sealed_directory_identity["attempt_directory_inode"]
    except (KeyError, TypeError) as exc:
        raise DatabasePortalBridgeError(
            "immutable database Portal artifact lacks a sealed directory identity"
        ) from exc
    if (
        type(expected_device) is not int
        or type(expected_inode) is not int
        or expected_device < 0
        or expected_inode < 1
    ):
        raise DatabasePortalBridgeError(
            "immutable database Portal artifact has an invalid directory identity"
        )
    target_name = path.name
    if not target_name or target_name in {".", ".."}:
        raise DatabasePortalBridgeError(
            "immutable database Portal artifact is not a direct child"
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_only is None:
        raise DatabasePortalBridgeError(
            "immutable database Portal artifact requires no-follow access"
        )

    directory_descriptor = -1
    temporary_descriptor = -1
    temporary_name = ""
    cleanup_failure: OSError | None = None
    verification_failure: Exception | None = None
    target_expected = False
    try:
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | directory_only
            | nofollow
            | getattr(os, "O_CLOEXEC", 0),
        )
        directory_identity = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_identity.st_mode)
            or directory_identity.st_dev != expected_device
            or directory_identity.st_ino != expected_inode
        ):
            raise DatabasePortalBridgeError(
                "immutable database Portal artifact parent identity changed"
            )
        try:
            observed, identity = _stable_regular_bytes_at(
                directory_descriptor,
                target_name,
                noun="immutable database Portal artifact",
            )
        except FileNotFoundError:
            pass
        else:
            if observed != payload or identity.st_nlink != 1:
                raise DatabasePortalBridgeError(
                    "immutable database Portal artifact conflicts"
                )
            target_expected = True
            return
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(16):
            candidate = f".{target_name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_descriptor = os.open(
                    candidate,
                    flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_descriptor < 0:
            raise DatabasePortalBridgeError(
                "immutable database Portal temporary name bound exhausted"
            )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(temporary_descriptor, remaining)
            if written < 1:
                raise OSError("short immutable database Portal artifact write")
            remaining = remaining[written:]
        os.fsync(temporary_descriptor)
        written_identity = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(written_identity.st_mode)
            or written_identity.st_nlink != 1
            or written_identity.st_size != len(payload)
        ):
            raise DatabasePortalBridgeError(
                "immutable database Portal temporary identity changed"
            )
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        try:
            os.link(
                temporary_name,
                target_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            pass
        target_expected = True
        os.fsync(directory_descriptor)
    except DatabasePortalBridgeError:
        raise
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "immutable database Portal artifact publication failed"
        ) from exc
    finally:
        if temporary_descriptor >= 0:
            try:
                os.close(temporary_descriptor)
            except OSError as exc:
                cleanup_failure = exc
        if temporary_name and directory_descriptor >= 0:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_failure = cleanup_failure or exc
            else:
                try:
                    os.fsync(directory_descriptor)
                except OSError as exc:
                    cleanup_failure = cleanup_failure or exc
        if (
            directory_descriptor >= 0
            and cleanup_failure is None
            and target_expected
        ):
            try:
                observed, identity = _stable_regular_bytes_at(
                    directory_descriptor,
                    target_name,
                    noun="immutable database Portal artifact",
                )
                if observed != payload or identity.st_nlink != 1:
                    raise DatabasePortalBridgeError(
                        "immutable database Portal artifact was not published exactly"
                    )
                directory_identity = os.fstat(directory_descriptor)
                named_directory_identity = os.stat(
                    path.parent,
                    follow_symlinks=False,
                )
                if (
                    directory_identity.st_dev != expected_device
                    or directory_identity.st_ino != expected_inode
                    or not stat.S_ISDIR(named_directory_identity.st_mode)
                    or named_directory_identity.st_dev != expected_device
                    or named_directory_identity.st_ino != expected_inode
                ):
                    raise DatabasePortalBridgeError(
                        "immutable database Portal artifact parent identity changed"
                    )
            except FileNotFoundError as exc:
                verification_failure = DatabasePortalBridgeError(
                    "immutable database Portal artifact disappeared"
                )
                verification_failure.__cause__ = exc
            except Exception as exc:
                verification_failure = exc
        if directory_descriptor >= 0:
            try:
                os.close(directory_descriptor)
            except OSError as exc:
                cleanup_failure = cleanup_failure or exc
        if cleanup_failure is not None:
            raise DatabasePortalBridgeError(
                "immutable database Portal artifact cleanup was not durable"
            ) from cleanup_failure
        if verification_failure is not None:
            raise verification_failure


def _stable_regular_utf8(path: Path, *, noun: str) -> str:
    """Read one bounded single-link regular file without following symlinks."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DatabasePortalBridgeError(f"{noun} is unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > _MAX_ATTEMPT_CONTROL_BYTES
        ):
            raise DatabasePortalBridgeError(
                f"{noun} is not a bounded regular file"
            )
        payload = bytearray()
        while len(payload) <= _MAX_ATTEMPT_CONTROL_BYTES:
            block = os.read(
                descriptor,
                min(
                    65_536,
                    _MAX_ATTEMPT_CONTROL_BYTES + 1 - len(payload),
                ),
            )
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DatabasePortalBridgeError(f"{noun} is unreadable") from exc
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or len(payload) != before.st_size
        or len(payload) > _MAX_ATTEMPT_CONTROL_BYTES
    ):
        raise DatabasePortalBridgeError(f"{noun} changed while read")
    try:
        return bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatabasePortalBridgeError(f"{noun} is not UTF-8") from exc


def _line_value(value: Any) -> str:
    if isinstance(value, str):
        selected = value
    elif isinstance(value, Mapping):
        selected = _canonical_json(dict(value)).decode("utf-8")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        selected = ", ".join(_line_value(item) for item in value)
    else:
        selected = str(value or "")
    return " ".join(selected.replace("\x00", "").splitlines()).strip()


def _canonical_declared_output_path(value: Any) -> str:
    """Return one exact repository-relative path or fail closed.

    This stricter profile applies only to the typed declared-output envelope.
    Legacy projected output strings retain their existing normalization in
    :func:`_mapping_path`.
    """

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value.splitlines() != [value]
        or value.lower() in {"none", "n/a"}
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output path is malformed"
        )
    if "," in value or "\\" in value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output path is malformed"
        )
    candidate = PurePosixPath(value)
    if (
        value.startswith("/")
        or candidate.is_absolute()
        or not candidate.parts
        or "." in candidate.parts
        or ".." in candidate.parts
        or candidate.as_posix() != value
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output path is not canonical and repo-relative"
        )
    return value


def _nested_declared_output_path(value: Mapping[str, Any]) -> str | None:
    """Resolve the closed IntentRepository declared-output envelope.

    IntentRepository deliberately stores the effect identity in the outer
    ``path`` column and the exact repository path in the canonical nested
    effect.  The nested path may replace that storage identity in a disposable
    Portal projection only when the whole typed declaration is exact and the
    two identity copies agree.
    """

    if (
        "declared_path" in value
        or value.get("effect") == "declared_output"
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output declaration must be nested"
        )

    effect = value.get("effect")
    if not isinstance(effect, Mapping):
        return None
    is_declared_output = (
        "declared_path" in effect
        or effect.get("effect") == "declared_output"
    )
    if not is_declared_output:
        return None
    if set(effect) != _DECLARED_OUTPUT_EFFECT_FIELDS:
        raise DatabasePortalBridgeError(
            "database task declared-output effect is not a closed record"
        )
    if effect.get("effect") != "declared_output":
        raise DatabasePortalBridgeError(
            "database task declared-output effect kind is invalid"
        )
    effect_id = effect.get("effect_id")
    if (
        not isinstance(effect_id, str)
        or not effect_id
        or effect_id != effect_id.strip()
        or any(character.isspace() for character in effect_id)
    ):
        raise DatabasePortalBridgeError(
            "database task declared-output effect identity is malformed"
        )

    outer_identities: list[str] = []
    for field in _OUTPUT_PATH_FIELDS:
        if field not in value:
            continue
        identity = value[field]
        if (
            not isinstance(identity, str)
            or not identity
            or identity != identity.strip()
            or any(character.isspace() for character in identity)
        ):
            raise DatabasePortalBridgeError(
                "database task declared-output outer identity is malformed"
            )
        outer_identities.append(identity)
    if not outer_identities:
        raise DatabasePortalBridgeError(
            "database task declared-output outer identity is missing"
        )
    if len(set(outer_identities)) != 1:
        raise DatabasePortalBridgeError(
            "database task declared-output outer identities conflict"
        )
    if outer_identities[0] != effect_id:
        raise DatabasePortalBridgeError(
            "database task declared-output effect identity conflicts with its outer identity"
        )
    return _canonical_declared_output_path(effect.get("declared_path"))


def _mapping_path(value: Mapping[str, Any]) -> str:
    declared_path = _nested_declared_output_path(value)
    if declared_path is not None:
        return declared_path

    selected = [
        _line_value(value[field])
        for field in _OUTPUT_PATH_FIELDS
        if value.get(field)
    ]
    selected = [item for item in selected if item]
    if len(set(selected)) > 1:
        raise DatabasePortalBridgeError(
            "database task output path declarations conflict"
        )
    return selected[0] if selected else _line_value(value)


def _output_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = getattr(record, "outputs", ()) or body.get("outputs") or ()
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    return list(
        dict.fromkeys(
            selected
            for item in raw
            if (
                selected := (
                    _mapping_path(item) if isinstance(item, Mapping) else _line_value(item)
                )
            )
        )
    )


def _validation_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = (
        getattr(record, "validations", ())
        or body.get("validations")
        or body.get("validation_commands")
        or body.get("validation")
        or ()
    )
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    selected: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            argv = item.get("argv")
            if isinstance(argv, Sequence) and not isinstance(
                argv, (str, bytes, bytearray, memoryview)
            ):
                value = shlex.join(str(part) for part in argv)
            else:
                value = _line_value(item.get("command") or item.get("value") or item)
        else:
            value = _line_value(item)
        if value and value not in selected:
            selected.append(value)
    return selected


def _acceptance_value(record: Any, body: Mapping[str, Any]) -> str:
    raw = (
        getattr(record, "acceptance", ())
        or body.get("acceptance")
        or body.get("completion_contract")
        or body.get("completion rule")
        or body.get("completion_rule")
        or ()
    )
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    values: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = _line_value(
                item.get("criterion") or item.get("statement") or item.get("value") or item
            )
        else:
            value = _line_value(item)
        if value:
            values.append(value)
    return " ; ".join(values)


def _projection_immutable_digest(text: str) -> str:
    normalized = _MUTABLE_PROJECTION_LINE.sub("- Status: <mutable>", text)
    return _sha256_bytes(normalized.encode("utf-8"))


def _projection_status(text: str) -> str:
    match = re.search(r"(?mi)^-\s*status\s*:\s*([^\r\n]+)$", text)
    return str(match.group(1) if match else "").strip().lower().replace("-", "_")


def _bounded_portal_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep control evidence while excluding raw provider/model payloads."""

    summary: dict[str, Any] = {}
    for key in (
        "task_count",
        "completed_count",
        "ready_count",
        "blocked_count",
        "active_task_id",
        "selection_idle_reason",
        "unchanged",
        "write_count",
        "blocked",
        "reason",
    ):
        if key in result:
            summary[key] = result[key]
    implementation = result.get("implementation_result")
    if isinstance(implementation, Mapping):
        summary["implementation"] = {
            key: implementation[key]
            for key in (
                "task_id",
                "attempt",
                "returncode",
                "reason",
                "deferred",
                "skipped",
                "implementation_commit",
                "branch",
                "merge_queued",
            )
            if key in implementation
        }
    reconciliation = result.get("merge_reconciliation")
    if isinstance(reconciliation, Sequence) and not isinstance(
        reconciliation, (str, bytes, bytearray, memoryview)
    ):
        summary["merge_reconciliation"] = [
            {
                key: item[key]
                for key in (
                    "task_id",
                    "returncode",
                    "reason",
                    "status",
                    "implementation_commit",
                    "merge_commit",
                    "resolved",
                )
                if key in item
            }
            for item in reconciliation[-8:]
            if isinstance(item, Mapping)
        ]
    return summary


class DatabasePortalExecutionBridge:
    """Run one database claim through a private Portal execution projection."""

    INTERFACE = DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
    RECEIPT_SCHEMA = DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA

    def __init__(
        self,
        *,
        task_source: Any,
        attempt_root: Path | str,
        portal_factory: PortalDaemonFactory,
        repo_root: Path | str | None = None,
        board_namespace: str = "",
        configured_board_admission_cid: str = "",
        configured_board_live_admission: Any | None = None,
        merge_target_branch: str = "",
        task_header_prefix: str = "## ",
        max_passes: int = 4,
        prior_attempt_authority: PriorAttemptAuthority | None = None,
    ) -> None:
        if not callable(portal_factory):
            raise TypeError("portal_factory must be callable")
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        self.task_source = task_source
        self.attempt_root = Path(attempt_root).absolute()
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.board_namespace = str(board_namespace or "").strip()
        self.configured_board_admission_cid = str(
            configured_board_admission_cid or ""
        ).strip()
        self._configured_board_live_admission_json = ""
        if configured_board_live_admission is not None:
            from ..runtime.configured_board_live_capsule import (
                parse_configured_board_live_capsule_admission,
            )

            raw_admission = (
                configured_board_live_admission.as_dict()
                if callable(
                    getattr(configured_board_live_admission, "as_dict", None)
                )
                else configured_board_live_admission
            )
            parsed_admission = parse_configured_board_live_capsule_admission(
                raw_admission
            )
            if (
                self.configured_board_admission_cid
                and parsed_admission.admission_cid
                != self.configured_board_admission_cid
            ):
                raise ValueError(
                    "configured-board admission CID disagrees with its live capsule"
                )
            if (
                self.board_namespace
                and parsed_admission.board_namespace != self.board_namespace
            ):
                raise ValueError(
                    "configured-board namespace disagrees with its live capsule"
                )
            self.configured_board_admission_cid = parsed_admission.admission_cid
            # Keep only a canonical immutable serialization.  Every use which
            # can grant source-transition authority reparses it and therefore
            # recomputes the admission CID; a caller cannot mutate a retained
            # mapping after construction to broaden the admitted source.
            self._configured_board_live_admission_json = (
                parsed_admission.to_json()
            )
        self.merge_target_branch = str(merge_target_branch or "").strip()
        self.portal_factory = portal_factory
        self.task_header_prefix = str(task_header_prefix or "## ")
        self.max_passes = max_passes
        if prior_attempt_authority is not None and not callable(
            prior_attempt_authority
        ):
            raise TypeError("prior_attempt_authority must be callable")
        self.prior_attempt_authority = prior_attempt_authority

    def _paths(self, attempt: Any) -> DatabasePortalAttemptPaths:
        root = self.attempt_root / _attempt_key(attempt.attempt_id)
        return DatabasePortalAttemptPaths(
            root=root,
            task_projection=root / "task-projection.runtime.todo.md",
            binding=root / "database-attempt-binding.json",
            state=root / "portal-task-state.json",
            strategy=root / "portal-strategy.json",
            events=root / "portal-events.jsonl",
            implementation_logs=root / "implementation-logs",
        )

    def _seal_attempt_directory(
        self,
        paths: DatabasePortalAttemptPaths,
        *,
        attempt_id: Any,
        create: bool,
    ) -> dict[str, int]:
        """Require a direct, non-symlink attempt directory before any write."""

        expected_name = _attempt_key(attempt_id)
        parent = self.attempt_root.parent
        try:
            parent_resolved = parent.resolve(strict=True)
            parent_identity = parent.lstat()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt parent is unavailable"
            ) from exc
        if not stat.S_ISDIR(parent_identity.st_mode) or parent.is_symlink():
            raise DatabasePortalBridgeError(
                "database Portal attempt parent is not a sealed directory"
            )

        if not os.path.lexists(self.attempt_root):
            if not create:
                raise DatabasePortalBridgeError(
                    "database Portal attempt root is unavailable"
                )
            try:
                os.mkdir(self.attempt_root, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal attempt root could not be created"
                ) from exc
        try:
            root_identity = self.attempt_root.lstat()
            root_resolved = self.attempt_root.resolve(strict=True)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt root is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(root_identity.st_mode)
            or self.attempt_root.is_symlink()
            or root_resolved.parent != parent_resolved
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt root is not a sealed direct child"
            )

        if paths.root.parent != self.attempt_root:
            raise DatabasePortalBridgeError(
                "database Portal attempt path escaped its sealed root"
            )
        if not os.path.lexists(paths.root):
            if not create:
                raise DatabasePortalBridgeError(
                    "database Portal attempt directory is unavailable"
                )
            try:
                os.mkdir(paths.root, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal attempt directory could not be created"
                ) from exc
        try:
            child_identity = paths.root.lstat()
            child_resolved = paths.root.resolve(strict=True)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt directory is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(child_identity.st_mode)
            or paths.root.is_symlink()
            or child_resolved.parent != root_resolved
            or paths.root.name != expected_name
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt directory is not a sealed direct child"
            )
        return {
            "attempt_root_device": int(root_identity.st_dev),
            "attempt_root_inode": int(root_identity.st_ino),
            "attempt_root_mode": int(root_identity.st_mode),
            "attempt_directory_device": int(child_identity.st_dev),
            "attempt_directory_inode": int(child_identity.st_ino),
            "attempt_directory_mode": int(child_identity.st_mode),
        }

    @staticmethod
    def _record_for_attempt(task_source: Any, attempt: Any) -> Any:
        getter = getattr(task_source, "get_task", None) or getattr(task_source, "get", None)
        if not callable(getter):
            raise DatabasePortalBridgeError("database task source does not expose get_task()")
        record = getter(str(attempt.task_cid))
        if record is None:
            raise DatabasePortalBridgeError(
                f"claimed database task {attempt.task_cid!r} disappeared"
            )
        if str(getattr(record, "task_cid", "")) != str(attempt.task_cid):
            raise DatabasePortalBridgeError("database task identity changed")
        attempt_alias = str(getattr(attempt, "task_alias", "") or "")
        record_alias = str(getattr(record, "task_alias", "") or "")
        if attempt_alias and record_alias and attempt_alias != record_alias:
            raise DatabasePortalBridgeError("database task alias changed")
        return record

    def _binding(
        self,
        attempt: Any,
        record: Any,
        seed: str,
        *,
        schema: str = "",
    ) -> dict[str, Any]:
        body = dict(getattr(record, "body", {}) or {})
        attempt_body = getattr(attempt, "body", {}) or {}
        control = (
            attempt_body.get("control_binding")
            if isinstance(attempt_body, Mapping)
            else None
        )
        portal_basis = (
            control.get("database_portal_binding_basis")
            if isinstance(control, Mapping)
            else None
        )
        has_control = False
        if control is not None:
            if type(control) is not dict:
                raise DatabasePortalBridgeError(
                    "database attempt control binding is malformed"
                )
            control_schema = control.get("schema")
            if control_schema == _CONTROL_CLAIM_BINDING_SCHEMA_V2:
                control_body = dict(control)
                control_binding_id = control_body.pop("binding_id", "")
                expected_attempt_control = {
                    "task_cid": str(attempt.task_cid),
                    "claim_id": str(attempt.claim_id),
                    "attempt_id": str(attempt.attempt_id),
                    "attempt_number": int(attempt.attempt_number),
                    "lease_id": str(getattr(attempt, "lease_id", "") or ""),
                    "owner_session_id": str(
                        getattr(attempt, "owner_session_id", "") or ""
                    ),
                    "fencing_token": int(attempt.fencing_token),
                    "fence_epoch": int(attempt.fence_epoch),
                }
                if (
                    set(control) != _CONTROL_CLAIM_BINDING_FIELDS_V2
                    or type(control_binding_id) is not str
                    or not control_binding_id
                    or content_identity(control_body) != control_binding_id
                    or any(
                        control.get(field) != value
                        for field, value in expected_attempt_control.items()
                    )
                    or control.get("control_expected_status") != "in_progress"
                    or type(control.get("control_expected_revision")) is not int
                    or int(control["control_expected_revision"]) < 1
                    or type(control.get("control_task_projection_cid")) is not str
                    or not control["control_task_projection_cid"]
                    or type(portal_basis) is not dict
                    or set(portal_basis) != _CONTROL_PORTAL_BASIS_FIELDS
                    or portal_basis.get("schema") != _CONTROL_PORTAL_BASIS_SCHEMA
                    or type(portal_basis.get("task_revision")) is not int
                    or portal_basis.get("task_revision")
                    != control["control_expected_revision"]
                    or portal_basis.get("control_task_projection_cid")
                    != control["control_task_projection_cid"]
                    or any(
                        type(portal_basis.get(field)) is not str
                        for field in _CONTROL_PORTAL_BASIS_FIELDS.difference(
                            {"task_revision"}
                        )
                    )
                    or control.get("database_portal_binding_basis_cid")
                    != content_identity(portal_basis)
                ):
                    raise DatabasePortalBridgeError(
                        "database attempt control binding is invalid"
                    )
                has_control = True
            elif control_schema != _CONTROL_CLAIM_BINDING_SCHEMA_V1:
                raise DatabasePortalBridgeError(
                    "database attempt control binding schema is unsupported"
                )
        if has_control:
            to_dict = getattr(record, "to_dict", None)
            record_revision = getattr(record, "revision", None)
            if not callable(to_dict) or type(record_revision) is not int or (
                record_revision != int(control["control_expected_revision"])
            ):
                raise DatabasePortalBridgeError(
                    "database task changed after claim-time control binding"
                )
            try:
                record_projection = dict(to_dict())
            except Exception as exc:
                raise DatabasePortalBridgeError(
                    "database task claim-time projection is unavailable"
                ) from exc
            if content_identity(record_projection) != str(
                control["control_task_projection_cid"]
            ):
                raise DatabasePortalBridgeError(
                    "database task projection changed after claim"
                )
        selected_schema = str(schema or "").strip() or (
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
            if has_control
            else DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1
        )
        if selected_schema not in {
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
            DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
        }:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding schema is unsupported"
            )
        if selected_schema == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA and not has_control:
            raise DatabasePortalBridgeError(
                "database Portal @2 binding lacks claim-time control authority"
            )
        payload = {
            "schema": selected_schema,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(
                getattr(record, "task_alias", "")
                or getattr(attempt, "task_alias", "")
                or attempt.task_cid
            ),
            "goal_cid": str(getattr(record, "goal_cid", "") or ""),
            "plan_cid": str(
                getattr(record, "plan_cid", "")
                or body.get("plan_cid")
                or body.get("plan_root_cid")
                or ""
            ),
            "task_revision": int(getattr(record, "revision", 0) or 0),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "task_body_digest": _sha256_bytes(_canonical_json(body)),
            "projection_seed_digest": _sha256_bytes(seed.encode("utf-8")),
            "projection_immutable_digest": _projection_immutable_digest(seed),
            "authoritative_task_store": "duckdb",
            "projection_authority": False,
        }
        if selected_schema == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA:
            assert isinstance(control, Mapping)
            assert isinstance(portal_basis, Mapping)
            basis_expected = {
                "task_alias": payload["task_alias"],
                "task_revision": payload["task_revision"],
                "goal_cid": payload["goal_cid"],
                "plan_cid": payload["plan_cid"],
                "task_body_digest": payload["task_body_digest"],
                "control_task_projection_cid": str(
                    control["control_task_projection_cid"]
                ),
            }
            if any(
                portal_basis.get(field) != value
                for field, value in basis_expected.items()
            ):
                raise DatabasePortalBridgeError(
                    "database Portal binding disagrees with claim-time task basis"
                )
            payload.update(
                {
                    "control_binding_id": str(control["binding_id"]),
                    "control_task_projection_cid": str(
                        control["control_task_projection_cid"]
                    ),
                    "control_expected_revision": int(
                        control["control_expected_revision"]
                    ),
                    "control_portal_binding_basis_cid": str(
                        control["database_portal_binding_basis_cid"]
                    ),
                }
            )
        payload["binding_id"] = _sha256_bytes(_canonical_json(payload))
        return payload

    def _render_projection(self, attempt: Any, record: Any) -> str:
        body = dict(getattr(record, "body", {}) or {})
        alias = _line_value(
            getattr(record, "task_alias", "")
            or getattr(attempt, "task_alias", "")
            or attempt.task_cid
        )
        if not alias or any(character.isspace() for character in alias):
            raise DatabasePortalBridgeError("database task alias is not projection-safe")
        title = _line_value(
            body.get("objective") or body.get("title") or body.get("description") or alias
        )
        outputs = _output_values(record, body)
        validations = _validation_values(record, body)
        acceptance = _acceptance_value(record, body)
        priority = _line_value(
            getattr(record, "priority", "") or body.get("priority") or "P2"
        )
        reserved = {
            "status",
            "completion",
            "priority",
            "track",
            "depends on",
            "depends_on",
            "outputs",
            "validation",
            "validations",
            "validation_commands",
            "acceptance",
        }
        lines = [
            "# Database attempt projection (non-authoritative)",
            "",
            f"## {alias} {title}",
            "",
            "- Status: ready",
            f"- Completion: {_line_value(body.get('completion') or 'auto')}",
            f"- Priority: {priority}",
            f"- Track: {_line_value(body.get('track') or 'implementation')}",
            "- Depends on:",
            f"- Outputs: {', '.join(outputs)}",
            f"- Validation: {' ; '.join(validations)}",
            f"- Acceptance: {acceptance}",
            f"- Database task CID: {_line_value(attempt.task_cid)}",
            f"- Database attempt ID: {_line_value(attempt.attempt_id)}",
            f"- Database claim ID: {_line_value(attempt.claim_id)}",
            f"- Database dependency CIDs: {_line_value(getattr(record, 'dependencies', ()))}",
            "- Projection authority: false",
        ]
        for key in sorted(body):
            normalized = str(key).strip().lower().replace("_", " ")
            if not normalized or normalized in reserved:
                continue
            if "credential" in normalized or "secret" in normalized:
                continue
            value = _line_value(body[key])
            if value:
                label = " ".join(word.capitalize() for word in normalized.split())
                lines.append(f"- {label}: {value}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _read_binding(path: Path) -> Mapping[str, Any]:
        try:
            value = json.loads(
                _stable_regular_utf8(
                    path,
                    noun="database Portal attempt binding",
                )
            )
        except DatabasePortalBridgeError:
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is unreadable"
            ) from exc
        if not isinstance(value, Mapping):
            raise DatabasePortalBridgeError("database Portal attempt binding is malformed")
        return value

    @classmethod
    def _strict_binding(cls, path: Path) -> dict[str, Any]:
        """Load one closed, self-hashed attempt binding without following links."""

        try:
            identity = path.lstat()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is unavailable"
            ) from exc
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is not a regular file"
            )
        value = cls._read_binding(path)
        if type(value) is not dict:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding shape is invalid"
            )
        schema = value.get("schema")
        expected_fields = (
            _BINDING_FIELDS
            if schema == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
            else _BINDING_FIELDS_V1
            if schema == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1
            else frozenset()
        )
        if not expected_fields or set(value) != expected_fields:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding shape is invalid"
            )
        normalized = dict(value)
        binding_id = normalized.pop("binding_id")
        integer_fields = ("task_revision", "fencing_token", "fence_epoch")
        if (
            any(type(value[field]) is not int for field in integer_fields)
            or any(int(value[field]) < 1 for field in integer_fields)
            or any(
                type(value[field]) is not str or not str(value[field])
                for field in (
                    "schema",
                    "interface",
                    "attempt_id",
                    "claim_id",
                    "task_cid",
                    "task_alias",
                    "lease_id",
                    "task_body_digest",
                    "projection_seed_digest",
                    "projection_immutable_digest",
                )
            )
            or type(value["goal_cid"]) is not str
            or type(value["plan_cid"]) is not str
            or value["schema"]
            not in {
                DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1,
                DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            }
            or value["interface"] != cls.INTERFACE
            or value["authoritative_task_store"] != "duckdb"
            or value["projection_authority"] is not False
            or type(binding_id) is not str
            or binding_id != _sha256_bytes(_canonical_json(normalized))
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt binding identity is invalid"
            )
        if value["schema"] == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA and (
            type(value["control_expected_revision"]) is not int
            or int(value["control_expected_revision"]) < 1
            or type(value["control_binding_id"]) is not str
            or not value["control_binding_id"]
            or type(value["control_task_projection_cid"]) is not str
            or not value["control_task_projection_cid"]
            or type(value["control_portal_binding_basis_cid"]) is not str
            or not value["control_portal_binding_basis_cid"]
            or int(value["control_expected_revision"])
            != int(value["task_revision"])
        ):
            raise DatabasePortalBridgeError(
                "database Portal @2 control binding identity is invalid"
            )
        return dict(value)

    @staticmethod
    def _direct_attempt_paths(root: Path) -> DatabasePortalAttemptPaths:
        return DatabasePortalAttemptPaths(
            root=root,
            task_projection=root / "task-projection.runtime.todo.md",
            binding=root / "database-attempt-binding.json",
            state=root / "portal-task-state.json",
            strategy=root / "portal-strategy.json",
            events=root / "portal-events.jsonl",
            implementation_logs=root / "implementation-logs",
        )

    def _prior_attempt_bindings(
        self,
        *,
        current_paths: DatabasePortalAttemptPaths,
        current_binding: Mapping[str, Any],
    ) -> list[tuple[DatabasePortalAttemptPaths, dict[str, Any]]]:
        """Return older, exact direct-sibling bindings for the same DB task."""

        try:
            attempt_root = self.attempt_root.resolve(strict=True)
            current_root = current_paths.root.resolve(strict=True)
            current_root.relative_to(attempt_root)
            children = tuple(attempt_root.iterdir())
        except (OSError, RuntimeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt root is unavailable"
            ) from exc
        if (
            current_root.parent != attempt_root
            or current_paths.root.is_symlink()
            or current_paths.root.name
            != _attempt_key(current_binding.get("attempt_id"))
        ):
            raise DatabasePortalBridgeError(
                "current database Portal attempt is not a direct sealed child"
            )
        if len(children) > 1024:
            raise DatabasePortalBridgeError(
                "database Portal attempt sibling bound exceeded"
            )

        candidates: list[tuple[DatabasePortalAttemptPaths, dict[str, Any]]] = []
        for child in sorted(children, key=lambda item: item.name):
            if child == current_paths.root or _ATTEMPT_DIRECTORY.fullmatch(
                child.name
            ) is None:
                continue
            try:
                child_identity = child.lstat()
                child_resolved = child.resolve(strict=True)
            except OSError:
                continue
            if (
                not stat.S_ISDIR(child_identity.st_mode)
                or child.is_symlink()
                or child_resolved.parent != attempt_root
            ):
                continue
            paths = self._direct_attempt_paths(child)
            if not paths.binding.exists():
                continue
            try:
                binding = self._strict_binding(paths.binding)
                self._verify_projection(paths, binding)
            except DatabasePortalBridgeError:
                # Malformed unrelated state cannot grant authority.  Leaving
                # it untouched avoids turning junk into a global denial of
                # service for every task in this lane.
                continue
            if child.name != _attempt_key(binding["attempt_id"]):
                continue
            prior_revision = int(binding["task_revision"])
            current_revision = int(current_binding.get("task_revision") or 0)
            if (
                binding["task_cid"] != current_binding.get("task_cid")
                or binding["task_alias"] != current_binding.get("task_alias")
                or binding["attempt_id"] == current_binding.get("attempt_id")
                or prior_revision > current_revision
                or int(binding["fencing_token"])
                >= int(current_binding.get("fencing_token") or 0)
                or int(binding["fence_epoch"])
                >= int(current_binding.get("fence_epoch") or 0)
            ):
                continue
            # A retry normally keeps the same authoritative task revision.
            # In that case, require the exact task payload and goal/plan
            # bindings to match.  Projection digests intentionally differ
            # because the attempt and claim identities are part of the
            # non-authoritative projection.
            if prior_revision == current_revision and any(
                binding[field] != current_binding.get(field)
                for field in ("task_body_digest", "goal_cid", "plan_cid")
            ):
                continue
            candidates.append((paths, binding))
        return candidates

    @staticmethod
    def _git_observation(
        repository: Path,
        *arguments: str,
        text: bool = True,
    ) -> subprocess.CompletedProcess[Any]:
        environment = {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
        try:
            return subprocess.run(
                ["/usr/bin/git", "--no-replace-objects", *arguments],
                cwd=repository,
                env=environment,
                capture_output=True,
                check=False,
                timeout=10.0,
                text=text,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross-attempt worktree Git observation is unavailable"
            ) from exc

    def _prior_projection_identity(
        self,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve one prior projection to its exact Portal task identity."""

        projection = self._verify_projection(paths, binding)
        try:
            from .implementation_daemon import (
                parse_task_text,
                task_declared_output_paths,
            )

            tasks = parse_task_text(
                projection,
                path=paths.task_projection,
                task_header_prefix=self.task_header_prefix,
            )
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_projection_identity_unavailable"
            ) from exc
        if len(tasks) != 1 or tasks[0].task_id != binding["task_alias"]:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_projection_identity_ambiguous"
            )
        task = tasks[0]
        metadata = dict(task.metadata)
        metadata.pop("canonical task cid", None)
        metadata.pop("canonical task key", None)
        canonical = canonical_task_identity(
            {
                "task_id": task.task_id,
                "title": task.title,
                "outputs": task_declared_output_paths(task),
                "acceptance": task.acceptance,
                "metadata": metadata,
            },
            board_namespace=(
                task.board_namespace
                or self.board_namespace
                or paths.task_projection.name
            ),
            source_path=paths.task_projection,
        )
        if not canonical.canonical_task_cid or not canonical.canonical_task_key:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_projection_identity_invalid"
            )
        return {
            "task_id": task.task_id,
            "canonical_task_cid": canonical.canonical_task_cid,
            "canonical_task_key": canonical.canonical_task_key,
        }

    def _prior_declared_output_paths(
        self,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Read the exact closed output scope from a sealed prior projection."""

        projection = self._verify_projection(paths, binding)
        try:
            from .implementation_daemon import (
                parse_task_text,
                task_declared_output_paths,
            )

            tasks = parse_task_text(
                projection,
                path=paths.task_projection,
                task_header_prefix=self.task_header_prefix,
            )
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_projection_unavailable"
            ) from exc
        if len(tasks) != 1 or tasks[0].task_id != binding["task_alias"]:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_projection_ambiguous"
            )
        outputs = tuple(task_declared_output_paths(tasks[0]))
        if (
            not outputs
            or len(outputs) > _MAX_PRESERVED_DECLARED_OUTPUT_FILES
            or len(set(outputs)) != len(outputs)
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_scope_invalid"
            )
        return tuple(sorted(outputs))

    def _prior_portal_state_binding(
        self,
        daemon: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        record: Any,
    ) -> dict[str, Any]:
        """Join a lifecycle claim to the old Portal active-task tuple."""

        identity = self._prior_projection_identity(paths, binding)
        try:
            state = json.loads(
                _stable_regular_utf8(
                    paths.state,
                    noun="prior Portal task state",
                )
            )
        except (DatabasePortalBridgeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_portal_state_unavailable"
            ) from exc
        if type(state) is not dict:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_portal_state_invalid"
            )
        try:
            workspace = str(Path(record.workspace_path).resolve(strict=True))
            state_workspace = str(
                Path(str(state["active_worktree_path"])).resolve(strict=True)
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_active_tuple_unavailable"
            ) from exc
        expected = {
            "implementation_in_progress": True,
            "active_task_id": identity["task_id"],
            "active_task_cid": identity["canonical_task_cid"],
            "active_task_key": identity["canonical_task_key"],
            "active_attempt": int(record.attempt),
            "active_branch": str(record.branch),
        }
        if (
            any(state.get(field) != value for field, value in expected.items())
            or state_workspace != workspace
            or record.task_id != identity["task_id"]
            or record.canonical_task_cid != identity["canonical_task_cid"]
            or not str(record.lane_id).startswith(
                f"{paths.root.resolve(strict=True)}:"
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_active_tuple_mismatch"
            )

        implementation_lock = paths.root / "implementation.lock"
        implementation_lock_id = "absent"
        if os.path.lexists(implementation_lock):
            try:
                lock = json.loads(
                    _stable_regular_utf8(
                        implementation_lock,
                        noun="prior Portal implementation lock",
                    )
                )
            except (DatabasePortalBridgeError, ValueError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_implementation_lock_invalid"
                ) from exc
            lock_active = getattr(
                daemon,
                "_lock_owner_is_active",
                None,
            )
            if (
                type(lock) is not dict
                or lock.get("kind") != "implementation"
                or str(Path(str(lock.get("state_dir") or "")).resolve(strict=False))
                != str(paths.root.resolve(strict=True))
                or lock.get("task_id") != identity["task_id"]
                or lock.get("canonical_task_cid")
                != identity["canonical_task_cid"]
                or type(lock.get("attempt")) is not int
                or int(lock["attempt"]) != int(record.attempt)
                or not callable(lock_active)
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_implementation_lock_invalid"
                )
            try:
                if lock_active(lock, expected_kind="implementation"):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_implementation_lock_active"
                    )
            except DatabasePortalBridgeDeferred:
                raise
            except Exception as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_implementation_lock_unavailable"
                ) from exc
            implementation_lock_id = content_identity(lock)
        state_id = content_identity(state)
        projection_identity_id = content_identity(identity)
        return {
            **identity,
            "projection_identity_id": projection_identity_id,
            "portal_state_id": state_id,
            "implementation_lock_id": implementation_lock_id,
            "active_attempt": int(record.attempt),
            "active_worktree_path": workspace,
            "active_branch": str(record.branch),
        }

    @staticmethod
    def _strict_workspace_process_scan(
        lifecycle_store: Any,
        workspace: Path,
    ) -> dict[str, Any]:
        """Fail closed while checking same-UID process argv and cwd via procfs."""

        proc_root = Path(getattr(lifecycle_store, "proc_root", Path("/proc")))
        try:
            root_identity = proc_root.lstat()
            entries = tuple(proc_root.iterdir())
        except OSError as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_process_inventory_unavailable"
            ) from exc
        if not stat.S_ISDIR(root_identity.st_mode) or proc_root.is_symlink():
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_process_inventory_unavailable"
            )
        if len(entries) > 1_000_000:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_process_inventory_bound_exceeded"
            )
        workspace_text = str(workspace)
        workspace_bytes = workspace_text.encode("utf-8")
        inspected = 0
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                identity = entry.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_process_inventory_unavailable"
                ) from exc
            if not stat.S_ISDIR(identity.st_mode) or identity.st_uid != os.geteuid():
                continue
            inspected += 1
            try:
                raw_cwd = os.readlink(entry / "cwd")
            except FileNotFoundError:
                raw_cwd = ""
            except OSError as exc:
                if exc.errno in {errno.ENOENT, errno.ESRCH}:
                    raw_cwd = ""
                else:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_process_inventory_unavailable"
                    ) from exc
            if raw_cwd:
                cwd = raw_cwd.removesuffix(" (deleted)")
                if cwd == workspace_text or cwd.startswith(f"{workspace_text}/"):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_worktree_process_active"
                    )

            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
                os,
                "O_NOFOLLOW",
                0,
            )
            try:
                descriptor = os.open(entry / "cmdline", flags)
            except FileNotFoundError:
                continue
            except OSError as exc:
                if exc.errno in {errno.ENOENT, errno.ESRCH}:
                    continue
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_process_inventory_unavailable"
                ) from exc
            try:
                command = bytearray()
                while len(command) <= 1024 * 1024:
                    block = os.read(
                        descriptor,
                        min(65_536, 1024 * 1024 + 1 - len(command)),
                    )
                    if not block:
                        break
                    command.extend(block)
            except OSError as exc:
                if exc.errno not in {errno.ENOENT, errno.ESRCH}:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_process_inventory_unavailable"
                    ) from exc
                command = bytearray()
            finally:
                os.close(descriptor)
            if len(command) > 1024 * 1024:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_process_inventory_bound_exceeded"
                )
            if workspace_bytes in bytes(command):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_worktree_process_active"
                )
        return {"same_uid_processes_inspected": inspected}

    @staticmethod
    def _mount_source_overlaps_workspace(
        raw_source: Any,
        workspace: Path,
    ) -> bool:
        """Resolve one host mount source and test bidirectional containment."""

        if raw_source in (None, ""):
            return False
        if type(raw_source) is not str or not raw_source.startswith("/"):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_container_inventory_invalid"
            )
        try:
            source = Path(raw_source).resolve(strict=True)
            resolved_workspace = workspace.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_container_inventory_unavailable"
            ) from exc
        return (
            source == resolved_workspace
            or source in resolved_workspace.parents
            or resolved_workspace in source.parents
        )

    @staticmethod
    def _strict_workspace_container_scan(workspace: Path) -> dict[str, Any]:
        """Fail closed on Docker list/inspect uncertainty for supervisor labels."""

        # Import lazily because implementation_daemon imports this bridge.
        # Invocation happens only after both modules have initialized.
        try:
            from .implementation_daemon import (
                AUTHORITY_VALIDATION_DOCKER_ENDPOINT,
                AUTHORITY_VALIDATION_DOCKER_PATH,
                IMPLEMENTATION_DOCKER_ISOLATION_LABELS,
            )
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_container_inventory_unavailable"
            ) from exc
        docker = Path(AUTHORITY_VALIDATION_DOCKER_PATH)
        socket_path = Path(
            str(AUTHORITY_VALIDATION_DOCKER_ENDPOINT).removeprefix("unix://")
        )
        try:
            socket_present = os.path.lexists(socket_path)
            docker_present = os.path.lexists(docker)
        except OSError as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_container_inventory_unavailable"
            ) from exc
        if not docker_present:
            if socket_present:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_unavailable"
                )
            return {"container_runtime": "unavailable", "containers_inspected": 0}
        try:
            docker_resolved = docker.resolve(strict=True)
            docker_identity = docker_resolved.stat()
            socket_resolved = socket_path.resolve(strict=True)
            socket_identity = socket_resolved.stat()
        except OSError as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_container_inventory_unavailable"
            ) from exc
        if (
            docker_resolved != docker
            or not stat.S_ISREG(docker_identity.st_mode)
            or int(docker_identity.st_uid) != 0
            or stat.S_IMODE(docker_identity.st_mode) & 0o022
            or not os.access(docker_resolved, os.X_OK)
            or socket_resolved != Path("/run/docker.sock")
            or not stat.S_ISSOCK(socket_identity.st_mode)
            or int(socket_identity.st_uid) != 0
            or stat.S_IMODE(socket_identity.st_mode) & 0o007
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_container_inventory_unavailable"
            )
        environment = {
            "DOCKER_CONFIG": "/nonexistent/ipfs-accelerate-docker-config",
            "DOCKER_HOST": str(AUTHORITY_VALIDATION_DOCKER_ENDPOINT),
            "HOME": "/nonexistent/ipfs-accelerate-docker-home",
            "PATH": "/usr/bin:/bin",
        }
        container_ids: set[str] = set()
        for label in tuple(IMPLEMENTATION_DOCKER_ISOLATION_LABELS):
            try:
                listed = subprocess.run(
                    [
                        str(docker_resolved),
                        "--host",
                        str(AUTHORITY_VALIDATION_DOCKER_ENDPOINT),
                        "ps",
                        "--filter",
                        f"label={label}",
                        "--format",
                        "{{.ID}}",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5.0,
                    env=environment,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_unavailable"
                ) from exc
            if listed.returncode != 0:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_unavailable"
                )
            container_ids.update(
                line.strip() for line in listed.stdout.splitlines() if line.strip()
            )
        for container_id in sorted(container_ids):
            if re.fullmatch(r"[0-9a-f]{12,64}", container_id) is None:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_invalid"
                )
            try:
                inspected = subprocess.run(
                    [
                        str(docker_resolved),
                        "--host",
                        str(AUTHORITY_VALIDATION_DOCKER_ENDPOINT),
                        "inspect",
                        "--format",
                        "{{json .Mounts}}",
                        container_id,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5.0,
                    env=environment,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_unavailable"
                ) from exc
            if inspected.returncode != 0:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_unavailable"
                )
            try:
                mounts = json.loads(str(inspected.stdout or ""))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_invalid"
                ) from exc
            if type(mounts) is not list or any(
                type(mount) is not dict for mount in mounts
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_container_inventory_invalid"
                )
            overlaps = False
            for mount in mounts:
                if DatabasePortalExecutionBridge._mount_source_overlaps_workspace(
                    mount.get("Source"),
                    workspace,
                ):
                    overlaps = True
                    break
            if overlaps:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_worktree_container_active"
                )
        return {
            "container_runtime": str(docker_resolved),
            "container_endpoint": str(AUTHORITY_VALIDATION_DOCKER_ENDPOINT),
            "isolation_labels": list(IMPLEMENTATION_DOCKER_ISOLATION_LABELS),
            "containers_inspected": len(container_ids),
        }

    @staticmethod
    def _canonical_recovery_path(value: Any) -> str:
        """Return one literal repository-relative path or fail closed."""

        if type(value) is not str or value != value.strip() or "\\" in value:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_path_invalid"
            )
        candidate = PurePosixPath(value)
        if (
            not value
            or value.startswith(("/", "./", "../", "//"))
            or candidate.as_posix() != value
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
            or any(character in value for character in "*?[")
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_path_invalid"
            )
        return value

    @staticmethod
    def _nul_path_records(raw: bytes, *, reason: str) -> tuple[str, ...]:
        records: list[str] = []
        for value in raw.split(b"\0"):
            if not value:
                continue
            try:
                decoded = value.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise DatabasePortalBridgeDeferred(reason) from exc
            records.append(
                DatabasePortalExecutionBridge._canonical_recovery_path(decoded)
            )
        if len(records) > 4096 or len(set(records)) != len(records):
            raise DatabasePortalBridgeDeferred(reason)
        return tuple(records)

    def _direct_gitlinks(self, workspace: Path) -> dict[str, str]:
        result = self._git_observation(
            workspace,
            "ls-files",
            "--stage",
            "-z",
            text=False,
        )
        if result.returncode != 0:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_gitlink_inventory_unavailable"
            )
        gitlinks: dict[str, str] = {}
        for raw in bytes(result.stdout or b"").split(b"\0"):
            if not raw:
                continue
            try:
                prefix, raw_path = raw.split(b"\t", 1)
                mode, object_id, stage = prefix.decode("ascii").split(" ", 2)
                path = self._canonical_recovery_path(
                    raw_path.decode("utf-8", errors="strict")
                )
            except (UnicodeDecodeError, ValueError) as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_gitlink_inventory_invalid"
                ) from exc
            if mode != "160000":
                continue
            if (
                stage != "0"
                or re.fullmatch(r"[0-9a-f]{40}", object_id) is None
                or path in gitlinks
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_gitlink_inventory_invalid"
                )
            gitlinks[path] = object_id
        if len(gitlinks) > 64:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_gitlink_inventory_bound_exceeded"
            )
        return gitlinks

    def _nested_gitlink_state_present(self, workspace: Path) -> bool:
        """Detect nested state, including ignored paths hidden from top Git status."""

        for gitlink_path in sorted(self._direct_gitlinks(workspace)):
            nested = workspace / gitlink_path
            if not os.path.lexists(nested):
                continue
            try:
                nested_identity = nested.lstat()
            except OSError as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_gitlink_inventory_unavailable"
                ) from exc
            if not stat.S_ISDIR(nested_identity.st_mode) or nested.is_symlink():
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_gitlink_identity_unsafe"
                )
            if not os.path.lexists(nested / ".git"):
                try:
                    first_entry = next(nested.iterdir(), None)
                except OSError as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_gitlink_inventory_unavailable"
                    ) from exc
                if first_entry is not None:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_uninitialized_gitlink_nonempty"
                    )
                continue
            status = self._git_observation(
                nested,
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=matching",
                "--ignore-submodules=none",
                text=False,
            )
            if status.returncode != 0:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_gitlink_inventory_unavailable"
                )
            if bytes(status.stdout or b""):
                return True
        return False

    @staticmethod
    def _declared_output_contains_secret(path: str, text: str) -> bool:
        if _SENSITIVE_DECLARED_OUTPUT_PATH.search(path):
            return True
        try:
            from ..validation.proposal_validation import (
                _PRIVATE_KEY_CONTENT_RE,
                _SECRET_ASSIGNMENT_RE,
                _is_concrete_secret_value,
            )
        except Exception:
            return True
        if _PRIVATE_KEY_CONTENT_RE.search(text):
            return True
        return any(
            _is_concrete_secret_value(match.group("value"))
            for match in _SECRET_ASSIGNMENT_RE.finditer(text)
        )

    def _declared_nested_output_observation(
        self,
        *,
        daemon: Any,
        record: Any,
        prior_paths: DatabasePortalAttemptPaths,
        prior_binding: Mapping[str, Any],
        workspace: Path,
        branch_name: str,
        head_id: str,
        tree_id: str,
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        """Prove that all dirtiness is bounded, declared nested output bytes."""

        declared_outputs = {
            self._canonical_recovery_path(path)
            for path in self._prior_declared_output_paths(
                prior_paths,
                prior_binding,
            )
        }
        configured_protected = tuple(
            str(path)
            for path in tuple(
                getattr(daemon, "implementation_protected_paths", ()) or ()
            )
        )
        if not configured_protected or any(
            output == protected
            or output.startswith(f"{protected.rstrip('/')}/")
            or protected.startswith(f"{output.rstrip('/')}/")
            for output in declared_outputs
            for protected in configured_protected
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_protected_overlap"
            )
        if any(_SENSITIVE_DECLARED_OUTPUT_PATH.search(path) for path in declared_outputs):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_sensitive_path"
            )

        for arguments, reason in (
            (
                ("diff", "--quiet", "--ignore-submodules=all"),
                "cross_attempt_declared_output_top_level_tracked_dirty",
            ),
            (
                ("diff", "--cached", "--quiet", "--ignore-submodules=all"),
                "cross_attempt_declared_output_top_level_staged",
            ),
        ):
            observed = self._git_observation(workspace, *arguments)
            if observed.returncode != 0:
                raise DatabasePortalBridgeDeferred(reason)
        for arguments, reason in (
            (
                ("ls-files", "--others", "--exclude-standard", "-z"),
                "cross_attempt_declared_output_top_level_untracked",
            ),
            (
                (
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "-z",
                ),
                "cross_attempt_declared_output_top_level_ignored",
            ),
        ):
            observed = self._git_observation(
                workspace,
                *arguments,
                text=False,
            )
            if observed.returncode != 0 or bytes(observed.stdout or b""):
                raise DatabasePortalBridgeDeferred(reason)

        gitlinks = self._direct_gitlinks(workspace)
        output_records: list[dict[str, Any]] = []
        blob_payloads: dict[str, bytes] = {}
        dirty_gitlinks: set[str] = set()
        total_bytes = 0
        for gitlink_path, expected_commit in sorted(gitlinks.items()):
            nested = workspace / gitlink_path
            try:
                resolved_nested = nested.resolve(strict=True)
                resolved_nested.relative_to(workspace.resolve(strict=True))
            except (OSError, RuntimeError, ValueError):
                continue
            if not resolved_nested.is_dir():
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_gitlink_unavailable"
                )
            nested_head = self._git_observation(
                resolved_nested,
                "rev-parse",
                "HEAD^{commit}",
            )
            if (
                nested_head.returncode != 0
                or str(nested_head.stdout or "").strip() != expected_commit
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_gitlink_head_changed"
                )
            for arguments, reason in (
                (
                    ("diff", "--quiet", "--ignore-submodules=none"),
                    "cross_attempt_declared_output_nested_tracked_dirty",
                ),
                (
                    (
                        "diff",
                        "--cached",
                        "--quiet",
                        "--ignore-submodules=none",
                    ),
                    "cross_attempt_declared_output_nested_staged",
                ),
            ):
                observed = self._git_observation(resolved_nested, *arguments)
                if observed.returncode != 0:
                    raise DatabasePortalBridgeDeferred(reason)
            untracked_result = self._git_observation(
                resolved_nested,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                text=False,
            )
            ignored_result = self._git_observation(
                resolved_nested,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
                text=False,
            )
            if untracked_result.returncode != 0 or ignored_result.returncode != 0:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_nested_inventory_unavailable"
                )
            untracked = self._nul_path_records(
                bytes(untracked_result.stdout or b""),
                reason="cross_attempt_declared_output_nested_inventory_invalid",
            )
            ignored = self._nul_path_records(
                bytes(ignored_result.stdout or b""),
                reason="cross_attempt_declared_output_nested_inventory_invalid",
            )
            if ignored:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_nested_ignored"
                )
            nested_status = self._git_observation(
                resolved_nested,
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
                text=False,
            )
            if nested_status.returncode != 0:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_nested_inventory_unavailable"
                )
            expected_status = {
                b"?? " + path.encode("utf-8") for path in untracked
            }
            actual_status = {
                value
                for value in bytes(nested_status.stdout or b"").split(b"\0")
                if value
            }
            if actual_status != expected_status:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_nested_status_ambiguous"
                )
            if not untracked:
                continue
            dirty_gitlinks.add(gitlink_path)
            for nested_path in untracked:
                repository_path = f"{gitlink_path}/{nested_path}"
                if repository_path not in declared_outputs:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_undeclared_nested_path"
                    )
                output_path = resolved_nested / nested_path
                try:
                    lexical_identity = output_path.lstat()
                    resolved_output = output_path.resolve(strict=True)
                    resolved_output.relative_to(resolved_nested)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_nested_path_escape"
                    ) from exc
                if (
                    not stat.S_ISREG(lexical_identity.st_mode)
                    or resolved_output != Path(os.path.abspath(output_path))
                ):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_nested_path_escape"
                    )
                payload, identity = _stable_regular_bytes(
                    output_path,
                    noun="declared nested output",
                )
                if (
                    identity.st_uid != os.geteuid()
                    or stat.S_IMODE(identity.st_mode) & 0o002
                    or identity.st_size
                    > _MAX_PRESERVED_DECLARED_OUTPUT_FILE_BYTES
                ):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_nested_identity_unsafe"
                    )
                try:
                    text = payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_nested_not_utf8"
                    ) from exc
                if "\x00" in text or self._declared_output_contains_secret(
                    repository_path,
                    text,
                ):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_secret_or_binary"
                    )
                total_bytes += len(payload)
                if total_bytes > _MAX_PRESERVED_DECLARED_OUTPUT_TOTAL_BYTES:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_declared_output_total_bound_exceeded"
                    )
                digest = hashlib.sha256(payload).hexdigest()
                blob_filename = f"{_DECLARED_OUTPUT_BLOB_PREFIX}{digest}.blob"
                blob_payloads[blob_filename] = payload
                output_records.append(
                    {
                        "repository_path": repository_path,
                        "gitlink_path": gitlink_path,
                        "nested_path": nested_path,
                        "gitlink_commit": expected_commit,
                        "sha256": f"sha256:{digest}",
                        "size": len(payload),
                        "mode": stat.S_IMODE(identity.st_mode),
                        "blob_filename": blob_filename,
                    }
                )
        if (
            not output_records
            or len(output_records) > _MAX_PRESERVED_DECLARED_OUTPUT_FILES
            or len({item["repository_path"] for item in output_records})
            != len(output_records)
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_nested_set_invalid"
            )

        top_status = self._git_observation(
            workspace,
            "status",
            "--ignore-submodules=none",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            text=False,
        )
        if top_status.returncode != 0:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_top_level_status_unavailable"
            )
        status_paths: set[str] = set()
        for value in bytes(top_status.stdout or b"").split(b"\0"):
            if not value:
                continue
            try:
                prefix = value[:3].decode("ascii")
                status_path = self._canonical_recovery_path(
                    value[3:].decode("utf-8", errors="strict")
                )
            except (UnicodeDecodeError, ValueError) as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_top_level_status_ambiguous"
                ) from exc
            if (
                len(value) < 4
                or prefix[0] != " "
                or prefix[1] not in {"?", "m", "M"}
                or prefix[2] != " "
                or status_path not in dirty_gitlinks
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_top_level_status_ambiguous"
                )
            status_paths.add(status_path)
        if status_paths != dirty_gitlinks:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_top_level_status_ambiguous"
            )
        workspace_identity = workspace.lstat()
        return (
            {
                "schema": CROSS_ATTEMPT_DECLARED_OUTPUT_PRESERVATION_SCHEMA,
                "task_id": str(record.task_id),
                "attempt": int(record.attempt),
                "prior_attempt_id": prior_binding["attempt_id"],
                "prior_binding_id": prior_binding["binding_id"],
                "lifecycle_record_id": str(record.record_id),
                "workspace_path": str(workspace),
                "workspace_device": int(workspace_identity.st_dev),
                "workspace_inode": int(workspace_identity.st_ino),
                "branch": branch_name,
                "head": head_id,
                "tree": tree_id,
                "projection_sha256": _sha256_file(prior_paths.task_projection),
                "declared_outputs": sorted(declared_outputs),
                "dirty_gitlinks": sorted(dirty_gitlinks),
                "outputs": sorted(
                    output_records,
                    key=lambda item: item["repository_path"],
                ),
                "total_bytes": total_bytes,
                "worktree_deleted": False,
                "provider_dispatched": False,
                "mutation_authority": False,
                "merge_authority": False,
                "task_completion_authority": False,
                "normal_validation_required": True,
            },
            blob_payloads,
        )

    def _preserve_declared_nested_outputs(
        self,
        *,
        daemon: Any,
        record: Any,
        prior_paths: DatabasePortalAttemptPaths,
        prior_binding: Mapping[str, Any],
        prior_directory_identity: Mapping[str, Any],
        workspace: Path,
        branch_name: str,
        head_id: str,
        tree_id: str,
    ) -> dict[str, Any]:
        """Content-address exact declared nested output bytes without admitting them."""

        first_body, blobs = self._declared_nested_output_observation(
            daemon=daemon,
            record=record,
            prior_paths=prior_paths,
            prior_binding=prior_binding,
            workspace=workspace,
            branch_name=branch_name,
            head_id=head_id,
            tree_id=tree_id,
        )
        preservation_id = _sha256_bytes(_canonical_json(first_body))
        receipt = {
            **first_body,
            "preservation_id": preservation_id,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        for filename, payload in sorted(blobs.items()):
            _publish_immutable_file(
                prior_paths.root / filename,
                payload,
                sealed_directory_identity=prior_directory_identity,
            )
        receipt_path = prior_paths.root / (
            f"{_DECLARED_OUTPUT_PRESERVATION_PREFIX}"
            f"{preservation_id.removeprefix('sha256:')[:24]}.json"
        )
        receipt_payload = json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        _publish_immutable_file(
            receipt_path,
            receipt_payload,
            sealed_directory_identity=prior_directory_identity,
        )
        second_body, second_blobs = self._declared_nested_output_observation(
            daemon=daemon,
            record=record,
            prior_paths=prior_paths,
            prior_binding=prior_binding,
            workspace=workspace,
            branch_name=branch_name,
            head_id=head_id,
            tree_id=tree_id,
        )
        if second_body != first_body or second_blobs != blobs:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_changed_during_preservation"
            )
        observed, _identity = _stable_regular_bytes(
            receipt_path,
            noun="declared nested output preservation receipt",
        )
        if observed != receipt_payload:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_preservation_receipt_changed"
            )
        return {
            "preservation_id": preservation_id,
            "receipt_id": receipt["receipt_id"],
            "receipt_path": str(receipt_path),
            "output_count": len(first_body["outputs"]),
            "total_bytes": int(first_body["total_bytes"]),
        }

    def _strict_active_protected_marker(
        self,
        *,
        path: Path,
        daemon: Any,
        record: Any,
        workspace: Path,
    ) -> tuple[dict[str, Any], bytes, os.stat_result]:
        raw, identity = _stable_regular_bytes(
            path,
            noun="cross-attempt active protected-path marker",
        )
        try:
            marker = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_control_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {value}")
                ),
            )
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_snapshot_malformed"
            ) from exc
        configured = tuple(
            str(value)
            for value in tuple(
                getattr(daemon, "implementation_protected_paths", ()) or ()
            )
        )
        snapshot = marker.get("snapshot") if type(marker) is dict else None
        if (
            type(marker) is not dict
            or set(marker) != _ACTIVE_PROTECTED_STATE_FIELDS
            or marker.get("schema") != "implementation-protected-path-active-v1"
            or type(marker.get("recorded_at")) is not str
            or not marker["recorded_at"]
            or marker.get("task_id") != record.task_id
            or type(marker.get("attempt")) is not int
            or int(marker["attempt"]) != int(record.attempt)
            or marker.get("workspace_path") != str(workspace)
            or marker.get("ephemeral_worktree") is not True
            or not configured
            or marker.get("protected_paths") != list(configured)
            or type(snapshot) is not dict
            or set(snapshot) != {"workspace", "shared_checkout"}
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_snapshot_binding_mismatch"
            )
        expected_roots = {
            "workspace": str(workspace),
            "shared_checkout": str(self.repo_root.resolve(strict=True)),
        }
        for scope, expected_root in expected_roots.items():
            scope_snapshot = snapshot.get(scope)
            if (
                type(scope_snapshot) is not dict
                or set(scope_snapshot) != {"root", "paths", "git_head"}
                or scope_snapshot.get("root") != expected_root
                or re.fullmatch(
                    r"[0-9a-f]{40}",
                    str(scope_snapshot.get("git_head") or ""),
                )
                is None
                or type(scope_snapshot.get("paths")) is not dict
                or set(scope_snapshot["paths"]) != set(configured)
                or len(scope_snapshot["paths"]) != len(configured)
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_protected_snapshot_malformed"
                )
            for protected_path, protected_identity in scope_snapshot["paths"].items():
                if type(protected_identity) is not dict:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_protected_snapshot_malformed"
                    )
                state = protected_identity.get("state")
                if state == "missing":
                    allowed_fields = {"state"}
                elif state == "present":
                    allowed_fields = {
                        "state",
                        "kind",
                        "device",
                        "inode",
                        "mode",
                        "links",
                        "uid",
                        "gid",
                        "size",
                        "mtime_ns",
                        "ctime_ns",
                    }
                    kind = protected_identity.get("kind")
                    if kind == "regular_file":
                        allowed_fields.add("sha256")
                    elif kind == "symlink":
                        allowed_fields.add("symlink_target")
                    elif kind not in {
                        "directory",
                        "fifo",
                        "socket",
                        "character_device",
                        "block_device",
                        "other",
                    }:
                        raise DatabasePortalBridgeDeferred(
                            "cross_attempt_lifecycle_protected_snapshot_malformed"
                        )
                else:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_protected_snapshot_malformed"
                    )
                if set(protected_identity) != allowed_fields:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_protected_snapshot_malformed"
                    )
                if state == "present" and (
                    any(
                        type(protected_identity[field]) is not int
                        or int(protected_identity[field]) < 0
                        for field in (
                            "device",
                            "inode",
                            "mode",
                            "links",
                            "uid",
                            "gid",
                            "size",
                            "mtime_ns",
                            "ctime_ns",
                        )
                    )
                    or (
                        protected_identity.get("kind") == "regular_file"
                        and re.fullmatch(
                            r"[0-9a-f]{64}",
                            str(protected_identity.get("sha256") or ""),
                        )
                        is None
                    )
                    or (
                        protected_identity.get("kind") == "symlink"
                        and type(protected_identity.get("symlink_target"))
                        is not str
                    )
                ):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_protected_snapshot_malformed"
                    )
                self._canonical_recovery_path(protected_path)
        return marker, raw, identity

    def _configured_source_transition_authority(
        self,
        *,
        workspace: Path,
        marker: Mapping[str, Any],
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        mutations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Verify an exact protected update against the admitted source capsule."""

        if not self._configured_board_live_admission_json or self.repo_root is None:
            return {}
        try:
            from ..runtime.configured_board_live_capsule import (
                parse_configured_board_live_capsule_admission,
            )

            parsed_admission = parse_configured_board_live_capsule_admission(
                self._configured_board_live_admission_json
            )
            admission = parsed_admission.as_dict()
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_configured_source_admission_invalid"
            ) from exc
        if (
            parsed_admission.admission_cid
            != self.configured_board_admission_cid
            or (
                self.board_namespace
                and parsed_admission.board_namespace != self.board_namespace
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_configured_source_admission_invalid"
            )
        before_workspace = before.get("workspace")
        after_workspace = after.get("workspace")
        before_shared = before.get("shared_checkout")
        after_shared = after.get("shared_checkout")
        protected_paths = marker.get("protected_paths")
        if (
            workspace.resolve() == self.repo_root.resolve()
            or type(before_workspace) is not dict
            or type(after_workspace) is not dict
            or before_workspace != after_workspace
            or type(before_shared) is not dict
            or type(after_shared) is not dict
            or type(protected_paths) is not list
            or not protected_paths
            or protected_paths != sorted(set(map(str, protected_paths)))
            or any(
                str(item.get("scope") or "") != "shared_checkout"
                for item in mutations
            )
        ):
            return {}
        before_head = str(before_shared.get("git_head") or "")
        after_head = str(after_shared.get("git_head") or "")
        source_head = str(admission.get("source_head") or "")
        source_tree = str(admission.get("source_tree") or "")
        if (
            re.fullmatch(r"[0-9a-f]{40}", before_head) is None
            or re.fullmatch(r"[0-9a-f]{40}", after_head) is None
            or source_head != after_head
            or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
            or admission.get("admission_cid")
            != self.configured_board_admission_cid
            or admission.get("board_namespace") != self.board_namespace
        ):
            return {}
        before_paths = before_shared.get("paths")
        after_paths = after_shared.get("paths")
        if (
            type(before_paths) is not dict
            or type(after_paths) is not dict
            or set(map(str, before_paths)) != set(protected_paths)
            or set(map(str, after_paths)) != set(protected_paths)
        ):
            return {}
        mutation_paths: list[str] = []
        for mutation in mutations:
            path = str(mutation.get("path") or "")
            if (
                path not in before_paths
                or path not in after_paths
                or mutation.get("before") != before_paths[path]
                or mutation.get("after") != after_paths[path]
            ):
                return {}
            mutation_paths.append(path)
        if (
            not mutation_paths
            or mutation_paths != sorted(set(mutation_paths))
        ):
            return {}

        observed_head = self._git_observation(
            self.repo_root, "rev-parse", "--verify", "HEAD^{commit}"
        )
        observed_tree = self._git_observation(
            self.repo_root, "rev-parse", "--verify", "HEAD^{tree}"
        )
        source_commit_tree = self._git_observation(
            self.repo_root,
            "rev-parse",
            "--verify",
            f"{source_head}^{{tree}}",
        )
        ancestry = self._git_observation(
            self.repo_root,
            "merge-base",
            "--is-ancestor",
            before_head,
            source_head,
        )
        status = self._git_observation(
            self.repo_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *protected_paths,
        )
        changed = self._git_observation(
            self.repo_root,
            "diff",
            "--name-only",
            "--no-renames",
            before_head,
            source_head,
            "--",
            *protected_paths,
        )
        if (
            any(
                result.returncode != 0
                for result in (
                    observed_head,
                    observed_tree,
                    source_commit_tree,
                    status,
                    changed,
                )
            )
            or ancestry.returncode != 0
            or observed_head.stdout.strip() != source_head
            or observed_tree.stdout.strip() != source_tree
            or source_commit_tree.stdout.strip() != source_tree
            or status.stdout
            or sorted(
                line.strip()
                for line in changed.stdout.splitlines()
                if line.strip()
            )
            != mutation_paths
        ):
            return {}
        artifacts = admission.get("control_artifacts")
        if type(artifacts) is not list or not artifacts:
            return {}
        if [str(item.get("path") or "") for item in artifacts] != protected_paths:
            return {}
        verified_artifacts: list[dict[str, Any]] = []
        for artifact in artifacts:
            if type(artifact) is not dict or set(artifact) != {
                "path",
                "sha256",
                "size",
            }:
                return {}
            relative = str(artifact.get("path") or "")
            try:
                canonical = self._canonical_recovery_path(relative)
                candidate = (self.repo_root / canonical).resolve(strict=True)
                candidate.relative_to(self.repo_root.resolve(strict=True))
                payload, _identity = _stable_regular_bytes(
                    candidate,
                    noun="configured-board control artifact",
                )
            except (
                DatabasePortalBridgeError,
                OSError,
                RuntimeError,
                ValueError,
            ):
                return {}
            if (
                len(payload) != artifact.get("size")
                or _sha256_bytes(payload) != artifact.get("sha256")
            ):
                return {}
            verified_artifacts.append(
                {
                    "path": canonical,
                    "sha256": artifact["sha256"],
                    "size": artifact["size"],
                }
            )
        changed_protected_blobs: list[dict[str, Any]] = []
        for path in mutation_paths:
            observed_entry = self._git_observation(
                self.repo_root,
                "ls-tree",
                "-z",
                source_head,
                "--",
                path,
                text=False,
            )
            if observed_entry.returncode != 0:
                return {}
            entries = observed_entry.stdout.split(b"\0")
            if entries[-1:] != [b""] or len(entries) != 2:
                return {}
            try:
                metadata, raw_path = entries[0].split(b"\t", 1)
                mode, kind, blob_oid = metadata.decode("ascii").split(" ", 2)
                decoded_path = raw_path.decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                return {}
            current_identity = after_paths[path]
            if (
                mode not in {"100644", "100755"}
                or kind != "blob"
                or re.fullmatch(r"[0-9a-f]{40}", blob_oid) is None
                or decoded_path != path
                or type(current_identity) is not dict
                or current_identity.get("kind") != "regular_file"
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(current_identity.get("sha256") or ""),
                )
                is None
            ):
                return {}
            changed_protected_blobs.append(
                {
                    "path": path,
                    "mode": mode,
                    "blob_oid": blob_oid,
                    "sha256": current_identity["sha256"],
                }
            )
        confirmed_head = self._git_observation(
            self.repo_root, "rev-parse", "--verify", "HEAD^{commit}"
        )
        confirmed_tree = self._git_observation(
            self.repo_root, "rev-parse", "--verify", "HEAD^{tree}"
        )
        confirmed_status = self._git_observation(
            self.repo_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *protected_paths,
        )
        if (
            confirmed_head.returncode != 0
            or confirmed_tree.returncode != 0
            or confirmed_status.returncode != 0
            or confirmed_head.stdout.strip() != source_head
            or confirmed_tree.stdout.strip() != source_tree
            or confirmed_status.stdout
        ):
            return {}
        return {
            "authority": "configured_board_live_capsule",
            "admission_cid": admission["admission_cid"],
            "before_head": before_head,
            "after_head": source_head,
            "after_tree": source_tree,
            "protected_paths": mutation_paths,
            "changed_protected_blobs": changed_protected_blobs,
            "verified_control_artifacts": verified_artifacts,
            "operator_or_worker_identity_inferred": False,
            "mutation_authority": False,
            "task_completion_authority": False,
        }

    def _protected_marker_snapshot_proof(
        self,
        *,
        daemon: Any,
        workspace: Path,
        marker: Mapping[str, Any],
    ) -> dict[str, Any]:
        snapshot = getattr(daemon, "_implementation_protected_path_snapshot", None)
        errors = getattr(daemon, "_implementation_protected_snapshot_errors", None)
        mutations_for = getattr(
            daemon,
            "_implementation_protected_path_mutations",
            None,
        )
        authorize = getattr(
            daemon,
            "_authorized_concurrent_protected_path_update",
            None,
        )
        if not all(callable(value) for value in (snapshot, errors, mutations_for, authorize)):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_snapshot_authority_unavailable"
            )
        before = marker["snapshot"]
        try:
            after = snapshot(workspace)
            snapshot_errors = errors(after)
            mutations = mutations_for(before, after)
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_snapshot_verification_unavailable"
            ) from exc
        if type(after) is not dict or snapshot_errors:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_snapshot_verification_failed"
            )
        trusted_update: dict[str, Any] = {}
        mode = "exact_snapshot"
        if mutations:
            if any(
                str(item.get("scope") or "") != "shared_checkout"
                for item in mutations
                if isinstance(item, Mapping)
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_protected_snapshot_mutated"
                )
            # A process born with a configured-board live capsule may accept
            # only that exact admitted source generation.  Never downgrade a
            # mismatch or malformed capsule to the older author-based rule.
            if self._configured_board_live_admission_json:
                trusted_update = self._configured_source_transition_authority(
                    workspace=workspace,
                    marker=marker,
                    before=before,
                    after=after,
                    mutations=mutations,
                )
            else:
                try:
                    trusted_update = dict(
                        authorize(
                            workspace_path=workspace,
                            before=before,
                            after=after,
                            mutations=mutations,
                        )
                        or {}
                    )
                except Exception as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_protected_snapshot_authority_unavailable"
                    ) from exc
            if not trusted_update:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_protected_snapshot_mutated"
                )
            mode = "current_trusted_shared_source"
        return {
            "mode": mode,
            "baseline_snapshot_id": _sha256_bytes(_canonical_json(before)),
            "current_snapshot_id": _sha256_bytes(_canonical_json(after)),
            "current_snapshot": after,
            "trusted_shared_update": trusted_update,
        }

    @staticmethod
    def _strict_protected_recovery_receipt(
        path: Path,
        *,
        fields: frozenset[str],
        schema: str,
        noun: str,
    ) -> dict[str, Any]:
        raw, _identity = _stable_regular_bytes(path, noun=noun)
        try:
            value = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_control_keys,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {constant}")
                ),
            )
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_receipt_malformed"
            ) from exc
        if type(value) is not dict or set(value) != fields or value.get("schema") != schema:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_receipt_malformed"
            )
        return value

    def _validated_prepared_protected_clearance(
        self,
        receipt: Mapping[str, Any],
        *,
        current_binding: Mapping[str, Any],
        prior_binding: Mapping[str, Any],
        database_authority: Mapping[str, Any],
        portal_state_binding: Mapping[str, Any],
        record: Any,
        marker: Mapping[str, Any],
        marker_raw: bytes,
        preservation: Mapping[str, Any],
        status_id: str,
        lifecycle_recovery_receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        value = dict(receipt)
        receipt_id = value.get("receipt_id")
        receipt_body = dict(value)
        receipt_body.pop("receipt_id", None)
        clearance_id = receipt_body.get("clearance_id")
        clearance_body = dict(receipt_body)
        clearance_body.pop("clearance_id", None)
        marker_digest = hashlib.sha256(marker_raw).hexdigest()
        if lifecycle_recovery_receipt is None:
            lifecycle_authority_id = _sha256_bytes(
                _canonical_json(record.to_dict())
            )
            lifecycle_transition_basis_id = self._lifecycle_transition_basis_id(
                record
            )
            prior_lifecycle_state = record.state.value
            prior_lifecycle_fence = int(record.fence)
        else:
            lifecycle_authority_id = lifecycle_recovery_receipt[
                "lifecycle_authority_id"
            ]
            lifecycle_transition_basis_id = lifecycle_recovery_receipt[
                "lifecycle_transition_basis_id"
            ]
            prior_lifecycle_state = lifecycle_recovery_receipt[
                "prior_lifecycle_state"
            ]
            prior_lifecycle_fence = int(
                lifecycle_recovery_receipt["prior_lifecycle_fence"]
            )
        if (
            record.is_terminal
            and (
                self._lifecycle_transition_basis_id(record)
                != lifecycle_transition_basis_id
                or int(record.fence) != prior_lifecycle_fence + 1
                or record.terminal_reason != _SUPERSEDED_LIFECYCLE_TERMINAL_REASON
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_terminal_successor_mismatch"
            )
        expected = {
            "task_id": str(record.task_id),
            "attempt": int(record.attempt),
            "current_attempt_id": current_binding["attempt_id"],
            "prior_attempt_id": prior_binding["attempt_id"],
            "current_binding_id": current_binding["binding_id"],
            "prior_binding_id": prior_binding["binding_id"],
            "lifecycle_record_id": str(record.record_id),
            "lifecycle_authority_id": lifecycle_authority_id,
            "lifecycle_transition_basis_id": lifecycle_transition_basis_id,
            "prior_lifecycle_state": prior_lifecycle_state,
            "prior_lifecycle_fence": prior_lifecycle_fence,
            "expected_terminal_lifecycle_fence": prior_lifecycle_fence + 1,
            "terminal_reason": _SUPERSEDED_LIFECYCLE_TERMINAL_REASON,
            "database_authority_id": content_identity(database_authority),
            "portal_state_binding_id": content_identity(portal_state_binding),
            "active_marker_sha256": f"sha256:{marker_digest}",
            "active_marker_blob_filename": (
                f"{_PROTECTED_STATE_MARKER_BLOB_PREFIX}{marker_digest}.json"
            ),
            "active_marker": dict(marker),
            "workspace_status_id": status_id,
            "clearance_phase": "prepared",
            "active_marker_retired": False,
            "retirement_operation": "atomic_noreplace_rename_after_receipt",
            "worktree_deleted": False,
            "provider_dispatched": False,
            "mutation_authority": False,
            "merge_authority": False,
            "task_completion_authority": False,
            "worker_self_approval": False,
            "normal_validation_required": True,
        }
        proof = value.get("protected_path_proof")
        if (
            set(value) != _PREPARED_PROTECTED_CLEARANCE_FIELDS
            or value.get("schema") != CROSS_ATTEMPT_PROTECTED_STATE_CLEARANCE_SCHEMA
            or type(receipt_id) is not str
            or receipt_id != _sha256_bytes(_canonical_json(receipt_body))
            or type(clearance_id) is not str
            or clearance_id != _sha256_bytes(_canonical_json(clearance_body))
            or any(value.get(field) != expected_value for field, expected_value in expected.items())
            or type(proof) is not dict
            or proof.get("baseline_snapshot_id")
            != _sha256_bytes(_canonical_json(marker["snapshot"]))
            or type(value.get("preservation")) is not dict
            or self._stable_preservation_authority(value["preservation"])
            != self._stable_preservation_authority(preservation)
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_clearance_authority_mismatch"
            )
        return value

    def _publish_protected_retirement_receipt(
        self,
        *,
        prior_paths: DatabasePortalAttemptPaths,
        prior_directory_identity: Mapping[str, Any],
        prepared: Mapping[str, Any],
        prepared_filename: str,
        retired_marker_filename: str,
    ) -> dict[str, Any]:
        body = {
            "schema": CROSS_ATTEMPT_PROTECTED_STATE_RETIREMENT_SCHEMA,
            "prepared_clearance": dict(prepared),
            "prepared_clearance_filename": prepared_filename,
            "retired_marker_filename": retired_marker_filename,
            "active_marker_sha256": prepared["active_marker_sha256"],
            "clearance_phase": "retired",
            "active_marker_retired": True,
            "retirement_operation": "atomic_noreplace_rename_after_receipt",
            "worktree_deleted": False,
            "provider_dispatched": False,
            "mutation_authority": False,
            "merge_authority": False,
            "task_completion_authority": False,
            "worker_self_approval": False,
            "normal_validation_required": True,
        }
        retirement_id = _sha256_bytes(_canonical_json(body))
        receipt = {**body, "retirement_id": retirement_id}
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        filename = (
            f"{_PROTECTED_STATE_RETIREMENT_PREFIX}"
            f"{retirement_id.removeprefix('sha256:')[:24]}.json"
        )
        payload = json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        _publish_immutable_file(
            prior_paths.root / filename,
            payload,
            sealed_directory_identity=prior_directory_identity,
        )
        observed = self._strict_protected_recovery_receipt(
            prior_paths.root / filename,
            fields=_PROTECTED_RETIREMENT_FIELDS,
            schema=CROSS_ATTEMPT_PROTECTED_STATE_RETIREMENT_SCHEMA,
            noun="protected-state retirement receipt",
        )
        if observed != receipt:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_receipt_changed"
            )
        return {**receipt, "retirement_receipt_filename": filename}

    def _verify_or_finish_protected_marker_retirement(
        self,
        *,
        current_binding: Mapping[str, Any],
        prior_binding: Mapping[str, Any],
        database_authority: Mapping[str, Any],
        portal_state_binding: Mapping[str, Any],
        daemon: Any,
        record: Any,
        prior_paths: DatabasePortalAttemptPaths,
        prior_directory_identity: Mapping[str, Any],
        workspace: Path,
        preservation: Mapping[str, Any],
        status_id: str,
        lifecycle_recovery_receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Verify, or finish only the receipt-bound post-rename crash boundary."""

        families = {
            "prepared": tuple(prior_paths.root.glob(f"{_PROTECTED_STATE_CLEARANCE_PREFIX}*.json")),
            "marker_blob": tuple(prior_paths.root.glob(f"{_PROTECTED_STATE_MARKER_BLOB_PREFIX}*.json")),
            "retired_marker": tuple(prior_paths.root.glob("implementation-protected-path-retired-*.json")),
            "retirement": tuple(prior_paths.root.glob(f"{_PROTECTED_STATE_RETIREMENT_PREFIX}*.json")),
        }
        if not any(families.values()):
            return {}
        if lifecycle_recovery_receipt is None or not record.is_terminal:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_retirement_lacks_terminal_authority"
            )
        if any(len(paths) != 1 for name, paths in families.items() if name != "retirement") or len(families["retirement"]) > 1:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_artifacts_ambiguous"
            )
        prepared_path = families["prepared"][0]
        marker_blob_path = families["marker_blob"][0]
        retired_path = families["retired_marker"][0]
        prepared = self._strict_protected_recovery_receipt(
            prepared_path,
            fields=_PREPARED_PROTECTED_CLEARANCE_FIELDS,
            schema=CROSS_ATTEMPT_PROTECTED_STATE_CLEARANCE_SCHEMA,
            noun="prepared protected-state clearance receipt",
        )
        retired_raw, _retired_identity = _stable_regular_bytes(
            retired_path,
            noun="retired protected-path marker",
        )
        marker_blob_raw, _blob_identity = _stable_regular_bytes(
            marker_blob_path,
            noun="protected-path marker blob",
        )
        marker, parsed_raw, _marker_identity = self._strict_active_protected_marker(
            path=retired_path,
            daemon=daemon,
            record=record,
            workspace=workspace,
        )
        if retired_raw != marker_blob_raw or parsed_raw != retired_raw:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retired_marker_changed"
            )
        prepared = self._validated_prepared_protected_clearance(
            prepared,
            current_binding=current_binding,
            prior_binding=prior_binding,
            database_authority=database_authority,
            portal_state_binding=portal_state_binding,
            record=record,
            marker=marker,
            marker_raw=retired_raw,
            preservation=preservation,
            status_id=status_id,
            lifecycle_recovery_receipt=lifecycle_recovery_receipt,
        )
        expected_prepared_filename = (
            f"{_PROTECTED_STATE_CLEARANCE_PREFIX}"
            f"{prepared['clearance_id'].removeprefix('sha256:')[:24]}.json"
        )
        expected_retired_filename = (
            "implementation-protected-path-retired-"
            f"{prepared['clearance_id'].removeprefix('sha256:')[:24]}.json"
        )
        expected_marker_blob = prepared["active_marker_blob_filename"]
        if (
            prepared_path.name != expected_prepared_filename
            or retired_path.name != expected_retired_filename
            or marker_blob_path.name != expected_marker_blob
            or os.path.lexists(
                prior_paths.root / "implementation-protected-path-active.json"
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_binding_mismatch"
            )
        current_proof = self._protected_marker_snapshot_proof(
            daemon=daemon,
            workspace=workspace,
            marker=marker,
        )
        if current_proof.get("baseline_snapshot_id") != prepared[
            "protected_path_proof"
        ].get("baseline_snapshot_id"):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_snapshot_changed"
            )
        terminal = self._publish_protected_retirement_receipt(
            prior_paths=prior_paths,
            prior_directory_identity=prior_directory_identity,
            prepared=prepared,
            prepared_filename=expected_prepared_filename,
            retired_marker_filename=expected_retired_filename,
        )
        if families["retirement"] and families["retirement"][0].name != terminal[
            "retirement_receipt_filename"
        ]:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_receipt_ambiguous"
            )
        return terminal

    def _retire_dead_protected_active_marker(
        self,
        *,
        attempt: Any,
        current_binding: Mapping[str, Any],
        prior_binding: Mapping[str, Any],
        database_authority: Mapping[str, Any],
        portal_state_binding: Mapping[str, Any],
        daemon: Any,
        record: Any,
        prior_paths: DatabasePortalAttemptPaths,
        prior_directory_identity: Mapping[str, Any],
        workspace: Path,
        preservation: Mapping[str, Any],
        status_id: str,
        lifecycle_recovery_receipt: Mapping[str, Any] | None,
        perform_retirement: bool,
        publish_prepared_clearance: bool,
        successor_adoption: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Preserve and retire one exact dead-attempt active marker."""

        active_path = prior_paths.root / "implementation-protected-path-active.json"
        incident_path = prior_paths.root / "implementation-protected-path-incident.json"
        if os.path.lexists(incident_path):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_incident_active"
            )
        marker, marker_raw, marker_identity = self._strict_active_protected_marker(
            path=active_path,
            daemon=daemon,
            record=record,
            workspace=workspace,
        )
        try:
            exact_dead = daemon.worktree_lifecycle.require_exact_dead_owner(
                record.workspace_path,
                allow_terminal=bool(record.is_terminal),
                **self._lifecycle_expected(record),
            )
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_marker_dead_owner_unproven"
            ) from exc
        if exact_dead != record:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_marker_dead_owner_changed"
            )
        first_proof = self._protected_marker_snapshot_proof(
            daemon=daemon,
            workspace=workspace,
            marker=marker,
        )
        marker_digest = hashlib.sha256(marker_raw).hexdigest()
        if lifecycle_recovery_receipt is None:
            lifecycle_authority_id = _sha256_bytes(
                _canonical_json(record.to_dict())
            )
            lifecycle_transition_basis_id = self._lifecycle_transition_basis_id(
                record
            )
            prior_lifecycle_state = record.state.value
            prior_lifecycle_fence = int(record.fence)
        else:
            lifecycle_authority_id = lifecycle_recovery_receipt[
                "lifecycle_authority_id"
            ]
            lifecycle_transition_basis_id = lifecycle_recovery_receipt[
                "lifecycle_transition_basis_id"
            ]
            prior_lifecycle_state = lifecycle_recovery_receipt[
                "prior_lifecycle_state"
            ]
            prior_lifecycle_fence = int(
                lifecycle_recovery_receipt["prior_lifecycle_fence"]
            )
        if (
            record.is_terminal
            and (
                self._lifecycle_transition_basis_id(record)
                != lifecycle_transition_basis_id
                or int(record.fence) != prior_lifecycle_fence + 1
                or record.terminal_reason != _SUPERSEDED_LIFECYCLE_TERMINAL_REASON
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_terminal_successor_mismatch"
            )
        marker_blob_filename = (
            f"{_PROTECTED_STATE_MARKER_BLOB_PREFIX}{marker_digest}.json"
        )
        clearance_body = {
            "schema": CROSS_ATTEMPT_PROTECTED_STATE_CLEARANCE_SCHEMA,
            "task_id": str(record.task_id),
            "attempt": int(record.attempt),
            "current_attempt_id": current_binding["attempt_id"],
            "prior_attempt_id": prior_binding["attempt_id"],
            "current_binding_id": current_binding["binding_id"],
            "prior_binding_id": prior_binding["binding_id"],
            "lifecycle_record_id": str(record.record_id),
            "lifecycle_authority_id": lifecycle_authority_id,
            "lifecycle_transition_basis_id": lifecycle_transition_basis_id,
            "prior_lifecycle_state": prior_lifecycle_state,
            "prior_lifecycle_fence": prior_lifecycle_fence,
            "expected_terminal_lifecycle_fence": prior_lifecycle_fence + 1,
            "terminal_reason": _SUPERSEDED_LIFECYCLE_TERMINAL_REASON,
            "database_authority_id": content_identity(database_authority),
            "portal_state_binding_id": content_identity(portal_state_binding),
            "active_marker_sha256": f"sha256:{marker_digest}",
            "active_marker_blob_filename": marker_blob_filename,
            "active_marker": marker,
            "protected_path_proof": first_proof,
            "preservation": self._stable_preservation_authority(preservation),
            "workspace_status_id": status_id,
            "clearance_phase": "prepared",
            "active_marker_retired": False,
            "retirement_operation": "atomic_noreplace_rename_after_receipt",
            "worktree_deleted": False,
            "provider_dispatched": False,
            "mutation_authority": False,
            "merge_authority": False,
            "task_completion_authority": False,
            "worker_self_approval": False,
            "normal_validation_required": True,
        }
        clearance_id = _sha256_bytes(_canonical_json(clearance_body))
        receipt = {**clearance_body, "clearance_id": clearance_id}
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        receipt_path = prior_paths.root / (
            f"{_PROTECTED_STATE_CLEARANCE_PREFIX}"
            f"{clearance_id.removeprefix('sha256:')[:24]}.json"
        )
        receipt_payload = json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        existing_prepared = tuple(
            prior_paths.root.glob(f"{_PROTECTED_STATE_CLEARANCE_PREFIX}*.json")
        )
        existing_blobs = tuple(
            prior_paths.root.glob(f"{_PROTECTED_STATE_MARKER_BLOB_PREFIX}*.json")
        )
        existing_retired = tuple(
            prior_paths.root.glob("implementation-protected-path-retired-*.json")
        )
        existing_terminal = tuple(
            prior_paths.root.glob(f"{_PROTECTED_STATE_RETIREMENT_PREFIX}*.json")
        )
        if (
            any(path.name != receipt_path.name for path in existing_prepared)
            or any(path.name != marker_blob_filename for path in existing_blobs)
            or existing_retired
            or existing_terminal
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_retirement_artifacts_ambiguous"
            )

        if successor_adoption is None:
            second_authority = self._validated_prior_authority(
                self.prior_attempt_authority(
                    attempt,
                    current_binding,
                    prior_binding,
                ),
                current_binding=current_binding,
                prior_binding=prior_binding,
            )
            expected_second_authority = dict(database_authority)
        else:
            successor_binding = successor_adoption.get("current_binding")
            successor_authority = successor_adoption.get("database_authority")
            if type(successor_binding) is not dict or type(successor_authority) is not dict:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_successor_adoption_malformed"
                )
            second_authority = self._validated_prior_authority(
                self.prior_attempt_authority(
                    attempt,
                    successor_binding,
                    current_binding,
                ),
                current_binding=successor_binding,
                prior_binding=current_binding,
            )
            expected_second_authority = successor_authority
        second_portal = self._prior_portal_state_binding(
            daemon,
            prior_paths,
            prior_binding,
            record,
        )
        second_portal["attempt_directory_identity"] = dict(
            portal_state_binding["attempt_directory_identity"]
        )
        second_process = self._strict_workspace_process_scan(
            daemon.worktree_lifecycle,
            workspace,
        )
        second_container = self._strict_workspace_container_scan(workspace)
        second_marker, second_raw, second_identity = (
            self._strict_active_protected_marker(
                path=active_path,
                daemon=daemon,
                record=record,
                workspace=workspace,
            )
        )
        second_proof = self._protected_marker_snapshot_proof(
            daemon=daemon,
            workspace=workspace,
            marker=second_marker,
        )
        second_status = self._git_observation(
            workspace,
            "status",
            "--ignore-submodules=none",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            text=False,
        )
        preservation_mode = str(preservation.get("preservation_mode") or "")
        if preservation_mode.startswith(
            "content_addressed_declared_nested_outputs:"
        ):
            expected_preservation_id = preservation_mode.split(":", 1)[1]
            revalidated_preservation = self._preserve_declared_nested_outputs(
                daemon=daemon,
                record=record,
                prior_paths=prior_paths,
                prior_binding=prior_binding,
                prior_directory_identity=prior_directory_identity,
                workspace=workspace,
                branch_name=str(preservation["branch"]),
                head_id=str(preservation["head"]),
                tree_id=str(preservation["tree"]),
            )
            if (
                revalidated_preservation.get("preservation_id")
                != expected_preservation_id
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_declared_output_changed_before_marker_retirement"
                )
        elif self._nested_gitlink_state_present(workspace):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_declared_output_appeared_before_marker_retirement"
            )
        if (
            second_authority != expected_second_authority
            or second_portal != dict(portal_state_binding)
            or second_marker != marker
            or second_raw != marker_raw
            or second_proof != first_proof
            or second_status.returncode != 0
            or _sha256_bytes(bytes(second_status.stdout or b""))
            != status_id
            or second_identity.st_dev != marker_identity.st_dev
            or second_identity.st_ino != marker_identity.st_ino
            or second_identity.st_mode != marker_identity.st_mode
            or second_identity.st_size != marker_identity.st_size
            or not isinstance(second_process, Mapping)
            or not isinstance(second_container, Mapping)
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_marker_evidence_changed"
            )
        try:
            exact_dead = daemon.worktree_lifecycle.require_exact_dead_owner(
                record.workspace_path,
                allow_terminal=bool(record.is_terminal),
                **self._lifecycle_expected(record),
            )
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_marker_dead_owner_unproven"
            ) from exc
        if exact_dead != record:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_marker_dead_owner_changed"
            )
        validated_prepared = self._validated_prepared_protected_clearance(
            receipt,
            current_binding=current_binding,
            prior_binding=prior_binding,
            database_authority=database_authority,
            portal_state_binding=portal_state_binding,
            record=record,
            marker=marker,
            marker_raw=marker_raw,
            preservation=preservation,
            status_id=status_id,
            lifecycle_recovery_receipt=lifecycle_recovery_receipt,
        )
        if publish_prepared_clearance:
            _publish_immutable_file(
                prior_paths.root / marker_blob_filename,
                marker_raw,
                sealed_directory_identity=prior_directory_identity,
            )
            _publish_immutable_file(
                receipt_path,
                receipt_payload,
                sealed_directory_identity=prior_directory_identity,
            )
        elif existing_prepared or existing_blobs:
            if len(existing_prepared) != 1 or len(existing_blobs) != 1:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_protected_preparation_ambiguous"
                )
            observed_prepared, _prepared_identity = _stable_regular_bytes(
                receipt_path,
                noun="prepared protected-state clearance receipt",
            )
            observed_blob, _blob_identity = _stable_regular_bytes(
                prior_paths.root / marker_blob_filename,
                noun="protected-state marker blob",
            )
            if observed_prepared != receipt_payload or observed_blob != marker_raw:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_protected_preparation_changed"
                )
        if not perform_retirement:
            return {
                "clearance_id": clearance_id,
                "receipt_id": validated_prepared["receipt_id"],
                "receipt_path": str(receipt_path),
                "active_marker_retired": False,
            }
        if (
            not publish_prepared_clearance
            or lifecycle_recovery_receipt is None
            or not record.is_terminal
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_marker_retirement_precedes_terminal_authority"
            )

        retired_path = prior_paths.root / (
            "implementation-protected-path-retired-"
            f"{clearance_id.removeprefix('sha256:')[:24]}.json"
        )
        if os.path.lexists(retired_path):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_marker_retirement_conflict"
            )
        directory_descriptor = -1
        try:
            directory_descriptor = os.open(
                prior_paths.root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            directory_identity = os.fstat(directory_descriptor)
            if (
                directory_identity.st_dev
                != prior_directory_identity["attempt_directory_device"]
                or directory_identity.st_ino
                != prior_directory_identity["attempt_directory_inode"]
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_protected_marker_parent_changed"
                )
            try:
                os.stat(
                    retired_path.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_protected_marker_retirement_conflict"
                )
            from ..task_sources.quack_owner_mutation import (
                _rename_noreplace_at,
            )

            _rename_noreplace_at(
                directory_descriptor,
                active_path.name,
                retired_path.name,
            )
            os.fsync(directory_descriptor)
        except DatabasePortalBridgeDeferred:
            raise
        except (ImportError, OSError, RuntimeError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_marker_retirement_failed"
            ) from exc
        finally:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
        retired_raw, _retired_identity = _stable_regular_bytes(
            retired_path,
            noun="retired protected-path marker",
        )
        observed_receipt, _receipt_identity = _stable_regular_bytes(
            receipt_path,
            noun="protected-path clearance receipt",
        )
        if (
            retired_raw != marker_raw
            or observed_receipt != receipt_payload
            or os.path.lexists(active_path)
        ):
            raise DatabasePortalBridgeError(
                "cross-attempt protected-path marker retirement was not durable"
            )
        terminal_receipt = self._publish_protected_retirement_receipt(
            prior_paths=prior_paths,
            prior_directory_identity=prior_directory_identity,
            prepared=receipt,
            prepared_filename=receipt_path.name,
            retired_marker_filename=retired_path.name,
        )
        return {
            "clearance_id": clearance_id,
            "receipt_id": receipt["receipt_id"],
            "receipt_path": str(receipt_path),
            "retired_marker_path": str(retired_path),
            "retirement_id": terminal_receipt["retirement_id"],
            "retirement_receipt_id": terminal_receipt["receipt_id"],
            "retirement_receipt_path": str(
                prior_paths.root
                / terminal_receipt["retirement_receipt_filename"]
            ),
        }

    def _preserved_quiescent_worktree(
        self,
        daemon: Any,
        record: Any,
        prior_paths: DatabasePortalAttemptPaths,
        *,
        attempt: Any,
        current_binding: Mapping[str, Any],
        prior_binding: Mapping[str, Any],
        database_authority: Mapping[str, Any],
        portal_state_binding: Mapping[str, Any],
        lifecycle_recovery_receipt: Mapping[str, Any] | None,
        perform_marker_retirement: bool,
        publish_marker_clearance: bool,
        successor_adoption: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prove that a dead attempt's bytes are committed and quiescent."""

        if self.repo_root is None:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_repository_unbound"
            )
        try:
            repository = self.repo_root.resolve(strict=True)
            workspace = Path(str(record.workspace_path)).resolve(strict=True)
            worktree_root = Path(str(daemon.worktree_root)).resolve(strict=True)
            workspace.relative_to(worktree_root)
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_worktree_unbound"
            ) from exc
        if (
            workspace == repository
            or str(Path(str(record.repo_root)).resolve(strict=False))
            != str(repository)
            or str(Path(str(record.state_dir)).resolve(strict=False))
            != str(prior_paths.root.resolve(strict=True))
            or str(record.merge_target).removeprefix("refs/heads/")
            != self.merge_target_branch.removeprefix("refs/heads/")
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_binding_mismatch"
            )
        incident_path = prior_paths.root / "implementation-protected-path-incident.json"
        active_path = prior_paths.root / "implementation-protected-path-active.json"
        if os.path.lexists(incident_path):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_incident_active"
            )

        process_inventory = self._strict_workspace_process_scan(
            daemon.worktree_lifecycle,
            workspace,
        )
        container_inventory = self._strict_workspace_container_scan(workspace)

        status = self._git_observation(
            workspace,
            "status",
            "--ignore-submodules=none",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            text=False,
        )
        head = self._git_observation(workspace, "rev-parse", "HEAD^{commit}")
        tree = self._git_observation(workspace, "rev-parse", "HEAD^{tree}")
        branch = self._git_observation(
            workspace,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        )
        try:
            workspace_identity = workspace.lstat()
        except OSError as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_worktree_identity_unavailable"
            ) from exc
        if (
            status.returncode != 0
            or head.returncode != 0
            or tree.returncode != 0
            or branch.returncode != 0
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_worktree_not_preserved"
            )
        head_id = str(head.stdout or "").strip()
        tree_id = str(tree.stdout or "").strip()
        branch_name = str(branch.stdout or "").strip()
        if (
            re.fullmatch(r"[0-9a-f]{40}", head_id) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree_id) is None
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_worktree_identity_invalid"
            )
        original_branch = str(record.branch).removeprefix("refs/heads/")
        rescue = branch_name.startswith("rescue/worktree/")
        if branch_name != original_branch and not rescue:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_preservation_branch_invalid"
            )
        ref = self._git_observation(
            repository,
            "rev-parse",
            f"refs/heads/{branch_name}^{{commit}}",
        )
        if ref.returncode != 0 or str(ref.stdout or "").strip() != head_id:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_preservation_ref_changed"
            )
        if rescue:
            metadata = self._git_observation(
                workspace,
                "show",
                "-s",
                "--format=%ae%x00%s%x00%b",
                "HEAD",
            )
            fields = str(metadata.stdout or "").split("\x00", 2)
            if (
                metadata.returncode != 0
                or len(fields) != 3
                or fields[0].strip()
                != "implementation-supervisor@example.invalid"
                or fields[1].strip()
                != f"Rescue dirty worktree {original_branch}"
                or f"Original branch: {original_branch}" not in fields[2]
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_rescue_receipt_invalid"
                )
        prior_directory_identity = self._seal_attempt_directory(
            prior_paths,
            attempt_id=prior_binding["attempt_id"],
            create=False,
        )
        status_bytes = bytes(status.stdout or b"")
        declared_output_preservation: dict[str, Any] = {}
        if status_bytes or self._nested_gitlink_state_present(workspace):
            declared_output_preservation = self._preserve_declared_nested_outputs(
                daemon=daemon,
                record=record,
                prior_paths=prior_paths,
                prior_binding=prior_binding,
                prior_directory_identity=prior_directory_identity,
                workspace=workspace,
                branch_name=branch_name,
                head_id=head_id,
                tree_id=tree_id,
            )
        preservation_mode = (
            "content_addressed_declared_nested_outputs:"
            f"{declared_output_preservation['preservation_id']}"
            if declared_output_preservation
            else ("supervisor_rescue_commit" if rescue else "clean_branch_commit")
        )
        preservation = {
            "workspace_path": str(workspace),
            "branch": branch_name,
            "head": head_id,
            "tree": tree_id,
            "workspace_device": int(workspace_identity.st_dev),
            "workspace_inode": int(workspace_identity.st_ino),
            "workspace_mode": int(workspace_identity.st_mode),
            "process_inventory": process_inventory,
            "container_inventory": container_inventory,
            "preservation_mode": preservation_mode,
        }
        if os.path.lexists(active_path):
            self._retire_dead_protected_active_marker(
                attempt=attempt,
                current_binding=current_binding,
                prior_binding=prior_binding,
                database_authority=database_authority,
                portal_state_binding=portal_state_binding,
                daemon=daemon,
                record=record,
                prior_paths=prior_paths,
                prior_directory_identity=prior_directory_identity,
                workspace=workspace,
                preservation=preservation,
                status_id=_sha256_bytes(status_bytes),
                lifecycle_recovery_receipt=lifecycle_recovery_receipt,
                perform_retirement=perform_marker_retirement,
                publish_prepared_clearance=publish_marker_clearance,
                successor_adoption=successor_adoption,
            )
        else:
            recovered_retirement = self._verify_or_finish_protected_marker_retirement(
                current_binding=current_binding,
                prior_binding=prior_binding,
                database_authority=database_authority,
                portal_state_binding=portal_state_binding,
                daemon=daemon,
                record=record,
                prior_paths=prior_paths,
                prior_directory_identity=prior_directory_identity,
                workspace=workspace,
                preservation=preservation,
                status_id=_sha256_bytes(status_bytes),
                lifecycle_recovery_receipt=lifecycle_recovery_receipt,
            )
            if recovered_retirement and (
                lifecycle_recovery_receipt is None or not record.is_terminal
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_retirement_lacks_terminal_authority"
                )
        return preservation

    @staticmethod
    def _stable_preservation_authority(
        preservation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return only decision-relevant preservation facts.

        Process and container totals are bounded audit observations, not
        authority.  Unrelated same-UID processes or labeled containers may
        appear between the two mandatory scans.  Each scan still fails closed
        when it finds an overlap; excluding ambient totals here prevents safe
        parallel activity from invalidating an otherwise exact recovery.
        """

        stable = {
            field: preservation[field]
            for field in _PRESERVATION_FIELDS.difference(
                {"process_inventory", "container_inventory"}
            )
        }
        container = dict(preservation["container_inventory"])
        container.pop("containers_inspected", None)
        stable["container_inventory"] = container
        return stable

    @staticmethod
    def _lifecycle_transition_basis_id(record: Any) -> str:
        """Hash exact lifecycle facts preserved by terminalization.

        The digest binds the owner and lease without serializing either into
        the recovery receipt.  Only fields normatively changed by
        ``finalize_exact_dead_owner`` are omitted.
        """

        payload = record.to_dict()
        basis = {
            field: payload[field]
            for field in payload
            if field
            not in {
                "state",
                "fence",
                "updated_at",
                "expires_at",
                "terminal_reason",
            }
        }
        return _sha256_bytes(_canonical_json(basis))

    @staticmethod
    def _validated_prior_authority(
        raw: Any,
        *,
        current_binding: Mapping[str, Any],
        prior_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Close and sanitize the database predecessor authority result."""

        if type(raw) is not dict or set(raw) != _PRIOR_AUTHORITY_FIELDS:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_database_authority_rejected"
            )
        authority = {field: raw[field] for field in sorted(_PRIOR_AUTHORITY_FIELDS)}
        string_fields = _PRIOR_AUTHORITY_FIELDS.difference(
            {
                "authorized",
                "current_attempt_number",
                "prior_attempt_number",
                "current_fencing_token",
                "prior_fencing_token",
                "current_control_expected_revision",
                "prior_control_expected_revision",
                "legacy_current_binding",
                "legacy_prior_binding",
                "mutation_authority",
                "completion_authority",
            }
        )
        if (
            any(type(authority[field]) is not str or not authority[field] for field in string_fields)
            or any(
                type(authority[field]) is not int or int(authority[field]) < 1
                for field in (
                    "current_attempt_number",
                    "prior_attempt_number",
                    "current_fencing_token",
                    "prior_fencing_token",
                    "current_control_expected_revision",
                    "prior_control_expected_revision",
                )
            )
            or authority["schema"] != CROSS_ATTEMPT_LIFECYCLE_AUTHORITY_SCHEMA
            or authority["authorized"] is not True
            or authority["mutation_authority"] is not False
            or authority["completion_authority"] is not False
            or type(authority["legacy_current_binding"]) is not bool
            or type(authority["legacy_prior_binding"]) is not bool
            or authority["current_attempt_id"] != current_binding["attempt_id"]
            or authority["prior_attempt_id"] != prior_binding["attempt_id"]
            or authority["task_cid"] != current_binding["task_cid"]
            or authority["task_alias"] != current_binding["task_alias"]
            or authority["current_binding_id"] != current_binding["binding_id"]
            or authority["prior_binding_id"] != prior_binding["binding_id"]
            or authority["current_fencing_token"]
            != current_binding["fencing_token"]
            or authority["prior_fencing_token"]
            != prior_binding["fencing_token"]
            or authority["prior_attempt_number"]
            >= authority["current_attempt_number"]
            or authority["prior_execution_status"]
            not in {"succeeded", "failed", "released", "expired"}
            or authority["prior_claim_state"]
            not in {"released", "expired", "superseded", "completed"}
            or authority["prior_coordination_status"]
            not in {"succeeded", "failed", "released", "expired"}
            or authority["current_control_expected_revision"]
            != current_binding["task_revision"]
            or authority["prior_control_expected_revision"]
            != prior_binding["task_revision"]
            or authority["legacy_current_binding"]
            is not (
                current_binding["schema"]
                == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1
            )
            or authority["legacy_prior_binding"]
            is not (
                prior_binding["schema"]
                == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_database_authority_rejected"
            )
        for prefix, selected in (
            ("current", current_binding),
            ("prior", prior_binding),
        ):
            if selected["schema"] != DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA:
                continue
            if (
                authority[f"{prefix}_control_binding_id"]
                != selected["control_binding_id"]
                or authority[f"{prefix}_control_task_projection_cid"]
                != selected["control_task_projection_cid"]
                or authority[f"{prefix}_control_expected_revision"]
                != selected["control_expected_revision"]
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_database_authority_rejected"
                )
        return authority

    @staticmethod
    def _recovery_id(receipt: Mapping[str, Any]) -> str:
        body = {
            field: receipt[field]
            for field in sorted(_RECOVERY_RECEIPT_FIELDS)
            if field not in _RECOVERY_ID_EXCLUDED_FIELDS
        }
        return _sha256_bytes(_canonical_json(body))

    @classmethod
    def _seal_recovery_receipt(
        cls,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = dict(body)
        receipt["recovery_id"] = cls._recovery_id(receipt)
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    @classmethod
    def _read_recovery_receipt(cls, path: Path) -> dict[str, Any]:
        try:
            receipt = json.loads(
                _stable_regular_utf8(
                    path,
                    noun="cross-attempt lifecycle recovery receipt",
                ),
                object_pairs_hook=_reject_duplicate_control_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {value}")
                ),
            )
        except DatabasePortalBridgeError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle recovery receipt is unreadable"
            ) from exc
        if type(receipt) is not dict or set(receipt) != _RECOVERY_RECEIPT_FIELDS:
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle recovery receipt is not closed"
            )
        receipt_id = receipt.get("receipt_id")
        receipt_body = dict(receipt)
        receipt_body.pop("receipt_id")
        if (
            type(receipt_id) is not str
            or receipt_id != _sha256_bytes(_canonical_json(receipt_body))
            or type(receipt.get("recovery_id")) is not str
            or receipt["recovery_id"] != cls._recovery_id(receipt)
        ):
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle recovery receipt identity is invalid"
            )
        return receipt

    @classmethod
    def _validated_recovery_receipt(
        cls,
        receipt: Mapping[str, Any],
        *,
        current_binding: Mapping[str, Any],
        prior_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = dict(receipt)
        authority = cls._validated_prior_authority(
            value.get("database_authority"),
            current_binding=current_binding,
            prior_binding=prior_binding,
        )
        portal = value.get("portal_state_binding")
        preservation = value.get("preservation")
        if (
            type(portal) is not dict
            or set(portal) != _PORTAL_STATE_BINDING_FIELDS
            or type(portal.get("attempt_directory_identity")) is not dict
            or set(portal["attempt_directory_identity"])
            != _ATTEMPT_DIRECTORY_IDENTITY_FIELDS
            or any(
                type(portal["attempt_directory_identity"][field]) is not int
                or int(portal["attempt_directory_identity"][field]) < 0
                for field in _ATTEMPT_DIRECTORY_IDENTITY_FIELDS
            )
            or type(portal.get("active_attempt")) is not int
            or int(portal["active_attempt"]) < 0
            or any(
                type(portal.get(field)) is not str or not portal[field]
                for field in _PORTAL_STATE_BINDING_FIELDS.difference(
                    {"active_attempt", "attempt_directory_identity"}
                )
            )
            or type(preservation) is not dict
            or set(preservation) != _PRESERVATION_FIELDS
            or any(
                type(preservation.get(field)) is not int
                or int(preservation[field]) < 0
                for field in (
                    "workspace_device",
                    "workspace_inode",
                    "workspace_mode",
                )
            )
            or any(
                type(preservation.get(field)) is not str
                or not preservation[field]
                for field in (
                    "workspace_path",
                    "branch",
                    "head",
                    "tree",
                    "preservation_mode",
                )
            )
            or type(preservation.get("process_inventory")) is not dict
            or type(preservation.get("container_inventory")) is not dict
        ):
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle recovery evidence is malformed"
            )
        process_inventory = preservation["process_inventory"]
        container_inventory = preservation["container_inventory"]
        if (
            set(process_inventory) != {"same_uid_processes_inspected"}
            or type(process_inventory["same_uid_processes_inspected"]) is not int
            or process_inventory["same_uid_processes_inspected"] < 0
            or set(container_inventory)
            not in (
                {"container_runtime", "containers_inspected"},
                {
                    "container_runtime",
                    "container_endpoint",
                    "isolation_labels",
                    "containers_inspected",
                },
            )
            or type(container_inventory.get("container_runtime")) is not str
            or not container_inventory["container_runtime"]
            or type(container_inventory.get("containers_inspected")) is not int
            or int(container_inventory["containers_inspected"]) < 0
        ):
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle recovery inventory is malformed"
            )
        if "isolation_labels" in container_inventory and (
            type(container_inventory.get("container_endpoint")) is not str
            or not container_inventory["container_endpoint"]
            or type(container_inventory["isolation_labels"]) is not list
            or not container_inventory["isolation_labels"]
            or any(
                type(label) is not str or not label
                for label in container_inventory["isolation_labels"]
            )
        ):
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle recovery inventory is malformed"
            )
        if value.get("prior_lifecycle_state") not in {
            "preparing",
            "active",
            "settling",
        }:
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle predecessor evidence is inconsistent"
            )
        integer_fields = (
            "current_attempt_number",
            "prior_attempt_number",
            "current_fencing_token",
            "prior_fencing_token",
            "prior_lifecycle_fence",
            "expected_terminal_lifecycle_fence",
        )
        string_fields = (
            "task_cid",
            "task_alias",
            "current_attempt_id",
            "prior_attempt_id",
            "current_binding_id",
            "prior_binding_id",
            "lifecycle_record_id",
            "lifecycle_authority_id",
            "lifecycle_transition_basis_id",
            "terminal_reason",
            "recovery_id",
            "receipt_id",
        )
        if (
            value.get("schema") != CROSS_ATTEMPT_LIFECYCLE_RECOVERY_SCHEMA
            or value.get("phase") not in {"prepared", "committed"}
            or any(
                type(value.get(field)) is not int or int(value[field]) < 1
                for field in integer_fields
            )
            or any(
                type(value.get(field)) is not str or not value[field]
                for field in string_fields
            )
            or type(value.get("terminal_lifecycle_authority_id")) is not str
            or (
                value["phase"] == "prepared"
                and value["terminal_lifecycle_authority_id"]
            )
            or (
                value["phase"] == "committed"
                and not value["terminal_lifecycle_authority_id"]
            )
            or value.get("worktree_deleted") is not False
            or value.get("provider_dispatched") is not False
            or value.get("task_completion_authority") is not False
            or value["task_cid"] != current_binding["task_cid"]
            or value["task_alias"] != current_binding["task_alias"]
            or value["current_attempt_id"] != current_binding["attempt_id"]
            or value["prior_attempt_id"] != prior_binding["attempt_id"]
            or value["current_binding_id"] != current_binding["binding_id"]
            or value["prior_binding_id"] != prior_binding["binding_id"]
            or value["current_fencing_token"]
            != current_binding["fencing_token"]
            or value["prior_fencing_token"] != prior_binding["fencing_token"]
            or value["current_attempt_number"]
            != authority["current_attempt_number"]
            or value["prior_attempt_number"]
            != authority["prior_attempt_number"]
            or value["expected_terminal_lifecycle_fence"]
            != value["prior_lifecycle_fence"] + 1
            or value["terminal_reason"]
            != "superseded_database_attempt_preserved"
        ):
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle recovery receipt is inconsistent"
            )
        return value

    @staticmethod
    def _lifecycle_expected(record: Any, *, fence: int | None = None) -> dict[str, Any]:
        return {
            "expected_record_id": record.record_id,
            "expected_fence": int(record.fence if fence is None else fence),
            "expected_lease_id": record.lease_id,
            "expected_task_id": record.task_id,
            "expected_canonical_task_cid": record.canonical_task_cid,
            "expected_attempt": int(record.attempt),
            "expected_branch": record.branch,
            "expected_merge_target": record.merge_target,
            "expected_repo_root": record.repo_root,
            "expected_state_dir": record.state_dir,
        }

    @classmethod
    def _verify_terminal_lifecycle(
        cls,
        lifecycle_store: Any,
        original: Any,
        *,
        terminal_reason: str,
    ) -> Any:
        try:
            terminal = lifecycle_store.require_exact_dead_owner(
                original.workspace_path,
                allow_terminal=True,
                **cls._lifecycle_expected(
                    original,
                    fence=int(original.fence) + 1,
                ),
            )
        except Exception as exc:
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle terminal evidence is invalid"
            ) from exc
        if (
            not terminal.is_terminal
            or terminal.record_id != original.record_id
            or terminal.owner != original.owner
            or terminal.lease_id != original.lease_id
            or int(terminal.fence) != int(original.fence) + 1
            or terminal.terminal_reason != terminal_reason
        ):
            raise DatabasePortalBridgeError(
                "cross-attempt lifecycle terminal evidence is inconsistent"
            )
        return terminal

    @staticmethod
    def _protected_recovery_artifacts(root: Path) -> tuple[dict[str, Any], ...]:
        """Read the bounded protected-recovery artifact set exactly once."""

        records: list[dict[str, Any]] = []
        try:
            entries = root.iterdir()
            for entry in entries:
                name = entry.name
                selected = (
                    name
                    in {
                        "implementation-protected-path-active.json",
                        "implementation-protected-path-incident.json",
                    }
                    or name.startswith(_PROTECTED_STATE_CLEARANCE_PREFIX)
                    or name.startswith(_PROTECTED_STATE_MARKER_BLOB_PREFIX)
                    or name.startswith("implementation-protected-path-retired-")
                    or name.startswith(_PROTECTED_STATE_RETIREMENT_PREFIX)
                    or name.startswith(_PROTECTED_STATE_ADOPTION_PREFIX)
                )
                if not selected:
                    continue
                if len(records) >= 32:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_protected_artifact_bound_exceeded"
                    )
                payload, _identity = _stable_regular_bytes(
                    entry,
                    noun="protected-state recovery artifact",
                )
                records.append(
                    {
                        "filename": name,
                        "sha256": _sha256_bytes(payload),
                        "size": len(payload),
                        "payload": payload,
                    }
                )
        except DatabasePortalBridgeDeferred:
            raise
        except OSError as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_protected_artifact_scan_failed"
            ) from exc
        return tuple(sorted(records, key=lambda item: item["filename"]))

    def _verify_completed_protected_history(
        self,
        *,
        recovery: Mapping[str, Any],
        record: Any,
        prior_paths: DatabasePortalAttemptPaths,
        abandoned_binding: Mapping[str, Any],
        prior_binding: Mapping[str, Any],
        artifacts: Sequence[Mapping[str, Any]],
    ) -> None:
        """Verify and ignore one exact closed historical marker recovery."""

        if (
            recovery.get("phase") != "committed"
            or not record.is_terminal
            or recovery.get("terminal_lifecycle_authority_id")
            != _sha256_bytes(_canonical_json(record.to_dict()))
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_completed_history_unproven"
            )
        if not artifacts:
            return
        by_prefix = {
            "active": [
                item
                for item in artifacts
                if item["filename"]
                == "implementation-protected-path-active.json"
            ],
            "incident": [
                item
                for item in artifacts
                if item["filename"]
                == "implementation-protected-path-incident.json"
            ],
            "prepared": [
                item
                for item in artifacts
                if item["filename"].startswith(_PROTECTED_STATE_CLEARANCE_PREFIX)
            ],
            "blob": [
                item
                for item in artifacts
                if item["filename"].startswith(_PROTECTED_STATE_MARKER_BLOB_PREFIX)
            ],
            "retired": [
                item
                for item in artifacts
                if item["filename"].startswith(
                    "implementation-protected-path-retired-"
                )
            ],
            "retirement": [
                item
                for item in artifacts
                if item["filename"].startswith(_PROTECTED_STATE_RETIREMENT_PREFIX)
            ],
            "adoption": [
                item
                for item in artifacts
                if item["filename"].startswith(_PROTECTED_STATE_ADOPTION_PREFIX)
            ],
        }
        if (
            by_prefix["active"]
            or by_prefix["incident"]
            or any(
                len(by_prefix[name]) != 1
                for name in ("prepared", "blob", "retired", "retirement")
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_completed_history_ambiguous"
            )
        prepared_item = by_prefix["prepared"][0]
        blob_item = by_prefix["blob"][0]
        retired_item = by_prefix["retired"][0]
        retirement_item = by_prefix["retirement"][0]
        prepared = self._strict_protected_recovery_receipt(
            prior_paths.root / prepared_item["filename"],
            fields=_PREPARED_PROTECTED_CLEARANCE_FIELDS,
            schema=CROSS_ATTEMPT_PROTECTED_STATE_CLEARANCE_SCHEMA,
            noun="completed protected-state clearance receipt",
        )
        retirement = self._strict_protected_recovery_receipt(
            prior_paths.root / retirement_item["filename"],
            fields=_PROTECTED_RETIREMENT_FIELDS,
            schema=CROSS_ATTEMPT_PROTECTED_STATE_RETIREMENT_SCHEMA,
            noun="completed protected-state retirement receipt",
        )
        clearance_receipt_id = prepared.get("receipt_id")
        clearance_body = dict(prepared)
        clearance_body.pop("receipt_id", None)
        clearance_id = clearance_body.pop("clearance_id", None)
        retirement_receipt_id = retirement.get("receipt_id")
        retirement_body = dict(retirement)
        retirement_body.pop("receipt_id", None)
        retirement_id = retirement_body.pop("retirement_id", None)
        marker_raw = bytes(blob_item["payload"])
        try:
            marker = json.loads(
                marker_raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_control_keys,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {constant}")
                ),
            )
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_completed_marker_malformed"
            ) from exc
        marker_digest = hashlib.sha256(marker_raw).hexdigest()
        stable_recovery_preservation = self._stable_preservation_authority(
            recovery["preservation"]
        )
        if (
            type(marker) is not dict
            or set(marker) != _ACTIVE_PROTECTED_STATE_FIELDS
            or bytes(retired_item["payload"]) != marker_raw
            or prepared.get("active_marker") != marker
            or prepared.get("active_marker_sha256") != f"sha256:{marker_digest}"
            or prepared.get("active_marker_blob_filename")
            != blob_item["filename"]
            or prepared.get("current_attempt_id")
            != abandoned_binding["attempt_id"]
            or prepared.get("prior_attempt_id") != prior_binding["attempt_id"]
            or prepared.get("current_binding_id")
            != abandoned_binding["binding_id"]
            or prepared.get("prior_binding_id") != prior_binding["binding_id"]
            or prepared.get("lifecycle_record_id")
            != recovery["lifecycle_record_id"]
            or prepared.get("lifecycle_authority_id")
            != recovery["lifecycle_authority_id"]
            or prepared.get("lifecycle_transition_basis_id")
            != recovery["lifecycle_transition_basis_id"]
            or prepared.get("prior_lifecycle_state")
            != recovery["prior_lifecycle_state"]
            or prepared.get("prior_lifecycle_fence")
            != recovery["prior_lifecycle_fence"]
            or prepared.get("expected_terminal_lifecycle_fence")
            != recovery["expected_terminal_lifecycle_fence"]
            or prepared.get("terminal_reason") != recovery["terminal_reason"]
            or prepared.get("database_authority_id")
            != content_identity(recovery["database_authority"])
            or prepared.get("portal_state_binding_id")
            != content_identity(recovery["portal_state_binding"])
            or type(prepared.get("preservation")) is not dict
            or self._stable_preservation_authority(prepared["preservation"])
            != stable_recovery_preservation
            or clearance_id != _sha256_bytes(_canonical_json(clearance_body))
            or clearance_receipt_id
            != _sha256_bytes(
                _canonical_json({**clearance_body, "clearance_id": clearance_id})
            )
            or prepared_item["filename"]
            != f"{_PROTECTED_STATE_CLEARANCE_PREFIX}{str(clearance_id).removeprefix('sha256:')[:24]}.json"
            or retired_item["filename"]
            != "implementation-protected-path-retired-"
            f"{str(clearance_id).removeprefix('sha256:')[:24]}.json"
            or retirement.get("prepared_clearance") != prepared
            or retirement.get("prepared_clearance_filename")
            != prepared_item["filename"]
            or retirement.get("retired_marker_filename")
            != retired_item["filename"]
            or retirement.get("active_marker_sha256")
            != prepared["active_marker_sha256"]
            or retirement_id != _sha256_bytes(_canonical_json(retirement_body))
            or retirement_receipt_id
            != _sha256_bytes(
                _canonical_json({**retirement_body, "retirement_id": retirement_id})
            )
            or any(
                prepared.get(field) is not expected
                for field, expected in {
                    "active_marker_retired": False,
                    "worktree_deleted": False,
                    "provider_dispatched": False,
                    "mutation_authority": False,
                    "merge_authority": False,
                    "task_completion_authority": False,
                    "worker_self_approval": False,
                    "normal_validation_required": True,
                }.items()
            )
            or any(
                retirement.get(field) is not expected
                for field, expected in {
                    "active_marker_retired": True,
                    "worktree_deleted": False,
                    "provider_dispatched": False,
                    "mutation_authority": False,
                    "merge_authority": False,
                    "task_completion_authority": False,
                    "worker_self_approval": False,
                    "normal_validation_required": True,
                }.items()
            )
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_completed_history_changed"
            )
        for item in by_prefix["adoption"]:
            adoption = self._strict_protected_recovery_receipt(
                prior_paths.root / item["filename"],
                fields=_PROTECTED_ADOPTION_FIELDS,
                schema=CROSS_ATTEMPT_PROTECTED_STATE_ADOPTION_SCHEMA,
                noun="completed protected-state adoption receipt",
            )
            adoption_receipt_id = adoption.get("receipt_id")
            adoption_body = dict(adoption)
            adoption_body.pop("receipt_id", None)
            adoption_id = adoption_body.pop("adoption_id", None)
            if (
                adoption.get("lifecycle_recovery_id") != recovery["recovery_id"]
                or adoption.get("abandoned_attempt_id")
                != abandoned_binding["attempt_id"]
                or adoption.get("abandoned_binding_id")
                != abandoned_binding["binding_id"]
                or adoption.get("prior_attempt_id") != prior_binding["attempt_id"]
                or adoption.get("prior_binding_id") != prior_binding["binding_id"]
                or adoption_id != _sha256_bytes(_canonical_json(adoption_body))
                or adoption_receipt_id
                != _sha256_bytes(
                    _canonical_json({**adoption_body, "adoption_id": adoption_id})
                )
                or any(
                    adoption.get(field) is not expected
                    for field, expected in {
                        "provider_dispatched": False,
                        "mutation_authority": False,
                        "merge_authority": False,
                        "task_completion_authority": False,
                        "worker_self_approval": False,
                        "normal_validation_required": True,
                    }.items()
                )
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_completed_adoption_changed"
                )

    def _resume_abandoned_recovery_transaction(
        self,
        *,
        attempt: Any,
        binding: Mapping[str, Any],
        daemon: Any,
        candidates: Sequence[tuple[DatabasePortalAttemptPaths, dict[str, Any]]],
        records: Sequence[Any],
    ) -> dict[str, Any] | None:
        """Adopt and finish one exact DB-authorized prior recovery transaction."""

        current_paths = self._paths(attempt)
        current_directory_identity = self._seal_attempt_directory(
            current_paths,
            attempt_id=binding["attempt_id"],
            create=False,
        )
        if (
            self._strict_binding(current_paths.binding) != dict(binding)
            or not self._verify_projection(current_paths, binding)
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_current_binding_changed"
            )
        by_binding = {candidate[1]["binding_id"]: candidate for candidate in candidates}
        if len(by_binding) != len(candidates):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_prior_binding_ambiguous"
            )
        transactions: list[
            tuple[
                DatabasePortalAttemptPaths,
                dict[str, Any],
                DatabasePortalAttemptPaths,
                dict[str, Any],
                dict[str, Any],
                Any,
                dict[str, int],
                dict[str, int],
                str,
                tuple[dict[str, Any], ...],
            ]
        ] = []
        for abandoned_paths, abandoned_binding in candidates:
            recovery_path = (
                abandoned_paths.root / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
            )
            if not os.path.lexists(recovery_path):
                continue
            recovery_raw, _recovery_identity = _stable_regular_bytes(
                recovery_path,
                noun="abandoned lifecycle recovery receipt",
            )
            recovery = self._read_recovery_receipt(recovery_path)
            if (
                recovery.get("current_binding_id")
                != abandoned_binding["binding_id"]
                or recovery.get("current_attempt_id")
                != abandoned_binding["attempt_id"]
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_abandoned_recovery_binding_mismatch"
                )
            prior_candidate = by_binding.get(recovery.get("prior_binding_id"))
            if prior_candidate is None:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_abandoned_recovery_prior_missing"
                )
            prior_paths, prior_binding = prior_candidate
            recovery = self._validated_recovery_receipt(
                recovery,
                current_binding=abandoned_binding,
                prior_binding=prior_binding,
            )
            matching_records = [
                record
                for record in records
                if record.record_id == recovery["lifecycle_record_id"]
                and str(Path(record.state_dir).resolve(strict=False))
                == str(prior_paths.root.resolve(strict=True))
                and self._lifecycle_transition_basis_id(record)
                == recovery["lifecycle_transition_basis_id"]
                and (
                    (
                        record.is_terminal
                        and int(record.fence)
                        == int(recovery["expected_terminal_lifecycle_fence"])
                        and record.terminal_reason == recovery["terminal_reason"]
                    )
                    or (
                        record.is_nonterminal
                        and int(record.fence)
                        == int(recovery["prior_lifecycle_fence"])
                        and record.state.value == recovery["prior_lifecycle_state"]
                        and _sha256_bytes(_canonical_json(record.to_dict()))
                        == recovery["lifecycle_authority_id"]
                    )
                )
            ]
            if len(matching_records) != 1:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_abandoned_transaction_unproven"
                )
            abandoned_directory_identity = self._seal_attempt_directory(
                abandoned_paths,
                attempt_id=abandoned_binding["attempt_id"],
                create=False,
            )
            prior_directory_identity = self._seal_attempt_directory(
                prior_paths,
                attempt_id=prior_binding["attempt_id"],
                create=False,
            )
            artifacts = self._protected_recovery_artifacts(prior_paths.root)
            if recovery["phase"] == "committed":
                self._verify_completed_protected_history(
                    recovery=recovery,
                    record=matching_records[0],
                    prior_paths=prior_paths,
                    abandoned_binding=abandoned_binding,
                    prior_binding=prior_binding,
                    artifacts=artifacts,
                )
                continue
            transactions.append(
                (
                    abandoned_paths,
                    abandoned_binding,
                    prior_paths,
                    prior_binding,
                    recovery,
                    matching_records[0],
                    abandoned_directory_identity,
                    prior_directory_identity,
                    _sha256_bytes(recovery_raw),
                    artifacts,
                )
            )
        if not transactions:
            return None
        if len(transactions) != 1:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_abandoned_recovery_ambiguous"
            )
        (
            abandoned_paths,
            abandoned_binding,
            prior_paths,
            prior_binding,
            recovery,
            selected_record,
            abandoned_directory_identity,
            prior_directory_identity,
            recovery_digest,
            selected_artifacts,
        ) = transactions[0]
        successor_authority = self._validated_prior_authority(
            self.prior_attempt_authority(
                attempt,
                binding,
                abandoned_binding,
            ),
            current_binding=binding,
            prior_binding=abandoned_binding,
        )
        if (
            int(successor_authority["current_attempt_number"])
            <= int(recovery["current_attempt_number"])
            or int(successor_authority["current_fencing_token"])
            <= int(recovery["current_fencing_token"])
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_successor_adoption_not_newer"
            )
        acquire = getattr(daemon, "_acquire_checkout_mutation_lease", None)
        release = getattr(daemon, "_release_checkout_mutation_lease", None)
        if not callable(acquire) or not callable(release):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_checkout_lock_unavailable"
            )
        checkout_lease: Any | None = None
        try:
            checkout_lease, outcome, _owner, _waited = acquire(
                task_id=selected_record.task_id,
                attempt=int(selected_record.attempt),
                branch=selected_record.branch,
                operation="cross_attempt_successor_adoption",
                timeout_seconds=0.0,
            )
            if checkout_lease is None or outcome != "acquired":
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_checkout_lock_busy"
                )
            if (
                self._seal_attempt_directory(
                    current_paths,
                    attempt_id=binding["attempt_id"],
                    create=False,
                )
                != current_directory_identity
                or self._seal_attempt_directory(
                    abandoned_paths,
                    attempt_id=abandoned_binding["attempt_id"],
                    create=False,
                )
                != abandoned_directory_identity
                or self._seal_attempt_directory(
                    prior_paths,
                    attempt_id=prior_binding["attempt_id"],
                    create=False,
                )
                != prior_directory_identity
                or self._strict_binding(current_paths.binding) != dict(binding)
                or self._strict_binding(abandoned_paths.binding)
                != abandoned_binding
                or self._strict_binding(prior_paths.binding) != prior_binding
                or not self._verify_projection(current_paths, binding)
                or not self._verify_projection(abandoned_paths, abandoned_binding)
                or not self._verify_projection(prior_paths, prior_binding)
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_adoption_binding_changed"
                )
            locked_recovery_raw, _locked_recovery_identity = _stable_regular_bytes(
                abandoned_paths.root
                / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME,
                noun="locked abandoned lifecycle recovery receipt",
            )
            locked_recovery = self._validated_recovery_receipt(
                self._read_recovery_receipt(
                    abandoned_paths.root
                    / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
                ),
                current_binding=abandoned_binding,
                prior_binding=prior_binding,
            )
            locked_successor_authority = self._validated_prior_authority(
                self.prior_attempt_authority(
                    attempt,
                    binding,
                    abandoned_binding,
                ),
                current_binding=binding,
                prior_binding=abandoned_binding,
            )
            locked_artifacts = self._protected_recovery_artifacts(
                prior_paths.root
            )
            if (
                _sha256_bytes(locked_recovery_raw) != recovery_digest
                or locked_recovery != recovery
                or locked_successor_authority != successor_authority
                or locked_artifacts != selected_artifacts
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_adoption_evidence_changed"
                )
            try:
                exact_record = daemon.worktree_lifecycle.require_exact_dead_owner(
                    selected_record.workspace_path,
                    allow_terminal=bool(selected_record.is_terminal),
                    **self._lifecycle_expected(selected_record),
                )
            except Exception as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_abandoned_record_changed"
                ) from exc
            if exact_record != selected_record:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_abandoned_record_changed"
                )
            portal_state = self._prior_portal_state_binding(
                daemon,
                prior_paths,
                prior_binding,
                exact_record,
            )
            portal_state["attempt_directory_identity"] = prior_directory_identity
            if portal_state != recovery["portal_state_binding"]:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_abandoned_portal_state_changed"
                )
            artifact_records = [
                {
                    "filename": item["filename"],
                    "sha256": item["sha256"],
                    "size": item["size"],
                }
                for item in locked_artifacts
            ]
            adoption_body = {
                "schema": CROSS_ATTEMPT_PROTECTED_STATE_ADOPTION_SCHEMA,
                "task_id": exact_record.task_id,
                "successor_attempt_id": binding["attempt_id"],
                "successor_binding_id": binding["binding_id"],
                "abandoned_attempt_id": abandoned_binding["attempt_id"],
                "abandoned_binding_id": abandoned_binding["binding_id"],
                "prior_attempt_id": prior_binding["attempt_id"],
                "prior_binding_id": prior_binding["binding_id"],
                "lifecycle_recovery_id": recovery["recovery_id"],
                "lifecycle_recovery_receipt_id": recovery["receipt_id"],
                "lifecycle_recovery_phase": recovery["phase"],
                "observed_lifecycle_state": exact_record.state.value,
                "observed_lifecycle_fence": int(exact_record.fence),
                "observed_lifecycle_authority_id": _sha256_bytes(
                    _canonical_json(exact_record.to_dict())
                ),
                "successor_database_authority": successor_authority,
                "protected_artifacts": sorted(
                    artifact_records,
                    key=lambda item: item["filename"],
                ),
                "adoption_operation": "finish_exact_prepared_recovery_only",
                "provider_dispatched": False,
                "mutation_authority": False,
                "merge_authority": False,
                "task_completion_authority": False,
                "worker_self_approval": False,
                "normal_validation_required": True,
            }
            adoption_id = _sha256_bytes(_canonical_json(adoption_body))
            adoption_receipt = {**adoption_body, "adoption_id": adoption_id}
            adoption_receipt["receipt_id"] = _sha256_bytes(
                _canonical_json(adoption_receipt)
            )
            adoption_filename = (
                f"{_PROTECTED_STATE_ADOPTION_PREFIX}"
                f"{adoption_id.removeprefix('sha256:')[:24]}.json"
            )
            _publish_immutable_file(
                prior_paths.root / adoption_filename,
                json.dumps(
                    adoption_receipt,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n",
                sealed_directory_identity=prior_directory_identity,
            )
            observed_adoption = self._strict_protected_recovery_receipt(
                prior_paths.root / adoption_filename,
                fields=_PROTECTED_ADOPTION_FIELDS,
                schema=CROSS_ATTEMPT_PROTECTED_STATE_ADOPTION_SCHEMA,
                noun="protected-state successor adoption receipt",
            )
            if observed_adoption != adoption_receipt:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_successor_adoption_changed"
                )
            successor_adoption = {
                "current_binding": dict(binding),
                "database_authority": successor_authority,
            }
            if exact_record.is_nonterminal:
                preterminal_preservation = self._preserved_quiescent_worktree(
                    daemon,
                    exact_record,
                    prior_paths,
                    attempt=attempt,
                    current_binding=abandoned_binding,
                    prior_binding=prior_binding,
                    database_authority=recovery["database_authority"],
                    portal_state_binding=portal_state,
                    lifecycle_recovery_receipt=recovery,
                    perform_marker_retirement=False,
                    publish_marker_clearance=True,
                    successor_adoption=successor_adoption,
                )
                if self._stable_preservation_authority(
                    preterminal_preservation
                ) != self._stable_preservation_authority(
                    recovery["preservation"]
                ):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_adoption_preservation_changed"
                    )
                repeated_authority = self._validated_prior_authority(
                    self.prior_attempt_authority(
                        attempt,
                        binding,
                        abandoned_binding,
                    ),
                    current_binding=binding,
                    prior_binding=abandoned_binding,
                )
                if repeated_authority != successor_authority:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_successor_authority_changed"
                    )
                try:
                    exact_dead = daemon.worktree_lifecycle.require_exact_dead_owner(
                        exact_record.workspace_path,
                        **self._lifecycle_expected(exact_record),
                    )
                    if exact_dead != exact_record:
                        raise DatabasePortalBridgeDeferred(
                            "cross_attempt_lifecycle_abandoned_record_changed"
                        )
                    daemon.worktree_lifecycle.finalize_exact_dead_owner(
                        exact_record.workspace_path,
                        expected_owner=exact_record.owner,
                        reason=recovery["terminal_reason"],
                        retain_terminal=True,
                        **self._lifecycle_expected(exact_record),
                    )
                except DatabasePortalBridgeDeferred:
                    raise
                except Exception as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_adoption_finalize_race"
                    ) from exc
                terminal = self._verify_terminal_lifecycle(
                    daemon.worktree_lifecycle,
                    exact_record,
                    terminal_reason=recovery["terminal_reason"],
                )
            else:
                terminal = exact_record
            preservation = self._preserved_quiescent_worktree(
                daemon,
                terminal,
                prior_paths,
                attempt=attempt,
                current_binding=abandoned_binding,
                prior_binding=prior_binding,
                database_authority=recovery["database_authority"],
                portal_state_binding=portal_state,
                lifecycle_recovery_receipt=recovery,
                perform_marker_retirement=True,
                publish_marker_clearance=True,
                successor_adoption=successor_adoption,
            )
            final_preservation = self._preserved_quiescent_worktree(
                daemon,
                terminal,
                prior_paths,
                attempt=attempt,
                current_binding=abandoned_binding,
                prior_binding=prior_binding,
                database_authority=recovery["database_authority"],
                portal_state_binding=portal_state,
                lifecycle_recovery_receipt=recovery,
                perform_marker_retirement=True,
                publish_marker_clearance=True,
                successor_adoption=successor_adoption,
            )
            if self._stable_preservation_authority(
                final_preservation
            ) != self._stable_preservation_authority(
                recovery["preservation"]
            ) or self._stable_preservation_authority(
                preservation
            ) != self._stable_preservation_authority(
                recovery["preservation"]
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_successor_adoption_changed"
                )
            repeated_authority = self._validated_prior_authority(
                self.prior_attempt_authority(
                    attempt,
                    binding,
                    abandoned_binding,
                ),
                current_binding=binding,
                prior_binding=abandoned_binding,
            )
            if repeated_authority != successor_authority:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_successor_authority_changed"
                )
            final_recovery_raw, _final_recovery_identity = _stable_regular_bytes(
                abandoned_paths.root
                / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME,
                noun="abandoned lifecycle recovery receipt before commit",
            )
            if (
                _sha256_bytes(final_recovery_raw) != recovery_digest
                or self._validated_recovery_receipt(
                    self._read_recovery_receipt(
                        abandoned_paths.root
                        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
                    ),
                    current_binding=abandoned_binding,
                    prior_binding=prior_binding,
                )
                != recovery
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_abandoned_recovery_changed"
                )
            committed_body = {
                field: recovery[field]
                for field in _RECOVERY_RECEIPT_FIELDS.difference(
                    {"recovery_id", "receipt_id"}
                )
            }
            committed_body["phase"] = "committed"
            committed_body["terminal_lifecycle_authority_id"] = _sha256_bytes(
                _canonical_json(terminal.to_dict())
            )
            committed = self._seal_recovery_receipt(committed_body)
            if committed["recovery_id"] != recovery["recovery_id"]:
                raise DatabasePortalBridgeError(
                    "cross-attempt lifecycle adoption changed recovery identity"
                )
            _atomic_write(
                abandoned_paths.root
                / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME,
                json.dumps(
                    committed,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n",
                sealed_directory_identity=abandoned_directory_identity,
            )
            observed_committed = self._validated_recovery_receipt(
                self._read_recovery_receipt(
                    abandoned_paths.root
                    / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
                ),
                current_binding=abandoned_binding,
                prior_binding=prior_binding,
            )
            if observed_committed != committed:
                raise DatabasePortalBridgeError(
                    "cross-attempt lifecycle adopted receipt was not committed"
                )
            self._verify_completed_protected_history(
                recovery=committed,
                record=terminal,
                prior_paths=prior_paths,
                abandoned_binding=abandoned_binding,
                prior_binding=prior_binding,
                artifacts=self._protected_recovery_artifacts(prior_paths.root),
            )
        finally:
            if checkout_lease is not None:
                try:
                    released = bool(release(checkout_lease))
                except Exception as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_checkout_lock_release_failed"
                    ) from exc
                if not released:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_checkout_lock_release_failed"
                    )
        return {
            "attempted": True,
            "recovered": True,
            "adopted": True,
            "adoption_id": adoption_id,
            "adoption_receipt_path": str(prior_paths.root / adoption_filename),
        }

    def _recover_superseded_attempt_lifecycle(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        daemon: Any,
    ) -> dict[str, Any]:
        """Retire one exact dead prior-attempt fence without deleting its worktree."""

        if self.prior_attempt_authority is None:
            return {"attempted": False, "reason": "authority_unavailable"}
        try:
            current_directory_identity = self._seal_attempt_directory(
                paths,
                attempt_id=binding["attempt_id"],
                create=False,
            )
            if self._strict_binding(paths.binding) != dict(binding):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_current_binding_changed"
                )
            self._verify_projection(paths, binding)
        except DatabasePortalBridgeDeferred:
            raise
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_current_binding_changed"
            ) from exc
        candidates = self._prior_attempt_bindings(
            current_paths=paths,
            current_binding=binding,
        )
        lifecycle_store = getattr(daemon, "worktree_lifecycle", None)
        if lifecycle_store is None or not callable(
            getattr(lifecycle_store, "iter_records", None)
        ):
            if candidates:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_store_unavailable"
                )
            return {"attempted": False, "reason": "no_prior_attempt"}

        try:
            records = tuple(lifecycle_store.iter_records())
            lexical_attempt_root = Path(
                os.path.abspath(os.fspath(self.attempt_root))
            )
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_store_unavailable"
            ) from exc
        candidate_roots = {
            str(prior_paths.root.resolve(strict=True))
            for prior_paths, _prior_binding in candidates
        }
        adopted = self._resume_abandoned_recovery_transaction(
            attempt=attempt,
            binding=binding,
            daemon=daemon,
            candidates=candidates,
            records=records,
        )
        if adopted is not None:
            return adopted
        # A malformed sibling joined to a live lifecycle record is relevant
        # authority, not ignorable junk.  It must block rather than disappear
        # from candidate discovery.
        for lifecycle_record in records:
            if (
                not lifecycle_record.is_nonterminal
                or lifecycle_record.task_id != binding["task_alias"]
            ):
                continue
            lexical_state_dir = Path(
                os.path.abspath(str(lifecycle_record.state_dir or ""))
            )
            if (
                lexical_state_dir.parent == lexical_attempt_root
                and _ATTEMPT_DIRECTORY.fullmatch(lexical_state_dir.name)
                and lexical_state_dir != paths.root
            ):
                try:
                    resolved_state_dir = str(
                        lexical_state_dir.resolve(strict=True)
                    )
                except OSError as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_relevant_binding_invalid"
                    ) from exc
                if resolved_state_dir not in candidate_roots:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_relevant_binding_invalid"
                    )

        receipt_path = paths.root / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
        existing_receipt: dict[str, Any] | None = None
        receipt_candidate: tuple[
            DatabasePortalAttemptPaths,
            dict[str, Any],
        ] | None = None
        if os.path.lexists(receipt_path):
            existing_receipt = self._read_recovery_receipt(receipt_path)
            bound_candidates = [
                candidate
                for candidate in candidates
                if candidate[1]["binding_id"]
                == existing_receipt.get("prior_binding_id")
                and candidate[1]["attempt_id"]
                == existing_receipt.get("prior_attempt_id")
            ]
            if len(bound_candidates) != 1:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_recovery_receipt_unbound"
                )
            receipt_candidate = bound_candidates[0]
            existing_receipt = self._validated_recovery_receipt(
                existing_receipt,
                current_binding=binding,
                prior_binding=receipt_candidate[1],
            )

        matches: list[tuple[DatabasePortalAttemptPaths, dict[str, Any], Any]] = []
        selected_candidates = (
            [receipt_candidate] if receipt_candidate is not None else candidates
        )
        for prior_paths, prior_binding in selected_candidates:
            try:
                prior_state_dir = str(prior_paths.root.resolve(strict=True))
            except OSError as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_relevant_binding_invalid"
                ) from exc
            bound = [
                lifecycle_record
                for lifecycle_record in records
                if lifecycle_record.task_id == prior_binding["task_alias"]
                and str(Path(lifecycle_record.state_dir).resolve(strict=False))
                == prior_state_dir
                and (
                    (
                        existing_receipt is None
                        and lifecycle_record.is_nonterminal
                    )
                    or (
                        existing_receipt is not None
                        and lifecycle_record.record_id
                        == existing_receipt["lifecycle_record_id"]
                        and (
                            (
                                lifecycle_record.is_nonterminal
                                and int(lifecycle_record.fence)
                                == int(existing_receipt["prior_lifecycle_fence"])
                            )
                            or (
                                lifecycle_record.is_terminal
                                and int(lifecycle_record.fence)
                                == int(
                                    existing_receipt[
                                        "expected_terminal_lifecycle_fence"
                                    ]
                                )
                                and lifecycle_record.terminal_reason
                                == existing_receipt["terminal_reason"]
                            )
                        )
                    )
                )
            ]
            if len(bound) > 1:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_authority_ambiguous"
                )
            if bound:
                matches.append((prior_paths, prior_binding, bound[0]))
        if not matches:
            if existing_receipt is not None:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_terminal_evidence_missing"
                )
            return {"attempted": False, "reason": "no_prior_lifecycle"}
        if len(matches) != 1:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_authority_ambiguous"
            )
        prior_paths, prior_binding, record = matches[0]
        terminal_reason = _SUPERSEDED_LIFECYCLE_TERMINAL_REASON
        if record.is_terminal and existing_receipt is None:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_terminal_evidence_unbound"
            )

        acquire = getattr(daemon, "_acquire_checkout_mutation_lease", None)
        release = getattr(daemon, "_release_checkout_mutation_lease", None)
        if not callable(acquire) or not callable(release):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_checkout_lock_unavailable"
            )
        try:
            lease_result = acquire(
                task_id=record.task_id,
                attempt=int(record.attempt),
                branch=record.branch,
                operation="cross_attempt_lifecycle_recovery",
                timeout_seconds=0.0,
                extra={
                    "current_binding_id": binding["binding_id"],
                    "prior_binding_id": prior_binding["binding_id"],
                },
                preserve_existing=True,
            )
        except Exception as exc:
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_checkout_lock_unavailable"
            ) from exc
        if (
            not isinstance(lease_result, tuple)
            or len(lease_result) != 4
            or lease_result[0] is None
        ):
            raise DatabasePortalBridgeDeferred(
                "cross_attempt_lifecycle_checkout_lock_contended"
            )
        checkout_lease = lease_result[0]
        result: dict[str, Any] | None = None
        try:
            try:
                prior_directory_identity = self._seal_attempt_directory(
                    prior_paths,
                    attempt_id=prior_binding["attempt_id"],
                    create=False,
                )
                if self._strict_binding(prior_paths.binding) != prior_binding:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_prior_binding_changed"
                    )
                if self._strict_binding(paths.binding) != dict(binding):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_current_binding_changed"
                    )
                portal_state_binding = self._prior_portal_state_binding(
                    daemon,
                    prior_paths,
                    prior_binding,
                    record,
                )
                portal_state_binding["attempt_directory_identity"] = (
                    prior_directory_identity
                )
                authority = self._validated_prior_authority(
                    self.prior_attempt_authority(
                        attempt,
                        binding,
                        prior_binding,
                    ),
                    current_binding=binding,
                    prior_binding=prior_binding,
                )
            except DatabasePortalBridgeDeferred:
                raise
            except Exception as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_database_authority_rejected"
                ) from exc

            if record.is_terminal:
                assert existing_receipt is not None
                prior_fence = int(existing_receipt["prior_lifecycle_fence"])
                if (
                    record.record_id != existing_receipt["lifecycle_record_id"]
                    or int(record.fence) != prior_fence + 1
                    or record.terminal_reason != terminal_reason
                ):
                    raise DatabasePortalBridgeError(
                        "cross-attempt lifecycle terminal evidence is inconsistent"
                    )
                # A partially finalized lifecycle can expose the successor record
                # before its task-index projection is durably replaced.  Authenticate
                # that terminal successor against the prepared receipt before the
                # repair helper is allowed to mutate the stale index.
                if (
                    self._lifecycle_transition_basis_id(record)
                    != existing_receipt["lifecycle_transition_basis_id"]
                ):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_receipt_authority_mismatch"
                    )
                try:
                    verified = lifecycle_store.require_exact_dead_owner(
                        record.workspace_path,
                        allow_terminal=True,
                        **self._lifecycle_expected(record),
                    )
                except Exception as exact_error:
                    repair = getattr(
                        lifecycle_store,
                        "repair_partial_finalize",
                        None,
                    )
                    if (
                        existing_receipt["phase"] != "prepared"
                        or not callable(repair)
                    ):
                        raise DatabasePortalBridgeDeferred(
                            "cross_attempt_lifecycle_exact_dead_owner_unproven"
                        ) from exact_error
                    try:
                        repaired = repair(
                            record.workspace_path,
                            expected_terminal=record,
                            expected_preterminal_state=existing_receipt[
                                "prior_lifecycle_state"
                            ],
                        )
                        verified = lifecycle_store.require_exact_dead_owner(
                            record.workspace_path,
                            allow_terminal=True,
                            **self._lifecycle_expected(record),
                        )
                    except Exception as repair_error:
                        raise DatabasePortalBridgeDeferred(
                            "cross_attempt_lifecycle_partial_finalize_unproven"
                        ) from repair_error
                    if repaired != record or verified != record:
                        raise DatabasePortalBridgeDeferred(
                            "cross_attempt_lifecycle_partial_finalize_changed"
                        )
            else:
                prior_fence = int(record.fence)
                try:
                    verified = lifecycle_store.require_exact_dead_owner(
                        record.workspace_path,
                        **self._lifecycle_expected(record),
                    )
                except Exception as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_exact_dead_owner_unproven"
                    ) from exc
            if verified != record:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_record_changed"
                )
            if existing_receipt is not None and (
                self._lifecycle_transition_basis_id(verified)
                != existing_receipt["lifecycle_transition_basis_id"]
                or (
                    verified.is_nonterminal
                    and (
                        verified.state.value
                        != existing_receipt["prior_lifecycle_state"]
                        or _sha256_bytes(
                            _canonical_json(verified.to_dict())
                        )
                        != existing_receipt["lifecycle_authority_id"]
                    )
                )
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_receipt_authority_mismatch"
                )
            preservation = self._preserved_quiescent_worktree(
                daemon,
                verified,
                prior_paths,
                attempt=attempt,
                current_binding=binding,
                prior_binding=prior_binding,
                database_authority=authority,
                portal_state_binding=portal_state_binding,
                lifecycle_recovery_receipt=existing_receipt,
                perform_marker_retirement=False,
                publish_marker_clearance=False,
            )
            receipt_preservation = preservation
            if existing_receipt is not None:
                receipt_preservation = existing_receipt["preservation"]
                if self._stable_preservation_authority(
                    preservation
                ) != self._stable_preservation_authority(receipt_preservation):
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_preservation_changed"
                    )
            lifecycle_authority_id = (
                existing_receipt["lifecycle_authority_id"]
                if record.is_terminal and existing_receipt is not None
                else _sha256_bytes(_canonical_json(verified.to_dict()))
            )
            lifecycle_transition_basis_id = (
                existing_receipt["lifecycle_transition_basis_id"]
                if existing_receipt is not None
                else self._lifecycle_transition_basis_id(verified)
            )
            prior_lifecycle_state = (
                existing_receipt["prior_lifecycle_state"]
                if existing_receipt is not None
                else verified.state.value
            )
            prepared = self._seal_recovery_receipt(
                {
                    "schema": CROSS_ATTEMPT_LIFECYCLE_RECOVERY_SCHEMA,
                    "phase": "prepared",
                    "task_cid": binding["task_cid"],
                    "task_alias": binding["task_alias"],
                    "current_attempt_id": binding["attempt_id"],
                    "prior_attempt_id": prior_binding["attempt_id"],
                    "current_attempt_number": authority[
                        "current_attempt_number"
                    ],
                    "prior_attempt_number": authority["prior_attempt_number"],
                    "current_binding_id": binding["binding_id"],
                    "prior_binding_id": prior_binding["binding_id"],
                    "current_fencing_token": binding["fencing_token"],
                    "prior_fencing_token": prior_binding["fencing_token"],
                    "lifecycle_record_id": record.record_id,
                    "lifecycle_authority_id": lifecycle_authority_id,
                    "lifecycle_transition_basis_id": (
                        lifecycle_transition_basis_id
                    ),
                    "prior_lifecycle_state": prior_lifecycle_state,
                    "prior_lifecycle_fence": prior_fence,
                    "expected_terminal_lifecycle_fence": prior_fence + 1,
                    "terminal_lifecycle_authority_id": "",
                    "terminal_reason": terminal_reason,
                    "database_authority": authority,
                    "portal_state_binding": portal_state_binding,
                    "preservation": receipt_preservation,
                    "worktree_deleted": False,
                    "provider_dispatched": False,
                    "task_completion_authority": False,
                }
            )
            if existing_receipt is None:
                write_directory_identity = self._seal_attempt_directory(
                    paths,
                    attempt_id=binding["attempt_id"],
                    create=False,
                )
                _atomic_write(
                    receipt_path,
                    json.dumps(prepared, indent=2, sort_keys=True).encode(
                        "utf-8"
                    )
                    + b"\n",
                    sealed_directory_identity=write_directory_identity,
                )
                existing_receipt = self._validated_recovery_receipt(
                    self._read_recovery_receipt(receipt_path),
                    current_binding=binding,
                    prior_binding=prior_binding,
                )
            elif existing_receipt["recovery_id"] != prepared["recovery_id"]:
                raise DatabasePortalBridgeError(
                    "cross-attempt lifecycle recovery receipt changed"
                )

            # Reauthorize every mutable observation immediately before the
            # lifecycle CAS.  The checkout lease serializes repository mutation;
            # these exact reloads close changes in the DB/Portal/lifecycle joins.
            if (
                self._seal_attempt_directory(
                    paths,
                    attempt_id=binding["attempt_id"],
                    create=False,
                )
                != current_directory_identity
                or self._seal_attempt_directory(
                    prior_paths,
                    attempt_id=prior_binding["attempt_id"],
                    create=False,
                )
                != prior_directory_identity
                or self._strict_binding(paths.binding) != dict(binding)
                or self._strict_binding(prior_paths.binding) != prior_binding
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_binding_changed_before_finalize"
                )
            second_portal_state = self._prior_portal_state_binding(
                daemon,
                prior_paths,
                prior_binding,
                record,
            )
            second_portal_state["attempt_directory_identity"] = (
                prior_directory_identity
            )
            second_preservation = self._preserved_quiescent_worktree(
                daemon,
                verified,
                prior_paths,
                attempt=attempt,
                current_binding=binding,
                prior_binding=prior_binding,
                database_authority=authority,
                portal_state_binding=portal_state_binding,
                lifecycle_recovery_receipt=existing_receipt,
                perform_marker_retirement=False,
                publish_marker_clearance=True,
            )
            try:
                second_authority = self._validated_prior_authority(
                    self.prior_attempt_authority(
                        attempt,
                        binding,
                        prior_binding,
                    ),
                    current_binding=binding,
                    prior_binding=prior_binding,
                )
            except DatabasePortalBridgeDeferred:
                raise
            except Exception as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_database_authority_rejected"
                ) from exc
            if (
                second_portal_state != portal_state_binding
                or self._stable_preservation_authority(second_preservation)
                != self._stable_preservation_authority(preservation)
                or second_authority != authority
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_evidence_changed_before_finalize"
                )

            if record.is_terminal:
                try:
                    exact_terminal = lifecycle_store.require_exact_dead_owner(
                        record.workspace_path,
                        allow_terminal=True,
                        **self._lifecycle_expected(record),
                    )
                except Exception as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_terminal_evidence_changed"
                    ) from exc
                if exact_terminal != record:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_terminal_evidence_changed"
                    )
                terminal = record
            else:
                try:
                    exact_dead_owner = lifecycle_store.require_exact_dead_owner(
                        record.workspace_path,
                        **self._lifecycle_expected(record),
                    )
                except Exception as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_exact_dead_owner_unproven"
                    ) from exc
                if exact_dead_owner != record:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_record_changed"
                    )
                try:
                    terminal = lifecycle_store.finalize_exact_dead_owner(
                        record.workspace_path,
                        expected_owner=record.owner,
                        reason=terminal_reason,
                        retain_terminal=True,
                        **self._lifecycle_expected(record),
                    )
                except Exception as exc:
                    raise DatabasePortalBridgeDeferred(
                        "cross_attempt_lifecycle_finalize_race"
                    ) from exc
            terminal = self._verify_terminal_lifecycle(
                lifecycle_store,
                terminal if record.is_terminal else record,
                terminal_reason=terminal_reason,
            ) if not record.is_terminal else terminal
            if record.is_terminal and (
                not terminal.is_terminal
                or terminal.terminal_reason != terminal_reason
                or int(terminal.fence) != prior_fence + 1
            ):
                raise DatabasePortalBridgeError(
                    "cross-attempt lifecycle terminal evidence is inconsistent"
                )
            terminal_authority_id = _sha256_bytes(
                _canonical_json(terminal.to_dict())
            )
            if (
                self._lifecycle_transition_basis_id(terminal)
                != prepared["lifecycle_transition_basis_id"]
            ):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_terminal_basis_mismatch"
                )
            post_terminal_preservation = self._preserved_quiescent_worktree(
                daemon,
                terminal,
                prior_paths,
                attempt=attempt,
                current_binding=binding,
                prior_binding=prior_binding,
                database_authority=authority,
                portal_state_binding=portal_state_binding,
                lifecycle_recovery_receipt=existing_receipt,
                perform_marker_retirement=True,
                publish_marker_clearance=True,
            )
            if self._stable_preservation_authority(
                post_terminal_preservation
            ) != self._stable_preservation_authority(preservation):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_post_terminal_preservation_changed"
                )
            final_preservation = self._preserved_quiescent_worktree(
                daemon,
                terminal,
                prior_paths,
                attempt=attempt,
                current_binding=binding,
                prior_binding=prior_binding,
                database_authority=authority,
                portal_state_binding=portal_state_binding,
                lifecycle_recovery_receipt=existing_receipt,
                perform_marker_retirement=True,
                publish_marker_clearance=True,
            )
            if self._stable_preservation_authority(
                final_preservation
            ) != self._stable_preservation_authority(preservation):
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_final_preservation_changed"
                )
            committed_body = {
                field: prepared[field]
                for field in _RECOVERY_RECEIPT_FIELDS.difference(
                    {"recovery_id", "receipt_id"}
                )
            }
            committed_body["phase"] = "committed"
            committed_body["terminal_lifecycle_authority_id"] = (
                terminal_authority_id
            )
            committed = self._seal_recovery_receipt(committed_body)
            if committed["recovery_id"] != prepared["recovery_id"]:
                raise DatabasePortalBridgeError(
                    "cross-attempt lifecycle recovery phase identity changed"
                )
            if existing_receipt["phase"] == "committed":
                if existing_receipt != committed:
                    raise DatabasePortalBridgeError(
                        "cross-attempt lifecycle committed receipt changed"
                    )
            else:
                write_directory_identity = self._seal_attempt_directory(
                    paths,
                    attempt_id=binding["attempt_id"],
                    create=False,
                )
                _atomic_write(
                    receipt_path,
                    json.dumps(committed, indent=2, sort_keys=True).encode(
                        "utf-8"
                    )
                    + b"\n",
                    sealed_directory_identity=write_directory_identity,
                )
                observed_committed = self._validated_recovery_receipt(
                    self._read_recovery_receipt(receipt_path),
                    current_binding=binding,
                    prior_binding=prior_binding,
                )
                if observed_committed != committed:
                    raise DatabasePortalBridgeError(
                        "cross-attempt lifecycle committed receipt was not durable"
                    )
            result = {
                "attempted": True,
                "recovered": True,
                "recovery_id": committed["recovery_id"],
                "receipt_path": str(receipt_path),
            }
        finally:
            try:
                released = bool(release(checkout_lease))
            except Exception as exc:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_checkout_lock_release_failed"
                ) from exc
            if not released:
                raise DatabasePortalBridgeDeferred(
                    "cross_attempt_lifecycle_checkout_lock_release_failed"
                )
        assert result is not None
        return result

    def _ensure_attempt_projection(
        self, attempt: Any, record: Any
    ) -> tuple[DatabasePortalAttemptPaths, Mapping[str, Any]]:
        paths = self._paths(attempt)
        seed = self._render_projection(attempt, record)
        self._seal_attempt_directory(
            paths,
            attempt_id=attempt.attempt_id,
            create=True,
        )
        if os.path.lexists(paths.binding):
            observed = self._strict_binding(paths.binding)
            expected = self._binding(
                attempt,
                record,
                seed,
                schema=str(observed.get("schema") or ""),
            )
            if observed != expected:
                raise DatabasePortalBridgeError(
                    "database Portal attempt binding changed across resume"
                )
        else:
            expected = self._binding(attempt, record, seed)
            write_directory_identity = self._seal_attempt_directory(
                paths,
                attempt_id=attempt.attempt_id,
                create=False,
            )
            _atomic_write(
                paths.binding,
                json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n",
                sealed_directory_identity=write_directory_identity,
            )
        if not os.path.lexists(paths.task_projection):
            write_directory_identity = self._seal_attempt_directory(
                paths,
                attempt_id=attempt.attempt_id,
                create=False,
            )
            _atomic_write(
                paths.task_projection,
                seed.encode("utf-8"),
                sealed_directory_identity=write_directory_identity,
            )
        self._verify_projection(paths, expected)
        return paths, expected

    @staticmethod
    def _verify_projection(paths: DatabasePortalAttemptPaths, binding: Mapping[str, Any]) -> str:
        text = _stable_regular_utf8(
            paths.task_projection,
            noun="Portal task projection",
        )
        if _projection_immutable_digest(text) != str(
            binding.get("projection_immutable_digest") or ""
        ):
            raise DatabasePortalBridgeError(
                "Portal task projection changed outside its mutable status field"
            )
        headers = _HEADER.findall(text)
        if headers != [str(binding.get("task_alias") or "")]:
            raise DatabasePortalBridgeError(
                "Portal task projection no longer contains exactly the claimed task"
            )
        return text

    @staticmethod
    def _has_completion_event(paths: DatabasePortalAttemptPaths, alias: str) -> bool:
        if not paths.events.is_file():
            return False
        try:
            lines = paths.events.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return False
        for line in reversed(lines[-4096:]):
            try:
                event = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                isinstance(event, Mapping)
                and event.get("type") == "task_completed"
                and str(event.get("task_id") or "") == alias
            ):
                return True
        return False

    def _accepted_source_transition(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        task_alias: str,
        task_cid: str,
        merge_request_loader: Callable[[str], Any] | None = None,
    ) -> dict[str, Any] | None:
        """Reconstruct one exact landed transition before Quack completion.

        The Portal event file is only an observation.  This method resolves its
        bounded commit claims against Git and emits a content-addressed packet;
        the database completion CAS later makes that packet authoritative for
        the exact task revision.
        """

        if not paths.events.is_file():
            return None
        event_records, event_log_sha256 = _accepted_source_events(paths.events)
        direct_candidates = [
            (index, event)
            for index, event in enumerate(event_records)
            if (
                event.get("type") == "implementation_finished"
                and str(event.get("task_id") or "") == task_alias
                and event.get("returncode") == 0
                and event.get("board_completion")
                == {
                    "complete": True,
                    "pending_merge": False,
                    "reason": "merged_into_target",
                }
            )
        ]
        queued_candidates = [
            (index, event)
            for index, event in enumerate(event_records)
            if (
                event.get("type") == "implementation_finished"
                and str(event.get("task_id") or "") == task_alias
                and event.get("returncode") == 0
                and event.get("board_completion")
                == {
                    "complete": False,
                    "pending_merge": True,
                    "reason": "merge_queued_awaiting_integration",
                }
                and isinstance(event.get("merge_result"), Mapping)
                and event["merge_result"].get("queued") is True
            )
        ]
        reconciled_pairs: list[
            tuple[Mapping[str, Any], Mapping[str, Any]]
        ] = []
        for queued_index, queued_event in queued_candidates:
            implementation_commit = str(
                queued_event.get("implementation_commit") or ""
            )
            portal_attempt_number = queued_event.get("attempt")
            canonical_task_cid = str(
                queued_event.get("canonical_task_cid") or ""
            )
            canonical_task_key = str(
                queued_event.get("canonical_task_key") or ""
            )
            for reconciliation_index, reconciliation in enumerate(
                event_records
            ):
                if reconciliation_index <= queued_index:
                    continue
                if (
                    reconciliation.get("type") == "merge_reconciled"
                    and reconciliation.get("resolved") is True
                    and str(reconciliation.get("task_id") or "")
                    == task_alias
                    and reconciliation.get("attempt")
                    == portal_attempt_number
                    and str(
                        reconciliation.get("implementation_commit") or ""
                    )
                    == implementation_commit
                    and str(
                        reconciliation.get("canonical_task_cid") or ""
                    )
                    == canonical_task_cid
                    and str(
                        reconciliation.get("canonical_task_key") or ""
                    )
                    == canonical_task_key
                ):
                    reconciled_pairs.append(
                        (queued_event, reconciliation)
                    )
        candidate_count = len(direct_candidates) + len(reconciled_pairs)
        if candidate_count == 0:
            return None
        if candidate_count != 1:
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition is not unique"
            )
        normalized_binding = dict(binding)
        binding_id = str(normalized_binding.pop("binding_id", "") or "")
        binding_schema = binding.get("schema")
        expected_binding_fields = (
            _BINDING_FIELDS.difference({"binding_id"})
            if binding_schema == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
            else _BINDING_FIELDS_V1.difference({"binding_id"})
            if binding_schema == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1
            else frozenset()
        )
        if (
            not expected_binding_fields
            or set(normalized_binding) != expected_binding_fields
            or binding.get("interface") != self.INTERFACE
            or binding.get("attempt_id") != str(attempt.attempt_id)
            or binding.get("claim_id") != str(attempt.claim_id)
            or binding.get("task_cid") != task_cid
            or binding.get("task_alias") != task_alias
            or binding.get("fencing_token") != int(attempt.fencing_token)
            or binding.get("fence_epoch") != int(attempt.fence_epoch)
            or binding.get("authoritative_task_store") != "duckdb"
            or binding.get("projection_authority") is not False
            or binding_id != _sha256_bytes(_canonical_json(normalized_binding))
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source database binding is inconsistent"
            )
        try:
            if self._strict_binding(paths.binding) != dict(binding):
                raise DatabasePortalBridgeError(
                    "Portal accepted-source database binding changed"
                )
        except DatabasePortalBridgeError:
            raise
        except Exception as exc:
            raise DatabasePortalBridgeError(
                "Portal accepted-source database binding is unavailable"
            ) from exc
        projection_text = self._verify_projection(paths, binding)
        try:
            # Imported lazily because implementation_daemon owns the parser and
            # imports this bridge.  Invocation happens only after both modules
            # are fully initialized.
            from .implementation_daemon import (
                parse_task_text,
                task_declared_output_paths,
            )

            parsed_tasks = parse_task_text(
                projection_text,
                path=paths.task_projection,
                task_header_prefix=self.task_header_prefix,
            )
            if len(parsed_tasks) != 1 or parsed_tasks[0].task_id != task_alias:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source projection identity is ambiguous"
                )
            parsed_task = parsed_tasks[0]
            identity_metadata = dict(parsed_task.metadata)
            if parsed_task.canonical_task_key:
                identity_metadata["canonical task key"] = (
                    parsed_task.canonical_task_key
                )
            if parsed_task.canonical_task_cid:
                identity_metadata["canonical task cid"] = (
                    parsed_task.canonical_task_cid
                )
            canonical_identity = canonical_task_identity(
                {
                    "task_id": parsed_task.task_id,
                    "title": parsed_task.title,
                    "outputs": task_declared_output_paths(parsed_task),
                    "acceptance": parsed_task.acceptance,
                    "metadata": identity_metadata,
                },
                board_namespace=(
                    parsed_task.board_namespace
                    or self.board_namespace
                    or paths.task_projection.name
                ),
                source_path=paths.task_projection,
            )
        except DatabasePortalBridgeError:
            raise
        except Exception as exc:
            raise DatabasePortalBridgeError(
                "Portal accepted-source projection identity is invalid"
            ) from exc
        if self.repo_root is None:
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no repository authority"
            )
        reconciliation: Mapping[str, Any] | None = None
        if direct_candidates:
            event = direct_candidates[0][1]
            transition_schema = (
                DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA
            )
        else:
            event, reconciliation = reconciled_pairs[0]
            transition_schema = (
                DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA
            )
        merge = (
            reconciliation.get("merge_result")
            if reconciliation is not None
            else event.get("merge_result")
        )
        if not isinstance(merge, Mapping):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no merge result"
            )
        queued_merge = event.get("merge_result")
        if not isinstance(queued_merge, Mapping):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no queue binding"
            )
        baseline = str(event.get("baseline_ref") or "")
        implementation = str(event.get("implementation_commit") or "")
        proof = (
            reconciliation.get("integration_commit_proof")
            if reconciliation is not None
            else merge.get("integration_commit_proof")
        )
        invariant = (
            reconciliation.get("post_merge_declared_output_invariant")
            if reconciliation is not None
            else merge.get("post_merge_declared_output_invariant")
        )
        merge_commit = str(
            merge.get("merge_commit")
            or (
                proof.get("integration_commit")
                if isinstance(proof, Mapping)
                else ""
            )
            or (
                reconciliation.get("merge_commit")
                if reconciliation is not None
                else ""
            )
            or ""
        )
        target_branch = str(
            merge.get("target_branch")
            or (
                proof.get("target_branch")
                if isinstance(proof, Mapping)
                else ""
            )
            or ""
        )
        canonical_task_cid = str(
            event.get("canonical_task_cid") or ""
        )
        canonical_task_key = str(
            event.get("canonical_task_key") or ""
        )
        request_id = str(queued_merge.get("request_id") or "")
        portal_attempt_number = event.get("attempt")
        event_target_repository_id = str(
            event.get("target_repository_id") or ""
        )
        merge_target_repository_id = str(
            queued_merge.get("target_repository_id")
            or merge.get("target_repository_id")
            or ""
        )
        target_repository_id = (
            event_target_repository_id or merge_target_repository_id
        )
        expected_repository_id = checkout_repository_id(self.repo_root)
        if (
            any(re.fullmatch(r"[0-9a-f]{40}", item) is None for item in (
                baseline,
                implementation,
                merge_commit,
            ))
            or not target_branch
            or not request_id
            or isinstance(portal_attempt_number, bool)
            or not isinstance(portal_attempt_number, int)
            or portal_attempt_number < 1
            or str(event.get("board_namespace") or "") != self.board_namespace
            or target_branch != self.merge_target_branch
            or (
                event_target_repository_id
                and merge_target_repository_id
                and event_target_repository_id != merge_target_repository_id
            )
            or target_repository_id != expected_repository_id
            or merge.get("merged") is not True
            or (
                reconciliation is None
                and merge.get("returncode") != 0
            )
            or (
                reconciliation is None
                and str(merge.get("implementation_commit") or "")
                != implementation
            )
            or not isinstance(proof, Mapping)
            or proof.get("passed") is not True
            or proof.get("implementation_commit") != implementation
            or proof.get("integration_commit") != merge_commit
            or proof.get("integration_ref") != merge_commit
            or proof.get("target_branch") != target_branch
            or not isinstance(invariant, Mapping)
            or invariant.get("passed") is not True
            or invariant.get("repository_ref") != merge_commit
            or canonical_task_cid != canonical_identity.canonical_task_cid
            or canonical_task_key != canonical_identity.canonical_task_key
            or event.get("canonical_task_cid") != canonical_task_cid
            or event.get("canonical_task_key") != canonical_task_key
            or queued_merge.get("canonical_task_cid") != canonical_task_cid
            or queued_merge.get("canonical_task_key") != canonical_task_key
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition is inconsistent"
            )
        if reconciliation is not None:
            completion_persistence = reconciliation.get(
                "completion_persistence"
            )
            runtime_binding = (
                completion_persistence.get("runtime_taskboard_binding")
                if isinstance(completion_persistence, Mapping)
                else None
            )
            taskboard_snapshot = (
                completion_persistence.get("fsynced_taskboard_snapshot")
                if isinstance(completion_persistence, Mapping)
                else None
            )
            expected_completion = {task_alias: canonical_task_cid}
            try:
                runtime_projection_path = Path(
                    str(runtime_binding.get("path") or "")
                ).resolve()
                snapshot_projection_path = Path(
                    str(taskboard_snapshot.get("path") or "")
                ).resolve()
                expected_projection_path = paths.task_projection.resolve()
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                raise DatabasePortalBridgeError(
                    "Portal reconciled-source completion is inconsistent"
                ) from exc
            if (
                reconciliation.get("reason")
                not in {
                    "merge_retried",
                    "completion_persistence_recovered_from_landed_rewrite",
                }
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(event.get("event_id") or ""),
                )
                is None
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(reconciliation.get("event_id") or ""),
                )
                is None
                or reconciliation.get("completion_task_cids")
                != expected_completion
                or not isinstance(completion_persistence, Mapping)
                or completion_persistence.get("passed") is not True
                or completion_persistence.get("reason")
                != "completion_persisted"
                or completion_persistence.get("durable_update") is not True
                or completion_persistence.get("status_persisted") is not True
                or completion_persistence.get("expected_task_ids")
                != [task_alias]
                or completion_persistence.get("completed_task_ids")
                != [task_alias]
                or completion_persistence.get("missing_task_ids") != []
                or completion_persistence.get("receipt_mismatches") != {}
                or not isinstance(runtime_binding, Mapping)
                or runtime_binding.get("passed") is not True
                or runtime_binding.get("authoritative") is not True
                or runtime_binding.get("ignored") is not True
                or runtime_binding.get("runtime_projection") is not True
                or runtime_projection_path != expected_projection_path
                or not isinstance(taskboard_snapshot, Mapping)
                or taskboard_snapshot.get("passed") is not True
                or taskboard_snapshot.get("reason")
                != "fsynced_taskboard_completion_proven"
                or taskboard_snapshot.get("runtime_projection") is not True
                or taskboard_snapshot.get("runtime_binding")
                != runtime_binding
                or taskboard_snapshot.get("expected_task_ids")
                != [task_alias]
                or taskboard_snapshot.get("observed_statuses")
                != {task_alias: "completed"}
                or taskboard_snapshot.get("observed_task_cids")
                != expected_completion
                or taskboard_snapshot.get("missing_task_ids") != []
                or taskboard_snapshot.get("ambiguous_task_ids") != []
                or taskboard_snapshot.get("status_mismatches") != {}
                or taskboard_snapshot.get("task_cid_mismatches") != {}
                or snapshot_projection_path != expected_projection_path
            ):
                raise DatabasePortalBridgeError(
                    "Portal reconciled-source completion is inconsistent"
                )
        if not callable(merge_request_loader):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition has no merge-queue authority"
            )
        try:
            request = merge_request_loader(request_id)
            converter = getattr(request, "to_dict", None)
            request_record = (
                dict(converter())
                if callable(converter)
                else dict(request)
                if isinstance(request, Mapping)
                else None
            )
        except Exception as exc:
            raise DatabasePortalBridgeError(
                "Portal accepted-source merge request is unavailable"
            ) from exc
        if not isinstance(request_record, dict):
            raise DatabasePortalBridgeError(
                "Portal accepted-source merge request is unavailable"
            )
        request_metadata = request_record.get("metadata")
        request_task = (
            request_metadata.get("task")
            if isinstance(request_metadata, Mapping)
            else None
        )
        request_task_metadata = (
            request_task.get("metadata")
            if isinstance(request_task, Mapping)
            else None
        )
        completion_task_cids = (
            request_metadata.get("completion_task_cids")
            if isinstance(request_metadata, Mapping)
            else None
        )
        request_dedupe_key = str(request_record.get("dedupe_key") or "")
        request_status = str(request_record.get("status") or "")
        request_attempt = request_record.get("attempt")
        cancellation = (
            request_metadata.get("cancellation")
            if isinstance(request_metadata, Mapping)
            else None
        )
        direct_queue_terminal = bool(
            reconciliation is None
            and request_status == "completed"
            and request_attempt == portal_attempt_number
        )
        reconciled_queue_terminal = bool(
            reconciliation is not None
            and request_status == "cancelled"
            and isinstance(request_attempt, int)
            and not isinstance(request_attempt, bool)
            and request_attempt >= portal_attempt_number
            and isinstance(cancellation, Mapping)
            and set(cancellation) == {"at", "reason"}
            and isinstance(cancellation.get("at"), (int, float))
            and not isinstance(cancellation.get("at"), bool)
            and float(cancellation["at"]) >= 0.0
            and cancellation.get("reason") == "stale_quarantined_merge"
            and request_record.get("branch_name")
            == str(event.get("branch") or "")
        )
        if (
            request_record.get("request_id") != request_id
            or not (direct_queue_terminal or reconciled_queue_terminal)
            or request_record.get("task_id") != task_alias
            or request_record.get("commit_sha") != implementation
            or request_record.get("canonical_task_id") != canonical_task_cid
            or request_record.get("canonical_task_key") != canonical_task_key
            or not re.fullmatch(r"[0-9a-f]{64}", request_dedupe_key)
            or not isinstance(request_metadata, Mapping)
            or request_metadata.get("baseline_ref") != baseline
            or request_metadata.get("implementation_commit") != implementation
            or request_metadata.get("target_binding_schema")
            != "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            or request_metadata.get("target_repository_id")
            != target_repository_id
            or request_metadata.get("target_branch") != target_branch
            or request_metadata.get("repo_root") != str(self.repo_root)
            or not isinstance(completion_task_cids, Mapping)
            or completion_task_cids.get(task_alias) != canonical_task_cid
            or not isinstance(request_task, Mapping)
            or request_task.get("task_id") != task_alias
            or request_task.get("board_namespace") != self.board_namespace
            or request_task.get("canonical_task_cid") != canonical_task_cid
            or request_task.get("canonical_task_key") != canonical_task_key
            or not isinstance(request_task_metadata, Mapping)
            or request_task_metadata.get("database attempt id")
            != str(attempt.attempt_id)
            or request_task_metadata.get("database claim id")
            != str(attempt.claim_id)
            or request_task_metadata.get("database task cid") != task_cid
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source merge request is inconsistent"
            )
        merge_request_digest = _sha256_bytes(_canonical_json(request_record))
        git_environment = {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }

        def git(*arguments: str) -> bytes:
            try:
                completed = subprocess.run(
                    ["/usr/bin/git", "--no-replace-objects", *arguments],
                    cwd=self.repo_root,
                    env=git_environment,
                    capture_output=True,
                    check=False,
                    timeout=10.0,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source Git proof is unavailable"
                ) from exc
            if completed.returncode != 0:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source Git proof failed"
                )
            return completed.stdout

        parents = git("rev-list", "--parents", "-n", "1", merge_commit)
        parent_fields = parents.decode("ascii").strip().split()
        if (
            len(parent_fields) != 3
            or parent_fields[0] != merge_commit
            or parent_fields[2] != implementation
        ):
            raise DatabasePortalBridgeError(
                "Portal accepted-source transition is not the exact Git merge"
            )
        integration_base_commit = parent_fields[1]
        target_advanced = integration_base_commit != baseline
        transition_proof = dict(proof)
        if target_advanced:
            # A task's dispatch baseline and the target's first parent at
            # integration are different identities when another fenced lane
            # lands first.  The completed merge request still binds the
            # immutable implementation parent; Git supplies the exact target
            # parent.  Both histories must descend from the dispatch baseline.
            try:
                git(
                    "merge-base",
                    "--is-ancestor",
                    baseline,
                    integration_base_commit,
                )
                git(
                    "merge-base",
                    "--is-ancestor",
                    baseline,
                    implementation,
                )
            except DatabasePortalBridgeError as exc:
                raise DatabasePortalBridgeError(
                    "Portal accepted-source transition is not the exact Git merge"
                ) from exc
            claimed_candidate_baselines = {
                str(value)
                for value in (
                    merge.get("candidate_baseline_ref"),
                    proof.get("candidate_baseline_ref"),
                )
                if value is not None and str(value)
            }
            claimed_integration_bases = {
                str(value)
                for value in (
                    merge.get("integration_base_commit"),
                    proof.get("integration_base_commit"),
                )
                if value is not None and str(value)
            }
            claimed_exact_topology = proof.get("exact_two_parent_merge")
            if (
                (
                    claimed_candidate_baselines
                    and claimed_candidate_baselines != {baseline}
                )
                or (
                    claimed_integration_bases
                    and claimed_integration_bases
                    != {integration_base_commit}
                )
                or claimed_exact_topology not in {None, True}
            ):
                raise DatabasePortalBridgeError(
                    "Portal accepted-source transition is not the exact Git merge"
                )
            transition_schema = (
                DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
            )
            transition_proof.update(
                {
                    "candidate_baseline_ref": baseline,
                    "integration_base_commit": integration_base_commit,
                    "exact_two_parent_merge": True,
                }
            )
        implementation_tree = git(
            "rev-parse", f"{implementation}^{{tree}}"
        ).decode("ascii").strip()
        merge_tree = git("rev-parse", f"{merge_commit}^{{tree}}").decode(
            "ascii"
        ).strip()
        changed_path_diff_sha256 = _sha256_bytes(
            git(
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                integration_base_commit,
                merge_commit,
            )
        )
        transition: dict[str, Any] = {
            "schema": transition_schema,
            "board_namespace": self.board_namespace,
            "configured_board_admission_cid": self.configured_board_admission_cid,
            "task_alias": task_alias,
            "database_task_cid": task_cid,
            "attempt_id": str(attempt.attempt_id),
            "attempt_number": int(attempt.attempt_number),
            "portal_attempt_number": portal_attempt_number,
            "claim_id": str(attempt.claim_id),
            "fencing_token": int(attempt.fencing_token),
            "database_attempt_binding": dict(binding),
            "canonical_task_cid": canonical_task_cid,
            "canonical_task_key": canonical_task_key,
            "request_id": request_id,
            "merge_request_digest": merge_request_digest,
            "merge_request_dedupe_key": request_dedupe_key,
            "target_repository_id": target_repository_id,
            "implementation_commit": implementation,
            "implementation_tree": implementation_tree,
            "merge_commit": merge_commit,
            "merge_tree": merge_tree,
            "target_branch": target_branch,
            "changed_path_diff_sha256": changed_path_diff_sha256,
            "integration_commit_proof": transition_proof,
            "declared_output_invariant": dict(invariant),
            "portal_event_log_sha256": event_log_sha256,
            "authority": "database_completion_cas_after_portal_and_git_verification",
            "task_completion_authority": False,
            "worker_self_approval": False,
        }
        if target_advanced:
            transition.update(
                {
                    "candidate_baseline_ref": baseline,
                    "integration_base_commit": integration_base_commit,
                }
            )
        else:
            # Preserve the accepted-source-transition@1/@2 byte vocabulary.
            transition["baseline_ref"] = baseline
        if reconciliation is not None:
            transition.update(
                {
                    "source_event_mode": "queued_merge_reconciliation",
                    "queued_implementation_event_id": str(
                        event.get("event_id") or ""
                    ),
                    "reconciliation_event_id": str(
                        reconciliation.get("event_id") or ""
                    ),
                    "merge_queue_terminal_status": request_status,
                    "merge_queue_attempt": request_attempt,
                    "merge_queue_cancellation_reason": str(
                        cancellation.get("reason") or ""
                    ),
                    "completion_persistence": dict(
                        reconciliation["completion_persistence"]
                    ),
                }
            )
        transition["transition_cid"] = _sha256_bytes(
            _canonical_transition_json(transition)
        )
        return transition

    @staticmethod
    def _terminal_failure(result: Mapping[str, Any]) -> str:
        if result.get("blocked") is True:
            return str(result.get("reason") or "portal_execution_blocked")
        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return ""
        if implementation.get("deferred") is True:
            return str(implementation.get("reason") or "portal_execution_deferred")
        returncode = implementation.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0:
            return str(implementation.get("reason") or "portal_provider_failed")
        if implementation.get("skipped") is True:
            return str(implementation.get("reason") or "portal_execution_skipped")
        return ""

    @staticmethod
    def _is_external_protected_recovery_deferral(
        result: Mapping[str, Any],
    ) -> bool:
        """Recognize only the daemon's exact no-write owner deferral."""

        recovery = result.get("protected_checkout_recovery")
        write_count = result.get("write_count")
        return bool(
            result.get("blocked") is True
            and result.get("unchanged") is True
            and isinstance(write_count, int)
            and not isinstance(write_count, bool)
            and write_count == 0
            and result.get("implementation_result") is None
            and result.get("reason")
            == "external_protected_checkout_recovery_required"
            and isinstance(recovery, Mapping)
            and recovery.get("required") is True
            and recovery.get("adopted") is False
            and recovery.get("blocked") is True
            and recovery.get("recovered") is False
            and recovery.get("reason")
            == "external_protected_checkout_recovery_required"
            and recovery.get("protected_recovery_owner")
            == "implementation_supervisor"
            and bool(str(recovery.get("lock_path") or "").strip())
            and result.get("projection_delta") == {}
            and result.get("merge_reconciliation") == []
        )

    def _acceptance_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        summaries: Sequence[Mapping[str, Any]],
        merge_request_loader: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        alias = str(binding.get("task_alias") or "")
        projection_text = self._verify_projection(paths, binding)
        if _projection_status(projection_text) not in _TERMINAL_STATUSES:
            raise DatabasePortalBridgeDeferred("Portal task projection is not complete")
        if not self._has_completion_event(paths, alias):
            raise DatabasePortalBridgeError(
                "Portal completion lacks a matching durable task_completed event"
            )
        evidence = {
            "binding_id": str(binding.get("binding_id") or ""),
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "projection_digest": _sha256_bytes(projection_text.encode("utf-8")),
            "projection_immutable_digest": str(binding.get("projection_immutable_digest") or ""),
            "state_digest": _sha256_file(paths.state) if paths.state.is_file() else "",
            "events_digest": _sha256_file(paths.events),
            "portal_passes": [dict(item) for item in summaries],
        }
        accepted_source_transition = self._accepted_source_transition(
            attempt=attempt,
            paths=paths,
            binding=binding,
            task_alias=alias,
            task_cid=str(attempt.task_cid),
            merge_request_loader=merge_request_loader,
        )
        if accepted_source_transition is not None:
            evidence["accepted_source_transition"] = accepted_source_transition
        evidence_digest = _sha256_bytes(_canonical_json(evidence))
        transition_schema = (
            accepted_source_transition.get("schema")
            if isinstance(accepted_source_transition, Mapping)
            else None
        )
        receipt = {
            "schema": (
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1
                if accepted_source_transition is None
                else self.RECEIPT_SCHEMA
                if transition_schema
                == DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
                else DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2
            ),
            "interface": self.INTERFACE,
            "status": "succeeded",
            "provider": "PortalImplementationDaemon",
            "execution_mode": "database-authoritative-portal-bridge",
            "accepted": True,
            "completion_authority": "DatabaseImplementationDaemon",
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "attempt_id": str(attempt.attempt_id),
            "binding_id": str(binding.get("binding_id") or ""),
            "evidence_digest": evidence_digest,
            "portal_evidence": evidence,
        }
        if accepted_source_transition is not None:
            receipt["accepted_source_transition"] = accepted_source_transition
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def run_provider(self, attempt: Any) -> Mapping[str, Any]:
        """Run bounded real Portal passes and return only accepted evidence."""

        record = self._record_for_attempt(self.task_source, attempt)
        paths, binding = self._ensure_attempt_projection(attempt, record)
        summaries: list[Mapping[str, Any]] = []
        daemon = self.portal_factory(
            paths,
            str(binding.get("task_alias") or attempt.task_cid),
        )
        if daemon is None or not callable(getattr(daemon, "run_once", None)):
            raise DatabasePortalBridgeError(
                "portal_factory did not return a Portal-compatible daemon"
            )
        merge_queue = getattr(daemon, "merge_queue", None)
        merge_request_loader = getattr(merge_queue, "get", None)
        try:
            self._recover_superseded_attempt_lifecycle(
                attempt=attempt,
                paths=paths,
                binding=binding,
                daemon=daemon,
            )
            for _pass_index in range(self.max_passes):
                projection = self._verify_projection(paths, binding)
                if _projection_status(
                    projection
                ) in _TERMINAL_STATUSES and self._has_completion_event(
                    paths, str(binding.get("task_alias") or "")
                ):
                    return self._acceptance_receipt(
                        attempt=attempt,
                        paths=paths,
                        binding=binding,
                        summaries=summaries,
                        merge_request_loader=merge_request_loader,
                    )
                raw_result = daemon.run_once()
                if not isinstance(raw_result, Mapping):
                    raise DatabasePortalBridgeError("Portal daemon returned a non-object result")
                summary = _bounded_portal_result(raw_result)
                summaries.append(summary)
                self._verify_projection(paths, binding)
                implementation = raw_result.get("implementation_result")
                if (
                    isinstance(implementation, Mapping)
                    and implementation.get("deferred") is True
                ):
                    raise DatabasePortalBridgeDeferred(
                        str(
                            implementation.get("reason")
                            or "portal_execution_deferred"
                        )
                    )
                if self._is_external_protected_recovery_deferral(raw_result):
                    raise DatabasePortalBridgeDeferred(
                        "external_protected_checkout_recovery_required"
                    )
                failure = self._terminal_failure(raw_result)
                if failure:
                    if (
                        "deferred" in failure
                        or "backoff" in failure
                        or "capacity" in failure
                        or "resource_claim" in failure
                        or failure
                        in {
                            "inflight_process",
                            "inflight_process_missing",
                            "worktree_lifecycle_claim_exists",
                        }
                    ):
                        raise DatabasePortalBridgeDeferred(failure)
                    raise DatabasePortalBridgeError(failure)
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=summaries,
                merge_request_loader=merge_request_loader,
            )
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(daemon, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _require_accepted_provider(attempt: Any, provider_result: Mapping[str, Any]) -> str:
        schema = provider_result.get("schema")
        if (
            schema
            not in {
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1,
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2,
                DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
            }
            or provider_result.get("interface") != DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
            or provider_result.get("accepted") is not True
            or provider_result.get("status") != "succeeded"
            or provider_result.get("provider") != "PortalImplementationDaemon"
            or str(provider_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(provider_result.get("attempt_id") or "") != str(attempt.attempt_id)
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected unaccepted Portal provider evidence"
            )
        digest = str(provider_result.get("evidence_digest") or "")
        evidence = provider_result.get("portal_evidence")
        normalized_receipt = dict(provider_result)
        receipt_id = str(normalized_receipt.pop("receipt_id", "") or "")
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or not isinstance(evidence, Mapping)
            or digest != _sha256_bytes(_canonical_json(evidence))
            or receipt_id != _sha256_bytes(_canonical_json(normalized_receipt))
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected malformed Portal evidence identity"
            )
        transition = provider_result.get("accepted_source_transition")
        if schema == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1 and transition is not None:
            raise DatabasePortalBridgeError(
                "database effect rejected a legacy receipt with a source transition"
            )
        if schema != DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1 and not isinstance(
            transition, Mapping
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected a source-transition receipt without its transition"
            )
        if transition is not None:
            if not isinstance(transition, Mapping):
                raise DatabasePortalBridgeError(
                    "database effect rejected malformed source transition"
                )
            normalized = dict(transition)
            transition_cid = str(normalized.pop("transition_cid", "") or "")
            if (
                transition.get("schema")
                not in {
                    DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA,
                    DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA,
                    DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA,
                }
                or (
                    schema == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2
                    and transition.get("schema")
                    == DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
                )
                or (
                    schema == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
                    and transition.get("schema")
                    != DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
                )
                or transition.get("database_task_cid") != str(attempt.task_cid)
                or transition.get("worker_self_approval") is not False
                or (
                    transition.get("schema")
                    == DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA
                    and transition.get("source_event_mode")
                    != "queued_merge_reconciliation"
                )
                or transition_cid
                != _sha256_bytes(_canonical_transition_json(normalized))
            ):
                raise DatabasePortalBridgeError(
                    "database effect rejected unbound source transition"
                )
        return digest

    def apply_effect(self, attempt: Any, provider_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind the already-applied Portal effect to the database phase."""

        digest = self._require_accepted_provider(attempt, provider_result)
        result = {
            "status": "applied",
            "effect": "portal-supervised-accepted-effect",
            "effect_key": f"portal:{attempt.task_cid}:{attempt.attempt_id}",
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(provider_result.get("receipt_id") or ""),
            "evidence_digest": digest,
        }
        if provider_result.get("accepted_source_transition") is not None:
            result["accepted_source_transition"] = dict(
                provider_result["accepted_source_transition"]
            )
        return result

    def validate_effect(self, attempt: Any, effect_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Admit only an exact effect derived from accepted Portal evidence."""

        digest = str(effect_result.get("evidence_digest") or "")
        if (
            effect_result.get("status") != "applied"
            or effect_result.get("effect") != "portal-supervised-accepted-effect"
            or str(effect_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(effect_result.get("attempt_id") or "") != str(attempt.attempt_id)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            raise DatabasePortalBridgeError(
                "database validation rejected unbound Portal effect evidence"
            )
        result = {
            "outcome": "passed",
            "evidence_digest": digest,
            "argv": ["portal-supervisor-gates"],
            "validator": self.INTERFACE,
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(effect_result.get("portal_receipt_id") or ""),
        }
        if effect_result.get("accepted_source_transition") is not None:
            result["accepted_source_transition"] = dict(
                effect_result["accepted_source_transition"]
            )
        return result


__all__ = (
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA",
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA_V1",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2",
    "DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA",
    "DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA",
    "DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeError",
    "DatabasePortalExecutionBridge",
    "PortalDaemonFactory",
)
