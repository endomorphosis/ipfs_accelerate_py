"""Governed, crash-recoverable retry reset for DuckDB task sources.

This owner is intentionally quiescent-only.  It does not stop supervisors or
workers.  A caller must bind every lane and lifecycle-owner PID/status file;
the operation refuses to start while any declared owner is live or the task is
reported active.  DuckDB and JSON sidecars cannot share one transaction, so a
fsynced intent journal makes every boundary replayable and observable to
launch preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .authorization_logic import ControlMutationAuthorizer, ControlMutationPolicy
from .checkout_lock import checkout_repository_id
from .control_contracts import (
    AuthorizationDecision,
    EffectKind,
    ExpectedEffect,
    Operation,
    OperationRequest,
    decode_operation_request,
)
from .duckdb_state import exclusive_file_lock
from .duckdb_task_source import DuckDBTaskSource
from .formal_verification_contracts import content_identity
from .todo_daemon.core import pid_alive

RETRY_RESET_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset@1"
RETRY_RESET_JOURNAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset-journal@1"
)
RETRY_RESET_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset-receipt@1"
)
RETRY_RESET_EVENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset-event@1"
)
RETRY_RESET_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-mutation-policy@1"
)
RETRY_RESET_OWNER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset-owner@1"
)
RETRY_RESET_OWNER_FILE: Final = "duckdb-retry-reset-owner.json"
RETRY_RESET_GRANT: Final = "grant:duckdb-retry-reset"
MAX_SIDECAR_BYTES: Final = 8 * 1024 * 1024
MAX_LANES: Final = 64
MAX_OWNER_PATHS: Final = 32
MAX_DISCOVERED_SIDECARS: Final = 512
_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RETRY_FIELDS: Final = (
    "selection_penalty",
    "consecutive_failures",
    "consecutive_no_change",
    "merge_failure_count",
    "cooldown_until",
    "notes",
)
_PHASES: Final = {
    "prepared",
    "database_committed",
    "sidecars_committed",
    "receipt_committed",
    "completed",
}
_ELIGIBLE_STATUSES: Final = frozenset(
    {
        "pending",
        "ready",
        "retrying",
        "completed",
        "failed",
        "quarantined",
    }
)
_QUEUE_TOP_FIELDS: Final = frozenset(
    {"schema", "updated_at", "entry_count", "entries", "aliases"}
)
_QUEUE_V2_ENTRY_FIELDS: Final = frozenset(
    {
        "task_id",
        "priority",
        "track",
        "canonical_task_cid",
        "canonical_task_key",
        "aliases",
        "provenance",
        "selection_penalty",
        "attempt_count",
        "last_selected_at",
        "last_completed_at",
        "consecutive_failures",
        "consecutive_no_change",
        "merge_failure_count",
        "cooldown_until",
        "notes",
    }
)
_QUEUE_V3_AUTHORITY_FIELDS: Final = frozenset(
    {
        "authority_renewal_key",
        "authority_renewal_failure_count",
        "authority_renewal_last_failure_at",
        "authority_renewal_cooldown_until",
        "authority_renewal_quarantined",
        "authority_renewal_reason",
    }
)
_STATE_FIELDS: Final = frozenset(
    {
        "heartbeat_at",
        "last_progress_at",
        "active_task_id",
        "active_task_key",
        "active_task_cid",
        "active_task_title",
        "active_task_track",
        "active_task_started_at",
        "active_attempt",
        "active_phase",
        "active_phase_started_at",
        "active_phase_detail",
        "active_log_path",
        "active_worktree_path",
        "active_branch",
        "implementation_in_progress",
        "recommended_task_id",
        "recommended_actions",
        "completed_task_ids",
        "ready_task_ids",
        "selectable_ready_task_ids",
        "external_reserved_task_ids",
        "assumed_completed_task_ids",
        "eligible_ready_task_ids",
        "strict_deprioritized_ready_task_ids",
        "waiting_task_ids",
        "blocked_task_ids",
        "task_statuses",
        "task_artifacts",
        "task_validation",
        "task_identities",
        "implementation_attempts",
        "implementation_attempts_by_cid",
        "retry_budget_repair_receipts",
        "last_implementation_task_id",
        "last_implementation_task_key",
        "last_implementation_task_cid",
        "last_implementation_started_at",
        "last_implementation_finished_at",
        "last_implementation_returncode",
        "last_implementation_log_path",
        "last_implementation_worktree_path",
        "last_implementation_branch",
        "last_implementation_commit",
        "last_proof_workflow",
        "last_merge_started_at",
        "last_merge_finished_at",
        "last_merge_branch",
        "last_merge_commit",
        "last_merge_returncode",
        "last_merge_error",
        "completed_count",
        "ready_count",
        "selectable_ready_count",
        "external_reserved_count",
        "assumed_completed_count",
        "eligible_ready_count",
        "strict_deprioritized_ready_count",
        "waiting_count",
        "blocked_count",
        "task_count",
        "strategy_generation",
        "selection_idle_reason",
    }
)
_TASK_IDENTITY_FIELDS: Final = frozenset(
    {
        "canonical_task_key",
        "canonical_task_cid",
        "semantic_fingerprint",
        "display_task_id",
        "board_namespace",
        "source_path",
        "identity_version",
        # Legacy projections used these equivalent identity labels.
        "task_cid",
        "content_id",
    }
)


class DuckDBRetryResetError(RuntimeError):
    """Base fail-closed retry-reset error."""


class DuckDBRetryResetConflict(DuckDBRetryResetError):
    """The requested identity, revision, owner, or replay state changed."""


class DuckDBRetryResetAuthorizationError(DuckDBRetryResetError):
    """The request is absent from the independently trusted live policy."""


class DuckDBRetryResetQuiescenceError(DuckDBRetryResetError):
    """A declared lifecycle owner or bound task lane is still active."""


class DuckDBRetryResetCorruptionError(DuckDBRetryResetError):
    """A governed journal, state, queue, or receipt is malformed."""


@dataclass(frozen=True)
class LaneBinding:
    state_prefix: str
    state_path: str
    queue_path: str

    @property
    def supervisor_lock_path(self) -> str:
        parent = Path(self.state_path).parent
        return (parent / f"{self.state_prefix}_supervisor.lock").as_posix()

    @property
    def supervisor_pid_path(self) -> str:
        parent = Path(self.state_path).parent
        return (parent / f"{self.state_prefix}_supervisor.pid").as_posix()

    @property
    def daemon_pid_path(self) -> str:
        parent = Path(self.state_path).parent
        return (parent / f"{self.state_prefix}_managed_daemon.pid").as_posix()

    @property
    def status_path(self) -> str:
        parent = Path(self.state_path).parent
        return (parent / f"{self.state_prefix}_supervisor_status.json").as_posix()

    def to_dict(self) -> dict[str, str]:
        return {
            "state_prefix": self.state_prefix,
            "state_path": self.state_path,
            "queue_path": self.queue_path,
            "supervisor_lock_path": self.supervisor_lock_path,
            "supervisor_pid_path": self.supervisor_pid_path,
            "daemon_pid_path": self.daemon_pid_path,
            "status_path": self.status_path,
        }


@dataclass(frozen=True)
class RetryResetBinding:
    database_path: str
    plan_root_cid: str
    task_source_repository_tree_id: str
    repository_head_commit: str
    task_cid: str
    task_alias: str
    task_revision: int
    expected_status: str
    writer_id: str
    writer_fencing_token: int
    lanes: tuple[LaneBinding, ...]
    lifecycle_owner_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RETRY_RESET_SCHEMA,
            "database_path": self.database_path,
            "plan_root_cid": self.plan_root_cid,
            "task_source_repository_tree_id": self.task_source_repository_tree_id,
            "repository_head_commit": self.repository_head_commit,
            "task_cid": self.task_cid,
            "task_alias": self.task_alias,
            "task_revision": self.task_revision,
            "expected_status": self.expected_status,
            "reopen_status": "retrying",
            "writer_id": self.writer_id,
            "writer_fencing_token": self.writer_fencing_token,
            "lanes": [item.to_dict() for item in self.lanes],
            "lifecycle_owner_paths": list(self.lifecycle_owner_paths),
        }


@dataclass(frozen=True)
class RetryResetOwnerConfig:
    """Owner-controlled trust anchor and complete scheduler topology."""

    repository_root: str
    repository_id: str
    database_path: str
    task_source_repository_tree_id: str
    policy_path: str
    policy_digest: str
    lanes: tuple[LaneBinding, ...]
    lifecycle_owner_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RETRY_RESET_OWNER_SCHEMA,
            "repository_root": self.repository_root,
            "repository_id": self.repository_id,
            "database_path": self.database_path,
            "task_source_repository_tree_id": self.task_source_repository_tree_id,
            "policy_path": self.policy_path,
            "policy_digest": self.policy_digest,
            "lanes": [
                {
                    "state_prefix": lane.state_prefix,
                    "state_path": lane.state_path,
                    "queue_path": lane.queue_path,
                }
                for lane in self.lanes
            ],
            "lifecycle_owner_paths": list(self.lifecycle_owner_paths),
        }


def _relative(value: Any, noun: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DuckDBRetryResetError(f"{noun} must be a non-empty relative path")
    candidate = Path(value.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DuckDBRetryResetError(f"{noun} must remain below state_root")
    normalized = candidate.as_posix().removeprefix("./")
    if normalized in {"", "."}:
        raise DuckDBRetryResetError(f"{noun} must not be empty")
    return normalized


def _positive_int(value: Any, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DuckDBRetryResetError(f"{noun} must be a positive integer")
    return value


def _binding_from_parameters(parameters: Mapping[str, Any]) -> RetryResetBinding:
    allowed = {
        "task_source_kind",
        "database_path",
        "plan_root_cid",
        "task_source_repository_tree_id",
        "repository_head_commit",
        "task_cid",
        "task_alias",
        "task_revision",
        "expected_status",
        "reopen_status",
        "writer_id",
        "writer_fencing_token",
        "lanes",
        "lifecycle_owner_paths",
    }
    unknown = set(parameters) - allowed
    if unknown:
        raise DuckDBRetryResetError(
            "retry reset parameters contain unsupported fields: "
            + ", ".join(sorted(unknown))
        )
    if parameters.get("task_source_kind") != "duckdb":
        raise DuckDBRetryResetError(
            "retry reset is only defined for task_source_kind=duckdb"
        )
    if parameters.get("reopen_status", "retrying") != "retrying":
        raise DuckDBRetryResetError("the governed reopen status is fixed at retrying")
    expected_status = str(parameters.get("expected_status") or "")
    if expected_status not in _ELIGIBLE_STATUSES:
        raise DuckDBRetryResetError(
            "expected_status is not eligible for a quiescent retry reset"
        )
    raw_lanes = parameters.get("lanes")
    if (
        isinstance(raw_lanes, (str, bytes, Mapping))
        or not isinstance(raw_lanes, Sequence)
        or not raw_lanes
        or len(raw_lanes) > MAX_LANES
    ):
        raise DuckDBRetryResetError("lanes must contain 1..64 lane bindings")
    lanes: list[LaneBinding] = []
    for raw in raw_lanes:
        if not isinstance(raw, Mapping) or set(raw) != {
            "state_prefix",
            "state_path",
            "queue_path",
        }:
            raise DuckDBRetryResetError(
                "each lane requires exactly state_prefix, state_path, and queue_path"
            )
        prefix = str(raw.get("state_prefix") or "")
        if not _PREFIX.fullmatch(prefix):
            raise DuckDBRetryResetError("lane state_prefix is malformed")
        state_path = _relative(raw.get("state_path"), "lane state_path")
        queue_path = _relative(raw.get("queue_path"), "lane queue path")
        if Path(queue_path).parent != Path(state_path).parent:
            raise DuckDBRetryResetError(
                "lane state and queue must share one state directory"
            )
        lanes.append(LaneBinding(prefix, state_path, queue_path))
    lanes.sort(key=lambda item: (item.state_path, item.state_prefix, item.queue_path))
    if len({item.state_path for item in lanes}) != len(lanes):
        raise DuckDBRetryResetError("lane state paths must be unique")
    raw_owners = parameters.get("lifecycle_owner_paths")
    if (
        isinstance(raw_owners, (str, bytes, Mapping))
        or not isinstance(raw_owners, Sequence)
        or not raw_owners
        or len(raw_owners) > MAX_OWNER_PATHS
    ):
        raise DuckDBRetryResetError(
            "lifecycle_owner_paths must bind 1..32 master PID/status paths"
        )
    owners = tuple(
        sorted({_relative(item, "lifecycle owner path") for item in raw_owners})
    )
    plan_root_cid = str(parameters.get("plan_root_cid") or "").strip()
    source_tree_id = str(parameters.get("task_source_repository_tree_id") or "").strip()
    head_commit = str(parameters.get("repository_head_commit") or "").strip()
    task_cid = str(parameters.get("task_cid") or "").strip()
    task_alias = str(parameters.get("task_alias") or "").strip()
    writer_id = str(parameters.get("writer_id") or "").strip()
    if not all(
        (plan_root_cid, source_tree_id, head_commit, task_cid, task_alias, writer_id)
    ):
        raise DuckDBRetryResetError(
            "plan/source-tree/HEAD/task/writer bindings must be non-empty"
        )
    if not re.fullmatch(r"[0-9a-f]{40,64}", head_commit):
        raise DuckDBRetryResetError(
            "repository_head_commit must be a full Git object ID"
        )
    return RetryResetBinding(
        database_path=_relative(parameters.get("database_path"), "database path"),
        plan_root_cid=plan_root_cid,
        task_source_repository_tree_id=source_tree_id,
        repository_head_commit=head_commit,
        task_cid=task_cid,
        task_alias=task_alias,
        task_revision=_positive_int(parameters.get("task_revision"), "task_revision"),
        expected_status=expected_status,
        writer_id=writer_id,
        writer_fencing_token=_positive_int(
            parameters.get("writer_fencing_token"), "writer_fencing_token"
        ),
        lanes=tuple(lanes),
        lifecycle_owner_paths=owners,
    )


def retry_reset_expected_effect(
    *,
    repository_root: str,
    state_root: str,
    repository_id: str,
    tree_id: str,
    parameters: Mapping[str, Any],
) -> ExpectedEffect:
    """Build the sole exact effect that a trusted policy must authorize."""

    binding = _binding_from_parameters(parameters)
    material = {
        "schema": RETRY_RESET_SCHEMA,
        "repository_root": str(Path(repository_root).resolve()),
        "state_root": str(Path(state_root).resolve()),
        "repository_id": repository_id,
        "repository_tree_id": tree_id,
        "binding": binding.to_dict(),
    }
    effect_id = content_identity({"namespace": "duckdb-retry-reset-effect", **material})
    paths = {
        binding.database_path,
        str(
            Path(binding.database_path).with_name(
                f".{Path(binding.database_path).name}.lock"
            )
        ),
        ".duckdb-retry-reset.lifecycle.lock",
        "duckdb-retry-reset/journals",
        "duckdb-retry-reset/receipts",
        *binding.lifecycle_owner_paths,
        *(lane.state_path for lane in binding.lanes),
        *(lane.queue_path for lane in binding.lanes),
        *(lane.supervisor_lock_path for lane in binding.lanes),
        *(lane.supervisor_pid_path for lane in binding.lanes),
        *(lane.daemon_pid_path for lane in binding.lanes),
        *(lane.status_path for lane in binding.lanes),
    }
    return ExpectedEffect(
        effect_id=effect_id,
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource=f"duckdb-retry-reset:{effect_id}",
        paths=tuple(sorted(paths)),
        description="Reopen one exact DuckDB task and reset bound lane retry penalties",
    )


def _resolve_under(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = root / _relative(relative, "bound path")
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise DuckDBRetryResetCorruptionError(
                f"cannot inspect governed path component: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise DuckDBRetryResetCorruptionError(
                f"governed path contains a symlink: {current}"
            )
    return candidate


def _read_bounded_bytes(path: Path, noun: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise DuckDBRetryResetCorruptionError(f"{noun} is missing: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DuckDBRetryResetCorruptionError(f"{noun} is not a regular file: {path}")
    if before.st_size <= 0 or before.st_size > MAX_SIDECAR_BYTES:
        raise DuckDBRetryResetCorruptionError(f"{noun} has an invalid size: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DuckDBRetryResetCorruptionError(f"cannot open {noun}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise DuckDBRetryResetCorruptionError(
                f"{noun} changed while opening: {path}"
            )
        chunks: list[bytes] = []
        remaining = MAX_SIDECAR_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) or len(payload) != opened.st_size:
            raise DuckDBRetryResetCorruptionError(
                f"{noun} changed while reading: {path}"
            )
        try:
            final_path = path.lstat()
        except OSError as exc:
            raise DuckDBRetryResetCorruptionError(
                f"{noun} disappeared: {path}"
            ) from exc
        if (final_path.st_dev, final_path.st_ino) != (opened.st_dev, opened.st_ino):
            raise DuckDBRetryResetCorruptionError(
                f"{noun} was replaced while reading: {path}"
            )
    finally:
        os.close(descriptor)
    if not payload or len(payload) > MAX_SIDECAR_BYTES:
        raise DuckDBRetryResetCorruptionError(f"{noun} has an invalid size: {path}")
    return payload


def _assert_regular_path(path: Path, noun: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DuckDBRetryResetCorruptionError(f"{noun} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DuckDBRetryResetCorruptionError(f"{noun} is not a regular file: {path}")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value} is forbidden")


def _read_bounded_json(path: Path, noun: str) -> dict[str, Any]:
    encoded = _read_bounded_bytes(path, noun)
    try:
        payload = json.loads(
            encoded.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DuckDBRetryResetCorruptionError(f"{noun} is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise DuckDBRetryResetCorruptionError(f"{noun} must contain an object: {path}")
    return payload


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest_path(path: Path) -> str:
    return _digest_bytes(_read_bounded_bytes(path, "governed file"))


def _git_head_binding(repository_root: Path) -> tuple[str, str]:
    values: list[str] = []
    for revision in ("HEAD", "HEAD^{tree}"):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", revision],
                cwd=repository_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DuckDBRetryResetConflict("cannot resolve canonical Git HEAD") from exc
        value = (result.stdout or "").strip()
        if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", value):
            raise DuckDBRetryResetConflict("repository root lacks a canonical Git HEAD")
        values.append(value)
    return values[0], values[1]


def _parse_owner_lanes(value: Any) -> tuple[LaneBinding, ...]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or not value
        or len(value) > MAX_LANES
    ):
        raise DuckDBRetryResetAuthorizationError("owner topology lanes are malformed")
    lanes: list[LaneBinding] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "state_prefix",
            "state_path",
            "queue_path",
        }:
            raise DuckDBRetryResetAuthorizationError("owner topology lane is malformed")
        prefix = raw.get("state_prefix")
        if not isinstance(prefix, str) or not _PREFIX.fullmatch(prefix):
            raise DuckDBRetryResetAuthorizationError("owner lane prefix is malformed")
        state_path = _relative(raw.get("state_path"), "owner lane state path")
        queue_path = _relative(raw.get("queue_path"), "owner lane queue path")
        if Path(state_path).parent != Path(queue_path).parent:
            raise DuckDBRetryResetAuthorizationError(
                "owner lane paths do not share a root"
            )
        lanes.append(LaneBinding(prefix, state_path, queue_path))
    lanes.sort(key=lambda item: (item.state_path, item.state_prefix, item.queue_path))
    if len({lane.state_path for lane in lanes}) != len(lanes):
        raise DuckDBRetryResetAuthorizationError("owner topology repeats a lane state")
    return tuple(lanes)


def _load_owner_config(state_root: Path) -> RetryResetOwnerConfig:
    path = _resolve_under(state_root, RETRY_RESET_OWNER_FILE)
    try:
        root_metadata = state_root.lstat()
        owner_metadata = path.lstat()
    except OSError as exc:
        raise DuckDBRetryResetAuthorizationError(
            "owner configuration trust anchor is unavailable"
        ) from exc
    if owner_metadata.st_uid != root_metadata.st_uid or owner_metadata.st_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise DuckDBRetryResetAuthorizationError(
            "owner configuration is not owner-controlled"
        )
    payload = _read_bounded_json(path, "retry-reset owner configuration")
    expected = {
        "schema",
        "repository_root",
        "repository_id",
        "database_path",
        "task_source_repository_tree_id",
        "policy_path",
        "policy_digest",
        "lanes",
        "lifecycle_owner_paths",
    }
    if set(payload) != expected or payload.get("schema") != RETRY_RESET_OWNER_SCHEMA:
        raise DuckDBRetryResetAuthorizationError(
            "owner configuration schema is invalid"
        )
    repository_root = payload.get("repository_root")
    repository_id = payload.get("repository_id")
    source_tree = payload.get("task_source_repository_tree_id")
    policy_digest = payload.get("policy_digest")
    if (
        not isinstance(repository_root, str)
        or not repository_root
        or str(Path(repository_root).resolve()) != repository_root
        or not isinstance(repository_id, str)
        or not repository_id
        or not isinstance(source_tree, str)
        or not source_tree
        or not isinstance(policy_digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", policy_digest)
    ):
        raise DuckDBRetryResetAuthorizationError(
            "owner configuration identity is malformed"
        )
    raw_owners = payload.get("lifecycle_owner_paths")
    if (
        isinstance(raw_owners, (str, bytes, Mapping))
        or not isinstance(raw_owners, Sequence)
        or not raw_owners
        or len(raw_owners) > MAX_OWNER_PATHS
    ):
        raise DuckDBRetryResetAuthorizationError(
            "owner lifecycle topology is malformed"
        )
    owners = tuple(
        sorted({_relative(item, "owner lifecycle path") for item in raw_owners})
    )
    return RetryResetOwnerConfig(
        repository_root=repository_root,
        repository_id=repository_id,
        database_path=_relative(payload.get("database_path"), "owner database path"),
        task_source_repository_tree_id=source_tree,
        policy_path=_relative(payload.get("policy_path"), "owner policy path"),
        policy_digest=policy_digest,
        lanes=_parse_owner_lanes(payload.get("lanes")),
        lifecycle_owner_paths=owners,
    )


def _assert_owner_binding(
    request: OperationRequest,
    binding: RetryResetBinding,
    trusted_owner: RetryResetOwnerConfig,
) -> None:
    if (
        trusted_owner.repository_root != request.repository_root
        or trusted_owner.repository_id != request.repository_id
        or trusted_owner.database_path != binding.database_path
        or trusted_owner.task_source_repository_tree_id
        != binding.task_source_repository_tree_id
        or trusted_owner.lanes != binding.lanes
        or trusted_owner.lifecycle_owner_paths != binding.lifecycle_owner_paths
    ):
        raise DuckDBRetryResetAuthorizationError(
            "request does not bind the complete owner-controlled topology"
        )


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_durable(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in (path.parent, *path.parent.parents):
        try:
            metadata = parent.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise DuckDBRetryResetCorruptionError(
                f"governed write parent is a symlink: {parent}"
            )
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None and (
        stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode)
    ):
        raise DuckDBRetryResetCorruptionError(
            f"governed write target is not a regular file: {path}"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _digest_bytes(encoded)


def _strict_attempt_map(payload: Mapping[str, Any], name: str) -> dict[str, int]:
    raw = payload.get(name, {})
    if not isinstance(raw, dict):
        raise DuckDBRetryResetCorruptionError(f"task state {name} must be an object")
    result: dict[str, int] = {}
    for key, value in raw.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise DuckDBRetryResetCorruptionError(f"task state {name} is malformed")
        result[key] = value
    return result


def _nonnegative_int_value(value: Any, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DuckDBRetryResetCorruptionError(f"{noun} must be a non-negative integer")
    return value


def _finite_nonnegative_number(value: Any, noun: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DuckDBRetryResetCorruptionError(f"{noun} must be a finite number")
    if not math.isfinite(float(value)) or value < 0:
        raise DuckDBRetryResetCorruptionError(f"{noun} must be finite and non-negative")
    return value


def _strict_string_list(value: Any, noun: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise DuckDBRetryResetCorruptionError(f"{noun} must be a string list")
    if len(set(value)) != len(value):
        raise DuckDBRetryResetCorruptionError(f"{noun} contains duplicates")
    return value


def _strict_state(path: Path) -> dict[str, Any]:
    payload = _read_bounded_json(path, "lane task state")
    unknown = set(payload) - _STATE_FIELDS
    if unknown:
        raise DuckDBRetryResetCorruptionError(
            "lane task state contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    required = {
        "implementation_in_progress",
        "active_task_id",
        "active_task_cid",
        "task_identities",
        "implementation_attempts",
        "implementation_attempts_by_cid",
    }
    if not required.issubset(payload):
        raise DuckDBRetryResetCorruptionError(
            "lane task state lacks required identity fields"
        )
    _strict_attempt_map(payload, "implementation_attempts")
    _strict_attempt_map(payload, "implementation_attempts_by_cid")
    identities = payload.get("task_identities", {})
    if not isinstance(identities, dict) or any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in identities.items()
    ):
        raise DuckDBRetryResetCorruptionError("task state task_identities is malformed")
    for key, value in identities.items():
        if not key or set(value) - _TASK_IDENTITY_FIELDS:
            raise DuckDBRetryResetCorruptionError(
                "task state identity has unsupported fields"
            )
        for name, member in value.items():
            if name == "identity_version":
                _nonnegative_int_value(member, "task identity version")
            elif not isinstance(member, str):
                raise DuckDBRetryResetCorruptionError(
                    "task state identity is malformed"
                )
    if type(payload.get("implementation_in_progress")) is not bool:
        raise DuckDBRetryResetCorruptionError("task state active flag is malformed")
    string_fields = {
        name
        for name in _STATE_FIELDS
        if name.endswith("_at")
        or name.endswith("_path")
        or name.endswith("_branch")
        or name.endswith("_commit")
        or name.endswith("_error")
        or name
        in {
            "active_task_id",
            "active_task_key",
            "active_task_cid",
            "active_task_title",
            "active_task_track",
            "active_phase",
            "active_phase_detail",
            "recommended_task_id",
            "last_implementation_task_id",
            "last_implementation_task_key",
            "last_implementation_task_cid",
            "selection_idle_reason",
        }
    }
    for name in string_fields.intersection(payload):
        if not isinstance(payload[name], str):
            raise DuckDBRetryResetCorruptionError(f"task state {name} is malformed")
    list_fields = {name for name in _STATE_FIELDS if name.endswith("_task_ids")} | {
        "recommended_actions"
    }
    for name in list_fields.intersection(payload):
        _strict_string_list(payload[name], f"task state {name}")
    int_fields = {
        "active_attempt",
        "completed_count",
        "ready_count",
        "selectable_ready_count",
        "external_reserved_count",
        "assumed_completed_count",
        "eligible_ready_count",
        "strict_deprioritized_ready_count",
        "waiting_count",
        "blocked_count",
        "task_count",
        "strategy_generation",
    }
    for name in int_fields.intersection(payload):
        _nonnegative_int_value(payload[name], f"task state {name}")
    for name in ("last_implementation_returncode", "last_merge_returncode"):
        if (
            name in payload
            and payload[name] is not None
            and (isinstance(payload[name], bool) or not isinstance(payload[name], int))
        ):
            raise DuckDBRetryResetCorruptionError(f"task state {name} is malformed")
    for name in ("task_statuses", "retry_budget_repair_receipts"):
        value = payload.get(name, {})
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not key or not isinstance(member, str)
            for key, member in value.items()
        ):
            raise DuckDBRetryResetCorruptionError(f"task state {name} is malformed")
    for name in ("task_artifacts", "task_validation"):
        value = payload.get(name, {})
        if not isinstance(value, dict):
            raise DuckDBRetryResetCorruptionError(f"task state {name} is malformed")
        for key, members in value.items():
            if not isinstance(key, str) or not key:
                raise DuckDBRetryResetCorruptionError(f"task state {name} is malformed")
            _strict_string_list(members, f"task state {name}")
    if "last_proof_workflow" in payload and not isinstance(
        payload["last_proof_workflow"], dict
    ):
        raise DuckDBRetryResetCorruptionError(
            "task state last_proof_workflow is malformed"
        )
    return payload


def _strict_queue(path: Path) -> dict[str, Any]:
    payload = _read_bounded_json(path, "persistent task queue")
    if set(payload) != _QUEUE_TOP_FIELDS:
        raise DuckDBRetryResetCorruptionError(
            "persistent task queue fields are unsupported"
        )
    schema = payload.get("schema")
    if schema not in {"persistent_task_queue_v2", "persistent_task_queue_v3"}:
        raise DuckDBRetryResetCorruptionError(
            "persistent task queue schema is unsupported"
        )
    entries = payload.get("entries")
    aliases = payload.get("aliases")
    if not isinstance(entries, dict) or not isinstance(aliases, dict):
        raise DuckDBRetryResetCorruptionError(
            "persistent task queue maps are malformed"
        )
    _finite_nonnegative_number(payload.get("updated_at"), "queue updated_at")
    _nonnegative_int_value(payload.get("entry_count"), "queue entry_count")
    if payload["entry_count"] != len(entries):
        raise DuckDBRetryResetCorruptionError(
            "persistent task queue entry_count is stale"
        )
    expected_fields = set(_QUEUE_V2_ENTRY_FIELDS)
    if schema == "persistent_task_queue_v3":
        expected_fields.update(_QUEUE_V3_AUTHORITY_FIELDS)
    for key, value in entries.items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise DuckDBRetryResetCorruptionError(
                "persistent task queue entry is malformed"
            )
        if set(value) != expected_fields:
            raise DuckDBRetryResetCorruptionError(
                "persistent task queue entry fields are unsupported"
            )
        for name in (
            "task_id",
            "priority",
            "track",
            "canonical_task_cid",
            "canonical_task_key",
            "notes",
        ):
            if not isinstance(value[name], str):
                raise DuckDBRetryResetCorruptionError(
                    f"queue entry {name} is malformed"
                )
        if not value["task_id"] or not value["canonical_task_cid"]:
            raise DuckDBRetryResetCorruptionError("queue entry identity is empty")
        if value["canonical_task_cid"] != key:
            raise DuckDBRetryResetCorruptionError(
                "queue entry key/CID binding is inconsistent"
            )
        _strict_string_list(value["aliases"], "queue entry aliases")
        provenance = value["provenance"]
        if not isinstance(provenance, list) or any(
            not isinstance(item, dict)
            or any(
                not isinstance(pkey, str) or not isinstance(pvalue, str)
                for pkey, pvalue in item.items()
            )
            for item in provenance
        ):
            raise DuckDBRetryResetCorruptionError("queue entry provenance is malformed")
        for name in (
            "selection_penalty",
            "attempt_count",
            "consecutive_failures",
            "consecutive_no_change",
            "merge_failure_count",
        ):
            _nonnegative_int_value(value[name], f"queue entry {name}")
        for name in ("last_selected_at", "last_completed_at", "cooldown_until"):
            _finite_nonnegative_number(value[name], f"queue entry {name}")
        if schema == "persistent_task_queue_v3":
            for name in ("authority_renewal_key", "authority_renewal_reason"):
                if not isinstance(value[name], str):
                    raise DuckDBRetryResetCorruptionError(
                        f"queue entry {name} is malformed"
                    )
            _nonnegative_int_value(
                value["authority_renewal_failure_count"],
                "queue entry authority_renewal_failure_count",
            )
            for name in (
                "authority_renewal_last_failure_at",
                "authority_renewal_cooldown_until",
            ):
                _finite_nonnegative_number(value[name], f"queue entry {name}")
            if type(value["authority_renewal_quarantined"]) is not bool:
                raise DuckDBRetryResetCorruptionError(
                    "queue entry authority_renewal_quarantined is malformed"
                )
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or value not in entries
        for key, value in aliases.items()
    ):
        raise DuckDBRetryResetCorruptionError(
            "persistent task queue aliases are malformed"
        )
    for key, value in entries.items():
        for alias in value["aliases"]:
            if aliases.get(alias) != key:
                raise DuckDBRetryResetCorruptionError(
                    "persistent task queue alias/entry binding is inconsistent"
                )
    return payload


def _identity_matches(value: Mapping[str, Any], task_cid: str) -> bool:
    return any(
        str(value.get(name) or "") == task_cid
        for name in ("canonical_task_cid", "task_cid", "content_id")
    )


def _queue_matching_entries(
    payload: Mapping[str, Any], task_cid: str, task_alias: str
) -> list[str]:
    entries = payload["entries"]
    aliases = payload["aliases"]
    assert isinstance(entries, dict) and isinstance(aliases, dict)
    alias_target = aliases.get(task_alias)
    if alias_target is not None and alias_target != task_cid:
        raise DuckDBRetryResetCorruptionError(
            "bound display alias resolves to a different queue CID"
        )
    result: list[str] = []
    for key, raw in entries.items():
        assert isinstance(raw, dict)
        bound_aliases = set(raw["aliases"])
        canonical = raw["canonical_task_cid"]
        if (
            raw["task_id"] == task_alias or task_alias in bound_aliases
        ) and canonical != task_cid:
            raise DuckDBRetryResetCorruptionError(
                "bound queue display alias resolves to a different CID"
            )
        if canonical == task_cid:
            if raw["task_id"] != task_alias and task_alias not in bound_aliases:
                raise DuckDBRetryResetCorruptionError(
                    "bound queue CID does not carry the requested display alias"
                )
            if aliases.get(task_alias) != key:
                raise DuckDBRetryResetCorruptionError(
                    "bound queue CID/display alias map is incomplete"
                )
            result.append(key)
    return sorted(set(result))


def _state_matches(payload: Mapping[str, Any], task_cid: str) -> bool:
    by_cid = payload.get("implementation_attempts_by_cid", {})
    identities = payload.get("task_identities", {})
    return bool(
        task_cid in by_cid
        or payload.get("active_task_cid") == task_cid
        or any(_identity_matches(value, task_cid) for value in identities.values())
    )


def _validate_state_target_binding(
    payload: Mapping[str, Any], task_cid: str, task_alias: str
) -> bool:
    identities = payload.get("task_identities", {})
    assert isinstance(identities, dict)
    identity = identities.get(task_alias)
    if identity is not None:
        assert isinstance(identity, dict)
        cited = {
            str(identity.get(name) or "")
            for name in ("canonical_task_cid", "task_cid", "content_id")
            if str(identity.get(name) or "")
        }
        if cited != {task_cid}:
            raise DuckDBRetryResetCorruptionError(
                "bound display alias/CID identity sidecar is inconsistent"
            )
    for alias, raw in identities.items():
        assert isinstance(raw, dict)
        if alias != task_alias and _identity_matches(raw, task_cid):
            raise DuckDBRetryResetCorruptionError(
                "bound CID is assigned to a different display alias"
            )
    by_cid = payload.get("implementation_attempts_by_cid", {})
    attempts = payload.get("implementation_attempts", {})
    assert isinstance(by_cid, dict) and isinstance(attempts, dict)
    bound = bool(identity is not None or task_cid in by_cid or task_alias in attempts)
    if task_alias in attempts and identity is None and task_cid not in by_cid:
        raise DuckDBRetryResetCorruptionError(
            "display attempt counter lacks a canonical CID binding"
        )
    return bound


def _retry_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        name: raw.get(name, "" if name == "notes" else 0) for name in _RETRY_FIELDS
    }
    # Durable receipt CIDs use DAG-JSON, whose numeric domain is integers.
    # Queue timestamps are retained exactly as deterministic decimal text.
    cooldown = result.get("cooldown_until", 0)
    if isinstance(cooldown, float):
        result["cooldown_until"] = 0 if cooldown == 0.0 else repr(cooldown)
    return result


def _lane_before(
    *, state_path: Path, queue_path: Path, task_cid: str, task_alias: str
) -> dict[str, Any]:
    state = _strict_state(state_path)
    queue = _strict_queue(queue_path)
    matches = _queue_matching_entries(queue, task_cid, task_alias)
    state_bound = _validate_state_target_binding(state, task_cid, task_alias)
    attempts = _strict_attempt_map(state, "implementation_attempts")
    by_cid = _strict_attempt_map(state, "implementation_attempts_by_cid")
    return {
        "matched": bool(state_bound or matches),
        "state_bound": state_bound,
        "state_digest": _digest_path(state_path),
        "queue_digest": _digest_path(queue_path),
        "display_attempt_count": attempts.get(task_alias, 0),
        "canonical_attempt_count": by_cid.get(task_cid, 0),
        "queue_entries": [
            {
                "entry_key": key,
                "retry": _retry_projection(queue["entries"][key]),
                "attempt_count": int(queue["entries"][key].get("attempt_count", 0)),
            }
            for key in matches
        ],
    }


def _reset_lane(
    *, state_path: Path, queue_path: Path, task_cid: str, task_alias: str
) -> dict[str, Any]:
    state = _strict_state(state_path)
    queue = _strict_queue(queue_path)
    matches = _queue_matching_entries(queue, task_cid, task_alias)
    matched = _validate_state_target_binding(state, task_cid, task_alias) or bool(
        matches
    )
    if matched:
        attempts = _strict_attempt_map(state, "implementation_attempts")
        by_cid = _strict_attempt_map(state, "implementation_attempts_by_cid")
        attempts.pop(task_alias, None)
        by_cid.pop(task_cid, None)
        state["implementation_attempts"] = attempts
        state["implementation_attempts_by_cid"] = by_cid
        for key in matches:
            raw = queue["entries"][key]
            before_attempts = int(raw.get("attempt_count", 0))
            for field in _RETRY_FIELDS:
                raw[field] = "" if field == "notes" else 0
            if raw["attempt_count"] != before_attempts:
                raise DuckDBRetryResetCorruptionError(
                    "retry reset attempted to alter lifetime queue history"
                )
        queue["entry_count"] = len(queue["entries"])
        _write_json_durable(state_path, state)
        _write_json_durable(queue_path, queue)
    after_state = _strict_state(state_path)
    after_queue = _strict_queue(queue_path)
    after_matches = _queue_matching_entries(after_queue, task_cid, task_alias)
    return {
        "matched": matched,
        "state_digest": _digest_path(state_path),
        "queue_digest": _digest_path(queue_path),
        "display_attempt_count": _strict_attempt_map(
            after_state, "implementation_attempts"
        ).get(task_alias, 0),
        "canonical_attempt_count": _strict_attempt_map(
            after_state, "implementation_attempts_by_cid"
        ).get(task_cid, 0),
        "queue_entries": [
            {
                "entry_key": key,
                "retry": _retry_projection(after_queue["entries"][key]),
                "attempt_count": int(
                    after_queue["entries"][key].get("attempt_count", 0)
                ),
            }
            for key in after_matches
        ],
    }


def _pids_from_payload(value: Any, *, key: str = "") -> set[int]:
    result: set[int] = set()
    if isinstance(value, Mapping):
        for name, member in value.items():
            result.update(_pids_from_payload(member, key=str(name)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if key.endswith("pids") or key.endswith("_pids"):
            for member in value:
                try:
                    result.add(int(member))
                except (TypeError, ValueError):
                    continue
    elif not key or key == "pid" or key.endswith("_pid"):
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            pass
    return {item for item in result if item > 0}


def _live_pids(path: Path) -> tuple[int, ...]:
    try:
        path.lstat()
    except FileNotFoundError:
        return ()
    try:
        text = (
            _read_bounded_bytes(path, "lifecycle owner record").decode("utf-8").strip()
        )
    except (DuckDBRetryResetCorruptionError, UnicodeError) as exc:
        raise DuckDBRetryResetQuiescenceError(
            f"cannot inspect owner path: {path}"
        ) from exc
    if not text:
        raise DuckDBRetryResetQuiescenceError(f"owner path is empty: {path}")
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        try:
            value = {"pid": int(text)}
        except ValueError as exc:
            raise DuckDBRetryResetQuiescenceError(
                f"owner path is neither a PID nor JSON: {path}"
            ) from exc
    return tuple(sorted(pid for pid in _pids_from_payload(value) if pid_alive(pid)))


def _assert_quiescent(
    state_root: Path,
    binding: RetryResetBinding,
    lane_states: Mapping[str, Mapping[str, Any]],
) -> None:
    owner_paths = set(binding.lifecycle_owner_paths)
    for lane in binding.lanes:
        owner_paths.update(
            {lane.supervisor_pid_path, lane.daemon_pid_path, lane.status_path}
        )
    live: dict[str, tuple[int, ...]] = {}
    for relative in sorted(owner_paths):
        pids = _live_pids(_resolve_under(state_root, relative))
        if pids:
            live[relative] = pids
    if live:
        raise DuckDBRetryResetQuiescenceError(
            "declared lifecycle owner is live: "
            + ", ".join(f"{path}={list(pids)}" for path, pids in live.items())
        )
    for lane in binding.lanes:
        state = lane_states[lane.state_path]
        active_alias = state.get("active_task_id") == binding.task_alias
        active_cid = state.get("active_task_cid") == binding.task_cid
        if active_alias and state.get("active_task_cid") not in {"", binding.task_cid}:
            raise DuckDBRetryResetCorruptionError(
                f"active display alias/CID disagree in lane {lane.state_prefix}"
            )
        if active_cid and state.get("active_task_id") not in {"", binding.task_alias}:
            raise DuckDBRetryResetCorruptionError(
                f"active CID/display alias disagree in lane {lane.state_prefix}"
            )
        active_target = active_alias or active_cid
        if active_target or (
            state.get("implementation_in_progress")
            and _validate_state_target_binding(
                state, binding.task_cid, binding.task_alias
            )
        ):
            raise DuckDBRetryResetQuiescenceError(
                f"bound task is active in lane {lane.state_prefix}"
            )


def _assert_no_undeclared_matching_lanes(
    state_root: Path, binding: RetryResetBinding
) -> None:
    """Fail if a task-bearing lane sidecar is absent from trusted topology."""

    bound_states = {
        _resolve_under(state_root, lane.state_path) for lane in binding.lanes
    }
    bound_queues = {
        _resolve_under(state_root, lane.queue_path) for lane in binding.lanes
    }
    state_candidates = sorted(state_root.rglob("*_task_state.json"))
    queue_candidates = sorted(state_root.rglob("*task_queue*.json"))
    if len(state_candidates) + len(queue_candidates) > MAX_DISCOVERED_SIDECARS:
        raise DuckDBRetryResetCorruptionError(
            "lane sidecar discovery exceeds its governed bound"
        )
    undeclared: list[str] = []
    for path in state_candidates:
        if path in bound_states:
            continue
        payload = _read_bounded_json(path, "discovered lane task state")
        identities = payload.get("task_identities")
        by_cid = payload.get("implementation_attempts_by_cid")
        attempts = payload.get("implementation_attempts")
        matches = bool(
            payload.get("active_task_cid") == binding.task_cid
            or payload.get("active_task_id") == binding.task_alias
            or isinstance(by_cid, Mapping)
            and binding.task_cid in by_cid
            or isinstance(attempts, Mapping)
            and binding.task_alias in attempts
            or isinstance(identities, Mapping)
            and any(
                isinstance(value, Mapping)
                and _identity_matches(value, binding.task_cid)
                for value in identities.values()
            )
        )
        if matches:
            undeclared.append(str(path))
    for path in queue_candidates:
        if path in bound_queues:
            continue
        payload = _read_bounded_json(path, "discovered persistent task queue")
        entries = payload.get("entries")
        aliases = payload.get("aliases")
        matches = bool(
            isinstance(entries, Mapping)
            and (
                binding.task_cid in entries
                or any(
                    isinstance(value, Mapping)
                    and str(value.get("canonical_task_cid") or "") == binding.task_cid
                    for value in entries.values()
                )
            )
            or isinstance(aliases, Mapping)
            and aliases.get(binding.task_alias) == binding.task_cid
        )
        if matches:
            _strict_queue(path)
            undeclared.append(str(path))
    if undeclared:
        raise DuckDBRetryResetAuthorizationError(
            "task appears in lanes absent from owner topology: "
            + ", ".join(sorted(undeclared))
        )


def _assert_no_completed_descendant(source: DuckDBTaskSource, task_cid: str) -> None:
    projection = source.read_consistent_projection(("tasks", "task_dependencies"))
    statuses = {
        str(row["task_cid"]): str(row["status"]) for row in projection.tables["tasks"]
    }
    children: dict[str, set[str]] = {}
    for edge in projection.tables["task_dependencies"]:
        children.setdefault(str(edge["dependency_task_cid"]), set()).add(
            str(edge["task_cid"])
        )
    pending = list(children.get(task_cid, ()))
    seen: set[str] = set()
    completed: list[str] = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if statuses.get(current) in {"completed", "skipped"}:
            completed.append(current)
        pending.extend(children.get(current, ()))
    if completed:
        raise DuckDBRetryResetConflict(
            "completed descendants prevent a non-cascading reopen: "
            + ", ".join(sorted(completed))
        )


def _journal_key_cid(idempotency: Mapping[str, Any]) -> str:
    return content_identity(
        {
            "namespace": "duckdb-retry-reset-idempotency",
            "key": dict(idempotency),
        }
    )


def _journal_paths(state_root: Path, request: OperationRequest) -> tuple[Path, Path]:
    assert request.idempotency is not None
    key_cid = _journal_key_cid(request.idempotency.to_dict())
    journals = _resolve_under(state_root, "duckdb-retry-reset/journals")
    receipts = _resolve_under(state_root, "duckdb-retry-reset/receipts")
    return journals / f"{key_cid}.json", receipts


def _load_policy(
    path: Path, *, expected_digest: str | None = None
) -> ControlMutationPolicy:
    encoded = _read_bounded_bytes(path, "trusted mutation policy")
    if expected_digest is not None and _digest_bytes(encoded) != expected_digest:
        raise DuckDBRetryResetAuthorizationError(
            "configured mutation policy digest does not match the owner trust anchor"
        )
    try:
        payload = json.loads(
            encoded.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DuckDBRetryResetAuthorizationError(
            "trusted mutation policy is malformed"
        ) from exc
    if not isinstance(payload, dict):
        raise DuckDBRetryResetAuthorizationError(
            "trusted mutation policy must be an object"
        )
    allowed = {
        "schema",
        "policy_id",
        "policy_revision",
        "permits",
        "current_tree_ids",
        "current_objective_revisions",
        "active_lease_fences",
    }
    if payload.get("schema") != RETRY_RESET_POLICY_SCHEMA or set(payload) - allowed:
        raise DuckDBRetryResetAuthorizationError(
            "trusted mutation policy schema is invalid"
        )
    try:
        permits = tuple(
            AuthorizationDecision.from_dict(item) for item in payload["permits"]
        )
        return ControlMutationPolicy(
            policy_id=payload["policy_id"],
            policy_revision=payload["policy_revision"],
            permits=permits,
            current_tree_ids=payload["current_tree_ids"],
            current_objective_revisions=payload["current_objective_revisions"],
            active_lease_fences=payload["active_lease_fences"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DuckDBRetryResetAuthorizationError(
            "trusted mutation policy is malformed"
        ) from exc


def _binding_from_journal(value: Any) -> RetryResetBinding:
    if not isinstance(value, Mapping):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal binding is malformed"
        )
    material = dict(value)
    if material.pop("schema", None) != RETRY_RESET_SCHEMA:
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal binding schema is invalid"
        )
    raw_lanes = material.get("lanes")
    if not isinstance(raw_lanes, list):
        raise DuckDBRetryResetCorruptionError("retry-reset journal lanes are malformed")
    material["lanes"] = [
        {
            "state_prefix": lane.get("state_prefix"),
            "state_path": lane.get("state_path"),
            "queue_path": lane.get("queue_path"),
        }
        for lane in raw_lanes
        if isinstance(lane, Mapping)
    ]
    if len(material["lanes"]) != len(raw_lanes):
        raise DuckDBRetryResetCorruptionError("retry-reset journal lane is malformed")
    material["task_source_kind"] = "duckdb"
    try:
        binding = _binding_from_parameters(material)
    except DuckDBRetryResetError as exc:
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal binding is malformed"
        ) from exc
    if binding.to_dict() != dict(value):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal contains a non-canonical binding"
        )
    return binding


def _verified_receipt(journal: Mapping[str, Any], receipt_root: Path) -> dict[str, Any]:
    receipt_cid = journal.get("receipt_cid")
    if not isinstance(receipt_cid, str) or not receipt_cid:
        raise DuckDBRetryResetCorruptionError("retry-reset journal lacks a receipt CID")
    expected_path = receipt_root / f"{receipt_cid}.json"
    if journal.get("receipt_path") != str(expected_path):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset receipt path is not canonical"
        )
    receipt = _read_bounded_json(expected_path, "retry-reset receipt")
    claimed = receipt.get("receipt_cid")
    material = {key: value for key, value in receipt.items() if key != "receipt_cid"}
    if claimed != receipt_cid or claimed != content_identity(material):
        raise DuckDBRetryResetCorruptionError("retry-reset receipt identity is invalid")
    if (
        receipt.get("schema") != RETRY_RESET_RECEIPT_SCHEMA
        or receipt.get("request_id") != journal.get("request_id")
        or receipt.get("intent_cid") != journal.get("intent_cid")
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset receipt/journal binding changed"
        )
    return receipt


def _find_event_by_cid(
    source: DuckDBTaskSource, event_cid: str
) -> dict[str, Any] | None:
    cursor = 0
    scanned = 0
    while scanned < 100_000:
        page = source.events(cursor=cursor, limit=1_000)
        if not page.events:
            return None
        for event in page.events:
            scanned += 1
            if event["event_cid"] == event_cid:
                return dict(event)
        if page.cursor <= cursor:
            raise DuckDBRetryResetCorruptionError("task event cursor did not advance")
        cursor = page.cursor
        if len(page.events) < 1_000:
            return None
    raise DuckDBRetryResetCorruptionError(
        "task event history exceeds its governed bound"
    )


def _verify_completed_journal(
    *, state_root: Path, path: Path, journal: Mapping[str, Any]
) -> dict[str, Any]:
    binding = _binding_from_journal(journal.get("binding"))
    idempotency = journal.get("idempotency")
    if not isinstance(idempotency, Mapping):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal idempotency is malformed"
        )
    expected_key = _journal_key_cid(idempotency)
    if path.stem != expected_key or journal.get("journal_key_cid") != expected_key:
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal filename is not content-bound"
        )
    if journal.get("phase") != "completed":
        raise DuckDBRetryResetCorruptionError("retry-reset journal is not completed")
    receipt_root = _resolve_under(state_root, "duckdb-retry-reset/receipts")
    receipt = _verified_receipt(journal, receipt_root)
    database_before = journal.get("database_before")
    database_after = journal.get("database_after")
    lane_before = journal.get("lane_before")
    lane_after = journal.get("lane_after")
    if not all(
        isinstance(item, Mapping)
        for item in (database_before, database_after, lane_before, lane_after)
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal transition records are missing"
        )
    expected_intent_cid = content_identity(
        {
            "namespace": "duckdb-retry-reset-intent",
            "request_id": journal.get("request_id"),
            "binding": binding.to_dict(),
            "database_before": dict(database_before),
            "lane_before": dict(lane_before),
        }
    )
    if journal.get("intent_cid") != expected_intent_cid:
        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal intent identity is invalid"
        )
    receipt_binding = {
        "repository_id": journal.get("repository_id"),
        "repository_tree_id": journal.get("repository_tree_id"),
        "repository_head_commit": binding.repository_head_commit,
        "task_source_repository_tree_id": binding.task_source_repository_tree_id,
        "plan_root_cid": binding.plan_root_cid,
        "task_cid": binding.task_cid,
        "task_alias": binding.task_alias,
        "task_revision_before": binding.task_revision,
        "task_revision_after": database_after.get("task_revision"),
        "status_before": binding.expected_status,
        "status_after": "retrying",
        "status_changed": bool(database_after.get("status_changed")),
        "writer_id": binding.writer_id,
        "writer_fencing_token": binding.writer_fencing_token,
        "intent_cid": expected_intent_cid,
        "status_receipt_cid": database_after.get("status_receipt_cid"),
    }
    if any(receipt.get(key) != value for key, value in receipt_binding.items()):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset receipt binding does not match its journal"
        )
    receipt_lanes = receipt.get("lanes")
    if not isinstance(receipt_lanes, list) or len(receipt_lanes) != len(binding.lanes):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset receipt lane topology is invalid"
        )
    lanes_by_state = {
        str(item.get("state_path") or ""): item
        for item in receipt_lanes
        if isinstance(item, Mapping)
    }
    if len(lanes_by_state) != len(receipt_lanes):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset receipt lane identities are invalid"
        )
    for lane in binding.lanes:
        before = lane_before.get(lane.state_path)
        after = lane_after.get(lane.state_path)
        recorded = lanes_by_state.get(lane.state_path)
        if not all(isinstance(item, Mapping) for item in (before, after, recorded)):
            raise DuckDBRetryResetCorruptionError(
                "retry-reset receipt lane transition is missing"
            )
        expected_lane = {
            **lane.to_dict(),
            "matched": bool(before.get("matched")),
            "state_digest_before": before.get("state_digest"),
            "state_digest_after": after.get("state_digest"),
            "queue_digest_before": before.get("queue_digest"),
            "queue_digest_after": after.get("queue_digest"),
            "display_attempt_count_before": before.get("display_attempt_count"),
            "display_attempt_count_after": after.get("display_attempt_count"),
            "canonical_attempt_count_before": before.get("canonical_attempt_count"),
            "canonical_attempt_count_after": after.get("canonical_attempt_count"),
            "queue_entries_before": before.get("queue_entries"),
            "queue_entries_after": after.get("queue_entries"),
        }
        if dict(recorded) != expected_lane:
            raise DuckDBRetryResetCorruptionError(
                "retry-reset receipt lane evidence does not match its journal"
            )
    database_path = _resolve_under(state_root, binding.database_path)
    _assert_regular_path(database_path, "DuckDB task source")
    source = DuckDBTaskSource(
        database_path,
        expected_plan_root_cid=binding.plan_root_cid,
        expected_repository_tree_id=binding.task_source_repository_tree_id,
        writer_id=binding.writer_id,
        fencing_token=binding.writer_fencing_token,
    )
    recorded_event = journal.get("completion_event")
    if not isinstance(recorded_event, Mapping):
        raise DuckDBRetryResetCorruptionError("retry-reset completion event is missing")
    event_cid = recorded_event.get("event_cid")
    if event_cid != receipt.get("receipt_cid"):
        raise DuckDBRetryResetCorruptionError(
            "retry-reset completion event CID changed"
        )
    durable = _find_event_by_cid(source, str(event_cid or ""))
    if durable is None:
        raise DuckDBRetryResetCorruptionError(
            "durable retry-reset completion event is missing"
        )
    body = durable.get("body")
    if (
        durable.get("event_type") != "retry_reset_completed"
        or durable.get("task_cid") != binding.task_cid
        or not isinstance(body, Mapping)
        or body.get("receipt_cid") != receipt.get("receipt_cid")
        or body.get("request_id") != journal.get("request_id")
        or body.get("intent_cid") != journal.get("intent_cid")
        or body.get("writer_id") != binding.writer_id
        or body.get("writer_fencing_token") != binding.writer_fencing_token
        or body.get("plan_root_cid") != binding.plan_root_cid
        or body.get("repository_id") != journal.get("repository_id")
        or body.get("repository_tree_id") != journal.get("repository_tree_id")
        or body.get("repository_head_commit") != binding.repository_head_commit
        or body.get("task_source_repository_tree_id")
        != binding.task_source_repository_tree_id
    ):
        raise DuckDBRetryResetCorruptionError(
            "durable retry-reset event binding is invalid"
        )
    if any(
        recorded_event.get(name) != durable.get(name)
        for name in ("event_cid", "sequence", "revision")
    ):
        raise DuckDBRetryResetCorruptionError(
            "recorded completion event does not match history"
        )
    intent = _find_status_event(
        source,
        cursor=int(database_before.get("event_cursor", -1)),
        request_id=str(journal.get("request_id") or ""),
        intent_cid=str(journal.get("intent_cid") or ""),
    )
    if intent is None:
        raise DuckDBRetryResetCorruptionError(
            "durable retry-reset intent event is missing"
        )
    if intent.get("task_cid") != binding.task_cid:
        raise DuckDBRetryResetCorruptionError(
            "durable retry-reset intent task binding is invalid"
        )
    return receipt


def inspect_incomplete_retry_resets(
    state_root: str | os.PathLike[str],
) -> tuple[dict[str, Any], ...]:
    """Verify every reset journal and return non-completed launch blockers."""

    state = Path(state_root).resolve()
    root = _resolve_under(state, "duckdb-retry-reset/journals")
    if not root.exists():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise DuckDBRetryResetCorruptionError("retry-reset journal root is unsafe")
    result: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = _read_bounded_json(path, "retry-reset journal")
        if payload.get("schema") != RETRY_RESET_JOURNAL_SCHEMA:
            raise DuckDBRetryResetCorruptionError(
                f"foreign retry-reset journal: {path}"
            )
        if payload.get("phase") not in _PHASES:
            raise DuckDBRetryResetCorruptionError(
                f"retry-reset journal phase is invalid: {path}"
            )
        _binding_from_journal(payload.get("binding"))
        idempotency = payload.get("idempotency")
        if (
            not isinstance(idempotency, Mapping)
            or _journal_key_cid(idempotency) != path.stem
            or payload.get("journal_key_cid") != path.stem
        ):
            raise DuckDBRetryResetCorruptionError(
                f"retry-reset journal filename is not content-bound: {path}"
            )
        if payload.get("phase") == "completed":
            _verify_completed_journal(state_root=state, path=path, journal=payload)
        else:
            result.append(
                {
                    "path": str(path),
                    "phase": payload["phase"],
                    "request_id": payload.get("request_id", ""),
                    "task_cid": payload.get("binding", {}).get("task_cid", ""),
                }
            )
    return tuple(result)


def _find_status_event(
    source: DuckDBTaskSource, *, cursor: int, request_id: str, intent_cid: str
) -> dict[str, Any] | None:
    selected_cursor = cursor
    scanned = 0
    while scanned < 100_000:
        page = source.events(cursor=selected_cursor, limit=1_000)
        if not page.events:
            break
        for event in page.events:
            scanned += 1
            body = event["body"]
            receipt = body.get("receipt", {}) if isinstance(body, Mapping) else {}
            material = receipt if event["event_type"] == "status_changed" else body
            if (
                event["event_type"] in {"status_changed", "retry_reset_intent"}
                and isinstance(material, Mapping)
                and material.get("request_id") == request_id
                and material.get("intent_cid") == intent_cid
            ):
                return dict(event)
        if page.cursor <= selected_cursor:
            raise DuckDBRetryResetCorruptionError("task event cursor did not advance")
        selected_cursor = page.cursor
        if len(page.events) < 1_000:
            break
    return None


def execute_duckdb_retry_reset(
    request: OperationRequest,
    *,
    trusted_policy: ControlMutationPolicy,
    trusted_owner: RetryResetOwnerConfig,
    clock_ms: Callable[[], int] | None = None,
    lock_timeout_seconds: float = 30.0,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute or recover one exact authorized retry reset."""

    if request.operation is not Operation.RETRY or request.dry_run:
        raise DuckDBRetryResetError(
            "retry reset requires a real Operation.RETRY request"
        )
    now_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
    try:
        ControlMutationAuthorizer(trusted_policy, clock_ms=now_ms).validate(request)
    except Exception as exc:
        raise DuckDBRetryResetAuthorizationError(str(exc)) from exc
    binding = _binding_from_parameters(request.parameters)
    _assert_owner_binding(request, binding, trusted_owner)
    if request.fencing_epoch != binding.writer_fencing_token:
        raise DuckDBRetryResetAuthorizationError(
            "request fencing_epoch does not bind the DuckDB writer fence"
        )
    decision = request.authorization
    assert decision is not None
    required_grants = {RETRY_RESET_GRANT, f"grant:duckdb-writer:{binding.writer_id}"}
    if not required_grants.issubset(decision.grant_ids):
        raise DuckDBRetryResetAuthorizationError(
            "permit lacks retry-reset writer grants"
        )
    expected_effect = retry_reset_expected_effect(
        repository_root=request.repository_root,
        state_root=request.state_root,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        parameters=request.parameters,
    )
    if request.expected_effects != (expected_effect,):
        raise DuckDBRetryResetAuthorizationError(
            "request effect is not the exact retry-reset binding"
        )

    repository_root = Path(request.repository_root).resolve()
    state_root = Path(request.state_root).resolve()
    if (
        str(repository_root) != request.repository_root
        or str(state_root) != request.state_root
    ):
        raise DuckDBRetryResetConflict("request roots are not canonical resolved paths")
    if checkout_repository_id(repository_root) != request.repository_id:
        raise DuckDBRetryResetConflict("repository root does not match repository_id")
    head_commit, head_tree = _git_head_binding(repository_root)
    if head_commit != binding.repository_head_commit or head_tree != request.tree_id:
        raise DuckDBRetryResetConflict(
            "request does not bind the repository's current HEAD commit and tree"
        )
    database_path = _resolve_under(state_root, binding.database_path)
    _assert_regular_path(database_path, "DuckDB task source")
    journal_path, receipt_root = _journal_paths(state_root, request)
    lifecycle_lock = _resolve_under(state_root, ".duckdb-retry-reset.lifecycle.lock")
    _resolve_under(
        state_root,
        str(
            Path(binding.database_path).with_name(
                f".{Path(binding.database_path).name}.lock"
            )
        ),
    )
    lane_lock_paths = sorted(
        {
            _resolve_under(state_root, lane.supervisor_lock_path)
            for lane in binding.lanes
        },
        key=lambda path: str(path),
    )

    with ExitStack() as locks:
        locks.enter_context(
            exclusive_file_lock(lifecycle_lock, timeout_seconds=lock_timeout_seconds)
        )
        for path in lane_lock_paths:
            locks.enter_context(
                exclusive_file_lock(path, timeout_seconds=lock_timeout_seconds)
            )
        lane_states = {
            lane.state_path: _strict_state(_resolve_under(state_root, lane.state_path))
            for lane in binding.lanes
        }
        for lane in binding.lanes:
            _strict_queue(_resolve_under(state_root, lane.queue_path))
        _assert_quiescent(state_root, binding, lane_states)
        _assert_no_undeclared_matching_lanes(state_root, binding)

        source = DuckDBTaskSource(
            database_path,
            expected_plan_root_cid=binding.plan_root_cid,
            expected_repository_tree_id=binding.task_source_repository_tree_id,
            writer_id=binding.writer_id,
            fencing_token=binding.writer_fencing_token,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        snapshot = source.snapshot()
        if (
            snapshot.plan_root_cid != binding.plan_root_cid
            or snapshot.repository_tree_id != binding.task_source_repository_tree_id
        ):
            raise DuckDBRetryResetConflict("DuckDB plan/repository binding changed")
        writer = source.current_writer_fence()
        if (writer.writer_id, writer.fencing_token) != (
            binding.writer_id,
            binding.writer_fencing_token,
        ):
            raise DuckDBRetryResetConflict("DuckDB writer owner/fence is stale")
        task = source.get_task(binding.task_cid)
        if task is None or task.task_alias != binding.task_alias:
            raise DuckDBRetryResetConflict(
                "task CID/alias binding does not resolve exactly"
            )

        journal: dict[str, Any] | None = None
        if journal_path.exists():
            journal = _read_bounded_json(journal_path, "retry-reset journal")
            if (
                journal.get("schema") != RETRY_RESET_JOURNAL_SCHEMA
                or journal.get("request_id") != request.request_id
                or journal.get("idempotency") != request.idempotency.to_dict()
                or journal.get("journal_key_cid") != journal_path.stem
                or journal.get("binding") != binding.to_dict()
                or journal.get("phase") not in _PHASES
                or journal.get("repository_id") != request.repository_id
                or journal.get("repository_tree_id") != request.tree_id
                or journal.get("repository_head_commit")
                != binding.repository_head_commit
                or journal.get("task_source_repository_tree_id")
                != binding.task_source_repository_tree_id
            ):
                raise DuckDBRetryResetConflict(
                    "idempotency journal belongs to another request"
                )
            if journal["phase"] == "completed":
                return _verify_completed_journal(
                    state_root=state_root,
                    path=journal_path,
                    journal=journal,
                )
        else:
            if (
                task.revision != binding.task_revision
                or task.status != binding.expected_status
            ):
                raise DuckDBRetryResetConflict("task status/revision is stale")
            _assert_no_completed_descendant(source, binding.task_cid)
            lane_before: dict[str, Any] = {}
            for lane in binding.lanes:
                lane_before[lane.state_path] = {
                    **lane.to_dict(),
                    **_lane_before(
                        state_path=_resolve_under(state_root, lane.state_path),
                        queue_path=_resolve_under(state_root, lane.queue_path),
                        task_cid=binding.task_cid,
                        task_alias=binding.task_alias,
                    ),
                }
            if not any(item["matched"] for item in lane_before.values()):
                raise DuckDBRetryResetConflict(
                    "no configured lane contains the bound task CID"
                )
            journal = {
                "schema": RETRY_RESET_JOURNAL_SCHEMA,
                "phase": "prepared",
                "request_id": request.request_id,
                "idempotency_key": request.idempotency_key,
                "idempotency": request.idempotency.to_dict(),
                "journal_key_cid": journal_path.stem,
                "authorization_decision_id": decision.decision_id,
                "policy_id": trusted_policy.policy_id,
                "policy_revision": trusted_policy.policy_revision,
                "repository_id": request.repository_id,
                "repository_tree_id": request.tree_id,
                "repository_head_commit": binding.repository_head_commit,
                "task_source_repository_tree_id": binding.task_source_repository_tree_id,
                "binding": binding.to_dict(),
                "database_before": {
                    "revision": snapshot.revision,
                    "event_cursor": snapshot.event_cursor,
                    "task_status": task.status,
                    "task_revision": task.revision,
                    "writer_id": writer.writer_id,
                    "writer_fencing_token": writer.fencing_token,
                },
                "lane_before": lane_before,
            }
            _write_json_durable(journal_path, journal)
            if fault_injector:
                fault_injector("prepared")

        assert journal is not None
        intent_cid = content_identity(
            {
                "namespace": "duckdb-retry-reset-intent",
                "request_id": request.request_id,
                "binding": binding.to_dict(),
                "database_before": journal["database_before"],
                "lane_before": journal["lane_before"],
            }
        )
        if journal["phase"] == "prepared":
            task = source.get_task(binding.task_cid)
            assert task is not None
            if (
                task.status == binding.expected_status
                and task.revision == binding.task_revision
            ):
                intent_receipt = {
                    "schema": RETRY_RESET_EVENT_SCHEMA,
                    "operation": "retry_reset_intent",
                    "request_id": request.request_id,
                    "intent_cid": intent_cid,
                    "authorization_decision_id": decision.decision_id,
                    "plan_root_cid": binding.plan_root_cid,
                    "repository_id": request.repository_id,
                    "repository_tree_id": request.tree_id,
                    "repository_head_commit": binding.repository_head_commit,
                    "task_source_repository_tree_id": binding.task_source_repository_tree_id,
                    "task_cid": binding.task_cid,
                    "task_alias": binding.task_alias,
                    "expected_task_revision": binding.task_revision,
                    "writer_id": binding.writer_id,
                    "writer_fencing_token": binding.writer_fencing_token,
                }
                if binding.expected_status == "retrying":
                    appended = source.append_event(
                        {
                            **intent_receipt,
                            "event_cid": intent_cid,
                            "event_type": "retry_reset_intent",
                            "task_cid": binding.task_cid,
                        },
                        lease={
                            "lease_id": request.lease_id,
                            "fencing_token": binding.writer_fencing_token,
                        },
                        fence=binding.writer_fencing_token,
                        writer_id=binding.writer_id,
                    )
                    database_after = {
                        "task_status": task.status,
                        "task_revision": task.revision,
                        "revision": int(appended["revision"]),
                        "event_cursor": int(appended["sequence"]),
                        "status_receipt_cid": str(appended["event_cid"]),
                        "status_changed": False,
                    }
                else:
                    cas = source.compare_and_set_status(
                        binding.task_cid,
                        binding.task_revision,
                        "retrying",
                        intent_receipt,
                    )
                    database_after = {
                        "task_status": cas.task.status,
                        "task_revision": cas.task.revision,
                        "revision": cas.revision,
                        "event_cursor": cas.event_cursor,
                        "status_receipt_cid": cas.receipt_cid,
                        "status_changed": True,
                    }
            elif (
                task.status == "retrying" and task.revision == binding.task_revision + 1
            ):
                event = _find_status_event(
                    source,
                    cursor=int(journal["database_before"]["event_cursor"]),
                    request_id=request.request_id,
                    intent_cid=intent_cid,
                )
                if event is None:
                    raise DuckDBRetryResetConflict(
                        "retrying task is not bound to this durable reset intent"
                    )
                event_receipt = event["body"].get("receipt", {})
                database_after = {
                    "task_status": task.status,
                    "task_revision": task.revision,
                    "revision": int(event["revision"]),
                    "event_cursor": int(event["sequence"]),
                    "status_receipt_cid": content_identity(
                        {
                            "namespace": "task-status-receipt",
                            "event_cid": event["event_cid"],
                            "receipt": event_receipt,
                        }
                    ),
                    "status_changed": True,
                }
            else:
                raise DuckDBRetryResetConflict(
                    "task changed after retry-reset preparation"
                )
            if fault_injector:
                fault_injector("database_mutated")
            journal["database_after"] = database_after
            journal["intent_cid"] = intent_cid
            journal["phase"] = "database_committed"
            _write_json_durable(journal_path, journal)
            if fault_injector:
                fault_injector("database_committed")

        if journal["phase"] == "database_committed":
            lane_after: dict[str, Any] = {}
            queue_results: dict[str, dict[str, Any]] = {}
            for lane in binding.lanes:
                if lane.queue_path not in queue_results:
                    queue_results[lane.queue_path] = _reset_lane(
                        state_path=_resolve_under(state_root, lane.state_path),
                        queue_path=_resolve_under(state_root, lane.queue_path),
                        task_cid=binding.task_cid,
                        task_alias=binding.task_alias,
                    )
                    lane_after[lane.state_path] = queue_results[lane.queue_path]
                else:
                    lane_after[lane.state_path] = _reset_lane(
                        state_path=_resolve_under(state_root, lane.state_path),
                        queue_path=_resolve_under(state_root, lane.queue_path),
                        task_cid=binding.task_cid,
                        task_alias=binding.task_alias,
                    )
                if fault_injector:
                    fault_injector(f"lane_sidecar_mutated:{lane.state_prefix}")
            journal["lane_after"] = lane_after
            journal["phase"] = "sidecars_committed"
            _write_json_durable(journal_path, journal)
            if fault_injector:
                fault_injector("sidecars_committed")

        if journal["phase"] == "sidecars_committed":
            lanes_receipt = []
            for lane in binding.lanes:
                before = journal["lane_before"][lane.state_path]
                after = journal["lane_after"][lane.state_path]
                lanes_receipt.append(
                    {
                        **lane.to_dict(),
                        "matched": bool(before["matched"]),
                        "state_digest_before": before["state_digest"],
                        "state_digest_after": after["state_digest"],
                        "queue_digest_before": before["queue_digest"],
                        "queue_digest_after": after["queue_digest"],
                        "display_attempt_count_before": before["display_attempt_count"],
                        "display_attempt_count_after": after["display_attempt_count"],
                        "canonical_attempt_count_before": before[
                            "canonical_attempt_count"
                        ],
                        "canonical_attempt_count_after": after[
                            "canonical_attempt_count"
                        ],
                        "queue_entries_before": before["queue_entries"],
                        "queue_entries_after": after["queue_entries"],
                    }
                )
            receipt: dict[str, Any] = {
                "schema": RETRY_RESET_RECEIPT_SCHEMA,
                "request_id": request.request_id,
                "idempotency_key": request.idempotency_key,
                "authorization_decision_id": decision.decision_id,
                "policy_id": trusted_policy.policy_id,
                "policy_revision": trusted_policy.policy_revision,
                "repository_root": request.repository_root,
                "state_root": request.state_root,
                "repository_id": request.repository_id,
                "repository_tree_id": request.tree_id,
                "repository_head_commit": binding.repository_head_commit,
                "task_source_repository_tree_id": binding.task_source_repository_tree_id,
                "database_path": str(database_path),
                "journal_path": str(journal_path),
                "plan_root_cid": binding.plan_root_cid,
                "task_cid": binding.task_cid,
                "task_alias": binding.task_alias,
                "task_revision_before": binding.task_revision,
                "task_revision_after": journal["database_after"]["task_revision"],
                "status_before": binding.expected_status,
                "status_after": "retrying",
                "status_changed": bool(journal["database_after"]["status_changed"]),
                "writer_id": binding.writer_id,
                "writer_fencing_token": binding.writer_fencing_token,
                "intent_cid": intent_cid,
                "status_receipt_cid": journal["database_after"]["status_receipt_cid"],
                "lifecycle_owner_paths": [
                    str(_resolve_under(state_root, item))
                    for item in binding.lifecycle_owner_paths
                ],
                "lanes": lanes_receipt,
            }
            receipt_cid = content_identity(receipt)
            receipt["receipt_cid"] = receipt_cid
            receipt_path = receipt_root / f"{receipt_cid}.json"
            _write_json_durable(receipt_path, receipt)
            if fault_injector:
                fault_injector("receipt_written")
            journal["receipt_cid"] = receipt_cid
            journal["receipt_path"] = str(receipt_path)
            journal["phase"] = "receipt_committed"
            _write_json_durable(journal_path, journal)
            if fault_injector:
                fault_injector("receipt_committed")

        if journal["phase"] == "receipt_committed":
            receipt = _verified_receipt(journal, receipt_root)
            receipt_cid = str(receipt["receipt_cid"])
            event = source.append_event(
                {
                    "schema": RETRY_RESET_EVENT_SCHEMA,
                    "event_cid": receipt_cid,
                    "event_type": "retry_reset_completed",
                    "task_cid": binding.task_cid,
                    "request_id": request.request_id,
                    "receipt_cid": receipt_cid,
                    "intent_cid": intent_cid,
                    "plan_root_cid": binding.plan_root_cid,
                    "repository_id": request.repository_id,
                    "repository_tree_id": request.tree_id,
                    "repository_head_commit": binding.repository_head_commit,
                    "task_source_repository_tree_id": binding.task_source_repository_tree_id,
                    "writer_id": binding.writer_id,
                    "writer_fencing_token": binding.writer_fencing_token,
                    "lane_state_digests": [
                        item["state_digest_after"] for item in receipt["lanes"]
                    ],
                    "lane_queue_digests": [
                        item["queue_digest_after"] for item in receipt["lanes"]
                    ],
                },
                lease={
                    "lease_id": request.lease_id,
                    "fencing_token": binding.writer_fencing_token,
                },
                fence=binding.writer_fencing_token,
                writer_id=binding.writer_id,
            )
            if fault_injector:
                fault_injector("completion_event_appended")
            journal["completion_event"] = dict(event)
            journal["phase"] = "completed"
            _write_json_durable(journal_path, journal)
            if fault_injector:
                fault_injector("completed")
            return _verify_completed_journal(
                state_root=state_root,
                path=journal_path,
                journal=journal,
            )

        raise DuckDBRetryResetCorruptionError(
            "retry-reset journal did not reach completion"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--inspect-state-root", type=Path)
    parser.add_argument("--lock-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.inspect_state_root is None and args.request_file is None:
        parser.error("execute mode requires --request-file")
    if args.inspect_state_root is not None and args.request_file is not None:
        parser.error("--inspect-state-root cannot be combined with execute mode")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.inspect_state_root is not None:
            incomplete = inspect_incomplete_retry_resets(args.inspect_state_root)
            payload = {"ok": not incomplete, "incomplete": list(incomplete)}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if not incomplete else 3
        assert args.request_file is not None
        request_payload = _read_bounded_json(args.request_file, "retry-reset request")
        request = decode_operation_request(request_payload)
        state_root = Path(request.state_root)
        if str(state_root.resolve()) != request.state_root:
            raise DuckDBRetryResetConflict("request state_root is not canonical")
        owner = _load_owner_config(state_root)
        policy = _load_policy(
            _resolve_under(state_root, owner.policy_path),
            expected_digest=owner.policy_digest,
        )
        receipt = execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            lock_timeout_seconds=args.lock_timeout_seconds,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DuckDBRetryResetAuthorizationError",
    "DuckDBRetryResetConflict",
    "DuckDBRetryResetCorruptionError",
    "DuckDBRetryResetError",
    "DuckDBRetryResetQuiescenceError",
    "LaneBinding",
    "RETRY_RESET_GRANT",
    "RETRY_RESET_OWNER_FILE",
    "RETRY_RESET_OWNER_SCHEMA",
    "RETRY_RESET_POLICY_SCHEMA",
    "RetryResetOwnerConfig",
    "execute_duckdb_retry_reset",
    "inspect_incomplete_retry_resets",
    "retry_reset_expected_effect",
]
