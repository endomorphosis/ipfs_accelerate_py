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
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from .authorization_logic import ControlMutationAuthorizer, ControlMutationPolicy
from .checkout_lock import checkout_mutation_lock_path, checkout_repository_id
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
RETRY_RESET_EXECUTION_INTENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset-execution-intent@1"
)
RETRY_RESET_EXECUTION_INTENT_EVENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset-execution-intent-event@1"
)
RETRY_RESET_EXECUTION_INTENT_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-retry-reset-execution-intent-binding@1"
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
MAX_EXECUTION_INTENTS: Final = 512
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


def _reject_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _read_bounded_json(path: Path, noun: str) -> dict[str, Any]:
    encoded = _read_bounded_bytes(path, noun)
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
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
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
        )
    except (json.JSONDecodeError, ValueError):
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
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
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


def _request_canonical_evidence(request: OperationRequest) -> dict[str, str]:
    """Return the exact request bytes used to authorize a durable intent."""

    try:
        encoded = bytes(request.canonical_bytes())
        canonical_json = encoded.decode("utf-8")
    except (AttributeError, TypeError, UnicodeError, ValueError) as exc:
        raise DuckDBRetryResetAuthorizationError(
            "retry request lacks canonical UTF-8 bytes"
        ) from exc
    return {
        "canonical_json": canonical_json,
        "digest": _digest_bytes(encoded),
    }


def _policy_payload(policy: ControlMutationPolicy) -> dict[str, Any]:
    return {
        "schema": RETRY_RESET_POLICY_SCHEMA,
        "policy_id": policy.policy_id,
        "policy_revision": policy.policy_revision,
        "permits": [item.to_record() for item in policy.permits],
        "current_tree_ids": dict(policy.current_tree_ids),
        "current_objective_revisions": dict(policy.current_objective_revisions),
        "active_lease_fences": dict(policy.active_lease_fences),
    }


def _git_output(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DuckDBRetryResetConflict("cannot inspect repository generation") from exc
    if result.returncode != 0:
        raise DuckDBRetryResetConflict("cannot inspect repository generation")
    return result.stdout


def _repository_generation(repository_root: Path) -> dict[str, Any]:
    head_commit, head_tree = _git_head_binding(repository_root)
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise DuckDBRetryResetConflict(
            "retry execution intent requires a clean repository generation"
        )
    submodules = _git_output(repository_root, "submodule", "status", "--recursive")
    for line in submodules.splitlines():
        if line and line[0] in {"-", "+", "U"}:
            raise DuckDBRetryResetConflict(
                "retry execution intent requires exact clean submodule gitlinks"
            )
    return {
        "repository_head_commit": head_commit,
        "repository_head_tree": head_tree,
        "worktree_status": status,
        "worktree_status_digest": _digest_bytes(status.encode("utf-8")),
        "submodule_status": submodules,
        "submodule_status_digest": _digest_bytes(submodules.encode("utf-8")),
    }


def _parent_intent_material(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": parent.get("schema"),
        "program_id": parent.get("program_id"),
        "request_id": parent.get("request_id"),
        "request_digest": parent.get("request_digest"),
        "repository_root": parent.get("repository_root"),
        "runtime_root": parent.get("runtime_root"),
        "database_path": parent.get("database_path"),
        "plan_root_cid": parent.get("plan_root_cid"),
        "task_source_repository_tree_id": parent.get(
            "task_source_repository_tree_id"
        ),
        "repository_head_commit": parent.get("repository_head_commit"),
        "repository_head_tree": parent.get("repository_head_tree"),
        "checkout_binding": parent.get("checkout_binding"),
        "task": parent.get("task"),
        "writer": parent.get("writer"),
        "owner_configuration": parent.get("owner_configuration"),
        "policy": parent.get("policy"),
        "authorization": parent.get("authorization"),
        "environment": parent.get("environment"),
        "old_master": parent.get("old_master"),
        "old_process_tree": parent.get("old_process_tree"),
    }


def _parent_intent_cid(parent: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _parent_intent_material(parent),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _parent_phase_cid(namespace: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"namespace": namespace, **dict(payload)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _parent_drain_cid(parent: Mapping[str, Any]) -> str:
    return _parent_phase_cid(
        "duckdb-quack-retry-drain",
        {
            "intent_cid": parent.get("intent_cid"),
            "drain_process_tree": parent.get("drain_process_tree"),
            "drain_started_at": parent.get("drain_started_at"),
        },
    )


def _parent_drained_cid(parent: Mapping[str, Any]) -> str:
    return _parent_phase_cid(
        "duckdb-quack-retry-drained",
        {
            "drain_cid": parent.get("drain_cid"),
            "drained_at": parent.get("drained_at"),
        },
    )


_PARENT_PREPARED_REQUIRED_FIELDS: Final = frozenset(
    {
        "schema",
        "program_id",
        "phase",
        "request_id",
        "request_digest",
        "request_file_digest",
        "repository_root",
        "runtime_root",
        "database_path",
        "plan_root_cid",
        "task_source_repository_tree_id",
        "repository_head_commit",
        "repository_head_tree",
        "checkout_binding",
        "task",
        "writer",
        "owner_configuration",
        "policy",
        "authorization",
        "environment",
        "old_master",
        "old_process_tree",
        "lifecycle_owners",
        "created_at",
        "updated_at",
        "intent_cid",
    }
)

_PARENT_LIFECYCLE_SCHEMA: Final = (
    "ipfs_datasets_py/duckdb-quack-retry-lifecycle-journal@1"
)
_PARENT_PROGRAM_ID: Final = "ipfs-datasets-duckdb-quack-v1"
_PARENT_LIFECYCLE_PHASE_RANK: Final = {
    "prepared": 0,
    "draining": 1,
    "drained": 2,
    "leased": 3,
    "reset_committed": 4,
    "relaunching": 5,
    "finalizing": 6,
    "completed": 7,
}
_PARENT_LIFECYCLE_ALLOWED_FIELDS: Final = frozenset(
    {
        *_PARENT_PREPARED_REQUIRED_FIELDS,
        "execution_intent",
        "drain_process_tree",
        "drain_started_at",
        "drain_cid",
        "drained_at",
        "drained_cid",
        "retry_reset_receipt",
        "retry_reset_anchor",
        "reset_committed_at",
        "reset_commit_cid",
        "relaunch",
        "relaunch_intent_cid",
        "new_master",
        "checkout_leases",
        "checkout_leased_at",
        "checkout_lease_set_cid",
        "checkout_finalization",
        "checkout_release_tombstones",
        "checkout_release_receipt",
        "lifecycle_receipt",
    }
)
_PROCESS_IDENTITY_FIELDS: Final = frozenset(
    {"pid", "boot_id", "start_ticks", "cmdline_sha256", "argv"}
)
_MASTER_STORED_FIELDS: Final = frozenset(
    {
        "schema",
        "program_id",
        "repository_root",
        "master_root",
        "master_pid_path",
        "plan_root_cid",
        "repository_tree_id",
        "execution_slice_sha256",
        "execution_slice_task_count",
        "authorization_held_set_sha256",
        "authorization_held_task_count",
        "bootstrap_completion_evidence_id",
        "lane_count",
        "created_at",
        "python_environment_sha256",
        "pid",
        "boot_id",
        "start_ticks",
        "cmdline_sha256",
    }
)
_ENVIRONMENT_EVIDENCE_FIELDS: Final = frozenset(
    {
        "receipt_path",
        "receipt_sha256",
        "receipt_id",
        "environment_root",
        "sealed_python_launcher_path",
        "sealed_python_launcher_sha256",
        "base_python_sha256",
        "site_packages_manifest_sha256",
        "duckdb_version",
        "duckdb_record_evidence_sha256",
    }
)
_CHECKOUT_BINDING_FIELDS: Final = frozenset(
    {
        "role",
        "repository_root",
        "repository_id",
        "lock_path",
        "branch",
        "head_commit",
        "head_tree",
        "parent_accelerator_gitlink",
    }
)


def _aware_iso8601(value: Any, noun: str) -> str:
    if not isinstance(value, str) or not value:
        raise DuckDBRetryResetAuthorizationError(f"{noun} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DuckDBRetryResetAuthorizationError(f"{noun} is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DuckDBRetryResetAuthorizationError(f"{noun} lacks a timezone")
    return value


def _strict_process_identity(value: Any, noun: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PROCESS_IDENTITY_FIELDS:
        raise DuckDBRetryResetAuthorizationError(f"{noun} shape is malformed")
    identity = dict(value)
    pid = identity.get("pid")
    ticks = identity.get("start_ticks")
    argv = identity.get("argv")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(ticks, int)
        or isinstance(ticks, bool)
        or ticks <= 0
        or not isinstance(identity.get("boot_id"), str)
        or not identity.get("boot_id")
        or not isinstance(identity.get("cmdline_sha256"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity["cmdline_sha256"])
        or isinstance(argv, (str, bytes, Mapping))
        or not isinstance(argv, Sequence)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise DuckDBRetryResetAuthorizationError(f"{noun} is malformed")
    identity["argv"] = list(argv)
    return identity


def _argv_option(argv: Sequence[str], name: str) -> str:
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if item.startswith(prefix):
            return item[len(prefix) :]
    return ""


def _master_execution_slice(argv: Sequence[str]) -> list[str]:
    aliases: list[str] = []
    marker = "--common-arg=--execution-slice-task-id"
    for index, item in enumerate(argv[:-1]):
        if item != marker:
            continue
        selected = argv[index + 1]
        prefix = "--common-arg="
        if not selected.startswith(prefix) or not selected[len(prefix) :]:
            raise DuckDBRetryResetAuthorizationError(
                "parent PREPARED master execution slice is malformed"
            )
        aliases.append(selected[len(prefix) :])
    if len(aliases) != len(set(aliases)):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master execution slice is duplicated"
        )
    return aliases


def _master_bootstrap_completion_evidence_id(argv: Sequence[str]) -> str:
    marker = "--common-arg=--duckdb-bootstrap-completion-evidence-id"
    prefix = "--common-arg="
    if any(str(item).startswith(marker + "=") for item in argv):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master bootstrap evidence uses an unsupported form"
        )
    values: list[str] = []
    for index, item in enumerate(argv):
        if item != marker:
            continue
        if index + 1 >= len(argv):
            raise DuckDBRetryResetAuthorizationError(
                "parent PREPARED master bootstrap evidence is malformed"
            )
        selected = argv[index + 1]
        if not selected.startswith(prefix) or not selected[len(prefix) :]:
            raise DuckDBRetryResetAuthorizationError(
                "parent PREPARED master bootstrap evidence is malformed"
            )
        values.append(selected[len(prefix) :])
    if len(values) > 1:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master bootstrap evidence is duplicated"
        )
    value = values[0] if values else ""
    if value and not re.fullmatch(r"baguqeera[a-z2-7]{52}", value):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master bootstrap evidence is not a canonical CIDv1"
        )
    return value


def _live_process_identity(pid: int) -> dict[str, Any] | None:
    try:
        os.kill(pid, 0)
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        start_ticks = int(stat_text.rsplit(") ", 1)[1].split()[19])
        command_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, IndexError, ValueError):
        return None
    return {
        "pid": pid,
        "boot_id": boot_id,
        "start_ticks": start_ticks,
        "cmdline_sha256": _digest_bytes(command_bytes),
        "argv": [
            item.decode("utf-8", errors="replace")
            for item in command_bytes.split(b"\0")
            if item
        ],
    }


def _process_session_members(session_id: int) -> tuple[int, ...]:
    if session_id <= 0:
        raise DuckDBRetryResetCorruptionError(
            "parent execution intent has an invalid process session"
        )
    members: list[int] = []
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        raise DuckDBRetryResetQuiescenceError(
            "cannot inspect the drained master process session"
        ) from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) == session_id:
                members.append(pid)
        except ProcessLookupError:
            continue
        except OSError as exc:
            raise DuckDBRetryResetQuiescenceError(
                "cannot inspect a member of the drained master process session"
            ) from exc
    return tuple(sorted(members))


def _validate_checkout_generation(
    value: Any,
    *,
    request: OperationRequest,
) -> list[dict[str, Any]]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent checkout binding is malformed"
        )
    records: list[dict[str, Any]] = []
    roles: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _CHECKOUT_BINDING_FIELDS:
            raise DuckDBRetryResetAuthorizationError(
                "parent checkout binding shape is malformed"
            )
        record = dict(item)
        role = record.get("role")
        root_text = record.get("repository_root")
        if role not in {"parent", "accelerator"} or role in roles:
            raise DuckDBRetryResetAuthorizationError(
                "parent checkout binding roles are invalid"
            )
        roles.add(str(role))
        if not isinstance(root_text, str) or str(Path(root_text).resolve()) != root_text:
            raise DuckDBRetryResetAuthorizationError(
                "parent checkout root is not canonical"
            )
        root = Path(root_text)
        actual_head, actual_tree = _git_head_binding(root)
        actual_branch = _git_output(root, "branch", "--show-current").strip()
        status = _git_output(root, "status", "--porcelain=v1")
        lock_path = Path(checkout_mutation_lock_path(root))
        canonical_lock = lock_path.parent.resolve() / lock_path.name
        if (
            not actual_branch
            or status
            or record.get("repository_id") != checkout_repository_id(root)
            or record.get("lock_path") != str(canonical_lock)
            or record.get("branch") != actual_branch
            or record.get("head_commit") != actual_head
            or record.get("head_tree") != actual_tree
            or not re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}",
                str(record.get("parent_accelerator_gitlink") or ""),
            )
        ):
            raise DuckDBRetryResetAuthorizationError(
                "parent checkout binding differs from the clean live checkout"
            )
        records.append(record)
    if roles != {"parent", "accelerator"}:
        raise DuckDBRetryResetAuthorizationError(
            "parent checkout binding must cover parent and accelerator"
        )
    parent = next(item for item in records if item["role"] == "parent")
    if (
        parent["repository_root"] != request.repository_root
        or parent["repository_id"] != request.repository_id
        or parent["head_commit"]
        != request.parameters.get("repository_head_commit")
        or parent["head_tree"] != request.tree_id
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent checkout does not bind the retry request"
        )
    accelerator = next(item for item in records if item["role"] == "accelerator")
    expected_accelerator = (Path(request.repository_root) / "ipfs_accelerate_py").resolve()
    parent_gitlink = _git_output(
        Path(request.repository_root), "rev-parse", "HEAD:ipfs_accelerate_py"
    ).strip()
    if (
        accelerator["repository_root"] != str(expected_accelerator)
        or accelerator["head_commit"] != parent_gitlink
        or accelerator["head_commit"] != parent["parent_accelerator_gitlink"]
        or accelerator["parent_accelerator_gitlink"]
        != parent["parent_accelerator_gitlink"]
    ):
        raise DuckDBRetryResetAuthorizationError(
            "accelerator checkout differs from the parent gitlink"
        )
    return sorted(records, key=lambda item: str(item["lock_path"]))


def _validate_environment_generation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ENVIRONMENT_EVIDENCE_FIELDS:
        raise DuckDBRetryResetAuthorizationError(
            "parent environment generation shape is malformed"
        )
    evidence = dict(value)
    receipt_path = evidence.get("receipt_path")
    if (
        not isinstance(receipt_path, str)
        or str(Path(receipt_path).resolve()) != receipt_path
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent environment receipt path is not canonical"
        )
    receipt_bytes = _read_bounded_bytes(
        Path(receipt_path), "sealed execution-environment receipt"
    )
    try:
        receipt = json.loads(
            receipt_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DuckDBRetryResetAuthorizationError(
            "sealed execution-environment receipt is malformed"
        ) from exc
    digest_fields = {
        "receipt_sha256",
        "sealed_python_launcher_sha256",
        "base_python_sha256",
        "site_packages_manifest_sha256",
        "duckdb_record_evidence_sha256",
    }
    probe = receipt.get("probe") if isinstance(receipt, Mapping) else None
    probe_binding = {
        name: evidence.get(name)
        for name in (
            "environment_root",
            "sealed_python_launcher_path",
            "sealed_python_launcher_sha256",
            "base_python_sha256",
            "site_packages_manifest_sha256",
            "duckdb_version",
            "duckdb_record_evidence_sha256",
        )
    }
    if (
        not isinstance(receipt, Mapping)
        or evidence.get("receipt_sha256") != _digest_bytes(receipt_bytes)
        or receipt.get("receipt_id") != evidence.get("receipt_id")
        or not isinstance(probe, Mapping)
        or any(probe.get(name) != value for name, value in probe_binding.items())
        or any(
            not isinstance(evidence.get(name), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(evidence.get(name)))
            for name in digest_fields
        )
        or any(
            not isinstance(evidence.get(name), str) or not evidence.get(name)
            for name in (
                "receipt_id",
                "environment_root",
                "sealed_python_launcher_path",
                "duckdb_version",
            )
        )
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent environment generation is not receipt-bound"
        )
    return evidence


def _request_file_evidence(
    encoded: bytes,
    *,
    request: OperationRequest,
) -> dict[str, str]:
    if not encoded or len(encoded) > MAX_SIDECAR_BYTES:
        raise DuckDBRetryResetAuthorizationError(
            "retry request file bytes have an invalid size"
        )
    try:
        raw_json = encoded.decode("utf-8")
        payload = json.loads(
            raw_json,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
        )
        decoded = decode_operation_request(payload)
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise DuckDBRetryResetAuthorizationError(
            "retry request file bytes are malformed"
        ) from exc
    if decoded != request:
        raise DuckDBRetryResetAuthorizationError(
            "retry request file bytes decode to another request"
        )
    return {"raw_json": raw_json, "digest": _digest_bytes(encoded)}


def _validate_parent_prepared(
    parent: Mapping[str, Any],
    *,
    request: OperationRequest,
    binding: RetryResetBinding,
    trusted_owner: RetryResetOwnerConfig,
    owner_digest: str,
    policy_digest: str,
    repository_generation: Mapping[str, Any],
    request_file_evidence: Mapping[str, str],
    require_live_identities: bool,
) -> dict[str, Any]:
    """Validate the complete pre-drain lifecycle record supplied by its owner."""

    if set(parent) != _PARENT_PREPARED_REQUIRED_FIELDS:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED lifecycle material has an unsupported shape"
        )
    try:
        material = json.loads(_canonical_bytes(parent).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED lifecycle material is not canonical JSON"
        ) from exc
    request_evidence = _request_canonical_evidence(request)
    task = material.get("task")
    writer = material.get("writer")
    owner = material.get("owner_configuration")
    policy = material.get("policy")
    authorization = material.get("authorization")
    old_master = material.get("old_master")
    process_tree = material.get("old_process_tree")
    checkout_binding = _validate_checkout_generation(
        material.get("checkout_binding"), request=request
    )
    environment = _validate_environment_generation(material.get("environment"))
    if not isinstance(old_master, Mapping) or set(old_master) != {
        "stored",
        "actual",
        "lane_count",
        "duration_seconds",
        "execution_slice",
        "dedicated_session_id",
    }:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master binding shape is malformed"
        )
    actual_master = _strict_process_identity(
        old_master.get("actual"), "parent PREPARED actual master"
    )
    stored_master = old_master.get("stored")
    if not isinstance(stored_master, Mapping) or set(stored_master) != _MASTER_STORED_FIELDS:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED stored master shape is malformed"
        )
    duration_text = old_master.get("duration_seconds")
    execution_slice = old_master.get("execution_slice")
    try:
        duration_seconds = float(duration_text)
    except (TypeError, ValueError) as exc:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master duration is malformed"
        ) from exc
    actual_argv = actual_master["argv"]
    derived_execution_slice = _master_execution_slice(actual_argv)
    derived_bootstrap_completion_evidence_id = (
        _master_bootstrap_completion_evidence_id(actual_argv)
    )
    try:
        derived_lane_count = int(
            _argv_option(
                actual_argv, "--implementation-supervisor-lanes-per-track"
            )
        )
    except ValueError as exc:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master lane topology is malformed"
        ) from exc
    stored_digest_fields = (
        "execution_slice_sha256",
        "authorization_held_set_sha256",
        "python_environment_sha256",
    )
    if (
        not isinstance(duration_text, str)
        or not duration_seconds > 0
        or isinstance(execution_slice, (str, bytes, Mapping))
        or not isinstance(execution_slice, Sequence)
        or not execution_slice
        or any(not isinstance(item, str) or not item for item in execution_slice)
        or list(execution_slice) != derived_execution_slice
        or _argv_option(actual_argv, "--duration-seconds") != duration_text
        or derived_lane_count != len(binding.lanes)
        or old_master.get("dedicated_session_id") != actual_master["pid"]
        or (
            require_live_identities
            and (
                os.getsid(actual_master["pid"]) != actual_master["pid"]
                or _live_process_identity(actual_master["pid"]) != actual_master
            )
        )
        or any(
            stored_master.get(name) != actual_master.get(name)
            for name in ("pid", "boot_id", "start_ticks", "cmdline_sha256")
        )
        or stored_master.get("schema")
        != "ipfs_datasets_py/duckdb-quack-master-identity@3"
        or stored_master.get("program_id") != _PARENT_PROGRAM_ID
        or stored_master.get("repository_root") != request.repository_root
        or stored_master.get("master_root")
        != str((Path(request.state_root) / "master").resolve())
        or stored_master.get("master_pid_path")
        != str((Path(request.state_root) / "master/supervisor.pid").resolve())
        or stored_master.get("plan_root_cid") != binding.plan_root_cid
        or stored_master.get("repository_tree_id")
        != binding.task_source_repository_tree_id
        or stored_master.get("lane_count") != len(binding.lanes)
        or stored_master.get("execution_slice_task_count") != len(execution_slice)
        or stored_master.get("bootstrap_completion_evidence_id")
        != derived_bootstrap_completion_evidence_id
        or not isinstance(stored_master.get("authorization_held_task_count"), int)
        or isinstance(stored_master.get("authorization_held_task_count"), bool)
        or stored_master.get("authorization_held_task_count", -1) < 0
        or any(
            not isinstance(stored_master.get(name), str)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(stored_master.get(name))
            )
            for name in stored_digest_fields
        )
        or not _aware_iso8601(
            stored_master.get("created_at"), "stored master creation time"
        )
        or old_master.get("lane_count") != len(binding.lanes)
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED master binding is not live and canonical"
        )
    if (
        isinstance(process_tree, (str, bytes, Mapping))
        or not isinstance(process_tree, Sequence)
        or not process_tree
        or len(process_tree) > 4_096
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED process tree is malformed"
        )
    identities = [
        _strict_process_identity(item, "parent PREPARED process identity")
        for item in process_tree
    ]
    if (
        identities[0] != actual_master
        or len({item["pid"] for item in identities}) != len(identities)
        or (
            require_live_identities
            and any(
                _live_process_identity(item["pid"]) != item for item in identities
            )
        )
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED process tree is not an exact live snapshot"
        )
    owners = material.get("lifecycle_owners")
    if not isinstance(owners, list) or len(owners) != 1:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED lifecycle owner chain is malformed"
        )
    owner_identities: list[dict[str, Any]] = []
    for owner_record in owners:
        if not isinstance(owner_record, Mapping) or set(owner_record) != {
            "adopted_at",
            *_PROCESS_IDENTITY_FIELDS,
        }:
            raise DuckDBRetryResetAuthorizationError(
                "parent PREPARED lifecycle owner record is malformed"
            )
        _aware_iso8601(owner_record.get("adopted_at"), "owner adoption time")
        owner_identities.append(
            _strict_process_identity(
                {key: owner_record[key] for key in _PROCESS_IDENTITY_FIELDS},
                "parent PREPARED lifecycle owner",
            )
        )
    if (
        len(
            {
                (item["pid"], item["boot_id"], item["start_ticks"])
                for item in owner_identities
            }
        )
        != len(owner_identities)
        or (
            require_live_identities
            and (
                owner_identities[-1]["pid"] != os.getpid()
                or _live_process_identity(os.getpid()) != owner_identities[-1]
                or os.getpid() in {item["pid"] for item in identities}
            )
        )
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED lifecycle owner chain is not caller-bound"
        )
    created_at = _aware_iso8601(
        material.get("created_at"), "parent PREPARED creation time"
    )
    updated_at = _aware_iso8601(
        material.get("updated_at"), "parent PREPARED update time"
    )
    decision = request.authorization
    if decision is None or decision.expires_at_ms is None:
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED lifecycle requires a finite authorization"
        )
    created_at_ms = int(
        datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        * 1_000
    )
    if (
        material.get("schema") != _PARENT_LIFECYCLE_SCHEMA
        or material.get("program_id") != _PARENT_PROGRAM_ID
        or material.get("phase") != "prepared"
        or material.get("request_id") != request.request_id
        or material.get("request_digest") != request_evidence["digest"]
        or material.get("request_file_digest")
        != request_file_evidence.get("digest")
        or material.get("repository_root") != request.repository_root
        or material.get("runtime_root") != request.state_root
        or material.get("database_path")
        != str((Path(request.state_root) / binding.database_path).resolve())
        or material.get("plan_root_cid") != binding.plan_root_cid
        or material.get("task_source_repository_tree_id")
        != binding.task_source_repository_tree_id
        or material.get("repository_head_commit")
        != repository_generation.get("repository_head_commit")
        or material.get("repository_head_tree")
        != repository_generation.get("repository_head_tree")
        or material.get("checkout_binding") != checkout_binding
        or task
        != {
            "task_cid": binding.task_cid,
            "task_alias": binding.task_alias,
            "status": binding.expected_status,
            "revision": binding.task_revision,
        }
        or writer
        != {
            "writer_id": binding.writer_id,
            "fencing_token": binding.writer_fencing_token,
        }
        or not isinstance(owner, Mapping)
        or owner.get("path")
        != str(Path(request.state_root) / RETRY_RESET_OWNER_FILE)
        or owner.get("digest") != owner_digest
        or owner.get("payload") != trusted_owner.to_dict()
        or not isinstance(policy, Mapping)
        or policy.get("path")
        != str((Path(request.state_root) / trusted_owner.policy_path).resolve())
        or policy.get("digest") != policy_digest
        or policy.get("policy_id") != request.policy_id
        or policy.get("policy_revision") != request.policy_revision
        or policy.get("authorization_decision_id")
        != (request.authorization.decision_id if request.authorization else "")
        or not isinstance(authorization, Mapping)
        or authorization
        != {
            "decision_id": request.authorization.decision_id,
            "evaluated_at_ms": request.authorization.evaluated_at_ms,
            "expires_at_ms": request.authorization.expires_at_ms,
            "lease_id": request.lease_id,
            "fencing_epoch": request.fencing_epoch,
        }
        or material.get("environment") != environment
        or created_at != updated_at
        or owners[0].get("adopted_at") != created_at
        or not decision.evaluated_at_ms <= created_at_ms < decision.expires_at_ms
        or material.get("intent_cid") != _parent_intent_cid(material)
    ):
        raise DuckDBRetryResetAuthorizationError(
            "parent PREPARED lifecycle material is not exactly authority-bound"
        )
    return material


def _execution_intent_root(state_root: Path) -> Path:
    return _resolve_under(state_root, "duckdb-retry-reset/execution-intents")


def _execution_intent_material_cid(material: Mapping[str, Any]) -> str:
    return content_identity(
        {"namespace": "duckdb-retry-reset-execution-intent", **dict(material)}
    )


def _execution_intent_binding(projection: Mapping[str, Any]) -> dict[str, Any]:
    event = projection.get("preparation_event")
    if not isinstance(event, Mapping):
        raise DuckDBRetryResetCorruptionError(
            "retry execution intent lacks its preparation event"
        )
    return {
        "schema": RETRY_RESET_EXECUTION_INTENT_BINDING_SCHEMA,
        "execution_intent_cid": projection.get("execution_intent_cid"),
        "projection_path": projection.get("projection_path"),
        "request_digest": projection.get("request", {}).get("digest")
        if isinstance(projection.get("request"), Mapping)
        else None,
        "parent_intent_cid": projection.get("parent_prepared", {}).get("intent_cid")
        if isinstance(projection.get("parent_prepared"), Mapping)
        else None,
        "preparation_event": {
            name: event.get(name) for name in ("event_cid", "sequence", "revision")
        },
    }


def retry_reset_execution_intent_binding(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the immutable fields a parent lifecycle journal must bind."""

    return _execution_intent_binding(projection)


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


def _all_execution_intent_events(
    source: DuckDBTaskSource,
) -> tuple[dict[str, Any], ...]:
    """Return every bounded durable pre-effect authorization event."""

    cursor = 0
    scanned = 0
    matches: list[dict[str, Any]] = []
    while scanned < 100_000:
        page = source.events(cursor=cursor, limit=1_000)
        if not page.events:
            break
        for event in page.events:
            scanned += 1
            body = event.get("body")
            if (
                event.get("event_type")
                == "retry_reset_execution_intent_prepared"
            ):
                matches.append(dict(event))
                if len(matches) > MAX_EXECUTION_INTENTS:
                    raise DuckDBRetryResetCorruptionError(
                        "retry execution-intent population exceeds its governed bound"
                    )
        if page.cursor <= cursor:
            raise DuckDBRetryResetCorruptionError(
                "task event cursor did not advance"
            )
        cursor = page.cursor
        if len(page.events) < 1_000:
            break
    if scanned >= 100_000:
        raise DuckDBRetryResetCorruptionError(
            "task event history exceeds its governed bound"
        )
    return tuple(matches)


def _execution_intent_events(
    source: DuckDBTaskSource, *, request_id: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        event
        for event in _all_execution_intent_events(source)
        if isinstance(event.get("body"), Mapping)
        and event["body"].get("request_id") == request_id
    )


def _current_trust_evidence(
    *,
    state_root: Path,
    trusted_policy: ControlMutationPolicy,
    trusted_owner: RetryResetOwnerConfig,
) -> dict[str, Any]:
    owner_path = _resolve_under(state_root, RETRY_RESET_OWNER_FILE)
    owner_bytes = _read_bounded_bytes(
        owner_path, "retry-reset owner configuration"
    )
    loaded_owner = _load_owner_config(state_root)
    if loaded_owner != trusted_owner:
        raise DuckDBRetryResetAuthorizationError(
            "current owner configuration differs from the pinned owner"
        )
    policy_path = _resolve_under(state_root, trusted_owner.policy_path)
    policy_bytes = _read_bounded_bytes(policy_path, "trusted mutation policy")
    if _digest_bytes(policy_bytes) != trusted_owner.policy_digest:
        raise DuckDBRetryResetAuthorizationError(
            "current policy differs from the owner-pinned digest"
        )
    loaded_policy = _load_policy(
        policy_path, expected_digest=trusted_owner.policy_digest
    )
    if _policy_payload(loaded_policy) != _policy_payload(trusted_policy):
        raise DuckDBRetryResetAuthorizationError(
            "current policy differs from the supplied pinned policy"
        )
    policy_payload = _policy_payload(loaded_policy)
    owner_payload = loaded_owner.to_dict()
    return {
        "owner": loaded_owner,
        "owner_path": str(owner_path),
        "owner_file_digest": _digest_bytes(owner_bytes),
        "owner_payload": owner_payload,
        "owner_payload_digest": _digest_bytes(_canonical_bytes(owner_payload)),
        "policy": loaded_policy,
        "policy_path": str(policy_path),
        "policy_file_digest": _digest_bytes(policy_bytes),
        "policy_payload": policy_payload,
        "policy_payload_digest": _digest_bytes(_canonical_bytes(policy_payload)),
    }


def _validate_retry_request_static(
    request: OperationRequest,
    *,
    trusted_policy: ControlMutationPolicy,
    trusted_owner: RetryResetOwnerConfig,
    clock_ms: Callable[[], int],
    require_fresh: bool,
) -> dict[str, Any]:
    if request.operation is not Operation.RETRY or request.dry_run:
        raise DuckDBRetryResetError(
            "retry reset requires a real Operation.RETRY request"
        )
    decision = request.authorization
    if decision is None:
        raise DuckDBRetryResetAuthorizationError(
            "retry reset requires an authorization decision"
        )
    if require_fresh:
        try:
            ControlMutationAuthorizer(trusted_policy, clock_ms=clock_ms).validate(
                request
            )
        except Exception as exc:
            raise DuckDBRetryResetAuthorizationError(str(exc)) from exc
    else:
        registered = {
            item.decision_id: item for item in trusted_policy.permits
        }.get(decision.decision_id)
        if registered is None or registered != decision:
            raise DuckDBRetryResetAuthorizationError(
                "permit decision was not issued by the current policy"
            )
        if (
            request.policy_id != trusted_policy.policy_id
            or request.policy_revision != trusted_policy.policy_revision
            or trusted_policy.current_tree_ids.get(request.repository_id)
            != request.tree_id
            or trusted_policy.current_objective_revisions.get(request.objective_id)
            != request.objective_revision
            or trusted_policy.active_lease_fences.get(request.lease_id)
            != request.fencing_epoch
        ):
            raise DuckDBRetryResetAuthorizationError(
                "retry request no longer matches the current pinned policy"
            )
    binding = _binding_from_parameters(request.parameters)
    _assert_owner_binding(request, binding, trusted_owner)
    if request.fencing_epoch != binding.writer_fencing_token:
        raise DuckDBRetryResetAuthorizationError(
            "request fencing_epoch does not bind the DuckDB writer fence"
        )
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
        raise DuckDBRetryResetConflict(
            "request roots are not canonical resolved paths"
        )
    if checkout_repository_id(repository_root) != request.repository_id:
        raise DuckDBRetryResetConflict(
            "repository root does not match repository_id"
        )
    generation = _repository_generation(repository_root)
    if (
        generation["repository_head_commit"] != binding.repository_head_commit
        or generation["repository_head_tree"] != request.tree_id
    ):
        raise DuckDBRetryResetConflict(
            "request does not bind the repository's current clean generation"
        )
    trust = _current_trust_evidence(
        state_root=state_root,
        trusted_policy=trusted_policy,
        trusted_owner=trusted_owner,
    )
    return {
        "binding": binding,
        "decision": decision,
        "expected_effect": expected_effect,
        "repository_root": repository_root,
        "state_root": state_root,
        "repository_generation": generation,
        "trust": trust,
    }


def _execution_intent_from_event(
    event: Mapping[str, Any],
    *,
    state_root: Path,
) -> dict[str, Any]:
    body = event.get("body")
    if not isinstance(body, Mapping):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent event body is malformed"
        )
    required_body = {
        "schema",
        "event_type",
        "task_cid",
        "request_id",
        "repository_id",
        "repository_tree_id",
        "repository_head_commit",
        "task_source_repository_tree_id",
        "plan_root_cid",
        "writer_id",
        "writer_fencing_token",
        "execution_intent_cid",
        "intent",
        "lease",
    }
    material = body.get("intent")
    intent_cid = body.get("execution_intent_cid")
    authorization = material.get("authorization") if isinstance(material, Mapping) else None
    decision = authorization.get("decision") if isinstance(authorization, Mapping) else None
    database = material.get("database") if isinstance(material, Mapping) else None
    writer = database.get("writer") if isinstance(database, Mapping) else None
    task = database.get("task") if isinstance(database, Mapping) else None
    request_record = material.get("request") if isinstance(material, Mapping) else None
    repository_generation = (
        material.get("repository_generation") if isinstance(material, Mapping) else None
    )
    parent = material.get("parent_prepared") if isinstance(material, Mapping) else None
    parent_checkout = (
        parent.get("checkout_binding") if isinstance(parent, Mapping) else None
    )
    parent_repository = next(
        (
            item
            for item in parent_checkout
            if isinstance(item, Mapping) and item.get("role") == "parent"
        ),
        None,
    ) if isinstance(parent_checkout, list) else None
    expected_lease = {
        "lease_id": decision.get("lease_id") if isinstance(decision, Mapping) else None,
        "fencing_token": writer.get("fencing_token")
        if isinstance(writer, Mapping)
        else None,
    }
    expected_event_cid = content_identity(dict(body))
    if (
        set(body) != required_body
        or body.get("schema") != RETRY_RESET_EXECUTION_INTENT_EVENT_SCHEMA
        or body.get("event_type")
        != "retry_reset_execution_intent_prepared"
        or event.get("event_type") != body.get("event_type")
        or event.get("task_cid") != body.get("task_cid")
        or event.get("event_cid") != expected_event_cid
        or not isinstance(task, Mapping)
        or not isinstance(writer, Mapping)
        or not isinstance(repository_generation, Mapping)
        or body.get("task_cid") != task.get("task_cid")
        or not isinstance(request_record, Mapping)
        or body.get("request_id") != request_record.get("request_id")
        or not isinstance(decision, Mapping)
        or body.get("repository_id") != decision.get("repository_id")
        or body.get("repository_tree_id") != decision.get("tree_id")
        or body.get("repository_head_commit")
        != repository_generation.get("repository_head_commit")
        or body.get("task_source_repository_tree_id")
        != database.get("task_source_repository_tree_id")
        or body.get("plan_root_cid") != database.get("plan_root_cid")
        or body.get("writer_id") != writer.get("writer_id")
        or body.get("writer_fencing_token") != writer.get("fencing_token")
        or body.get("lease") != expected_lease
        or not isinstance(material, Mapping)
        or material.get("schema") != RETRY_RESET_EXECUTION_INTENT_SCHEMA
        or not isinstance(parent, Mapping)
        or parent.get("request_id") != request_record.get("request_id")
        or not isinstance(parent_repository, Mapping)
        or parent_repository.get("repository_id") != body.get("repository_id")
        or parent_repository.get("head_commit")
        != body.get("repository_head_commit")
        or parent_repository.get("head_tree") != body.get("repository_tree_id")
        or parent.get("task") != task
        or parent.get("writer") != writer
        or parent.get("repository_head_commit")
        != repository_generation.get("repository_head_commit")
        or parent.get("repository_head_tree")
        != repository_generation.get("repository_head_tree")
        or parent.get("plan_root_cid") != database.get("plan_root_cid")
        or parent.get("task_source_repository_tree_id")
        != database.get("task_source_repository_tree_id")
        or _execution_intent_material_cid(material) != intent_cid
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent event is not content-bound"
        )
    projection_path = _execution_intent_root(state_root) / f"{intent_cid}.json"
    projection = {
        **dict(material),
        "execution_intent_cid": intent_cid,
        "projection_path": str(projection_path),
        "preparation_event": {
            name: event.get(name) for name in ("event_cid", "sequence", "revision")
        },
    }
    return projection


def _verify_execution_intent_projection(
    projection: Mapping[str, Any],
    *,
    request: OperationRequest,
    trusted_policy: ControlMutationPolicy,
    trusted_owner: RetryResetOwnerConfig,
    expected_parent_journal_path: Path,
    source: DuckDBTaskSource,
    require_original_task: bool,
) -> dict[str, Any]:
    context = _validate_retry_request_static(
        request,
        trusted_policy=trusted_policy,
        trusted_owner=trusted_owner,
        clock_ms=lambda: 0,
        require_fresh=False,
    )
    intent_cid = projection.get("execution_intent_cid")
    projection_path = _execution_intent_root(context["state_root"]) / f"{intent_cid}.json"
    preparation_event = projection.get("preparation_event")
    material = {
        key: value
        for key, value in projection.items()
        if key
        not in {"execution_intent_cid", "projection_path", "preparation_event"}
    }
    if (
        not isinstance(intent_cid, str)
        or not intent_cid
        or projection.get("projection_path") != str(projection_path)
        or _execution_intent_material_cid(material) != intent_cid
        or not isinstance(preparation_event, Mapping)
        or not isinstance(preparation_event.get("event_cid"), str)
        or not preparation_event.get("event_cid")
        or not isinstance(preparation_event.get("sequence"), int)
        or isinstance(preparation_event.get("sequence"), bool)
        or not isinstance(preparation_event.get("revision"), int)
        or isinstance(preparation_event.get("revision"), bool)
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent projection is not content-bound"
        )
    request_evidence = _request_canonical_evidence(request)
    request_file_record = material.get("request_file")
    if (
        not isinstance(request_file_record, Mapping)
        or set(request_file_record) != {"raw_json", "digest"}
        or not isinstance(request_file_record.get("raw_json"), str)
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent request-file evidence is malformed"
        )
    request_file = _request_file_evidence(
        request_file_record["raw_json"].encode("utf-8"), request=request
    )
    decision = context["decision"]
    effect = context["expected_effect"]
    trust = context["trust"]
    parent = material.get("parent_prepared")
    parent_path = material.get("parent_journal_path")
    database = material.get("database")
    if (
        material.get("schema") != RETRY_RESET_EXECUTION_INTENT_SCHEMA
        or material.get("request")
        != {
            "request_id": request.request_id,
            **request_evidence,
        }
        or dict(request_file_record) != request_file
        or material.get("authorization")
        != {
            "decision": decision.to_record(),
            "decision_digest": decision.decision_id,
            "evaluated_at_ms": decision.evaluated_at_ms,
            "expires_at_ms": decision.expires_at_ms,
        }
        or material.get("policy")
        != {
            "path": trust["policy_path"],
            "file_digest": trust["policy_file_digest"],
            "payload_digest": trust["policy_payload_digest"],
            "payload": trust["policy_payload"],
        }
        or material.get("owner")
        != {
            "path": trust["owner_path"],
            "file_digest": trust["owner_file_digest"],
            "payload_digest": trust["owner_payload_digest"],
            "payload": trust["owner_payload"],
        }
        or material.get("expected_effect")
        != {
            "effect": effect.to_record(),
            "effect_id": effect.effect_id,
        }
        or material.get("idempotency")
        != {
            "record": request.idempotency.to_record(),
            "content_id": request.idempotency.content_id,
        }
        or material.get("repository_generation")
        != context["repository_generation"]
        or parent_path != str(expected_parent_journal_path)
        or not isinstance(parent, Mapping)
        or parent.get("environment") != material.get("environment_generation")
        or parent.get("old_master") != material.get("old_master")
        or parent.get("old_process_tree") != material.get("old_process_tree")
        or material.get("lanes")
        != [lane.to_dict() for lane in context["binding"].lanes]
        or not isinstance(database, Mapping)
        or database.get("plan_root_cid") != context["binding"].plan_root_cid
        or database.get("task_source_repository_tree_id")
        != context["binding"].task_source_repository_tree_id
        or database.get("task")
        != {
            "task_cid": context["binding"].task_cid,
            "task_alias": context["binding"].task_alias,
            "status": context["binding"].expected_status,
            "revision": context["binding"].task_revision,
        }
        or database.get("writer")
        != {
            "writer_id": context["binding"].writer_id,
            "fencing_token": context["binding"].writer_fencing_token,
        }
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent authority binding changed"
        )
    _validate_parent_prepared(
        parent,
        request=request,
        binding=context["binding"],
        trusted_owner=trusted_owner,
        owner_digest=trust["owner_file_digest"],
        policy_digest=trust["policy_file_digest"],
        repository_generation=context["repository_generation"],
        request_file_evidence=request_file,
        require_live_identities=False,
    )
    prepared_at_ms = material.get("prepared_at_ms")
    if (
        not isinstance(prepared_at_ms, int)
        or isinstance(prepared_at_ms, bool)
        or decision.expires_at_ms is None
        or not decision.evaluated_at_ms <= prepared_at_ms < decision.expires_at_ms
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution intent was not prepared under a finite fresh permit"
        )
    events = _execution_intent_events(source, request_id=request.request_id)
    if len(events) != 1:
        raise DuckDBRetryResetCorruptionError(
            "retry execution intent requires exactly one durable preparation event"
        )
    event_body = events[0].get("body")
    database_task = database.get("task") if isinstance(database, Mapping) else None
    if (
        not isinstance(event_body, Mapping)
        or not isinstance(database_task, Mapping)
        or event_body.get("task_cid") != database_task.get("task_cid")
        or event_body.get("request_id") != request.request_id
        or event_body.get("repository_id") != request.repository_id
        or event_body.get("repository_tree_id") != request.tree_id
        or event_body.get("repository_head_commit")
        != context["binding"].repository_head_commit
        or event_body.get("task_source_repository_tree_id")
        != context["binding"].task_source_repository_tree_id
        or event_body.get("plan_root_cid") != context["binding"].plan_root_cid
        or event_body.get("writer_id") != context["binding"].writer_id
        or event_body.get("writer_fencing_token")
        != context["binding"].writer_fencing_token
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent event envelope is cross-bound incorrectly"
        )
    derived = _execution_intent_from_event(events[0], state_root=context["state_root"])
    if derived != dict(projection):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent projection differs from durable event history"
        )
    current_task = source.get_task(context["binding"].task_cid)
    if current_task is None or current_task.task_alias != context["binding"].task_alias:
        raise DuckDBRetryResetConflict("retry execution-intent task disappeared")
    if require_original_task and (
        current_task.status != context["binding"].expected_status
        or current_task.revision != context["binding"].task_revision
    ):
        raise DuckDBRetryResetConflict(
            "retry execution-intent task status/revision changed"
        )
    writer = source.current_writer_fence()
    if (writer.writer_id, writer.fencing_token) != (
        context["binding"].writer_id,
        context["binding"].writer_fencing_token,
    ):
        raise DuckDBRetryResetConflict(
            "retry execution-intent writer owner/fence changed"
        )
    return dict(projection)


def prepare_duckdb_retry_reset_execution_intent(
    request: OperationRequest,
    *,
    trusted_policy: ControlMutationPolicy,
    trusted_owner: RetryResetOwnerConfig,
    parent_prepared: Mapping[str, Any],
    parent_journal_path: str | os.PathLike[str],
    request_file_bytes: bytes,
    clock_ms: Callable[[], int] | None = None,
    lock_timeout_seconds: float = 30.0,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Durably authorize one later quiescent reset while its permit is fresh.

    The custom DuckDB event is the authorization boundary.  It is deliberately
    distinct from the later reset intent/status event and contains enough exact
    parent material to recover a crash before the parent journal is published.
    """

    selected_clock = clock_ms or (lambda: time.time_ns() // 1_000_000)
    context = _validate_retry_request_static(
        request,
        trusted_policy=trusted_policy,
        trusted_owner=trusted_owner,
        clock_ms=selected_clock,
        require_fresh=True,
    )
    decision = context["decision"]
    if decision.expires_at_ms is None:
        raise DuckDBRetryResetAuthorizationError(
            "retry execution intent requires an explicitly finite permit"
        )
    request_file = _request_file_evidence(request_file_bytes, request=request)
    parent_path = Path(parent_journal_path)
    expected_parent_root = _resolve_under(
        context["state_root"], "duckdb-retry-reset/journals/lifecycle"
    )
    expected_parent_path = expected_parent_root / (
        _request_canonical_evidence(request)["digest"].removeprefix("sha256:")
        + ".json"
    )
    if parent_path != expected_parent_path or not parent_path.is_absolute():
        raise DuckDBRetryResetAuthorizationError(
            "parent lifecycle journal path is not canonical for this request"
        )
    trust = context["trust"]
    parent = _validate_parent_prepared(
        parent_prepared,
        request=request,
        binding=context["binding"],
        trusted_owner=trusted_owner,
        owner_digest=trust["owner_file_digest"],
        policy_digest=trust["policy_file_digest"],
        repository_generation=context["repository_generation"],
        request_file_evidence=request_file,
        require_live_identities=True,
    )
    database_path = _resolve_under(
        context["state_root"], context["binding"].database_path
    )
    _assert_regular_path(database_path, "DuckDB task source")
    lifecycle_lock = _resolve_under(
        context["state_root"], ".duckdb-retry-reset.lifecycle.lock"
    )
    with exclusive_file_lock(
        lifecycle_lock, timeout_seconds=lock_timeout_seconds
    ):
        # Re-read every mutable authority immediately adjacent to the append.
        context = _validate_retry_request_static(
            request,
            trusted_policy=trusted_policy,
            trusted_owner=trusted_owner,
            clock_ms=selected_clock,
            require_fresh=True,
        )
        trust = context["trust"]
        parent = _validate_parent_prepared(
            parent,
            request=request,
            binding=context["binding"],
            trusted_owner=trusted_owner,
            owner_digest=trust["owner_file_digest"],
            policy_digest=trust["policy_file_digest"],
            repository_generation=context["repository_generation"],
            request_file_evidence=request_file,
            require_live_identities=True,
        )
        source = DuckDBTaskSource(
            database_path,
            expected_plan_root_cid=context["binding"].plan_root_cid,
            expected_repository_tree_id=context["binding"].task_source_repository_tree_id,
            writer_id=context["binding"].writer_id,
            fencing_token=context["binding"].writer_fencing_token,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        snapshot = source.snapshot()
        task = source.get_task(context["binding"].task_cid)
        writer = source.current_writer_fence()
        if (
            snapshot.plan_root_cid != context["binding"].plan_root_cid
            or snapshot.repository_tree_id
            != context["binding"].task_source_repository_tree_id
            or task is None
            or task.task_alias != context["binding"].task_alias
            or task.status != context["binding"].expected_status
            or task.revision != context["binding"].task_revision
            or (writer.writer_id, writer.fencing_token)
            != (
                context["binding"].writer_id,
                context["binding"].writer_fencing_token,
            )
        ):
            raise DuckDBRetryResetConflict(
                "retry execution-intent DuckDB binding changed before append"
            )
        existing_events = _execution_intent_events(
            source, request_id=request.request_id
        )
        if len(existing_events) > 1:
            raise DuckDBRetryResetConflict(
                "retry request has conflicting execution-intent events"
            )
        if existing_events:
            projection = _execution_intent_from_event(
                existing_events[0], state_root=context["state_root"]
            )
            projection_path = Path(str(projection["projection_path"]))
            if projection.get("parent_prepared") != parent:
                raise DuckDBRetryResetConflict(
                    "retry request was already prepared for another parent intent"
                )
            if projection_path.exists():
                stored = _read_bounded_json(
                    projection_path, "retry execution-intent projection"
                )
                if stored != projection:
                    raise DuckDBRetryResetConflict(
                        "retry execution-intent projection conflicts with event history"
                    )
            else:
                _write_json_durable(projection_path, projection)
            return _verify_execution_intent_projection(
                projection,
                request=request,
                trusted_policy=trusted_policy,
                trusted_owner=trusted_owner,
                expected_parent_journal_path=expected_parent_path,
                source=source,
                require_original_task=True,
            )
        prepared_at_ms = selected_clock()
        # This check is deliberately repeated after all reads and immediately
        # before constructing/appending the durable authorization event.
        try:
            ControlMutationAuthorizer(
                trust["policy"], clock_ms=lambda: prepared_at_ms
            ).validate(request)
        except Exception as exc:
            raise DuckDBRetryResetAuthorizationError(str(exc)) from exc
        request_evidence = _request_canonical_evidence(request)
        decision = context["decision"]
        effect = context["expected_effect"]
        assert request.idempotency is not None
        intent_material: dict[str, Any] = {
            "schema": RETRY_RESET_EXECUTION_INTENT_SCHEMA,
            "prepared_at_ms": prepared_at_ms,
            "request": {
                "request_id": request.request_id,
                **request_evidence,
            },
            "request_file": request_file,
            "authorization": {
                "decision": decision.to_record(),
                "decision_digest": decision.decision_id,
                "evaluated_at_ms": decision.evaluated_at_ms,
                "expires_at_ms": decision.expires_at_ms,
            },
            "policy": {
                "path": trust["policy_path"],
                "file_digest": trust["policy_file_digest"],
                "payload_digest": trust["policy_payload_digest"],
                "payload": trust["policy_payload"],
            },
            "owner": {
                "path": trust["owner_path"],
                "file_digest": trust["owner_file_digest"],
                "payload_digest": trust["owner_payload_digest"],
                "payload": trust["owner_payload"],
            },
            "expected_effect": {
                "effect": effect.to_record(),
                "effect_id": effect.effect_id,
            },
            "idempotency": {
                "record": request.idempotency.to_record(),
                "content_id": request.idempotency.content_id,
            },
            "repository_generation": context["repository_generation"],
            "database": {
                "database_path": str(database_path),
                "plan_root_cid": snapshot.plan_root_cid,
                "task_source_repository_tree_id": snapshot.repository_tree_id,
                "revision_before": snapshot.revision,
                "event_cursor_before": snapshot.event_cursor,
                "task": {
                    "task_cid": task.task_cid,
                    "task_alias": task.task_alias,
                    "status": task.status,
                    "revision": task.revision,
                },
                "writer": {
                    "writer_id": writer.writer_id,
                    "fencing_token": writer.fencing_token,
                },
            },
            "lanes": [lane.to_dict() for lane in context["binding"].lanes],
            "environment_generation": parent["environment"],
            "old_master": parent["old_master"],
            "old_process_tree": parent["old_process_tree"],
            "parent_journal_path": str(expected_parent_path),
            "parent_prepared": parent,
        }
        intent_cid = _execution_intent_material_cid(intent_material)

        def assert_preparation_write_precondition() -> None:
            """Revalidate mutable authority while the task-source lock is held."""

            boundary = _validate_retry_request_static(
                request,
                trusted_policy=trusted_policy,
                trusted_owner=trusted_owner,
                clock_ms=selected_clock,
                require_fresh=True,
            )
            if (
                boundary["binding"] != context["binding"]
                or boundary["repository_generation"]
                != context["repository_generation"]
                or boundary["trust"]["owner_file_digest"]
                != trust["owner_file_digest"]
                or boundary["trust"]["policy_file_digest"]
                != trust["policy_file_digest"]
            ):
                raise DuckDBRetryResetConflict(
                    "retry execution-intent authority changed at the write boundary"
                )

        appended = source.append_event(
            {
                "schema": RETRY_RESET_EXECUTION_INTENT_EVENT_SCHEMA,
                "event_type": "retry_reset_execution_intent_prepared",
                "task_cid": context["binding"].task_cid,
                "request_id": request.request_id,
                "repository_id": request.repository_id,
                "repository_tree_id": request.tree_id,
                "repository_head_commit": context["binding"].repository_head_commit,
                "task_source_repository_tree_id": (
                    context["binding"].task_source_repository_tree_id
                ),
                "plan_root_cid": context["binding"].plan_root_cid,
                "writer_id": context["binding"].writer_id,
                "writer_fencing_token": (
                    context["binding"].writer_fencing_token
                ),
                "execution_intent_cid": intent_cid,
                "intent": intent_material,
            },
            lease={
                "lease_id": request.lease_id,
                "fencing_token": context["binding"].writer_fencing_token,
            },
            fence=context["binding"].writer_fencing_token,
            writer_id=context["binding"].writer_id,
            write_precondition=assert_preparation_write_precondition,
            expected_task_status=context["binding"].expected_status,
            expected_task_revision=context["binding"].task_revision,
        )
        if fault_injector:
            fault_injector("execution_intent_event_appended")
        projection = {
            **intent_material,
            "execution_intent_cid": intent_cid,
            "projection_path": str(
                _execution_intent_root(context["state_root"])
                / f"{intent_cid}.json"
            ),
            "preparation_event": {
                name: appended[name]
                for name in ("event_cid", "sequence", "revision")
            },
        }
        projection_path = Path(str(projection["projection_path"]))
        if projection_path.exists():
            existing = _read_bounded_json(
                projection_path, "retry execution-intent projection"
            )
            if existing != projection:
                raise DuckDBRetryResetConflict(
                    "retry execution-intent projection conflicts with event history"
                )
        else:
            _write_json_durable(projection_path, projection)
            if fault_injector:
                fault_injector("execution_intent_projection_written")
        return _verify_execution_intent_projection(
            projection,
            request=request,
            trusted_policy=trusted_policy,
            trusted_owner=trusted_owner,
            expected_parent_journal_path=expected_parent_path,
            source=source,
            require_original_task=True,
        )


def recover_duckdb_retry_reset_execution_intent(
    request: OperationRequest,
    *,
    trusted_policy: ControlMutationPolicy,
    trusted_owner: RetryResetOwnerConfig,
    expected_parent_journal_path: str | os.PathLike[str],
    lock_timeout_seconds: float = 30.0,
    repair_projection: bool = True,
) -> dict[str, Any] | None:
    """Recover and verify the one pre-effect event for a request, if present."""

    context = _validate_retry_request_static(
        request,
        trusted_policy=trusted_policy,
        trusted_owner=trusted_owner,
        clock_ms=lambda: 0,
        require_fresh=False,
    )
    parent_path = Path(expected_parent_journal_path)
    database_path = _resolve_under(
        context["state_root"], context["binding"].database_path
    )
    source = DuckDBTaskSource(
        database_path,
        expected_plan_root_cid=context["binding"].plan_root_cid,
        expected_repository_tree_id=context["binding"].task_source_repository_tree_id,
        writer_id=context["binding"].writer_id,
        fencing_token=context["binding"].writer_fencing_token,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    lifecycle_lock = _resolve_under(
        context["state_root"], ".duckdb-retry-reset.lifecycle.lock"
    )
    with exclusive_file_lock(
        lifecycle_lock, timeout_seconds=lock_timeout_seconds
    ):
        events = _execution_intent_events(source, request_id=request.request_id)
        if not events:
            return None
        if len(events) != 1:
            raise DuckDBRetryResetConflict(
                "retry request has duplicate execution-intent events"
            )
        projection = _execution_intent_from_event(
            events[0], state_root=context["state_root"]
        )
        projection_path = Path(str(projection["projection_path"]))
        if projection_path.exists():
            stored = _read_bounded_json(
                projection_path, "retry execution-intent projection"
            )
            if stored != projection:
                raise DuckDBRetryResetCorruptionError(
                    "retry execution-intent projection differs from its event"
                )
        elif repair_projection:
            # Event-before-projection is a safe pre-effect crash boundary: the
            # writer-fenced event embeds the complete immutable projection.
            _write_json_durable(projection_path, projection)
        return _verify_execution_intent_projection(
            projection,
            request=request,
            trusted_policy=trusted_policy,
            trusted_owner=trusted_owner,
            expected_parent_journal_path=parent_path,
            source=source,
            require_original_task=True,
        )


def _load_bound_execution_intent(
    binding_record: Mapping[str, Any],
    *,
    state_root: Path,
) -> dict[str, Any]:
    if set(binding_record) != {
        "schema",
        "execution_intent_cid",
        "projection_path",
        "request_digest",
        "parent_intent_cid",
        "preparation_event",
    } or binding_record.get("schema") != RETRY_RESET_EXECUTION_INTENT_BINDING_SCHEMA:
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent parent binding is malformed"
        )
    intent_cid = binding_record.get("execution_intent_cid")
    if not isinstance(intent_cid, str) or not re.fullmatch(
        r"b[a-z2-7]{20,100}", intent_cid
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent CID is malformed"
        )
    expected_path = _execution_intent_root(state_root) / f"{intent_cid}.json"
    if binding_record.get("projection_path") != str(expected_path):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent projection path is not canonical"
        )
    projection = _read_bounded_json(
        expected_path, "retry execution-intent projection"
    )
    if _execution_intent_binding(projection) != dict(binding_record):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent parent binding changed"
        )
    return projection


def _validate_drained_parent_journal(
    parent_journal: Mapping[str, Any],
    *,
    projection: Mapping[str, Any],
    binding_record: Mapping[str, Any],
) -> None:
    prepared = projection.get("parent_prepared")
    if not isinstance(prepared, Mapping):
        raise DuckDBRetryResetCorruptionError(
            "retry execution intent lacks parent PREPARED material"
        )
    immutable_prepared = {
        key: value
        for key, value in prepared.items()
        if key not in {"phase", "updated_at", "lifecycle_owners"}
    }
    prepared_owners = prepared.get("lifecycle_owners")
    current_owners = parent_journal.get("lifecycle_owners")
    if (
        any(
            parent_journal.get(key) != value
            for key, value in immutable_prepared.items()
        )
        or not isinstance(prepared_owners, list)
        or not isinstance(current_owners, list)
        or current_owners[: len(prepared_owners)] != prepared_owners
    ):
        raise DuckDBRetryResetCorruptionError(
            "drained parent journal differs from its durable PREPARED event"
        )
    if (
        parent_journal.get("phase") != "leased"
        or parent_journal.get("intent_cid") != prepared.get("intent_cid")
        or parent_journal.get("execution_intent") != dict(binding_record)
        or not isinstance(parent_journal.get("drain_process_tree"), list)
        or not isinstance(parent_journal.get("drain_started_at"), str)
        or not parent_journal.get("drain_started_at")
        or not isinstance(parent_journal.get("drain_cid"), str)
        or parent_journal.get("drain_cid") != _parent_drain_cid(parent_journal)
        or not isinstance(parent_journal.get("drained_at"), str)
        or not parent_journal.get("drained_at")
        or not isinstance(parent_journal.get("drained_cid"), str)
        or parent_journal.get("drained_cid")
        != _parent_drained_cid(parent_journal)
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution requires an exact quiescent leased parent lifecycle"
        )
    drain_tree = parent_journal.get("drain_process_tree")
    assert isinstance(drain_tree, list)
    if len(drain_tree) > 4_096:
        raise DuckDBRetryResetCorruptionError(
            "drained parent process tree exceeds its governed bound"
        )
    drain_identities = [
        _strict_process_identity(item, "drained parent process identity")
        for item in drain_tree
    ]
    old_master = prepared.get("old_master")
    old_tree = prepared.get("old_process_tree")
    if not isinstance(old_master, Mapping) or not isinstance(old_tree, list):
        raise DuckDBRetryResetCorruptionError(
            "retry execution intent lost its old process identities"
        )
    master_identity = _strict_process_identity(
        old_master.get("actual"), "drained parent master identity"
    )
    old_identities = [
        _strict_process_identity(item, "old parent process identity")
        for item in old_tree
    ]
    if (
        drain_identities
        and drain_identities[0] != master_identity
        or len({item["pid"] for item in drain_identities})
        != len(drain_identities)
    ):
        raise DuckDBRetryResetCorruptionError(
            "drained parent process tree is not an exact identity snapshot"
        )
    captured = old_identities + drain_identities
    still_live = sorted(
        {
            item["pid"]
            for item in captured
            if _live_process_identity(item["pid"]) == item
        }
    )
    if still_live:
        raise DuckDBRetryResetQuiescenceError(
            "drained parent process identities remain live: "
            + ", ".join(str(pid) for pid in still_live)
        )
    session_id = old_master.get("dedicated_session_id")
    if (
        not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id != master_identity["pid"]
    ):
        raise DuckDBRetryResetCorruptionError(
            "drained parent session identity is malformed"
        )
    session_members = _process_session_members(session_id)
    if session_members:
        raise DuckDBRetryResetQuiescenceError(
            "drained parent process session remains live: "
            + ", ".join(str(pid) for pid in session_members)
        )
    expected_path = Path(str(projection.get("parent_journal_path") or ""))
    stored = _read_bounded_json(expected_path, "drained parent lifecycle journal")
    if stored != dict(parent_journal):
        raise DuckDBRetryResetCorruptionError(
            "drained parent lifecycle changed at the reset boundary"
        )


def _verify_checkout_lease_assertion(
    assertion: Mapping[str, Any] | None,
    verifier: Callable[[Mapping[str, Any]], bool] | None,
) -> None:
    if not isinstance(assertion, Mapping) or not assertion:
        raise DuckDBRetryResetAuthorizationError(
            "retry execution intent requires a checkout-lease assertion record"
        )
    if verifier is None:
        raise DuckDBRetryResetAuthorizationError(
            "retry execution intent requires a checkout-lease verifier"
        )
    try:
        verified = verifier(assertion)
    except Exception as exc:
        raise DuckDBRetryResetAuthorizationError(
            "checkout-lease assertion could not be verified"
        ) from exc
    if verified is not True:
        raise DuckDBRetryResetAuthorizationError(
            "checkout-lease assertion was not accepted"
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
            "execution_intent": journal.get("execution_intent"),
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
        "execution_intent_cid": (
            journal.get("execution_intent", {}).get("execution_intent_cid")
            if isinstance(journal.get("execution_intent"), Mapping)
            else None
        ),
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
    # Completion is historical authority.  Later governed plan/writer rollover
    # must not erase an already committed one-shot result; the original
    # generation is verified from the content-bound receipt and event bodies.
    source = DuckDBTaskSource(database_path)
    execution_binding = journal.get("execution_intent")
    if execution_binding is not None:
        if not isinstance(execution_binding, Mapping):
            raise DuckDBRetryResetCorruptionError(
                "completed retry-reset execution intent is malformed"
            )
        events = _execution_intent_events(
            source, request_id=str(journal.get("request_id") or "")
        )
        if len(events) != 1:
            raise DuckDBRetryResetCorruptionError(
                "completed retry-reset lacks its durable execution-intent event"
            )
        event_projection = _execution_intent_from_event(
            events[0], state_root=state_root
        )
        if _execution_intent_binding(event_projection) != dict(execution_binding):
            raise DuckDBRetryResetCorruptionError(
                "completed retry-reset execution-intent event changed"
            )
        projection_path = Path(str(event_projection["projection_path"]))
        if projection_path.exists() and _read_bounded_json(
            projection_path, "retry execution-intent projection"
        ) != event_projection:
            raise DuckDBRetryResetCorruptionError(
                "completed retry-reset projection conflicts with event history"
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
        or body.get("execution_intent_cid")
        != receipt.get("execution_intent_cid")
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


def _execution_intent_request(projection: Mapping[str, Any]) -> OperationRequest:
    """Decode and rederive the canonical request embedded in one intent."""

    record = projection.get("request")
    if not isinstance(record, Mapping) or set(record) != {
        "request_id",
        "canonical_json",
        "digest",
    }:
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent request evidence is malformed"
        )
    raw_json = record.get("canonical_json")
    if not isinstance(raw_json, str) or not raw_json:
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent canonical request is missing"
        )
    try:
        payload = json.loads(
            raw_json,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
        )
        request = decode_operation_request(payload)
    except Exception as exc:
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent canonical request is malformed"
        ) from exc
    if (
        _request_canonical_evidence(request) != {
            "canonical_json": raw_json,
            "digest": record.get("digest"),
        }
        or record.get("request_id") != request.request_id
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent request evidence is not canonical"
        )
    return request


def _validate_historical_execution_intent_projection(
    projection: Mapping[str, Any],
    *,
    request: OperationRequest,
    state_root: Path,
    database_path: Path,
    expected_parent_path: Path,
) -> None:
    """Validate immutable intent authority without requiring live policy freshness."""

    decision = request.authorization
    idempotency = request.idempotency
    if decision is None or idempotency is None or len(request.expected_effects) != 1:
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent request lacks immutable mutation authority"
        )
    binding = _binding_from_parameters(request.parameters)
    authorization = projection.get("authorization")
    expected_effect = projection.get("expected_effect")
    idempotency_record = projection.get("idempotency")
    request_file_record = projection.get("request_file")
    database = projection.get("database")
    repository_generation = projection.get("repository_generation")
    owner_record = projection.get("owner")
    policy_record = projection.get("policy")
    parent = projection.get("parent_prepared")
    prepared_at_ms = projection.get("prepared_at_ms")
    if not all(
        isinstance(item, Mapping)
        for item in (
            authorization,
            expected_effect,
            idempotency_record,
            request_file_record,
            database,
            repository_generation,
            owner_record,
            policy_record,
            parent,
        )
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent immutable authority is malformed"
        )
    assert isinstance(authorization, Mapping)
    assert isinstance(expected_effect, Mapping)
    assert isinstance(idempotency_record, Mapping)
    assert isinstance(request_file_record, Mapping)
    assert isinstance(database, Mapping)
    assert isinstance(repository_generation, Mapping)
    assert isinstance(owner_record, Mapping)
    assert isinstance(policy_record, Mapping)
    assert isinstance(parent, Mapping)
    effect = request.expected_effects[0]
    try:
        request_file = _request_file_evidence(
            str(request_file_record.get("raw_json") or "").encode("utf-8"),
            request=request,
        )
    except DuckDBRetryResetError as exc:
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent request-file evidence changed"
        ) from exc
    owner_payload = owner_record.get("payload")
    policy_payload = policy_record.get("payload")
    if not isinstance(owner_payload, Mapping) or not isinstance(
        policy_payload, Mapping
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent historical trust payload is malformed"
        )
    parent_owner = parent.get("owner_configuration")
    parent_policy = parent.get("policy")
    policy_permits = policy_payload.get("permits")
    current_trees = policy_payload.get("current_tree_ids")
    current_objectives = policy_payload.get("current_objective_revisions")
    active_fences = policy_payload.get("active_lease_fences")
    if (
        not isinstance(parent_owner, Mapping)
        or not isinstance(parent_policy, Mapping)
        or not isinstance(current_trees, Mapping)
        or not isinstance(current_objectives, Mapping)
        or not isinstance(active_fences, Mapping)
        or projection.get("schema") != RETRY_RESET_EXECUTION_INTENT_SCHEMA
        or request.state_root != str(state_root)
        or projection.get("parent_journal_path") != str(expected_parent_path)
        or projection.get("request") != {
            "request_id": request.request_id,
            **_request_canonical_evidence(request),
        }
        or dict(request_file_record) != request_file
        or dict(authorization)
        != {
            "decision": decision.to_record(),
            "decision_digest": decision.decision_id,
            "evaluated_at_ms": decision.evaluated_at_ms,
            "expires_at_ms": decision.expires_at_ms,
        }
        or dict(expected_effect)
        != {"effect": effect.to_record(), "effect_id": effect.effect_id}
        or dict(idempotency_record)
        != {
            "record": idempotency.to_record(),
            "content_id": idempotency.content_id,
        }
        or not isinstance(prepared_at_ms, int)
        or isinstance(prepared_at_ms, bool)
        or decision.expires_at_ms is None
        or not decision.evaluated_at_ms <= prepared_at_ms < decision.expires_at_ms
        or owner_record.get("payload_digest")
        != _digest_bytes(_canonical_bytes(owner_payload))
        or policy_record.get("payload_digest")
        != _digest_bytes(_canonical_bytes(policy_payload))
        or parent_owner.get("payload") != owner_payload
        or parent_owner.get("digest") != owner_record.get("file_digest")
        or parent_policy.get("digest") != policy_record.get("file_digest")
        or owner_payload.get("schema") != RETRY_RESET_OWNER_SCHEMA
        or owner_payload.get("repository_root") != request.repository_root
        or owner_payload.get("repository_id") != request.repository_id
        or owner_payload.get("database_path") != binding.database_path
        or owner_payload.get("task_source_repository_tree_id")
        != binding.task_source_repository_tree_id
        or owner_payload.get("policy_digest") != policy_record.get("file_digest")
        or owner_payload.get("lanes")
        != [
            {
                "state_prefix": lane.state_prefix,
                "state_path": lane.state_path,
                "queue_path": lane.queue_path,
            }
            for lane in binding.lanes
        ]
        or owner_payload.get("lifecycle_owner_paths")
        != list(binding.lifecycle_owner_paths)
        or policy_payload.get("schema") != RETRY_RESET_POLICY_SCHEMA
        or policy_payload.get("policy_id") != request.policy_id
        or policy_payload.get("policy_revision") != request.policy_revision
        or not isinstance(policy_permits, list)
        or decision.to_record() not in policy_permits
        or current_trees.get(request.repository_id) != request.tree_id
        or current_objectives.get(request.objective_id)
        != request.objective_revision
        or active_fences.get(request.lease_id)
        != request.fencing_epoch
        or database.get("database_path") != str(database_path)
        or database.get("plan_root_cid") != binding.plan_root_cid
        or database.get("task_source_repository_tree_id")
        != binding.task_source_repository_tree_id
        or database.get("task")
        != {
            "task_cid": binding.task_cid,
            "task_alias": binding.task_alias,
            "status": binding.expected_status,
            "revision": binding.task_revision,
        }
        or database.get("writer")
        != {
            "writer_id": binding.writer_id,
            "fencing_token": binding.writer_fencing_token,
        }
        or repository_generation.get("repository_head_commit")
        != binding.repository_head_commit
        or repository_generation.get("repository_head_tree") != request.tree_id
        or repository_generation.get("worktree_status") != ""
        or repository_generation.get("worktree_status_digest")
        != _digest_bytes(b"")
        or not isinstance(repository_generation.get("submodule_status"), str)
        or repository_generation.get("submodule_status_digest")
        != _digest_bytes(
            str(repository_generation.get("submodule_status") or "").encode("utf-8")
        )
        or projection.get("lanes") != [lane.to_dict() for lane in binding.lanes]
        or parent.get("environment") != projection.get("environment_generation")
        or parent.get("old_master") != projection.get("old_master")
        or parent.get("old_process_tree") != projection.get("old_process_tree")
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry execution-intent historical authority is not content-bound"
        )


def _canonical_parent_journal_path(
    state_root: Path, request: OperationRequest
) -> Path:
    digest = _request_canonical_evidence(request)["digest"]
    return _resolve_under(
        state_root, "duckdb-retry-reset/journals/lifecycle"
    ) / f"{digest.removeprefix('sha256:')}.json"


def _assert_owner_controlled_journal(path: Path, state_root: Path) -> None:
    try:
        root_metadata = state_root.lstat()
        metadata = path.lstat()
    except OSError as exc:
        raise DuckDBRetryResetCorruptionError(
            f"retry lifecycle journal is unavailable: {path}"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != root_metadata.st_uid
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise DuckDBRetryResetCorruptionError(
            f"retry lifecycle journal is not owner-controlled: {path}"
        )


def _validate_parent_execution_intent_correlation(
    parent: Mapping[str, Any],
    *,
    projection: Mapping[str, Any],
    binding_record: Mapping[str, Any],
) -> str:
    """Verify that a current parent journal descends from the embedded PREPARED state."""

    prepared = projection.get("parent_prepared")
    if not isinstance(prepared, Mapping):
        raise DuckDBRetryResetCorruptionError(
            "retry execution intent lacks parent PREPARED material"
        )
    phase = parent.get("phase")
    if (
        not isinstance(phase, str)
        or phase not in _PARENT_LIFECYCLE_PHASE_RANK
        or set(parent).difference(_PARENT_LIFECYCLE_ALLOWED_FIELDS)
        or parent.get("schema") != _PARENT_LIFECYCLE_SCHEMA
        or parent.get("program_id") != _PARENT_PROGRAM_ID
        or parent.get("intent_cid") != prepared.get("intent_cid")
        or parent.get("execution_intent") != dict(binding_record)
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry parent lifecycle journal is not canonical"
        )
    immutable_prepared = {
        key: value
        for key, value in prepared.items()
        if key not in {"phase", "updated_at", "lifecycle_owners"}
    }
    prepared_owners = prepared.get("lifecycle_owners")
    current_owners = parent.get("lifecycle_owners")
    if (
        any(parent.get(key) != value for key, value in immutable_prepared.items())
        or not isinstance(prepared_owners, list)
        or not isinstance(current_owners, list)
        or current_owners[: len(prepared_owners)] != prepared_owners
        or len(current_owners) > MAX_OWNER_PATHS
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry parent lifecycle differs from its durable PREPARED event"
        )
    rank = _PARENT_LIFECYCLE_PHASE_RANK[phase]
    if rank >= 1:
        drain_tree = parent.get("drain_process_tree")
        if (
            not isinstance(drain_tree, list)
            or len(drain_tree) > 4_096
            or not isinstance(parent.get("drain_started_at"), str)
            or not parent.get("drain_started_at")
            or parent.get("drain_cid") != _parent_drain_cid(parent)
        ):
            raise DuckDBRetryResetCorruptionError(
                "retry parent drain evidence is not content-bound"
            )
        for identity in drain_tree:
            _strict_process_identity(identity, "drained parent process identity")
    if rank >= 2 and (
        not isinstance(parent.get("drained_at"), str)
        or not parent.get("drained_at")
        or parent.get("drained_cid") != _parent_drained_cid(parent)
    ):
        raise DuckDBRetryResetCorruptionError(
            "retry parent quiescence evidence is not content-bound"
        )
    return phase


def inspect_incomplete_retry_resets(
    state_root: str | os.PathLike[str],
) -> tuple[dict[str, Any], ...]:
    """Verify journals plus DuckDB pre-effect events and return launch blockers."""

    state = Path(state_root).resolve()
    root = _resolve_under(state, "duckdb-retry-reset/journals")
    result: list[dict[str, Any]] = []
    journals: dict[Path, dict[str, Any]] = {}
    completed_receipts: dict[Path, dict[str, Any]] = {}
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise DuckDBRetryResetCorruptionError("retry-reset journal root is unsafe")
        journal_entries = tuple(sorted(root.iterdir(), key=lambda item: item.name))
        reset_entries = tuple(
            path
            for path in journal_entries
            if not (
                path.name == "lifecycle"
                and path.is_dir()
                and not path.is_symlink()
            )
        )
        if len(reset_entries) > MAX_EXECUTION_INTENTS:
            raise DuckDBRetryResetCorruptionError(
                "retry-reset journal population exceeds its governed bound"
            )
        for path in reset_entries:
            if path.suffix != ".json" or path.is_symlink() or not path.is_file():
                raise DuckDBRetryResetCorruptionError(
                    f"foreign retry-reset journal entry: {path}"
                )
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
            journals[path] = payload
            if payload.get("phase") == "completed":
                completed_receipts[path] = _verify_completed_journal(
                    state_root=state, path=path, journal=payload
                )
            else:
                result.append(
                    {
                        "path": str(path),
                        "phase": payload["phase"],
                        "request_id": payload.get("request_id", ""),
                        "task_cid": payload.get("binding", {}).get("task_cid", ""),
                    }
                )

    owner_path = _resolve_under(state, RETRY_RESET_OWNER_FILE)
    projection_root = _execution_intent_root(state)
    if not owner_path.exists():
        evidence_paths = (
            root.parent,
            _resolve_under(state, ".duckdb-retry-reset.lifecycle.lock"),
        )
        governed_evidence = bool(journals)
        for path in evidence_paths:
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise DuckDBRetryResetCorruptionError(
                    f"cannot inspect retry-reset runtime evidence: {path}"
                ) from exc
            governed_evidence = True
        if state.exists():
            if state.is_symlink() or not state.is_dir():
                raise DuckDBRetryResetCorruptionError(
                    "retry-reset state root is unsafe"
                )
            try:
                top_level_entries = state.iterdir()
                governed_evidence = governed_evidence or any(
                    entry.name.endswith((".duckdb", ".duckdb.lock"))
                    for entry in top_level_entries
                )
            except OSError as exc:
                raise DuckDBRetryResetCorruptionError(
                    "cannot inspect retry-reset state root"
                ) from exc
        if governed_evidence:
            raise DuckDBRetryResetAuthorizationError(
                "governed DuckDB/runtime evidence has no canonical owner configuration"
            )
        return tuple(result)

    owner = _load_owner_config(state)
    policy_path = _resolve_under(state, owner.policy_path)
    policy = _load_policy(policy_path, expected_digest=owner.policy_digest)
    database_path = _resolve_under(state, owner.database_path)
    _assert_regular_path(database_path, "DuckDB task source")
    source = DuckDBTaskSource(
        database_path,
        expected_repository_tree_id=owner.task_source_repository_tree_id,
    )
    events = _all_execution_intent_events(source)
    seen_request_ids: set[str] = set()
    seen_intent_cids: set[str] = set()
    seen_projection_paths: set[Path] = set()
    seen_reset_paths: set[Path] = set()

    for event in events:
        projection = _execution_intent_from_event(event, state_root=state)
        request = _execution_intent_request(projection)
        intent_cid = str(projection.get("execution_intent_cid") or "")
        if (
            request.repository_root != owner.repository_root
            or request.repository_id != owner.repository_id
            or request.request_id in seen_request_ids
            or intent_cid in seen_intent_cids
        ):
            raise DuckDBRetryResetCorruptionError(
                "retry execution-intent history conflicts with canonical ownership"
            )
        seen_request_ids.add(request.request_id)
        seen_intent_cids.add(intent_cid)

        expected_parent_path = _canonical_parent_journal_path(state, request)
        projection_path = Path(str(projection.get("projection_path") or ""))
        seen_projection_paths.add(projection_path)
        assert request.idempotency is not None
        reset_path = root / (
            _journal_key_cid(request.idempotency.to_dict()) + ".json"
        )
        seen_reset_paths.add(reset_path)
        reset_journal = journals.get(reset_path)
        _validate_historical_execution_intent_projection(
            projection,
            request=request,
            state_root=state,
            database_path=database_path,
            expected_parent_path=expected_parent_path,
        )
        if reset_journal is None or reset_journal.get("phase") != "completed":
            _verify_execution_intent_projection(
                projection,
                request=request,
                trusted_policy=policy,
                trusted_owner=owner,
                expected_parent_journal_path=expected_parent_path,
                source=source,
                require_original_task=reset_journal is None,
            )

        projection_missing = not projection_path.exists()
        if not projection_missing:
            stored_projection = _read_bounded_json(
                projection_path, "retry execution-intent projection"
            )
            if stored_projection != projection:
                raise DuckDBRetryResetCorruptionError(
                    "retry execution-intent projection conflicts with event history"
                )

        if not expected_parent_path.exists():
            if reset_journal is not None:
                raise DuckDBRetryResetCorruptionError(
                    "retry reset has no canonical parent lifecycle journal"
                )
            result.append(
                {
                    "path": str(projection_path),
                    "phase": (
                        "execution_intent_event_appended"
                        if projection_missing
                        else "execution_intent_prepared"
                    ),
                    "request_id": request.request_id,
                    "task_cid": event.get("task_cid", ""),
                }
            )
            continue

        _assert_owner_controlled_journal(expected_parent_path, state)
        parent = _read_bounded_json(
            expected_parent_path, "retry parent lifecycle journal"
        )
        binding_record = _execution_intent_binding(projection)
        parent_phase = _validate_parent_execution_intent_correlation(
            parent,
            projection=projection,
            binding_record=binding_record,
        )

        if reset_journal is None:
            result.append(
                {
                    "path": str(expected_parent_path),
                    "phase": (
                        "execution_intent_projection_missing"
                        if projection_missing
                        else f"parent_{parent_phase}"
                    ),
                    "request_id": request.request_id,
                    "task_cid": event.get("task_cid", ""),
                }
            )
            continue
        if (
            reset_journal.get("request_id") != request.request_id
            or reset_journal.get("execution_intent") != binding_record
        ):
            raise DuckDBRetryResetCorruptionError(
                "retry-reset journal conflicts with its execution-intent event"
            )
        if reset_journal.get("phase") == "completed":
            receipt = completed_receipts[reset_path]
            parent_receipt = parent.get("retry_reset_receipt")
            if _PARENT_LIFECYCLE_PHASE_RANK[parent_phase] >= 4 and (
                not isinstance(parent_receipt, Mapping)
                or dict(parent_receipt) != receipt
            ):
                raise DuckDBRetryResetCorruptionError(
                    "retry parent terminal evidence conflicts with the reset receipt"
                )
        if projection_missing:
            result.append(
                {
                    "path": str(projection_path),
                    "phase": "execution_intent_projection_missing",
                    "request_id": request.request_id,
                    "task_cid": event.get("task_cid", ""),
                }
            )

    for path, journal in journals.items():
        execution_binding = journal.get("execution_intent")
        if execution_binding is not None and path not in seen_reset_paths:
            raise DuckDBRetryResetCorruptionError(
                "retry-reset journal lacks its durable execution-intent event"
            )

    if projection_root.exists():
        if projection_root.is_symlink() or not projection_root.is_dir():
            raise DuckDBRetryResetCorruptionError(
                "retry execution-intent projection root is unsafe"
            )
        projection_entries = tuple(
            sorted(projection_root.iterdir(), key=lambda item: item.name)
        )
        if len(projection_entries) > MAX_EXECUTION_INTENTS:
            raise DuckDBRetryResetCorruptionError(
                "retry execution-intent projection population exceeds its governed bound"
            )
        for path in projection_entries:
            if (
                path.suffix != ".json"
                or path.is_symlink()
                or not path.is_file()
                or path not in seen_projection_paths
            ):
                raise DuckDBRetryResetCorruptionError(
                    f"orphan retry execution-intent projection: {path}"
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
    execution_intent: Mapping[str, Any] | None = None,
    parent_journal: Mapping[str, Any] | None = None,
    checkout_lease_assertion: Mapping[str, Any] | None = None,
    checkout_lease_verifier: Callable[[Mapping[str, Any]], bool] | None = None,
    clock_ms: Callable[[], int] | None = None,
    lock_timeout_seconds: float = 30.0,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute or recover one exact authorized retry reset."""

    if request.operation is not Operation.RETRY or request.dry_run:
        raise DuckDBRetryResetError(
            "retry reset requires a real Operation.RETRY request"
        )
    if not isinstance(execution_intent, Mapping):
        raise DuckDBRetryResetAuthorizationError(
            "retry reset execution requires a durable execution intent"
        )
    if not isinstance(parent_journal, Mapping):
        raise DuckDBRetryResetAuthorizationError(
            "retry reset execution requires its parent lifecycle journal"
        )
    if not isinstance(checkout_lease_assertion, Mapping) or not checkout_lease_assertion:
        raise DuckDBRetryResetAuthorizationError(
            "retry reset execution requires a checkout-lease assertion record"
        )
    if checkout_lease_verifier is None:
        raise DuckDBRetryResetAuthorizationError(
            "retry reset execution requires a checkout-lease verifier"
        )
    now_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
    selected_now_ms = now_ms()
    fresh_at_entry = True
    try:
        ControlMutationAuthorizer(
            trusted_policy, clock_ms=lambda: selected_now_ms
        ).validate(request)
    except Exception as exc:
        decision = request.authorization
        expired_execution = bool(
            decision is not None
            and decision.expires_at_ms is not None
            and selected_now_ms >= decision.expires_at_ms
        )
        if not expired_execution:
            raise DuckDBRetryResetAuthorizationError(str(exc)) from exc
        fresh_at_entry = False

    def assert_boundary_authorization(label: str) -> None:
        boundary_now_ms = now_ms()
        try:
            _validate_retry_request_static(
                request,
                trusted_policy=trusted_policy,
                trusted_owner=trusted_owner,
                clock_ms=lambda: boundary_now_ms,
                require_fresh=fresh_at_entry,
            )
        except Exception as exc:
            raise DuckDBRetryResetAuthorizationError(
                f"retry permit expired or changed {label}: {exc}"
            ) from exc

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
        assert_boundary_authorization("while awaiting mutation locks")
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

        projection = _load_bound_execution_intent(
            execution_intent, state_root=state_root
        )
        reset_journal_exists = journal_path.exists()
        _verify_execution_intent_projection(
            projection,
            request=request,
            trusted_policy=trusted_policy,
            trusted_owner=trusted_owner,
            expected_parent_journal_path=Path(
                str(projection.get("parent_journal_path") or "")
            ),
            source=source,
            require_original_task=not reset_journal_exists,
        )
        _validate_drained_parent_journal(
            parent_journal,
            projection=projection,
            binding_record=execution_intent,
        )
        _verify_checkout_lease_assertion(
            checkout_lease_assertion,
            checkout_lease_verifier,
        )
        assert_boundary_authorization("at the mutation boundary")

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
                or journal.get("execution_intent") != dict(execution_intent)
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
                "execution_intent": dict(execution_intent),
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
                "execution_intent": journal.get("execution_intent"),
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
                    "execution_intent_cid": execution_intent.get(
                        "execution_intent_cid"
                    ),
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
                        write_precondition=lambda: assert_boundary_authorization(
                            "while awaiting the DuckDB task-source write lock"
                        ),
                        expected_task_status=binding.expected_status,
                        expected_task_revision=binding.task_revision,
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
                        writer_id=binding.writer_id,
                        fencing_token=binding.writer_fencing_token,
                        write_precondition=lambda: assert_boundary_authorization(
                            "while awaiting the DuckDB task-source write lock"
                        ),
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
                "execution_intent_cid": execution_intent.get(
                    "execution_intent_cid"
                ),
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
                    "execution_intent_cid": execution_intent.get(
                        "execution_intent_cid"
                    ),
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Mutation is lifecycle-owner-only. --request-file validates trust "
            "inputs but exits nonzero because a CLI process cannot supply the "
            "in-process checkout-lease verifier."
        ),
    )
    parser.add_argument(
        "--request-file",
        type=Path,
        help=(
            "request to validate and refuse for direct mutation; use the parent "
            "retry lifecycle owner to execute"
        ),
    )
    parser.add_argument("--inspect-state-root", type=Path)
    parser.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=30.0,
        help=(
            "reserved for lifecycle-owned execution; direct CLI mutation "
            "remains disabled"
        ),
    )
    args = parser.parse_args(argv)
    if args.inspect_state_root is None and args.request_file is None:
        parser.error(
            "choose --inspect-state-root or --request-file trust validation; "
            "direct mutation is lifecycle-owner-only"
        )
    if args.inspect_state_root is not None and args.request_file is not None:
        parser.error("--inspect-state-root cannot be combined with --request-file")
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
        _load_policy(
            _resolve_under(state_root, owner.policy_path),
            expected_digest=owner.policy_digest,
        )
        raise DuckDBRetryResetAuthorizationError(
            "direct CLI retry mutation is lifecycle-owner-only; the parent "
            "lifecycle must call execute_duckdb_retry_reset with its durable "
            "execution intent and in-process checkout-lease verifier"
        )
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
    "RETRY_RESET_EXECUTION_INTENT_BINDING_SCHEMA",
    "RETRY_RESET_EXECUTION_INTENT_EVENT_SCHEMA",
    "RETRY_RESET_EXECUTION_INTENT_SCHEMA",
    "RETRY_RESET_GRANT",
    "RETRY_RESET_OWNER_FILE",
    "RETRY_RESET_OWNER_SCHEMA",
    "RETRY_RESET_POLICY_SCHEMA",
    "RetryResetOwnerConfig",
    "execute_duckdb_retry_reset",
    "inspect_incomplete_retry_resets",
    "prepare_duckdb_retry_reset_execution_intent",
    "recover_duckdb_retry_reset_execution_intent",
    "retry_reset_execution_intent_binding",
    "retry_reset_expected_effect",
]
